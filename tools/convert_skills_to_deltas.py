#!/usr/bin/env python
"""Convert an existing skill bank from COPIES to DELTAS.

    python tools/convert_skills_to_deltas.py <bank_dir>            # dry run
    python tools/convert_skills_to_deltas.py <bank_dir> --apply    # write

WHAT IT DOES
  1. Reads every skill's actor from the bank.
  2. Builds ONE shared base actor (the elementwise mean, which minimises the
     total delta the factorization then has to represent) and writes it as
     `base_actor.pt`.
  3. Re-expresses each skill's actor as base + low-rank modulation, choosing
     the smallest rank per tensor that stays inside the error tolerance.
  4. VERIFIES each conversion behaviourally before accepting it, and refuses
     to write any skill that fails.

WHY THE VERIFY GATE IS NOT OPTIONAL
  These files are the agent's learned skills. A conversion that saves space
  while shifting behaviour has not compressed a memory, it has replaced one.
  So every skill is checked on the thing that matters — does the rebuilt
  actor choose the same actions, with the same distribution — and a skill
  that does not pass is left DENSE. A partially-converted bank is fine
  (loading is transparent either way); a silently-altered one is not.

SAFETY
  Dry run by default. With --apply the original `policy.pt` is preserved as
  `policy.dense.pt` before anything is overwritten, so the conversion is
  reversible with a file move. Nothing is deleted, ever.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.skill_bank import skill_delta as sdelta  # noqa: E402
from developmental_ai.policy.actor_critic import ActorNetwork  # noqa: E402


def _iter_skills(bank_dir):
    for name in sorted(os.listdir(bank_dir)):
        d = os.path.join(bank_dir, name)
        p = os.path.join(d, "policy.pt")
        if os.path.isdir(d) and os.path.exists(p):
            yield name, p


def _actor_dims(actor_sd):
    """Recover (in_dim, action_dim, hidden) from the actor's own tensors, so
    the verifier can rebuild it without consulting the registry (which, as
    the bank's own comments record, has been wrong about arch metadata)."""
    w0 = actor_sd.get("shared.0.weight")
    wh = actor_sd.get("action_head.weight")
    if w0 is None or wh is None:
        return None
    return int(w0.shape[1]), int(wh.shape[0]), int(w0.shape[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bank_dir")
    ap.add_argument("--apply", action="store_true",
                    help="write files (default: dry run)")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max relative reconstruction error per tensor")
    ap.add_argument("--max-rank", type=int, default=64)
    ap.add_argument("--min-agree", type=float, default=1.0,
                    help="required DECISIVE-argmax agreement (probes where "
                         "the skill had a real preference) to accept")
    ap.add_argument("--max-kl", type=float, default=1e-3,
                    help="max allowed action-distribution KL")
    args = ap.parse_args()

    bank = args.bank_dir
    if not os.path.isdir(bank):
        print(f"no such bank dir: {bank}")
        return 2

    skills = list(_iter_skills(bank))
    if not skills:
        print(f"no skills found in {bank}")
        return 1
    print(f"found {len(skills)} skills in {bank}\n")

    # ---- load actors, grouped by shape (a bank can hold several arch eras) --
    loaded, groups = {}, {}
    for name, path in skills:
        sd = torch.load(path, map_location="cpu")
        if sdelta.is_delta(sd.get("actor")):
            print(f"  {name}: already delta-form, skipping")
            continue
        actor = sd.get("actor")
        if not isinstance(actor, dict):
            print(f"  {name}: no actor tensor dict, skipping")
            continue
        loaded[name] = (path, sd, actor)
        key = tuple(sorted((k, tuple(v.shape)) for k, v in actor.items()))
        groups.setdefault(key, []).append(name)

    if not loaded:
        print("nothing to convert")
        return 0
    if len(groups) > 1:
        print(f"NOTE: {len(groups)} distinct actor shapes in this bank; each "
              f"gets its own base (a base must apply to all its tenants).\n")

    total_before = total_after = 0
    converted, refused = [], []

    for gi, (_key, names) in enumerate(sorted(groups.items(),
                                              key=lambda kv: -len(kv[1]))):
        base = sdelta.make_base([loaded[n][2] for n in names])
        base_file = (sdelta and os.path.join(
            bank, "base_actor.pt" if gi == 0 else f"base_actor.g{gi}.pt"))
        print(f"group {gi}: {len(names)} skills, base -> "
              f"{os.path.basename(base_file)}")
        if gi > 0:
            print("  (only group 0's base is auto-loaded by SkillBank; other "
                  "groups are reported but left DENSE)")

        for n in names:
            path, sd, actor = loaded[n]
            delta = sdelta.to_delta(base, actor, tol=args.tol,
                                    max_rank=args.max_rank)
            rebuilt = sdelta.from_delta(base, delta)
            werr = sdelta.weight_error(actor, rebuilt)

            dims = _actor_dims(actor)
            if dims is None:
                print(f"  {n}: unrecognised actor layout — left dense")
                refused.append(n)
                continue
            in_dim, act_dim, hidden = dims
            beh = sdelta.verify_behaviour(
                ActorNetwork,
                {"obs_dim": in_dim, "action_dim": act_dim,
                 "hidden_dim": hidden, "continuous": False},
                actor, rebuilt, in_dim=in_dim)

            n_before = sum(int(t.numel()) for t in actor.values())
            n_after = sdelta.param_count(delta)
            ok = (beh["decisive_agree"] >= args.min_agree
                  and beh["max_kl"] <= args.max_kl
                  and n_after < n_before and gi == 0)
            flag = "OK " if ok else "REFUSED"
            print(f"  {flag} {n:34} {n_before:>9,} -> {n_after:>8,} params "
                  f"({n_before / max(1, n_after):5.1f}x)  "
                  f"werr={werr:.2e} agree={beh['decisive_agree']:.3f} "
                  f"maxKL={beh['max_kl']:.2e}")
            if not ok:
                refused.append(n)
                continue
            converted.append((n, path, sd, delta))
            total_before += n_before
            total_after += n_after

        if args.apply and gi == 0:
            torch.save(base, base_file)

    print()
    print(f"convertible: {len(converted)}   refused (left dense): "
          f"{len(refused)}")
    if converted:
        print(f"actor params {total_before:,} -> {total_after:,} "
              f"({total_before / max(1, total_after):.1f}x smaller)")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to convert.")
        return 0

    for n, path, sd, delta in converted:
        backup = os.path.join(os.path.dirname(path), "policy.dense.pt")
        if not os.path.exists(backup):
            shutil.copy2(path, backup)      # reversible by a file move
        out = dict(sd)
        out["actor"] = delta
        tmp = path + ".tmp"
        torch.save(out, tmp)
        os.replace(tmp, path)               # atomic
    print(f"\nwrote {len(converted)} delta skills + base_actor.pt")
    print("originals preserved alongside as policy.dense.pt")
    with open(os.path.join(bank, "delta_conversion.json"), "w") as f:
        json.dump({"converted": [c[0] for c in converted],
                   "refused": refused,
                   "params_before": total_before,
                   "params_after": total_after}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
