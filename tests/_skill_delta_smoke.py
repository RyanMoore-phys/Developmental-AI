"""Smoke test: skills as DELTAS, and old skills still work after conversion.

The question this has to answer is not "does it compress" — it is "are the
agent's learned skills still the same skills afterwards". So every check is
about FIDELITY first and size second.

  1. round trip            — base + delta rebuilds the original weights
  2. behavioural identity  — the rebuilt actor picks the same actions, with
                             the same distribution (this is the real test)
  3. compression           — a bank of similar skills actually gets smaller
  4. genuinely different   — a skill that is NOT near the base is not mangled
                             (it falls back to dense rather than lying)
  5. transparent load      — SkillBank.load_skill_policy returns a normal
                             dense actor for a delta skill; callers unchanged
  6. missing base          — a delta skill without its base is REFUSED loudly,
                             never returned half-formed
  7. end-to-end converter  — the CLI converts a real bank and the converted
                             skills bind into option slots and act
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.skill_bank import skill_delta as sdelta
from developmental_ai.skill_bank.skill_bank import SkillBank
from developmental_ai.policy.actor_critic import ActorNetwork

IN_DIM, ACT_DIM, HID = 264, 36, 256


def _actor(seed, jitter=0.0, base=None):
    """An actor near `base` (like a real bank: all descended from one policy)."""
    torch.manual_seed(seed)
    a = ActorNetwork(IN_DIM, ACT_DIM, HID, continuous=False)
    sd = {k: v.detach().clone() for k, v in a.state_dict().items()}
    if base is not None:
        g = torch.Generator().manual_seed(seed)
        for k in sd:
            noise = torch.randn(sd[k].shape, generator=g) * jitter
            sd[k] = base[k].clone() + noise
    return sd


def test_round_trip_and_behaviour():
    torch.manual_seed(0)
    root = _actor(1)
    sds = [_actor(10 + i, jitter=0.002, base=root) for i in range(6)]
    base = sdelta.make_base(sds)

    worst_agree, worst_kl, ratios = 1.0, 0.0, []
    for sd in sds:
        d = sdelta.to_delta(base, sd, tol=0.01)
        rb = sdelta.from_delta(base, d)
        assert set(rb) == set(sd), "reconstruction lost tensors"
        beh = sdelta.verify_behaviour(
            ActorNetwork,
            {"obs_dim": IN_DIM, "action_dim": ACT_DIM,
             "hidden_dim": HID, "continuous": False},
            sd, rb, in_dim=IN_DIM)
        worst_agree = min(worst_agree, beh["decisive_agree"])
        worst_kl = max(worst_kl, beh["max_kl"])
        ratios.append(sum(t.numel() for t in sd.values())
                      / max(1, sdelta.param_count(d)))
    assert worst_agree == 1.0, f"decisive action changed ({worst_agree})"
    assert worst_kl < 1e-3, f"action distribution moved (KL {worst_kl})"
    # The jitter here is iid Gaussian, i.e. FULL RANK by construction — which
    # is also what real skill deltas measured as (rank 128 still 3.8% error on
    # the pod bank). So the expected saving is the int8 dense path's ~4x, NOT
    # a rank win. Asserting a rank win here would be asserting a property the
    # real data does not have, and would only be satisfiable by loosening the
    # fidelity gate — i.e. by damaging the skills.
    assert min(ratios) > 3.0, f"expected ~4x from int8 ({min(ratios):.2f}x)"
    print(f"  1-3. round trip + behaviour + compression ok "
          f"(agree=1.000, maxKL={worst_kl:.1e}, "
          f"{min(ratios):.1f}-{max(ratios):.1f}x smaller via int8 dense)")


def test_dissimilar_skill_not_mangled():
    """A skill far from the base must stay FAITHFUL, even if that costs size."""
    torch.manual_seed(0)
    root = _actor(1)
    base = sdelta.make_base([_actor(10 + i, jitter=0.002, base=root)
                             for i in range(4)])
    odd = _actor(999)                      # unrelated init: nothing like base
    d = sdelta.to_delta(base, odd, tol=0.01)
    rb = sdelta.from_delta(base, d)
    beh = sdelta.verify_behaviour(
        ActorNetwork,
        {"obs_dim": IN_DIM, "action_dim": ACT_DIM,
         "hidden_dim": HID, "continuous": False},
        odd, rb, in_dim=IN_DIM)
    assert beh["decisive_agree"] == 1.0 and beh["max_kl"] < 1e-3, (
        "a dissimilar skill was distorted by conversion")
    print(f"  4. dissimilar skill preserved (decisive agree=1.000, "
          f"maxKL={beh['max_kl']:.1e}) — falls back to dense, never lies")


def _make_bank(tmp, n=5):
    bank = SkillBank(storage_dir=tmp)
    root = _actor(1)
    for i in range(n):
        sd = {"actor": _actor(10 + i, jitter=0.002, base=root),
              "critic": {"w": torch.randn(4)}}
        bank.save_skill(skill_id=f"ach_{i:02d}_break_x", name=f"skill{i}",
                        policy_state_dict=sd, success_rate=0.8,
                        total_episodes=50, obs_dim=IN_DIM,
                        action_dim=ACT_DIM)
    return bank


def test_transparent_load_and_missing_base():
    tmp = tempfile.mkdtemp()
    try:
        bank = _make_bank(tmp, n=4)
        sid = "ach_00_break_x"
        orig = bank.load_skill_policy(sid)["actor"]

        actors = [bank.load_skill_policy(f"ach_{i:02d}_break_x")["actor"]
                  for i in range(4)]
        base = sdelta.make_base(actors)
        delta = sdelta.to_delta(base, orig, tol=0.01)

        # rewrite that one skill in delta form
        p = bank.skills[sid].policy_path
        sd = torch.load(p, map_location="cpu")
        sd["actor"] = delta
        torch.save(sd, p)

        # (6) base missing -> must refuse LOUDLY, not return a broken actor
        bank._base_actor_cache = None
        assert bank.load_skill_policy(sid) is None, (
            "delta skill loaded without its base — would surface as a "
            "baffling shape error later")

        # (5) base present -> transparent, dense actor, unchanged behaviour
        torch.save(base, os.path.join(tmp, SkillBank.BASE_ACTOR_FILE))
        bank._base_actor_cache = None
        got = bank.load_skill_policy(sid)
        assert got is not None and not sdelta.is_delta(got["actor"])
        beh = sdelta.verify_behaviour(
            ActorNetwork,
            {"obs_dim": IN_DIM, "action_dim": ACT_DIM,
             "hidden_dim": HID, "continuous": False},
            orig, got["actor"], in_dim=IN_DIM)
        assert beh["decisive_agree"] == 1.0 and beh["max_kl"] < 1e-3
        print("  5-6. transparent load ok; missing base refused loudly")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_converter_cli_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        bank = _make_bank(tmp, n=5)
        before = {sid: bank.load_skill_policy(sid)["actor"]
                  for sid in list(bank.skills)}
        tool = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "tools",
            "convert_skills_to_deltas.py")

        dry = subprocess.run([sys.executable, tool, tmp],
                             capture_output=True, text=True)
        assert dry.returncode == 0, dry.stderr[-800:]
        assert "DRY RUN" in dry.stdout
        assert not os.path.exists(os.path.join(tmp, "base_actor.pt")), (
            "dry run wrote files")

        run = subprocess.run([sys.executable, tool, tmp, "--apply"],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-800:]
        assert os.path.exists(os.path.join(tmp, "base_actor.pt"))

        # every skill still loads, and still behaves identically
        bank2 = SkillBank(storage_dir=tmp)
        for sid, orig in before.items():
            got = bank2.load_skill_policy(sid)
            assert got is not None, f"{sid} unloadable after conversion"
            beh = sdelta.verify_behaviour(
                ActorNetwork,
                {"obs_dim": IN_DIM, "action_dim": ACT_DIM,
                 "hidden_dim": HID, "continuous": False},
                orig, got["actor"], in_dim=IN_DIM)
            assert beh["decisive_agree"] == 1.0, f"{sid} changed behaviour"
            assert beh["max_kl"] < 1e-3, f"{sid} distribution moved"
            # originals kept, so the conversion is reversible
            assert os.path.exists(os.path.join(
                os.path.dirname(bank2.skills[sid].policy_path),
                "policy.dense.pt"))

        dense = sum(os.path.getsize(os.path.join(tmp, d, "policy.dense.pt"))
                    for d in os.listdir(tmp)
                    if os.path.isdir(os.path.join(tmp, d)))
        delta = sum(os.path.getsize(os.path.join(tmp, d, "policy.pt"))
                    for d in os.listdir(tmp)
                    if os.path.isdir(os.path.join(tmp, d)))
        base_sz = os.path.getsize(os.path.join(tmp, "base_actor.pt"))
        print(f"  7. converter end-to-end ok: {len(before)} skills, "
              f"{dense/1e6:.2f} MB -> {(delta+base_sz)/1e6:.2f} MB "
              f"(incl. one shared base), behaviour identical, reversible")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for fn in (test_round_trip_and_behaviour,
               test_dissimilar_skill_not_mangled,
               test_transparent_load_and_missing_base,
               test_converter_cli_end_to_end):
        print(f"[skill-delta-smoke] {fn.__name__}")
        fn()
    print("[skill-delta-smoke] ALL PASS")
