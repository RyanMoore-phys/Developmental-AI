"""Merge duplicate-named skills in the skill bank — NON-DESTRUCTIVELY.

WHY (2026-07-25): the junk-goal factory (AUDIT_FINDINGS #42) minted many
skills for one behaviour and named them all the same — live: 14x
`act_where_breakable_in_reach`, 8x `act_where_dirt_visible`, 22 of 51 skills
across two names. Each copy accumulated its own practice, so per-skill
competence never rose and **0 of 51 skills ever reached mastery**. Pooling a
name's practice into one canonical skill is what makes mastery reachable.

SAFETY MODEL — the skill bank is IRREPLACEABLE, so this tool:
  * NEVER writes to the source directory. It builds a complete merged copy at
    --out and leaves the original byte-identical.
  * copies the canonical skill's FULL payload (policy.pt, note.md, any
    discovery frame) rather than regenerating anything.
  * keeps every NON-duplicate skill exactly as-is.
  * verifies the result before reporting success (counts, payload presence,
    conservation of pooled episode counts).
  * is a no-op on an already-merged bank (idempotent).

CANONICAL CHOICE: the copy with the most `total_episodes` (the most-practised
one, so the best-trained policy survives), tie-broken by earliest `created_at`
so identity is stable across re-runs.

POOLING: episode/invocation/spike counts SUM; rates and averages are recomputed
as episode-weighted means (never a mean-of-means, which would silently
mis-weight a slot that was practised 30x against one practised once);
preconditions are OR-ed (a precondition observed under any copy is real);
created_at = min, updated_at = max.

Usage (read-only against a LIVE bank is safe):
    python scripts/merge_duplicate_skills.py \
        --bank skill_bank_mc_curiosity \
        --out  skill_bank_mc_curiosity.merged
    python scripts/merge_duplicate_skills.py --bank ... --out ... --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List


def _wmean(pairs) -> float:
    """Episode-weighted mean. pairs = [(value, weight), ...]."""
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        vals = [v for v, _ in pairs]
        return float(sum(vals) / len(vals)) if vals else 0.0
    return float(sum(v * w for v, w in pairs) / tot)


def merge_group(group: List[Dict]) -> Dict:
    """Pool a list of same-named skill records into one canonical record."""
    canon = max(group, key=lambda s: (int(s.get("total_episodes", 0) or 0),
                                      -float(s.get("created_at", 0) or 0)))
    out = dict(canon)                       # start from the canonical payload

    eps = [int(s.get("total_episodes", 0) or 0) for s in group]
    tot_eps = sum(eps)
    out["total_episodes"] = tot_eps
    out["successful_episodes"] = sum(
        int(s.get("successful_episodes", 0) or 0) for s in group)
    out["invocations"] = sum(int(s.get("invocations", 0) or 0) for s in group)
    out["option_spikes"] = sum(
        int(s.get("option_spikes", 0) or 0) for s in group)

    # rates/averages: episode-weighted, never mean-of-means
    for field in ("avg_reward", "avg_episode_length", "avg_option_return",
                  "option_spike_ema", "success_rate", "mastery_level"):
        pairs = [(float(s.get(field, 0.0) or 0.0),
                  int(s.get("total_episodes", 0) or 0)) for s in group]
        out[field] = _wmean(pairs)
    # success_rate is better recomputed from the pooled counts when we have them
    if tot_eps > 0:
        out["success_rate"] = float(out["successful_episodes"]) / tot_eps

    out["created_at"] = min(float(s.get("created_at", 0) or 0) for s in group)
    out["updated_at"] = max(float(s.get("updated_at", 0) or 0) for s in group)
    out["last_invoked_at"] = max(
        float(s.get("last_invoked_at", 0) or 0) for s in group)

    # preconditions: OR across copies (observed under any copy = real)
    merged_pre: Dict[str, bool] = {}
    for s in group:
        for k, v in (s.get("preconditions") or {}).items():
            merged_pre[k] = bool(merged_pre.get(k, False) or v)
    if merged_pre:
        out["preconditions"] = merged_pre

    # provenance: record what was folded in, so a merge is auditable/reversible
    out["merged_from"] = sorted(
        s.get("skill_id") for s in group
        if s.get("skill_id") != canon.get("skill_id"))
    out["merged_episode_counts"] = {
        s.get("skill_id"): int(s.get("total_episodes", 0) or 0) for s in group}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument(
        "--repoint", action="store_true",
        help="rewrite policy_path in --bank to the bank's CURRENT location "
             "and exit. REQUIRED after renaming a merged bank into place: the "
             "merge writes paths pointing at --out, so `mv out bank` leaves "
             "every policy_path dangling (hit live 2026-07-25: 0/16 "
             "resolvable after the swap). Idempotent.")
    a = ap.parse_args()

    if a.repoint:
        bank = os.path.abspath(a.bank)
        p = os.path.join(bank, "registry.json")
        reg = json.load(open(p))
        fixed = 0
        for s in reg["skills"]:
            want = os.path.join(bank, s.get("skill_id") or "", "policy.pt")
            if s.get("policy_path") != want and os.path.exists(want):
                s["policy_path"] = want
                fixed += 1
        tmp = p + ".tmp"
        json.dump(reg, open(tmp, "w"))
        os.replace(tmp, p)                      # atomic
        r = json.load(open(p))["skills"]
        ok = sum(1 for s in r if os.path.exists(s.get("policy_path", "")))
        print(f"repointed {fixed} | resolvable: {ok}/{len(r)}")
        return 0 if ok == len(r) else 1

    reg_path = os.path.join(a.bank, "registry.json")
    reg = json.load(open(reg_path))
    skills = reg["skills"]

    by_name: Dict[str, List[Dict]] = defaultdict(list)
    for s in skills:
        by_name[s.get("name") or s.get("skill_id")].append(s)

    dupes = {n: g for n, g in by_name.items() if len(g) > 1}
    n_dupe_skills = sum(len(g) for g in dupes.values())
    print(f"bank: {len(skills)} skills, {len(by_name)} distinct names")
    if not dupes:
        print("no duplicates — nothing to merge (idempotent no-op)")
        return 0
    print(f"duplicates: {len(dupes)} names covering {n_dupe_skills} skills "
          f"-> would become {len(dupes)} (net -{n_dupe_skills - len(dupes)})")
    for n, g in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
        eps = sum(int(s.get("total_episodes", 0) or 0) for s in g)
        canon = max(g, key=lambda s: (int(s.get("total_episodes", 0) or 0),
                                      -float(s.get("created_at", 0) or 0)))
        print(f"   {len(g):3d}x {n:<38} pooled_episodes={eps:<5} "
              f"canonical={canon.get('skill_id')}")

    if a.report_only:
        return 0
    if not a.out:
        print("ERROR: --out required unless --report-only", file=sys.stderr)
        return 2
    if os.path.abspath(a.out) == os.path.abspath(a.bank):
        print("REFUSED: --out must differ from --bank (never in-place)",
              file=sys.stderr)
        return 2

    merged: List[Dict] = []
    keep_dirs = []
    for name, g in by_name.items():
        rec = merge_group(g) if len(g) > 1 else dict(g[0])
        merged.append(rec)
        keep_dirs.append(rec.get("skill_id"))

    # ---- build the merged bank as a COPY; source is never touched ----
    os.makedirs(a.out, exist_ok=True)
    copied = 0
    for sid in keep_dirs:
        src = os.path.join(a.bank, sid or "")
        dst = os.path.join(a.out, sid or "")
        if sid and os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied += 1
    # carry over non-skill state files verbatim (goal slots, LTM, etc.)
    for extra in os.listdir(a.bank):
        p = os.path.join(a.bank, extra)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(a.out, extra))
        elif os.path.isdir(p) and extra not in keep_dirs \
                and not extra.startswith("ach_") and not extra.startswith("skill_"):
            d = os.path.join(a.out, extra)
            if os.path.isdir(d):
                shutil.rmtree(d)
            shutil.copytree(p, d)

    for rec in merged:                      # repoint policy paths into --out
        pp = rec.get("policy_path")
        if pp:
            rec["policy_path"] = pp.replace(
                os.path.abspath(a.bank), os.path.abspath(a.out))
    json.dump({"skills": merged,
               "last_updated": reg.get("last_updated")},
              open(os.path.join(a.out, "registry.json"), "w"))

    # ---- VERIFY before claiming success ----
    v = json.load(open(os.path.join(a.out, "registry.json")))["skills"]
    errs = []
    if len(v) != len(by_name):
        errs.append(f"expected {len(by_name)} skills, wrote {len(v)}")
    if len({s["name"] for s in v}) != len(v):
        errs.append("duplicate names survived the merge")
    src_eps = sum(int(s.get("total_episodes", 0) or 0) for s in skills)
    out_eps = sum(int(s.get("total_episodes", 0) or 0) for s in v)
    if src_eps != out_eps:
        errs.append(f"episode counts not conserved: {src_eps} -> {out_eps}")
    missing = [s["skill_id"] for s in v
               if s.get("policy_path") and not os.path.exists(s["policy_path"])]
    if missing:
        errs.append(f"{len(missing)} merged skills lost their policy payload")
    # the source must be untouched
    if len(json.load(open(reg_path))["skills"]) != len(skills):
        errs.append("SOURCE BANK WAS MODIFIED — this tool must be read-only")

    if errs:
        for e in errs:
            print(f"VERIFY FAILED: {e}", file=sys.stderr)
        return 1
    print(f"\nMERGED OK -> {a.out}")
    print(f"  {len(skills)} -> {len(v)} skills, {copied} payload dirs copied, "
          f"{src_eps} episodes conserved, source untouched")
    print("  swap in only while the run is STOPPED; keep the original as backup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
