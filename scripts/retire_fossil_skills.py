"""Retire FOSSIL skills from the bank — non-destructively.

A fossil is a skill whose GOAL can never be achieved, so it can never earn
competence, can never be offered, and only wastes an option slot and a row of
the meta-policy's action head. Two families exist in this bank, both minted
before later fixes landed:

  * `break_air` — air is not a breakable block at all. Minted before the
    `_NON_BLOCKS` denylist was added to `minerl_env._mine_counts`; the denylist
    stops NEW ones being minted but cannot remove the one already stored.
  * `discovered_N` — ungrounded signature-cluster goals from the junk-goal
    factory. Grounded-effect keying (`{"t": ["effect", X]}`) replaced this
    scheme, but slots minted under the old `{"t": ["disc", N]}` keying persist.

NOT fossils, and deliberately KEPT even though they are rare in a treechop
world: `break_sand`, `break_gravel`, `break_pumpkin`. These name real,
breakable blocks — a grounded effect the agent genuinely can produce. Rarity is
a reason for low competence, not a reason to delete earned learning.

NON-DESTRUCTIVE by construction:
  * every retired skill DIRECTORY is MOVED (never deleted) into
    `<bank>.retired_<stamp>/`, so it can be moved back;
  * `registry.json` is copied to `registry.json.pre_retire_<stamp>` first;
  * `--report-only` (the default) changes NOTHING.

Run the mutating form ONLY while the training run is STOPPED — the bank is
rewritten in place and a live run holds it open.

    python scripts/retire_fossil_skills.py --bank skill_bank_mc_curiosity
    python scripts/retire_fossil_skills.py --bank skill_bank_mc_curiosity --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Tuple

# A skill is a fossil iff its NAME half matches one of these.
FOSSIL_EXACT = {"break_air"}
FOSSIL_RE = re.compile(r"^discovered_\d+$")

# Kept on purpose — see the module docstring. Listed so the intent is explicit
# and a future reader does not "helpfully" add them to the fossil set.
KEEP_RARE_BUT_REAL = {"break_sand", "break_gravel", "break_pumpkin"}

_ACH_RE = re.compile(r"^ach_(\d+)_(.+)$")


def name_half(skill_id: str) -> str:
    """The goal-name half of an `ach_NN_<name>` id (the whole id otherwise)."""
    m = _ACH_RE.match(skill_id or "")
    return m.group(2) if m else (skill_id or "")


def is_fossil(skill_id: str) -> bool:
    n = name_half(skill_id)
    if n in KEEP_RARE_BUT_REAL:
        return False
    return n in FOSSIL_EXACT or bool(FOSSIL_RE.match(n))


def classify(skills: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    keep, drop = [], []
    for s in skills:
        (drop if is_fossil(s.get("skill_id", "")) else keep).append(s)
    return keep, drop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="skill bank directory")
    ap.add_argument("--apply", action="store_true",
                    help="actually retire (default is report-only)")
    ap.add_argument("--stamp", default="manual",
                    help="suffix for backup//retired dirs (pass a timestamp)")
    a = ap.parse_args()

    reg_path = os.path.join(a.bank, "registry.json")
    if not os.path.exists(reg_path):
        print(f"ERROR: no registry.json in {a.bank}", file=sys.stderr)
        return 2
    reg = json.load(open(reg_path))
    skills = reg.get("skills")
    if not isinstance(skills, list):
        print("ERROR: registry.json has no 'skills' list", file=sys.stderr)
        return 2

    keep, drop = classify(skills)
    print(f"bank         : {a.bank}")
    print(f"skills total : {len(skills)}")
    print(f"  KEEP       : {len(keep)}")
    print(f"  RETIRE     : {len(drop)}")
    for s in drop:
        print(f"     - {s['skill_id']:<40} "
              f"inv={s.get('invocations')} eps={s.get('total_episodes')} "
              f"succ={s.get('success_rate') or 0.0:.2f}")
    kept_rare = [s["skill_id"] for s in keep
                 if name_half(s["skill_id"]) in KEEP_RARE_BUT_REAL]
    if kept_rare:
        print("  kept on purpose (rare but REAL breakable blocks):")
        for k in kept_rare:
            print(f"     . {k}")

    # A fossil that was somehow invoked is NOT inert — refuse to touch it
    # rather than silently discarding real experience.
    invoked = [s["skill_id"] for s in drop if int(s.get("invocations") or 0) > 0]
    if invoked:
        print("\nREFUSING: these 'fossils' have invocations > 0 — they carry "
              "real experience, so the fossil rule is wrong for them:",
              file=sys.stderr)
        for i in invoked:
            print(f"   {i}", file=sys.stderr)
        return 3

    if not a.apply:
        print("\n(report-only — nothing changed; pass --apply to retire)")
        return 0
    if not drop:
        print("\nnothing to retire")
        return 0

    retired_dir = f"{a.bank.rstrip('/')}.retired_{a.stamp}"
    os.makedirs(retired_dir, exist_ok=True)
    shutil.copy2(reg_path, os.path.join(
        a.bank, f"registry.json.pre_retire_{a.stamp}"))

    moved = 0
    for s in drop:
        sid = s["skill_id"]
        src = os.path.join(a.bank, sid)
        if os.path.isdir(src):
            shutil.move(src, os.path.join(retired_dir, sid))
            moved += 1

    reg["skills"] = keep
    tmp = reg_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, reg_path)     # atomic: never leaves a truncated registry

    print(f"\nRETIRED {len(drop)} skills ({moved} directories moved)")
    print(f"  policies -> {retired_dir}")
    print(f"  registry backup -> registry.json.pre_retire_{a.stamp}")
    print(f"  bank now holds {len(keep)} skills")
    print("\nreversible: move the directories back and restore the backup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
