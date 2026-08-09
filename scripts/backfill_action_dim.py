"""Backfill `action_dim` on legacy skills before the 10->12 action widening.

WHY THIS MUST RUN BEFORE THE WIDENED BUILD BINDS THE BANK: three of the ten
skills carry `action_dim: null`, and every stored head is meta-width (34,256),
so a skill's own primitive count is NOT inferable from its tensor. Under the
new P_old-aware truncation, an unrecorded `action_dim` with a 34-row head in a
12-primitive world is a HARD SlotRefused — better than the silent
option-slot-as-primitive corruption it replaces, but it would still gate three
real skills out of the bank.

The backfill is safe precisely because of WHEN it runs: every skill currently
in the bank was minted in the 10-action era — the widening does not exist on
the pod yet. Writing `action_dim=10` records a historical fact, not a guess.
(Inferring `rows - k_slots` at bind time remains forbidden: k_slots is a
config knob that has changed across eras. This script encodes era knowledge
instead, once, explicitly, with a backup.)

    python scripts/backfill_action_dim.py --bank skill_bank_mc_curiosity          # report
    python scripts/backfill_action_dim.py --bank skill_bank_mc_curiosity --apply
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--era-dim", type=int, default=10,
                    help="primitive count of the era ALL current skills were "
                         "minted in (10 = pre-crafting TREECHOP_MACROS)")
    a = ap.parse_args()

    reg_path = os.path.join(a.bank, "registry.json")
    if not os.path.exists(reg_path):
        print(f"ERROR: no registry at {reg_path}", file=sys.stderr)
        return 2
    reg = json.load(open(reg_path))
    skills = reg.get("skills", [])

    nulls = [s for s in skills if s.get("action_dim") is None]
    wrong = [s for s in skills
             if s.get("action_dim") not in (None, a.era_dim)]
    print(f"bank: {a.bank} | skills: {len(skills)} | "
          f"action_dim null: {len(nulls)} | other values: {len(wrong)}")
    for s_ in nulls:
        print(f"  null -> {a.era_dim}: {s_['skill_id']}")
    if wrong:
        # a skill already recording a DIFFERENT era must not be overwritten —
        # that would falsify history, the exact thing this script exists to
        # record faithfully.
        print("  leaving untouched (recorded era differs):")
        for s_ in wrong:
            print(f"    {s_['skill_id']}: action_dim={s_['action_dim']}")

    if not a.apply:
        print("(report-only — pass --apply to write)")
        return 0
    if not nulls:
        print("nothing to backfill")
        return 0

    shutil.copy2(reg_path, reg_path + ".pre_backfill")
    for s_ in nulls:
        s_["action_dim"] = int(a.era_dim)
    tmp = reg_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, reg_path)
    print(f"backfilled {len(nulls)} skills -> action_dim={a.era_dim} "
          f"(backup: registry.json.pre_backfill)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
