"""Smoke: stop manufacturing junk goals/skills (2026-07-25).

Live finding: ALL 48 goal slots were ungrounded `discovered_N`, and 22 of 51
skills shared two names (14x act_where_breakable_in_reach, 8x
act_where_dirt_visible). Nothing ever reached mastery (0/51), because one
behaviour's practice was split across many slots — and the goal frontier
scored every candidate identically (2.7952), making goal selection carry no
information at all.

Five independent producers, all fixed here:
  1. NON-BLOCK junk: mine_block reports `air`/fluids; these minted `break_air`
     AND paid a first-break bonus.
  2. TIER SUMMATION: six 0.15 plants summed to 0.90 and crossed the
     goal-discovery spike threshold the tiering existed to stay under.
  3. LOG-PICKUP -> UNGROUNDED: the pickup guard passed effect_key=None, so
     every novel viewpoint of picking up a log minted a fresh `discovered_N`.
     The core task reward was the dominant junk producer.
  4. ALPHABETICAL NAMING: sorted(preds)[0] collapsed distinct behaviours onto
     whichever predicate sorts first.
  5. WEAK-NAME LATCH: `act_where_*` counted as "grounded", so a real
     `break_oak_log` could never replace a wrong fallback name.

Run: PYTHONPATH=. python tests/_junk_goal_fix_smoke.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from developmental_ai.environments.minerl_env import MineRLEnvAdapter
from developmental_ai.skill_bank.skill_bank import (
    is_grounded_name, is_placeholder_name, is_weakly_grounded_name)


def main() -> None:
    # ---- 1. NON-BLOCK filter: air/fluids never become achievements ----
    obs = {"mine_block": {
        "air": 12, "cave_air": 3, "water": 5, "flowing_lava": 2,
        "minecraft.air": 7,                 # namespaced form
        "oak_log": 4, "dirt": 9}}
    counts = MineRLEnvAdapter._mine_counts(obs)
    for junk in ("air", "cave_air", "water", "flowing_lava", "minecraft.air"):
        assert junk not in counts, f"{junk} still counts as a break -> break_air junk"
    assert counts.get("oak_log") == 4 and counts.get("dirt") == 9, counts

    # ---- 2. TIER SUMMATION: many trivial breaks must stay UNDER the 0.9
    #      goal-discovery spike threshold (max, not sum) ----
    src = open("developmental_ai/environments/minerl_env.py").read()
    # Assert the INTENT (max, never sum), not one exact spelling. The
    # expression gained a familiarity-decay multiplier on 2026-08-06
    # (tier x 1/(1+n/scale)); pinning the literal string made this fail on a
    # change that preserves the property it guards.
    import re as _re
    _m = _re.search(r"total_reward \+= (max|sum)\(self\._break_reward\(b\)"
                    r"[^)]*(?:\)[^)]*)*? for b in _new_breaks\)", src)
    assert _m is not None, (
        "the first-break payment no longer matches the expected shape — "
        "check it still aggregates over _new_breaks")
    assert _m.group(1) == "max", (
        "first-break bonus is SUMMED — six trivial plants at 0.15 sum to 0.90 "
        "and cross the goal-discovery spike threshold, minting junk goals")
    # decay may scale each term, but must never turn max into sum
    assert "sum(self._break_reward" not in src
    a = object.__new__(MineRLEnvAdapter)
    trivial = ["grass", "fern", "tall_grass", "poppy", "dandelion", "vine"]
    summed = sum(a._break_reward(b) for b in trivial)
    maxed = max(a._break_reward(b) for b in trivial)
    assert summed >= 0.9, f"test premise broken: {summed}"   # the old bug
    assert maxed < 0.9, (
        f"max over trivial breaks is {maxed} — must stay under the 0.9 spike "
        f"threshold or plants still mint goals")

    # ---- 3. LOG PICKUP gets a STABLE grounded key (not None) ----
    loop = open("developmental_ai/core/developmental_loop.py").read()
    assert '_best = "log_pickup"' in loop, \
        "log pickups still pass None -> ungrounded per-viewpoint junk goals"
    assert loop.count('_best = "log_pickup"') == 2, \
        "only one of the two pickup-guard sites was fixed"

    # ---- 4. NAMING: specificity + uniqueness, not alphabet ----
    assert "sorted(true_preds)[0]" not in loop, \
        "still naming by the ALPHABETICALLY first predicate"
    assert "min(true_preds, key=lambda p: (counts.get(p, 0), p))" in loop, \
        "fallback name is not ranked by predicate rarity"
    assert "_s{slot:02d}" in loop, \
        "fallback names lack the slot suffix -> two behaviours can collide"

    # ---- 5. WEAK-NAME LATCH: break_* may replace act_where_* ----
    assert is_weakly_grounded_name("act_where_dirt_visible_s07")
    assert is_weakly_grounded_name("skill_slot_03")
    assert not is_weakly_grounded_name("break_oak_log")
    assert is_grounded_name("break_oak_log")
    bank_src = open("developmental_ai/skill_bank/skill_bank.py").read()
    assert "is_weakly_grounded_name(existing.name)" in bank_src, \
        "a real break_* name still cannot replace a weak fallback name"
    # placeholders remain replaceable; real names are not placeholders
    assert is_placeholder_name("Achieve: discovered_3")
    assert not is_placeholder_name("break_oak_log")

    print("[junk-goal-fix-smoke] ALL PASS: non-block filter (air/fluids "
          f"dropped), tiers MAXed ({maxed:.2f} < 0.9, was {summed:.2f}), "
          "pickups keyed 'log_pickup' at both sites, names ranked by rarity "
          "+ slot-unique, weak names no longer latch")


if __name__ == "__main__":
    main()
