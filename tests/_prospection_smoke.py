"""Smoke: goal-level prospection re-ranking in AchievementGoalBroadcast.

Contracts (selection-level, no torch / no env):
  1. hook=None / weight=0  -> byte-identical frontier argmax (and the hook is
     never called when the weight gate is closed)
  2. hook promoting a shortlist candidate with weight=1 -> selection flips to
     it, and last_selection records candidates/frontier/prospective/chosen
  3. hook raising / returning {} / returning equal scores -> baseline choice
  4. a slot OUTSIDE the frontier's top-M can never be promoted
Run: PYTHONPATH=. python tests/_prospection_smoke.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from developmental_ai.core.achievement_goals import make_goal_broadcast


def fresh_broadcaster():
    b = make_goal_broadcast({"mode": "given", "max_slots": 8})
    names = ["g0", "g1", "g2", "g3", "g4"]
    b.slot_names = list(names)
    b.slot_of = {n: i for i, n in enumerate(names)}
    b._free_slots = []          # falsy -> no occupancy masking path
    b._attempts[:5] = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    b.epsilon = 0.0             # deterministic
    b.dag_prior = {}
    b._frontier_bonus = {}
    # frontier score = (1-p) + 1/sqrt(1+attempts): make slot ranks distinct
    # p: g0 hardest-unknown (highest frontier), then g1, g2, g3, g4
    b.competence.predict_all = lambda: np.array(
        [0.10, 0.20, 0.30, 0.40, 0.95, 0, 0, 0], dtype=np.float64)
    return b


def main() -> None:
    calls = []

    # 1. baseline (no hook): deterministic frontier argmax = slot 0
    b = fresh_broadcaster()
    base = b._select_target()
    assert base == 0, base

    # weight gate closed -> hook never called
    b = fresh_broadcaster()
    b.prospection_hook = lambda cand: calls.append(cand) or {}
    b.prospection_weight = 0.0
    assert b._select_target() == base and calls == []

    # 2. hook promotes candidate slot 2 with weight 1.0 -> selection flips
    b = fresh_broadcaster()
    b.prospection_weight = 1.0
    b.prospection_top_m = 4

    def hook(cand):
        calls.append(list(cand))
        return {g: (100.0 if g == 2 else 0.0) for g in cand}

    b.prospection_hook = hook
    chosen = b._select_target()
    assert chosen == 2, chosen
    assert calls and 0 in calls[-1] and 2 in calls[-1], calls
    ls = b.last_selection
    assert ls["chosen"] == 2 and 2 in ls["candidates"]
    assert len(ls["frontier"]) == len(ls["candidates"]) == len(
        ls["prospective"])

    # blended (weight=0.5): prospection must overcome half the frontier gap —
    # a huge prospective margin still flips the choice
    b = fresh_broadcaster()
    b.prospection_weight = 0.5
    b.prospection_top_m = 4
    b.prospection_hook = hook
    assert b._select_target() == 2

    # 3a. hook raising -> baseline
    b = fresh_broadcaster()
    b.prospection_weight = 1.0
    b.prospection_hook = lambda cand: 1 / 0
    assert b._select_target() == base

    # 3b. hook returning {} -> baseline
    b = fresh_broadcaster()
    b.prospection_weight = 1.0
    b.prospection_hook = lambda cand: {}
    assert b._select_target() == base

    # 3c. equal prospective scores -> frontier order preserved
    b = fresh_broadcaster()
    b.prospection_weight = 1.0
    b.prospection_hook = lambda cand: {g: 7.7 for g in cand}
    assert b._select_target() == base

    # 4. a slot outside top-M can't be promoted: top_m=2 shortlists {0,1};
    #    a rogue hook scoring slot 4 sky-high must not elect slot 4
    b = fresh_broadcaster()
    b.prospection_weight = 1.0
    b.prospection_top_m = 2
    b.prospection_hook = lambda cand: {4: 1e9, **{g: 0.0 for g in cand}}
    chosen = b._select_target()
    assert chosen in (0, 1), chosen

    # 5. FLAT frontier (all candidates tied) + distinct prospective scores ->
    #    prospection alone decides (the span-floor fix; pre-fix the blend
    #    collapsed under selection jitter and this assert fails)
    b = fresh_broadcaster()
    b.competence.predict_all = lambda: np.array(
        [0.30, 0.30, 0.30, 0.30, 0.30, 0, 0, 0], dtype=np.float64)
    b.prospection_weight = 0.5
    b.prospection_top_m = 4
    b.prospection_hook = lambda cand: {
        g: (9.0 if g == 3 else 1.0) for g in cand}
    chosen = b._select_target()
    assert chosen == 3, chosen

    # 6. stale-record clear: a later hook-less selection must null the record
    b.prospection_hook = None
    b._select_target()
    assert b.last_selection is None

    print("[prospection-smoke] ALL PASS: gate, flip, blend, exception/empty/"
          "degenerate fallbacks, shortlist containment, flat-frontier "
          "tie-break, stale-record clear")


if __name__ == "__main__":
    main()
