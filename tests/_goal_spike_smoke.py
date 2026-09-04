"""Goal-discovery spike-threshold smoke (Mac-ok; numpy only).

    PYTHONPATH=. ./venv/bin/python tests/_goal_spike_smoke.py

THE MEASURED STALL (2026-09-04)
-------------------------------
    Goals known:   3 active / 0 dormant (total known 3)
    Working set:   paged out=0, recalled=0

Three goals, for an entire run, on a system whose whole claim is open-ended
self-directed development. The working-set pager — built for UNBOUNDED goals —
had never had anything to page.

CAUSE. `DiscoveredAchievementGoals.update()` treated a step as an unlock event
only when `env.reward_history[-1] >= spike_threshold`, with spike_threshold
0.9. That number comes from achievement environments, where unlocking
something literally pays 1.0. Minecraft does not pay in those units: extrinsic
env reward appeared in roughly 1 segment in 39, and 3,699 block breaks
produced 9 logs. The test was, in practice, never true — so no new goal could
ever be discovered, no matter what the agent did. A curriculum of three,
forever.

THE FIX. The bar is now a fraction of the largest positive reward actually
seen (decaying running max), so "spike" means "unusually large FOR THIS
ENVIRONMENT". In an achievement env rmax converges to 1.0 and the bar returns
to ~0.5 — a 1.0 unlock still spikes, noise still does not — so that arm is
behaviourally unchanged. Same absolute->relative correction as
EmpowermentPotential, for the same reason: a hard constant cannot serve a
system that claims to transfer across games.

WHY THIS IS NOT THE JUNK-GOAL FACTORY RETURNING. That incident (48/48
ungrounded slots, 0/51 mastered) was NOT caused by this threshold and is not
fixed by it. Its fix was the REPEATABILITY GATE — an effect must recur across
distinct episodes before it earns a skill, because "one spike is evidence
something happened, not evidence of a capability". That gate is untouched and
contract 5 asserts so. Lowering this bar creates goal CANDIDATES; earning a
skill still costs the same repeated evidence.

CONTRACTS
  1. The old absolute mode is preserved exactly (a true revert path).
  2. Relative mode fires on a small reward that is large FOR ITS SCALE — the
     Minecraft case that was impossible before.
  3. Relative mode still rejects noise, and an achievement env is unchanged:
     1.0 spikes, 0.01 chatter does not.
  4. No latch: the bar DECAYS, so one freak reward cannot pin it forever.
  5. The repeatability gate that actually stopped junk goals is still there.
  6. Non-positive / non-finite rewards are never spikes.
"""
import math
from pathlib import Path

from developmental_ai.core.achievement_goals import (
    DiscoveredAchievementGoals)

SRC = (Path(__file__).resolve().parents[1] / "developmental_ai" / "core" /
       "achievement_goals.py").read_text()


def _mk(**kw):
    return DiscoveredAchievementGoals(max_slots=8, **kw)


def test_absolute_mode_is_preserved():
    """Contract 1 — `spike_mode: absolute` must be a true revert."""
    g = _mk(spike_mode="absolute", spike_threshold=0.9)
    assert g._is_spike(0.9) and g._is_spike(1.0)
    assert not g._is_spike(0.89)
    assert not g._is_spike(0.05)
    print("[goal-spike] 1. absolute mode unchanged: >=0.9 spikes, 0.89 does "
          "not")


def test_relative_fires_on_a_small_but_unusual_reward():
    """Contract 2 — the Minecraft case that was structurally impossible."""
    g = _mk()
    old = _mk(spike_mode="absolute", spike_threshold=0.9)
    rewards = [0.0] * 50 + [0.05] + [0.0] * 50      # a break pays 0.05
    fired_new = [r for r in rewards if g._is_spike(r)]
    fired_old = [r for r in rewards if old._is_spike(r)]
    assert fired_old == [], (
        "precondition: under the old rule this run discovers NOTHING")
    assert fired_new == [0.05], (
        f"a reward 0.05 in a world whose max is 0.05 must register as an "
        f"event; got {fired_new}")
    print(f"[goal-spike] 2. reward 0.05 -> spike under relative mode, "
          f"{len(fired_old)} spikes under the old >=0.9 rule")


def test_relative_rejects_noise_and_leaves_achievement_envs_alone():
    """Contract 3 — a lower bar must not become NO bar."""
    g = _mk()
    for _ in range(200):                            # an achievement env
        g._is_spike(1.0)
    assert g._is_spike(1.0), "a real unlock must still spike"
    assert not g._is_spike(0.01), "chatter at 1% of scale must NOT spike"
    assert not g._is_spike(0.4), "below half the scale must not spike"
    assert g._is_spike(0.6), "above half the scale should spike"
    print(f"[goal-spike] 3. with rmax={g._rmax:.2f}: 1.0 and 0.6 spike; "
          f"0.4 and 0.01 do not")


def test_no_latch_the_bar_decays():
    """Contract 4 — the failure this repo has hit nine times."""
    g = _mk(spike_decay=0.99)
    g._is_spike(100.0)                              # one freak reward
    assert not g._is_spike(0.05), "precondition: the bar is high now"
    for _ in range(2000):                           # the world's real scale
        g._is_spike(0.05)
    assert g._is_spike(0.05), (
        f"the bar never came down (rmax={g._rmax:.4f}) — one outlier would "
        f"freeze goal discovery for the rest of the run, exactly like the "
        f"competence gate frozen at an inherited -5.68")
    print(f"[goal-spike] 4. one 100.0 outlier decayed away; rmax now "
          f"{g._rmax:.4f} and 0.05 spikes again")


def test_the_repeatability_gate_is_untouched():
    """Contract 5 — what ACTUALLY stopped the junk-goal factory."""
    g = _mk()
    assert g.mint_min_repeats >= 2, (
        f"mint_min_repeats={g.mint_min_repeats}: one spike would mint a "
        f"skill again. This — not the spike threshold — is what produced "
        f"48/48 ungrounded slots and 0/51 mastered.")
    assert "DISTINCT EPISODES" in SRC
    assert "is_fossil_slot_name" in SRC, "the fossil-name mint guard is gone"
    print(f"[goal-spike] 5. repeatability gate intact "
          f"(mint_min_repeats={g.mint_min_repeats}, "
          f"window={g.mint_window_eps} eps) + fossil-name guard present")


def test_non_positive_and_non_finite_are_never_spikes():
    g = _mk()
    for bad in (0.0, -1.0, -0.0, float("nan"), float("inf"), float("-inf")):
        assert not g._is_spike(bad), bad
    assert g._rmax == 0.0, "a bad reward must not move the bar"
    g._is_spike(1.0)
    assert math.isfinite(g._rmax) and g._rmax > 0.0
    # persistence: a restart must not reset the bar to 0, or the first
    # positive reward after boot is a "spike" by definition
    st = g._extra_state()
    assert "rmax" in st and st["rmax"] > 0.0
    g2 = _mk()
    g2._load_extra_state(st)
    assert abs(g2._rmax - g._rmax) < 1e-9
    print(f"[goal-spike] 6. 0/negative/NaN/inf never spike; rmax persists "
          f"across restart ({st['rmax']:.3f})")


if __name__ == "__main__":
    for fn in (test_absolute_mode_is_preserved,
               test_relative_fires_on_a_small_but_unusual_reward,
               test_relative_rejects_noise_and_leaves_achievement_envs_alone,
               test_no_latch_the_bar_decays,
               test_the_repeatability_gate_is_untouched,
               test_non_positive_and_non_finite_are_never_spikes):
        fn()
    print("[goal-spike] ALL PASS")
