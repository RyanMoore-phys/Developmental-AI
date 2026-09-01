"""Behavioural preference suite (Mac-ok; heavy import, no env boot).

WHY THIS SUITE EXISTS: 13 mechanism smokes were all green while the
assembled agent stood on a pillar doing nothing. Mechanism tests assert
that each part computes what it should; nothing asserted what the SUM of
the parts would make the agent PREFER. This suite scores whole canned
trajectories through the real reward primitives (via tools/reward_replay)
and asserts the preference ORDER of the shipped economy — so the next
economy tweak that quietly re-funds a degenerate behaviour fails a test in
seconds instead of costing a live run.

Contracts:
  1. Under the shipped economy (log_break_reward=20, whitelist,
     habituation 50) the chop-a-log trajectory out-scores dirt-farming,
     sky-staring and pillar-building by >= 3x each.
  2. The dirt farm is genuinely defunded: its whole-trajectory total is
     below 1.0 (whitelist pays 0 extrinsic, habituation crushes the
     intrinsic at 400+ lifetime breaks).
  3. Malformed steps are contained as data ("errors"), never exceptions,
     and the remaining steps still score — replay is monitoring code and
     must not be able to crash whatever drives it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.reward_replay import score_trajectory  # noqa: E402


# --------------------------------------------------------------------------
# Canned trajectories, ~60 steps each. Counts are LIFETIME-BEFORE-STEP, so
# each one pins the familiarity level at which the behaviour is observed
# live: the farm at 400+ breaks, the pillar at 60+ placements, the chop at
# its second-ever log.
# --------------------------------------------------------------------------

def _chop_log():
    """Walk with the trunk growing in the fovea, then fell one oak_log."""
    steps = []
    for i in range(50):
        steps.append({"fovea": {"tree_visible": 0.2 + 0.014 * i},
                      "pitch": 5.0, "new_view": (i % 2 == 0)})
    steps.append({"events": [("break", "oak_log")],
                  "counts": {"break:oak_log": 1},
                  "fovea": {"tree_visible": 0.9}, "pitch": 10.0,
                  "new_view": True})
    for i in range(9):
        steps.append({"fovea": {"tree_visible": 0.6}, "pitch": 5.0,
                      "new_view": (i % 2 == 0)})
    return steps


def _dirt_farm():
    """The measured stall: grinding the ground at its feet, 400+ lifetime."""
    return [dict({"events": [("break", "dirt")], "pitch": 70.0,
                  "new_view": True},
                 **({"counts": {"break:dirt": 400}} if i == 0 else {}))
            for i in range(60)]


def _sky_stare():
    """Clouds drift forever: every step a 'new view', nothing ever caused."""
    return [{"pitch": -85.0, "new_view": True,
             "fovea": {"sky_visible": 0.95}} for _ in range(60)]


def _pillar():
    """The green-smokes failure mode itself: stacking the 60th+ dirt while
    staring up its own tower."""
    return [dict({"events": [("place", "dirt")], "pitch": -70.0,
                  "new_view": True},
                 **({"counts": {"place:dirt": 60}} if i == 0 else {}))
            for i in range(60)]


_SHIPPED = dict(habituation_scale=50.0, log_break_reward=20.0,
                break_decay_scale=25.0)


def _score_all():
    return {name: score_trajectory(traj, **_SHIPPED)
            for name, traj in (("chop_log", _chop_log()),
                               ("dirt_farm", _dirt_farm()),
                               ("sky_stare", _sky_stare()),
                               ("pillar", _pillar()))}


def test_preference_order():
    scores = _score_all()
    header = f"    {'trajectory':<12} {'extrinsic':>10} {'intrinsic':>10} " \
             f"{'total':>10}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name, r in scores.items():
        print(f"    {name:<12} {r['extrinsic']:>10.3f} "
              f"{r['intrinsic_proxy']:>10.3f} {r['total']:>10.3f}")
        assert not r["errors"], (name, r["errors"])
    chop = scores["chop_log"]["total"]
    for rival in ("dirt_farm", "sky_stare", "pillar"):
        other = scores[rival]["total"]
        assert chop >= 3.0 * other, (
            f"economy regression: chop_log ({chop:.3f}) does not out-pay "
            f"{rival} ({other:.3f}) by 3x")
    print("[behavior-suite] 1. chop_log out-pays dirt_farm/sky_stare/pillar"
          " by >= 3x under the shipped economy")


def test_dirt_farm_defunded():
    total = score_trajectory(_dirt_farm(), **_SHIPPED)["total"]
    assert total < 1.0, (
        f"dirt farm re-funded: 60 breaks at 400+ lifetime pay {total:.3f}")
    print(f"[behavior-suite] 2. dirt_farm defunded: 60-step total "
          f"{total:.3f} < 1.0")


def test_errors_are_data():
    steps = [
        {"events": 42},                       # not iterable as pairs
        None,                                 # not even a dict
        {"events": [("break", "oak_log")],    # a healthy step AFTER the junk
         "counts": {"break:oak_log": 0}},
    ]
    r = score_trajectory(steps, **_SHIPPED)   # must not raise
    assert len(r["errors"]) == 2, r["errors"]
    assert {e["i"] for e in r["errors"]} == {0, 1}, r["errors"]
    assert len(r["per_step"]) == 3, len(r["per_step"])
    # the healthy step still scored through the real economy
    assert r["per_step"][2]["extrinsic"] > 0.0, r["per_step"][2]
    print("[behavior-suite] 3. malformed steps contained as errors-as-data;"
          " later steps still score")


if __name__ == "__main__":
    for fn in (test_preference_order, test_dirt_farm_defunded,
               test_errors_are_data):
        fn()
    print("[behavior-suite] ALL PASS")
