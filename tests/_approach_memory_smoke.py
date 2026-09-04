"""Approach-a-remembered-place smoke (Mac-ok; stdlib only, no torch).

    PYTHONPATH=. ./venv/bin/python tests/_approach_memory_smoke.py

WHY THIS EXISTS
---------------
`infra/episodic` has recorded sightings WITH COORDINATES
(`sighting:tree_visible @(14,13)`) since the infrastructure wave, and nothing
ever consumed them. The agent could therefore only pursue what was on screen:
look away and the tree stopped existing. Curiosity cannot close that gap by
itself — once a category is familiar, learning progress retires it, so the
agent stops seeking the very thing it just learned to recognise. That is a
large part of why 3,699 block breaks produced 9 logs and chopping a tree, the
original objective, has never been learned.

THE FAILURE THIS TEST GUARDS
----------------------------
CLAUDE.md 4.3. A potential written the textbook way, `F = w(gamma*Phi' - Phi)`,
pays `-w(1-gamma)*Phi` on EVERY step Phi is merely held. For a NEGATIVE
potential (distance-to-target is a cost) that flips sign and standing still
becomes a WAGE. It shipped once and the log showed it:
`Gaze level: +0.00014/step` — the agent was being paid to stare at the pitch
clamp. So this term uses the PLAIN DIFFERENCE (gamma=1), and contract 1 is the
one that matters: **stationary must pay exactly 0.0**, not 1e-9.

CONTRACTS
  1. Standing still pays EXACTLY 0.0, for many consecutive steps.
  2. Closing the distance pays positive; walking away pays negative and the
     round trip nets ~0 (no farm in either direction).
  3. Adoption is free: the first sight of a target, and any change of target
     identity, pays 0 — otherwise a newly-recorded nearer sighting would be a
     windfall for standing still.
  4. Teleports/respawns are capped, so dying cannot be cashed in.
  5. No memory in range, weight 0, or no position -> exactly 0.0, never raises.
  6. Episode boundaries re-adopt rather than carry a stale distance.
  7. Wired into BOTH stepping bodies, gated on gui_open, in prim_extrinsic
     (4.2 + 4.4).
"""
import math
from pathlib import Path

from developmental_ai.infra.stack import InfraStack

ROOT = Path(__file__).resolve().parents[1] / "developmental_ai"
LOOP_SRC = (ROOT / "core" / "developmental_loop.py").read_text()
W = 0.02


class _Mem:
    """Stand-in for EpisodicEventMemory.near() — records are dicts."""

    def __init__(self, recs=()):
        self.recs = list(recs)

    def near(self, position, radius, kind=None):
        qx, qz = position[0], position[2]
        return [r for r in self.recs
                if (kind is None or r["kind"] == kind)
                and r["position"] is not None
                and math.hypot(r["position"][0] - qx,
                               r["position"][2] - qz) <= radius]


def _rec(x, z, sub="tree_visible"):
    return {"kind": "sighting", "subtype": sub, "position": (x, 0.0, z)}


class _Stack:
    """Minimal carrier; binds the REAL methods so this tests shipped code."""

    def __init__(self, recs=(), weight=W):
        self.episodic = _Mem(recs)
        self.approach_weight = weight
        self.approach_kind = "sighting"
        self.approach_radius = 64.0
        self.approach_max_delta = 1.5
        self.last_approach_reward = 0.0
        self._appr_prev_d = None
        self._appr_target = None

    _approach_step = InfraStack._approach_step
    approach_reset = InfraStack.approach_reset


def test_standing_still_pays_exactly_zero():
    """THE contract. 4.3 — a wage for doing nothing is how this fails."""
    s = _Stack([_rec(20.0, 0.0)])
    s._approach_step((0.0, 0.0, 0.0))          # adopt
    paid = []
    for _ in range(200):                        # never moves
        s._approach_step((0.0, 0.0, 0.0))
        paid.append(s.last_approach_reward)
    assert all(p == 0.0 for p in paid), (
        f"standing still paid {sum(paid):+.9f} over 200 steps "
        f"(max single {max(map(abs, paid)):.3e}). A telescoping potential "
        f"structurally cannot discourage dwelling — it pays for it.")
    print(f"[approach] 1. 200 stationary steps paid EXACTLY 0.0 "
          f"(sum {sum(paid):.1f})")


def test_closing_pays_and_a_round_trip_nets_zero():
    s = _Stack([_rec(20.0, 0.0)])
    s._approach_step((0.0, 0.0, 0.0))          # adopt at d=20
    out = 0.0
    for x in range(1, 11):                      # walk toward it
        s._approach_step((float(x), 0.0, 0.0))
        out += s.last_approach_reward
    assert out > 0.0, f"closing 10 blocks must pay, got {out}"
    assert abs(out - W * 10.0) < 1e-9, (out, W * 10.0)
    back = 0.0
    for x in range(9, -1, -1):                  # walk back
        s._approach_step((float(x), 0.0, 0.0))
        back += s.last_approach_reward
    assert abs(out + back) < 1e-9, (
        f"round trip must net 0, got {out + back:+.9f} — otherwise pacing "
        f"toward and away from a remembered point is a farm")
    print(f"[approach] 2. closing 10 blocks paid {out:+.3f}; return "
          f"{back:+.3f}; round trip nets {out + back:+.1e}")


def test_adoption_and_target_switch_are_free():
    s = _Stack([_rec(20.0, 0.0)])
    s._approach_step((0.0, 0.0, 0.0))
    assert s.last_approach_reward == 0.0, "first adoption must not pay"
    # a NEARER sighting appears while the agent stands perfectly still
    s.episodic.recs.append(_rec(1.0, 0.0, sub="other"))
    s._approach_step((0.0, 0.0, 0.0))
    assert s.last_approach_reward == 0.0, (
        f"a newly-recorded nearer target paid {s.last_approach_reward:+.4f} "
        f"for standing still — target identity must be re-adopted, not cashed")
    print("[approach] 3. adoption free; target switch re-adopts and pays 0")


def test_teleport_is_capped():
    # The target must stay INSIDE approach_radius at both ends, or the cap is
    # never exercised and this test passes for the wrong reason (it did once).
    s = _Stack([_rec(60.0, 0.0)])
    s._approach_step((0.0, 0.0, 0.0))           # adopt at d=60 (< radius 64)
    assert s._appr_prev_d is not None, "precondition: a target was adopted"
    s._approach_step((55.0, 0.0, 0.0))          # jump 55 blocks: d 60 -> 5
    cap = W * s.approach_max_delta
    assert s.last_approach_reward > 0.0, "precondition: the cap path ran"
    assert abs(s.last_approach_reward - cap) < 1e-12, (
        f"teleport paid {s.last_approach_reward:+.5f}, expected the cap "
        f"{cap:.5f}; dying/respawning must not be cashable")
    assert s.last_approach_reward < W * 55.0, "uncapped would pay 55 blocks"
    print(f"[approach] 4. 55-block jump paid the cap "
          f"{s.last_approach_reward:+.5f} instead of {W * 55.0:+.3f} "
          f"(uncapped)")


def test_degenerate_inputs_pay_zero_and_never_raise():
    empty = _Stack([])
    empty._approach_step((0.0, 0.0, 0.0))
    assert empty.last_approach_reward == 0.0
    off = _Stack([_rec(5.0, 0.0)], weight=0.0)
    off._approach_step((0.0, 0.0, 0.0))
    off._approach_step((1.0, 0.0, 0.0))
    assert off.last_approach_reward == 0.0, "weight 0 must be a hard off"
    nopos = _Stack([_rec(5.0, 0.0)])
    nopos._approach_step(None)
    assert nopos.last_approach_reward == 0.0
    far = _Stack([_rec(10_000.0, 0.0)])         # outside radius
    far._approach_step((0.0, 0.0, 0.0))
    assert far.last_approach_reward == 0.0
    broken = _Stack([_rec(5.0, 0.0)])
    broken.episodic = object()                  # no .near at all
    broken._approach_step((0.0, 0.0, 0.0))
    assert broken.last_approach_reward == 0.0, "must fail neutral, not raise"
    print("[approach] 5. empty/off/no-position/out-of-range/broken-memory "
          "all pay exactly 0.0 without raising")


def test_reset_re_adopts():
    s = _Stack([_rec(20.0, 0.0)])
    s._approach_step((0.0, 0.0, 0.0))
    s._approach_step((5.0, 0.0, 0.0))
    assert s.last_approach_reward > 0.0
    s.approach_reset()
    assert s._appr_prev_d is None and s.last_approach_reward == 0.0
    s._approach_step((19.0, 0.0, 0.0))          # new world, near the target
    assert s.last_approach_reward == 0.0, (
        "the first step after a boundary must re-adopt, not collect a "
        "windfall for a move the agent never made")
    print("[approach] 6. episode boundary re-adopts (pays 0 on the next step)")


def test_wired_into_both_bodies_and_gated():
    n = LOOP_SRC.count("prim_extrinsic = prim_extrinsic + _ar")
    assert n == 2, (
        f"expected the approach term in BOTH stepping bodies, found {n}. "
        f"_collect_segment is the one SkyBot actually runs (4.2).")
    assert "if not _gui_now2 and self.infra is not None:" in LOOP_SRC, (
        "must not pay while a GUI is open — inside a menu the agent cannot "
        "walk, so anything paid there is paying for blindness (4.4)")
    assert LOOP_SRC.count("self.infra.approach_reset()") >= 2, \
        "the reference must be re-adopted at episode boundaries"
    # it must land in prim_extrinsic, NOT intrinsic (which is zeroed on GUI)
    assert "intrinsic[0] = intrinsic[0] + _ar" not in LOOP_SRC
    print(f"[approach] 7. wired into {n}/2 bodies, gui-gated, in "
          f"prim_extrinsic, reset at boundaries")


if __name__ == "__main__":
    for fn in (test_standing_still_pays_exactly_zero,
               test_closing_pays_and_a_round_trip_nets_zero,
               test_adoption_and_target_switch_are_free,
               test_teleport_is_capped,
               test_degenerate_inputs_pay_zero_and_never_raise,
               test_reset_re_adopts,
               test_wired_into_both_bodies_and_gated):
        fn()
    print("[approach] ALL PASS")
