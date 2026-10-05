"""Farm-damping smoke (Mac-ok; stdlib only, no torch, no env).

    PYTHONPATH=. ./venv/bin/python tests/_farm_damping_smoke.py

THE INCIDENT (measured 2026-09-04)
----------------------------------
`infra/farm` reported a live reward farm — **2,213 laps in one cell at
+0.026/step**, with others at +0.092/step — and **nothing consumed it**. The
detector's docstring said so outright: "the detector only ranks, it never
intervenes."

Detection without response is the shape of every entry on the §9 scoreboard:
96% of drive to sky-staring, 77% of income to a villager trade menu, 96% of
option activity to holding attack against an unreachable trunk. Each was
visible in a log while it happened.

`loop_damp()` closes the loop: sustained positive loop income now costs the
farm its profitability. Under potential-based shaping the shaping terms
telescope to zero over any closed loop, so positive smoothed loop income is
the farm signature — that argument is the whole basis for damping on it.

THE THING THAT MUST NOT HAPPEN. This repo has produced a guard-becomes-latch
failure **nine** times (competence gate frozen at an inherited -5.68 → all 16
skills, 0 invocations, forever; the magnet's cold-start BUDGET latch → 19
hours of zero reward; `disable_macros: [10]` removing the only GUI exit →
10,149 consecutive steps in a trade menu). A damp that only ever tightens
would be the tenth. Contract 3 is therefore the most important test here.

CONTRACTS
  1. A quiet or briefly-visited cell is never damped (returns exactly 1.0).
  2. A paying loop IS damped, monotonically with lap count, never below the
     floor.
  3. ESCAPE PATH: staying away longer than revisit_horizon restores the
     multiplier toward 1.0, and repeating that fully restores it. The re-open
     condition is an action the AGENT can take.
  4. Damping is applied to SHAPING ONLY — real environment reward is never
     multiplied (asserted against the shipped loop source).
  5. Applied in BOTH stepping bodies (§4.2 duplicated-body drift) — to the
     magnet shaping AND (2026-10-04) to stream-0 intrinsic, since the magnet
     channel is dead on the live config.
  6. Never raises, and never returns a non-finite or out-of-range factor.
"""
import math
import re
from pathlib import Path

from developmental_ai.infra.ledger import FarmDetector

LOOP_SRC = (Path(__file__).resolve().parents[1] / "developmental_ai" /
            "core" / "developmental_loop.py").read_text()
STACK_SRC = (Path(__file__).resolve().parents[1] / "developmental_ai" /
             "infra" / "stack.py").read_text()

CELL, OTHER, PAY = 7, 99, 0.30


def _farm(det, laps, start=0, cell=CELL, pay=PAY, gap=10):
    """Drive `laps` closed loops through one cell, paying on each arrival."""
    s = start
    for _ in range(laps):
        det.step(cell, 1, pay, s)
        for j in range(1, gap):
            det.step(OTHER + j, 2, 0.0, s + j)
        s += gap
    return s


def test_quiet_cell_is_never_damped():
    d = FarmDetector()
    assert d.loop_damp(CELL) == 1.0, "unseen cell must be neutral"
    s = _farm(d, d.min_loops - 1)                 # just under the threshold
    assert d.loop_damp(CELL) == 1.0, "below min_loops must not be damped"
    # a heavily revisited cell that pays NOTHING is not a farm
    d2 = FarmDetector()
    _farm(d2, 40, pay=0.0)
    assert d2.loop_damp(CELL) == 1.0, "zero-income loop must not be damped"
    print(f"[farm-damp] 1. unseen / under-threshold / zero-income cells all "
          f"return 1.0 (laps used {d._cells[CELL][2]}, s={s})")


def test_a_paying_loop_is_damped_monotonically():
    d = FarmDetector()
    seen = []
    # Prime PAST min_loops before sampling: below the threshold the correct
    # answer is 1.0 (contract 1), so sampling there would only re-test that.
    s = _farm(d, d.min_loops + 2)
    for _ in range(12):
        seen.append(d.loop_damp(CELL))
        s = _farm(d, 4, start=s)
    assert seen[0] < 1.0, f"a paying loop must be damped, got {seen[0]}"
    assert all(b <= a + 1e-12 for a, b in zip(seen, seen[1:])), seen
    assert min(seen) >= d.damp_floor - 1e-12, (
        f"damp fell below the floor {d.damp_floor}: {min(seen)}. A hard zero "
        f"would also erase genuine repeated achievement, which this detector "
        f"cannot distinguish from a farm.")
    print(f"[farm-damp] 2. paying loop damped {seen[0]:.3f} -> {seen[-1]:.3f}, "
          f"monotone, floor {d.damp_floor} respected")


def test_escape_path_restores_value():
    """Contract 3 — the tenth guard-becomes-latch must not ship."""
    d = FarmDetector()
    s = _farm(d, 40)
    damped = d.loop_damp(CELL)
    assert damped < 1.0, damped

    # THE AGENT'S ESCAPE: stay away longer than one revisit horizon.
    s += d.revisit_horizon + 50
    d.step(CELL, 1, 0.0, s)
    after_one = d.loop_damp(CELL)
    assert after_one > damped, (
        f"leaving the farm must restore value: {damped:.3f} -> "
        f"{after_one:.3f}. If absence changes nothing, the damp is a latch "
        f"and only a human could reset it.")

    for _ in range(12):                       # keep leaving
        s += d.revisit_horizon + 50
        d.step(CELL, 1, 0.0, s)
    assert d.loop_damp(CELL) == 1.0, (
        f"repeated absence must FULLY restore the multiplier, got "
        f"{d.loop_damp(CELL)}")
    print(f"[farm-damp] 3. ESCAPE PATH: {damped:.3f} -> {after_one:.3f} after "
          f"one absence -> 1.0 after repeats (re-open condition: leave the "
          f"cell for > {d.revisit_horizon} steps)")


def test_shaping_only_never_env_reward():
    """Contract 4 — the scoreboard must stay honest."""
    assert "_sr = _sr * _damp" in LOOP_SRC, "damp not applied to shaping"
    for bad in ("rewards[0] = rewards[0] * _damp",
                "prim_extrinsic = (rewards[0] + _sr) * _damp",
                "prim_extrinsic * _damp"):
        assert bad not in LOOP_SRC, (
            f"env reward is being damped ({bad!r}) — a real block break in a "
            f"circled cell must still pay in full")
    assert "self.last_loop_damp = self.farm.loop_damp(" in STACK_SRC
    assert "self.last_loop_damp: float = 1.0" in STACK_SRC, (
        "last_loop_damp must exist from construction: on_step has an early "
        "return, so the first tick could read a missing attribute")
    print("[farm-damp] 4. shaping damped; rewards[0] untouched; stack "
          "publishes a neutral default from construction")


def test_applied_in_both_bodies():
    """Contract 5 — §4.2: an edit landing in one body only is silent."""
    n = LOOP_SRC.count("_sr = _sr * _damp")
    assert n == 2, (
        f"expected the damp in BOTH _run_episode_parallel and "
        f"_collect_segment, found {n}. _collect_segment is the one SkyBot "
        f"actually runs, so a single-site edit would look correct and do "
        f"nothing live.")
    # 2026-10-04: `_sr` is always 0 on the live config (vision off), so the
    # damp must ALSO reach stream-0 intrinsic — in both bodies, identically
    # (behaviour driven for real in _gui_farm_smoke contract F).
    m = LOOP_SRC.count("_ldi = self._loop_damp_factor()")
    k = LOOP_SRC.count("intrinsic[0] = intrinsic[0] * _ldi")
    assert m == 2 and k == 2, (
        f"intrinsic loop damp must be in BOTH bodies: factor {m}, apply {k}")
    print(f"[farm-damp] 5. damp present in {n}/2 stepping bodies (shaping) "
          f"and {k}/2 (stream-0 intrinsic)")


def test_never_raises_and_stays_in_range():
    d = FarmDetector()
    _farm(d, 20)
    for cell in (CELL, OTHER, -1, 0, 10 ** 9):
        v = d.loop_damp(cell)
        assert isinstance(v, float) and math.isfinite(v), (cell, v)
        assert d.damp_floor - 1e-12 <= v <= 1.0, (cell, v)
    assert d.loop_damp("not-an-int") == 1.0, "bad input must fail neutral"
    assert d.loop_damp(None) == 1.0
    # a non-finite reward must not poison the damp
    d.step(CELL, 1, float("inf"), 10 ** 6)
    v = d.loop_damp(CELL)
    assert math.isfinite(v) and d.damp_floor <= v <= 1.0, v
    # the detector's own report still works after damping was added
    assert isinstance(d.segment_report(), list)
    print("[farm-damp] 6. bad cells/inf reward fail NEUTRAL (1.0), factor "
          "always finite and in range")


def test_docstring_states_the_reopen_condition():
    """CLAUDE.md §4.1: 'you must state what re-opens it'."""
    doc = FarmDetector.loop_damp.__doc__ or ""
    assert "latch" in doc.lower() and "revisit_horizon" in doc, (
        "loop_damp must document its escape path by name")
    src = (Path(__file__).resolve().parents[1] / "developmental_ai" /
           "infra" / "ledger.py").read_text()
    assert re.search(r"ESCAPE PATH", src), "the decay branch must be labelled"
    print("[farm-damp] 7. re-open condition documented at both the decay "
          "branch and the accessor")


if __name__ == "__main__":
    for fn in (test_quiet_cell_is_never_damped,
               test_a_paying_loop_is_damped_monotonically,
               test_escape_path_restores_value,
               test_shaping_only_never_env_reward,
               test_applied_in_both_bodies,
               test_never_raises_and_stays_in_range,
               test_docstring_states_the_reopen_condition):
        fn()
    print("[farm-damp] ALL PASS")
