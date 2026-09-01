"""Proof-of-life lifecycle smoke (pure stdlib, no env needed).

Contracts:
  1. A registered subsystem is healthy inside 2x its expected cadence
     and OVERDUE strictly beyond it (grace factor 2).
  2. A subsystem that registers but NEVER beats is flagged once
     2*expected_every elapses since registration (the silent-inertness
     case this module exists for).
  3. Beating an unregistered name auto-registers it (visible in the
     table) with expected_every=0.
  4. expected_every=0 means "may be silent": listed, never overdue,
     no matter how much time passes.
  5. A raising invariant fn is contained: evaluate() returns an
     "ERRORED" line and never propagates the exception.
  6. Streaks increment on consecutive violations and reset to 0 on a
     pass.
"""
from developmental_ai.infra.lifecycle import Heartbeat, InvariantSet


def test_overdue_at_grace():
    hb = Heartbeat()
    hb.register("trainer", expected_every=100, step=0)
    hb.beat("trainer", step=50)
    # 200 steps of silence == exactly 2x cadence: still inside grace.
    lines, overdue = hb.report(step=250)
    assert overdue == [], overdue
    assert lines == ["trainer: last=50 (200 ago, every~100)"], lines
    # One step past 2x cadence: overdue, and the table line says so.
    lines, overdue = hb.report(step=251)
    assert overdue == ["trainer"], overdue
    assert lines == ["trainer: last=50 (201 ago, every~100) <-- OVERDUE"], \
        lines
    # A fresh beat clears the flag.
    hb.beat("trainer", step=260)
    _, overdue = hb.report(step=300)
    assert overdue == [], overdue


def test_never_fired():
    hb = Heartbeat()
    hb.register("dreamer", expected_every=10, step=100)
    # Inside grace since registration: listed, not yet overdue.
    lines, overdue = hb.report(step=120)
    assert overdue == [], overdue
    assert lines == ["dreamer: last=never (20 ago, every~10)"], lines
    # Past 2*expected_every since registration with zero beats: the
    # exact signature of a configured-but-dead subsystem.
    lines, overdue = hb.report(step=121)
    assert overdue == ["dreamer"], overdue
    assert lines == \
        ["dreamer: last=never (21 ago, every~10) <-- NEVER FIRED"], lines


def test_auto_register_on_beat():
    hb = Heartbeat()
    hb.beat("stray", step=7)               # never register()-ed
    lines, overdue = hb.report(step=9)
    assert overdue == [], overdue
    assert lines == ["stray: last=7 (2 ago, every~0)"], lines
    # Auto-registered names carry no cadence claim: silent forever is ok.
    _, overdue = hb.report(step=1_000_000)
    assert overdue == [], overdue


def test_zero_cadence_never_overdue():
    hb = Heartbeat()
    hb.register("rare_hook", expected_every=0, step=0)
    # Never beats AND expected_every=0: listed for visibility, never
    # flagged — 0 is the explicit "may be silent" contract.
    lines, overdue = hb.report(step=500_000)
    assert overdue == [], overdue
    assert len(lines) == 1 and lines[0].startswith("rare_hook: last=never"), \
        lines
    assert "OVERDUE" not in lines[0] and "NEVER FIRED" not in lines[0], lines


def test_raising_invariant_contained():
    inv = InvariantSet()
    inv.add("healthy", lambda ctx: None)
    inv.add("bomb", lambda ctx: 1 / 0)
    out = inv.evaluate({})                  # must NOT raise
    assert len(out) == 1 and out[0].startswith("INVARIANT bomb ERRORED:"), out
    assert "division" in out[0], out
    # A check that cannot run cannot vouch for its property: streaks up.
    assert inv.streak("bomb") == 1 and inv.streak("healthy") == 0


def test_streaks():
    inv = InvariantSet()
    inv.add("gate_open", lambda ctx: None if ctx["open"] else "gate closed")
    assert inv.streak("gate_open") == 0
    assert inv.evaluate({"open": False}) == \
        ["INVARIANT gate_open: gate closed"]
    assert inv.evaluate({"open": False}) == \
        ["INVARIANT gate_open: gate closed"]
    assert inv.streak("gate_open") == 2     # consecutive violations count up
    assert inv.evaluate({"open": True}) == []
    assert inv.streak("gate_open") == 0     # a pass wipes the streak
    inv.evaluate({"open": False})
    assert inv.streak("gate_open") == 1     # and it restarts from 1
    assert inv.streak("no_such_invariant") == 0   # unknown name: 0, no raise


if __name__ == "__main__":
    for n, (fn, contract) in enumerate((
        (test_overdue_at_grace,
         "healthy inside 2x cadence, OVERDUE strictly beyond it"),
        (test_never_fired,
         "never-fired flagged once 2*expected elapses from registration"),
        (test_auto_register_on_beat,
         "beat() on unknown name auto-registers with expected_every=0"),
        (test_zero_cadence_never_overdue,
         "expected_every=0 is listed but never overdue"),
        (test_raising_invariant_contained,
         "raising invariant fn contained as ERRORED line, never raises"),
        (test_streaks,
         "streaks increment on repeat violation, reset on pass"),
    ), start=1):
        print(f"[infra-lifecycle] {n}. {contract}")
        fn()
    print("[infra-lifecycle] ALL PASS")
