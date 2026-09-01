"""AffordanceMap smoke — the unwilling-vs-unable instrument.

Contracts:
  1. observe credits uses + effects; effects_of/producers answer at both
     granularities (bare kind and kind:subtype); unknowns return empty.
  2. actions_without_effect respects min_uses and clears on any effect.
  3. unproduced is silent before min_steps and specific after, formatted
     "<effect>: NEVER PRODUCED in <now> steps".
  4. report ranks by use count, formats effect profiles, and names
     zero-effect actions on their own line.
  5. Malformed input never raises; errors surface as data; verdicts survive.
"""
from developmental_ai.infra.affordance import AffordanceMap


def test_observe_effects_producers():
    m = AffordanceMap()
    m.observe(3, [("break", "x"), ("break", "y")], step=10)
    m.observe(3, [("break", "x")], step=20)
    m.observe(5, [("place", "y")], step=30)
    m.observe(5, [("break", "z")], step=40)
    m.observe(7, [], step=50)
    assert m.effects_of(3) == {"break:x": 2, "break:y": 1}, m.effects_of(3)
    assert m.effects_of(5) == {"place:y": 1, "break:z": 1}, m.effects_of(5)
    assert m.effects_of(7) == {}, "used-but-effectless action must read {}"
    assert m.effects_of(99) == {}, "never-used action must read {}"
    # effects_of returns a COPY: mutating it must not corrupt the evidence
    m.effects_of(3)["break:x"] = 999
    assert m.effects_of(3)["break:x"] == 2
    # bare-kind query unions subtypes; full-key query stays specific
    assert m.producers("break") == [3, 5], m.producers("break")
    assert m.producers("break:x") == [3], m.producers("break:x")
    assert m.producers("break:z") == [5], m.producers("break:z")
    assert m.producers("place") == [5] and m.producers("place:y") == [5]
    assert m.producers("craft") == [] and m.producers("break:q") == []


def test_actions_without_effect_threshold():
    m = AffordanceMap()
    for i in range(500):
        m.observe(1, [], i)                       # exactly at the default bar
    for i in range(499):
        m.observe(2, [], i)                       # one short of it
    for i in range(600):                          # heavy use, ONE effect ever
        m.observe(3, [("hit", "t")] if i == 0 else [], i)
    assert m.actions_without_effect() == [1], m.actions_without_effect()
    assert m.actions_without_effect(min_uses=499) == [1, 2]
    assert m.actions_without_effect(min_uses=1000) == []
    assert 3 not in m.actions_without_effect(min_uses=1), \
        "a single lifetime effect must clear the no-op verdict"


def test_unproduced_silence_then_alarm():
    m = AffordanceMap()
    m.observe(0, [("break", "x")], 5)
    req = ["break", "acquire:tool_a", "break:y"]
    # too early: absence is not yet evidence
    assert m.unproduced(req, min_steps=20000, now=100) == []
    assert m.unproduced(req) == [], "default now=0 must stay silent"
    out = m.unproduced(req, min_steps=20000, now=25000)
    assert out == ["acquire:tool_a: NEVER PRODUCED in 25000 steps",
                   "break:y: NEVER PRODUCED in 25000 steps"], out
    # bare kind is satisfied by ANY subtype; the specific subtype still alarms
    assert not any(s.startswith("break:") and ":y" not in s for s in out)
    # producing the specific effect retires its alarm
    m.observe(0, [("break", "y")], 30)
    assert m.unproduced(req, min_steps=20000, now=25000) == [
        "acquire:tool_a: NEVER PRODUCED in 25000 steps"]


def test_report_format():
    m = AffordanceMap()
    for i in range(510):
        m.observe(4, [], i)                       # busiest AND effectless
    for i in range(20):
        m.observe(1, [("break", "x")], i)
    m.observe(2, [("place", "y"), ("place", "y"), ("break", "x")], 30)
    lines = m.report(top=8)
    assert lines[0] == "action 4: 510 uses -> {}", lines[0]
    assert lines[1] == "action 1: 20 uses -> {break:x 20}", lines[1]
    assert lines[2] == "action 2: 1 uses -> {place:y 2, break:x 1}", lines[2]
    assert lines[3] == "no observed effect: actions [4]", lines[3]
    assert len(lines) == 4, lines
    # top truncates by use count
    short = m.report(top=1)
    assert short[0].startswith("action 4:") and \
        short[1].startswith("no observed effect"), short
    # empty map: report must not crash and must not invent lines
    assert AffordanceMap().report() == []


def test_garbage_never_raises():
    m = AffordanceMap()
    m.observe("not-an-int", [("a", "b")], 0)      # type: ignore[arg-type]
    m.observe(1, [None, ("k",), 42, ("ok", "sub"), ("x", "y", "z")], 5)  # type: ignore[list-item]
    m.observe(1, "junk", "weird-step")            # type: ignore[arg-type]
    m.observe(1, None, 7)                         # type: ignore[arg-type]
    # the good event among the garbage was kept; garbage was tallied as data
    assert m.effects_of(1) == {"ok:sub": 1}, m.effects_of(1)
    assert m._uses[1] == 3, "well-formed uses must still be counted"
    assert m.errors >= 4 and m.last_error is not None, (m.errors, m.last_error)
    # queries with junk arguments degrade to empty, never raise
    assert m.effects_of(None) == {}               # type: ignore[arg-type]
    assert m.producers(None) == []                # type: ignore[arg-type]
    assert m.actions_without_effect(min_uses="x") == []  # type: ignore[arg-type]
    assert m.unproduced(None, now=99999) == []
    # the ingest trouble shows up in the report as data
    assert any("ingest errors" in ln for ln in m.report()), m.report()


if __name__ == "__main__":
    tests = [
        (test_observe_effects_producers,
         "observe/effects_of/producers at kind and kind:subtype granularity"),
        (test_actions_without_effect_threshold,
         "actions_without_effect respects min_uses; one effect clears it"),
        (test_unproduced_silence_then_alarm,
         "unproduced silent before min_steps, specific after, retirable"),
        (test_report_format,
         "report ranks by uses, sane with zero-event actions"),
        (test_garbage_never_raises,
         "malformed input contained; errors exposed as data"),
    ]
    for i, (fn, contract) in enumerate(tests, 1):
        print(f"[infra-affordance] {i}. {contract}")
        fn()
    print("[infra-affordance] ALL PASS")
