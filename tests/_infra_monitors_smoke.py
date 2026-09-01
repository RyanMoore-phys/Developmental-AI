"""Smoke test for developmental_ai/infra/monitors.py (pure CPU, no env).

Contracts:
  1. StuckMonitor escalation timing: patience windows + cooldown gate
     between steps, ceiling at 3.
  2. StuckMonitor instant reset: one recovered metric -> level 0 at once
     (cooldown gates re-escalation, not de-escalation), fresh cycle after.
  3. StuckMonitor reason names the stuck metrics and values; defensive
     handling of missing/non-numeric metrics never crashes.
  4. BehaviorDrift JS math matches a hand-computed distribution shift and
     the report names the largest-moving bin with before/after shares.
  5. BehaviorDrift stays None on a stable stream and on too little data
     (and an under-filled window keeps accumulating, not resetting).
  6. DecisionTrace dump writes header+records (numpy float and a raw
     object survive via default=str), ring truncates to maxlen.
  7. DecisionTrace rate-limits repeat dumps and contains I/O errors.
"""
import json
import os
import tempfile

import numpy as np

from developmental_ai.infra.monitors import (BehaviorDrift, DecisionTrace,
                                             StuckMonitor)


def test_stuck_escalation_timing():
    m = StuckMonitor({"reward": 0.0, "novelty": 0.1}, patience=2, cooldown=3)
    dead = {"reward": 0.0, "novelty": 0.05}
    levels = [m.update(dead)[0] for _ in range(10)]
    # seg1: streak 1 -> 0; seg2: streak 2 = patience -> 1; then re-escalation
    # every max(patience, cooldown)=3 stuck segments; capped at 3 forever.
    assert levels == [0, 1, 1, 1, 2, 2, 2, 3, 3, 3], levels
    print("[infra-monitors] 1. stuck escalation: patience -> 1, cooldown-gated "
          "-> 2 -> 3, capped at 3")


def test_stuck_instant_reset():
    m = StuckMonitor({"reward": 0.0}, patience=2, cooldown=5)
    for _ in range(4):
        m.update({"reward": 0.0})
    assert m.update({"reward": 0.0})[0] >= 1
    # Recovery resets IMMEDIATELY, even though cooldown=5 has not elapsed:
    # cooldown gates re-escalation only, never de-escalation.
    lvl, reason = m.update({"reward": 0.7})
    assert lvl == 0 and reason == "", (lvl, reason)
    # Fresh cycle afterwards: patience segments again before level 1.
    assert m.update({"reward": 0.0})[0] == 0
    assert m.update({"reward": 0.0})[0] == 1
    print("[infra-monitors] 2. recovery above floor -> instant level 0; "
          "fresh patience cycle after")


def test_stuck_reason_and_defensiveness():
    m = StuckMonitor({"reward": 0.0, "novelty": 0.1}, patience=1, cooldown=1)
    lvl, reason = m.update({"reward": -0.5, "novelty": 0.02})
    assert lvl == 1, (lvl, reason)
    assert "reward" in reason and "-0.5" in reason, reason
    assert "novelty" in reason and "0.02" in reason, reason
    # Missing metric is ignored (not stuckness, not recovery): state holds.
    lvl2, _ = m.update({"reward": -0.5})
    assert lvl2 >= 1, lvl2
    # Non-numeric junk and an empty dict must not crash; empty holds state.
    lvl3, _ = m.update({"reward": "garbage", "novelty": None})
    lvl4, _ = m.update({})
    assert lvl3 == lvl4 == lvl2, (lvl2, lvl3, lvl4)
    print("[infra-monitors] 3. reason names stuck metrics+values; missing/"
          "non-numeric metrics ignored, never crash")


def _feed(drift, counts):
    for b, n in enumerate(counts):
        for _ in range(n):
            drift.update(b)


def test_drift_js_math_and_report():
    d = BehaviorDrift(n_bins=4, recent_window=100, alarm_js=0.12)
    _feed(d, [70, 10, 10, 10])           # window 1 -> becomes the baseline
    assert d.segment_report() is None    # cold start: nothing to compare yet
    _feed(d, [20, 50, 20, 10])           # window 2: the regime change
    rep = d.segment_report()
    assert rep is not None, "shift this large must alarm"
    # Hand-computed JS (base 2) between p=[.7,.1,.1,.1] and q=[.2,.5,.2,.1].
    p = np.array([0.7, 0.1, 0.1, 0.1])
    q = np.array([0.2, 0.5, 0.2, 0.1])
    mm = 0.5 * (p + q)
    expect = float(0.5 * (p * np.log2(p / mm)).sum()
                   + 0.5 * (q * np.log2(q / mm)).sum())
    assert rep.startswith("bin 0 share 70.0%->20.0%"), rep
    js = float(rep.split("JS=")[1].rstrip(")"))
    assert abs(js - expect) < 1e-3, (js, expect)
    print("[infra-monitors] 4. JS matches hand computation "
          f"({js:.3f}~{expect:.3f}); largest-moving bin named with shares")


def test_drift_quiet_and_min_data():
    d = BehaviorDrift(n_bins=4, recent_window=100, alarm_js=0.12)
    _feed(d, [70, 10, 10, 10])
    assert d.segment_report() is None    # baseline init
    _feed(d, [69, 11, 10, 10])           # same regime -> below alarm
    assert d.segment_report() is None
    # Too little data: no report, and the window is NOT thrown away —
    # (10+15)=25 observations then reach the /4 threshold and get judged.
    _feed(d, [10, 0, 0, 0])
    assert d.segment_report() is None    # 10 < 100/4
    _feed(d, [15, 0, 0, 0])
    rep = d.segment_report()             # 25 obs, 100% bin 0 vs ~70% baseline
    assert rep is not None and rep.startswith("bin 0"), rep
    # Bad indices are dropped silently, never counted, never raise.
    d.update(-1); d.update(99); d.update("x")
    print("[infra-monitors] 5. stable stream -> None; under-filled window "
          "accumulates instead of resetting; bad bins dropped")


def test_trace_dump_contents():
    t = DecisionTrace(maxlen=10, min_pushes_between_dumps=3)
    for i in range(15):                  # ring: only the last 10 survive
        t.push({"i": i, "v": np.float32(1.5), "obj": object()})
    path = os.path.join(tempfile.mkdtemp(), "trace.jsonl")
    out = t.dump(path, "unit-test")
    assert out == path, out
    lines = open(path).read().splitlines()
    assert len(lines) == 11, len(lines)
    head = json.loads(lines[0])
    assert head == {"__dump__": "unit-test", "n": 10}, head
    recs = [json.loads(ln) for ln in lines[1:]]
    assert [r["i"] for r in recs] == list(range(5, 15)), "ring truncation"
    # numpy float and raw object went through default=str, not an exception
    assert recs[0]["v"] == "1.5" and "object" in recs[0]["obj"], recs[0]
    print("[infra-monitors] 6. dump: header+records, maxlen ring, numpy "
          "float + object serialised via default=str")


def test_trace_rate_limit_and_io_containment():
    t = DecisionTrace(maxlen=10, min_pushes_between_dumps=3)
    t.push({"a": 1})
    path = os.path.join(tempfile.mkdtemp(), "trace.jsonl")
    assert t.dump(path, "first") == path      # first dump: always allowed
    assert t.dump(path, "again") is None      # 0 pushes since -> limited
    t.push({"a": 2}); t.push({"a": 3})
    assert t.dump(path, "still") is None      # 2 < 3 -> limited
    t.push({"a": 4})
    assert t.dump(path, "ok") == path         # 3 >= 3 -> appends 2nd chunk
    lines = open(path).read().splitlines()
    heads = [json.loads(ln) for ln in lines if "__dump__" in ln]
    assert [h["__dump__"] for h in heads] == ["first", "ok"], heads
    # I/O error contained: bogus directory -> None, no exception, and the
    # failed attempt does not consume the rate-limit budget.
    t2 = DecisionTrace(maxlen=4, min_pushes_between_dumps=2)
    t2.push({"x": 1})
    assert t2.dump("/no/such/dir/deep/trace.jsonl", "io") is None
    good = os.path.join(tempfile.mkdtemp(), "t2.jsonl")
    assert t2.dump(good, "retry") == good
    print("[infra-monitors] 7. rate limit between dumps; I/O errors return "
          "None and do not spend the budget")


if __name__ == "__main__":
    for fn in (test_stuck_escalation_timing, test_stuck_instant_reset,
               test_stuck_reason_and_defensiveness,
               test_drift_js_math_and_report, test_drift_quiet_and_min_data,
               test_trace_dump_contents,
               test_trace_rate_limit_and_io_containment):
        fn()
    print("[infra-monitors] ALL PASS")
