"""Infra ledger smoke (pure stdlib, no env needed).

Contracts:
  1. Shares/HHI math over abs sums — a negative shaping stream cannot hide a
     positive farm stream.
  2. segment() resets accumulators; an empty follow-up segment is zero-total
     with empty shares.
  3. Dominance alarm needs N CONSECUTIVE dominant segments; a balanced or
     zero-total segment breaks the streak.
  4. FarmDetector flags a synthetic pace-in-circle-with-income pattern with
     the exact report format, best-first, capped at 5 lines.
  5. The SAME circle under telescoping (potential-based) shaping nets ~0 per
     loop and is NOT flagged.
  6. LRU bound: distinct-cell flood never grows state past max_cells.
  7. Defensive contract: garbage inputs never raise; errors surface as data.
"""
import sys

sys.path.insert(0, ".")

from developmental_ai.infra.ledger import FarmDetector, RewardLedger


def test_shares_hhi():
    led = RewardLedger()
    led.record("a", 3.0)
    led.record("b", -1.0)
    out = led.segment()
    assert abs(out["total"] - 2.0) < 1e-9, out
    assert abs(out["shares"]["a"] - 0.75) < 1e-9, out
    assert abs(out["shares"]["b"] - 0.25) < 1e-9, out
    assert abs(out["hhi"] - (0.75 ** 2 + 0.25 ** 2)) < 1e-9, out
    # negative shaping cannot hide a farm: gross (abs) shares, not net
    led.record("shaper", -10.0)
    led.record("farm", 2.0)
    out = led.segment()
    assert abs(out["total"] - (-8.0)) < 1e-9, out
    assert abs(out["shares"]["shaper"] - 10.0 / 12.0) < 1e-9, out
    assert abs(out["shares"]["farm"] - 2.0 / 12.0) < 1e-9, out


def test_segment_reset():
    led = RewardLedger()
    led.record("a", 5.0)
    led.segment()
    led.record("c", 1.0)
    out = led.segment()
    assert out["shares"] == {"c": 1.0} and abs(out["total"] - 1.0) < 1e-9, out
    assert abs(out["hhi"] - 1.0) < 1e-9, out
    # nothing recorded since -> zero-total segment, empty shares, hhi 0
    out = led.segment()
    assert out["total"] == 0.0 and out["shares"] == {} and out["hhi"] == 0.0, \
        out
    assert out["alarms"] == [], out


def test_alarm_consecutive_and_streak_breaks():
    led = RewardLedger(share_alarm=0.8, consecutive=3)

    def seg(dom, other):
        led.record("x", dom)
        led.record("y", other)
        return led.segment()

    # needs exactly 3 consecutive dominant segments
    assert seg(9, 1)["alarms"] == []
    assert seg(9, 1)["alarms"] == []
    r = seg(9, 1)
    assert any("x" in a and "DOMINANT" in a for a in r["alarms"]), r
    # a balanced segment resets the streak...
    assert seg(5, 5)["alarms"] == []
    assert seg(9, 1)["alarms"] == []
    assert seg(9, 1)["alarms"] == []
    r = seg(9, 1)
    assert any("x" in a for a in r["alarms"]), r
    # ...and so does a ZERO-TOTAL segment (idle stretch = not a farm)
    seg(5, 5)                                   # clear the running streak
    assert seg(9, 1)["alarms"] == []
    assert seg(9, 1)["alarms"] == []            # streak at 2
    z = led.segment()                           # nothing recorded
    assert z["shares"] == {} and z["alarms"] == [], z
    assert seg(9, 1)["alarms"] == []            # streak restarted: 1
    assert seg(9, 1)["alarms"] == []            # 2 — still silent
    r = seg(9, 1)                               # 3 — fires again
    assert any("x" in a for a in r["alarms"]), r


def test_farm_detector_flags_income_circle():
    fd = FarmDetector(revisit_horizon=50, min_loops=6, income_alarm=0.02)
    acts = [0, 0, 0, 0, 1, 1, 2, 0]             # 5x action0, 2x action1, 1x2
    step = 0
    for _lap in range(10):                       # 10 laps around 8 cells
        for i in range(8):
            r = 1.0 if i == 0 else 0.0           # +1 per lap, paid at cell 0
            fd.step(cell=i, action=acts[i], step_reward=r, step=step)
            step += 1
    lines = fd.segment_report()
    # 8 cells qualify (9 loops each, rate 1/8 = 0.125) -> capped at 5 lines,
    # deterministic tie-break puts cell 0 first
    assert len(lines) == 5, lines
    assert lines[0] == "FARM? cell=0 loops=9 +0.125/step actions=[0, 1, 2]", \
        lines[0]
    assert all(ln.startswith("FARM? cell=") for ln in lines), lines


def test_no_flag_on_telescoping_shaping():
    # identical circling behaviour, but reward is a pure potential difference
    # phi(next)-phi(cur): sums to EXACTLY 0 over every closed lap, so the
    # income-rate filter (not the loop-count filter) must clear it
    fd = FarmDetector(revisit_horizon=50, min_loops=6, income_alarm=0.02)
    phi = [float(i) for i in range(8)]
    step = 0
    prev = 7                                     # entering cell 0 from cell 7
    for _lap in range(10):
        for i in range(8):
            r = phi[i] - phi[prev]               # telescopes to 0 per lap
            fd.step(cell=i, action=0, step_reward=r, step=step)
            prev = i
            step += 1
    assert fd.segment_report() == [], fd.segment_report()


def test_lru_bound():
    fd = FarmDetector(max_cells=16)
    for s in range(100):                         # 100 distinct cells
        fd.step(cell=s, action=0, step_reward=0.0, step=s)
    assert len(fd._cells) == 16, len(fd._cells)
    # most recent cells survive, oldest were evicted
    assert 99 in fd._cells and 0 not in fd._cells


def test_defensive_containment():
    led = RewardLedger()
    led.record("n", float("nan"))                # dropped, not booked
    led.record(object(), "not-a-number")         # contained
    out = led.segment()
    assert out["total"] == 0.0, out
    assert any("LEDGER-ERROR" in a for a in out["alarms"]), out
    # error memory clears once surfaced
    assert led.segment()["alarms"] == []

    fd = FarmDetector()
    fd.step("weird", None, "x", object())        # contained, never raises
    fd.step(cell=1, action=0, step_reward=float("inf"), step=0)
    rep = fd.segment_report()
    assert isinstance(rep, list), rep
    assert any("DETECTOR-ERROR" in ln for ln in rep), rep
    # healthy traffic afterwards still works
    fd.step(cell=2, action=1, step_reward=0.5, step=10)
    assert isinstance(fd.segment_report(), list)


if __name__ == "__main__":
    tests = [
        (test_shares_hhi,
         "shares/hhi over abs sums; negative shaping cannot hide a farm"),
        (test_segment_reset,
         "segment() resets; empty segment is zero-total/empty-shares"),
        (test_alarm_consecutive_and_streak_breaks,
         "alarm needs N consecutive; balanced or zero segments break streak"),
        (test_farm_detector_flags_income_circle,
         "detector flags circle-with-income; format + best-first + 5-cap"),
        (test_no_flag_on_telescoping_shaping,
         "telescoping shaping nets ~0/loop and is NOT flagged"),
        (test_lru_bound,
         "LRU bound holds under distinct-cell flood"),
        (test_defensive_containment,
         "garbage input never raises; errors surface as data"),
    ]
    for n, (fn, contract) in enumerate(tests, 1):
        print(f"[infra-ledger] {n}. {contract}")
        fn()
    print("[infra-ledger] ALL PASS")
