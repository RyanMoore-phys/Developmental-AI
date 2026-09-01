"""MonotoneCounter smoke (pure stdlib, no env).

Contracts:
  1. First real reading becomes the baseline and emits 0 events (the
     73-deaths-at-join case); None before it is a no-op.
  2. +1 within sanity emits exactly 1 event and advances the baseline.
  3. +73 in one tick is a RESYNC: 0 events, (old, new) recorded, pop_resync
     returns it once and clears it.
  4. A decrease is a REBASELINE: 0 events, new lower baseline adopted, and
     the next +1 counts from there (no phantom climb-back deltas).
  5. None mid-stream is tolerated: 0 events, baseline untouched.
  6. Fractional creep (+0.4) emits 0 events but HOLDS the baseline, so the
     creep accumulates into a real event instead of rounding away forever.
  7. Garbage readings (NaN, inf, strings) never raise; they count as
     bad_readings and leave the baseline intact.
"""

from developmental_ai.infra.counters import MonotoneCounter


def test_first_read_baseline():
    c = MonotoneCounter()
    assert c.update(None) == 0 and c.value is None
    assert c.update(73.0) == 0, "join burst counted as events"
    assert c.value == 73.0
    print("[infra-counters] 1. first reading -> baseline, 0 events "
          "(0->73 join burst absorbed)")


def test_small_increase_counts():
    c = MonotoneCounter()
    c.update(10.0)
    assert c.update(11.0) == 1
    assert c.value == 11.0
    assert c.update(13.0) == 2, "+2 within sanity_delta=2.5 must count"
    print("[infra-counters] 2. +1 -> 1 event; +2 -> 2 events; baseline "
          "advances")


def test_resync_recorded_and_popped():
    c = MonotoneCounter()
    c.update(5.0)
    assert c.update(78.0) == 0, "resync jump counted as events"
    assert c.last_resync == (5.0, 78.0)
    assert c.value == 78.0, "resync must still adopt the new baseline"
    assert c.pop_resync() == (5.0, 78.0)
    assert c.pop_resync() is None and c.last_resync is None
    assert c.update(79.0) == 1, "counting must resume from the resynced base"
    print("[infra-counters] 3. +73 -> 0 events, resync recorded, pop "
          "clears, counting resumes")


def test_decrease_rebaselines():
    c = MonotoneCounter()
    c.update(40.0)
    assert c.update(3.0) == 0, "external reset counted as events"
    assert c.value == 3.0
    assert c.last_resync is None, "a reset is not a resync record"
    assert c.update(4.0) == 1, "post-reset counting must use the new base"
    print("[infra-counters] 4. decrease -> rebaseline, 0 events, next +1 "
          "counts")


def test_none_midstream():
    c = MonotoneCounter()
    c.update(7.0)
    assert c.update(None) == 0
    assert c.value == 7.0, "None must not disturb the baseline"
    assert c.update(8.0) == 1
    print("[infra-counters] 5. None mid-stream tolerated, baseline intact")


def test_fractional_creep():
    c = MonotoneCounter()
    c.update(100.0)
    assert c.update(100.4) == 0, "0.4 is not an event"
    assert c.value == 100.0, "baseline must be HELD so creep accumulates"
    assert c.update(100.4) == 0            # same creepy reading again: still 0
    assert c.update(101.0) == 1, "accumulated +1.0 creep must finally count"
    assert c.value == 101.0
    print("[infra-counters] 6. +0.4 creep -> 0 events, baseline held, "
          "creep eventually counts")


def test_garbage_never_raises():
    c = MonotoneCounter()
    c.update(2.0)
    for junk in (float("nan"), float("inf"), float("-inf"), "seventy-three",
                 [], {}):
        assert c.update(junk) == 0          # type: ignore[arg-type]
    assert c.value == 2.0, "garbage must not move the baseline"
    assert c.bad_readings == 6
    assert c.update(3.0) == 1, "counter must survive a garbage flood"
    # A broken sanity_delta config degrades to the default, never a dead
    # counter or a constructor crash.
    c2 = MonotoneCounter(sanity_delta="oops")     # type: ignore[arg-type]
    c2.update(0.0)
    assert c2.update(1.0) == 1
    print("[infra-counters] 7. NaN/inf/strings absorbed as bad_readings; "
          "bad config degrades safely")


if __name__ == "__main__":
    for fn in (test_first_read_baseline, test_small_increase_counts,
               test_resync_recorded_and_popped, test_decrease_rebaselines,
               test_none_midstream, test_fractional_creep,
               test_garbage_never_raises):
        fn()
    print("[infra-counters] ALL PASS")
