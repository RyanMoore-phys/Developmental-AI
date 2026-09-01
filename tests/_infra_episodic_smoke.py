"""EpisodicEventMemory smoke — pure stdlib, no env needed.

Contracts:
  1. record/last/count honour the kind filter, with and without a subtype.
  2. Ring buffer evicts oldest-first at capacity; count() reflects only
     retained records.
  3. near() measures (x, z) only — a record far away in y is still "here" —
     skips positionless records, honours the kind filter, and returns
     most-recent-first copies.
  4. bearing_from() on known geometry: due +x -> 90 deg, due +z -> 0 deg,
     correct distance; skips positionless newer records back to the last
     positioned one; None when nothing positioned matches.
  5. summary() reports position + age for seen kinds and the literal
     "never" for a requested-but-unseen kind; kind:subtype specs work.
  6. Defensive containment: malformed position/salience never raise —
     the event is kept (position dropped), errors surface as data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.infra.episodic import EpisodicEventMemory


def test_record_last_count():
    m = EpisodicEventMemory()
    m.record("sighting", "goal", step=10, position=(1.0, 2.0, 3.0))
    m.record("sighting", "landmark", step=20, position=(4.0, 5.0, 6.0))
    m.record("achieve", "goal", step=30, salience=2.5)
    # last() without subtype = most recent of the kind
    r = m.last("sighting")
    assert r is not None and r["subtype"] == "landmark" and r["step"] == 20, r
    # last() with subtype filters within the kind
    r = m.last("sighting", "goal")
    assert r is not None and r["step"] == 10 and r["position"] == (1.0, 2.0, 3.0), r
    assert r["salience"] == 1.0, r
    # unseen kind/subtype -> None
    assert m.last("sighting", "unicorn") is None
    assert m.last("dream") is None
    # counts, filtered and unfiltered
    assert m.count("sighting") == 2
    assert m.count("sighting", "goal") == 1
    assert m.count("achieve") == 1
    assert m.count("dream") == 0
    # the returned dict is a copy — mutating it must not corrupt the store
    r["step"] = 999
    assert m.last("sighting", "goal")["step"] == 10
    print("[infra-episodic] 1. record/last/count with and without subtype")


def test_capacity_eviction():
    m = EpisodicEventMemory(capacity=3)
    for i in range(5):
        m.record("evt", f"s{i}", step=i)
    assert len(m) == 3
    assert m.count("evt") == 3
    # oldest two evicted, newest three retained
    assert m.last("evt", "s0") is None and m.last("evt", "s1") is None
    assert m.last("evt")["subtype"] == "s4"
    assert m.count("evt", "s2") == 1
    print("[infra-episodic] 2. capacity eviction is oldest-first")


def test_near_ignores_y():
    m = EpisodicEventMemory()
    m.record("sighting", "high", step=1, position=(1.0, 250.0, 1.0))   # far in y
    m.record("sighting", "far", step=2, position=(100.0, 0.0, 0.0))    # far in x
    m.record("sighting", "flat", step=3, position=(2.0, 0.0, -2.0))
    m.record("sighting", "nowhere", step=4)                            # no position
    m.record("noise", "close", step=5, position=(0.0, 0.0, 0.0))
    hits = m.near((0.0, 0.0, 0.0), radius=5.0)
    subs = [h["subtype"] for h in hits]
    # y=250 record is IN (height is not distance), x=100 is OUT,
    # positionless is skipped; most-recent-first ordering
    assert subs == ["close", "flat", "high"], subs
    # kind filter
    subs = [h["subtype"] for h in m.near((0.0, 0.0, 0.0), 5.0, kind="sighting")]
    assert subs == ["flat", "high"], subs
    # results are copies
    hits[0]["subtype"] = "mutated"
    assert m.last("noise")["subtype"] == "close"
    print("[infra-episodic] 3. near() ignores y, skips positionless, copies")


def test_bearing_geometry():
    m = EpisodicEventMemory()
    m.record("sighting", "goal", step=1, position=(5.0, 60.0, 0.0))
    got = m.bearing_from((0.0, 0.0, 0.0), "sighting", "goal")
    assert got is not None
    dist, brg = got
    assert abs(dist - 5.0) < 1e-9 and abs(brg - 90.0) < 1e-9, got  # due +x
    m.record("sighting", "goal", step=2, position=(0.0, -30.0, 7.0))
    dist, brg = m.bearing_from((0.0, 0.0, 0.0), "sighting", "goal")
    assert abs(dist - 7.0) < 1e-9 and abs(brg - 0.0) < 1e-9, (dist, brg)  # due +z
    # a NEWER positionless record must not blind the query — falls back to
    # the most recent record that HAS a position
    m.record("sighting", "goal", step=3)
    dist, brg = m.bearing_from((0.0, 0.0, 0.0), "sighting", "goal")
    assert abs(dist - 7.0) < 1e-9 and abs(brg - 0.0) < 1e-9, (dist, brg)
    # nothing positioned matches -> None
    m.record("achieve", "goal", step=4)
    assert m.bearing_from((0.0, 0.0, 0.0), "achieve", "goal") is None
    assert m.bearing_from((0.0, 0.0, 0.0), "dream") is None
    print("[infra-episodic] 4. bearing: +x=90deg, +z=0deg, positionless skip")


def test_summary():
    m = EpisodicEventMemory()
    m.record("sighting", "goal", step=100, position=(12.4, 70.0, -40.2))
    s = m.summary(now_step=440, kinds=["sighting", "achieve:goal"])
    assert "sighting:goal" in s and "@(12,-40)" in s and "340 steps ago" in s, s
    assert "achieve:goal never" in s, s
    assert " | " in s, s
    # positionless records still report age, just without a location
    m.record("achieve", "goal", step=430)
    s = m.summary(now_step=440, kinds=["achieve:goal"])
    assert s == "achieve:goal 10 steps ago", s
    # kinds=None covers every kind seen; empty memory has a stable marker
    s = m.summary(now_step=440)
    assert "sighting:goal" in s and "achieve:goal" in s, s
    assert EpisodicEventMemory().summary(0) == "(no episodic records)"
    print("[infra-episodic] 5. summary: position+age, literal 'never'")


def test_defensive_containment():
    m = EpisodicEventMemory(capacity=-1)         # bad capacity -> default
    m.record("evt", "bad_pos", step=1, position=("x", None))   # malformed
    m.record("evt", "nan_pos", step=2, position=(float("nan"), 0.0, 0.0))
    m.record("evt", "bad_sal", step=3, position=(1.0, 2.0, 3.0),
             salience="loud")
    # events survive with the bad field contained, never an exception
    assert m.count("evt") == 3
    assert m.last("evt", "bad_pos")["position"] is None
    assert m.last("evt", "nan_pos")["position"] is None
    assert m.last("evt", "bad_sal")["salience"] == 1.0
    assert m.error_count == 3 and m.last_error is not None
    # queries with garbage input return empty values, not exceptions
    assert m.near(("a", "b", "c"), 5.0) == []
    assert m.bearing_from((None,), "evt") is None
    print("[infra-episodic] 6. malformed input contained, errors are data")


if __name__ == "__main__":
    for fn in (test_record_last_count, test_capacity_eviction,
               test_near_ignores_y, test_bearing_geometry,
               test_summary, test_defensive_containment):
        fn()
    print("[infra-episodic] ALL PASS")
