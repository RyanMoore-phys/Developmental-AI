"""Unit cases for developmental_ai.infra.exploration (ExplorationTracker).

Parts only, each against a geometry whose answer is known in closed form:
a straight line (path = displacement), a closed circle (path = 2*pi*r, rg = r,
displacement ~ 0), standing still (stationary_frac = 1, one cell), a teleport
(excluded from the path, counted), an episode change (no step across it),
cell flooring incl. negatives, cross-segment steps_since_new_cell, the
state_dict round trip, trace line schema and trace rotation (keep 3).
The design claims (stuck vs exploring separate; no 50k latch) live in
tests/_exploration_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_exploration_unit.py
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, ".")

from tests.unit._runner import case, run_all, close, raises
from developmental_ai.infra.exploration import (
    CompactCellSet, ExplorationTracker, pack_cell, unpack_cell)


def _tr(**kw):
    kw.setdefault("trace_path", None)
    return ExplorationTracker(**kw)


def _feed(tr, pts, stream=0, t0=0.0, dt=1.0, episodes=None):
    for i, p in enumerate(pts):
        tr.observe(stream, i, t0 + i * dt, p,
                   episode=None if episodes is None else episodes[i])


@case
def straight_line_path_equals_displacement():
    tr = _tr()
    _feed(tr, [(float(i), 64.0, 0.0) for i in range(100)])
    s = tr.segment_summary()["stream-0"]
    close(s["path_blocks"], 99.0, 1e-9, "path")
    close(s["net_displacement"], 99.0, 1e-9, "net")
    close(s["radius_of_gyration"], math.sqrt((100 ** 2 - 1) / 12.0), 1e-6, "rg")
    close(s["stationary_frac"], 0.0, 0, "stationary")
    assert s["unique_cells_segment"] == 100 and s["new_cells_segment"] == 100
    assert s["n_obs"] == 100 and s["teleports"] == 0
    close(s["y_min"], 64.0) and close(s["y_max"], 64.0)


@case
def circle_path_2pi_r_rg_r_displacement_small():
    tr = _tr()
    r, n = 10.0, 360
    pts = [(r * math.cos(2 * math.pi * k / n), 70.0,
            r * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]
    _feed(tr, pts)
    s = tr.segment_summary()["stream-0"]
    close(s["path_blocks"], n * 2 * r * math.sin(math.pi / n), 1e-6, "path")
    close(s["net_displacement"], 0.0, 1e-9, "net")
    close(s["radius_of_gyration"], r, 0.05, "rg")


@case
def standing_still_is_stationary_one_cell():
    tr = _tr()
    _feed(tr, [(5.5 + 0.01 * (i % 2), 64.0, 5.5) for i in range(100)])
    s = tr.segment_summary()["stream-0"]
    close(s["stationary_frac"], 1.0, 0, "stationary")
    assert s["unique_cells_segment"] == 1 and s["unique_cells_total"] == 1
    assert s["steps_since_new_cell"] == 99
    close(s["seconds_since_new_cell"], 99.0)
    assert s["path_blocks"] < 1.0


@case
def teleport_excluded_and_counted():
    tr = _tr()
    _feed(tr, [(0, 0, 0), (1, 0, 0), (100, 0, 0), (101, 0, 0)])
    s = tr.segment_summary()["stream-0"]
    close(s["path_blocks"], 2.0, 1e-9, "path")
    assert s["teleports"] == 1
    close(s["net_displacement"], 101.0, 1e-9, "net")


@case
def no_step_across_episode_change():
    tr = _tr()
    _feed(tr, [(0, 0, 0), (1, 0, 0), (5, 0, 0), (6, 0, 0)],
          episodes=["a", "a", "b", "b"])
    s = tr.segment_summary()["stream-0"]
    close(s["path_blocks"], 2.0, 1e-9, "path")
    assert s["teleports"] == 0
    close(s["stationary_frac"], 0.0, 0)


@case
def cells_floor_negative_and_cell_size():
    tr = _tr(cell_size=4.0)
    _feed(tr, [(-0.5, 0, 0), (3.9, 0, 0), (4.0, 0, 0), (4.0, 0, -0.1)])
    s = tr.segment_summary()["stream-0"]
    assert s["unique_cells_segment"] == 4, s
    for cx, cz in [(-1, 0), (0, 0), (1, -1), (-(2 ** 31), 2 ** 31 - 1), (7, -9)]:
        assert unpack_cell(pack_cell(cx, cz)) == (cx, cz)
    assert pack_cell(-1, 0) != pack_cell(0, -1)


@case
def steps_since_new_cell_persists_across_segments():
    tr = _tr()
    _feed(tr, [(0.5, 0, 0.5)] * 10)
    s1 = tr.segment_summary()["stream-0"]
    for i in range(5):
        tr.observe(0, 10 + i, 10.0 + i, (0.5, 0, 0.5))
    s2 = tr.segment_summary()["stream-0"]
    assert s1["steps_since_new_cell"] == 9 and s2["steps_since_new_cell"] == 14
    assert s2["new_cells_segment"] == 0 and s2["unique_cells_total"] == 1
    close(s2["seconds_since_new_cell"], 14.0)
    tr.observe(0, 15, 15.0, (9.5, 0, 0.5))
    assert tr.segment_summary()["stream-0"]["steps_since_new_cell"] == 0


@case
def cells_per_hour_uses_wall_time():
    tr = _tr()
    _feed(tr, [(float(i), 0, 0) for i in range(11)], dt=360.0)   # 1 hour
    s = tr.segment_summary()["stream-0"]
    close(s["cells_per_hour"], 11.0, 1e-9, "cells/h")
    # zero elapsed time is undefined, not 0
    tr2 = _tr()
    tr2.observe(0, 0, 5.0, (0, 0, 0))
    assert tr2.segment_summary()["stream-0"]["cells_per_hour"] is None


@case
def empty_segment_values_are_none_not_zero():
    tr = _tr()
    _feed(tr, [(0, 0, 0), (3, 0, 0)])
    tr.segment_summary()
    s = tr.segment_summary()["stream-0"]
    assert s["n_obs"] == 0 and s["net_displacement"] is None
    assert s["stationary_frac"] is None and s["radius_of_gyration"] is None
    assert s["unique_cells_total"] == 2


@case
def invalid_positions_rejected_not_raised():
    tr = _tr()
    tr.observe(0, 0, 0.0, (float("nan"), 0, 0))
    tr.observe(0, 1, 1.0, None)
    tr.observe(0, 2, 2.0, (1, 2))
    assert tr.rejected == 3 and tr.segment_summary() == {}


@case
def streams_are_independent():
    tr = _tr()
    _feed(tr, [(float(i), 0, 0) for i in range(10)], stream=0)
    _feed(tr, [(0.5, 0, 0.5)] * 10, stream=1)
    s = tr.segment_summary()
    close(s["stream-0"]["path_blocks"], 9.0, 1e-9)
    close(s["stream-1"]["path_blocks"], 0.0, 1e-9)


@case
def state_round_trip():
    tr = _tr()
    _feed(tr, [(float(i) * 3, 0, -float(i)) for i in range(50)])
    tr.observe(0, 50, 50.0, (0.0, 0, 0.0))     # revisit
    tr.segment_summary()
    d = json.loads(json.dumps(tr.state_dict()))
    cells = d["streams"]["0"]["cells"]
    assert cells == sorted(cells) and len(cells) == 50
    tr2 = _tr()
    tr2.load_state_dict(d)
    tr2.observe(0, 0, 60.0, (3.0, 0, -1.0))     # already known
    s = tr2.segment_summary()["stream-0"]
    assert s["new_cells_segment"] == 0 and s["unique_cells_total"] == 50
    assert s["steps_since_new_cell"] == 2
    close(s["seconds_since_new_cell"], 11.0)
    raises(lambda: _tr(cell_size=2.0).load_state_dict(d), ValueError)
    raises(lambda: tr2.load_state_dict({"v": 9}), ValueError)


@case
def compact_set_membership_and_merge():
    cs = CompactCellSet()
    cs.MERGE_AT = 7
    keys = [pack_cell(i, -i) for i in range(100)]
    assert all(cs.add(k) for k in keys) and len(cs) == 100
    assert not any(cs.add(k) for k in keys) and len(cs) == 100
    assert pack_cell(1000, 0) not in cs and keys[50] in cs
    capped = CompactCellSet(max_cells=10)
    for k in keys:
        capped.add(k)
    assert len(capped) == 10 and capped.overflow == 90


@case
def trace_lines_every_n_and_schema():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sub", "trace.jsonl")
    tr = ExplorationTracker(trace_every_steps=4, trace_path=p, run_id="r1")
    for i in range(10):
        tr.observe(1, 100 + i, 1000.0 + i, (i, 64, 0), yaw=90.0, pitch=-3.0,
                   episode="e")
    assert not os.path.exists(p)                 # buffered until summary
    tr.segment_summary()
    rows = [json.loads(l) for l in open(p)]
    assert [r["step"] for r in rows] == [100, 104, 108]
    want = {"schema", "v", "run_id", "stream", "episode", "step", "t_wall",
            "x", "y", "z", "yaw", "pitch"}
    assert set(rows[0]) == want and rows[0]["schema"] == "skybot.position"
    assert rows[0]["run_id"] == "r1" and rows[0]["stream"] == 1


@case
def trace_rotation_keeps_three():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "trace.jsonl")
    tr = ExplorationTracker(trace_every_steps=1, trace_path=p)
    tr.rotate_bytes = 400
    for seg in range(20):
        for i in range(5):
            tr.observe(0, seg * 5 + i, float(seg * 5 + i), (i, 0, 0))
        tr.segment_summary()
    names = sorted(os.listdir(d))
    assert names == ["trace.jsonl", "trace.jsonl.1", "trace.jsonl.2",
                     "trace.jsonl.3"], names
    last = [json.loads(l) for l in open(p)][-1]
    assert last["step"] == 99


@case
def unwritable_trace_never_raises():
    d = tempfile.mkdtemp()
    blocker = os.path.join(d, "f")
    open(blocker, "w").close()
    tr = ExplorationTracker(trace_every_steps=1,
                            trace_path=os.path.join(blocker, "x.jsonl"))
    tr.observe(0, 0, 0.0, (0, 0, 0))
    assert "stream-0" in tr.segment_summary()


if __name__ == "__main__":
    sys.exit(run_all("exploration_unit"))
