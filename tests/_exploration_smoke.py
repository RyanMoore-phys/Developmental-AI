"""Exploration telemetry: can we tell STUCK from EXPLORING, and does it latch?

LIVE INCIDENT. The only territory number the run reported was the env's
`cells_seen = len(self._visits)`, a career visit table FIFO-capped at 50,000
cells. Once the cap was reached the length never changed again, so the loop's
per-segment `cells_delta` (the stuck monitor's "new territory" input in
infra/stack.py) read 0 FOREVER — identical for an agent pinned to a tree and
one walking across the map. That is CLAUDE.md §4.1, a guard become a latch,
on a measurement channel. Meanwhile the scoreboard (§9) is dominated by
"found a way to get paid for doing nothing" — standing at a trunk, spinning,
pacing — which is exactly what a movement measurement must expose.

CONTRACTS
  A. Separation. Synthetic agents over the same number of steps:
       stuck   — jitters within ~1 block of a tree,
       spinner — stands still and turns (yaw changes, position does not),
       pacer   — walks back and forth on a 10-block line,
       explorer— persistent random walk at walking speed.
     The explorer beats every stuck agent by >= 10x on unique cells and
     radius of gyration, and its steps_since_new_cell is far smaller.
  B. Falsification of the naive metric: path length ALONE does not separate
     the pacer from the explorer (pacer path >= 0.5 x explorer's) — which is
     why the tracker also reports displacement, rg and cells. If this ever
     fails, the synthetic pacer is not pacing and A proves less than it says.
  C. No latch in the tracker: walking 60,000 distinct cells, new_cells keeps
     counting in segments PAST 50,000 and unique_cells_total == 60,000.
  D. No latch in the env: MineRLEnvAdapter._world_events' cells_seen keeps
     rising past 50,000 cells (segment delta > 0), while the old formula
     len(_visits) is shown frozen at 50,000 (regression witness). Below the
     cap the new counter equals len(_visits) EXACTLY at every step (with
     revisits mixed in), and it is seeded from a restored visit table.
  E. Compact + persistent: 200,000 cells cost 8 bytes each once merged, and a
     state_dict round trip means a restarted stuck agent is NOT re-credited
     with "new" territory it had already visited.
  F. Pure module: exploration.py imports only stdlib + numpy (it cannot reach
     policy, replay or reward).

Run: PYTHONPATH=. python tests/_exploration_smoke.py
"""
import ast
import math
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.infra.exploration import CompactCellSet, ExplorationTracker

_n = [0]


def ok(cond, msg):
    _n[0] += 1
    if not cond:
        print(f"  [{_n[0]}] FAIL {msg}")
        sys.exit(1)
    print(f"  [{_n[0]}] {msg}")


def _run(gen, steps=3000):
    tr = ExplorationTracker(trace_path=None)
    for i in range(steps):
        p, yaw = gen(i)
        tr.observe(0, i, i * 0.25, p, yaw=yaw)
    return tr.segment_summary()["stream-0"]


def agents(seed=0):
    rng = np.random.default_rng(seed)
    jit = rng.normal(0, 0.3, size=(3000, 2))
    stuck = lambda i: ((100.5 + float(np.clip(jit[i, 0], -1, 1)), 64.0,
                        -40.5 + float(np.clip(jit[i, 1], -1, 1))), 0.0)
    spinner = lambda i: ((100.5, 64.0, -40.5), (i * 15.0) % 360.0)
    pacer = lambda i: ((100.0 + 10.0 * abs(((i * 0.9) / 10.0) % 2 - 1), 64.0,
                        -40.5), 0.0)
    head = np.cumsum(rng.normal(0, 0.15, size=3000))
    xs = 100.0 + np.cumsum(0.9 * np.cos(head))
    zs = -40.0 + np.cumsum(0.9 * np.sin(head))
    explorer = lambda i: ((float(xs[i]), 64.0, float(zs[i])),
                          float(np.degrees(head[i])))
    return {"stuck": stuck, "spinner": spinner, "pacer": pacer,
            "explorer": explorer}


def contract_a_b():
    print("A/B. stuck vs exploring")
    s = {k: _run(g) for k, g in agents().items()}
    for k, v in s.items():
        print(f"      {k:9s} path={v['path_blocks']:8.1f} "
              f"net={v['net_displacement']:7.1f} rg={v['radius_of_gyration']:6.2f} "
              f"cells={v['unique_cells_segment']:5d} "
              f"since_new={v['steps_since_new_cell']:5d} "
              f"still={v['stationary_frac']:.2f}")
    e = s["explorer"]
    for k in ("stuck", "spinner", "pacer"):
        ok(e["unique_cells_segment"] >= 10 * s[k]["unique_cells_segment"],
           f"explorer cells >= 10x {k}")
        ok(e["radius_of_gyration"] >= 10 * max(s[k]["radius_of_gyration"], 0.01),
           f"explorer rg >= 10x {k}")
        ok(s[k]["steps_since_new_cell"] > 10 * (e["steps_since_new_cell"] + 1),
           f"{k} steps_since_new_cell >> explorer's")
    ok(s["spinner"]["stationary_frac"] == 1.0 and s["spinner"]["path_blocks"] == 0,
       "spinner: turning in place is fully stationary")
    ok(s["pacer"]["path_blocks"] >= 0.5 * e["path_blocks"],
       "B: path alone does NOT separate pacer from explorer (falsification)")


def contract_c():
    print("C. tracker has no 50k latch")
    tr = ExplorationTracker(trace_path=None)
    news, i = [], 0
    for seg in range(6):
        for _ in range(10000):
            tr.observe(0, i, float(i), (i + 0.5, 64.0, 0.5))
            i += 1
        news.append(tr.segment_summary()["stream-0"]["new_cells_segment"])
    ok(news == [10000] * 6, f"new cells per segment past 50k: {news}")
    ok(tr.segment_summary()["stream-0"]["unique_cells_total"] == 60000,
       "unique_cells_total == 60000")


def _env():
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter as A
    e = A.__new__(A)
    e.coverage_cell = 8
    e._visits = {}
    e._save_break_memory = lambda **k: None
    e._prev_life = None
    e._prev_food = None
    e._prev_tool_damage = {}
    return e


def _cell_obs(cx, cz):
    return {"xpos": cx * 8.0 + 4.0, "zpos": cz * 8.0 + 4.0, "ypos": 64.0,
            "life": 20.0}


def contract_d():
    print("D. env cells_seen has no 50k latch")
    e = _env()
    rng = np.random.default_rng(1)
    exact = True
    seen_at, old_at = {}, {}
    for k in range(60000):
        if k % 7 == 3:                                   # a revisit
            j = int(rng.integers(0, k))
            w = e._world_events(_cell_obs(j % 300, j // 300))
        w = e._world_events(_cell_obs(k % 300, k // 300))
        if k < 50000 and w["cells_seen"] != len(e._visits):
            exact = False
        if k in (49999, 54999, 59999):
            seen_at[k], old_at[k] = w["cells_seen"], len(e._visits)
    ok(exact, "below the cap cells_seen == len(_visits) at every step")
    ok(seen_at[54999] - seen_at[49999] > 0 and seen_at[59999] - seen_at[54999] > 0,
       f"cells_seen keeps rising past 50k: {seen_at}")
    ok(old_at[54999] == old_at[59999] == 50000,
       f"witness: old len(_visits) is frozen at 50000: {old_at}")
    e2 = _env()
    e2._visits = {(i, 0): 3 for i in range(100)}         # restored table
    ok(e2._world_events(_cell_obs(5, 0))["cells_seen"] == 100,
       "seeded from a restored table (revisit first)")
    ok(e2._world_events(_cell_obs(500, 0))["cells_seen"] == 101,
       "then a new cell counts once")


def contract_e():
    print("E. compact + persistent")
    cs = CompactCellSet()
    for k in range(200000):
        cs.add(k * 2654435761 % (2 ** 40))
    cs.to_list()
    ok(cs._arr.nbytes == 8 * len(cs) and len(cs) == 200000,
       f"200k cells = {cs._arr.nbytes / 1e6:.1f} MB (8 B/cell)")
    tr = ExplorationTracker(trace_path=None)
    for i in range(100):
        tr.observe(0, i, float(i), ((i % 10) + 0.5, 64.0, 0.5))
    tr.segment_summary()
    tr2 = ExplorationTracker(trace_path=None)
    tr2.load_state_dict(tr.state_dict())
    for i in range(100):
        tr2.observe(0, i, 100.0 + i, ((i % 10) + 0.5, 64.0, 0.5))
    s = tr2.segment_summary()["stream-0"]
    ok(s["new_cells_segment"] == 0 and s["steps_since_new_cell"] == 190,
       "restarted stuck agent is not re-credited with new territory")


def contract_f():
    print("F. pure module")
    path = os.path.join("developmental_ai", "infra", "exploration.py")
    tree = ast.parse(open(path).read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    allowed = {"__future__", "json", "logging", "math", "os", "typing", "numpy"}
    ok(mods <= allowed, f"imports {sorted(mods)} within stdlib+numpy")


if __name__ == "__main__":
    contract_a_b()
    contract_c()
    contract_d()
    contract_e()
    contract_f()
    print("[exploration_smoke] ALL PASS")
