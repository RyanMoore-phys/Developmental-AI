#!/usr/bin/env python3
"""Stage 2 — integration against real MineRL. 500 steps, random policy.

NOT LEARNING. This proves the plumbing carries REAL data, which is exactly
what an offline suite cannot: every unit test feeds the bus a synthetic
context, so a sensor that is wired but permanently neutral passes all of
them and is indistinguishable from one that works.

THE CENTRAL CHECK IS 2.3 — every sensor must VARY. A constant sensor is the
signature of this project's most expensive failure class: machinery that
runs, logs, and does nothing.

Run on the pod:
    PYTHONPATH=. python scripts/host_stage2_integration.py --steps 500
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--config", default="configs/minecraft_skybot.yaml")
    ap.add_argument("--out", default="runlogs/stage2")
    ap.add_argument("--baseline-rate", type=float, default=0.0,
                    help="pre-change steps/s; 2.7 is skipped if 0")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    env_cfg = cfg["environment"]
    sen_cfg = cfg.get("sensors") or None

    from developmental_ai.environments.minerl_env import (
        MineRLEnvAdapter, TREECHOP_MACROS)
    from developmental_ai.sensors import build_default_bus

    env = MineRLEnvAdapter(
        image_size=env_cfg["image_size"],
        action_repeat=env_cfg.get("action_repeat", 4),
        render_size=env_cfg.get("render_size", 0),
        sensors_cfg=sen_cfg)
    bus = env._sensor_bus
    if bus is None:
        print("FAIL: no sensor bus was built; check config `sensors.enabled`")
        return 1

    want_w = bus.width
    want_hash = bus.layout_hash()
    print(f"bus: {bus.describe()}")

    obs, info = env.reset()
    rows, oracle_rows, layouts, widths = [], [], set(), set()
    nan_steps, missing = 0, 0
    action_seen = {}
    t0 = time.time()

    for i in range(a.steps):
        act = i % env.action_space.n              # sweep EVERY macro
        obs, r, term, trunc, info = env.step(act)
        v = info.get("sensors")
        if v is None:
            missing += 1
        else:
            v = np.asarray(v, dtype=np.float32)
            widths.add(int(v.shape[0]))
            if not np.all(np.isfinite(v)):
                nan_steps += 1
            rows.append(v)
        layouts.add(info.get("sensor_layout"))
        orc = info.get("oracle") or {}
        if "true_position" in orc:
            oracle_rows.append(np.asarray(orc["true_position"], np.float32))
        action_seen.setdefault(act, 0)
        action_seen[act] += 1
        if term or trunc:
            obs, info = env.reset()

    dt = time.time() - t0
    rate = a.steps / max(1e-6, dt)
    X = np.stack(rows) if rows else np.zeros((0, want_w), np.float32)

    res, fails = {}, []

    def check(name, ok, detail):
        res[name] = {"ok": bool(ok), "detail": detail}
        print(f"  [{'OK ' if ok else 'NO '}] {name}: {detail}")
        if not ok:
            fails.append(name)

    check("2.1_width", widths == {want_w} and missing == 0,
          f"widths seen {sorted(widths)}, want {want_w}; "
          f"{missing} steps had no sensors key")
    check("2.2_layout_stable", len(layouts) == 1 and want_hash in layouts,
          f"layout hashes seen: {layouts}")

    # 2.3 — THE ONE THAT MATTERS.
    dead, alive = [], []
    for s in bus.policy_sensors():
        sl = bus.slice_of(s.name, np.arange(want_w))
        block = X[:, sl.astype(int)] if X.size else np.zeros((1, s.width))
        sd = float(block.std())
        (alive if sd > 1e-6 else dead).append((s.name, sd))
    check("2.3_every_sensor_varies", not dead,
          (f"all {len(alive)} vary" if not dead
           else f"CONSTANT: {[n for n, _ in dead]} — wired but never changing, "
                f"which no unit test can distinguish from working"))
    res["sensor_std"] = {n: sd for n, sd in alive + dead}

    check("2.4_no_nan", nan_steps == 0, f"{nan_steps} steps had non-finite values")

    leak = False
    if oracle_rows and X.size:
        O = np.stack(oracle_rows)
        for col in range(O.shape[1]):
            vals = np.unique(np.round(O[:, col], 3))
            if len(vals) > 3 and np.isin(np.round(X, 3), vals).any():
                leak = True
    check("2.5_oracle_isolated", len(oracle_rows) > 0 and not leak,
          f"{len(oracle_rows)} oracle readings, transport leak={leak}")

    # 2.6 — A MACRO IS (SOCKET ACTION, HOW LONG IT IS HELD).
    #
    # THE FIRST VERSION OF THIS CHECK COMPARED ONLY THE ACTION DICT AND WAS
    # WRONG. It reported 25/28 and named macros 5/12/26/27 as duplicates —
    # all four emit `{attack: 1}`. They are not duplicates: `_ticks` is a
    # SCHEDULING directive, not an engine key, so `_macro_to_action`
    # deliberately strips it, and the step loop honours it at
    # minerl_env.py:1864 (`for _ in range(self._macro_ticks(action))`).
    # Those four hold attack for 4, 40, 10 and 120 ticks — which is the
    # entire point of A8, since a barehanded log needs ~60 ticks and an axe
    # ~8.
    #
    # So the honest identity of a macro is the pair. Comparing the dict
    # alone would have condemned a working feature; comparing the pair still
    # catches the failure this check exists for — two macros the policy can
    # select separately that do exactly the same thing.
    n_macros = len(TREECHOP_MACROS)
    sigs = {}
    for i in range(n_macros):
        d = env._macro_to_action(i)
        act_sig = tuple(sorted((k, str(np.asarray(v).tolist()))
                               for k, v in d.items() if np.any(v)))
        sigs[i] = (act_sig, int(env._macro_ticks(i)))
    uniq = len(set(sigs.values()))
    dupes = {}
    for i, sig in sigs.items():
        dupes.setdefault(sig, []).append(i)
    collide = [v for v in dupes.values() if len(v) > 1]
    check("2.6_macros_distinct", uniq == n_macros,
          f"{uniq}/{n_macros} macros are distinct as (action, ticks)"
          + (f"; COLLISIONS {collide}" if collide else ""))

    if a.baseline_rate > 0:
        check("2.7_throughput", rate >= 0.9 * a.baseline_rate,
              f"{rate:.2f} steps/s vs baseline {a.baseline_rate:.2f}")
    else:
        print(f"  [-- ] 2.7_throughput: {rate:.2f} steps/s "
              f"(no baseline given; record this as the baseline)")
    res["steps_per_sec"] = rate

    # 2.8 — the probe->scan->map chain, if spatial memory is on.
    if (cfg.get("spatial") or {}).get("enabled"):
        print("  [-- ] 2.8_occupancy: needs the loop, not the bare env — "
              "checked in Stage 3 from the metrics dump")

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "stage2.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    if X.size:
        np.save(os.path.join(a.out, "stage2_sensors.npy"), X)
    print(f"\nwrote {p}")
    try:
        env.close()
    except Exception:
        pass

    if fails:
        print(f"\nSTAGE 2 FAILED: {fails}")
        return 1
    print("\nSTAGE 2 PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
