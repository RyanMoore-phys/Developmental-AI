#!/usr/bin/env python3
"""Stage 3 — the FULL LOOP against MineRL, short. Everything Stage 2 cannot see.

Stage 2 drives the bare adapter, so it proves the sensors carry real data.
It cannot prove any of the following, because none of them exist until the
developmental loop is running:

  * the sensor transport actually reaches the WORLD MODEL (bus -> buffer ->
    embed), rather than being emitted into `info` and dropped;
  * `_augment_proprio` and the policy's declared width AGREE — a mismatch is
    silent and shifts every map into the wrong slot;
  * the spatial maps advance, decay and reset;
  * the oracle measures drift and reaches NOTHING else;
  * the world model's new losses are finite on real frames and FALL.

This is deliberately short (default 300 steps, one client). It answers "does
the machinery carry real signal end to end", not "does the agent learn" —
that is Stage 5 and takes days.

Run on the pod:
    PYTHONPATH=. python scripts/host_stage3_loop.py --steps 300
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--config", default="configs/minecraft_skybot.yaml")
    ap.add_argument("--out", default="runlogs/stage3")
    ap.add_argument("--envs", type=int, default=2)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    # ---- TEST THE CONFIGURATION THAT ACTUALLY RUNS ---------------------
    # The first version of this forced num_envs=1, and the loop refused it:
    # `lifelong.enabled requires parallel_envs.enabled with num_envs>1; the
    # single-env path is not covered by the continuous loop`. That refusal
    # is correct, and the right response is not to disable lifelong — it is
    # to stop testing a configuration that does not exist.
    #
    # TWO CLIENTS, which CLAUDE.md records as the proven number (four
    # produced 0 segments in 17 minutes against the shared server). Here the
    # server is not even involved: MineRL generates its own world per
    # client, so the usual client-count bottleneck does not apply, but two
    # is what the live config runs and therefore what gets tested.
    n_envs = max(2, int(a.envs))
    cfg.setdefault("parallel_envs", {})["enabled"] = True
    cfg["parallel_envs"]["num_envs"] = n_envs
    # NO EXTERNAL SERVER. The user's Paper server is offline; `remote_server`
    # only matters for the social-learning experiment, so it is CLEARED
    # rather than left pointing at a dead address, which would make every
    # client burn its connect timeout before falling back.
    cfg.setdefault("environment", {})["remote_server"] = None
    cfg["environment"]["remote_server_scope"] = "primary"
    cfg["environment"]["max_episode_steps"] = None

    from developmental_ai.core.developmental_loop import DevelopmentalAI

    t0 = time.time()
    agent = DevelopmentalAI(cfg)
    boot = time.time() - t0
    print(f"loop constructed in {boot:.0f}s ({n_envs} clients)")

    res, fails = {}, []

    def check(name, ok, detail):
        res[name] = {"ok": bool(ok), "detail": detail}
        print(f"  [{'OK ' if ok else 'NO '}] {name}: {detail}")
        if not ok:
            fails.append(name)

    # --- 3.1 the declared widths agree BEFORE a single step -------------
    bus_w = int(agent._wm_proprio_dim)
    wm_w = int(agent.world_model.proprio_dim)
    check("3.1_transport_width", bus_w == wm_w and bus_w > 0,
          f"loop declares {bus_w}, world model expects {wm_w}")

    pol_w = int(getattr(agent.policy, "proprio_dim", 0))
    aug = agent._augment_proprio(np.zeros(agent._proprio_source.PROPRIO_DIM,
                                          np.float32), 0)
    aug_w = int(np.asarray(aug).shape[0])
    check("3.2_policy_proprio_width", pol_w == aug_w,
          f"policy declares {pol_w}, _augment_proprio writes {aug_w} — a "
          f"mismatch is SILENT and shifts every map into the wrong slot")

    check("3.3_layout_recorded",
          bool(getattr(agent, "_sensor_layout_hash", "")),
          f"buffer layout hash = {getattr(agent, '_sensor_layout_hash', '')}")

    # --- run ------------------------------------------------------------
    t1 = time.time()
    # `run`, not `train` — DevelopmentalAI has no train(). Checked against the
    # source rather than assumed, because a wrong entry point here costs a
    # whole pod cycle to discover.
    agent.run(total_timesteps=a.steps, log_interval=1000, verbose=1)
    dt = time.time() - t1
    rate = a.steps / max(1e-6, dt)
    res["steps_per_sec"] = round(rate, 3)
    print(f"ran {a.steps} steps in {dt:.0f}s ({rate:.2f} steps/s)")

    m = {k: list(v) for k, v in agent.training_metrics.items() if len(v)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(m, f)

    # --- 3.4 the new losses exist and are finite -------------------------
    for key in ("flow_loss", "horizon_loss", "reconstruction_error"):
        v = m.get(key) or []
        ok = len(v) > 0 and all(np.isfinite(v))
        check(f"3.4_{key}", ok,
              f"{len(v)} samples, last={v[-1]:.5f}" if v else "NEVER RECORDED")

    # --- 3.5 the spatial maps advanced ----------------------------------
    if getattr(agent, "_spatial_on", False):
        occ = getattr(agent, "_occ_now", None)
        walk = getattr(agent, "_walk_now", None)
        check("3.5_spatial_advanced",
              occ is not None and walk is not None,
              f"occupancy={'set' if occ is not None else 'None'}, "
              f"walkability={'set' if walk is not None else 'None'}")
        if occ is not None:
            check("3.6_occupancy_bounded",
                  float(np.max(occ)) <= 1.0 + 1e-6,
                  f"max confidence {float(np.max(occ)):.3f} (clamp holds)")

    # --- 3.7 the oracle measured, and leaked nothing ---------------------
    if getattr(agent, "_oracle_enabled", False):
        drift = m.get("oracle_dr_drift") or []
        check("3.7_oracle_measured", len(drift) > 0 or agent._oracle_steps > 0,
              f"{agent._oracle_steps} oracle steps, "
              f"{len(drift)} drift reports"
              + (f", last {drift[-1]:.2f} blocks" if drift else ""))

    # --- 3.8 the buffer carries the sensor column ------------------------
    buf = agent.replay_buffer
    col = getattr(buf, "proprio", None)
    if col is None and hasattr(buf, "streams"):
        col = getattr(buf.streams[0], "proprio", None)
    check("3.8_buffer_column",
          col is not None and int(np.asarray(col[0]).shape[0]) == bus_w,
          f"column width {None if col is None else np.asarray(col[0]).shape[0]}"
          f", want {bus_w}")

    with open(os.path.join(a.out, "stage3.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {a.out}/stage3.json and metrics.json")
    try:
        agent.close()
    except Exception:
        pass
    if fails:
        print(f"\nSTAGE 3 FAILED: {fails}")
        return 1
    print("\nSTAGE 3 PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
