#!/usr/bin/env python3
"""Where does the step time actually go? (roadmap W2.7)

The loop already accumulates per-phase wall clock in `_phase_acc` via
`_phase_mark`, but only PRINTS it at a segment boundary a short run never
reaches — which is why this measurement has been "blocked" for two waves
while the instrument sat there fully built.

This reads the accumulator directly. No new instrumentation, no guessing.

    PYTHONPATH=. python scripts/host_profile_step.py --steps 120
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--config", default="configs/minecraft_skybot.yaml")
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--out", default="runlogs/profile")
    ap.add_argument("--no-vlm", action="store_true",
                    help="disable the symbolizer to isolate its cost")
    ap.add_argument("--no-async", action="store_true",
                    help="train the world model on the ACTING thread. The "
                         "A/B for buffer-lock contention: sample_sequences "
                         "holds ReplayBuffer._lock for a 25MB->100MB gather, "
                         "and add() blocks on the same lock.")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    cfg.setdefault("parallel_envs", {})["enabled"] = True
    cfg["parallel_envs"]["num_envs"] = max(2, a.envs)
    cfg.setdefault("environment", {})["remote_server"] = None
    cfg["environment"]["remote_server_scope"] = "primary"
    if a.no_vlm:
        cfg.setdefault("symbolic_grounding", {})["enabled"] = False
    if a.no_async:
        cfg.setdefault("async_wm", {})["enabled"] = False
        cfg.setdefault("async_replay", {})["enabled"] = False

    from developmental_ai.core.developmental_loop import DevelopmentalAI
    agent = DevelopmentalAI(cfg)
    t0 = time.time()
    agent.run(total_timesteps=a.steps, log_interval=10 ** 9, verbose=0)
    wall = time.time() - t0

    acc = dict(getattr(agent, "_phase_acc", None) or {})
    n = max(1, int(getattr(agent, "total_timesteps", a.steps)))
    step_ms = 1000.0 * wall / n

    rows, summed = [], 0.0
    for ph in getattr(agent, "_PHASES", ()):
        ms = 1000.0 * acc.get(ph, 0.0) / n
        summed += ms
        rows.append((ph, ms))
    rows.sort(key=lambda r: -r[1])
    unacc = step_ms - summed

    print(f"\n{n} steps in {wall:.0f}s  =  {n / wall:.2f} steps/s  "
          f"({step_ms:.0f} ms/step)\n")
    for ph, ms in rows:
        print(f"  {ph:12s} {ms:8.1f} ms  {100.0 * ms / max(step_ms, 1e-9):5.1f}%")
    # UNACCOUNTED IS PRINTED ON PURPOSE: if the phases do not sum to the step
    # time, the gap is real work nobody is timing, and pretending the buckets
    # are exhaustive would hide it.
    print(f"  {'UNACCOUNTED':12s} {unacc:8.1f} ms  "
          f"{100.0 * unacc / max(step_ms, 1e-9):5.1f}%")

    # ---- THE store_buf SPLIT: waiting vs working ----------------------
    # sample_sequences holds ReplayBuffer._lock through a 25 MB -> 100 MB
    # gather. If `wait` dominates `held`, the LEARNER IS STARVING THE ACTOR
    # and the fix is to stop gathering under the lock — not to make the
    # write faster, which would change nothing.
    buf = getattr(agent, "replay_buffer", None)
    streams = getattr(buf, "streams", None) or ([buf] if buf else [])
    wait = sum(float(getattr(b, "lock_wait_s", 0.0)) for b in streams)
    held = sum(float(getattr(b, "lock_held_s", 0.0)) for b in streams)
    if wait or held:
        print(f"\n  buffer lock   wait {1000.0 * wait / n:7.1f} ms/step   "
              f"held {1000.0 * held / n:7.1f} ms/step   "
              f"({100.0 * wait / max(wait + held, 1e-9):.0f}% of buffer time "
              f"is WAITING)")

    os.makedirs(a.out, exist_ok=True)
    tag = ("novlm" if a.no_vlm else
           "noasync" if a.no_async else "full")
    with open(os.path.join(a.out, f"phases_{tag}.json"), "w") as f:
        json.dump({"steps": n, "wall_s": wall, "step_ms": step_ms,
                   "phases_ms": dict(rows), "unaccounted_ms": unacc,
                   "lock_wait_ms": 1000.0 * wait / n,
                   "lock_held_ms": 1000.0 * held / n}, f, indent=2)
    try:
        agent.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
