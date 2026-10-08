"""Minecraft (MineRL Treechop) training run — with gameplay VIDEO.

Real Minecraft via the MineRL adapter, driven by the full developmental stack
(CNN world model + dreaming + DISCOVERED goal channel + per-achievement skill
minting). Discovered goals are the only option here — Minecraft has no
achievements dict — and they're the natural fit: Treechop's +1-per-log reward
spikes are exactly what the discovered clusterer consumes. (They also BEAT
the oracle goal space on Crafter; see rung 10.)

Writes watchable MP4s of what the agent actually sees/does:
  * the first `head` episodes (an untrained agent flailing),
  * every `every`-th episode (development over time),
  * every new best-reward episode (the highlights).

MUST run headless under xvfb:
    xvfb-run -a -s "-screen 0 640x480x24" \
      ./venv_mc/bin/python -u run_minecraft.py --timesteps 150000
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys

import numpy as np
import yaml

from developmental_ai.core.stats import save_results

# ---- MAKE logger.info VISIBLE (2026-07-27) --------------------------------
# THIS PROJECT HAD NO `logging.basicConfig` ANYWHERE. With no handler on the
# root logger, Python falls back to `lastResort`, which emits WARNING and
# above only — so every `logger.info` in the entire codebase has been silently
# discarded for the life of every run. Measured on a complete 34-segment log:
# "Minted achievement skill" 0 hits, "Skills-as-options ACTIVE" 0, "option
# slot <- ..." 0, "Re-distilled" 0 — while every `print()` line appeared 34
# times. Mints, option binds, re-distills, warm-start outcomes and skill
# refusals were all invisible, which is a large part of why diagnosing this
# system has repeatedly required reconstructing events from side effects.
#
# stdout (not stderr) because launch_lifelong.sh redirects both to the run
# log; `force=True` so an import that already touched logging cannot win.
logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, force=True,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S")
# MineRL/urllib3/matplotlib are extremely chatty at INFO and would bury the
# agent's own diagnostics in a multi-week log.
for _noisy in ("minerl", "urllib3", "matplotlib", "PIL", "coloredlogs",
               "httpx", "httpcore", "numba", "gym", "gymnasium"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--config", default="configs/minecraft.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="minecraft_results")
    # LIFELONG: run as a never-ending stream (no step budget). Ends only via
    # the stop-file (touch <lifelong.stop_file>) or SIGTERM/SIGINT. Turns on
    # lifelong.enabled + lifelong.forever regardless of the config defaults.
    ap.add_argument("--forever", action="store_true",
                    help="continuous lifelong run (no timestep budget)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["seed"] = args.seed
    if args.forever:
        cfg.setdefault("lifelong", {})
        cfg["lifelong"]["enabled"] = True
        cfg["lifelong"]["forever"] = True
        print("[mc] LIFELONG forever-mode ON — ends on stop-file / SIGTERM")

    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.environments.video_recorder import VideoRecorder

    print(f"[mc] building agent ({args.config})... first Minecraft launch "
          f"takes ~90s", flush=True)
    agent = DevelopmentalAI(config=copy.deepcopy(cfg))
    print(f"[mc] obs_dim={agent.obs_dim} action_dim={agent.action_dim} "
          f"pixel={agent.pixel_obs} device={agent.device} "
          f"goals={type(agent.broadcaster).__name__}", flush=True)

    # ---- wrap the env in the video recorder ----
    # With environment.render_size set, the game renders natively at that
    # resolution and render() returns the FULL Minecraft frame — record that
    # (source="render", no upscale needed). Otherwise record the agent's own
    # 64px POV as before.
    vcfg = cfg.get("video", {})
    if vcfg.get("enabled", True):
        rsize = int(cfg.get("environment", {}).get("render_size", 0) or 0)
        hires = rsize > agent.image_size
        agent.env = VideoRecorder(
            agent.env,
            out_dir=vcfg.get("out_dir", "runlogs/mc_videos"),
            source="render" if hires else "obs",
            image_hw=(agent.image_size, agent.image_size),
            channels=3,
            upscale=1 if hires else int(vcfg.get("upscale", 6)),
            head=int(vcfg.get("head", 3)), every=int(vcfg.get("every", 25)),
            fps=int(vcfg.get("fps", 12)),
            # Lifelong stream never resets -> window-flush every max_frames
            # steps (else: zero videos + unbounded frame-buffer RAM growth).
            # 1500 steps ~ 2 min of POV at 12fps. Episodic runs: 0 (off).
            max_frames=int(vcfg.get(
                "max_frames",
                1500 if cfg.get("lifelong", {}).get("enabled") else 0)))
        print(f"[mc] recording videos -> {vcfg.get('out_dir')} "
              f"({'full ' + str(rsize) + 'px render' if hires else 'agent POV'})",
              flush=True)

    # ---- measurement tap: TRUE logs collected (inventory delta), NOT total
    # reward. The step reward now bundles first-break achievements (+1/type)
    # and vision-scaffold shaping, so summing out[1] would overcount "logs".
    # Read the adapter's own log counter (info['logs']) instead.
    # info['logs'] is the inventory count WITHIN an episode (resets to 0 each
    # episode). Sum per-episode gains: add positive deltas, resync on reset.
    logs = [0]        # cumulative logs across the whole run
    _prev = [0]       # last-seen in-episode count
    orig_step = agent.env.step

    def tap(action):
        out = orig_step(action)
        info = out[4] if len(out) >= 5 else {}
        if isinstance(info, dict) and "logs" in info:
            cur = int(info["logs"])
            if cur > _prev[0]:
                logs[0] += cur - _prev[0]
            _prev[0] = cur           # drop (episode reset) just resyncs
        return out

    agent.env.step = tap

    # forever-mode passes None so the loop runs until stop-file/SIGTERM
    # (the LifelongController resolves the budget); otherwise the CLI budget.
    _budget = None if args.forever else args.timesteps
    # close() on EVERY exit path (clean STOP, budget reached, exception):
    # it stops the async WM trainer before CUDA/env teardown. See
    # _close_bounded for the live incident.
    try:
        agent.run(total_timesteps=_budget, verbose=1, log_interval=1)

        if hasattr(agent.env, "finalize"):
            agent.env.finalize()

        rewards = list(agent.training_metrics["episode_reward"]) or [0.0]
        q = max(1, len(rewards) // 4)
        quarters = [float(np.mean(rewards[i * q:(i + 1) * q])) for i in range(4)]
        res = {
            "total_logs_chopped": logs[0],
            "episodes": agent.total_episodes,
            "quarter_rewards": quarters,
            "best_episode_reward": float(max(rewards)),
            "skills_minted": agent.skill_bank.get_stats()["total_skills"],
            "goal_stats": agent.broadcaster.stats,
        }
        print("\n===== MINECRAFT (Treechop) =====", flush=True)
        print(f"  logs chopped total: {res['total_logs_chopped']:.0f}")
        print(f"  episodes: {res['episodes']}  best episode: "
              f"{res['best_episode_reward']:.0f} logs")
        print(f"  reward quarters: {[round(x, 2) for x in quarters]}")
        print(f"  skills minted: {res['skills_minted']}  "
              f"goal slots: {res['goal_stats']['n_slots']}")
        path = save_results(args.out, res, extra={"protocol": vars(args)})
        print(f"results -> {path}", flush=True)
    finally:
        _close_bounded(agent)


def _close_bounded(agent, timeout_s: float = 150.0) -> None:
    """Call agent.close() on EVERY exit, bounded in time (2026-10-07).

    LIVE INCIDENT: close() -- which stops and joins the async WM trainer
    BEFORE env/CUDA teardown -- was never called by anything. Every exit
    (including a clean runlogs/STOP) killed that daemon thread mid optimizer
    step at interpreter shutdown, and both 2026-10-05 runs ended in
    `terminate called without an active exception` (a C++ abort -> SIGABRT
    -> a core dump of a ~8-10 GB CUDA process on a 16 GB box). The host
    froze 22 s after the second one. close()'s own docstring names the
    failure. Run it on a worker thread so a hung MineRL client inside
    env.close() cannot keep the process from exiting: the trainer is
    stopped FIRST inside close(), so that is the part that must finish.
    """
    import threading

    err = []

    def _c():
        try:
            agent.close()
        except Exception as e:          # never mask the run's own outcome
            err.append(e)
    t = threading.Thread(target=_c, name="agent-close", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        print(f"agent.close() still running after {timeout_s:.0f}s "
              f"(a client is probably hung); exiting anyway", flush=True)
    elif err:
        print(f"agent.close() raised {type(err[0]).__name__}: {err[0]}",
              flush=True)
    else:
        print("agent closed cleanly (async WM trainer joined)", flush=True)


if __name__ == "__main__":
    main()
