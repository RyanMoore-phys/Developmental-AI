"""CHOP PROBE 3 — same scripted contact chop, but in HumanSurvival
(the VPT env family, where held-attack tree chopping demonstrably works
with this exact MCP-Reborn input bridge).

  logs collected here  -> engine fine; Treechop spec is the broken piece
                          -> fix = custom HumanSurvival-style spec (forest
                             spawn, axe, inventory obs, adapter reward).
  nothing here either  -> engine/input-level failure (mouse grab / press
                          semantics) -> patch at the client level.

HumanSurvival: random spawn (any biome!), no axe. Barehanded oak = 60
consecutive ticks; we hold 200 per burst — plenty IF a tree is reachable.
Random biome risk is handled by walking far (~12 cycles) and jumping; a
plains spawn may still miss trees — treat 'no trees seen in frames' as
inconclusive rather than negative, so we also save frames every cycle.

TRAINING HOST ONLY:  xvfb-run -a ./venv_mc/bin/python _chop_probe3.py
"""
import os
import time

import numpy as np

FRAMES = "runlogs/probe3_frames"


def act_of(env, **kw):
    a = env.action_space.noop()
    for k, v in kw.items():
        a[k] = np.array(v, dtype=np.float32) if k == "camera" else v
    return a


def save_frame(obs, name):
    try:
        import imageio.v2 as imageio
        f = np.asarray(obs["pov"], dtype=np.uint8)
        # HumanSurvival POV is bigger; save as-is
        imageio.imwrite(os.path.join(FRAMES, name + ".png"), f)
    except Exception as e:
        print(f"  (frame save failed: {e})")


def main():
    os.makedirs(FRAMES, exist_ok=True)
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

    print("[probe3] building HumanSurvival + launching Minecraft (~90s)...")
    spec = HumanSurvival()
    env = spec.make()
    obs = env.reset()
    print(f"[probe3] obs keys: {sorted(obs.keys())}")

    def logs_of(o):
        inv = o.get("inventory", {})
        # any *log* item counts (oak_log, birch_log, ... or legacy 'log')
        total = 0
        for k, v in inv.items():
            if "log" in k:
                total += int(np.asarray(v).flatten()[0])
        return total

    print(f"[probe3] initial logs in inventory: {logs_of(obs)}")

    t0 = time.time()
    tick, total_r = 0, 0.0
    hist = [(0, logs_of(obs))]

    def step(a):
        nonlocal obs, tick, total_r
        obs, r, done, _ = env.step(a)
        tick += 1
        total_r += float(r)
        lc = logs_of(obs)
        if lc != hist[-1][1]:
            hist.append((tick, lc))
            print(f"  [tick {tick:5d}] LOGS -> {lc}")
        return done

    done, cycle = False, 0
    while not done and tick < 4200 and cycle < 14:
        cycle += 1
        walk = act_of(env, forward=1, jump=1 if cycle % 2 == 0 else 0,
                      sprint=1)
        for _ in range(60):
            done = step(walk)
            if done:
                break
        if done:
            break
        save_frame(obs, f"c{cycle:02d}_pre")
        step(act_of(env, camera=[10.0, 0.0], attack=1, forward=1))
        chop = act_of(env, attack=1, forward=1)
        for i in range(200):
            done = step(chop)
            if i == 100:
                save_frame(obs, f"c{cycle:02d}_mid")
            if done:
                break
        save_frame(obs, f"c{cycle:02d}_post")
        if done:
            break
        step(act_of(env, camera=[-10.0, 43.0]))

    dt = time.time() - t0
    final = hist[-1][1]
    print(f"[probe3] {tick} ticks in {dt:.0f}s, {cycle} cycles")
    print(f"[probe3] reward total (HumanSurvival pays none): {total_r:.0f}")
    print(f"[probe3] log trajectory: {hist}")
    env.close()
    if final > 0:
        print("[probe3] VERDICT: HumanSurvival COLLECTS LOGS — engine + "
              "attack-hold WORK; Treechop spec is the broken piece. Fix: "
              "custom HumanSurvival-style env (forest, axe, inventory-delta "
              "reward in the adapter).")
    else:
        print("[probe3] VERDICT: no logs in HumanSurvival either — engine/"
              "input-level failure; inspect frames (trees seen at all?) "
              "before concluding.")


if __name__ == "__main__":
    main()
