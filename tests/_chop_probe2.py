"""CHOP PROBE 2 — isolate WHICH stage of chop->reward is broken.

Probe 1 proved scripted contact chopping yields zero REWARD. Three stages
could be at fault:
  A. block BREAKING (attack-hold semantics dead in the 1.0 port)
  B. item PICKUP (log drops but never enters inventory)
  C. REWARD handler (inventory gains the log, RewardForCollectingItems dead)

This probe watches the INVENTORY directly (custom Treechop spec + Flat-
InventoryObservation) and snapshots POV frames around each chop burst so a
human can see whether trunks visibly break.

  inventory log count rises + reward 0    -> stage C broken (adapter-side
                                              inventory-delta reward fixes it)
  inventory flat, frames show broken logs -> stage B broken
  inventory flat, trunks never break      -> stage A broken

TRAINING HOST ONLY:  xvfb-run -a ./venv_mc/bin/python _chop_probe2.py
"""
import os
import time

import numpy as np

FRAMES = "runlogs/probe2_frames"


def act_of(env, **kw):
    a = env.action_space.noop()
    for k, v in kw.items():
        a[k] = np.array(v, dtype=np.float32) if k == "camera" else v
    return a


def save_frame(obs, name):
    try:
        import imageio.v2 as imageio
        f = np.asarray(obs["pov"], dtype=np.uint8)
        f = np.repeat(np.repeat(f, 4, 0), 4, 1)
        imageio.imwrite(os.path.join(FRAMES, name + ".png"), f)
    except Exception as e:
        print(f"  (frame save failed: {e})")


def main():
    os.makedirs(FRAMES, exist_ok=True)
    from minerl.herobraine.env_specs.treechop_specs import Treechop
    import minerl.herobraine.hero.handlers as handlers

    class TreechopDiag(Treechop):
        def create_observables(self):
            obs = super().create_observables()
            obs.append(handlers.FlatInventoryObservation(["log"]))
            return obs

    print("[probe2] building diagnostic spec + launching Minecraft (~90s)...")
    spec = TreechopDiag()
    env = spec.make()
    obs = env.reset()
    print(f"[probe2] obs keys: {sorted(obs.keys())}")
    inv0 = obs.get("inventory", {})
    print(f"[probe2] initial inventory: { {k: np.asarray(v).tolist() for k, v in inv0.items()} }")

    t0 = time.time()
    total_r, tick = 0.0, 0
    log_count_hist = []

    def logs_of(o):
        inv = o.get("inventory", {})
        v = inv.get("log", 0)
        return int(np.asarray(v).flatten()[0]) if np.size(v) else 0

    def step(a):
        nonlocal total_r, tick, obs
        obs, r, done, _ = env.step(a)
        tick += 1
        total_r += float(r)
        lc = logs_of(obs)
        if not log_count_hist or lc != log_count_hist[-1][1]:
            log_count_hist.append((tick, lc))
            print(f"  [tick {tick:5d}] inventory log -> {lc}  "
                  f"(reward so far {total_r:.0f})")
        return done

    done, cycle = False, 0
    while not done and tick < 3000 and cycle < 12:
        cycle += 1
        walk = act_of(env, forward=1, jump=1 if cycle % 3 == 0 else 0)
        for _ in range(40):
            done = step(walk)
            if done:
                break
        if done:
            break
        save_frame(obs, f"c{cycle:02d}_pre")
        # look slightly down at trunk base, then chop pinned for 150 ticks
        step(act_of(env, camera=[10.0, 0.0], attack=1, forward=1))
        chop = act_of(env, attack=1, forward=1)
        for i in range(150):
            done = step(chop)
            if i == 75:
                save_frame(obs, f"c{cycle:02d}_mid")
            if done:
                break
        save_frame(obs, f"c{cycle:02d}_post")
        if done:
            break
        step(act_of(env, camera=[-10.0, 47.0]))  # level out + new heading

    dt = time.time() - t0
    print(f"[probe2] {tick} ticks in {dt:.0f}s, {cycle} cycles")
    print(f"[probe2] REWARD total: {total_r:.0f}")
    print(f"[probe2] inventory log trajectory: {log_count_hist}")
    print(f"[probe2] frames -> {FRAMES}/")
    env.close()

    final_logs = log_count_hist[-1][1] if log_count_hist else 0
    if final_logs > 0 and total_r == 0:
        print("[probe2] VERDICT: logs ARE collected but reward handler is "
              "DEAD -> fix = adapter-side inventory-delta reward.")
    elif final_logs > 0 and total_r > 0:
        print("[probe2] VERDICT: everything works here?! (spec-dependent "
              "reward bug — diag spec pays, stock spec doesn't)")
    else:
        print("[probe2] VERDICT: no logs ever entered inventory — breaking "
              "or pickup is broken; INSPECT THE FRAMES for crack/hole "
              "evidence to split A vs B.")


if __name__ == "__main__":
    main()
