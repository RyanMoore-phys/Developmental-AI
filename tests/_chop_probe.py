"""CHOP MECHANISM PROBE — answers one question empirically:

    Can holding attack against a tree in MineRL 1.0 Treechop actually
    break a log and produce +1 reward AT ALL?

Three zero-reward runs (800k+ ticks) are consistent with BOTH "exploration
never aimed at a trunk" AND "the mechanism is broken" (attack not treated
as held across ticks / RewardForCollectingItems dead in the 1.0 port).
This script removes aim-luck entirely with SCRIPTED contact chopping on the
RAW MineRL env (no macros, no learning):

  cycle:  walk forward 40 ticks (into whatever is ahead — forest biome, so
          usually a tree pins the agent), then hold attack 150 ticks with a
          slight downward pitch first (trunk base) then level (trunk body),
          then turn ~45° and repeat. The spawn axe (iron) breaks oak in ~13
          ticks, so ANY tick window with the crosshair on a trunk should pay.

Prints every reward event with its tick. Verdict:
  reward > 0  -> mechanism WORKS; failures are exploration/guidance.
  reward == 0 -> mechanism (or reward wiring) is BROKEN — env-level bug.

TRAINING HOST ONLY, ~6-8 min:  xvfb-run -a ./venv_mc/bin/python _chop_probe.py
"""
import time

import numpy as np


def act_of(env, **kw):
    a = env.action_space.noop()
    for k, v in kw.items():
        a[k] = np.array(v, dtype=np.float32) if k == "camera" else v
    return a


def main():
    import gym as old_gym
    import minerl  # noqa: F401

    print("[probe] launching Minecraft (~90s)...")
    env = old_gym.make("MineRLTreechop-v0")
    env.reset()
    t0 = time.time()

    total, tick, events = 0.0, 0, []

    def step(a):
        nonlocal total, tick
        _, r, done, _ = env.step(a)
        tick += 1
        if r > 0:
            total += r
            events.append((tick, float(r)))
            print(f"  [tick {tick:5d}] +{r:.0f} REWARD  (total={total:.0f})")
        return done

    done = False
    cycle = 0
    # ~5600 ticks of scripted contact chopping (episode cap is 8000)
    while not done and tick < 5600:
        cycle += 1
        # 1) close distance: walk (with jump every 3rd cycle for ledges)
        walk = act_of(env, forward=1, jump=1 if cycle % 3 == 0 else 0)
        for _ in range(40):
            done = step(walk)
            if done:
                break
        if done:
            break
        # 2) chop: pitch slightly down at trunk base for 75 ticks, then level
        #    for 75 — while STILL holding forward (stay pinned to the trunk).
        for pitch in (10.0, -10.0):
            done = step(act_of(env, camera=[pitch, 0.0],
                               attack=1, forward=1))
            if done:
                break
            chop = act_of(env, attack=1, forward=1)
            for _ in range(74):
                done = step(chop)
                if done:
                    break
            if done:
                break
        if done:
            break
        # 3) new heading
        step(act_of(env, camera=[0.0, 47.0]))

    dt = time.time() - t0
    print(f"[probe] {tick} ticks in {dt:.0f}s ({tick/max(dt,1):.1f} t/s), "
          f"{cycle} contact-chop cycles")
    print(f"[probe] TOTAL REWARD: {total:.0f}  events={events}")
    env.close()
    if total > 0:
        print("[probe] VERDICT: chop mechanism WORKS — zero-reward runs were "
              "an exploration/guidance failure, not an env bug.")
    else:
        print("[probe] VERDICT: NO reward from scripted contact chopping — "
              "the chop/reward mechanism itself is broken at the env level "
              "(attack-hold semantics or RewardForCollectingItems).")


if __name__ == "__main__":
    main()
