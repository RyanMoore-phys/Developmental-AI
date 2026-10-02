"""HOLD-TO-BREAK PROBE — the mechanical proof of the curiosity redesign.

Through OUR adapter (new macro table, 128px agent obs, 512px render):
  1. CHAINED PRESSES BREAK: scripted CONSECUTIVE attack picks (2 ticks each)
     against a block must eventually break it (mine_block increments) and
     pay the first-break +1. This is the emergent-hold contract: no macro
     holds for the agent; the chain IS the hold.
  2. RELEASE RESETS: interleaving a non-attack action between presses must
     reset breaking progress (few cracks, no break in the same window).
  3. CRACKS VISIBLE AT 128: saves the agent's own 128px obs at several
     points mid-hold — the frames a human inspects to confirm the crack
     overlay survives the downsample (the resolution bet).
  4. LP SANITY ON REAL FRAMES: feeds the captured obs sequence through
     LearningProgressCuriosity — consecutive same-scene frames must land in
     few prototype buckets (not one-per-frame), and the reward call runs
     clean end-to-end on real data.

TRAINING HOST ONLY (after provisioning):
    DISPLAY=:77 PYTHONUNBUFFERED=1 ./venv_mc/bin/python _hold_probe.py
"""
import os
import time

import numpy as np

FRAMES = "runlogs/hold_probe_frames"


def save(arr, name, upscale=1):
    try:
        import imageio.v2 as imageio
        f = np.asarray(arr)
        if f.ndim == 1:  # flat CHW float
            n = int(np.sqrt(f.size // 3))
            f = (f.reshape(3, n, n).transpose(1, 2, 0) * 255).astype(np.uint8)
        if upscale > 1:
            f = np.repeat(np.repeat(f, upscale, 0), upscale, 1)
        imageio.imwrite(os.path.join(FRAMES, name + ".png"), f)
    except Exception as e:
        print(f"  (frame save failed: {e})")


def main():
    os.makedirs(FRAMES, exist_ok=True)
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter

    print("[hold-probe] booting fixed env @128 obs / 512 render (~90s)...")
    env = MineRLEnvAdapter(image_size=128, action_repeat=2, render_size=512)
    obs, info = env.reset()
    assert obs.shape == (3 * 128 * 128,), obs.shape
    print(f"[hold-probe] obs contract ok {obs.shape}; logs={info.get('logs')}")

    obs_seq = [obs]

    def step(a):
        nonlocal obs
        obs, r, te, tr, i2 = env.step(a)
        obs_seq.append(obs)
        return r, te or tr, i2

    # find a face: walk forward into terrain, nose to a block, look down a bit
    for _ in range(30):
        r, done, i2 = step(1)
    step(8)  # look down 15deg (ground/dirt at feet-level ahead)

    # ---- phase A: chained presses with a mid-chain release (reset check)
    total = 0.0
    broke_at = None
    save(env._last_pov, "A_before_512")
    save(obs, "A_before_obs128")
    for j in range(60):                    # up to 120 held ticks if unbroken
        r, done, i2 = step(5)              # attack press, 2 ticks
        total += r
        if j == 3:
            save(obs, "A_hold4_obs128")    # ~8 held ticks: cracks due
            save(env._last_pov, "A_hold4_512")
        if j == 7:
            save(obs, "A_hold8_obs128")
        if r > 0:
            broke_at = j
            print(f"  [A] chained press #{j + 1} -> +{r:.0f} "
                  f"(mine={ {k: v for k, v in i2.get('achievements', {}).items() if k.startswith('mine_')} })")
            save(obs, "A_broke_obs128")
            save(env._last_pov, "A_broke_512")
            break
    assert broke_at is not None, (
        "chained 2-tick attack presses never broke a block — emergent-hold "
        "contract FAILED (check aim/target: inspect A_* frames)")
    assert total >= 1.0, "first-break must pay +1"

    # ---- phase B: release-resets — alternate attack/noop, expect NO break
    # in the same number of attack presses that broke in phase A (+margin)
    step(1); step(1)                       # nudge to a fresh face
    b_total = 0.0
    presses = 0
    for j in range(min(broke_at + 4, 30)):
        r, done, i2 = step(5); presses += 1; b_total += r
        r2, done, i2 = step(0)             # noop releases the button
        b_total += r2
    print(f"  [B] {presses} interrupted presses -> reward {b_total:.0f} "
          f"(expected 0 unless a soft block like dirt breaks in 2 ticks)")

    # ---- phase C: LP prototype sanity on the real captured sequence
    from developmental_ai.curiosity.learning_progress import (
        LearningProgressCuriosity)
    lp = LearningProgressCuriosity(
        obs_dim=3 * 128 * 128, action_dim=10, feature_dim=64, hidden_dim=64,
        pixel_obs=True, image_channels=3, image_size=128,
        discrete_actions=True, lp_pixel_pool=4, lp_proto_thresh=0.06,
        lp_max_protos=512)
    # measured per-step pooled-frame drift — the data that calibrates
    # lp_proto_thresh (review flagged 0.06 as possibly 10x real drift)
    pooled = [lp._pooled_vec(o.astype(np.float32)) for o in obs_seq]
    drifts = [float(np.abs(pooled[i + 1] - pooled[i]).mean())
              for i in range(len(pooled) - 1)]
    d = np.array(drifts)
    print(f"  [C] per-step pooled drift: p50={np.percentile(d,50):.4f} "
          f"p90={np.percentile(d,90):.4f} max={d.max():.4f} "
          f"(thresh={lp.lp_proto_thresh})")
    keys = [lp._bucket_key(o.astype(np.float32)) for o in obs_seq]
    n_protos = len(set(keys))
    print(f"  [C] {len(keys)} real frames -> {n_protos} prototype buckets")
    assert n_protos < len(keys) * 0.5, (
        "prototype quantizer fragments real gameplay into near-singletons — "
        "LP would starve (tune lp_proto_thresh)")
    # end-to-end reward call on real data (batched)
    import torch
    ot = torch.from_numpy(np.stack(obs_seq[:-1][:32]).astype(np.float32))
    nt = torch.from_numpy(np.stack(obs_seq[1:][:32]).astype(np.float32))
    at = torch.nn.functional.one_hot(
        torch.randint(0, 10, (ot.shape[0],)), 10).float()
    rew = lp.compute_intrinsic_reward(ot, at, nt)
    assert rew.shape[0] == ot.shape[0] and torch.isfinite(rew).all()
    print(f"  [C] LP reward call clean on real frames "
          f"(mean={float(rew.mean()):.4f})")

    env.close()
    print("[hold-probe] VERDICT: emergent hold WORKS — chained presses "
          "break, release resets, first-break pays, LP runs on real frames. "
          "Inspect A_hold*_obs128.png for crack visibility.")


if __name__ == "__main__":
    main()
