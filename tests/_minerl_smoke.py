"""Minecraft integration smoke — POD ONLY, under xvfb, in venv_mc.

    xvfb-run -a -s "-screen 0 640x480x24" ./venv_mc/bin/python _minerl_smoke.py

  1. Adapter API: gymnasium spaces/5-tuple, flat CHW float [0,1] obs of the
     right dim, macros expand to valid MineRL action dicts, CHW round-trip.
  2. make_env("MineRLTreechop-v0"): full-stack dims + a real rollout.
  3. Video: an MP4 of real Minecraft gameplay is written and non-trivial.

NOTE: each env reset launches a Minecraft client (~90s), so this smoke uses
ONE env instance and few steps.
"""
import glob
import os
import shutil

import numpy as np

VID = "/tmp/mc_smoke_videos"


def test_adapter_and_video():
    from developmental_ai.environments.minerl_env import (
        MineRLEnvAdapter, TREECHOP_MACROS)
    from developmental_ai.environments.video_recorder import VideoRecorder
    import gymnasium as gym

    shutil.rmtree(VID, ignore_errors=True)
    raw = MineRLEnvAdapter()
    assert isinstance(raw.observation_space, gym.spaces.Box)
    assert isinstance(raw.action_space, gym.spaces.Discrete)
    assert raw.action_space.n == len(TREECHOP_MACROS)
    assert raw.observation_space.shape == (3 * 64 * 64,)

    # macros must expand to valid action dicts
    for i in range(raw.action_space.n):
        a = raw._macro_to_action(i)
        assert set(a).issuperset({"attack", "forward", "camera"}), a
    print(f"  adapter ok: Box({raw.observation_space.shape[0]}) / "
          f"Discrete({raw.action_space.n}), macros valid")

    env = VideoRecorder(raw, out_dir=VID, source="obs", image_hw=(64, 64),
                        channels=3, upscale=4, head=1, every=0, fps=10)
    obs, info = env.reset()
    assert obs.shape == (12288,) and obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    # CHW round-trip against the raw POV
    chw = obs.reshape(3, 64, 64)
    hwc = (chw.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    assert np.array_equal(hwc, raw._last_pov), "CHW layout mismatch"
    print("  obs ok: flat CHW float32 [0,1], POV round-trip exact")

    total = 0.0
    for i in range(120):
        a = 6 if i % 3 == 0 else (1 if i % 3 == 1 else 4)  # chop/walk/turn
        obs, r, term, trunc, info = env.step(a)
        total += r
        assert obs.shape == (12288,)
        if term or trunc:
            obs, info = env.reset()
    env.finalize()
    vids = glob.glob(f"{VID}/*.mp4")
    assert vids, "no gameplay video written"
    size = os.path.getsize(vids[0])
    assert size > 5000, f"video suspiciously small ({size}B)"
    print(f"  rollout ok: 120 steps, reward {total:.1f} logs, "
          f"video {os.path.basename(vids[0])} ({size//1024}KB)")
    raw.close()


def test_make_env_stack():
    from developmental_ai.environments.wrappers import make_env
    env, _ = make_env("MineRLTreechop-v0", curriculum=False)
    assert env.obs_dim == 12288 and env.action_dim == 10 and env.is_discrete
    obs, info = env.reset()
    assert obs.shape == (12288,)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all()
    print(f"  make_env ok: dims (12288/10/discrete), 20 steps clean")
    env.close()


if __name__ == "__main__":
    for fn in (test_adapter_and_video, test_make_env_stack):
        print(f"[minerl-smoke] {fn.__name__}")
        fn()
    print("[minerl-smoke] ALL PASS")
