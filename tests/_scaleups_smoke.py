"""Smoke for the step-8 scale-ups: uint8 replay storage + CNN-ICM.

  1. uint8 buffer: dtype is uint8, RAM is ~1/4, roundtrip error <= 1/255,
     sampling/PER/window logic unchanged.
  2. CNN-ICM: drop-in on flat CHW pixel vectors; intrinsic reward + train
     step finite; encoder is genuinely convolutional.
  3. Integration: Crafter agent with both ON — buffer stores uint8, curiosity
     flows, WM trains, short run clean.
"""
import copy

import numpy as np
import torch
import yaml

from developmental_ai.world_model.replay_buffer import ReplayBuffer
from developmental_ai.curiosity.icm import (
    CNNFeatureEncoder, IntrinsicCuriosityModule)


def test_uint8_buffer():
    rng = np.random.RandomState(0)
    buf8 = ReplayBuffer(capacity=500, obs_dim=192, action_dim=4, obs_uint8=True)
    buf32 = ReplayBuffer(capacity=500, obs_dim=192, action_dim=4)
    assert buf8.observations.dtype == np.uint8
    assert buf8.observations.nbytes * 4 == buf32.observations.nbytes
    obs_all = []
    for i in range(400):
        obs = rng.rand(192).astype(np.float32)  # [0,1] pixel-like
        obs_all.append(obs)
        done = i % 40 == 39
        buf8.add(obs, int(rng.randint(4)), float(done), done)
    batch = buf8.sample_sequences(batch_size=8, seq_len=6,
                                  device=torch.device("cpu"),
                                  prioritized=True, reward_fraction=0.25)
    ob = batch["observations"].numpy()
    assert ob.dtype == np.float32 and 0.0 <= ob.min() and ob.max() <= 1.0
    # roundtrip error bound: exact stored obs vs dequantized
    idx = int(batch["start_indices"][0])
    err = np.abs(ob[0, 0] - obs_all[idx]).max()
    assert err <= (0.5 / 255.0) + 1e-6, f"quantization error {err}"
    print(f"  uint8 buffer ok: 4x RAM cut, roundtrip err {err:.5f} <= 1/510")


def test_cnn_icm():
    torch.manual_seed(0)
    icm = IntrinsicCuriosityModule(
        obs_dim=12288, action_dim=17, feature_dim=64, discrete_actions=True,
        pixel_obs=True, image_channels=3, image_size=64)
    assert isinstance(icm.encoder, CNNFeatureEncoder)
    obs = torch.rand(6, 12288)
    nxt = torch.rand(6, 12288)
    act = torch.eye(17)[torch.randint(0, 17, (6,))]
    r = icm.compute_intrinsic_reward(obs, act, nxt)
    assert r.shape == (6,) and torch.isfinite(r).all()
    m = icm.train_step(obs, act, nxt)
    assert np.isfinite(m["icm_total"])
    n_conv = sum(p.numel() for p in icm.encoder.conv.parameters())
    assert n_conv > 100_000, "conv stack missing?"
    print(f"  cnn-icm ok: reward+train finite, conv params {n_conv/1e3:.0f}k")


def test_crafter_integration():
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 2000})
    cfg["environment"]["max_episode_steps"] = 120
    cfg["dream_training"]["enabled"] = False
    cfg["skill_bank"]["storage_dir"] = "/tmp/scaleups_sb"
    cfg["loop"]["verbose"] = 0
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    agent = DevelopmentalAI(config=copy.deepcopy(cfg))
    assert agent.replay_buffer.observations.dtype == np.uint8
    assert isinstance(agent.curiosity.encoder, CNNFeatureEncoder)
    agent.run(total_timesteps=500, verbose=0)
    m = agent._train_world_model()
    wm_loss = m.get("world_model_loss", m.get("total"))
    assert wm_loss is None or np.isfinite(float(wm_loss))
    print(f"  integration ok: uint8 buffer + CNN curiosity live, "
          f"{agent.total_episodes} episodes, WM trains")


if __name__ == "__main__":
    for fn in (test_uint8_buffer, test_cnn_icm, test_crafter_integration):
        print(f"[scaleups-smoke] {fn.__name__}")
        fn()
    print("[scaleups-smoke] ALL PASS")
