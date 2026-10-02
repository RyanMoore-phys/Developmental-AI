"""Crafter integration smoke (NEXT_OBJECTIVES.md #1).

Verifies the full pixel pipeline on real Crafter frames, bottom-up:
  1. Adapter API: gymnasium 5-tuple/2-tuple, gymnasium spaces, seeded
     reproducibility, terminated-vs-truncated split semantics.
  2. NativePixelObsWrapper: flat CHW float32 [0,1], obs_dim 12288, channel
     order actually CHW (round-trip test), no Welford on images.
  3. make_env("CrafterReward-v1"): full stack dims + one env rollout.
  4. WorldModel pixel path: CNN encoder/decoder shapes, observe_sequence,
     compute_loss (finite), imagine_trajectory on REAL crafter obs.
  5. DevelopmentalAI end-to-end from configs/crafter.yaml (reduced sizes):
     real steps + a WM train call + no per-pixel fact/decoder taxes.

CPU-sized: a few minutes on the Mac. Re-run on the training host with CUDA visible
to prove the GPU path (with _gpu_device_smoke.py).
"""
import copy

import numpy as np
import torch
import yaml

SB_DIR = "/tmp/crafter_smoke_sb"


def test_adapter_api():
    from developmental_ai.environments.crafter_env import CrafterEnvAdapter
    import gymnasium as gym
    env = CrafterEnvAdapter()
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 17
    obs, info = env.reset(seed=7)
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    out = env.step(2)
    assert len(out) == 5
    obs2, r, term, trunc, info = out
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    # seeded reproducibility
    o1, _ = CrafterEnvAdapter(seed=11).reset(seed=11)
    o2, _ = CrafterEnvAdapter(seed=11).reset(seed=11)
    o3, _ = CrafterEnvAdapter(seed=12).reset(seed=12)
    assert np.array_equal(o1, o2), "same seed differs"
    assert not np.array_equal(o1, o3), "different seeds identical"
    # truncation semantics: a short length cap fires as TRUNCATED (alive at cap)
    env_short = CrafterEnvAdapter(length=12)
    env_short.reset(seed=3)
    term = trunc = False
    for _ in range(12):
        _, _, term, trunc, info = env_short.step(0)  # noop until the cap
        if term or trunc:
            break
    assert trunc and not term, f"length cap should truncate (term={term}, trunc={trunc})"
    print("  adapter ok: spaces, 5-tuple, seeding, truncation split")


def test_native_pixel_wrapper():
    from developmental_ai.environments.crafter_env import CrafterEnvAdapter
    from developmental_ai.environments.wrappers import NativePixelObsWrapper
    env = NativePixelObsWrapper(CrafterEnvAdapter())
    obs, _ = env.reset(seed=5)
    assert obs.shape == (12288,) and obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0 and obs.max() > 0.05
    # channel order: reconstruct CHW and compare against the raw HWC frame
    raw, _ = CrafterEnvAdapter().reset(seed=5)
    chw = obs.reshape(3, 64, 64)
    hwc = (chw.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    assert np.array_equal(hwc, raw), "CHW round-trip mismatch — layout bug"
    print("  native pixel wrapper ok: flat CHW float32 [0,1], round-trip exact")


def test_make_env_stack():
    from developmental_ai.environments.wrappers import make_env
    env, _ = make_env("CrafterReward-v1", curriculum=False)
    assert env.obs_dim == 12288 and env.action_dim == 17 and env.is_discrete
    obs, info = env.reset(seed=1)
    assert obs.shape == (12288,) and obs.dtype == np.float32
    total_r = 0.0
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        assert obs.shape == (12288,)
        if term or trunc:
            obs, info = env.reset()
    assert "extrinsic_reward" in info
    print(f"  make_env ok: dims (12288/17/discrete), 50 steps walked")


def test_world_model_pixel_path():
    from developmental_ai.environments.wrappers import make_env
    from developmental_ai.world_model.rssm import WorldModel
    torch.manual_seed(0)
    env, _ = make_env("CrafterReward-v1", curriculum=False)
    wm = WorldModel(obs_dim=12288, action_dim=17, stochastic_size=8,
                    stochastic_classes=8, deterministic_size=64,
                    hidden_dim=64, pixel_obs=True, image_channels=3,
                    image_size=64, film_conditioning=True,
                    inverse_dynamics=True, discrete_actions=True)
    # collect a real short sequence
    obs_seq, act_seq = [], []
    o, _ = env.reset(seed=2)
    for _ in range(9):
        a = env.action_space.sample()
        obs_seq.append(o)
        act_seq.append(np.eye(17, dtype=np.float32)[a])
        o, r, term, trunc, _ = env.step(a)
    obs_t = torch.tensor(np.stack(obs_seq)).unsqueeze(0)         # (1, 9, 12288)
    act_t = torch.tensor(np.stack(act_seq)).unsqueeze(0)         # (1, 9, 17)
    states, infos = wm.observe_sequence(obs_t, act_t)
    assert states["h"].shape[:2] == (1, 9)
    losses = wm.compute_loss(obs_t, act_t,
                             torch.zeros(1, 9), torch.ones(1, 9))
    if isinstance(losses, tuple):
        losses = losses[0]
    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"non-finite {k}"
    # decoder emits flat CHW matching obs layout
    lat = wm.rssm.get_latent({"h": states["h"][:, -1], "z": states["z"][:, -1]})
    dec = wm.decoder(lat)
    assert dec.shape[-1] == 12288
    # imagination rolls
    policy = lambda l: torch.eye(17)[torch.randint(0, 17, (l.shape[0],))]
    start = {"h": states["h"][:, -1], "z": states["z"][:, -1]}
    traj = wm.imagine_trajectory(start, policy, horizon=5)
    assert torch.isfinite(traj["latents"]).all()
    print(f"  WM pixel path ok: recon={losses['reconstruction'].item():.4f}, "
          f"kl={losses['kl'].item():.4f}, imagination rolls")


def test_agent_end_to_end():
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    # CPU-smoke sizing: tiny WM + buffer so this runs in minutes on 6 cores
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 2000})
    cfg["environment"]["max_episode_steps"] = 120   # short episodes for the smoke
    cfg["dream_training"].update({"batch_size": 4, "imagination_horizon": 5,
                                  "activate_after_episodes": 2,
                                  "warmup_episodes": 2})
    cfg["skill_bank"]["storage_dir"] = SB_DIR
    cfg["loop"]["verbose"] = 0
    from developmental_ai.core.developmental_loop import (
        DevelopmentalAI, _NullSymbolicDecoder)
    agent = DevelopmentalAI(config=copy.deepcopy(cfg))
    assert agent.pixel_obs and agent.obs_dim == 12288 and agent.action_dim == 17
    assert isinstance(agent.symbolic_decoder, _NullSymbolicDecoder), (
        "pixel env must get the no-op symbolic decoder")
    assert agent.world_model.pixel_obs, "WM must be on the CNN path"
    agent.run(total_timesteps=700, verbose=0)
    assert agent.total_episodes >= 2, f"only {agent.total_episodes} episodes"
    assert len(agent.replay_buffer) >= 600
    kg = agent.knowledge_graph.get_stats()
    assert kg["num_facts"] == 0, f"pixel gates leaked {kg['num_facts']} facts"
    m = agent._train_world_model()
    wm_loss = m.get("world_model_loss", m.get("total"))
    assert wm_loss is None or np.isfinite(float(wm_loss))
    print(f"  agent ok: {agent.total_episodes} episodes, buffer "
          f"{len(agent.replay_buffer)}, 0 pixel-facts, WM trains")


if __name__ == "__main__":
    for fn in (test_adapter_api, test_native_pixel_wrapper,
               test_make_env_stack, test_world_model_pixel_path,
               test_agent_end_to_end):
        print(f"[crafter-smoke] {fn.__name__}")
        fn()
    print("[crafter-smoke] ALL PASS")
