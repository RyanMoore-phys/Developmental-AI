"""Craftax integration smoke (NEXT_OBJECTIVES.md — the deeper env).

Runs on the POD ONLY (Craftax needs Python 3.10+ and JAX). Verifies the full
Craftax-Symbolic pipeline bottom-up:
  1. Adapter API: gymnasium 5-tuple/2-tuple, gymnasium Box(8268)/Discrete(43),
     NO jax arrays leak into obs/reward/info, seeded reset, terminated split.
  2. make_env("Craftax-Symbolic-v1"): full-stack dims + one rollout.
  3. WorldModel VECTOR path (pixel_obs=false) on real Craftax obs:
     observe_sequence + compute_loss finite + imagination rolls.
  4. DevelopmentalAI end-to-end from configs/craftax.yaml (reduced sizes):
     the per-dimension symbolic machinery is auto-SKIPPED (obs_dim 8268>2000),
     0 KG facts, real steps, WM trains, no NaN.
"""
import copy

import numpy as np
import torch
import yaml

SB_DIR = "/tmp/craftax_smoke_sb"


def test_adapter_api():
    from developmental_ai.environments.craftax_env import CraftaxEnvAdapter
    import gymnasium as gym
    env = CraftaxEnvAdapter()
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 43, env.action_space.n
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.ndim == 1 and obs.shape[0] == 8268
    assert isinstance(obs, np.ndarray) and not hasattr(obs, "block_until_ready")
    out = env.step(0)
    assert len(out) == 5
    o2, r, term, trunc, info = out
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    assert isinstance(o2, np.ndarray) and o2.shape == (8268,)
    # no jax arrays leak into info
    for k, v in info.items():
        assert not hasattr(v, "block_until_ready"), f"jax array leaked in info[{k}]"
    # achievements contract (2026-07-15 fix): Craftax keeps achievements in
    # STATE, not info — the adapter must surface a Crafter-style dict or the
    # goal channel + breadth measurement are silently blind.
    ach = info.get("achievements")
    assert isinstance(ach, dict) and len(ach) >= 60, f"achievements missing: {type(ach)}"
    assert "collect_wood" in ach, list(ach)[:5]
    assert all(isinstance(v, int) for v in ach.values())
    # seeded reproducibility
    o_a, _ = CraftaxEnvAdapter().reset(seed=3)
    o_b, _ = CraftaxEnvAdapter().reset(seed=3)
    o_c, _ = CraftaxEnvAdapter().reset(seed=4)
    assert np.allclose(o_a, o_b), "same seed differs"
    assert not np.allclose(o_a, o_c), "different seeds identical"
    print(f"  adapter ok: Box(8268)/Discrete(43), numpy 5-tuple, no jax leak, seeding")


def test_make_env_stack():
    from developmental_ai.environments.wrappers import make_env
    env, _ = make_env("Craftax-Symbolic-v1", curriculum=False)
    assert env.obs_dim == 8268 and env.action_dim == 43 and env.is_discrete
    obs, info = env.reset(seed=1)
    assert obs.shape == (8268,) and obs.dtype == np.float32
    total_r = 0.0
    for _ in range(60):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        assert obs.shape == (8268,) and np.isfinite(obs).all()
        if term or trunc:
            obs, info = env.reset()
    print(f"  make_env ok: dims (8268/43/discrete), 60 steps, reward so far {total_r:.1f}")


def test_world_model_vector_path():
    from developmental_ai.environments.wrappers import make_env
    from developmental_ai.world_model.rssm import WorldModel
    torch.manual_seed(0)
    env, _ = make_env("Craftax-Symbolic-v1", curriculum=False)
    wm = WorldModel(obs_dim=8268, action_dim=43, stochastic_size=8,
                    stochastic_classes=8, deterministic_size=64, hidden_dim=64,
                    pixel_obs=False, film_conditioning=True,
                    inverse_dynamics=True, discrete_actions=True)
    obs_seq, act_seq = [], []
    o, _ = env.reset(seed=2)
    for _ in range(9):
        a = env.action_space.sample()
        obs_seq.append(o)
        act_seq.append(np.eye(43, dtype=np.float32)[a])
        o, r, term, trunc, _ = env.step(a)
        if term or trunc:
            o, _ = env.reset()
    obs_t = torch.tensor(np.stack(obs_seq)).unsqueeze(0)
    act_t = torch.tensor(np.stack(act_seq)).unsqueeze(0)
    states, infos = wm.observe_sequence(obs_t, act_t)
    assert states["h"].shape[:2] == (1, 9)
    losses = wm.compute_loss(obs_t, act_t, torch.zeros(1, 9), torch.ones(1, 9))
    if isinstance(losses, tuple):
        losses = losses[0]
    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"non-finite {k}"
    policy = lambda l: torch.eye(43)[torch.randint(0, 43, (l.shape[0],))]
    start = {"h": states["h"][:, -1], "z": states["z"][:, -1]}
    traj = wm.imagine_trajectory(start, policy, horizon=5)
    assert torch.isfinite(traj["latents"]).all()
    print(f"  WM vector path ok: recon={losses['reconstruction'].item():.4f}, "
          f"kl={losses['kl'].item():.4f}, imagination rolls")


def test_agent_end_to_end():
    cfg = yaml.safe_load(open("configs/craftax.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 1500})
    cfg["dream_training"].update({"batch_size": 4, "imagination_horizon": 5,
                                  "activate_after_episodes": 1, "warmup_episodes": 1})
    cfg["skill_bank"]["storage_dir"] = SB_DIR
    cfg["loop"]["verbose"] = 0
    from developmental_ai.core.developmental_loop import (
        DevelopmentalAI, _NullSymbolicDecoder)
    agent = DevelopmentalAI(config=copy.deepcopy(cfg))
    assert not agent.pixel_obs and agent.obs_dim == 8268 and agent.action_dim == 43
    assert agent._skip_perdim_symbolic, "8268-d obs must skip per-dim symbolic"
    assert isinstance(agent.symbolic_decoder, _NullSymbolicDecoder)
    assert not agent.world_model.pixel_obs, "must be on the MLP (vector) path"
    agent.run(total_timesteps=700, verbose=0)
    assert agent.total_episodes >= 1, f"only {agent.total_episodes} episodes"
    assert len(agent.replay_buffer) >= 600
    kg = agent.knowledge_graph.get_stats()
    assert kg["num_facts"] == 0, f"per-dim gate leaked {kg['num_facts']} facts"
    m = agent._train_world_model()
    wm_loss = m.get("world_model_loss", m.get("total"))
    assert wm_loss is None or np.isfinite(float(wm_loss))
    print(f"  agent ok: {agent.total_episodes} episodes, buffer "
          f"{len(agent.replay_buffer)}, 0 facts (gate), WM trains")


if __name__ == "__main__":
    for fn in (test_adapter_api, test_make_env_stack,
               test_world_model_vector_path, test_agent_end_to_end):
        print(f"[craftax-smoke] {fn.__name__}")
        fn()
    print("[craftax-smoke] ALL PASS")
