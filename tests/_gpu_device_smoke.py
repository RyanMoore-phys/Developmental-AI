"""Device-correctness smoke (July 2026 GPU fix, NEXT_OBJECTIVES.md #2).

StandaloneActorCritic was a plain class: never `.to(device)`'d, built every
input tensor on CPU, and crashed under CUDA ("mat1 on cuda:0 vs cpu"). This
smoke verifies the fix on WHATEVER device is available:

  - on the Mac (CPU-only) it proves the plumbing routes through
    `policy.device` and nothing regressed;
  - on the training host, run WITH the GPU visible — it then proves the CUDA path
    end-to-end (constructor placement, select_action, train_step, the
    dream-distill mix of WM latents + dream actor + policy).

Run on GPU box:  ./venv/bin/python _gpu_device_smoke.py       (CUDA auto)
Force CPU:       CUDA_VISIBLE_DEVICES="" ./venv/bin/python _gpu_device_smoke.py
"""
import numpy as np
import torch

from developmental_ai.policy.actor_critic import (
    DreamActorCritic, StandaloneActorCritic)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[gpu-smoke] running on device: {DEV}")


def test_policy_device_placement():
    pol = StandaloneActorCritic(obs_dim=12, action_dim=5, knowledge_dim=8,
                                device=DEV)
    for name, module in (("actor", pol.actor), ("critic", pol.critic),
                         ("conditioner", pol.conditioner)):
        p_dev = next(module.parameters()).device
        assert p_dev.type == DEV.type, f"{name} params on {p_dev}, want {DEV}"
    # select_action from a numpy obs (the crash site) + knowledge path
    obs = np.random.rand(12).astype(np.float32)
    a, info = pol.select_action(obs, knowledge=np.random.rand(8))
    assert isinstance(a, int) and np.isfinite(info["log_prob"])
    a2, _ = pol.select_action(obs, deterministic=True)
    assert isinstance(a2, int)
    print(f"  policy placement ok: params + select_action on {DEV.type}")


def test_policy_train_step_on_device():
    torch.manual_seed(0)
    pol = StandaloneActorCritic(obs_dim=12, action_dim=5, knowledge_dim=8,
                                device=DEV)
    rng = np.random.RandomState(0)
    for i in range(64):
        obs = rng.rand(12).astype(np.float32)
        a, info = pol.select_action(obs, knowledge=rng.rand(8))
        pol.store_transition(obs, a, float(rng.rand()), i % 16 == 15,
                             info["log_prob"], info["value"],
                             knowledge=rng.rand(8))
    m = pol.train_step(n_epochs=2)
    assert np.isfinite(m["policy_loss"]) and np.isfinite(m["value_loss"])
    assert len(pol.rollout_obs) == 0  # train_step clears the rollout buffer
    print(f"  train_step ok on {DEV.type}: policy_loss={m['policy_loss']:.4f}")


def test_dream_actor_on_device():
    torch.manual_seed(0)
    da = DreamActorCritic(latent_dim=32, action_dim=5).to(DEV)
    latents = torch.randn(4, 32, device=DEV)
    dist = da.get_action_dist(latents)
    act = da.policy_fn(latents)
    assert act.device.type == DEV.type
    rewards = torch.randn(4, 7, device=DEV)
    values = torch.randn(4, 7, device=DEV)
    conts = torch.ones(4, 7, device=DEV)
    ret = da.compute_lambda_returns(rewards, values, conts)
    assert ret.device.type == DEV.type and torch.isfinite(ret).all()
    a, info = da.select_action(latents[:1])
    assert np.isfinite(info["value"])
    print(f"  dream actor ok on {DEV.type}")


def test_agent_end_to_end():
    """Full DevelopmentalAI on the auto-selected device: a few hundred real
    steps + a distill call — the exact CUDA crash path (WM latents + dream
    actor + policy actor/critic all mixed in one graph)."""
    import yaml
    cfg = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
    cfg["seed"] = 0
    cfg.setdefault("symbolic", {})["enabled"] = False
    cfg.setdefault("llm", {})["enabled"] = False
    cfg.setdefault("dream_training", {}).update(
        {"enabled": True, "control": False, "activate_after_episodes": 1})
    cfg.setdefault("skill_bank", {})["storage_dir"] = "/tmp/gpu_smoke_sb"
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    agent = DevelopmentalAI(config=cfg)
    assert agent.device.type == DEV.type, (agent.device, DEV)
    agent.run(total_timesteps=600, verbose=0)
    rng = np.random.RandomState(0)
    for i in range(600):
        agent.replay_buffer.add(rng.rand(agent.obs_dim).astype(np.float32),
                                int(rng.randint(agent.action_dim)),
                                float(i % 40 == 39), bool(i % 40 == 39))
    agent.dream_distill_trust_kl_max = 1e9
    m = agent._distill_dream_to_real()
    assert np.isfinite(m["dream_distill_loss"])
    print(f"  end-to-end ok on {DEV.type}: {agent.total_episodes} episodes, "
          f"distill loss {m['dream_distill_loss']:.4f}")


if __name__ == "__main__":
    for fn in (test_policy_device_placement, test_policy_train_step_on_device,
               test_dream_actor_on_device, test_agent_end_to_end):
        print(f"[gpu-smoke] {fn.__name__}")
        fn()
    print(f"[gpu-smoke] ALL PASS on {DEV.type.upper()}")
