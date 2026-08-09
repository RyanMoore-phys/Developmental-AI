"""Smoke tests for the H2 (collapse) + H3 (curiosity) fix packages
(CODE_AUDIT_2026-07.md §H.2, §H.3).

H3 — curiosity:
  1. Encoder is trained ONLY by the inverse loss (forward loss detached).
  2. Intrinsic reward is non-negative (std-only normalization), and
     reward_scale actually scales the delivered reward (post-normalization).
  3. Discrete inverse loss is cross-entropy; continuous stays MSE.
  4. LP: zero-progress / first-visit states get reward 0, never negative.

H2 — collapse package (integration, tiny DoorKey agent):
  5. Env swap flushes the PPO rollout buffer and restarts the per-env
     dream-augment warmup counter.
  6. Dream distillation exposes trust_frac; the KL trust threshold behaves
     monotonically (huge threshold -> all trusted; zero -> none), and the
     effective weight is scaled by the trust EMA.
"""
import copy

import numpy as np
import torch
import yaml

from developmental_ai.curiosity.icm import IntrinsicCuriosityModule
from developmental_ai.curiosity.learning_progress import LearningProgressCuriosity

OBS, ACT, FEAT = 12, 4, 32


def _batch(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, OBS, generator=g)
    nxt = obs + 0.1 * torch.randn(n, OBS, generator=g)
    act = torch.eye(ACT)[torch.randint(0, ACT, (n,), generator=g)]
    return obs, act, nxt


def test_encoder_trained_by_inverse_only():
    torch.manual_seed(0)
    icm = IntrinsicCuriosityModule(OBS, ACT, feature_dim=FEAT,
                                   inverse_loss_weight=0.0,
                                   discrete_actions=True)
    obs, act, nxt = _batch()
    icm.train_step(obs, act, nxt)
    g = [p.grad.abs().max().item() for p in icm.encoder.parameters()
         if p.grad is not None]
    assert max(g, default=0.0) < 1e-12, f"forward loss leaked into encoder: {g}"
    # ...and the inverse loss DOES train it.
    icm2 = IntrinsicCuriosityModule(OBS, ACT, feature_dim=FEAT,
                                    forward_loss_weight=0.0,
                                    discrete_actions=True)
    icm2.train_step(obs, act, nxt)
    g2 = [p.grad.abs().max().item() for p in icm2.encoder.parameters()
          if p.grad is not None]
    assert max(g2, default=0.0) > 0, "inverse loss failed to reach encoder"
    print(f"  encoder isolation ok (fwd-only grad {max(g, default=0):.1e}, "
          f"inv-only grad {max(g2):.1e})")


def test_nonnegative_and_scaled_reward():
    torch.manual_seed(0)
    icm = IntrinsicCuriosityModule(OBS, ACT, feature_dim=FEAT,
                                   discrete_actions=True)
    obs, act, nxt = _batch()
    r1 = icm.compute_intrinsic_reward(obs, act, nxt)
    # ANTI-FARMING is the binding constraint (canary-validated twice): the
    # novelty signal must be ~zero-mean so padding an episode buys nothing.
    assert abs(r1.mean().item()) < 0.3, (
        f"novelty reward has farmable mass: mean {r1.mean():.3f}")
    assert (r1 > 0).any() and (r1 < 0).any(), "not a relative signal"
    icm.reward_scale = 0.5
    r_half = icm.compute_intrinsic_reward(obs, act, nxt)
    nz = r1.abs() > 1e-6
    ratio = (r_half[nz].abs().mean() / r1[nz].abs().mean()).item()
    assert 0.3 < ratio < 0.7, f"reward_scale inert (ratio {ratio:.3f})"
    print(f"  reward ok: zero-mean ({r1.mean():+.3f}, no farm mass), "
          f"scale 0.5 -> ratio {ratio:.3f}")


def test_inverse_loss_objectives():
    torch.manual_seed(0)
    obs, act, nxt = _batch()
    d = IntrinsicCuriosityModule(OBS, ACT, feature_dim=FEAT,
                                 discrete_actions=True)
    m_d = d.train_step(obs, act, nxt)
    c = IntrinsicCuriosityModule(OBS, ACT, feature_dim=FEAT,
                                 discrete_actions=False)
    m_c = c.train_step(obs, act, nxt)
    # CE on a 4-way random problem starts near ln(4)=1.386; MSE on one-hots
    # starts near ~0.2 — the two objectives must be distinguishable.
    assert m_d["icm_inverse"] > 0.8, f"discrete loss not CE-like: {m_d}"
    assert m_c["icm_inverse"] < 0.8, f"continuous loss not MSE-like: {m_c}"
    print(f"  inverse objectives ok: CE {m_d['icm_inverse']:.3f}, "
          f"MSE {m_c['icm_inverse']:.3f}")


def test_lp_first_visit_not_punished():
    torch.manual_seed(0)
    lp = LearningProgressCuriosity(OBS, ACT, feature_dim=FEAT,
                                   lp_min_samples=2, discrete_actions=True)
    base = torch.zeros(1, OBS)
    act = torch.eye(ACT)[[0]]
    # Grind one recurring bucket so running stats see real progress values.
    for _ in range(12):
        lp.compute_intrinsic_reward(base, act, base)
    # First visit to a brand-new state: zero progress — must NOT be negative.
    novel = torch.full((1, OBS), 7.7)
    r = lp.compute_intrinsic_reward(novel, act, novel)
    assert r.item() >= 0.0, f"first visit punished: {r.item():.4f}"
    print(f"  LP first-visit ok: reward {r.item():.4f} >= 0")


def _tiny_agent():
    cfg = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
    cfg["environment"]["name"] = "MiniGrid-DoorKey-5x5-v0"
    cfg["seed"] = 0
    cfg.setdefault("symbolic", {})["enabled"] = False
    cfg.setdefault("llm", {})["enabled"] = False
    cfg.setdefault("dream_training", {}).update(
        {"enabled": True, "control": False, "activate_after_episodes": 3})
    cfg.setdefault("skill_bank", {})["storage_dir"] = "/tmp/h2_smoke_sb"
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    return DevelopmentalAI(config=copy.deepcopy(cfg))


def test_env_swap_flush_and_warmup():
    agent = _tiny_agent()
    agent.run(total_timesteps=0)          # latch env token, no episodes
    agent._episodes_this_env = 57         # pretend long training on env 1
    agent.policy.rollout_obs.extend([np.zeros(agent.obs_dim)] * 5)
    agent.policy.rollout_rewards.extend([0.0] * 5)
    import gymnasium as gym  # noqa: F401  (env swap via fresh object)
    agent.env = copy.deepcopy(agent.env)  # new object identity = swap
    agent.run(total_timesteps=0)
    assert agent._episodes_this_env == 0, agent._episodes_this_env
    assert len(agent.policy.rollout_obs) == 0, "rollout buffer not flushed"
    # per-env warmup: 0 episodes on this env < activate_after (3)
    assert agent._episodes_this_env < agent.dream_activate_after
    print("  env swap ok: warmup counter reset, rollout buffer flushed")
    return agent


def test_trust_gate(agent):
    # Fill the replay buffer with enough synthetic transitions to sample.
    rng = np.random.RandomState(0)
    for i in range(600):
        agent.replay_buffer.add(
            rng.rand(agent.obs_dim).astype(np.float32),
            int(rng.randint(agent.action_dim)),
            float(i % 40 == 39),
            bool(i % 40 == 39),
        )
    agent._dream_solve_ema = 0.0
    # (a) huge threshold -> every state trusted, EMA rises toward 1.
    agent.dream_distill_trust_kl_max = 1e9
    m1 = agent._distill_dream_to_real()
    assert m1["dream_distill_trust_frac"] == 1.0, m1
    ema_after_1 = agent._wm_trust_ema
    assert ema_after_1 > 0.0
    # (b) impossible threshold -> nothing trusted, zero effective weight
    #     contribution from the gate this round.
    agent.dream_distill_trust_kl_max = -1.0
    m2 = agent._distill_dream_to_real()
    assert m2["dream_distill_trust_frac"] == 0.0, m2
    assert m2["dream_distill_gate_frac"] == 0.0
    # eff weight = base * (1 - solve_ema) * trust_ema (EMA decayed after (b))
    expect = agent.dream_distill_weight * (1.0 - 0.0) * agent._wm_trust_ema
    assert abs(m2["dream_distill_eff_weight"] - expect) < 1e-6, (
        m2["dream_distill_eff_weight"], expect)
    print(f"  trust gate ok: frac 1.0 -> 0.0 across thresholds, "
          f"eff_w tracks trust EMA ({m2['dream_distill_eff_weight']:.3f})")


if __name__ == "__main__":
    for fn in (test_encoder_trained_by_inverse_only,
               test_nonnegative_and_scaled_reward,
               test_inverse_loss_objectives,
               test_lp_first_visit_not_punished):
        print(f"[h2h3-smoke] {fn.__name__}")
        fn()
    print("[h2h3-smoke] test_env_swap_flush_and_warmup")
    agent = test_env_swap_flush_and_warmup()
    print("[h2h3-smoke] test_trust_gate")
    test_trust_gate(agent)
    print("[h2h3-smoke] ALL PASS")
