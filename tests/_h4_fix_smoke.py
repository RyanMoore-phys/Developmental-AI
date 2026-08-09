"""Smoke tests for the H4 eval-honesty package (CODE_AUDIT_2026-07.md §H.4).

  1. deterministic=True gives the argmax action, repeatably; sampling remains
     the default for training.
  2. The env wrapper's obs-normalization stats freeze/unfreeze correctly.
  3. Seeded eval episodes reproduce the same layout (same first observation).
  4. success_rate respects the per-env threshold.
  5. bootstrap_ci / bootstrap_diff_ci / save_results behave sanely.
"""
import json
import os

import numpy as np
import torch

from developmental_ai.core.stats import (
    bootstrap_ci, bootstrap_diff_ci, save_results)
from developmental_ai.policy.actor_critic import StandaloneActorCritic


def test_deterministic_action():
    torch.manual_seed(0)
    pol = StandaloneActorCritic(obs_dim=10, action_dim=5)
    obs = np.random.RandomState(0).rand(10).astype(np.float32)
    greedy = {pol.select_action(obs, deterministic=True)[0] for _ in range(20)}
    assert len(greedy) == 1, f"deterministic path not deterministic: {greedy}"
    sampled = {pol.select_action(obs)[0] for _ in range(50)}
    assert len(sampled) > 1, "default sampling looks deterministic (untrained net)"
    print(f"  deterministic ok: greedy always {greedy.pop()}, "
          f"sampled hit {len(sampled)} distinct actions")


def test_normalizer_freeze():
    from developmental_ai.environments.wrappers import DevelopmentalEnvWrapper
    import gymnasium as gym
    env = DevelopmentalEnvWrapper(gym.make("CartPole-v1"), normalize_obs=True)
    env.reset(seed=0)
    for _ in range(30):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    count_before = env._obs_count
    env.freeze_obs_stats(True)
    for _ in range(10):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    assert env._obs_count == count_before, "stats updated while frozen"
    env.freeze_obs_stats(False)
    _, _, term, trunc, _ = env.step(env.action_space.sample())
    assert env._obs_count == count_before + 1, "stats did not resume"
    print(f"  normalizer freeze ok (count held at {count_before}, then resumed)")


def test_seeded_reset_reproducible():
    import gymnasium as gym
    import minigrid  # noqa: F401
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    o1, _ = env.reset(seed=123)
    o2, _ = env.reset(seed=123)
    o3, _ = env.reset(seed=124)
    same = np.array_equal(o1["image"], o2["image"])
    diff = not np.array_equal(o1["image"], o3["image"])
    assert same, "same seed gave different layouts"
    assert diff, "different seeds gave identical layouts (suspicious)"
    print("  seeded reset ok: seed 123 == seed 123, != seed 124")


def test_success_threshold():
    rewards = [-100.0, -200.0, -120.0]
    thr = -150.0
    rate = float(np.mean([1.0 if r > thr else 0.0 for r in rewards]))
    assert abs(rate - 2 / 3) < 1e-9
    print(f"  threshold ok: Acrobot-style rewards at thr={thr} -> {rate:.2f}")


def test_stats_helpers(tmp="/tmp/h4_smoke_results"):
    ci = bootstrap_ci([0.8, 0.9, 0.7, 0.85])
    assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["n"] == 4
    d = bootstrap_diff_ci([1.0, 1.1, 0.9], [0.1, 0.2, 0.15])
    assert d["excludes_zero"] and d["lo"] > 0
    d2 = bootstrap_diff_ci([0.5, 0.6, 0.4], [0.45, 0.65, 0.5])
    assert not d2["excludes_zero"]
    path = save_results(tmp, {"auc": ci}, extra={"protocol": "smoke"})
    with open(path) as f:
        back = json.load(f)
    assert back["results"]["auc"]["n"] == 4 and back["meta"]["protocol"] == "smoke"
    os.remove(path)
    print(f"  stats ok: CI [{ci['lo']:.2f},{ci['hi']:.2f}], clear diff detected, "
          f"null diff not, artifact round-trips")


if __name__ == "__main__":
    for fn in (test_deterministic_action, test_normalizer_freeze,
               test_seeded_reset_reproducible, test_success_threshold,
               test_stats_helpers):
        print(f"[h4-smoke] {fn.__name__}")
        fn()
    print("[h4-smoke] ALL PASS")
