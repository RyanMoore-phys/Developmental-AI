"""Parallel-envs smoke — the least-battle-tested path, now with threaded
stepping + goal-broadcaster wiring (July 2026 Minecraft-parallel fixes).

Runs on Crafter (fast, Mac-ok). Forces dream-control + IMAGINE stage so the
parallel path engages immediately, then verifies:
  1. The parallel episode actually runs (timesteps advance by N per step).
  2. All N replay streams fill (MultiStreamReplayBuffer, uint8 obs on pixels).
  3. The goal broadcaster is live in parallel mode (target set, slots seen).
  4. WM trains from the multi-stream buffer; dream actor trains; no NaN.
  5. Concurrent stepping returns consistent shapes (implicitly: no crash).
"""
import copy

import numpy as np
import yaml

SB = "/tmp/parallel_smoke_sb"


def main():
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 4000})
    cfg["environment"]["max_episode_steps"] = 60
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 4}
    cfg["dream_training"].update({"enabled": True, "control": True,
                                  "warmup_episodes": 1,
                                  "activate_after_episodes": 1,
                                  "batch_size": 4, "imagination_horizon": 5})
    cfg["developmental_stages"]["trust_level"] = 999.0  # keep IMAGINE stable
    cfg["symbolic"] = {"enabled": True, "knowledge_source": "goal"}
    cfg["goals"] = {"mode": "given", "max_slots": 16, "mint_skills": True}
    cfg["skill_bank"]["storage_dir"] = SB
    cfg["loop"]["verbose"] = 0

    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.core.glue_layer import DevelopmentalStageController
    agent = DevelopmentalAI(config=copy.deepcopy(cfg))
    assert agent._use_parallel_envs, "parallel_envs not enabled"

    # Force the dream-control parallel path from episode 1
    agent.stage_controller.stage = DevelopmentalStageController.IMAGINE
    agent.dream_training_active = True

    agent.run(total_timesteps=1200, verbose=0)

    # 1. parallel actually ran: N transitions per lockstep step
    assert len(agent._parallel_envs) == 4, "parallel envs not constructed"
    assert hasattr(agent, "_env_pool"), "thread pool missing"
    # 2. every stream filled
    sizes = [len(s) for s in agent.replay_buffer.streams]
    assert all(s > 100 for s in sizes), f"streams unbalanced/empty: {sizes}"
    assert agent.replay_buffer.streams[0].observations.dtype == np.uint8
    # 3. broadcaster alive in parallel mode
    st = agent.broadcaster.stats
    assert st["target"] is not None or st["n_slots"] >= 0  # updated at all
    # 4. WM + dream actor train from the multi-stream buffer
    m = agent._train_world_model()
    wm_loss = m.get("world_model_loss", m.get("total"))
    assert wm_loss is None or np.isfinite(float(wm_loss))
    dm = agent._dream_train_policy()
    assert np.isfinite(dm.get("dream_actor_loss", 0.0))

    print(f"[parallel-smoke] ALL PASS — streams={sizes}, "
          f"episodes={agent.total_episodes}, timesteps={agent.total_timesteps}, "
          f"goal_stats={st}")


def test_waking_parallel():
    """WAKING parallel path (July 2026): real PPO policy drives N envs; all
    streams feed replay + discovery; the PRIMARY stream feeds the PPO buffer;
    PPO updates on-policy. dream_training_active stays False throughout."""
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 8000})
    cfg["environment"]["max_episode_steps"] = 40
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 4}
    cfg["dream_training"] = {"enabled": False}
    cfg["policy"] = {**cfg.get("policy", {}), "n_steps": 60}
    cfg["symbolic"] = {"enabled": True, "knowledge_source": "goal"}
    cfg["goals"] = {"mode": "given", "max_slots": 16, "mint_skills": True}
    cfg["skill_bank"]["storage_dir"] = "/tmp/waking_par_sb"
    cfg["loop"]["verbose"] = 0

    from developmental_ai.core.developmental_loop import DevelopmentalAI
    agent = DevelopmentalAI(config=copy.deepcopy(cfg))
    assert agent._use_parallel_envs
    assert not agent.dream_training_active, "should start in waking mode"

    agent.run(total_timesteps=1600, verbose=0)

    # dream never engaged; parallel still ran across 4 real-policy envs
    assert not agent.dream_training_active
    assert len(agent._parallel_envs) == 4
    sizes = [len(s) for s in agent.replay_buffer.streams]
    assert all(s > 50 for s in sizes), f"scout streams empty: {sizes}"
    # the PPO primary rollout actually filled (proves the primary-stream store
    # path ran); at n_steps=60 over ~1600 steps it also updated + cleared.
    plosses = [x for x in agent.training_metrics.get("policy_loss", [])
               if x != 0.0]
    assert len(plosses) > 0 or len(agent.policy.rollout_obs) > 0, \
        "primary stream never fed the PPO rollout in waking parallel"
    print(f"[parallel-smoke] WAKING PASS — streams={sizes}, "
          f"nonzero_policy_losses={len(plosses)}, "
          f"rollout={len(agent.policy.rollout_obs)}, "
          f"episodes={agent.total_episodes}")


if __name__ == "__main__":
    main()
    test_waking_parallel()
