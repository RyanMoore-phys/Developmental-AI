"""24/48 working-set widening smoke (Mac-ok, Crafter parallel lifelong).

Certifies that `lifelong.active_goals`/`active_skills` are the SINGLE knob for
the resident working-set widths:
  1. the goal registry constructs at active_goals (48), overriding goals.max_slots,
  2. the policy option head widens by active_skills (24),
  3. the policy knowledge-input width tracks broadcaster.DIM = 2*48+2,
  4. a short lifelong segment runs at the new width without error,
  5. a FRESH skill-bank dir starts clean (0 skills) — the certified-run shape.
"""
import copy
import shutil

import yaml


def _tmp(name):
    import os
    d = os.path.join("/tmp", name)
    shutil.rmtree(d, ignore_errors=True)
    return d


def test_widening_construct_and_run():
    from developmental_ai.core.developmental_loop import DevelopmentalAI

    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 2000})
    cfg["environment"]["max_episode_steps"] = 60
    cfg["skill_bank"]["storage_dir"] = _tmp("widen_sb")     # FRESH bank
    cfg["loop"]["verbose"] = 0
    cfg.setdefault("llm", {})["enabled"] = False
    cfg["dream_training"]["enabled"] = False
    cfg["symbolic"] = {"enabled": True, "knowledge_source": "goal"}
    cfg["goals"] = {"mode": "discovered", "max_slots": 16,  # deliberately small
                    "spike_threshold": 0.5, "match_cosine": 0.8, "seed": 0}
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["skills_as_options"] = {"enabled": True, "k_slots": 8}  # small default
    # the widening: active_* OVERRIDE the small config widths
    cfg["lifelong"] = {"enabled": True, "forever": False, "segment_steps": 128,
                       "segment_len": 128, "wm_train_every": 64,
                       "goal_horizon": 256, "state_decay": 1.0,
                       "active_goals": 48, "active_skills": 24,
                       "paging": {"enabled": True, "min_idle_episodes": 4,
                                  "recall_threshold": 0.8}}

    agent = DevelopmentalAI(config=copy.deepcopy(cfg))
    # 1+2+3: active_* drove the widths, overriding the small config values
    assert agent.broadcaster.max_slots == 48, (
        f"active_goals did not widen the registry: {agent.broadcaster.max_slots}")
    assert agent.meta_action_dim == agent.action_dim + 24, (
        f"active_skills did not widen the option head: {agent.meta_action_dim} "
        f"(action_dim={agent.action_dim})")
    assert agent.policy.knowledge_dim == agent.broadcaster.DIM == 2 * 48 + 2, (
        f"policy knowledge width != broadcaster.DIM: "
        f"{agent.policy.knowledge_dim} vs {agent.broadcaster.DIM}")
    # 5: fresh bank starts clean
    assert agent.skill_bank.get_stats()["total_skills"] == 0, "bank not fresh"
    # 4: a short lifelong segment runs at the new width without error
    agent.run(total_timesteps=300, verbose=0)
    assert agent.total_timesteps >= 256, "lifelong stream did not advance"
    # paging store attached and sized to the signature cue dim
    assert getattr(agent, "_goal_ltm", None) is not None, "paging not attached"
    agent.close()
    print(f"  1. widening ok: 48 goal slots, +24 option head, "
          f"knowledge_dim={agent.policy.knowledge_dim}, ran to "
          f"{agent.total_timesteps} steps, fresh bank")


if __name__ == "__main__":
    print("[widening-smoke] test_widening_construct_and_run")
    test_widening_construct_and_run()
    print("[widening-smoke] ALL PASS")
