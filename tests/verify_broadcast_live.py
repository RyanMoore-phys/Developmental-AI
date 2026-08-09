#!/usr/bin/env python3
"""
Live broadcast verifier — proves the affordance/knowledge channel actually
carries signal through the SAME data path the running experiment uses:

    env  ->  broadcaster.update(env)  ->  agent._current_knowledge_feature()
         ->  policy.select_action(obs, knowledge=...)  (gate fixed OPEN)

It builds a real DevelopmentalAI agent with the EXACT Phase-1 transfer config
(affordance, force-concat), drives a live DoorKey rollout, and prints the live
broadcast vector at each visited task state. It then toggles the lesion flag on
the SAME agent/broadcaster state to show the content is severed to zeros — the
clean control. Does NOT call agent.run(): no skill minting, no checkpoints.
"""
import copy, logging, numpy as np, yaml, torch

logging.basicConfig(level=logging.WARNING)
from developmental_ai.core.developmental_loop import DevelopmentalAI

PHASE = ["NEED_KEY", "NEED_OPEN_DOOR", "GO_TO_GOAL"]

base = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
cfg = copy.deepcopy(base)
cfg.setdefault("environment", {})["name"] = "MiniGrid-DoorKey-5x5-v0"
cfg["environment"]["max_episode_steps"] = 250
cfg["seed"] = 42
sym = cfg.setdefault("symbolic", {})
sym["enabled"] = True
sym["knowledge_source"] = "affordance"
sym["broadcast_gate"] = "open"
sym["broadcast_lesion"] = False
cfg.setdefault("llm", {})["enabled"] = False
cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
cfg.setdefault("loop", {})["curriculum_enabled"] = False
cfg.setdefault("skill_bank", {})["storage_dir"] = "/tmp/rung6_transfer_verify_sb"

agent = DevelopmentalAI(config=cfg)
print(f"[config] policy.knowledge_dim={agent.policy.knowledge_dim} "
      f"gate_mode={agent.policy.knowledge_gate_mode} "
      f"broadcaster={type(agent.broadcaster).__name__} obs_dim={agent.obs_dim}")
assert agent.policy.knowledge_dim == 5 and agent.broadcaster is not None

# (1) Open gate must pass knowledge through UNCHANGED (force-concat).
o = torch.zeros(1, agent.obs_dim)
k = (torch.arange(agent.policy.knowledge_dim, dtype=torch.float32) + 1).unsqueeze(0)
gated = agent.policy.conditioner(o, k)
print(f"[gate ] open-gate identity (gated==knowledge): {torch.allclose(gated, k)}")

# (2) Policy input is genuinely augmented to obs_dim + 5.
aug = agent.policy._augment(o, k)
print(f"[shape] policy input dim: obs {agent.obs_dim} + knowledge "
      f"{agent.policy.knowledge_dim} -> augmented {aug.shape[-1]}")

# (3) LIVE rollout: print the broadcast vector at each newly-visited task state,
#     plus the lesion view of the identical state.
seen_phase, flips = {}, {"has_key": False, "door_open": False}
steps = 0
obs, info = agent._seeded_reset()
agent.broadcaster.reset(); agent.broadcaster.update(agent.env)
for ep in range(60):
    if ep > 0:
        obs, info = agent._seeded_reset()
        agent.broadcaster.reset(); agent.broadcaster.update(agent.env)
    done = False
    while not done and steps < 1200:
        kv = agent._current_knowledge_feature()            # INTACT (live path)
        agent.symbolic_broadcast_lesion = True
        kv_les = agent._current_knowledge_feature()         # LESION (same state)
        agent.symbolic_broadcast_lesion = False
        flips["has_key"] |= bool(kv[0] == 1.0)
        flips["door_open"] |= bool(kv[1] == 1.0)
        ph = PHASE[int(np.argmax(kv[2:]))]
        key = (int(kv[0]), int(kv[1]), ph)
        if key not in seen_phase:
            seen_phase[key] = (kv.copy(), kv_les.copy())
            print(f"  [live state #{len(seen_phase)}] has_key={int(kv[0])} "
                  f"door_open={int(kv[1])} phase={ph:14s} "
                  f"intact={np.array2string(kv, precision=0)} "
                  f"lesion={np.array2string(kv_les, precision=0)}")
        action, _ = agent.policy.select_action(obs, knowledge=kv)
        obs, r, term, trunc, _ = agent.env.step(action)
        agent.broadcaster.update(agent.env)
        done = term or trunc
        steps += 1
    if steps >= 1200:
        break

print(f"\n[live ] stepped {steps} steps; phases observed live: "
      f"{sorted({k[2] for k in seen_phase})}")
print(f"[live ] has_key flipped live: {flips['has_key']} | "
      f"door_open flipped live: {flips['door_open']}")

# (4) Forced-state check: drive the broadcaster to the door-open state so the
#     GO_TO_GOAL phase is demonstrated even if the untrained policy never solves.
agent.broadcaster._has_key = 1.0
agent.broadcaster._door_open = 1.0
print(f"[forced] has_key=1 door_open=1 -> {agent.broadcaster.feature()} "
      f"(expect GO_TO_GOAL: idx4=1)")

intact_nonzero = any(v[0].any() for v in seen_phase.values())
lesion_all_zero = all(not v[1].any() for v in seen_phase.values())
print(f"\n[VERDICT] intact carries content: {intact_nonzero} | "
      f"lesion severed to zeros: {lesion_all_zero}")
print("LIVE BROADCAST OK" if (intact_nonzero and lesion_all_zero) else "LIVE BROADCAST PROBLEM")
