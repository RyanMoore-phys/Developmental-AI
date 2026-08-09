"""Verify the dream-distill fix composes with SYMBOLIC ON (the capstone bug)."""
import copy, yaml
from developmental_ai.core.developmental_loop import DevelopmentalAI

cfg = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
cfg.setdefault("environment", {})["name"] = "MiniGrid-DoorKey-6x6-v0"
cfg["environment"]["max_episode_steps"] = 360
cfg["seed"] = 42
cfg.setdefault("symbolic", {})["enabled"] = True          # the key: symbolic ON
cfg.setdefault("llm", {})["enabled"] = False
cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
cfg.setdefault("loop", {})["curriculum_enabled"] = False
cfg.setdefault("world_model", {})["goal_replay_fraction"] = 0.25
cfg["world_model"]["causal_align"] = True
dt = cfg.setdefault("dream_training", {})
dt["enabled"] = True; dt["control"] = False; dt["activate_after_episodes"] = 20
cfg.setdefault("skill_bank", {})["storage_dir"] = "/tmp/dream_sym_smoke_sb"

a = DevelopmentalAI(config=cfg)
print(f"[build] symbolic_on={a.policy.conditioner is not None} dream_augment={a.dream_augment}", flush=True)
a.run(total_timesteps=30000, log_interval=20000, verbose=0)
dl = list(a.training_metrics["dream_distill_loss"])
gf = list(a.training_metrics["dream_distill_gate_frac"])
print(f"[end] dream_distill_updates={len(dl)} (was 0 with the bug; >0 = FIXED)", flush=True)
print(f"[end] distill_loss last={dl[-1] if dl else 'NA'} gate_frac last={gf[-1] if gf else 'NA'} "
      f"warned={getattr(a,'_dream_augment_warned',False)}", flush=True)
ok = len(dl) > 0 and not getattr(a, "_dream_augment_warned", False)
print("\nDREAM+SYMBOLIC FIX OK" if ok else "\nDREAM+SYMBOLIC STILL BROKEN", flush=True)
