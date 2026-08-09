import copy, yaml
from developmental_ai.core.developmental_loop import DevelopmentalAI

base = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
cfg = copy.deepcopy(base)
cfg.setdefault("environment", {})["name"] = "MiniGrid-DoorKey-5x5-v0"
cfg["environment"]["max_episode_steps"] = 250
cfg["seed"] = 42
cfg.setdefault("symbolic", {})["enabled"] = False
cfg.setdefault("llm", {})["enabled"] = False
cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
cfg.setdefault("loop", {})["curriculum_enabled"] = False
dt = cfg.setdefault("dream_training", {})
dt["enabled"] = True
dt["control"] = False               # AUGMENT mode (real policy keeps control)
dt["activate_after_episodes"] = 20
dt["distill_weight"] = 1.0
cfg.setdefault("skill_bank", {})["storage_dir"] = "/tmp/dream_smoke_sb"

a = DevelopmentalAI(config=cfg)
print(f"[build] dream_augment={a.dream_augment} dream_control={a.dream_control} "
      f"activate_after={a.dream_activate_after} distill_opt={a._distill_opt is not None} "
      f"latent_dim={a.world_model.rssm.latent_dim} action_dim={a.action_dim}", flush=True)
a.run(total_timesteps=40000, log_interval=40, verbose=0)
dl = list(a.training_metrics["dream_distill_loss"])
da = list(a.training_metrics["dream_actor_loss"])
print(f"[end] dream_training_active(should be False)={a.dream_training_active}  "
      f"dream_augment_active={a.dream_augment_active}", flush=True)
print(f"[end] distill_loss: n={len(dl)} first={(dl[0] if dl else 0):.4f} "
      f"last={(dl[-1] if dl else 0):.4f}  dream_actor_loss n={len(da)}", flush=True)
frac = sum(1 for r in a.training_metrics['episode_reward'] if r > 0.1) / max(1, len(a.training_metrics['episode_reward']))
print(f"[end] PPO-controlled run frac_solved={frac:.3f} episodes={len(a.training_metrics['episode_reward'])}", flush=True)
ok = (len(dl) > 0) and (not a.dream_training_active)
print("\nDREAM AUGMENT SMOKE OK" if ok else "\nDREAM AUGMENT SMOKE PROBLEM", flush=True)
