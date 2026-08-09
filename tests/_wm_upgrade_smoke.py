"""Integration smoke for the WM-accuracy upgrades (twohot head + goal replay)
running through the full agent with dream-augment ON. Confirms the pipeline
trains end-to-end and the reward head discriminates the goal in TRAINING
alignment (same measurement as wm_reward_probe2.py)."""
import copy, time
import numpy as np
import torch
import yaml
from developmental_ai.core.developmental_loop import DevelopmentalAI

base = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
cfg = copy.deepcopy(base)
cfg.setdefault("environment", {})["name"] = "MiniGrid-DoorKey-6x6-v0"
cfg["environment"]["max_episode_steps"] = 360
cfg["seed"] = 42
cfg.setdefault("symbolic", {})["enabled"] = False
cfg.setdefault("llm", {})["enabled"] = False
cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
cfg.setdefault("loop", {})["curriculum_enabled"] = False
cfg.setdefault("world_model", {})["goal_replay_fraction"] = 0.25      # NEW
dt = cfg.setdefault("dream_training", {})
dt["enabled"] = True; dt["control"] = False
dt["activate_after_episodes"] = 20; dt["distill_weight"] = 1.0
cfg.setdefault("skill_bank", {})["storage_dir"] = "/tmp/wm_upgrade_smoke_sb"

a = DevelopmentalAI(config=cfg)
print(f"[build] goal_replay={a._goal_replay_fraction} reward_bins={a.world_model.reward_bins.numel()} "
      f"dream_augment={a.dream_augment}", flush=True)
t0 = time.time()
a.run(total_timesteps=40000, log_interval=40, verbose=0)
print(f"[run] {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def measure(agent, n_seq=512, seq_len=32):
    wm = agent.world_model; wm.eval()
    dev = next(wm.parameters()).device
    data = agent.replay_buffer.sample_sequences(n_seq, seq_len, device=dev)
    obs, acts, rew = data["observations"], data["actions"], data["rewards"]
    states, _ = wm.observe_sequence(obs, acts)
    lat = torch.cat([states["h"], states["z"]], dim=-1).reshape(-1, wm.rssm.latent_dim)
    pred = wm.predict_reward(lat).reshape(-1)
    rflat = rew.reshape(-1); goal = rflat > 0.1
    pg = pred[goal]; pn = pred[~goal]
    aa = pred - pred.mean(); bb = rflat - rflat.mean()
    corr = float((aa*bb).sum() / (aa.norm()*bb.norm() + 1e-8))
    disc = float(pg.mean() - pn.mean()) if goal.any() else float("nan")
    return int(goal.sum()), int(rflat.numel()), (float(pg.mean()) if goal.any() else float("nan")), float(pn.mean()), disc, corr

ng, nt, pgm, pnm, disc, corr = measure(a)
gf = list(a.training_metrics["dream_distill_gate_frac"])
dl = list(a.training_metrics["dream_distill_loss"])
print(f"[reward-head] goal {ng}/{nt}  pred@goal={pgm:.3f} pred@nongoal={pnm:.3f} "
      f"DISCRIM={disc:+.3f} corr={corr:+.3f}", flush=True)
print(f"[dream] distill_n={len(dl)} gate_frac_mean={(sum(gf)/len(gf) if gf else 0):.3f} "
      f"solve_ema={a._dream_solve_ema:.3f}", flush=True)
frac = sum(1 for r in a.training_metrics['episode_reward'] if r > 0.1) / max(1, len(a.training_metrics['episode_reward']))
print(f"[run] frac_solved={frac:.3f} episodes={len(a.training_metrics['episode_reward'])}", flush=True)

ok = (not np.isnan(disc)) and disc > 0.2 and len(dl) > 0 and a.world_model.reward_bins.numel() == 255
print("\nWM UPGRADE SMOKE OK" if ok else f"\nWM UPGRADE SMOKE PROBLEM (disc={disc} distill_n={len(dl)})", flush=True)
