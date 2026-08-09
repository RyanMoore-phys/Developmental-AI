#!/usr/bin/env python3
"""
Rung 4 DIAGNOSIS v2 — reward head accuracy in the EXACT TRAINING ALIGNMENT.

The first reward probe shifted the action sequence (act_aligned=[0]+acts), which
fed the reward head latents computed with a DIFFERENT action alignment than the
WM was trained on (training calls observe_sequence(obs, actions) with the replay's
action_t paired with obs_t, no shift). So the head may have looked blind only
because it was being read at the wrong latents.

This probe removes all ambiguity: it samples sequences from the agent's OWN replay
buffer (the exact training data) and runs observe_sequence(obs, actions) exactly as
compute_loss does, then compares the reward head's prediction to the same
symlog(reward) target, segregated by goal (reward>0.1) vs non-goal steps. If the
head discriminates HERE, it learned fine and the earlier blindness was a probe bug.
"""
import argparse, copy, time
import numpy as np
import torch
import yaml

from developmental_ai.core.developmental_loop import DevelopmentalAI
from developmental_ai.world_model.rssm import symexp


@torch.no_grad()
def measure(agent, n_seq=512, seq_len=32):
    wm = agent.world_model; wm.eval()
    dev = next(wm.parameters()).device
    data = agent.replay_buffer.sample_sequences(n_seq, seq_len, device=dev)
    obs, acts, rew = data["observations"], data["actions"], data["rewards"]
    states, _ = wm.observe_sequence(obs, acts)                 # EXACT training path
    lat = torch.cat([states["h"], states["z"]], dim=-1).reshape(-1, wm.rssm.latent_dim)
    pred_real = wm.predict_reward(lat).reshape(-1)             # real scale (twohot head)
    rflat = rew.reshape(-1)
    goal = rflat > 0.1
    pg, pn = pred_real[goal], pred_real[~goal]
    # correlation of predicted vs actual over all steps
    a = pred_real - pred_real.mean(); b = rflat - rflat.mean()
    corr = float((a * b).sum() / (a.norm() * b.norm() + 1e-8))
    return {
        "n_goal": int(goal.sum().item()), "n_total": int(rflat.numel()),
        "pred_goal_mean": float(pg.mean()) if goal.any() else float("nan"),
        "pred_nongoal_mean": float(pn.mean()),
        "actual_goal_mean": float(rflat[goal].mean()) if goal.any() else float("nan"),
        "discrimination": (float(pg.mean() - pn.mean()) if goal.any() else float("nan")),
        "corr_pred_actual": corr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--train-timesteps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", default="configs/minigrid_doorkey.yaml")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base))
    cfg = copy.deepcopy(base)
    cfg.setdefault("environment", {})["name"] = args.env
    cfg["environment"]["max_episode_steps"] = args.max_steps
    cfg["seed"] = args.seed
    cfg.setdefault("symbolic", {})["enabled"] = False
    cfg.setdefault("llm", {})["enabled"] = False
    cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
    cfg.setdefault("loop", {})["curriculum_enabled"] = False
    cfg.setdefault("skill_bank", {})["storage_dir"] = f"/tmp/wm_rp2_sb_{args.seed}"

    print(f"[rp2] training WM on {args.env} for {args.train_timesteps} steps...", flush=True)
    agent = DevelopmentalAI(config=cfg)
    t0 = time.time()
    agent.run(total_timesteps=args.train_timesteps, log_interval=20, verbose=1)
    print(f"[rp2] trained in {time.time()-t0:.0f}s; measuring in TRAINING alignment...", flush=True)
    r = measure(agent)
    print("\n==== REWARD HEAD — TRAINING-ALIGNED ====", flush=True)
    print(f"goal steps: {r['n_goal']} / {r['n_total']}", flush=True)
    print(f"predicted reward @ GOAL    : {r['pred_goal_mean']:.4f}  (actual ~{r['actual_goal_mean']:.3f})", flush=True)
    print(f"predicted reward @ NONGOAL : {r['pred_nongoal_mean']:.4f}", flush=True)
    print(f"DISCRIMINATION (goal-nongoal): {r['discrimination']:+.4f}", flush=True)
    print(f"corr(pred, actual) over all steps: {r['corr_pred_actual']:+.3f}", flush=True)
    good = (not np.isnan(r["discrimination"])) and r["discrimination"] > 0.2
    print("\nVERDICT: " + ("REWARD HEAD WORKS in training alignment -> earlier blindness "
          "was a PROBE alignment bug; dreaming failure is elsewhere" if good else
          "reward head genuinely weak even in training alignment -> MSE head is the issue "
          "(twohot needed)"), flush=True)


if __name__ == "__main__":
    main()
