#!/usr/bin/env python3
"""
Rung 4 DIAGNOSIS — world-model REWARD/CONTINUE accuracy.

The obs-rollout gate (wm_rollout_probe.py) passed, yet dream-augment HURT. The
dream actor optimizes IMAGINED REWARDS, so if the WM's reward predictor is wrong,
the dreamed policy is misguided and distilling it harms PPO. The obs gate never
tested this. This probe does.

Tested on POSTERIOR (true) latents — decoupled from rollout divergence: given the
real state, does the reward head know the reward, and the continue head know
termination? Sparse-reward task (DoorKey: ~0 everywhere, a spike at the goal), so
the key question is DISCRIMINATION: does predicted reward differ between goal and
non-goal states? If not, the reward signal the dream actor learns from is blind to
the goal — the smoking gun.
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import yaml

from developmental_ai.core.developmental_loop import DevelopmentalAI
from developmental_ai.world_model.rssm import symexp


def collect(agent, traj_len):
    obs, acts, rew_at_obs, done_at_obs = [], [], [0.0], [0.0]
    o, _ = agent._seeded_reset()
    obs.append(np.asarray(o, dtype=np.float32))
    for _ in range(traj_len):
        a, _ = agent.policy.select_action(o, knowledge=None)
        o, r, term, trunc, _ = agent.env.step(a)
        acts.append(int(a)); obs.append(np.asarray(o, dtype=np.float32))
        rew_at_obs.append(float(r)); done_at_obs.append(1.0 if term else 0.0)
        if term or trunc:
            break
    return obs, acts, rew_at_obs, done_at_obs


@torch.no_grad()
def probe(agent, n_traj, traj_len):
    wm = agent.world_model; wm.eval()
    A = agent.action_dim; dev = next(wm.parameters()).device
    pr_goal, pr_nongoal = [], []        # predicted reward (real scale) by class
    act_goal_n = 0
    cont_term, cont_nonterm = [], []    # predicted continue prob by class
    collected = 0; n_goal = 0
    while collected < n_traj:
        obs, acts, rew, done = collect(agent, traj_len)
        T = len(acts)
        if T < 5:
            continue
        collected += 1
        obs_t = torch.tensor(np.stack(obs), dtype=torch.float32, device=dev).unsqueeze(0)
        act_aligned = [0] + acts
        act_seq = torch.zeros(1, len(act_aligned), A, device=dev)
        for i, a in enumerate(act_aligned):
            act_seq[0, i, int(a)] = 1.0
        states, _ = wm.observe_sequence(obs_t, act_seq)
        for t in range(1, T + 1):                      # obs index (reward arrives at obs[t])
            lat = wm.rssm.get_latent({"h": states["h"][:, t], "z": states["z"][:, t]})
            pred_r = float(wm.predict_reward(lat).mean())  # real scale (twohot head)
            pred_c = float(torch.sigmoid(wm.continue_predictor(lat)).mean())
            if rew[t] > 0.1:
                pr_goal.append(pred_r); n_goal += 1
            else:
                pr_nongoal.append(pred_r)
            if done[t] > 0.5:
                cont_term.append(pred_c)
            else:
                cont_nonterm.append(pred_c)
    def ms(x):
        return (float(np.mean(x)), float(np.std(x)), len(x)) if x else (float("nan"), 0.0, 0)
    return {"reward_goal": ms(pr_goal), "reward_nongoal": ms(pr_nongoal),
            "continue_terminal": ms(cont_term), "continue_nonterminal": ms(cont_nonterm),
            "n_traj": collected, "n_goal_states": n_goal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--train-timesteps", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-traj", type=int, default=80)
    ap.add_argument("--traj-len", type=int, default=120)
    ap.add_argument("--base", default="configs/minigrid_doorkey.yaml")
    ap.add_argument("--out", default="wm_reward_probe_results")
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
    cfg.setdefault("skill_bank", {})["storage_dir"] = f"/tmp/wm_reward_probe_sb_{args.seed}"

    print(f"[reward-probe] training WM on {args.env} for {args.train_timesteps} steps...",
          flush=True)
    agent = DevelopmentalAI(config=cfg)
    t0 = time.time()
    agent.run(total_timesteps=args.train_timesteps, log_interval=20, verbose=1)
    print(f"[reward-probe] trained in {time.time()-t0:.0f}s; probing reward/continue...",
          flush=True)
    res = probe(agent, args.n_traj, args.traj_len)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"env": args.env, **res}, open(os.path.join(args.out, "results.json"), "w"), indent=2)

    rg, rn = res["reward_goal"], res["reward_nongoal"]
    ct, cn = res["continue_terminal"], res["continue_nonterminal"]
    print("\n==== WM REWARD / CONTINUE ACCURACY (posterior states) ====", flush=True)
    print(f"goal states seen: {res['n_goal_states']} over {res['n_traj']} trajectories", flush=True)
    print(f"predicted reward @ GOAL    states: mean={rg[0]:.4f} std={rg[1]:.4f} (n={rg[2]})", flush=True)
    print(f"predicted reward @ NONGOAL states: mean={rn[0]:.4f} std={rn[1]:.4f} (n={rn[2]})", flush=True)
    sep = (rg[0] - rn[0]) if not np.isnan(rg[0]) else float("nan")
    print(f"  -> reward DISCRIMINATION (goal - nongoal): {sep:+.4f}  "
          f"(actual gap ~0.9; near 0 = reward head BLIND to goal)", flush=True)
    print(f"predicted continue @ TERMINAL    : mean={ct[0]:.3f} (n={ct[2]})  (should be LOW)", flush=True)
    print(f"predicted continue @ NONTERMINAL : mean={cn[0]:.3f} (n={cn[2]})  (should be HIGH)", flush=True)
    blind = (np.isnan(rg[0]) or rg[2] == 0 or sep < 0.2)
    print("\nDIAGNOSIS: " + ("REWARD HEAD BLIND/WEAK -> dream actor optimizes a bad signal "
          "-> distillation hurts (root cause confirmed)" if blind else
          "reward head discriminates the goal -> root cause is elsewhere"), flush=True)


if __name__ == "__main__":
    main()
