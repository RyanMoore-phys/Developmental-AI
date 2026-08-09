#!/usr/bin/env python3
"""
Rung 4 VALIDITY GATE — world-model rollout accuracy.

Dreaming can only help if the world model's imagined rollouts are ACCURATE: you
cannot learn anything useful from dreams that diverge from reality after one or
two steps. This probe measures exactly that.

  1. Train an agent on a learnable env (the RSSM world model trains alongside).
  2. Collect held-out real trajectories with the trained policy.
  3. From a posterior latent state, IMAGINE forward H steps applying the REAL
     actions (open-loop, no observations), decode each imagined latent back to a
     predicted observation, and compare to the actual observation at that horizon
     (in the WM's symlog space — the same space it is trained on).
  4. Compare against a PERSISTENCE baseline (predict "no change" = the obs at the
     start of the rollout). The world model is only useful out to the horizon
     where it still BEATS persistence.

Reported: WM error vs horizon, persistence error vs horizon, and the
USEFUL HORIZON = the largest h where WM error < persistence error. A self-check
confirms the 1-step imagined error matches the WM's own get_prediction_error
(validates the action/obs alignment).

PASS (gate): useful horizon comfortably > 1 (dreams carry signal several steps
out). FAIL: WM error >= persistence even at h=1 (the WM can't dream) -> dreaming
is blocked until the world model is fixed.
"""
import argparse, copy, json, time
import numpy as np
import torch
import yaml

from developmental_ai.core.developmental_loop import DevelopmentalAI
from developmental_ai.world_model.rssm import symlog


def _onehot_seq(acts, n, device):
    t = torch.zeros(1, len(acts), n, device=device)
    for i, a in enumerate(acts):
        t[0, i, int(a)] = 1.0
    return t


def collect_traj(agent, traj_len):
    obs, acts = [], []
    o, _ = agent._seeded_reset()
    obs.append(np.asarray(o, dtype=np.float32))
    for _ in range(traj_len):
        a, _ = agent.policy.select_action(o, knowledge=None)
        o, r, term, trunc, _ = agent.env.step(a)
        acts.append(int(a))
        obs.append(np.asarray(o, dtype=np.float32))
        if term or trunc:
            break
    return obs, acts  # len(obs) == len(acts)+1


@torch.no_grad()
def probe(agent, n_traj, traj_len, context, horizon):
    wm = agent.world_model
    wm.eval()
    A, D = agent.action_dim, agent.obs_dim
    dev = next(wm.parameters()).device
    wm_err = np.zeros(horizon)
    per_err = np.zeros(horizon)
    cnt = np.zeros(horizon)
    selfcheck = []  # (h1_imagined, get_prediction_error) pairs

    collected = 0
    while collected < n_traj:
        obs, acts = collect_traj(agent, traj_len)
        T = len(acts)
        if T < context + horizon + 1:
            continue
        collected += 1
        obs_t = torch.tensor(np.stack(obs), dtype=torch.float32, device=dev).unsqueeze(0)  # (1,T+1,D)
        # DreamerV3 convention: action aligned with obs[t] is the action that
        # PRODUCED obs[t] (i.e. act[t-1]); obs[0] has a null action.
        act_aligned = [0] + acts  # act_aligned[t] produced obs[t]
        act_seq = _onehot_seq(act_aligned, A, dev)            # (1,T+1,A)
        states, _ = wm.observe_sequence(obs_t, act_seq)        # posterior latents

        for s in range(context, T - horizon):
            st = {"h": states["h"][:, s], "z": states["z"][:, s]}
            base_symlog = symlog(obs_t[:, s])                 # persistence target
            for h in range(horizon):
                a_next = act_seq[:, s + 1 + h]                 # action taken at obs[s+h]
                st = wm.rssm.imagine_step(st, a_next)
                pred = wm.decoder(wm.rssm.get_latent(st))      # symlog space
                target = symlog(obs_t[:, s + 1 + h])
                wm_err[h] += float(((pred - target) ** 2).mean())
                per_err[h] += float(((base_symlog - target) ** 2).mean())
                cnt[h] += 1
                if h == 0:
                    gpe = wm.get_prediction_error(
                        {"h": states["h"][:, s], "z": states["z"][:, s]},
                        act_seq[:, s + 1], obs_t[:, s + 1]).item()
                    selfcheck.append((float(((pred - target) ** 2).mean()), gpe))

    wm_err /= np.maximum(cnt, 1)
    per_err /= np.maximum(cnt, 1)
    useful_h = 0
    for h in range(horizon):
        if wm_err[h] < per_err[h]:
            useful_h = h + 1
        else:
            break
    sc = np.array(selfcheck)
    sc_ok = bool(len(sc) and np.allclose(sc[:, 0], sc[:, 1], rtol=0.05, atol=1e-4))
    return {"wm_err": wm_err.tolist(), "persistence_err": per_err.tolist(),
            "useful_horizon": int(useful_h), "n_rollouts": int(cnt[0]),
            "selfcheck_h1_matches_gpe": sc_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--train-timesteps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--n-traj", type=int, default=40)
    ap.add_argument("--traj-len", type=int, default=60)
    ap.add_argument("--base", default="configs/minigrid_doorkey.yaml")
    ap.add_argument("--out", default="wm_rollout_probe_results")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base))
    cfg = copy.deepcopy(base)
    cfg.setdefault("environment", {})["name"] = args.env
    cfg["environment"]["max_episode_steps"] = args.max_steps
    cfg["seed"] = args.seed
    cfg.setdefault("symbolic", {})["enabled"] = False       # WM probe needs no knowledge
    cfg.setdefault("llm", {})["enabled"] = False
    cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
    cfg.setdefault("loop", {})["curriculum_enabled"] = False
    cfg.setdefault("skill_bank", {})["storage_dir"] = f"/tmp/wm_probe_sb_seed{args.seed}"

    print(f"[wm-probe] building agent on {args.env}; training WM for "
          f"{args.train_timesteps} steps...", flush=True)
    agent = DevelopmentalAI(config=cfg)
    t0 = time.time()
    agent.run(total_timesteps=args.train_timesteps, log_interval=20, verbose=1)
    print(f"[wm-probe] trained in {time.time()-t0:.0f}s; probing rollout accuracy "
          f"(context={args.context} horizon={args.horizon} n_traj={args.n_traj})...",
          flush=True)

    res = probe(agent, args.n_traj, args.traj_len, args.context, args.horizon)
    import os
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"env": args.env, "args": vars(args), **res}, f, indent=2)

    print("\n==== WM ROLLOUT ACCURACY ====", flush=True)
    print(f"self-check (h1 imagined == get_prediction_error): {res['selfcheck_h1_matches_gpe']}",
          flush=True)
    print(f"{'h':>3} | {'WM err':>10} | {'persist':>10} | {'WM<persist?':>11}", flush=True)
    for h in range(args.horizon):
        w, p = res["wm_err"][h], res["persistence_err"][h]
        print(f"{h+1:>3} | {w:>10.4f} | {p:>10.4f} | {('yes' if w < p else 'NO'):>11}", flush=True)
    print(f"\nUSEFUL HORIZON (WM beats persistence): {res['useful_horizon']} steps "
          f"(from {res['n_rollouts']} rollouts)", flush=True)
    print("GATE: " + ("PASS — dreams carry signal" if res["useful_horizon"] >= 3
                      else "WEAK/FAIL — WM rollouts diverge too fast for useful dreaming"),
          flush=True)


if __name__ == "__main__":
    main()
