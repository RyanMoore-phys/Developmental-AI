"""A/B test the causal action-alignment fix.

Hypothesis: the original observe_sequence pairs a_t with o_t, so the prior learns
to predict the next obs from the NEXT action (policy-correlation leak), not the
causal one — crippling action-conditioned prediction (and plausibly Rung-4
dreaming). The fix (world_model.causal_align) builds state_t from a_{t-1}, so
imagine_step(state_t, a_t) predicts o_{t+1} causally.

This trains ONE WM at the given alignment and measures, on real trajectories:
  - next-obs prediction error vs persistence, on FULL obs AND on the CHANGED dims
    (where o_{t+1} != o_t — the action-relevant part; full-obs MSE is swamped by
    the static background, which is why the first probe looked weak).
  - action-sensitivity: how much the predicted next-obs varies across the 7
    candidate actions (≈0 => the action input is ignored).
  - which action slot (a[t] vs a[t+1]) is the factual one.
Run with --causal 0 and --causal 1 and compare.
"""
import argparse, time
import numpy as np
import torch
import yaml

from developmental_ai.core.developmental_loop import DevelopmentalAI
from developmental_ai.world_model.rssm import symlog

ap = argparse.ArgumentParser()
ap.add_argument("--causal", type=int, default=0)
ap.add_argument("--steps", type=int, default=70000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

cfg = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
cfg.setdefault("environment", {})["name"] = "MiniGrid-DoorKey-6x6-v0"
cfg["environment"]["max_episode_steps"] = 360
cfg["seed"] = args.seed
cfg.setdefault("symbolic", {})["enabled"] = False
cfg.setdefault("llm", {})["enabled"] = False
cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
cfg.setdefault("loop", {})["curriculum_enabled"] = False
cfg.setdefault("dream_training", {})["enabled"] = False
cfg.setdefault("world_model", {})["goal_replay_fraction"] = 0.25
cfg["world_model"]["causal_align"] = bool(args.causal)
cfg.setdefault("skill_bank", {})["storage_dir"] = f"/tmp/causal_ab_sb_{args.causal}"

print(f"[ab causal={args.causal}] training WM {args.steps} steps...", flush=True)
agent = DevelopmentalAI(config=cfg)
print(f"[ab causal={args.causal}] world_model.action_shift={agent.world_model.action_shift}", flush=True)
t0 = time.time()
agent.run(total_timesteps=args.steps, log_interval=20000, verbose=0)
print(f"[ab causal={args.causal}] trained {time.time()-t0:.0f}s", flush=True)

wm = agent.world_model
wm.eval()
dev = next(wm.parameters()).device
A = agent.action_dim
CH = 0.05  # symlog-diff threshold for a "changed" obs dim


def oh(a):
    v = torch.zeros(1, A, device=dev)
    v[0, int(a)] = 1.0
    return v


@torch.no_grad()
def collect():
    o, _ = agent._seeded_reset()
    obs, acts = [np.asarray(o, np.float32)], []
    for _ in range(120):
        a, _ = agent.policy.select_action(o, knowledge=None)
        o, r, term, trunc, _ = agent.env.step(a)
        acts.append(int(a)); obs.append(np.asarray(o, np.float32))
        if term or trunc:
            break
    return obs, acts


@torch.no_grad()
def measure():
    full = {"a[t]": [], "a[t+1]": [], "persist": []}
    chg = {"a[t]": [], "a[t+1]": [], "persist": []}
    act_sens, n_changed = [], []
    for _ in range(50):
        obs, acts = collect()
        T = len(acts)
        if T < 8:
            continue
        obs_t = torch.tensor(np.stack(obs), dtype=torch.float32, device=dev).unsqueeze(0)
        for t in range(3, T - 1):
            a_seq = torch.zeros(1, t + 1, A, device=dev)
            for i in range(t + 1):
                a_seq[0, i, int(acts[i])] = 1.0
            states, _ = wm.observe_sequence(obs_t[:, : t + 1], a_seq)
            s_t = {"h": states["h"][:, t], "z": states["z"][:, t]}
            tgt = symlog(obs_t[:, t + 1]); cur = symlog(obs_t[:, t])
            changed = (tgt - cur).abs() > CH                    # (1, obs)
            nc = int(changed.sum().item())
            n_changed.append(nc)
            def err(pred, mask=None):
                if mask is None:
                    return float(torch.nn.functional.mse_loss(pred, tgt).item())
                if mask.sum() == 0:
                    return None
                return float(((pred - tgt)[mask] ** 2).mean().item())
            preds = {}
            for a in range(A):
                preds[a] = wm.decoder(wm.rssm.get_latent(wm.rssm.imagine_step(s_t, oh(a))))
            # action sensitivity: std across actions, averaged over dims
            stack = torch.stack([preds[a] for a in range(A)], dim=0)  # (A,1,obs)
            act_sens.append(float(stack.std(dim=0).mean().item()))
            for label, slot in (("a[t]", acts[t]), ("a[t+1]", acts[t + 1])):
                full[label].append(err(preds[slot]))
                e = err(preds[slot], changed)
                if e is not None:
                    chg[label].append(e)
            full["persist"].append(err(cur))
            ec = err(cur, changed)
            if ec is not None:
                chg["persist"].append(ec)
    return full, chg, act_sens, n_changed


full, chg, act_sens, n_changed = measure()
print(f"\n==== causal={args.causal}  (n_changed_dims avg={np.mean(n_changed):.1f}) ====", flush=True)
print("FULL-obs MSE to real next obs (swamped by static bg):", flush=True)
for k in ("a[t]", "a[t+1]", "persist"):
    print(f"   {k:>8}: {np.mean(full[k]):.4f}", flush=True)
print("CHANGED-dims MSE (the action-relevant part — THIS is the discriminating metric):", flush=True)
for k in ("a[t]", "a[t+1]", "persist"):
    print(f"   {k:>8}: {np.mean(chg[k]):.4f}", flush=True)
best = min(("a[t]", "a[t+1]"), key=lambda k: np.mean(chg[k]))
gain = np.mean(chg["persist"]) - np.mean(chg[best])
print(f"action-sensitivity (std across 7 actions; higher=action matters): {np.mean(act_sens):.4f}", flush=True)
print(f"BEST factual slot = {best}; changed-dim gain over persistence = {gain:+.4f}", flush=True)
print(f"RESULT causal={args.causal}: changed_best={np.mean(chg[best]):.4f} "
      f"persist={np.mean(chg['persist']):.4f} act_sens={np.mean(act_sens):.4f}", flush=True)
