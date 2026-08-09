"""Rung 8 prep — empirically nail the RSSM action alignment.

observe_step consumes action_t to BUILD state_t; imagine_step consumes an action
to transition OUT of a state. Mixing them (abduce posterior, then imagine a
counterfactual action) needs the right slot, or the counterfactual is garbage
(this is the exact class of bug that fooled the reward probe). So: train a WM,
take real trajectories, abduce s_t = observe_sequence(o[0:t+1], a[0:t+1]), and
test which action predicts the REAL o[t+1]:
    cand a[t]   : decode(imagine_step(s_t, a[t]))
    cand a[t+1] : decode(imagine_step(s_t, a[t+1]))
vs a persistence baseline (predict o[t]). Also sanity: decode(s_t) ~ o[t].
The slot with the lowest error to o[t+1] is the FACTUAL transition; the
counterfactual then swaps in a' at that same slot.
"""
import copy, time
import numpy as np
import torch
import yaml

from developmental_ai.core.developmental_loop import DevelopmentalAI
from developmental_ai.world_model.rssm import symlog

cfg = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
cfg.setdefault("environment", {})["name"] = "MiniGrid-DoorKey-6x6-v0"
cfg["environment"]["max_episode_steps"] = 360
cfg["seed"] = 42
cfg.setdefault("symbolic", {})["enabled"] = False
cfg.setdefault("llm", {})["enabled"] = False
cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
cfg.setdefault("loop", {})["curriculum_enabled"] = False
cfg.setdefault("dream_training", {})["enabled"] = False
cfg.setdefault("world_model", {})["goal_replay_fraction"] = 0.25
cfg.setdefault("skill_bank", {})["storage_dir"] = "/tmp/cf_align_sb"

print("[cf-align] training WM ~70k steps...", flush=True)
agent = DevelopmentalAI(config=cfg)
t0 = time.time()
agent.run(total_timesteps=70000, log_interval=20000, verbose=0)
print(f"[cf-align] trained in {time.time()-t0:.0f}s", flush=True)

wm = agent.world_model
wm.eval()
dev = next(wm.parameters()).device
A = agent.action_dim


@torch.no_grad()
def collect():
    o, _ = agent._seeded_reset()
    obs = [np.asarray(o, np.float32)]
    acts = []
    for _ in range(120):
        a, _ = agent.policy.select_action(o, knowledge=None)
        o, r, term, trunc, _ = agent.env.step(a)
        acts.append(int(a))
        obs.append(np.asarray(o, np.float32))
        if term or trunc:
            break
    return obs, acts


def oh(a):
    v = torch.zeros(1, A, device=dev)
    v[0, int(a)] = 1.0
    return v


@torch.no_grad()
def run():
    errs = {"a[t]": [], "a[t+1]": [], "persist": [], "recon s_t~o[t]": []}
    for _ in range(40):
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
            tgt = symlog(obs_t[:, t + 1])                       # real next obs (symlog)
            recon_t = symlog(obs_t[:, t])
            def derr(pred):
                return float(torch.nn.functional.mse_loss(pred, tgt).item())
            s_at = wm.rssm.imagine_step(s_t, oh(acts[t]))
            s_at1 = wm.rssm.imagine_step(s_t, oh(acts[t + 1]))
            errs["a[t]"].append(derr(wm.decoder(wm.rssm.get_latent(s_at))))
            errs["a[t+1]"].append(derr(wm.decoder(wm.rssm.get_latent(s_at1))))
            errs["persist"].append(derr(recon_t))
            errs["recon s_t~o[t]"].append(
                float(torch.nn.functional.mse_loss(
                    wm.decoder(wm.rssm.get_latent(s_t)), recon_t).item()))
    return errs


errs = run()
print("\n==== ALIGNMENT (mean MSE to REAL next obs symlog; lower=better) ====", flush=True)
for k in ("a[t]", "a[t+1]", "persist", "recon s_t~o[t]"):
    v = errs[k]
    print(f"  {k:>16}: {np.mean(v):.4f}  (n={len(v)})", flush=True)
best = min(("a[t]", "a[t+1]"), key=lambda k: np.mean(errs[k]))
beats = np.mean(errs[best]) < np.mean(errs["persist"])
print(f"\nFACTUAL next-obs slot = {best}; beats persistence: {beats}", flush=True)
print("USE THIS SLOT for the counterfactual action in rung8.", flush=True)
