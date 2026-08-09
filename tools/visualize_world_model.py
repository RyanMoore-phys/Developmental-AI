"""
Peer into the world model: train a compact RSSM (FiLM + inverse dynamics ON)
on raw CartPole, then VISUALIZE what it has learned.

Produces three lenses into the model's "mind":
  1. dream_vs_reality.gif  — condition on a few real frames, then let the model
     IMAGINE forward under the real action sequence. Left = reality, right = the
     model's dream, decoded from latent space back into renderable CartPole.
  2. action_probe.png      — from one state, imagine LEFT vs RIGHT. Demonstrates
     the FiLM action-conditioning fix: the action actually moves the prediction.
  3. latent_map.png        — PCA of the model's internal latent states along a
     trajectory, colored by pole angle. A 2D map of the representation.

Usage:  python visualize_world_model.py [train_iters]
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from developmental_ai.world_model.rssm import WorldModel, symexp

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

OBS_DIM = 4          # CartPole: [cart_x, cart_vel, pole_angle, pole_ang_vel]
ACTION_DIM = 2       # {0: push left, 1: push right}
SEQ_LEN = 15
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "outputs", "world_model_view")
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. DATA COLLECTION  (mix of random + a crude balancer for diverse coverage)
# ----------------------------------------------------------------------------
def heuristic_action(obs):
    # Push in the direction the pole is falling — keeps some episodes alive long
    # enough to give the model balanced, upright trajectories to learn from.
    angle, ang_vel = obs[2], obs[3]
    return 1 if (angle + 0.5 * ang_vel) > 0 else 0


def collect(n_transitions):
    env = gym.make("CartPole-v1")
    episodes, ep = [], []
    obs, _ = env.reset(seed=SEED)
    total = 0
    while total < n_transitions:
        a = heuristic_action(obs) if np.random.rand() < 0.5 else env.action_space.sample()
        nobs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        ep.append((obs.astype(np.float32), a, float(r), 0.0 if done else 1.0))
        obs = nobs
        total += 1
        if done:
            episodes.append(ep); ep = []
            obs, _ = env.reset()
    if len(ep) > 1:
        episodes.append(ep)
    env.close()
    return episodes


def make_batches(episodes, seq_len):
    """Slice episodes into contiguous (obs, act_onehot, rew, cont) sequences."""
    seqs = []
    for ep in episodes:
        if len(ep) < seq_len:
            continue
        for s in range(0, len(ep) - seq_len + 1, seq_len // 2):  # 50% overlap
            chunk = ep[s:s + seq_len]
            o = np.stack([c[0] for c in chunk])
            a = np.array([c[1] for c in chunk])
            r = np.array([c[2] for c in chunk], dtype=np.float32)
            c = np.array([c[3] for c in chunk], dtype=np.float32)
            a_oh = np.eye(ACTION_DIM, dtype=np.float32)[a]
            seqs.append((o, a_oh, r, c))
    return seqs


# ----------------------------------------------------------------------------
# 2. TRAIN THE WORLD MODEL
# ----------------------------------------------------------------------------
def train_world_model(iters):
    print(f"[collect] gathering transitions...")
    episodes = collect(12000)
    seqs = make_batches(episodes, SEQ_LEN)
    print(f"[collect] {len(episodes)} episodes -> {len(seqs)} training sequences")

    wm = WorldModel(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        film_conditioning=True, inverse_dynamics=True,
        inverse_dynamics_scale=1.0, discrete_actions=True,
    )
    opt = torch.optim.Adam(wm.parameters(), lr=3e-4)

    O = torch.tensor(np.stack([s[0] for s in seqs]))
    A = torch.tensor(np.stack([s[1] for s in seqs]))
    R = torch.tensor(np.stack([s[2] for s in seqs]))
    C = torch.tensor(np.stack([s[3] for s in seqs]))
    N = O.shape[0]
    bs = min(32, N)

    print(f"[train] {iters} gradient steps (batch {bs})...")
    wm.train()
    curve = []
    for it in range(iters):
        idx = torch.randint(0, N, (bs,))
        losses = wm.compute_loss(O[idx], A[idx], R[idx], C[idx])
        loss = losses["total"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
        curve.append(loss.item())
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            print(f"  iter {it:5d}  total={loss.item():7.3f}  "
                  f"recon={losses['reconstruction'].item():.3f}  "
                  f"inverse={losses['inverse'].item():.3f}")
    wm.eval()
    return wm, curve


# ----------------------------------------------------------------------------
# 3. RENDERING — draw a CartPole from a 4-D state vector
# ----------------------------------------------------------------------------
def draw_cartpole(ax, state, title=None, cart_color="steelblue",
                  pole_color="darkorange", subtitle=None):
    x, theta = float(state[0]), float(state[2])
    ax.clear()
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.45, 1.7)
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot([-2.4, 2.4], [0, 0], color="0.6", lw=2, zorder=0)
    for bx in (-2.4, 2.4):                       # failure bounds
        ax.plot([bx, bx], [-0.05, 0.08], color="crimson", lw=2, zorder=1)
    cw, ch = 0.5, 0.3
    ax.add_patch(plt.Rectangle((x - cw / 2, 0), cw, ch, color=cart_color, zorder=2))
    L = 1.0
    tx, ty = x + L * math.sin(theta), ch + L * math.cos(theta)
    ax.plot([x, tx], [ch, ty], color=pole_color, lw=6, zorder=3,
            solid_capstyle="round")
    ax.plot([tx], [ty], "o", color=pole_color, ms=10, zorder=4)
    ax.plot([x], [ch], "o", color="0.2", ms=6, zorder=4)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if subtitle:
        ax.text(0, -0.38, subtitle, ha="center", fontsize=9, color="0.3")


def fig_to_image(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")


# ----------------------------------------------------------------------------
# 4. OPEN-LOOP DREAM: condition on W real frames, then imagine the rest
# ----------------------------------------------------------------------------
def onehot(a):
    v = torch.zeros(1, ACTION_DIM); v[0, a] = 1.0
    return v


def decode_state(wm, state):
    latent = wm.rssm.get_latent(state)
    return symexp(wm.decoder(latent)).squeeze(0).detach().numpy()


def rollout_real(policy_warmup=5, horizon=18):
    """Run a real episode, return obs[], act[] with a balanced start."""
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=SEED + 7)
    O, A = [obs.astype(np.float32)], []
    for _ in range(policy_warmup + horizon):
        a = heuristic_action(obs)               # keep it alive & coherent
        obs, _, term, trunc, _ = env.step(a)
        A.append(a); O.append(obs.astype(np.float32))
        if term or trunc:
            break
    env.close()
    return np.stack(O), np.array(A)


def dream_vs_reality(wm, warmup=5, horizon=18):
    shift = getattr(wm, "action_shift", True)
    O, A = rollout_real(warmup, horizon)
    T = len(A)
    warmup = min(warmup, T - 2)

    # --- warm the recurrent state on the first `warmup` real observations ---
    state = wm.rssm.initial_state(1, torch.device("cpu"))
    with torch.no_grad():
        for t in range(warmup):
            causal_a = onehot(A[t - 1]) if (shift and t > 0) else \
                       (onehot(A[t]) if not shift else torch.zeros(1, ACTION_DIM))
            emb = wm.encoder(torch.tensor(O[t]).unsqueeze(0))
            state, _ = wm.rssm.observe_step(state, causal_a, emb)

        # --- branch into pure imagination under the REAL action sequence ---
        dream_states = [state]
        for t in range(warmup, T):
            state = wm.rssm.imagine_step(state, onehot(A[t]))
            dream_states.append(state)

    real_track = O[warmup:warmup + len(dream_states)]
    dream_track = [O[warmup]] + [decode_state(wm, s) for s in dream_states[1:]]

    # --- render side-by-side frames -> GIF ---
    frames = []
    for k in range(len(dream_track)):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        rs = real_track[k] if k < len(real_track) else real_track[-1]
        ds = dream_track[k]
        draw_cartpole(axes[0], rs, "REALITY", cart_color="steelblue",
                      subtitle=f"x={rs[0]:+.2f}  angle={math.degrees(rs[2]):+.1f}deg")
        draw_cartpole(axes[1], ds, "DREAM (imagined)", cart_color="seagreen",
                      pole_color="orangered",
                      subtitle=f"x={ds[0]:+.2f}  angle={math.degrees(ds[2]):+.1f}deg")
        tag = "conditioning on reality" if k == 0 else f"imagined step {k}"
        fig.suptitle(f"World model open-loop prediction — {tag}",
                     fontsize=12, y=0.97)
        fig.subplots_adjust(top=0.80, bottom=0.06, left=0.03, right=0.97)
        frames.append(fig_to_image(fig))
        plt.close(fig)

    # hold the final frame a beat
    frames += [frames[-1]] * 6
    path = os.path.join(OUTDIR, "dream_vs_reality.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=260, loop=0)

    # divergence over horizon
    div = [float(np.abs(real_track[k] - dream_track[k]).mean())
           for k in range(min(len(real_track), len(dream_track)))]
    print(f"[dream] saved {path}")
    print(f"[dream] mean |reality - dream| per step: "
          f"{', '.join(f'{d:.3f}' for d in div)}")
    return path


# ----------------------------------------------------------------------------
# 5. ACTION PROBE: does the action move the prediction?  (FiLM check)
# ----------------------------------------------------------------------------
def action_probe(wm, warmup=6):
    shift = getattr(wm, "action_shift", True)
    O, A = rollout_real(warmup, 4)
    state = wm.rssm.initial_state(1, torch.device("cpu"))
    with torch.no_grad():
        for t in range(warmup):
            causal_a = onehot(A[t - 1]) if (shift and t > 0) else \
                       (onehot(A[t]) if not shift else torch.zeros(1, ACTION_DIM))
            emb = wm.encoder(torch.tensor(O[t]).unsqueeze(0))
            state, _ = wm.rssm.observe_step(state, causal_a, emb)
        left = decode_state(wm, wm.rssm.imagine_step(state, onehot(0)))
        right = decode_state(wm, wm.rssm.imagine_step(state, onehot(1)))

    cur = O[warmup]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    draw_cartpole(axes[0], cur, "current state", cart_color="0.4",
                  pole_color="0.4",
                  subtitle=f"cart_vel={cur[1]:+.2f}")
    draw_cartpole(axes[1], left, "imagine: PUSH LEFT", cart_color="indianred",
                  pole_color="darkred",
                  subtitle=f"pred cart_vel={left[1]:+.2f}")
    draw_cartpole(axes[2], right, "imagine: PUSH RIGHT", cart_color="royalblue",
                  pole_color="navy",
                  subtitle=f"pred cart_vel={right[1]:+.2f}")
    sens = float(np.abs(left - right).mean())
    fig.suptitle(f"Action conditioning probe — same state, opposite actions   "
                 f"|Δprediction| = {sens:.3f}   (Δcart_vel = {right[1]-left[1]:+.3f})",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "action_probe.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)

    # aggregate sensitivity over many states
    divs = []
    with torch.no_grad():
        for _ in range(60):
            s = wm.rssm.initial_state(1, torch.device("cpu"))
            o, a = rollout_real(np.random.randint(3, 8), 2)
            w = min(len(a), 6)
            for t in range(w):
                ca = onehot(a[t - 1]) if (shift and t > 0) else \
                     (onehot(a[t]) if not shift else torch.zeros(1, ACTION_DIM))
                s, _ = wm.rssm.observe_step(
                    s, ca, wm.encoder(torch.tensor(o[t]).unsqueeze(0)))
            l = decode_state(wm, wm.rssm.imagine_step(s, onehot(0)))
            r = decode_state(wm, wm.rssm.imagine_step(s, onehot(1)))
            divs.append(float(np.abs(l - r).mean()))
    print(f"[probe] saved {path}")
    print(f"[probe] mean action sensitivity over 60 states: {np.mean(divs):.4f} "
          f"(higher = action has more control over predictions)")
    return path


# ----------------------------------------------------------------------------
# 6. LATENT MAP: PCA of internal states along a trajectory
# ----------------------------------------------------------------------------
def latent_map(wm):
    shift = getattr(wm, "action_shift", True)
    O, A = rollout_real(2, 120)
    state = wm.rssm.initial_state(1, torch.device("cpu"))
    latents, angles = [], []
    with torch.no_grad():
        for t in range(len(A)):
            causal_a = onehot(A[t - 1]) if (shift and t > 0) else \
                       (onehot(A[t]) if not shift else torch.zeros(1, ACTION_DIM))
            emb = wm.encoder(torch.tensor(O[t]).unsqueeze(0))
            state, _ = wm.rssm.observe_step(state, causal_a, emb)
            latents.append(wm.rssm.get_latent(state).squeeze(0).numpy())
            angles.append(math.degrees(O[t][2]))
    X = np.stack(latents)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=angles, cmap="coolwarm",
                    s=28, edgecolor="k", linewidth=0.3)
    ax.plot(pcs[:, 0], pcs[:, 1], color="0.6", lw=0.6, alpha=0.6, zorder=0)
    cb = plt.colorbar(sc); cb.set_label("pole angle (deg)")
    ax.set_title("Latent state map (PCA of h⊕z along a trajectory)",
                 fontsize=12)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "latent_map.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[latent] saved {path}")
    return path


def training_curve(curve):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(curve, lw=1, color="purple")
    ax.set_xlabel("gradient step"); ax.set_ylabel("total WM loss")
    ax.set_title("World model training loss"); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "training_curve.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


CACHE = os.path.join(OUTDIR, "wm_cache.pt")


def build_wm():
    return WorldModel(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        film_conditioning=True, inverse_dynamics=True,
        inverse_dynamics_scale=1.0, discrete_actions=True,
    )


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "2000"
    if arg == "viz":                       # reuse cached model, just re-render
        print(f"[cache] loading trained model from {CACHE}")
        wm = build_wm()
        wm.load_state_dict(torch.load(CACHE, map_location="cpu"))
        wm.eval()
        curve = []
    else:
        wm, curve = train_world_model(int(arg))
        torch.save(wm.state_dict(), CACHE)
        print(f"[cache] saved trained model -> {CACHE} "
              f"(re-render later with: python visualize_world_model.py viz)")
    print("\n--- generating visualizations ---")
    g = dream_vs_reality(wm)
    p = action_probe(wm)
    m = latent_map(wm)
    outs = [g, p, m]
    if curve:
        outs.append(training_curve(curve))
    print("\nDONE. Artifacts in:", OUTDIR)
    for f in outs:
        print("   ", f)
