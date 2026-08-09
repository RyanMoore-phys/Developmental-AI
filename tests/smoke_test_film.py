"""Smoke test: FiLM conditioning + inverse dynamics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from developmental_ai.world_model.rssm import (
    WorldModel, FiLMActionConditioner, InverseDynamicsHead,
)

device = torch.device("cpu")
obs_dim, action_dim, batch, seq_len = 4, 2, 8, 10

wm = WorldModel(
    obs_dim=obs_dim, action_dim=action_dim,
    film_conditioning=True, inverse_dynamics=True,
    inverse_dynamics_scale=1.0, discrete_actions=True,
)
print("[1] WorldModel created (film=True, inverse=True)")

assert isinstance(wm.rssm.film, FiLMActionConditioner)
print("[2] FiLM module present")

assert isinstance(wm.inverse_head, InverseDynamicsHead)
print("[3] Inverse dynamics head present")

prev_state = wm.rssm.initial_state(batch, device)
action = torch.nn.functional.one_hot(
    torch.randint(0, action_dim, (batch,)), action_dim
).float()
obs_embed = wm.encoder(torch.randn(batch, obs_dim))
new_state, model_info = wm.rssm.observe_step(prev_state, action, obs_embed)
h_shape = new_state["h"].shape
z_shape = new_state["z"].shape
print("[4] observe_step OK - h:", h_shape, "z:", z_shape)

imag_state = wm.rssm.imagine_step(new_state, action)
print("[5] imagine_step OK - h:", imag_state["h"].shape)

obs_seq = torch.randn(batch, seq_len, obs_dim)
act_seq = torch.nn.functional.one_hot(
    torch.randint(0, action_dim, (batch, seq_len)), action_dim
).float()
rew_seq = torch.randn(batch, seq_len)
cont_seq = torch.ones(batch, seq_len)
losses = wm.compute_loss(obs_seq, act_seq, rew_seq, cont_seq)
total_loss = losses["total"]
print("[6] compute_loss OK - total:", round(total_loss.item(), 4))
for k, v in losses.items():
    val = round(v.item(), 4) if hasattr(v, "item") else v
    print("   ", k, ":", val)

assert losses["inverse"].item() > 0
print("[7] Inverse loss nonzero:", round(losses["inverse"].item(), 4))

total_loss.backward()

# Check all FiLM parameter gradients
film_has_grad = False
for name, p in wm.rssm.film.named_parameters():
    g = p.grad
    if g is not None:
        gn = g.norm().item()
        print("    FiLM", name, "grad norm:", round(gn, 8))
        if gn > 0:
            film_has_grad = True
assert film_has_grad, "No FiLM parameter received nonzero gradients!"
print("[8] FiLM gradients flowing")

ig = wm.inverse_head.net[0].weight.grad
assert ig is not None and ig.abs().sum() > 0
print("    Inverse grad norm:", round(ig.norm().item(), 6))

wm.eval()
with torch.no_grad():
    state = wm.rssm.initial_state(1, device)
    a1 = torch.zeros(1, action_dim)
    a1[0, 0] = 1.0
    a2 = torch.zeros(1, action_dim)
    a2[0, 1] = 1.0
    obs_e = wm.encoder(torch.randn(1, obs_dim))
    s1, _ = wm.rssm.observe_step(state, a1, obs_e)
    s2, _ = wm.rssm.observe_step(state, a2, obs_e)
    z_diff = (s1["z"] - s2["z"]).abs().mean().item()
    h_diff = (s1["h"] - s2["h"]).abs().mean().item()
print("[9] Action sensitivity: z_diff=", round(z_diff, 4), "h_diff=", round(h_diff, 4))

wm.train()
opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
losses = []
for i in range(3):
    opt.zero_grad()
    ls = wm.compute_loss(obs_seq, act_seq, rew_seq, cont_seq)
    l = ls["total"]
    l.backward()
    opt.step()
    losses.append(l.item())
print("[10] 3 grad steps:", round(losses[0], 3), "->", round(losses[-1], 3))

print()
print("=" * 50)
print("ALL SMOKE TESTS PASSED")
print("=" * 50)
