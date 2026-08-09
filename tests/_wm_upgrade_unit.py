"""Unit tests for the WM-accuracy upgrades: twohot reward head + goal replay."""
import numpy as np
import torch
from developmental_ai.world_model.rssm import (
    symlog, symexp, twohot_encode, twohot_decode, TWOHOT_BINS, TWOHOT_LOW, TWOHOT_HIGH)
from developmental_ai.world_model.replay_buffer import ReplayBuffer

bins = torch.linspace(TWOHOT_LOW, TWOHOT_HIGH, TWOHOT_BINS)

# 1) twohot encode expectation == x  (and decode(log p) == x)
x = torch.tensor([0.0, 0.69, -0.69, 2.5, -2.5, 4.99, symlog(torch.tensor(0.95)).item()])
enc = twohot_encode(x, bins)                       # (N, B) probs
exp_from_enc = (enc * bins).sum(-1)
dec = twohot_decode((enc.clamp_min(1e-12)).log(), bins)
err_enc = (exp_from_enc - x).abs().max().item()
err_dec = (dec - x).abs().max().item()
mass = enc.sum(-1)                                 # should all be ~1
print(f"[twohot] max|E[enc]-x|={err_enc:.2e}  max|decode-x|={err_dec:.2e}  "
      f"mass∈[{mass.min():.4f},{mass.max():.4f}]")
# real-scale round trip for a goal reward 0.95
r = 0.95
sl = symlog(torch.tensor([r]))
back = symexp(twohot_decode(twohot_encode(sl, bins).clamp_min(1e-12).log(), bins))
print(f"[twohot] reward 0.95 round-trip -> {back.item():.4f}")
twohot_ok = err_enc < 1e-4 and err_dec < 1e-4 and abs(back.item() - r) < 1e-2

# 2) goal-prioritized replay: build a buffer where only ~20% of episodes reach a
#    goal (reward 0.9 at the last step), rest time out (reward 0). Check that
#    reward_fraction lifts the share of sampled windows that contain a reward.
rb = ReplayBuffer(capacity=20000, obs_dim=4, action_dim=3)
rng = np.random.default_rng(0)
ep_len = 30
for ep in range(200):
    solved = (ep % 5 == 0)            # 20% solve
    for t in range(ep_len):
        last = (t == ep_len - 1)
        rew = 0.9 if (last and solved) else 0.0
        rb.add(rng.normal(size=4).astype(np.float32),
               int(rng.integers(0, 3)), rew, done=last)

def realized_reward_share(reward_fraction, n=64, seq_len=20, trials=20):
    shares = []
    for _ in range(trials):
        b = rb.sample_sequences(n, seq_len, reward_fraction=reward_fraction)
        has = (b["rewards"].abs() > 1e-3).any(dim=1).float().mean().item()
        shares.append(has)
    return float(np.mean(shares))

base = realized_reward_share(0.0)
goal = realized_reward_share(0.5)
print(f"[goal-replay] reward-bearing window share: off={base:.3f}  frac=0.5 -> {goal:.3f}")
# off includes the existing 25% terminal oversampling; goal should be clearly higher
goal_ok = goal > base + 0.15 and goal >= 0.5

# 3) shapes: a fresh WorldModel reward head returns logits over bins; predict_reward scalar
from developmental_ai.world_model.rssm import WorldModel
wm = WorldModel(obs_dim=4, action_dim=3)
lat = torch.randn(8, wm.rssm.latent_dim)
logits = wm.reward_predictor(lat)
pr = wm.predict_reward(lat)
print(f"[shapes] reward logits {tuple(logits.shape)} (want (8,{TWOHOT_BINS}))  "
      f"predict_reward {tuple(pr.shape)} (want (8,))")
shape_ok = tuple(logits.shape) == (8, TWOHOT_BINS) and tuple(pr.shape) == (8,)

print("\nUNIT OK" if (twohot_ok and goal_ok and shape_ok) else
      f"\nUNIT PROBLEM (twohot={twohot_ok} goal={goal_ok} shape={shape_ok})")
