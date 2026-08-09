"""Smoke tests for the H1 action-causality fix package (CODE_AUDIT_2026-07.md §H.1).

Verifies, with hand-built data:
  1. Terminal replay windows end EXACTLY on a done (no next-episode bleed).
  2. Reward windows never cross an episode boundary.
  3. No sampled window straddles the circular write seam after wrap.
  4. PER importance weights reflect the true mixture probabilities
     (every sampled start has a mixture entry; no uniform fallback).
  5. StraightThroughCategorical applies exactly ONE straight-through path
     (gradient magnitude ~halved vs the old double-ST).
  6. Inverse-dynamics target under action_shift=True is a_t (the causal
     action), not a_{t+1} (the policy leak).
"""
import numpy as np
import torch

from developmental_ai.world_model.replay_buffer import ReplayBuffer
from developmental_ai.world_model.rssm import (
    StraightThroughCategorical, WorldModel)

SEQ = 8


def _fill(buf, n, episode_len=10, obs_dim=6, act_dim=4, reward_on_done=True):
    """Write n transitions of synthetic episodes into the buffer."""
    for i in range(n):
        step = i % episode_len
        done = step == episode_len - 1
        obs = np.full(obs_dim, float(i), dtype=np.float32)
        reward = 1.0 if (done and reward_on_done) else 0.0
        buf.add(obs, int(i % act_dim), reward, done)


def test_windows_respect_boundaries_and_seam():
    buf = ReplayBuffer(capacity=200, obs_dim=6, action_dim=4)
    _fill(buf, 200 + 57)  # force wrap: write head mid-buffer
    normal, terminal = buf._find_valid_starts(SEQ)
    reward = buf._find_reward_starts(SEQ)
    pos = buf.position

    def window_ok(s, require_done_last=False, allow_done_last=False):
        idx = (s + np.arange(SEQ)) % buf.capacity
        dones = buf.dones[idx] > 0.5
        # seam: write position may only appear at offset 0
        off = (pos - s) % buf.capacity
        assert not (0 < off < SEQ), f"start {s} straddles write seam at {pos}"
        if require_done_last:
            assert dones[-1] and not dones[:-1].any(), (
                f"terminal window {s} has done pattern {dones}")
        elif allow_done_last:
            assert not dones[:-1].any(), (
                f"window {s} crosses an episode boundary: {dones}")
        else:
            assert not dones.any(), f"normal window {s} contains a done"

    for s in normal:
        window_ok(s)
    for s in terminal:
        window_ok(s, require_done_last=True)
    for s in reward:
        window_ok(s, allow_done_last=True)
    assert len(terminal) > 0 and len(reward) > 0 and len(normal) > 0
    print(f"  windows ok: {len(normal)} normal, {len(terminal)} terminal, "
          f"{len(reward)} reward — all boundary- and seam-clean")


def test_per_mixture_weights():
    buf = ReplayBuffer(capacity=300, obs_dim=6, action_dim=4)
    _fill(buf, 290)
    # Give the buffer non-trivial priorities so strata probs are non-uniform.
    buf.priorities[: buf.size] = np.random.RandomState(0).rand(buf.size) + 0.1
    batch = buf.sample_sequences(
        batch_size=32, seq_len=SEQ, device=torch.device("cpu"),
        terminal_fraction=0.25, reward_fraction=0.25, prioritized=True)
    w = batch["weights"].numpy()
    assert w.shape == (32,) and np.isfinite(w).all() and w.max() <= 1.0 + 1e-6
    # Weights must differentiate strata (old code gave oversampled strata a
    # uniform-fallback probability -> after max-normalization the bug showed
    # as many weights pinned to the same fallback value).
    assert len(np.unique(np.round(w, 6))) > 3, f"degenerate IS weights: {w}"
    print(f"  PER mixture weights ok: min {w.min():.3f} max {w.max():.3f} "
          f"unique {len(np.unique(np.round(w, 6)))}")


def test_single_straight_through():
    torch.manual_seed(0)
    st = StraightThroughCategorical(4, 8).train()
    logits = torch.randn(64, 32, requires_grad=True)
    torch.manual_seed(1234)
    out = st(logits)
    assert out.shape == (64, 32)
    # forward is exactly one-hot per categorical
    hard = out.view(64, 4, 8)
    assert torch.allclose(hard.sum(-1), torch.ones(64, 4))
    proj = torch.randn(32)  # non-degenerate loss (sum of one-hots is constant)
    (out * proj).sum().backward()
    g_new = logits.grad.abs().mean().item()
    # Reference: the OLD double-ST path on the same logits (same gumbel draw).
    logits2 = logits.detach().clone().requires_grad_(True)
    probs = torch.softmax(logits2.view(64, 4, 8), -1).view(64, -1)
    torch.manual_seed(1234)
    samples = torch.nn.functional.gumbel_softmax(
        logits2.view(64, 4, 8), tau=1.0, hard=True, dim=-1).view(64, -1)
    ((samples + probs - probs.detach()) * proj).sum().backward()
    g_old = logits2.grad.abs().mean().item()
    assert g_new > 1e-4, "gradient did not flow through the ST path"
    assert g_new < g_old * 0.75, (
        f"expected single-ST gradient well below double-ST ({g_new} vs {g_old})")
    print(f"  single ST ok: grad {g_new:.4f} vs double-ST {g_old:.4f}")


def test_inverse_target_alignment():
    torch.manual_seed(0)
    wm = WorldModel(obs_dim=6, action_dim=4, stochastic_size=4,
                    stochastic_classes=4, deterministic_size=16,
                    hidden_dim=16, inverse_dynamics=True,
                    discrete_actions=True)
    B, L = 2, 5
    obs = torch.randn(B, L, 6)
    actions = torch.eye(4)[torch.randint(0, 4, (B, L))]

    captured = {}
    orig_ce = torch.nn.functional.cross_entropy

    def spy_ce(pred, target, *a, **k):
        captured["target"] = target.detach().clone()
        return orig_ce(pred, target, *a, **k)

    torch.nn.functional.cross_entropy = spy_ce
    try:
        for shift, want_slice in ((True, actions[:, :-1]), (False, actions[:, 1:])):
            wm.action_shift = shift
            wm.compute_loss(obs, actions, torch.zeros(B, L), torch.ones(B, L))
            want = want_slice.argmax(-1).reshape(-1)
            got = captured["target"]
            assert torch.equal(got, want), (
                f"action_shift={shift}: inverse target {got.tolist()} != "
                f"expected {want.tolist()}")
    finally:
        torch.nn.functional.cross_entropy = orig_ce
    print("  inverse target ok: a_t under causal_align, a_t+1 otherwise")


if __name__ == "__main__":
    for fn in (test_windows_respect_boundaries_and_seam,
               test_per_mixture_weights,
               test_single_straight_through,
               test_inverse_target_alignment):
        print(f"[h1-smoke] {fn.__name__}")
        fn()
    print("[h1-smoke] ALL PASS")
