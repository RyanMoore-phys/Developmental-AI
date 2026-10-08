"""ML producer telemetry: the numbers are right, and collecting them changes NOTHING.

WHAT IS CLAIMED (2026-10-07 telemetry build, Agent P). Four producers now
publish diagnostics for the loop to log:

  * StandaloneActorCritic.last_update_stats   (policy/actor_critic.py)
  * WorldModel.last_step_stats                (world_model/rssm.py)
  * LearningProgressCuriosity.pop_gate_stats() (curiosity/learning_progress.py)
  * ReplayBuffer / MultiStreamReplayBuffer.pop_sample_stats()
                                              (world_model/replay_buffer.py)

WHY A TEST. This project's history is a reward number that was wrong every
time (CLAUDE.md §9). A diagnostic that is itself wrong, or that quietly
perturbs training (an extra RNG draw, a stray gradient, a changed order of
ops), would be worse than none: it would move the run AND mis-describe it.
So the central contract is NON-INTERFERENCE, checked byte-for-byte against a
reference update run with collection disabled (the module flags
`COLLECT_STATS`), plus RNG state equality. The secondary contracts check the
numbers against independent recomputation, not against themselves.

CONTRACTS
  A. PPO stats: every key finite, clip_frac in [0,1], n_samples/epochs right.
  B. PPO grad_norm equals what clip_grad_norm_ returned (recorded by a
     wrapper) for the final epoch's minibatches; grad_norm_clipped counts
     the steps whose pre-clip norm exceeded 0.5.
  C. explained_variance: 1 for a perfect fit, 0 for predicting the mean,
     exactly 1 - Var(e)/Var(y) on a constructed residual, NaN for a constant
     target.
  D. PPO NON-INTERFERENCE: parameters, optimizer state and torch/numpy RNG
     after an update are byte-identical with COLLECT_STATS on vs off.
  E. WM stats: every loss head reported, finite; kl_raw >= 0; entropies in
     [0, log C]; grad_norm equals clip_grad_norm_'s return.
  F. WM NON-INTERFERENCE: parameters and RNG byte-identical on vs off.
  G. LP gate counters: paid <= passed_sig <= passed_abs <= evaluated;
     evaluated/paid match an independent count of the gate's own returns;
     collapsed_visits counts continuation steps; read-only calls are not
     counted; pop resets.
  H. Sampler counters: sequences_sampled counts what was drawn; boundary
     rejections are counted; mean_priority is the drawn windows' max priority;
     pop resets; the multi-stream pop aggregates; counting draws no RNG.

Run: PYTHONPATH=. ./venv/bin/python tests/_ml_stats_smoke.py
"""
import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn

import developmental_ai.policy.actor_critic as ac_mod
import developmental_ai.world_model.rssm as rssm_mod
from developmental_ai.policy.actor_critic import (StandaloneActorCritic,
                                                  explained_variance)
from developmental_ai.world_model.rssm import WorldModel
from developmental_ai.curiosity.learning_progress import \
    LearningProgressCuriosity
from developmental_ai.world_model.replay_buffer import (
    ReplayBuffer, MultiStreamReplayBuffer)

N = [0]


def ok(msg):
    N[0] += 1
    print(f"  {N[0]}. {msg}")


# ----------------------------------------------------------------- PPO ----
OD, AD, ROWS = 8, 4, 64


def _ppo(seed=0):
    torch.manual_seed(seed)
    p = StandaloneActorCritic(obs_dim=OD, action_dim=AD, hidden_dim=16,
                              minibatch_size=16)
    rng = np.random.default_rng(seed)
    for i in range(ROWS):
        p.store_transition(rng.standard_normal(OD).astype(np.float32),
                           int(rng.integers(AD)), float(rng.standard_normal()),
                           bool(i % 20 == 19), float(math.log(1.0 / AD)
                                                     + 0.3 * rng.standard_normal()),
                           float(rng.standard_normal()))
    return p


def _state_bytes(module_or_list):
    mods = module_or_list if isinstance(module_or_list, list) else [module_or_list]
    out = []
    for m in mods:
        for k, v in m.state_dict().items():
            out.append((k, v.detach().cpu().numpy().tobytes()))
    return out


def _opt_bytes(opt):
    out = []
    for st in opt.state_dict()["state"].values():
        for k, v in sorted(st.items()):
            out.append(v.cpu().numpy().tobytes() if torch.is_tensor(v) else repr(v))
    return out


def test_ppo():
    rec = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        r = real(params, max_norm, *a, **k)
        rec.append(float(r))
        return r

    p = _ppo()
    torch.nn.utils.clip_grad_norm_ = spy
    try:
        p.train_step(n_epochs=3)
    finally:
        torch.nn.utils.clip_grad_norm_ = real
    s = p.last_update_stats
    assert isinstance(s, dict), s
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
              "explained_variance", "grad_norm", "grad_norm_clipped", "lr",
              "adv_mean", "adv_std", "ret_mean", "n_samples", "epochs"):
        assert k in s and s[k] is not None, (k, s.get(k))
        assert math.isfinite(float(s[k])), (k, s[k])
    assert 0.0 <= s["clip_frac"] <= 1.0, s["clip_frac"]
    assert s["n_samples"] == ROWS and s["epochs"] == p.last_epochs_run, s
    assert s["explained_variance"] <= 1.0 + 1e-9
    ok(f"A PPO stats finite (clip_frac {s['clip_frac']:.3f}, EV "
       f"{s['explained_variance']:+.3f}, {s['minibatches']} final-epoch mb)")

    k = s["minibatches"]
    tail = rec[-2 * k:]
    ga, gc = np.array(tail[0::2]), np.array(tail[1::2])
    assert abs(s["grad_norm_actor"] - ga.mean()) < 1e-5 * max(1, ga.mean()), \
        (s["grad_norm_actor"], ga.mean())
    assert abs(s["grad_norm_critic"] - gc.mean()) < 1e-5 * max(1, gc.mean())
    gn = np.sqrt(ga ** 2 + gc ** 2).mean()
    assert abs(s["grad_norm"] - gn) < 1e-5 * max(1, gn), (s["grad_norm"], gn)
    assert s["grad_norm_clipped"] == int(((ga > 0.5) | (gc > 0.5)).sum())
    ok(f"B grad_norm {s['grad_norm']:.4f} == clip_grad_norm_ returns over the "
       f"final epoch; clipped {s['grad_norm_clipped']}/{k}")

    y = np.random.default_rng(3).standard_normal(500)
    assert abs(explained_variance(y, y) - 1.0) < 1e-12
    assert abs(explained_variance(np.full_like(y, y.mean()), y)) < 1e-12
    e = 0.5 * np.random.default_rng(4).standard_normal(500)
    want = 1.0 - np.var(e) / np.var(y)
    assert abs(explained_variance(y - e, y) - want) < 1e-12
    assert math.isnan(explained_variance(y, np.ones(500)))
    ok(f"C explained_variance exact on constructed cases ({want:+.4f})")

    # D. non-interference
    runs = {}
    for flag in (True, False):
        ac_mod.COLLECT_STATS = flag
        try:
            q = _ppo(seed=7)
            torch.manual_seed(123)
            np.random.seed(123)
            q.train_step(n_epochs=3)
            runs[flag] = (_state_bytes([q.actor, q.critic]),
                          _opt_bytes(q.optimizer),
                          torch.get_rng_state().numpy().tobytes(),
                          repr(np.random.get_state()[1][:8].tolist())
                          + str(np.random.get_state()[2]),
                          q.last_update_stats)
        finally:
            ac_mod.COLLECT_STATS = True
    assert runs[True][0] == runs[False][0], "PPO params differ with stats on"
    assert runs[True][1] == runs[False][1], "optimizer state differs"
    assert runs[True][2] == runs[False][2], "torch RNG state differs"
    assert runs[True][3] == runs[False][3], "numpy RNG state differs"
    assert runs[False][4] is None and isinstance(runs[True][4], dict)
    ok("D PPO params/optimizer/RNG byte-identical with COLLECT_STATS on vs off")


# ------------------------------------------------------------------ WM ----
B, L, WOD, WAD, S, C = 4, 6, 8, 4, 4, 4


def _wm(seed=0):
    torch.manual_seed(seed)
    return WorldModel(obs_dim=WOD, action_dim=WAD, stochastic_size=S,
                      stochastic_classes=C, deterministic_size=16,
                      hidden_dim=16)


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(B, L, WOD, generator=g)
    act = torch.nn.functional.one_hot(
        torch.randint(0, WAD, (B, L), generator=g), WAD).float()
    rew = torch.randn(B, L, generator=g)
    cont = torch.ones(B, L)
    return obs, act, rew, cont


def test_wm():
    rec = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        r = real(params, max_norm, *a, **k)
        rec.append(float(r))
        return r

    wm = _wm()
    torch.nn.utils.clip_grad_norm_ = spy
    try:
        m = wm.train_step(*_batch())
    finally:
        torch.nn.utils.clip_grad_norm_ = real
    s = wm.last_step_stats
    assert isinstance(s, dict), s
    for k in ("loss_total", "loss_recon", "loss_kl", "loss_reward",
              "loss_continue", "kl_raw", "grad_norm", "prior_entropy",
              "posterior_entropy"):
        assert s.get(k) is not None and math.isfinite(s[k]), (k, s.get(k))
    assert len([k for k in s if k.startswith("loss_")]) == len(m), (s, m)
    assert abs(s["loss_total"] - m["total"]) < 1e-12
    assert s["kl_raw"] >= -1e-6
    for k in ("prior_entropy", "posterior_entropy"):
        assert -1e-6 <= s[k] <= math.log(C) + 1e-5, (k, s[k])
    assert len(rec) == 1 and abs(s["grad_norm"] - rec[0]) < 1e-6 * max(1, rec[0])
    assert wm._stats_logits is None, "logit stash not cleared"
    ok(f"E WM stats finite ({len(m)} loss heads, kl_raw {s['kl_raw']:.4f}, "
       f"H_prior {s['prior_entropy']:.3f}, grad_norm {s['grad_norm']:.3f} "
       f"== clip_grad_norm_)")

    runs = {}
    for flag in (True, False):
        rssm_mod.COLLECT_STATS = flag
        try:
            w = _wm(seed=5)
            torch.manual_seed(99)
            w.train_step(*_batch(2))
            w.train_step(*_batch(3))
            runs[flag] = (_state_bytes(w), _opt_bytes(w.optimizer),
                          torch.get_rng_state().numpy().tobytes(),
                          w.last_step_stats)
        finally:
            rssm_mod.COLLECT_STATS = True
    assert runs[True][0] == runs[False][0], "WM params differ with stats on"
    assert runs[True][1] == runs[False][1], "WM optimizer state differs"
    assert runs[True][2] == runs[False][2], "torch RNG state differs"
    assert runs[False][3] is None and isinstance(runs[True][3], dict)
    ok("F WM params/optimizer/RNG byte-identical with COLLECT_STATS on vs off")


# ------------------------------------------------------------------ LP ----
LD, LA = 4, 3


class _Decaying(nn.Module):
    """Prediction off by C; the loop shrinks C (plus jitter) -> error falls
    on revisit, i.e. a learnable bucket that should be paid."""

    def __init__(self):
        super().__init__()
        self.C = 1.0

    def forward(self, f, a):
        return f + self.C


def test_lp():
    lp = LearningProgressCuriosity(obs_dim=LD, action_dim=LA, feature_dim=LD,
                                   hidden_dim=8, discrete_actions=True,
                                   lp_history=30, lp_min_samples=4,
                                   lp_abs_frac=0.05)
    lp.encoder = nn.Identity()
    lp.forward_model = _Decaying()
    seen = []
    real = lp._lp_from_history

    def spy(h):
        r = real(h)
        seen.append(r)
        return r

    lp._lp_from_history = spy
    states = [np.full(LD, float(v), np.float32) for v in (0.0, 3.0, 7.0)]
    act = torch.nn.functional.one_hot(torch.tensor([1]), LA).float()
    steps = 0
    jit = np.random.default_rng(2)
    for t in range(240):
        o = torch.from_numpy(states[(t // 2) % 3])[None]  # each state twice
        # error = C^2 steps down 1.0 -> 0.3 with iid jitter: real progress
        lp.forward_model.C = float(np.sqrt(max(
            1e-4, (1.0 if t < 150 else 0.3)
            + 0.03 * jit.standard_normal())))
        lp.compute_intrinsic_reward(o, act, o, stream_ids=[0])
        steps += 1
    g = lp.pop_gate_stats()
    assert g["paid"] <= g["passed_sig"] <= g["passed_abs"] <= g["evaluated"], g
    assert g["evaluated"] == len(seen), (g, len(seen))
    assert g["paid"] == sum(1 for r in seen if r > 0.0), g
    assert g["paid"] > 0, f"construction never paid: {g}"
    assert g["collapsed_visits"] == steps // 2, g
    assert g["buckets"] == 3, g
    paid = [r for r in seen if r > 0.0]
    # lp is stored in a float32 array before it is summed
    assert abs(g["mean_paid"] - float(np.mean(paid))) < 1e-6 * max(1.0, float(np.mean(paid)))
    # read-only calls are not counted, pop reset
    n0 = len(seen)
    lp.compute_intrinsic_reward(torch.from_numpy(states[0])[None], act,
                                torch.from_numpy(states[0])[None],
                                update_state=False)
    g2 = lp.pop_gate_stats()
    assert g2["evaluated"] == 0 and g2["paid"] == 0 \
        and g2["collapsed_visits"] == 0 and g2["mean_paid"] is None, g2
    assert len(seen) == n0 + 1  # it WAS scored, just not counted
    ok(f"G LP gates: evaluated {g['evaluated']} >= abs {g['passed_abs']} >= "
       f"sig {g['passed_sig']} >= paid {g['paid']}; collapsed "
       f"{g['collapsed_visits']}; read-only uncounted; pop resets")


# -------------------------------------------------------------- replay ----
def _rb(cap=1000, n=300, ep=50):
    b = ReplayBuffer(capacity=cap, obs_dim=4, action_dim=2)
    rng = np.random.default_rng(0)
    for i in range(n):
        b.add(rng.random(4).astype(np.float32), i % 2, 0.0, i % ep == ep - 1)
    return b


def test_replay():
    b = _rb()
    b.priorities[:] = np.random.default_rng(8).random(b.capacity)  # non-flat
    np.random.seed(11)
    b1 = b.sample_sequences(batch_size=8, seq_len=10)
    b2 = b.sample_sequences(batch_size=8, seq_len=10)
    pri = np.asarray(b.priorities[np.arange(b.capacity)], dtype=np.float64)
    want = []
    for bt in (b1, b2):
        for s0 in bt["start_indices"]:
            want.append(pri[(s0 + np.arange(10)) % b.capacity].max())
    st = b.pop_sample_stats()
    assert st["sequences_sampled"] == 16 and st["sample_calls"] == 2, st
    assert st["rejected_boundary"] > 0, st
    assert 0.0 < st["rejected_boundary_frac"] < 1.0, st
    assert abs(st["mean_priority"] - float(np.mean(want))) < 1e-6, st
    z = b.pop_sample_stats()
    assert z["sequences_sampled"] == 0 and z["mean_priority"] is None, z
    ok(f"H1 sampler: 16 sequences, {st['rejected_boundary']} boundary "
       f"rejections ({st['rejected_boundary_frac']:.3f}), mean_priority "
       f"{st['mean_priority']:.3f} matches; pop resets")

    # counting draws no RNG and changes no indices
    res = {}
    for count in (True, False):
        bb = _rb()
        if not count:
            bb._count_sample = lambda *a, **k: None
        np.random.seed(5)
        idx = bb.sample_sequences(batch_size=8, seq_len=10,
                                  prioritized=True)["start_indices"]
        res[count] = (idx.tobytes(), np.random.get_state()[1].tobytes(),
                      np.random.get_state()[2])
    assert res[True] == res[False], "sampler counting changed the draw"
    ok("H2 sampler counting draws no RNG (indices + numpy state identical)")

    ms = MultiStreamReplayBuffer(num_streams=2, capacity=200, obs_dim=4,
                                 action_dim=2)
    rs = np.random.RandomState(1)
    for i in range(240):
        ms.add(rs.rand(4).astype(np.float32), i % 2, 0.0, i % 40 == 39,
               stream=i % 2)
    ms.sample_sequences(batch_size=6, seq_len=8)
    ms.sample_sequences(batch_size=5, seq_len=8)
    agg = ms.pop_sample_stats()
    assert agg["sequences_sampled"] == 11, agg
    assert agg["sample_calls"] == 4, agg        # 2 streams x 2 batches
    assert agg["mean_priority"] is not None
    assert ms.pop_sample_stats()["sequences_sampled"] == 0
    assert all(s.pop_sample_stats()["sequences_sampled"] == 0
               for s in ms.streams), "multi pop did not reset streams"
    ok("H3 multi-stream pop aggregates 11 sequences over 2 streams, resets")


if __name__ == "__main__":
    print("[ml_stats_smoke]")
    test_ppo()
    test_wm()
    test_lp()
    test_replay()
    print("[ml_stats_smoke] ALL PASS")
    sys.exit(0)
