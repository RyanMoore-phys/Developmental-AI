"""Unit cases for developmental_ai.foundation.mechanisms (plan Stage 6).

Parts only: the MixedOutcome algebra (log-density, mixture moments, CDF,
sampling, entropy, records), state/transition validation, the record ->
transition stream rule, the proper scores, the constant-velocity floor,
rollout shapes, Prediction records, the fixture's physics, and shadow
safety (fitting does not advance the global torch RNG). The design claims
(held-out scores, symmetry falsification, reset exclusion on a full stream,
law diagnostics, drift) live in tests/_foundation_mechanisms_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_foundation_mechanisms_unit.py
"""
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*requires_grad.*")

import numpy as np
import torch
from scipy.stats import norm

from tests.unit._runner import case, close, run_all, raises
from developmental_ai.foundation import mechanisms as M
from developmental_ai.foundation.contracts import (ABSENT, UNKNOWN, Action, ContractError,
                                                   Observation)
from developmental_ai.foundation.mechanisms.api import _regroup

LAY = M.OutcomeLayout(("a", "b"), ("e",), ("none", "hit", "other"))


def _single(B=4, seed=0):
    r = np.random.default_rng(seed)
    p = r.dirichlet(np.ones(3), (B, 1))
    return M.MixedOutcome.gaussian_categorical(LAY, r.normal(size=(B, 2)),
                                               r.uniform(0.5, 2, (B, 2)), p)


def _batch(B=6, N=3, seed=0):
    r = np.random.default_rng(seed)
    s = M.EntityBatch(r.normal(size=(B, N, 2)), r.normal(size=(B, N, 2)),
                      np.ones((B, N, 5)))
    s2 = M.EntityBatch(s.pos + 0.1, s.vel, s.attrs)
    return M.TransitionBatch(s, np.zeros((B, N, 2)), np.full(B, 0.05), s2,
                             np.zeros((B, N), int))


# ---- layout / distribution ----------------------------------------------------
@case
def layout_validates():
    raises(lambda: M.OutcomeLayout(("a", "a")), ContractError, "dup names")
    raises(lambda: M.OutcomeLayout(("a",), ("e",), ("only",)), ContractError, "1 class")
    raises(lambda: M.OutcomeLayout(()), ContractError, "empty")
    lay = M.entity_layout(2, 2, ("n", "h"))
    assert lay.Dc == 8 and lay.G == 2 and lay.cont_names[4] == "e1.pos0"


@case
def single_logprob_matches_scipy():
    d = _single()
    y = np.random.default_rng(1).normal(size=(4, 2))
    k = np.array([[0], [1], [2], [M.MISSING]])
    lp = d.log_prob({"continuous": y, "discrete": k})
    ref = norm.logpdf(y, d.means[0], np.sqrt(d.vars[0])).sum(1)
    pk = d.probs[0, np.arange(4), 0, np.maximum(k[:, 0], 0)]
    ref = ref + np.where(k[:, 0] >= 0, np.log(pk), 0.0)
    assert np.allclose(lp, ref), (lp, ref)
    parts = d.log_prob_parts({"continuous": y})
    assert set(parts) == {"continuous"}
    assert np.array_equal(d.log_prob({}), np.zeros(4)), "no evidence must be zero"
    assert np.allclose(d.log_prob({"discrete": k, "continuous": ABSENT}),
                       ref - norm.logpdf(y, d.means[0], np.sqrt(d.vars[0])).sum(1))


@case
def mixture_moments_and_logprob():
    a, b = _single(seed=0), _single(seed=1)
    m = M.MixedOutcome.mixture([a, b], [0.3, 0.7])
    assert m.n_components == 2
    mu = 0.3 * a.mean() + 0.7 * b.mean()
    assert np.allclose(m.mean(), mu)
    var = 0.3 * (a.variance() + a.mean() ** 2) + 0.7 * (b.variance() + b.mean() ** 2) - mu ** 2
    assert np.allclose(m.variance(), var)
    w, e = m.decompose_variance()
    assert np.allclose(w + e, m.variance()) and np.all(e >= 0)
    y = {"continuous": np.zeros((4, 2)), "discrete": np.zeros((4, 1), int)}
    ref = np.logaddexp(np.log(0.3) + a.log_prob(y), np.log(0.7) + b.log_prob(y))
    assert np.allclose(m.log_prob(y), ref)
    assert np.allclose(m.probs_marginal(), 0.3 * a.probs_marginal() + 0.7 * b.probs_marginal())
    assert np.allclose(m.cdf(m.means[0]) * 0 + a.cdf(a.mean()), 0.5)


@case
def sampling_matches_moments():
    m = M.MixedOutcome.mixture([_single(seed=0), _single(seed=1)])
    s = m.sample(40000, np.random.default_rng(0))
    assert s["continuous"].shape == (40000, 4, 2) and s["discrete"].shape == (40000, 4, 1)
    assert np.allclose(s["continuous"].mean(0), m.mean(), atol=0.05)
    assert np.allclose(s["continuous"].var(0), m.variance(), rtol=0.06)
    freq = np.stack([(s["discrete"][..., 0] == k).mean(0) for k in range(3)], -1)
    assert np.allclose(freq, m.probs_marginal()[:, 0], atol=0.02)
    raises(lambda: m.sample(3, None), ContractError, "global RNG")


@case
def entropy_exact_or_unknown():
    d = _single()
    h = d.entropy()
    ref = norm.entropy(d.means[0], np.sqrt(d.vars[0])).sum(1)
    assert np.allclose(h["continuous"], ref)
    m = M.MixedOutcome.mixture([_single(seed=0), _single(seed=1)])
    assert m.entropy()["continuous"] is UNKNOWN
    p = m.probs_marginal()[:, 0]
    assert np.allclose(m.entropy()["discrete"], -(p * np.log(p)).sum(-1))


@case
def outcome_validation():
    raises(lambda: M.MixedOutcome.gaussian_categorical(LAY, np.zeros((2, 2)), np.zeros((2, 2)),
                                                       np.full((2, 1, 3), 1 / 3)),
           ContractError, "zero variance")
    raises(lambda: M.MixedOutcome.gaussian_categorical(LAY, np.zeros((2, 2)), np.ones((2, 2)),
                                                       np.full((2, 1, 3), 0.5)),
           ContractError, "probs not normalised")
    raises(lambda: M.MixedOutcome.gaussian_categorical(LAY, np.zeros((2, 3)), np.ones((2, 3))),
           ContractError, "wrong Dc")
    raises(lambda: _single().log_prob({"discrete": np.full((4, 1), 3)}), ContractError,
           "class out of range")
    raises(lambda: _single().log_prob({"discrete": np.zeros((4, 1))}), ContractError,
           "float classes")


@case
def record_roundtrip_select_index():
    m = M.MixedOutcome.mixture([_single(seed=0), _single(seed=1)])
    back = M.MixedOutcome.from_record(m.to_record())
    assert np.allclose(back.mean(), m.mean()) and back.layout == m.layout
    raises(lambda: M.MixedOutcome.from_record({"kind": "other"}), ContractError, "kind")
    s = m.select([1], [])
    assert s.layout.cont_names == ("b",) and s.layout.G == 0
    assert np.allclose(s.mean()[:, 0], m.mean()[:, 1])
    assert m.index([2]).batch_size == 1


# ---- state / transitions -------------------------------------------------------
@case
def entity_and_transition_validation():
    b = _batch()
    assert b.state.continuous().shape == (6, 12)
    assert np.allclose(b.state.with_continuous(b.state.continuous()).pos, b.state.pos)
    assert np.allclose(b.state.translate([1, 2]).pos - b.state.pos, [1, 2])
    raises(lambda: M.EntityBatch(np.full((1, 2, 2), np.nan), np.zeros((1, 2, 2)),
                                 np.zeros((1, 2, 1))), ContractError, "NaN state")
    raises(lambda: M.TransitionBatch(b.state, b.action, np.zeros(6), b.next_state),
           ContractError, "dt = 0")
    other = M.EntityBatch(b.state.pos, b.state.vel, b.state.attrs, frame="other")
    raises(lambda: M.TransitionBatch(b.state, b.action, b.dt, other), ContractError,
           "frame mismatch")
    assert np.all(M.TransitionBatch(b.state, b.action, b.dt, b.next_state).events == M.MISSING)
    assert len(M.TransitionBatch.concat([b, b])) == 12 and len(b.subset([0, 2])) == 2


def _obs(ep, seq, payload, prov="sensor"):
    return Observation("env", "s", ep, seq, float(seq), float(seq), "entities", prov, payload)


def _act(ep, seq, dur=0.05):
    return Action("env", "s", ep, seq, "f", np.zeros((2, 2)), dur, float(seq))


@case
def stream_refuses_bad_pairs():
    st = {"pos": np.zeros((2, 2)), "vel": np.zeros((2, 2)), "attrs": np.zeros((2, 1)),
          "events": np.zeros(2, int)}
    obs = [_obs("e1", i, dict(st)) for i in range(4)] + [_obs("e2", i, dict(st)) for i in range(3)]
    obs[2] = _obs("e1", 2, dict(st, pos=ABSENT))             # dropped frame
    acts = [_act("e1", 0), _act("e1", 1), _act("e1", 2, UNKNOWN), _act("e2", 0)]
    s = M.EntityStream(obs, acts)
    # pairs: e1 0-1 ok, 1-2 absent, 2-3 absent+unknown dur, e1->e2 splice, e2 0-1 ok,
    # e2 1-2 no action
    assert s.pair_valid.tolist() == [True, False, False, False, True, False], s.pair_valid
    assert any("ABSENT" in p for p in s.pair_problems[1])
    assert any("UNKNOWN" in p for p in s.pair_problems[2])
    assert any("reset splice" in p for p in s.pair_problems[3])
    assert any("no action" in p for p in s.pair_problems[5])
    assert s.windows(1).tolist() == [0, 4] and len(s.windows(2)) == 0
    assert s.refused_windows(2) == 5
    tb = s.transitions()
    assert len(tb) == 2 and np.allclose(tb.dt, 0.05)
    raises(lambda: s.transitions([1]), ContractError, "refused transition served")


@case
def stream_absent_events_are_missing_not_zero():
    st = {"pos": np.zeros((1, 2)), "vel": np.zeros((1, 2)), "attrs": np.zeros((1, 1))}
    obs = [_obs("e", 0, dict(st, events=ABSENT)), _obs("e", 1, dict(st, events=ABSENT))]
    acts = [Action("env", "s", "e", 0, "f", np.zeros((1, 2)), 0.1, 0.0)]
    tb = M.EntityStream(obs, acts).transitions()
    assert tb.events.tolist() == [[M.MISSING]]


# ---- scores ----------------------------------------------------------------------
@case
def crps_closed_form_matches_samples():
    r = np.random.default_rng(0)
    mu, var, y = np.zeros((50, 1)), np.full((50, 1), 2.0), r.normal(size=(50, 1))
    s = r.normal(0, np.sqrt(2.0), (6000, 50, 1))
    a = M.gaussian_crps(mu, var, y).mean()
    b = M.crps_samples(s, y).mean()
    close(a, b, 0.03, "CRPS closed form vs samples")
    close(M.energy_score(s[:1500], y).mean(), M.crps_samples(s[:1500], y).mean(), 1e-9,
          "1-D energy score == CRPS")
    cov = M.coverage(r.random(20000))
    close(cov["cov50"], 0.5, 0.02, "uniform PIT cov50")
    close(cov["cov90"], 0.9, 0.02, "uniform PIT cov90")
    assert np.isclose(M.pit_histogram(r.random(1000)).sum(), 1.0)


# ---- mechanisms --------------------------------------------------------------------
ARGS = (3, 2, 5, 2, ("none", "wall", "ball"))


@case
def constant_velocity_fit_and_applicability():
    b = _batch()
    cv = M.ConstantVelocityMechanism(*ARGS)
    assert cv.applicability(b.state) is UNKNOWN and cv.version == 0
    cv.fit(b)
    d = cv.predict(b.state, b.action, b.dt)
    exp = np.concatenate([b.state.pos + b.state.vel * 0.05, b.state.vel], -1).reshape(6, -1)
    assert np.allclose(d.mean(), exp) and cv.version == 1
    assert np.allclose(cv.applicability(b.state), 1.0)
    far = M.EntityBatch(b.state.pos, b.state.vel * 100, b.state.attrs)
    assert np.all(cv.applicability(far) < 1.0)
    raises(lambda: cv.predict(b.state, b.action[:, :2], b.dt), ContractError, "wrong action")


@case
def rollout_modes_and_regroup():
    b = _batch()
    cv = M.ConstantVelocityMechanism(*ARGS)
    cv.fit(b)
    acts, dts = np.zeros((4, 6, 3, 2)), np.full((4, 6), 0.05)
    rm = cv.rollout(b.state, acts, dts)
    assert len(rm) == 4 and rm[0].batch_size == 6 and rm[0].n_components == 1
    rs = cv.rollout(b.state, acts, dts, mode="sample", n_samples=5,
                    rng=np.random.default_rng(0))
    assert rs[0].n_components == 5 and rs[0].batch_size == 6
    assert np.allclose(rs[0].mean(), rm[0].mean())        # identical particles at h=1
    assert np.all(rs[3].variance() > rm[3].variance())    # mean-mode under-dispersed
    raises(lambda: cv.rollout(b.state, acts, dts, mode="sample"), ContractError, "no rng")
    raises(lambda: cv.rollout(b.state, acts, dts, mode="other"), ContractError, "mode")
    d = cv.predict(b.state.repeat(2), np.zeros((12, 3, 2)), np.full(12, 0.05))
    g = _regroup(d, 6, 2)
    assert g.n_components == 2 and np.allclose(g.means[1], d.means[0][1::2])


@case
def prediction_record_rules():
    b = _batch()
    cv = M.ConstantVelocityMechanism(*ARGS)
    d = cv.predict(b.state.index([0]), b.action[:1], b.dt[:1])
    raises(lambda: M.prediction_record(cv, d, b.action[:1], b.dt[:1], prediction_id="p",
                                       source_belief="b", snapshot="s", action_spec_id="a"),
           ContractError, "version 0 recorded")
    cv.fit(b)
    d = cv.predict(b.state.index([0]), b.action[:1], b.dt[:1])
    raises(lambda: M.prediction_record(cv, d, b.action[:1], b.dt[:1], prediction_id="p",
                                       source_belief="b", snapshot="", action_spec_id="a"),
           ContractError, "empty snapshot id")
    p = M.prediction_record(cv, d, b.action[:1], b.dt[:1], prediction_id="p",
                            source_belief="b", snapshot="snap-1", action_spec_id="a")
    assert p.horizon == 1 and p.payload["provenance"] == "imagined"
    assert np.allclose(M.outcomes_from_prediction(p)[0].mean(), d.mean())


@case
def fixture_physics():
    w = M.BoxWorld(n_balls=2, force_scale=0.0)
    tr = w.simulate(0, 60)
    nb = w.n_balls
    assert np.allclose(tr["pos"][:, nb:], tr["pos"][0, nb:]), "walls moved"
    r = w.radius
    assert np.all(tr["pos"][:, :nb] >= r - 1e-9) and np.all(tr["pos"][:, :nb] <= w.size - r + 1e-9)
    m = tr["attrs"][:nb, 1]
    E = (0.5 * m * (tr["vel"][:, :nb] ** 2).sum(-1) + m * w.gravity * tr["pos"][:, :nb, 1]).sum(1)
    assert np.abs(E - E[0]).max() < 0.05 * abs(E[0]) + 0.5, np.abs(E - E[0]).max()
    assert set(np.unique(tr["events"])) <= {0, 1, 2} and np.all(tr["events"][:, nb:] == 0)
    obs, acts = M.trajectory_records(tr, episode="e0")
    assert len(obs) == 61 and len(acts) == 60 and obs[0].payload["events"] is ABSENT


@case
def learned_models_shapes_and_shadow_safety():
    import torch
    b = _batch(B=32)
    rng_before = torch.get_rng_state().clone()
    for m in (M.InteractionNetwork(*ARGS, hidden=16), M.MLPMechanism(*ARGS, hidden=16),
              M.InteractionNetwork(*ARGS, hidden=16, translation_invariant=False,
                                   mechanism_id="in_nti")):
        d0 = m.predict(b.state, b.action, b.dt)               # untrained: valid, wide
        assert d0.means.shape == (1, 32, 12) and d0.probs.shape == (1, 32, 3, 3)
        rep = m.fit(b, epochs=2, batch_size=16)
        assert m.version == 1 and rep["n"] == 32 and rep["train_seconds"] >= 0
        assert m.resources(b.state, b.action, b.dt)["flops_per_sample"] > 0
        assert set(m.state_dict()) and m.applicability(b.state).shape == (32,)
    assert torch.equal(torch.get_rng_state(), rng_before), "fit advanced the global RNG"
    assert M.InteractionNetwork(*ARGS).assumptions["translation_invariant"] is True


# ---- static dims + resumable training (2026-10-03 regression) -----------------
# Independent A/B verifier, experiments/foundation_ab/RESULTS_stage6_8.md:
# bug 1 — never-moving outcome dims (walls) sat at the variance floor and
# dominated the training NLL; bug 2 — every fit() built a fresh Adam, so
# chunked/continual training reset its moments (MLP loss 9.3 -> 310 at toy
# scale). Each case below fails on the pre-fix _nets.py.
from developmental_ai.foundation.mechanisms import _nets  # noqa: E402


@case
def static_mask_rules():
    ch = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0]])
    no = np.zeros(3, bool)
    assert _nets.update_static_mask(ch, None, no).tolist() == [True, False, True]
    ch2 = np.array([[0.0, 0.0, 0.5]])                   # dim 2 moves now
    assert _nets.update_static_mask(ch2, np.array([True, False, True]), no).tolist() \
        == [True, False, False], "mask must only shrink (escape path)"
    raises(lambda: _nets.update_static_mask(ch, None, np.array([False, True, False])),
           ContractError, "declared static dim that moves")
    mu, var = torch.zeros(4, 3), torch.ones(4, 3)
    y = torch.zeros(4, 3)
    y2 = y.clone()
    y2[:, 0] = 1e6                                         # junk on a static dim
    st = torch.tensor([True, False, False])
    close(float(_nets.head_loss(mu, var, None, y, None, st)),
          float(_nets.head_loss(mu, var, None, y2, None, st)), 1e-12, "masked dim ignored")


@case
def static_dims_pass_through_bit_exact():
    b = _batch(B=32)                                       # vel never changes in _batch
    m = M.MLPMechanism(*ARGS, hidden=16, seed=0)
    rep = m.fit(b, epochs=2, batch_size=16)
    vel = [i for i, n in enumerate(m.layout.cont_names) if ".vel" in n]
    assert rep["static_dims"] == len(vel) and m.static_dims.tolist() == vel
    d = m.predict(b.state, b.action, b.dt)
    assert np.array_equal(d.mean()[:, vel], b.state.continuous()[:, vel])
    assert np.allclose(d.vars[0][:, vel], _nets.VAR_FLOOR)


@case
def optimizer_persists_across_fits():
    b = _batch(B=32)
    m = M.MLPMechanism(*ARGS, hidden=16, seed=0)
    m.fit(b, epochs=2, batch_size=16, lr=1e-3)              # 4 updates
    opt = m._opt
    m.fit(b, epochs=3, batch_size=16, lr=5e-4)              # 6 more, SAME optimizer
    assert m._opt is opt
    p0 = m._trainable_params()[0]
    assert int(opt.state[p0]["step"]) == 10, int(opt.state[p0]["step"])
    assert opt.param_groups[0]["lr"] == 5e-4
    m.reset_optimizer()
    m.fit(b, epochs=1, batch_size=16)
    assert int(m._opt.state[p0]["step"]) == 2


def _weights(m):
    return [p.detach().clone() for p in m._trainable_params()]


@case
def training_state_round_trip_resumes_bit_exact():
    b = _batch(B=32)
    mk = {"mlp": lambda: M.MLPMechanism(*ARGS, hidden=16, seed=0),
          "in": lambda: M.InteractionNetwork(*ARGS, hidden=16, seed=0)}
    for name, f in mk.items():
        a = f()
        a.fit(b, epochs=2, batch_size=16)
        st = a.training_state_dict()
        c, w_only = f(), f()
        c.load_training_state_dict(st)
        w_only.module.load_state_dict(a.module.state_dict())   # weights, no moments
        w_only.version = a.version
        assert c.version == 1 and c.static_dims.tolist() == a.static_dims.tolist()
        for m in (a, c, w_only):
            m.fit(b, epochs=2, batch_size=16, seed=5)
        assert all(torch.equal(x, y) for x, y in zip(_weights(a), _weights(c))), name
        assert not all(torch.equal(x, y) for x, y in zip(_weights(a), _weights(w_only))), \
            f"{name}: optimizer moments made no difference — the test cannot see"
    other = M.MLPMechanism(*ARGS, hidden=16, seed=0, mechanism_id="other")
    raises(lambda: other.load_training_state_dict(st), ContractError, "wrong mechanism")
    base = M.ConstantVelocityMechanism(*ARGS)
    base.fit(b)
    r1, r2 = M.ResidualMechanism(base, hidden=8), M.ResidualMechanism(base, hidden=8)
    r1.fit(b, epochs=2, batch_size=16)
    r2.load_training_state_dict(r1.training_state_dict())
    r1.fit(b, epochs=1, batch_size=16, seed=3)
    r2.fit(b, epochs=1, batch_size=16, seed=3)
    assert all(torch.equal(x, y) for x, y in zip(_weights(r1), _weights(r2)))


# ---- early stopping + failed fits (2026-10-05 review) -------------------------
# docs/foundation/LEARNING_PROCESS_IMPROVEMENTS.md §5: a non-finite loss raised
# mid-fit and left weights + the persistent Adam at the LAST epoch while
# `version` was not bumped; a NaN min_delta passed the budget check; the
# validation batch skipped _check; loss_last described a discarded epoch.
# The toy cases drive _nets.train with a SCRIPTED validation curve so the
# epoch that must be selected is known exactly; validation_fn snapshots the
# weights it is called on, so "restored the best epoch" is checked bit-exact.
def _toy(vals, nan_at=None, raise_at=None):
    """-> (module, params, loss_fn, validation_fn, snaps). 4 rows, so
    batch_size=2 is 2 optimizer steps per epoch. Loss call number `nan_at`
    (1-based) returns NaN; call `raise_at` raises RuntimeError."""
    with _nets.seeded(0):
        mod = torch.nn.Linear(1, 1)
    x = torch.linspace(0, 1, 4)[:, None]
    y = 2 * x + 1
    calls, snaps = [0], []

    def loss_fn(idx):
        calls[0] += 1
        if calls[0] == raise_at:
            raise RuntimeError("boom mid-fit")
        if calls[0] == nan_at:
            return torch.tensor(float("nan"))
        i = torch.as_tensor(idx)
        return ((mod(x[i]) - y[i]) ** 2).mean()

    def validation_fn():
        snaps.append([p.detach().clone() for p in mod.parameters()])
        return vals[len(snaps) - 1]
    return mod, list(mod.parameters()), loss_fn, validation_fn, snaps


def _train_toy(vals, epochs, **kw):
    nan_at, raise_at = kw.pop("nan_at", None), kw.pop("raise_at", None)
    mod, ps, lf, vf, snaps = _toy(vals, nan_at, raise_at)
    opt = torch.optim.Adam(ps, lr=0.05)
    info = {}
    hist = _nets.train(ps, lf, 4, epochs=epochs, batch_size=2, lr=0.05, seed=0,
                       optimizer=opt, validation_fn=vf, module=mod, report=info, **kw)
    return mod, ps, opt, snaps, hist, info


def _is(ps, snap):
    return all(torch.equal(p.detach(), q) for p, q in zip(ps, snap))


@case
def early_stopping_patience_restores_best_weights_and_moments():
    mod, ps, opt, snaps, hist, info = _train_toy([3.0, 2.0, 2.5, 2.4, 1.0], 10,
                                                 patience=2)
    assert len(hist) == 4 and info["stop_reason"] == "patience", (hist, info)
    assert info["selected_epoch"] == 2 and info["gradient_steps"] == 8, info
    assert _is(ps, snaps[1]) and not _is(ps, snaps[3]), "best epoch not restored"
    assert int(opt.state[ps[0]]["step"]) == 4, "optimizer not restored WITH weights"
    assert _nets.DEFAULT_PATIENCE == 5


@case
def early_stopping_min_delta():
    vals = [3.0, 2.95, 2.92, 2.91]
    _, ps, _, snaps, hist, info = _train_toy(vals, 4, patience=3, min_delta=0.1)
    assert info == {"selected_epoch": 1, "stop_reason": "patience",
                    "gradient_steps": 8}, info
    assert _is(ps, snaps[0])
    _, ps, _, snaps, hist, info = _train_toy(vals, 4, patience=3, min_delta=0.0)
    assert info["selected_epoch"] == 4 and info["stop_reason"] == "epochs", info
    for bad in (float("nan"), float("inf"), -0.1):
        raises(lambda: _train_toy(vals, 4, min_delta=bad), ValueError,
               f"min_delta={bad}")
    raises(lambda: _train_toy(vals, 4, patience=0), ValueError, "patience 0")


@case
def nan_validation_is_no_improvement():
    nan = float("nan")
    _, ps, _, snaps, hist, info = _train_toy([3.0, nan, 2.5], 3, patience=5)
    assert info["selected_epoch"] == 3 and len(hist) == 3, info   # NaN did not stop it
    _, ps, opt, snaps, hist, info = _train_toy([3.0, nan, nan, 0.1], 10, patience=2)
    assert info["selected_epoch"] == 1 and info["stop_reason"] == "patience", info
    assert _is(ps, snaps[0]) and int(opt.state[ps[0]]["step"]) == 2
    raises(lambda: _train_toy([nan, nan], 2), FloatingPointError, "no finite val")


@case
def nonfinite_training_loss_and_exceptions_restore_best():
    # call 6 = epoch 3, 2nd batch (one step of epoch 3 already taken)
    _, ps, opt, snaps, hist, info = _train_toy([2.0, 1.0, 0.5], 5, nan_at=6)
    assert info["stop_reason"] == "diverged" and info["selected_epoch"] == 2, info
    assert len(hist) == 2 and _is(ps, snaps[1])
    assert int(opt.state[ps[0]]["step"]) == 4, "epoch-3 step not undone"
    raises(lambda: _train_toy([2.0], 5, nan_at=1), FloatingPointError, "nothing selected")
    mod, ps, lf, vf, snaps = _toy([2.0, 1.0, 0.5], raise_at=6)
    opt = torch.optim.Adam(ps, lr=0.05)
    raises(lambda: _nets.train(ps, lf, 4, epochs=5, batch_size=2, lr=0.05, seed=0,
                               optimizer=opt, validation_fn=vf, module=mod),
           RuntimeError, "exception propagates")
    assert _is(ps, snaps[1]) and int(opt.state[ps[0]]["step"]) == 4, \
        "exception left the module at a discarded epoch"
    mod, ps, lf, _, _ = _toy([], nan_at=3)                 # no validation -> raise
    raises(lambda: _nets.train(ps, lf, 4, epochs=5, batch_size=2, lr=0.05, seed=0),
           FloatingPointError, "no-validation divergence")


def _full_state(m):
    sd = {k: v.clone() for k, v in m.module.state_dict().items()}
    st = None if m._opt is None else {
        i: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s_.items()}
        for i, s_ in m._opt.state_dict()["state"].items()}
    return sd, st


def _same_state(a, b):
    (sa, oa), (sb, ob) = a, b
    if set(sa) != set(sb) or not all(torch.equal(sa[k], sb[k]) for k in sa):
        return False
    if (oa is None) != (ob is None):
        return False
    return oa is None or all(
        set(oa[i]) == set(ob[i]) and all(
            torch.equal(oa[i][k], ob[i][k]) if torch.is_tensor(oa[i][k])
            else oa[i][k] == ob[i][k] for k in oa[i]) for i in oa)


@case
def failed_fit_is_atomic_and_version_consistent():
    b = _batch(B=32)
    m = M.MLPMechanism(*ARGS, hidden=16, seed=0)
    orig = m._forward
    m._forward = lambda prep: (lambda o: (o[0] * float("nan"),) + tuple(o[1:]))(orig(prep))
    raises(lambda: m.fit(b, epochs=2, batch_size=16), FloatingPointError, "first fit")
    assert m.version == 0 and m._opt is None and not bool(m.module.out_std.fitted), \
        "a failed FIRST fit left the standardiser frozen / an optimizer behind"
    del m._forward
    m.fit(b, epochs=2, batch_size=16)
    before = _full_state(m)
    calls = [0]

    def late_nan(prep):                                     # diverge in epoch 2
        calls[0] += 1
        o = orig(prep)
        return (o[0] * float("nan"),) + tuple(o[1:]) if calls[0] == 4 else o
    m._forward = late_nan
    raises(lambda: m.fit(b, epochs=3, batch_size=16), FloatingPointError, "second fit")
    del m._forward
    assert m.version == 1 and _same_state(_full_state(m), before), \
        "failed fit changed state without a version bump"
    rep = m.fit(b, epochs=1, batch_size=16)                 # recovers, same seed
    assert m.version == 2 and rep["stop_reason"] == "epochs"


@case
def validation_batch_checked_and_selected_loss_reported():
    b, v = _batch(B=32), _batch(B=8, seed=1)
    m = M.MLPMechanism(*ARGS, hidden=16, seed=0)
    raises(lambda: m.fit(b, validation="not a batch"), ContractError, "type")
    raises(lambda: m.fit(b, validation=_batch(B=8, N=2)), ContractError, "wrong N")
    assert m.version == 0
    rep = m.fit(b, epochs=6, batch_size=16, validation=v, patience=2)
    k = rep["selected_epoch"]
    assert 1 <= k <= rep["epochs"] == len(rep["loss_hist"]) == len(rep["validation_loss"])
    assert rep["loss_selected"] == rep["loss_hist"][k - 1]
    assert rep["loss_last"] == rep["loss_hist"][-1]       # existing key unchanged
    assert rep["gradient_steps"] >= rep["epochs"] * 2


if __name__ == "__main__":
    sys.exit(run_all("foundation-mechanisms-unit"))
