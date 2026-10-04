"""Unit cases for developmental_ai.foundation.inference (plan Stage 7).

Parts only: parameter validation, bounds (floor, min_keep, K <= 8),
explicit errors (partial / non-finite evidence, non-PSD covariances, shape
changes, total likelihood collapse), resampling arithmetic, CUSUM reset
(it can fire again — not a latch) and the ensemble variance identity. The
design claims (analytic posteriors, recovery, change vs noise, calibration,
OOD epistemic growth) live in tests/_foundation_inference_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_foundation_inference_unit.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, close, run_all, raises
from developmental_ai.foundation.contracts import ContractError
from developmental_ai.foundation.inference import (MAX_MEMBERS, BootstrapEnsemble,
                                                   HypothesisSet, KalmanFilter,
                                                   ParticleFilter, PredictiveCUSUM, ess,
                                                   systematic_resample)


# ---- hypotheses -------------------------------------------------------------
@case
def hypothesis_params_validated():
    raises(lambda: HypothesisSet(temperature=0), ContractError, "tau 0")
    raises(lambda: HypothesisSet(floor=0.1, max_hypotheses=10), ContractError, "floor*K>=1")
    raises(lambda: HypothesisSet(min_keep=5, max_hypotheses=3), ContractError, "min>max")
    hs = HypothesisSet()
    hs.add_hypothesis("a")
    raises(lambda: hs.add_hypothesis("a"), ContractError, "duplicate")
    raises(lambda: hs.add_hypothesis("b", prior=1.0), ContractError, "prior 1")
    raises(lambda: hs.add_hypothesis(""), ContractError, "empty name")


@case
def hypothesis_floor_and_errors():
    hs = HypothesisSet(floor=1e-3)
    for n in "abc":
        hs.add_hypothesis(n)
    close(hs.entropy(), np.log(3), 1e-12, "uniform entropy")
    hs.update([0.0, -1e4, -1e4])
    assert hs.probs().min() >= 1e-3 - 1e-15 and hs.map() == "a"
    close(hs.probs().sum(), 1.0, 1e-12, "normalised")
    raises(lambda: hs.update([0.0, -np.inf, 0.0]), ContractError, "-inf")
    raises(lambda: hs.update([0.0, 0.0]), ContractError, "wrong length")
    for _ in range(10000):                                  # no underflow
        hs.update([-50.0, -60.0, -70.0])
    assert np.all(np.isfinite(hs.log_probs()))


@case
def hypothesis_limits_and_reversible_pruning():
    hs = HypothesisSet(floor=0.0, max_hypotheses=3, min_keep=2)
    for n in "abc":
        hs.add_hypothesis(n)
    hs.update([0.0, -5.0, -1.0])
    hs.add_hypothesis("d")                                  # retires the least: b
    assert hs.names == ["a", "c", "d"] and hs.retired[-1]["name"] == "b"
    hs.update([0.0, -20.0, -20.0])
    gone = hs.prune(0.01)
    assert len(hs) == 2 and hs.map() == "a" and len(gone) == 1, (hs.names, gone)
    assert hs.prune(0.99) == [] and len(hs) == 2, "pruned below min_keep"
    hs.add_hypothesis("b", prior=0.1)                       # re-add a retired one
    assert "b" in hs.names
    tiny = HypothesisSet(max_hypotheses=2, min_keep=2)
    tiny.add_hypothesis("x")
    tiny.add_hypothesis("y")
    raises(lambda: tiny.add_hypothesis("z"), ContractError, "cannot retire below min_keep")


# ---- kalman -------------------------------------------------------------------
@case
def kalman_validation_and_symmetry():
    F, Q, H, R = np.eye(2), np.eye(2) * 0.1, np.eye(2), np.eye(2)
    raises(lambda: KalmanFilter(F, Q, H, R, [0, 0, 0], np.eye(2)), ContractError, "x0 shape")
    raises(lambda: KalmanFilter(F, -Q, H, R, [0, 0], np.eye(2)), ContractError, "Q not PSD")
    kf = KalmanFilter(F, Q, H, R, [0, 0], np.eye(2))
    raises(lambda: kf.predict(u=[1, 1]), ContractError, "u without B")
    raises(lambda: kf.update(np.zeros(3)), ContractError, "y shape")
    r = np.random.default_rng(0)
    for _ in range(500):
        kf.predict()
        inn = kf.update(r.normal(size=2))
        assert np.isfinite(inn["loglik"])
    assert np.allclose(kf.P, kf.P.T) and np.min(np.linalg.eigvalsh(kf.P)) > 0
    mu, S = kf.predictive(mask=[True, False])
    assert mu.shape == (1,) and S.shape == (1, 1)


# ---- particles -----------------------------------------------------------------
@case
def resampling_arithmetic():
    close(ess(np.full(10, 0.1)), 10.0, 1e-12, "uniform ESS")
    close(ess(np.eye(1, 10)[0]), 1.0, 1e-12, "degenerate ESS")
    idx = systematic_resample(np.array([0.5, 0.25, 0.25]), np.random.default_rng(0))
    assert len(idx) == 3 and np.bincount(idx, minlength=3).tolist() in ([2, 1, 0], [1, 1, 1],
                                                                         [2, 0, 1])
    idx = systematic_resample(np.array([0.5, 0.25, 0.125, 0.125]), np.random.default_rng(1))
    c = np.bincount(idx, minlength=4)
    assert c[0] == 2 and c[1] == 1 and c[2] + c[3] == 1, c


@case
def particle_filter_errors():
    rng = np.random.default_rng(0)
    raises(lambda: ParticleFilter(np.zeros(1), None, None, rng), ContractError, "1 particle")
    raises(lambda: ParticleFilter(np.zeros(5), None, None, None), ContractError, "no rng")
    pf = ParticleFilter(np.zeros(5), lambda p, r: p[:3], lambda p, y: np.zeros(len(p)), rng)
    raises(pf.predict, ContractError, "shape change")
    pf = ParticleFilter(np.zeros(5), lambda p, r: p, lambda p, y: np.full(len(p), -np.inf), rng)
    raises(lambda: pf.update(1.0), ContractError, "total likelihood collapse")


# ---- change -----------------------------------------------------------------------
@case
def cusum_validation_and_rearm():
    raises(lambda: PredictiveCUSUM(h_mean=0), ContractError, "h 0")
    cs = PredictiveCUSUM()
    raises(lambda: cs.update([0, 0], [1], [0, 0]), ContractError, "shape")
    raises(lambda: cs.update([0], [0], [0]), ContractError, "var 0")
    for _ in range(200):
        cs.update([0.0], [1.0], [3.0])                     # persistent 3-sd bias
    assert len(cs.alarms) >= 5 and all(k == "mean_shift" for _, k in cs.alarms), cs.alarms
    cs2 = PredictiveCUSUM()
    for _ in range(200):
        cs2.update([0.0], [1.0], [-3.0])
    assert len(cs2.alarms) >= 5, "two-sided: negative bias missed"
    cs3 = PredictiveCUSUM()
    r = np.random.default_rng(0)
    for _ in range(300):
        cs3.update([0.0], [1.0], r.normal(0, 4.0, 1))      # noise 4x what model claims
    assert any(k == "dispersion" for _, k in cs3.alarms), cs3.alarms


# ---- ensemble --------------------------------------------------------------------
@case
def ensemble_bounds_and_identity():
    from developmental_ai.foundation import mechanisms as M
    args = (2, 2, 1, 2, ("n", "h"))
    mk = lambda i: M.ConstantVelocityMechanism(*args, mechanism_id=f"cv{i}")
    raises(lambda: BootstrapEnsemble(mk, k=MAX_MEMBERS + 1), ContractError, "K > 8")
    raises(lambda: BootstrapEnsemble(mk, k=1), ContractError, "K = 1")
    raises(lambda: BootstrapEnsemble(lambda i: M.ConstantVelocityMechanism(*args), k=3),
           ContractError, "duplicate ids")
    r = np.random.default_rng(0)
    s = M.EntityBatch(r.normal(size=(40, 2, 2)), r.normal(size=(40, 2, 2)), np.ones((40, 2, 1)))
    s2 = M.EntityBatch(s.pos + s.vel * 0.1 + r.normal(0, 0.1, s.pos.shape), s.vel, s.attrs)
    b = M.TransitionBatch(s, np.zeros((40, 2, 2)), np.full(40, 0.1), s2, np.zeros((40, 2), int))
    ens = BootstrapEnsemble(mk, k=4)
    from developmental_ai.foundation.contracts import UNKNOWN
    assert ens.applicability(s) is UNKNOWN
    ens.fit(b, groups=np.arange(40) // 10)
    dec = ens.decompose(s, b.action, b.dt)
    assert np.allclose(dec["aleatoric"] + dec["epistemic"], dec["total"])
    # Constant-velocity members share a fit-independent mean, so they AGREE
    # exactly: epistemic is 0 by construction, whatever the truth — the
    # degenerate case of "agreement is not correctness". The bootstrap still
    # reached them: their fitted variances differ.
    assert np.all(dec["epistemic"] == 0)
    v = [m.predict(s, b.action, b.dt).variance() for m in ens.members]
    assert not np.allclose(v[0], v[1]), "bootstrap had no effect"
    assert ens.predict(s, b.action, b.dt).n_components == 4
    assert len(ens.rollout(s, np.zeros((3, 40, 2, 2)), np.full((3, 40), 0.1))) == 3
    raises(lambda: ens.fit(b, groups=np.arange(5)), ContractError, "groups length")


@case
def ensemble_training_state_resumes_bit_exact():
    # 2026-10-03 regression (RESULTS_stage6_8.md bug 2): members now keep one
    # Adam across fits; the ensemble's training state carries every member's
    # moments AND its version (which seeds the next bootstrap draw).
    import torch
    from developmental_ai.foundation import mechanisms as M
    args = (2, 2, 1, 2, ("n", "h"))
    mk = lambda i: M.MLPMechanism(*args, hidden=8, seed=i, mechanism_id=f"m{i}")
    r = np.random.default_rng(1)
    s = M.EntityBatch(r.normal(size=(40, 2, 2)), r.normal(size=(40, 2, 2)), np.ones((40, 2, 1)))
    s2 = M.EntityBatch(s.pos + s.vel * 0.1 + r.normal(0, 0.1, s.pos.shape), s.vel, s.attrs)
    b = M.TransitionBatch(s, np.zeros((40, 2, 2)), np.full(40, 0.1), s2, np.zeros((40, 2), int))
    a, c = BootstrapEnsemble(mk, k=2), BootstrapEnsemble(mk, k=2)
    a.fit(b, epochs=2, batch_size=16)
    c.load_training_state_dict(a.training_state_dict())
    assert c.version == 1
    a.fit(b, epochs=2, batch_size=16)
    c.fit(b, epochs=2, batch_size=16)
    for ma, mc in zip(a.members, c.members):
        assert all(torch.equal(x, y) for x, y in
                   zip(ma._trainable_params(), mc._trainable_params()))
    cv = BootstrapEnsemble(lambda i: M.ConstantVelocityMechanism(*args, mechanism_id=f"c{i}"),
                           k=2)
    raises(lambda: cv.training_state_dict(), ContractError, "member without state")


# ---- misspecification monitor (verifier F6, 2026-10-03) ---------------------------
@case
def misspec_monitor_parts():
    from developmental_ai.foundation.inference import (MisspecificationMonitor,
                                                       categorical_predictive)
    raises(lambda: MisspecificationMonitor(contamination=1.0), ContractError, "lam 1")
    raises(lambda: MisspecificationMonitor(h=float("nan")), ContractError, "h nan")
    raises(lambda: MisspecificationMonitor(tolerance=1.0), ContractError, "tol 1")
    raises(lambda: MisspecificationMonitor(inflate=1.0), ContractError, "inflate 1")
    m = MisspecificationMonitor(contamination=0.5, h=1.0)
    raises(lambda: m.observe_categorical([0.5, 0.6], 0), ContractError, "not a distribution")
    raises(lambda: m.observe_categorical([0.5, 0.5], 2), ContractError, "outcome idx")
    # increment = log(0.5 q + 0.25) - log q for a binary predictive
    r = m.observe_categorical([0.9, 0.1], 1)
    close(r["increment"], np.log(0.5 * 0.1 + 0.25) - np.log(0.1))
    assert r["statistic"] > 1.0 and r["misspecified"] and r["raised"] and m.flags == 1
    # a bounded step: even a 1e-9 outcome adds log(1 + lam/(M q)) only
    r2 = MisspecificationMonitor().observe_categorical([1 - 1e-9, 1e-9], 1)
    close(r2["increment"], np.log(0.5 + 0.25 / 1e-9), tol=1e-6)
    for _ in range(30):                       # consistent evidence clears it (hysteresis)
        m.observe_categorical([0.9, 0.1], 0)
    assert not m.misspecified and m.statistic == 0.0 and m.flags == 1
    m.reset("test")
    assert m.resets[-1]["reason"] == "test" and m.observed == 31
    # Wald: under the predictive, E[exp(increment)] = 1 exactly
    q = np.array([0.7, 0.2, 0.1])
    e = sum(q[o] * np.exp(MisspecificationMonitor().observe_categorical(q, o)["increment"])
            for o in range(3))
    close(e, 1.0)
    # Gaussian: LLR of the 4x-variance mixture against N(mu, var)
    g = MisspecificationMonitor(contamination=0.5, h=100.0)
    out = g.observe_gaussian([0.0], [1.0], [2.0])
    l0 = -0.5 * np.log(2 * np.pi) - 2.0
    l1 = np.log(0.5 * np.exp(l0) + 0.5 * np.exp(-0.5 * np.log(2 * np.pi * 4) - 0.5))
    close(out["increment"], l1 - l0)
    raises(lambda: g.observe_gaussian([0.0], [0.0], [1.0]), ContractError, "var 0")
    # categorical predictive: by hand
    L = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
    assert np.allclose(categorical_predictive([0.5, 0.5], L), [0.55, 0.45])
    raises(lambda: categorical_predictive([0.5, 0.5], np.log([[0.9, 0.2], [0.2, 0.8]])),
           ContractError, "unnormalised row")


@case
def hypothesis_update_categorical():
    L = np.log(np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]))
    a, b = HypothesisSet(), HypothesisSet()
    for n in ("A", "B", "C"):
        a.add_hypothesis(n)
        b.add_hypothesis(n)
    for o in (0, 1, 1, 0, 1):
        assert a.update_categorical(L, o)
        b.update(L[:, o])
    assert np.allclose(a.probs(), b.probs())          # same posterior as update()
    assert a.misspecification()["observed"] == 5 and a.misspecification()["unmonitored"] == 0
    assert b.misspecification()["unmonitored"] == 5 and b.misspecification()["observed"] == 0
    assert a.update_categorical(L, None) is False
    raises(lambda: a.update_categorical({"A": L[0], "B": L[1]}, 0), ContractError, "no row C")
    assert a.update_categorical({"A": L[0], "B": L[1], "C": L[2]}, 0)
    # a set whose members all predict outcome 0 at 0.9 but outcome 1 keeps coming
    hs = HypothesisSet(misspec={"h": 5.0})
    for n in ("A", "B"):
        hs.add_hypothesis(n)
    T = np.log(np.array([[0.9, 0.1], [0.9, 0.1]]))
    t = 0
    while not hs.misspecified and t < 50:
        hs.update_categorical(T, 1)
        t += 1
    assert hs.misspecified and t < 10, t
    rep = hs.misspecification()
    assert rep["evidence"] >= 2 and rep["flags"] == 1 and rep["flagged_at"] == t
    hs.add_hypothesis("C")                           # the set changed: reset (escape path)
    assert not hs.misspecified and hs.monitor.resets[-1]["was_misspecified"]


@case
def cusum_dispersion_params():
    raises(lambda: PredictiveCUSUM(disp_ratio=1.0), ContractError, "ratio 1")
    raises(lambda: PredictiveCUSUM(disp_clip=0.5), ContractError, "clip < 1")
    cs = PredictiveCUSUM()
    out = cs.update([0.0], [1.0], [100.0])                 # one 100-sd point
    assert not out["alarm"] and out["s_disp"] < cs.h_disp and out["s_mean"] < cs.h_mean


if __name__ == "__main__":
    sys.exit(run_all("foundation-inference-unit"))
