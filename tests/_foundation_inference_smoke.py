"""Foundation inference smoke (2026-10-03) — plan Stage 7 test list.

WHAT IS CLAIMED

    "improvement plan.md" Stage 7 tests: "ambiguous observations; missing
    data; known posterior fixtures; contradictory evidence; changed
    dynamics; calibration curves; interval coverage; particle degeneration
    if applicable." Gate: "uncertainty has demonstrated predictive value and
    can increase appropriately. Agreement among models alone is not
    accepted as proof of correctness."

WHY IT IS WORTH A CONTRACT

    Every one of these has a cheap fake. A posterior that never moves passes
    "ambiguous observations"; one that ignores missing data by treating it
    as a zero reading passes nothing but looks fine in a dashboard (the
    sensor bus once turned every absent frame into a confident NEUTRAL
    reading); a hypothesis that hit probability 0 can never come back — a
    guard-becomes-latch (CLAUDE.md §4.1); an ensemble whose members agree
    can still all be wrong. So each claim below is paired with the case that
    would expose the fake.

Contracts:
    A. Kalman == analytic posterior: the filtered mean and covariance of
       x_T equal the batch Gaussian conditional of x_T on every observation
       (built from the joint covariance by hand), including a step whose
       observation is ABSENT (skipped, not zero) and steps observed only in
       part (mask).
    B. Known hypothesis posterior: coin-bias hypotheses after 60 flips equal
       prior x likelihood normalised, exactly (floor 0); tempering by tau
       equals likelihood^tau.
    C. Ambiguous observations keep mass split: hypotheses that predict the
       same thing for what was observed stay at their prior however much
       such data arrives; a particle filter on a sign-ambiguous sensor
       (y = |x| + noise) keeps BOTH modes (each 30-70% of mass) — the case
       one Gaussian cannot represent.
    D. Missing data leaves every belief unchanged: HypothesisSet,
       KalmanFilter, ParticleFilter and PredictiveCUSUM all treat
       None/ABSENT as a no-op, bit-for-bit; a PARTIAL hypothesis update is
       refused rather than rewarding the unscored hypothesis.
    E. Contradictory evidence + floor recovers: after 300 flips favouring A
       the truth switches to B; with floor 1e-3 B becomes MAP within 40
       flips, without a floor it takes > 150 (the floor is the escape path
       of the overconfidence latch, and the test shows it is the reason).
    F. Changed dynamics vs noise: on 5 x 4000 steps of pure noise of KNOWN
       variance the CUSUM never fires (while a one-step |z| > 3 rule fires
       dozens of times — the distinction is real); a Kalman-tracked object
       that starts accelerating trips it within 40 steps, never before.
    G. Ensemble: in-distribution 50/90% interval coverage near nominal;
       epistemic variance INCREASES out of distribution (4x speeds) by >2x.
       AGREEMENT IS NOT CORRECTNESS: on data whose gravity flipped, the
       states look familiar (applicability ~1) and epistemic grows far less
       than out of distribution, yet NLL is much worse — and the CUSUM on
       the ensemble's own predictive fires there and not in distribution.
    H. Particle degeneracy: a sharp likelihood drives ESS below threshold
       and triggers systematic resampling (ESS back to n, fewer distinct
       particles recorded); a flat likelihood triggers none.

    I. (verifier F5, stage9_11 E9b) a VARIANCE-INFLATING change with zero
       mean is detected: the slope change 2 -> 3 at noise 0.5 leaves
       residuals x + N(0, 0.5^2), x ~ U(-1, 1), scored by the stale model at
       sd 0.5 — E[z^2] = 2.33, unconditional mean 0. MEASURED BEFORE THE FIX
       (old dispersion drift k_disp = 1 needed E[z^2] > 2.41): first alarm
       within 160 rows in 66% of runs, median > 80 rows. Contract over 200
       runs: >= 95% alarm within 160 rows (8 episodes of 20), median <= 80.
       NOT AT THE PRICE OF NOISE: 50 x 4000 steps of unit noise of the
       KNOWN variance at D = 1 give 0 dispersion alarms (the mean-shift
       statistic gives its usual ~1 per 10^5 steps, as before; contract F
       still holds at D = 2), and ONE 8-sd outlier in clean data is not an
       alarm (the old statistic fired on any single |z| > 6.5 at D = 1).
    J. Misspecification is flagged, not silent (verifier F6/E10d): a
       HypothesisSet whose hypotheses all miss the truth (cue -> response
       shifts; the truth agrees with one shift on half the cues only; slip
       0.077, floor 1e-4, information-gain queries as in E10d) converges
       confidently on a wrong shift in 40/40 scenarios — and `misspecified`
       is raised, with an evidence count, in >= 36/40 within 80
       interactions. Before the fix there was no flag at all (no API).
       A WELL-SPECIFIED set under the same noise: <= 1 of 200 scenarios
       flagged in 200 interactions (MEASURED 1/300 over seeds 1000-1299; a
       sequential test cannot promise zero — the monitor's false-alarm run
       length is >= e^8 ~ 3000 observations for ANY predictive). Adding a
       hypothesis resets the flag (the escape path: the flag concerns the
       set, and a changed set is a new question).

Run: PYTHONPATH=. python tests/_foundation_inference_smoke.py
"""
import sys
import time
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*requires_grad.*")

import numpy as np

from developmental_ai.foundation.contracts import ABSENT, UNKNOWN, ContractError
from developmental_ai.foundation.inference import (BootstrapEnsemble, HypothesisSet,
                                                   KalmanFilter, ParticleFilter,
                                                   PredictiveCUSUM)

T0 = time.time()


# --------------------------------------------------------------------- A

def _cv_model(dt=0.1, q=0.05, r=0.3):
    F = np.array([[1, dt], [0, 1]])
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    return F, Q, np.eye(2), np.diag([r ** 2, (2 * r) ** 2])


def test_kalman_matches_analytic():
    rng = np.random.default_rng(0)
    F, Q, H, R = _cv_model()
    m0, P0 = np.array([0.0, 1.0]), np.diag([1.0, 0.5])
    T, n = 30, 2
    x = rng.multivariate_normal(m0, P0)
    ys, masks = [], []
    for t in range(1, T + 1):
        x = F @ x + rng.multivariate_normal(np.zeros(2), Q)
        y = H @ x + rng.multivariate_normal(np.zeros(2), R)
        mask = np.array([True, t % 3 != 0])            # every 3rd step: pos only
        ys.append(ABSENT if t == 17 else y)            # one dropped frame
        masks.append(mask)
    kf = KalmanFilter(F, Q, H, R, m0, P0)
    for y, mk in zip(ys, masks):
        kf.predict()
        kf.update(y, mask=mk)
    assert kf.skipped == 1 and kf.updates == T - 1
    # Batch: X = [x_0..x_T] = mu + A eps, eps ~ N(0, blockdiag(P0, Q..Q)).
    N = n * (T + 1)
    A = np.zeros((N, N))
    Fp = [np.linalg.matrix_power(F, k) for k in range(T + 1)]
    for t in range(T + 1):
        for s in range(t + 1):
            A[n * t:n * t + n, n * s:n * s + n] = Fp[t - s]
    Se = np.zeros((N, N))
    Se[:n, :n] = P0
    for s in range(1, T + 1):
        Se[n * s:n * s + n, n * s:n * s + n] = Q
    mu = np.concatenate([Fp[t] @ m0 for t in range(T + 1)])
    Sx = A @ Se @ A.T
    rows, yv, rv = [], [], []
    for t, (y, mk) in enumerate(zip(ys, masks), start=1):
        if y is ABSENT:
            continue
        for d in np.flatnonzero(mk):
            row = np.zeros(N)
            row[n * t:n * t + n] = H[d]
            rows.append(row)
            yv.append(y[d])
            rv.append(R[d, d])
    G = np.array(rows)
    Sy = G @ Sx @ G.T + np.diag(rv)
    C = Sx[-n:] @ G.T
    post_m = mu[-n:] + C @ np.linalg.solve(Sy, np.array(yv) - G @ mu)
    post_P = Sx[-n:, -n:] - C @ np.linalg.solve(Sy, C.T)
    assert np.allclose(kf.x, post_m, atol=1e-9), (kf.x, post_m)
    assert np.allclose(kf.P, post_P, atol=1e-9), (kf.P, post_P)
    print(f"  A. Kalman x_T/P_T == batch Gaussian conditional over {len(yv)} scalar "
          f"observations (max err {max(np.abs(kf.x - post_m).max(), np.abs(kf.P - post_P).max()):.1e}),"
          f" with one ABSENT frame skipped and 10 pos-only steps")


# --------------------------------------------------------------------- B

def _coin_ll(ps, k):
    return np.log(np.where(k == 1, ps, 1 - ps))


def test_known_hypothesis_posterior():
    ps = np.array([0.2, 0.5, 0.8])
    prior = np.array([0.5, 0.3, 0.2])
    flips = (np.random.default_rng(1).random(60) < 0.8).astype(int)
    for tau in (1.0, 0.5):
        hs = HypothesisSet(floor=0.0, temperature=tau)
        hs.add_hypothesis("p0.2")
        hs.add_hypothesis("p0.5", prior=0.3 / 0.8)
        hs.add_hypothesis("p0.8", prior=0.2)
        assert np.allclose(hs.probs(), prior), hs.probs()
        for k in flips:
            hs.update(_coin_ll(ps, k))
        nh = flips.sum()
        logpost = np.log(prior) + tau * (nh * np.log(ps) + (60 - nh) * np.log(1 - ps))
        exact = np.exp(logpost - np.logaddexp.reduce(logpost))
        assert np.allclose(hs.probs(), exact, atol=1e-12), (hs.probs(), exact)
    print(f"  B. coin-bias posterior after 60 flips ({nh} heads) == prior x likelihood "
          f"exactly; tempered (tau=0.5) == likelihood^0.5; MAP {hs.map()}")


# --------------------------------------------------------------------- C

def test_ambiguity_keeps_mass_split():
    hs = HypothesisSet(floor=0.0)
    hs.add_hypothesis("mu=+1")
    hs.add_hypothesis("mu=-1")
    rng = np.random.default_rng(2)
    for _ in range(500):                       # sensor reads |y|: sign invisible
        a = abs(rng.normal(1.0, 0.5))
        ll = [np.logaddexp(-0.5 * ((a - m) / 0.5) ** 2, -0.5 * ((-a - m) / 0.5) ** 2)
              for m in (1.0, -1.0)]
        hs.update(ll)
    assert np.allclose(hs.probs(), [0.5, 0.5], atol=1e-12), hs.probs()
    # Particle filter on y = |x| + noise, static x = 2.
    rng = np.random.default_rng(3)
    pf = ParticleFilter(rng.normal(0, 3, 4000), lambda p, r: p + r.normal(0, 0.02, p.shape),
                        lambda p, y: -0.5 * ((y - np.abs(p[:, 0])) / 0.3) ** 2, rng)
    for _ in range(20):
        pf.predict()
        pf.update(2.0 + rng.normal(0, 0.3))
    w, x = pf.weights, pf.particles[:, 0]
    pos, neg = w[x > 0].sum(), w[x < 0].sum()
    assert 0.3 < pos < 0.7 and 0.3 < neg < 0.7, (pos, neg)
    near = w[np.abs(np.abs(x) - 2.0) < 0.5].sum()
    assert near > 0.9, near
    print(f"  C. 500 sign-ambiguous readings leave two hypotheses at exactly 50/50; "
          f"particle filter on y=|x| keeps both modes: mass x>0 {pos:.2f}, x<0 {neg:.2f} "
          f"({near:.0%} within 0.5 of +-2) — a single Gaussian could not")


# --------------------------------------------------------------------- D

def test_missing_data_is_a_noop():
    hs = HypothesisSet()
    for n in ("a", "b", "c"):
        hs.add_hypothesis(n)
    hs.update([0.0, -1.0, -2.0])
    before = hs.log_probs()
    for miss in (None, ABSENT, UNKNOWN, {"a": ABSENT, "b": ABSENT, "c": ABSENT}):
        assert hs.update(miss) is False
    assert np.array_equal(hs.log_probs(), before) and hs.steps == 1
    try:
        hs.update({"a": -1.0, "b": ABSENT, "c": -2.0})
        raise AssertionError("partial evidence accepted")
    except ContractError:
        pass
    F, Q, H, R = _cv_model()
    kf = KalmanFilter(F, Q, H, R, [0, 1], np.eye(2))
    kf.predict()
    x, P = kf.x.copy(), kf.P.copy()
    assert kf.update(ABSENT) is None and kf.update(None) is None
    assert kf.update(np.zeros(2), mask=[False, False]) is None
    assert np.array_equal(kf.x, x) and np.array_equal(kf.P, P)
    rng = np.random.default_rng(0)
    pf = ParticleFilter(rng.normal(size=100), lambda p, r: p, lambda p, y: -p[:, 0] ** 2, rng)
    pf.update(1.0)
    lw = pf.logw.copy()
    assert pf.update(ABSENT) is None and np.array_equal(pf.logw, lw)
    cs = PredictiveCUSUM()
    cs.update([0.0], [1.0], [2.5])
    st = (cs.s_hi, cs.s_lo, cs.s_disp, cs.step)
    assert cs.update([0.0], [1.0], ABSENT) is None
    assert (cs.s_hi, cs.s_lo, cs.s_disp, cs.step) == st and cs.skipped == 1
    print("  D. None/ABSENT/UNKNOWN leave hypothesis, Kalman, particle and CUSUM state "
          "bit-identical; partial hypothesis evidence refused")


# --------------------------------------------------------------------- E

def _recovery_steps(floor):
    hs = HypothesisSet(floor=floor)
    hs.add_hypothesis("A")
    hs.add_hypothesis("B")
    ps = np.array([0.8, 0.2])
    rng = np.random.default_rng(4)
    pmin = 1.0
    for k in (rng.random(300) < 0.8).astype(int):
        hs.update(_coin_ll(ps, k))
        pmin = min(pmin, hs.probs().min())
    for t, k in enumerate((rng.random(1000) < 0.2).astype(int), start=1):
        hs.update(_coin_ll(ps, k))
        if hs.map() == "B":
            return t, pmin
    return 10 ** 6, pmin


def test_contradiction_and_floor_recovery():
    t_floor, pmin = _recovery_steps(1e-3)
    t_exact, pmin0 = _recovery_steps(0.0)
    assert pmin >= 1e-3 - 1e-15, pmin
    assert t_floor <= 40, t_floor
    assert t_exact > 150, t_exact
    print(f"  E. after 300 flips for A, truth switches to B: MAP flips in {t_floor} flips "
          f"with floor 1e-3 (min prob held at {pmin:.1e}) vs {t_exact} without "
          f"(min prob fell to {pmin0:.1e})")


# --------------------------------------------------------------------- F

def test_change_vs_noise():
    fired, naive = 0, 0
    for seed in range(5):
        rng = np.random.default_rng(10 + seed)
        cs = PredictiveCUSUM()
        sd = np.array([0.3, 2.0])                       # known, unequal noise
        for _ in range(4000):
            y = rng.normal(0, sd)
            cs.update(np.zeros(2), sd ** 2, y)
            naive += int(np.any(np.abs(y / sd) > 3))
        fired += len(cs.alarms)
    assert fired == 0, fired
    assert naive > 20, naive
    # Kalman-tracked object; dynamics change at t=300 (constant acceleration).
    F, Q, H, R = _cv_model(q=0.01)
    rng = np.random.default_rng(5)
    kf = KalmanFilter(F, Q, H[:1], R[:1, :1], [0, 1], np.eye(2) * 0.1)
    cs = PredictiveCUSUM()
    x = np.array([0.0, 1.0])
    first = None
    for t in range(600):
        acc = 0.0 if t < 300 else 0.5
        x = F @ x + np.array([0.5 * acc * 0.01, acc * 0.1]) + \
            rng.multivariate_normal(np.zeros(2), Q)
        y = H[:1] @ x + rng.normal(0, np.sqrt(R[0, 0]), 1)
        kf.predict()
        mu, S = kf.predictive()
        cs.update_gaussian(mu, S, y)
        kf.update(y)
        if cs.alarms and first is None:
            first = cs.alarms[0][0]
    assert first is not None and 300 <= first < 340, (first, cs.alarms[:3])
    print(f"  F. pure noise of known variance, 5 x 4000 steps: 0 CUSUM alarms (a one-step "
          f"|z|>3 rule fired {naive} times); dynamics change at t=300 detected at "
          f"t={first} ({cs.alarms[0][1]})")


# --------------------------------------------------------------------- G

def test_ensemble_uncertainty():
    from developmental_ai.foundation import mechanisms as M
    w = M.BoxWorld(n_balls=2, size=40.0, obs_noise=0.2)
    d = M.make_streams(w, 120, 20, seed=3)
    tr = d["train"].transitions()
    args = (w.n_entities, 2, 5, 2, M.EVENT_CLASSES)
    ci, di = w.ball_cont_index(), w.ball_disc_index()
    ens = BootstrapEnsemble(lambda i: M.MLPMechanism(*args, hidden=64, seed=i,
                                                     mechanism_id=f"mlp.m{i}"), k=5)
    ens.fit(tr, epochs=80, lr=3e-3, batch_size=64)
    assert ens.version == 1 and set(ens.model_versions().values()) == {1}

    def free(b):
        return b.subset(np.flatnonzero(np.all(b.events[:, :2] == 0, 1)))

    te = free(d["heldout"].transitions())
    r = M.one_step(ens, te, ci, di, split_contact=False)["all"]
    assert abs(r["cov50"] - 0.5) < 0.1 and abs(r["cov90"] - 0.9) < 0.05, r
    dec = ens.decompose(te.state, te.action, te.dt)
    assert np.allclose(dec["total"], ens.predict(te.state, te.action, te.dt).variance())
    ep_in = dec["epistemic"][:, ci].mean()
    ood = free(M.make_streams(w, 20, 20, seed=11, speed_scale=4.0)["heldout"].transitions())
    ep_ood = ens.decompose(ood.state, ood.action, ood.dt)["epistemic"][:, ci].mean()
    assert ep_ood > 2 * ep_in, (ep_ood, ep_in)
    # Agreement is not correctness: gravity reversed, states otherwise familiar.
    flip = free(M.make_streams(w, 20, 20, seed=11, gravity=-w.gravity)["heldout"].transitions())
    ep_flip = ens.decompose(flip.state, flip.action, flip.dt)["epistemic"][:, ci].mean()
    rf = M.one_step(ens, flip, ci, di, split_contact=False)["all"]
    app_in = float(np.mean(ens.applicability(te.state)))
    app_flip = float(np.mean(ens.applicability(flip.state)))
    assert app_flip > 0.95 * app_in, (app_flip, app_in)
    assert ep_flip / ep_in < 0.75 * ep_ood / ep_in, (ep_flip, ep_ood, ep_in)
    assert rf["nll_cont"] > r["nll_cont"] + 1.0, (rf["nll_cont"], r["nll_cont"])

    def alarms(b):
        dist = ens.predict(b.state, b.action, b.dt)
        mu, var, y = dist.mean()[:, ci], dist.variance()[:, ci], b.next_state.continuous()[:, ci]
        cs = PredictiveCUSUM()
        for i in range(len(b)):
            cs.update(mu[i], var[i], y[i])
        return cs.alarms

    a_in, a_flip = alarms(te), alarms(flip)
    assert not a_in, a_in
    assert a_flip and a_flip[0][0] < 30, a_flip[:3]
    print(f"  G. ensemble K=5 held-out coverage 50/90 = {r['cov50']:.2f}/{r['cov90']:.2f}; "
          f"epistemic x{ep_ood / ep_in:.1f} at 4x speed; gravity flipped: applicability "
          f"{app_flip:.2f} (in-dist {app_in:.2f}), epistemic only x{ep_flip / ep_in:.1f}, "
          f"but NLL {rf['nll_cont']:.2f} vs {r['nll_cont']:.2f} and CUSUM fires at step "
          f"{a_flip[0][0]} ({a_flip[0][1]}); 0 alarms in distribution")


# --------------------------------------------------------------------- H

def test_particle_degeneracy():
    rng = np.random.default_rng(6)
    pf = ParticleFilter(rng.normal(0, 1, (500, 2)), lambda p, r: p, lambda p, y: np.zeros(len(p)),
                        rng, ess_threshold=0.5)
    for _ in range(10):
        pf.predict()
        pf.update(0.0)
    assert pf.diag["resamples"] == 0, "flat likelihood triggered resampling"
    pf.loglik = lambda p, y: -0.5 * ((p - y) ** 2).sum(1) / 0.05 ** 2
    out = pf.update(np.array([0.5, -0.5]))
    assert out["ess"] < 0.5 * pf.n and out["resampled"], out
    assert pf.diag["resamples"] == 1 and np.isclose(1.0 / np.sum(pf.weights ** 2), pf.n)
    assert pf.diag["distinct_after_resample"][-1] < pf.n
    assert pf.diag["collapses"] == int(out["ess"] < 2.0)
    print(f"  H. flat likelihood x10: 0 resamples; sharp likelihood: ESS {out['ess']:.1f}/"
          f"{pf.n} -> systematic resample, ESS back to {pf.n}, "
          f"{pf.diag['distinct_after_resample'][-1]} distinct particles kept")


# --------------------------------------------------------------------- I

def _slope_change_first_alarms(factory, runs=200, rows=400, seed=21):
    rng = np.random.default_rng(seed)
    firsts = []
    for _ in range(runs):
        cs = factory()
        y = rng.uniform(-1, 1, rows) + rng.normal(0, 0.5, rows)   # (3x+1) - (2x+1) + noise
        first = rows + 1
        for t in range(rows):
            if cs.update([0.0], [0.25], [y[t]])["alarm"]:
                first = t
                break
        firsts.append(first)
    return np.array(firsts)


def test_variance_inflating_change(factory=PredictiveCUSUM):
    f = _slope_change_first_alarms(factory)
    within, med = float(np.mean(f < 160)), float(np.median(f))
    null = mean_null = 0
    for seed in range(50):
        rng = np.random.default_rng(1000 + seed)
        cs = factory()
        for v in rng.normal(0, 1, 4000):
            cs.update([0.0], [1.0], [v])
        null += sum(k == "dispersion" for _, k in cs.alarms)
        mean_null += sum(k == "mean_shift" for _, k in cs.alarms)
    cs = factory()
    rng = np.random.default_rng(7)
    for t in range(400):
        cs.update([0.0], [1.0], [8.0 if t == 200 else rng.normal()])
    one_outlier = len(cs.alarms)
    print(f"  I. slope 2->3 at noise 0.5 (E[z^2] 2.33, mean 0): alarm within 160 rows in "
          f"{within:.0%} of 200 runs, median first alarm {med:.0f} rows; unit noise of known "
          f"variance 50 x 4000 steps: {null} dispersion alarms ({mean_null} mean-shift, "
          f"as many as before the fix: the 4-sd clip leaves the null rate alone); one 8-sd "
          f"outlier: {one_outlier} alarms")
    assert within >= 0.95 and med <= 80, (within, med)
    assert null == 0, null
    assert mean_null <= 6, mean_null
    assert one_outlier == 0, cs.alarms


# --------------------------------------------------------------------- J

def _cue_world(seed, misspecified, K=6):
    """Truth: cue c -> response (c + shift) % K, or (misspecified) a map
    agreeing with that shift on half the cues only and with no shift."""
    r = np.random.default_rng(seed)
    shift = int(r.integers(K))
    resp = [(c + shift) % K for c in range(K)]
    if misspecified:
        half = sorted(r.choice(K, K // 2, replace=False).tolist())
        for c in half:
            resp[c] = (resp[c] + 1 + int(r.integers(K - 1))) % K
        for s in range(K):
            if all(resp[c] == (c + s) % K for c in range(K)):
                resp[half[0]] = (resp[half[0]] + 1) % K
    return resp, shift


def _cue_run(seed, misspecified, steps, K=6, slip=0.0766, floor=1e-4):
    resp, shift = _cue_world(seed, misspecified, K)
    rng = np.random.default_rng(10_000 + seed)
    hs = HypothesisSet(floor=floor, max_hypotheses=K)
    for s in range(K):
        hs.add_hypothesis(f"s{s}")
    shifts = np.arange(K)
    conf_t = flag_t = None
    for t in range(1, steps + 1):
        cue = int(rng.integers(K))
        p = hs.probs()
        tables = []
        for cmd in range(K):                # the outcome table of every command
            hit = np.where((cue + shifts) % K == cmd, 1 - slip, slip)
            tables.append(np.stack([1 - hit, hit], 1))
        # information-gain selection (I(H;O) = H(pred) - E_h H(O|h)), eps 0.05
        ent = lambda q: -np.sum(q * np.log(q), -1)
        gain = [ent(p @ T) - p @ ent(T) for T in tables]
        cmd = int(np.argmax(gain)) if rng.random() > 0.05 else int(rng.integers(K))
        table = np.log(tables[cmd])
        o = int(rng.random() < (1 - slip if cmd == resp[cue] else slip))
        hs.update_categorical(table, o)
        if conf_t is None and hs.probs().max() > 0.9:
            conf_t = t
        if flag_t is None and hs.misspecified:
            flag_t = t
    return conf_t, flag_t, hs


def test_misspecification_flag():
    if not hasattr(HypothesisSet, "update_categorical"):
        raise AssertionError("no misspecification API on HypothesisSet (pre-fix state)")
    conf_wrong = flagged = 0
    t_flags, evid = [], []
    for seed in range(40):
        conf_t, flag_t, hs = _cue_run(seed, True, 80)
        conf_wrong += conf_t is not None
        if flag_t is not None:
            flagged += 1
            t_flags.append(flag_t)
            evid.append(hs.misspecification()["evidence"])
    false_flags = 0
    for seed in range(200):
        _, flag_t, _ = _cue_run(1000 + seed, False, 200)
        false_flags += flag_t is not None
    hs = _cue_run(0, True, 80)[2]
    was = hs.misspecified
    hs.add_hypothesis("other")
    print(f"  J. truth NOT in the set: confident (>0.9) on a wrong shift in {conf_wrong}/40, "
          f"`misspecified` raised in {flagged}/40 within 80 interactions (median at "
          f"{np.median(t_flags) if t_flags else '-'}; evidence {min(evid) if evid else '-'}-"
          f"{max(evid) if evid else '-'} observations); well-specified: {false_flags}/200 "
          f"flagged in 200 interactions; adding a hypothesis reset the flag "
          f"({was} -> {hs.misspecified})")
    assert flagged >= 36, flagged
    assert false_flags <= 1, false_flags
    assert was and not hs.misspecified


if __name__ == "__main__":
    test_kalman_matches_analytic()
    test_known_hypothesis_posterior()
    test_ambiguity_keeps_mass_split()
    test_missing_data_is_a_noop()
    test_contradiction_and_floor_recovery()
    test_change_vs_noise()
    test_ensemble_uncertainty()
    test_particle_degeneracy()
    test_variance_inflating_change()
    test_misspecification_flag()
    print(f"  ({time.time() - T0:.1f}s)")
    print("[foundation-inference] ALL PASS")
