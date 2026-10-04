"""Foundation mechanisms smoke (2026-10-03) — plan Stage 6 test list.

WHAT IS CLAIMED

    "improvement plan.md" Stage 6 tests: "held-out trajectories; contact
    and non-contact examples; duration sensitivity; reset exclusion;
    transformed inputs; law violations; stochastic observations;
    long-horizon drift." And the Stage 6 rule that architecture
    comparisons run on IDENTICAL data with resource costs reported ("extra
    inputs and extra compute must not be confused with architectural
    gains").

    This suite does NOT claim the relational model beats the MLP. That is a
    predeclared A/B experiment (foundation/experiments), run separately with
    seeds and an acceptance rule. Here we only claim that every model trains
    and is evaluable on the same data through the same scorer, and that each
    design claim in foundation/mechanisms holds — including the ones that
    could be quietly false.

WHY IT IS WORTH A CONTRACT

    The repo's world model was only ever scored by its own loss. Two
    incidents this mirrors: a training window that crossed the replay write
    head taught "a splice between two unrelated worlds as if it were
    dynamics" (tests/_replay_sampling_smoke.py), and a horizon metric that
    sat on its own free-nats floor (0.1100 for three different models,
    rssm.prior_divergence docstring) — an instrument that could not see.
    So: scores on observable targets, reset splices refused by the shared
    stream rule, and every symmetry claim paired with its falsification.

Fixture: foundation.mechanisms.fixtures.BoxWorld — 3 balls + 4 wall
entities in a 4 m box, gravity, elastic contacts, variable dt in
[0.02, 0.1] s, ball 0 pushed by a random force. Split by EPISODE via
runtime.splits.

Contracts:
    A. Held-out trajectories: the split is by episode (assert_no_leak); the
       Interaction Network, the same-input MLP, the RSSM adapter and the
       constant-velocity floor all fit on the SAME TransitionBatch object
       and are scored by the same scorer on the same held-out batch; all
       scores finite; learned losses fall; params / FLOPs / train time /
       latency reported for each.
    B. Contact vs non-contact: both splits non-empty; the free-flight law
       (constant velocity) fails far worse on contact steps than free ones
       (so the split means something); the relational event head assigns
       more contact probability to actual contact steps than to free ones.
    C. Duration sensitivity: on held-out (state, ball) pairs where NO contact
       is physically possible within the step (an INPUT-side test: the ball's
       reach |v|dt + (g + F/m)dt^2/2 + 5 cm is shorter than its gap to every
       wall and every other ball's reach), the relational model's predicted
       dv_y/dt recovers gravity (within 25%), and the same pairs predicted at
       dt = 0.03 vs 0.09 (safe at 0.09) change dv_y by > 1.5x (physics: 3x;
       a dt-blind model: 1x — dt is used, not ignored). The selector is
       checked: the TRUE dv_y on it gives g = 9.81 exactly.
       WHY THE SELECTOR (2026-10-03): this used to select steps by their
       OUTCOME (no contact event happened). The model predicts E[dv | state],
       which near a wall or ball includes the chance of a bounce (dv_y ~ +6
       m/s), so outcome-selected steps carried that mass and g_hat read ~25%
       low on every seed and both torch stacks (anaconda 6.5/7.2/7.7, torch
       2.6 7.2/7.2/6.6/7.3/8.3). That was the probe conditioning on the
       future, not the model missing gravity: on the input-safe pairs the same
       models read 9.0/9.2/9.2 and 8.6/8.3/8.0.
    D. Reset exclusion: episodes stored back to back yield refused windows;
       every accepted window lies inside one episode; a refused window
       cannot be requested; scored windows never include one; and the cost
       of NOT refusing is measured (a splice pair's error dwarfs a real
       one), so the rule is known to bite.
    E. Transformed inputs: with the translation symmetry DECLARED, shifting
       every position (walls included) by c shifts predicted positions by
       exactly c and leaves velocities and event probabilities unchanged.
       FALSIFICATION: the same architecture without the declaration, and
       the MLP, are measurably NOT equivariant — the symmetry comes from the
       declaration, not for free. (The RSSM adapter's error is reported,
       never credited: equivariance is evidence only in a model shown to
       read its input.)
    F. Law violations (geometry.constraints): on true next states box
       containment and free-flight energy are never violated and energy is
       INAPPLICABLE exactly on steps where every unpushed ball touched
       something; the constant-velocity model violates both; learned
       models' violation counts are reported.
    G. Stochastic observations: a model trained on noisy observations
       predicts variance well above the noiseless model's on the same
       states (the noise is absorbed as aleatoric spread, not claimed as
       certainty) and its 90% intervals cover ~90% of held-out outcomes.
    H. Long-horizon drift: 10-step rollouts over accepted windows report
       RMSE per horizon and drift; error grows with horizon for every
       model; "mean" rollouts are under-dispersed vs "sample" rollouts at
       h = 10 (the documented caveat is real).
    I. RSSM comparability: native_prediction (through adapters.RSSMPredictor)
       and the batched path give the same decoded outcome; the RSSM layout
       has no event part, so event scores are absent, not zero.
    J. Records and residuals: a Prediction built from a published snapshot
       round-trips through JSON and decodes to the same distribution; a
       never-fitted model is refused; a residual on the constant-velocity
       base learns the missing gravity term, goes STALE (raises) when the
       base is refitted, and recovers on refit (not a latch).
    K. Static dims (regression, 2026-10-03; independent A/B verifier,
       experiments/foundation_ab/RESULTS_stage6_8.md bug 1). The walls'
       16 change dims are exactly zero, so they could sit at the variance
       floor and supply ~-66 of a training NLL whose floor was -73.7:
       the IN's per-epoch training loss then JUMPED by +16 to +76 nats
       mid-training on 360-520 transitions (seeds 3/7/11, lr 1e-3; the
       verifier's own repro went 9.5 -> -15.9 -> 332). Claimed now: (1) on
       those three small datasets the IN's loss never rises by more than 1
       nat between epochs; (2) the mask in force is EXACTLY the wall dims of
       the trained models (no ball dim, every IN/MLP/RSSM); (3) static dims
       are predicted as the identity, bit-exact, with the floor variance;
       (4) ESCAPE PATH (§4.1, the mask must not latch): a refit on data
       where one wall dim moves makes that dim dynamic and it is then
       predicted off-identity; (5) a dim DECLARED static that moves in the
       training data is refused with ContractError, not silently learned,
       and undeclare_static() lifts the refusal (it is not a latch either).

Run: PYTHONPATH=. python tests/_foundation_mechanisms_smoke.py
"""
import sys
import time
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*requires_grad.*")

import numpy as np
import torch

from developmental_ai.foundation import mechanisms as M
from developmental_ai.foundation.contracts import ContractError, from_json, to_json
from developmental_ai.foundation.geometry.constraints import ConstraintRegistry
from developmental_ai.foundation.runtime.snapshots import SnapshotRegistry
from developmental_ai.foundation.runtime.splits import assert_no_leak

T0 = time.time()
W = M.BoxWorld()
DATA = M.make_streams(W, 36, 80, seed=7)
TRAIN, HELD = DATA["train"].transitions(), DATA["heldout"].transitions()
ARGS = (W.n_entities, 2, 5, 2, M.EVENT_CLASSES)
CI, DI = W.ball_cont_index(), W.ball_disc_index()
FIT = dict(epochs=60, lr=3e-3, batch_size=64)
# Budget (2026-10-03): the suite must stay under ~60 s on the CI stack (py3.10,
# torch 2.6), where it measured 121-218 s. These models are tiny (hidden <= 128,
# batch 64): extra intra-op threads only add contention (IN 60-epoch fit under
# load: 1 thread 29.8 s, 2 threads 12.8 s, 4 threads 22.9 s), so pin 2. The RSSM
# trains 20 epochs (its claims — learns, finite, drift grows, both record paths
# agree — do not need a converged RSSM); G's two noise models train 30.
torch.set_num_threads(2)
FIT_RSSM = dict(FIT, epochs=20)
FIT_G = dict(FIT, epochs=30)
MODELS = {}


def fmt(d, keys=("rmse", "nll_cont", "cov50", "cov90", "crps", "nll_event")):
    return " ".join(f"{k}={d[k]:.3f}" for k in keys if k in d)


def _free(batch, n=W.n_balls):
    return np.all(batch.events[:, :n] == 0, 1)


# --------------------------------------------------------------------- A

def test_heldout_identical_data():
    scope = "box2d:s0"
    assert_no_leak([(scope, e) for e in DATA["train_episodes"]],
                   [(scope, e) for e in DATA["heldout_episodes"]])
    assert len(DATA["heldout_episodes"]) >= 3, DATA["heldout_episodes"]
    MODELS["cv"] = M.ConstantVelocityMechanism(*ARGS)
    MODELS["in"] = M.InteractionNetwork(*ARGS, hidden=48, seed=0)
    MODELS["mlp"] = M.MLPMechanism(*ARGS, hidden=128, seed=0)
    MODELS["rssm"] = M.RSSMMechanism(*ARGS, seed=0)
    sub = HELD.subset(np.arange(64))
    for name, m in MODELS.items():
        rep = m.fit(TRAIN) if name == "cv" else \
            m.fit(TRAIN, **(FIT_RSSM if name == "rssm" else FIT))         # SAME object
        assert rep["n"] == len(TRAIN) and m.version == 1
        if name != "cv":
            assert rep["loss_last"] < rep["loss_first"], f"{name} did not learn: {rep}"
        r = M.one_step(m, HELD, CI, DI if m.layout.G else None)
        for k, v in r["all"].items():
            if isinstance(v, float):
                assert np.isfinite(v), f"{name} {k} not finite"
        rc = M.resource_cost(m, sub)
        print(f"     {name:5s} params={rc['params']:6d} "
              f"flops/sample={rc.get('flops_per_sample', 0):8d} "
              f"train={rc['train_seconds']:.1f}s predict={rc['predict_ms']:.2f}ms/64 | "
              f"{fmt(r['all'])}")
    print(f"  A. split by episode ({len(DATA['train_episodes'])} train / "
          f"{len(DATA['heldout_episodes'])} held-out, no leak); 4 models fit on the "
          f"SAME {len(TRAIN)} transitions, scored on the same {len(HELD)}; all finite")


# --------------------------------------------------------------------- B

def test_contact_vs_free():
    r = M.one_step(MODELS["cv"], HELD, CI, DI)
    assert "contact" in r and "free" in r, "a split is empty"
    assert r["contact"]["rmse"] > 3 * r["free"]["rmse"], (r["contact"]["rmse"], r["free"]["rmse"])
    p = MODELS["in"].predict(HELD.state, HELD.action, HELD.dt).probs_marginal()[:, :W.n_balls]
    ev = HELD.events[:, :W.n_balls]
    p_hit = 1 - p[..., 0]
    on_contact, on_free = p_hit[ev > 0].mean(), p_hit[ev == 0].mean()
    assert on_contact > on_free + 0.1, (on_contact, on_free)
    ri = M.one_step(MODELS["in"], HELD, CI, DI)
    print(f"  B. contact n={r['contact']['n']} free n={r['free']['n']}; CV rmse contact "
          f"{r['contact']['rmse']:.2f} vs free {r['free']['rmse']:.2f}; IN P(contact) "
          f"{on_contact:.2f} on contacts vs {on_free:.2f} on free; IN contact "
          f"[{fmt(ri['contact'])}] free [{fmt(ri['free'])}]")


# --------------------------------------------------------------------- C

def _no_contact_possible(batch, dt=None, margin=0.05):
    """(B, n_balls) True where ball i cannot reach a wall or another ball
    within the step — decided from the INPUT state only, never the outcome."""
    nb, r, sz = W.n_balls, W.radius, W.size
    p, v = batch.state.pos[:, :nb], batch.state.vel[:, :nb]
    d = (batch.dt if dt is None else np.full(len(batch), float(dt)))[:, None]
    acc = W.gravity + W.force_scale * np.sqrt(2) / W.mass_range[0]
    reach = np.linalg.norm(v, axis=-1) * d + 0.5 * acc * d ** 2 + margin
    wall = np.minimum(np.minimum(p[..., 0], sz - p[..., 0]),
                      np.minimum(p[..., 1], sz - p[..., 1])) - r
    gap = np.linalg.norm(p[:, :, None] - p[:, None], axis=-1) - 2 * r \
        - reach[:, :, None] - reach[:, None, :]
    gap[:, np.arange(nb), np.arange(nb)] = np.inf
    return (wall > reach) & (gap.min(-1) > 0)


def test_duration_sensitivity():
    m, b, nb = MODELS["in"], HELD, W.n_balls
    vy = [4 * i + 3 for i in range(nb)]

    def g_of(dvy, dt, sel):
        dt = np.broadcast_to(dt[:, None], dvy.shape)
        return -float(np.sum(dvy[sel] * dt[sel]) / np.sum(dt[sel] ** 2))

    sel = _no_contact_possible(b)
    sel[:, 0] = False                                      # ball 0 is pushed
    assert sel.sum() >= 200, sel.sum()
    true_dv = (b.next_state.continuous() - b.state.continuous())[:, vy]
    g_true = g_of(true_dv, b.dt, sel)
    assert abs(g_true - W.gravity) < 1e-6, g_true          # the selector is clean
    dv = (m.predict(b.state, b.action, b.dt).mean() - b.state.continuous())[:, vy]
    g_hat = g_of(dv, b.dt, sel)
    assert abs(g_hat - W.gravity) < 0.25 * W.gravity, g_hat
    s9 = _no_contact_possible(b, dt=0.09)
    s9[:, 0] = False
    n = len(b.dt)
    lo = (m.predict(b.state, b.action, np.full(n, 0.03)).mean() - b.state.continuous())[:, vy]
    hi = (m.predict(b.state, b.action, np.full(n, 0.09)).mean() - b.state.continuous())[:, vy]
    ratio = float(hi[s9].mean() / lo[s9].mean())
    assert 1.5 < ratio < 4.5, ratio
    print(f"  C. IN recovers gravity from variable-dt data on {int(sel.sum())} held-out "
          f"(state, ball) pairs where no contact is possible: g_hat={g_hat:.2f} (true "
          f"{W.gravity}; truth on the same pairs {g_true:.2f}); "
          f"dv_y(dt=0.09)/dv_y(dt=0.03)={ratio:.2f} on {int(s9.sum())} pairs (physics: 3)")


# --------------------------------------------------------------------- D

def test_reset_exclusion():
    s = DATA["train"]
    H = 10
    st = s.windows(H)
    refused = s.refused_windows(H)
    assert refused > 0, "no window was refused: the test cannot bite"
    for i in st:
        assert s.episode_of(i) == s.episode_of(i + H), f"window {i} spans episodes"
    n_eps = len(DATA["train_episodes"])
    assert len(TRAIN) == len(s) - n_eps, "a reset splice became a transition"
    bad = int(np.flatnonzero(~s.pair_valid)[0])
    assert any("reset splice" in p for p in s.pair_problems[bad]), s.pair_problems[bad]
    try:
        s.window_batch([bad - 2], H)
        raise AssertionError("refused window was served")
    except ContractError:
        pass
    hz = M.horizons(MODELS["cv"], s, H, CI, DI, max_windows=50)
    assert hz["refused"] == refused
    # The cost of not refusing: the splice "transition" vs real ones.
    s0, a, d, tg = s.window_batch(np.flatnonzero(~s.pair_valid), 1, allow_invalid=True)
    err_splice = np.abs(MODELS["cv"].predict(s0, a[0], d[0]).mean()[:, CI]
                        - tg[0]["continuous"][:, CI]).mean()
    err_real = np.abs(MODELS["cv"].predict(TRAIN.state, TRAIN.action, TRAIN.dt).mean()[:, CI]
                      - TRAIN.next_state.continuous()[:, CI]).mean()
    assert err_splice > 3 * err_real, (err_splice, err_real)
    print(f"  D. {refused} of {len(st) + refused} naive {H}-step windows refused (reset "
          f"splices); accepted windows each inside one episode; splice error "
          f"{err_splice:.2f} vs real {err_real:.2f} — refusing them matters")


# --------------------------------------------------------------------- E

def _equiv_err(m, b, c):
    d0 = m.predict(b.state, b.action, b.dt)
    d1 = m.predict(b.state.translate(c), b.action, b.dt)
    mu0, mu1 = d0.mean().reshape(len(b), -1, 4), d1.mean().reshape(len(b), -1, 4)
    pos = np.abs(mu1[..., :2] - mu0[..., :2] - c).max()
    vel = np.abs(mu1[..., 2:] - mu0[..., 2:]).max()
    pr = np.abs(d1.probs_marginal() - d0.probs_marginal()).max() if m.layout.G else 0.0
    return pos, vel, pr


def test_translation_symmetry():
    b = HELD.subset(np.arange(200))
    c = np.array([3.7, -2.2])
    m = MODELS["in"]
    assert m.assumptions["translation_invariant"] is True
    for cc in (c, np.array([50.0, 50.0])):
        pos, vel, pr = _equiv_err(m, b, cc)
        assert pos < 2e-3 and vel < 2e-3 and pr < 1e-3, (cc, pos, vel, pr)
    nti = M.InteractionNetwork(*ARGS, hidden=48, seed=0, translation_invariant=False,
                               mechanism_id="relational.in_no_symmetry")
    nti.fit(TRAIN, epochs=15, lr=3e-3, batch_size=64)
    e_nti = _equiv_err(nti, b, c)
    e_mlp = _equiv_err(MODELS["mlp"], b, c)
    e_rssm = _equiv_err(MODELS["rssm"], b, c)
    for name, e in (("IN without declaration", e_nti), ("MLP", e_mlp)):
        assert max(e[:2]) > 0.05, f"{name} is equivariant without declaring it: {e}"
    # The RSSM is NOT asserted either way. Before the static-dim fix it read
    # ~1e-4 (near-equivariant for a bad reason: its one-step latent barely
    # read the state); since the fix it reads the state and is clearly NOT
    # equivariant (~0.6/3.6 at 20 epochs). A model that ignores its input is
    # trivially "equivariant"; equivariance is evidence of structure only in a
    # model shown to use the input.
    print(f"  E. declared-TI IN: shift by {c.tolist()} and [50,50] moves predicted pos by "
          f"exactly c (err {_equiv_err(m, b, c)[0]:.1e}), vel/probs unchanged; WITHOUT the "
          f"declaration err pos/vel = {e_nti[0]:.2f}/{e_nti[1]:.2f}, MLP "
          f"{e_mlp[0]:.2f}/{e_mlp[1]:.2f} (not free); RSSM {e_rssm[0]:.4f}/{e_rssm[1]:.4f} "
          f"(reported, not credited)")


# --------------------------------------------------------------------- F

def _law_states(batch, after_cont):
    out = []
    for i in range(len(batch)):
        att = batch.state.attrs[i]
        nxt = after_cont[i].reshape(-1, 4)
        out.append({"boundary": "box", "pos": nxt[:, :2], "attrs": att,
                    "before": {"pos": batch.state.pos[i], "vel": batch.state.vel[i]},
                    "after": {"pos": nxt[:, :2], "vel": nxt[:, 2:]},
                    "events": batch.events[i], "action": batch.action[i]})
    return out


def test_law_violations():
    reg = ConstraintRegistry()
    reg.register(M.box_containment_constraint(W.size))
    reg.register(M.free_flight_energy_constraint(W.gravity))
    b = HELD
    truth = M.law_diagnostics(reg, _law_states(b, b.next_state.continuous()))
    assert truth["box_containment"]["violated"] == 0, truth
    assert truth["free_flight_energy"]["violated"] == 0, truth
    nb = W.n_balls
    expect_inapp = int(np.sum(np.all((b.events[:, 1:nb] > 0), 1)))   # ball 0 always pushed
    assert truth["free_flight_energy"]["inapplicable"] == expect_inapp, \
        (truth["free_flight_energy"], expect_inapp)
    res = {}
    for name in ("cv", "in", "mlp"):
        mu = MODELS[name].predict(b.state, b.action, b.dt).mean()
        res[name] = M.law_diagnostics(reg, _law_states(b, mu))
    assert res["cv"]["free_flight_energy"]["violated"] > 0, res["cv"]
    assert res["cv"]["box_containment"]["violated"] > 0, res["cv"]
    show = {n: (r["box_containment"]["violated"], r["free_flight_energy"]["violated"])
            for n, r in res.items()}
    print(f"  F. truth: 0 violations of either law, energy inapplicable on exactly "
          f"{expect_inapp} contact steps; (containment, energy) violations of "
          f"{len(b)} predictions: {show}")


# --------------------------------------------------------------------- G

_BIG = {}


def _big(noise):
    """2 balls in a 40 m box, 20-step episodes: contacts are rare. Same seeds
    for every noise level, so the episodes (and split) are identical."""
    if noise not in _BIG:
        wn = M.BoxWorld(n_balls=2, size=40.0, obs_noise=noise)
        _BIG[noise] = (wn, M.make_streams(wn, 80, 20, seed=3))
    return _BIG[noise]


def test_stochastic_observations():
    # A big box: contacts are rare, so free-flight error is not swamped by
    # contact variance and observation noise is a visible share of it.
    out = {}
    for noise in (0.0, 0.2):
        wn, dn = _big(noise)
        args = (wn.n_entities, 2, 5, 2, M.EVENT_CLASSES)
        m = M.MLPMechanism(*args, hidden=64, seed=0, mechanism_id=f"mlp_noise{noise}")
        m.fit(dn["train"].transitions(), **FIT_G)
        hen = dn["heldout"].transitions()
        f = hen.subset(np.flatnonzero(_free(hen, 2)))
        ci = wn.ball_cont_index()
        out[noise] = (m.predict(f.state, f.action, f.dt).variance()[:, ci].mean(),
                      M.one_step(m, f, ci, wn.ball_disc_index(), split_contact=False)["all"])
    (v_clean, _), (v_noisy, r) = out[0.0], out[0.2]
    assert v_noisy > 1.5 * v_clean, (v_noisy, v_clean)        # measured 2.4x
    assert v_noisy > 0.5 * 0.2 ** 2, v_noisy
    assert 0.82 <= r["cov90"] <= 0.97, r["cov90"]
    print(f"  G. obs noise sd=0.2: predictive var {v_noisy:.4f} vs noiseless-trained "
          f"{v_clean:.4f} on the same held-out states; noisy held-out free-flight "
          f"coverage 50/90 = {r['cov50']:.2f}/{r['cov90']:.2f}")


# --------------------------------------------------------------------- H

def test_long_horizon_drift():
    H = 10
    lines = []
    for name in ("cv", "in", "mlp", "rssm"):
        m = MODELS[name]
        hz = M.horizons(m, DATA["heldout"], H, CI, DI if m.layout.G else None,
                        max_windows=150)
        r = [p["rmse"] for p in hz["per_h"]]
        assert len(r) == H and all(np.isfinite(r))
        assert r[-1] > r[0] and hz["drift"] > 1.0, (name, r)
        lines.append(f"{name} rmse h1={r[0]:.2f} h5={r[4]:.2f} h10={r[-1]:.2f} "
                     f"drift={hz['drift']:.1f}x")
    m = MODELS["in"]
    hs = M.horizons(m, DATA["heldout"], H, CI, DI, mode="sample", n_samples=12,
                    max_windows=100)
    hm = M.horizons(m, DATA["heldout"], H, CI, DI, mode="mean", max_windows=100)
    vs = hs["per_h"][-1]["var_within"] + hs["per_h"][-1]["var_between"]
    vm = hm["per_h"][-1]["var_within"] + hm["per_h"][-1]["var_between"]
    assert vs > vm, (vs, vm)
    print(f"  H. {hz['windows']} windows scored, {hz['refused']} refused; " + "; ".join(lines)
          + f"; IN h10 predictive var sample={vs:.2f} > mean-mode={vm:.2f} "
          f"(cov90 {hs['per_h'][-1]['cov90']:.2f} vs {hm['per_h'][-1]['cov90']:.2f})")


# --------------------------------------------------------------------- I

def test_rssm_comparability():
    m = MODELS["rssm"]
    b = HELD.subset([5])
    pred, dec = m.native_prediction(b.state, b.action, b.dt, prediction_id="p-rssm",
                                    belief_id="bel-0", snapshot_id="snap-test")
    direct = m.predict(b.state, b.action, b.dt)
    assert np.allclose(dec.mean(), direct.mean(), atol=1e-5), \
        np.abs(dec.mean() - direct.mean()).max()
    assert np.allclose(dec.variance(), direct.variance(), rtol=1e-4)
    assert pred.model_versions == {m.mechanism_id: str(m.version)}
    assert m.layout.G == 0
    r = M.one_step(m, HELD, CI, None)["all"]
    assert "nll_event" not in r and "rmse" in r
    print(f"  I. RSSM via adapters.RSSMPredictor == batched path (max diff "
          f"{np.abs(dec.mean() - direct.mean()).max():.1e}); events INAPPLICABLE "
          f"(no event head) -> no event score, not a zero")


# --------------------------------------------------------------------- J

def test_records_and_residual():
    m = MODELS["in"]
    reg = SnapshotRegistry()
    snap = reg.publish(m.mechanism_id, m, meta={"version": m.version})
    win = DATA["heldout"].windows(3)[:1]
    s0, acts, dts, _ = DATA["heldout"].window_batch(win, 3)
    outs = m.rollout(s0, acts, dts)
    p = M.prediction_record(m, outs, acts, dts, prediction_id="pred-1",
                            source_belief="obs:heldout", snapshot=snap,
                            action_spec_id="box2d.force", t_wall=1.0)
    assert p.model_versions == {m.mechanism_id: 1} and p.horizon == 3
    assert p.snapshot_id == f"{snap.name}#{snap.snapshot_id}"
    assert p.payload["snapshots"][snap.name]["content_hash"] == snap.content_hash
    back = from_json(to_json(p))
    assert back == p
    for a, b in zip(M.outcomes_from_prediction(back), outs):
        assert np.allclose(a.mean(), b.mean()) and np.allclose(a.probs_marginal(),
                                                               b.probs_marginal())
    try:
        M.prediction_record(M.MLPMechanism(*ARGS, mechanism_id="fresh"), outs, acts, dts,
                            prediction_id="x", source_belief="b", snapshot="s",
                            action_spec_id="a")
        raise AssertionError("version-0 model's prediction recorded")
    except ContractError:
        pass
    wb, db = _big(0.0)
    tb, hb = db["train"].transitions(), db["heldout"].transitions()
    base = M.ConstantVelocityMechanism(wb.n_entities, 2, 5, 2, M.EVENT_CLASSES,
                                       mechanism_id="cv_base")
    base.fit(tb)
    res = M.ResidualMechanism(base, hidden=64, seed=0)
    res.fit(tb, epochs=40, lr=3e-3, batch_size=64)
    f = hb.subset(np.flatnonzero(_free(hb, wb.n_balls)))
    u = res.unexplained(f)
    vy = [4 * b + 3 for b in range(1, wb.n_balls)]
    corr = u["mean_correction"][vy].mean()
    expect = -wb.gravity * f.dt.mean()
    assert abs(corr - expect) < 0.2 * abs(expect), (corr, expect)
    assert u["explained_fraction"] > 0.5, u["explained_fraction"]
    assert res.model_versions() == {res.mechanism_id: 1, "cv_base": 1}
    base.fit(tb)
    try:
        res.predict(f.state, f.action, f.dt)
        raise AssertionError("stale residual used after base refit")
    except ContractError:
        pass
    res.fit(tb, epochs=5)
    res.predict(f.state, f.action, f.dt)
    print(f"  J. Prediction (snapshot {p.snapshot_id}, horizon 3) round-trips and decodes; "
          f"version-0 refused; residual on constant velocity learns gravity: mean "
          f"dv_y correction {corr:.3f} (expect {expect:.3f}), explains "
          f"{u['explained_fraction']:.0%} of free-flight error; stale after base refit, "
          f"recovers on refit")


# --------------------------------------------------------------------- K

def test_static_dims():
    jumps = []
    for ds in (3, 7, 11):
        tr = M.make_streams(W, 16, 40, seed=ds)["train"].transitions()
        rep = M.InteractionNetwork(*ARGS, hidden=48, seed=ds).fit(
            tr, epochs=10, lr=1e-3, batch_size=64)
        h = np.asarray(rep["loss_hist"])
        jumps.append(float(np.diff(h).max()))
        assert jumps[-1] < 1.0, (ds, np.round(h, 1))
        assert rep["static_dims"] == 4 * (W.n_entities - W.n_balls), rep["static_dims"]
    walls = np.arange(4 * W.n_balls, 4 * W.n_entities)
    for k in ("in", "mlp", "rssm"):
        st = MODELS[k].module.static.numpy()
        assert np.array_equal(np.flatnonzero(st), walls), (k, np.flatnonzero(st))
    m = MODELS["in"]
    d = m.predict(HELD.state, HELD.action, HELD.dt)
    cur = HELD.state.continuous()
    assert np.array_equal(d.mean()[:, walls], cur[:, walls])
    assert np.allclose(d.vars[0][:, walls], 1e-4)
    assert not np.allclose(d.mean()[:, CI], cur[:, CI])
    # escape path: one wall coordinate drifts in new data -> it is learned
    sub = TRAIN.subset(np.arange(min(400, len(TRAIN))))
    e = M.MLPMechanism(*ARGS, hidden=32, seed=1, mechanism_id="k.escape")
    e.fit(sub, epochs=2)
    assert walls[0] in e.static_dims
    npos = np.array(sub.next_state.pos, dtype=np.float64, copy=True)
    npos[:, W.n_balls, 0] += 0.05
    nxt = M.EntityBatch(npos, sub.next_state.vel, sub.next_state.attrs, sub.next_state.frame)
    moved = M.TransitionBatch(sub.state, sub.action, sub.dt, nxt, sub.events)
    e.fit(moved, epochs=30, lr=3e-3)
    assert walls[0] not in e.static_dims and set(e.static_dims) == set(walls[1:])
    dm = e.predict(sub.state, sub.action, sub.dt).mean()[:, walls[0]] - \
        sub.state.continuous()[:, walls[0]]
    assert abs(dm.mean() - 0.05) < 0.02, dm.mean()
    # a declared static dim that moves is refused
    dcl = M.MLPMechanism(*ARGS, hidden=16, seed=2, mechanism_id="k.declared")
    dcl.declare_static([0])                              # ball 0 pos x: it moves
    try:
        dcl.fit(sub, epochs=1)
        raise AssertionError("declared-static dim that moves was accepted")
    except ContractError:
        pass
    dcl.undeclare_static([0])                            # the refusal's escape path
    dcl.fit(sub, epochs=1)
    assert 0 not in dcl.static_dims
    print(f"  K. static dims: IN max per-epoch loss rise {max(jumps):+.2f} nats on 3 small "
          f"datasets (was +16..+76 before the fix); mask = the {len(walls)} wall dims for "
          f"IN/MLP/RSSM; walls predicted bit-exact identity; a moving wall dim is "
          f"re-learned (mean shift {dm.mean():.3f}, true 0.05); declared-static "
          f"violation refused")


if __name__ == "__main__":
    test_heldout_identical_data()
    test_contact_vs_free()
    test_duration_sensitivity()
    test_reset_exclusion()
    test_translation_symmetry()
    test_law_violations()
    test_stochastic_observations()
    test_long_horizon_drift()
    test_rssm_comparability()
    test_records_and_residual()
    test_static_dims()
    print(f"  ({time.time() - T0:.1f}s)")
    print("[foundation-mechanisms] ALL PASS")
