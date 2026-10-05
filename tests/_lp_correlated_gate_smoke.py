"""Learning-progress significance gate under AUTOCORRELATED errors (Mac-ok).

    PYTHONPATH=. python tests/_lp_correlated_gate_smoke.py

THE LIVE INCIDENT (2026-10, 13 h shadow run, discrete28 Treechop on Paper)
    LP pays `older-half mean - recent-half mean` of a bucket's error history
    when the drop clears `0.5 * std(history)` (and the 5% absolute floor).
    That test is only valid if the history entries are INDEPENDENT. The two
    smallest non-noop actions — the 2-degree micro-turn (action 17) and the
    120-tick attack hold (action 27) — leave the 4x4-pooled frame almost
    unchanged, so the agent stays in ONE prototype bucket for many steps and
    its 30-entry history fills with back-to-back, strongly AUTOCORRELATED
    errors from a model that has not changed. A random walk has a large
    half-mean difference relative to its own std, so the gate passes chance:
        frozen model, iid errors           ->  9% of transitions pay
        frozen model, AR(1) rho 0.5..0.95  -> 22-39% pay
        live: action 17 passed 24-28%, action 27 27%, everything else ~7%
    and the LP income of action 17 GREW 0.03 -> 0.26/step over 13 hours —
    fabricated learning progress (improvement plan §7.4, the first regression
    case). Noop pays exactly 0 through the action-attribution gate, so the
    farm settled on the smallest action that is not noop.

THE FIX THIS PINS (curiosity/learning_progress.py)
    1. VISIT COLLAPSE: consecutive same-bucket transitions of one stream are
       ONE visit. The history gets one entry per visit (the visit mean) and
       LP is scored once, on ENTRY — so a dwell cannot be paid N times for
       one piece of evidence, and "error fell over repeated visits" means
       visits again.
    2. A VALID TEST: the drop is compared with the standard error of the
       half-mean difference (pooled within-half variance), inflated by
       (1+r)/(1-r) for the history's own lag-1 autocorrelation r, against a
       one-sided ~1% quantile with the effective degrees of freedom.

CONTRACTS (all through the REAL LearningProgressCuriosity; its encoder and
forward model are replaced by identity / no-change so the per-transition
error is exactly the value fed in — the bucketing, history, gate and
normaliser are the production code)
    A  FROZEN model, iid errors, hopping buckets: per-step pay rate <= 2%.
    B  FROZEN model, AR(1) rho in {0.5, 0.8, 0.95}, the micro-turn pattern
       (dwell ~10 steps in one bucket, brief excursion, back): <= 2%.
    C  FROZEN model, AR(1) rho in {0.5, 0.8, 0.95}, rapid A/B oscillation
       (every step is a bucket entry, so collapse alone cannot help): <= 2%.
    D  PINNED in one bucket forever (the pure dwell farm): pays nothing.
    E  GENUINE LEARNING (error mean decays 1.0 -> 0.3) still pays — iid
       hopping AND the correlated micro-turn pattern — and the payments
       concentrate in the learning phase, not after it.
    F  MASTERED bucket (flat error, 1% jitter) pays exactly 0.
    G  NOOP pays exactly 0 even on a genuinely improving bucket, while a
       non-noop action on the same schedule is paid (attribution gate).
    H  READ-ONLY calls (imagination) change no visit/history state.
    I  the legacy gate is retained behind lp_sig_z=0 / lp_visit_collapse=
       False as a revert AND a regression witness: on contract B it must
       still pass > 10% — if it does not, this harness lost its teeth.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

from developmental_ai.curiosity.learning_progress import (
    LearningProgressCuriosity)

D = 4           # obs dim (identity encoder -> feature dim)
A = 3           # discrete actions; 0 = noop
SIGMA = 0.3     # log-normal error spread (CV ~0.3)


class _NoChange(nn.Module):
    """Predicts 'nothing changes' for every action -> error = mse(next-obs)."""

    def forward(self, f, a):
        return f


class _ActionAccounts(nn.Module):
    """Non-noop actions predict a fixed +C shift; noop predicts no change.
    So a non-noop transition's error is attributable to the action (gate>0)
    and a noop's err_a == err_null exactly (gate == 0)."""
    C = 0.5

    def forward(self, f, a):
        return f + (1.0 - a[:, :1]) * self.C


def _make(legacy=False, ac=False, model=None):
    kw = dict(obs_dim=D, action_dim=A, feature_dim=D, hidden_dim=8,
              discrete_actions=True, lp_history=30, lp_min_samples=4,
              lp_abs_frac=0.05, action_conditional=ac, null_action=0)
    if legacy:
        kw.update(lp_sig_z=0.0, lp_visit_collapse=False)
    try:
        lp = LearningProgressCuriosity(**kw)
    except TypeError:   # pre-fix code: no such knobs -> it IS the legacy gate
        kw.pop("lp_sig_z", None)
        kw.pop("lp_visit_collapse", None)
        lp = LearningProgressCuriosity(**kw)
    lp.encoder = nn.Identity()
    lp.forward_model = model if model is not None else _NoChange()
    lp._raw = []
    orig = lp._update_stats

    def _cap(r):   # capture the raw (pre-normalisation) LP the gate emitted
        lp._raw.append(r.detach().cpu().numpy().copy())
        orig(r)
    lp._update_stats = _cap
    return lp


def _obs(bucket):
    return np.full(D, 10.0 * bucket + 1.0, np.float32)


def _step(lp, buckets, errs, action=1, shift=0.0, update_state=True):
    """One batched call; row i is in `buckets[i]` with exact error errs[i]."""
    o = np.stack([_obs(b) for b in buckets])
    n = o + (shift + np.sqrt(np.asarray(errs, np.float32)))[:, None]
    a = torch.zeros(len(buckets), A)
    a[:, action] = 1.0
    r = lp.compute_intrinsic_reward(torch.from_numpy(o), a,
                                    torch.from_numpy(n.astype(np.float32)),
                                    update_state=update_state)
    return r.numpy()


def _ar1(rng, n, rho):
    z = np.empty(n)
    z[0] = rng.normal()
    s = np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + s * rng.normal()
    return np.exp(SIGMA * z - SIGMA * SIGMA / 2)


def _schedule(rng, n, pattern, base):
    if pattern == "hop":           # every step a different bucket (3-cycle)
        return [base + (t % 3) for t in range(n)]
    if pattern == "oscillate":     # A B A B ...
        return [base + (t % 2) for t in range(n)]
    if pattern == "pinned":
        return [base] * n
    if pattern == "microturn":     # dwell ~10 in bucket 0, 1-step excursion
        out = []
        while len(out) < n:
            out += [base] * int(rng.geometric(0.1))
            out.append(base + 1 + int(rng.integers(0, 2)))
        return out[:n]
    raise ValueError(pattern)


def _run(lp, pattern, rho, n=6000, seed=0, mean_fn=None, warm=300):
    """Two independent streams (rows) per call. Returns per-step raw LP,
    per-step error, and the step index, after `warm` steps."""
    rng = np.random.default_rng(seed)
    sch = [_schedule(rng, n, pattern, base) for base in (0, 10)]
    noise = [(_ar1(rng, n, rho) if rho > 0 else
              np.exp(SIGMA * rng.normal(size=n) - SIGMA * SIGMA / 2))
             for _ in range(2)]
    lp._raw.clear()
    errs = []
    for t in range(n):
        m = 1.0 if mean_fn is None else mean_fn(t)
        e = [m * noise[0][t], m * noise[1][t]]
        errs.append(e)
        _step(lp, [sch[0][t], sch[1][t]], e)
    raw = np.concatenate(lp._raw).reshape(n, 2)[warm:]
    return raw, np.asarray(errs)[warm:]


def _rate(raw):
    return float((raw > 0).mean())


def _pay(raw, errs):
    return float(raw.mean() / errs.mean())


FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    print("[lp-correlated-gate] A: frozen model, iid errors, hopping")
    raw, e = _run(_make(), "hop", 0.0, seed=1)
    rl, el = _run(_make(legacy=True), "hop", 0.0, seed=1)
    check(_rate(raw) <= 0.02,
          f"A1 iid hop pay rate {_rate(raw):.4f} <= 0.02 "
          f"(legacy {_rate(rl):.4f}); mean pay/err {_pay(raw, e):.5f} "
          f"(legacy {_pay(rl, el):.5f})")

    print("[lp-correlated-gate] B: frozen model, AR(1), micro-turn dwell")
    legacy_b = []
    for rho in (0.5, 0.8, 0.95):
        raw, e = _run(_make(), "microturn", rho, seed=2)
        rl, el = _run(_make(legacy=True), "microturn", rho, seed=2)
        legacy_b.append(_rate(rl))
        check(_rate(raw) <= 0.02 and _pay(raw, e) <= 0.005,
              f"B rho={rho} pay rate {_rate(raw):.4f} <= 0.02, pay/err "
              f"{_pay(raw, e):.5f} <= 0.005 (legacy {_rate(rl):.4f} / "
              f"{_pay(rl, el):.5f})")

    print("[lp-correlated-gate] C: frozen model, AR(1), A/B oscillation")
    for rho in (0.5, 0.8, 0.95):
        raw, e = _run(_make(), "oscillate", rho, seed=3)
        rl, _ = _run(_make(legacy=True), "oscillate", rho, seed=3)
        check(_rate(raw) <= 0.02,
              f"C rho={rho} pay rate {_rate(raw):.4f} <= 0.02 "
              f"(legacy {_rate(rl):.4f})")

    print("[lp-correlated-gate] D: pinned in one bucket")
    raw, e = _run(_make(), "pinned", 0.9, seed=4)
    rl, _ = _run(_make(legacy=True), "pinned", 0.9, seed=4)
    check(_rate(raw) == 0.0,
          f"D pinned pay rate {_rate(raw):.4f} == 0 (legacy {_rate(rl):.4f})")

    print("[lp-correlated-gate] E: genuine learning still pays")
    for pattern, rho, tau in (("hop", 0.0, 400.0),
                              ("microturn", 0.8, 1500.0)):
        def mean_fn(t, tau=tau):
            return 0.3 + 0.7 * np.exp(-t / tau)
        n = int(6 * tau)
        raw, e = _run(_make(), pattern, rho, n=n, seed=5, mean_fn=mean_fn,
                      warm=0)
        rl, _ = _run(_make(legacy=True), pattern, rho, n=n, seed=5,
                     mean_fn=mean_fn, warm=0)
        lp_phase = raw[: int(2 * tau)]
        late = raw[int(4 * tau):]
        entries_paid = int((lp_phase > 0).sum())
        check(entries_paid >= 10 and lp_phase.sum() > 5 * late.sum(),
              f"E {pattern} rho={rho}: {entries_paid} paid steps in the "
              f"learning phase (legacy {int((rl[: int(2 * tau)] > 0).sum())})"
              f", learning-phase pay {lp_phase.sum():.3f} > 5x late "
              f"{late.sum():.3f}")

    print("[lp-correlated-gate] F: mastered bucket")
    lp = _make()
    rng = np.random.default_rng(6)
    lp._raw.clear()
    for t in range(3000):
        _step(lp, [t % 3, 10 + t % 3],
              list(0.02 * (1.0 + 0.01 * rng.normal(size=2))))
    raw = np.concatenate(lp._raw)
    check(float(raw.max()) == 0.0, f"F mastered max raw LP {raw.max():.2e} == 0")

    print("[lp-correlated-gate] G: noop pays exactly 0 on a learning bucket")

    def mean_fn(t):
        return 0.3 + 0.7 * np.exp(-t / 400.0)
    paid = {}
    raw_pos = {}
    for act in (0, 1):
        lp = _make(ac=True, model=_ActionAccounts())
        rng = np.random.default_rng(7)
        tot = 0.0
        lp._raw.clear()
        for t in range(2400):
            e = mean_fn(t) * np.exp(SIGMA * rng.normal(size=2) - SIGMA ** 2 / 2)
            sh = 0.0 if act == 0 else _ActionAccounts.C
            r = _step(lp, [t % 3, 10 + t % 3], e, action=act, shift=sh)
            tot += float(np.abs(r).sum())
        paid[act] = tot
        raw_pos[act] = int((np.concatenate(lp._raw) > 0).sum())
    check(paid[0] == 0.0 and raw_pos[0] > 0,
          f"G noop paid {paid[0]} == 0 exactly although raw LP fired "
          f"{raw_pos[0]}x")
    check(paid[1] > 0.0, f"G non-noop on the same schedule paid {paid[1]:.3f}")

    print("[lp-correlated-gate] H: read-only calls are stateless")
    lp = _make()
    for t in range(200):
        _step(lp, [t % 3, 10], [1.0, 1.0])
    before = ({k: list(v) for k, v in lp._bucket_err.items()},
              {k: list(v) for k, v in getattr(lp, "_lp_runs", {}).items()})
    for t in range(50):
        _step(lp, [5, 10, t % 3], [0.5, 0.5, 0.5], update_state=False)
    after = ({k: list(v) for k, v in lp._bucket_err.items()},
             {k: list(v) for k, v in getattr(lp, "_lp_runs", {}).items()})
    check(before == after, "H read-only left histories and visit runs intact")

    print("[lp-correlated-gate] I: legacy gate is still the farm (witness)")
    check(min(legacy_b) > 0.10,
          f"I legacy micro-turn pay rates {[round(x, 3) for x in legacy_b]} "
          f"all > 0.10")

    if FAILS:
        print(f"[lp-correlated-gate] {len(FAILS)} FAILED")
        sys.exit(1)
    print("[lp-correlated-gate] ALL PASS")


if __name__ == "__main__":
    main()
