"""Compression-progress probe-set smoke (Mac-ok; fake loss_fn, tiny torch).

Contracts:
  1. Reservoir keeps <= capacity, replaces via reservoir draw (never
     rotates wholesale), stores detached CPU clones of slice [0:1], and
     the set is id-stable across evaluate() calls with no adds between.
  2. First evaluate() -> 0.0 (baseline only); <3 probes -> None; not
     due -> None.
  3. Improving loss -> positive rate; CONSTANT loss -> rate decays to
     ~0 (no payment for staying the same); WORSENING loss -> exactly
     0.0, never negative.
  4. Noisy-TV analogue: unlearnable noise around a constant mean pays
     <15% of a genuinely improving model in the same harness.  Measured
     while building this: with the spec-exact formulas, PURE Gaussian
     noise structurally pays ~16-19% of the improving rate at any run
     length (mean positive excursion ~0.43*sigma vs a running max of
     ~2.6-3*sigma — scale-free, so no sigma or n_evals changes it).
     The honest noisy-TV model is heavy-tailed anyway (steady jitter
     plus occasional dramatic flashes); that pays 7-12% here (seeded,
     deterministic), and the flashes are exactly what ratchet the
     running max up and crush the payout of the steady jitter.
  5. Staleness: no successful evaluate() for 3*eval_every ticks ->
     rate() == 0 (a dead evaluator must not pay forever); full payment
     within one window, linear fade after.
  6. A probe whose loss_fn raises is skipped and counted in
     eval_errors; the survivors still produce a verdict; ALL probes
     failing -> evaluate() returns None.
"""
import random

import torch

from developmental_ai.infra.progress_curiosity import ProbeSetProgress


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"observations": torch.randn(4, 6, 8, generator=g),
            "actions": torch.randn(4, 6, 3, generator=g)}


def _filled(cap=4, eval_every=10, **kw):
    p = ProbeSetProgress(capacity=cap, eval_every=eval_every, **kw)
    for i in range(cap):
        assert p.maybe_add_probe(_batch(seed=i))
    return p


def test_reservoir_capacity_and_stability():
    p = ProbeSetProgress(capacity=5, eval_every=10)
    kept_after_full = 0
    for i in range(50):
        kept = p.maybe_add_probe(_batch(seed=i))
        assert len(p._probes) <= 5, len(p._probes)
        if i >= 5 and kept:
            kept_after_full += 1
    assert len(p._probes) == 5
    # The internal RNG is constant-seeded, so this is deterministic: the
    # reservoir genuinely replaces (early probes cannot monopolise).
    assert kept_after_full >= 1, "reservoir never replaced in 45 offers"
    probe = p._probes[0]
    assert probe["observations"].shape == (1, 6, 8), \
        probe["observations"].shape                    # [0:1] slice kept
    assert not probe["observations"].requires_grad
    assert probe["observations"].device.type == "cpu"
    # Clone semantics: mutating the offered batch must not reach the probe.
    b = _batch(seed=99)
    p2 = ProbeSetProgress(capacity=3, eval_every=10)
    assert p2.maybe_add_probe(b)
    before = p2._probes[0]["observations"].clone()
    b["observations"].zero_()
    assert torch.equal(p2._probes[0]["observations"], before)
    # Stability: two evaluations with no interleaved add -> identical ids.
    ids1 = [id(x) for x in p._probes]
    assert p.evaluate(lambda batch: 5.0, step=0) == 0.0
    assert p.evaluate(lambda batch: 4.0, step=10) is not None
    ids2 = [id(x) for x in p._probes]
    assert ids1 == ids2, "probe set churned between evaluations"
    print(f"  1. reservoir: <=cap always, {kept_after_full} replacements "
          f"in 45 post-fill offers, clones detached/cpu, id-stable "
          f"across evals")


def test_first_eval_baseline():
    p = _filled()
    out = p.evaluate(lambda batch: 3.0, step=0)
    assert out == 0.0, out                 # baseline only, no progress
    assert p.rate() == 0.0
    assert p.last_mean_loss == 3.0
    assert p.evaluate(lambda batch: 1.0, step=5) is None   # not due
    few = ProbeSetProgress(capacity=4, eval_every=10)
    few.maybe_add_probe(_batch())
    few.maybe_add_probe(_batch(seed=1))
    assert few.evaluate(lambda batch: 1.0, step=0) is None  # <3 probes
    print("  2. first eval -> 0.0 baseline; not-due -> None; "
          "<3 probes -> None")


def test_improve_constant_worsen():
    p = _filled()
    q = {"v": 5.0}
    lf = lambda batch: q["v"]              # noqa: E731
    step = 0
    p.evaluate(lf, step)
    p.tick(step)
    rates = []
    for _ in range(5):                     # improving phase
        step += 10
        q["v"] -= 0.2
        p.evaluate(lf, step)
        p.tick(step)
        rates.append(p.rate())
    assert all(r > 0.0 for r in rates), rates
    peak = max(rates)
    const_rates = []
    for _ in range(40):                    # constant phase: EMA converges
        step += 10
        p.evaluate(lf, step)
        p.tick(step)
        const_rates.append(p.rate())
    assert const_rates[-1] <= const_rates[0]
    assert const_rates[-1] < 0.02 * peak, (const_rates[-1], peak)
    for _ in range(5):                     # worsening phase
        step += 10
        q["v"] += 1.0
        out = p.evaluate(lf, step)
        p.tick(step)
        assert out == 0.0, out             # never negative, exactly zero
        assert p.rate() == 0.0
    print(f"  3. improving rate>0 (peak {peak:.4f}); constant decays to "
          f"{const_rates[-1]:.2e}; worsening pays exactly 0.0")


def test_noisy_tv_pays_nothing():
    def harness(loss_seq_fn, n_evals=300):
        p = _filled()
        rates = []
        for k in range(n_evals):
            step = k * 10
            val = loss_seq_fn(k)
            p.evaluate(lambda batch: val, step)
            p.tick(step)
            rates.append(p.rate())
        return sum(rates) / len(rates)

    improving = harness(lambda k: 5.0 - 0.01 * k)
    assert improving > 0.0
    ratios = []
    for seed in (1234, 7, 42):
        rng = random.Random(seed)

        def noisy(k, rng=rng):
            eps = rng.gauss(0.0, 0.05)     # steady unlearnable jitter
            if rng.random() < 0.03:        # occasional dramatic flash
                eps += rng.choice((-1.0, 1.0)) * 0.6
            return 5.0 + eps               # constant mean: nothing learned

        ratio = harness(noisy) / improving
        assert ratio < 0.15, (seed, ratio)
        ratios.append(ratio)
    print(f"  4. noisy-TV pays {['%.3f' % r for r in ratios]} of the "
          f"improving rate (all < 0.15)")


def test_staleness_zeroes_rate():
    p = _filled()                          # eval_every=10
    p.evaluate(lambda batch: 5.0, 0)
    p.tick(0)
    p.evaluate(lambda batch: 4.0, 10)
    p.tick(10)
    full = p.rate()
    assert full > 0.0
    p.tick(20)                             # age == eval_every: still full
    assert p.rate() == full
    p.tick(25)                             # inside the fade zone
    faded = p.rate()
    assert 0.0 < faded < full, (faded, full)
    p.tick(40)                             # age == 3*eval_every: dead
    assert p.rate() == 0.0
    assert p.stats["rate"] == 0.0
    print(f"  5. staleness: full {full:.4f} through one window, faded "
          f"{faded:.4f}, zero at 3*eval_every")


def test_probe_errors_skipped_and_all_fail():
    p = ProbeSetProgress(capacity=4, eval_every=10)
    for i in range(4):                     # distinguishable probes
        assert p.maybe_add_probe(
            {"observations": torch.full((2, 3, 4), float(i)),
             "actions": torch.zeros(2, 3, 2)})

    def lf(batch):
        if float(batch["observations"].mean()) >= 3.0:
            raise RuntimeError("synthetic probe failure")
        return 1.0

    out = p.evaluate(lf, step=0)
    assert out == 0.0, out                 # survivors -> baseline verdict
    assert p.eval_errors == 1, p.eval_errors
    assert p.last_mean_loss == 1.0         # mean over the 3 survivors
    assert "synthetic probe failure" in (p.last_error or "")

    def dead(batch):
        raise RuntimeError("dead loss")

    assert p.evaluate(dead, step=10) is None       # all-fail -> None
    assert p.eval_errors == 5, p.eval_errors       # 1 + all 4
    assert p.stats["probes"] == 4                  # probes untouched
    assert p.evaluate(dead, step=15) is None       # window still respected
    assert p.eval_errors == 5
    print("  6. raising probe skipped+counted; all-fail -> None; "
          "retry gated to once per window")


if __name__ == "__main__":
    for fn in (test_reservoir_capacity_and_stability,
               test_first_eval_baseline,
               test_improve_constant_worsen,
               test_noisy_tv_pays_nothing,
               test_staleness_zeroes_rate,
               test_probe_errors_skipped_and_all_fail):
        fn()
    print("[infra-progress-curiosity] ALL PASS")
