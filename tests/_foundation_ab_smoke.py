"""A/B harness contracts (plan §7.2; foundation/experiments/ab.py).

WHY THIS EXISTS

    CLAUDE.md §9: "The reward number has been wrong every single time."
    The project has read favourable numbers after the fact (96% of drive
    from staring at the sky, 77% of income from a menu) and called them
    progress. This harness only means something if it REFUSES the moves
    that produce those readings, so each contract below is written to catch
    the harness being permissive, not to confirm that it runs.

Contracts:
    A. TAMPERING IS REFUSED. A prereg edited after freezing (replace or
       hand-edited JSON), re-frozen after the results, or frozen "later"
       than the first recorded result gets NO verdict. An unfrozen prereg
       or changed seeds cannot even start a run.
    B. FEWER THAN 3 SEEDS -> "inconclusive (screening only)", even for an
       effect fifty standard deviations wide.
    C. A KNOWN-BETTER ARM IS ACCEPTED (and a known-worse one rejected).
    D. AN IDENTICAL ARM IS NOT ACCEPTED: over 20 null repeats the acceptance
       rate is <= 10%. And the small-sample CI expansion is load-bearing:
       at n=3 the raw percentile bootstrap accepts a null difference far
       more often than the expanded interval does (measured here).
    E. A RAISING ARM IS A RECORDED FAILURE, not a skipped run; a candidate
       that fails beyond the preregistered allowance is rejected; a budget
       overrun is recorded as budget_exceeded.
    F. REPORTS ARE WRITTEN and never overwritten: JSON + markdown per run,
       one index line each, negative results included; a written report
       re-derives the same verdict.
    G. THE SAME SPLIT FOR ALL ARMS: every arm sees the identical DataSplit
       per seed; a split with a leaked episode is refused.
    H. MISSING ABLATION WARNS LOUDLY; a named ablation that was not supplied
       is refused.
    I. EQUAL INTERACTIONS: a candidate that only "wins" by using twice the
       interactions is accepted on basis=final and NOT on equal_interactions.

Run: PYTHONPATH=. python tests/_foundation_ab_smoke.py   (< 60 s)
"""
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import warnings

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.foundation.experiments import ab
from developmental_ai.foundation.runtime.splits import SplitLeakError

SPLIT = ab.make_split("smoke:ab", [f"e{i}" for i in range(30)], 0.3)


def prereg(seeds=(0, 1, 2, 3, 4), **kw):
    base = dict(name="smoke", hypothesis="B beats A", primary_metric="score",
                direction="higher", baseline_arm="A", candidate_arm="B",
                ablation_arm="B_minus", seeds=tuple(seeds), basis="final",
                n_bootstrap=1000)
    base.update(kw)
    return ab.Preregistration(**base)


def noisy(effect, tag, noise=0.5, base_scale=3.0):
    """score = shared per-seed baseline + effect + arm-specific noise."""
    def arm(seed, split):
        shared = np.random.default_rng(seed).normal(scale=base_scale)
        own = np.random.default_rng([seed, tag]).normal(scale=noise)
        return {"score": float(shared + effect + own), "interactions": 100}
    return arm


def arms(effect_b=0.0, effect_ablation=0.0, **kw):
    return {"A": noisy(0.0, 1, **kw), "B": noisy(effect_b, 2, **kw),
            "B_minus": noisy(effect_ablation, 3, **kw)}


def run(p, a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ab.run_ab(p, a, splits=[SPLIT], measure_memory=False, **kw)


def expect_violation(fn, what):
    try:
        fn()
    except ab.PreregistrationViolation:
        return
    raise AssertionError(f"{what}: no PreregistrationViolation")


def test_tampering_refused(tmp):
    p = prereg().freeze()
    rep = run(p, arms(2.0), out_dir=tmp)
    assert ab.decide(p, rep.results)["verdict"] == "accept"
    expect_violation(lambda: ab.decide(dataclasses.replace(p, delta=5.0), rep.results),
                     "edited after freezing")
    refrozen = dataclasses.replace(p, delta=5.0, frozen_at_ns=None,
                                   frozen_hash=None).freeze()
    expect_violation(lambda: ab.decide(refrozen, rep.results), "re-frozen after results")
    with open(rep.json_path) as f:
        d = json.load(f)
    d["prereg"]["delta"] = 0.0001
    d["prereg"]["acceptance_rule"] = "edited"
    expect_violation(lambda: ab.decide(ab.Preregistration.from_dict(d["prereg"]),
                                       ab.ABResults.from_dict(d["results"])),
                     "hand-edited JSON prereg")
    # Frozen after the first result: simulate with a clock that freezes the
    # prereg in the future relative to the run's own clock.
    late = prereg().freeze(clock_ns=lambda: 10 ** 30)
    expect_violation(lambda: run(late, arms(2.0)), "prereg frozen after results")
    expect_violation(lambda: run(prereg(), arms(2.0)), "unfrozen prereg")
    expect_violation(lambda: ab.run_ab(p, arms(2.0), seeds=(0, 1, 2, 3, 99),
                                       splits=[SPLIT]), "changed seeds")
    print("  A. edited / re-frozen / JSON-edited / late-frozen / unfrozen prereg "
          "and changed seeds all refused a verdict")


def test_screening_only():
    rep = run(prereg(seeds=(0, 1)).freeze(), arms(50.0, noise=0.01, base_scale=0.0))
    v = rep.verdict
    assert v["verdict"] == "inconclusive" and v["label"] == "inconclusive (screening only)", v
    assert v["comparisons"]["final"]["mean_diff"] > 49, "effect not even measured"
    print(f"  B. 2 seeds with a +{v['comparisons']['final']['mean_diff']:.1f} effect "
          f"-> '{v['label']}'")


def test_known_better_and_worse():
    p = prereg(delta=0.5).freeze()
    good = run(p, arms(3.0))
    assert good.verdict["verdict"] == "accept", good.verdict["reasons"]
    assert "ablation_arm" in good.verdict["secondary"], "ablation not compared"
    bad = run(p, arms(-3.0))
    assert bad.verdict["verdict"] == "reject", bad.verdict["reasons"]
    lower = prereg(direction="lower").freeze()
    assert run(lower, arms(3.0)).verdict["verdict"] == "reject", "direction ignored"
    c = good.verdict["comparisons"]["final"]
    print(f"  C. +3 effect accepted (diff {c['mean_diff']:.2f}, CI "
          f"[{c['ci_low']:.2f}, {c['ci_high']:.2f}]); -3 rejected; "
          f"'lower is better' flips it to reject")


def test_null_false_positive_rate():
    accepts, verdicts = 0, []
    for rep_i in range(20):
        seeds = tuple(range(100 * rep_i, 100 * rep_i + 5))
        r = run(prereg(seeds=seeds).freeze(), arms(0.0))
        verdicts.append(r.verdict["verdict"])
        accepts += r.verdict["verdict"] == "accept"
    rate = accepts / 20.0
    assert rate <= 0.10, f"null arm accepted {accepts}/20 times"
    # Is the small-sample expansion load-bearing? Measure both at n = 3.
    rng = np.random.default_rng(12345)
    raw = exp = 0
    trials = 400
    for t in range(trials):
        d = rng.normal(size=3)
        _, lo_r, _ = ab.paired_bootstrap_ci(d, 400, t, small_sample_correction=False)
        _, lo_e, _ = ab.paired_bootstrap_ci(d, 400, t, small_sample_correction=True)
        raw += lo_r > 0
        exp += lo_e > 0
    assert exp / trials <= 0.06, f"expanded CI false-accepts {exp}/{trials} at n=3"
    assert raw > 2 * exp, ("raw percentile bootstrap is not worse at n=3 — "
                           "then the expansion is unjustified")
    print(f"  D. identical arms accepted {accepts}/20 null repeats (<= 10%; "
          f"verdicts {sorted(set(verdicts))}); at n=3 the raw bootstrap "
          f"false-accepts {raw / trials:.1%} vs expanded {exp / trials:.1%}")


def test_failures_recorded():
    def flaky(seed, split):
        if seed == 1:
            raise RuntimeError("diverged at seed 1")
        return noisy(3.0, 2)(seed, split)
    a = arms(3.0)
    a["B"] = flaky
    rep = run(prereg().freeze(), a)
    fails = [r for r in rep.results.runs if r.status == "failed"]
    assert len(fails) == 1 and fails[0].arm == "B" and fails[0].seed == 1
    assert "diverged at seed 1" in fails[0].error
    assert len(rep.results.runs) == 15, "a failed run was dropped"
    assert rep.verdict["verdict"] == "reject", rep.verdict["reasons"]
    lenient = run(prereg(max_candidate_failure_fraction=0.25).freeze(), a)
    assert lenient.verdict["verdict"] == "accept", lenient.verdict["reasons"]
    assert lenient.verdict["comparisons"]["final"]["n_pairs"] == 4
    a2 = arms(3.0)
    a2["B"] = lambda s, sp: {"score": float("nan")}
    assert all(r.status == "failed" for r in run(prereg().freeze(), a2).results.runs
               if r.arm == "B"), "NaN metric accepted"
    slow = run(prereg(resource_budget={"max_interactions": 50}).freeze(), arms(3.0))
    assert all(r.status == "budget_exceeded" for r in slow.results.runs)
    assert slow.verdict["verdict"] == "reject"
    print("  E. raising arm recorded (seed 1, error kept, 15/15 runs present) -> "
          "reject at 0% allowance, accept on 4 pairs at 25%; NaN metric = failure; "
          "budget overrun = budget_exceeded")


def test_reports_written(tmp):
    p = prereg().freeze()
    r1 = run(p, arms(-3.0), out_dir=tmp)          # a NEGATIVE result
    r2 = run(p, arms(-3.0), out_dir=tmp)
    assert r1.json_path != r2.json_path, "report overwritten"
    for r in (r1, r2):
        assert os.path.exists(r.json_path) and os.path.exists(r.md_path)
        assert "Verdict: reject" in open(r.md_path).read()
    lines = open(os.path.join(tmp, "index.jsonl")).read().strip().splitlines()
    assert sum(json.loads(l)["verdict"] == "reject" for l in lines) >= 2
    p2, res2 = ab.load_report(r1.json_path)
    assert ab.decide(p2, res2)["verdict"] == r1.verdict["verdict"]
    print(f"  F. negative result written twice without overwrite, "
          f"{len(lines)} index lines, reloaded report re-derives '{r1.verdict['label']}'")


def test_same_split_for_all_arms():
    seen = []

    def spy(name, effect):
        inner = noisy(effect, hash(name) % 97)

        def arm(seed, split):
            seen.append((name, seed, split.fingerprint, split.dev, split.heldout))
            try:
                split.dev = ()
                raise AssertionError("arm could mutate the split")
            except dataclasses.FrozenInstanceError:
                pass
            return inner(seed, split)
        return arm
    run(prereg().freeze(), {"A": spy("A", 0), "B": spy("B", 1), "B_minus": spy("C", 0)})
    by_seed = {}
    for name, seed, fp, dev, held in seen:
        by_seed.setdefault(seed, set()).add((fp, dev, held))
    assert all(len(v) == 1 for v in by_seed.values()), "arms saw different data"
    assert len(seen) == 15
    leaky = dataclasses.replace(SPLIT, heldout=SPLIT.heldout + (SPLIT.dev[0],))
    try:
        ab.run_ab(prereg().freeze(), arms(), splits=[leaky], measure_memory=False)
        raise AssertionError("leaky split accepted")
    except SplitLeakError:
        pass
    print("  G. 3 arms x 5 seeds saw one identical frozen split per seed; "
          "a split leaking one episode is refused")


def test_ablation_slot():
    p = prereg(ablation_arm=None).freeze()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rep = ab.run_ab(p, arms(3.0), splits=[SPLIT], measure_memory=False)
    assert any("NO ABLATION ARM" in str(x.message) for x in w), "no warning"
    assert any("NO ABLATION ARM" in s for s in rep.results.warnings)
    assert "NO ABLATION ARM" in rep.to_markdown()
    a = arms(3.0)
    del a["B_minus"]
    try:
        run(prereg().freeze(), a)
        raise AssertionError("named ablation missing but accepted")
    except ab.PreregistrationError:
        pass
    print("  H. absent ablation -> warning + report line; named-but-missing "
          "ablation refused")


def test_equal_interactions():
    def curve_arm(n, tag):
        def arm(seed, split):
            jitter = np.random.default_rng([seed, tag]).normal(scale=0.01)
            pts = [{"interactions": k, "seconds": k * 1e-3, "value": k / 100.0 + jitter}
                   for k in range(10, n + 1, 10)]
            return {"score": pts[-1]["value"], "curve": pts}
        return arm
    a = {"A": curve_arm(100, 1), "B": curve_arm(200, 2), "B_minus": curve_arm(100, 3)}
    fin = run(prereg(basis="final").freeze(), a).verdict
    eq = run(prereg(basis="equal_interactions").freeze(), a).verdict
    assert fin["verdict"] == "accept", fin["reasons"]
    assert eq["verdict"] != "accept", eq["reasons"]
    c = eq["comparisons"]["equal_interactions"]
    assert all(u["budget"] == 100 for u in c["units"])
    print(f"  I. 2x-interaction 'win': final={fin['verdict']}, "
          f"equal_interactions={eq['verdict']} (diff {c['mean_diff']:+.3f} at 100)")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="ab_smoke_")
    try:
        test_tampering_refused(tmp)
        test_screening_only()
        test_known_better_and_worse()
        test_null_false_positive_rate()
        test_failures_recorded()
        test_reports_written(os.path.join(tmp, "reports"))
        test_same_split_for_all_arms()
        test_ablation_slot()
        test_equal_interactions()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[foundation-ab] ALL PASS")
