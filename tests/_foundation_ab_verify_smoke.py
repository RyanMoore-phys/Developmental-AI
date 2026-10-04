"""Stage 6-8 verification A/Bs, reduced scale: the PROTOCOL runs end to end.

WHAT THIS IS
    experiments/foundation_ab/ holds an independent verifier's preregistered
    A/B comparisons for the plan's Stage 6 (relational mechanism vs a
    same-input MLP, FLOP- and time-matched), Stage 7 (does ensemble
    uncertainty predict error, in and out of distribution) and Stage 8
    (tracker state vs per-frame features on next-frame prediction). The full
    runs and their verdicts are in experiments/foundation_ab/RESULTS_stage6_8.md.

    This smoke re-runs all three at toy scale (~20 s). It deliberately does
    NOT assert who wins: at this scale the numbers mean nothing, and a test
    that pinned a winner would turn a finding into a chore (CLAUDE.md §5).
    It asserts that the comparison MACHINERY is honest, because each of
    these has failed silently somewhere in this project before:

    A. Every prereg of an experiment is frozen BEFORE its first recorded
       result, verifies against its hash, and yields a verdict in
       {accept, reject, inconclusive} (no PreregistrationViolation).
    B. No arm silently failed: every arm the prereg names (baseline,
       candidate, ablation, alternative) and every exploratory arm has
       exactly one run per preregistered seed, status "ok", a finite
       primary metric. (The first draft of E1 had the FLOP-matched MLP and an
       ablation fail the preregistered "loss must fall" condition at toy
       scale — those were RECORDED as failed by the harness, which is the
       behaviour this contract protects; the toy config was then made stable
       so a failure here means something changed.)
    C. Identical data: within each (seed, split) every arm reports the same
       `data_fp` (sha256 of the exact arrays it consumed). The harness only
       fingerprints episode IDS; data here is generated from (seed, id), so
       this is the check that the arrays really match.
    D. The ablation slot is real: each prereg names an ablation, it ran for
       every seed, and the secondary comparison against it has one pair per
       seed (except on basis equal_time, where single-fit arms have no common
       time budget with the candidate — B already proves the ablation ran). The decisive comparison has one usable pair per seed (for E1c,
       the equal-time basis, that requires the long-trained MLP's curve to
       reach the IN's training time — if it stops reaching, E1c silently
       degrades to "inconclusive (0 pairs)", which this catches).
    E. Compute is reported, not inferred: E1 arms report FLOPs/sample and
       train seconds, and the FLOP-matched MLP is within 5% of the IN's FLOPs.

Run:  PYTHONPATH=. python tests/_foundation_ab_verify_smoke.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.foundation.experiments import run_ab  # noqa: E402
from developmental_ai.foundation.experiments.ab import VERDICTS  # noqa: E402

from experiments.foundation_ab import e1_mechanisms, e2_uncertainty, e3_perception  # noqa: E402

NAME = "foundation_ab_verify_smoke"
_n = [0]


def check(cond, msg):
    _n[0] += 1
    print(f"  {_n[0]:2d}. {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        raise SystemExit(f"[{NAME}] FAIL: {msg}")


def run(mod, out):
    specs = mod.experiments("smoke")
    t_frozen = max(sp["prereg"].frozen_at_ns for sp in specs)
    reps = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sp in specs:
            reps.append((sp, run_ab(sp["prereg"], sp["arms"], splits=sp["splits"],
                                    out_dir=out, measure_memory=False)))
    return reps, t_frozen


def contracts(tag, reps, t_frozen):
    first = min(r.t_recorded_ns for _, rep in reps for r in rep.results.runs)
    check(t_frozen < first,
          f"{tag} A. all {len(reps)} preregs frozen before the first result")
    for sp, rep in reps:
        p, v, res = rep.prereg, rep.verdict, rep.results
        p.verify()
        check(v["verdict"] in VERDICTS and v["acceptance_rule"],
              f"{tag} A. {p.name}: verdict '{v['label']}' (not asserted; just produced)")
        n_seeds = len(p.seeds)
        bad = [(r.arm, r.seed, r.status, (r.error or "")[:80]) for r in res.runs
               if r.status != "ok" or r.primary is None or not math.isfinite(r.primary)]
        per_arm = {a: sorted(r.seed for r in res.runs if r.arm == a) for a in res.arms}
        check(not bad and all(s == sorted(p.seeds) for s in per_arm.values())
              and set(sp["arms"]) == set(res.arms),
              f"{tag} B. {p.name}: {len(res.arms)} arms x {n_seeds} seeds all ok, "
              f"finite primary (failures: {bad})")
        fps = {}
        for r in res.runs:
            fps.setdefault(r.seed, set()).add(r.metrics.get("data_fp"))
        check(all(len(s) == 1 and None not in s for s in fps.values()),
              f"{tag} C. {p.name}: one data_fp per seed across all arms")
        sec = v["secondary"].get("ablation_arm")
        # On basis equal_time the secondary comparison needs both arms' curves to
        # reach a common budget; single-fit arms have one curve point each, so it
        # is legitimately unpaired there (reported in RESULTS). Elsewhere: paired.
        abl_ok = (sec is not None and sec["n_pairs"] == n_seeds) if p.basis != "equal_time" \
            else sec is not None
        check(p.ablation_arm in res.arms and abl_ok
              and v["comparisons"][p.basis]["n_pairs"] == n_seeds,
              f"{tag} D. {p.name}: ablation '{p.ablation_arm}' paired {n_seeds}/"
              f"{n_seeds}; decisive basis '{p.basis}' paired "
              f"{v['comparisons'][p.basis]['n_pairs']}/{n_seeds}")


def main():
    t0 = time.time()
    print(f"[{NAME}] reduced-scale Stage 6-8 A/Bs (no winner is asserted)")
    with tempfile.TemporaryDirectory() as out:
        reps, tf = run(e1_mechanisms, out)
        contracts("E1", reps, tf)
        fm = reps[0][0]["flop_match"]
        runs = reps[0][1].results.runs
        fl = {r.arm: r.metrics["flops_per_sample"] for r in runs}
        ts = all(r.metrics.get("train_seconds", 0) > 0 for r in runs)
        check(abs(fl["mlp_flops"] / fl["in"] - 1) < 0.05 and fl["mlp128"] < fl["in"] / 5
              and ts and fm["hidden"] > 128,
              f"E1 E. FLOPs/sample measured: IN {fl['in']}, mlp_flops {fl['mlp_flops']} "
              f"(hidden {fm['hidden']}), mlp128 {fl['mlp128']}; train seconds recorded")
        reps, tf = run(e2_uncertainty, out)
        contracts("E2", reps, tf)
        reps, tf = run(e3_perception, out)
        contracts("E3", reps, tf)
        n_reports = len([f for f in os.listdir(out) if f.endswith(".json")])
        check(n_reports == 8, f"8 reports written (negative results kept): {n_reports}")
    dt = time.time() - t0
    check(dt < 60, f"runtime {dt:.1f}s < 60s")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
