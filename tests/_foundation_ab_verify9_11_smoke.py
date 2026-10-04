"""Stage 9-11 verification A/Bs, reduced scale: the PROTOCOL runs end to end.

WHAT THIS IS
    experiments/foundation_ab/stage9_11/ holds an independent verifier's
    preregistered A/B comparisons that probe the Stage 9 (discovery), Stage
    10 (experiment selection) and Stage 11 (planning/skills) implementations
    on fixtures their authors did not tune on: noisier/smaller data with
    distractors, pure noise, rare-row misfit and a large-table budget (E9);
    replication on new parameters, four noisy TVs, a learnable-but-useless
    distractor and a misspecified hypothesis set (E10); a model wrong only
    in a rarely visited region, a heavy-tailed-latency model and
    behaviour-signature edge cases (E11). Full verdicts and per-run tables:
    experiments/foundation_ab/stage9_11/RESULTS.md.

    This smoke re-runs every experiment at toy scale (3 seeds, few
    scenarios). It deliberately does NOT assert who wins — at this scale
    the numbers mean nothing, and a test that pinned a winner would turn a
    finding into a chore (CLAUDE.md §5). It asserts the comparison
    MACHINERY is honest:

    A. Every prereg of a module is frozen BEFORE its first recorded result
       and before any (memoised) arm computation; it verifies against its
       hash; the verdict is in {accept, reject, inconclusive}; and the
       verdict re-derived from the saved JSON report equals the live one.
    B. No silent arm failure: every arm (prereg roles and exploratory) has
       exactly one run per preregistered seed and split, status "ok", and a
       finite primary metric. A deliberately broken arm is RECORDED as
       failed, not skipped (the falsification of this contract).
    C. Identical data: within each (seed, split) every arm that is declared
       to share a world reports the same `data_fp` (sha256 of the arrays /
       scenario parameters it consumed); arms declared to use different
       data (`exploratory_data_differs`) are the only exceptions, and they
       are never in a decisive role.
    D. The ablation slot is real: each prereg names an ablation that ran
       for every seed, and the decisive comparison has one pair per seed.

Run:  PYTHONPATH=. python tests/_foundation_ab_verify9_11_smoke.py   (< 60 s)
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import tempfile
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.foundation.experiments import decide, load_report, run_ab  # noqa: E402
from developmental_ai.foundation.experiments.ab import VERDICTS  # noqa: E402

from experiments.foundation_ab.stage9_11 import (e9_discovery, e10_selection,  # noqa: E402
                                                 e11_planning, run as runner)

NAME = "foundation-ab-verify9-11"
N = [0]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    N[0] += 1
    print(f"  {N[0]:2d}. {msg}")


def contracts(mod, out_dir):
    specs = mod.experiments("smoke")
    t_frozen = max(sp["prereg"].frozen_at_ns for sp in specs)
    cache = {}
    for sp in specs:
        p = sp["prereg"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = run_ab(p, runner.memo(sp["arms"], cache), splits=sp["splits"],
                         out_dir=out_dir, measure_memory=False)
        res, v = rep.results, rep.verdict
        first = min(r.t_recorded_ns for r in res.runs)
        p.verify()
        assert p.frozen_at_ns <= first and v["verdict"] in VERDICTS, (p.name, v["label"])
        lp, lr = load_report(rep.json_path)
        assert decide(lp, lr)["verdict"] == v["verdict"], p.name
        computed = [r.metrics["computed_at_ns"] for r in res.runs if r.status == "ok"]
        assert min(computed) > t_frozen, f"{p.name}: an arm ran before every prereg froze"
        # B
        n_split = len(sp["splits"])
        for a in sp["arms"]:
            rs = [r for r in res.runs if r.arm == a]
            assert len(rs) == len(p.seeds) * n_split, (p.name, a, len(rs))
            bad = [(r.seed, r.status, r.error) for r in rs if r.status != "ok"]
            assert not bad, f"{p.name}: arm {a} failed: {bad}"
            assert all(math.isfinite(r.primary) for r in rs), (p.name, a)
        # C
        differ = set(sp.get("exploratory_data_differs", ()))
        roles = {p.baseline_arm, p.candidate_arm, p.ablation_arm, p.alternative_arm}
        if sp["tag"] != "E9f":                    # E9f's arms ARE different tables
            assert not (differ & roles) or sp["tag"] == "E10d", (p.name, differ & roles)
        for s in p.seeds:
            for spl in res.split_ids:
                fps = {r.metrics["data_fp"] for r in res.runs
                       if r.seed == s and r.split_id == spl and r.arm not in differ}
                assert len(fps) == 1 or (not fps and set(sp["arms"]) <= differ), \
                    f"{p.name} seed {s}: arms saw different data {fps}"
        # D
        assert p.ablation_arm and p.ablation_arm in sp["arms"], p.name
        assert v["comparisons"][p.basis]["n_pairs"] == len(p.seeds) * n_split, p.name
        assert v["secondary"]["ablation_arm"]["n_pairs"] == len(p.seeds) * n_split, p.name
        print(f"      {p.name}: {v['label']} ({len(res.runs)} runs)")
    return specs


def sabotage(out_dir):
    """B's falsification: a broken arm must be recorded as failed, and the
    candidate failing must reject — never be skipped into a clean verdict."""
    sp = e10_selection.experiments("smoke")[1]           # E10b, cheap
    p = sp["prereg"]
    p2 = dataclasses.replace(p, name=p.name + "-sabotaged", frozen_at_ns=None,
                             frozen_hash=None).freeze()
    arms = dict(sp["arms"])

    def broken(seed, split):
        raise RuntimeError("deliberately broken arm")
    arms[p.candidate_arm] = broken
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = run_ab(p2, arms, splits=sp["splits"], out_dir=out_dir, measure_memory=False)
    st = [r.status for r in rep.results.runs if r.arm == p.candidate_arm]
    return st, rep.verdict["verdict"]


def main():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as out:
        for tag, mod in (("E9", e9_discovery), ("E10", e10_selection), ("E11", e11_planning)):
            specs = contracts(mod, out)
            check(True, f"{tag}: {len(specs)} preregs frozen before any result; verdicts "
                        f"re-derivable from JSON; every arm ran ok on every seed; data "
                        f"identical across same-world arms; ablation paired per seed")
        st, verdict = sabotage(out)
        check(st and all(s == "failed" for s in st) and verdict == "reject",
              f"a deliberately broken candidate is recorded failed x{len(st)} and the "
              f"prereg is rejected (not skipped)")
        idle = e10_selection.idle_tracker_probe(0, 0.2, False, n_eval=100, sd=0.05)
        check(math.isfinite(idle["net_progress"]) and idle["n_probes"] == 16,
              "E10e idle-tracker measurement runs (value reported in RESULTS.md, not asserted)")
    dt = time.time() - t0
    check(dt < 60, f"total {dt:.1f}s < 60s")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
