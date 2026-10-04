"""Run the Stage 9-11 verification A/Bs and write the generated tables.

    cd <repo> && PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.run [full|smoke] [E9 E10 E11]

Every prereg of a module is frozen inside its `experiments()` BEFORE any arm
runs (run_ab re-checks the order for each report). Arms shared by several
preregs (same function object) are computed once per (seed, split) by
`memo`, which stamps `computed_at_ns` so a reader can see each computation
happened after every freeze. Reports: stage9_11/reports/<scale>/ (JSON + md,
never overwritten, + index.jsonl). Tables: stage9_11/generated_<scale>.md.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from developmental_ai.foundation.experiments import run_ab

from . import e9_discovery, e10_selection, e11_planning

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = {"E9": e9_discovery, "E10": e10_selection, "E11": e11_planning}
KEYS = {
    "E9a": ("nll_robust", "nll_gauss", "incumbent_op", "deps", "n_distractor_deps",
            "alarms", "searches", "candidates_evaluated", "stopped", "search_seconds"),
    "E9b": ("nll_robust", "nll_gauss", "incumbent_op", "deps", "alarms", "searches",
            "candidates_evaluated", "stopped"),
    "E9c": ("false_struct_promotions", "struct_accepted", "struct_accepted_ops", "admitted",
            "promotions", "incumbent_op", "deps", "nll_gauss", "evaluated", "val_episodes"),
    "E9d": ("nll_gauss", "nll_robust", "nll_gauss_rare_rows", "captured_r", "best_op",
            "deps", "rare_rows_search", "cusum_alarms_on_change", "evaluated"),
    "E9f": ("overshoot_ratio", "elapsed", "internal_elapsed", "rows", "vars", "evaluated",
            "stopped", "leaked_threads"),
    "E10a": ("interactions_to_identify", "censored", "tv_fraction", "map_correct",
             "realized_nats_per_interaction"),
    "E10b": ("interactions_to_identify", "censored", "tv_fraction"),
    "E10c": ("interactions_to_identify", "censored", "lamp_fraction", "tv_fraction"),
    "E10d": ("confident_wrong_unflagged", "confident", "time_to_confidence", "ppc_flagged",
             "interactions_to_identify"),
    "E11a": ("mean_dist", "frac_steps_in_mud", "planner_steps_in_mud",
             "planner_entries_into_mud", "gate_closes", "planner_share"),
    "E11c": ("overrun_fraction", "late_fraction", "max_latency_over_deadline",
             "p99_latency_ms", "median_latency_ms", "planner_fraction", "timeout_fraction",
             "probes", "late_chunks"),
    "E11d": ("flag_rate", "js_bits_mean", "js_bits_max", "deploy_argmax_disagreement",
             "probe_sets_touching_region"),
}
KEYS["E9e"] = KEYS["E9d"]
KEYS["E11b"] = KEYS["E11a"]
KEYS["E11e"] = KEYS["E11d"]


def memo(arms: Dict[str, Any], cache: Dict[Any, Any]) -> Dict[str, Any]:
    out = {}
    for name, fn in arms.items():
        def run(seed, split, _fn=fn):
            k = (id(_fn), seed, split.split_id)
            if k not in cache:
                r = dict(_fn(seed, split))
                r["computed_at_ns"] = time.time_ns()
                cache[k] = r
            return dict(cache[k])
        out[name] = run
    return out


def run_module(tag: str, scale: str, out_dir: str):
    specs = MODULES[tag].experiments(scale)          # ALL preregs frozen here
    t_frozen = max(sp["prereg"].frozen_at_ns for sp in specs)
    cache: Dict[Any, Any] = {}
    reps = []
    for sp in specs:
        t0 = time.perf_counter()
        rep = run_ab(sp["prereg"], memo(sp["arms"], cache), splits=sp["splits"],
                     out_dir=out_dir, measure_memory=False)
        rep.verdict["_elapsed_s"] = time.perf_counter() - t0
        for r in rep.results.runs:
            if r.status == "ok":
                assert r.metrics["computed_at_ns"] > t_frozen, "result predates a freeze"
        reps.append((sp, rep))
    return reps


def _f(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def tables(reps) -> str:
    L: List[str] = []
    for sp, rep in reps:
        p, v, res = rep.prereg, rep.verdict, rep.results
        L += [f"### {p.name}", "",
              f"- **verdict: {v['label']}** — " + "; ".join(v["reasons"]),
              f"- hypothesis: {p.hypothesis}",
              f"- primary `{p.primary_metric}` ({p.direction} better), delta {p.delta:g}, "
              f"basis `{p.basis}`; baseline `{p.baseline_arm}`, candidate `{p.candidate_arm}`, "
              f"ablation `{p.ablation_arm}`, alternative `{p.alternative_arm}`; seeds "
              f"{list(p.seeds)}",
              f"- prereg hash `{p.frozen_hash[:12]}`, frozen {p.frozen_at_ns} ns; first "
              f"result {min(r.t_recorded_ns for r in res.runs)} ns",
              f"- report: `{os.path.relpath(rep.json_path, HERE) if rep.json_path else '-'}`",
              "", "| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low "
              "| CI high |", "|---|---|---|---|---|---|"]
        for b, c in v["comparisons"].items():
            L.append(f"| {p.candidate_arm} vs {p.baseline_arm} | {b}"
                     f"{' (decisive)' if b == p.basis else ''} | {c['n_pairs']} | "
                     f"{_f(c['mean_diff'])} | {_f(c['ci_low'])} | {_f(c['ci_high'])} |")
        for role, c in v["secondary"].items():
            L.append(f"| {p.candidate_arm} vs {c['a']} ({role}) | {c['basis']} | "
                     f"{c['n_pairs']} | {_f(c['mean_diff'])} | {_f(c['ci_low'])} | "
                     f"{_f(c['ci_high'])} |")
        L += [""] + [f"- note: {n}" for n in v["notes"]]
        L += [f"- harness warning: {w}" for w in res.warnings]
        L += ["", "| arm | ok/runs | failed | budget | mean primary | std |",
              "|---|---|---|---|---|---|"]
        for a, s in v["arms"].items():
            L.append(f"| {a} | {s['ok']}/{s['runs']} | {s['failed']} | {s['budget_exceeded']}"
                     f" | {_f(s['mean'])} | {_f(s['std'])} |")
        for r in res.runs:
            if r.status != "ok":
                L.append(f"\n- FAILED RUN {r.arm} seed {r.seed}: {r.status} {r.error}")
        keys = KEYS[sp["tag"]]
        L += ["", f"Per-run outcomes ({sp['tag']}):", "",
              "| arm | seed | status | wall s | " + " | ".join(keys) + " |",
              "|---|---|---|---|" + "---|" * len(keys)]
        for r in sorted(res.runs, key=lambda r: (r.arm, r.seed)):
            m = dict(r.metrics)
            if r.primary is not None:
                m[p.primary_metric] = r.primary
            L.append(f"| {r.arm} | {r.seed} | {r.status} | {r.wall_seconds:.2f} | "
                     + " | ".join(_f(m.get(k)) for k in keys) + " |")
        L += ["", "Per-arm means:", "", "| arm | " + " | ".join(keys) + " |",
              "|---|" + "---|" * len(keys)]
        for a in res.arms:
            row = []
            for k in keys:
                vals = []
                for r in res.runs:
                    if r.arm != a or r.status != "ok":
                        continue
                    m = dict(r.metrics)
                    if r.primary is not None:
                        m[p.primary_metric] = r.primary
                    x = m.get(k)
                    if isinstance(x, (int, float, np.integer, np.floating)):
                        vals.append(float(x))
                row.append(_f(float(np.mean(vals))) if vals else "-")
            L.append(f"| {a} | " + " | ".join(row) + " |")
        L.append("")
    return "\n".join(L)


def idle_table(scale: str) -> str:
    seeds = (0, 1, 2, 3, 4) if scale == "full" else (0,)
    n = 2000 if scale == "full" else 300
    cfgs = (("straddle 0.5+-0.2", 0.5, 0.2, False), ("wide 0.3+-0.2", 0.3, 0.2, False),
            ("far 0.2+-0.05", 0.2, 0.05, False), ("straddle annealed", 0.5, 0.2, True))
    L = ["## E10e idle / stationary-probe retention tracker (measurement, not an A/B)", "",
         "RetentionTracker(known_below=0.45, forgotten_above=0.55); 16 probes with i.i.d. "
         f"N(center, sd) losses, {n} evaluations, seeds {list(seeds)}.", "",
         "| config | seed | net progress | telescoped first-last | raw (clipped LP) gain | "
         "relearn events | flagged probes /16 |", "|---|---|---|---|---|---|---|"]
    for nm, c, sd, an in cfgs:
        for s in seeds:
            r = e10_selection.idle_tracker_probe(s, c, an, n_eval=n, sd=sd)
            L.append(f"| {nm} | {s} | {r['net_progress']:.1f} | "
                     f"{r['telescoped_first_minus_last']:.2f} | {r['raw_gain']:.1f} | "
                     f"{r['relearn_events']} | {r['flagged_probes']} |")
    return "\n".join(L) + "\n"


def followup() -> str:
    """POST-HOC (written after the preregistered run; exploratory, not a
    verdict): E9c's null on 60 more seeds, to bound the false-structure rate
    that 0/5 cannot."""
    from developmental_ai.foundation.experiments import make_split
    sp = make_split("e9:noise", [f"g{i}" for i in range(36)])
    rows, t0 = [], time.perf_counter()
    for seed in range(1000, 1060):
        m = e9_discovery.noise_arm(e9_discovery.ALL)(seed, sp)
        rows.append((m["struct_accepted"], m["admitted"], m["false_struct_promotions"],
                     m["evaluated"]))
    a = np.array(rows, float)
    return ("## Post-hoc follow-up: E9c null sweep (60 seeds, 1000-1059; exploratory)\n\n"
            f"- searches with >= 1 structural candidate ACCEPTED: {int((a[:, 0] > 0).sum())}/60 "
            f"(total accepted {int(a[:, 0].sum())}); admitted (any op) {int(a[:, 1].sum())}; "
            f"structural PROMOTIONS {int(a[:, 2].sum())}/60; candidates evaluated per search "
            f"{a[:, 3].mean():.0f}; {time.perf_counter() - t0:.0f}s\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "followup":
        txt = followup()
        print(txt)
        with open(os.path.join(HERE, "generated_followup.md"), "w") as f:
            f.write(txt)
        return
    scale = argv.pop(0) if argv and argv[0] in ("full", "smoke") else "full"
    tags = argv or list(MODULES)
    out_dir = os.path.join(HERE, "reports", scale)
    parts, T0 = [], time.perf_counter()
    for tag in tags:
        t0 = time.perf_counter()
        reps = run_module(tag, scale, out_dir)
        el = time.perf_counter() - t0
        print(f"[{tag}] {[(r.prereg.name, r.verdict['label']) for _, r in reps]} "
              f"({el:.1f}s)", flush=True)
        parts.append(f"## {tag} ({el:.0f}s wall)\n\n" + tables(reps))
        if tag == "E10":
            parts.append(idle_table(scale))
    parts.append(f"\nTotal wall {time.perf_counter() - T0:.0f}s\n")
    path = os.path.join(HERE, f"generated_{scale}_{'_'.join(tags)}.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"tables -> {path}")


if __name__ == "__main__":
    main()
