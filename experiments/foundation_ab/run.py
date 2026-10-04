"""Run the Stage 6-8 verification A/Bs and write RESULTS tables.

    cd <repo> && PYTHONPATH=. python -m experiments.foundation_ab.run [full|smoke] [E1 E2 E3]

Every prereg of an experiment is frozen inside `experiments()` BEFORE any arm
runs; run_ab then refuses a verdict if that ordering is ever violated.
Reports (JSON + md per prereg, never overwritten, plus index.jsonl) go to
experiments/foundation_ab/reports/<scale>/. The generated tables go to
experiments/foundation_ab/generated_<scale>.md.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np

from developmental_ai.foundation.experiments import run_ab

from . import e1_mechanisms, e2_uncertainty, e3_perception

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = {"E1": e1_mechanisms, "E2": e2_uncertainty, "E3": e3_perception}
KEY_METRICS = {
    "E1": ("nll1", "rmse1", "cov90_1", "nll_event", "rmse10", "nll10", "cov90_10",
           "rmse1_contact", "rmse1_free", "flops_per_sample", "params", "train_seconds",
           "predict_ms_per64", "epochs"),
    "E2": ("unc_err_spearman", "spearman_id", "spearman_g4", "spearman_g20", "spearman_box6",
           "cov90_id", "cov90_g4", "cov90_g20", "cov90_box6", "cov90_shift_abs_err",
           "nll_id", "nll_g4", "nll_g20", "nll_box6",
           "signal_ratio_g4", "signal_ratio_g20", "signal_ratio_box6",
           "conf_wrong_frac_id", "conf_wrong_frac_g4", "conf_wrong_frac_g20",
           "conf_wrong_frac_box6", "outside3sd_frac_id", "outside3sd_frac_g4",
           "outside3sd_frac_g20", "outside3sd_frac_box6", "train_seconds"),
    "E3": ("ospa_obs", "ospa_oracle_PRIVILEGED", "ospa_obs_p90", "mean_predicted_count",
           "ms_per_frame", "frames_scored"),
}


def run_experiment(tag: str, scale: str, out_dir: str):
    mod = MODULES[tag]
    specs = mod.experiments(scale)
    reps = []
    for sp in specs:
        t0 = time.perf_counter()
        rep = run_ab(sp["prereg"], sp["arms"], splits=sp["splits"], out_dir=out_dir,
                     measure_memory=False)
        rep.verdict["_elapsed_s"] = time.perf_counter() - t0
        reps.append((sp, rep))
    return reps


def _f(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def tables(tag: str, reps) -> str:
    L: List[str] = []
    for sp, rep in reps:
        p, v, res = rep.prereg, rep.verdict, rep.results
        c = v["comparisons"][p.basis]
        L += [f"### {p.name}", "",
              f"- **verdict: {v['label']}** — " + "; ".join(v["reasons"]),
              f"- hypothesis: {p.hypothesis}",
              f"- primary `{p.primary_metric}` ({p.direction} better), delta {p.delta:g}, "
              f"basis `{p.basis}`; baseline `{p.baseline_arm}`, candidate "
              f"`{p.candidate_arm}`, ablation `{p.ablation_arm}`, alternative "
              f"`{p.alternative_arm}`; seeds {list(p.seeds)}",
              f"- prereg hash `{p.frozen_hash[:12]}`, frozen {p.frozen_at_ns} ns; first "
              f"result {min(r.t_recorded_ns for r in res.runs)} ns",
              f"- report: `{os.path.relpath(rep.json_path, HERE) if rep.json_path else '-'}`",
              ""]
        L += ["| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | "
              "mean diff | CI low | CI high | skipped |", "|---|---|---|---|---|---|---|"]
        for b, cc in v["comparisons"].items():
            L.append(f"| {p.candidate_arm} vs {p.baseline_arm} | {b}"
                     f"{' (decisive)' if b == p.basis else ''} | {cc['n_pairs']} | "
                     f"{_f(cc['mean_diff'])} | {_f(cc['ci_low'])} | {_f(cc['ci_high'])} | "
                     f"{len(cc['skipped'])} |")
        for role, cc in v["secondary"].items():
            L.append(f"| {p.candidate_arm} vs {cc['a']} ({role}) | {cc['basis']} | "
                     f"{cc['n_pairs']} | {_f(cc['mean_diff'])} | {_f(cc['ci_low'])} | "
                     f"{_f(cc['ci_high'])} | {len(cc['skipped'])} |")
        if v["notes"]:
            L += [""] + [f"- note: {n}" for n in v["notes"]]
        if res.warnings:
            L += [f"- harness warning: {w}" for w in res.warnings]
        L.append("")
        st = v["arms"]
        L += ["| arm | ok/runs | failed | budget | mean primary | std |",
              "|---|---|---|---|---|---|"]
        for a, s in st.items():
            L.append(f"| {a} | {s['ok']}/{s['runs']} | {s['failed']} | {s['budget_exceeded']}"
                     f" | {_f(s['mean'])} | {_f(s['std'])} |")
        fails = [r for r in res.runs if r.status != "ok"]
        for r in fails:
            L.append(f"\n- FAILED RUN {r.arm} seed {r.seed}: {r.status} {r.error}")
        L.append("")
    # per-run table once per experiment (arm results are shared across its preregs)
    sp, rep = reps[0]
    keys = KEY_METRICS[tag]
    L += [f"### {tag} per-run outcomes (shared by every {tag} prereg)", "",
          "| arm | seed | status | data_fp | " + " | ".join(keys) + " |",
          "|---|---|---|---|" + "---|" * len(keys)]
    for r in sorted(rep.results.runs, key=lambda r: (r.arm, r.seed)):
        m = dict(r.metrics)
        if r.primary is not None:
            m[rep.prereg.primary_metric] = r.primary
        L.append(f"| {r.arm} | {r.seed} | {r.status} | {m.get('data_fp', '-')} | "
                 + " | ".join(_f(m.get(k)) for k in keys) + " |")
    L += ["", f"{tag} per-arm means over seeds:", "",
          "| arm | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for a in rep.results.arms:
        rs = [r for r in rep.results.runs if r.arm == a and r.status == "ok"]
        row = []
        for k in keys:
            vals = []
            for r in rs:
                m = dict(r.metrics)
                if r.primary is not None:
                    m[rep.prereg.primary_metric] = r.primary
                if isinstance(m.get(k), (int, float)):
                    vals.append(float(m[k]))
            row.append(_f(float(np.mean(vals))) if vals else "-")
        L.append(f"| {a} | " + " | ".join(row) + " |")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    scale = argv.pop(0) if argv and argv[0] in ("full", "smoke") else "full"
    tags = argv or list(MODULES)
    out_dir = os.path.join(HERE, "reports", scale)
    parts = []
    for tag in tags:
        t0 = time.perf_counter()
        reps = run_experiment(tag, scale, out_dir)
        el = time.perf_counter() - t0
        print(f"[{tag}] {[(r.prereg.name, r.verdict['label']) for _, r in reps]} "
              f"({el:.1f}s)", flush=True)
        parts.append(f"## {tag} ({el:.0f}s wall)\n\n" + tables(tag, reps))
    path = os.path.join(HERE, f"generated_{scale}_{'_'.join(tags)}.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"tables -> {path}")


if __name__ == "__main__":
    main()
