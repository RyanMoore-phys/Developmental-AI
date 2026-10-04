"""E9 — Stage 9 (foundation/discovery) probed on fixtures the implementers
did not tune on.

Implementers' claims under test
  * full proposal language beats REFIT-only by ~1.82 nats/row held-out after
    a structural change; adds nothing in a slope-only control;
  * pure-noise data yields no structure (their check: y shuffled, <=3 vars);
  * the 1% outlier component only stops one boundary row deciding an episode;
  * the search budget bounds time.

New fixtures (all generated here; nothing reused from discovery/fixtures):
  nregime   noise sd 0.5 (vs 0.2), 20 rows/episode (vs 30), 16 post-change
            dev episodes (vs 24), SEVEN distractors (4 continuous, 3
            discrete). Change: y = 1+2x  ->  y = 1+2x if temp < 0.3 else
            1.5-2x (threshold 0.3, not the 0.5 the fixtures use).
  nparam    same noise/size/distractors; slope-only change 1+2x -> 1+4x
            (pilot, seed 100: 1+3x — the implementers' control — raised ZERO
            change alarms at noise 0.5, so every arm stayed on the root and the
            control was vacuous; it is kept as exploratory arm full_slope3).
  noise     y ~ N(0, 1), TWELVE distractors, nothing ever depends on
            anything. A search is FORCED (the CUSUM would rarely alarm), its
            accepted candidates are admitted and judged by the registry on
            fresh 4-episode windows — the whole path to promotion.
  rare      y = 1+2x, then y = 1+2x + 3*r with r ~ Bernoulli(f): misfit
            confined to a fraction f of rows (f = 0.03 and 0.01). The
            ablation arm sets forms.ROBUST_EPS ~ 0 (pure Gaussian scoring)
            inside the arm only (restored in `finally`) — the code under
            test is not edited.
  budget    search() wall-clock overshoot on a small vs a large/wide table.

Primary held-out metric: per-row NLL of the final incumbent on the split's
final held-out episodes. `nll_robust` is the implementers' scoring
(mech.loglik, with the 1% outlier component); `nll_gauss` is the plain
Gaussian predictive (no outlier component) — the honest predictive score
when the question is whether misfit is being hidden.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List

import numpy as np

from developmental_ai.foundation.discovery import (
    REFIT, ChangeTriggeredReviser, EvidenceTable, MechanismRegistry, SearchBudget,
    fit_mechanism, initial_mechanism, search)
from developmental_ai.foundation.discovery import forms as _forms
from developmental_ai.foundation.experiments import Preregistration, make_split

from ..common import fingerprint_arrays, finite_metrics, sim_seed

ALL = ("ADD_DEPENDENCY", "SPLIT_APPLICABILITY", "COMPOSE", "REPLACE_TRANSITION", REFIT)
NO_SPLIT = ("ADD_DEPENDENCY", "COMPOSE", "REPLACE_TRANSITION", REFIT)
ONLY_REFIT = (REFIT,)
BUDGET = dict(wall_s=2.0, max_candidates=150)   # the implementers' gate budget


# ------------------------------------------------------------------ worlds
def _cols(world: str, rng, n: int, regime: str, f: float = 0.03,
          slope: float = 4.0) -> Dict[str, np.ndarray]:
    if world in ("nregime", "nparam"):
        x, temp = rng.uniform(-1, 1, n), rng.uniform(0, 1, n)
        c = {f"c{i}": rng.uniform(-1, 1, n) for i in range(1, 5)}
        k = {"k1": rng.integers(0, 3, n).astype(float),
             "k2": rng.integers(0, 3, n).astype(float),
             "k3": rng.integers(0, 2, n).astype(float)}
        if regime == "A":
            mu = 1 + 2 * x
        elif world == "nregime":
            mu = np.where(temp < 0.3, 1 + 2 * x, 1.5 - 2 * x)
        else:
            mu = 1 + slope * x
        return {"x": x, "temp": temp, **c, **k, "y": mu + rng.normal(0, 0.5, n)}
    if world == "noise":
        c = {f"c{i}": rng.uniform(-1, 1, n) for i in range(1, 9)}
        k = {f"k{i}": rng.integers(0, 2 + i % 2, n).astype(float) for i in range(1, 5)}
        return {**c, **k, "y": rng.normal(0, 1, n)}
    if world == "rare":
        x = rng.uniform(-1, 1, n)
        r = (rng.uniform(size=n) < f).astype(float)
        c1, c2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        mu = 1 + 2 * x + (3.0 * r if regime == "B" else 0.0)
        return {"x": x, "r": r, "c1": c1, "c2": c2, "y": mu + rng.normal(0, 0.2, n)}
    raise ValueError(world)


def episodes(world, tag, seed, eids, scope, regime, n_rows, fp, **kw) -> List[EvidenceTable]:
    out = []
    for e in eids:
        cols = _cols(world, np.random.default_rng(sim_seed(tag, seed, e)), n_rows, regime, **kw)
        fp.extend(cols[k] for k in sorted(cols))
        out.append(EvidenceTable(cols, "y", [e] * n_rows, scope))
    return out


def _cat(eps):
    return EvidenceTable.concat(eps)


def nll_robust(mech, tab) -> float:
    return float(-np.nanmean(mech.loglik(tab)))


def nll_gauss(mech, tab) -> float:
    mu, sd = mech.predict(tab)
    ok = np.isfinite(mu) & np.isfinite(tab.y)
    y = tab.y[ok]
    return float(np.mean(0.5 * np.log(2 * np.pi * sd[ok] ** 2) + 0.5 * ((y - mu[ok]) / sd[ok]) ** 2))


def _desc(mech) -> Dict[str, Any]:
    deps = list(mech.dependencies())
    return {"deps": ",".join(deps), "n_deps": len(deps), "complexity": int(mech.complexity()),
            "n_distractor_deps": sum(d[0] in "ck" and d[1:].isdigit() for d in deps)}


# --------------------------------------------------------------- reviser arm
def reviser_arm(world: str, ops, n_pre=10, n_post=16, n_rows=20, update=True, **wkw):
    def arm(seed: int, split) -> Dict[str, Any]:
        tag, scope = f"e9:{world}", split.scope
        dev = list(split.dev)
        fp: List[np.ndarray] = []
        pre = episodes(world, tag, seed, dev[:n_pre], scope, "A", n_rows, fp, **wkw)
        post = episodes(world, tag, seed, dev[n_pre:n_pre + n_post], scope, "B", n_rows, fp,
                        **wkw)
        held = episodes(world, tag, seed, list(split.heldout), scope, "B", n_rows, fp, **wkw)
        root = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(pre[:6]))
        H = _cat(held)
        t0 = time.perf_counter()
        if not update:
            inc, rv, reg = root, None, None
        else:
            reg = MechanismRegistry(promote_windows=2)
            reg.install(root)
            rv = ChangeTriggeredReviser(reg, "y", SearchBudget(**BUDGET), ops=ops)
            for ep in pre[6:] + post:
                rv.observe_episode(ep)
            inc = reg.incumbent("y")
        secs = time.perf_counter() - t0
        out = {"nll_robust": nll_robust(inc, H), "nll_gauss": nll_gauss(inc, H),
               "root_nll_robust": nll_robust(root, H), "data_fp": fingerprint_arrays(*fp),
               "replaced_root": inc.ref() != root.ref(), "arm_seconds": secs, **_desc(inc)}
        if rv is not None:
            op = reg.incumbent_entry("y").provenance[0].get("proposal", {}).get("op", "root")
            out.update(alarms=rv.alarms, searches=len(rv.searches),
                       candidates_evaluated=int(sum(r.evaluated for r in rv.searches)),
                       search_seconds=float(sum(r.elapsed for r in rv.searches)),
                       stopped=",".join(r.stopped for r in rv.searches) or "-",
                       incumbent_op=op, uses_temp="temp" in inc.dependencies(),
                       interactions=n_post * n_rows)
        return finite_metrics(out)
    return arm


# ------------------------------------------------------- forced-search arms
def noise_arm(ops, n_pre=6, n_search=10, n_win=12, n_rows=20, win=4):
    """Force a search on pure noise, admit its top-2, judge on fresh windows."""
    def arm(seed, split):
        tag, scope, dev = "e9:noise", split.scope, list(split.dev)
        fp: List[np.ndarray] = []
        pre = episodes("noise", tag, seed, dev[:n_pre], scope, "A", n_rows, fp)
        se = episodes("noise", tag, seed, dev[n_pre:n_pre + n_search], scope, "A", n_rows, fp)
        wi = episodes("noise", tag, seed, dev[n_pre + n_search:n_pre + n_search + n_win],
                      scope, "A", n_rows, fp)
        held = episodes("noise", tag, seed, list(split.heldout), scope, "A", n_rows, fp)
        root = fit_mechanism(initial_mechanism("y"), _cat(pre))
        reg = MechanismRegistry(promote_windows=2)
        reg.install(root)
        rep = search(root, _cat(se), SearchBudget(**BUDGET), ops=ops)
        struct_acc = [c for c in rep.accepted if c.proposal.op != REFIT]
        admitted = 0
        for c in sorted(rep.accepted, key=lambda c: -c.val["mean"])[:2]:
            admitted += reg.admit(c.mechanism, c.provenance())[0]
        promos = 0
        for i in range(0, len(wi) - win + 1, win):
            for ev in reg.evaluate_window("y", _cat(wi[i:i + win])):
                promos += ev["event"] == "promoted"
        inc = reg.incumbent("y")
        ent = reg.incumbent_entry("y")
        op = ent.provenance[0].get("proposal", {}).get("op", "root")
        H = _cat(held)
        return finite_metrics({
            "false_struct_promotions": float(promos > 0 and op != REFIT),
            "struct_accepted": len(struct_acc), "accepted": len(rep.accepted),
            "admitted": admitted, "promotions": promos, "incumbent_op": op,
            "evaluated": rep.evaluated, "stopped": rep.stopped,
            "val_episodes": len(rep.val_episodes), "search_seconds": rep.elapsed,
            "struct_accepted_ops": ",".join(sorted({c.proposal.op for c in struct_acc})) or "-",
            "nll_gauss": nll_gauss(inc, H), "nll_robust": nll_robust(inc, H),
            "data_fp": fingerprint_arrays(*fp), "interactions": (n_search + n_win) * n_rows,
            **_desc(inc)})
    return arm


def rare_arm(ops, f, eps=None, n_pre=6, n_search=20, n_rows=30):
    """Forced search after a change that touches only a fraction f of rows.
    eps: None = code default ROBUST_EPS; else temporarily patched."""
    def arm(seed, split):
        tag, scope, dev = f"e9:rare{f}", split.scope, list(split.dev)
        fp: List[np.ndarray] = []
        pre = episodes("rare", tag, seed, dev[:n_pre], scope, "A", n_rows, fp, f=f)
        se = episodes("rare", tag, seed, dev[n_pre:n_pre + n_search], scope, "B", n_rows, fp, f=f)
        held = episodes("rare", tag, seed, list(split.heldout), scope, "B", n_rows, fp, f=f)
        root = fit_mechanism(initial_mechanism("y", "linear", ("x",)), _cat(pre))
        old = _forms.ROBUST_EPS
        try:
            if eps is not None:
                _forms.ROBUST_EPS = eps
            rep = search(root, _cat(se), SearchBudget(**BUDGET), ops=ops)
        finally:
            _forms.ROBUST_EPS = old
        best = rep.best.mechanism if rep.best is not None else root
        # does the change detector even notice? (root's predictive residuals)
        from developmental_ai.foundation.inference.change import PredictiveCUSUM
        cu = PredictiveCUSUM()
        alarms = 0
        for ep in se:
            mu, sd = root.predict(ep)
            for i in range(len(ep)):
                alarms += bool(cu.update(mu[i], sd[i] ** 2, ep.y[i])["alarm"])
        H = _cat(held)
        Hr = H.select(H.col("r") == 1)
        return finite_metrics({
            "nll_gauss": nll_gauss(best, H), "nll_robust": nll_robust(best, H),
            "nll_gauss_rare_rows": nll_gauss(best, Hr) if len(Hr) else 0.0,
            "rare_rows_heldout": len(Hr), "rare_rows_search": int(_cat(se).col("r").sum()),
            "captured_r": "r" in best.dependencies(),
            "best_op": rep.best.proposal.op if rep.best is not None else "none",
            "accepted": len(rep.accepted), "evaluated": rep.evaluated, "stopped": rep.stopped,
            "cusum_alarms_on_change": alarms, "search_seconds": rep.elapsed,
            "data_fp": fingerprint_arrays(*fp), "interactions": n_search * n_rows,
            **_desc(best)})
    return arm


# ---------------------------------------------------------------- budget arm
def budget_arm(n_vars: int, n_eps: int, rows: int = 100, wall_s: float = 0.25,
               hang: bool = False):
    def arm(seed, split):
        rng = np.random.default_rng(sim_seed("e9:budget", seed, f"{n_vars}x{n_eps}"))
        eids = list(split.dev)[:n_eps]
        n = rows * len(eids)
        cols = {f"v{i}": rng.uniform(-1, 1, n) for i in range(n_vars)}
        cols["y"] = rng.normal(0, 1, n)
        tab = EvidenceTable(cols, "y", np.repeat(eids, rows), split.scope)
        kw = {}
        if hang:
            def fitter(m, t, d=None):
                time.sleep(5.0)
                return fit_mechanism(m, t, d)
            kw["fitter"] = fitter
        th0 = threading.active_count()
        t0 = time.perf_counter()
        rep = search(initial_mechanism("y"), tab,
                     SearchBudget(wall_s=wall_s, max_candidates=10_000, max_bytes=100_000,
                                  max_rows=10 ** 7), **kw)
        el = time.perf_counter() - t0
        return finite_metrics({
            "overshoot_ratio": el / wall_s, "elapsed": el, "internal_elapsed": rep.elapsed,
            "wall_s": wall_s, "rows": n, "vars": n_vars, "evaluated": rep.evaluated,
            "stopped": rep.stopped, "leaked_threads": threading.active_count() - th0,
            "data_fp": fingerprint_arrays(*[cols[k] for k in sorted(cols)])})
    return arm


# ------------------------------------------------------------- experiments
def experiments(scale: str = "full") -> List[Dict[str, Any]]:
    """Every prereg below is FROZEN here, before any arm is called."""
    seeds = (0, 1, 2, 3, 4) if scale == "full" else (0, 1, 2)
    specs = []
    tot = 48 if scale == "full" else 40

    sp = make_split("e9:nregime", [f"g{i}" for i in range(tot)])
    specs.append({"prereg": Preregistration(
        name="E9a-structural-vs-refit-noisy-regime",
        hypothesis=("after a structural rule change (threshold 0.3 on temp), on NEW noisier "
                    "(sd 0.5), smaller (16x20 post rows) data with 7 distractors, the full "
                    "proposal language yields lower held-out NLL than REFIT-only under the "
                    "same SearchBudget (implementers report 1.82 nats/row on their fixture)"),
        primary_metric="nll_robust", direction="lower", delta=0.1,
        baseline_arm="refit_only", candidate_arm="full", ablation_arm="full_no_split",
        alternative_arm="no_update", seeds=seeds, basis="final",
        resource_budget={"max_wall_seconds": 120.0},
        failure_conditions=("any arm raises or exceeds 120 s",
                            "held-out NLL non-finite",
                            "delta 0.1 nats/row is the minimum effect worth a structure"),
        split_fingerprints=(sp.fingerprint,),
        notes=f"budget {BUDGET} per search, ChangeTriggeredReviser defaults, promote_windows=2"
    ).freeze(), "arms": {
        "refit_only": reviser_arm("nregime", ONLY_REFIT), "full": reviser_arm("nregime", ALL),
        "full_no_split": reviser_arm("nregime", NO_SPLIT),
        "no_update": reviser_arm("nregime", ALL, update=False)}, "splits": [sp], "tag": "E9a"})

    sp = make_split("e9:nparam", [f"g{i}" for i in range(tot)])
    specs.append({"prereg": Preregistration(
        name="E9b-control-slope-change-structure-hurts",
        hypothesis=("CONTROL (falsification direction): after a slope-only change (2 -> 4), with "
                    "noise and 7 distractors, REFIT-only beats the full language on held-out "
                    "NLL by >= 0.02 nats/row, i.e. structure search invents structure that "
                    "costs prediction"),
        primary_metric="nll_robust", direction="lower", delta=0.02,
        baseline_arm="full", candidate_arm="refit_only", ablation_arm="no_update",
        seeds=seeds, basis="final", resource_budget={"max_wall_seconds": 120.0},
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
        "full": reviser_arm("nparam", ALL), "refit_only": reviser_arm("nparam", ONLY_REFIT),
        "no_update": reviser_arm("nparam", ALL, update=False),
        "full_slope3": reviser_arm("nparam", ALL, slope=3.0)}, "splits": [sp], "tag": "E9b",
        "exploratory_data_differs": ["full_slope3"]})

    sp = make_split("e9:noise", [f"g{i}" for i in range(36)])
    specs.append({"prereg": Preregistration(
        name="E9c-false-structure-on-pure-noise",
        hypothesis=("on pure noise with 12 distractors and a forced search (10x20 rows), the "
                    "full language gets a STRUCTURAL mechanism promoted through the registry "
                    "in >= 20% of runs more than REFIT-only (which cannot)"),
        primary_metric="false_struct_promotions", direction="higher", delta=0.2,
        baseline_arm="refit_only", candidate_arm="full", ablation_arm="full_no_split",
        seeds=seeds, basis="final", resource_budget={"max_wall_seconds": 60.0},
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
        "refit_only": noise_arm(ONLY_REFIT), "full": noise_arm(ALL),
        "full_no_split": noise_arm(NO_SPLIT)}, "splits": [sp], "tag": "E9c"})

    for f, tagx in ((0.03, "E9d"), (0.01, "E9e")):
        sp = make_split(f"e9:rare{f}", [f"g{i}" for i in range(40)])
        specs.append({"prereg": Preregistration(
            name=f"{tagx}-rare-misfit-f{f}",
            hypothesis=(f"a change touching only {f:.0%} of rows (y += 3 when r == 1) is "
                        f"found by the full language: held-out GAUSSIAN NLL beats REFIT-only "
                        f"by >= 0.05 nats/row (if the 1% outlier component hides the misfit, "
                        f"this fails while the eps~0 ablation succeeds)"),
            primary_metric="nll_gauss", direction="lower", delta=0.05,
            baseline_arm="refit_only", candidate_arm="full_robust",
            ablation_arm="full_eps0", seeds=seeds, basis="final",
            resource_budget={"max_wall_seconds": 60.0},
            failure_conditions=("any arm raises",),
            split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
            "refit_only": rare_arm(ONLY_REFIT, f), "full_robust": rare_arm(ALL, f),
            "full_eps0": rare_arm(ALL, f, eps=1e-12)}, "splits": [sp], "tag": tagx})

    sp = make_split("e9:budget", [f"g{i}" for i in range(800)])
    specs.append({"prereg": Preregistration(
        name="E9f-search-wall-budget-overshoot",
        hypothesis=("search() under wall_s = 0.25 s overshoots its budget by > 50% "
                    "(elapsed/wall_s - small-table ratio >= 0.5) on a large table "
                    "(150 variables x 60000 rows) — work outside the guarded fits is unbudgeted"),
        primary_metric="overshoot_ratio", direction="higher", delta=0.5,
        baseline_arm="small_table", candidate_arm="large_table", ablation_arm="hung_fitter",
        seeds=seeds, basis="final", resource_budget={"max_wall_seconds": 60.0},
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
        "small_table": budget_arm(5, 20), "large_table": budget_arm(150, 600),
        "hung_fitter": budget_arm(5, 20, hang=True)}, "splits": [sp], "tag": "E9f",
        "exploratory_data_differs": ["small_table", "large_table", "hung_fitter"]})
    return specs
