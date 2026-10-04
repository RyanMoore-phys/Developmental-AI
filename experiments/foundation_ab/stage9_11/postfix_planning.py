"""Post-fix re-run of the verifier's E11 (planning) A/Bs, after the planning
fixes for B3 (deadline), F7 (region-aware reliability) and F8 (duplicate
detection needs evidence coverage). Summary: POSTFIX_planning.md.

    cd <repo> && PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.postfix_planning

PREREGS. The ORIGINAL frozen preregistrations are re-loaded from the
verifier's own reports (reports/full/E11*.json) — same content, same hash,
same freeze time — and handed to run_ab unchanged; run_ab re-verifies each
hash. Nothing in a prereg was edited, so no *_postfix prereg was needed.

ARMS. The verifier's arm functions (e11_planning.py) are re-used verbatim
wherever they exercise the fix through the default API:
  E11a/b  region_arm(mode) unchanged. "gated" now gets the region-aware
          gate by default. Exploratory arm "gated_global" = the same arm
          with ReliabilityMonitor(regions=False) (the pre-fix global gate),
          injected by swapping the name the verifier's module looks up for
          the duration of the call.
  E11c    deadline_arm(kind) unchanged (primary: decisions > 2x deadline).
          The 1.25x distribution is measured separately by deadline_dist(),
          a line-for-line copy of deadline_arm that keeps every latency.
  E11d/e  run TWICE under the same frozen prereg:
          (i)  the verifier's skill_arm unchanged — it builds signatures on
               the bare 32-probe array and counts len(find_duplicates()),
               which now returns "indistinguishable" findings (no evidence,
               no duplicate verdict), so the count is unchanged by design;
          (ii) skill_arm_evidence(pair): identical pairs and probes, but the
               probe set is built with build_probe_set from each skill's own
               evidenced states (64 draws each from the deployment
               distribution N(0, 4^2), where the verifier measured
               disagreement) plus the 32 shared probes, and a trial is
               flagged only on verdict == "duplicate". Exploratory arm
               "offprobe_inside": evidence drawn from N(0, 2^2) only — two
               skills only ever used where they agree; a duplicate verdict
               there is the honest one.
Reports: reports/postfix/ (never overwritten). Tables:
generated_postfix_planning.md.
"""

from __future__ import annotations

import functools
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from developmental_ai.foundation.experiments import load_report, run_ab
from developmental_ai.foundation.planning import (
    BoundedPlanner, PlanningController, ReliabilityMonitor, behaviour_signature,
    behavioural_divergence, build_probe_set, find_duplicates)
from developmental_ai.foundation.planning.fixtures import (
    DT, GoalReward, PointMassAdapter, ProportionalController, encode_discrete,
    policy_vector, state_from_obs)

from ..common import fingerprint_arrays, finite_metrics, sim_seed
from . import e11_planning as E
from . import run as runner

HERE = os.path.dirname(os.path.abspath(__file__))


def original_prereg(prefix: str):
    paths = sorted(glob.glob(os.path.join(HERE, "reports", "full", f"{prefix}-*.json")))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one original report for {prefix}: {paths}")
    p, _ = load_report(paths[0])
    p.verify()
    return p, paths[0]


# ------------------------------------------------------------------ E11a/b
def gated_global_arm(n_eps: int):
    inner = E.region_arm("gated", n_eps)

    def arm(seed, split):
        orig = E.ReliabilityMonitor
        E.ReliabilityMonitor = functools.partial(ReliabilityMonitor, regions=False)
        try:
            return inner(seed, split)
        finally:
            E.ReliabilityMonitor = orig
    return arm


# ------------------------------------------------------------------ E11c
def deadline_dist(kind: str, seed: int, n_dec: int = 150, deadline: float = 0.010):
    """Line-for-line copy of e11_planning.deadline_arm, returning every
    decision latency (in units of the deadline) and the sources."""
    if kind == "constant":
        m = E._SlowModel(0.0, 0.0, 0.0006, seed)
    else:
        m = E._SlowModel(0.02, 0.030, 0.0, sim_seed("lat", seed, kind) % 2 ** 31)
    m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
    env = PointMassAdapter("discrete", seed=0)
    gr = GoalReward()
    pl = BoundedPlanner(m, gr, env.action_spec(), n_entities=1, action_dim=2,
                        encode=encode_discrete, horizon=4, n_candidates=64, n_iters=2,
                        deadline_s=deadline, reprobe_after=5,
                        chunk=8 if kind == "heavy_tail_chunk8" else None, seed=seed)
    pc = ProportionalController(seed=3)
    ctl = PlanningController(pl, lambda o: pc(policy_vector(o)),
                             ReliabilityMonitor(window=10, min_count=3, max_z2=9.0))
    for _ in range(3):
        ctl.reliability.validate([0], [1], [0], provenance="sensor",
                                 model_version=m.version)
    obs = env.reset(seed=sim_seed("dl", seed, "0") % 2 ** 31)
    s, g, _ = state_from_obs(obs)
    gr.goal = g
    el, src = [], []
    for _ in range(n_dec):
        t0 = time.perf_counter()
        d = ctl.decide(s, DT, obs)
        el.append(time.perf_counter() - t0)
        src.append(d.source)
    pl.wait_idle(1.0)
    return np.array(el) / deadline, src


def deadline_table(seeds=(0, 1, 2, 3, 4)) -> str:
    L = ["## E11c latency distribution (decide() latency / 10 ms deadline; 5 seeds x 150 "
         "decisions; measurement, not an A/B)", "",
         "| model | > 1x | > 1.25x | > 2x | p50 | p99 | max | planner | fallback_timeout |",
         "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for kind in ("constant", "heavy_tail", "heavy_tail_chunk8"):
        rs = [deadline_dist(kind, s) for s in seeds]
        x = np.concatenate([a for a, _ in rs])
        src = sum([b for _, b in rs], [])
        row = {"over1": float(np.mean(x > 1)), "over1.25": float(np.mean(x > 1.25)),
               "over2": float(np.mean(x > 2)), "p50": float(np.median(x)),
               "p99": float(np.quantile(x, 0.99)), "max": float(x.max()),
               "planner": src.count("planner") / len(src),
               "timeout": src.count("fallback_timeout") / len(src)}
        out[kind] = row
        L.append(f"| {kind} | {row['over1']:.1%} | {row['over1.25']:.1%} | {row['over2']:.1%}"
                 f" | {row['p50']:.2f} | {row['p99']:.2f} | {row['max']:.2f} | "
                 f"{row['planner']:.1%} | {row['timeout']:.1%} |")
    with open(os.path.join(HERE, "reports", "postfix", "e11c_latency.json"), "w") as f:
        json.dump(out, f, indent=1)
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ E11d/e
def skill_arm_evidence(pair: str, n_trials: int = 20, drift: float = 0.0,
                       evidence_sd: float = 4.0, n_evidence: int = 64):
    """The verifier's skill_arm, with the duplicate verdict taken on a probe
    set built from each skill's own evidenced states."""
    def arm(seed, split):
        flags, js, dis, cover, nprobes = [], [], [], [], []
        base_fp = []
        for tr in range(n_trials):
            rng = np.random.default_rng(sim_seed("e11:skill", seed, f"t{tr}"))
            A = E._policy(rng)
            base_fp.append(A["W1"].ravel())
            probes = rng.normal(size=(32, 4)) * 2.0
            deploy = rng.normal(size=(400, 4)) * 4.0
            other = np.random.default_rng(sim_seed("e11:other", seed, f"t{tr}"))
            fa = E._probs_fn(A)
            if pair == "name_only":
                fb = E._probs_fn({k: v.copy() for k, v in A.items()})
            elif pair == "samename_diff":
                fb = E._probs_fn(E._policy(other))
            elif pair == "offprobe":
                fb = E._probs_fn(A, offprobe=True)
            elif pair == "tempered":
                fb = E._probs_fn(A, temp=3.0)
            elif pair == "drift":
                fb = E._probs_fn({k: v + drift * other.normal(size=v.shape) * np.abs(v).mean()
                                  for k, v in A.items()})
            else:
                raise ValueError(pair)
            ev_rng = np.random.default_rng(sim_seed("e11:evidence", seed, f"t{tr}"))
            ia, ib = f"skill-{tr}-a", f"skill-{tr}-b"
            evidence = {ia: ev_rng.normal(size=(n_evidence, 4)) * evidence_sd,
                        ib: ev_rng.normal(size=(n_evidence, 4)) * evidence_sd}
            ps = build_probe_set(evidence, shared=probes)
            sigs = {ia: behaviour_signature(ps, probs_fn=fa),
                    ib: behaviour_signature(ps, probs_fn=fb)}
            found = find_duplicates(sigs, threshold=1e-3, probe_set=ps)
            flags.append(float(any(f.verdict == "duplicate" for f in found)))
            js.append(behavioural_divergence(*sigs.values()))
            dis.append(float(np.mean([np.argmax(fa(s)) != np.argmax(fb(s)) for s in deploy])))
            cover.append(min(ps.coverage.values()))
            nprobes.append(len(ps))
        return finite_metrics({
            "flag_rate": float(np.mean(flags)), "js_bits_mean": float(np.mean(js)),
            "js_bits_max": float(np.max(js)),
            "deploy_argmax_disagreement": float(np.mean(dis)),
            "probe_sets_touching_region": float(np.mean(cover)),   # = min coverage here
            "n_probes_mean": float(np.mean(nprobes)),
            "interactions": n_trials, "data_fp": fingerprint_arrays(*base_fp)})
    return arm


# ------------------------------------------------------------------ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = os.path.join(HERE, "reports", "postfix")
    os.makedirs(out_dir, exist_ok=True)
    specs = {sp["tag"]: sp for sp in E.experiments("full")}   # splits + verifier arms
    for sp in specs.values():                                  # (fresh freezes unused)
        assert sp["prereg"].split_fingerprints == tuple(s.fingerprint for s in sp["splits"])
    nt, n_eps = 20, 20
    sk_orig = specs["E11d"]["arms"]
    sk_ev = {p: skill_arm_evidence(p, nt) for p in ("name_only", "samename_diff", "offprobe",
                                                     "tempered")}
    sk_ev.update({f"drift_{d:g}": skill_arm_evidence("drift", nt, d) for d in (1e-3, 1e-2, 1e-1)})
    sk_ev["offprobe_inside"] = skill_arm_evidence("offprobe", nt, evidence_sd=2.0)
    region = dict(specs["E11a"]["arms"])
    region["gated_global"] = gated_global_arm(n_eps)
    plan = [("E11a", region, "E11a"), ("E11b", region, "E11b"),
            ("E11c", specs["E11c"]["arms"], "E11c"),
            ("E11d", sk_orig, "E11d"), ("E11e", sk_orig, "E11e"),
            ("E11d", sk_ev, "E11d+evidence"), ("E11e", sk_ev, "E11e+evidence")]
    if argv:
        plan = [x for x in plan if x[2] in argv or x[0] in argv]
    caches: Dict[int, Dict] = {}
    reps, parts, T0 = [], [], time.perf_counter()
    for tag, arms, label in plan:
        prereg, src = original_prereg(specs[tag]["prereg"].name)
        if prereg.content_hash() != prereg.frozen_hash:          # pragma: no cover
            raise RuntimeError(f"{label}: original prereg hash does not verify")
        t0 = time.perf_counter()
        cache = caches.setdefault(id(arms), {})
        rep = run_ab(prereg, runner.memo(arms, cache), splits=specs[tag]["splits"],
                     out_dir=out_dir, measure_memory=False)
        rep.verdict["_elapsed_s"] = time.perf_counter() - t0
        print(f"[{label}] {prereg.name} hash {prereg.frozen_hash[:12]} (original, from "
              f"{os.path.basename(src)}): {rep.verdict['label']} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        reps.append(({"tag": tag, "prereg": prereg}, rep))
        parts.append(f"## {label}\n\n" + runner.tables([({"tag": tag, "prereg": prereg},
                                                          rep)]))
    if not argv or "E11c" in argv:
        parts.append(deadline_table())
    parts.append(f"\nTotal wall {time.perf_counter() - T0:.0f}s\n")
    path = os.path.join(HERE, "generated_postfix_planning.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"tables -> {path}")


if __name__ == "__main__":
    main()
