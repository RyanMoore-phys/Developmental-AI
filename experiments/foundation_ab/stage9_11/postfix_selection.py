"""Post-fix re-run of the verifier's E10 experiments against the fixed
foundation/experiments selection code (targeted information, the
misspecification check, the noise-aware retention tracker).

    cd <repo> && PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.postfix_selection

ORIGINAL FROZEN PREREGS. Every prereg is loaded from the verifier's own
full-scale JSON report (reports/full/E10*.json), so its hash and freeze time
are the original ones (prereg.verify() passes; the freeze predates every
result here by construction). Nothing in the verifier's files is edited.

Two runs per prereg, both under that original prereg:
  "original arms"  the verifier's arm functions, unmodified (E10a's arms go
                   through the library's fixture, so they ARE post-fix code;
                   E10b-d's sims call the library's joint I(H;O) and are a
                   reproduction of the pre-fix behaviour).
  "library arms"   the same arm NAMES bound to the post-fix library path
                   (`lib_sim` below): question.ExperimentQuestion supplies
                   targeted I(shift; O), the real-evidence update and the
                   resolved/misspecified status. The world (scenario,
                   cue/outcome RNG streams, selector seeds) is generated
                   exactly as e10_selection.run_sim does, and data_fp is the
                   same fingerprint, so the identical-data contract holds.

E10d's primary metric `confident_wrong_unflagged` means, for library arms:
the library REPORTED a resolved conclusion (ExperimentQuestion state
"resolved") naming a shift although the truth is not a shift, without the
set being flagged. `posterior_confident_wrong` (the posterior alone went
> 0.9 on a wrong shift) is kept alongside: Bayes over a set without the
truth still goes confident — what changed is whether that is reported.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Any, Dict

import numpy as np

from developmental_ai.foundation.contracts import Action, ActionSpec, Observation
from developmental_ai.foundation.experiments import (
    RESOLVED, ExperimentQuestion, ExperimentSelector, default_terms, effort_of,
    enumerate_candidates, load_report, run_ab, slip_probability)
from developmental_ai.foundation.inference import HypothesisSet

from . import e10_selection as E
from .run import KEYS, memo, tables
from ..common import fingerprint_arrays, finite_metrics

HERE = os.path.dirname(os.path.abspath(__file__))


def lib_sim(arm: str, sid: str, seed: int, K: int = 6, reward_noise: float = 0.35,
            n_tv: int = 1, lamp_bits: int = 0, misspecified: bool = False,
            budget: int = 60, threshold: float = 0.9, epsilon: float = 0.05,
            floor: float = 1e-4, confirm: float = 5.0,
            stop_report: bool = None) -> Dict[str, Any]:
    """stop_report (default: misspecified) runs until the question REPORTS
    (resolved) or the budget, instead of stopping at the evaluator-side
    identification time."""
    # ---- the world: verbatim from e10_selection.run_sim -----------------------
    r = np.random.default_rng(E._h("e10", sid, seed))
    shift = int(r.integers(K))
    D = 2 ** lamp_bits
    lamp = int(r.integers(D))
    resp = [(c + shift) % K for c in range(K)]
    if misspecified:
        half = sorted(r.choice(K, K // 2, replace=False).tolist())
        tgt = [(resp[c] + 1 + int(r.integers(K - 1))) % K for c in half]
        for c, t in zip(half, tgt):
            resp[c] = t
        for s in range(K):
            if all(resp[c] == (c + s) % K for c in range(K)):
                resp[half[0]] = (resp[half[0]] + 1) % K
    slip = slip_probability(reward_noise)
    n_act = K + n_tv + lamp_bits
    spec = ActionSpec.discrete("sim.act", n_act)
    S_of = np.repeat(np.arange(K), D)
    L_of = np.tile(np.arange(D), K)
    env_rng = np.random.default_rng(E._h("env", sid, seed))
    pick_rng = np.random.default_rng(E._h("pick", sid, seed))
    sel = ExperimentSelector(default_terms(info_gain=1.0, effort=0.01), epsilon=epsilon,
                             seed=E._h("sel", sid, seed) % 2 ** 31)

    def table(c, cmd):
        if cmd < K:
            hit = np.where((c + S_of) % K == cmd, 1 - slip, slip)
        elif cmd < K + n_tv:
            hit = np.full(K * D, 0.5)
        else:
            j = cmd - K - n_tv
            hit = np.where((L_of >> j) & 1 == 1, 1 - E.LAMP_EPS, E.LAMP_EPS)
        return np.stack([1 - hit, hit], 1)

    # ---- the post-fix library learner -------------------------------------------
    hs = HypothesisSet(floor=floor, max_hypotheses=K * D)
    for s in range(K):
        for L in range(D):
            hs.add_hypothesis(f"s{s}L{L}")
    q = ExperimentQuestion(hs, lambda n: n.split("L")[0], threshold=threshold,
                           confirm=confirm)
    scope = ("e10-lib-sim", f"{arm}-{seed}", f"{sid}")
    cue = int(env_rng.integers(K))
    first = conf_t = conf_ans = flag_t = None
    tv = lamp_n = 0
    stop_report = misspecified if stop_report is None else bool(stop_report)
    for t in range(1, budget + 1):
        cs = enumerate_candidates(spec, durations=(1,), max_candidates=min(64, n_act),
                                  max_horizon=1)
        vals = {}
        for cnd in cs:
            ig = q.information(table(cue, cnd.command))["info"] if arm != "random" else 0.0
            vals[cnd.candidate_id] = {"info_gain": ig, "usefulness": 0.0, "task": 0.0,
                                      "effort": effort_of(cnd), "risk": 0.0}
        status = q.status()
        if arm == "random":
            ch = cs.candidates[int(pick_rng.integers(len(cs)))]
            s_ = sel.select([ch], {ch.candidate_id: vals[ch.candidate_id]}, status=status)
        else:
            s_ = sel.select(cs, vals, status=status)
        cmd = int(s_.chosen.command)
        if cmd < K:
            o = int(env_rng.random() < (1 - slip if cmd == resp[cue] else slip))
        elif cmd < K + n_tv:
            o = int(env_rng.integers(2))
            tv += 1
        else:
            o = (lamp >> (cmd - K - n_tv)) & 1
            lamp_n += 1
        now = time.time()
        act = Action(*scope, t - 1, spec.spec_id, cmd, 1.0, now, t_complete=now)
        ev = [Observation(*scope, t, float(t), now, "outcome", "sensor",
                          {"value": np.int64(o)})]
        q.observe(table(cue, cmd), o, executed_action=act, evidence=ev)
        if cmd < K:
            cue = int(env_rng.integers(K))
        st = q.status()
        ps = q.answer_probs()
        if flag_t is None and st["misspecified"]:
            flag_t = t
        if conf_t is None and st["p_leading"] > threshold:
            conf_t, conf_ans = t, st["leading"]
        if not misspecified and first is None and ps.get(f"s{shift}", 0.0) > threshold:
            first = t
            if not stop_report:
                break
        if stop_report and st["state"] == RESOLVED:
            break
    st = q.status()
    reported = st["state"] == RESOLVED
    reported_wrong = reported and (misspecified or st["answer"] != f"s{shift}")
    post_wrong = conf_t is not None and (misspecified or conf_ans != f"s{shift}")
    n = t
    return {"interactions_to_identify": first if first is not None else budget + 1,
            "censored": first is None and not misspecified,
            "confident_wrong_unflagged": float(reported_wrong),
            "posterior_confident_wrong": float(post_wrong),
            "flag_before_confidence": float(flag_t is not None and
                                            (conf_t is None or flag_t <= conf_t)),
            "confident": float(conf_t is not None),
            "time_to_confidence": conf_t if conf_t is not None else budget + 1,
            "ppc_flagged": float(flag_t is not None),        # the LIBRARY flag here
            "final_state": st["state"],
            "interactions_to_resolved": q.resolved_at if reported else budget + 1,
            "tv_fraction": tv / n, "lamp_fraction": lamp_n / n, "interactions": n,
            "scenario": (shift, lamp, tuple(resp))}


def lib_arm(arm: str, **kw):
    def run(seed, split):
        res = [lib_sim(arm, sid, seed, **kw) for sid in split.heldout]
        out = {k: float(np.mean([r[k] for r in res])) for k in
               ("interactions_to_identify", "censored", "confident_wrong_unflagged",
                "posterior_confident_wrong", "flag_before_confidence", "confident",
                "time_to_confidence", "ppc_flagged", "tv_fraction", "lamp_fraction",
               "interactions_to_resolved")}
        out["final_states"] = {s: sum(r["final_state"] == s for r in res)
                               for s in ("open", "provisional", "resolved", "misspecified")}
        out["per_scenario"] = [r["interactions_to_identify"] for r in res]
        out["interactions"] = int(sum(r["interactions"] for r in res))
        out["data_fp"] = fingerprint_arrays(
            np.array([[r["scenario"][0], r["scenario"][1]] + list(r["scenario"][2])
                      for r in res], float))
        return finite_metrics(out)
    return run


def original_prereg(tag: str):
    hits = sorted(glob.glob(os.path.join(HERE, "reports", "full", f"{tag}-*.json")))
    if len(hits) != 1:
        raise RuntimeError(f"expected one full report for {tag}, found {hits}")
    p, _ = load_report(hits[0])
    p.verify()                                  # the ORIGINAL frozen hash
    return p, hits[0]


LIB_ARMS = {
    "E10b": {"info_joint": lib_arm("info", n_tv=4)},
    "E10c": {"info_marginal": lib_arm("info", lamp_bits=4)},
    "E10d": {"info_wellspec": lib_arm("info", misspecified=False, stop_report=True),
             "info_misspec": lib_arm("info", misspecified=True),
             "random_misspec": lib_arm("random", misspecified=True)},
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = os.path.join(HERE, "reports", "postfix_selection")
    specs = {sp["tag"]: sp for sp in E.experiments("full")}
    parts, T0 = [], time.perf_counter()
    for tag in (argv or ["E10a", "E10b", "E10c", "E10d"]):
        sp = specs[tag]
        p, src = original_prereg(tag)
        assert p.name == sp["prereg"].name
        variants = [("original arms", dict(sp["arms"]))]
        if tag in LIB_ARMS:
            arms = dict(sp["arms"])
            arms.update(LIB_ARMS[tag])
            variants.append(("library arms", arms))
        for label, arms in variants:
            t0 = time.perf_counter()
            rep = run_ab(p, memo(arms, {}), splits=sp["splits"], out_dir=out_dir,
                         measure_memory=False)
            print(f"[{tag} / {label}] {rep.verdict['label']} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
            parts.append(f"## {tag} — {label} (original prereg "
                         f"`{p.frozen_hash[:12]}` from `{os.path.relpath(src, HERE)}`)\n\n"
                         + tables([(sp, rep)]))
    parts.append(E10e_table())
    parts.append(f"\nTotal wall {time.perf_counter() - T0:.0f}s\n")
    path = os.path.join(HERE, "generated_postfix_selection.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"tables -> {path}")


def E10e_table() -> str:
    from .run import idle_table
    return "## E10e (post-fix tracker)\n\n" + idle_table("full")


KEYS["E10d"] = KEYS["E10d"] + ("posterior_confident_wrong", "flag_before_confidence",
                                "interactions_to_resolved")

if __name__ == "__main__":
    main()
