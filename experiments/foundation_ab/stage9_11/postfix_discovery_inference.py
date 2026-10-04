"""POST-FIX re-run of the stage 9-11 verifier experiments touched by the
2026-10-03 discovery/inference fixes (B1 search budget, F5 change detector,
F6/E10d misspecification). Written by the fixer, NOT the verifier.

    cd <repo> && PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.postfix_discovery_inference

1. The ORIGINAL E9 and E10 modules are re-run unchanged: their preregs are
   built and frozen by the verifier's own `experiments()`; arms are the
   verifier's code; only developmental_ai/ changed underneath. The frozen
   hash covers the freeze timestamp, so it differs per run by design; the
   prereg CONTENT was checked field by field against the verifier's saved
   reports/full/*.json — identical except frozen_at_ns / frozen_hash. Reports go
   to reports/postfix/ (never the verifier's reports/full/), tables to
   generated_postfix_discovery_inference.md. RESULTS.md and the verifier's
   generated_* files are not touched.

2. E10d's original arms never read a misspecification signal (there was no
   API); re-running them can only reproduce 100%. A NEW prereg,
   "E10d-postfix-misspecification-monitor", frozen here before its arms
   run, asks whether the library's monitor (HypothesisSet.update_categorical
   -> hs.misspecified) flags the same misspecified scenarios. Its simulator
   is the verifier's run_sim with exactly three changes, all marked
   POSTFIX: the update goes through update_categorical (same posterior),
   the run continues to the budget instead of stopping at confidence (a
   flag raised after confidence can only be seen if the run goes on), and
   `flagged` = hs.misspecified ever raised. Same scenarios, same seeds,
   same selection, same floor.

3. EXPLORATORY (not a verdict): E9b's slope 2 -> 3 arm with a 32-episode
   post-change horizon instead of 16, to separate "the detector does not
   fire" (F5, fixed) from "the 16-episode horizon is too short for search
   (8 episodes) + 2 judging windows (8 episodes) after any alarm later than
   the first post episode".
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from developmental_ai.foundation.contracts import ActionSpec
from developmental_ai.foundation.experiments import (
    ExperimentSelector, Preregistration, default_terms, effort_of, enumerate_candidates,
    hypothesis_information, make_split, run_ab, slip_probability)
from developmental_ai.foundation.inference import HypothesisSet

from ..common import finite_metrics, fingerprint_arrays
from . import e9_discovery, e10_selection, run as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports", "postfix")
_h = e10_selection._h
LAMP_EPS = e10_selection.LAMP_EPS


def run_sim_monitor(arm: str, sid: str, seed: int, K: int = 6, reward_noise: float = 0.35,
                    misspecified: bool = False, budget: int = 60, threshold: float = 0.9,
                    epsilon: float = 0.05, floor: float = 1e-4,
                    use_monitor: bool = True) -> Dict[str, Any]:
    """e10_selection.run_sim (n_tv=1, lamp_bits=0) + the library monitor."""
    n_tv, lamp_bits = 1, 0
    r = np.random.default_rng(_h("e10", sid, seed))
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
    hs = HypothesisSet(floor=floor, max_hypotheses=K * D)
    for s in range(K):
        for L in range(D):
            hs.add_hypothesis(f"s{s}L{L}")
    S_of = np.repeat(np.arange(K), D)
    L_of = np.tile(np.arange(D), K)
    sel = ExperimentSelector(default_terms(info_gain=1.0, effort=0.01), epsilon=epsilon,
                             seed=_h("sel", sid, seed) % 2 ** 31)
    env_rng = np.random.default_rng(_h("env", sid, seed))
    pick_rng = np.random.default_rng(_h("pick", sid, seed))
    cue = int(env_rng.integers(K))
    first = conf_t = flag_t = None
    conf_arg = None

    def table(c, cmd):
        if cmd < K:
            hit = np.where((c + S_of) % K == cmd, 1 - slip, slip)
        elif cmd < K + n_tv:
            hit = np.full(K * D, 0.5)
        else:
            j = cmd - K - n_tv
            hit = np.where((L_of >> j) & 1 == 1, 1 - LAMP_EPS, LAMP_EPS)
        return np.stack([1 - hit, hit], 1)

    for t in range(1, budget + 1):
        cs = enumerate_candidates(spec, durations=(1,), max_candidates=min(64, n_act),
                                  max_horizon=1)
        p = hs.probs()
        vals = {}
        for cnd in cs:
            T = table(cue, cnd.command)
            ig = hypothesis_information(p, T)["info"] if arm == "info_joint" else 0.0
            vals[cnd.candidate_id] = {"info_gain": ig, "usefulness": 0.0, "task": 0.0,
                                      "effort": effort_of(cnd), "risk": 0.0}
        if arm == "random":
            ch = cs.candidates[int(pick_rng.integers(len(cs)))]
            s_ = sel.select([ch], {ch.candidate_id: vals[ch.candidate_id]})
        else:
            s_ = sel.select(cs, vals)
        cmd = int(s_.chosen.command)
        if cmd < K:
            o = int(env_rng.random() < (1 - slip if cmd == resp[cue] else slip))
        else:
            o = int(env_rng.integers(2))
        T = table(cue, cmd)
        if use_monitor:
            hs.update_categorical(np.log(T), o)              # POSTFIX: same posterior + monitor
        else:
            hs.update(np.log(T[:, o]))
        if flag_t is None and hs.misspecified:
            flag_t = t                                       # POSTFIX: library flag
        if cmd < K:
            cue = int(env_rng.integers(K))
        ps = np.bincount(S_of, weights=hs.probs(), minlength=K)
        if conf_t is None and ps.max() > threshold:
            conf_t, conf_arg = t, int(np.argmax(ps))
        if not misspecified and first is None and ps[shift] > threshold:
            first = t                                        # POSTFIX: no break, run on
    wrong_conf = conf_t is not None and (misspecified or conf_arg != shift)
    return {"interactions_to_identify": first if first is not None else budget + 1,
            "confident_wrong_unflagged": float(wrong_conf and flag_t is None),
            "confident_wrong": float(wrong_conf),
            "confident": float(conf_t is not None),
            "time_to_confidence": conf_t if conf_t is not None else budget + 1,
            "flagged": float(flag_t is not None),
            "time_to_flag": flag_t if flag_t is not None else budget + 1,
            "flag_before_confidence": float(flag_t is not None and
                                            (conf_t is None or flag_t <= conf_t)),
            "scenario": (shift, lamp, tuple(resp))}


def sim_arm(arm: str, **kw):
    def run(seed, split):
        res = [run_sim_monitor(arm, sid, seed, **kw) for sid in split.heldout]
        out = {k: float(np.mean([r[k] for r in res])) for k in
               ("confident_wrong_unflagged", "confident_wrong", "confident",
                "time_to_confidence", "flagged", "time_to_flag", "flag_before_confidence",
                "interactions_to_identify")}
        out["data_fp"] = fingerprint_arrays(
            np.array([[r["scenario"][0], r["scenario"][1]] + list(r["scenario"][2])
                      for r in res], float))
        return finite_metrics(out)
    return run


def postfix_specs():
    seeds = (0, 1, 2, 3, 4)
    n_ids = 50
    sp = make_split("e10:misspec", [f"sc{i}" for i in range(n_ids)])   # verifier's split
    pre = Preregistration(
        name="E10d-postfix-misspecification-monitor",
        hypothesis=("on the verifier's E10d misspecified scenarios (truth not a shift; budget "
                    "60), the library misspecification monitor (HypothesisSet."
                    "update_categorical -> misspecified) lowers the rate of confident-wrong-"
                    "UNFLAGGED scenarios by >= 0.5 against the same runs without it; "
                    "well-specified false flags reported as the alternative arm"),
        primary_metric="confident_wrong_unflagged", direction="lower", delta=0.5,
        baseline_arm="info_misspec_nomon", candidate_arm="info_misspec_mon",
        ablation_arm="random_misspec_mon", alternative_arm="info_wellspec_mon",
        seeds=seeds, basis="final", resource_budget={"max_wall_seconds": 120.0},
        failure_conditions=("any arm raises",
                            "well-specified arm flags in > 5% of scenarios"),
        split_fingerprints=(sp.fingerprint,),
        notes="fixer-written postfix prereg; verifier's run_sim + update_categorical, no "
              "early stop at confidence").freeze()
    return [{"prereg": pre, "arms": {
        "info_misspec_nomon": sim_arm("info_joint", misspecified=True, use_monitor=False),
        "info_misspec_mon": sim_arm("info_joint", misspecified=True),
        "random_misspec_mon": sim_arm("random", misspecified=True),
        "info_wellspec_mon": sim_arm("info_joint", misspecified=False)},
        "splits": [sp], "tag": "E10dpf", "exploratory_data_differs": ["info_wellspec_mon"]}]


R.KEYS["E10dpf"] = ("confident_wrong_unflagged", "confident_wrong", "flagged", "time_to_flag",
                    "time_to_confidence", "flag_before_confidence",
                    "interactions_to_identify")


def slope3_long_horizon(seeds=(0, 1, 2, 3, 4)) -> str:
    """EXPLORATORY: E9b full_slope3 with 16 (original) vs 32 post episodes."""
    sp = make_split("e9:nparam", [f"g{i}" for i in range(64)])
    L = ["## Exploratory: E9b slope 2 -> 3 at noise 0.5, post-change horizon 16 vs 32 "
         "episodes (not preregistered; no verdict)", "",
         "Same world/arm code as the verifier's `full_slope3` (reviser_arm('nparam', ALL, "
         "slope=3.0)); a 64-id split so 32 dev post episodes exist. The 16-episode column "
         "is NOT the prereg's run (different split size, so different episode ids).", "",
         "| seed | n_post | alarms | searches | replaced root | incumbent op | NLL robust | "
         "root NLL |", "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        for n_post in (16, 32):
            m = e9_discovery.reviser_arm("nparam", e9_discovery.ALL, n_post=n_post,
                                         slope=3.0)(s, sp)
            L.append(f"| {s} | {n_post} | {m['alarms']} | {m['searches']} | "
                     f"{bool(m['replaced_root'])} | {m['incumbent_op']} | "
                     f"{m['nll_robust']:.3f} | {m['root_nll_robust']:.3f} |")
    return "\n".join(L) + "\n"


def main(argv: List[str]):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    parts = ["# POSTFIX generated tables (discovery/inference fixes, 2026-10-03)", "",
             "Generated by `postfix_discovery_inference.py`; see "
             "POSTFIX_discovery_inference.md for the write-up.", ""]
    hashes = []
    for tag in ("E9", "E10"):
        reps = R.run_module(tag, "full", OUT)
        hashes += [(sp["prereg"].name, sp["prereg"].frozen_hash[:12]) for sp, _ in reps]
        parts.append(R.tables(reps))
    specs = postfix_specs()
    cache: Dict[Any, Any] = {}
    reps = []
    for sp in specs:
        rep = run_ab(sp["prereg"], R.memo(sp["arms"], cache), splits=sp["splits"],
                     out_dir=OUT, measure_memory=False)
        reps.append((sp, rep))
        hashes.append((sp["prereg"].name, sp["prereg"].frozen_hash[:12]))
    parts.append(R.tables(reps))
    parts.append(slope3_long_horizon())
    parts.insert(4, "Prereg content hashes this run: " +
                 ", ".join(f"{n} `{h}`" for n, h in hashes) + "\n")
    parts.append(f"\n(total {time.perf_counter() - t0:.0f} s)\n")
    with open(os.path.join(HERE, "generated_postfix_discovery_inference.md"), "w") as f:
        f.write("\n".join(parts))
    for n, h in hashes:
        print(n, h)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
