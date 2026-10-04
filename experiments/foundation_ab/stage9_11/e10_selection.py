"""E10 — Stage 10 (foundation/experiments selection) probed off the fixture it
was built on.

Implementers' claim: info-gain identifies the hidden shift in 5.48
interactions vs 10.50 for random (K=6, reward noise 0.35, one coin-flip TV).

  E10a  REPLICATION ON NEW PARAMETERS through the library's own path
        (fixture.make_arm / run_identification, unmodified): K=8 cues,
        reward noise 0.5 (slip ~0.16), budget 80.
  E10b  FOUR noisy TVs (K=6, noise 0.35).
  E10c  A LEARNABLE BUT USELESS distractor: a hidden 4-bit "lamp" L, four
        lamp actions each returning one bit of L deterministically. The
        hypothesis set is the JOINT (shift x lamp) — what a learner that
        models everything it can see would carry. Joint I(H;O) values a lamp
        query at log 2 = 0.69 nats against ~0.25 for a response. Candidate:
        info-gain on the TASK MARGINAL I(shift; O).
  E10d  MISSPECIFIED hypothesis set: the true cue->response map is a
        permutation agreeing with some shift on half the cues and with no
        shift overall (truth NOT in the set). Does selection flag it or
        converge confidently? Alternative arm adds a simple posterior-
        predictive check (PPC) written here, to show a flag is cheap.

E10b-d use a small simulator written here with the library's selection
pieces (enumerate_candidates, ExperimentSelector, default_terms,
hypothesis_information, HypothesisSet). The posterior update calls
HypothesisSet.update directly (the Observation/Action plumbing of
update_from_real_outcome is a provenance contract, tested by the
implementers' own suite, not a selection behaviour). Floor 1e-4 (the joint
set has up to 96 hypotheses; at the library's 1e-3 the shift marginal could
not exceed ~0.92). Metric: interactions until the shift-marginal posterior
on the truth first exceeds 0.9, censored at budget + 1.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List

import numpy as np

from developmental_ai.foundation.contracts import ActionSpec
from developmental_ai.foundation.experiments import (
    ExperimentSelector, Preregistration, RetentionTracker, default_terms, effort_of,
    enumerate_candidates, hypothesis_information, make_arm, make_split,
    slip_probability)
from developmental_ai.foundation.experiments.fixture import scenario as lib_scenario
from developmental_ai.foundation.inference import HypothesisSet

from ..common import finite_metrics

LAMP_EPS = 1e-3          # deterministic lamp outcome, never a hard zero


def _h(*parts) -> int:
    return int(hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class _Err:
    """Prediction-error / learning-progress statistics (same rule as
    fixture._ErrorStats: running-mean predictor, EMA squared error,
    optimistic init 0.25, LP = older-half minus newer-half mean error)."""

    def __init__(self, init=0.25, beta=0.3):
        self.mean, self.n, self.ema, self.hist = {}, {}, {}, {}
        self.init, self.beta = init, beta

    def update(self, k, y):
        m = self.mean.get(k, 0.5)
        e = (y - m) ** 2
        self.n[k] = self.n.get(k, 0) + 1
        self.mean[k] = m + (y - m) / self.n[k]
        self.ema[k] = (1 - self.beta) * self.ema.get(k, self.init) + self.beta * e
        self.hist.setdefault(k, []).append(e)
        del self.hist[k][:-30]

    def error(self, k):
        return self.ema.get(k, self.init)

    def progress(self, k):
        h = self.hist.get(k, [])
        if len(h) < 4:
            return 0.0
        j = len(h) // 2
        return max(0.0, float(np.mean(h[:j]) - np.mean(h[j:])))


def run_sim(arm: str, sid: str, seed: int, K: int = 6, reward_noise: float = 0.35,
            n_tv: int = 1, lamp_bits: int = 0, misspecified: bool = False,
            budget: int = 60, threshold: float = 0.9, epsilon: float = 0.05,
            floor: float = 1e-4) -> Dict[str, Any]:
    r = np.random.default_rng(_h("e10", sid, seed))
    shift = int(r.integers(K))
    D = 2 ** lamp_bits
    lamp = int(r.integers(D))
    resp = [(c + shift) % K for c in range(K)]
    if misspecified:                    # half the cues follow `shift`, half do not
        half = sorted(r.choice(K, K // 2, replace=False).tolist())
        tgt = [(resp[c] + 1 + int(r.integers(K - 1))) % K for c in half]
        for c, t in zip(half, tgt):
            resp[c] = t
        for s in range(K):              # guarantee truth is not a shift
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
    stats = _Err()
    env_rng = np.random.default_rng(_h("env", sid, seed))
    pick_rng = np.random.default_rng(_h("pick", sid, seed))
    cue = int(env_rng.integers(K))
    first = None
    conf_t = None
    tv = lamp_n = 0
    ppc_sum = ppc_mu = ppc_var = 0.0
    flag_t = None

    def table(c, cmd):
        if cmd < K:
            hit = np.where((c + S_of) % K == cmd, 1 - slip, slip)
        elif cmd < K + n_tv:
            hit = np.full(K * D, 0.5)
        else:
            j = cmd - K - n_tv
            hit = np.where((L_of >> j) & 1 == 1, 1 - LAMP_EPS, LAMP_EPS)
        return np.stack([1 - hit, hit], 1)

    def key(c, cmd):
        return (c, cmd) if cmd < K else ("x", cmd)

    for t in range(1, budget + 1):
        cs = enumerate_candidates(spec, durations=(1,), max_candidates=min(64, n_act),
                                  max_horizon=1)
        p = hs.probs()
        ps = np.bincount(S_of, weights=p, minlength=K)
        vals = {}
        for cnd in cs:
            T = table(cue, cnd.command)
            if arm in ("info_joint", "info_ppc"):
                ig = hypothesis_information(p, T)["info"]
            elif arm == "info_marginal":
                Ts = np.stack([(p[S_of == s] @ T[S_of == s]) / max(ps[s], 1e-300)
                               for s in range(K)])
                ig = hypothesis_information(ps / ps.sum(), Ts / Ts.sum(1, keepdims=True))["info"]
            elif arm == "entropy":
                ig = hypothesis_information(p, T)["predictive_entropy"]
            elif arm == "icm":
                ig = stats.error(key(cue, cnd.command))
            elif arm == "lp":
                ig = stats.progress(key(cue, cnd.command))
            elif arm == "random":
                ig = 0.0
            else:
                raise ValueError(arm)
            vals[cnd.candidate_id] = {"info_gain": ig, "usefulness": 0.0, "task": 0.0,
                                      "effort": effort_of(cnd), "risk": 0.0}
        if arm == "random":
            ch = cs.candidates[int(pick_rng.integers(len(cs)))]
            s_ = sel.select([ch], {ch.candidate_id: vals[ch.candidate_id]})
        else:
            s_ = sel.select(cs, vals)
        cmd = int(s_.chosen.command)
        if cmd < K:
            truth_hit = cmd == resp[cue]
            o = int(env_rng.random() < (1 - slip if truth_hit else slip))
        elif cmd < K + n_tv:
            o = int(env_rng.integers(2))
            tv += 1
        else:
            o = (lamp >> (cmd - K - n_tv)) & 1
            lamp_n += 1
        T = table(cue, cmd)
        pred = p @ T                                   # posterior predictive
        lp_o = math.log(max(pred[o], 1e-300))
        e_lp = float(np.sum(pred * np.log(np.maximum(pred, 1e-300))))
        ppc_sum += lp_o
        ppc_mu += e_lp
        ppc_var += float(np.sum(pred * (np.log(np.maximum(pred, 1e-300)) - e_lp) ** 2))
        if flag_t is None and ppc_var > 0 and t >= 10 and \
                (ppc_sum - ppc_mu) / math.sqrt(ppc_var) < -3.0:
            flag_t = t
        hs.update(np.log(T[:, o]))
        stats.update(key(cue, cmd), float(o))
        if cmd < K:
            cue = int(env_rng.integers(K))
        ps = np.bincount(S_of, weights=hs.probs(), minlength=K)
        if conf_t is None and ps.max() > threshold:
            conf_t, conf_arg = t, int(np.argmax(ps))
        if not misspecified and first is None and ps[shift] > threshold:
            first = t
            break
        if misspecified and conf_t is not None:
            break
    flagged_first = flag_t is not None and (conf_t is None or flag_t <= conf_t)
    wrong_conf = conf_t is not None and (misspecified or conf_arg != shift)
    n = t
    return {"interactions_to_identify": first if first is not None else budget + 1,
            "censored": first is None and not misspecified,
            "confident_wrong_unflagged": float(wrong_conf and not
                                               (arm == "info_ppc" and flagged_first)),
            "confident": float(conf_t is not None),
            "time_to_confidence": conf_t if conf_t is not None else budget + 1,
            "ppc_flagged": float(flag_t is not None),
            "tv_fraction": tv / n, "lamp_fraction": lamp_n / n, "interactions": n,
            "scenario": (shift, lamp, tuple(resp))}


def sim_arm(arm: str, **kw):
    def run(seed, split):
        res = [run_sim(arm, sid, seed, **kw) for sid in split.heldout]
        out = {k: float(np.mean([r[k] for r in res])) for k in
               ("interactions_to_identify", "censored", "confident_wrong_unflagged",
                "confident", "time_to_confidence", "ppc_flagged", "tv_fraction",
                "lamp_fraction")}
        out["per_scenario"] = [r["interactions_to_identify"] for r in res]
        out["interactions"] = int(sum(r["interactions"] for r in res))
        from ..common import fingerprint_arrays
        out["data_fp"] = fingerprint_arrays(
            np.array([[r["scenario"][0], r["scenario"][1]] + list(r["scenario"][2])
                      for r in res], float))
        return finite_metrics(out)
    return run


def lib_arm(arm: str, n_cues: int, **kw):
    inner = make_arm(arm, n_cues=n_cues, **kw)

    def run(seed, split):
        m = inner(seed, split)
        from ..common import fingerprint_arrays
        m["data_fp"] = fingerprint_arrays(np.array(
            [lib_scenario(s, seed, n_cues) for s in split.heldout], float))
        return finite_metrics(m)
    return run


# ------------------------------------------------- idle / forgetting probe
def idle_tracker_probe(seed: int, center: float = 0.5, anneal: bool = False,
                       n_eval: int = 2000, n_probes: int = 16, sd: float = 0.2
                       ) -> Dict[str, Any]:
    """Idle / stationary probe losses: N(center, sd) i.i.d. per evaluation
    (optionally with sd shrinking 3x -> 1x over time). forgetting.py claims
    NET progress has E = 0 on a stationary model (telescoping), and flags
    only real forget->relearn loops. Thresholds: known < 0.45, forgotten > 0.55."""
    rng = np.random.default_rng(seed)
    tr = RetentionTracker(known_below=0.45, forgotten_above=0.55)
    clipped, prev, first = 0.0, None, None
    for t in range(n_eval):
        amp = sd * (1.0 + 2.0 * (1 - t / n_eval)) if anneal else sd
        losses = {f"p{i}": float(center + amp * rng.normal()) for i in range(n_probes)}
        tr.observe(t, losses)
        if prev is not None:
            clipped += sum(max(0.0, prev[k] - v) for k, v in losses.items())
        else:
            first = losses
        prev = losses
    telescoped = sum(first[k] - prev[k] for k in first)
    return {"net_progress": tr.totals["progress"], "raw_gain": tr.totals["raw_gain"],
            "telescoped_first_minus_last": telescoped, "clipped_lp_total": clipped,
            "relearn_events": tr.totals["relearn_events"], "flagged_probes": len(tr.flags()),
            "n_probes": n_probes, "n_eval": n_eval}


# ------------------------------------------------------------- experiments
def experiments(scale: str = "full") -> List[Dict[str, Any]]:
    seeds = (0, 1, 2, 3, 4) if scale == "full" else (0, 1, 2)
    n_ids = 50 if scale == "full" else 20
    specs = []

    sp = make_split("e10:lib-k8", [f"sc{i}" for i in range(n_ids)])
    kw = dict(budget=80, reward_noise=0.5)
    specs.append({"prereg": Preregistration(
        name="E10a-replication-K8-noise0.5",
        hypothesis=("through the library's own fixture path with NEW parameters (K=8, reward "
                    "noise 0.5, budget 80), info-gain identifies the hidden shift in >= 1 "
                    "fewer real interactions than random"),
        primary_metric="interactions_to_identify", direction="lower", delta=1.0,
        baseline_arm="random", candidate_arm="info_gain", ablation_arm="entropy",
        alternative_arm="icm", seeds=seeds, basis="final",
        resource_budget={"max_wall_seconds": 60.0},
        failure_conditions=("any arm raises", "censored at budget+1=81, counted"),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
        a: lib_arm(a, 8, **kw) for a in ("info_gain", "random", "icm", "entropy", "lp")},
        "splits": [sp], "tag": "E10a"})

    sp = make_split("e10:tv4", [f"sc{i}" for i in range(n_ids)])
    specs.append({"prereg": Preregistration(
        name="E10b-four-noisy-TVs",
        hypothesis=("with FOUR coin-flip TVs (K=6, noise 0.35, budget 60), info-gain "
                    "identifies the shift in >= 1 fewer interactions than random"),
        primary_metric="interactions_to_identify", direction="lower", delta=1.0,
        baseline_arm="random", candidate_arm="info_joint", ablation_arm="entropy",
        alternative_arm="icm", seeds=seeds, basis="final",
        resource_budget={"max_wall_seconds": 60.0},
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
        a: sim_arm(a, n_tv=4) for a in ("info_joint", "random", "icm", "entropy", "lp")},
        "splits": [sp], "tag": "E10b"})

    sp = make_split("e10:lamp", [f"sc{i}" for i in range(n_ids)])
    specs.append({"prereg": Preregistration(
        name="E10c-learnable-useless-distractor",
        hypothesis=("with a deterministic 4-bit lamp the task does not need (joint shift x "
                    "lamp hypotheses), info-gain on the JOINT set wastes interactions on the "
                    "lamp: task-marginal info-gain identifies the shift in >= 1 fewer "
                    "interactions"),
        primary_metric="interactions_to_identify", direction="lower", delta=1.0,
        baseline_arm="info_joint", candidate_arm="info_marginal", ablation_arm="random",
        alternative_arm="icm", seeds=seeds, basis="final",
        resource_budget={"max_wall_seconds": 60.0},
        failure_conditions=("any arm raises",),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
        a: sim_arm(a, lamp_bits=4) for a in ("info_joint", "info_marginal", "random", "icm")},
        "splits": [sp], "tag": "E10c"})

    sp = make_split("e10:misspec", [f"sc{i}" for i in range(n_ids)])
    specs.append({"prereg": Preregistration(
        name="E10d-misspecified-hypothesis-set",
        hypothesis=("when the truth is NOT in the hypothesis set, info-gain selection "
                    "reaches > 0.9 posterior on a (necessarily wrong) shift with no flag in "
                    ">= 30 percentage points more scenarios than it reaches confident-wrong "
                    "when the set is well specified"),
        primary_metric="confident_wrong_unflagged", direction="higher", delta=0.3,
        baseline_arm="info_wellspec", candidate_arm="info_misspec",
        ablation_arm="random_misspec", alternative_arm="info_ppc_misspec",
        seeds=seeds, basis="final", resource_budget={"max_wall_seconds": 60.0},
        failure_conditions=("any arm raises",
                            "the alternative arm's PPC flag is the verifier's, not the "
                            "library's; it is reported to show feasibility only"),
        split_fingerprints=(sp.fingerprint,)).freeze(), "arms": {
        "info_wellspec": sim_arm("info_joint", misspecified=False),
        "info_misspec": sim_arm("info_joint", misspecified=True),
        "random_misspec": sim_arm("random", misspecified=True),
        "info_ppc_misspec": sim_arm("info_ppc", misspecified=True)},
        "splits": [sp], "tag": "E10d", "exploratory_data_differs": ["info_wellspec"]})
    return specs
