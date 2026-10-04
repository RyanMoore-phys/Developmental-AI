"""E2 — Stage 7 gate: does ensemble uncertainty have PREDICTIVE value?

Gate wording (plan Stage 7): "uncertainty has demonstrated predictive value
and can increase appropriately. Agreement among models alone is not accepted
as proof of correctness." So the test is error-vs-uncertainty on data the
models never saw, including data whose RULE changed.

DATA. BoxWorld(size 4, contacts ON, obs_noise 0.05) — deliberately NOT the
inference agent's setting (size 40, free flight only, noise 0.2). Training on
the split's dev episodes; evaluation on the held-out episodes simulated
  id        unchanged physics
  g4        gravity 9.81 -> 4.0   (rule change; states look familiar)
  g20       gravity 9.81 -> 20.0  (rule change)
  box6      box size 4 -> 6       (walls move: an input-distribution shift)

ARMS (MLPMechanism hidden 64 members, 60 epochs, lr 3e-3, batch 64)
    ens5        BootstrapEnsemble K=5, bootstrap over EPISODES (groups=)
                signal = epistemic variance (variance of member means)
    k1          ABLATION K=1: one MLP on all data; signal = its own predictive
                variance (the only uncertainty a single model has)
    null_perm   signal = ens5's epistemic, randomly permuted (same ensemble):
                the "no predictive value" reference
    ens5_noboot ensemble without resampling (seed diversity only)
    ens5_iid    exploratory: transition-level i.i.d. bootstrap

PRIMARY METRIC `unc_err_spearman`: Spearman rank correlation, over every
held-out transition pooled across id+g4+g20+box6, between the arm's
uncertainty signal (mean over ball dims) and its squared error (mean over
ball dims). Per-set correlations, coverage, NLL are reported beside it.

FALSIFICATION (reported, not a winner): "confidently wrong" =
signal <= the id median signal AND squared error >= the id 99th percentile.
On id this is <= ~0.5% by construction; any OOD set far above that is a
case where AGREEMENT COINCIDED WITH A LARGE ERROR.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from scipy.stats import spearmanr

from developmental_ai.foundation import mechanisms as M
from developmental_ai.foundation.experiments import Preregistration, make_split
from developmental_ai.foundation.inference import BootstrapEnsemble

from .common import box_data, cached, finite_metrics, fingerprint_arrays, scale_key

FULL = dict(n_traj=40, T=60, epochs=60, k=5, seeds=(21, 22, 23, 24, 25), threads=4,
            heldout=0.25, noise=0.05)
SMOKE = dict(n_traj=10, T=20, epochs=4, k=2, seeds=(21, 22, 23), threads=2,
             heldout=0.3, noise=0.05)
TAG = "e2-box"
SHIFTS = {"id": {}, "g4": {"gravity": 4.0}, "g20": {"gravity": 20.0}, "box6": {"size": 6.0}}


def _world(sc, size=4.0):
    return M.BoxWorld(obs_noise=sc["noise"], size=size)


def _datasets(seed, split, sc):
    out = {}
    base = box_data(_world(sc), split, seed, sc["T"], TAG)
    out["train"] = base["train"]
    fps = [base["data_fp"]]
    for name, sh in SHIFTS.items():
        if name == "id":
            out[name] = base["heldout"]
            continue
        w = _world(sc, sh.get("size", 4.0))
        kw = {"gravity": sh["gravity"]} if "gravity" in sh else {}
        d = box_data(w, split, seed, sc["T"], f"{TAG}-{name}", episodes="heldout", **kw)
        out[name] = d["heldout"]
        fps.append(d["data_fp"])
    out["data_fp"] = fingerprint_arrays(*[np.frombuffer(f.encode(), np.uint8) for f in fps])
    return out


def _groups(stream):
    idx = np.flatnonzero(stream.pair_valid)
    return np.array([stream.episode_of(int(i))[1] for i in idx])


def _fit_model(kind, seed, data, sc):
    w = _world(sc)
    args = (w.n_entities, 2, 5, 2, M.EVENT_CLASSES)
    tr = data["train"].transitions()
    fit_kw = dict(epochs=sc["epochs"], lr=3e-3, batch_size=64)
    if kind == "k1":
        m = M.MLPMechanism(*args, hidden=64, seed=1000 * seed, mechanism_id="mlp.k1")
        rep = m.fit(tr, **fit_kw)
        return m, rep["train_seconds"]
    mk = lambda i: M.MLPMechanism(*args, hidden=64, seed=1000 * seed + i,
                                  mechanism_id=f"mlp.m{i}")
    ens = BootstrapEnsemble(mk, k=sc["k"], seed=seed, bootstrap=(kind != "ens5_noboot"))
    groups = _groups(data["train"]) if kind in ("ens5", "ens5_noboot") else None
    rep = ens.fit(tr, groups=groups, **fit_kw)
    return ens, rep["train_seconds"]


def _evaluate(model, data, kind, seed):
    w = M.BoxWorld()
    ci, di = w.ball_cont_index(), w.ball_disc_index()
    sig, err, z2, per = {}, {}, {}, {}
    for name in SHIFTS:
        b = data[name].transitions()
        dist = model.predict(b.state, b.action, b.dt)
        within, between = dist.decompose_variance()
        mu = dist.mean()[:, ci]
        y = b.next_state.continuous()[:, ci]
        tot = (within + between)[:, ci]
        e = ((mu - y) ** 2).mean(1)
        s = between[:, ci].mean(1) if kind != "k1" else tot.mean(1)
        sig[name], err[name] = s, e
        z2[name] = ((mu - y) ** 2 / tot).mean(1)
        sc_ = M.one_step(model, b, ci, di, split_contact=False)["all"]
        per[name] = {"cov90": sc_["cov90"], "cov50": sc_["cov50"], "nll": sc_["nll_cont"],
                     "rmse": sc_["rmse"], "mean_signal": float(s.mean()),
                     "mean_epistemic": float(between[:, ci].mean()),
                     "mean_aleatoric": float(within[:, ci].mean()),
                     "applicability": float(np.mean(model.applicability(b.state))),
                     "n": int(len(b))}
    return sig, err, z2, per


def _corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0          # declared: a constant signal ranks nothing
    return float(spearmanr(a, b).correlation)


def _compute(kind, seed, split, sc) -> Dict[str, Any]:
    torch.set_num_threads(int(sc["threads"]))
    data = _datasets(seed, split, sc)
    base_kind = "ens5" if kind == "null_perm" else kind
    model, secs = _fit_model(base_kind, seed, data, sc)
    sig, err, z2, per = _evaluate(model, data, base_kind, seed)
    if kind == "null_perm":
        rng = np.random.default_rng(10_000 + seed)
        allsig = rng.permutation(np.concatenate([sig[n] for n in SHIFTS]))
        cut = np.cumsum([len(sig[n]) for n in SHIFTS])[:-1]
        sig = dict(zip(SHIFTS, np.split(allsig, cut)))
    S = np.concatenate([sig[n] for n in SHIFTS])
    E = np.concatenate([err[n] for n in SHIFTS])
    med_id, q99_id = np.median(sig["id"]), np.quantile(err["id"], 0.99)
    out: Dict[str, Any] = {"unc_err_spearman": _corr(S, E)}
    for n in SHIFTS:
        out[f"spearman_{n}"] = _corr(sig[n], err[n])
        out[f"cov90_{n}"] = per[n]["cov90"]
        out[f"nll_{n}"] = per[n]["nll"]
        out[f"rmse_{n}"] = per[n]["rmse"]
        out[f"signal_ratio_{n}"] = per[n]["mean_signal"] / max(per["id"]["mean_signal"], 1e-12) \
            if kind != "null_perm" else float(np.mean(sig[n]) / max(np.mean(sig["id"]), 1e-12))
        conf_wrong = (sig[n] <= med_id) & (err[n] >= q99_id)
        out[f"conf_wrong_frac_{n}"] = float(conf_wrong.mean())
        out[f"conf_wrong_n_{n}"] = int(conf_wrong.sum())
        out[f"outside3sd_frac_{n}"] = float((z2[n] > 9.0).mean())
        out[f"applicability_{n}"] = per[n]["applicability"]
        out[f"epistemic_{n}"] = per[n]["mean_epistemic"]
        out[f"aleatoric_{n}"] = per[n]["mean_aleatoric"]
        out[f"n_{n}"] = per[n]["n"]
    shifts = [n for n in SHIFTS if n != "id"]
    out["cov90_shift_abs_err"] = float(np.mean([abs(per[n]["cov90"] - 0.9) for n in shifts]))
    out["train_seconds"] = secs
    out["interactions"] = int(len(data["train"].transitions()))
    out["data_fp"] = data["data_fp"]
    out["members"] = 1 if kind == "k1" else int(sc["k"])
    return finite_metrics(out)


def arm(kind, sc):
    def fn(seed, split):
        return cached((TAG, kind, seed, split.fingerprint, scale_key(sc)),
                      lambda: _compute(kind, seed, split, sc))
    return fn


ARMS = ("ens5", "k1", "null_perm", "ens5_noboot", "ens5_iid")


def experiments(scale: str = "full"):
    sc = dict(FULL if scale == "full" else SMOKE)
    split = make_split("box2d:e2", [f"ep{k}" for k in range(sc["n_traj"])], sc["heldout"])
    common = dict(seeds=tuple(sc["seeds"]), split_fingerprints=(split.fingerprint,),
                  resource_budget={"max_wall_seconds": 300.0}, basis="final",
                  failure_conditions=("any arm raises", "non-finite metric"),
                  notes=f"scale={scale}; K={sc['k']}; shifts {SHIFTS}; tracemalloc off; "
                        "arm results cached across E2 preregs, all frozen before any run")
    p_a = Preregistration(
        name="E2a-epistemic-has-predictive-value",
        hypothesis="ens5 epistemic variance rank-correlates with held-out squared error "
                   "(pooled id + gravity/box shifts) by >= 0.1 more than a permuted "
                   "(information-free) signal",
        primary_metric="unc_err_spearman", direction="higher", delta=0.1,
        baseline_arm="null_perm", candidate_arm="ens5", ablation_arm="k1",
        alternative_arm="ens5_noboot", **common).freeze()
    p_b = Preregistration(
        name="E2b-epistemic-beats-single-model-variance",
        hypothesis="ens5 epistemic variance predicts error better (Spearman, pooled) than "
                   "a single model's own predictive variance (K=1)",
        primary_metric="unc_err_spearman", direction="higher", delta=0.05,
        baseline_arm="k1", candidate_arm="ens5", ablation_arm="ens5_noboot",
        alternative_arm="null_perm", **common).freeze()
    p_c = Preregistration(
        name="E2c-coverage-under-shift",
        hypothesis="the K=5 mixture's 90% intervals stay closer to nominal under shift "
                   "(mean |cov90-0.9| over g4, g20, box6) than K=1's by >= 0.02",
        primary_metric="cov90_shift_abs_err", direction="lower", delta=0.02,
        baseline_arm="k1", candidate_arm="ens5", ablation_arm="ens5_noboot",
        alternative_arm="ens5_iid", **common).freeze()
    specs = []
    for p in (p_a, p_b, p_c):
        specs.append({"prereg": p, "arms": {k: arm(k, sc) for k in ARMS},
                      "splits": [split], "scale": sc})
    return specs
