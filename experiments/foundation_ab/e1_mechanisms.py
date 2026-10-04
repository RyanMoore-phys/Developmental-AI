"""E1 — Stage 6 gate: relational Interaction Network vs same-input MLP.

QUESTION. Does the relational structure (plan Stage 6.2) improve held-out
prediction once EXTRA COMPUTE is taken out of the comparison? The mechanisms
agent measured the IN at ~15x the MLP's FLOPs/sample, so "IN beats MLP-128"
alone cannot be attributed to architecture (plan Stage 6 gate).

ARMS (all fit on the SAME arrays per (seed, split); see data_fp)
    in          InteractionNetwork(hidden=48), translation-invariant (declared)
    mlp_flops   MLPMechanism, 2 layers, hidden chosen so FLOPs/sample ~= the IN's
                (equal-FLOP, equal-epoch baseline)
    mlp128      MLPMechanism(hidden=128), the mechanisms agent's baseline, trained
                5x the epochs and curve-logged, so an EQUAL-WALL-CLOCK point exists
    in_noedges  ABLATION: IN with the edge model removed (aggregate = 0): a
                per-entity MLP over the IN's node features
    in_norel    exploratory ablation: IN with relative geometry zeroed on the edges
                (edges keep h_i, h_j only)

TRAINING. Every arm is ONE fit of `epochs` epochs (60; lr 3e-3, batch 64, as
the mechanisms smoke). mlp128 additionally fits FRESH models for 2x/4x/8x the
epochs; each fit's held-out 1-step NLL / RMSE is a curve point at that fit's
TRAIN seconds (eval excluded) -> the harness's equal-time comparison. (A first
draft resumed training in chunks; BaseMechanism.fit re-creates Adam on every
call, and the FLOP-matched MLP's loss ROSE across resets in the pilot, so
chunking was dropped as unfair to the wide MLP.)

METRICS (ball dims only; static walls excluded)
    nll1, rmse1, cov90_1, nll_event   one-step held-out
    rmse10                            10-step mean-mode rollout error
    nll10, cov90_10                   10-step sample-mode rollout (16 particles)
    rmse1_contact / rmse1_free        split by contact events
    flops_per_sample, params, train_seconds, predict_ms
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch

from developmental_ai.foundation import mechanisms as M
from developmental_ai.foundation.experiments import Preregistration, make_split
from developmental_ai.foundation.mechanisms._nets import count_flops, split_head

from .common import box_data, cached, finite_metrics, scale_key

FULL = dict(n_traj=36, T=80, epochs=60, long_factors=(2, 4, 8), lr=3e-3,
            batch_size=64, max_windows=200, n_samples=16, seeds=(11, 12, 13, 14, 15),
            threads=4, heldout=0.25)
SMOKE = dict(n_traj=16, T=40, epochs=5, long_factors=(4, 16, 64), lr=5e-4,
             batch_size=64, max_windows=20, n_samples=4, seeds=(11, 12, 13),
             threads=2, heldout=0.3)
TAG = "e1-box"


class AblatedIN(M.InteractionNetwork):
    """IN with a component removed. mode 'noedges': aggregate := 0 and the
    edge model is never called (so its FLOPs are not counted); 'norel':
    the relative-geometry edge input is zeroed. Mirrors relational._forward."""

    def __init__(self, *a, mode: str, **k):
        if mode not in ("noedges", "norel"):
            raise ValueError(mode)
        self.mode = mode
        super().__init__(*a, mechanism_id=f"ablation.in_{mode}", **k)

    def _forward(self, prep):
        m = self.module
        x = m.node_std(prep["node"])
        B, N, _ = x.shape
        h = m.enc(x)
        H = h.shape[-1]
        if self.mode == "noedges":
            agg = torch.zeros(B, N, H, dtype=h.dtype)
        else:
            r = torch.zeros_like(m.rel_std(prep["rel"]))
            e = m.edge(torch.cat([h[:, :, None].expand(B, N, N, H),
                                  h[:, None].expand(B, N, N, H), r], -1))
            mask = (~torch.eye(N, dtype=torch.bool)).to(e.dtype)[None, :, :, None]
            agg = (e * mask).sum(2)
        out = m.upd(torch.cat([h, agg, x], -1))
        D2 = 2 * self.D
        mu, var, _ = split_head(torch.cat([out[..., :D2].reshape(B, -1),
                                           out[..., D2:2 * D2].reshape(B, -1)], -1),
                                N * D2, 0, 0)
        return mu, var, out[..., 2 * D2:]


def _args(w):
    return (w.n_entities, 2, 5, 2, M.EVENT_CLASSES)


def flop_matched_hidden(w, in_hidden: int = 48) -> Dict[str, int]:
    """Smallest 2-layer MLP hidden size whose FLOPs/sample >= the IN's."""
    a = _args(w)
    rng = np.random.default_rng(0)
    tr = w.simulate(0, 4)
    st = M.EntityBatch(tr["pos"][:4], tr["vel"][:4],
                       np.repeat(tr["attrs"][None], 4, 0), "box_world")
    act, dt = tr["actions"][:4], tr["dts"][:4]
    inn = M.InteractionNetwork(*a, hidden=in_hidden, seed=0)
    f_in = inn.resources(st, act, dt)["flops_per_sample"]
    lo, hi = 16, 4096
    while lo < hi:
        mid = (lo + hi) // 2
        f = M.MLPMechanism(*a, hidden=mid, seed=0).resources(st, act, dt)["flops_per_sample"]
        if f >= f_in:
            hi = mid
        else:
            lo = mid + 1
    f_mlp = M.MLPMechanism(*a, hidden=lo, seed=0).resources(st, act, dt)["flops_per_sample"]
    del rng
    return {"hidden": lo, "in_flops": int(f_in), "mlp_flops": int(f_mlp)}


def _build(name, w, seed, h_match):
    a = _args(w)
    if name == "in":
        return M.InteractionNetwork(*a, hidden=48, seed=seed)
    if name == "in_noedges":
        return AblatedIN(*a, hidden=48, seed=seed, mode="noedges")
    if name == "in_norel":
        return AblatedIN(*a, hidden=48, seed=seed, mode="norel")
    if name == "mlp_flops":
        return M.MLPMechanism(*a, hidden=h_match, seed=seed, mechanism_id="baseline.mlp_flops")
    if name == "mlp128":
        return M.MLPMechanism(*a, hidden=128, seed=seed)
    raise KeyError(name)


def _compute(name, seed, split, sc, h_match) -> Dict[str, Any]:
    torch.set_num_threads(int(sc["threads"]))
    w = M.BoxWorld()
    data = box_data(w, split, seed, sc["T"], TAG)
    tr, held = data["train"].transitions(), data["heldout"].transitions()
    ci, di = w.ball_cont_index(), w.ball_disc_index()
    def fit(epochs):
        m = _build(name, w, seed, h_match)
        rep = m.fit(tr, epochs=epochs, lr=sc["lr"], batch_size=sc["batch_size"])
        if not rep["loss_last"] < rep["loss_first"]:
            raise RuntimeError(f"{name}@{epochs}ep: training loss did not fall "
                               f"{rep['loss_first']:.3f} -> {rep['loss_last']:.3f} "
                               f"(preregistered failure condition)")
        r = M.one_step(m, held, ci, di, split_contact=False)["all"]
        return m, rep, r

    # The reported model is ALWAYS the `epochs`-epoch single fit (equal epochs for
    # every arm). mlp128 additionally refits FRESH models at 2x, 4x, 8x the
    # epochs (no optimizer resets) to provide the equal-wall-clock curve.
    model, rep0, r0 = fit(sc["epochs"])
    secs, losses = rep0["train_seconds"], (rep0["loss_first"], rep0["loss_last"])
    updates = sc["epochs"] * int(np.ceil(len(tr) / sc["batch_size"]))
    pts = [(rep0["train_seconds"], r0)]
    if name == "mlp128":
        for f in sc["long_factors"]:
            _, rp, rr = fit(sc["epochs"] * f)
            pts.append((rp["train_seconds"], rr))
    pts.sort(key=lambda p: p[0])
    curve_nll = [{"interactions": len(tr), "seconds": s_, "value": r_["nll_cont"]}
                 for s_, r_ in pts]
    curve_rmse = [{"interactions": len(tr), "seconds": s_, "value": r_["rmse"]}
                  for s_, r_ in pts]
    one = M.one_step(model, held, ci, di)
    hm = M.horizons(model, data["heldout"], 10, ci, di, mode="mean",
                    max_windows=sc["max_windows"], seed=seed)
    hs = M.horizons(model, data["heldout"], 10, ci, di, mode="sample",
                    n_samples=sc["n_samples"], max_windows=sc["max_windows"], seed=seed)
    sub = held.subset(np.arange(min(64, len(held))))
    rc = M.resource_cost(model, sub)
    a = one["all"]
    out = {
        "nll1": a["nll_cont"], "rmse1": a["rmse"], "cov90_1": a["cov90"],
        "cov50_1": a["cov50"], "crps1": a["crps"], "nll_event": a.get("nll_event", 0.0),
        "rmse1_contact": one.get("contact", {}).get("rmse", float("nan")),
        "rmse1_free": one.get("free", {}).get("rmse", float("nan")),
        "rmse10": hm["per_h"][-1]["rmse"], "drift10": hm["drift"],
        "nll10": hs["per_h"][-1]["nll_cont"], "cov90_10": hs["per_h"][-1]["cov90"],
        "windows10": hm["windows"], "refused10": hm["refused"],
        "flops_per_sample": int(rc.get("flops_per_sample", 0)), "params": int(rc["params"]),
        "train_seconds": secs, "predict_ms_per64": rc["predict_ms"],
        "n_train": len(tr), "n_heldout": len(held), "epochs": sc["epochs"],
        "interactions": len(tr), "model_updates": updates, "data_fp": data["data_fp"],
        "torch_threads": int(sc["threads"]),
        "loss_first": losses[0], "loss_last": losses[1],
        "curve_points": len(pts),
        "_curve_nll1": curve_nll, "_curve_rmse1": curve_rmse,
    }
    if not np.isfinite(out["rmse1_contact"]):
        out["rmse1_contact"] = -1.0          # declared: no contact subset (tiny smoke data)
    if not np.isfinite(out["rmse1_free"]):
        out["rmse1_free"] = -1.0
    return finite_metrics({k: v for k, v in out.items() if not k.startswith("_")}) and out


def arm(name, metric, sc, h_match):
    curve_key = {"nll1": "_curve_nll1", "rmse1": "_curve_rmse1"}.get(metric)

    def fn(seed, split):
        r = cached((TAG, name, seed, split.fingerprint, scale_key(sc)),
                   lambda: _compute(name, seed, split, sc, h_match))
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        if curve_key is not None:
            out["curve"] = r[curve_key]
        return out
    return fn


ARMS = ("in", "mlp_flops", "mlp128", "in_noedges", "in_norel")


def experiments(scale: str = "full"):
    """-> list of (spec, note). ALL preregs are frozen here, before any arm runs."""
    sc = dict(FULL if scale == "full" else SMOKE)
    w = M.BoxWorld()
    fm = flop_matched_hidden(w)
    split = make_split("box2d:e1", [f"ep{k}" for k in range(sc["n_traj"])], sc["heldout"])
    common = dict(seeds=tuple(sc["seeds"]), split_fingerprints=(split.fingerprint,),
                  resource_budget={"max_wall_seconds": 300.0},
                  failure_conditions=("any arm raises", "non-finite metric",
                                      "loss does not fall"),
                  notes=f"scale={scale}; flop match {fm}; torch threads {sc['threads']}; "
                        "tracemalloc off (timing fidelity); arm results cached across "
                        "the E1 preregs, all of which were frozen before any arm ran")
    p_a = Preregistration(
        name="E1a-IN-vs-flopmatched-MLP-nll1",
        hypothesis="At equal FLOPs/sample, equal data and equal epochs, the relational IN "
                   "has lower held-out 1-step Gaussian NLL (ball dims) than an MLP on the "
                   "same structured inputs",
        primary_metric="nll1", direction="lower", delta=0.05, basis="final",
        baseline_arm="mlp_flops", candidate_arm="in", ablation_arm="in_noedges",
        alternative_arm="mlp128", **common).freeze()
    p_b = Preregistration(
        name="E1b-IN-vs-flopmatched-MLP-rmse10",
        hypothesis="At equal FLOPs/sample, the IN has lower 10-step mean-rollout RMSE "
                   "(ball dims) than the FLOP-matched MLP",
        primary_metric="rmse10", direction="lower", delta=0.05, basis="final",
        baseline_arm="mlp_flops", candidate_arm="in", ablation_arm="in_noedges",
        alternative_arm="mlp128", **common).freeze()
    p_c = Preregistration(
        name="E1c-IN-vs-MLP128-equal-time-nll1",
        hypothesis="At equal TRAINING WALL-CLOCK, the IN has lower held-out 1-step NLL "
                   "than the mechanisms agent's MLP-128 given as many epochs as fit",
        primary_metric="nll1", direction="lower", delta=0.05, basis="equal_time",
        baseline_arm="mlp128", candidate_arm="in", ablation_arm="in_noedges",
        alternative_arm="mlp_flops", **common).freeze()
    specs = []
    for p in (p_a, p_b, p_c):
        arms = {n: arm(n, p.primary_metric, sc, fm["hidden"]) for n in ARMS}
        specs.append({"prereg": p, "arms": arms, "splits": [split], "scale": sc,
                      "flop_match": fm})
    return specs
