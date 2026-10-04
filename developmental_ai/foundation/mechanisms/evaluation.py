"""Scoring mechanisms on held-out experience (plan Stage 6.5, §7.3).

Every score is computed on a COMMON OBSERVABLE TARGET (the next entity
state / event), never on latents, so different model families compare
(plan §7.3: "Do not compare arbitrary latent distances across models").

    one_step(pred, batch, ...)       RMSE, Gaussian NLL, CRPS, PIT histogram,
                                     50/90% interval coverage, event NLL /
                                     accuracy / Brier, contact vs no-contact
    horizons(pred, stream, H, ...)   the same per horizon h = 1..H over
                                     windows that the stream rule ACCEPTS;
                                     refused windows (reset splices) are
                                     counted and never scored; plus drift
    law_diagnostics(...)             geometry.constraints over predictions
    resource_cost(pred, batch)       params, FLOPs, train time, latency

Scores on a subset of outcome dims (e.g. balls, not static walls) via
cont_index / disc_index.

Coverage at level q is the fraction of PIT values in [(1-q)/2, (1+q)/2],
i.e. inside the central q interval of each dimension's marginal predictive.
It is exact for mixtures too (the mixture CDF is closed form).

CRPS: closed form for a single Gaussian; for a mixture, the sample
estimator E|X - y| - 0.5 E|X - X'| (unbiased-ish; n_samples controls
noise). Energy score is the multivariate analogue over the selected dims.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.special import ndtr

from ..contracts import ContractError
from ..geometry.constraints import ConstraintRegistry
from .api import MISSING, EntityBatch, MixedOutcome, TransitionBatch
from .data import EntityStream

LEVELS = (0.5, 0.9)


# ---- proper scores ---------------------------------------------------------

def gaussian_crps(mu, var, y) -> np.ndarray:
    sd = np.sqrt(var)
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return sd * (z * (2 * ndtr(z) - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def crps_samples(samples, y) -> np.ndarray:
    """samples (n, B, D), y (B, D) -> (B, D)."""
    s = np.sort(np.asarray(samples), axis=0)
    n = s.shape[0]
    a = np.abs(s - y[None]).mean(0)
    w = (2 * np.arange(1, n + 1) - n - 1).reshape((n,) + (1,) * (s.ndim - 1))
    b = 2.0 * (w * s).sum(0) / (n * n)          # E|X - X'| via order statistics
    return a - 0.5 * b


def energy_score(samples, y) -> np.ndarray:
    """samples (n, B, D), y (B, D) -> (B,)."""
    s = np.asarray(samples)
    a = np.linalg.norm(s - y[None], axis=-1).mean(0)
    b = sum(np.linalg.norm(s - s[i:i + 1], axis=-1).mean(0) for i in range(len(s))) / len(s)
    return a - 0.5 * b


def coverage(pit, levels: Sequence[float] = LEVELS) -> Dict[str, float]:
    p = np.asarray(pit).ravel()
    return {f"cov{int(round(q * 100))}": float(np.mean((p >= (1 - q) / 2) & (p <= (1 + q) / 2)))
            for q in levels}


def pit_histogram(pit, bins: int = 10) -> np.ndarray:
    h, _ = np.histogram(np.asarray(pit).ravel(), bins=bins, range=(0, 1))
    return h / max(h.sum(), 1)


# ---- distribution vs observed outcome -------------------------------------------

def score(dist: MixedOutcome, target: Mapping[str, np.ndarray], cont_index=None,
          disc_index=None, n_samples: int = 64,
          rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    d = dist.select(cont_index, disc_index)
    y = np.asarray(target["continuous"])
    if cont_index is not None:
        y = y[:, np.asarray(cont_index)]
    out: Dict[str, Any] = {"n": int(d.batch_size)}
    if d.layout.Dc:
        mu = d.mean()
        out["rmse"] = float(np.sqrt(((mu - y) ** 2).mean()))
        out["nll_cont"] = float(-d.log_prob_parts({"continuous": y})["continuous"].mean()
                                / d.layout.Dc)
        pit = d.cdf(y)
        out.update(coverage(pit))
        out["pit_hist"] = pit_histogram(pit)
        if d.n_components == 1:
            out["crps"] = float(gaussian_crps(d.means[0], d.vars[0], y).mean())
        else:
            rng = rng or np.random.default_rng(0)
            smp = d.sample(n_samples, rng)["continuous"]
            out["crps"] = float(crps_samples(smp, y).mean())
            out["energy"] = float(energy_score(smp, y).mean())
        within, between = d.decompose_variance()
        out["var_within"] = float(within.mean())
        out["var_between"] = float(between.mean())
    ev = target.get("discrete")
    if d.layout.G and ev is not None:
        k = np.asarray(ev)
        if disc_index is not None:
            k = k[:, np.asarray(disc_index)]
        obs = k != MISSING
        if obs.any():
            p = d.probs_marginal()
            pk = np.take_along_axis(p, np.where(obs, k, 0)[..., None], -1)[..., 0]
            out["nll_event"] = float(-np.log(np.maximum(pk[obs], 1e-12)).mean())
            out["event_acc"] = float((np.argmax(p, -1) == k)[obs].mean())
            onehot = np.eye(d.layout.K)[np.where(obs, k, 0)]
            out["brier"] = float(((p - onehot) ** 2).sum(-1)[obs].mean())
    return out


def one_step(pred, batch: TransitionBatch, cont_index=None, disc_index=None,
             split_contact: bool = True, **kw) -> Dict[str, Any]:
    """Score one-step predictions; with split_contact, also separately on
    transitions where any selected entity had an event vs none did."""
    dist = pred.predict(batch.state, batch.action, batch.dt)
    tgt = batch.target()
    res = {"all": score(dist, tgt, cont_index, disc_index, **kw)}
    if split_contact:
        ev = batch.events if disc_index is None else batch.events[:, np.asarray(disc_index)]
        known = np.all(ev != MISSING, 1)
        contact = known & np.any(ev > 0, 1)
        free = known & ~np.any(ev > 0, 1)
        for name, m in (("contact", contact), ("free", free)):
            if m.sum() >= 2:
                idx = np.flatnonzero(m)
                res[name] = score(dist.index(idx), {k: v[idx] for k, v in tgt.items()},
                                  cont_index, disc_index, **kw)
    return res


def horizons(pred, stream: EntityStream, horizon: int, cont_index=None,
             disc_index=None, mode: str = "mean", n_samples: int = 16,
             max_windows: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """Score h = 1..horizon over windows the stream rule accepts.

    Returns {"per_h": [score dicts], "windows": n, "refused": n refused,
    "drift": rmse[H] / rmse[1], "rmse_slope": least-squares slope per step}.
    """
    st = stream.windows(horizon)
    refused = stream.refused_windows(horizon)
    if len(st) == 0:
        raise ContractError(f"no valid {horizon}-step windows in the stream")
    rng = np.random.default_rng(seed)
    if max_windows is not None and len(st) > max_windows:
        st = np.sort(rng.choice(st, max_windows, replace=False))
    s0, acts, dts, targets = stream.window_batch(st, horizon)
    rl = pred.rollout(s0, acts, dts, mode=mode, n_samples=n_samples,
                      rng=rng if mode == "sample" else None)
    per_h = [score(rl[h], targets[h], cont_index, disc_index, rng=rng)
             for h in range(horizon)]
    r = np.array([p["rmse"] for p in per_h])
    slope = float(np.polyfit(np.arange(1, horizon + 1), r, 1)[0]) if horizon > 1 else 0.0
    return {"per_h": per_h, "windows": int(len(st)), "refused": int(refused),
            "drift": float(r[-1] / max(r[0], 1e-12)), "rmse_slope": slope}


# ---- laws -------------------------------------------------------------------------

def law_diagnostics(registry: ConstraintRegistry, states: Sequence[Mapping[str, Any]]
                    ) -> Dict[str, Dict[str, int]]:
    """Evaluate every registered constraint on every state dict; per-law
    status counts (satisfied / violated / inapplicable / unknown / error)."""
    out: Dict[str, Dict[str, int]] = {}
    for st in states:
        for name, dg in registry.evaluate(st).items():
            c = out.setdefault(name, {s: 0 for s in
                                      ("satisfied", "violated", "inapplicable",
                                       "unknown", "error")})
            c[dg.status] += 1
    return out


# ---- resources ----------------------------------------------------------------------

def resource_cost(pred, batch: TransitionBatch, repeats: int = 5) -> Dict[str, Any]:
    """params, FLOPs per sample (where countable), last fit's train seconds,
    and median predict latency per batch (ms)."""
    r = dict(pred.resources(batch.state, batch.action, batch.dt)
             if _accepts_batch(pred) else pred.resources())
    lat = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        pred.predict(batch.state, batch.action, batch.dt)
        lat.append(time.perf_counter() - t0)
    r["predict_ms"] = 1e3 * float(np.median(lat))
    r["batch"] = len(batch)
    r["train_seconds"] = float(getattr(pred, "last_fit", {}).get("train_seconds", np.nan))
    return r


def _accepts_batch(pred) -> bool:
    import inspect
    try:
        return len(inspect.signature(pred.resources).parameters) >= 3
    except (TypeError, ValueError):
        return False
