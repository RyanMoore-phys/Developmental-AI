"""E3 — Stage 8 gate: does tracker state improve a DOWNSTREAM outcome?

Downstream outcome: predict the next frame's detections (positions). The
perception agent reported (contract H, seeds 0-2, 6 bouncing entities,
clutter 1.0, miss 0.1) OSPA vs next detections: tracker 0.0399 vs best
per-frame 0.0506. This re-runs it on DIFFERENT seeds and fixture parameters,
plus a harder variant.

SCOPE CAVEAT (stated up front). IdentityTracker is a hand-specified
constant-velocity Kalman filter with Hungarian association; nothing in it is
LEARNED. So this tests "persistent tracked state vs per-frame features", not
the plan's "learned state improves a downstream outcome".

FIXTURE (re-implemented from tests/_foundation_perception_smoke.py::_bounce,
with knobs): N entities bounce in [-0.9, 0.9]^2, each frame each is detected
with prob 1-pmiss (+ an optional OCCLUDER band |x| < occ_halfwidth where
nothing is detected), Gaussian position noise, appearance feature = the
entity's prototype + noise; clutter ~ Poisson(clutter) uniform detections.
`n_protos` < N makes several entities share one prototype (look-alikes).

VARIANTS
    std    N=6, T=150, clutter 1.0, pmiss 0.1, speed .015-.035, feat noise .1,
           distinct appearance (the perception agent's fixture, new seeds)
    hard   N=10, T=150, clutter 2.0, pmiss 0.15, speed .025-.05, feat noise .3,
           3 shared appearance prototypes, occluder band |x| < 0.15

ARMS (all read the SAME frames; burn-in 20 frames is not scored for any arm)
    tracker      IdentityTracker, TrackerConfig(clutter_density = clutter/4)
    nnvel        BASELINE per-frame: current detections + velocity from a
                 Hungarian match to the previous frame
    static       per-frame: next = current detections
    tracker_noapp ABLATION: same tracker with appearance ignored (kappa 1e-6)

PRIMARY `ospa_obs` (OSPA, cutoff 0.1, vs the NEXT FRAME'S DETECTIONS: a
non-privileged target). `ospa_oracle` (vs true positions of entities visible
next frame) is PRIVILEGED and reported only as a diagnostic.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

from developmental_ai.foundation.contracts import Scope
from developmental_ai.foundation.experiments import Preregistration, make_split
from developmental_ai.foundation.perception import IdentityTracker, TrackerConfig

from .common import cached, finite_metrics, fingerprint_arrays, scale_key, sim_seed

F = 8
BURN = 20
VARIANTS = {
    "std": dict(N=6, T=150, clutter=1.0, pmiss=0.1, noise=0.01, speed=(.015, .035),
                fnoise=0.1, n_protos=None, occ=0.0),
    "hard": dict(N=10, T=150, clutter=2.0, pmiss=0.15, noise=0.01, speed=(.025, .05),
                 fnoise=0.3, n_protos=3, occ=0.15),
}
FULL = dict(n_scenes=16, heldout=0.5, seeds=(31, 32, 33, 34, 35), T=None)
SMOKE = dict(n_scenes=6, heldout=0.5, seeds=(31, 32, 33), T=45)
TAG = "e3-bounce"


def scene(rng, N, T, clutter, pmiss, noise, speed, fnoise, n_protos, occ):
    P = rng.uniform(-.8, .8, (N, 2))
    ang = rng.uniform(0, 2 * np.pi, N)
    sp = rng.uniform(speed[0], speed[1], N)
    V = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)
    if n_protos:
        protos = rng.normal(size=(n_protos, F))
        feats = protos[np.arange(N) % n_protos]
    else:
        feats = rng.normal(size=(N, F))
    fr, truth = [], []
    for _ in range(T):
        ps, fs, es = [], [], []
        for i in range(N):
            if rng.random() < pmiss or abs(P[i, 0]) < occ:
                continue
            ps.append(P[i] + rng.normal(0, noise, 2))
            fs.append(feats[i] + rng.normal(0, fnoise, F))
            es.append(i)
        for _ in range(rng.poisson(clutter)):
            ps.append(rng.uniform(-1, 1, 2))
            fs.append(rng.normal(size=F))
            es.append(-1)
        idx = rng.permutation(len(ps))
        fr.append((np.array(ps).reshape(-1, 2)[idx], np.array(fs).reshape(-1, F)[idx],
                   np.array(es, int)[idx]))
        truth.append(P.copy())
        P = P + V
        for i in range(N):
            for d in range(2):
                if abs(P[i, d]) > 0.9:
                    V[i, d] *= -1
                    P[i, d] = np.clip(P[i, d], -.9, .9)
    return fr, truth


def ospa(A, B, c=0.1):
    A, B = np.asarray(A).reshape(-1, 2), np.asarray(B).reshape(-1, 2)
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return c
    D = np.minimum(np.linalg.norm(A[:, None] - B[None], axis=-1), c)
    r, cc = linear_sum_assignment(D)
    return float((D[r, cc].sum() + c * abs(n - m)) / max(n, m))


def _scenes(variant, seed, split, sc):
    v = dict(VARIANTS[variant])
    if sc["T"]:
        v["T"] = sc["T"]
    out = {}
    for e in split.heldout:
        rng = np.random.default_rng(sim_seed(f"{TAG}-{variant}", seed, e))
        out[e] = scene(rng, **v)
    fp = fingerprint_arrays(*[np.concatenate([f[0].ravel() for f in out[e][0]])
                              for e in split.heldout])
    return out, fp, v


def _predict_run(kind, frames, v, ep):
    clut = v["clutter"] / 4.0
    if kind in ("tracker", "tracker_noapp"):
        cfg = TrackerConfig(clutter_density=clut,
                            feature_kappa=(1e-6 if kind == "tracker_noapp" else 4.0))
        tr = IdentityTracker(Scope("fixture", "s0", ep), cfg)
    preds, prev = [], None
    for t in range(len(frames) - 1):
        p = frames[t][0]
        if kind.startswith("tracker"):
            tr.step(p, frames[t][1])
            pred = np.array(list(tr.predict_positions().values())).reshape(-1, 2)
        elif kind == "static":
            pred = p.copy()
        elif kind == "nnvel":
            pred = p.copy()
            if prev is not None and len(prev) and len(p):
                D = np.linalg.norm(p[:, None] - prev[None], axis=-1)
                r, c = linear_sum_assignment(D)
                for a, b in zip(r, c):
                    pred[a] = p[a] + (p[a] - prev[b])
            prev = p
        else:
            raise KeyError(kind)
        preds.append(pred)
    return preds


def _compute(kind, variant, seed, split, sc) -> Dict[str, Any]:
    scenes, fp, v = _scenes(variant, seed, split, sc)
    obs, orc, n_pred, frames_scored, secs = [], [], [], 0, 0.0
    for e in split.heldout:
        fr, truth = scenes[e]
        t0 = time.perf_counter()
        preds = _predict_run(kind, fr, v, f"{variant}-{seed}-{e}")
        secs += time.perf_counter() - t0
        for t in range(BURN, len(fr) - 1):
            nxt = fr[t + 1][0]
            ids = fr[t + 1][2]
            vis = sorted(set(ids[ids >= 0].tolist()))
            tt = truth[t + 1][vis]
            obs.append(ospa(preds[t], nxt))
            orc.append(ospa(preds[t], tt))
            n_pred.append(len(preds[t]))
            frames_scored += 1
    return finite_metrics({
        "ospa_obs": float(np.mean(obs)), "ospa_oracle_PRIVILEGED": float(np.mean(orc)),
        "ospa_obs_p90": float(np.quantile(obs, 0.9)),
        "mean_predicted_count": float(np.mean(n_pred)),
        "frames_scored": frames_scored, "scenes": len(split.heldout),
        "ms_per_frame": 1e3 * secs / max(frames_scored + BURN * len(split.heldout), 1),
        "interactions": frames_scored, "data_fp": fp, "variant": variant,
    })


def arm(kind, variant, sc):
    def fn(seed, split):
        return cached((TAG, kind, variant, seed, split.fingerprint, scale_key(sc)),
                      lambda: _compute(kind, variant, seed, split, sc))
    return fn


ARMS = ("tracker", "nnvel", "static", "tracker_noapp")


def experiments(scale: str = "full"):
    sc = dict(FULL if scale == "full" else SMOKE)
    split = make_split("bounce:e3", [f"sc{k}" for k in range(sc["n_scenes"])], sc["heldout"])
    specs = []
    for variant in ("std", "hard"):
        p = Preregistration(
            name=f"E3{'a' if variant == 'std' else 'b'}-tracker-vs-perframe-{variant}",
            hypothesis=f"[{variant}] tracker state predicts next-frame detections with "
                       f"OSPA (c=0.1) lower by >= 0.005 than per-frame NN-velocity features",
            primary_metric="ospa_obs", direction="lower", delta=0.005, basis="final",
            baseline_arm="nnvel", candidate_arm="tracker", ablation_arm="tracker_noapp",
            alternative_arm="static", seeds=tuple(sc["seeds"]),
            split_fingerprints=(split.fingerprint,),
            resource_budget={"max_wall_seconds": 300.0},
            failure_conditions=("any arm raises", "non-finite metric"),
            notes=f"scale={scale}; variant {VARIANTS[variant]}; dev scenes unused "
                  f"(nothing is fitted; no tuning); tracemalloc off").freeze()
        specs.append({"prereg": p, "arms": {k: arm(k, variant, sc) for k in ARMS},
                      "splits": [split], "scale": sc})
    return specs
