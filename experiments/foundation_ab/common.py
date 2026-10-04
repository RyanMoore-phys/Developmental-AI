"""Shared plumbing for the Stage 6-8 verification A/Bs (independent verifier).

Nothing here changes developmental_ai/: it only CALLS the foundation API.

    sim_seed        deterministic simulator seed from (tag, seed, episode id)
    box_data        BoxWorld trajectories for a DataSplit's dev / held-out
                    episodes -> EntityStreams + a fingerprint of the arrays
    cached          per-process memo so one arm computation can serve several
                    preregistrations that were ALL frozen before it ran
    finite_metrics  refuse NaN/inf in any numeric metric (a silent NaN is a
                    failure, never a number)

DATA IDENTITY. The harness fingerprints the split (episode ids). That alone
does not prove two arms saw the same ARRAYS, because data here is generated
from (seed, episode id). Every arm therefore also reports `data_fp`, a sha256
of the exact train / held-out arrays it consumed, and the smoke test asserts
it is identical across arms for each (seed, split).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Dict, Sequence

import numpy as np

from developmental_ai.foundation.mechanisms import EntityStream, trajectory_records
from developmental_ai.foundation.mechanisms.fixtures import FRAME

_CACHE: Dict[Any, Dict[str, Any]] = {}


def sim_seed(tag: str, seed: int, eid: str) -> int:
    return int(hashlib.sha256(f"{tag}|{seed}|{eid}".encode()).hexdigest()[:8], 16)


def _stream(trajs: Dict[str, Dict[str, np.ndarray]], eids: Sequence[str],
            stream: str) -> EntityStream:
    O, A, tw = [], [], 0.0
    for e in eids:
        o, a = trajectory_records(trajs[e], episode=e, stream=stream, t_wall0=tw)
        O += o
        A += a
        tw = o[-1].t_wall + 1.0
    return EntityStream(O, A, frame=FRAME)


def fingerprint_arrays(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def box_data(world, split, seed: int, T: int, tag: str, stream: str = "s0",
             episodes: str = "both", **sim_kw) -> Dict[str, Any]:
    """Simulate the split's episodes for this seed. `episodes` in
    {"both", "heldout"} (OOD sets only need the held-out side)."""
    sides = {"heldout": list(split.heldout)}
    if episodes == "both":
        sides["train"] = list(split.dev)
    out: Dict[str, Any] = {}
    fp_parts = []
    for side, eids in sides.items():
        trajs = {e: world.simulate(sim_seed(tag, seed, e), T, **sim_kw) for e in eids}
        st = _stream(trajs, eids, stream)
        out[side] = st
        for e in eids:
            tj = trajs[e]
            fp_parts += [tj["pos"], tj["vel"], tj["actions"], tj["dts"], tj["events"]]
    out["data_fp"] = fingerprint_arrays(*fp_parts)
    return out


def cached(key, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Compute once per process. The cached dict records `computed_at_ns`
    so a report can show it was computed AFTER every prereg was frozen."""
    if key not in _CACHE:
        import time
        r = fn()
        r["computed_at_ns"] = time.time_ns()
        _CACHE[key] = r
    return dict(_CACHE[key])


def clear_cache() -> None:
    _CACHE.clear()


def finite_metrics(d: Dict[str, Any], path: str = "metrics") -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, bool) or v is None or isinstance(v, str):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            if not math.isfinite(float(v)):
                raise FloatingPointError(f"{path}.{k} is non-finite ({v})")
        elif isinstance(v, dict):
            finite_metrics(v, f"{path}.{k}")
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, dict):
                    finite_metrics(x, f"{path}.{k}[{i}]")
                elif isinstance(x, (int, float, np.floating)) and not math.isfinite(float(x)):
                    raise FloatingPointError(f"{path}.{k}[{i}] is non-finite")
    return d


def scale_key(scale: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(scale, sort_keys=True, default=str)
                          .encode()).hexdigest()[:10]
