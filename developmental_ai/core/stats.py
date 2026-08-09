"""Statistics + results-artifact helpers (July 2026 audit H4).

The audit found (a) no confidence intervals anywhere except the Hopper
ablation, violating the project's own evidence rule #2, and (b) only one raw
results file in the repo — headline numbers lived solely as ROADMAP prose.
New harnesses (and retrofits) should use these two helpers:

    from developmental_ai.core.stats import bootstrap_ci, save_results

    ci = bootstrap_ci([0.8, 0.9, 0.7])           # mean + 95% CI
    d  = bootstrap_diff_ci(arm_a, arm_b)          # difference A-B + 95% CI
    save_results("rung9_lawtransfer_results", {...})
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic=np.mean,
) -> Dict[str, float]:
    """Percentile-bootstrap confidence interval for a statistic of `values`.

    Returns {mean, lo, hi, n}. With very small n (2-4 seeds, the project
    norm) the CI is honest about its own width — that is the point.
    """
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, vals.size, size=(n_boot, vals.size))
    stats = statistic(vals[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(statistic(vals)), "lo": float(lo),
            "hi": float(hi), "n": int(vals.size)}


def bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap CI for mean(a) - mean(b) (independent resampling per arm).
    The comparison is 'significant at alpha' iff the CI excludes 0 — report
    the interval either way; never report a bare mean difference."""
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n_a": int(a.size), "n_b": int(b.size)}
    rng = np.random.RandomState(seed)
    ia = rng.randint(0, a.size, size=(n_boot, a.size))
    ib = rng.randint(0, b.size, size=(n_boot, b.size))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(a.mean() - b.mean()), "lo": float(lo),
            "hi": float(hi), "n_a": int(a.size), "n_b": int(b.size),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def save_results(
    out_dir: str,
    results: Dict[str, Any],
    filename: str = "results.json",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a results artifact next to the harness (auditable, re-derivable).

    Adds a timestamp and any `extra` metadata (protocol notes, seed lists,
    reduced-protocol flags). Returns the written path."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    if extra:
        payload["meta"] = extra
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return path
