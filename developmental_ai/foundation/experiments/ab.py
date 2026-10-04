"""The plan §7.2 learning-comparison harness: preregister, run, decide.

WHY THIS EXISTS
    CLAUDE.md §9: "The reward number has been wrong every single time." The
    project has repeatedly read a favourable number after the fact and
    called it progress. This harness makes the comparison an argument that
    was written down BEFORE the data existed, and refuses to give a verdict
    otherwise.

WHAT IT CLAIMS
    1. PREREGISTRATION IS FROZEN BEFORE RESULTS. A `Preregistration` holds
       the hypothesis, primary metric and direction, minimum effect `delta`,
       the comparison basis, resource budget, failure conditions, seeds and
       arm roles. `freeze()` stamps a content hash and a timestamp.
       `decide()` REFUSES (PreregistrationViolation) when
         - the content no longer matches its frozen hash (edited after
           freezing, e.g. `dataclasses.replace` or a hand-edited JSON);
         - the results were recorded under a different prereg hash;
         - the prereg was frozen AFTER any result was recorded (re-freezing
           an edited prereg gives a new, later timestamp, so it is caught by
           the order check even though its hash is self-consistent).
    2. EVERY ARM SEES THE SAME DATA. For each (seed, split) every arm gets
       the same immutable `DataSplit` (episode-level, from runtime.splits —
       adjacent frames never straddle it). Its fingerprint is recorded per
       run and `decide()` refuses a pair whose fingerprints differ.
    3. FAILURES ARE OUTCOMES. An arm that raises, returns malformed metrics,
       or exceeds the preregistered budget is recorded with its error and
       status; it is never skipped or retried. A candidate that fails more
       often than `max_candidate_failure_fraction` is REJECTED.
    4. RESOURCES ARE REPORTED per run: wall-clock seconds, the Python
       allocation peak (tracemalloc, reset per run) and the process RSS
       high-water mark (getrusage; monotone over the process, labelled so).
    5. EQUAL-INTERACTION AND EQUAL-TIME comparisons are both computed from
       per-run learning curves; the preregistered `basis` decides. A pair
       whose basis is unavailable does not count. At equal interactions a
       run that used exactly the common budget contributes its reported
       primary, never a later curve point at the same count (more compute
       on the same data is not more interactions; see _at_interactions).
    6. THE ABLATION SLOT IS MANDATORY in spirit: a prereg without an
       `ablation_arm` runs, but warns loudly (warnings + stderr + report).
       A prereg that names an ablation the caller did not supply is refused.
    7. NEGATIVE RESULTS ARE PRESERVED. Every run writes a JSON + markdown
       report under a unique name (never overwritten) and appends one line
       to `index.jsonl`, whatever the verdict.

THE DECISION RULE (verdict in {accept, reject, inconclusive})
    d_i = sign * (candidate_i - baseline_i) per paired (seed, split) unit,
    sign = +1 for "higher is better", -1 for "lower".
    CI = paired percentile bootstrap of mean(d), EXPANDED for small n by
    t_{n-1,.975}/1.96 * sqrt(n/(n-1)). At n = 3 the raw percentile bootstrap
    is roughly the range of three numbers and accepts a null difference far
    more often than 5% (tests/_foundation_ab_smoke.py measures it); the
    expansion brings it to t-interval coverage.
      fewer than 3 preregistered seeds  -> inconclusive (screening only)
      candidate failures over the limit -> reject
      fewer than 3 usable pairs         -> inconclusive
      CI_low > 0 and mean(d) >= delta   -> accept
      CI_high < delta                   -> reject (the effect is ruled out)
      otherwise                         -> inconclusive
    Three seeds are not a guarantee of reliability (plan §7.2.6); the report
    says so.

ARM CONTRACT
    arm(seed: int, split: DataSplit) -> dict with
      <primary_metric>  finite float                       (required)
      "curve"           list of {"interactions": int, "seconds": float,
                        "value": float}, both axes non-decreasing (optional;
                        needed for equal-interaction/-time with unequal use)
      "interactions"    int environment interactions used (optional)
      "model_updates"   int (optional)
      anything else     JSON-able, carried into the report
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
import tracemalloc
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..runtime.splits import (DEFAULT_HELDOUT_FRACTION, SPLIT_SALT,
                              assert_no_leak, partition, split_spec)

try:
    import resource as _resource
except ImportError:                       # pragma: no cover  (Windows)
    _resource = None

SCHEMA_VERSION = 1
DIRECTIONS = ("higher", "lower")
BASES = ("final", "equal_interactions", "equal_time")
VERDICTS = ("accept", "reject", "inconclusive")
MIN_SEEDS = 3
CI_LEVEL = 0.95
BUDGET_KEYS = ("max_wall_seconds", "max_py_alloc_mb", "max_interactions")
RUN_STATUSES = ("ok", "failed", "budget_exceeded")


class PreregistrationError(ValueError):
    """A malformed preregistration or a protocol misuse."""


class PreregistrationViolation(PreregistrationError):
    """The prereg was changed, or written, after results existed. No verdict."""


# ---------------------------------------------------------------- statistics
_T975 = ((1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
         (6, 2.447), (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228),
         (12, 2.179), (15, 2.131), (20, 2.086), (25, 2.060), (30, 2.042),
         (40, 2.021), (60, 2.000), (120, 1.980))


def t_quantile_975(df: int) -> float:
    """Two-sided 95% Student-t critical value (table + linear interpolation;
    no scipy dependency). df >= 1."""
    if isinstance(df, bool) or not isinstance(df, (int, np.integer)) or df < 1:
        raise ValueError(f"df must be an int >= 1, got {df!r}")
    df = int(df)
    if df >= 120:
        return 1.960 + (1.980 - 1.960) * 120.0 / df
    for (d0, t0), (d1, t1) in zip(_T975, _T975[1:]):
        if d0 <= df <= d1:
            return t0 + (t1 - t0) * (df - d0) / float(d1 - d0)
    return _T975[0][1]                     # pragma: no cover


def paired_bootstrap_ci(diffs: Sequence[float], n_boot: int, seed: int,
                        small_sample_correction: bool = True
                        ) -> Tuple[float, Optional[float], Optional[float]]:
    """(mean, lo, hi) of mean(diffs), 95% paired percentile bootstrap.
    lo/hi are None for n < 2 (no interval exists). Deterministic in seed."""
    d = np.asarray(list(diffs), dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("diffs must be a non-empty 1-D sequence")
    if not np.isfinite(d).all():
        raise ValueError("diffs must be finite")
    mean = float(d.mean())
    n = d.size
    if n < 2:
        return mean, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boots = d[idx].mean(axis=1)
    a = (1.0 - CI_LEVEL) / 2.0
    p_lo, p_hi = (float(x) for x in np.quantile(boots, [a, 1.0 - a]))
    if small_sample_correction:
        f = t_quantile_975(n - 1) / 1.959964 * math.sqrt(n / (n - 1.0))
        p_lo = mean - f * (mean - p_lo)
        p_hi = mean + f * (p_hi - mean)
    return mean, p_lo, p_hi


# ----------------------------------------------------------- preregistration
def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _req_str(name, v):
    if not isinstance(v, str) or not v.strip():
        raise PreregistrationError(f"{name} must be a non-empty string, got {v!r}")


@dataclass(frozen=True)
class Preregistration:
    name: str
    hypothesis: str
    primary_metric: str
    direction: str
    baseline_arm: str
    candidate_arm: str
    seeds: Tuple[int, ...]
    delta: float = 0.0
    basis: str = "equal_interactions"
    ablation_arm: Optional[str] = None
    alternative_arm: Optional[str] = None
    resource_budget: Dict[str, float] = field(default_factory=dict)
    failure_conditions: Tuple[str, ...] = ()
    max_candidate_failure_fraction: float = 0.0
    split_fingerprints: Tuple[str, ...] = ()
    n_bootstrap: int = 4000
    notes: str = ""
    frozen_at_ns: Optional[int] = None
    frozen_hash: Optional[str] = None

    def __post_init__(self):
        for nm in ("name", "hypothesis", "primary_metric", "baseline_arm",
                   "candidate_arm"):
            _req_str(nm, getattr(self, nm))
        if self.direction not in DIRECTIONS:
            raise PreregistrationError(f"direction must be one of {DIRECTIONS}")
        if self.basis not in BASES:
            raise PreregistrationError(f"basis must be one of {BASES}")
        seeds = tuple(self.seeds)
        if not seeds or any(isinstance(s, bool) or not isinstance(
                s, (int, np.integer)) for s in seeds):
            raise PreregistrationError("seeds must be a non-empty tuple of int")
        seeds = tuple(int(s) for s in seeds)
        if len(set(seeds)) != len(seeds):
            raise PreregistrationError(f"duplicate seeds {seeds}")
        object.__setattr__(self, "seeds", seeds)
        if not isinstance(self.delta, (int, float)) or isinstance(self.delta, bool) \
                or not math.isfinite(self.delta) or self.delta < 0:
            raise PreregistrationError("delta must be a finite float >= 0 "
                                       "(an improvement, after the direction sign)")
        object.__setattr__(self, "delta", float(self.delta))
        roles = [self.baseline_arm, self.candidate_arm]
        for nm in ("ablation_arm", "alternative_arm"):
            v = getattr(self, nm)
            if v is not None:
                _req_str(nm, v)
                roles.append(v)
        if len(set(roles)) != len(roles):
            raise PreregistrationError(f"arm roles must be distinct, got {roles}")
        b = dict(self.resource_budget)
        for k, v in b.items():
            if k not in BUDGET_KEYS:
                raise PreregistrationError(f"unknown budget key {k!r}; known {BUDGET_KEYS}")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or \
                    not math.isfinite(v) or v <= 0:
                raise PreregistrationError(f"budget {k} must be > 0, got {v!r}")
        object.__setattr__(self, "resource_budget",
                           {k: float(b[k]) for k in sorted(b)})
        fc = tuple(self.failure_conditions)
        for s in fc:
            _req_str("failure condition", s)
        object.__setattr__(self, "failure_conditions", fc)
        object.__setattr__(self, "split_fingerprints", tuple(self.split_fingerprints))
        f = self.max_candidate_failure_fraction
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not (0.0 <= f <= 1.0):
            raise PreregistrationError("max_candidate_failure_fraction must be in [0,1]")
        object.__setattr__(self, "max_candidate_failure_fraction", float(f))
        if isinstance(self.n_bootstrap, bool) or not isinstance(self.n_bootstrap, int) \
                or self.n_bootstrap < 200:
            raise PreregistrationError("n_bootstrap must be an int >= 200")
        if (self.frozen_at_ns is None) != (self.frozen_hash is None):
            raise PreregistrationError("frozen_at_ns and frozen_hash go together")

    # ---- identity ------------------------------------------------------
    @property
    def acceptance_rule(self) -> str:
        better = "higher" if self.direction == "higher" else "lower"
        return (f"accept iff {self.candidate_arm} beats {self.baseline_arm} on "
                f"{self.primary_metric} ({better} is better, basis={self.basis}) "
                f"by mean >= {self.delta:g} with the paired bootstrap 95% CI "
                f"(small-sample expanded) excluding 0, across >= {MIN_SEEDS} "
                f"seeds; reject iff the CI upper bound < {self.delta:g} or "
                f"candidate failures exceed "
                f"{self.max_candidate_failure_fraction:g}; else inconclusive")

    def content(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("frozen_hash")
        d["seeds"] = list(self.seeds)
        d["failure_conditions"] = list(self.failure_conditions)
        d["split_fingerprints"] = list(self.split_fingerprints)
        d["acceptance_rule"] = self.acceptance_rule
        d["schema_version"] = SCHEMA_VERSION
        return d

    def content_hash(self) -> str:
        return hashlib.sha256(_canon(self.content()).encode()).hexdigest()

    @property
    def frozen(self) -> bool:
        return self.frozen_hash is not None

    def freeze(self, clock_ns: Callable[[], int] = time.time_ns) -> "Preregistration":
        if self.frozen:
            raise PreregistrationError(
                "already frozen; an edited prereg is a NEW prereg: "
                "dataclasses.replace(p, ..., frozen_at_ns=None, "
                "frozen_hash=None).freeze()")
        p = dataclasses.replace(self, frozen_at_ns=int(clock_ns()), frozen_hash="x")
        object.__setattr__(p, "frozen_hash", p.content_hash())
        return p

    def verify(self) -> None:
        if not self.frozen:
            raise PreregistrationViolation(
                "prereg is not frozen; freeze() it before running anything")
        if self.content_hash() != self.frozen_hash:
            raise PreregistrationViolation(
                f"prereg {self.name!r} was modified after it was frozen "
                f"(hash {self.content_hash()[:12]} != frozen "
                f"{self.frozen_hash[:12]})")

    def to_dict(self) -> Dict[str, Any]:
        d = self.content()
        d["frozen_hash"] = self.frozen_hash
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Preregistration":
        d = dict(d)
        if d.pop("schema_version", None) != SCHEMA_VERSION:
            raise PreregistrationError("unsupported prereg schema_version")
        d.pop("acceptance_rule", None)
        for k in ("seeds", "failure_conditions", "split_fingerprints"):
            d[k] = tuple(d.get(k, ()))
        return cls(**d)


# ---------------------------------------------------------------- data split
@dataclass(frozen=True)
class DataSplit:
    """An immutable episode-level split. Arms read `dev` / `heldout`."""
    split_id: str
    scope: str
    dev: Tuple[str, ...]
    heldout: Tuple[str, ...]
    spec: Tuple[Tuple[str, Any], ...]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canon([self.split_id, self.scope, list(self.dev),
                                      list(self.heldout), list(self.spec)])
                              .encode()).hexdigest()


def make_split(scope: str, episode_ids: Sequence[Any],
               heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
               salt: str = SPLIT_SALT, split_id: Optional[str] = None) -> DataSplit:
    """Deterministic episode-level split (runtime.splits). Refuses an empty
    side: a comparison with no held-out episodes evaluates on training data."""
    ids = [str(e) for e in episode_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode ids")
    dev, held = partition([(scope, e) for e in ids], heldout_fraction, salt)
    if not dev or not held:
        raise ValueError(f"split of {len(ids)} episodes left an empty side "
                         f"(dev={len(dev)}, heldout={len(held)})")
    spec = split_spec(heldout_fraction, salt)
    return DataSplit(split_id or f"{scope}#{salt}",
                     scope, tuple(e for _, e in dev), tuple(e for _, e in held),
                     tuple(sorted((k, v) for k, v in spec.items())))


# ------------------------------------------------------------------ outcomes
@dataclass(frozen=True)
class RunOutcome:
    arm: str
    seed: int
    split_id: str
    split_fingerprint: str
    status: str
    primary: Optional[float]
    metrics: Dict[str, Any]
    curve: Tuple[Tuple[int, float, float], ...]
    interactions: Optional[int]
    model_updates: Optional[int]
    wall_seconds: float
    py_alloc_peak_mb: Optional[float]
    process_peak_rss_mb: Optional[float]
    error: Optional[str]
    t_recorded_ns: int

    def to_dict(self):
        d = dataclasses.asdict(self)
        d["curve"] = [list(p) for p in self.curve]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["curve"] = tuple(tuple(p) for p in d["curve"])
        return cls(**d)


def _jsonable(v, path="metrics"):
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"{path} is non-finite")
        return f
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist(), path)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, f"{path}[{i}]") for i, x in enumerate(v)]
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x, f"{path}.{k}") for k, x in v.items()}
    raise ValueError(f"{path}: {type(v).__name__} is not JSON-able")


def validate_metrics(m: Any, primary_metric: str):
    """-> (primary, curve, interactions, model_updates, other) or ValueError."""
    if not isinstance(m, Mapping):
        raise ValueError(f"arm must return a dict, got {type(m).__name__}")
    if primary_metric not in m:
        raise ValueError(f"primary metric {primary_metric!r} missing "
                         f"(got {sorted(m)})")
    pv = m[primary_metric]
    if isinstance(pv, bool) or not isinstance(pv, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(pv)):
        raise ValueError(f"primary metric must be a finite number, got {pv!r}")
    curve = []
    for i, p in enumerate(m.get("curve", ()) or ()):
        try:
            it, sec, val = int(p["interactions"]), float(p["seconds"]), float(p["value"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"curve[{i}] needs interactions/seconds/value") from None
        if it < 0 or sec < 0 or not math.isfinite(sec) or not math.isfinite(val):
            raise ValueError(f"curve[{i}] out of range: {p}")
        if curve and (it < curve[-1][0] or sec < curve[-1][1]):
            raise ValueError(f"curve[{i}] axes must be non-decreasing")
        curve.append((it, sec, val))
    inter = m.get("interactions")
    if inter is None and curve:
        inter = curve[-1][0]
    if inter is not None:
        if isinstance(inter, bool) or not isinstance(inter, (int, np.integer)) or inter < 0:
            raise ValueError(f"interactions must be an int >= 0, got {inter!r}")
        inter = int(inter)
    upd = m.get("model_updates")
    if upd is not None:
        if isinstance(upd, bool) or not isinstance(upd, (int, np.integer)) or upd < 0:
            raise ValueError(f"model_updates must be an int >= 0, got {upd!r}")
        upd = int(upd)
    other = {k: _jsonable(v, f"metrics.{k}") for k, v in m.items()
             if k not in (primary_metric, "curve", "interactions", "model_updates")}
    return float(pv), tuple(curve), inter, upd, other


def _rss_peak_mb() -> Optional[float]:
    if _resource is None:
        return None
    r = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0


def _run_one(prereg, arm_name, fn, seed, split, measure_memory, clock_ns):
    started_tm = False
    if measure_memory:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            started_tm = True
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    status, err = "ok", None
    primary, curve, inter, upd, other = None, (), None, None, {}
    try:
        out = fn(seed, split)
        primary, curve, inter, upd, other = validate_metrics(out, prereg.primary_metric)
    except Exception as e:                       # a failure is an outcome
        status = "failed"
        tb = traceback.format_exc().strip().splitlines()
        err = f"{type(e).__name__}: {e} | {' / '.join(tb[-3:])}"
    wall = time.perf_counter() - t0
    py_peak = None
    if measure_memory:
        py_peak = tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)
        if started_tm:
            tracemalloc.stop()
    if status == "ok":
        b = prereg.resource_budget
        over = []
        if "max_wall_seconds" in b and wall > b["max_wall_seconds"]:
            over.append(f"wall {wall:.3f}s > {b['max_wall_seconds']:g}s")
        if "max_py_alloc_mb" in b and py_peak is not None and py_peak > b["max_py_alloc_mb"]:
            over.append(f"py alloc {py_peak:.1f}MB > {b['max_py_alloc_mb']:g}MB")
        if "max_interactions" in b and inter is not None and inter > b["max_interactions"]:
            over.append(f"interactions {inter} > {b['max_interactions']:g}")
        if over:
            status, err = "budget_exceeded", "; ".join(over)
    return RunOutcome(arm_name, int(seed), split.split_id, split.fingerprint,
                      status, primary, other, curve, inter, upd, float(wall),
                      py_peak, _rss_peak_mb(), err, int(clock_ns()))


# ----------------------------------------------------------------- results
@dataclass
class ABResults:
    prereg_hash: str
    prereg: Dict[str, Any]
    arms: Tuple[str, ...]
    seeds: Tuple[int, ...]
    split_ids: Tuple[str, ...]
    runs: List[RunOutcome]
    started_at_ns: int
    finished_at_ns: int
    warnings: List[str]
    environment: Dict[str, str]

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "prereg_hash": self.prereg_hash,
                "prereg": self.prereg, "arms": list(self.arms),
                "seeds": list(self.seeds), "split_ids": list(self.split_ids),
                "runs": [r.to_dict() for r in self.runs],
                "started_at_ns": self.started_at_ns,
                "finished_at_ns": self.finished_at_ns,
                "warnings": list(self.warnings), "environment": self.environment}

    @classmethod
    def from_dict(cls, d):
        if d.get("schema_version") != SCHEMA_VERSION:
            raise PreregistrationError("unsupported results schema_version")
        return cls(d["prereg_hash"], d["prereg"], tuple(d["arms"]),
                   tuple(d["seeds"]), tuple(d["split_ids"]),
                   [RunOutcome.from_dict(r) for r in d["runs"]],
                   d["started_at_ns"], d["finished_at_ns"], list(d["warnings"]),
                   d["environment"])


def value_at(curve, axis: int, budget: float) -> Optional[float]:
    """Last curve value whose axis coordinate is <= budget (step function);
    None if the curve has no point that early."""
    v = None
    for p in curve:
        if p[axis] <= budget + 1e-12:
            v = p[2]
        else:
            break
    return v


def _at_interactions(r: RunOutcome, budget: float) -> Optional[float]:
    """The run's value at `budget` interactions.

    A run that used EXACTLY `budget` interactions is read at its reported
    primary: that is its value at that count. Fixed 2026-10-03 (independent
    A/B verifier, RESULTS_stage6_8.md bug 3): this used to be value_at's
    LAST curve point at <= budget, and when every point of a curve shares
    one interaction count (extra epochs on FIXED data add seconds, not
    interactions) that is the most-trained point, not the reported one —
    E1c printed a +118 "equal_interactions" difference that was really the
    480-epoch overfit MLP-128 vs the IN, while the 60-epoch runs that the
    primary reports differed by 0.45. A run that used MORE interactions is
    truncated with value_at (its state after the last update at that count)."""
    if r.interactions is not None and abs(r.interactions - budget) <= 1e-12:
        return r.primary
    return value_at(r.curve, 0, budget)


def _pair_value(ra: RunOutcome, rb: RunOutcome, basis: str):
    """(value_a, value_b, budget) on `basis`, or (None, None, reason)."""
    if basis == "final":
        return ra.primary, rb.primary, None
    axis = 0 if basis == "equal_interactions" else 1
    if not ra.curve or not rb.curve:
        if basis == "equal_interactions" and ra.interactions is not None \
                and ra.interactions == rb.interactions:
            return ra.primary, rb.primary, ra.interactions
        return None, None, f"no curve and unequal/unknown {basis.split('_')[1]}"
    budget = min(ra.curve[-1][axis], rb.curve[-1][axis])
    if axis == 0:
        va, vb = _at_interactions(ra, budget), _at_interactions(rb, budget)
    else:
        va, vb = value_at(ra.curve, axis, budget), value_at(rb.curve, axis, budget)
    if va is None or vb is None:
        return None, None, f"no curve point at common budget {budget:g}"
    return va, vb, budget


def compare_arms(results: ABResults, arm_a: str, arm_b: str, basis: str,
                 direction: str, n_boot: int, seed: int) -> Dict[str, Any]:
    """Paired comparison of arm_b over arm_a. Units are (seed, split)."""
    sign = 1.0 if direction == "higher" else -1.0
    by = {(r.arm, r.seed, r.split_id): r for r in results.runs}
    diffs, units, skipped = [], [], []
    for s in results.seeds:
        for sp in results.split_ids:
            ra, rb = by.get((arm_a, s, sp)), by.get((arm_b, s, sp))
            if ra is None or rb is None:
                skipped.append([s, sp, "run missing"])
                continue
            if ra.split_fingerprint != rb.split_fingerprint:
                raise PreregistrationViolation(
                    f"arms {arm_a}/{arm_b} saw DIFFERENT data for seed {s}, "
                    f"split {sp}; the comparison is void")
            if ra.status != "ok" or rb.status != "ok":
                skipped.append([s, sp, f"{arm_a}:{ra.status} {arm_b}:{rb.status}"])
                continue
            va, vb, budget = _pair_value(ra, rb, basis)
            if va is None:
                skipped.append([s, sp, budget])
                continue
            diffs.append(sign * (vb - va))
            units.append({"seed": s, "split": sp, "a": va, "b": vb,
                          "budget": budget, "diff": sign * (vb - va)})
    out = {"a": arm_a, "b": arm_b, "basis": basis, "n_pairs": len(diffs),
           "units": units, "skipped": skipped, "mean_diff": None,
           "ci_low": None, "ci_high": None}
    if diffs:
        m, lo, hi = paired_bootstrap_ci(diffs, n_boot, seed)
        out.update(mean_diff=m, ci_low=lo, ci_high=hi)
    return out


def _arm_summary(results: ABResults):
    out = {}
    for a in results.arms:
        rs = [r for r in results.runs if r.arm == a]
        ok = [r.primary for r in rs if r.status == "ok"]
        walls = [r.wall_seconds for r in rs]
        py = [r.py_alloc_peak_mb for r in rs if r.py_alloc_peak_mb is not None]
        out[a] = {
            "runs": len(rs), "ok": len(ok),
            "failed": sum(r.status == "failed" for r in rs),
            "budget_exceeded": sum(r.status == "budget_exceeded" for r in rs),
            "mean": float(np.mean(ok)) if ok else None,
            "std": float(np.std(ok, ddof=1)) if len(ok) > 1 else None,
            "interactions_total": sum(r.interactions or 0 for r in rs),
            "model_updates_total": sum(r.model_updates or 0 for r in rs),
            "wall_seconds_total": float(sum(walls)),
            "py_alloc_peak_mb_max": max(py) if py else None,
        }
    return out


def decide(prereg: Preregistration, results: ABResults) -> Dict[str, Any]:
    """The verdict, or PreregistrationViolation if the prereg cannot be
    trusted to predate the results."""
    prereg.verify()
    if results.prereg_hash != prereg.frozen_hash:
        raise PreregistrationViolation(
            f"results were recorded under prereg {results.prereg_hash[:12]}, "
            f"not {prereg.frozen_hash[:12]}")
    t_first = min([r.t_recorded_ns for r in results.runs] + [results.started_at_ns])
    if prereg.frozen_at_ns > t_first:
        raise PreregistrationViolation(
            f"prereg {prereg.name!r} was frozen at {prereg.frozen_at_ns} ns, "
            f"AFTER the first result ({t_first} ns); no verdict")
    seed = int(prereg.frozen_hash[:8], 16)
    comps = {b: compare_arms(results, prereg.baseline_arm, prereg.candidate_arm,
                             b, prereg.direction, prereg.n_bootstrap, seed)
             for b in BASES}
    secondary = {}
    for role in ("ablation_arm", "alternative_arm"):
        other = getattr(prereg, role)
        if other is not None and other in results.arms:
            secondary[role] = compare_arms(results, other, prereg.candidate_arm,
                                           prereg.basis, prereg.direction,
                                           prereg.n_bootstrap, seed)
    c = comps[prereg.basis]
    cand = [r for r in results.runs if r.arm == prereg.candidate_arm]
    cand_fail = sum(r.status != "ok" for r in cand)
    frac = cand_fail / float(len(cand)) if cand else 1.0
    reasons, qualifier = [], None
    if len(prereg.seeds) < MIN_SEEDS:
        verdict, qualifier = "inconclusive", "screening only"
        reasons.append(f"{len(prereg.seeds)} seed(s) < {MIN_SEEDS}: a screening "
                       f"run cannot accept or reject")
    elif frac > prereg.max_candidate_failure_fraction:
        verdict = "reject"
        reasons.append(f"candidate failed {cand_fail}/{len(cand)} runs "
                       f"(> {prereg.max_candidate_failure_fraction:g} allowed)")
    elif c["n_pairs"] < MIN_SEEDS:
        verdict = "inconclusive"
        reasons.append(f"only {c['n_pairs']} usable pair(s) on basis "
                       f"{prereg.basis} (< {MIN_SEEDS})")
    elif c["ci_low"] > 0 and c["mean_diff"] >= prereg.delta:
        verdict = "accept"
        reasons.append(f"mean improvement {c['mean_diff']:.4g} >= delta "
                       f"{prereg.delta:g}, CI [{c['ci_low']:.4g}, "
                       f"{c['ci_high']:.4g}] excludes 0")
    elif c["ci_high"] < prereg.delta:
        verdict = "reject"
        reasons.append(f"CI upper bound {c['ci_high']:.4g} < delta "
                       f"{prereg.delta:g}: the hypothesised effect is ruled out")
    else:
        verdict = "inconclusive"
        reasons.append(f"CI [{c['ci_low']:.4g}, {c['ci_high']:.4g}] with mean "
                       f"{c['mean_diff']:.4g} neither meets nor rules out "
                       f"delta {prereg.delta:g}")
    signs = {b: (np.sign(comps[b]["mean_diff"]) if comps[b]["mean_diff"] is not None
                 else None) for b in BASES}
    notes = []
    known = [s for s in signs.values() if s is not None and s != 0]
    if len(set(known)) > 1:
        notes.append("equal-interaction / equal-time / final comparisons "
                     f"DISAGREE in sign: {signs}")
    if verdict != "accept":
        notes.append(f"criterion unmet: baseline {prereg.baseline_arm!r} is preserved")
    if len(prereg.seeds) <= 5:
        notes.append("few seeds: three seeds are not a guarantee of statistical "
                     "reliability (plan §7.2.6)")
    label = verdict + (f" ({qualifier})" if qualifier else "")
    return {"verdict": verdict, "qualifier": qualifier, "label": label,
            "reasons": reasons, "notes": notes,
            "acceptance_rule": prereg.acceptance_rule,
            "comparisons": comps, "secondary": secondary,
            "arms": _arm_summary(results),
            "candidate_failure_fraction": frac}


# ---------------------------------------------------------------- the runner
@dataclass
class ABReport:
    prereg: Preregistration
    results: ABResults
    verdict: Dict[str, Any]
    json_path: Optional[str] = None
    md_path: Optional[str] = None

    def to_dict(self):
        return {"prereg": self.prereg.to_dict(), "results": self.results.to_dict(),
                "verdict": self.verdict}

    def to_markdown(self) -> str:
        return format_markdown(self.prereg, self.results, self.verdict)


def run_ab(prereg: Preregistration, arms: Mapping[str, Callable],
           seeds: Optional[Sequence[int]] = None,
           splits: Any = None, out_dir: Optional[str] = None,
           measure_memory: bool = True,
           clock_ns: Callable[[], int] = time.time_ns) -> ABReport:
    """Run every arm on every (seed, split), record everything, decide."""
    if not isinstance(prereg, Preregistration):
        raise TypeError("prereg must be a Preregistration")
    prereg.verify()
    if seeds is None:
        seeds = prereg.seeds
    if tuple(int(s) for s in seeds) != prereg.seeds:
        raise PreregistrationViolation(
            f"seeds {tuple(seeds)} differ from the preregistered {prereg.seeds}; "
            f"seeds are chosen before results, not after")
    if splits is None:
        raise PreregistrationError("splits are required (DataSplit or list of them)")
    if isinstance(splits, DataSplit):
        splits = [splits]
    splits = list(splits)
    if not splits or not all(isinstance(s, DataSplit) for s in splits):
        raise PreregistrationError("splits must be DataSplit objects (make_split)")
    if len({s.split_id for s in splits}) != len(splits):
        raise PreregistrationError("duplicate split ids")
    for s in splits:
        assert_no_leak([(s.scope, e) for e in s.dev], [(s.scope, e) for e in s.heldout])
    if prereg.split_fingerprints and tuple(s.fingerprint for s in splits) \
            != prereg.split_fingerprints:
        raise PreregistrationViolation("splits differ from the preregistered ones")
    arms = dict(arms)
    for role in ("baseline_arm", "candidate_arm", "ablation_arm", "alternative_arm"):
        nm = getattr(prereg, role)
        if nm is not None and nm not in arms:
            raise PreregistrationError(f"prereg {role}={nm!r} but no such arm supplied")
        if nm is not None and not callable(arms[nm]):
            raise PreregistrationError(f"arm {nm!r} is not callable")
    warns = []
    if prereg.ablation_arm is None:
        msg = (f"NO ABLATION ARM in prereg {prereg.name!r}: plan §7.2.8 requires "
               f"an arm that removes the new component; without it an "
               f"improvement cannot be attributed to that component")
        warns.append(msg)
        warnings.warn(msg, stacklevel=2)
        print(f"WARNING: {msg}", file=sys.stderr)
    roles = {prereg.baseline_arm, prereg.candidate_arm, prereg.ablation_arm,
             prereg.alternative_arm}
    extra = sorted(a for a in arms if a not in roles)
    if extra:
        warns.append(f"exploratory arms not in the prereg (reported, never "
                     f"decisive): {extra}")
    order = list(arms)
    started = int(clock_ns())
    runs = []
    for si, s in enumerate(prereg.seeds):
        for sp in splits:
            k = si % len(order)            # rotate start: no arm always first
            for name in order[k:] + order[:k]:
                runs.append(_run_one(prereg, name, arms[name], s, sp,
                                     measure_memory, clock_ns))
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "platform": platform.platform()}
    res = ABResults(prereg.frozen_hash, prereg.to_dict(), tuple(order),
                    prereg.seeds, tuple(s.split_id for s in splits), runs,
                    started, int(clock_ns()), warns, env)
    rep = ABReport(prereg, res, decide(prereg, res))
    if out_dir is not None:
        rep.json_path, rep.md_path = write_report(rep, out_dir)
    return rep


def _unique(path_base: str) -> str:
    i = 0
    while True:
        p = path_base + (f"-{i}" if i else "")
        if not os.path.exists(p + ".json") and not os.path.exists(p + ".md"):
            return p
        i += 1


def write_report(rep: ABReport, out_dir: str) -> Tuple[str, str]:
    """Write JSON + markdown under a unique name (never overwrites) and
    append one line to index.jsonl. Negative results are kept the same way."""
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rep.prereg.name)[:60]
    base = _unique(os.path.join(out_dir, f"{safe}-{rep.prereg.frozen_hash[:10]}-"
                                         f"{rep.results.finished_at_ns}"))
    jp, mp = base + ".json", base + ".md"
    with open(jp, "x") as f:
        json.dump(rep.to_dict(), f, indent=2, sort_keys=True, allow_nan=False)
    with open(mp, "x") as f:
        f.write(rep.to_markdown())
    with open(os.path.join(out_dir, "index.jsonl"), "a") as f:
        f.write(_canon({"name": rep.prereg.name, "prereg_hash": rep.prereg.frozen_hash,
                        "verdict": rep.verdict["label"], "json": os.path.basename(jp),
                        "md": os.path.basename(mp),
                        "finished_at_ns": rep.results.finished_at_ns}) + "\n")
    return jp, mp


def load_report(json_path: str) -> Tuple[Preregistration, ABResults]:
    """Re-read a report so its verdict can be re-derived (decide()) later."""
    with open(json_path) as f:
        d = json.load(f)
    return Preregistration.from_dict(d["prereg"]), ABResults.from_dict(d["results"])


def _f(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def format_markdown(prereg: Preregistration, res: ABResults, v: Dict[str, Any]) -> str:
    L = [f"# A/B report: {prereg.name}", "",
         f"**Verdict: {v['label']}**", ""]
    L += [f"- {r}" for r in v["reasons"]] + [f"- note: {n}" for n in v["notes"]]
    L += [f"- WARNING: {w}" for w in res.warnings]
    L += ["", "## Preregistration", "",
          f"- hypothesis: {prereg.hypothesis}",
          f"- primary metric: `{prereg.primary_metric}` ({prereg.direction} is better)",
          f"- acceptance rule: {prereg.acceptance_rule}",
          f"- arms: baseline `{prereg.baseline_arm}`, candidate "
          f"`{prereg.candidate_arm}`, ablation `{prereg.ablation_arm}`, "
          f"alternative `{prereg.alternative_arm}`",
          f"- seeds: {list(prereg.seeds)}; splits: {list(res.split_ids)}",
          f"- resource budget: {prereg.resource_budget or 'none declared'}",
          f"- failure conditions: {list(prereg.failure_conditions) or 'none declared'}",
          f"- frozen: {prereg.frozen_at_ns} ns, hash `{prereg.frozen_hash}`", "",
          "## Arms", "",
          "| arm | ok/runs | failed | budget | mean | std | interactions | "
          "updates | wall s | py peak MB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a, s in v["arms"].items():
        L.append(f"| {a} | {s['ok']}/{s['runs']} | {s['failed']} | "
                 f"{s['budget_exceeded']} | {_f(s['mean'])} | {_f(s['std'])} | "
                 f"{s['interactions_total']} | {s['model_updates_total']} | "
                 f"{s['wall_seconds_total']:.3f} | {_f(s['py_alloc_peak_mb_max'], 3)} |")
    L += ["", "## Comparisons (candidate minus baseline, sign-adjusted)", "",
          "| basis | pairs | mean diff | CI low | CI high | skipped |",
          "|---|---|---|---|---|---|"]
    for b, c in list(v["comparisons"].items()) + [
            (f"vs {k}", c) for k, c in v["secondary"].items()]:
        mark = " (decisive)" if b == prereg.basis else ""
        L.append(f"| {b}{mark} | {c['n_pairs']} | {_f(c['mean_diff'])} | "
                 f"{_f(c['ci_low'])} | {_f(c['ci_high'])} | {len(c['skipped'])} |")
    L += ["", "## Per-run outcomes", "",
          "| arm | seed | split | status | primary | interactions | wall s | "
          "py peak MB | rss hwm MB | error |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.runs:
        err = (r.error or "").replace("|", "/")[:120]
        L.append(f"| {r.arm} | {r.seed} | {r.split_id} | {r.status} | "
                 f"{_f(r.primary)} | {_f(r.interactions)} | {r.wall_seconds:.3f} | "
                 f"{_f(r.py_alloc_peak_mb, 3)} | {_f(r.process_peak_rss_mb, 4)} | {err} |")
    L += ["", "rss hwm = process RSS high-water mark (monotone across runs); "
          "py peak = tracemalloc peak within the run.", ""]
    return "\n".join(L)


# -------------------------------------------------------------- demo fixture
def _episode_data(eid: str, n: int = 8):
    h = int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16)
    r = np.random.default_rng(h)
    x = r.normal(size=n)
    return x, 2.0 * x + 0.5 + r.normal(scale=0.5, size=n)


def _demo_arm(use_slope: bool, intercept: bool = True):
    def arm(seed: int, split: DataSplit):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(split.dev))
        xh = np.concatenate([_episode_data(e)[0] for e in split.heldout])
        yh = np.concatenate([_episode_data(e)[1] for e in split.heldout])
        xs, ys, curve, t0 = [], [], [], time.perf_counter()
        for k, i in enumerate(order):
            x, y = _episode_data(split.dev[i])
            keep = rng.random(x.size) < 0.6            # seed-dependent subsample
            xs.append(x[keep]), ys.append(y[keep])
            X, Y = np.concatenate(xs), np.concatenate(ys)
            if use_slope and X.size >= 2:
                A = np.stack([X, np.ones_like(X)], 1)
                w, b = np.linalg.lstsq(A, Y, rcond=None)[0]
            else:
                w, b = 0.0, (float(Y.mean()) if Y.size else 0.0)
            mse = float(np.mean((w * xh + b - yh) ** 2))
            curve.append({"interactions": k + 1,
                          "seconds": time.perf_counter() - t0, "value": mse})
        return {"heldout_mse": curve[-1]["value"], "curve": curve,
                "interactions": len(order), "model_updates": len(order)}
    return arm


def demo_experiment(seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> Dict[str, Any]:
    """A registered-experiment example: least squares vs a mean predictor on
    a synthetic linear task. `tools/ab_compare.py
    developmental_ai.foundation.experiments.ab:demo_experiment`."""
    split = make_split("demo:linear", [f"ep{i}" for i in range(40)], 0.25)
    prereg = Preregistration(
        name="demo-lstsq-vs-mean",
        hypothesis="a fitted slope lowers held-out MSE versus predicting the mean",
        primary_metric="heldout_mse", direction="lower", delta=0.5,
        baseline_arm="mean", candidate_arm="lstsq", ablation_arm="lstsq_no_slope",
        seeds=tuple(seeds), basis="equal_interactions",
        resource_budget={"max_wall_seconds": 30.0},
        failure_conditions=("any arm raises", "held-out MSE non-finite"),
        split_fingerprints=(split.fingerprint,)).freeze()
    return {"prereg": prereg,
            "arms": {"mean": _demo_arm(False), "lstsq": _demo_arm(True),
                     "lstsq_no_slope": _demo_arm(False)},
            "splits": [split]}
