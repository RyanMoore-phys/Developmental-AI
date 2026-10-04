"""Bounded, uncertainty-aware planning with an explicit fallback (plan Stage 11
items 1-2).

WHAT IS CLAIMED
    1. BoundedPlanner runs CEM (box or discrete ActionSpec) over a mechanism
       predictor's `rollout` (foundation.mechanisms BaseMechanism, or the
       inference.BootstrapEnsemble). Each candidate's imagined return is
       CUT where the predicted uncertainty u_h (default: EPISTEMIC variance =
       the between-member variance of the step-h mixture; a single model has
       none, so pass uncertainty="total" to use its full predictive variance)
       exceeds `epistemic_threshold`, and every counted step pays
       `uncertainty_penalty * u_h`.
    2. TRUNCATION IS NEVER A REWARD. A naive cut that simply stops summing
       makes a candidate that wanders into the unknown look better whenever
       rewards are negative (fewer negative terms). Steps past the cut are
       filled with `reward_floor` (declared, or the worst counted reward in
       the chunk) — so entering uncertainty is at best as good as the worst
       thing the model is confident about. If NO candidate has a confident
       first step, the plan is refused (the model knows nothing here).
    3. HARD WALL-CLOCK DEADLINE. Candidates are evaluated in chunks; before
       each chunk the planner checks `now + estimated_chunk_cost <= deadline`
       (the estimate is the MEDIAN of the last 16 measured chunk costs, so
       one heavy-tailed spike does not shut the planner for many calls).
       With the real clock (default) every model call runs on ONE worker
       thread and the planner WAITS for it only until the deadline: a single
       slow call can no longer carry the decision past it (verifier finding
       B3: a synchronous chunk overran 2x on 8.5% of decisions). An
       abandoned call cannot be killed in Python; it keeps the worker busy
       until it returns, and while it does the model is UNAVAILABLE (the
       next plan waits only as long as its own budget allows, then returns
       timed_out) — no thread leaks per timeout. A call still running
       after `hang_after_s` is declared HUNG (see the watchdog below). Inside a chunk the deadline is re-checked between
       horizon steps of the reward pass (cooperative cancel). On timeout the
       best plan of the completed iterations is returned; if none completed
       the result is `timed_out` and the controller falls back. Residual
       overrun = thread wake-up latency (sub-ms when the model call releases
       the GIL; up to the interpreter switch interval, 5 ms, for a model
       that spins in pure Python). With an injected clock (tests) calls run
       synchronously and a late chunk is discarded after the fact.
    4. PlanningController.decide() returns the planner's command only when
       the ReliabilityMonitor says the model has been validated against REAL
       outcomes recently (globally AND at the current state's region) and
       the planner produced an on-time, confident plan; otherwise it returns the existing learned controller's action
       (`fallback_policy(obs)`). Every decision is logged with source in
       SOURCES = {planner, fallback_timeout, fallback_unreliable} and a
       reason.
    5. ReliabilityMonitor scores the model's ONE-STEP prediction for the
       action actually executed (whoever chose it) against the real next
       state: z^2 = (y - mu)^2 / var (catches wrong-but-confident models)
       and, if a scale is declared, nmse = (y - mu)^2 / scale^2 (catches
       wrong-and-vague ones). Imagined or evaluator outcomes RAISE
       ProvenanceError: a model cannot certify itself (plan §7.4). A
       PredictiveCUSUM on the same residuals reports dynamics changes.
    6. RELIABILITY IS REGION-AWARE (verifier finding F7). A global window
       of recent z^2 forgets WHERE the model failed: fallback steps outside
       a bad region refill it, the gate reopens, and the planner walks back
       into the region it was wrong about (4.8 re-entries per 20 episodes).
       RegionMemory keeps every validation's input state with a pass/fail
       label (fail: z^2 > max_z2, or nmse > max_nmse) in bounded memory and
       scores a query state by kernel-weighted local evidence (standardised
       RMS distance, bandwidth `region_bandwidth`). A state is UNRELIABLE
       when the local failure weight >= `region_min_fail_weight` and the
       local failure fraction > `region_max_fail_frac`. The controller
       falls back while the CURRENT state is unreliable, and the planner
       CUTS each imagined rollout at the first step whose input state is
       unreliable (filled with the reward floor, exactly like the
       epistemic cut — entering a region the model failed in is never
       better than the worst confident outcome). A failure whose input
       state is ALREADY in a region marked unreliable is "explained" and
       does not enter the global window (it still updates the region and
       the CUSUM); failures anywhere new close the global gate as before.

WHY
    The scoreboard (CLAUDE.md §9) is full of reward numbers that were wrong.
    A planner that optimises a model is a machine for finding where the
    model is wrong in your favour ("model exploitation"). The only defence
    that does not rely on the model grading itself is validation against
    what actually happened, and the only safe action while that validation
    fails is the controller that already exists.

ESCAPE PATHS (CLAUDE.md §4.1 — every guard here can reopen)
    * Unreliable gate: validation runs on EVERY real step, including the
      fallback's, so the window keeps refilling. It reopens when `min_count`
      validations of the CURRENT model version pass. A refit bumps the
      model version and restarts the window, so a repaired model is judged
      on its own record, not its predecessor's.
    * Region gate — three reachable ways back, none needs a human:
        (a) the CURRENT model passes validation there again. Fallback steps
            are validated wherever they go, so the existing controller keeps
            sampling the region the planner may no longer enter;
        (b) a REFIT (model version bump) multiplies every older-version
            record by `region_stale_weight`. Default 0: a refit model is
            judged on its OWN record in every region, exactly like the
            global window — the old model's failures say nothing about the
            new one, and keeping them blocked the correctly refitted model
            out of every place the broken one had failed (smoke D: real late
            distance after the refit 0.155 -> 0.815 at weight 0.25). Cost:
            a refit that did NOT fix a region is re-entered once there. With
            0 < weight < 1 (for learners that refit very often) the region
            stays closed after a refit until the new model's own passes
            there outweigh it — reachable through fallback steps, which are
            validated wherever they go, and through (c);
        (c) every record decays with half-life `region_half_life`
            validations; a region nobody revisits returns to UNKNOWN, i.e.
            it defers to the global gate. Bounded memory (max_fails /
            max_passes FIFO) also forgets.
      tests/unit/test_foundation_planning_unit.py
      (region_memory_marks_failures_and_reopens) exercises all three.
    * Timeout gate: a planner whose cost estimate says "nothing fits" would
      never run again and never remeasure — a latch if the slowness was
      transient. After `reprobe_after` consecutive pre-emptive timeouts the
      next call runs one probe chunk regardless to refresh the estimate (in
      thread mode the probe is itself deadline-bounded).
    * Hung-model WATCHDOG (closed 2026-10-03; before it, one model call
      that never returned kept the worker busy forever, so the planner was
      unavailable for good and the fallback permanently in control — a
      guard-becomes-latch, CLAUDE.md §4.1). A call older than
      `hang_after_s` (default max(1 s, 20 x deadline_s)) is HUNG: its
      worker is marked POISONED and abandoned (its thread exits on its own
      if the call ever returns), a FRESH worker is spawned, and the next
      chunk is a forced PROBE that re-measures the model from scratch.
      Bounded: at most `max_respawns` poisoned workers may be alive at once
      (each is a thread stuck in the model), and successive respawns back
      off exponentially (`respawn_backoff_s` x 2^(k-1) after the k-th hang
      in a row, capped at MAX_RESPAWN_BACKOFF_S) — a model that hangs on
      every call costs at most 1 + max_respawns threads, while the fallback
      keeps control. REACHABLE RECOVERY CONDITION: the planner regains
      control as soon as ONE model call on any live worker returns —
      either the fresh worker's probe completes on time (a model that hung
      once and works now: back after ~hang_after_s), or a poisoned call
      returns, which frees its slot so a respawn is allowed again. A
      completed on-time call resets the backoff. The only state with no way
      back is a model on which EVERY call hangs forever — then the fallback
      is the correct controller, not a latch. PlanningController.observe
      waits for an in-flight call only until it would count as hung.
    * Epistemic refusal: depends only on the model; fitting more data where
      it is uncertain reopens it.

Offline/shadow only (plan §8): nothing here is wired into the live loop.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import ActionSpec, ContractError, UNKNOWN
from ..experience.errors import ProvenanceError
from ..inference.change import PredictiveCUSUM

SOURCES = ("planner", "fallback_timeout", "fallback_unreliable")
REAL_PROVENANCES = ("sensor", "inferred")
STATE_SCHEMA = 2                  # 2: planner cost history (median estimate)
RELIABILITY_SCHEMA = 2            # 2: + region memory (verifier finding F7)


def _check_real(provenance: str, what: str) -> None:
    if provenance not in REAL_PROVENANCES:
        raise ProvenanceError(
            f"{what}: provenance {provenance!r} is not a real outcome; only "
            f"{REAL_PROVENANCES} may validate a model or a skill (a model "
            f"cannot certify its own imagined outcomes; evaluator data never "
            f"reaches learning)")


def _model_version(model) -> int:
    v = getattr(model, "version", None)
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise ContractError(f"model needs an int `version`, got {v!r}")
    return int(v)


def _model_versions(model) -> Dict[str, int]:
    if hasattr(model, "model_versions"):
        return dict(model.model_versions())
    return {str(getattr(model, "mechanism_id", "model")): _model_version(model)}


# ======================================================================
# Reliability: real-outcome validation of the model
# ======================================================================

class _Ring:
    """Fixed-capacity FIFO of (x, version, t) records as preallocated arrays
    (the region query is on the planner's hot path: no per-call stacking)."""

    def __init__(self, cap: int):
        self.cap, self.n, self.head = int(cap), 0, 0
        self.X: Optional[np.ndarray] = None
        self.ver = np.zeros(self.cap, np.int64)
        self.t = np.zeros(self.cap, np.int64)

    def __len__(self) -> int:
        return self.n

    def add(self, x: np.ndarray, version: int, t: int) -> None:
        if self.X is None:
            self.X = np.zeros((self.cap, x.size))
        self.X[self.head], self.ver[self.head], self.t[self.head] = x, version, t
        self.head = (self.head + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def arrays(self):
        return self.X[:self.n], self.ver[:self.n], self.t[:self.n]

    def rows(self) -> List[List[Any]]:                     # chronological
        order = np.argsort(self.t[:self.n], kind="stable")
        return [[self.X[i].tolist(), int(self.ver[i]), int(self.t[i])] for i in order]

    def load(self, rows) -> None:
        self.n, self.head, self.X = 0, 0, None
        for x, v, t in rows:
            self.add(np.asarray(x, np.float64), int(v), int(t))


class RegionMemory:
    """WHERE real validation failed: bounded, decaying, kernel-weighted.

    Records are (input state x, pass/fail, model version, validation index
    t). Weights and kernel:
        w_j = 0.5 ** ((t_now - t_j) / half_life) * (1 if version_j is the
              current model version else stale_weight)
        k(a, b) = exp(-0.5 * |(a - b) / scale|^2 / (D * bandwidth^2))
    (scale = per-dimension std of every validated state, frozen between
    re-anchorings; a dim that never varied gets 0.1 x the median positive
    std, or 1). Every FAILURE record i keeps its local evidence
        F_i = sum_fail k(x_i, x_j) w_j      P_i = sum_pass k(x_i, x_j) w_j
    updated incrementally on each add (no full rescan on the planner's hot
    path), and is BAD while F_i / (F_i + P_i) > max_fail_frac — the model
    still fails more than it passes around there. A query state q is
    UNRELIABLE iff sum_{bad i} k(q, x_i) w_i >= min_fail_weight. Too little
    evidence is UNKNOWN, reported as "not unreliable" (the global gate and
    the epistemic cut still apply). Decay multiplies F_i and P_i alike, so
    it never flips BAD by itself; it shrinks the query weight to UNKNOWN."""

    RECOMPUTE_EVERY = 1024         # re-anchor: bounds float drift of the sums
    SCALE_DRIFT = 0.25             # re-anchor when the state scale moved > 25%

    def __init__(self, bandwidth: float = 0.25, half_life: float = 1000.0,
                 stale_weight: float = 0.0, max_fail_frac: float = 0.25,
                 min_fail_weight: float = 0.5, max_fails: int = 512,
                 max_passes: int = 4096):
        if not (bandwidth > 0 and half_life > 0 and min_fail_weight > 0):
            raise ContractError("bandwidth, half_life and min_fail_weight must be > 0")
        if not (0 <= stale_weight <= 1 and 0 <= max_fail_frac < 1):
            raise ContractError("need 0 <= stale_weight <= 1 and 0 <= max_fail_frac < 1")
        if int(max_fails) < 1 or int(max_passes) < 1:
            raise ContractError("max_fails and max_passes must be >= 1")
        self.bandwidth, self.half_life = float(bandwidth), float(half_life)
        self.stale_weight, self.max_fail_frac = float(stale_weight), float(max_fail_frac)
        self.min_fail_weight = float(min_fail_weight)
        self.max_fails, self.max_passes = int(max_fails), int(max_passes)
        self._reset()

    def _reset(self) -> None:
        self.fails, self.passes = _Ring(self.max_fails), _Ring(self.max_passes)
        self.acc_f = np.zeros(self.max_fails)
        self.acc_p = np.zeros(self.max_fails)
        self.t = 0
        self.version: Optional[int] = None
        self.dim: Optional[int] = None
        self._n, self._mean, self._m2 = 0, None, None
        self._scale_ref: Optional[np.ndarray] = None
        self._since_full = 0

    # ---- helpers ------------------------------------------------------------------
    def _vec(self, x) -> np.ndarray:
        x = np.asarray(x, np.float64).reshape(-1)
        if x.size < 1 or not np.all(np.isfinite(x)):
            raise ContractError("region state must be a finite, non-empty vector")
        if self.dim is not None and x.size != self.dim:
            raise ContractError(f"region state has {x.size} dims, memory holds {self.dim}")
        return x

    def scale(self) -> np.ndarray:
        sd = np.sqrt(self._m2 / max(self._n - 1, 1))
        pos = sd[sd > 0]
        floor = 0.1 * float(np.median(pos)) if pos.size else 1.0
        return np.maximum(sd, floor)

    def _w(self, ring: _Ring) -> np.ndarray:
        X, ver, tt = ring.arrays()
        return 0.5 ** ((self.t - tt) / self.half_life) * \
            np.where(ver == self.version, 1.0, self.stale_weight)

    def _k(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """(n, D), (m, D) -> (n, m) kernel under the anchored scale."""
        sc = self._scale_ref
        a, b = A / sc, B / sc
        d2 = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2.0 * a @ b.T, 0.0)
        return np.exp(-0.5 * d2 / (self.dim * self.bandwidth ** 2))

    def _full(self) -> None:
        """Re-anchor the scale and recompute every failure's local sums."""
        self._scale_ref = self.scale()
        self._since_full = 0
        nf = len(self.fails)
        if not nf:
            return
        Xf = self.fails.arrays()[0]
        self.acc_f[:nf] = self._k(Xf, Xf) @ self._w(self.fails)
        self.acc_p[:nf] = 0.0
        if len(self.passes):
            Xp = self.passes.arrays()[0]
            wp = self._w(self.passes)
            for i in range(0, len(Xp), 2048):
                self.acc_p[:nf] += self._k(Xf, Xp[i:i + 2048]) @ wp[i:i + 2048]

    # ---- updates ------------------------------------------------------------------
    def add(self, x, failed: bool, version: int) -> None:
        x = self._vec(x)
        if self.dim is None:
            self.dim = x.size
            self._mean, self._m2 = np.zeros(x.size), np.zeros(x.size)
        self._n += 1
        d = x - self._mean
        self._mean = self._mean + d / self._n
        self._m2 = self._m2 + d * (x - self._mean)
        version = int(version)
        full = self._scale_ref is None or version != self.version
        self.version = version
        self.t += 1
        nf = len(self.fails)
        self.acc_f[:nf] *= 0.5 ** (1.0 / self.half_life)       # sums are "as of now"
        self.acc_p[:nf] *= 0.5 ** (1.0 / self.half_life)
        if not full:
            sc = self.scale()
            full = bool(np.max(np.abs(sc / self._scale_ref - 1.0)) > self.SCALE_DRIFT) or \
                self._since_full >= self.RECOMPUTE_EVERY
        if full:                                   # the new record is folded in below
            ring = self.fails if failed else self.passes
            ring.add(x, version, self.t)
            self._full()
            return
        self._since_full += 1
        xb = x[None]
        if failed:
            slot, ring = self.fails.head, self.fails
            if len(ring) == ring.cap:              # evict the oldest failure
                e = ring.X[slot][None]
                we = self._w(ring)[slot]
                self.acc_f[:nf] -= self._k(ring.X[:nf], e)[:, 0] * we
            ring.add(x, version, self.t)
            nf = len(ring)
            Xf = ring.arrays()[0]
            kx = self._k(Xf, xb)[:, 0]
            self.acc_f[:nf] += kx                   # weight 1: newest, current version
            self.acc_f[slot] = float(kx @ self._w(ring))
            self.acc_p[slot] = 0.0
            if len(self.passes):
                self.acc_p[slot] = float(self._k(xb, self.passes.arrays()[0])[0] @
                                         self._w(self.passes))
        else:
            ring = self.passes
            if nf and len(ring) == ring.cap:       # evict the oldest pass
                slot = ring.head
                self.acc_p[:nf] -= self._k(self.fails.arrays()[0],
                                           ring.X[slot][None])[:, 0] * self._w(ring)[slot]
            ring.add(x, version, self.t)
            if nf:
                self.acc_p[:nf] += self._k(self.fails.arrays()[0], xb)[:, 0]

    # ---- queries ------------------------------------------------------------------
    def bad(self) -> np.ndarray:
        nf = len(self.fails)
        f, p = self.acc_f[:nf], self.acc_p[:nf]
        tot = f + p
        with np.errstate(invalid="ignore", divide="ignore"):
            return (tot > 0) & (f / np.where(tot > 0, tot, 1.0) > self.max_fail_frac)

    def assess(self, X, version: Optional[int] = None):
        """-> (unreliable (n,) bool, bad-failure weight (n,), local failure
        fraction at the nearest bad failure record (n,), NaN if none)."""
        Q = np.asarray(X, np.float64)
        if Q.ndim == 1:
            Q = Q[None]
        n = len(Q)
        nanv = np.full(n, np.nan)
        if self.dim is None or not len(self.fails):
            return np.zeros(n, bool), np.zeros(n), nanv
        if Q.ndim != 2 or Q.shape[1] != self.dim or not np.all(np.isfinite(Q)):
            raise ContractError(f"region queries must be finite (n, {self.dim}), got {Q.shape}")
        b = self.bad()
        w = self._w(self.fails)
        keep = b & (w > 1e-6)
        if not keep.any():
            return np.zeros(n, bool), np.zeros(n), nanv
        Xf = self.fails.arrays()[0][keep]
        wk = w[keep]
        tot = self.acc_f[:len(self.fails)][keep] + self.acc_p[:len(self.fails)][keep]
        frac = self.acc_f[:len(self.fails)][keep] / tot
        bw = np.empty(n)
        near = np.empty(n)
        for i in range(0, n, 2048):
            K = self._k(Q[i:i + 2048], Xf)
            bw[i:i + 2048] = K @ wk
            near[i:i + 2048] = frac[K.argmax(1)]
        return bw >= self.min_fail_weight, bw, near

    def active(self) -> bool:
        return bool(len(self.fails)) and bool(self.bad().any())

    # ---- persistence -----------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        L = lambda a: None if a is None else a.tolist()
        return {"t": self.t, "dim": self.dim, "n": self._n, "version": self.version,
                "mean": L(self._mean), "m2": L(self._m2),
                "fails": self.fails.rows(), "passes": self.passes.rows()}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self._reset()
        self.t, self.dim, self._n = int(d["t"]), d["dim"], int(d["n"])
        self.version = d["version"]
        self._mean = None if d["mean"] is None else np.asarray(d["mean"], np.float64)
        self._m2 = None if d["m2"] is None else np.asarray(d["m2"], np.float64)
        self.fails.load(d["fails"])
        self.passes.load(d["passes"])
        if self.dim is not None:
            self._full()


class ReliabilityMonitor:
    """Recent real-outcome validation error of ONE model, by model version,
    globally (a sliding window) and by REGION of state space (RegionMemory;
    `regions=False` gives the plain global gate).

    reliable() -> (bool, reason). Cold start (< min_count validations of the
    current version) is UNRELIABLE: an unvalidated model is not trusted.
    reliable_at(x) / region_unreliable(X): the region verdict at states."""

    def __init__(self, window: int = 20, min_count: int = 5, max_z2: float = 9.0,
                 max_nmse: Optional[float] = None, scale=None,
                 cusum: Optional[PredictiveCUSUM] = None, *, regions: bool = True,
                 region_bandwidth: float = 0.25, region_half_life: float = 1000.0,
                 region_stale_weight: float = 0.0, region_max_fail_frac: float = 0.25,
                 region_min_fail_weight: float = 0.5, region_max_fails: int = 512,
                 region_max_passes: int = 4096):
        if window < 1 or min_count < 1 or min_count > window:
            raise ContractError("need 1 <= min_count <= window")
        if not (np.isfinite(max_z2) and max_z2 > 0):
            raise ContractError("max_z2 must be finite and > 0")
        if (max_nmse is None) != (scale is None):
            raise ContractError("max_nmse and scale are declared together")
        self.window, self.min_count = int(window), int(min_count)
        self.max_z2 = float(max_z2)
        self.max_nmse = None if max_nmse is None else float(max_nmse)
        self.scale = None if scale is None else np.asarray(scale, np.float64)
        if self.scale is not None and np.any(self.scale <= 0):
            raise ContractError("scale must be > 0")
        self.cusum = cusum if cusum is not None else PredictiveCUSUM()
        self.model_version: Optional[int] = None
        self.z2: Deque[float] = deque(maxlen=self.window)
        self.nmse: Deque[float] = deque(maxlen=self.window)
        self.total = 0
        self.alarms: List[Dict[str, Any]] = []
        self.regions: Optional[RegionMemory] = RegionMemory(
            region_bandwidth, region_half_life, region_stale_weight, region_max_fail_frac,
            region_min_fail_weight, region_max_fails, region_max_passes) if regions else None

    def validate(self, mean, var, observed, *, provenance: str,
                 model_version: int, where=None) -> Dict[str, Any]:
        """`where` = the INPUT state of the validated prediction (the state
        the model predicted from); it is what the region memory is keyed on.
        Omitted -> global window only."""
        _check_real(provenance, "ReliabilityMonitor.validate")
        mu = np.asarray(mean, np.float64).reshape(-1)
        va = np.asarray(var, np.float64).reshape(-1)
        y = np.asarray(observed, np.float64).reshape(-1)
        if not (mu.shape == va.shape == y.shape):
            raise ContractError(f"mean/var/observed shapes differ: {mu.shape} "
                                f"{va.shape} {y.shape}")
        if np.any(va <= 0) or not np.all(np.isfinite(np.concatenate([mu, va, y]))):
            raise ContractError("validation needs finite inputs and var > 0")
        if self.scale is not None and self.scale.size not in (1, y.size):
            raise ContractError("scale must be scalar or one per dimension")
        if where is not None and self.regions is not None:
            where = self.regions._vec(where)          # validate before any mutation
        mv = int(model_version)
        if mv != self.model_version:                 # new model: new record
            self.model_version = mv
            self.z2.clear()
            self.nmse.clear()
            self.cusum.reset()
        err2 = (y - mu) ** 2
        z2 = float(np.mean(err2 / va))
        nm = UNKNOWN
        failed = z2 > self.max_z2
        if self.scale is not None:
            nm = float(np.mean(err2 / self.scale ** 2))
            failed = failed or nm > self.max_nmse
        # a failure INSIDE a region already marked unreliable is explained by
        # it: the region gate handles it, so it does not close the global
        # gate for everywhere else (that was the planner's other big cost on
        # the mud fixture: 10 steps of lost control every time the FALLBACK
        # crossed known mud). Failures anywhere new still count globally.
        explained = bool(failed and where is not None and self.regions is not None
                         and self.regions.assess(where[None])[0][0])
        if not explained:
            self.z2.append(z2)
            if self.scale is not None:
                self.nmse.append(nm)
        self.total += 1
        if where is not None and self.regions is not None:
            self.regions.add(where, failed, mv)
        c = self.cusum.update(mu, va, y)
        alarm = bool(c and c["alarm"])
        if alarm:
            self.alarms.append({"validation": self.total, "kind": c["kind"],
                                "model_version": mv})
        return {"z2": z2, "nmse": nm, "alarm": alarm, "failed": bool(failed),
                "explained_by_region": explained, "kind": c["kind"] if c else None}

    def reliable(self) -> Tuple[bool, str]:
        n = len(self.z2)
        if n < self.min_count:
            return False, (f"cold: {n}/{self.min_count} real validations of "
                           f"model v{self.model_version}")
        mz = float(np.mean(self.z2))
        if mz > self.max_z2:
            return False, f"overconfident/wrong: mean z^2 {mz:.3g} > {self.max_z2}"
        if self.max_nmse is not None:
            mn = float(np.mean(self.nmse))
            if mn > self.max_nmse:
                return False, f"inaccurate: nmse {mn:.3g} > {self.max_nmse}"
        return True, f"validated: mean z^2 {mz:.3g} over {n}"

    def regions_active(self) -> bool:
        return self.regions is not None and self.regions.active()

    def region_unreliable(self, X) -> np.ndarray:
        """(n, D) states -> (n,) bool: True where the model has failed real
        validation locally (see RegionMemory). False = reliable OR unknown."""
        Q = np.asarray(X, np.float64)
        if self.regions is None:
            return np.zeros(len(Q) if Q.ndim > 1 else 1, bool)
        return self.regions.assess(Q)[0]

    def reliable_at(self, x) -> Tuple[bool, str]:
        if not self.regions_active():
            return True, "region: no recorded failures"
        bad, bw, frac = self.regions.assess(np.asarray(x, np.float64).reshape(1, -1))
        if bad[0]:
            return False, (f"region: model failed real validation near this state "
                           f"(failure weight {bw[0]:.2g}; {frac[0]:.0%} of the evidence "
                           f"there is failure); reopens on current-model passes here, "
                           f"after a refit, or by decay (half-life "
                           f"{self.regions.half_life:g} validations)")
        return True, f"region: failure weight {bw[0]:.2g} here"

    def state_dict(self) -> Dict[str, Any]:
        return {"schema": RELIABILITY_SCHEMA, "model_version": self.model_version,
                "z2": list(self.z2), "nmse": list(self.nmse), "total": self.total,
                "alarms": [dict(a) for a in self.alarms],
                "cusum": {"s_hi": self.cusum.s_hi, "s_lo": self.cusum.s_lo,
                          "s_disp": self.cusum.s_disp, "step": self.cusum.step},
                "regions": None if self.regions is None else self.regions.state_dict()}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        if d.get("schema") != RELIABILITY_SCHEMA:
            raise ContractError(f"reliability state schema {d.get('schema')!r}, this code "
                                f"reads {RELIABILITY_SCHEMA} (region memory added)")
        if (d["regions"] is None) != (self.regions is None):
            raise ContractError("saved reliability state and this monitor disagree on "
                                "whether region memory is enabled")
        self.model_version = d["model_version"]
        self.z2 = deque((float(x) for x in d["z2"]), maxlen=self.window)
        self.nmse = deque((float(x) for x in d["nmse"]), maxlen=self.window)
        self.total = int(d["total"])
        self.alarms = [dict(a) for a in d["alarms"]]
        for k in ("s_hi", "s_lo", "s_disp"):
            setattr(self.cusum, k, float(d["cusum"][k]))
        self.cusum.step = int(d["cusum"]["step"])
        if self.regions is not None:
            self.regions.load_state_dict(d["regions"])


# ======================================================================
# Planner
# ======================================================================

class _Cancelled(Exception):
    """A chunk stopped at a horizon-step deadline check."""


WAIT_SLICE_S = 0.0005


def _wait(ev: threading.Event, timeout: Optional[float]) -> bool:
    """Event.wait with a deadline that is actually kept. A single long timed
    wait on macOS overshoots by up to ~50% (timer coalescing: a 20 ms wait
    measured 22-30 ms), which alone would break a 1.25x deadline. So wait
    HALF the remaining time, repeatedly, down to 0.5 ms slices: the
    overshoot of each wait stays inside what is left, and a 1 s wait costs
    ~12 wake-ups, not 1000. A set event still wakes immediately."""
    if timeout is None:
        return ev.wait()
    end = time.perf_counter() + max(0.0, float(timeout))
    while True:
        left = end - time.perf_counter()
        if left <= 0:
            return ev.is_set()
        if ev.wait(left if left <= WAIT_SLICE_S else max(WAIT_SLICE_S, 0.5 * left)):
            return True


class _Job:
    __slots__ = ("fn", "done", "result", "error", "cancelled", "t0")

    def __init__(self, fn):
        self.fn, self.done = fn, threading.Event()
        self.result = self.error = None
        self.cancelled = False
        self.t0 = time.perf_counter()

    def age(self) -> float:
        return time.perf_counter() - self.t0


class _CallWorker:
    """ONE daemon thread that runs the planner's model calls so the planner
    can stop WAITING at its deadline. Python cannot kill a running call, so
    an abandoned call keeps this worker busy until it returns; submit()
    refuses while busy (no queueing, no second thread: at most one model
    call in flight per planner, ever). The thread is started lazily and
    exits after IDLE_EXIT_S without work, so idle planners hold no thread."""

    IDLE_EXIT_S = 5.0

    def __init__(self):
        self._q: "queue.Queue[_Job]" = queue.Queue()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.job: Optional[_Job] = None
        self.poisoned = False          # set by the planner's watchdog; never reused

    def busy(self) -> bool:
        j = self.job
        return j is not None and not j.done.is_set()

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        j = self.job
        return True if j is None else _wait(j.done, timeout)

    def submit(self, fn) -> _Job:
        if self.busy() or self.poisoned:
            raise ContractError("model call worker is busy")     # pragma: no cover
        job = _Job(fn)
        self.job = job
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True,
                                                name="foundation-planner-model-call")
                self._thread.start()
            self._q.put(job)
        return job

    def _loop(self) -> None:
        while True:
            try:
                job = self._q.get(timeout=self.IDLE_EXIT_S)
            except queue.Empty:
                with self._lock:
                    if self._q.empty():
                        self._thread = None
                        return
                continue
            try:
                job.result = job.fn(job)
            except BaseException as e:          # re-raised in the planner's thread
                job.error = e
            finally:
                job.done.set()
            if self.poisoned:                   # a hung call finally returned:
                with self._lock:                # the planner moved on, so exit
                    self._thread = None
                return


_LATE = object()
COST_HISTORY = 16
MAX_RESPAWN_BACKOFF_S = 60.0


@dataclass
class PlanResult:
    ok: bool
    timed_out: bool
    command: Any                      # first-step command (spec space) or None
    imagined_value: Any               # float, or UNKNOWN when no plan
    effective_horizon: int            # steps of the best plan inside the
                                      # uncertainty threshold
    elapsed: float
    iterations: int
    evaluated: int
    reason: str
    model_versions: Dict[str, int] = field(default_factory=dict)
    sequence: Any = None


class BoundedPlanner:
    """CEM over action sequences of an ActionSpec ("box" or "discrete").

    model     has rollout(EntityBatch, actions (H,B,N,Ad), dts (H,B),
              mode="mean") -> [MixedOutcome] * H, and an int `version`
    reward_fn (next_continuous (C, Dc), model_action (C, N, Ad)) -> (C,)
    encode    spec commands (H, C, *cmd) -> model actions (H, C, N, Ad).
              Box default: reshape when sizes match. Discrete: REQUIRED
              (index -> actuation is environment knowledge the planner must
              be told or must learn; it is never guessed here).
    """

    def __init__(self, model, reward_fn: Callable, action_spec: ActionSpec, *,
                 n_entities: int, action_dim: int, encode: Optional[Callable] = None,
                 horizon: int = 8, n_candidates: int = 64, n_iters: int = 3,
                 elite_frac: float = 0.15, gamma: float = 0.95,
                 epistemic_threshold: float = float("inf"),
                 uncertainty_penalty: float = 0.0, uncertainty: str = "epistemic",
                 reward_floor: Optional[float] = None, deadline_s: float = 0.05,
                 chunk: Optional[int] = None, reprobe_after: int = 20,
                 min_std: float = 0.05, smoothing: float = 0.25, seed: int = 0,
                 clock: Callable[[], float] = time.perf_counter,
                 isolate_calls: Optional[bool] = None,
                 hang_after_s: Optional[float] = None, max_respawns: int = 3,
                 respawn_backoff_s: float = 0.5):
        if not isinstance(action_spec, ActionSpec):
            raise ContractError("action_spec must be a contracts.ActionSpec")
        if action_spec.kind not in ("box", "discrete"):
            raise ContractError(f"BoundedPlanner supports box/discrete specs, not "
                                f"{action_spec.kind!r} (no silent flattening)")
        if action_spec.kind == "discrete" and encode is None:
            raise ContractError("a discrete spec needs `encode` (index -> actuation)")
        if not hasattr(model, "rollout"):
            raise ContractError("model must provide rollout()")
        _model_version(model)
        if uncertainty not in ("epistemic", "total"):
            raise ContractError("uncertainty must be 'epistemic' or 'total'")
        for nm, v, lo in (("horizon", horizon, 1), ("n_candidates", n_candidates, 2),
                          ("n_iters", n_iters, 1), ("reprobe_after", reprobe_after, 1)):
            if int(v) < lo:
                raise ContractError(f"{nm} must be >= {lo}")
        if not (0 < elite_frac < 1) or not (0 < gamma <= 1):
            raise ContractError("need 0 < elite_frac < 1 and 0 < gamma <= 1")
        if not (deadline_s > 0) or uncertainty_penalty < 0 or not (epistemic_threshold > 0):
            raise ContractError("deadline_s > 0, uncertainty_penalty >= 0 and "
                                "epistemic_threshold > 0 required")
        self.model, self.reward_fn, self.spec = model, reward_fn, action_spec
        self.N, self.Ad = int(n_entities), int(action_dim)
        self.H, self.C, self.n_iters = int(horizon), int(n_candidates), int(n_iters)
        self.n_elite = max(2, int(round(elite_frac * self.C)))
        self.gamma = float(gamma)
        self.threshold = float(epistemic_threshold)
        self.penalty = float(uncertainty_penalty)
        self.uncertainty = uncertainty
        self.reward_floor = reward_floor
        self.deadline_s = float(deadline_s)
        self.chunk = int(chunk or self.C)
        if self.chunk < 1:
            raise ContractError("chunk must be >= 1")
        self.reprobe_after = int(reprobe_after)
        self.min_std, self.smoothing = float(min_std), float(smoothing)
        self.clock = clock
        # thread-isolated model calls need the REAL clock (Event.wait is real
        # time); an injected clock means deterministic synchronous mode
        self.isolate = (clock is time.perf_counter) if isolate_calls is None \
            else bool(isolate_calls)
        self._worker = _CallWorker() if self.isolate else None
        # hung-model watchdog (thread mode only; see ESCAPE PATHS)
        self.hang_after_s = max(1.0, 20.0 * self.deadline_s) if hang_after_s is None \
            else float(hang_after_s)
        if not (self.hang_after_s > 0) or int(max_respawns) < 0 or \
                not (respawn_backoff_s >= 0):
            raise ContractError("need hang_after_s > 0, max_respawns >= 0 and "
                                "respawn_backoff_s >= 0")
        self.max_respawns = int(max_respawns)
        self.respawn_backoff_s = float(respawn_backoff_s)
        self._poisoned: List[_CallWorker] = []
        self._consec_hangs = 0
        self._respawn_not_before = float("-inf")
        self._force_probe = False
        self.rng = np.random.default_rng(seed)
        self._encode_fn = encode
        if action_spec.kind == "box":
            self.cmd_shape = tuple(action_spec.shape)
            self.low = np.where(np.isfinite(action_spec.low), action_spec.low, -1.0)
            self.high = np.where(np.isfinite(action_spec.high), action_spec.high, 1.0)
            if encode is None and int(np.prod(self.cmd_shape)) != self.N * self.Ad:
                raise ContractError(f"box spec {self.cmd_shape} cannot be reshaped "
                                    f"to (N={self.N}, Ad={self.Ad}); pass encode")
        else:
            self.cmd_shape = ()
            self.n_actions = int(action_spec.n)
        # persistent state (saved/restored)
        self.chunk_cost: Optional[float] = None       # median seconds per chunk
        self._costs: Deque[float] = deque(maxlen=COST_HISTORY)
        self.consecutive_preempt = 0
        self.warm: Optional[np.ndarray] = None        # previous plan's dist params
        self.stats = {"calls": 0, "timed_out": 0, "probes": 0, "late_chunks": 0,
                      "refused_epistemic": 0, "model_busy": 0, "region_cut": 0,
                      "hung_calls": 0, "respawns": 0, "respawn_deferred": 0}

    # ---- helpers ----------------------------------------------------------
    def _encode(self, cmds: np.ndarray) -> np.ndarray:
        """cmds (H, C, *cmd) -> (H, C, N, Ad)."""
        if self._encode_fn is not None:
            out = np.asarray(self._encode_fn(cmds), np.float64)
        else:
            out = np.asarray(cmds, np.float64).reshape(cmds.shape[:2] + (self.N, self.Ad))
        if out.shape != cmds.shape[:2] + (self.N, self.Ad):
            raise ContractError(f"encode returned {out.shape}, expected "
                                f"{cmds.shape[:2] + (self.N, self.Ad)}")
        return out

    def _evaluate(self, state, dt: float, seqs: np.ndarray, region_check=None,
                  cancel: Optional[Callable[[], bool]] = None):
        """seqs (C, H, *cmd) -> values (C,), effective horizons (C,), number
        of candidates whose cut came from a failed region. region_check
        (n, Dc) -> (n,) bool marks input states the model failed in; step h
        is cut when ITS INPUT (the start state for h = 0, else the imagined
        state after step h-1) is marked. cancel() is polled between horizon
        steps and raises _Cancelled."""
        C = seqs.shape[0]
        acts = self._encode(np.swapaxes(seqs, 0, 1))
        outs = self.model.rollout(state.repeat(C), acts, np.full((self.H, C), float(dt)),
                                  mode="mean")
        if len(outs) != self.H:
            raise ContractError(f"rollout returned {len(outs)} steps for H={self.H}")
        region = np.zeros((self.H, C), bool)
        if region_check is not None:
            x0 = np.asarray(state.continuous(), np.float64)
            X = np.concatenate([np.repeat(x0, C, 0)] +
                               [np.asarray(d.mean(), np.float64) for d in outs[:-1]], 0)
            region = np.asarray(region_check(X), bool).reshape(self.H, C)
        r = np.empty((self.H, C))
        u = np.empty((self.H, C))
        for h, d in enumerate(outs):
            if cancel is not None and cancel():
                raise _Cancelled()
            within, between = d.decompose_variance()
            u[h] = (between if self.uncertainty == "epistemic" else within + between).mean(1)
            rh = np.asarray(self.reward_fn(d.mean(), acts[h]), np.float64)
            if rh.shape != (C,) or not np.all(np.isfinite(rh)):
                raise ContractError(f"reward_fn must return finite (C,), got {rh.shape}")
            r[h] = rh
        over = (u > self.threshold) | region
        eff = np.where(over.any(0), over.argmax(0), self.H)          # first cut
        unc = u > self.threshold
        eff_unc = np.where(unc.any(0), unc.argmax(0), self.H)
        n_region = int((eff < eff_unc).sum())
        counted = np.arange(self.H)[:, None] < eff[None, :]
        if self.reward_floor is not None:
            floor = float(self.reward_floor)
        else:
            floor = float(r[counted].min()) if counted.any() else float(r.min())
        disc = self.gamma ** np.arange(self.H)[:, None]
        per = np.where(counted, r - self.penalty * u, floor)
        return (disc * per).sum(0), eff, n_region

    def _sample(self, params) -> np.ndarray:
        if self.spec.kind == "box":
            mu, sd = params
            x = mu[None] + sd[None] * self.rng.standard_normal((self.C,) + mu.shape)
            x[0] = mu                                           # keep the mean
            return np.clip(x, self.low, self.high)
        probs = params
        u = self.rng.random((self.C, self.H, 1))
        x = (u > np.cumsum(probs, -1)[None]).sum(-1).clip(0, self.n_actions - 1)
        x[0] = probs.argmax(-1)                                 # keep the mode
        return x

    def _init_params(self):
        if self.spec.kind == "box":
            mid = (self.low + self.high) / 2.0
            sd0 = np.broadcast_to((self.high - self.low) / 2.0, (self.H,) + self.cmd_shape).copy()
            mu = np.broadcast_to(mid, (self.H,) + self.cmd_shape).copy()
            if self.warm is not None:
                mu[:-1] = self.warm[1:]
                mu[-1] = self.warm[-1]
            return mu, sd0
        p = np.full((self.H, self.n_actions), 1.0 / self.n_actions)
        if self.warm is not None:
            w = np.concatenate([self.warm[1:], self.warm[-1:]], 0)
            p = 0.5 * p + 0.5 * w
        return p

    def _refit(self, params, elites: np.ndarray):
        a = self.smoothing
        if self.spec.kind == "box":
            mu, sd = params
            mu = a * mu + (1 - a) * elites.mean(0)
            sd = np.maximum(a * sd + (1 - a) * elites.std(0), self.min_std)
            return mu, sd
        counts = np.stack([(elites == k).mean(0) for k in range(self.n_actions)], -1)
        p = a * params + (1 - a) * counts
        p = np.maximum(p, self.min_std / self.n_actions)
        return p / p.sum(-1, keepdims=True)

    # ---- deadline machinery -------------------------------------------------------
    def _record_cost(self, cost: float, probe: bool) -> None:
        """A probe REPLACES the history it was sent to remeasure. Abandoned
        calls enter as their (censored) waited time: a lower bound, which
        the median shrugs off when rare and adopts when typical."""
        if probe or self.chunk_cost is None:
            self._costs.clear()
            self.consecutive_preempt = 0
        self._costs.append(float(cost))
        self.chunk_cost = float(np.median(self._costs))

    def _call(self, fn, deadline: float):
        """Run fn(cancel) -> result, returning _LATE if it cannot finish by
        `deadline`. Thread mode: wait at most until the deadline, then
        abandon (the worker stays busy until the call returns). Sync mode:
        run inline; cancel() polls the clock between horizon steps."""
        if not self.isolate:
            try:
                return fn(lambda: self.clock() > deadline)
            except _Cancelled:
                return _LATE
        job = self._worker.submit(lambda j: fn(lambda: j.cancelled))
        if _wait(job.done, deadline - self.clock()):
            if isinstance(job.error, _Cancelled):
                return _LATE
            if job.error is not None:
                raise job.error
            return job.result
        job.cancelled = True
        return _LATE

    def busy(self) -> bool:
        """True while an abandoned model call still runs on the worker."""
        return self._worker is not None and self._worker.busy()

    def live_poisoned(self) -> int:
        """Poisoned workers whose hung call is still running (threads stuck
        in the model). A returned call frees its slot."""
        self._poisoned = [w for w in self._poisoned if w.busy()]
        return len(self._poisoned)

    def _watchdog(self) -> None:
        """Declare a call older than hang_after_s HUNG: poison its worker,
        spawn a fresh one and force a probe — unless max_respawns poisoned
        workers are still alive or the backoff has not elapsed (then the
        model simply stays unavailable and the fallback controls; checked
        again on every plan/observe)."""
        w = self._worker
        if w is None or not w.busy() or w.job.age() < self.hang_after_s:
            return
        now = time.perf_counter()
        if self.live_poisoned() >= self.max_respawns or now < self._respawn_not_before:
            self.stats["respawn_deferred"] += 1
            return
        w.job.cancelled = True              # cooperative models stop at the next poll
        w.poisoned = True
        self._poisoned.append(w)
        self._worker = _CallWorker()
        self._consec_hangs += 1
        self.stats["hung_calls"] += 1
        self.stats["respawns"] += 1
        self._respawn_not_before = now + min(
            MAX_RESPAWN_BACKOFF_S, self.respawn_backoff_s * 2.0 ** (self._consec_hangs - 1))
        self._force_probe = True            # re-probe: the old cost history is stale

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until no model call is in flight (or timeout). Anything else
        that touches the model (e.g. PlanningController.observe) waits here
        first, so the model is never used from two threads at once."""
        return True if self._worker is None else self._worker.wait_idle(timeout)

    def wait_usable(self) -> bool:
        """Wait for the in-flight call only until it would count as HUNG,
        then run the watchdog. Returns True when no live worker is busy. A
        call declared hung is presumed stuck, and the model is used beside
        it (the alternative — waiting forever — is the latch this replaces)."""
        w = self._worker
        if w is None or not w.busy():
            return True
        w.wait_idle(max(0.0, self.hang_after_s - w.job.age()))
        self._watchdog()
        return not self.busy()

    # ---- main ------------------------------------------------------------------
    def plan(self, state, dt: float, region_check: Optional[Callable] = None) -> PlanResult:
        """region_check (n, Dc) -> (n,) bool: states the model FAILED real
        validation in (ReliabilityMonitor.region_unreliable). Rollouts are
        cut at the first step whose input state is marked."""
        if getattr(state, "B", None) != 1:
            raise ContractError("plan() takes a single state (EntityBatch with B == 1)")
        if not (np.isfinite(dt) and dt > 0):
            raise ContractError("dt must be finite and > 0")
        self.stats["calls"] += 1
        t0 = self.clock()
        deadline = t0 + self.deadline_s
        self._watchdog()
        params = self._init_params()
        best_v, best_seq, best_eff = -np.inf, None, 0
        evaluated, iters, max_eff, n_region = 0, 0, 0, 0
        stop, preempted, late = False, False, False
        for it in range(self.n_iters):
            seqs = self._sample(params)
            vals, effs, done = [], [], []
            for s in range(0, self.C, self.chunk):
                now = self.clock()
                self._watchdog()
                if self.busy():                # an abandoned call holds the model
                    est = self.chunk_cost or 0.0
                    if not self._worker.wait_idle(max(0.0, deadline - now - est)):
                        self.stats["model_busy"] += 1
                        stop, preempted = True, True
                        break
                    now = self.clock()
                probe = False
                if self._force_probe and evaluated == 0:
                    probe = True                       # fresh worker: remeasure ONCE
                    self._force_probe = False          # (a late probe still resets the
                    self.stats["probes"] += 1          # estimate; a hung one re-arms it)
                elif self.chunk_cost is not None and now + self.chunk_cost > deadline:
                    if self.consecutive_preempt >= self.reprobe_after and evaluated == 0:
                        probe = True                   # escape: remeasure cost
                        self.stats["probes"] += 1
                    else:
                        stop, preempted = True, True
                        break
                part = seqs[s:s + self.chunk]
                c0 = self.clock()
                out = self._call(lambda cancel, _p=part: self._evaluate(
                    state, dt, _p, region_check, cancel), deadline)
                c1 = self.clock()
                self._record_cost(c1 - c0, probe)
                if out is not _LATE:                   # the model answered: not hung
                    self._consec_hangs = 0
                if out is _LATE or c1 > deadline:      # late answers are discarded
                    self.stats["late_chunks"] += 1
                    stop, late = True, True
                    break
                v, e, nr = out
                vals.append(v)
                effs.append(e)
                done.append(part)
                n_region += nr
            if vals:
                v = np.concatenate(vals)
                e = np.concatenate(effs)
                sq = np.concatenate(done)
                evaluated += len(v)
                iters += 1
                max_eff = max(max_eff, int(e.max()))
                i = int(np.argmax(v))
                if v[i] > best_v:
                    best_v, best_seq, best_eff = float(v[i]), sq[i].copy(), int(e[i])
                k = min(self.n_elite, len(v))
                if k >= 2:
                    params = self._refit(params, sq[np.argsort(-v)[:k]])
            if stop:
                break
        elapsed = self.clock() - t0
        mv = _model_versions(self.model)
        self.stats["region_cut"] += n_region
        if evaluated == 0:
            self.stats["timed_out"] += 1
            if preempted:
                self.consecutive_preempt += 1
            why = ("model busy with an abandoned call" if self.busy() and not late
                   else "no candidate evaluated on time")
            return PlanResult(False, True, None, UNKNOWN, 0, elapsed, 0, 0,
                              f"deadline {self.deadline_s * 1e3:.1f} ms: {why} (chunk cost "
                              f"estimate {(self.chunk_cost or 0) * 1e3:.1f} ms)", mv)
        self.consecutive_preempt = 0
        if max_eff == 0:
            self.stats["refused_epistemic"] += 1
            src = "region" if n_region else "epistemic"
            return PlanResult(False, False, None, UNKNOWN, 0, elapsed, iters, evaluated,
                              f"{src}: no candidate has a first step under threshold "
                              f"{self.threshold:g}" + (" outside a region the model "
                                                       "failed in" if n_region else ""), mv)
        self.warm = (params[0].copy() if self.spec.kind == "box" else params.copy())
        cmd = best_seq[0]
        cmd = int(cmd) if self.spec.kind == "discrete" else np.asarray(cmd, np.float64)
        return PlanResult(True, False, cmd, best_v, best_eff, elapsed, iters, evaluated,
                          "planned" + (" (best so far: deadline)" if stop else ""), mv,
                          best_seq)

    # ---- persistence --------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"schema": STATE_SCHEMA, "chunk_cost": self.chunk_cost,
                "costs": list(self._costs),
                "consecutive_preempt": self.consecutive_preempt,
                "warm": None if self.warm is None else np.array(self.warm),
                "stats": dict(self.stats),
                "rng": _rng_state(self.rng)}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        if d.get("schema") != STATE_SCHEMA:
            raise ContractError(f"planner state schema {d.get('schema')!r}")
        self.chunk_cost = d["chunk_cost"]
        self._costs = deque((float(x) for x in d["costs"]), maxlen=COST_HISTORY)
        self.consecutive_preempt = int(d["consecutive_preempt"])
        self.warm = None if d["warm"] is None else np.array(d["warm"])
        self.stats = {**self.stats, **dict(d["stats"])}
        self.rng.bit_generator.state = _rng_restore(d["rng"])


def _rng_state(rng: np.random.Generator) -> Dict[str, Any]:
    s = rng.bit_generator.state
    return {"bit_generator": s["bit_generator"],
            "state": {k: str(v) for k, v in s["state"].items()},
            "has_uint32": int(s["has_uint32"]), "uinteger": int(s["uinteger"])}


def _rng_restore(d: Dict[str, Any]) -> Dict[str, Any]:
    return {"bit_generator": d["bit_generator"],
            "state": {k: int(v) for k, v in d["state"].items()},
            "has_uint32": int(d["has_uint32"]), "uinteger": int(d["uinteger"])}


# ======================================================================
# Decisions: planner or the existing controller
# ======================================================================

@dataclass
class Decision:
    command: Any
    source: str
    reason: str
    elapsed: float
    plan: Optional[PlanResult] = None


class PlanningController:
    """planner + existing controller + real-outcome reliability gate.

    fallback_policy(obs) -> command in the SAME spec space as the planner.
    knowledge_gate (optional) () -> (usable: bool, reason): e.g. the
    consolidation store saying the model's mechanism needs revalidation
    after a representation bump.
    region_key     what a REGION is keyed on (validation records and the
                   planner's imagined states alike): "position" (default:
                   entity positions — regions of SPACE), "state" (every
                   continuous dim) or a callable (n, Dc) -> (n, K).
                   Measured on the verifier's mud fixture: keyed on the full
                   state, failures recorded at the low speeds mud forces did
                   not generalise to the high speeds the planner enters at
                   (re-entries 4.2/seed at bandwidth 0.25; at 0.5 the region
                   blurred 1-2 m beyond the mud instead). Position keys
                   over-block a velocity-dependent failure — the safe side,
                   with the same three ways back.
    """

    REGION_KEYS = ("position", "state")

    def __init__(self, planner: BoundedPlanner, fallback_policy: Callable,
                 reliability: ReliabilityMonitor, knowledge_gate: Optional[Callable] = None,
                 log_size: int = 10000, region_key: Any = "position"):
        if not callable(fallback_policy):
            raise ContractError("fallback_policy must be callable(obs) -> command")
        if not (callable(region_key) or region_key in self.REGION_KEYS):
            raise ContractError(f"region_key must be one of {self.REGION_KEYS} or a "
                                f"callable (n, Dc) continuous states -> (n, K)")
        self.region_key = region_key
        self.planner, self.fallback = planner, fallback_policy
        self.reliability, self.knowledge_gate = reliability, knowledge_gate
        self.log: Deque[Dict[str, Any]] = deque(maxlen=int(log_size))
        self.counts = {s: 0 for s in SOURCES}
        self.t = 0

    def _key(self, state, X: np.ndarray) -> np.ndarray:
        """Continuous states (n, Dc) in EntityBatch.continuous() layout
        (entity-major [pos_i, vel_i]) -> region keys (n, K)."""
        X = np.asarray(X, np.float64)
        if callable(self.region_key):
            K = np.asarray(self.region_key(X), np.float64)
            if K.ndim != 2 or len(K) != len(X):
                raise ContractError(f"region_key returned {K.shape} for {X.shape}")
            return K
        if self.region_key == "state":
            return X
        N, D = state.N, state.D
        idx = (np.arange(N)[:, None] * 2 * D + np.arange(D)[None]).reshape(-1)
        return X[:, idx]

    def _fallback(self, obs, source, reason, t0, plan=None) -> Decision:
        cmd = self.fallback(obs)
        cmd = self.planner.spec.validate_command(cmd)
        return self._record(Decision(cmd, source, reason,
                                     self.planner.clock() - t0, plan))

    def _record(self, d: Decision) -> Decision:
        if d.source not in SOURCES:                    # pragma: no cover
            raise ContractError(f"unknown decision source {d.source!r}")
        self.counts[d.source] += 1
        p = d.plan
        self.log.append({"t": self.t, "source": d.source, "reason": d.reason,
                         "elapsed": d.elapsed,
                         "effective_horizon": None if p is None else p.effective_horizon,
                         "imagined_value": None if p is None else p.imagined_value,
                         "model_versions": None if p is None else p.model_versions})
        self.t += 1
        return d

    def decide(self, state, dt: float, obs) -> Decision:
        t0 = self.planner.clock()
        if self.knowledge_gate is not None:
            ok, why = self.knowledge_gate()
            if not ok:
                return self._fallback(obs, "fallback_unreliable", f"knowledge: {why}", t0)
        ok, why = self.reliability.reliable()
        if not ok:
            return self._fallback(obs, "fallback_unreliable", why, t0)
        rc = None
        if self.reliability.regions_active():
            ok, why = self.reliability.reliable_at(self._key(state, state.continuous())[0])
            if not ok:
                return self._fallback(obs, "fallback_unreliable", why, t0)
            rc = lambda X, _s=state: self.reliability.region_unreliable(self._key(_s, X))
        res = self.planner.plan(state, dt, region_check=rc)
        if res.timed_out:
            return self._fallback(obs, "fallback_timeout", res.reason, t0, res)
        if not res.ok:
            return self._fallback(obs, "fallback_unreliable", res.reason, t0, res)
        cmd = self.planner.spec.validate_command(res.command)
        return self._record(Decision(cmd, "planner", res.reason,
                                     self.planner.clock() - t0, res))

    def observe(self, state, command, dt: float, next_state, *, provenance: str
                ) -> Dict[str, Any]:
        """Validate the model's one-step prediction for the EXECUTED command
        (whoever chose it) against the real next state."""
        _check_real(provenance, "PlanningController.observe")
        cmd = self.planner.spec.validate_command(command)
        arr = np.asarray(cmd).reshape((1, 1) + self.planner.cmd_shape)
        act = self.planner._encode(arr)[0]
        self.planner.wait_usable()             # never wait on a HUNG call forever
        d = self.planner.model.predict(state, act, np.array([float(dt)]))
        return self.reliability.validate(d.mean()[0], d.variance()[0],
                                         next_state.continuous()[0],
                                         provenance=provenance,
                                         model_version=_model_version(self.planner.model),
                                         where=self._key(state, state.continuous())[0])

    def state_dict(self) -> Dict[str, Any]:
        return {"schema": STATE_SCHEMA, "planner": self.planner.state_dict(),
                "reliability": self.reliability.state_dict(),
                "counts": dict(self.counts), "t": self.t,
                "log": [dict(x) for x in self.log]}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        if d.get("schema") != STATE_SCHEMA:
            raise ContractError(f"controller state schema {d.get('schema')!r}")
        self.planner.load_state_dict(d["planner"])
        self.reliability.load_state_dict(d["reliability"])
        self.counts = {s: int(d["counts"].get(s, 0)) for s in SOURCES}
        self.t = int(d["t"])
        self.log = deque((dict(x) for x in d["log"]), maxlen=self.log.maxlen)


# ======================================================================
# Adapter: the existing PPO actor-critic as the fallback controller
# ======================================================================

class ActorCriticFallback:
    """Wrap policy.actor_critic.StandaloneActorCritic as `fallback_policy`.

    __call__(obs) -> command   via the public select_action(obs,
                               deterministic=...), mapped through
                               `command_map` (index -> spec command) if given
    action_probs(obs) -> (K,)  the discrete action distribution, computed by
                               the same internal hops select_action uses
                               (_encode -> _augment -> actor.shared ->
                               _bound_logits). Used for behavioural
                               signatures (skills.behaviour_signature).
    weights_hash()             sha1 over the actor's state dict — a
                               DIAGNOSTIC only: identity is behaviour, not
                               weights (CLAUDE.md §4.6)

    Supports arch "flat"/"conv" (which encode an observation). "rssm"
    cannot encode one frame (its latent is recurrent) and raises.
    """

    def __init__(self, actor_critic, deterministic: bool = True,
                 command_map: Optional[Callable] = None, knowledge=None, proprio=None):
        if not hasattr(actor_critic, "select_action"):
            raise ContractError("actor_critic must expose select_action()")
        if getattr(actor_critic, "continuous", False):
            raise ContractError("ActorCriticFallback supports discrete policies")
        self.ac, self.deterministic = actor_critic, bool(deterministic)
        self.command_map, self.knowledge, self.proprio = command_map, knowledge, proprio
        self.calls = 0

    def __call__(self, obs):
        a, _info = self.ac.select_action(np.asarray(obs, np.float32),
                                         knowledge=self.knowledge,
                                         deterministic=self.deterministic,
                                         proprio=self.proprio)
        self.calls += 1
        return self.command_map(int(a)) if self.command_map is not None else int(a)

    def action_probs(self, obs) -> np.ndarray:
        import torch
        ac = self.ac
        x = torch.as_tensor(np.asarray(obs, np.float32)).reshape(1, -1).to(ac.device)
        with torch.no_grad():
            aug = ac._augment(ac._encode(x), ac._prep_knowledge(self.knowledge),
                              ac._prep_proprio(self.proprio))
            logits = ac._bound_logits(ac.actor.action_head(ac.actor.shared(aug)))
            return torch.softmax(logits, -1).reshape(-1).cpu().numpy().astype(np.float64)

    def weights_hash(self) -> str:
        import hashlib
        h = hashlib.sha1()
        for k, v in sorted(self.ac.actor.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
