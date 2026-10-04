"""Bounded split/merge revision of tracks, and residual retention (plan
Stage 8 item 5).

WHAT IS CLAIMED
    1. PROPOSALS ARE BOUNDED. `RevisionEngine.step` generates candidate
       structural changes to the tracker's entity set and EVALUATES at most
       `max_proposals` of them per call (highest prior score first). Nothing
       is evaluated speculatively beyond that budget.
    2. ACCEPTANCE IS BY PREDICTIVE LIKELIHOOD, on data the proposal was not
       fitted to. Each proposal is scored as
           gain = log p(held-out | revised) - log p(held-out | current)
       using Bayesian constant-velocity regressions (predictive variance
       sigma^2 (1 + x^T (X^T X)^-1 x), sigma = the tracker's measurement
       noise). Accepted only if gain >= `min_gain_nats`.
         MERGE (tracklet stitching): live track b and occluded/lost track a,
           never observed in the same frame, b's start near a's
           extrapolation. Revised model fits ONE trajectory to a's tail + b's
           early points; current model fits b alone. Held-out = b's later
           points. A real fragment is predicted better by the longer
           baseline; a different entity is predicted much worse.
         SPLIT: one track whose detections come from two entities
           (alternating). Revised = two CV components (hard-EM), mixture
           predictive; current = one CV line. Held-out = the track's last
           points. Unimodal noise gains nothing and is rejected.
    3. RESIDUALS ARE RETAINED. `ResidualStore` keeps, per kind (position
       innovation, appearance residual, unexplained detection), a bounded
       ring of vectors the current model did NOT explain, and `summary`
       reports whether their mean is structured (|mean|/sem above a z
       threshold) — the trace an unmodelled mechanism (e.g. a constant
       acceleration the CV model lacks) leaves behind, kept for Stage 9.

ESCAPE PATHS (CLAUDE.md §4.1)
    A rejected proposal is put on `cooldown` steps, then becomes eligible
    again — with more data it may now pass. Proposals over budget are not
    dropped: candidates are regenerated from track histories every step, so
    a deferred one reappears. Nothing here is permanent except an accepted
    revision, and that is itself recorded (aliases / new ids / ambiguity log).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .errors import PerceptionError
from .identity import IdentityTracker, Track

RESIDUAL_KINDS = ("position_innovation", "feature_residual",
                  "unexplained_detection")


class ResidualStore:
    """Bounded ring buffers of what the current model left unexplained."""

    def __init__(self, capacity: int = 2000):
        if int(capacity) < 1:
            raise PerceptionError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._buf: Dict[str, Deque[Tuple[int, str, np.ndarray]]] = {
            k: deque(maxlen=self.capacity) for k in RESIDUAL_KINDS}

    def add(self, frame: int, kind: str, local: str, vec) -> None:
        if kind not in self._buf:
            raise PerceptionError(f"unknown residual kind {kind!r}")
        v = np.asarray(vec, dtype=np.float64).ravel()
        if not np.all(np.isfinite(v)):
            raise PerceptionError("residual must be finite")
        self._buf[kind].append((int(frame), str(local), v.copy()))

    def __len__(self) -> int:
        return sum(len(b) for b in self._buf.values())

    def vectors(self, kind: str) -> np.ndarray:
        b = self._buf[kind]
        return np.stack([v for _, _, v in b]) if b else np.zeros((0, 0))

    def summary(self, kind: str, z: float = 4.0) -> Dict[str, Any]:
        X = self.vectors(kind)
        n = int(X.shape[0])
        if n < 3:
            return {"n": n, "mean": None, "z": None, "structured": False}
        mean = X.mean(0)
        sem = X.std(0, ddof=1) / math.sqrt(n) + 1e-12
        zs = np.abs(mean) / sem
        return {"n": n, "mean": mean, "z": zs,
                "structured": bool(np.any(zs > z))}


# ---------------------------------------------------------------- CV fits
def _design(times: np.ndarray, t0: float) -> np.ndarray:
    return np.stack([np.ones_like(times), times - t0], axis=1)


def fit_cv(times: np.ndarray, pos: np.ndarray, t0: float):
    """Least-squares x(t) = p + v (t - t0) per coordinate.
    -> (beta (2, 2), (X^T X)^-1)."""
    X = _design(times.astype(np.float64), t0)
    XtX = X.T @ X + 1e-9 * np.eye(2)
    inv = np.linalg.inv(XtX)
    beta = inv @ X.T @ pos
    return beta, inv


def predictive_loglik(beta, inv, t0, times, pos, sigma) -> float:
    X = _design(times.astype(np.float64), t0)
    mu = X @ beta
    var = sigma ** 2 * (1.0 + np.einsum("ni,ij,nj->n", X, inv, X))
    r2 = ((pos - mu) ** 2).sum(1)
    return float(np.sum(-0.5 * r2 / var - np.log(2 * math.pi * var)))


def _mixture_loglik(comps, times, pos, sigma) -> float:
    lls = []
    for beta, inv, t0 in comps:
        X = _design(times.astype(np.float64), t0)
        mu = X @ beta
        var = sigma ** 2 * (1.0 + np.einsum("ni,ij,nj->n", X, inv, X))
        lls.append(-0.5 * ((pos - mu) ** 2).sum(1) / var
                   - np.log(2 * math.pi * var) + math.log(1.0 / len(comps)))
    L = np.stack(lls, 0)
    mx = L.max(0)
    return float(np.sum(mx + np.log(np.exp(L - mx).sum(0))))


@dataclass(frozen=True)
class Proposal:
    kind: str                       # "merge" | "split"
    tracks: Tuple[str, ...]
    prior_score: float              # lower = evaluated first
    key: str = ""


@dataclass
class RevisionOutcome:
    proposal: Proposal
    gain_nats: float
    accepted: bool
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)


class RevisionEngine:
    def __init__(self, max_proposals: int = 2, min_gain_nats: float = 5.0,
                 holdout: int = 6, min_train: int = 3, merge_window: int = 12,
                 merge_max_gap: int = 30, merge_radius_sd: float = 4.0,
                 split_window: int = 24, split_holdout: int = 8,
                 cooldown: int = 10):
        if int(max_proposals) < 1:
            raise PerceptionError("max_proposals must be >= 1")
        self.max_proposals = int(max_proposals)
        self.min_gain = float(min_gain_nats)
        self.holdout = int(holdout)
        self.min_train = int(min_train)
        self.merge_window = int(merge_window)
        self.merge_max_gap = int(merge_max_gap)
        self.merge_radius_sd = float(merge_radius_sd)
        self.split_window = int(split_window)
        self.split_holdout = int(split_holdout)
        if self.split_window < self.split_holdout + 6:
            raise PerceptionError("split_window must exceed split_holdout + 6")
        self.cooldown = int(cooldown)
        self._cool: Dict[str, int] = {}
        self._step = 0
        self.evaluated_per_step: List[int] = []
        self.log: List[RevisionOutcome] = []

    # ----------------------------------------------------------- propose
    def propose(self, tracker: IdentityTracker) -> List[Proposal]:
        sigma = tracker.cfg.meas_std
        out: List[Proposal] = []
        live = tracker.tracks(("confirmed", "occluded"))
        dead = [t for t in tracker.tracks(("lost",))
                if not t.end_reason.startswith("merged_into")
                and t.end_reason != "tentative_died"]
        # merge: b live & young-ish, a occluded/lost ended before b began
        for b in live:
            if b.state != "confirmed":
                continue
            fb = {f for f, _ in b.history}
            b0 = min(fb)
            for a in dead + [t for t in live if t.state == "occluded"]:
                if a.local == b.local:
                    continue
                fa = {f for f, _ in a.history}
                if fa & fb or max(fa) >= b0 or b0 - max(fa) > self.merge_max_gap:
                    continue
                if len(a.history) < self.min_train:
                    continue
                ta = np.array([f for f, _ in a.history[-self.merge_window:]], float)
                pa = np.stack([p for _, p in a.history[-self.merge_window:]])
                beta, inv = fit_cv(ta, pa, ta[-1])
                pred = beta[0] + beta[1] * (b0 - ta[-1])
                xb = np.array([1.0, b0 - ta[-1]])
                sd = sigma * math.sqrt(1 + xb @ inv @ xb) + \
                    tracker.cfg.process_std * (b0 - ta[-1]) ** 1.5
                dist = float(np.linalg.norm(b.history[0][1] - pred))
                if dist > self.merge_radius_sd * sd * math.sqrt(2):
                    continue
                out.append(Proposal("merge", (a.local, b.local), dist / sd,
                                    f"merge:{a.local}:{b.local}"))
        # split: confirmed tracks with enough history whose innovations
        # are larger than the measurement model allows
        for t in live:
            if t.state != "confirmed" or len(t.history) < self.split_window:
                continue
            inn = np.stack([v for _, v in t.innovations[-self.split_window:]]) \
                if len(t.innovations) >= self.split_window else None
            if inn is None:
                continue
            ratio = float(np.mean((inn ** 2).sum(1)) / (2 * sigma ** 2))
            if ratio < 2.0:
                continue
            out.append(Proposal("split", (t.local,), 1.0 / ratio,
                                f"split:{t.local}"))
        out.sort(key=lambda p: p.prior_score)
        return out

    # ---------------------------------------------------------- evaluate
    def evaluate_merge(self, tracker, a_local: str, b_local: str):
        sigma = tracker.cfg.meas_std
        a, b = tracker.track(a_local), tracker.track(b_local)
        if len(b.history) < self.min_train + self.holdout:
            return None, {"why": "b too short"}
        # FIXED DESIGN: train on b's first min_train points, hold out the
        # next `holdout`. The question is "does a's trajectory predict b's
        # early life better than b's own first points do" — re-asking it
        # later with b's long history would only measure b's own fit.
        tb = np.array([f for f, _ in b.history], float)
        pb = np.stack([p for _, p in b.history])
        k, h = self.min_train, self.holdout
        tr_t, tr_p = tb[:k], pb[:k]
        ho_t, ho_p = tb[k:k + h], pb[k:k + h]
        ta = np.array([f for f, _ in a.history[-self.merge_window:]], float)
        pa = np.stack([p for _, p in a.history[-self.merge_window:]])
        t0 = tr_t[-1]
        bs, ins = fit_cv(tr_t, tr_p, t0)
        bm, inm = fit_cv(np.concatenate([ta, tr_t]),
                         np.concatenate([pa, tr_p]), t0)
        ll_cur = predictive_loglik(bs, ins, t0, ho_t, ho_p, sigma)
        ll_rev = predictive_loglik(bm, inm, t0, ho_t, ho_p, sigma)
        return ll_rev - ll_cur, {"ll_current": ll_cur, "ll_revised": ll_rev}

    def _two_lines(self, times, pos, t0):
        b1, inv1 = fit_cv(times, pos, t0)
        r = pos - _design(times, t0) @ b1
        u, s, vt = np.linalg.svd(r - r.mean(0), full_matrices=False)
        lab = (r @ vt[0] > np.median(r @ vt[0])).astype(int)
        comps = None
        for _ in range(8):
            if lab.min() == lab.max() or min(np.bincount(lab, minlength=2)) < 2:
                return None
            comps = [fit_cv(times[lab == k], pos[lab == k], t0) for k in (0, 1)]
            d = np.stack([((pos - _design(times, t0) @ c[0]) ** 2).sum(1)
                          for c in comps], 1)
            new = d.argmin(1)
            if np.array_equal(new, lab):
                break
            lab = new
        return comps, lab

    def evaluate_split(self, tracker, local: str):
        sigma = tracker.cfg.meas_std
        t = tracker.track(local)
        h = t.history[-self.split_window:]
        times = np.array([f for f, _ in h], float)
        pos = np.stack([p for _, p in h])
        tr_t, tr_p = times[:-self.split_holdout], pos[:-self.split_holdout]
        ho_t, ho_p = times[-self.split_holdout:], pos[-self.split_holdout:]
        t0 = tr_t[-1]
        b1, inv1 = fit_cv(tr_t, tr_p, t0)
        ll_cur = predictive_loglik(b1, inv1, t0, ho_t, ho_p, sigma)
        two = self._two_lines(tr_t, tr_p, t0)
        if two is None:
            return None, {"why": "degenerate split"}
        comps, lab = two
        ll_rev = _mixture_loglik([(c[0], c[1], t0) for c in comps],
                                 ho_t, ho_p, sigma)
        return ll_rev - ll_cur, {"ll_current": ll_cur, "ll_revised": ll_rev,
                                 "comps": comps, "labels": lab,
                                 "train_frames": tr_t, "t0": t0}

    # -------------------------------------------------------------- step
    def step(self, tracker: IdentityTracker) -> List[RevisionOutcome]:
        self._step += 1
        cands = [p for p in self.propose(tracker)
                 if self._cool.get(p.key, -10 ** 9) <= self._step]
        outcomes = []
        touched = set()
        for p in cands:
            if len(outcomes) >= self.max_proposals:
                break
            if touched & set(p.tracks):
                continue
            if p.kind == "merge":
                gain, det = self.evaluate_merge(tracker, *p.tracks)
            else:
                gain, det = self.evaluate_split(tracker, p.tracks[0])
            if gain is None:
                continue            # not enough data yet: not counted, retried later
            ok = gain >= self.min_gain
            reason = "accepted" if ok else "insufficient predictive gain"
            if ok:
                touched |= set(p.tracks)
                if p.kind == "merge":
                    tracker.apply_merge(p.tracks[0], p.tracks[1],
                                        f"gain={gain:.2f}")
                else:
                    self._apply_split(tracker, p.tracks[0], det, gain)
            else:
                self._cool[p.key] = self._step + self.cooldown
            o = RevisionOutcome(p, float(gain), ok, reason,
                                {k: v for k, v in det.items()
                                 if k in ("ll_current", "ll_revised")})
            outcomes.append(o)
        self.evaluated_per_step.append(len(outcomes))
        self.log.extend(outcomes)
        return outcomes

    def _apply_split(self, tracker, local, det, gain) -> None:
        t = tracker.track(local)
        comps, t0 = det["comps"], det["t0"]
        now = float(tracker.frame)
        sigma = tracker.cfg.meas_std
        # Re-label the WHOLE window (train + holdout) by nearest component,
        # so neither track keeps the other's points, then REFIT each
        # component on all its points at t0 = now (extrapolating the
        # train-only fit across the holdout was measured to misplace the
        # components by ~2 sigma and cause re-mixing).
        hist = t.history[-self.split_window:]
        groups: List[List[Tuple[int, np.ndarray]]] = [[], []]
        for f, p in hist:
            d = [float(np.linalg.norm(p - (beta[0] + beta[1] * (f - t0))))
                 for beta, _ in comps]
            groups[int(np.argmin(d))].append((f, p))
        if min(len(g) for g in groups) < 2:
            return
        states, covs = [], []
        for g in groups:
            tt = np.array([f for f, _ in g], float)
            pp = np.stack([p for _, p in g])
            beta, inv = fit_cv(tt, pp, now)
            states.append((beta[0].copy(), beta[1].copy()))
            # Covariance from THIS component's regression, not from the
            # pre-split track (which was confidently wrong about both).
            pv, vv, pc = (sigma ** 2 * inv[0, 0], sigma ** 2 * inv[1, 1],
                          sigma ** 2 * inv[0, 1])
            covs.append(np.array([[pv, 0, pc, 0], [0, pv, 0, pc],
                                  [pc, 0, vv, 0], [0, pc, 0, vv]]))
        # Component a = the one that explains the track's latest detection.
        last = t.history[-1][1]
        ia = int(np.argmin([np.linalg.norm(s[0] - last) for s in states]))
        ib = 1 - ia
        tracker.apply_split(local, states[ia], states[ib], groups[ia],
                            groups[ib], f"gain={gain:.2f}",
                            covs=(covs[ia], covs[ib]))
