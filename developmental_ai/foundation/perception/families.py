"""Alternative state families behind one StateBelief interface (plan Stage 8
item 4).

WHAT IS CLAIMED
    Object decomposition is one hypothesis about the world, not a given. A
    diffusing field has no persistent compact things; a blinking indicator is
    one thing whose STATE is the information. Three families are provided:

      ObjectFamily          IdentityTracker over field peaks; predicts the
                            next frame by rendering tracked blobs forward.
      FieldFamily           a grid of values with a learned shift-invariant
                            3x3 linear operator (+ bias), fit by least squares.
      DiscreteProcessFamily K latent states (k-means on frames) with a
                            Dirichlet-smoothed transition matrix and an HMM
                            forward filter.

    All three emit `StateBelief`s with per-variable uncertainty and the same
    record fields, and all three implement `fit(train)` and
    `one_step_predictions(frames)` so they can be compared on held-out
    one-step prediction error by `compare_families`.

SELECTION IS DECLARED, NOT ASSUMED
    `compare_families` returns scores and a recommendation and NOTHING ELSE —
    it hands back no family object to encode with. Beliefs are produced only
    through a `FamilyDeclaration` (`declare_family`), which names the family,
    states a non-empty reason, carries the evidence it was declared on (if
    any) and records `against_evidence=True` when the declaration contradicts
    the comparison. Every belief carries that declaration in its payload.
    There is no default family: code that never declared one cannot encode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..contracts import ObsRef, Scope, StateBelief, UNKNOWN
from .errors import PerceptionError, UndeclaredFamilyError
from .identity import IdentityTracker, TrackerConfig

FIELD_REPRESENTATION = "field_grid"
DISCRETE_REPRESENTATION = "discrete_process"
OBJECT_REPRESENTATION = "object_tracks"


def _frames(frames) -> np.ndarray:
    a = np.asarray(frames, dtype=np.float64)
    if a.ndim != 3 or a.shape[0] < 2:
        raise PerceptionError(f"frames must be (T>=2, H, W), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise PerceptionError("frames must be finite")
    return a


def _belief_common(rep, registry, scope, seq, support, tag):
    if not isinstance(scope, Scope):
        raise PerceptionError("scope must be a contracts Scope")
    sup = tuple(support)
    if not sup:
        raise PerceptionError("a belief needs at least one supporting "
                              "observation (support=())")
    for s in sup:
        if not isinstance(s, ObsRef) or s.scope != scope or s.seq > seq:
            raise PerceptionError(f"support {s!r} is not an earlier/same-step "
                                  f"observation in scope {scope}")
    major, semver = registry.stamp(rep)
    bid = f"{rep}:{scope.environment}/{scope.stream}/{scope.episode}:{seq}:{tag}"
    return major, semver, sup, bid


class StateFamily:
    name = ""
    representation = ""

    def fit(self, train: np.ndarray) -> "StateFamily":       # pragma: no cover
        raise NotImplementedError

    def one_step_predictions(self, frames: np.ndarray) -> np.ndarray:  # pragma: no cover
        """pred[t] predicts frames[t+1] from frames[:t+1]; shape (T-1, H, W)."""
        raise NotImplementedError


# ------------------------------------------------------------------ field
def _patches(f: np.ndarray) -> np.ndarray:
    p = np.pad(f, 1, mode="edge")
    h, w = f.shape
    cols = [p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]
    return np.stack(cols, -1)                                  # (H, W, 9)


class FieldFamily(StateFamily):
    name = "field"
    representation = FIELD_REPRESENTATION

    def __init__(self, ridge: float = 1e-6):
        self.ridge = float(ridge)
        self.w: Optional[np.ndarray] = None
        self.resid_std: Optional[float] = None

    def fit(self, train) -> "FieldFamily":
        tr = _frames(train)
        X = np.concatenate([_patches(f).reshape(-1, 9) for f in tr[:-1]])
        X = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
        y = tr[1:].reshape(-1)
        self.w = np.linalg.solve(X.T @ X + self.ridge * np.eye(10), X.T @ y)
        self.resid_std = float(np.std(X @ self.w - y))
        return self

    def predict(self, f: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise PerceptionError("FieldFamily.fit() first")
        return _patches(f) @ self.w[:9] + self.w[9]

    def one_step_predictions(self, frames) -> np.ndarray:
        fr = _frames(frames)
        return np.stack([self.predict(f) for f in fr[:-1]])

    def _belief(self, field_values, registry, scope, seq, support, *,
                std=None, extra: Optional[Dict[str, Any]] = None,
                extra_unc: Optional[Dict[str, Any]] = None,
                representation: Optional[str] = None,
                payload: Optional[Dict[str, Any]] = None) -> StateBelief:
        rep = representation or self.representation
        major, semver, sup, bid = _belief_common(rep, registry, scope, seq,
                                                 support, "field")
        f = np.asarray(field_values, dtype=np.float64)
        if f.ndim != 2 or not np.all(np.isfinite(f)):
            raise PerceptionError(f"field must be finite (H, W), got {f.shape}")
        if std is None:
            unc = UNKNOWN
        else:
            unc = np.broadcast_to(np.asarray(std, np.float64), f.shape).copy()
            if np.any(unc < 0):
                raise PerceptionError("field std must be >= 0")
        variables = {"field": f.copy()}
        uncertainty = {"field": unc}
        for k, v in (extra or {}).items():
            variables[k] = v
        for k, v in (extra_unc or {}).items():
            uncertainty[k] = v
        pl = {"representation_semver": semver, "family": self.name,
              "shape": tuple(int(s) for s in f.shape)}
        pl.update(payload or {})
        return StateBelief(belief_id=bid, representation=rep,
                           representation_version=major, variables=variables,
                           uncertainty=uncertainty, support=sup, payload=pl)


# --------------------------------------------------------------- discrete
class DiscreteProcessFamily(StateFamily):
    name = "discrete"
    representation = DISCRETE_REPRESENTATION

    def __init__(self, n_states: int = 3, alpha: float = 1.0, seed: int = 0):
        if int(n_states) < 2:
            raise PerceptionError("n_states must be >= 2")
        self.k = int(n_states)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.centroids = None
        self.T = None
        self.sigma = None

    def fit(self, train) -> "DiscreteProcessFamily":
        tr = _frames(train)
        X = tr.reshape(tr.shape[0], -1)
        rng = np.random.default_rng(self.seed)
        # k-means++ init, then Lloyd
        c = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d = np.min(((X[:, None] - np.array(c)[None]) ** 2).sum(-1), 1)
            c.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0
                     else X[rng.integers(len(X))])
        C = np.array(c)
        for _ in range(50):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == k].mean(0) if np.any(lab == k) else C[k]
                             for k in range(self.k)])
            if np.allclose(newC, C):
                break
            C = newC
        self.centroids = C
        cnt = np.full((self.k, self.k), self.alpha)
        for a, b in zip(lab[:-1], lab[1:]):
            cnt[a, b] += 1
        self.T = cnt / cnt.sum(1, keepdims=True)
        self.sigma = float(max(1e-3, np.sqrt(np.mean((X - C[lab]) ** 2))))
        return self

    def _emission_logp(self, x: np.ndarray) -> np.ndarray:
        d = ((self.centroids - x[None]) ** 2).sum(1)
        return -0.5 * d / self.sigma ** 2

    def filter(self, frames) -> np.ndarray:
        """Posterior state probabilities after each frame, (T, K)."""
        if self.T is None:
            raise PerceptionError("DiscreteProcessFamily.fit() first")
        fr = _frames(frames)
        X = fr.reshape(fr.shape[0], -1)
        prior = np.full(self.k, 1.0 / self.k)
        out = []
        for x in X:
            lp = np.log(prior + 1e-300) + self._emission_logp(x)
            p = np.exp(lp - lp.max())
            p /= p.sum()
            out.append(p)
            prior = p @ self.T
        return np.stack(out)

    def one_step_predictions(self, frames) -> np.ndarray:
        fr = _frames(frames)
        post = self.filter(fr)
        nxt = post[:-1] @ self.T                                  # (T-1, K)
        return (nxt @ self.centroids).reshape((-1,) + fr.shape[1:])

    def _belief(self, state_probs, registry, scope, seq, support,
                payload: Optional[Dict[str, Any]] = None) -> StateBelief:
        major, semver, sup, bid = _belief_common(self.representation, registry,
                                                 scope, seq, support, "proc")
        p = np.asarray(state_probs, dtype=np.float64)
        if p.shape != (self.k,) or np.any(p < 0) or abs(p.sum() - 1) > 1e-6:
            raise PerceptionError(f"state_probs must be a ({self.k},) "
                                  f"distribution")
        ent = float(-(p * np.log(p + 1e-300)).sum())
        pl = {"representation_semver": semver, "family": self.name,
              "transition": self.T.copy(), "uncertainty_kind": "entropy_nats"}
        pl.update(payload or {})
        return StateBelief(belief_id=bid, representation=self.representation,
                           representation_version=major,
                           variables={"state_probs": p.copy(),
                                      "map_state": int(p.argmax())},
                           uncertainty={"state_probs": ent, "map_state": ent},
                           support=sup, payload=pl)


# ----------------------------------------------------------------- object
def detect_peaks(f: np.ndarray, thresh: float):
    """Strict-ish 3x3 local maxima above thresh -> (positions (M,2) as
    (col, row) sub-pixel, amplitudes (M,))."""
    h, w = f.shape
    P = _patches(f)
    centre = P[..., 4]
    is_max = (centre >= P.max(-1)) & (centre > thresh)
    rows, cols = np.nonzero(is_max)
    pos, amp = [], []
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        win = np.clip(f[r0:r1, c0:c1], 0, None)
        s = win.sum()
        yy, xx = np.mgrid[r0:r1, c0:c1]
        pos.append(((xx * win).sum() / s, (yy * win).sum() / s) if s > 0
                   else (float(c), float(r)))
        amp.append(float(f[r, c]))
    return (np.array(pos, dtype=np.float64).reshape(-1, 2),
            np.array(amp, dtype=np.float64))


class ObjectFamily(StateFamily):
    name = "objects"
    representation = OBJECT_REPRESENTATION

    def __init__(self, scope: Scope, peak_thresh: float = 0.3,
                 sigmas: Sequence[float] = (0.8, 1.2, 1.6, 2.0, 2.5),
                 config: Optional[TrackerConfig] = None):
        self.scope = scope
        self.peak_thresh = float(peak_thresh)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.cfg = config or TrackerConfig(
            meas_std=0.3, process_std=0.1, init_vel_std=1.0,
            clutter_density=0.01, feature_kappa=1.0, gate_maha2=25.0,
            confirm_hits=2, reid_window=3)
        self.sigma = self.sigmas[0]
        self._n = 0

    def _render(self, shape, centres, amps, sigma) -> np.ndarray:
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        out = np.zeros(shape)
        for (cx, cy), a in zip(centres, amps):
            out += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        return out

    def _run(self, frames, sigma) -> np.ndarray:
        self._n += 1
        tr = IdentityTracker(Scope(self.scope.environment, self.scope.stream,
                                   f"{self.scope.episode}#objfam{self._n}"),
                             self.cfg)
        preds = []
        for f in frames[:-1]:
            pos, amp = detect_peaks(f, self.peak_thresh)
            tr.step(pos, np.ones((len(pos), 1)), attrs=amp.reshape(-1, 1))
            nxt = tr.predict_positions(states=("confirmed", "tentative"))
            centres = list(nxt.values())
            amps = [float(tr.track(k).attrs[0]) for k in nxt]
            preds.append(self._render(f.shape, centres, amps, sigma))
        return np.stack(preds)

    def fit(self, train) -> "ObjectFamily":
        tr = _frames(train)
        errs = [float(np.mean((self._run(tr, s) - tr[1:]) ** 2))
                for s in self.sigmas]
        self.sigma = self.sigmas[int(np.argmin(errs))]
        return self

    def one_step_predictions(self, frames) -> np.ndarray:
        return self._run(_frames(frames), self.sigma)


# ------------------------------------------------------------- selection
@dataclass(frozen=True)
class FamilyComparison:
    scores: Dict[str, float]           # held-out one-step MSE (lower = better)
    recommended: str
    n_train: int
    n_test: int


def compare_families(frames, families: Dict[str, StateFamily],
                     train_frac: float = 0.6) -> FamilyComparison:
    fr = _frames(frames)
    n_train = int(round(fr.shape[0] * float(train_frac)))
    if n_train < 2 or fr.shape[0] - n_train < 2:
        raise PerceptionError("not enough frames to compare families")
    scores = {}
    for name, fam in families.items():
        fam.fit(fr[:n_train])
        pred = fam.one_step_predictions(fr)
        # pred[t] targets fr[t+1]; held-out targets are t+1 >= n_train
        scores[name] = float(np.mean((pred[n_train - 1:] - fr[n_train:]) ** 2))
    best = min(scores, key=scores.get)
    return FamilyComparison(scores, best, n_train, fr.shape[0] - n_train)


@dataclass(frozen=True)
class FamilyDeclaration:
    name: str
    family: StateFamily
    reason: str
    evidence: Optional[FamilyComparison] = None
    against_evidence: bool = False

    def record(self) -> Dict[str, Any]:
        return {"name": self.name, "reason": self.reason,
                "against_evidence": bool(self.against_evidence),
                "evidence_scores": (dict(self.evidence.scores)
                                    if self.evidence else UNKNOWN)}

    def belief(self, *args, **kwargs) -> StateBelief:
        pl = dict(kwargs.pop("payload", None) or {})
        pl["family_declaration"] = self.record()
        return self.family._belief(*args, payload=pl, **kwargs)


def declare_family(name: str, family: StateFamily, reason: str,
                   evidence: Optional[FamilyComparison] = None
                   ) -> FamilyDeclaration:
    if not isinstance(family, StateFamily):
        raise UndeclaredFamilyError(f"{name!r} is not a StateFamily")
    if not hasattr(family, "_belief"):
        raise UndeclaredFamilyError(f"family {name!r} cannot emit beliefs")
    if not isinstance(reason, str) or not reason.strip():
        raise UndeclaredFamilyError("a family declaration must state a reason")
    against = bool(evidence is not None and evidence.recommended != name)
    return FamilyDeclaration(name, family, reason, evidence, against)


def _object_belief(self, tracker, registry, scope, seq, support,
                   payload: Optional[Dict[str, Any]] = None) -> List[StateBelief]:
    """Object family beliefs ARE the tracker's per-entity beliefs, stamped
    with the family declaration. Returns a list (one per entity)."""
    if not isinstance(tracker, IdentityTracker) or tracker.scope != scope:
        raise PerceptionError("object beliefs need the IdentityTracker of "
                              "this scope")
    _belief_common(self.representation, registry, scope, seq, support, "obj")
    out = []
    for b in tracker.beliefs(registry, support=support):
        pl = dict(b.payload)
        pl.update(payload or {})
        pl["family"] = self.name
        out.append(b.replace(payload=pl))
    return out


ObjectFamily._belief = _object_belief
