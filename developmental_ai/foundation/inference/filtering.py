"""State filtering: a predict / update cycle (plan Stage 7.3-7.4).

KalmanFilter — linear-Gaussian, exact. x' = F x + B u + w, w ~ N(0, Q);
    y = H x + v, v ~ N(0, R). update() uses the Joseph form and
    re-symmetrises P, so P stays symmetric positive semi-definite under
    round-off. Partial observations via a boolean `mask` over y; a missing
    observation (None / ABSENT / UNKNOWN) skips the update and leaves the
    predicted belief exactly as it was — absence is not a zero reading.
    `predictive()` is the distribution of the NEXT observation (H x, H P H'
    + R) — the input change.py scores to tell rule changes from noise.

ParticleFilter — sequential importance resampling for when the posterior
    is MULTIMODAL (e.g. a sign-ambiguous or aliased observation) or the
    dynamics are nonlinear enough that a single Gaussian misrepresents the
    belief. It is the expensive tool: n particles x a model call each step,
    Monte Carlo error O(1/sqrt(ESS)). Use the Kalman filter whenever one
    Gaussian is adequate; the smoke test shows the case where it is not.

    Resampling is SYSTEMATIC and triggered when ESS = 1 / sum(w^2) falls
    below `ess_threshold * n` (not every step: resampling itself throws
    away diversity). Degeneracy diagnostics are recorded every update:
    ess, max weight, resample count, distinct particles after resampling,
    and `collapses` (ESS < 2: essentially one particle carries the belief).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.special import logsumexp

from ..contracts import ContractError, is_missing

_LOG2PI = float(np.log(2 * np.pi))


def _missing(y) -> bool:
    return y is None or is_missing(y)


class KalmanFilter:
    def __init__(self, F, Q, H, R, x0, P0, B=None):
        self.F, self.Q = np.atleast_2d(np.asarray(F, float)), np.atleast_2d(np.asarray(Q, float))
        self.H, self.R = np.atleast_2d(np.asarray(H, float)), np.atleast_2d(np.asarray(R, float))
        self.B = None if B is None else np.atleast_2d(np.asarray(B, float))
        self.x = np.asarray(x0, float).reshape(-1)
        self.P = np.atleast_2d(np.asarray(P0, float))
        n, m = len(self.x), self.H.shape[0]
        for nm, M, sh in (("F", self.F, (n, n)), ("Q", self.Q, (n, n)), ("H", self.H, (m, n)),
                          ("R", self.R, (m, m)), ("P0", self.P, (n, n))):
            if M.shape != sh:
                raise ContractError(f"Kalman {nm} shape {M.shape} != {sh}")
        for nm, M in (("Q", self.Q), ("R", self.R), ("P0", self.P)):
            if not np.allclose(M, M.T) or np.min(np.linalg.eigvalsh(M)) < -1e-12:
                raise ContractError(f"Kalman {nm} must be symmetric PSD")
        self.updates = 0
        self.skipped = 0

    def predict(self, u=None):
        self.x = self.F @ self.x
        if u is not None:
            if self.B is None:
                raise ContractError("control given but no B matrix")
            self.x = self.x + self.B @ np.asarray(u, float).reshape(-1)
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy()

    def predictive(self, mask=None):
        H, R = self._hr(mask)
        return H @ self.x, H @ self.P @ H.T + R

    def _hr(self, mask):
        if mask is None:
            return self.H, self.R
        m = np.asarray(mask, bool)
        return self.H[m], self.R[np.ix_(m, m)]

    def update(self, y, mask=None) -> Optional[Dict[str, Any]]:
        """Returns the innovation {nu, S, loglik}, or None if y is missing."""
        if _missing(y):
            self.skipped += 1
            return None
        y = np.asarray(y, float).reshape(-1)
        if mask is not None:
            m = np.asarray(mask, bool)
            if not m.any():
                self.skipped += 1
                return None
            y = y[m] if len(y) == len(m) else y
        H, R = self._hr(mask)
        if y.shape != (H.shape[0],):
            raise ContractError(f"observation shape {y.shape} != ({H.shape[0]},)")
        nu = y - H @ self.x
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ nu
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        sign, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (len(nu) * _LOG2PI + logdet + nu @ np.linalg.solve(S, nu))
        self.updates += 1
        return {"nu": nu, "S": S, "loglik": float(ll)}


def ess(weights) -> float:
    w = np.asarray(weights, float)
    return float(1.0 / np.sum(w * w))


def systematic_resample(weights, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(weights, float)
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, pos, side="left")


class ParticleFilter:
    def __init__(self, particles, transition: Callable[[np.ndarray, np.random.Generator], np.ndarray],
                 loglik: Callable[[np.ndarray, Any], np.ndarray], rng: np.random.Generator,
                 ess_threshold: float = 0.5):
        self.particles = np.asarray(particles, float)
        if self.particles.ndim == 1:
            self.particles = self.particles[:, None]
        self.n = len(self.particles)
        if self.n < 2:
            raise ContractError("a particle filter needs >= 2 particles")
        if not 0 < ess_threshold <= 1:
            raise ContractError("ess_threshold must be in (0, 1]")
        if not isinstance(rng, np.random.Generator):
            raise ContractError("ParticleFilter needs an explicit numpy Generator")
        self.transition, self.loglik, self.rng = transition, loglik, rng
        self.ess_threshold = float(ess_threshold)
        self.logw = np.full(self.n, -np.log(self.n))
        self.diag: Dict[str, Any] = {"ess": [], "max_weight": [], "resamples": 0,
                                     "distinct_after_resample": [], "collapses": 0,
                                     "log_evidence": []}

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.logw)

    def predict(self):
        p = np.asarray(self.transition(self.particles, self.rng), float)
        if p.shape != self.particles.shape:
            raise ContractError(f"transition changed particle shape {self.particles.shape} "
                                f"-> {p.shape}")
        self.particles = p

    def update(self, y) -> Optional[Dict[str, float]]:
        if _missing(y):
            return None
        ll = np.asarray(self.loglik(self.particles, y), float)
        if ll.shape != (self.n,):
            raise ContractError(f"loglik must return ({self.n},), got {ll.shape}")
        if np.all(ll == -np.inf):
            raise ContractError("every particle has zero likelihood: the filter has "
                                "lost the state (model mismatch, not noise)")
        joint = self.logw + ll
        z = logsumexp(joint)
        self.logw = joint - z
        e = ess(self.weights)
        d = self.diag
        d["log_evidence"].append(float(z))
        d["ess"].append(e)
        d["max_weight"].append(float(self.weights.max()))
        if e < 2.0:
            d["collapses"] += 1
        resampled = False
        if e < self.ess_threshold * self.n:
            idx = systematic_resample(self.weights, self.rng)
            self.particles = self.particles[idx]
            self.logw = np.full(self.n, -np.log(self.n))
            d["resamples"] += 1
            d["distinct_after_resample"].append(int(len(np.unique(idx))))
            resampled = True
        return {"ess": e, "resampled": resampled, "log_evidence": float(z)}

    def mean(self) -> np.ndarray:
        return self.weights @ self.particles

    def cov(self) -> np.ndarray:
        d = self.particles - self.mean()
        return (self.weights[:, None] * d).T @ d
