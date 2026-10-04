"""Model mismatch / rule change, kept distinct from measurement noise
(plan Stage 7.5).

Input each step: the model's PREDICTIVE distribution for an observation
(mean mu, variance var — per dimension, the variance INCLUDING the noise the
model expects) and the observation y. Standardise: z = (y - mu)/sqrt(var).
If the model is right, z ~ N(0, I) independently over time, whatever the
noise level, because var already accounts for it. Two statistics then have
mean 0 and unit variance under the model:

    mean shift   m_t = sum_d z_d / sqrt(D)
                 a rule change that biases the outcome (a new force, a
                 changed gain) makes z consistently signed
    dispersion   q_t = (sum_d z_d^2 - D) / sqrt(2D)
                 a rule change that makes outcomes less predictable than
                 the model claims (or a noise level the model no longer
                 knows) inflates z^2

Each feeds a CUSUM: S_t = max(0, S_{t-1} + g_t), alarm when S_t > h.

    mean shift   g = +-clip(m_t, mean_clip) - k_mean (two-sided), h_mean = 10.
                 A 1-sd bias is caught in ~8-30 steps at D = 1 and ~4 at
                 D = 6; mean_clip = 4 means one step adds at most 3.5.
    dispersion   g = sum_d [ a * min(z_d^2, disp_clip) - b ], the LOG-
                 LIKELIHOOD RATIO of "variance is disp_ratio x what the
                 model claims" against "the model is right":
                     a = (1 - 1/disp_ratio) / 2,  b = log(disp_ratio) / 2.
                 h_disp is in nats (default 16).

WHY THE DISPERSION STATISTIC CHANGED (2026-10-03, verifier finding F5,
experiments/foundation_ab/stage9_11). It used to be g = q_t - k_disp with
k_disp = 1, i.e. it drifted UP only when E[z^2] > 1 + sqrt(2) = 2.41. A
slope change 2 -> 3 at noise 0.5 inflates E[z^2] to 2.33 with an
unconditional mean of zero (the error is x-dependent and symmetric), so the
statistic drifted DOWN and alarms came only from chance excursions: the
reviser kept the stale root on 4/5 seeds. The LLR form with
disp_ratio = 3 drifts up for any variance ratio above
r ln r / (r - 1) = 1.65, and is the Page-optimal CUSUM at 3x. MEASURED
(D = 1, 400 runs of that residual, x + N(0, 0.5^2) scored at sd 0.5):
old: first alarm within 80 / 160 rows in 49% / 76%, median 83 rows; new
(disp_ratio 3, h_disp 16): 74% / 99%, median 61. Null (unit noise of the
KNOWN variance), 250 x 4000 steps at each of D = 1, 2, 6: 0 dispersion
alarms (old: 1 at D = 1). A model whose variance is understated 1.2x: 0
alarms per 1000 steps (old 0); 1.4x: 0.08 (old 0.03); 2x: 6.9 (old 2.5).
WHY NOT MORE SENSITIVE: h_disp 12 / disp_ratio 2.5 caught the slope change
faster (median 46) but fired once on the stage 7 ensemble's IN-distribution
predictive (8 dims, per-dim E[z^2] 0.95-1.31, temporally correlated) —
tests/_foundation_inference_smoke.py G — where the old statistic peaked far
below its threshold; at 3 / 16 that stream peaks at 13.3 of 16.

disp_clip (default 16 = a 4-sd residual) caps one dimension's contribution
per step at a * 16 - b = 4.8 nats, and mean_clip caps the mean statistic's
step at 3.5, so a SINGLE large |z| — one unlucky draw, one outlier — is
never an alarm by itself (before: the dispersion statistic fired on any one
|z| > 6.5 at D = 1, the mean statistic on any |z| > 10.5); a real change
persists and still fires within a few steps (variance 16x the model's:
median 6 steps, old 2).

MEASURED with the defaults at D = 1 for the mean statistic: one alarm in
50 x 4000 steps of pure unit noise. After an alarm the statistics reset,
so the detector can fire again (it is not a latch, CLAUDE.md §4.1);
`alarms` keeps (step, kind). A missing observation is skipped, not scored
as zero error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import ContractError, is_missing


class PredictiveCUSUM:
    def __init__(self, k_mean: float = 0.5, h_mean: float = 10.0,
                 disp_ratio: float = 3.0, h_disp: float = 16.0, disp_clip: float = 16.0,
                 mean_clip: float = 4.0):
        for nm, v in (("k_mean", k_mean), ("h_mean", h_mean), ("disp_ratio", disp_ratio),
                      ("h_disp", h_disp), ("disp_clip", disp_clip), ("mean_clip", mean_clip)):
            if not (np.isfinite(v) and v > 0):
                raise ContractError(f"{nm} must be finite and > 0")
        if disp_ratio <= 1:
            raise ContractError("disp_ratio must be > 1 (the variance inflation to detect)")
        if disp_clip <= 1:
            raise ContractError("disp_clip must be > 1 (it caps z^2; E[z^2] = 1 under the model)")
        self.k_mean, self.h_mean, self.mean_clip = float(k_mean), float(h_mean), float(mean_clip)
        self.disp_ratio, self.h_disp, self.disp_clip = (float(disp_ratio), float(h_disp),
                                                        float(disp_clip))
        self._disp_a = 0.5 * (1.0 - 1.0 / self.disp_ratio)
        self._disp_b = 0.5 * float(np.log(self.disp_ratio))
        self.s_hi = self.s_lo = self.s_disp = 0.0
        self.step = 0
        self.skipped = 0
        self.alarms: List[Tuple[int, str]] = []

    def reset(self):
        self.s_hi = self.s_lo = self.s_disp = 0.0

    def update(self, mu, var, y) -> Optional[Dict[str, Any]]:
        if y is None or is_missing(y):
            self.skipped += 1
            return None
        mu = np.atleast_1d(np.asarray(mu, float))
        var = np.atleast_1d(np.asarray(var, float))
        y = np.atleast_1d(np.asarray(y, float))
        if not (mu.shape == var.shape == y.shape) or mu.ndim != 1:
            raise ContractError(f"mu/var/y shapes differ: {mu.shape} {var.shape} {y.shape}")
        if np.any(var <= 0) or not np.all(np.isfinite(np.concatenate([mu, var, y]))):
            raise ContractError("predictive variance must be > 0 and inputs finite")
        z = (y - mu) / np.sqrt(var)
        D = len(z)
        m = z.sum() / np.sqrt(D)
        q = ((z * z).sum() - D) / np.sqrt(2 * D)
        mc = float(np.clip(m, -self.mean_clip, self.mean_clip))
        self.s_hi = max(0.0, self.s_hi + mc - self.k_mean)
        self.s_lo = max(0.0, self.s_lo - mc - self.k_mean)
        g = float(np.sum(self._disp_a * np.minimum(z * z, self.disp_clip) - self._disp_b))
        self.s_disp = max(0.0, self.s_disp + g)
        kind = None
        if max(self.s_hi, self.s_lo) > self.h_mean:
            kind = "mean_shift"
        elif self.s_disp > self.h_disp:
            kind = "dispersion"
        out = {"step": self.step, "z": z, "m": float(m), "q": float(q),
               "s_mean": max(self.s_hi, self.s_lo), "s_disp": self.s_disp,
               "alarm": kind is not None, "kind": kind}
        if kind is not None:
            self.alarms.append((self.step, kind))
            self.reset()
        self.step += 1
        return out

    def update_gaussian(self, mu, cov, y):
        """Full-covariance predictive (e.g. a Kalman innovation): whiten with
        the Cholesky factor, then score as independent unit dims."""
        cov = np.atleast_2d(np.asarray(cov, float))
        if y is None or is_missing(y):
            self.skipped += 1
            return None
        L = np.linalg.cholesky(cov)
        w = np.linalg.solve(L, np.atleast_1d(np.asarray(y, float)) - np.atleast_1d(mu))
        return self.update(np.zeros_like(w), np.ones_like(w), w)
