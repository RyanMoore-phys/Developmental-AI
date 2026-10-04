"""Reference scorer: a point/gaussian Prediction against its outcome.

The producer for the store's prediction_score interpretations (what failure
and contradiction retrieval read). Scores are interpretations because the
scoring rule is revisable; the prediction and the outcome are not.

    error        RMSE between outcome["mean"] and the observed value of
                 `channel` at seq context_seq + horizon
    contradicts  gaussian: any component outside z_tol standard deviations;
                 point: error > tol (a point prediction must declare tol —
                 there is no default threshold for "disagrees")
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..contracts import Evidence, Prediction, is_missing
from .errors import ExperienceError


def score_point_prediction(pred: Prediction, ev: Evidence, channel: str,
                           context_seq: int, tol: Optional[float] = None,
                           z_tol: float = 3.0) -> Tuple[float, bool]:
    kind = pred.outcome.get("kind")
    if kind not in ("point", "gaussian"):
        raise ExperienceError(f"no reference scorer for outcome kind {kind!r}")
    target = context_seq + pred.horizon
    obs = [o for o in ev.observations if o.channel == channel and o.seq == target]
    if not obs:
        raise ExperienceError(f"evidence has no {channel!r} observation at seq "
                              f"{target} (context {context_seq} + horizon "
                              f"{pred.horizon})")
    val = obs[0].value
    if is_missing(val):
        raise ExperienceError(f"{channel}@{target} is {val!r}; a missing "
                              f"reading cannot score a prediction")
    y = np.asarray(val, dtype=float).ravel()
    mu = np.asarray(pred.outcome["mean"], dtype=float).ravel()
    if y.shape != mu.shape:
        raise ExperienceError(f"predicted shape {mu.shape} != observed {y.shape}")
    err = float(math.sqrt(float(np.mean((y - mu) ** 2))))
    if kind == "gaussian":
        sd = np.asarray(pred.outcome["std"], dtype=float).ravel()
        if sd.shape != mu.shape or (sd <= 0).any():
            raise ExperienceError("gaussian outcome needs std > 0 per component")
        return err, bool((np.abs(y - mu) / sd > z_tol).any())
    if tol is None:
        raise ExperienceError("a point prediction needs an explicit tol to "
                              "decide contradiction")
    return err, err > float(tol)
