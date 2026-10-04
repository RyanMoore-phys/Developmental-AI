"""A finite hypothesis set with log-space Bayesian updating (plan Stage 7.1, 7.6).

    add_hypothesis(name, prior, payload)   new hypothesis takes `prior` mass,
                                           the rest are scaled by (1 - prior)
    update(logliks)                        log w += temperature * loglik, then
                                           normalise with logsumexp, then floor
    probs / map / entropy / log_evidence   reading the belief
    prune(threshold)                       explicit pruning, bounded below

NUMERICAL STABILISATION. Weights live in log space and are renormalised by
logsumexp after every update, so 10^4 updates of log-likelihood -50 do not
underflow. Non-finite log-likelihoods are refused (ContractError): a -inf
would be a hard zero no evidence could ever undo.

RECOVERY FROM OVERCONFIDENCE — the floor (§4.1: a guard needs an escape
path). After each update the posterior is mixed with uniform:

        p <- (1 - K*floor) * p + floor

so every retained hypothesis keeps at least `floor`. Without it, a
hypothesis that lost for T steps is T*L nats behind and needs ~T steps of
contrary evidence to recover; with it, the deficit is capped at about
log(1/floor) nats, so recovery takes ~log(1/floor)/L steps whatever
happened before. The price is a ceiling on certainty: no hypothesis exceeds
1 - (K-1)*floor. floor = 0 is exact Bayes (use it for known-posterior
checks). `temperature` in (0, 1] tempers each likelihood (a deliberately
slower learner, robust to an overconfident likelihood model).

MISSING DATA. `update(None)` / `update(ABSENT)` / `update(UNKNOWN)` changes
nothing and says so (returns False). A partial update — some hypotheses
scored, some not — is refused: comparing a scored hypothesis against an
unscored one would reward the unscored one for silence.

MISSPECIFICATION (2026-10-03, verifier finding F6 / E10d). Bayes over a
set that does not contain the truth still converges — confidently, on the
least-wrong member — and the floor does not stop it: in E10d a shift set
facing a non-shift truth reached > 0.9 on a wrong shift in 100% of
scenarios with nothing to say so. The posterior cannot see this; the
POSTERIOR PREDICTIVE can. `update_categorical(table, outcome)` scores each
observed outcome under the predictive the set implied BEFORE seeing it, and
a MisspecificationMonitor (Page CUSUM of "noisier than predicted" vs "as
predicted", false-alarm run length >= e^h for any predictive) raises
`misspecified` once the evidence passes its threshold, with the evidence
count. It is a STATE with hysteresis: it clears when the statistic returns
to 0 (evidence consistent with the set again) or when the set changes
(add_hypothesis resets it — a changed set is a new question; that is the
escape path, §4.1). A plain `update(logliks)` cannot be monitored (the
predictive needs the whole outcome distribution) and is counted as
`unmonitored`. Consumers (experiment selection, discovery's reviser) read
`hs.misspecified` / `hs.misspecification()`.

LIMITS. `max_hypotheses` bounds the set; adding beyond it first retires the
least probable hypothesis, but pruning NEVER goes below `min_keep` and never
removes the MAP. Retired hypotheses are recorded in `retired` and can be
added again by name — pruning is a reversible decision, not a one-way door.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from scipy.special import logsumexp

from ..contracts import ContractError, is_missing


class MisspecificationMonitor:
    """Page's CUSUM of the log-likelihood ratio "outcomes are noisier than
    the predictive says" against "the predictive is right".

        null   q0 = the predictive (optionally widened by `tolerance`)
        alt    q1 = (1 - contamination) * q0 + contamination * wide
               (categorical: wide = uniform over the M outcomes;
                Gaussian: wide = the same mean with `inflate` x the variance)
        S <- max(0, S + log q1(o) - log q0(o));  misspecified once S > h,
        cleared when S returns to 0 (evidence consistent again) or reset().

    WHY THIS FORM. q1 is a proper distribution, so under q0 the increments
    satisfy E[exp(inc)] = 1 exactly and the false-alarm run length is >= e^h
    WHATEVER the predictive looks like (Wald / Lorden) — one calibrated
    threshold for a sharp binary predictive, a flat one, or a Gaussian. A
    single surprising outcome adds at most log(1 + contamination / (M q0(o)))
    — bounded by the contamination, never a verdict on its own unless the
    predictive called it (near-)impossible, which is what a hard
    contradiction is. An earlier draft standardised the score deficit
    (u = (log q - E log q)/sd) and ran a fixed-allowance CUSUM: on the
    skewed binary scores of a confident set its false-alarm length grew
    only like exp(0.4 h) — MEASURED 20-45% of well-specified 200-step runs
    flagged at the thresholds that caught misspecification — so it was
    replaced.

    DEFAULTS (contamination 0.5, h 8: false-alarm run length >= e^8 ~ 3000
    observations). MEASURED on a 6-shift cue/response set (slip 0.077,
    floor 1e-4, information-gain queries — the verifier's E10d world):
    truth NOT in the set flagged in 48/60 runs by 60 interactions, 56/60 by
    80, 59/60 by 120; well-specified 1/300 runs flagged in 200 interactions.
    h 10: 38 / 50 / 59 of 60, well-specified 0/300. h 7: 50 / 57 / 59,
    1/300.

    `tolerance` widens the null on purpose: a model whose noise scale was
    estimated from a few hundred rows is off by +-10-15% from sampling
    alone, and that is not misspecification (the discovery reviser uses
    tolerance 0.2 on Gaussian scores). `evidence` is the number of
    observations since the statistic last sat at 0; `llr` their summed log
    likelihood ratio.
    """

    def __init__(self, contamination: float = 0.5, h: float = 8.0,
                 tolerance: float = 0.0, inflate: float = 4.0):
        for nm, v in (("contamination", contamination), ("h", h), ("inflate", inflate)):
            if not (np.isfinite(v) and v > 0):
                raise ContractError(f"{nm} must be finite and > 0")
        if not contamination < 1:
            raise ContractError("contamination must be in (0, 1)")
        if not (np.isfinite(tolerance) and 0 <= tolerance < 1):
            raise ContractError("tolerance must be in [0, 1)")
        if inflate <= 1:
            raise ContractError("inflate must be > 1")
        self.contamination, self.h = float(contamination), float(h)
        self.tolerance, self.inflate = float(tolerance), float(inflate)
        self.observed = 0
        self.flags = 0
        self.resets: List[Dict[str, Any]] = []
        self._clear()

    def _clear(self):
        self.statistic = 0.0
        self.misspecified = False
        self.evidence = 0
        self.llr = 0.0
        self.flagged_at: Optional[int] = None

    def reset(self, reason: str = "reset") -> None:
        self.resets.append({"observed": self.observed, "reason": reason,
                            "was_misspecified": self.misspecified})
        self._clear()

    def observe(self, lp_null: float, lp_alt: float) -> Dict[str, Any]:
        """Generic step: log q0(o) and log q1(o) of what happened."""
        lp_null, lp_alt = float(lp_null), float(lp_alt)
        if not (np.isfinite(lp_null) and np.isfinite(lp_alt)):
            raise ContractError("log predictive probabilities must be finite")
        inc = lp_alt - lp_null
        self.observed += 1
        self.statistic = max(0.0, self.statistic + inc)
        if self.statistic > 0:
            self.evidence += 1
            self.llr += inc
        else:
            self.evidence, self.llr = 0, 0.0
        raised = False
        if self.statistic > self.h and not self.misspecified:
            self.misspecified, raised = True, True
            self.flags += 1
            self.flagged_at = self.observed
        elif self.statistic == 0.0 and self.misspecified:
            self.misspecified = False
            self.flagged_at = None
        return {"increment": inc, "statistic": self.statistic,
                "misspecified": self.misspecified, "raised": raised}

    def observe_categorical(self, pred, outcome: int) -> Dict[str, Any]:
        """pred: the predictive distribution over M outcomes (sums to 1)."""
        q = np.asarray(pred, np.float64)
        if q.ndim != 1 or len(q) < 2 or np.any(q < 0) or not np.isclose(q.sum(), 1.0, atol=1e-6):
            raise ContractError("pred must be a probability vector over >= 2 outcomes")
        o = int(outcome)
        if not 0 <= o < len(q):
            raise ContractError(f"outcome {outcome!r} outside 0..{len(q) - 1}")
        u = 1.0 / len(q)
        q0 = (1 - self.tolerance) * q[o] + self.tolerance * u
        q1 = (1 - self.contamination) * q0 + self.contamination * u
        if q0 <= 0:
            raise ContractError("outcome had predictive probability 0 (a hard zero)")
        return self.observe(math.log(q0), math.log(q1))

    def observe_gaussian(self, mu, var, y) -> Dict[str, Any]:
        """Gaussian predictive N(mu, var), independent dimensions."""
        mu = np.atleast_1d(np.asarray(mu, float))
        var = np.atleast_1d(np.asarray(var, float))
        y = np.atleast_1d(np.asarray(y, float))
        if not (mu.shape == var.shape == y.shape) or mu.ndim != 1:
            raise ContractError(f"mu/var/y shapes differ: {mu.shape} {var.shape} {y.shape}")
        if np.any(var <= 0) or not np.all(np.isfinite(np.concatenate([mu, var, y]))):
            raise ContractError("predictive variance must be > 0 and inputs finite")
        v0 = var * (1 + self.tolerance)
        r2 = (y - mu) ** 2
        l0 = -0.5 * np.log(2 * np.pi * v0) - 0.5 * r2 / v0
        lw = -0.5 * np.log(2 * np.pi * self.inflate * v0) - 0.5 * r2 / (self.inflate * v0)
        l1 = np.logaddexp(math.log1p(-self.contamination) + l0,
                          math.log(self.contamination) + lw)
        return self.observe(float(l0.sum()), float(l1.sum()))

    def report(self) -> Dict[str, Any]:
        return {"misspecified": self.misspecified, "statistic": self.statistic,
                "threshold": self.h, "evidence": self.evidence, "llr": self.llr,
                "observed": self.observed, "flags": self.flags,
                "flagged_at": self.flagged_at}


def categorical_predictive(probs, loglik_table) -> np.ndarray:
    """The mixture predictive sum_i probs_i * exp(loglik_table[i]) over M
    outcomes; every row must be a normalised outcome distribution."""
    L = np.asarray(loglik_table, np.float64)
    p = np.asarray(probs, np.float64)
    if L.ndim != 2 or L.shape[0] != len(p) or L.shape[1] < 2:
        raise ContractError(f"loglik table shape {L.shape} must be ({len(p)}, M>=2)")
    if not np.all(np.isfinite(L)):
        raise ContractError("outcome log-likelihoods must be finite (no hard zeros)")
    if not np.allclose(logsumexp(L, axis=1), 0.0, atol=1e-6):
        raise ContractError("each hypothesis' outcome distribution must be normalised "
                            "(logsumexp of its row = 0)")
    lpred = logsumexp(L + np.log(np.maximum(p, 1e-300))[:, None], axis=0)
    return np.exp(lpred - logsumexp(lpred))


class HypothesisSet:
    def __init__(self, floor: float = 1e-3, temperature: float = 1.0,
                 max_hypotheses: int = 16, min_keep: int = 2,
                 misspec: Optional[Mapping[str, float]] = None):
        self.floor = float(floor)
        self.temperature = float(temperature)
        self.max_hypotheses = int(max_hypotheses)
        self.min_keep = int(min_keep)
        if not 0 < self.temperature <= 1:
            raise ContractError("temperature must be in (0, 1]")
        if self.min_keep < 1 or self.max_hypotheses < self.min_keep:
            raise ContractError("need 1 <= min_keep <= max_hypotheses")
        if not 0 <= self.floor or self.floor * self.max_hypotheses >= 1:
            raise ContractError("floor must satisfy 0 <= floor * max_hypotheses < 1")
        self.names: List[str] = []
        self.payloads: Dict[str, Any] = {}
        self._logp = np.zeros(0)
        self.retired: List[Dict[str, Any]] = []
        self.steps = 0
        self.log_evidence: List[float] = []
        self.monitor = MisspecificationMonitor(**dict(misspec or {}))
        self.unmonitored = 0

    # ---- membership --------------------------------------------------------
    def add_hypothesis(self, name: str, prior: Optional[float] = None,
                       payload: Any = None) -> None:
        if not isinstance(name, str) or not name:
            raise ContractError("hypothesis name must be a non-empty string")
        if name in self.names:
            raise ContractError(f"hypothesis {name!r} already present")
        K = len(self.names)
        if K >= self.max_hypotheses:
            self._retire_least(reason="max_hypotheses reached on add")
            K = len(self.names)
        if K == 0:
            self.names, self._logp = [name], np.zeros(1)
        else:
            p = 1.0 / (K + 1) if prior is None else float(prior)
            if not 0 < p < 1:
                raise ContractError("prior of a new hypothesis must be in (0, 1)")
            self._logp = np.append(self._logp + math.log1p(-p), math.log(p))
            self.names.append(name)
        self.payloads[name] = payload
        self._apply_floor()
        if self.monitor.observed:
            self.monitor.reset(f"hypothesis {name!r} added: the set changed")

    def _retire_least(self, reason: str) -> None:
        if len(self.names) <= self.min_keep:
            raise ContractError(f"cannot retire below min_keep={self.min_keep}")
        order = np.argsort(self._logp)
        i = int(order[0])
        if i == int(np.argmax(self._logp)):              # never the MAP
            i = int(order[1])
        self._drop([i], reason)

    def _drop(self, idx, reason):
        for i in sorted(idx, reverse=True):
            n = self.names[i]
            self.retired.append({"name": n, "step": self.steps, "reason": reason,
                                 "prob": float(np.exp(self._logp[i]))})
            del self.names[i]
            self.payloads.pop(n, None)
            self._logp = np.delete(self._logp, i)
        self._logp -= logsumexp(self._logp)

    def prune(self, threshold: float) -> List[str]:
        """Retire hypotheses below `threshold`, keeping >= min_keep (the most
        probable) and always the MAP. Returns the retired names."""
        if len(self.names) <= self.min_keep:
            return []
        p = self.probs()
        order = np.argsort(-p)
        keep = set(order[:self.min_keep].tolist())
        drop = [i for i in range(len(p)) if p[i] < threshold and i not in keep]
        names = [self.names[i] for i in drop]
        if drop:
            self._drop(drop, f"prob < {threshold}")
            self._apply_floor()
        return names

    # ---- updating --------------------------------------------------------------
    def update_categorical(self, loglik_table, outcome) -> bool:
        """Update on a DISCRETE outcome, given every hypothesis' full outcome
        distribution: loglik_table is (K, M) log-probabilities (rows in
        `names` order, or a Mapping name -> length-M array), `outcome` the
        index observed. Same posterior as update(table[:, outcome]); in
        addition the observation is scored under the posterior predictive
        BEFORE the update and fed to the misspecification monitor."""
        if outcome is None or is_missing(outcome):
            return False
        if isinstance(loglik_table, Mapping):
            missing = [n for n in self.names if n not in loglik_table]
            if missing:
                raise ContractError(f"no outcome distribution for {missing}")
            L = np.stack([np.asarray(loglik_table[n], np.float64) for n in self.names])
        else:
            L = np.asarray(loglik_table, np.float64)
        pred = categorical_predictive(self.probs(), L)
        o = int(outcome)
        if not 0 <= o < L.shape[1]:
            raise ContractError(f"outcome {outcome!r} outside 0..{L.shape[1] - 1}")
        ok = self.update(L[:, o], _monitored=True)
        if ok:
            self.monitor.observe_categorical(pred, o)
        return ok

    @property
    def misspecified(self) -> bool:
        """True while the posterior-predictive score deficit is beyond the
        monitor's threshold (see MISSPECIFICATION above)."""
        return self.monitor.misspecified

    def misspecification(self) -> Dict[str, Any]:
        """The monitor's report: misspecified, statistic/threshold, evidence
        (observations in the current deficit run), mean deficit (sd per
        observation), observed, flags raised so far, unmonitored updates."""
        r = self.monitor.report()
        r["unmonitored"] = self.unmonitored
        return r

    def update(self, logliks, _monitored: bool = False) -> bool:
        if logliks is None or is_missing(logliks):
            return False
        if isinstance(logliks, Mapping):
            missing = [n for n in self.names if n not in logliks or is_missing(logliks[n])]
            if missing:
                if len(missing) == len(self.names):
                    return False
                raise ContractError(f"partial evidence: no likelihood for {missing}; "
                                    f"refusing to compare scored and unscored hypotheses")
            ll = np.array([float(logliks[n]) for n in self.names])
        else:
            ll = np.asarray(logliks, np.float64)
            if ll.shape != (len(self.names),):
                raise ContractError(f"loglik shape {ll.shape} != ({len(self.names)},)")
        if not np.all(np.isfinite(ll)):
            raise ContractError("log-likelihoods must be finite (a -inf is an "
                                "irrecoverable hard zero)")
        joint = self._logp + self.temperature * ll
        z = logsumexp(joint)
        self.log_evidence.append(float(z))
        self._logp = joint - z
        self._apply_floor()
        self.steps += 1
        if not _monitored:
            self.unmonitored += 1
        return True

    def _apply_floor(self):
        K = len(self.names)
        if self.floor > 0 and K > 1:
            self._logp = np.logaddexp(math.log1p(-K * self.floor) + self._logp,
                                      math.log(self.floor))
        self._logp -= logsumexp(self._logp)

    # ---- reading --------------------------------------------------------------
    def probs(self) -> np.ndarray:
        return np.exp(self._logp)

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(self.names, self.probs().tolist()))

    def log_probs(self) -> np.ndarray:
        return self._logp.copy()

    def map(self) -> str:
        return self.names[int(np.argmax(self._logp))]

    def entropy(self) -> float:
        p = self.probs()
        return float(-(p * self._logp).sum())

    def __len__(self):
        return len(self.names)
