"""Skills learned from experience, identified by behaviour (plan Stage 11
items 3-4).

WHAT IS CLAIMED
    1. INITIATION and TERMINATION are learned, not declared: SetClassifier
       (L2 logistic regression, standardised, optional quadratic features)
       is fitted on states where the skill was invoked (label: it
       succeeded) and on states along successful executions (label: this
       is where it ended). Below `min_samples` a classifier says UNKNOWN —
       never 0.5 dressed up as knowledge.
    2. COMPETENCE is a Beta posterior over the success probability, fed by
       REAL outcomes only. An imagined or evaluator outcome RAISES
       ProvenanceError (plan §7.4: a model cannot certify its own imagined
       outcomes). Evidence ids are kept so the contracts.Skill record names
       what backs its competence. `calibration` / `interval_coverage`
       measure whether the posterior means what it says.
    3. IDENTITY IS BEHAVIOUR. A behavioural signature is the controller's
       action distribution (discrete) or command (continuous) on a fixed,
       hashed probe set. Two skills whose mean Jensen-Shannon divergence
       (bits, in [0, 1]) is below a threshold are INDISTINGUISHABLE ON THOSE
       PROBES — whatever their names, and even if their weights differ.
       Names never enter any comparison here (CLAUDE.md §4.6: minted skills
       are byte copies of the shared policy, cosine 0.869-1.000, three pairs
       bit-identical).
       A DUPLICATE verdict needs more (verifier finding F8: skills that
       differ only off a caller-chosen probe set were flagged duplicates in
       92% of trials while disagreeing on 29% of deployment states). The
       probe set must be built by `build_probe_set` from EACH skill's own
       evidenced states (LearnedSkill.evidence_states(): where it was
       invoked and where it went, from real attempts) plus any shared
       probes, and must COVER both skills' evidence: >= `min_evidence`
       states each, and >= `min_coverage` of them within `radius`
       (standardised RMS) of a probe. Otherwise find_duplicates reports
       "indistinguishable on N probes covering X" and never "duplicate".
       Coverage is relative to evidence: two skills that have only ever
       been used where they agree ARE duplicates as far as anything has
       shown; a new state where they differ is new evidence.
       THE VERDICT IS GRADED (closed 2026-10-03, verifier E11e: a copy with
       logits x3 — same argmax on every state, so the same deterministic
       behaviour — was reported distinct at JS 0.10 bits):
         duplicate       JS < threshold (default 1e-3) and covered;
         near_duplicate  JS >= threshold, but the EXECUTED (modal)
                         behaviour agrees within sampling noise: on every
                         probe where BOTH skills' modal actions are
                         resolvable by `noise_samples` executions (top-two
                         gap > z * sqrt((p1 + p2) / n), the 95% two-sided
                         multinomial bound, z 1.96, n 64 by default) they
                         pick the same action, and such probes are at least
                         `min_resolved` of the set; and covered. The pair
                         differs in confidence/temperature, not in what it
                         does. Discrete signatures only (a continuous
                         signature has no sampling model);
         distinct        anything else — a measured difference is real
                         whatever the coverage;
         indistinguishable_on_probes
                         duplicate or near_duplicate on a probe set that
                         does not cover both skills' evidence (or bare
                         probes): nothing more can be claimed.
    4. OPTIONS ARE SMDP: `smdp_return` discounts rewards inside an option by
       gamma^k and the bootstrap by gamma^tau, tau = the option's real
       duration. Treating a tau-step option as one step overvalues long
       options (tests/_foundation_planning_smoke.py has the witness).

ESCAPE PATHS (CLAUDE.md §4.1)
    * A learned initiation set that says "never" would starve itself of the
      data that could correct it. `can_initiate` therefore admits a state
      outside the learned set with probability `explore_floor` (> 0), and an
      UNKNOWN classifier admits everything.
    * A termination classifier that never fires would make the option a
      latch; `max_duration` is a hard timeout that always ends it.

Offline/shadow only (plan §8).
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import UNKNOWN, ContractError, Evidence, Skill, is_missing
from ..experience.errors import ProvenanceError
from .planner import REAL_PROVENANCES, _check_real

STATE_SCHEMA = 1


# ======================================================================
# Learned state sets
# ======================================================================

class SetClassifier:
    """P(label = 1 | state) by L2-regularised logistic regression (IRLS).

    features "linear" -> [1, x]; "quadratic" -> [1, x, x_i x_j (i <= j)],
    which represents balls and slabs (a goal radius, a reach limit).
    Data are kept in a bounded FIFO (`max_samples`) so the set can be
    refitted when the world changes."""

    def __init__(self, dim: int, features: str = "linear", l2: float = 1e-4,
                 min_samples: int = 10, max_samples: int = 4000, iters: int = 30):
        if int(dim) < 1:
            raise ContractError("dim must be >= 1")
        if features not in ("linear", "quadratic"):
            raise ContractError("features must be 'linear' or 'quadratic'")
        if l2 <= 0 or min_samples < 2 or max_samples < min_samples:
            raise ContractError("need l2 > 0 and 2 <= min_samples <= max_samples")
        self.dim, self.features, self.l2 = int(dim), features, float(l2)
        self.min_samples, self.max_samples, self.iters = int(min_samples), int(max_samples), int(iters)
        self.X: Deque[np.ndarray] = deque(maxlen=self.max_samples)
        self.y: Deque[int] = deque(maxlen=self.max_samples)
        self.w: Optional[np.ndarray] = None
        self.mu = self.sd = None
        self.const: Optional[float] = None       # one-class data: smoothed rate

    def _check(self, X) -> np.ndarray:
        X = np.asarray(X, np.float64)
        if X.ndim == 1:
            X = X[None]
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ContractError(f"states must be (n, {self.dim}), got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ContractError("states must be finite")
        return X

    def _phi(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        parts = [np.ones((len(Z), 1)), Z]
        if self.features == "quadratic":
            iu = np.triu_indices(self.dim)
            parts.append((Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]])
        return np.concatenate(parts, 1)

    def add(self, X, y) -> None:
        X = self._check(X)
        y = np.atleast_1d(np.asarray(y))
        if y.shape != (len(X),) or not np.isin(y, (0, 1)).all():
            raise ContractError("labels must be 0/1, one per state")
        for x, t in zip(X, y):
            self.X.append(x.copy())
            self.y.append(int(t))

    @property
    def n(self) -> int:
        return len(self.y)

    def fit(self) -> Dict[str, Any]:
        if self.n < self.min_samples:
            self.w, self.const = None, None
            return {"fitted": False, "n": self.n}
        X = np.stack(self.X)
        y = np.asarray(self.y, np.float64)
        if y.min() == y.max():                       # one class: no boundary
            self.w, self.const = None, float((y.sum() + 1) / (len(y) + 2))
            return {"fitted": True, "one_class": True, "n": self.n}
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 1e-8)
        P = self._phi(X)
        w = np.zeros(P.shape[1])
        reg = self.l2 * len(y) * np.eye(P.shape[1])
        reg[0, 0] = 0.0                              # do not shrink the bias
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(P @ w, -30, 30)))
            g = P.T @ (p - y) + reg @ w
            Hm = (P * (p * (1 - p))[:, None]).T @ P + reg + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(Hm, g)
            w = w - step
            if np.abs(step).max() < 1e-8:
                break
        self.w, self.const = w, None
        return {"fitted": True, "one_class": False, "n": self.n}

    def prob(self, X):
        X = self._check(X)
        if self.const is not None:
            return np.full(len(X), self.const)
        if self.w is None:
            return UNKNOWN
        return 1.0 / (1.0 + np.exp(-np.clip(self._phi(X) @ self.w, -30, 30)))

    def state_dict(self) -> Dict[str, Any]:
        return {"schema": STATE_SCHEMA, "dim": self.dim, "features": self.features,
                "l2": self.l2, "min_samples": self.min_samples,
                "max_samples": self.max_samples, "iters": self.iters,
                "X": np.array(list(self.X)).reshape(-1, self.dim),
                "y": np.asarray(list(self.y), np.int64),
                "w": self.w, "mu": self.mu, "sd": self.sd, "const": self.const}

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "SetClassifier":
        if d.get("schema") != STATE_SCHEMA:
            raise ContractError(f"classifier state schema {d.get('schema')!r}")
        c = cls(d["dim"], d["features"], d["l2"], d["min_samples"], d["max_samples"],
                d["iters"])
        for x, t in zip(np.asarray(d["X"]).reshape(-1, c.dim), np.asarray(d["y"])):
            c.X.append(np.array(x, np.float64))
            c.y.append(int(t))
        c.w, c.mu, c.sd, c.const = d["w"], d["mu"], d["sd"], d["const"]
        return c


# ======================================================================
# Competence
# ======================================================================

class BetaCompetence:
    """Beta(a0 + successes, b0 + failures) over a skill's success
    probability, from REAL outcomes only."""

    def __init__(self, a0: float = 1.0, b0: float = 1.0):
        if not (a0 > 0 and b0 > 0):
            raise ContractError("Beta prior parameters must be > 0")
        self.a0, self.b0 = float(a0), float(b0)
        self.successes = 0
        self.failures = 0
        self.rejected = 0
        self.evidence_ids: List[str] = []

    def record(self, success: bool, *, provenance: str,
               evidence_id: Optional[str] = None) -> None:
        if provenance not in REAL_PROVENANCES:
            self.rejected += 1
        _check_real(provenance, "BetaCompetence.record")
        if not isinstance(success, (bool, np.bool_)):
            raise ContractError(f"success must be a bool, got {success!r}")
        if success:
            self.successes += 1
        else:
            self.failures += 1
        if evidence_id is not None:
            self.evidence_ids.append(str(evidence_id))

    def record_evidence(self, evidence: Evidence, success: bool) -> None:
        """An outcome backed by a contracts.Evidence record: it must be
        valid and real (sensor/inferred) — imagined evidence cannot even be
        constructed with sensor observations, and is refused here too."""
        if not isinstance(evidence, Evidence):
            raise ContractError("record_evidence takes a contracts.Evidence")
        if evidence.validity != "valid":
            raise ContractError(f"evidence {evidence.evidence_id!r} is "
                                f"{evidence.validity}: {evidence.validity_reason}")
        self.record(success, provenance=evidence.provenance,
                    evidence_id=evidence.evidence_id)

    @property
    def a(self) -> float:
        return self.a0 + self.successes

    @property
    def b(self) -> float:
        return self.b0 + self.failures

    @property
    def n(self) -> int:
        return self.successes + self.failures

    def mean(self) -> float:
        """Posterior predictive P(next real attempt succeeds)."""
        return self.a / (self.a + self.b)

    def interval(self, level: float = 0.9) -> Tuple[float, float]:
        from scipy.stats import beta
        if not 0 < level < 1:
            raise ContractError("level must be in (0, 1)")
        lo = (1 - level) / 2
        return float(beta.ppf(lo, self.a, self.b)), float(beta.ppf(1 - lo, self.a, self.b))

    def state_dict(self) -> Dict[str, Any]:
        return {"schema": STATE_SCHEMA, "a0": self.a0, "b0": self.b0,
                "successes": self.successes, "failures": self.failures,
                "rejected": self.rejected, "evidence_ids": list(self.evidence_ids)}

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "BetaCompetence":
        if d.get("schema") != STATE_SCHEMA:
            raise ContractError(f"competence state schema {d.get('schema')!r}")
        c = cls(d["a0"], d["b0"])
        c.successes, c.failures = int(d["successes"]), int(d["failures"])
        c.rejected, c.evidence_ids = int(d["rejected"]), list(d["evidence_ids"])
        return c


def calibration(probs, outcomes, n_bins: int = 10) -> Dict[str, Any]:
    """Expected calibration error, Brier score and the reliability bins of
    predicted probabilities against binary real outcomes."""
    p = np.asarray(probs, np.float64).reshape(-1)
    y = np.asarray(outcomes, np.float64).reshape(-1)
    if p.shape != y.shape or len(p) == 0:
        raise ContractError("probs and outcomes must be equal-length and non-empty")
    if np.any((p < 0) | (p > 1)) or not np.isin(y, (0, 1)).all():
        raise ContractError("probs in [0, 1], outcomes in {0, 1}")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        bins.append({"bin": b, "n": int(m.sum()), "confidence": conf, "accuracy": acc})
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)), "bins": bins}


def interval_coverage(competences: Sequence[BetaCompetence], true_p, level: float = 0.9
                      ) -> float:
    """Fraction of true success probabilities inside each posterior's
    central `level` interval (fixture use: the truth is known there)."""
    tp = np.asarray(true_p, np.float64)
    if len(tp) != len(competences):
        raise ContractError("one true probability per competence")
    hits = [lo <= t <= hi for c, t in zip(competences, tp)
            for lo, hi in [c.interval(level)]]
    return float(np.mean(hits))


# ======================================================================
# Behavioural identity
# ======================================================================

def probe_hash(probes) -> str:
    a = np.ascontiguousarray(np.asarray(probes, np.float64))
    return hashlib.sha1(repr(a.shape).encode() + a.tobytes()).hexdigest()


@dataclass(frozen=True)
class BehaviourSignature:
    kind: str                # "discrete" (rows are distributions) | "continuous"
    probe_hash: str
    values: np.ndarray       # (P, K) or (P, Ad)
    scale: float = 1.0       # continuous: command scale for the distance

    def to_state(self) -> Dict[str, Any]:
        return {"kind": self.kind, "probe_hash": self.probe_hash,
                "values": np.array(self.values), "scale": float(self.scale)}

    @classmethod
    def from_state(cls, d) -> "BehaviourSignature":
        return cls(d["kind"], d["probe_hash"], np.asarray(d["values"], np.float64),
                   float(d["scale"]))

    def digest(self) -> str:
        return hashlib.sha1(self.kind.encode() + self.probe_hash.encode() +
                            np.round(self.values, 6).tobytes()).hexdigest()


def behaviour_signature(probes, *, probs_fn: Optional[Callable] = None,
                        command_fn: Optional[Callable] = None,
                        scale: float = 1.0) -> BehaviourSignature:
    """Exactly one of probs_fn(state) -> (K,) distribution (discrete) or
    command_fn(state) -> (Ad,) command (continuous, deterministic).
    `probes` is a (P, d) array or a ProbeSet (build_probe_set)."""
    if (probs_fn is None) == (command_fn is None):
        raise ContractError("pass exactly one of probs_fn / command_fn")
    if isinstance(probes, ProbeSet):
        probes = probes.states
    P = np.asarray(probes, np.float64)
    if P.ndim != 2 or len(P) < 1:
        raise ContractError("probes must be (P, state_dim)")
    if probs_fn is not None:
        rows = np.stack([np.asarray(probs_fn(s), np.float64).reshape(-1) for s in P])
        if np.any(rows < -1e-9) or not np.allclose(rows.sum(1), 1.0, atol=1e-5):
            raise ContractError("probs_fn must return probability vectors")
        return BehaviourSignature("discrete", probe_hash(P), np.clip(rows, 0, 1))
    rows = np.stack([np.asarray(command_fn(s), np.float64).reshape(-1) for s in P])
    if not np.all(np.isfinite(rows)) or not scale > 0:
        raise ContractError("commands must be finite and scale > 0")
    return BehaviourSignature("continuous", probe_hash(P), rows, float(scale))


def behavioural_divergence(a: BehaviourSignature, b: BehaviourSignature) -> float:
    """Mean JS divergence in bits ([0, 1]) for discrete signatures; RMS
    command difference / scale for continuous. Refuses signatures taken on
    different probe sets or of different kinds — there is no meaningful
    number to return then."""
    if a.kind != b.kind or a.probe_hash != b.probe_hash or a.values.shape != b.values.shape:
        raise ContractError("signatures are not comparable (kind, probe set or shape "
                            "differ); recompute on a shared probe set")
    if a.kind == "continuous":
        return float(np.sqrt(np.mean((a.values - b.values) ** 2)) / max(a.scale, b.scale))
    p, q = a.values, b.values
    m = 0.5 * (p + q)

    def kl(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(x > 0, x * np.log2(x / np.maximum(y, 1e-300)), 0.0)
        return t.sum(1)

    return float(np.mean(0.5 * kl(p, m) + 0.5 * kl(q, m)))


@dataclass(frozen=True)
class ProbeSet:
    """Probe states plus WHOSE evidence they cover (build_probe_set)."""
    states: np.ndarray                        # (P, d)
    sources: Tuple[str, ...]                  # per row: a skill id or "shared"
    coverage: Dict[str, float]                # per evidenced skill, in [0, 1]
    n_evidence: Dict[str, int]
    radius: float

    @property
    def hash(self) -> str:
        return probe_hash(self.states)

    def __len__(self) -> int:
        return len(self.states)


def _farthest_points(X: np.ndarray, k: int) -> np.ndarray:
    """Deterministic farthest-point subsample (indices), starting at the
    point nearest the centroid: spreads the probes over the evidence."""
    i0 = int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))
    idx, d = [i0], ((X - X[i0]) ** 2).sum(1)
    while len(idx) < k:
        j = int(np.argmax(d))
        if d[j] <= 0:
            break
        idx.append(j)
        d = np.minimum(d, ((X - X[j]) ** 2).sum(1))
    return np.asarray(idx)


def build_probe_set(evidence: Dict[str, Any], shared=None, *, max_per_skill: int = 64,
                    radius: float = 0.5) -> ProbeSet:
    """evidence: skill_id -> (n, d) states that skill was REALLY in
    (LearnedSkill.evidence_states()). Each skill contributes up to
    `max_per_skill` farthest-point-sampled states; `shared` (P0, d) probes
    are appended. coverage[id] = fraction of that skill's evidence within
    `radius` (RMS over dims, in units of the pooled per-dim std) of some
    probe."""
    if not isinstance(evidence, dict):
        raise ContractError("evidence must be {skill_id: (n, d) states}")
    if int(max_per_skill) < 1 or not radius > 0:
        raise ContractError("max_per_skill >= 1 and radius > 0 required")
    ev: Dict[str, np.ndarray] = {}
    dim = None
    for sid in sorted(evidence):
        X = np.asarray(evidence[sid], np.float64)
        X = X.reshape(0, dim or 0) if X.size == 0 else (X[None] if X.ndim == 1 else X)
        if X.ndim != 2 or (dim is not None and X.shape[1] != dim) or \
                not np.all(np.isfinite(X)):
            raise ContractError(f"evidence for {sid!r} must be finite (n, d), got {X.shape}")
        dim = X.shape[1] if X.size else dim
        ev[str(sid)] = X
    S0 = None
    if shared is not None:
        S0 = np.asarray(shared, np.float64)
        if S0.ndim != 2 or (dim is not None and S0.shape[1] != dim) or \
                not np.all(np.isfinite(S0)):
            raise ContractError("shared probes must be finite (P, d)")
        dim = S0.shape[1]
    if dim is None:
        raise ContractError("no evidence and no shared probes: nothing to probe")
    rows, src = [], []
    for sid, X in ev.items():
        if len(X):
            pick = X[_farthest_points(X, min(len(X), int(max_per_skill)))]
            rows.append(pick)
            src += [sid] * len(pick)
    if S0 is not None and len(S0):
        rows.append(S0)
        src += ["shared"] * len(S0)
    if not rows:
        raise ContractError("probe set would be empty")
    P = np.concatenate(rows, 0)
    pool = np.concatenate([X for X in ev.values() if len(X)] + [P], 0)
    sd = pool.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Ps = P / sd
    cov: Dict[str, float] = {}
    for sid, X in ev.items():
        if not len(X):
            cov[sid] = 0.0
            continue
        Xs = X / sd
        d2 = np.maximum((Xs ** 2).sum(1)[:, None] + (Ps ** 2).sum(1)[None]
                        - 2.0 * Xs @ Ps.T, 0.0).min(1) / dim
        cov[sid] = float(np.mean(np.sqrt(d2) <= radius))
    return ProbeSet(P, tuple(src), cov, {k: len(v) for k, v in ev.items()}, float(radius))


DUPLICATE, NEAR_DUPLICATE, DISTINCT = "duplicate", "near_duplicate", "distinct"
INDISTINGUISHABLE = "indistinguishable_on_probes"
VERDICTS = (DUPLICATE, NEAR_DUPLICATE, DISTINCT, INDISTINGUISHABLE)
MODE_Z = 1.959964                              # 95% two-sided


def modal_agreement(a: BehaviourSignature, b: BehaviourSignature, n_samples: int = 64,
                    z: float = MODE_Z) -> Dict[str, Any]:
    """Executed-behaviour comparison of two DISCRETE signatures, calibrated
    against sampling noise. A probe's modal action is RESOLVED for a skill
    when n_samples executions would identify it: top-two gap > z *
    sqrt((p1 + p2) / n) (normal bound on the difference of two multinomial
    counts). Returns {"resolved": probes resolved for both, "disagree":
    those where the modes differ, "exec_divergence": disagree / resolved
    (UNKNOWN when nothing is resolved)}."""
    if a.kind != "discrete" or b.kind != "discrete":
        raise ContractError("modal agreement needs discrete signatures")
    if a.probe_hash != b.probe_hash or a.values.shape != b.values.shape:
        raise ContractError("signatures are not comparable (probe set or shape differ)")
    if int(n_samples) < 1 or not z > 0:
        raise ContractError("n_samples >= 1 and z > 0 required")

    def resolved(p):
        top = np.sort(p, 1)[:, ::-1]
        if p.shape[1] < 2:
            return np.ones(len(p), bool)
        return (top[:, 0] - top[:, 1]) > z * np.sqrt((top[:, 0] + top[:, 1]) / int(n_samples))
    both = resolved(a.values) & resolved(b.values)
    diff = both & (a.values.argmax(1) != b.values.argmax(1))
    n_r, n_d = int(both.sum()), int(diff.sum())
    return {"resolved": n_r, "disagree": n_d, "n_probes": int(len(both)),
            "exec_divergence": (n_d / n_r) if n_r else UNKNOWN}


@dataclass(frozen=True)
class DuplicateFinding:
    """A pair of skills and a GRADED verdict (VERDICTS; see WHAT IS CLAIMED
    3). "duplicate" / "near_duplicate" only when the probe set covers both
    skills' evidence; otherwise "indistinguishable_on_probes" — read
    `statement`, not just the pair. Deliberately NOT a tuple: code that
    unpacked (a, b, d) and treated every pair as a duplicate must now read
    the verdict."""
    a: str
    b: str
    divergence: float
    verdict: str                              # one of VERDICTS
    n_probes: int
    coverage: Dict[str, Any]                  # id -> fraction, or UNKNOWN
    statement: str
    modal: Any = UNKNOWN                      # modal_agreement(...) dict, discrete only


def find_duplicates(signatures: Dict[str, BehaviourSignature], threshold: float = 1e-3, *,
                    probe_set: Optional[ProbeSet] = None, min_coverage: float = 0.9,
                    min_evidence: int = 8, noise_samples: int = 64,
                    min_resolved: float = 0.25,
                    include_distinct: bool = False) -> List[DuplicateFinding]:
    """Every pair that is NOT distinct (duplicate, near_duplicate or
    indistinguishable_on_probes), with a graded verdict; every pair when
    `include_distinct`. Keys are skill ids; nothing else about a skill is
    consulted. Without a `probe_set` (signatures taken on bare probe
    arrays) no pair can be a duplicate or near-duplicate: nothing says the
    probes cover where either skill acts. near_duplicate needs modal
    agreement on every probe both skills resolve with `noise_samples`
    executions, on at least max(8, min_resolved x probes) such probes."""
    if not 0 < min_coverage <= 1 or int(min_evidence) < 1:
        raise ContractError("need 0 < min_coverage <= 1 and min_evidence >= 1")
    if not 0 <= min_resolved <= 1 or int(noise_samples) < 1:
        raise ContractError("need 0 <= min_resolved <= 1 and noise_samples >= 1")
    if probe_set is not None:
        bad = [k for k, v in signatures.items() if v.probe_hash != probe_set.hash]
        if bad:
            raise ContractError(f"signatures {bad} were not taken on this probe set")
    ids = sorted(signatures)
    out = []
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            d = behavioural_divergence(signatures[x], signatures[y])
            P = len(signatures[x].values)
            modal: Any = UNKNOWN
            if d < threshold:
                base = DUPLICATE
            elif signatures[x].kind == "discrete":
                modal = modal_agreement(signatures[x], signatures[y], noise_samples)
                need = max(8, int(np.ceil(min_resolved * P)))
                base = NEAR_DUPLICATE if (modal["disagree"] == 0 and
                                          modal["resolved"] >= need) else DISTINCT
            else:
                base = DISTINCT
            mtxt = "" if is_missing(modal) else (
                f"; modal action agrees on {modal['resolved'] - modal['disagree']}/"
                f"{modal['resolved']} probes resolvable by {int(noise_samples)} executions")
            if base == DISTINCT:
                if include_distinct:
                    out.append(DuplicateFinding(
                        x, y, d, DISTINCT, P, {x: UNKNOWN, y: UNKNOWN} if probe_set is None
                        else {k: probe_set.coverage.get(k, UNKNOWN) for k in (x, y)},
                        f"distinct: JS {d:.3g} bits >= {threshold:g} on {P} probes" + mtxt,
                        modal))
                continue
            what = ("indistinguishable" if base == DUPLICATE else
                    f"same executed behaviour, JS {d:.3g} bits")
            if probe_set is None:
                cov = {x: UNKNOWN, y: UNKNOWN}
                out.append(DuplicateFinding(
                    x, y, d, INDISTINGUISHABLE, P, cov,
                    f"indistinguishable on {P} probes covering an UNKNOWN fraction of "
                    f"either skill's evidenced states (no evidence-built probe set: "
                    f"not a {base} verdict)" + mtxt, modal))
                continue
            cov = {k: probe_set.coverage.get(k, UNKNOWN) for k in (x, y)}
            nev = {k: probe_set.n_evidence.get(k, 0) for k in (x, y)}
            ok = all(not is_missing(cov[k]) and cov[k] >= min_coverage and
                     nev[k] >= int(min_evidence) for k in (x, y))
            desc = " and ".join(
                f"{'UNKNOWN' if is_missing(cov[k]) else f'{cov[k]:.0%}'} of {k}'s "
                f"{nev[k]} evidenced states" for k in (x, y))
            if ok:
                out.append(DuplicateFinding(
                    x, y, d, base, P, cov,
                    f"{base}: {what} on {P} probes covering {desc}" + mtxt, modal))
            else:
                out.append(DuplicateFinding(
                    x, y, d, INDISTINGUISHABLE, P, cov,
                    f"indistinguishable on {P} probes covering {desc} (needs >= "
                    f"{min_coverage:.0%} of >= {int(min_evidence)} states each for a "
                    f"{base} verdict)" + mtxt, modal))
    return out


# ======================================================================
# Options: SMDP credit
# ======================================================================

def smdp_return(rewards, gamma: float) -> Tuple[float, float]:
    """(sum_k gamma^k r_k, gamma^tau) for an option that ran tau = len(rewards)
    primitive steps."""
    r = np.asarray(rewards, np.float64).reshape(-1)
    if len(r) < 1:
        raise ContractError("an option lasts at least one step")
    if not (0 < gamma <= 1) or not np.all(np.isfinite(r)):
        raise ContractError("0 < gamma <= 1 and finite rewards required")
    return float((gamma ** np.arange(len(r)) * r).sum()), float(gamma ** len(r))


def smdp_target(rewards, gamma: float, next_value: float, terminal: bool) -> float:
    g, disc = smdp_return(rewards, gamma)
    return g if terminal else g + disc * float(next_value)


# ======================================================================
# A learned skill
# ======================================================================

class LearnedSkill:
    """One skill: a controller REFERENCE plus everything learned about it.

    The controller itself (weights) is not owned here — `controller_ref`
    names it, and `signature` pins what it does. version bumps when the
    controller is replaced; a new version starts a fresh competence record
    (an old controller's successes are not the new one's)."""

    def __init__(self, skill_id: str, controller_ref: str, *, state_dim: int,
                 representation: str, representation_version: str, version: int = 1,
                 features: str = "quadratic", init_threshold: float = 0.5,
                 explore_floor: float = 0.05, term_threshold: float = 0.5,
                 max_duration: int = 50, min_samples: int = 10,
                 payload: Optional[Dict[str, Any]] = None):
        if not isinstance(skill_id, str) or not skill_id:
            raise ContractError("skill_id must be a non-empty str")
        if not 0 < explore_floor <= 1:
            raise ContractError("explore_floor must be in (0, 1]: a zero floor makes "
                                "a learned 'never' a latch (CLAUDE.md §4.1)")
        if int(max_duration) < 1:
            raise ContractError("max_duration must be >= 1 (the termination escape)")
        self.skill_id, self.controller_ref = skill_id, str(controller_ref)
        self.version = int(version)
        self.state_dim = int(state_dim)
        self.representation, self.representation_version = representation, representation_version
        self.init_threshold, self.explore_floor = float(init_threshold), float(explore_floor)
        self.term_threshold, self.max_duration = float(term_threshold), int(max_duration)
        self.initiation = SetClassifier(state_dim, features, min_samples=min_samples)
        self.termination = SetClassifier(state_dim, features, min_samples=min_samples)
        self.competence = BetaCompetence()
        self.durations: List[int] = []
        self.signature: Optional[BehaviourSignature] = None
        self.payload = dict(payload or {})

    # ---- experience -----------------------------------------------------------
    def record_attempt(self, start_state, visited_states, success: bool, *,
                       provenance: str, evidence_id: Optional[str] = None) -> None:
        """One REAL execution: invoked in start_state, passed through
        visited_states (T, d) — the last row is where it ended — and
        succeeded or not."""
        _check_real(provenance, f"skill {self.skill_id} attempt")
        V = np.asarray(visited_states, np.float64).reshape(-1, self.state_dim)
        if len(V) < 1:
            raise ContractError("an attempt visits at least its end state")
        self.competence.record(bool(success), provenance=provenance, evidence_id=evidence_id)
        self.initiation.add(np.asarray(start_state, np.float64), [int(bool(success))])
        if success:
            self.termination.add(V[-1:], [1])
            if len(V) > 1:
                self.termination.add(V[:-1], [0] * (len(V) - 1))
        self.durations.append(len(V))

    def refit(self) -> Dict[str, Any]:
        return {"initiation": self.initiation.fit(), "termination": self.termination.fit()}

    def evidence_states(self) -> np.ndarray:
        """(n, state_dim) states this skill was REALLY in: every recorded
        start state (initiation data) and every state along successful
        executions (termination data). Feeds build_probe_set."""
        rows = list(self.initiation.X) + list(self.termination.X)
        return (np.stack(rows) if rows else np.zeros((0, self.state_dim)))

    # ---- use ------------------------------------------------------------------
    def can_initiate(self, state, rng: np.random.Generator) -> Tuple[bool, str]:
        p = self.initiation.prob(state)
        if is_missing(p):
            return True, "initiation unknown: exploring"
        p = float(np.asarray(p)[0])
        if p >= self.init_threshold:
            return True, f"in learned initiation set (p={p:.2f})"
        if rng.random() < self.explore_floor:
            return True, f"explore floor (p={p:.2f} < {self.init_threshold})"
        return False, f"outside learned initiation set (p={p:.2f})"

    def should_terminate(self, state, t: int) -> Tuple[bool, str]:
        if t >= self.max_duration:
            return True, f"timeout at {t} >= max_duration {self.max_duration}"
        p = self.termination.prob(state)
        if is_missing(p):
            return False, "termination unknown: running to timeout"
        p = float(np.asarray(p)[0])
        return (p >= self.term_threshold), f"termination p={p:.2f}"

    def expected_discount(self, gamma: float):
        """E[gamma^tau] over observed real durations, or UNKNOWN."""
        if not self.durations:
            return UNKNOWN
        return float(np.mean(gamma ** np.asarray(self.durations, np.float64)))

    def set_signature(self, sig: BehaviourSignature) -> None:
        self.signature = sig

    def verify_signature(self, sig: BehaviourSignature, tol: float = 1e-3) -> Tuple[bool, float]:
        """Does the controller now behind controller_ref still behave as the
        one this skill's competence was earned with?"""
        if self.signature is None:
            return False, float("inf")
        d = behavioural_divergence(self.signature, sig)
        return d < tol, d

    def new_version(self, controller_ref: str, reason: str) -> None:
        if not reason:
            raise ContractError("a skill version bump must state its reason")
        self.version += 1
        self.controller_ref = str(controller_ref)
        self.competence = BetaCompetence(self.competence.a0, self.competence.b0)
        self.durations = []
        self.signature = None
        self.payload.setdefault("version_history", []).append(
            {"version": self.version, "reason": reason})

    # ---- records / persistence ---------------------------------------------------
    def to_record(self) -> Skill:
        c = self.competence
        return Skill(
            skill_id=self.skill_id, version=self.version,
            controller_ref=self.controller_ref,
            initiation={"kind": "learned_logistic", "features": self.initiation.features,
                        "threshold": self.init_threshold,
                        "explore_floor": self.explore_floor, "n": self.initiation.n},
            termination={"kind": "learned_logistic", "features": self.termination.features,
                         "threshold": self.term_threshold,
                         "max_duration": self.max_duration, "n": self.termination.n},
            competence_evidence=tuple(c.evidence_ids),
            payload={"competence": {"a": c.a, "b": c.b, "n": c.n,
                                    "source": "real_outcomes_only"},
                     "representation": self.representation,
                     "representation_version": self.representation_version,
                     "signature_digest": (UNKNOWN if self.signature is None
                                          else self.signature.digest()),
                     **{k: v for k, v in self.payload.items() if k != "version_history"}})

    def state_dict(self) -> Dict[str, Any]:
        return {"schema": STATE_SCHEMA, "skill_id": self.skill_id, "version": self.version,
                "controller_ref": self.controller_ref, "state_dim": self.state_dim,
                "representation": self.representation,
                "representation_version": self.representation_version,
                "init_threshold": self.init_threshold, "explore_floor": self.explore_floor,
                "term_threshold": self.term_threshold, "max_duration": self.max_duration,
                "initiation": self.initiation.state_dict(),
                "termination": self.termination.state_dict(),
                "competence": self.competence.state_dict(),
                "durations": list(self.durations),
                "signature": None if self.signature is None else self.signature.to_state(),
                "payload": dict(self.payload)}

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "LearnedSkill":
        if d.get("schema") != STATE_SCHEMA:
            raise ContractError(f"skill state schema {d.get('schema')!r}")
        s = cls(d["skill_id"], d["controller_ref"], state_dim=d["state_dim"],
                representation=d["representation"],
                representation_version=d["representation_version"], version=d["version"],
                init_threshold=d["init_threshold"], explore_floor=d["explore_floor"],
                term_threshold=d["term_threshold"], max_duration=d["max_duration"],
                payload=d["payload"])
        s.initiation = SetClassifier.from_state_dict(d["initiation"])
        s.termination = SetClassifier.from_state_dict(d["termination"])
        s.competence = BetaCompetence.from_state_dict(d["competence"])
        s.durations = [int(x) for x in d["durations"]]
        s.signature = None if d["signature"] is None else \
            BehaviourSignature.from_state(d["signature"])
        return s
