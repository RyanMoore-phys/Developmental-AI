"""The mechanism prediction API (plan Stage 6 item 1).

A MECHANISM maps (state, action, elapsed time dt) to a DISTRIBUTION over the
outcome, never to a point. Everything downstream — scoring, ensembles,
hypothesis comparison, change detection, experiment selection — consumes the
distribution, so this module fixes its shape once:

    OutcomeLayout     names of the continuous dimensions and of the discrete
                      groups (each group is one categorical over `classes`)
    MixedOutcome      THE concrete OutcomeDistribution: a mixture of M >= 1
                      components, each a diagonal Gaussian over the continuous
                      part times independent categoricals over the discrete
                      part. M == 1 is the plain Gaussian+categorical a single
                      model emits; M > 1 is what an ensemble or a sampled
                      rollout emits. One class, so no consumer special-cases.
    EntityBatch       the structured state: per-entity position, velocity and
                      static attributes in one declared frame
    TransitionBatch   (state, action, dt) -> (next state, discrete events)
    BaseMechanism     shared machinery: version counter, support-box
                      applicability, multi-step rollout
    prediction_record contracts.Prediction from a rollout, naming the model
                      versions and the snapshot that made it

THE OUTCOME OF AN ENTITY STEP. For an EntityBatch with N entities in D
dimensions the continuous outcome is the NEXT state, entity-major
[pos_0, vel_0, pos_1, vel_1, ...] (N*2D dims), and the discrete outcome is
one event class per entity (N groups). Class index MISSING (-1) in an
observed discrete outcome means "not observed": it contributes nothing to a
log-probability — it is not class 0.

UNKNOWN IS NOT A NUMBER. `applicability` of a never-fitted mechanism is the
UNKNOWN sentinel, not 0.5 or 1.0; `entropy()` of a continuous mixture (no
closed form) is UNKNOWN, not an approximation silently passed off as exact.

VARIANCE IS ALL-IN. A model's predictive variance covers everything it could
not predict: process noise, observation noise and its own error. It does
NOT separate them. The only decomposition offered here is the mixture one
(`decompose_variance`): mean of component variances vs variance of component
means, which is aleatoric vs epistemic ONLY when the components are
ensemble members (see foundation.inference.ensemble).

numpy + scipy only; the torch models live in relational.py / baselines.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import logsumexp, ndtr

try:                                           # py3.8+: typing.Protocol
    from typing import Protocol, runtime_checkable
except ImportError:                            # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable

from ..contracts import UNKNOWN, ContractError, Prediction

MISSING = -1
OUTCOME_KIND = "mixed_gaussian_categorical"
OUTCOME_SCHEMA = 1
ROLLOUT_KIND = "mechanism_rollout"
_LOG2PI = float(np.log(2.0 * np.pi))


def _arr(x, name, ndim=None, dtype=np.float64) -> np.ndarray:
    a = np.asarray(x, dtype=dtype)
    if ndim is not None and a.ndim != ndim:
        raise ContractError(f"{name} must be {ndim}-D, got shape {a.shape}")
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        raise ContractError(f"{name} has non-finite entries")
    return a


# ======================================================================
# Outcome layout and distribution
# ======================================================================

@dataclass(frozen=True)
class OutcomeLayout:
    cont_names: Tuple[str, ...]
    disc_names: Tuple[str, ...] = ()
    classes: Tuple[str, ...] = ()

    def __post_init__(self):
        for f in ("cont_names", "disc_names", "classes"):
            v = tuple(getattr(self, f))
            if any(not isinstance(s, str) or not s for s in v):
                raise ContractError(f"OutcomeLayout.{f} entries must be non-empty str")
            if len(set(v)) != len(v):
                raise ContractError(f"OutcomeLayout.{f} entries must be unique")
            object.__setattr__(self, f, v)
        if self.disc_names and len(self.classes) < 2:
            raise ContractError("discrete groups need >= 2 classes")
        if not self.cont_names and not self.disc_names:
            raise ContractError("an outcome needs at least one dimension")

    @property
    def Dc(self) -> int:
        return len(self.cont_names)

    @property
    def G(self) -> int:
        return len(self.disc_names)

    @property
    def K(self) -> int:
        return len(self.classes) if self.disc_names else 0

    def select(self, cont_index=None, disc_index=None) -> "OutcomeLayout":
        ci = range(self.Dc) if cont_index is None else cont_index
        di = range(self.G) if disc_index is None else disc_index
        dn = tuple(self.disc_names[i] for i in di)
        return OutcomeLayout(tuple(self.cont_names[i] for i in ci), dn,
                             self.classes if dn else ())


def entity_layout(n_entities: int, dim: int, classes: Sequence[str]) -> OutcomeLayout:
    cont = []
    for i in range(n_entities):
        cont += [f"e{i}.pos{d}" for d in range(dim)]
        cont += [f"e{i}.vel{d}" for d in range(dim)]
    return OutcomeLayout(tuple(cont), tuple(f"e{i}.event" for i in range(n_entities)),
                         tuple(classes))


class MixedOutcome:
    """Mixture of M components of (diag Gaussian) x (independent categoricals).

    Arrays: means/vars (M, B, Dc), probs (M, B, G, K), log_weights (B, M).
    Construct with `gaussian_categorical` (M=1) or `mixture` (stack).
    """

    __slots__ = ("layout", "means", "vars", "probs", "log_weights")

    def __init__(self, layout: OutcomeLayout, means, vars_, probs, log_weights):
        if not isinstance(layout, OutcomeLayout):
            raise ContractError("MixedOutcome needs an OutcomeLayout")
        means = _arr(means, "means", 3)
        vars_ = _arr(vars_, "vars", 3)
        M, B, Dc = means.shape
        if Dc != layout.Dc or vars_.shape != means.shape:
            raise ContractError(f"continuous arrays {means.shape}/{vars_.shape} do "
                                f"not match layout Dc={layout.Dc}")
        if np.any(vars_ <= 0):
            raise ContractError("variances must be > 0 (a zero-variance Gaussian "
                                "claims certainty the model does not have)")
        probs = _arr(probs, "probs", 4)
        if probs.shape != (M, B, layout.G, layout.K):
            raise ContractError(f"probs shape {probs.shape} != {(M, B, layout.G, layout.K)}")
        if layout.G and (np.any(probs < 0) or
                         not np.allclose(probs.sum(-1), 1.0, atol=1e-5)):
            raise ContractError("categorical probs must be >= 0 and sum to 1")
        lw = _arr(log_weights, "log_weights", 2)
        if lw.shape != (B, M):
            raise ContractError(f"log_weights shape {lw.shape} != {(B, M)}")
        lw = lw - logsumexp(lw, axis=1, keepdims=True)
        self.layout, self.means, self.vars, self.probs, self.log_weights = \
            layout, means, vars_, probs, lw

    # ---- constructors ----------------------------------------------------
    @classmethod
    def gaussian_categorical(cls, layout, mean, var, probs=None) -> "MixedOutcome":
        mean = _arr(mean, "mean", 2)
        B = mean.shape[0]
        if probs is None:
            probs = np.zeros((B, layout.G, layout.K))
        probs = np.asarray(probs, np.float64)
        return cls(layout, mean[None], np.asarray(var, np.float64)[None],
                   probs[None], np.zeros((B, 1)))

    @classmethod
    def mixture(cls, parts: Sequence["MixedOutcome"], weights=None) -> "MixedOutcome":
        if not parts:
            raise ContractError("mixture of zero parts")
        lay = parts[0].layout
        if any(p.layout != lay for p in parts):
            raise ContractError("mixture parts have different layouts")
        B = parts[0].batch_size
        if any(p.batch_size != B for p in parts):
            raise ContractError("mixture parts have different batch sizes")
        w = np.full(len(parts), 1.0 / len(parts)) if weights is None else \
            np.asarray(weights, np.float64)
        if w.shape != (len(parts),) or np.any(w < 0) or w.sum() <= 0:
            raise ContractError("mixture weights must be (P,) non-negative, sum > 0")
        with np.errstate(divide="ignore"):
            lw = np.concatenate([p.log_weights + np.log(wi) for p, wi in zip(parts, w)], 1)
        return cls(lay, np.concatenate([p.means for p in parts], 0),
                   np.concatenate([p.vars for p in parts], 0),
                   np.concatenate([p.probs for p in parts], 0), lw)

    # ---- shape -------------------------------------------------------------
    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    @property
    def batch_size(self) -> int:
        return self.means.shape[1]

    def weights(self) -> np.ndarray:
        return np.exp(self.log_weights)                       # (B, M)

    # ---- moments -------------------------------------------------------------
    def mean(self) -> np.ndarray:
        """(B, Dc) mixture mean of the continuous part."""
        return np.einsum("bm,mbd->bd", self.weights(), self.means)

    def decompose_variance(self) -> Tuple[np.ndarray, np.ndarray]:
        """(within, between): mean of component variances and variance of
        component means, both (B, Dc). total = within + between."""
        w = self.weights()
        mu = self.mean()
        within = np.einsum("bm,mbd->bd", w, self.vars)
        between = np.einsum("bm,mbd->bd", w, (self.means - mu[None]) ** 2)
        return within, between

    def variance(self) -> np.ndarray:
        a, e = self.decompose_variance()
        return a + e

    def probs_marginal(self) -> np.ndarray:
        """(B, G, K) mixture class probabilities per discrete group."""
        return np.einsum("bm,mbgk->bgk", self.weights(), self.probs)

    def mode_discrete(self) -> np.ndarray:
        return np.argmax(self.probs_marginal(), -1) if self.layout.G else \
            np.zeros((self.batch_size, 0), int)

    # ---- likelihood ------------------------------------------------------------
    def _component_logp(self, x: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        out = {}
        xc = x.get("continuous", UNKNOWN) if isinstance(x, Mapping) else x
        xd = x.get("discrete", UNKNOWN) if isinstance(x, Mapping) else UNKNOWN
        M, B = self.n_components, self.batch_size
        if xc is not UNKNOWN and xc is not None and not _is_sentinel(xc) and self.layout.Dc:
            y = _arr(xc, "x['continuous']", 2)
            if y.shape != (B, self.layout.Dc):
                raise ContractError(f"continuous outcome shape {y.shape} != "
                                    f"{(B, self.layout.Dc)}")
            out["continuous"] = -0.5 * np.sum(
                _LOG2PI + np.log(self.vars) + (y[None] - self.means) ** 2 / self.vars, -1)
        if xd is not UNKNOWN and xd is not None and not _is_sentinel(xd) and self.layout.G:
            k = np.asarray(xd)
            if k.shape != (B, self.layout.G) or k.dtype.kind not in "iu":
                raise ContractError(f"discrete outcome must be int {(B, self.layout.G)}")
            if np.any((k >= self.layout.K) | (k < MISSING)):
                raise ContractError("discrete outcome class out of range")
            obs = k != MISSING
            kk = np.where(obs, k, 0)
            p = np.take_along_axis(self.probs, kk[None, :, :, None].repeat(M, 0), -1)[..., 0]
            with np.errstate(divide="ignore"):
                lp = np.where(obs[None], np.log(p), 0.0)
            out["discrete"] = lp.sum(-1)
        return out                                              # each (M, B)

    def log_prob_parts(self, x) -> Dict[str, np.ndarray]:
        """Marginal log-density of each observed part, (B,) each. A part that
        is absent from `x` (or a sentinel) is simply not in the result."""
        comp = self._component_logp(x)
        return {k: logsumexp(self.log_weights.T + v, axis=0) for k, v in comp.items()}

    def log_prob(self, x) -> np.ndarray:
        """(B,) joint log-density of the observed parts under the mixture.
        `x` = {"continuous": (B, Dc), "discrete": (B, G) int}; either may be
        omitted. Nothing observed -> zeros (no evidence, no change)."""
        comp = self._component_logp(x)
        if not comp:
            return np.zeros(self.batch_size)
        joint = sum(comp.values())
        return logsumexp(self.log_weights.T + joint, axis=0)

    def cdf(self, y) -> np.ndarray:
        """(B, Dc) per-dimension marginal CDF at y (the PIT value)."""
        y = _arr(y, "y", 2)
        z = (y[None] - self.means) / np.sqrt(self.vars)
        return np.einsum("bm,mbd->bd", self.weights(), ndtr(z))

    # ---- sampling --------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """{"continuous": (n, B, Dc), "discrete": (n, B, G) int}."""
        if not isinstance(rng, np.random.Generator):
            raise ContractError("sample needs a numpy Generator (no global RNG)")
        n = int(n)
        B, Dc, G = self.batch_size, self.layout.Dc, self.layout.G
        cw = np.cumsum(self.weights(), 1)                      # (B, M)
        u = rng.random((n, B, 1))
        m = np.minimum((u > cw[None]).sum(-1), self.n_components - 1)   # (n, B)
        bi = np.arange(B)[None].repeat(n, 0)
        mu, var = self.means[m, bi], self.vars[m, bi]          # (n, B, Dc)
        cont = mu + np.sqrt(var) * rng.standard_normal((n, B, Dc))
        if G:
            pc = np.cumsum(self.probs[m, bi], -1)              # (n, B, G, K)
            uu = rng.random((n, B, G, 1))
            disc = np.minimum((uu > pc).sum(-1), self.layout.K - 1)
        else:
            disc = np.zeros((n, B, 0), int)
        return {"continuous": cont, "discrete": disc.astype(np.int64)}

    # ---- entropy ---------------------------------------------------------------
    def entropy(self) -> Dict[str, Any]:
        """Exact entropy where it has a closed form, else UNKNOWN:
        continuous exact only for M == 1; discrete exact for M == 1 or G == 1."""
        out: Dict[str, Any] = {}
        if self.layout.Dc:
            out["continuous"] = (0.5 * np.sum(1.0 + _LOG2PI + np.log(self.vars[0]), -1)
                                 if self.n_components == 1 else UNKNOWN)
        if self.layout.G:
            if self.n_components == 1 or self.layout.G == 1:
                p = self.probs_marginal()
                with np.errstate(divide="ignore", invalid="ignore"):
                    h = -np.where(p > 0, p * np.log(p), 0.0)
                out["discrete"] = h.sum((-1, -2))
            else:
                out["discrete"] = UNKNOWN
        return out

    # ---- views -------------------------------------------------------------
    def select(self, cont_index=None, disc_index=None) -> "MixedOutcome":
        lay = self.layout.select(cont_index, disc_index)
        ci = slice(None) if cont_index is None else np.asarray(cont_index, int)
        di = slice(None) if disc_index is None else np.asarray(disc_index, int)
        probs = self.probs[:, :, di] if lay.G else \
            np.zeros(self.probs.shape[:2] + (0, 0))
        return MixedOutcome(lay, self.means[:, :, ci], self.vars[:, :, ci],
                            probs, self.log_weights)

    def index(self, b) -> "MixedOutcome":
        b = np.atleast_1d(np.asarray(b, int))
        return MixedOutcome(self.layout, self.means[:, b], self.vars[:, b],
                            self.probs[:, b], self.log_weights[b])

    # ---- records ------------------------------------------------------------
    def to_record(self) -> Dict[str, Any]:
        return {"kind": OUTCOME_KIND, "schema": OUTCOME_SCHEMA,
                "cont_names": self.layout.cont_names,
                "disc_names": self.layout.disc_names,
                "classes": self.layout.classes,
                "means": self.means.astype(np.float64),
                "vars": self.vars.astype(np.float64),
                "probs": self.probs.astype(np.float64),
                "log_weights": self.log_weights.astype(np.float64)}

    @classmethod
    def from_record(cls, d: Mapping[str, Any]) -> "MixedOutcome":
        if d.get("kind") != OUTCOME_KIND or d.get("schema") != OUTCOME_SCHEMA:
            raise ContractError(f"not a {OUTCOME_KIND} v{OUTCOME_SCHEMA} outcome: "
                                f"{d.get('kind')!r} v{d.get('schema')!r}")
        lay = OutcomeLayout(tuple(d["cont_names"]), tuple(d["disc_names"]),
                            tuple(d["classes"]))
        return cls(lay, d["means"], d["vars"], d["probs"], d["log_weights"])

    def __repr__(self):
        return (f"MixedOutcome(M={self.n_components}, B={self.batch_size}, "
                f"Dc={self.layout.Dc}, G={self.layout.G}, K={self.layout.K})")


def _is_sentinel(x) -> bool:
    from ..contracts import is_missing
    return is_missing(x)


@runtime_checkable
class OutcomeDistribution(Protocol):
    """What every predictive distribution offers. MixedOutcome is the one
    implementation; the protocol exists so a consumer states its needs."""

    layout: OutcomeLayout

    def mean(self) -> np.ndarray: ...
    def variance(self) -> np.ndarray: ...
    def probs_marginal(self) -> np.ndarray: ...
    def log_prob(self, x) -> np.ndarray: ...
    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]: ...
    def entropy(self) -> Dict[str, Any]: ...
    def cdf(self, y) -> np.ndarray: ...


# ======================================================================
# Structured state
# ======================================================================

class EntityBatch:
    """B states of N entities in D dims: pos/vel (B, N, D), attrs (B, N, A),
    all in `frame` (positions metres, velocities m/s by fixture convention).
    attrs are static per episode and never predicted."""

    __slots__ = ("pos", "vel", "attrs", "frame")

    def __init__(self, pos, vel, attrs, frame: str = "world"):
        pos, vel, attrs = _arr(pos, "pos", 3), _arr(vel, "vel", 3), _arr(attrs, "attrs", 3)
        if pos.shape != vel.shape or attrs.shape[:2] != pos.shape[:2]:
            raise ContractError(f"EntityBatch shapes disagree: pos {pos.shape}, "
                                f"vel {vel.shape}, attrs {attrs.shape}")
        if not isinstance(frame, str) or not frame:
            raise ContractError("EntityBatch needs a frame name")
        self.pos, self.vel, self.attrs, self.frame = pos, vel, attrs, frame

    B = property(lambda s: s.pos.shape[0])
    N = property(lambda s: s.pos.shape[1])
    D = property(lambda s: s.pos.shape[2])
    A = property(lambda s: s.attrs.shape[2])

    def continuous(self) -> np.ndarray:
        """(B, N*2D) entity-major [pos_i, vel_i]."""
        return np.concatenate([self.pos, self.vel], -1).reshape(self.B, -1)

    def with_continuous(self, x) -> "EntityBatch":
        x = _arr(x, "continuous", 2).reshape(self.B, self.N, 2 * self.D)
        return EntityBatch(x[..., :self.D], x[..., self.D:], self.attrs, self.frame)

    def translate(self, shift) -> "EntityBatch":
        """Coordinate translation: every position moves, nothing else."""
        s = _arr(shift, "shift", 1)
        return EntityBatch(self.pos + s, self.vel, self.attrs, self.frame)

    def index(self, idx) -> "EntityBatch":
        return EntityBatch(self.pos[idx], self.vel[idx], self.attrs[idx], self.frame)

    def repeat(self, n: int) -> "EntityBatch":
        """Each state repeated n times consecutively (B*n)."""
        return EntityBatch(np.repeat(self.pos, n, 0), np.repeat(self.vel, n, 0),
                           np.repeat(self.attrs, n, 0), self.frame)

    @staticmethod
    def concat(parts: Sequence["EntityBatch"]) -> "EntityBatch":
        fr = {p.frame for p in parts}
        if len(fr) != 1:
            raise ContractError(f"cannot concatenate frames {sorted(fr)}")
        return EntityBatch(np.concatenate([p.pos for p in parts]),
                           np.concatenate([p.vel for p in parts]),
                           np.concatenate([p.attrs for p in parts]), fr.pop())


class TransitionBatch:
    """state, action (B, N, Ad), dt (B,) seconds > 0, next_state, events
    (B, N) int (MISSING = not observed)."""

    __slots__ = ("state", "action", "dt", "next_state", "events")

    def __init__(self, state: EntityBatch, action, dt, next_state: EntityBatch, events=None):
        if not isinstance(state, EntityBatch) or not isinstance(next_state, EntityBatch):
            raise ContractError("state and next_state must be EntityBatch")
        if state.frame != next_state.frame:
            raise ContractError("state and next_state are in different frames")
        if state.pos.shape != next_state.pos.shape:
            raise ContractError("state and next_state shapes disagree")
        action = _arr(action, "action", 3)
        dt = _arr(dt, "dt", 1)
        if action.shape[:2] != (state.B, state.N) or dt.shape != (state.B,):
            raise ContractError(f"action {action.shape} / dt {dt.shape} do not match "
                                f"B={state.B}, N={state.N}")
        if np.any(dt <= 0):
            raise ContractError("dt must be > 0 seconds")
        ev = np.full((state.B, state.N), MISSING, np.int64) if events is None \
            else np.asarray(events)
        if ev.shape != (state.B, state.N) or ev.dtype.kind not in "iu":
            raise ContractError(f"events must be int (B, N), got {ev.dtype} {ev.shape}")
        self.state, self.action, self.dt, self.next_state, self.events = \
            state, action, dt, next_state, ev.astype(np.int64)

    def __len__(self):
        return self.state.B

    def subset(self, idx) -> "TransitionBatch":
        idx = np.asarray(idx)
        return TransitionBatch(self.state.index(idx), self.action[idx], self.dt[idx],
                               self.next_state.index(idx), self.events[idx])

    def target(self) -> Dict[str, np.ndarray]:
        """The observed outcome in MixedOutcome.log_prob form."""
        return {"continuous": self.next_state.continuous(), "discrete": self.events}

    @staticmethod
    def concat(parts: Sequence["TransitionBatch"]) -> "TransitionBatch":
        return TransitionBatch(EntityBatch.concat([p.state for p in parts]),
                               np.concatenate([p.action for p in parts]),
                               np.concatenate([p.dt for p in parts]),
                               EntityBatch.concat([p.next_state for p in parts]),
                               np.concatenate([p.events for p in parts]))


# ======================================================================
# Mechanism predictor
# ======================================================================

@runtime_checkable
class MechanismPredictor(Protocol):
    """mechanism_id   stable identity (never a display name; CLAUDE.md §4.6)
    version        int; 0 = never fitted, +1 per fit()
    assumptions    declared structure, e.g. {"translation_invariant": True}
    predict        (EntityBatch, action (B,N,Ad), dt (B,)) -> MixedOutcome
    fit            TransitionBatch -> report dict (incl. "train_seconds")
    applicability  EntityBatch -> (B,) in [0, 1], or UNKNOWN before any fit
    rollout        H-step prediction (see BaseMechanism.rollout)"""

    mechanism_id: str
    version: int
    assumptions: Dict[str, Any]

    def predict(self, state, action, dt) -> MixedOutcome: ...
    def fit(self, batch: TransitionBatch, **kw) -> Dict[str, Any]: ...
    def applicability(self, state) -> Any: ...
    def rollout(self, state, actions, dts, mode: str = "mean", n_samples: int = 1,
                rng: Optional[np.random.Generator] = None) -> List[MixedOutcome]: ...


class BaseMechanism:
    """Shared machinery. Subclasses implement `_predict` and `_fit` and may
    override `_support_features` (default: velocities + attributes, which are
    translation invariant; positions are added by models that read them)."""

    SUPPORT_MARGIN = 0.1      # fraction of the training range tolerated outside it

    def __init__(self, mechanism_id: str, n_entities: int, dim: int, attr_dim: int,
                 action_dim: int, classes: Sequence[str], assumptions=None):
        if not isinstance(mechanism_id, str) or not mechanism_id:
            raise ContractError("mechanism_id must be a non-empty string")
        self.mechanism_id = mechanism_id
        self.version = 0
        self.N, self.D, self.A, self.Ad = int(n_entities), int(dim), int(attr_dim), int(action_dim)
        self.layout = entity_layout(self.N, self.D, classes)
        self.assumptions: Dict[str, Any] = dict(assumptions or {})
        self.last_fit: Dict[str, Any] = {}
        self._lo = self._hi = None
        self.uses_absolute_position = False

    # ---- validation ---------------------------------------------------------
    def _check(self, state, action, dt):
        if not isinstance(state, EntityBatch):
            raise ContractError("state must be an EntityBatch")
        if (state.N, state.D, state.A) != (self.N, self.D, self.A):
            raise ContractError(f"{self.mechanism_id}: state (N={state.N}, D={state.D}, "
                                f"A={state.A}) != declared ({self.N}, {self.D}, {self.A})")
        action = _arr(action, "action", 3)
        dt = _arr(dt, "dt", 1)
        if action.shape != (state.B, self.N, self.Ad) or dt.shape != (state.B,):
            raise ContractError(f"action {action.shape} / dt {dt.shape} do not match "
                                f"state B={state.B}")
        if np.any(dt <= 0):
            raise ContractError("dt must be > 0")
        return action, dt

    def predict(self, state: EntityBatch, action, dt) -> MixedOutcome:
        action, dt = self._check(state, action, dt)
        return self._predict(state, action, dt)

    def fit(self, batch: TransitionBatch, **kw) -> Dict[str, Any]:
        if not isinstance(batch, TransitionBatch) or len(batch) == 0:
            raise ContractError("fit needs a non-empty TransitionBatch")
        self._check(batch.state, batch.action, batch.dt)
        t0 = time.perf_counter()
        rep = dict(self._fit(batch, **kw) or {})
        self._record_support(batch.state)
        self.version += 1
        rep.update(train_seconds=time.perf_counter() - t0, n=len(batch),
                   version=self.version)
        self.last_fit = rep
        return rep

    def _predict(self, state, action, dt) -> MixedOutcome:     # pragma: no cover
        raise NotImplementedError

    def _fit(self, batch, **kw):                                 # pragma: no cover
        raise NotImplementedError

    # ---- applicability ------------------------------------------------------
    def _support_features(self, state: EntityBatch) -> np.ndarray:
        f = [state.vel.reshape(state.B, -1), state.attrs.reshape(state.B, -1)]
        if self.uses_absolute_position:
            f.append(state.pos.reshape(state.B, -1))
        return np.concatenate(f, 1)

    def _record_support(self, state):
        f = self._support_features(state)
        lo, hi = f.min(0), f.max(0)
        if self._lo is not None:                     # support only grows
            lo, hi = np.minimum(lo, self._lo), np.maximum(hi, self._hi)
        self._lo, self._hi = lo, hi

    def applicability(self, state: EntityBatch):
        """Fraction of support features inside the (margin-widened) training
        range: 1.0 = every feature seen before; UNKNOWN if never fitted."""
        if self._lo is None:
            return UNKNOWN
        f = self._support_features(state)
        m = self.SUPPORT_MARGIN * np.maximum(self._hi - self._lo, 1e-9)
        inside = (f >= self._lo - m) & (f <= self._hi + m)
        return inside.mean(1)

    # ---- rollout --------------------------------------------------------------
    def rollout(self, state: EntityBatch, actions, dts, mode: str = "mean",
                n_samples: int = 1, rng: Optional[np.random.Generator] = None
                ) -> List[MixedOutcome]:
        """H-step prediction under given actions (H, B, N, Ad) and dts (H, B).

        mode "mean"   feed the predictive mean back in; returns the ONE-STEP
                      distribution around each fed-back mean — cheap, and
                      UNDER-dispersed beyond h = 1 (it ignores how uncertainty
                      compounds). Use for point error and drift.
        mode "sample" propagate n_samples particles, each advanced by a draw
                      from its own predictive; step h returns the equal-weight
                      mixture of the particles' predictives (M = n_samples x
                      the predictor's own M). Treats ALL predictive variance as
                      process noise, so it is OVER-dispersed when observation
                      noise dominates. Use for distributional scores.
        """
        actions = _arr(actions, "actions", 4)
        dts = _arr(dts, "dts", 2)
        H = actions.shape[0]
        if H < 1 or dts.shape != (H, state.B):
            raise ContractError(f"actions {actions.shape} / dts {dts.shape} disagree")
        out: List[MixedOutcome] = []
        if mode == "mean":
            s = state
            for h in range(H):
                d = self.predict(s, actions[h], dts[h])
                out.append(d)
                s = s.with_continuous(d.mean())
            return out
        if mode != "sample":
            raise ContractError(f"rollout mode must be 'mean' or 'sample', got {mode!r}")
        if rng is None:
            raise ContractError("sample rollout needs an explicit rng")
        n = int(n_samples)
        if n < 1:
            raise ContractError("n_samples must be >= 1")
        B = state.B
        s = state.repeat(n)                                   # (B*n) b-major
        for h in range(H):
            d = self.predict(s, np.repeat(actions[h], n, 0), np.repeat(dts[h], n, 0))
            out.append(_regroup(d, B, n))
            x = d.sample(1, rng)["continuous"][0]
            s = s.with_continuous(x)
        return out

    def state_dict(self):
        """Weights for runtime.SnapshotRegistry.publish (torch models override)."""
        return {}

    def resources(self) -> Dict[str, Any]:
        return {"params": 0}


def _regroup(d: MixedOutcome, B: int, n: int) -> MixedOutcome:
    """(M, B*n) b-major -> (M*n, B) mixture over the n particles."""
    M = d.n_components
    r = lambda a: a.reshape((M, B, n) + a.shape[2:]).transpose(
        (0, 2, 1) + tuple(range(3, a.ndim + 1))).reshape((M * n, B) + a.shape[2:])
    lw = d.log_weights.reshape(B, n, M).transpose(0, 2, 1).reshape(B, M * n) - np.log(n)
    return MixedOutcome(d.layout, r(d.means), r(d.vars), r(d.probs), lw)


# ======================================================================
# Records
# ======================================================================

def prediction_record(predictor, outcomes: Union[MixedOutcome, Sequence[MixedOutcome]],
                      actions, dts, *, prediction_id: str, source_belief: str,
                      snapshot, action_spec_id: str, t_wall: Optional[float] = None,
                      payload: Optional[Dict[str, Any]] = None) -> Prediction:
    """A contracts.Prediction for a (possibly 1-step) rollout.

    model_versions = predictor.model_versions() if it has one (an ensemble
    names every member) else {mechanism_id: version}. `snapshot` is a
    runtime.Snapshot (its provenance is stamped into the payload and its id
    becomes snapshot_id) or an explicit non-empty id string. A version-0
    (never fitted) predictor is refused: its output is a guess with no
    evidence behind it and must not be scored as a model's prediction.
    """
    from ..runtime.snapshots import Snapshot, stamp_prediction
    if isinstance(outcomes, MixedOutcome):
        outcomes = [outcomes]
    outcomes = list(outcomes)
    if int(getattr(predictor, "version", 0)) < 1:
        raise ContractError(f"{getattr(predictor, 'mechanism_id', predictor)!r} has "
                            f"never been fitted (version 0); refusing to record its "
                            f"output as a prediction")
    actions = np.asarray(actions, np.float64)
    dts = np.asarray(dts, np.float64)
    if actions.ndim == 3:
        actions, dts = actions[None], dts.reshape(1, -1)
    if len(actions) != len(outcomes) or dts.shape[0] != len(outcomes):
        raise ContractError(f"{len(outcomes)} outcomes for {len(actions)} actions / "
                            f"{dts.shape[0]} dts")
    mv = (predictor.model_versions() if hasattr(predictor, "model_versions")
          else {predictor.mechanism_id: int(predictor.version)})
    pl = dict(payload or {})
    pl.setdefault("provenance", "imagined")
    pl["assumptions"] = {k: v for k, v in getattr(predictor, "assumptions", {}).items()
                         if isinstance(v, (bool, int, float, str))}
    if isinstance(snapshot, Snapshot):
        sid = f"{snapshot.name}#{snapshot.snapshot_id}"
        pl = stamp_prediction(pl, {snapshot.name: snapshot})
    elif isinstance(snapshot, str) and snapshot:
        sid = snapshot
    else:
        raise ContractError("snapshot must be a runtime Snapshot or a non-empty id")
    return Prediction(
        prediction_id, source_belief, mv, sid, action_spec_id,
        tuple(a for a in actions), len(outcomes),
        {"kind": ROLLOUT_KIND, "steps": [o.to_record() for o in outcomes], "dt": dts},
        float(time.time() if t_wall is None else t_wall), pl)


def outcomes_from_prediction(pred: Prediction) -> List[MixedOutcome]:
    if pred.outcome.get("kind") != ROLLOUT_KIND:
        raise ContractError(f"prediction outcome kind {pred.outcome.get('kind')!r} "
                            f"is not {ROLLOUT_KIND!r}")
    return [MixedOutcome.from_record(s) for s in pred.outcome["steps"]]
