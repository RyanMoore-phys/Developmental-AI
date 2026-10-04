"""Persistent identity over per-frame entity features (plan Stage 8 items 2-3).

WHAT IS CLAIMED
    Given, each frame, an UNORDERED set of detections (2-d position + an
    appearance feature vector, e.g. slot centroids + slot identity vectors
    from `slots/attention.py`), the tracker maintains entities whose
    identity persists across frames:

    1. CORRESPONDENCE IS PROBABILISTIC. Each (track, detection) pair gets a
       negative log-likelihood: constant-velocity Kalman position term
       (Mahalanobis + log det of the innovation covariance) + an appearance
       term kappa*(1 - cos) - log P_D. The best joint assignment is Hungarian
       (scipy.optimize.linear_sum_assignment) over the matrix augmented with
       "track missed" and "detection is new/clutter" options, and every
       track also gets SOFT association probabilities (PDA-style: a
       likelihood ratio against clutter, plus a miss option).
    2. TRACK STATES: tentative -> confirmed (confirm_hits consecutive hits)
       -> occluded (missed; coasting on the motion model with growing
       covariance) -> confirmed again on re-identification, or -> lost after
       `reid_window` missed frames (not a latch: revision.RevisionEngine
       can stitch a lost identity back onto a new track when the evidence
       says it is the same entity). Tentative tracks die on their first
       `tentative_max_misses`+1 misses, so single-frame flicker never
       becomes a confirmed entity. Lost is terminal; local ids are never
       reused within a scope.
    3. IDENTITY-SWAP AMBIGUITY IS REPORTED, NOT RESOLVED. For each pair of
       confirmed/occluded tracks the cost of the chosen assignment is
       compared with the one-swap alternative; q = P(swap) =
       sigmoid(c_chosen - c_swapped). Above `ambiguity_threshold` an
       ambiguity event is recorded and both tracks' `identity_confidence`
       (P(track still follows the entity it was born on)) is updated by the
       closed-pair rule c_i <- (1-q) c_i + q (1 - c_j). Two indistinguishable
       entities that meet therefore end at ~0.5 ("coin flip"), never at a
       confident wrong answer.
    4. EGO-MOTION: `ego_shift` is the image displacement that the agent's
       OWN motion predicts for static scenery (for a camera rotation it is
       depth-independent — docs/PERSPECTIVE_LEARNING.md contract O — so it is
       exactly compensable; translational parallax is not, and is left to
       process noise). It is added to every prediction.
    5. SCOPE: EntityIds are EntityId(scope, "t<n>"). `reset(new_scope)`
       drops every track; resetting into the SAME scope raises (an episode
       cannot restart), so no identity survives a world reset.
    6. PROVENANCE: the tracker accepts "sensor"/"inferred" detections only.
       Evaluator truth and imagined detections raise PrivilegedInputError —
       identity must be earned from permitted observations.

DECLARED APPROXIMATIONS
    Soft probabilities are per-track marginals (not full JPDA exclusivity).
    The confidence rule is exact for a closed pair and approximate for
    larger groups. The appearance likelihood of an unmatched detection is
    scored as cos = 0 (an unrelated direction).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..contracts import EntityId, ObsRef, Scope, StateBelief, UNKNOWN
from .errors import PerceptionError, PrivilegedInputError

TRACK_STATES = ("tentative", "confirmed", "occluded", "lost")
ACCEPTED_PROVENANCE = ("sensor", "inferred")
TRACKER_REPRESENTATION = "identity_tracker"
_BIG = 1e9

_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
              dtype=np.float64)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)


@dataclass
class TrackerConfig:
    meas_std: float = 0.01          # detection position noise (position units)
    process_std: float = 0.006      # white-noise acceleration per frame
    init_vel_std: float = 0.02      # prior velocity spread of a new track
    p_detect: float = 0.9
    clutter_density: float = 0.25   # expected clutter detections / unit area
    feature_kappa: float = 4.0      # appearance concentration
    confirm_hits: int = 3
    tentative_max_misses: int = 0
    reid_window: int = 10           # max coasting frames before "lost"
    feature_lr: float = 0.3
    ambiguity_threshold: float = 0.05
    gate_maha2: float = 25.0        # hard gate on squared Mahalanobis

    def validate(self) -> "TrackerConfig":
        pos = ("meas_std", "process_std", "init_vel_std", "clutter_density",
               "feature_kappa", "gate_maha2")
        for k in pos:
            v = getattr(self, k)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise PerceptionError(f"TrackerConfig.{k} must be > 0, got {v!r}")
        if not 0.0 < self.p_detect < 1.0:
            raise PerceptionError("p_detect must be in (0, 1)")
        if not 0.0 < self.feature_lr <= 1.0:
            raise PerceptionError("feature_lr must be in (0, 1]")
        if not 0.0 < self.ambiguity_threshold < 0.5:
            raise PerceptionError("ambiguity_threshold must be in (0, 0.5)")
        for k in ("confirm_hits", "reid_window"):
            if int(getattr(self, k)) < 1:
                raise PerceptionError(f"{k} must be >= 1")
        if int(self.tentative_max_misses) < 0:
            raise PerceptionError("tentative_max_misses must be >= 0")
        return self


@dataclass
class Track:
    local: str
    entity_id: EntityId
    state: str
    x: np.ndarray                   # (4,) x, y, vx, vy
    P: np.ndarray                   # (4, 4)
    feature: np.ndarray             # (F,) unit vector
    attrs: Optional[np.ndarray]
    born_frame: int
    last_seen: int
    hits: int = 1                   # consecutive hits while tentative
    total_hits: int = 1
    misses: int = 0
    identity_confidence: float = 1.0
    swap_partners: Dict[str, float] = field(default_factory=dict)
    history: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    innovations: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    last_innovation: Optional[np.ndarray] = None
    end_reason: str = ""

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()


@dataclass
class StepResult:
    frame: int
    assignments: Dict[int, EntityId]          # detection index -> entity
    births: List[EntityId]
    deaths: List[EntityId]
    reidentified: List[EntityId]
    ambiguities: List[Dict[str, Any]]
    assoc_probs: Dict[str, np.ndarray]        # track local -> (M+1,), last = miss


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


class IdentityTracker:
    """Multi-entity tracker with probabilistic correspondence. See module
    docstring for the claims."""

    def __init__(self, scope: Scope, config: Optional[TrackerConfig] = None,
                 residuals=None):
        self.cfg = (config or TrackerConfig()).validate()
        self.residuals = residuals          # duck-typed: .add(frame, kind, local, vec)
        self._q = self._process_cov()
        self._R = np.eye(2) * self.cfg.meas_std ** 2
        self._init_scope(scope)

    # ------------------------------------------------------------ scope
    def _init_scope(self, scope: Scope) -> None:
        if not isinstance(scope, Scope):
            raise PerceptionError(f"scope must be a contracts Scope, got "
                                  f"{type(scope).__name__}")
        self.scope = scope
        self.frame = -1
        self._next = 0
        self._tracks: Dict[str, Track] = {}
        self._ended: Dict[str, Track] = {}
        self.aliases: Dict[str, str] = {}     # absorbed local -> surviving local
        self.ambiguity_log: List[Dict[str, Any]] = []
        self._log: List[Dict[str, Any]] = []
        self._feat_dim: Optional[int] = None

    def reset(self, scope: Scope) -> None:
        """New episode: every identity is dropped. The same scope twice is
        refused — an episode does not restart."""
        if scope == self.scope:
            raise PerceptionError(
                f"reset into the same scope {scope}; identities never "
                f"persist across a reset, so a reset needs a new episode id")
        self._init_scope(scope)

    # ------------------------------------------------------------ model
    def _process_cov(self) -> np.ndarray:
        q = self.cfg.process_std ** 2
        g = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        return q * g @ g.T

    def _new_track(self, z, f, attrs) -> Track:
        local = f"t{self._next}"
        self._next += 1
        P = np.diag([self.cfg.meas_std ** 2] * 2 + [self.cfg.init_vel_std ** 2] * 2)
        t = Track(local=local, entity_id=EntityId(self.scope, local),
                  state="tentative", x=np.array([z[0], z[1], 0.0, 0.0]),
                  P=P, feature=_unit(f.copy()),
                  attrs=None if attrs is None else attrs.copy(),
                  born_frame=self.frame, last_seen=self.frame)
        t.history.append((self.frame, z.copy()))
        if self.cfg.confirm_hits <= 1:
            t.state = "confirmed"
        return t

    def _live(self) -> List[Track]:
        return list(self._tracks.values())

    # ------------------------------------------------------------- step
    def _validate_inputs(self, positions, features, ego_shift, attrs,
                         provenance):
        if provenance not in ACCEPTED_PROVENANCE:
            raise PrivilegedInputError(
                f"tracker accepts {ACCEPTED_PROVENANCE} detections only, got "
                f"{provenance!r}: identity must be earned from permitted "
                f"observations, not from evaluator truth or imagination")
        for name, v in (("positions", positions), ("features", features)):
            if not isinstance(v, np.ndarray):
                raise PrivilegedInputError(
                    f"{name} must be a numpy array, got {type(v).__name__} "
                    f"(oracle/evaluator objects are refused)")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise PerceptionError(f"positions must be (M, 2), got {positions.shape}")
        pos = positions.astype(np.float64)
        m = pos.shape[0]
        if features.ndim != 2 or features.shape[0] != m:
            raise PerceptionError(f"features must be (M={m}, F), got {features.shape}")
        feats = features.astype(np.float64)
        if m:
            if self._feat_dim is None:
                self._feat_dim = feats.shape[1]
            elif feats.shape[1] != self._feat_dim:
                raise PerceptionError(f"feature dim changed {self._feat_dim} -> "
                                      f"{feats.shape[1]} within a scope")
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(feats))):
            raise PerceptionError("positions/features must be finite")
        ego = np.zeros(2) if ego_shift is None else np.asarray(ego_shift, dtype=np.float64)
        if ego.shape != (2,) or not np.all(np.isfinite(ego)):
            raise PerceptionError(f"ego_shift must be finite (2,), got {ego!r}")
        at = None
        if attrs is not None:
            at = np.asarray(attrs, dtype=np.float64)
            if at.ndim != 2 or at.shape[0] != m:
                raise PerceptionError(f"attrs must be (M={m}, A), got {at.shape}")
        return pos, feats, ego, at

    def _pair_cost(self, t: Track, z: np.ndarray, f: np.ndarray):
        S = _H @ t.P @ _H.T + self._R
        Si = np.linalg.inv(S)
        d = z - t.x[:2]
        m2 = float(d @ Si @ d)
        if m2 > self.cfg.gate_maha2:
            return _BIG, m2
        _, logdet = np.linalg.slogdet(2.0 * math.pi * S)
        cos = float(t.feature @ _unit(f)) if f.size else 1.0
        c = (0.5 * m2 + 0.5 * logdet + self.cfg.feature_kappa * (1.0 - cos)
             - math.log(self.cfg.p_detect))
        return c, m2

    def step(self, positions: np.ndarray, features: np.ndarray,
             ego_shift=None, attrs=None, support: Optional[ObsRef] = None,
             provenance: str = "sensor") -> StepResult:
        pos, feats, ego, at = self._validate_inputs(
            positions, features, ego_shift, attrs, provenance)
        if support is not None and (not isinstance(support, ObsRef)
                                    or support.scope != self.scope):
            raise PerceptionError(f"support must be an ObsRef in scope "
                                  f"{self.scope}, got {support!r}")
        self.frame += 1
        cfg = self.cfg
        # 1. predict (ego-motion is added to every track's prediction)
        live = self._live()
        for t in live:
            t.x = _F @ t.x
            t.x[:2] += ego
            t.P = _F @ t.P @ _F.T + self._q
        T, M = len(live), pos.shape[0]
        c_miss = -math.log(1.0 - cfg.p_detect)
        c_birth = -math.log(cfg.clutter_density) + cfg.feature_kappa
        # 2. costs
        C = np.full((T, M), _BIG)
        for i, t in enumerate(live):
            for j in range(M):
                C[i, j], _ = self._pair_cost(t, pos[j], feats[j])
        # 3. augmented Hungarian
        n = T + M
        A = np.full((n, n), _BIG)
        A[:T, :M] = C
        if T:
            A[:T, M:] = np.where(np.eye(T, dtype=bool), c_miss, _BIG)
        if M:
            A[T:, :M] = np.where(np.eye(M, dtype=bool), c_birth, _BIG)
        A[T:, M:] = 0.0
        rows, cols = linear_sum_assignment(A) if n else ([], [])
        t2d: Dict[int, int] = {}
        for r, c in zip(rows, cols):
            if r < T and c < M and C[r, c] < _BIG:
                t2d[r] = c
        d2t = {d: t for t, d in t2d.items()}
        # 4. soft association probabilities (per-track marginals)
        probs: Dict[str, np.ndarray] = {}
        for i, t in enumerate(live):
            logits = np.concatenate([-(C[i] - c_birth), [-c_miss]])
            logits = np.where(np.concatenate([C[i], [0.0]]) >= _BIG, -np.inf, logits)
            mx = np.max(logits)
            p = np.exp(logits - mx)
            probs[t.local] = p / p.sum()
        # 5. swap ambiguity between identity-bearing tracks
        ambig = self._swap_ambiguity(live, C, t2d, c_miss)
        # 6. update matched
        births, deaths, reid = [], [], []
        result_assign: Dict[int, EntityId] = {}
        for i, t in enumerate(live):
            if i in t2d:
                j = t2d[i]
                z = pos[j]
                S = _H @ t.P @ _H.T + self._R
                K = t.P @ _H.T @ np.linalg.inv(S)
                innov = z - t.x[:2]
                t.x = t.x + K @ innov
                t.P = (np.eye(4) - K @ _H) @ t.P
                t.feature = _unit((1 - cfg.feature_lr) * t.feature
                                  + cfg.feature_lr * _unit(feats[j]))
                if at is not None:
                    t.attrs = at[j].copy() if t.attrs is None else \
                        (1 - cfg.feature_lr) * t.attrs + cfg.feature_lr * at[j]
                t.history.append((self.frame, z.copy()))
                t.innovations.append((self.frame, innov.copy()))
                t.last_innovation = innov.copy()
                if self.residuals is not None and t.state != "tentative":
                    self.residuals.add(self.frame, "position_innovation",
                                       t.local, innov)
                    self.residuals.add(self.frame, "feature_residual", t.local,
                                       _unit(feats[j]) - t.feature)
                t.last_seen = self.frame
                t.total_hits += 1
                if t.state == "tentative":
                    t.hits += 1
                    if t.hits >= cfg.confirm_hits:
                        t.state = "confirmed"
                elif t.state == "occluded":
                    t.state = "confirmed"
                    reid.append(t.entity_id)
                t.misses = 0
                result_assign[j] = t.entity_id
            else:
                t.misses += 1
                if t.state == "tentative":
                    t.hits = 0
                    if t.misses > cfg.tentative_max_misses:
                        self._end(t, "tentative_died")
                        deaths.append(t.entity_id)
                elif t.state == "confirmed":
                    t.state = "occluded"
                elif t.state == "occluded" and t.misses > cfg.reid_window:
                    self._end(t, "reid_window_expired")
                    deaths.append(t.entity_id)
        # 7. births
        for j in range(M):
            if j not in d2t:
                nt = self._new_track(pos[j], feats[j],
                                     None if at is None else at[j])
                self._tracks[nt.local] = nt
                births.append(nt.entity_id)
                result_assign[j] = nt.entity_id
                if self.residuals is not None:
                    self.residuals.add(self.frame, "unexplained_detection",
                                       nt.local, pos[j])
        # 8. log (the oracle reads this; nothing reads the oracle back)
        assign_local = [result_assign[j].local if j in result_assign else None
                        for j in range(M)]
        self._log.append({
            "frame": self.frame,
            "assign": tuple(assign_local),
            "states": {t.local: t.state for t in self._tracks.values()},
            "confidence": {t.local: t.identity_confidence
                           for t in self._tracks.values()},
            "support": support,
        })
        return StepResult(self.frame, result_assign, births, deaths, reid,
                          ambig, probs)

    def _swap_ambiguity(self, live, C, t2d, c_miss) -> List[Dict[str, Any]]:
        cfg = self.cfg
        idx = [i for i, t in enumerate(live) if t.state in ("confirmed", "occluded")]
        events = []
        updates: Dict[int, float] = {}
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                di, dj = t2d.get(i), t2d.get(j)
                if di is None and dj is None:
                    continue
                ci = C[i, di] if di is not None else c_miss
                cj = C[j, dj] if dj is not None else c_miss
                si = C[i, dj] if dj is not None else c_miss
                sj = C[j, di] if di is not None else c_miss
                if si >= _BIG or sj >= _BIG:
                    continue
                d = (si + sj) - (ci + cj)       # swapped minus chosen
                q = 1.0 / (1.0 + math.exp(min(50.0, max(-50.0, d))))
                if q <= cfg.ambiguity_threshold:
                    continue
                ti, tj = live[i], live[j]
                events.append({"frame": self.frame,
                               "tracks": (ti.entity_id, tj.entity_id),
                               "p_swap": q})
                ci_new = (1 - q) * ti.identity_confidence + q * (1 - tj.identity_confidence)
                cj_new = (1 - q) * tj.identity_confidence + q * (1 - ti.identity_confidence)
                updates[i] = min(updates.get(i, 1.0), ci_new)
                updates[j] = min(updates.get(j, 1.0), cj_new)
                ti.swap_partners[tj.local] = max(ti.swap_partners.get(tj.local, 0.0), q)
                tj.swap_partners[ti.local] = max(tj.swap_partners.get(ti.local, 0.0), q)
        for i, c in updates.items():
            live[i].identity_confidence = float(min(1.0, max(0.0, c)))
        self.ambiguity_log.extend(events)
        return events

    def _end(self, t: Track, reason: str) -> None:
        t.state = "lost"
        t.end_reason = reason
        self._tracks.pop(t.local, None)
        self._ended[t.local] = t

    # ---------------------------------------------------------- queries
    def tracks(self, states: Sequence[str] = ("confirmed", "occluded")
               ) -> List[Track]:
        bad = set(states) - set(TRACK_STATES)
        if bad:
            raise PerceptionError(f"unknown track states {sorted(bad)}")
        out = [t for t in self._tracks.values() if t.state in states]
        if "lost" in states:
            out += list(self._ended.values())
        return out

    def track(self, local: str) -> Track:
        if local in self._tracks:
            return self._tracks[local]
        if local in self._ended:
            return self._ended[local]
        raise PerceptionError(f"no track {local!r} in scope {self.scope}")

    def resolve(self, local: str) -> str:
        """Follow merge aliases to the surviving local id."""
        seen = set()
        while local in self.aliases and local not in seen:
            seen.add(local)
            local = self.aliases[local]
        return local

    def predict_positions(self, ego_shift=None,
                          states: Sequence[str] = ("confirmed",)
                          ) -> Dict[str, np.ndarray]:
        """Next-frame predicted positions, WITHOUT mutating any track."""
        ego = np.zeros(2) if ego_shift is None else np.asarray(ego_shift, float)
        return {t.local: (_F @ t.x)[:2] + ego for t in self.tracks(states)}

    def export_log(self) -> List[Dict[str, Any]]:
        out = copy.deepcopy(self._log)
        for rec in out:
            rec["aliases"] = dict(self.aliases)
        return out

    # ----------------------------------------------------- revision hooks
    def apply_merge(self, keep_local: str, absorb_local: str, reason: str) -> None:
        """`absorb` (a live track) is the same entity as `keep` (occluded or
        lost): the live track takes keep's identity; absorb's id becomes an
        alias. Called by revision.RevisionEngine."""
        keep, absorb = self.track(keep_local), self.track(absorb_local)
        if absorb.local not in self._tracks:
            raise PerceptionError("the absorbed track must be live")
        if keep.state not in ("occluded", "lost"):
            raise PerceptionError("only an occluded/lost track can be merged into")
        absorb_ghost = copy.copy(absorb)        # keeps absorb's own EntityId
        absorb_ghost.state = "lost"
        absorb_ghost.end_reason = f"merged_into:{keep.local}:{reason}"
        self._tracks.pop(absorb.local)
        self._tracks.pop(keep.local, None)
        self._ended.pop(keep.local, None)
        merged = absorb
        merged.history = keep.history + absorb.history
        merged.innovations = keep.innovations + absorb.innovations
        merged.local, merged.entity_id = keep.local, keep.entity_id
        merged.born_frame = keep.born_frame
        merged.identity_confidence = min(keep.identity_confidence,
                                         absorb.identity_confidence)
        merged.state = "confirmed" if absorb.state != "occluded" else "occluded"
        self._tracks[merged.local] = merged
        self._ended[absorb_local] = absorb_ghost
        self.aliases[absorb_local] = keep.local

    def apply_split(self, local: str, comp_a: Tuple[np.ndarray, np.ndarray],
                    comp_b: Tuple[np.ndarray, np.ndarray],
                    history_a: List[Tuple[int, np.ndarray]],
                    history_b: List[Tuple[int, np.ndarray]], reason: str,
                    covs: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> EntityId:
        """Track `local` was two entities. It keeps component a; a NEW
        identity is minted for component b (its identity is new evidence,
        so its confidence starts at 0.5, not 1)."""
        t = self.track(local)
        if t.local not in self._tracks:
            raise PerceptionError("only a live track can be split")
        (pa, va), (pb, vb) = comp_a, comp_b
        t.x = np.array([pa[0], pa[1], va[0], va[1]], dtype=np.float64)
        # Both histories restart from the re-labelled window: the older
        # points were a mixture of the two entities and would re-propose
        # the same split forever.
        t.history = list(history_a)
        t.innovations = []
        nt = self._new_track(np.asarray(pb, float), t.feature.copy(), t.attrs)
        nt.x = np.array([pb[0], pb[1], vb[0], vb[1]], dtype=np.float64)
        if covs is not None:
            for c in covs:
                if np.shape(c) != (4, 4) or not np.all(np.isfinite(c)):
                    raise PerceptionError("split covariances must be finite (4, 4)")
            t.P = np.asarray(covs[0], float) + self._q
            nt.P = np.asarray(covs[1], float) + self._q
        else:
            nt.P = t.P.copy()
        nt.state = "confirmed"
        nt.history = list(history_b)
        nt.identity_confidence = 0.5
        nt.end_reason = ""
        t.identity_confidence = min(t.identity_confidence, 0.5)
        t.swap_partners[nt.local] = 0.5
        nt.swap_partners[t.local] = 0.5
        self._tracks[nt.local] = nt
        self.ambiguity_log.append({"frame": self.frame,
                                   "tracks": (t.entity_id, nt.entity_id),
                                   "p_swap": 0.5, "split": reason})
        return nt.entity_id

    # ---------------------------------------------------------- beliefs
    def beliefs(self, registry, support: Sequence[ObsRef] = (),
                states: Sequence[str] = ("confirmed", "occluded")
                ) -> List[StateBelief]:
        """One StateBelief per identity-bearing track."""
        major, semver = registry.stamp(TRACKER_REPRESENTATION)
        sup = tuple(support)
        for s in sup:
            if not isinstance(s, ObsRef) or s.scope != self.scope:
                raise PerceptionError(f"support {s!r} not in scope {self.scope}")
        out = []
        for t in self.tracks(states):
            pstd = np.sqrt(np.clip(np.diag(t.P)[:2], 0, None))
            vstd = np.sqrt(np.clip(np.diag(t.P)[2:], 0, None))
            out.append(StateBelief(
                belief_id=(f"{TRACKER_REPRESENTATION}:{self.scope.environment}/"
                           f"{self.scope.stream}/{self.scope.episode}:"
                           f"{self.frame}:{t.local}"),
                representation=TRACKER_REPRESENTATION,
                representation_version=major,
                variables={"position": t.position, "velocity": t.velocity,
                           "appearance": t.feature.copy(),
                           "identity_confidence": float(t.identity_confidence),
                           "track_state": t.state},
                uncertainty={"position": pstd, "velocity": vstd,
                             "appearance": UNKNOWN,
                             "identity_confidence": UNKNOWN},
                support=sup,
                payload={"representation_semver": semver,
                         "entity_id": t.entity_id,
                         "swap_partners": dict(t.swap_partners),
                         "frame": int(self.frame),
                         "misses": int(t.misses),
                         "residual": (UNKNOWN if t.last_innovation is None
                                      else t.last_innovation.copy())}))
        return out
