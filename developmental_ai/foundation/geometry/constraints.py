"""Constraint registry: laws with scope, applicability, evidence and diagnostics.

WHAT IS CLAIMED. A physical or game law is not "true" or "false" in the
abstract; it holds within a SCOPE, under APPLICABILITY conditions, on stated
EVIDENCE (plan §2.3). So a `Constraint` is evaluated against a state and
returns a `Diagnostic` with one of five statuses:

    satisfied     applicable, residual <= tolerance
    violated      applicable, residual >  tolerance
    inapplicable  the applicability predicate said no (with its reason)
    unknown       applicable, but an input needed for the residual is UNKNOWN
    error         the residual function raised (recorded, not swallowed
                  into "satisfied")

`diag.violated` is True / False only when the constraint was actually
checked; otherwise it is the INAPPLICABLE or UNKNOWN sentinel, which refuses
`bool()`. So `if not diag.violated:` cannot mistake an unchecked law for a
satisfied one.

HARD vs SOFT. Hard constraints are claimed exact within tolerance; soft ones
are approximate/uncertain laws and report a graded `severity` =
residual / tolerance. Neither kind changes anything by itself — this registry
only DIAGNOSES; consumers decide what a violation means.

EXAMPLES (each is also a producer/consumer fixture for the smoke test):
  * `rigid_distance_constraint` — pairwise distances preserved, applicable
    only to entities DECLARED rigid with evidence.
  * `gravity_coordinate_covariance_constraint` — a predictor's output must
    transform covariantly under a COORDINATE CHANGE (gravity re-expressed
    along with everything else). It is INAPPLICABLE to a PHYSICAL rotation,
    where gravity stays fixed in the world and the outcome may legitimately
    change. `incline_acceleration` is the reference predictor.
  * `conservation_constraint` — a soft count conservation inside a declared
    system boundary, with explicit exception events (e.g. block created /
    removed, item crossing the boundary) accounted for in the residual.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import ConstraintError
from .transforms import SE3, quat_rotate
from .values import INAPPLICABLE, UNKNOWN

STATUSES = ("satisfied", "violated", "inapplicable", "unknown", "error")
KINDS = ("hard", "soft")

Applicability = Callable[[Mapping[str, Any]], Union[bool, Tuple[bool, str]]]
Residual = Callable[[Mapping[str, Any]], Any]


class Diagnostic:
    __slots__ = ("name", "kind", "scope", "status", "applicable", "residual",
                 "tolerance", "violated", "severity", "reason", "evidence")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return (f"Diagnostic({self.name}: {self.status}, residual={self.residual!r}, "
                f"reason={self.reason!r})")


class Constraint:
    def __init__(self, name: str, scope: Any, applicable: Applicability,
                 residual: Residual, tolerance: float, kind: str = "hard",
                 evidence: Sequence[str] = ()):
        if not name:
            raise ConstraintError("constraint needs a name")
        if kind not in KINDS:
            raise ConstraintError(f"kind must be one of {KINDS}, got {kind!r}")
        tol = float(tolerance)
        if not (tol >= 0.0 and math.isfinite(tol)):
            raise ConstraintError("tolerance must be finite and >= 0")
        if kind == "soft" and tol == 0.0:
            raise ConstraintError("a soft constraint needs a positive tolerance")
        self.name = name
        self.scope = scope
        self.applicable = applicable
        self.residual = residual
        self.tolerance = tol
        self.kind = kind
        self.evidence = tuple(evidence)

    def evaluate(self, state: Mapping[str, Any]) -> Diagnostic:
        base = dict(name=self.name, kind=self.kind, scope=self.scope,
                    tolerance=self.tolerance, evidence=self.evidence)
        try:
            a = self.applicable(state)
        except Exception as e:  # applicability itself failed: report, never pass
            return Diagnostic(status="error", applicable=UNKNOWN, residual=UNKNOWN,
                              violated=UNKNOWN, severity=UNKNOWN,
                              reason=f"applicability check raised {type(e).__name__}: {e}",
                              **base)
        ok, why = (a if isinstance(a, tuple) else (bool(a), ""))
        if not ok:
            return Diagnostic(status="inapplicable", applicable=False,
                              residual=INAPPLICABLE, violated=INAPPLICABLE,
                              severity=INAPPLICABLE, reason=why or "not applicable",
                              **base)
        try:
            r = self.residual(state)
        except Exception as e:
            return Diagnostic(status="error", applicable=True, residual=UNKNOWN,
                              violated=UNKNOWN, severity=UNKNOWN,
                              reason=f"residual raised {type(e).__name__}: {e}", **base)
        if r is UNKNOWN:
            return Diagnostic(status="unknown", applicable=True, residual=UNKNOWN,
                              violated=UNKNOWN, severity=UNKNOWN,
                              reason="an input to the residual is UNKNOWN", **base)
        r = float(r)
        if not math.isfinite(r):
            return Diagnostic(status="error", applicable=True, residual=r,
                              violated=UNKNOWN, severity=UNKNOWN,
                              reason="non-finite residual", **base)
        r = abs(r)
        viol = r > self.tolerance
        sev = (r / self.tolerance) if self.tolerance > 0 else (math.inf if viol else 0.0)
        return Diagnostic(status="violated" if viol else "satisfied", applicable=True,
                          residual=r, violated=viol, severity=sev,
                          reason=why or ("residual exceeds tolerance" if viol else "within tolerance"),
                          **base)


class ConstraintRegistry:
    def __init__(self):
        self._c: Dict[str, Constraint] = {}

    def register(self, c: Constraint) -> Constraint:
        if not isinstance(c, Constraint):
            raise ConstraintError("register() takes a Constraint")
        if c.name in self._c:
            raise ConstraintError(f"constraint {c.name!r} already registered")
        self._c[c.name] = c
        return c

    def unregister(self, name: str) -> None:
        self._c.pop(name)

    def names(self) -> List[str]:
        return list(self._c)

    def evaluate(self, state: Mapping[str, Any],
                 names: Optional[Iterable[str]] = None) -> Dict[str, Diagnostic]:
        sel = self._c.keys() if names is None else list(names)
        return {n: self._c[n].evaluate(state) for n in sel}

    @staticmethod
    def summary(diags: Mapping[str, Diagnostic]) -> Dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for d in diags.values():
            out[d.status] += 1
        return out


# ===================================================================== examples

def _pairwise(p: np.ndarray) -> np.ndarray:
    d = p[:, None, :] - p[None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def rigid_distance_constraint(entity: str, tolerance: float = 1e-6,
                              evidence: Sequence[str] = ()) -> Constraint:
    """Pairwise distances of `entity`'s points are preserved before -> after.

    state = {"before": {"entities": {id: rec}}, "after": {...}} with
    rec = {"points": (N,3) metric array, "rigid_evidence": [str, ...]}.
    Applies only when the entity is present in both snapshots, has >= 2
    points in each with matching count, and carries non-empty
    `rigid_evidence` (rigidity is a declaration needing evidence, not a
    default).
    """
    def _rec(state, which):
        return state.get(which, {}).get("entities", {}).get(entity)

    def applicable(state):
        b, a = _rec(state, "before"), _rec(state, "after")
        if b is None or a is None:
            return False, f"{entity} not present in both snapshots"
        if not (b.get("rigid_evidence") and a.get("rigid_evidence")):
            return False, f"{entity} is not declared rigid (no rigid_evidence)"
        pb, pa = np.asarray(b["points"]), np.asarray(a["points"])
        if pb.ndim != 2 or pb.shape[-1] != 3 or pb.shape != pa.shape or len(pb) < 2:
            return False, f"{entity}: need matching (N>=2, 3) point sets"
        return True, ""

    def residual(state):
        pb = np.asarray(_rec(state, "before")["points"], dtype=np.float64)
        pa = np.asarray(_rec(state, "after")["points"], dtype=np.float64)
        return float(np.max(np.abs(_pairwise(pa) - _pairwise(pb))))

    return Constraint(f"rigid_distance[{entity}]", scope=("entity", entity),
                      applicable=applicable, residual=residual, tolerance=tolerance,
                      kind="hard", evidence=evidence)


def incline_acceleration(scene: Mapping[str, Any]) -> np.ndarray:
    """Frictionless block on an inclined plane: a = g - (g.n) n.

    `scene` = {"normal": unit plane normal, "gravity": gravity vector}, both
    expressed in the SAME frame. The reference covariant predictor.
    """
    n = np.asarray(scene["normal"], dtype=np.float64)
    n = n / np.linalg.norm(n)
    g = np.asarray(scene["gravity"], dtype=np.float64)
    return g - float(g @ n) * n


def coordinate_change(scene: Mapping[str, Any], new_T_old: SE3) -> Dict[str, Any]:
    """Re-express a scene in new coordinates: EVERY direction rotates,
    gravity included. The world has not changed."""
    return {"normal": new_T_old.apply_direction(scene["normal"]),
            "gravity": new_T_old.apply_direction(scene["gravity"])}


def physical_rotation(scene: Mapping[str, Any], rot: SE3) -> Dict[str, Any]:
    """Physically rotate the plane in the world: its normal turns, gravity
    (an external field fixed to the world) does not."""
    return {"normal": rot.apply_direction(scene["normal"]),
            "gravity": np.asarray(scene["gravity"], dtype=np.float64)}


def gravity_coordinate_covariance_constraint(
        predictor: Callable[[Mapping[str, Any]], np.ndarray],
        tolerance: float = 1e-9, evidence: Sequence[str] = ()) -> Constraint:
    """predictor(T scene) == T predictor(scene) for a COORDINATE change T.

    state = {"scene": {...}, "transformation": {"kind": "coordinate_change" |
    "physical_rotation", "T": SE3}}. A physical rotation is not a symmetry
    while gravity is held fixed, so the constraint is INAPPLICABLE to it —
    not satisfied, not violated.
    """
    def applicable(state):
        tr = state.get("transformation") or {}
        k = tr.get("kind")
        if k == "coordinate_change":
            return True, ""
        if k == "physical_rotation":
            return False, ("physical rotation with gravity fixed in the world is "
                           "not a symmetry of the outcome; covariance does not apply")
        return False, f"no declared transformation kind (got {k!r})"

    def residual(state):
        T = state["transformation"]["T"]
        scene = state["scene"]
        direct = T.apply_direction(predictor(scene))
        via = predictor(coordinate_change(scene, T))
        return float(np.max(np.abs(direct - via)))

    return Constraint("gravity_coordinate_covariance", scope=("predictor", getattr(
        predictor, "__name__", "predictor")), applicable=applicable, residual=residual,
        tolerance=tolerance, kind="hard", evidence=evidence)


def conservation_constraint(quantity: str, boundary: str,
                            exceptions: Mapping[str, float],
                            tolerance: float = 0.5,
                            evidence: Sequence[str] = ()) -> Constraint:
    """Soft conservation of a count inside a declared system boundary.

    state = {"boundary": str, "before": {quantity: n}, "after": {quantity: n},
    "events": [{"type": str, "count": k}, ...]}. Each event whose type is in
    `exceptions` contributes exceptions[type] * count to the EXPECTED change
    (e.g. {"block_created": +1, "block_removed": -1, "item_entered": +1,
    "item_left": -1}). residual = (after - before) - expected. Measured over
    a different boundary -> INAPPLICABLE; a count of UNKNOWN -> unknown.
    """
    ex = {str(k): float(v) for k, v in exceptions.items()}

    def applicable(state):
        b = state.get("boundary")
        if b != boundary:
            return False, f"measured over boundary {b!r}, law declared for {boundary!r}"
        if quantity not in state.get("before", {}) or quantity not in state.get("after", {}):
            return False, f"{quantity} not measured in both snapshots"
        return True, ""

    def residual(state):
        nb, na = state["before"][quantity], state["after"][quantity]
        if nb is UNKNOWN or na is UNKNOWN:
            return UNKNOWN
        expected = 0.0
        for ev in state.get("events", ()):
            t = ev.get("type")
            if t in ex:
                c = ev.get("count", 1)
                if c is UNKNOWN:
                    return UNKNOWN
                expected += ex[t] * float(c)
        return (float(na) - float(nb)) - expected

    return Constraint(f"conservation[{quantity}@{boundary}]",
                      scope=("boundary", boundary, quantity), applicable=applicable,
                      residual=residual, tolerance=tolerance, kind="soft",
                      evidence=evidence)
