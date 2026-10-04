"""Named coordinate frames and explicit conversion between them.

`FrameGraph` is a forest: each frame has at most one parent and a rigid
transform parent_T_child (see transforms.py for the convention). `convert`
composes along the tree through the common root. Asking for a frame that was
never registered raises `UnknownFrameError`; asking to convert between two
trees raises `DisconnectedFramesError` — there is no "assume identity".

POINTS vs DIRECTIONS. Points are moved by rotation AND translation;
directions (velocities, normals, gravity) by rotation only. `convert` makes
the caller say which (`kind=`), because guessing is how a gravity vector
ends up translated by the camera's position.

SCALE. Frame translations are in the graph's length unit (metric). A point
whose scale status is not metric cannot cross an edge with a non-zero
translation: the sum "unknown-scale point + metric offset" has no meaning.
Directions can — rotation commutes with an unknown positive scale.

AXIS CONVENTIONS are METADATA declared by an adapter (`AxisConvention`), e.g.
Minecraft's "yaw 0 faces +z, yaw 90 faces -x". The graph stores the
declaration on a frame; none of the maths here reads it. That keeps the core
engine-agnostic (cf. developmental_ai/sensors/heading.py, which made the same
choice for the same reason).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import (DegenerateError, DisconnectedFramesError, FrameMismatchError,
                     ScaleStatusError, UnitError, UnknownFrameError)
from .transforms import SE3, quat_from_axis_angle, quat_mul, quat_rotate
from .units import LENGTH, M, Quantity, Unit


class AxisConvention:
    """An adapter's declaration of what its axes and heading angles mean.

    up               world up axis (unit 3-vector)
    forward_at_zero  body forward when yaw = pitch = 0
    yaw_sign         heading(yaw) = R(up, yaw_sign * yaw) forward_at_zero
    pitch_sign       pitch rotates about lateral = up x forward_at_zero
                     by pitch_sign * pitch (applied in the body, before yaw)
    declared_by      who claims this, and `cases` = known-answer checks
                     [(yaw_deg, pitch_deg, expected_forward_xyz)] that the
                     declaration must reproduce (`verify()`).
    """

    def __init__(self, name: str, up: Sequence[float], forward_at_zero: Sequence[float],
                 yaw_sign: float, pitch_sign: float, declared_by: str,
                 cases: Sequence[Tuple[float, float, Sequence[float]]] = ()):
        up_v = np.asarray(up, dtype=np.float64)
        f_v = np.asarray(forward_at_zero, dtype=np.float64)
        if abs(np.linalg.norm(up_v) - 1) > 1e-12 or abs(np.linalg.norm(f_v) - 1) > 1e-12:
            raise DegenerateError(f"{name}: up and forward must be unit vectors")
        if abs(float(up_v @ f_v)) > 1e-12:
            raise DegenerateError(f"{name}: forward must be orthogonal to up")
        if yaw_sign not in (1, -1, 1.0, -1.0) or pitch_sign not in (1, -1, 1.0, -1.0):
            raise DegenerateError(f"{name}: yaw/pitch signs must be +-1")
        if not declared_by:
            raise DegenerateError(f"{name}: a convention needs a declared_by source")
        self.name = name
        self.up = up_v
        self.forward_at_zero = f_v
        self.lateral = np.cross(up_v, f_v)
        self.yaw_sign = float(yaw_sign)
        self.pitch_sign = float(pitch_sign)
        self.declared_by = declared_by
        self.cases = tuple((float(a), float(b), tuple(float(x) for x in c))
                           for a, b, c in cases)

    def orientation(self, yaw_rad: float, pitch_rad: float = 0.0) -> np.ndarray:
        """world_q_body for a body whose axes coincide with the world's at
        yaw = pitch = 0."""
        qy = quat_from_axis_angle(self.up, self.yaw_sign * float(yaw_rad))
        qp = quat_from_axis_angle(self.lateral, self.pitch_sign * float(pitch_rad))
        return quat_mul(qy, qp)

    def forward(self, yaw_rad: float, pitch_rad: float = 0.0) -> np.ndarray:
        return quat_rotate(self.orientation(yaw_rad, pitch_rad), self.forward_at_zero)

    def verify(self, tol: float = 1e-9) -> List[str]:
        """Return a list of failed known-answer cases (empty = consistent)."""
        bad = []
        for yd, pd, exp in self.cases:
            got = self.forward(math.radians(yd), math.radians(pd))
            if np.max(np.abs(got - np.asarray(exp))) > tol:
                bad.append(f"{self.name}: yaw {yd} pitch {pd} -> {np.round(got, 6).tolist()}, "
                           f"declared {list(exp)}")
        return bad

    def __repr__(self) -> str:
        return f"AxisConvention({self.name}, declared_by={self.declared_by!r})"


class FrameGraph:
    """A forest of named frames joined by rigid parent_T_child transforms."""

    def __init__(self, length_unit: Unit = M):
        if length_unit.dims != LENGTH:
            raise UnitError("frame graph length unit must be a length")
        self.length_unit = length_unit
        self._parent: Dict[str, Optional[str]] = {}
        self._T: Dict[str, SE3] = {}           # parent_T_child (identity for roots)
        self._convention: Dict[str, Optional[AxisConvention]] = {}
        self._meta: Dict[str, dict] = {}

    # ---------------------------------------------------------------- registry
    def add_frame(self, name: str, parent: Optional[str] = None,
                  parent_T_child: Optional[SE3] = None,
                  convention: Optional[AxisConvention] = None,
                  metadata: Optional[dict] = None) -> None:
        if not name or not isinstance(name, str):
            raise UnknownFrameError("frame name must be a non-empty string")
        if name in self._parent:
            raise FrameMismatchError(f"frame {name!r} already registered")
        if parent is not None:
            self._require(parent)
        elif parent_T_child is not None:
            raise FrameMismatchError(f"root frame {name!r} cannot have a parent transform")
        if parent_T_child is not None and not isinstance(parent_T_child, SE3):
            raise TypeError("parent_T_child must be an SE3")
        self._parent[name] = parent
        self._T[name] = SE3() if parent_T_child is None else parent_T_child
        self._convention[name] = convention
        self._meta[name] = dict(metadata or {})

    def set_transform(self, name: str, parent_T_child: SE3) -> None:
        """Update a moving frame (e.g. the body each step)."""
        self._require(name)
        if self._parent[name] is None:
            raise FrameMismatchError(f"{name!r} is a root; it has no parent transform")
        if not isinstance(parent_T_child, SE3):
            raise TypeError("parent_T_child must be an SE3")
        self._T[name] = parent_T_child

    def frames(self) -> List[str]:
        return list(self._parent)

    def convention(self, name: str) -> Optional[AxisConvention]:
        self._require(name)
        return self._convention[name]

    def metadata(self, name: str) -> dict:
        self._require(name)
        return dict(self._meta[name])

    def _require(self, name: str) -> None:
        if name not in self._parent:
            raise UnknownFrameError(
                f"unknown frame {name!r}; registered: {sorted(self._parent)}")

    # ---------------------------------------------------------------- paths
    def _root_T(self, name: str) -> Tuple[str, SE3]:
        """(root, root_T_name)."""
        self._require(name)
        T = SE3()
        cur = name
        seen = set()
        while self._parent[cur] is not None:
            if cur in seen:  # pragma: no cover - add_frame cannot create cycles
                raise FrameMismatchError("cycle in frame graph")
            seen.add(cur)
            T = self._T[cur] @ T
            cur = self._parent[cur]  # type: ignore
        return cur, T

    def transform(self, src: str, dst: str) -> SE3:
        """dst_T_src: maps coordinates expressed in `src` into `dst`."""
        r1, root_T_src = self._root_T(src)
        r2, root_T_dst = self._root_T(dst)
        if r1 != r2:
            raise DisconnectedFramesError(
                f"no transform path from {src!r} (tree {r1!r}) to {dst!r} "
                f"(tree {r2!r}); frames in different trees are unrelated")
        return root_T_dst.inverse() @ root_T_src

    def convert(self, x, src: str, dst: str, kind: str = "point"):
        """Re-express x (Quantity or raw (...,3) array) from `src` into `dst`.

        A Quantity must declare `frame == src`. Raw arrays are UNTYPED: they
        are taken to be in the graph's length unit, metric.
        """
        if kind not in ("point", "direction"):
            raise ValueError(f"kind must be 'point' or 'direction', got {kind!r}")
        T = self.transform(src, dst)
        if isinstance(x, Quantity):
            if x.frame != src:
                raise FrameMismatchError(
                    f"quantity is expressed in {x.frame!r}, not {src!r}")
            if x.value.ndim == 0 or x.value.shape[-1] != 3:
                raise DegenerateError("frame conversion needs (...,3) vectors")
            if x.std is not None and not np.allclose(
                    x.std, x.std[..., :1], rtol=0.0, atol=1e-15):
                raise DegenerateError(
                    "per-component std is only rotation-invariant when isotropic; "
                    "an anisotropic one needs a covariance, which this type does not carry")
            if kind == "direction":
                return x._replace(value=T.apply_direction(x.value), frame=dst)
            if x.unit.dims != LENGTH:
                raise UnitError(f"a point must have length dimension, got {x.unit}")
            if not x.is_metric and np.any(T.t != 0.0):
                raise ScaleStatusError(
                    f"cannot move a {x.scale_status} point across a translated "
                    f"frame edge ({src!r}->{dst!r}): metric offset + unknown scale")
            v = x.value * (x.unit.factor / self.length_unit.factor)
            out = T.apply_point(v) * (self.length_unit.factor / x.unit.factor)
            return x._replace(value=out, frame=dst)
        if kind == "direction":
            return T.apply_direction(x)
        return T.apply_point(x)
