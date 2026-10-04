"""Rigid transforms: unit quaternions, SE(3) and SE(2), NumPy float64 reference.

DECLARED CONVENTIONS (every function here obeys these; nothing else does):

  * Quaternions are HAMILTON, stored as (w, x, y, z), scalar first.
  * Rotations are ACTIVE and RIGHT-HANDED: `quat_rotate(q, v) = q v q*`, and
    `quat_from_axis_angle(n, theta)` turns vectors counter-clockwise by
    theta when viewed looking DOWN the axis n towards the origin.
  * `quat_mul(a, b)` applies b FIRST, then a (like matrices: R_a R_b).
  * q and -q are the same rotation; `slerp` takes the shorter arc.
  * `SE3(q, t)` written a_T_b maps a POINT expressed in frame b to frame a:
    p_a = R(q) p_b + t. DIRECTIONS (velocities, normals, gravity) ignore t.
    Composition: (a_T_b @ b_T_c) == a_T_c.
  * Translation is in the owning FrameGraph's length unit (metres by default).

DEGENERACIES are errors, not silent fixes: a quaternion whose norm is below
`EPS_NORM` is rejected (normalising it would invent a rotation out of
rounding noise); a zero axis with a non-zero angle is rejected; any
non-finite input is rejected. 180-degree rotations (w = 0) and pitch = +-90
degrees are ordinary cases here — the quaternion path has no gimbal lock.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .errors import DegenerateError

EPS_NORM = 1e-12


def _arr(x, last: Optional[int] = None, what: str = "array") -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if last is not None and (a.ndim == 0 or a.shape[-1] != last):
        raise DegenerateError(f"{what} must have trailing dimension {last}, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise DegenerateError(f"{what} contains non-finite values")
    return a


# ----------------------------------------------------------------- quaternion

def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q) -> np.ndarray:
    q = _arr(q, 4, "quaternion")
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < EPS_NORM):
        raise DegenerateError(
            f"quaternion norm {float(n.min()):.3g} < {EPS_NORM}; a zero quaternion "
            f"is not a rotation and normalising it would invent one")
    return q / n


def quat_check_unit(q, tol: float = 1e-9) -> np.ndarray:
    q = _arr(q, 4, "quaternion")
    n = np.linalg.norm(q, axis=-1)
    if np.any(np.abs(n - 1.0) > tol):
        raise DegenerateError(f"quaternion is not unit (|q|={n})")
    return q


def quat_mul(a, b) -> np.ndarray:
    a = _arr(a, 4, "quaternion")
    b = _arr(b, 4, "quaternion")
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conj(q) -> np.ndarray:
    q = _arr(q, 4, "quaternion")
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_inv(q) -> np.ndarray:
    q = _arr(q, 4, "quaternion")
    n2 = np.sum(q * q, axis=-1, keepdims=True)
    if np.any(n2 < EPS_NORM ** 2):
        raise DegenerateError("cannot invert a zero quaternion")
    return quat_conj(q) / n2


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector(s) v by unit quaternion(s) q: q v q*."""
    q = _arr(q, 4, "quaternion")
    v = _arr(v, 3, "vector")
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = _arr(axis, 3, "axis")
    ang = float(angle)
    if not math.isfinite(ang):
        raise DegenerateError("angle must be finite")
    n = float(np.linalg.norm(axis))
    if n < EPS_NORM:
        if ang == 0.0:
            return quat_identity()
        raise DegenerateError("zero rotation axis with a non-zero angle")
    h = 0.5 * ang
    return np.concatenate([[math.cos(h)], math.sin(h) * axis / n])


def quat_angle(q) -> float:
    """Rotation angle in [0, pi], numerically stable near 0 and pi."""
    q = quat_normalize(q)
    return float(2.0 * math.atan2(float(np.linalg.norm(q[1:])), abs(float(q[0]))))


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> np.ndarray:
    """Shepperd's method: picks the largest diagonal term, so it is exact at
    180 degrees where the trace form divides by ~0. Returns w >= 0."""
    R = _arr(R, 3, "rotation matrix")
    if R.shape != (3, 3):
        raise DegenerateError(f"rotation matrix must be 3x3, got {R.shape}")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-8) or np.linalg.det(R) < 0:
        raise DegenerateError("matrix is not a proper rotation")
    tr = float(np.trace(R))
    cands = [tr, R[0, 0], R[1, 1], R[2, 2]]
    i = int(np.argmax(cands))
    if i == 0:
        s = math.sqrt(1.0 + tr) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = quat_normalize(np.array(q))
    return -q if q[0] < 0 else q


def quat_equal(a, b, tol: float = 1e-9) -> bool:
    """Same ROTATION (q ~ -q)."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    return bool(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))) <= tol)


def slerp(q0, q1, s: float) -> np.ndarray:
    """Spherical interpolation along the shorter arc, s in [0, 1].

    Near identity (dot -> 1) the sin(Omega) denominator vanishes; there the
    normalised linear form is used, whose error is O(Omega^3) — below 1e-18
    at the switch-over, so continuity is not visibly broken.
    """
    s = float(s)
    if not (0.0 <= s <= 1.0):
        raise DegenerateError(f"slerp parameter must be in [0, 1], got {s}")
    a = quat_normalize(q0)
    b = quat_normalize(q1)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 1.0 - 1e-12:
        return quat_normalize((1.0 - s) * a + s * b)
    om = math.acos(min(1.0, d))
    so = math.sin(om)
    return (math.sin((1.0 - s) * om) / so) * a + (math.sin(s * om) / so) * b


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform on SO(3) (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    q = np.array([a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2),
                  b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3)])
    return q if q[0] >= 0 else -q


# ----------------------------------------------------------------- SE(3)

class SE3:
    """A rigid transform a_T_b: p_a = R p_b + t (see module conventions)."""

    __slots__ = ("q", "t")

    def __init__(self, q=None, t=None):
        self.q = quat_identity() if q is None else quat_normalize(q)
        if self.q.shape != (4,):
            raise DegenerateError("SE3 holds a single quaternion")
        self.t = np.zeros(3) if t is None else _arr(t, 3, "translation").reshape(3).copy()

    @classmethod
    def identity(cls) -> "SE3":
        return cls()

    @classmethod
    def from_matrix(cls, T) -> "SE3":
        T = _arr(T, 4, "homogeneous matrix")
        if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
            raise DegenerateError("not a 4x4 homogeneous rigid transform")
        return cls(matrix_to_quat(T[:3, :3]), T[:3, 3])

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 10.0) -> "SE3":
        return cls(random_quat(rng), rng.normal(size=3) * scale)

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(self.q)
        T[:3, 3] = self.t
        return T

    def __matmul__(self, other: "SE3") -> "SE3":
        if not isinstance(other, SE3):
            raise TypeError("SE3 @ SE3 only; use apply_point / apply_direction for vectors")
        return SE3(quat_mul(self.q, other.q), quat_rotate(self.q, other.t) + self.t)

    def inverse(self) -> "SE3":
        qi = quat_conj(self.q)
        return SE3(qi, -quat_rotate(qi, self.t))

    def apply_point(self, p) -> np.ndarray:
        return quat_rotate(self.q, _arr(p, 3, "point")) + self.t

    def apply_direction(self, d) -> np.ndarray:
        return quat_rotate(self.q, _arr(d, 3, "direction"))

    def interpolate(self, other: "SE3", s: float) -> "SE3":
        """Slerp on rotation, linear on translation (declared; not a screw
        motion — the path of the origin is a straight line)."""
        return SE3(slerp(self.q, other.q, s), (1.0 - s) * self.t + s * other.t)

    def is_identity(self, tol: float = 1e-12) -> bool:
        return quat_equal(self.q, quat_identity(), tol) and bool(np.all(np.abs(self.t) <= tol))

    def almost_equal(self, other: "SE3", tol: float = 1e-9) -> bool:
        return quat_equal(self.q, other.q, tol) and bool(np.max(np.abs(self.t - other.t)) <= tol)

    def __repr__(self) -> str:
        return f"SE3(q={np.round(self.q, 6).tolist()}, t={np.round(self.t, 6).tolist()})"


# ----------------------------------------------------------------- SE(2)

def _wrap(a: float) -> float:
    """Wrap to (-pi, pi]."""
    w = math.remainder(float(a), 2.0 * math.pi)
    return math.pi if w == -math.pi else w


class SE2:
    """Planar rigid transform: p_a = R(theta) p_b + t, theta CCW-positive in
    the (first, second) axis plane of whatever 2-D chart the caller declares."""

    __slots__ = ("theta", "t")

    def __init__(self, theta: float = 0.0, t=None):
        th = float(theta)
        if not math.isfinite(th):
            raise DegenerateError("SE2 angle must be finite")
        self.theta = _wrap(th)
        self.t = np.zeros(2) if t is None else _arr(t, 2, "translation").reshape(2).copy()

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 10.0) -> "SE2":
        return cls(rng.uniform(-math.pi, math.pi), rng.normal(size=2) * scale)

    def _R(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def as_matrix(self) -> np.ndarray:
        T = np.eye(3)
        T[:2, :2] = self._R()
        T[:2, 2] = self.t
        return T

    def __matmul__(self, other: "SE2") -> "SE2":
        if not isinstance(other, SE2):
            raise TypeError("SE2 @ SE2 only")
        return SE2(self.theta + other.theta, self._R() @ other.t + self.t)

    def inverse(self) -> "SE2":
        Rt = self._R().T
        return SE2(-self.theta, -Rt @ self.t)

    def apply_point(self, p) -> np.ndarray:
        return _arr(p, 2, "point") @ self._R().T + self.t

    def apply_direction(self, d) -> np.ndarray:
        return _arr(d, 2, "direction") @ self._R().T

    def interpolate(self, other: "SE2", s: float) -> "SE2":
        s = float(s)
        if not (0.0 <= s <= 1.0):
            raise DegenerateError("interpolation parameter must be in [0, 1]")
        d = _wrap(other.theta - self.theta)
        return SE2(self.theta + s * d, (1.0 - s) * self.t + s * other.t)

    def to_se3(self) -> SE3:
        """Embed as a rotation about +z of the (x, y) plane."""
        return SE3(quat_from_axis_angle([0.0, 0.0, 1.0], self.theta),
                   np.array([self.t[0], self.t[1], 0.0]))

    def almost_equal(self, other: "SE2", tol: float = 1e-9) -> bool:
        return (abs(_wrap(self.theta - other.theta)) <= tol
                and bool(np.max(np.abs(self.t - other.t)) <= tol))

    def __repr__(self) -> str:
        return f"SE2(theta={self.theta:.6f}, t={np.round(self.t, 6).tolist()})"
