"""Cl(3,0) geometric algebra — used ONLY for rotors acting on vectors.

SCOPE (plan Stage 4.4, §2.2). This module exists for one selected use: a
rotation representation whose composition and action are a single product,
checked against the quaternion path. It does not provide uncertainty, units,
causality or conservation, and nothing else in the library depends on it.

DECLARED ALGEBRA AND CONVENTIONS

  * Algebra: Cl(3,0), orthonormal basis e1, e2, e3 with e_i e_i = +1.
  * Storage: 8 coefficients indexed by blade BITMASK
        index: 0   1   2   3    4   5    6    7
        blade: 1   e1  e2  e12  e3  e13  e23  e123
    e_ij means the ordered product e_i e_j (i < j). The pseudoscalar is
    I = e123, with I*I = -1.
  * Named bivector basis (for humans): e23, e31, e12, where e31 = -e13.
    The dual map is I e1 = e23, I e2 = e31, I e3 = e12.
  * Rotation: rotating by angle theta about unit axis n (right-handed, same
    sense as transforms.quat_from_axis_angle) is the rotor
        R = exp(-I n theta / 2) = cos(theta/2) - sin(theta/2) (I n),
    acting by the SANDWICH  v' = R v ~R, where ~ is the reverse.
  * Quaternion correspondence (Hamilton (w, x, y, z)):
        R = w - x e23 - y e31 - z e12
          = w - x e23 + y e13 - z e12      (in stored blades)
    R2 R1 corresponds to quat_mul(q2, q1): apply R1 first.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .errors import DegenerateError
from .transforms import EPS_NORM

N_BLADES = 8
GRADE = np.array([bin(i).count("1") for i in range(N_BLADES)])
BLADE_NAMES = ("1", "e1", "e2", "e12", "e3", "e13", "e23", "e123")


def _reorder_sign(a: int, b: int) -> int:
    """Sign from sorting the basis vectors of blade a*b into canonical order."""
    a >>= 1
    s = 0
    while a:
        s += bin(a & b).count("1")
        a >>= 1
    return -1 if (s & 1) else 1


# (i, j, k, sign): blade_i * blade_j = sign * blade_k  (Euclidean metric)
PRODUCT_TABLE: List[Tuple[int, int, int, int]] = [
    (i, j, i ^ j, _reorder_sign(i, j)) for i in range(N_BLADES) for j in range(N_BLADES)]

_REV_SIGN = np.array([(-1) ** (g * (g - 1) // 2) for g in GRADE], dtype=np.float64)


def gp(a, b) -> np.ndarray:
    """Geometric product of multivectors (trailing dim 8)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[-1] != N_BLADES or b.shape[-1] != N_BLADES:
        raise DegenerateError("multivectors must have trailing dimension 8")
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape))
    for i, j, k, s in PRODUCT_TABLE:
        out[..., k] += s * a[..., i] * b[..., j]
    return out


def reverse(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) * _REV_SIGN


def grade(a, g: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a * (GRADE == g)


def vector(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] != 3:
        raise DegenerateError("vector must have trailing dimension 3")
    mv = np.zeros(v.shape[:-1] + (N_BLADES,))
    mv[..., 1], mv[..., 2], mv[..., 4] = v[..., 0], v[..., 1], v[..., 2]
    return mv


def to_vector(mv, tol: float = 1e-9) -> np.ndarray:
    mv = np.asarray(mv, dtype=np.float64)
    other = mv * (GRADE != 1)
    if np.max(np.abs(other), initial=0.0) > tol:
        raise DegenerateError("multivector has non-vector parts; not a vector")
    return np.stack([mv[..., 1], mv[..., 2], mv[..., 4]], axis=-1)


PSEUDOSCALAR = np.zeros(N_BLADES)
PSEUDOSCALAR[7] = 1.0


class Rotor:
    """An even-grade unit multivector R with R ~R = 1."""

    __slots__ = ("mv",)

    def __init__(self, mv, tol: float = 1e-9):
        m = np.asarray(mv, dtype=np.float64).reshape(N_BLADES)
        if not np.all(np.isfinite(m)):
            raise DegenerateError("rotor contains non-finite values")
        if np.max(np.abs(m * (GRADE % 2 == 1))) > tol:
            raise DegenerateError("a rotor lies in the even subalgebra (grades 0, 2)")
        n2 = float(gp(m, reverse(m))[0])
        if n2 < EPS_NORM ** 2:
            raise DegenerateError("zero-norm rotor")
        self.mv = m / math.sqrt(n2)

    @classmethod
    def identity(cls) -> "Rotor":
        m = np.zeros(N_BLADES)
        m[0] = 1.0
        return cls(m)

    @classmethod
    def from_axis_angle(cls, axis, angle: float) -> "Rotor":
        n = np.asarray(axis, dtype=np.float64)
        nn = float(np.linalg.norm(n))
        if nn < EPS_NORM:
            if float(angle) == 0.0:
                return cls.identity()
            raise DegenerateError("zero rotation axis with a non-zero angle")
        B = gp(PSEUDOSCALAR, vector(n / nn))          # I n, a unit bivector
        h = 0.5 * float(angle)
        m = -math.sin(h) * B
        m[0] += math.cos(h)
        return cls(m)

    @classmethod
    def from_quaternion(cls, q) -> "Rotor":
        w, x, y, z = np.asarray(q, dtype=np.float64)
        m = np.zeros(N_BLADES)
        m[0], m[6], m[5], m[3] = w, -x, +y, -z   # w - x e23 + y e13 - z e12
        return cls(m)

    def to_quaternion(self) -> np.ndarray:
        m = self.mv
        q = np.array([m[0], -m[6], m[5], -m[3]])
        return -q if q[0] < 0 else q

    def __mul__(self, other: "Rotor") -> "Rotor":
        """Composition: (R2 * R1) applies R1 first."""
        if not isinstance(other, Rotor):
            raise TypeError("Rotor * Rotor only; use .apply(v) for vectors")
        return Rotor(gp(self.mv, other.mv))

    def inverse(self) -> "Rotor":
        return Rotor(reverse(self.mv))

    def apply(self, v) -> np.ndarray:
        """Sandwich R v ~R on (...,3) vectors."""
        out = gp(gp(self.mv, vector(v)), reverse(self.mv))
        return to_vector(out)

    def __repr__(self) -> str:
        parts = [f"{c:+.6f}{n}" for c, n in zip(self.mv, BLADE_NAMES) if abs(c) > 1e-12]
        return "Rotor(" + " ".join(parts) + ")"
