"""Dimensioned quantities with frames and an explicit SCALE STATUS.

WHAT IS CLAIMED. A `Quantity` carries, besides its number:

  unit          a `Unit` = base-dimension vector (length, time, mass, angle)
                plus a factor to the base unit (m, s, kg, rad). Angle is kept
                as its own pseudo-dimension, deliberately unlike SI, because
                "added a yaw to a distance" is a bug this repo can make.
  frame         the coordinate (or clock) frame the value is expressed in, or
                None for frame-free scalars. Add/sub requires equal frames.
  scale_status  "metric"      — the number is in true units.
                "up_to_scale" — true value = s**k * value for an UNKNOWN
                                positive s shared by everything carrying the
                                same `scale_ref` (one monocular
                                reconstruction). k = `scale_exponent`.
                "unknown"     — we do not know whether it is metric at all.
  std           optional 1-sigma uncertainty in the same unit (propagated to
                first order under a DECLARED independence assumption).
  provenance    tuple of strings recording how the value was obtained.

WHY. Plan Stage 4.3: "monocular estimates must not silently become metric
ground truth." The failure mode is not a crash — it is a depth-from-motion
estimate in arbitrary units being added to a dead-reckoned position in blocks
and the sum used as if it were metres. So:

  * metric + up_to_scale raises; up_to_scale values from DIFFERENT
    reconstructions (scale_refs) never combine additively.
  * multiplying/dividing tracks the scale exponent, so a RATIO of two lengths
    from the same reconstruction is genuinely scale-free and becomes metric
    (exponent 0) — the one honest way out without calibration.
  * the only other way to metric is `calibrate(scale, evidence)` (or
    `declare_metric(evidence)` for "unknown"), which REQUIRE evidence and
    record it in provenance.
  * `float(q)` refuses non-metric values; `np.asarray(q)` refuses always
    (it would strip the unit); use `.metric_value(unit)` or `.value`.

ADAPTER UNITS. Only m, s, kg, rad and dimensionless exist by default. A unit
like Minecraft's "block" is created by `declare_unit("block", M, 1.0,
declared_by=...)` — "1 block = 1 m" is an adapter's claim, not physics, and
it is recorded as such on the Unit.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .errors import DegenerateError, FrameMismatchError, ScaleStatusError, UnitError

Dims = Tuple[int, int, int, int]  # (length, time, mass, angle)
DIM_NAMES = ("m", "s", "kg", "rad")

DIMENSIONLESS_DIMS: Dims = (0, 0, 0, 0)
LENGTH: Dims = (1, 0, 0, 0)
TIME: Dims = (0, 1, 0, 0)
MASS: Dims = (0, 0, 1, 0)
ANGLE: Dims = (0, 0, 0, 1)

SCALE_STATUSES = ("metric", "up_to_scale", "unknown")


def _dims_add(a: Dims, b: Dims, sign: int = 1) -> Dims:
    return tuple(int(x) + sign * int(y) for x, y in zip(a, b))  # type: ignore


class Unit:
    """A named unit: dimension vector + factor to the base unit."""

    __slots__ = ("name", "dims", "factor", "declared_by")

    def __init__(self, name: str, dims: Sequence[int], factor: float = 1.0,
                 declared_by: str = "core"):
        d = tuple(int(x) for x in dims)
        if len(d) != 4:
            raise UnitError(f"unit dims must have 4 entries {DIM_NAMES}, got {d}")
        f = float(factor)
        if not (f > 0.0 and math.isfinite(f)):
            raise UnitError(f"unit factor must be finite and > 0, got {factor}")
        self.name = name
        self.dims: Dims = d  # type: ignore
        self.factor = f
        self.declared_by = declared_by

    def compatible(self, other: "Unit") -> bool:
        return self.dims == other.dims

    def __mul__(self, other: "Unit") -> "Unit":
        return Unit(f"{self.name}*{other.name}", _dims_add(self.dims, other.dims),
                    self.factor * other.factor, "derived")

    def __truediv__(self, other: "Unit") -> "Unit":
        return Unit(f"{self.name}/{other.name}",
                    _dims_add(self.dims, other.dims, -1),
                    self.factor / other.factor, "derived")

    def __pow__(self, k: int) -> "Unit":
        k = int(k)
        return Unit(f"{self.name}^{k}", tuple(x * k for x in self.dims),
                    self.factor ** k, "derived")

    def __eq__(self, other) -> bool:
        return (isinstance(other, Unit) and self.dims == other.dims
                and abs(self.factor - other.factor) <= 1e-15 * max(1.0, self.factor))

    def __hash__(self):
        return hash((self.dims, round(self.factor, 12)))

    def __repr__(self) -> str:
        return f"Unit({self.name})"


DIMENSIONLESS = Unit("1", DIMENSIONLESS_DIMS)
M = Unit("m", LENGTH)
S = Unit("s", TIME)
KG = Unit("kg", MASS)
RAD = Unit("rad", ANGLE)


def declare_unit(name: str, base: Unit, factor: float, declared_by: str) -> Unit:
    """An adapter-declared unit: `1 name = factor * base`.

    `declared_by` is mandatory and non-empty — the declaration is a claim
    somebody made, and the Unit remembers who.
    """
    if not declared_by or not str(declared_by).strip():
        raise UnitError(f"declaring unit {name!r} requires a declared_by source")
    return Unit(name, base.dims, base.factor * float(factor), str(declared_by))


def _as_array(value) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise DegenerateError("quantity value must be finite")
    return a


class Quantity:
    """A value with unit, frame, scale status, optional std and provenance."""

    __slots__ = ("value", "unit", "frame", "scale_status", "scale_ref",
                 "scale_exponent", "std", "provenance")
    # Make numpy defer to our reflected operators instead of broadcasting
    # into an object array that silently drops the unit.
    __array_ufunc__ = None

    def __init__(self, value, unit: Unit, frame: Optional[str] = None,
                 scale_status: str = "metric", scale_ref: Optional[str] = None,
                 scale_exponent: Optional[int] = None, std=None,
                 provenance: Sequence[str] = ()):
        if not isinstance(unit, Unit):
            raise UnitError(f"unit must be a Unit, got {type(unit).__name__}")
        if scale_status not in SCALE_STATUSES:
            raise ScaleStatusError(
                f"scale_status must be one of {SCALE_STATUSES}, got {scale_status!r}")
        if scale_status == "metric":
            if scale_ref is not None or scale_exponent not in (None, 0):
                raise ScaleStatusError("a metric quantity has no scale_ref/exponent")
            k = 0
        elif scale_status == "up_to_scale":
            if not scale_ref:
                raise ScaleStatusError(
                    "an up_to_scale quantity needs a scale_ref naming the "
                    "reconstruction whose unknown scale it shares")
            k = unit.dims[0] if scale_exponent is None else int(scale_exponent)
            if k == 0:
                raise ScaleStatusError(
                    "scale exponent 0 means the value does not depend on the "
                    "unknown scale; construct it as metric instead")
        else:
            k = 0 if scale_exponent is None else int(scale_exponent)
        self.value = _as_array(value)
        self.unit = unit
        self.frame = frame
        self.scale_status = scale_status
        self.scale_ref = scale_ref
        self.scale_exponent = k
        if std is not None:
            sd = _as_array(std)
            if np.any(sd < 0):
                raise DegenerateError("std must be non-negative")
            std = np.broadcast_to(sd, self.value.shape).copy()
        self.std = std
        self.provenance = tuple(provenance)

    # ------------------------------------------------------------ inspection
    @property
    def is_metric(self) -> bool:
        return self.scale_status == "metric"

    @property
    def shape(self):
        return self.value.shape

    def _replace(self, **kw) -> "Quantity":
        d = dict(value=self.value, unit=self.unit, frame=self.frame,
                 scale_status=self.scale_status, scale_ref=self.scale_ref,
                 scale_exponent=self.scale_exponent, std=self.std,
                 provenance=self.provenance)
        d.update(kw)
        if d["scale_status"] == "metric":
            d["scale_ref"], d["scale_exponent"] = None, None
        return Quantity(**d)

    def to(self, unit: Unit) -> "Quantity":
        """Same quantity in another unit of the same dimension (any status)."""
        if unit.dims != self.unit.dims:
            raise UnitError(f"cannot convert {self.unit} to {unit}: dimensions "
                            f"{self.unit.dims} vs {unit.dims}")
        r = self.unit.factor / unit.factor
        return self._replace(value=self.value * r, unit=unit,
                             std=None if self.std is None else self.std * r)

    def metric_value(self, unit: Optional[Unit] = None) -> np.ndarray:
        """The number in `unit` — ONLY for metric quantities."""
        if not self.is_metric:
            raise ScaleStatusError(
                f"{self.scale_status} quantity (scale_ref={self.scale_ref!r}) "
                f"has no metric value; calibrate it with evidence first")
        return (self if unit is None else self.to(unit)).value

    # ------------------------------------------------------------ scale
    def calibrate(self, scale: float, evidence: Sequence[str],
                  scale_std: float = 0.0) -> "Quantity":
        """up_to_scale -> metric, using `scale` = true units per estimate unit.

        Requires at least one evidence string. Provenance records the
        reconstruction, the scale and the evidence.
        """
        if self.scale_status != "up_to_scale":
            raise ScaleStatusError(
                f"calibrate() applies to up_to_scale quantities, not "
                f"{self.scale_status}")
        ev = tuple(str(e) for e in evidence)
        if not ev or not all(e.strip() for e in ev):
            raise ScaleStatusError("calibration requires non-empty evidence")
        s = float(scale)
        if not (s > 0.0 and math.isfinite(s)):
            raise DegenerateError(f"calibration scale must be finite and > 0, got {scale}")
        k = self.scale_exponent
        val = self.value * s ** k
        std = None
        if self.std is not None or scale_std:
            rel_s = float(scale_std) / s
            base = (np.zeros_like(self.value) if self.std is None
                    else self.std * s ** k)
            std = np.sqrt(base ** 2 + (np.abs(val) * abs(k) * rel_s) ** 2)
        prov = self.provenance + (
            f"calibrated:{self.scale_ref}:scale={s!r}^{k}",) + tuple(
            f"evidence:{e}" for e in ev)
        return self._replace(value=val, scale_status="metric", std=std,
                             provenance=prov)

    def declare_metric(self, evidence: Sequence[str]) -> "Quantity":
        """unknown -> metric, on stated evidence (e.g. adapter documentation)."""
        if self.scale_status != "unknown":
            raise ScaleStatusError(
                f"declare_metric() applies to unknown-scale quantities, not "
                f"{self.scale_status}")
        ev = tuple(str(e) for e in evidence)
        if not ev or not all(e.strip() for e in ev):
            raise ScaleStatusError("declare_metric requires non-empty evidence")
        return self._replace(scale_status="metric", provenance=self.provenance
                             + ("declared_metric",) + tuple(f"evidence:{e}" for e in ev))

    # ------------------------------------------------------------ checks
    def _check_additive(self, other: "Quantity", op: str) -> None:
        if not isinstance(other, Quantity):
            raise UnitError(
                f"cannot {op} a bare {type(other).__name__} and a Quantity; "
                f"wrap it with a unit")
        if other.unit.dims != self.unit.dims:
            raise UnitError(f"cannot {op} {self.unit} and {other.unit}: "
                            f"dimensions {self.unit.dims} vs {other.unit.dims}")
        if other.frame != self.frame:
            raise FrameMismatchError(
                f"cannot {op} values in frame {self.frame!r} and "
                f"{other.frame!r}; convert one first")
        if other.scale_status != self.scale_status:
            raise ScaleStatusError(
                f"cannot {op} a {self.scale_status} and a {other.scale_status} "
                f"quantity; an estimate of unknown scale must be calibrated "
                f"before it can meet a metric value")
        if self.scale_status != "metric":
            if self.scale_ref is None or other.scale_ref != self.scale_ref:
                raise ScaleStatusError(
                    f"cannot {op} {self.scale_status} values from different "
                    f"scale references ({self.scale_ref!r} vs "
                    f"{other.scale_ref!r}); their unknown scales are unrelated")
            if other.scale_exponent != self.scale_exponent:
                raise ScaleStatusError("scale exponents differ")

    def _other_in_my_unit(self, other: "Quantity"):
        r = other.unit.factor / self.unit.factor
        sd = None if other.std is None else other.std * r
        return other.value * r, sd

    @staticmethod
    def _quad(a, b, shape):
        if a is None and b is None:
            return None
        a = 0.0 if a is None else a
        b = 0.0 if b is None else b
        return np.broadcast_to(np.sqrt(np.square(a) + np.square(b)), shape)

    # ------------------------------------------------------------ arithmetic
    def __add__(self, other):
        self._check_additive(other, "add")
        ov, osd = self._other_in_my_unit(other)
        v = self.value + ov
        return self._replace(value=v, std=self._quad(self.std, osd, v.shape),
                             provenance=())

    def __sub__(self, other):
        self._check_additive(other, "subtract")
        ov, osd = self._other_in_my_unit(other)
        v = self.value - ov
        return self._replace(value=v, std=self._quad(self.std, osd, v.shape),
                             provenance=())

    def __neg__(self):
        return self._replace(value=-self.value)

    def _combine_scale(self, other: "Quantity", sign: int):
        a, b = self, other
        if a.scale_status == "unknown" or b.scale_status == "unknown":
            ref = a.scale_ref if a.scale_ref == b.scale_ref else None
            return dict(scale_status="unknown", scale_ref=ref, scale_exponent=0)
        if a.is_metric and b.is_metric:
            return dict(scale_status="metric")
        if a.is_metric:
            return dict(scale_status="up_to_scale", scale_ref=b.scale_ref,
                        scale_exponent=sign * b.scale_exponent)
        if b.is_metric:
            return dict(scale_status="up_to_scale", scale_ref=a.scale_ref,
                        scale_exponent=a.scale_exponent)
        if a.scale_ref != b.scale_ref:
            raise ScaleStatusError(
                f"product of values from two different unknown scales "
                f"({a.scale_ref!r}, {b.scale_ref!r}) is not representable")
        k = a.scale_exponent + sign * b.scale_exponent
        if k == 0:
            return dict(scale_status="metric")
        return dict(scale_status="up_to_scale", scale_ref=a.scale_ref,
                    scale_exponent=k)

    def _combine_frame(self, other: "Quantity"):
        if self.frame is not None and other.frame is not None and self.frame != other.frame:
            raise FrameMismatchError(
                f"cannot multiply values in frames {self.frame!r} and {other.frame!r}")
        return self.frame if self.frame is not None else other.frame

    def __mul__(self, other):
        if isinstance(other, Quantity):
            sc = self._combine_scale(other, +1)
            v = self.value * other.value
            std = None
            if self.std is not None or other.std is not None:
                sa = 0.0 if self.std is None else self.std
                sb = 0.0 if other.std is None else other.std
                std = np.sqrt((sa * other.value) ** 2 + (sb * self.value) ** 2)
            return Quantity(v, self.unit * other.unit,
                            frame=self._combine_frame(other), std=std, **sc)
        if isinstance(other, (int, float, np.ndarray, np.floating, np.integer)):
            o = _as_array(other)
            return self._replace(value=self.value * o,
                                 std=None if self.std is None else self.std * np.abs(o),
                                 provenance=())
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            if np.any(other.value == 0):
                raise DegenerateError("division by a zero quantity")
            sc = self._combine_scale(other, -1)
            v = self.value / other.value
            std = None
            if self.std is not None or other.std is not None:
                sa = 0.0 if self.std is None else self.std
                sb = 0.0 if other.std is None else other.std
                std = np.sqrt((sa / other.value) ** 2
                              + (sb * self.value / other.value ** 2) ** 2)
            return Quantity(v, self.unit / other.unit,
                            frame=self._combine_frame(other), std=std, **sc)
        if isinstance(other, (int, float, np.ndarray, np.floating, np.integer)):
            o = _as_array(other)
            if np.any(o == 0):
                raise DegenerateError("division by zero")
            return self._replace(value=self.value / o,
                                 std=None if self.std is None else self.std / np.abs(o),
                                 provenance=())
        return NotImplemented

    # ------------------------------------------------------------ comparison
    def _cmp(self, other, op):
        self._check_additive(other, "compare")
        ov, _ = self._other_in_my_unit(other)
        return op(self.value, ov)

    def __lt__(self, o):
        return self._cmp(o, np.less)

    def __le__(self, o):
        return self._cmp(o, np.less_equal)

    def __gt__(self, o):
        return self._cmp(o, np.greater)

    def __ge__(self, o):
        return self._cmp(o, np.greater_equal)

    def allclose(self, other: "Quantity", atol: float = 1e-9) -> bool:
        self._check_additive(other, "compare")
        ov, _ = self._other_in_my_unit(other)
        return bool(np.allclose(self.value, ov, rtol=0.0, atol=atol))

    # ------------------------------------------------------------ escapes
    def __float__(self):
        if not self.is_metric:
            raise ScaleStatusError(
                f"float() of a {self.scale_status} quantity would discard its "
                f"scale status; calibrate it or use .value explicitly")
        if self.value.size != 1:
            raise TypeError("float() of a non-scalar quantity")
        return float(self.value.reshape(()))

    def __array__(self, *args, **kwargs):
        raise UnitError(
            "implicit array conversion would drop unit/frame/scale status; "
            "use .metric_value(unit) or .value")

    def __bool__(self):
        raise TypeError("a Quantity has no truth value; compare it explicitly")

    def __repr__(self) -> str:
        extra = "" if self.frame is None else f" @{self.frame}"
        if self.scale_status != "metric":
            extra += f" [{self.scale_status}:{self.scale_ref}^{self.scale_exponent}]"
        v = self.value.tolist()
        return f"Quantity({v} {self.unit.name}{extra})"
