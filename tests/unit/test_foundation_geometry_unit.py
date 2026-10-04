"""Unit cases for foundation.geometry (plan Stage 4) — the PARTS.

tests/_foundation_geometry_smoke.py argues the design claims (covariance,
coordinate change vs physical rotation, rotor/quaternion parity, gradients).
This file checks each piece in isolation: constructor validation, the right
error CLASS for each invalid operation, and the small algebraic facts the
contracts build on. Each `raises` names the specific error type — a check
that "something raised" would pass on a typo.

Run: PYTHONPATH=. python tests/unit/test_foundation_geometry_unit.py
"""
import math
import sys

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, run_all, close, raises
from developmental_ai.foundation.geometry import (
    INAPPLICABLE, KG, M, RAD, S, SE2, SE3, UNKNOWN, ClockMismatchError,
    Constraint, ConstraintError, ConstraintRegistry, DegenerateError,
    DisconnectedFramesError, DiscreteVar, DomainError, FrameGraph,
    FrameMismatchError, Quantity, RelationType, Rotor, ScaleStatusError,
    Timestamp, UnitError, UnknownFrameError, declare_unit, matrix_to_quat,
    quat_angle, quat_conj, quat_equal, quat_from_axis_angle, quat_identity,
    quat_inv, quat_mul, quat_normalize, quat_rotate, quat_to_matrix, slerp)
from developmental_ai.foundation.contracts.sentinels import UNKNOWN as C_UNKNOWN

BLOCK = declare_unit("block", M, 1.0, declared_by="unit-test adapter")


# ---------------------------------------------------------------- units

@case
def unit_algebra_and_declaration():
    v = M / S
    assert v.dims == (1, -1, 0, 0)
    assert (M * M).dims == (2, 0, 0, 0)
    assert (M ** 3).dims == (3, 0, 0, 0)
    raises(lambda: declare_unit("block", M, 1.0, declared_by=""), UnitError,
           "undeclared adapter unit")
    assert BLOCK.declared_by == "unit-test adapter" and BLOCK == M


@case
def quantity_add_converts_units_and_rejects_dims():
    a = Quantity(2.0, M)
    b = Quantity(300.0, declare_unit("cm", M, 0.01, "si"))
    close(float(a + b), 5.0, 1e-12, "2 m + 300 cm")
    raises(lambda: a + Quantity(1.0, S), UnitError, "m + s")
    raises(lambda: a + Quantity(1.0, RAD), UnitError, "m + rad (angle is a dimension here)")
    raises(lambda: a + 1.0, UnitError, "bare float + quantity")
    close(float(Quantity(3.0, M) * Quantity(2.0, KG) / Quantity(1.0, S) / Quantity(6.0, KG * M / S)),
          1.0, 1e-12, "dimensionless round trip")


@case
def quantity_frame_mismatch():
    a = Quantity([1.0, 0, 0], M, frame="world")
    raises(lambda: a + Quantity([1.0, 0, 0], M, frame="body"), FrameMismatchError, "frames")
    raises(lambda: a + Quantity([1.0, 0, 0], M), FrameMismatchError, "framed + frameless")


@case
def scale_status_rules():
    up = Quantity(2.0, M, scale_status="up_to_scale", scale_ref="recon1")
    met = Quantity(2.0, M)
    raises(lambda: up + met, ScaleStatusError, "up_to_scale + metric")
    raises(lambda: up + Quantity(1.0, M, scale_status="up_to_scale", scale_ref="recon2"),
           ScaleStatusError, "different reconstructions")
    raises(lambda: float(up), ScaleStatusError, "float() of up_to_scale")
    raises(lambda: up.metric_value(), ScaleStatusError, "metric_value of up_to_scale")
    raises(lambda: np.asarray(up), UnitError, "implicit array conversion")
    raises(lambda: Quantity(1.0, M, scale_status="up_to_scale"), ScaleStatusError, "no scale_ref")
    raises(lambda: Quantity(1.0, M, scale_status="metricish"), ScaleStatusError, "bad status")
    s = up + Quantity(1.0, M, scale_status="up_to_scale", scale_ref="recon1")
    assert s.scale_status == "up_to_scale" and s.scale_ref == "recon1"
    area = up * up
    assert area.scale_exponent == 2
    ratio = up / Quantity(4.0, M, scale_status="up_to_scale", scale_ref="recon1")
    assert ratio.is_metric, "same-reconstruction ratio is scale-free"
    close(float(ratio), 0.5, 1e-12)
    raises(lambda: up * Quantity(1.0, M, scale_status="up_to_scale", scale_ref="r2"),
           ScaleStatusError, "product across reconstructions")


@case
def calibrate_requires_evidence_and_records_it():
    up = Quantity([1.0, 2.0], M, scale_status="up_to_scale", scale_ref="recon1")
    raises(lambda: up.calibrate(2.0, []), ScaleStatusError, "no evidence")
    raises(lambda: up.calibrate(2.0, ["  "]), ScaleStatusError, "blank evidence")
    raises(lambda: up.calibrate(0.0, ["x"]), DegenerateError, "zero scale")
    met = up.calibrate(2.0, ["block edge = 1 m in frame 17"])
    assert met.is_metric and np.allclose(met.value, [2.0, 4.0])
    assert any("evidence:block edge" in p for p in met.provenance)
    assert any(p.startswith("calibrated:recon1") for p in met.provenance)
    raises(lambda: met.calibrate(2.0, ["x"]), ScaleStatusError, "calibrate metric twice")
    area = (up * up).calibrate(3.0, ["x"])
    assert np.allclose(area.value, [9.0, 36.0]), "exponent-2 quantity scales by s^2"


@case
def unknown_scale_status():
    u = Quantity(1.0, M, scale_status="unknown")
    raises(lambda: u + Quantity(1.0, M, scale_status="unknown"), ScaleStatusError,
           "unknown + unknown with no shared ref")
    raises(lambda: u.calibrate(2.0, ["x"]), ScaleStatusError, "calibrate unknown")
    m = u.declare_metric(["adapter documents metres"])
    assert m.is_metric


@case
def std_propagation():
    a = Quantity(1.0, M, std=0.3)
    b = Quantity(2.0, M, std=0.4)
    close(float((a + b).std), 0.5, 1e-12, "quadrature")
    close(float((a * 2.0).std), 0.6, 1e-12, "scalar")
    raises(lambda: Quantity(1.0, M, std=-1.0), DegenerateError, "negative std")


# ---------------------------------------------------------------- values

@case
def sentinels_are_canonical_and_refuse_truthiness():
    assert UNKNOWN is C_UNKNOWN
    assert UNKNOWN is not INAPPLICABLE
    raises(lambda: bool(UNKNOWN), TypeError, "bool(UNKNOWN)")
    raises(lambda: bool(INAPPLICABLE), TypeError, "bool(INAPPLICABLE)")


@case
def discrete_var():
    v = DiscreteVar("block", ["oak_log", "birch_log", "stone"])
    assert not v.is_known and v.support == {"oak_log", "birch_log", "stone"}
    r = v.restrict(["oak_log", "birch_log"])
    assert not r.is_known and len(r.support) == 2
    k = r.restrict(["oak_log"])
    assert k.is_known and k.value == "oak_log"
    raises(lambda: r.restrict(["stone"]), DomainError, "contradiction")
    raises(lambda: DiscreteVar("x", ["a"], "b"), DomainError, "out of domain")
    raises(lambda: DiscreteVar("x", []), DomainError, "empty domain")
    raises(lambda: DiscreteVar("x", ["a", UNKNOWN]), DomainError, "sentinel in domain")


@case
def relations():
    on = RelationType("on_top_of", 2, (str, str))
    r = on("log", "dirt")
    assert r.truth is True and r.key == ("on_top_of", ("log", "dirt"))
    u = on("log", "stone", truth=UNKNOWN)
    assert u.truth is UNKNOWN
    raises(lambda: on("log"), DomainError, "arity")
    raises(lambda: on("log", 3), DomainError, "arg type")
    raises(lambda: on("a", "b", truth=1), DomainError, "truth must be bool/UNKNOWN")


@case
def timestamps():
    a = Timestamp(10.0, "env", epoch=3)
    b = Timestamp(12.5, "env", epoch=3)
    d = b - a
    close(float(d), 2.5, 1e-12)
    assert d.frame == "clock:env/3"
    assert (a + d) == b
    raises(lambda: b - Timestamp(1.0, "wall"), ClockMismatchError, "env vs wall")
    raises(lambda: b < Timestamp(1.0, "env", epoch=4), ClockMismatchError, "epochs")
    wall_d = Timestamp(5.0, "wall") - Timestamp(1.0, "wall")
    raises(lambda: a + wall_d, FrameMismatchError, "wall duration on env stamp")
    raises(lambda: d + wall_d, FrameMismatchError, "env duration + wall duration")
    raises(lambda: Timestamp(1.0, "env", unit=M), UnitError, "non-time unit")
    tick = declare_unit("tick", S, 0.05, declared_by="minecraft adapter: 20 tps nominal")
    t0, t1 = Timestamp(0, "env", 1, tick), Timestamp(40, "env", 1, tick)
    close(float((t1 - t0).to(S)), 2.0, 1e-12, "ticks to seconds")


# ---------------------------------------------------------------- quaternions

@case
def quaternion_basics():
    q = quat_from_axis_angle([0, 0, 1], math.pi / 2)
    assert np.allclose(quat_rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-15)
    assert np.allclose(quat_mul(q, quat_inv(q)), quat_identity(), atol=1e-15)
    assert np.allclose(quat_conj(q), quat_inv(q))
    close(quat_angle(q), math.pi / 2, 1e-15)
    raises(lambda: quat_normalize([0, 0, 0, 0]), DegenerateError, "zero quaternion")
    raises(lambda: quat_normalize([1e-14, 0, 0, 0]), DegenerateError, "near-zero quaternion")
    raises(lambda: quat_from_axis_angle([0, 0, 0], 0.3), DegenerateError, "zero axis")
    assert np.allclose(quat_from_axis_angle([0, 0, 0], 0.0), quat_identity())
    raises(lambda: quat_rotate([np.nan, 0, 0, 1], [1, 0, 0]), DegenerateError, "nan")
    raises(lambda: quat_rotate([1, 0, 0], [1, 0, 0]), DegenerateError, "wrong width")


@case
def matrix_round_trip_at_180():
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -2, 3]):
        q = quat_from_axis_angle(axis, math.pi)
        assert abs(q[0]) < 1e-15
        assert quat_equal(matrix_to_quat(quat_to_matrix(q)), q, 1e-12), axis
    raises(lambda: matrix_to_quat(np.diag([1.0, 1.0, -1.0])), DegenerateError, "reflection")


@case
def slerp_cases():
    a = quat_identity()
    b = quat_from_axis_angle([0, 1, 0], 1.0)
    assert quat_equal(slerp(a, b, 0.0), a) and quat_equal(slerp(a, b, 1.0), b)
    close(quat_angle(slerp(a, b, 0.25)), 0.25, 1e-13)
    assert quat_equal(slerp(a, -b, 0.5), slerp(a, b, 0.5)), "q and -q interpolate the same"
    raises(lambda: slerp(a, b, 1.5), DegenerateError, "s outside [0,1]")


@case
def se3_and_se2_parts():
    T = SE3(quat_from_axis_angle([0, 0, 1], 0.7), [1, 2, 3])
    assert np.allclose(T.apply_direction([1, 0, 0]), quat_rotate(T.q, [1, 0, 0])), \
        "directions ignore translation"
    assert (T @ T.inverse()).is_identity(1e-12)
    assert np.allclose(T.as_matrix() @ np.array([1, 1, 1, 1.0]),
                       np.r_[T.apply_point([1, 1, 1]), 1.0])
    assert SE3.from_matrix(T.as_matrix()).almost_equal(T, 1e-12)
    raises(lambda: T @ np.zeros(3), TypeError, "SE3 @ array")
    P = SE2(math.pi * 0.9, [1, 2])
    Q = SE2(math.pi * 0.3, [0, -1])
    assert (P @ Q).to_se3().almost_equal(P.to_se3() @ Q.to_se3(), 1e-12)
    assert (P @ P.inverse()).almost_equal(SE2(), 1e-12)
    close(SE2(3 * math.pi).theta, math.pi, 1e-12, "wrap to (-pi, pi]")


# ---------------------------------------------------------------- frames

@case
def frame_graph_errors():
    g = FrameGraph()
    g.add_frame("world")
    g.add_frame("body", "world", SE3(t=[1, 0, 0]))
    g.add_frame("other_world")
    raises(lambda: g.transform("body", "nope"), UnknownFrameError, "unknown frame")
    raises(lambda: g.transform("body", "other_world"), DisconnectedFramesError, "disconnected")
    raises(lambda: g.add_frame("body", "world"), FrameMismatchError, "duplicate")
    raises(lambda: g.add_frame("x", "missing"), UnknownFrameError, "missing parent")
    raises(lambda: g.set_transform("world", SE3()), FrameMismatchError, "root transform")
    p = Quantity([0.0, 0, 0], M, frame="body")
    raises(lambda: g.convert(p, "world", "body"), FrameMismatchError, "declared frame != src")
    raises(lambda: g.convert(Quantity([0.0, 0, 0], S, frame="body"), "body", "world"),
           UnitError, "point without length dimension")
    raises(lambda: g.convert(p, "body", "world", kind="vector"), ValueError, "kind")
    assert np.allclose(g.convert(p, "body", "world").value, [1, 0, 0])


# ---------------------------------------------------------------- rotor

@case
def rotor_parts():
    raises(lambda: Rotor(np.r_[0, 1.0, 0, 0, 0, 0, 0, 0]), DegenerateError, "odd grade")
    raises(lambda: Rotor(np.zeros(8)), DegenerateError, "zero rotor")
    R = Rotor.from_axis_angle([0, 0, 1], math.pi / 2)
    assert np.allclose(R.apply([1, 0, 0]), [0, 1, 0], atol=1e-15)
    assert np.allclose((R * R.inverse()).mv, Rotor.identity().mv, atol=1e-15)


# ---------------------------------------------------------------- constraints

@case
def constraint_validation():
    raises(lambda: Constraint("x", None, lambda s: True, lambda s: 0.0, 1.0, kind="firm"),
           ConstraintError, "bad kind")
    raises(lambda: Constraint("x", None, lambda s: True, lambda s: 0.0, 0.0, kind="soft"),
           ConstraintError, "soft with zero tolerance")
    reg = ConstraintRegistry()
    reg.register(Constraint("x", None, lambda s: True, lambda s: 0.0, 0.0))
    raises(lambda: reg.register(Constraint("x", None, lambda s: True, lambda s: 0.0, 0.0)),
           ConstraintError, "duplicate")
    d = reg.evaluate({})["x"]
    assert d.status == "satisfied" and d.violated is False


@case
def constraint_never_reports_failure_as_satisfied():
    def boom(s):
        raise KeyError("missing")
    c = Constraint("e", None, lambda s: True, boom, 1.0)
    d = c.evaluate({})
    assert d.status == "error" and d.violated is UNKNOWN
    c2 = Constraint("e2", None, boom, lambda s: 0.0, 1.0)
    assert c2.evaluate({}).status == "error"
    c3 = Constraint("e3", None, lambda s: True, lambda s: float("nan"), 1.0)
    assert c3.evaluate({}).status == "error"
    c4 = Constraint("e4", None, lambda s: (False, "no"), lambda s: 0.0, 1.0)
    d4 = c4.evaluate({})
    assert d4.status == "inapplicable" and d4.violated is INAPPLICABLE and d4.reason == "no"


if __name__ == "__main__":
    sys.exit(1 if run_all("foundation_geometry_unit") else 0)
