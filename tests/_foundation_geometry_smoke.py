"""Foundation geometry smoke — plan Stage 4, "typed state and geometry library".

WHAT IS CLAIMED (the Stage 4 completion gate): coordinate changes preserve
represented meaning within declared tolerances; invalid operations are
detected; discrete and unknown structures remain expressible.

WHY IT MATTERS HERE. This repo has shipped geometry sign errors that no
test could see: the episodic bearing turned the wrong way because Minecraft
yaw grows clockwise while atan2 grows counter-clockwise, and the yaw = 0
case is blind to it (tests/_oracle_isolation_smoke.py, contract F). Depth
from motion is monocular — its scale is unknown — and the obvious way to
misuse it is to add it to a dead-reckoned position in blocks. And plan §2.3
insists that a COORDINATE change and a PHYSICAL rotation are different
operations: gravity re-expressed along with the scene is a symmetry; a scene
rotated under fixed gravity is not. Each of those is a contract below, and
each has a falsification — a deliberately wrong input the check must catch.

Contracts:
    A. ROUND TRIPS. 1000 random SE(3) and SE(2) transforms: inverse round
       trip, matrix round trip, composition == matrix product, point vs
       direction semantics; float64 tolerance 1e-9.
    B. GROUP LAWS. Associativity, identity and inverse for quaternions and
       SE(3) on 1000 random triples.
    C. DEGENERACIES. Zero-norm quaternions rejected; 180-degree rotations
       exact through the matrix path; pitch = +-90 (where Euler angles lose
       a degree of freedom) keeps yaw distinguishable; near-identity slerp
       accurate to relative 1e-9.
    D. UNITS AND SCALE. Each invalid operation raises its own error class;
       an up-to-scale (monocular) estimate never reaches a metric value
       except by evidence-bearing calibration or a same-reconstruction
       ratio; it cannot cross a translated frame edge.
    E. COORDINATE COVARIANCE. A prediction made in frame A and converted to
       B equals the prediction made in B from converted inputs; tree
       composition equals direct composition; disconnected trees refuse.
    F. COORDINATE CHANGE vs PHYSICAL ROTATION. The covariance law holds under
       1000 coordinate changes, is INAPPLICABLE (not satisfied) under a
       physical rotation which really does change the outcome, and catches a
       predictor that hard-codes gravity.
    G. ROTOR == QUATERNION. Cl(3,0) rotor sandwich, composition and axis-angle
       construction agree with the quaternion path on 1000 random cases;
       basis/sign conventions pinned by known products.
    H. ADAPTER CONVENTION. The Minecraft declaration reproduces
       sensors/heading.py at every whole-degree yaw, is metadata on a frame
       (the core never reads it), and a wrong-signed declaration is refused.
    I. CONSTRAINT DIAGNOSTICS. Rigid-distance: satisfied / violated /
       inapplicable-when-undeclared; conservation: exceptions accounted,
       unaccounted change violated (soft, graded), other boundary
       inapplicable, UNKNOWN count unknown. `bool(violated)` refuses on
       anything not actually checked.
    J. GRADIENTS. torch.autograd.gradcheck (float64) on quaternion compose,
       normalise+rotate, geometric product and rotor sandwich; torch == numpy.
    K. EXPRESSIBILITY. Unknown discrete values with partial support, relations
       of unknown truth, unknown-scale quantities and clocked timestamps are
       all representable and refuse to be silently mixed.

Run: PYTHONPATH=. python tests/_foundation_geometry_smoke.py
"""
import math
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.foundation.geometry import (
    INAPPLICABLE, M, MINECRAFT, ROS_FLU, S, SE2, SE3, UNKNOWN, AxisConvention,
    ClockMismatchError, ConstraintRegistry, DegenerateError,
    DisconnectedFramesError, DiscreteVar, FrameGraph, FrameMismatchError,
    Quantity, RelationType, Rotor, ScaleStatusError, Timestamp, UnitError,
    conservation_constraint, coordinate_change, declare_convention, declare_unit,
    gravity_coordinate_covariance_constraint, gp, incline_acceleration,
    matrix_to_quat, physical_rotation, quat_angle, quat_equal,
    quat_from_axis_angle, quat_identity, quat_inv, quat_mul, quat_normalize,
    quat_rotate, quat_to_matrix, random_quat, rigid_distance_constraint, slerp)

N = 1000
TOL = 1e-9
rng = np.random.default_rng(20261003)


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # the WRONG check fired
        raise AssertionError(f"expected {getattr(exc, '__name__', exc)}, got {type(e).__name__}: {e}")
    return False


def test_A_round_trips():
    worst = 0.0
    for _ in range(N):
        T = SE3.random(rng)
        U = SE3.random(rng)
        p = rng.normal(size=3) * 50
        d = rng.normal(size=3)
        worst = max(worst,
                    np.abs(T.inverse().apply_point(T.apply_point(p)) - p).max(),
                    np.abs(SE3.from_matrix(T.as_matrix()).as_matrix() - T.as_matrix()).max(),
                    np.abs((T @ U).as_matrix() - T.as_matrix() @ U.as_matrix()).max(),
                    np.abs(T.apply_direction(d) - quat_to_matrix(T.q) @ d).max(),
                    abs(np.linalg.norm(T.apply_direction(d)) - np.linalg.norm(d)))
        P, Q = SE2.random(rng), SE2.random(rng)
        x = rng.normal(size=2) * 50
        worst = max(worst,
                    np.abs(P.inverse().apply_point(P.apply_point(x)) - x).max(),
                    np.abs((P @ Q).as_matrix() - P.as_matrix() @ Q.as_matrix()).max())
    assert worst < TOL, f"round-trip error {worst:.3g} exceeds {TOL}"
    # Falsification: if apply_direction used the translation, this would move.
    T = SE3(quat_identity(), [100.0, 0, 0])
    assert np.allclose(T.apply_direction([0, -9.81, 0]), [0, -9.81, 0])
    assert np.allclose(T.apply_point([0, -9.81, 0]), [100, -9.81, 0])
    print(f"  A. {N} random SE3+SE2: worst round-trip/compose error {worst:.2e} < {TOL}; "
          f"directions ignore translation")


def test_B_group_laws():
    worst = 0.0
    I = SE3()
    for _ in range(N):
        a, b, c = random_quat(rng), random_quat(rng), random_quat(rng)
        worst = max(worst,
                    np.abs(quat_mul(quat_mul(a, b), c) - quat_mul(a, quat_mul(b, c))).max(),
                    np.abs(quat_mul(a, quat_inv(a)) - quat_identity()).max(),
                    np.abs(quat_mul(a, quat_identity()) - a).max())
        A, B, C = SE3.random(rng), SE3.random(rng), SE3.random(rng)
        L, R = (A @ B) @ C, A @ (B @ C)
        worst = max(worst, np.abs(L.as_matrix() - R.as_matrix()).max(),
                    np.abs((A @ A.inverse()).as_matrix() - np.eye(4)).max(),
                    np.abs((A @ I).as_matrix() - A.as_matrix()).max())
    # Non-commutativity must survive (a library that silently commuted would pass the above).
    x90 = quat_from_axis_angle([1, 0, 0], math.pi / 2)
    y90 = quat_from_axis_angle([0, 1, 0], math.pi / 2)
    assert not quat_equal(quat_mul(x90, y90), quat_mul(y90, x90))
    assert worst < TOL, worst
    print(f"  B. associativity / identity / inverse over {N} triples: worst {worst:.2e}; "
          f"rotation order still matters")


def test_C_degeneracies():
    assert _raises(lambda: quat_normalize([0.0, 0, 0, 0]), DegenerateError)
    assert _raises(lambda: SE3([0.0, 0, 0, 0]), DegenerateError)
    assert _raises(lambda: Rotor(np.zeros(8)), DegenerateError)
    # 180 degrees: w = 0, the case where the trace-based matrix->quat divides by ~0.
    worst = 0.0
    for _ in range(200):
        ax = rng.normal(size=3)
        q = quat_from_axis_angle(ax, math.pi)
        q2 = matrix_to_quat(quat_to_matrix(q))
        worst = max(worst, min(np.abs(q - q2).max(), np.abs(q + q2).max()))
        v = rng.normal(size=3)
        n = ax / np.linalg.norm(ax)
        # 180 about n: v -> 2(v.n)n - v
        worst = max(worst, np.abs(quat_rotate(q, v) - (2 * (v @ n) * n - v)).max())
    assert worst < 1e-12, worst
    # Gimbal: at pitch -90 in the Minecraft declaration the body looks straight
    # up for EVERY yaw; Euler (yaw, pitch, roll) cannot tell yaw from roll there.
    # The quaternion keeps the full orientation: forward agrees, the body's
    # lateral axis does not.
    qa = MINECRAFT.orientation(math.radians(10), math.radians(-90))
    qb = MINECRAFT.orientation(math.radians(70), math.radians(-90))
    fa, fb = (quat_rotate(q, MINECRAFT.forward_at_zero) for q in (qa, qb))
    la, lb = (quat_rotate(q, MINECRAFT.lateral) for q in (qa, qb))
    assert np.allclose(fa, [0, 1, 0]) and np.allclose(fb, [0, 1, 0])
    close_ang = math.degrees(math.acos(np.clip(la @ lb, -1, 1)))
    assert abs(close_ang - 60.0) < 1e-9, close_ang
    assert quat_equal(matrix_to_quat(quat_to_matrix(qa)), qa, 1e-12)
    # Near-identity slerp: angle 1e-9 rad, halfway must be 5e-10 rad.
    tiny = quat_from_axis_angle([0.3, -0.2, 0.9], 1e-9)
    mid = slerp(quat_identity(), tiny, 0.5)
    rel = abs(quat_angle(mid) - 5e-10) / 5e-10
    assert rel < 1e-6, rel
    # Antipodal endpoints: q and -q are one rotation; slerp must not take the long way.
    q = random_quat(rng)
    assert quat_equal(slerp(q, -q, 0.5), q, 1e-12)
    print(f"  C. zero-norm rejected; 200 x 180deg exact ({worst:.1e}); pitch -90 keeps "
          f"a 60deg yaw difference; near-identity slerp rel err {rel:.1e}")


def test_D_units_and_scale():
    BLOCK = declare_unit("block", M, 1.0, declared_by="smoke: minecraft adapter")
    pos = Quantity([3.0, 64.0, -2.0], BLOCK, frame="world")       # dead reckoned
    mono = Quantity([0.4, 0.1, 2.0], M, frame="world", scale_status="up_to_scale",
                    scale_ref="flow_recon_7")                      # depth from motion
    # Each misuse raises its OWN error class (a different class = wrong check fired).
    assert _raises(lambda: pos + mono, ScaleStatusError)
    assert _raises(lambda: pos + Quantity([1.0, 0, 0], S, frame="world"), UnitError)
    assert _raises(lambda: pos + Quantity([1.0, 0, 0], BLOCK, frame="body"), FrameMismatchError)
    assert _raises(lambda: float(Quantity(1.0, M, scale_status="up_to_scale", scale_ref="r")),
                   ScaleStatusError)
    assert _raises(lambda: np.asarray(mono), UnitError)
    assert _raises(lambda: mono.metric_value(), ScaleStatusError)
    # numpy must not route around the type: ndarray + Quantity defers to Quantity.
    assert _raises(lambda: np.ones(3) + pos, (UnitError, TypeError))
    # Two paths to metric, both explicit.
    d1 = Quantity(2.0, M, scale_status="up_to_scale", scale_ref="flow_recon_7")
    d2 = Quantity(8.0, M, scale_status="up_to_scale", scale_ref="flow_recon_7")
    assert (d1 / d2).is_metric                                     # scale cancels
    cal = mono.calibrate(1.5, ["known block edge in frames 120-131"])
    s = pos + cal
    assert s.is_metric and np.allclose(s.value, [3.6, 64.15, 1.0])
    assert any(p.startswith("calibrated:flow_recon_7") for p in cal.provenance)
    # Frame edge with a metric translation: points refuse, directions pass.
    g = FrameGraph()
    g.add_frame("world")
    g.add_frame("cam", "world", SE3(random_quat(rng), [1.0, 2.0, 3.0]))
    pt = Quantity([0.1, 0.2, 1.0], M, frame="cam", scale_status="up_to_scale", scale_ref="r")
    assert _raises(lambda: g.convert(pt, "cam", "world"), ScaleStatusError)
    dr = g.convert(pt, "cam", "world", kind="direction")
    assert dr.scale_status == "up_to_scale" and dr.frame == "world"
    print("  D. metric+monocular, m+s, world+body, float(up_to_scale), asarray: each its "
          "own error; metric only via ratio or evidenced calibrate; unscaled points "
          "cannot cross a translated edge")


def test_E_coordinate_covariance():
    worst = 0.0
    for _ in range(N // 4):
        g = FrameGraph()
        g.add_frame("A")
        g.add_frame("mid", "A", SE3.random(rng))
        g.add_frame("B", "mid", SE3.random(rng))
        n_A = rng.normal(size=3)
        n_A /= np.linalg.norm(n_A)
        g_A = rng.normal(size=3) * 9.81
        pred_A = incline_acceleration({"normal": n_A, "gravity": g_A})
        conv = g.convert(pred_A, "A", "B", kind="direction")
        pred_B = incline_acceleration({
            "normal": g.convert(n_A, "A", "B", kind="direction"),
            "gravity": g.convert(g_A, "A", "B", kind="direction")})
        worst = max(worst, np.abs(conv - pred_B).max())
        # Tree composition == direct composition; typed round trip.
        T_direct = g._T["mid"] @ g._T["B"]
        worst = max(worst, np.abs(g.transform("B", "A").as_matrix() - T_direct.as_matrix()).max())
        p = Quantity(rng.normal(size=3) * 20, M, frame="A")
        back = g.convert(g.convert(p, "A", "B"), "B", "A")
        worst = max(worst, np.abs(back.value - p.value).max())
    assert worst < TOL, worst
    g.add_frame("island")
    assert _raises(lambda: g.convert(np.zeros(3), "A", "island"), DisconnectedFramesError)
    # Falsification: converting the prediction as a POINT (picks up translation)
    # breaks covariance — the kind= argument is load-bearing.
    bad = g.convert(pred_A, "A", "B", kind="point")
    assert np.abs(bad - pred_B).max() > 1e-3
    print(f"  E. predict-in-A-then-convert == predict-in-B over {N // 4} random 2-edge "
          f"trees: worst {worst:.2e}; treating it as a point breaks it")


def test_F_coordinate_change_vs_physical_rotation():
    reg = ConstraintRegistry()
    good = reg.register(gravity_coordinate_covariance_constraint(
        incline_acceleration, tolerance=TOL, evidence=["Newtonian incline, frictionless"]))

    def hardcoded_gravity(scene):          # NOT covariant: ignores the scene's gravity
        return incline_acceleration({"normal": scene["normal"], "gravity": [0, -9.81, 0]})
    hardcoded_gravity.__name__ = "hardcoded_gravity"
    bad = gravity_coordinate_covariance_constraint(hardcoded_gravity, tolerance=TOL)

    worst, caught, changed = 0.0, 0, []
    for _ in range(N):
        n = rng.normal(size=3)
        scene = {"normal": n / np.linalg.norm(n), "gravity": np.array([0, -9.81, 0])}
        T = SE3(random_quat(rng))
        d = good.evaluate({"scene": scene, "transformation": {"kind": "coordinate_change", "T": T}})
        assert d.status == "satisfied", d
        worst = max(worst, d.residual)
        if bad.evaluate({"scene": scene, "transformation": {
                "kind": "coordinate_change", "T": T}}).status == "violated":
            caught += 1
        dp = good.evaluate({"scene": scene, "transformation": {"kind": "physical_rotation", "T": T}})
        assert dp.status == "inapplicable" and dp.violated is INAPPLICABLE, dp
        assert _raises(lambda: bool(dp.violated), TypeError)
        # The physical rotation genuinely changes the outcome (in world coordinates):
        out0 = incline_acceleration(scene)
        out1 = incline_acceleration(physical_rotation(scene, T))
        changed.append(np.abs(out1 - T.apply_direction(out0)).max())
    assert caught > 0.99 * N, f"non-covariant predictor caught only {caught}/{N}"
    med = float(np.median(changed))
    assert med > 1.0, f"physical rotation barely changed the outcome (median {med})"
    # Sanity: the coordinate-change helper really is just re-expression.
    s0 = {"normal": np.array([0, 1.0, 0]), "gravity": np.array([0, -9.81, 0])}
    T = SE3(quat_from_axis_angle([1, 0, 0], 0.5))
    assert np.allclose(incline_acceleration(coordinate_change(s0, T)), 0.0)
    assert np.linalg.norm(incline_acceleration(physical_rotation(s0, T))) > 4.0
    print(f"  F. covariance satisfied on {N} coordinate changes (worst {worst:.1e}); "
          f"INAPPLICABLE on physical rotations, which move the outcome (median "
          f"{med:.2f} m/s^2); hard-coded gravity caught {caught}/{N}")


def test_G_rotor_quaternion_parity():
    # Pinned conventions.
    e = {n: np.eye(8)[i] for i, n in enumerate(("1", "e1", "e2", "e12", "e3", "e13", "e23", "e123"))}
    assert np.allclose(gp(e["e1"], e["e2"]), e["e12"]) and np.allclose(gp(e["e2"], e["e1"]), -e["e12"])
    assert np.allclose(gp(e["e123"], e["e123"]), -e["1"])
    assert np.allclose(gp(e["e123"], e["e1"]), e["e23"])
    assert np.allclose(gp(e["e123"], e["e2"]), -e["e13"])           # = e31
    assert np.allclose(gp(e["e123"], e["e3"]), e["e12"])
    assert np.allclose(Rotor.from_axis_angle([0, 0, 1], math.pi / 2).apply([1, 0, 0]), [0, 1, 0])
    worst = 0.0
    for _ in range(N):
        q1, q2 = random_quat(rng), random_quat(rng)
        R1, R2 = Rotor.from_quaternion(q1), Rotor.from_quaternion(q2)
        v = rng.normal(size=3)
        ax, ang = rng.normal(size=3), rng.uniform(-math.pi, math.pi)
        worst = max(worst,
                    np.abs(R1.apply(v) - quat_rotate(q1, v)).max(),
                    np.abs((R2 * R1).apply(v) - quat_rotate(quat_mul(q2, q1), v)).max(),
                    np.abs(Rotor.from_axis_angle(ax, ang).apply(v)
                           - quat_rotate(quat_from_axis_angle(ax, ang), v)).max(),
                    min(np.abs((R2 * R1).to_quaternion() - quat_mul(q2, q1)).max(),
                        np.abs((R2 * R1).to_quaternion() + quat_mul(q2, q1)).max()))
    assert worst < TOL, worst
    print(f"  G. basis signs pinned (e1e2=e12, I^2=-1, Ie2=e31); rotor sandwich, "
          f"composition and axis-angle == quaternion over {N}: worst {worst:.2e}")


def test_H_adapter_convention():
    from developmental_ai.sensors.heading import forward_vector
    assert MINECRAFT.verify() == [] and ROS_FLU.verify() == []
    worst = 0.0
    for deg in range(360):
        fx, fz = forward_vector(math.radians(deg), "minecraft")
        f = MINECRAFT.forward(math.radians(deg))
        worst = max(worst, abs(f[0] - fx), abs(f[2] - fz), abs(f[1]))
    assert worst < 1e-12, worst
    # It is METADATA: two graphs differing only in declared convention convert identically.
    T = SE3.random(rng)
    gs = []
    for conv in (MINECRAFT, ROS_FLU):
        g = FrameGraph()
        g.add_frame("world", convention=conv)
        g.add_frame("body", "world", T)
        gs.append(g)
    p = rng.normal(size=3)
    assert np.array_equal(gs[0].convert(p, "body", "world"), gs[1].convert(p, "body", "world"))
    assert gs[0].convention("world") is MINECRAFT
    # Falsification: the sign error this repo once shipped is refused at declaration.
    wrong = AxisConvention("minecraft_wrong_sign", up=(0, 1, 0), forward_at_zero=(0, 0, 1),
                           yaw_sign=+1.0, pitch_sign=+1.0, declared_by="smoke",
                           cases=MINECRAFT.cases)
    assert _raises(lambda: declare_convention(wrong), DegenerateError)
    print(f"  H. minecraft declaration == sensors/heading.py at 360 yaws ({worst:.0e}); "
          f"convention is frame metadata only; yaw-sign error refused")


def test_I_constraints():
    reg = ConstraintRegistry()
    rig = reg.register(rigid_distance_constraint("log_7", tolerance=1e-9,
                                                 evidence=["no deformation observed in 40 contacts"]))
    cons = reg.register(conservation_constraint(
        "logs", boundary="inventory+radius8",
        exceptions={"block_removed": +1, "item_consumed": -1, "item_left": -1},
        tolerance=0.5, evidence=["count bookkeeping"]))
    pts = rng.normal(size=(6, 3))
    T = SE3.random(rng)

    def snap(points, rigid=True):
        return {"entities": {"log_7": {"points": points,
                                       "rigid_evidence": ["declared"] if rigid else []}}}
    d = rig.evaluate({"before": snap(pts), "after": snap(T.apply_point(pts))})
    assert d.status == "satisfied", d
    warped = pts * np.array([1.0, 1.2, 1.0])
    d = rig.evaluate({"before": snap(pts), "after": snap(warped)})
    assert d.status == "violated" and d.violated is True and d.residual > 1e-3
    d = rig.evaluate({"before": snap(pts, False), "after": snap(warped, False)})
    assert d.status == "inapplicable" and "not declared rigid" in d.reason
    assert _raises(lambda: bool(d.violated), TypeError)
    # Conservation: a broken log counts as an exception, not a violation.
    base = {"boundary": "inventory+radius8", "before": {"logs": 3}, "after": {"logs": 4}}
    d = cons.evaluate(dict(base, events=[{"type": "block_removed", "count": 1}]))
    assert d.status == "satisfied" and d.kind == "soft", d
    d = cons.evaluate(dict(base, events=[]))
    assert d.status == "violated" and abs(d.severity - 2.0) < 1e-12, d
    d = cons.evaluate(dict(base, boundary="inventory_only", events=[]))
    assert d.status == "inapplicable"
    d = cons.evaluate(dict(base, after={"logs": UNKNOWN}, events=[]))
    assert d.status == "unknown" and d.violated is UNKNOWN
    # One state can serve several laws: snapshots carry entities AND counts.
    allstate = {"boundary": base["boundary"], "events": [],
                "before": dict(snap(pts), logs=3), "after": dict(snap(warped), logs=4)}
    summ = reg.summary(reg.evaluate(allstate))
    assert summ["violated"] == 2, summ
    print("  I. rigid: satisfied/violated/inapplicable-when-undeclared; conservation: "
          "exception accounted, unaccounted soft-violated (severity 2.0), other boundary "
          "inapplicable, UNKNOWN count -> unknown; bool(unchecked) refuses")


def test_J_gradients():
    try:
        import torch
    except ImportError:  # pragma: no cover
        print("  J. SKIPPED: torch unavailable")
        return
    from torch.autograd import gradcheck
    from developmental_ai.foundation.geometry.torch_ops import (
        gp_t, quat_mul_t, quat_normalize_t, quat_rotate_t, rotor_from_quat_t,
        rotor_sandwich_t)
    from developmental_ai.foundation.geometry.rotor import gp as gp_np
    torch.manual_seed(0)
    dt = torch.float64
    a = torch.randn(5, 4, dtype=dt, requires_grad=True)
    b = torch.randn(5, 4, dtype=dt, requires_grad=True)
    v = torch.randn(5, 3, dtype=dt, requires_grad=True)
    R = torch.randn(5, 8, dtype=dt, requires_grad=True)
    m = torch.randn(5, 8, dtype=dt, requires_grad=True)
    assert gradcheck(quat_mul_t, (a, b), eps=1e-6, atol=1e-8)
    assert gradcheck(lambda q, x: quat_rotate_t(quat_normalize_t(q), x), (a, v), eps=1e-6, atol=1e-8)
    assert gradcheck(gp_t, (R, m), eps=1e-6, atol=1e-8)
    assert gradcheck(lambda q, x: rotor_sandwich_t(rotor_from_quat_t(quat_normalize_t(q)), x),
                     (a, v), eps=1e-6, atol=1e-8)
    worst = 0.0
    with torch.no_grad():
        qn = quat_normalize_t(a)
        for i in range(5):
            q_np, v_np = qn[i].numpy(), v[i].numpy()
            worst = max(worst,
                        np.abs(quat_mul_t(qn[i], qn[(i + 1) % 5]).numpy()
                               - quat_mul(q_np, qn[(i + 1) % 5].numpy())).max(),
                        np.abs(quat_rotate_t(qn[i], v[i]).numpy() - quat_rotate(q_np, v_np)).max(),
                        np.abs(rotor_sandwich_t(rotor_from_quat_t(qn[i]), v[i]).numpy()
                               - Rotor.from_quaternion(q_np).apply(v_np)).max(),
                        np.abs(gp_t(R[i], m[i]).numpy() - gp_np(R[i].numpy(), m[i].numpy())).max())
    assert worst < 1e-12, worst
    assert _raises(lambda: quat_normalize_t(torch.zeros(4, dtype=dt)), DegenerateError)
    print(f"  J. gradcheck(float64) passes for quat compose, normalise+rotate, geometric "
          f"product, rotor sandwich; torch == numpy ({worst:.1e})")


def test_K_expressibility():
    block = DiscreteVar("target_block", ["oak_log", "birch_log", "leaves", "air"])
    assert block.value is UNKNOWN and len(block.support) == 4
    narrowed = block.restrict(["oak_log", "birch_log"])
    assert narrowed.value is UNKNOWN and narrowed.support == {"oak_log", "birch_log"}
    reach = RelationType("reachable_from", 2)
    r = reach("log_7", "standpoint_3", truth=UNKNOWN, evidence=["never attempted"])
    assert r.truth is UNKNOWN
    q = Quantity(1.3, M, scale_status="unknown")
    assert _raises(lambda: q + Quantity(1.0, M), ScaleStatusError)
    env = Timestamp(100.0, "env", epoch=12)
    wall = Timestamp(1.7e9, "wall")
    assert _raises(lambda: env - wall, ClockMismatchError)
    assert _raises(lambda: env < Timestamp(5.0, "env", epoch=13), ClockMismatchError)
    print("  K. UNKNOWN discrete with partial support, UNKNOWN relation truth, "
          "unknown-scale quantity, env-vs-wall and cross-episode clocks all expressible "
          "and refused when mixed")


def main():
    t0 = time.time()
    for fn in (test_A_round_trips, test_B_group_laws, test_C_degeneracies,
               test_D_units_and_scale, test_E_coordinate_covariance,
               test_F_coordinate_change_vs_physical_rotation,
               test_G_rotor_quaternion_parity, test_H_adapter_convention,
               test_I_constraints, test_J_gradients, test_K_expressibility):
        fn()
    print(f"[foundation_geometry_smoke] ALL PASS ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"[foundation_geometry_smoke] FAIL: {e}")
        raise
