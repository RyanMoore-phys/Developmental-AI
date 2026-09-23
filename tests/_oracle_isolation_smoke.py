"""Oracle-isolation smoke (2026-09-19, roadmap item O1).

WHAT THE ORACLE IS FOR, AND WHAT IT MUST NEVER BE

    This project's founding thesis is that MEANING IS EARNED, NEVER DECLARED.
    A player can press F3 and read `minecraft:oak_log` — that string is
    player-perceivable and still forbidden, because a human reading it
    already knows English and already knows what a log is. The string merely
    indexes knowledge earned elsewhere. SkyBot has no elsewhere.

    But the same information is enormously valuable as MEASUREMENT. Until
    now the project could not answer "does a learned predicate track
    anything real", "how far off is the agent's sense of where it is", or
    "what was actually under the crosshair when a long attack run ended".
    MineDojo calls exactly this class of observation PRIVILEGED INFORMATION;
    that is the right term, and the right treatment is a channel that can be
    read by the experimenter and never by the agent.

WHY ISOLATION IS STRUCTURAL RATHER THAN CAREFUL

    A filter you must remember to apply is a filter you will eventually
    forget — this repo has nine recorded instances of a guard that stopped
    working because a second code path grew around it. So RED sensors are
    not filtered out of the transport vector: `SensorBus.read_policy` never
    VISITS them. There is no step to omit. RED readings are returned by a
    different method, into a different dict, which the replay buffer does
    not carry.

Contracts:
    A. WIDTH ARITHMETIC. Registering a RED sensor does not change the
       transport width by a single float, and the vector is byte-identical
       to the same bus without it.
    B. NO ADDRESS. A RED sensor cannot be sliced out of the transport, so
       even a caller who went looking could not find it there.
    C. SEPARATE VESSEL. RED readings appear only via read_oracle, and the
       replay buffer's column is built from the transport, so a meaning
       has no route into the world model.
    D. ONE CONSUMER. Exactly one method in developmental_loop.py touches
       info["oracle"], and it writes only to telemetry.
    E. THE MEASUREMENT IS REAL. Dead reckoning over a known trajectory
       reproduces truth, and a sensing error shows up as drift rather than
       being silently absorbed.
    F. HEADING SIGN. Minecraft yaw grows CLOCKWISE while atan2 grows
       counter-clockwise; this repo has already shipped that sign error once
       (the episodic bearing), and the yaw=0 case cannot see it.

Run: PYTHONPATH=. python tests/_oracle_isolation_smoke.py
"""
import ast
import math
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.sensors import (
    GREEN, RED, DeadReckoner, Sensor, SensorBus, build_default_bus)
from developmental_ai.sensors.builtins import DR_SCALES

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
CFG = os.path.join("configs", "minecraft_skybot.yaml")
PD = 13


def _bus(enabled, fovea=4):
    return build_default_bus(
        PD, lambda ctx: np.full(PD, 0.5, np.float32),
        enabled=enabled, fovea_size=fovea, move_scale=1.0)


def _ctx(**world):
    w = {"x": 0.0, "y": 64.0, "z": 0.0, "yaw": 0.0,
         "dr_x": 0.0, "dr_z": 0.0, "move_fwd": 0.0, "move_lat": 0.0,
         "move_dy": 0.0, "fall_dist": 0.0}
    w.update(world)
    return {"world": w, "info": {}, "pov_native": None, "prev_fovea": None}


def test_red_costs_zero_transport_width():
    """A. A meaning does not occupy one float of the agent's input."""
    green_only = ["proprio", "dead_reckon", "motion"]
    a = _bus(green_only)
    b = _bus(green_only + ["true_position"])

    assert a.width == b.width, (
        f"enabling true_position changed the transport width "
        f"{a.width} -> {b.width}; a meaning is reaching the policy")
    assert a.layout_hash() == b.layout_hash(), (
        "a RED sensor changed the layout hash — it is participating in the "
        "wire format it must be absent from")

    va = a.read_policy(_ctx(x=123.0, z=-77.0))
    vb = b.read_policy(_ctx(x=123.0, z=-77.0))
    assert np.array_equal(va, vb), (
        "the transport differs with the oracle enabled; the values are "
        "leaking even though the width matches")
    print(f"  A. width {a.width} with and without the oracle; transport "
          f"byte-identical; hash unchanged ({a.layout_hash()})")


def test_red_has_no_address_in_the_transport():
    """B. Even a caller who went looking could not find it."""
    b = _bus(["proprio", "true_position"])
    vec = b.read_policy(_ctx(x=9.0, z=9.0))
    assert b.slice_of("true_position", vec) is None
    assert [s.name for s in b.policy_sensors()] == ["proprio"], (
        "policy_sensors() returned a RED sensor — that list IS the "
        "definition of what the world model may see")
    assert [s.name for s in b.oracle_sensors()] == ["true_position"]
    # And the values are genuinely absent, not merely unaddressed.
    assert not np.any(np.isclose(vec, 9.0)), (
        "the true position appears inside the transport vector")
    print("  B. not sliceable, not in policy_sensors(), values absent")


def test_red_travels_in_a_separate_vessel():
    """C. read_oracle is the only way out, and it is not the buffer's column."""
    b = _bus(["proprio", "true_position"])
    orc = b.read_oracle(_ctx(x=1.5, y=70.0, z=-2.5))
    assert set(orc) == {"true_position"}
    assert np.allclose(orc["true_position"], [1.5, 70.0, -2.5])

    # The buffer column is built from the TRANSPORT. Same width, no oracle.
    from developmental_ai.world_model.replay_buffer import ReplayBuffer
    buf = ReplayBuffer(capacity=64, obs_dim=4, action_dim=2,
                       proprio_dim=b.width, sensor_layout=b.layout_hash())
    buf.add(np.zeros(4, np.float32), 0, 0.0, False,
            proprio=b.read_policy(_ctx(x=1.5, z=-2.5)))
    stored = np.asarray(buf.proprio[0])
    assert stored.shape[0] == b.width
    assert not np.any(np.isclose(stored, 1.5)), (
        "a true coordinate reached the replay buffer")
    print(f"  C. oracle readable by name; buffer column is {b.width} wide "
          f"and carries none of it")


def test_exactly_one_consumer():
    """D. One method touches the oracle, and it only writes telemetry."""
    src = open(LOOP).read()
    # AST, NOT TEXT SEARCH. The comments and docstrings that explain the
    # barrier necessarily quote the key they are explaining, and counting
    # those would make the contract fail for being well documented. A string
    # CONSTANT equal to "oracle" only appears where code actually uses it —
    # docstrings are a single constant holding the whole prose.
    hits = sum(1 for node in ast.walk(ast.parse(src))
               if isinstance(node, ast.Constant) and node.value == "oracle")
    assert hits == 1, (
        f'the literal "oracle" is used {hits} times in loop CODE; there must '
        f'be exactly ONE consumer (_oracle_observe) so the isolation has a '
        f'single thing to audit rather than a habit to maintain')
    assert "def _oracle_observe" in src
    # The consumer must not feed the model or the policy.
    i = src.index("def _oracle_observe")
    body = src[i:src.index("\n    def ", i + 10)]
    for forbidden in ("self.policy", "world_model", "replay_buffer",
                      "curiosity", "reward"):
        assert forbidden not in body, (
            f"_oracle_observe references {forbidden!r} — the evaluation sink "
            f"must not touch anything the agent learns from")
    print("  D. one consumer; it touches no model, policy, buffer or reward")


def test_dead_reckoning_measures_something_real():
    """E. The estimate tracks truth, and a sensing gap shows up as drift."""
    dr = DeadReckoner()
    # A 10-block square, turning 90 degrees at each corner. Minecraft yaw 0
    # faces +z, so leg one must move +z.
    for yaw in (0.0, 90.0, 180.0, 270.0):
        for _ in range(10):
            dr.step(1.0, 0.0, yaw)
    assert abs(dr.x) < 1e-6 and abs(dr.z) < 1e-6, (
        f"a closed square did not close: ({dr.x:.3f}, {dr.z:.3f})")

    dr.reset()
    for _ in range(10):
        dr.step(1.0, 0.0, 0.0)
    assert abs(dr.z - 10.0) < 1e-6 and abs(dr.x) < 1e-6, (
        f"10 forward steps at yaw 0 gave ({dr.x:.3f}, {dr.z:.3f}); "
        f"yaw 0 faces +z in Minecraft")

    # A SENSING GAP MUST SHOW. If the body is shoved sideways and the
    # lateral component is not sensed, the estimate has to be wrong — that
    # error is exactly what the oracle exists to quantify.
    blind = DeadReckoner()
    seeing = DeadReckoner()
    for _ in range(10):
        blind.step(1.0, 0.0, 0.0)          # lateral push not sensed
        seeing.step(1.0, 0.5, 0.0)         # lateral push sensed
    drift = math.hypot(seeing.x - blind.x, seeing.z - blind.z)
    assert drift > 4.0, (
        f"an unsensed 0.5-block/step sideways push produced only {drift:.2f} "
        f"blocks of divergence over 10 steps; the measurement is not "
        f"sensitive to the thing it exists to catch")
    print(f"  E. square closes to <1e-6; yaw 0 -> +z; an unsensed lateral "
          f"push shows as {drift:.1f} blocks of drift")


def test_heading_sign_and_periodic_code():
    """F. The sign this repo has already got wrong once, pinned by cases."""
    # yaw 90 faces WEST (-x) in Minecraft.
    dr = DeadReckoner()
    dr.step(1.0, 0.0, 90.0)
    assert dr.x < -0.99 and abs(dr.z) < 1e-6, (
        f"yaw 90 moved to ({dr.x:.3f}, {dr.z:.3f}); it must face -x (west). "
        f"This is the episodic-bearing sign error, one module over")

    dr.reset()
    dr.step(1.0, 0.0, 270.0)
    assert dr.x > 0.99, "yaw 270 must face +x (east)"

    # The periodic code must distinguish nearby positions and REPEAT at the
    # scale period — that repetition is the point, and is why four scales
    # are used rather than one.
    b = _bus(["dead_reckon"])
    at0 = b.read_policy(_ctx(dr_x=0.0, dr_z=0.0))
    at1 = b.read_policy(_ctx(dr_x=1.0, dr_z=0.0))
    assert np.abs(at0 - at1).max() > 0.5, (
        "one block apart is indistinguishable in the positional code")
    fine = float(DR_SCALES[0])
    atp = b.read_policy(_ctx(dr_x=fine, dr_z=0.0))
    assert np.allclose(at0[:2], atp[:2], atol=1e-5), (
        f"the finest scale ({fine} blocks) did not repeat at its own period")
    assert b.width == len(DR_SCALES) * 4
    print(f"  F. yaw 90 -> west, 270 -> east; code distinct at 1 block and "
          f"periodic at {fine:.0f} blocks over {len(DR_SCALES)} scales")


def test_config_is_coherent():
    import yaml
    cfg = yaml.safe_load(open(CFG))
    sen = cfg.get("sensors") or {}
    en = list(sen.get("enabled") or [])
    if "true_position" in en:
        assert "dead_reckon" in en, (
            "the oracle is enabled but dead_reckon is not, so there is "
            "nothing for it to measure")
    if "motion" in en:
        assert "proprio" in en, "motion refines `moved`, which lives in proprio"
    print(f"  G. config: {len(en)} sensors, oracle "
          f"{'on' if 'true_position' in en else 'off'}")


if __name__ == "__main__":
    test_red_costs_zero_transport_width()
    test_red_has_no_address_in_the_transport()
    test_red_travels_in_a_separate_vessel()
    test_exactly_one_consumer()
    test_dead_reckoning_measures_something_real()
    test_heading_sign_and_periodic_code()
    test_config_is_coherent()
    print("[oracle-isolation] ALL PASS")
