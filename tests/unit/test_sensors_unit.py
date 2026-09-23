"""Unit cases for the sensor bus and every registered sensor.

CONTRACTS vs UNITS. tests/_sensor_bus_smoke.py argues the DESIGN — that RED
cannot reach the policy, that a native crop recovers what downsampling
destroys. This file checks the PARTS: every sensor's width, range, neutral
value, and behaviour on the malformed inputs a live client actually produces
(a frame that failed to render, a world dict missing a key, a wrong-shaped
buffer). Those are the failures that happen at 3am on the pod, and none of
them are interesting enough to deserve a contract.

Run: PYTHONPATH=. python tests/unit/test_sensors_unit.py
"""
import math
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, run_all, close, between, finite, raises
from developmental_ai.sensors import (
    GREEN, AMBER, RED, Sensor, SensorBus, build_default_bus, crop_centre,
    DeadReckoner)
from developmental_ai.sensors.builtins import (
    DR_SCALES, AUDIO_CHANNELS, AUDIO_MELS, AUDIO_FRAMES, HUD_ROWS, HUD_COLS)

PD = 13
ALL = ["proprio", "screen_fx", "fovea_native", "fovea_delta", "dead_reckon",
       "motion", "light", "sky", "audio", "hud", "true_position"]


def _bus(enabled=None, fovea=8, ms=1.0):
    return build_default_bus(PD, lambda c: np.full(PD, 0.25, np.float32),
                             enabled=enabled or ALL, fovea_size=fovea,
                             move_scale=ms)


def _ctx(frame=None, prev=None, **world):
    w = {"x": 0.0, "y": 64.0, "z": 0.0, "yaw": 0.0, "dr_x": 0.0, "dr_z": 0.0,
         "move_fwd": 0.0, "move_lat": 0.0, "move_dy": 0.0, "fall_dist": 0.0}
    w.update(world)
    return {"world": w, "info": {}, "pov_native": frame, "prev_fovea": prev}


# ---------------------------------------------------------------- registry

@case
def sensor_rejects_bad_construction():
    raises(lambda: Sensor("x", 0, lambda c: None), ValueError, "zero width")
    raises(lambda: Sensor("x", 4, lambda c: None, classification="purple"),
           ValueError, "unknown classification")
    raises(lambda: Sensor("x", 5, lambda c: None, shape=(1, 2, 2)),
           ValueError, "shape inconsistent with width")
    raises(lambda: Sensor("x", 4, lambda c: None, neutral=np.zeros(3)),
           ValueError, "neutral of the wrong width")
    raises(lambda: Sensor("x", 3, lambda c: None, classification=RED,
                          shape=(1, 1, 3)), ValueError, "RED image sensor")


@case
def bus_rejects_duplicate_and_unknown():
    a = Sensor("a", 2, lambda c: None)
    raises(lambda: SensorBus([a, Sensor("a", 2, lambda c: None)]),
           ValueError, "duplicate name")
    raises(lambda: SensorBus([a], enabled=["nope"]), ValueError, "unknown")


@case
def default_enabled_is_green_only():
    g = Sensor("g", 1, lambda c: None, classification=GREEN)
    m = Sensor("m", 1, lambda c: None, classification=AMBER)
    r = Sensor("r", 1, lambda c: None, classification=RED)
    b = SensorBus([g, m, r])                      # enabled=None
    names = [s.name for s in b.policy_sensors()]
    assert names == ["g"], f"default enabled AMBER or RED: {names}"


@case
def transport_orders_vectors_before_images():
    b = _bus(["proprio", "fovea_native", "screen_fx"])
    assert [n for n, _o, _w in b.layout()] == ["proprio", "screen_fx"]
    assert [n for n, _o, _w, _s in b.image_layout()] == ["fovea_native"]
    assert b.width == b.vector_width + b.image_width


@case
def slice_of_round_trips_every_sensor():
    b = _bus()
    f = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    v = b.read_policy(_ctx(frame=f, prev=crop_centre(f, 8)))
    seen = 0
    for s in b.policy_sensors():
        sl = b.slice_of(s.name, v)
        assert sl is not None and sl.shape[0] == s.width, s.name
        seen += s.width
    assert seen == b.width


@case
def layout_hash_is_stable_and_short():
    a, b = _bus(), _bus()
    assert a.layout_hash() == b.layout_hash()
    assert len(a.layout_hash()) == 16


# ----------------------------------------------------------------- reading

@case
def every_sensor_is_finite_and_in_range_on_a_real_frame():
    b = _bus()
    f = np.random.randint(0, 255, (96, 96, 3), np.uint8)
    v = b.read_policy(_ctx(frame=f, prev=crop_centre(f, 8), moved=0.4))
    finite(v, "transport vector")
    for s in b.policy_sensors():
        sl = b.slice_of(s.name, v)
        # Every sensor must stay in [-1, 1]; the proprio vector's own
        # standing invariant is [0, 1] and the signed ones extend it.
        between(sl.min(), -1.0, 1.0, f"{s.name} min")
        between(sl.max(), -1.0, 1.0, f"{s.name} max")


@case
def every_sensor_reads_neutral_on_a_missing_frame():
    b = _bus()
    v = b.read_policy(_ctx(frame=None))
    finite(v)
    for s in b.image_sensors():
        assert np.allclose(b.slice_of(s.name, v), 0.0), s.name


@case
def malformed_frames_do_not_raise():
    b = _bus(["screen_fx", "light", "sky", "fovea_native"])
    for bad in (np.zeros((4, 4), np.uint8),            # no channel axis
                np.zeros((4, 4, 4), np.uint8),         # RGBA
                np.zeros((0, 0, 3), np.uint8)):        # empty
        v = b.read_policy(_ctx(frame=bad))
        finite(v, f"transport on {bad.shape}")


@case
def world_dict_missing_keys_reads_neutral():
    b = _bus(["dead_reckon", "motion"])
    v = b.read_policy({"world": {}, "info": {}, "pov_native": None,
                       "prev_fovea": None})
    assert np.allclose(v, 0.0)


# ------------------------------------------------------------ crop / fovea

@case
def crop_centre_is_exact_and_never_upscales():
    a = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    c = crop_centre(a, 4)
    assert c.shape == (4, 4, 3)
    assert np.array_equal(c, a[2:6, 2:6])
    small = np.zeros((2, 2, 3), np.uint8)
    assert crop_centre(small, 8).shape == (2, 2, 3), "upscaled a small frame"
    assert crop_centre(None, 4) is None


@case
def fovea_delta_is_zero_on_identical_frames():
    b = _bus(["fovea_delta"], fovea=8)
    f = np.random.randint(0, 255, (32, 32, 3), np.uint8)
    v = b.read_policy(_ctx(frame=f, prev=crop_centre(f, 8)))
    assert float(np.abs(v).sum()) == 0.0


@case
def fovea_delta_localizes_a_single_pixel_change():
    b = _bus(["fovea_delta"], fovea=8)
    f = np.full((32, 32, 3), 100, np.uint8)
    prev = crop_centre(f, 8).copy()
    f[16, 16] = 255
    v = b.read_policy(_ctx(frame=f, prev=prev)).reshape(8, 8)
    assert int((v > 0).sum()) == 1, f"{int((v > 0).sum())} cells changed"


# ---------------------------------------------------------------- physical

@case
def screen_fx_detects_a_red_vignette_but_not_a_red_scene():
    b = _bus(["screen_fx"])
    plain = np.full((64, 64, 3), 90, np.uint8)
    vig = plain.copy()
    vig[:8], vig[-8:], vig[:, :8], vig[:, -8:] = 220, 220, 220, 220
    vig[..., 1] //= 3
    vig[..., 2] //= 3
    red_scene = np.zeros((64, 64, 3), np.uint8)
    red_scene[..., 0] = 200                       # a uniformly red world
    p = float(b.slice_of("screen_fx", b.read_policy(_ctx(frame=plain)))[0])
    g = float(b.slice_of("screen_fx", b.read_policy(_ctx(frame=vig)))[0])
    r = float(b.slice_of("screen_fx", b.read_policy(_ctx(frame=red_scene)))[2])
    assert g > p + 0.2, f"vignette {g:.2f} vs plain {p:.2f}"
    close(r, 0.0, 0.05, "border-vs-centre red on a uniformly red scene")


@case
def light_orders_dark_typical_bright():
    b = _bus(["light"])
    f = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    v = b.slice_of("light", b.read_policy(_ctx(frame=f)))
    assert v[0] <= v[1] <= v[2], f"percentiles out of order: {v[:3]}"
    between(v[3], 0.0, 1.0, "dark fraction")


@case
def sky_reads_the_top_of_the_frame_only():
    b = _bus(["sky"])
    f = np.zeros((64, 64, 3), np.uint8)
    f[:8] = 240                                   # bright sky, dark ground
    bright = float(b.slice_of("sky", b.read_policy(_ctx(frame=f)))[0])
    g = np.zeros((64, 64, 3), np.uint8)
    g[-8:] = 240                                  # bright GROUND only
    dark = float(b.slice_of("sky", b.read_policy(_ctx(frame=g)))[0])
    assert bright > dark + 0.4, "sky is reading the whole frame"


@case
def motion_is_signed_and_clipped():
    b = _bus(["motion"], ms=1.0)
    fwd = b.slice_of("motion", b.read_policy(_ctx(move_fwd=0.5)))
    back = b.slice_of("motion", b.read_policy(_ctx(move_fwd=-0.5)))
    close(fwd[0], 0.5, 1e-6); close(back[0], -0.5, 1e-6)
    fast = b.slice_of("motion", b.read_policy(_ctx(move_fwd=99.0)))
    close(fast[0], 1.0, 1e-6, "clipped forward speed")


@case
def motion_flags_descent_and_accumulates_fall():
    b = _bus(["motion"])
    v = b.slice_of("motion", b.read_policy(_ctx(move_dy=-0.5, fall_dist=8.0)))
    close(v[4], 1.0, 1e-6, "descending flag")
    close(v[3], 0.5, 1e-6, "fall distance normalised by 16 blocks")
    up = b.slice_of("motion", b.read_policy(_ctx(move_dy=0.5)))
    close(up[4], 0.0, 1e-6, "ascending must not read as descending")


@case
def dead_reckon_is_periodic_at_every_scale():
    b = _bus(["dead_reckon"])
    at0 = b.read_policy(_ctx(dr_x=0.0, dr_z=0.0))
    for i, s in enumerate(DR_SCALES):
        shifted = b.read_policy(_ctx(dr_x=float(s), dr_z=0.0))
        close(at0[4 * i], shifted[4 * i], 1e-4, f"scale {s} x-sin")
        close(at0[4 * i + 1], shifted[4 * i + 1], 1e-4, f"scale {s} x-cos")


@case
def heading_conventions_match_their_known_answers():
    """WHICH WAY IS FORWARD is an engine's choice, not a fact of geometry.

    Minecraft/MineRL says yaw 0 faces +z and yaw 90 faces -x. A different
    engine says something else. Baking either into the maths makes the
    geometry quietly engine-shaped — and MineRL is a STAND-IN here, in use
    only while the external server is offline, so that cost would come due.

    This is a known-answer test rather than a derivation because reasoning
    from the phrase "yaw grows clockwise" produced the WRONG sign on the
    first attempt: yaw 90 came out facing east. Whether a rotation reads
    clockwise depends on which way you draw the axes.
    """
    from developmental_ai.sensors import (
        CONVENTIONS, forward_vector, right_vector, heading_cases)
    for conv in CONVENTIONS:
        cases = heading_cases(conv)
        assert cases, f"{conv} has no known-answer cases; it is unfinished"
        for yaw, (ex, ez) in cases:
            fx, fz = forward_vector(math.radians(yaw), conv)
            close(fx, ex, 1e-9, f"{conv} yaw {yaw} x")
            close(fz, ez, 1e-9, f"{conv} yaw {yaw} z")
        # right is forward + 90 degrees, and must be a unit vector at right
        # angles to it — otherwise strafe and walk would not be independent.
        for yaw in (0.0, 37.0, 180.0, 300.0):
            f = forward_vector(math.radians(yaw), conv)
            r = right_vector(math.radians(yaw), conv)
            close(f[0] * r[0] + f[1] * r[1], 0.0, 1e-9, f"{conv} orthogonal")
            close(math.hypot(*r), 1.0, 1e-9, f"{conv} right is unit")
    raises(lambda: forward_vector(0.0, "klingon"), ValueError, "unknown engine")


@case
def dead_reckoner_respects_the_convention_it_was_given():
    """The same commanded motion must go opposite ways under opposite
    conventions — which is the whole point of the convention being a value."""
    from developmental_ai.sensors import DeadReckoner
    mc = DeadReckoner("minecraft")
    st = DeadReckoner("standard")
    for d in (mc, st):
        d.step(1.0, 0.0, 0.0)
    close(mc.z, 1.0, 1e-9, "minecraft yaw 0 walks +z")
    close(st.x, 1.0, 1e-9, "standard yaw 0 walks +x")
    assert abs(mc.x) < 1e-9 and abs(st.z) < 1e-9


@case
def dead_reckoner_closes_a_square_and_resets():
    dr = DeadReckoner()
    for yaw in (0.0, 90.0, 180.0, 270.0):
        for _ in range(5):
            dr.step(1.0, 0.0, yaw)
    close(dr.x, 0.0, 1e-5, "square closure x")
    close(dr.z, 0.0, 1e-5, "square closure z")
    assert dr.steps == 20
    dr.reset()
    assert dr.x == 0.0 and dr.steps == 0


@case
def dead_reckoner_ignores_a_missing_heading():
    dr = DeadReckoner()
    dr.step(1.0, 0.0, None)
    assert dr.steps == 0, "integrated without knowing which way it was facing"


# --------------------------------------------------------- blocked sensors

@case
def audio_is_deaf_until_the_key_exists():
    b = _bus(["audio"])
    assert np.allclose(b.read_policy(_ctx()), 0.0)
    ctx = _ctx()
    ctx["info"]["audio"] = np.full(
        (AUDIO_CHANNELS, AUDIO_MELS, AUDIO_FRAMES), 0.5, np.float32)
    close(float(b.read_policy(ctx).mean()), 0.5, 1e-6)


@case
def audio_rejects_a_wrong_shaped_buffer():
    b = _bus(["audio"])
    ctx = _ctx()
    ctx["info"]["audio"] = np.zeros((1, 8, 8), np.float32)   # mono, wrong size
    assert np.allclose(b.read_policy(ctx), 0.0), (
        "a wrong-shaped audio buffer was accepted; it would be reshaped into "
        "nonsense and the agent would learn from a scrambled spectrogram")


@case
def hud_reads_the_bottom_strip_at_the_declared_shape():
    b = _bus(["hud"])
    f = np.zeros((256, 256, 3), np.uint8)
    f[-24:] = 200                                 # a bright HUD band
    v = b.slice_of("hud", b.read_policy(_ctx(frame=f)))
    assert v.shape[0] == 3 * HUD_ROWS * HUD_COLS
    assert float(v.mean()) > 0.3, "the HUD strip did not read the bottom"
    top_only = np.zeros((256, 256, 3), np.uint8)
    top_only[:24] = 200
    v2 = b.slice_of("hud", b.read_policy(_ctx(frame=top_only)))
    assert float(v2.mean()) < 0.1, "the HUD strip is reading the whole frame"


if __name__ == "__main__":
    sys.exit(run_all("sensors-unit"))
