"""Sensor-bus smoke (2026-09-19).

WHY THE BUS EXISTS
    The agent's observation tensor is pure pixels and `obs_dim` is
    load-bearing — minerl_env.create_observables records that location and
    life stats were routed through `info` rather than the obs tensor
    precisely so "obs_dim is unchanged and every saved skill policy stays
    loadable". Wave 1 added ONE side-channel, the 13-field proprio column,
    with optional-on-load semantics so the host's buffer survived.

    That does not scale. Every further sense would repeat the migration
    against the buffer, the world model and BOTH loop bodies. The bus turns
    adding a sense into a registration.

THE RULE IT ENFORCES
    A SENSE is a transducer: it reports a physical quantity the agent's own
    body or screen makes available, WITHOUT NAMING WHAT ANYTHING IS. A
    MEANING names, classifies or identifies. Senses may be policy input;
    meanings may only ever be evaluation.

    "Player-perceivable" is NOT the test. A player can press F3 and read
    `minecraft:oak_log` — player-perceivable, and still forbidden, because
    the thesis is "meaning is earned, never declared". A human reading
    `oak_log` already knows English; SkyBot has no elsewhere.

Contracts:
    A. RED IS EXCLUDED STRUCTURALLY. A RED sensor never appears in the
       transport vector — not because a caller filtered it, but because
       read_policy does not visit it. Asserted by width ARITHMETIC.
    B. LAYOUT HASH pins order and width together. Width alone is not enough:
       two different sensor sets can be the same width and mean entirely
       different things field-for-field.
    C. A buffer whose stored layout differs restores NEUTRAL, never
       misaligned — and a buffer saved before the bus existed still loads.
    D. NEUTRAL IS NOT ALWAYS ZERO, and a broken sensor cannot stall the step.
    E. proprio stays FIRST and keeps its width, so a Wave 1 column lines up
       field-for-field.
    F. IMAGE SENSORS ARE ENCODED, NOT CONCATENATED — 3,072 pixels beside 13
       body scalars would drown the body at 236:1.
    G. The live config is coherent and the A1 claim is true: a feature
       visible at native resolution is DESTROYED by the 128px downsample.

Run: PYTHONPATH=. python tests/_sensor_bus_smoke.py
"""
import os
import sys

sys.path.insert(0, ".")

import pathlib

import numpy as np
import torch

from developmental_ai.sensors import GREEN, AMBER, RED, Sensor, SensorBus
from developmental_ai.sensors import builtins as B
from developmental_ai.sensors.builtins import build_default_bus, crop_centre
from developmental_ai.world_model.rssm import WorldModel

CFG = os.path.join("configs", "minecraft_skybot.yaml")
PD = 13


def _proprio_read(ctx):
    w = ctx.get("world") or {}
    return np.full(PD, float(w.get("food", 0.0)), dtype=np.float32)


def _bus(enabled=None, fovea=8):
    return build_default_bus(PD, _proprio_read, enabled=enabled,
                             fovea_size=fovea)


def _ctx(frame=None, prev=None, food=0.5):
    return {"world": {"food": food}, "info": {},
            "pov_native": frame, "prev_fovea": prev}


def test_red_is_structurally_excluded():
    """A. A meaning cannot reach the world model, by arithmetic.

    Not "we remembered to filter it" — read_policy never visits a RED
    sensor, so there is no path to forget. The oracle travels in a separate
    dict the replay buffer does not carry.
    """
    bus = _bus(enabled=["proprio"])
    w_before = bus.width

    oracle = Sensor("targeted_block", 4, lambda c: np.arange(4, dtype=np.float32),
                    classification=RED)
    bus2 = SensorBus([bus.get("proprio"), oracle],
                     enabled=["proprio", "targeted_block"])

    assert bus2.width == w_before, (
        f"registering a RED sensor changed the transport width "
        f"{w_before} -> {bus2.width}; a meaning is reaching the policy")
    vec = bus2.read_policy(_ctx())
    assert vec.shape[0] == w_before

    # It IS readable — as evaluation, by name, from a separate call.
    orc = bus2.read_oracle(_ctx())
    assert "targeted_block" in orc and orc["targeted_block"].shape == (4,)
    assert bus2.slice_of("targeted_block", vec) is None, (
        "a RED sensor must not be addressable inside the transport vector")

    # An image sensor cannot be RED at all — a shape implies a network.
    try:
        Sensor("bad", 3, lambda c: None, classification=RED, shape=(1, 1, 3))
        raise AssertionError("an image sensor was allowed to be RED")
    except ValueError:
        pass
    print(f"  A. RED registered: width unchanged at {w_before}; readable "
          f"only via read_oracle; image-RED refused")


def test_layout_hash_pins_order_and_width():
    """B. Width alone cannot distinguish two different meanings."""
    a = Sensor("a", 4, lambda c: np.zeros(4, np.float32))
    b = Sensor("b", 4, lambda c: np.ones(4, np.float32))
    ab = SensorBus([a, b], enabled=["a", "b"])
    ba = SensorBus([b, a], enabled=["a", "b"])
    assert ab.width == ba.width == 8
    assert ab.layout_hash() != ba.layout_hash(), (
        "two buses with the same width but opposite field order hash the "
        "same — a restored buffer would be read with the wrong boundaries")

    # Reclassifying does NOT invalidate stored bytes: it changes whether a
    # sensor is enabled by default, not what the numbers mean.
    g = Sensor("a", 4, lambda c: None, classification=GREEN)
    m = Sensor("a", 4, lambda c: None, classification=AMBER)
    assert (SensorBus([g], enabled=["a"]).layout_hash()
            == SensorBus([m], enabled=["a"]).layout_hash())
    print(f"  B. order changes the hash ({ab.layout_hash()} vs "
          f"{ba.layout_hash()}); reclassification does not")


def test_buffer_restores_neutral_on_layout_change():
    """C. A layout change must degrade to neutral, never to misalignment.

    The precedent is Wave 1's proprio width guard and, before it, `restarts`
    being optional on load. The host's buffer is the only copy of the agent's
    experience; a new column that refuses to load bricks it.
    """
    import tempfile
    from developmental_ai.world_model.replay_buffer import ReplayBuffer
    OD, SEQ, W = 8, 4, 6

    b = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=2,
                     proprio_dim=W, sensor_layout="AAAA")
    for i in range(40):
        b.add(np.zeros(OD, np.float32), 0, 0.0, False,
              proprio=np.full(W, 0.7, np.float32))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "buf")
        b.save(p)

        same = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=2,
                            proprio_dim=W, sensor_layout="AAAA")
        n = same.load(p)
        assert np.allclose(same.proprio[:n], 0.7), "same layout must restore"

        # SAME WIDTH, DIFFERENT MEANING — the case width alone cannot catch.
        other = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=2,
                             proprio_dim=W, sensor_layout="BBBB")
        n2 = other.load(p)
        assert float(np.abs(other.proprio[:n2]).sum()) == 0.0, (
            "a buffer with the same WIDTH but a different LAYOUT restored "
            "its values — those fields mean something else now")

        # A pre-bus buffer (no layout recorded) still loads.
        legacy = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=2,
                              proprio_dim=W, sensor_layout="")
        assert legacy.load(p) > 0
    print("  C. same layout restores; same width + different layout -> "
          "neutral; pre-bus buffer still loads")


def test_neutral_and_failure_are_safe():
    """D. Absence must not become a claim, and a broken sensor must not stall."""
    half = Sensor("ratio", 1, lambda c: None,
                  neutral=np.array([0.5], np.float32))
    assert float(half.neutral()[0]) == 0.5, (
        "a ratio whose midpoint is 0.5 must read 0.5 when absent — 0.0 is a "
        "CLAIM that the quantity is at its minimum")

    boom = Sensor("boom", 2, lambda c: 1 / 0)
    assert np.allclose(boom.read({}), 0.0), "a throwing sensor must read neutral"

    wrong = Sensor("wrong", 3, lambda c: np.zeros(5, np.float32))
    assert wrong.read({}).shape == (3,), "a wrong-width sensor must read neutral"

    nan = Sensor("nan", 2, lambda c: np.array([np.nan, 1.0], np.float32))
    assert np.all(np.isfinite(nan.read({}))), (
        "a NaN would propagate into the RSSM posterior and poison every "
        "latent downstream of it, silently and permanently")
    print("  D. neutral != zero where it matters; throw / wrong-width / NaN "
          "all read neutral")


def test_proprio_stays_first_and_intact():
    """E. Wave 1's column must line up field-for-field under the bus."""
    bus = _bus(enabled=["proprio"])
    assert bus.layout()[0][0] == "proprio"
    assert bus.layout()[0][1] == 0, "proprio must start at offset 0"
    assert bus.width == PD, f"proprio-only bus is {bus.width} wide, want {PD}"

    full = _bus(enabled=["proprio", "screen_fx", "fovea_native", "fovea_delta"])
    assert full.layout()[0] == ("proprio", 0, PD), (
        "adding sensors moved proprio — every Wave 1 row would be misread")
    vec = full.read_policy(_ctx(food=0.25))
    assert np.allclose(full.slice_of("proprio", vec), 0.25)
    print(f"  E. proprio at offset 0, width {PD}, unmoved by 3 new sensors")


def test_image_sensors_are_encoded_not_concatenated():
    """F. A 3,072-pixel crop must not be flat-concatenated onto 13 scalars.

    The config already flags a 17:1 compression through one Linear as a
    problem. Raw concatenation here would be 236:1 and the body vector would
    simply stop mattering.
    """
    S, IMG = 8, 16
    bus = _bus(enabled=["proprio", "fovea_native"], fovea=S)
    specs = [(off + bus.vector_width, c, h, w)
             for _n, off, _w, (c, h, w) in bus.image_layout()]
    assert specs, "fovea_native did not register as an image sensor"

    wm = WorldModel(obs_dim=3 * IMG * IMG, action_dim=3, stochastic_size=4,
                    stochastic_classes=4, deterministic_size=16, hidden_dim=16,
                    pixel_obs=True, image_size=IMG, proprio_dim=bus.width,
                    sensor_image_specs=specs, sensor_feature_dim=8)

    raw = bus.width                                   # 13 + 3*S*S
    assert wm.sensor_embed_dim < raw, (
        f"the sensor side reaches the RSSM at width {wm.sensor_embed_dim}, "
        f"no smaller than the raw {raw} — images are not being encoded")
    assert wm.sensor_embed_dim == PD + 8, (
        f"expected {PD} body fields + one 8-d image block, got "
        f"{wm.sensor_embed_dim}")
    assert wm.rssm.obs_dim == 16 + wm.sensor_embed_dim

    out = wm.embed(torch.rand(2, 3 * IMG * IMG), torch.rand(2, bus.width))
    assert out.shape == (2, wm.rssm.obs_dim)
    print(f"  F. raw sensor width {raw} -> encoded {wm.sensor_embed_dim} "
          f"({PD} body + 8 image features); rssm.obs_dim {wm.rssm.obs_dim}")


def test_light_reports_structure_a_mean_cannot():
    """H. G5 — illumination SHAPE, not brightness.

    `screen_fx` already carries frame mean. The point of `light` is the
    structure a mean destroys: a cave mouth in a sunlit field is a dark
    region beside a bright one, and their average is an ordinary afternoon.

    Read off PIXELS, not F3's light-level readout. The pixels are the
    player's own retina (GREEN); the readout is a quantised truth (AMBER).
    """
    bus = _bus(enabled=["proprio", "light"])

    flat = np.full((64, 64, 3), 128, np.uint8)
    split = np.full((64, 64, 3), 200, np.uint8)
    split[:, :32] = 10                                   # half in shadow

    lf = bus.slice_of("light", bus.read_policy(_ctx(frame=flat)))
    ls = bus.slice_of("light", bus.read_policy(_ctx(frame=split)))

    assert abs(float(lf[0]) - float(lf[2])) < 0.05, (
        "a uniformly lit frame must have p05 ~ p95")
    assert float(ls[2]) - float(ls[0]) > 0.5, (
        f"the split frame reported p05={ls[0]:.2f} p95={ls[2]:.2f}; a dark "
        f"region beside a bright one must separate them")
    assert 0.4 < float(ls[3]) < 0.6, (
        f"dark fraction {ls[3]:.2f}; half the frame is unlit")
    assert float(lf[3]) == 0.0, "a lit frame has no unlit fraction"

    # The mean CANNOT tell these apart, which is the whole argument.
    assert abs(float(np.mean(flat) - np.mean(split)) / 255.0) < 0.12, (
        "pick frames whose means are closer, or this contract does not "
        "demonstrate that structure beats a mean")
    print(f"  H. flat p05/p95 {lf[0]:.2f}/{lf[2]:.2f} vs split "
          f"{ls[0]:.2f}/{ls[2]:.2f}, dark frac {ls[3]:.2f} — separated where "
          f"the means are not")


def test_sky_transduces_and_claims_no_phase():
    """I. G6 — and the OVERCLAIM this sensor was corrected out of.

    The roadmap item said "time-of-day phase (sin, cos)". That is NOT
    derivable from a single frame: dawn and dusk look nearly identical. A
    player separates them by which horizon the sun is on (a compass the
    agent does not have) and by remembering which way the last hour went.

    So `sky` reports APPEARANCE and claims nothing about phase, and this
    contract pins BOTH halves — that day and night separate, and that dawn
    and dusk do NOT. The second assertion is the honest limitation, kept as
    a witness so nobody later reads a clock out of four numbers that cannot
    contain one.
    """
    bus = _bus(enabled=["proprio", "sky"])

    def frame(top_rgb, bottom=60):
        f = np.full((64, 64, 3), bottom, np.uint8)
        f[:16] = top_rgb
        return f

    day = bus.slice_of("sky", bus.read_policy(_ctx(frame=frame((120, 165, 255)))))
    night = bus.slice_of("sky", bus.read_policy(_ctx(frame=frame((8, 8, 20)))))
    dawn = bus.slice_of("sky", bus.read_policy(_ctx(frame=frame((255, 150, 90)))))
    dusk = bus.slice_of("sky", bus.read_policy(_ctx(frame=frame((255, 150, 90)))))

    assert float(day[0]) - float(night[0]) > 0.4, (
        "day and night must separate on brightness")
    assert float(dawn[1]) > float(day[1]) + 0.2, (
        f"warm-cool did not separate sunrise ({dawn[1]:.2f}) from midday "
        f"({day[1]:.2f}); R-B is the transducer for that")

    assert np.allclose(dawn, dusk), (
        "dawn and dusk came out DIFFERENT — either the frames differ or this "
        "sensor is claiming a phase it cannot know from one image")
    assert len(day) == 4, (
        "sky is 4 wide; a (sin, cos) phase pair would imply the sensor knows "
        "where in the cycle it is, which it does not")
    print(f"  I. day-night brightness gap {float(day[0]) - float(night[0]):.2f}; "
          f"sunrise warmer than midday ({dawn[1]:.2f} vs {day[1]:.2f}); "
          f"dawn == dusk (no phase claimed)")


def test_native_fovea_recovers_what_downsampling_destroys():
    """G. THE A1 CLAIM, asserted rather than described.

    The agent sees a block-mean downsample of the native frame. The crack
    overlay, the in-reach wireframe and break particles are fine detail at
    the crosshair. If a feature survives the native crop and is destroyed by
    the downsample, then A1 recovers signal that genuinely does not exist
    anywhere else in the agent's input.
    """
    import yaml
    NATIVE, AGENT = 96, 32
    k = NATIVE // AGENT

    frame = np.full((NATIVE, NATIVE, 3), 100, dtype=np.uint8)
    c = NATIVE // 2
    frame[c - 1:c + 1, c - 1:c + 1] = 220            # a 2px crosshair feature

    crop = crop_centre(frame, 16)
    assert float(crop.max()) >= 220, "the native crop lost the feature"

    small = frame.reshape(AGENT, k, AGENT, k, 3).mean(axis=(1, 3))
    centre = small[AGENT // 2 - 1:AGENT // 2 + 1, AGENT // 2 - 1:AGENT // 2 + 1]
    contrast = float(centre.max() - 100)
    assert contrast < 40, (
        f"the downsample preserved {contrast:.0f} of 120 contrast — pick a "
        f"smaller feature or this contract proves nothing")

    # And the delta sensor is exactly zero on a still frame.
    bus = _bus(enabled=["proprio", "fovea_delta"], fovea=8)
    still = np.full((32, 32, 3), 50, np.uint8)
    v = bus.read_policy(_ctx(frame=still, prev=crop_centre(still, 8)))
    assert float(np.abs(bus.slice_of("fovea_delta", v)).sum()) == 0.0, (
        "a static crosshair must read exactly zero change")

    cfg = yaml.safe_load(open(CFG))
    sen = cfg.get("sensors") or {}
    if sen.get("enabled"):
        assert "proprio" in sen["enabled"], "proprio must stay registered"
        assert sen["enabled"][0] == "proprio", (
            "proprio must be FIRST or every Wave 1 buffer row is misread")
        assert cfg["world_model"].get("proprio"), (
            "the sensor bus is configured but world_model.proprio is off, so "
            "nothing consumes it")
        assert int(cfg["environment"]["render_size"]) > int(
            cfg["environment"]["image_size"]), (
            "fovea_native is enabled but render_size == image_size, so the "
            "'native' crop is the same pixels the agent already sees")
    print(f"  G. native crop keeps the feature (max {float(crop.max()):.0f}); "
          f"downsample leaves {contrast:.0f}/120 contrast; still frame reads "
          f"0 delta; config coherent")


def test_sensors_see_this_step_not_the_last_one():
    """J. THE FRAME THE BUS READS IS THE ONE THE ACTION JUST PRODUCED.

    THE LIVE DEFECT (found 2026-09-21, fixed the same day). In
    `MineRLTreechopEnv.step`, `_pov_to_obs` is the ONLY writer of
    `self._last_pov` and of the `_prev_fovea`/`_cur_fovea` pair, and it was
    called inside the RETURN EXPRESSION — i.e. after the sensor block had
    already read `self._last_pov`. Every image sensor therefore described the
    PREVIOUS step's frame:

      * `screen_fx` reported the damage vignette from before the action
      * `light` reported the illumination of the frame before the action
      * `fovea_delta` differenced t-1 against t-2, so the ONE sensor whose
        entire purpose is "did THIS swing land on THIS block" structurally
        could not see this swing

    That is §4.2 duplicated-body drift's quieter cousin: an edit that lands
    in the right file, runs every step, and silently measures the wrong
    thing. Nothing crashes and no number looks wrong — the sensors are all
    perfectly plausible readings of a frame one decision (4 game ticks) old.

    WHY THIS CONTRACT IS SOURCE-LEVEL. Exercising it live needs a booted
    Minecraft client, which is exactly what this offline tier cannot have,
    and a hand-built stand-in would be testing the stand-in. The ordering IS
    the invariant, so the ordering is what gets pinned.
    """
    src = pathlib.Path("developmental_ai/environments/minerl_env.py").read_text()
    step_at = src.index("    def step(self, action: int")
    body = src[step_at:src.index("\n    def render(self)", step_at)]

    i_obs = body.index("_obs_vec = self._pov_to_obs(obs)")
    i_read = body.index("self._sensor_bus.read_policy(")
    assert i_obs < i_read, (
        "the observation (and therefore _last_pov and the foveal crop pair) "
        "must be built BEFORE the sensor bus reads it")

    # ...and the return must hand back the PRE-BUILT vector, not rebuild it:
    # a second call would rotate the fovea pair a second time per step.
    assert "return (_obs_vec," in body
    assert body.count("self._pov_to_obs(obs)") == 1, (
        "exactly one _pov_to_obs per step — a second call would advance "
        "_prev_fovea twice and make fovea_delta a two-step difference")

    # The ONLY writer assumption the argument above rests on.
    assert src.count("self._last_pov = pov") == 1
    print("  J. bus reads THIS step's frame: _pov_to_obs at "
          f"{i_obs} < read_policy at {i_read}; exactly 1 call/step")


def test_coarse_sensors_read_the_downsample_and_agree_with_native():
    """K. WHOLE-SCENE STATISTICS DO NOT NEED NATIVE PIXELS.

    `light`, `sky` and `screen_fx` are percentiles, band means and a dark
    fraction. None asks a question that 384px answers better than 128px, and
    at 384 they cost 5.0 ms of every step for the live sensor set — measured
    2026-09-20, against a ~400 ms step. They now read the same downsample the
    agent itself sees (0.56 ms, 9.0x), while `fovea_native`, `fovea_delta`
    and `hud` keep the native frame because sub-128px detail is their whole
    point.

    THE THING THIS HAS TO PROVE is that the readings did not move. On a
    STRUCTURED frame — which is what a renderer emits — `_block_mean` is an
    exact box filter, so every mean-derived field is preserved outright and
    the order statistics shift only as far as 3x3 averaging moves a tail.
    Measured max|delta| <= 0.015 across all fourteen fields.

    THE HONEST CAVEAT, recorded so nobody rediscovers it as a bug: on UNIFORM
    NOISE the same fields move by up to 0.21, because averaging 9 iid samples
    collapses the tails by 3x. That is the pathological case for a box filter
    and it is not what Minecraft renders. If this assertion ever fires on a
    real frame, the resolution choice is what to revisit.
    """
    rng = np.random.default_rng(1)
    nat = np.zeros((384, 384, 3), np.uint8)
    nat[:120] = [135, 180, 235]                  # sky
    nat[120:] = [90, 110, 70]                    # terrain
    nat[200:290, 60:150] = [12, 10, 14]          # a contiguous cave mouth
    nat[:120, 300:340] = [255, 250, 210]         # sun disc
    nat = np.clip(nat.astype(np.int16)
                  + rng.integers(-8, 8, nat.shape), 0, 255).astype(np.uint8)
    small = nat.reshape(128, 3, 128, 3, 3).mean(axis=(1, 3)).astype(np.uint8)

    worst = 0.0
    for name, fn in (("screen_fx", B._read_screen_fx),
                     ("light", B._read_light), ("sky", B._read_sky)):
        a = fn(_ctx(nat))
        b = fn(dict(_ctx(nat), pov_small=small))
        assert a is not None and b is not None
        d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
        worst = max(worst, d)
        assert d <= 0.02, f"{name} moved {d:.4f} on a structured frame"

    # The fallback: with no pov_small the native frame is still read, so a
    # frame source that never sets the key keeps working unchanged.
    assert B._read_light(_ctx(nat)) is not None
    assert B._read_light(_ctx(None)) is None

    # And the sensors that exist FOR native detail must not have been moved.
    src = pathlib.Path("developmental_ai/sensors/builtins.py").read_text()
    for fn_name in ("_read_fovea", "_read_fovea_delta", "_read_hud"):
        at = src.index(f"def {fn_name}(")
        chunk = src[at:at + 400]
        assert "pov_small" not in chunk and "_scene(" not in chunk, (
            f"{fn_name} must keep native pixels")
    print(f"  K. coarse sensors on the 128px frame: max delta {worst:.4f} "
          f"(<=0.02); fovea/hud still native")


def test_one_frame_conversion_per_step():
    """L. THE FRAME IS CONVERTED TO FLOAT ONCE, NOT ONCE PER SENSOR.

    Four sensors read the same frame and each used to call
    `astype(float32)/255` on it independently — four full conversions of one
    array per step (0.169 ms each at 384px). The context dict is rebuilt by
    the env every step, so memoising inside it is scoped to exactly one
    frame and cannot go stale across steps.

    This is the LOSSLESS half of the change above: it must not move a single
    number, which is what the equality below asserts.
    """
    rng = np.random.default_rng(3)
    nat = rng.integers(0, 255, (384, 384, 3), dtype=np.uint8)

    ctx = _ctx(nat)
    first = B._as_float(ctx, "pov_native")
    second = B._as_float(ctx, "pov_native")
    assert first is second, "the second read must be the SAME array, not a copy"

    # a fresh ctx (i.e. the next step) must NOT reuse the previous frame
    other = B._as_float(_ctx(nat), "pov_native")
    assert other is not first

    # lossless: identical to the unmemoised conversion, value for value
    assert np.array_equal(first, nat.astype(np.float32) / 255.0)

    # a missing or malformed frame caches the None rather than retrying
    bad = _ctx(np.zeros((4, 4), np.uint8))
    assert B._as_float(bad, "pov_native") is None
    assert bad["_f32"]["pov_native"] is None
    print("  L. one conversion per frame, cached in the step's ctx; "
          "values identical to the unmemoised path")


if __name__ == "__main__":
    test_red_is_structurally_excluded()
    test_layout_hash_pins_order_and_width()
    test_buffer_restores_neutral_on_layout_change()
    test_neutral_and_failure_are_safe()
    test_proprio_stays_first_and_intact()
    test_image_sensors_are_encoded_not_concatenated()
    test_light_reports_structure_a_mean_cannot()
    test_sky_transduces_and_claims_no_phase()
    test_native_fovea_recovers_what_downsampling_destroys()
    test_sensors_see_this_step_not_the_last_one()
    test_coarse_sensors_read_the_downsample_and_agree_with_native()
    test_one_frame_conversion_per_step()
    print("[sensor-bus] ALL PASS")
