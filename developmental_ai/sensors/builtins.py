"""The sensors that exist today, and the default bus.

Every sensor here is GREEN — a pure transducer over the agent's own body or
its own screen. Nothing in this file names a block, an item or a creature.

READING CONTEXT. Each `read` is handed a dict the env assembles once per
step:

    world       the env's `world` dict (food, life, pitch, yaw, moved, ...)
    pov_native  the FULL-RESOLUTION frame (render_size), HWC uint8, or None
    prev_fovea  last step's foveal crop, HWC uint8, or None
    info        the step info dict

Nothing here reaches into the env object, so a sensor can be exercised in a
test with a hand-built context and no Minecraft at all — which is what makes
every contract in tests/_sensor_bus_smoke.py runnable offline.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .heading import DEFAULT_CONVENTION, forward_vector, right_vector
from .registry import GREEN, RED, Sensor, SensorBus

# ---------------------------------------------------------------------------
# A1 / A3 — the crosshair fovea at NATIVE resolution
# ---------------------------------------------------------------------------
# THE SIGNAL THIS RECOVERS. The game renders at `render_size` (384 live) and
# the agent sees a block-mean downsample to `image_size` (128). The native
# frame is already retained on the adapter for video and the VLM; the agent
# has simply never been shown it.
#
# Three PLAYER-CRITICAL signals live below the 128px floor, all of them at the
# crosshair and all of them currently discarded:
#
#   * THE BLOCK-BREAKING CRACK OVERLAY. The progressive cracking texture is
#     how a player knows a swing is landing AND progressing. The scoreboard
#     reads 13,305 blocks broken for 35 logs; "the agent cannot see whether
#     its attack is doing anything" is the simplest account of that available.
#   * THE BLOCK OUTLINE WIREFRAME, which Minecraft draws ONLY when a block is
#     within reach. `_reach_sense` currently LEARNS that fact from break
#     events while the game renders the ground truth and we throw it away.
#   * BREAK PARTICLES, whose colour carries the material.
#
# A 32x32 centre crop of a 384px frame covers ~6 degrees at full game acuity —
# 3x the resolution the agent gets anywhere else, for 6% more numbers.
#
# Research: "Higher Resolution, Better Generalization" (arXiv 2605.10546)
# finds resolution gains concentrate exactly where small/distant objects must
# be perceived. The wide-low-acuity + narrow-high-acuity construction is the
# standard foveal-peripheral design (arXiv 2306.00975, arXiv 1903.09950).

FOVEA_SIZE = 32


def crop_centre(frame: np.ndarray, size: int) -> Optional[np.ndarray]:
    """Centre `size`x`size` crop, HWC uint8. None if the frame is unusable.

    NO RESAMPLING, deliberately: the entire point is to keep native pixels.
    A frame smaller than the crop is returned whole rather than upscaled —
    upscaling would manufacture detail that is the one thing this sensor
    exists to preserve.
    """
    if frame is None:
        return None
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] != 3:
        return None
    h, w = a.shape[:2]
    if h < size or w < size:
        return a
    t, l = (h - size) // 2, (w - size) // 2
    return a[t:t + size, l:l + size]


def _read_fovea(ctx: Dict) -> Optional[np.ndarray]:
    c = crop_centre(ctx.get("pov_native"), FOVEA_SIZE)
    if c is None or c.shape[:2] != (FOVEA_SIZE, FOVEA_SIZE):
        return None
    return (c.astype(np.float32) / 255.0).transpose(2, 0, 1).reshape(-1)


def _read_fovea_delta(ctx: Dict) -> Optional[np.ndarray]:
    """|fovea_t - fovea_{t-1}|, greyscale.

    THE CRACK OVERLAY IS AN ANIMATION. A difference channel states "something
    at the crosshair is changing" outright, instead of asking the RSSM to
    difference it out of a 4,352-d latent. A still frame reads exactly zero,
    which is also the honest reading on the first step of an episode.
    """
    cur = crop_centre(ctx.get("pov_native"), FOVEA_SIZE)
    prev = ctx.get("prev_fovea")
    if cur is None or prev is None or cur.shape != np.asarray(prev).shape:
        return None
    d = np.abs(cur.astype(np.float32) - np.asarray(prev, np.float32))
    return (d.mean(axis=2) / 255.0).reshape(-1)


# ---------------------------------------------------------------------------
# A4 — screen effects: the damage vignette and friends
# ---------------------------------------------------------------------------
# `damage_taken` is a COUNTER in info. What a player actually experiences is a
# full-screen red flash. The config records that the creature vocabulary never
# produced a single fact because "nothing measurable ever told the agent it
# was hurt" — this is the measurable thing, and it has been rendered on screen
# the whole time.
#
# Six numbers, measured on the frame BORDER because that is where Minecraft
# draws its overlays (damage vignette, underwater tint, fire, low-health
# pulse) while the centre stays world-coloured.

_SCREEN_FX_WIDTH = 6


def _as_float(ctx: Dict, key: str) -> Optional[np.ndarray]:
    """`ctx[key]` as HWC float in [0,1], converted ONCE PER STEP.

    WHY THIS EXISTS. Four sensors read the same frame, and each used to call
    `astype(float32)/255` on it independently -- four full conversions of one
    array per step (0.169 ms each at 384px, measured 2026-09-20). The context
    dict is rebuilt every step by the env, so memoising inside it is scoped
    exactly to one frame and cannot go stale across steps.
    """
    cache = ctx.get("_f32")
    if cache is None:
        cache = ctx["_f32"] = {}
    if key in cache:
        return cache[key]
    frame = ctx.get(key)
    out = None
    if frame is not None:
        a = np.asarray(frame)
        if a.ndim == 3 and a.shape[2] == 3:
            # Unconditional /255 is the long-standing contract: every caller
            # passes a uint8 frame, and changing it here would silently
            # rescale every sensor built on it.
            out = a.astype(np.float32) / 255.0
    cache[key] = out
    return out


# THE SCENE FRAME versus THE NATIVE FRAME. `light`, `sky` and `screen_fx` are
# COARSE STATISTICS -- percentiles, band means, a dark fraction. None of them
# asks a question that native pixels answer better, so they read the same
# 128px downsample the agent itself sees, at 1/9 the pixels (5.0 ms -> 0.6 ms
# for the live sensor set, measured 2026-09-20).
#
# WHAT THIS DOES AND DOES NOT CHANGE. `_block_mean` is an exact k-x box
# filter, so every quantity that is a MEAN over a band whose edge falls on a
# block boundary is preserved exactly: screen_fx's five mean-derived fields
# and sky's brightness, warm-cool and gradient. What DOES move is the three
# order statistics -- screen_fx's `border.std()`, sky's bright fraction, and
# all four of `light` -- because averaging 3x3 blocks compresses the tails.
# At k=3 that shift is small, and p05 still rests on ~800 pixels.
#
# `fovea_native`, `fovea_delta` and `hud` keep the native frame: those are
# the sensors whose entire purpose is sub-128px detail.

def _scene(ctx: Dict, min_side: int = 4) -> Optional[np.ndarray]:
    """The coarse frame for whole-scene statistics, native as a fallback."""
    out = _usable(ctx, "pov_small", min_side)
    return out if out is not None else _usable(ctx, "pov_native", min_side)


def _usable(ctx: Dict, key: str = "pov_native",
            min_side: int = 4) -> Optional[np.ndarray]:
    """HWC float frame in [0,1], or None if it cannot be read.

    A DEGENERATE FRAME IS A REAL INPUT, not a hypothetical: a client that is
    rebuilding, a render that failed, a first step before anything has been
    drawn. numpy's mean over an empty slice returns NaN, and a NaN reaching
    the RSSM posterior poisons every latent downstream of it — permanently
    and silently. The Sensor wrapper catches that, but a guard that fires on
    every bad frame is a log full of warnings, so the check belongs here.
    """
    frame = ctx.get(key)
    if frame is None:
        return None
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] != 3:
        return None
    if a.shape[0] < min_side or a.shape[1] < min_side:
        return None
    return _as_float(ctx, key)


def _read_screen_fx(ctx: Dict) -> Optional[np.ndarray]:
    a = _scene(ctx, min_side=8)
    if a is None:
        return None
    h, w = a.shape[:2]
    b = max(1, min(h, w) // 8)                      # border band
    border = np.concatenate([
        a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
        a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    bm = border.mean(axis=0)                        # R, G, B at the edge
    full = a.reshape(-1, 3).mean(axis=0)
    # Channel EXCESS, not raw channel value: a red vignette is red RELATIVE to
    # the rest of the frame. Raw redness would fire on a desert at sunset.
    red_excess = float(bm[0] - 0.5 * (bm[1] + bm[2]))
    blue_excess = float(bm[2] - 0.5 * (bm[0] + bm[1]))
    return np.array([
        red_excess,                                  # damage / low health
        blue_excess,                                 # underwater
        float(bm[0] - full[0]),                      # border-vs-centre red
        float(border.mean() - full.mean()),          # overall vignetting
        float(full.mean()),                          # scene brightness
        float(border.std()),                         # flat overlay vs world
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# proprio — the 13 body fields, unchanged
# ---------------------------------------------------------------------------
# Kept first in registration order so its offsets are byte-identical to the
# Wave 1 column. A restored buffer written before the bus existed therefore
# lines up field-for-field, which is what lets the pod keep its experience.

def make_proprio_sensor(width: int, read) -> Sensor:
    return Sensor("proprio", width, read, classification=GREEN, version=1)


def build_default_bus(proprio_width: int, proprio_read,
                      enabled: Optional[List[str]] = None,
                      fovea_size: int = FOVEA_SIZE,
                      move_scale: float = 1.0) -> SensorBus:
    """Every sensor that exists today, in a stable registration order.

    ORDER IS THE WIRE FORMAT. `proprio` stays first, and new sensors append.
    Reordering would silently reinterpret every stored row, which is why
    layout_hash exists and why it covers order.
    """
    global FOVEA_SIZE
    FOVEA_SIZE = int(fovea_size)
    s = int(fovea_size)
    sensors = [
        make_proprio_sensor(proprio_width, proprio_read),
        Sensor("screen_fx", _SCREEN_FX_WIDTH, _read_screen_fx,
               classification=GREEN, version=1),
        Sensor("fovea_native", 3 * s * s, _read_fovea,
               classification=GREEN, version=1, shape=(3, s, s)),
        Sensor("fovea_delta", s * s, _read_fovea_delta,
               classification=GREEN, version=1, shape=(1, s, s)),
        # G3/G4 append AFTER the image sensors in registration order. The bus
        # reorders vectors ahead of images in the transport itself, so this
        # ordering is only about which vector field comes last — and proprio
        # staying first is the part that matters for a restored buffer.
        Sensor("dead_reckon", len(DR_SCALES) * 4, _read_dead_reckon,
               classification=GREEN, version=1),
        make_motion_sensor(move_scale),
        Sensor("light", _LIGHT_WIDTH, _read_light,
               classification=GREEN, version=1),
        Sensor("sky", _SKY_WIDTH, _read_sky,
               classification=GREEN, version=1),
        Sensor("audio", AUDIO_CHANNELS * AUDIO_MELS * AUDIO_FRAMES,
               _read_audio, classification=GREEN, version=1,
               shape=(AUDIO_CHANNELS, AUDIO_MELS, AUDIO_FRAMES)),
        Sensor("hud", 3 * HUD_ROWS * HUD_COLS, _read_hud,
               classification=GREEN, version=1,
               shape=(3, HUD_ROWS, HUD_COLS)),
        # RED. Registered so it can be MEASURED, never so it can be used.
        Sensor("true_position", 3, _read_true_position,
               classification=RED, version=1),
    ]
    return SensorBus(sensors, enabled=enabled)


# ---------------------------------------------------------------------------
# G3 — dead reckoning, and why it is not GPS
# ---------------------------------------------------------------------------
# The engine's absolute XYZ is already in `info` (ObservationFromCurrentLocation
# is requested). Using it as policy input would be GPS: a player without F3
# does not have world coordinates, they have a running sense of how far they
# have come and in what direction. So the agent integrates its OWN sensed
# body-relative displacement, from an origin at the episode start.
#
# WHAT THIS DELIBERATELY CANNOT DO. It has no world frame, so two episodes
# never share coordinates; it accumulates whatever error the displacement
# sense carries; and it is reset by anything that teleports the body. The
# true position is registered separately as a RED oracle (item O1) so the
# error can be MEASURED rather than assumed.
#
# WHY THE PERIODIC ENCODING. Raw (x, z) is poor network input: unbounded, and
# a linear layer has no reason to treat nearby positions as similar. A
# multi-scale periodic code fixes both, and is what the brain uses — grid
# cells are a multi-scale periodic metric for space, and Banino et al.
# (Nature 2018) showed grid-like codes EMERGE from training a recurrent
# network to path-integrate, then let agents outperform expert humans at
# goal-directed navigation. The scales below span one block to a long walk.

DR_SCALES = (4.0, 16.0, 64.0, 256.0)     # blocks per cycle


class DeadReckoner:
    """Integrates body-relative displacement into an episode-local position.

    Stateful on purpose and owned by the env adapter, which is the only place
    that sees consecutive positions. Kept here rather than in the adapter so
    it can be exercised offline with no Minecraft at all.
    """

    __slots__ = ("x", "z", "steps", "convention")

    def __init__(self, convention: str = DEFAULT_CONVENTION):
        self.convention = str(convention)
        self.reset()

    def reset(self) -> None:
        self.x = 0.0
        self.z = 0.0
        self.steps = 0

    def step(self, fwd: float, lat: float, yaw_deg) -> None:
        """Advance by one step's body-relative displacement.

        THE HEADING CONVENTION IS A VALUE, NOT AN ASSUMPTION. Which axis yaw
        0 points along, and which way the angle grows, are facts the ENGINE
        chooses — see sensors/heading.py. MineRL is a stand-in while the
        external server is offline, so baking its convention into the maths
        here would make the geometry quietly Minecraft-shaped and turn a
        later migration into a hunt for sign errors.

        Re-deriving the displacement from the body's own heading, rather than
        taking a world delta, is the whole point: this class must only ever
        see quantities the body itself can feel.
        """
        if yaw_deg is None:
            return
        r = math.radians(float(yaw_deg))
        fx, fz = forward_vector(r, self.convention)
        rx, rz = right_vector(r, self.convention)
        self.x += float(fwd) * fx + float(lat) * rx
        self.z += float(fwd) * fz + float(lat) * rz
        self.steps += 1


def _read_dead_reckon(ctx: Dict) -> Optional[np.ndarray]:
    w = ctx.get("world") or {}
    x, z = w.get("dr_x"), w.get("dr_z")
    if x is None or z is None:
        return None
    out = np.empty(len(DR_SCALES) * 4, dtype=np.float32)
    for i, s in enumerate(DR_SCALES):
        out[4 * i + 0] = math.sin(2.0 * math.pi * float(x) / s)
        out[4 * i + 1] = math.cos(2.0 * math.pi * float(x) / s)
        out[4 * i + 2] = math.sin(2.0 * math.pi * float(z) / s)
        out[4 * i + 3] = math.cos(2.0 * math.pi * float(z) / s)
    return out


# ---------------------------------------------------------------------------
# G4 — the motion the body feels
# ---------------------------------------------------------------------------
# `moved` in the proprio vector is a MAGNITUDE. Nothing downstream can tell
# walking forward from being shoved sideways by a mob, or from pressing
# `back`. A player feels all three, plus falling. Five numbers.
#
# `move_scale` normalizes speed the way the proprio vector already does, so
# these read on the same scale as `moved` and stay in [-1, 1].

_MOTION_WIDTH = 5
_FALL_SCALE = 16.0          # blocks; a fatal fall is ~23


def make_motion_sensor(move_scale: float) -> Sensor:
    ms = max(float(move_scale), 1e-6)

    def _read(ctx: Dict) -> Optional[np.ndarray]:
        w = ctx.get("world") or {}
        if w.get("move_fwd") is None:
            return None
        dy = float(w.get("move_dy", 0.0) or 0.0)
        return np.array([
            float(np.clip(float(w.get("move_fwd", 0.0)) / ms, -1.0, 1.0)),
            float(np.clip(float(w.get("move_lat", 0.0)) / ms, -1.0, 1.0)),
            float(np.clip(dy / ms, -1.0, 1.0)),
            float(np.clip(float(w.get("fall_dist", 0.0)) / _FALL_SCALE,
                          0.0, 1.0)),
            1.0 if dy < -1e-3 else 0.0,              # descending right now
        ], dtype=np.float32)

    return Sensor("motion", _MOTION_WIDTH, _read, classification=GREEN,
                  version=1)


# ---------------------------------------------------------------------------
# O1 — the privileged channel. RED: EVALUATION ONLY, NEVER INPUT.
# ---------------------------------------------------------------------------
# This is F3's information, and it is registered so the project can finally
# MEASURE itself — not so the agent can use it. MineDojo calls exactly this
# class of observation "privileged information"; that is the right term.
#
# What it buys, none of which requires the agent to ever see it:
#   * dead-reckoning drift (G3) against truth;
#   * whether a learned predicate tracks anything real;
#   * depth accuracy, once there is a ray to compare against.
#
# It cannot reach the world model: SensorBus.read_policy does not visit RED
# sensors, so there is no code path to forget to filter, and RED readings
# travel in a separate dict the replay buffer does not carry.

def _read_true_position(ctx: Dict) -> Optional[np.ndarray]:
    w = ctx.get("world") or {}
    x, y, z = w.get("x"), w.get("y"), w.get("z")
    if x is None or z is None:
        return None
    return np.array([float(x), float(y or 0.0), float(z)], dtype=np.float32)


# ---------------------------------------------------------------------------
# G5 — illumination structure, read off the retina
# ---------------------------------------------------------------------------
# A player judges "am I somewhere dark", "is there a cave mouth over there",
# "is it night" by brightness, and acts on it — light level governs mob
# spawning and visibility. The agent has never had any representation of it.
#
# TAKEN FROM PIXELS, NOT FROM F3. The debug screen reports a quantised light
# level for the block underfoot; that is a READOUT, and the pixels are the
# player's own retina. Pixels are GREEN and the readout would be AMBER, so
# pixels it is — and it costs nothing, because the frame is already here.
#
# DELIBERATELY NOT MEAN BRIGHTNESS. `screen_fx` already carries frame mean.
# These four describe the SHAPE of the illumination instead: how dark the
# darkest part is, the median, the brightest part, and how much of the view
# is dark. A cave mouth in a sunlit field is a low p05 with a high p95 — a
# fact no single mean can express, and exactly the kind of structure this
# roadmap is about.

_LIGHT_WIDTH = 4
_DARK_THRESHOLD = 0.18          # below this a region reads as unlit


def _luma(a: np.ndarray) -> np.ndarray:
    # Rec. 601 luma. The green weight dominates because the eye's does.
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])


def _read_light(ctx: Dict) -> Optional[np.ndarray]:
    a = _scene(ctx)
    if a is None:
        return None
    y = _luma(a)
    p05, p50, p95 = np.percentile(y, (5.0, 50.0, 95.0))
    return np.array([
        float(p05),                                   # darkest region
        float(p50),                                   # typical
        float(p95),                                   # brightest region
        float((y < _DARK_THRESHOLD).mean()),          # how much is unlit
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# G6 — sky appearance, and a correction to what it can honestly claim
# ---------------------------------------------------------------------------
# THE ROADMAP ITEM SAID "time-of-day PHASE (sin, cos)". That is not derivable
# from a single frame and shipping it would have been a fabrication: dawn and
# dusk look nearly identical from one image. A player tells them apart by
# which horizon the sun is on and by remembering which way the last hour
# went — the first needs a compass the agent does not have, the second is
# memory, not perception.
#
# So this sensor reports SKY APPEARANCE and claims nothing about phase. Four
# numbers over the top eighth of the frame:
#
#   brightness      day/night, and overcast versus clear
#   warm-cool       R minus B: sunrise and sunset are warm, midday is cool,
#                   night is cool and dark
#   gradient        how fast the sky brightens toward the top — a proxy for
#                   sun elevation, and the thing that separates "sun is low"
#                   from "sun is high"
#   bright fraction the sun or moon disc as a share of the strip
#
# The CLOCK is then something the RSSM can learn across steps, because these
# four move monotonically through a cycle even though no single frame pins
# where in the cycle it is. That is the honest division of labour: the sensor
# transduces, the recurrent model integrates.

_SKY_WIDTH = 4


def _read_sky(ctx: Dict) -> Optional[np.ndarray]:
    a = _scene(ctx, min_side=8)
    if a is None:
        return None
    h = a.shape[0]
    band = max(2, h // 8)
    top = a[:band]
    y = _luma(top)
    # Gradient measured top-half versus bottom-half OF THE STRIP, so it is a
    # property of the sky itself rather than of whatever terrain intrudes
    # into the lower frame.
    half = max(1, band // 2)
    grad = float(y[:half].mean() - y[half:].mean())
    return np.array([
        float(y.mean()),
        float(top[..., 0].mean() - top[..., 2].mean()),   # warm - cool
        grad,
        float((y > 0.85).mean()),                          # sun/moon disc
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# A2 — the HUD, at native resolution
# ---------------------------------------------------------------------------
# WHAT IS RENDERED THERE AND NOWHERE ELSE: the hotbar and which slot is
# selected, the held-item model, and the health and hunger bars. The agent
# reconstructs none of it today — `FlatInventoryObservation` gives item->count
# with NO SLOT INDICES, and `equipped_items` is dead in MineRL 1.0, so
# mainhand is derived from evidence (`_last_placed_item`). The game draws the
# answer every frame.
#
# THIS IS ALSO THE OBSERVATION HALF OF A9. The hotbar KEYS were added to the
# macro table so the agent can change its selection; without seeing the
# highlight it cannot tell that anything changed.
#
# SHIPPED DISABLED, and the reason is a fact I could not check rather than a
# preference: it is unverified whether MineRL's POV includes the HUD overlay
# at all. One frame dump from the pod settles it. If the HUD is not rendered
# this sensor reads a strip of world and is worse than nothing, so it stays
# off until somebody looks.
#
# The strip is downsampled HORIZONTALLY ONLY. Vertical detail is what
# distinguishes a half-full hunger shank from a full one; horizontal
# resolution beyond one column per hotbar slot buys nothing.

HUD_ROWS = 12
HUD_COLS = 96
_HUD_FRACTION = 0.12          # bottom share of the frame the HUD occupies


def _read_hud(ctx: Dict) -> Optional[np.ndarray]:
    a = _usable(ctx, "pov_native", min_side=max(HUD_ROWS, 16))
    if a is None:
        return None
    h, w = a.shape[:2]
    band = max(HUD_ROWS, int(round(h * _HUD_FRACTION)))
    strip = a[h - band:]
    # Block-mean to (HUD_ROWS, HUD_COLS) without scipy: index-and-average.
    ri = np.linspace(0, strip.shape[0], HUD_ROWS + 1).astype(int)
    ci = np.linspace(0, strip.shape[1], HUD_COLS + 1).astype(int)
    out = np.empty((HUD_ROWS, HUD_COLS, 3), dtype=np.float32)
    for r in range(HUD_ROWS):
        rs = strip[ri[r]:max(ri[r] + 1, ri[r + 1])]
        for c in range(HUD_COLS):
            out[r, c] = rs[:, ci[c]:max(ci[c] + 1, ci[c + 1])].mean(axis=(0, 1))
    return out.transpose(2, 0, 1).reshape(-1)


# ---------------------------------------------------------------------------
# M1 — AUDIO. The modality the agent does not have at all.
# ---------------------------------------------------------------------------
# WHAT A PLAYER GETS AND SKYBOT DOES NOT:
#
#   * BLOCK-BREAK SOUNDS CARRY MATERIAL IDENTITY. Wood, stone, dirt and glass
#     are unmistakable to a player and indistinguishable to this agent. On a
#     scoreboard reading 13,305 blocks broken for 35 logs, "it cannot hear
#     what it is hitting" is not a small gap.
#   * FOOTSTEPS carry the block underfoot, continuously and for free.
#   * MOB SOUNDS carry identity, DIRECTION and DISTANCE — Minecraft audio is
#     positional and stereo, so it is a SPATIAL sense, not a label channel.
#   * WATER, LAVA AND CAVE AMBIENCE carry structure that is not on screen.
#
# WHY IT IS THE RIGHT KIND OF COMPLEXITY. Sound is raw sensory data. It names
# nothing. Learning that THIS sound co-occurs with THAT visual texture is
# exactly the earned grounding the thesis wants, and it is a textbook
# self-supervised objective — SoundSpaces (arXiv 1912.11474) opens with
# "today's embodied agents are deaf", and audio-visual correspondence
# (Objects that Sound; arXiv 2104.06401) yields object localisation AND
# material prediction with no labels at all.
#
# ---- THE BLOCKER, STATED PLAINLY -----------------------------------------
# MINERL DOES NOT EXPOSE AUDIO. Its observation space is POV + inventory +
# stats; there is public evidence of people trying to add audio to
# MCP-Reborn and no shipped handler. Making this real means MODDING THE JAVA
# CLIENT to tap the sound engine and pipe a buffer through the observation
# socket — days of Java work on the critical path of an environment whose
# boot sequence is already delicate (VirtualGL EGL + per-core taskset).
#
# SO THIS IS THE PYTHON HALF ONLY, and it is deliberately the easy half. It
# reads `info["audio"]` and returns NEUTRAL when the key is absent, which is
# what it will do on every frame until the Java work happens. Shipping it now
# costs nothing and means the agent side is not what blocks the spike.
#
# WHAT THE JAVA SIDE MUST PRODUCE, so the contract is unambiguous:
#   a float32 array of shape (2, AUDIO_MELS, AUDIO_FRAMES) in [0, 1] —
#   two channels (LEFT and RIGHT, because the direction is half the
#   information), a mel-scaled magnitude spectrogram, covering the ticks
#   since the last agent step. Mel rather than raw samples because raw audio
#   at 44.1 kHz is ~7,000 numbers per agent step at action_repeat 4, and
#   because the mel scale is the ear's own transducer.
#
# Registered as an IMAGE sensor, so the bus gives it its own small conv
# encoder exactly like the foveal crop — a spectrogram is a picture of sound
# and the same argument against flat-concatenating pixels applies.

AUDIO_MELS = 32
AUDIO_FRAMES = 16
AUDIO_CHANNELS = 2          # stereo: direction is half the information


def _read_audio(ctx: Dict) -> Optional[np.ndarray]:
    info = ctx.get("info") or {}
    buf = info.get("audio")
    if buf is None:
        return None                      # deaf, and honest about it
    a = np.asarray(buf, dtype=np.float32)
    want = (AUDIO_CHANNELS, AUDIO_MELS, AUDIO_FRAMES)
    if a.shape != want:
        return None
    return np.clip(a, 0.0, 1.0).reshape(-1)
