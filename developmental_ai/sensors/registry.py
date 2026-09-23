"""The SENSOR BUS — one registry for every transducer the agent has.

WHY THIS EXISTS (2026-09-19)
    The agent's observation tensor is pure pixels and `obs_dim` is
    load-bearing: minerl_env.py's create_observables records that location
    and life stats were deliberately routed through `info` rather than the
    obs tensor precisely so "obs_dim is unchanged and every saved skill
    policy stays loadable". Wave 1 then added ONE side-channel — the 13-field
    proprio column — with optional-on-load semantics so the pod's buffer
    survived the schema change.

    That pattern works. It does not scale: every further sense (a foveal
    crop, a damage vignette, a depth map, an occupancy grid) would repeat the
    same migration against the buffer, the world model and both loop bodies.
    This generalises the one column into a REGISTRY, so adding a sense is a
    registration rather than a schema change.

THE RULE THIS CLASS ENFORCES
    A SENSE is a transducer: it reports a physical quantity the agent's own
    body or its own screen makes available, WITHOUT NAMING WHAT ANYTHING IS.
    A MEANING names, classifies or identifies.

        GREEN  a pure sense                      -> policy input
        AMBER  a sense that leaks structure the
               agent should have earned          -> opt-in, default OFF
        RED    a meaning (block names, entity
               lists, true coordinates)          -> EVALUATION ONLY

    RED is excluded STRUCTURALLY, not by convention: `read_policy` builds the
    transport vector and simply never visits a RED sensor, so there is no
    code path along which a meaning can reach the world model. RED values
    travel in a separate dict that the replay buffer does not carry. A caller
    cannot "forget to filter" because filtering is not a step.

WHY "PLAYER-PERCEIVABLE" IS NOT THE TEST
    A player can press F3 and read `minecraft:oak_log`, so that string is
    player-perceivable and still forbidden. The thesis is not "only what a
    player can see"; it is "meaning is earned, never declared". A human
    reading `oak_log` already knows English and already knows what a log is —
    the string indexes knowledge earned elsewhere. SkyBot has no elsewhere.

TWO KINDS OF SENSOR, AND WHY THE DISTINCTION IS LOAD-BEARING
    VECTOR sensors (proprio, screen effects) are a handful of scalars and are
    concatenated straight onto the encoder embedding.

    IMAGE sensors (the native-resolution foveal crop) are thousands of
    numbers. Flat-concatenating 3072 raw pixels next to 13 body scalars would
    drown the scalars outright — the config already flags a 17:1 compression
    as a problem at far gentler ratios. Image sensors therefore declare a
    (C, H, W) shape and are given their OWN small convolutional encoder by
    the world model, so each modality arrives at comparable width.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GREEN = "green"
AMBER = "amber"
RED = "red"

_CLASSES = (GREEN, AMBER, RED)

# The proprio sensor keeps its historical name because the replay buffer
# column, the world-model kwarg and both loop bodies all spell it that way.
# Renaming the wire format to match the new concept would be churn with no
# behavioural gain and a real chance of silently misaligning a restored
# buffer.
PROPRIO_SENSOR_NAME = "proprio"


class Sensor:
    """One named transducer.

    `read(ctx) -> np.ndarray` returns EXACTLY `width` float32 values, or None
    when this step has no reading. None is not an error: a client rebuilding
    mid-step, a scout with no body, a frame that failed to render — all of
    them must still produce a row, and `neutral()` is what they produce.

    `neutral` is the value that means "no information", which is NOT always
    zero. A ratio whose natural midpoint is 0.5 must read 0.5 when absent, or
    the absence itself becomes a claim.
    """

    __slots__ = ("name", "width", "classification", "version", "_read",
                 "_neutral", "shape", "kind")

    def __init__(self, name: str, width: int, read: Callable,
                 classification: str = GREEN, version: int = 1,
                 neutral: Optional[np.ndarray] = None,
                 shape: Optional[Tuple[int, int, int]] = None):
        if classification not in _CLASSES:
            raise ValueError(
                f"sensor {name!r}: classification must be one of {_CLASSES}, "
                f"got {classification!r}")
        if width <= 0:
            raise ValueError(f"sensor {name!r}: width must be positive")
        if shape is not None:
            c, h, w = shape
            if c * h * w != width:
                raise ValueError(
                    f"sensor {name!r}: shape {shape} holds {c * h * w} values "
                    f"but width is {width}")
            if classification == RED:
                raise ValueError(
                    f"sensor {name!r}: an image sensor cannot be RED — RED "
                    f"values never reach a network, so a shape is meaningless")
        self.name = str(name)
        self.width = int(width)
        self.classification = classification
        self.version = int(version)
        self._read = read
        self.shape = shape
        self.kind = "image" if shape is not None else "vector"
        self._neutral = (np.zeros(self.width, np.float32) if neutral is None
                         else np.asarray(neutral, np.float32).reshape(-1))
        if self._neutral.shape[0] != self.width:
            raise ValueError(
                f"sensor {name!r}: neutral has {self._neutral.shape[0]} "
                f"values, expected {self.width}")

    def neutral(self) -> np.ndarray:
        return self._neutral.copy()

    def read(self, ctx: Dict) -> np.ndarray:
        """Never raises. A sensor that throws reads NEUTRAL and logs once —
        a broken transducer must not be able to stop the agent from acting."""
        try:
            v = self._read(ctx)
        except Exception as e:                      # pragma: no cover
            logger.debug("sensor %s failed: %s", self.name, e)
            return self.neutral()
        if v is None:
            return self.neutral()
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.width:
            logger.warning(
                "sensor %s returned %d values, expected %d — reading NEUTRAL",
                self.name, v.shape[0], self.width)
            return self.neutral()
        if not np.all(np.isfinite(v)):
            # A NaN here would propagate into the RSSM posterior and poison
            # every latent downstream of it, silently and permanently.
            logger.warning("sensor %s returned non-finite values", self.name)
            return self.neutral()
        return v


class SensorBus:
    """An ordered, hashed collection of sensors.

    LAYOUT HASH. The transport vector is a flat concatenation, so its meaning
    depends entirely on the order and widths of its fields. The hash pins
    exactly that. It is written into the buffer manifest and the world-model
    checkpoint; on mismatch the stored column restores as NEUTRAL rather than
    being read with the wrong field boundaries. The precedent is Wave 1's
    proprio width guard, which already refuses to carry misaligned senses.
    """

    def __init__(self, sensors: Sequence[Sensor], enabled: Optional[Sequence[str]] = None):
        self._all: Dict[str, Sensor] = {}
        for s in sensors:
            if s.name in self._all:
                raise ValueError(f"duplicate sensor name {s.name!r}")
            self._all[s.name] = s
        self._order: List[str] = [s.name for s in sensors]
        # None means "the default set": every GREEN sensor, no AMBER, no RED.
        # AMBER is never on by default — that is the whole point of the
        # classification.
        if enabled is None:
            self._enabled = {n for n in self._order
                             if self._all[n].classification == GREEN}
        else:
            want = set(enabled)
            unknown = want - set(self._order)
            if unknown:
                raise ValueError(f"unknown sensors enabled: {sorted(unknown)}")
            self._enabled = want

    # ---- introspection ------------------------------------------------
    def names(self) -> List[str]:
        return list(self._order)

    def get(self, name: str) -> Sensor:
        return self._all[name]

    def policy_sensors(self) -> List[Sensor]:
        """Enabled, non-RED, in registration order. THE definition of what
        the policy and world model may see."""
        return [self._all[n] for n in self._order
                if n in self._enabled
                and self._all[n].classification != RED]

    def oracle_sensors(self) -> List[Sensor]:
        return [self._all[n] for n in self._order
                if n in self._enabled and self._all[n].classification == RED]

    def vector_sensors(self) -> List[Sensor]:
        return [s for s in self.policy_sensors() if s.kind == "vector"]

    def image_sensors(self) -> List[Sensor]:
        return [s for s in self.policy_sensors() if s.kind == "image"]

    def layout(self) -> List[Tuple[str, int, int]]:
        """[(name, offset, width)] over the VECTOR transport only."""
        out, off = [], 0
        for s in self.vector_sensors():
            out.append((s.name, off, s.width))
            off += s.width
        return out

    def image_layout(self) -> List[Tuple[str, int, int, Tuple[int, int, int]]]:
        out, off = [], 0
        for s in self.image_sensors():
            out.append((s.name, off, s.width, s.shape))
            off += s.width
        return out

    @property
    def vector_width(self) -> int:
        return sum(s.width for s in self.vector_sensors())

    @property
    def image_width(self) -> int:
        return sum(s.width for s in self.image_sensors())

    @property
    def width(self) -> int:
        """Total transport width: vectors first, then images."""
        return self.vector_width + self.image_width

    def layout_hash(self) -> str:
        """Stable over (name, width, version, kind, shape), in order.

        Deliberately NOT over `classification`: reclassifying a sensor from
        GREEN to AMBER changes whether it is enabled by default, not what the
        stored bytes mean, and invalidating a pod buffer over a policy
        decision would be exactly the kind of cost this project does not pay.
        """
        h = hashlib.sha256()
        for s in self.policy_sensors():
            h.update(f"{s.name}:{s.width}:{s.version}:{s.kind}:"
                     f"{s.shape}\n".encode())
        return h.hexdigest()[:16]

    # ---- reading ------------------------------------------------------
    def read_policy(self, ctx: Dict) -> np.ndarray:
        """The transport vector: VECTORS first, then IMAGES, in order.

        RED sensors are not visited. There is no filter here to forget.
        """
        parts = [s.read(ctx) for s in self.vector_sensors()]
        parts += [s.read(ctx) for s in self.image_sensors()]
        if not parts:
            return np.zeros(0, np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def neutral_policy(self) -> np.ndarray:
        parts = [s.neutral() for s in self.vector_sensors()]
        parts += [s.neutral() for s in self.image_sensors()]
        if not parts:
            return np.zeros(0, np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def read_oracle(self, ctx: Dict) -> Dict[str, np.ndarray]:
        """RED readings, by name. Goes to the evaluation sink and NOWHERE
        else — in particular never into the replay buffer's sensor column."""
        return {s.name: s.read(ctx) for s in self.oracle_sensors()}

    def slice_of(self, name: str, vec: np.ndarray) -> Optional[np.ndarray]:
        """Pull one sensor's fields back out of a transport vector."""
        off = 0
        for s in self.vector_sensors():
            if s.name == name:
                return vec[off:off + s.width]
            off += s.width
        for s in self.image_sensors():
            if s.name == name:
                return vec[off:off + s.width]
            off += s.width
        return None

    def describe(self) -> str:
        v = ", ".join(f"{n}[{w}]" for n, _o, w in self.layout()) or "-"
        i = ", ".join(f"{n}{shp}" for n, _o, _w, shp in self.image_layout()) or "-"
        r = ", ".join(s.name for s in self.oracle_sensors()) or "-"
        return (f"vectors: {v} | images: {i} | oracle(RED, never input): {r} "
                f"| width {self.width} | hash {self.layout_hash()}")
