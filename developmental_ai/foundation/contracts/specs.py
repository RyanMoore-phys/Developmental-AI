"""What an adapter DECLARES: action shapes and observation-channel metadata.

ActionSpec   the available control (plan §5 "Action: available control
             specification"). Continuous box, discrete, or multi-discrete;
             per-dimension units; and explicit duration semantics:

               "tick"            one environment tick per Action; Action.
                                 duration is the tick length if known
               "held"            the command is held for Action.duration
                                 seconds, then released
               "until_complete"  the environment decides when it is done;
                                 Action.t_complete records when

ChannelSpec  one observation channel. REQUIRED metadata: name, kind, shape,
             dtype, provenance, policy_visible, required (is it emitted on
             every step). OPTIONAL metadata, UNKNOWN unless declared: units,
             frame (a local frame name; INAPPLICABLE for a nonspatial
             channel), rate_hz, neutral (the "no information" value).

ObservationSpec  the environment's full channel list. `policy_channels()`
             is THE definition of what learning may consume: evaluator
             channels are excluded by construction — a channel cannot be
             both evaluator-provenance and policy-visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from .base import Record, register_record
from .sentinels import ABSENT, INAPPLICABLE, UNKNOWN, Sentinel
from .values import (PROVENANCES, fail, req_bool, req_enum, req_finite,
                     req_int, req_mapping, req_str, req_str_tuple, req_tuple)

ACTION_KINDS = ("box", "discrete", "multi_discrete")
DURATION_MODES = ("tick", "held", "until_complete")
CHANNEL_KINDS = ("scalar", "vector", "image", "discrete")
# An environment adapter reports what it sensed, what it derived, or what
# only an evaluator may see. It never emits imagined data.
CHANNEL_PROVENANCES = ("sensor", "inferred", "evaluator")


def _shape(owner, name, v) -> Tuple[int, ...]:
    t = req_tuple(owner, name, v)
    for d in t:
        req_int(owner, f"{name} entry", d, minimum=1)
    return tuple(int(d) for d in t)


@register_record
@dataclass(frozen=True, eq=False)
class ActionSpec(Record):
    SCHEMA = "action_spec"
    VERSION = 1

    spec_id: str
    kind: str
    shape: Tuple[int, ...]
    low: Any            # box: float64 array of `shape`; else INAPPLICABLE
    high: Any
    n: Any              # discrete: int >= 1; else INAPPLICABLE
    nvec: Any           # multi_discrete: tuple of int >= 1; else INAPPLICABLE
    units: Tuple[str, ...]
    duration_mode: str
    tick_seconds: Any = UNKNOWN
    payload: Dict[str, Any] = field(default_factory=dict)

    # ---- constructors --------------------------------------------------
    @classmethod
    def box(cls, spec_id, low, high, units, duration_mode="tick",
            tick_seconds=UNKNOWN, payload=None):
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        return cls(spec_id, "box", tuple(low.shape), low, high, INAPPLICABLE,
                   INAPPLICABLE, tuple(units), duration_mode, tick_seconds,
                   dict(payload or {}))

    @classmethod
    def discrete(cls, spec_id, n, units=("index",), duration_mode="tick",
                 tick_seconds=UNKNOWN, payload=None):
        return cls(spec_id, "discrete", (), INAPPLICABLE, INAPPLICABLE, n,
                   INAPPLICABLE, tuple(units), duration_mode, tick_seconds,
                   dict(payload or {}))

    @classmethod
    def multi_discrete(cls, spec_id, nvec, units=None, duration_mode="tick",
                       tick_seconds=UNKNOWN, payload=None):
        nvec = tuple(nvec)
        units = tuple(units) if units is not None else ("index",) * len(nvec)
        return cls(spec_id, "multi_discrete", (len(nvec),), INAPPLICABLE,
                   INAPPLICABLE, INAPPLICABLE, nvec, units, duration_mode,
                   tick_seconds, dict(payload or {}))

    # ---- validation ----------------------------------------------------
    def _validate(self):
        o = "ActionSpec"
        req_str(o, "spec_id", self.spec_id)
        req_enum(o, "kind", self.kind, ACTION_KINDS)
        self._set("shape", _shape(o, "shape", self.shape))
        self._set("units", req_str_tuple(o, "units", self.units))
        req_enum(o, "duration_mode", self.duration_mode, DURATION_MODES)
        self._set("tick_seconds", req_finite(o, "tick_seconds", self.tick_seconds,
                                             allow=(UNKNOWN, INAPPLICABLE)))
        if not isinstance(self.tick_seconds, Sentinel) and self.tick_seconds <= 0:
            fail(o, f"tick_seconds must be > 0, got {self.tick_seconds}")
        self._set("payload", req_mapping(o, "payload", self.payload))
        if self.kind == "box":
            for nm in ("low", "high"):
                a = getattr(self, nm)
                if not isinstance(a, np.ndarray) or a.dtype.kind != "f":
                    fail(o, f"box {nm} must be a float ndarray")
                if a.shape != self.shape:
                    fail(o, f"box {nm} shape {a.shape} != shape {self.shape}")
                if np.isnan(a).any():
                    fail(o, f"box {nm} contains NaN")
                self._set(nm, a.astype(np.float64))
            if (self.low > self.high).any():
                fail(o, "box low > high")
            for nm in ("n", "nvec"):
                if getattr(self, nm) is not INAPPLICABLE:
                    fail(o, f"box spec must have {nm}=INAPPLICABLE")
            ndim = int(np.prod(self.shape)) if self.shape else 1
        elif self.kind == "discrete":
            self._set("n", req_int(o, "n", self.n, minimum=1))
            if self.shape != ():
                fail(o, "discrete spec shape must be ()")
            for nm in ("low", "high", "nvec"):
                if getattr(self, nm) is not INAPPLICABLE:
                    fail(o, f"discrete spec must have {nm}=INAPPLICABLE")
            ndim = 1
        else:
            nv = req_tuple(o, "nvec", self.nvec, nonempty=True)
            self._set("nvec", tuple(req_int(o, "nvec entry", x, minimum=1)
                                    for x in nv))
            if self.shape != (len(self.nvec),):
                fail(o, f"multi_discrete shape must be ({len(self.nvec)},)")
            for nm in ("low", "high", "n"):
                if getattr(self, nm) is not INAPPLICABLE:
                    fail(o, f"multi_discrete spec must have {nm}=INAPPLICABLE")
            ndim = len(self.nvec)
        if len(self.units) != ndim:
            fail(o, f"units has {len(self.units)} entries for {ndim} "
                    f"action dimension(s)")

    # ---- use -----------------------------------------------------------
    def validate_command(self, command):
        """Return the command in canonical form, or raise
        RecordValidationError. Canonical: box -> float64 ndarray of
        `shape`; discrete -> int; multi_discrete -> tuple of int."""
        o = f"ActionSpec[{self.spec_id}]"
        if self.kind == "box":
            if isinstance(command, (bool, np.bool_)):
                fail(o, "box command may not be bool")
            try:
                a = np.asarray(command, dtype=np.float64)
            except (TypeError, ValueError):
                fail(o, f"box command not numeric: {command!r}")
            if a.shape != self.shape:
                fail(o, f"box command shape {a.shape} != {self.shape}")
            if not np.isfinite(a).all():
                fail(o, "box command must be finite")
            if (a < self.low).any() or (a > self.high).any():
                fail(o, f"box command {a.tolist()} outside "
                        f"[{self.low.tolist()}, {self.high.tolist()}]")
            return a
        if self.kind == "discrete":
            c = req_int(o, "command", command, minimum=0)
            if c >= self.n:
                fail(o, f"discrete command {c} >= n={self.n}")
            return c
        if isinstance(command, np.ndarray):
            command = command.tolist()
        t = req_tuple(o, "command", command)
        if len(t) != len(self.nvec):
            fail(o, f"multi_discrete command has {len(t)} entries, "
                    f"expected {len(self.nvec)}")
        out = tuple(req_int(o, "command entry", x, minimum=0) for x in t)
        for x, n in zip(out, self.nvec):
            if x >= n:
                fail(o, f"multi_discrete entry {x} >= {n}")
        return out

    def sample(self, rng: np.random.Generator):
        if self.kind == "box":
            lo = np.where(np.isfinite(self.low), self.low, -1.0)
            hi = np.where(np.isfinite(self.high), self.high, 1.0)
            return rng.uniform(lo, hi).astype(np.float64).reshape(self.shape)
        if self.kind == "discrete":
            return int(rng.integers(self.n))
        return tuple(int(rng.integers(n)) for n in self.nvec)

    def check_action(self, action) -> Any:
        """Validate an Action against this spec; returns canonical command."""
        if action.spec_id != self.spec_id:
            fail(f"ActionSpec[{self.spec_id}]",
                 f"action targets spec {action.spec_id!r}")
        return self.validate_command(action.command)


@register_record
@dataclass(frozen=True, eq=False)
class ChannelSpec(Record):
    SCHEMA = "channel_spec"
    VERSION = 1

    name: str
    kind: str
    shape: Tuple[int, ...]
    dtype: str
    provenance: str
    policy_visible: bool
    required: bool = True
    units: Any = UNKNOWN
    frame: Any = UNKNOWN        # local frame name (str), UNKNOWN, or INAPPLICABLE
    rate_hz: Any = UNKNOWN
    neutral: Any = UNKNOWN
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = f"ChannelSpec[{self.name!r}]"
        req_str(o, "name", self.name)
        req_enum(o, "kind", self.kind, CHANNEL_KINDS)
        self._set("shape", _shape(o, "shape", self.shape))
        req_str(o, "dtype", self.dtype)
        try:
            dt = np.dtype(self.dtype)
        except TypeError:
            fail(o, f"dtype {self.dtype!r} is not a numpy dtype")
        if dt.kind not in "biuf":
            fail(o, f"dtype {self.dtype!r} must be bool/int/float")
        self._set("dtype", dt.name)
        req_enum(o, "provenance", self.provenance, CHANNEL_PROVENANCES)
        self._set("policy_visible", req_bool(o, "policy_visible", self.policy_visible))
        self._set("required", req_bool(o, "required", self.required))
        if self.provenance == "evaluator" and self.policy_visible:
            fail(o, "an evaluator channel can never be policy_visible")
        if self.kind == "scalar" and self.shape != ():
            fail(o, "scalar channel must have shape ()")
        if self.kind == "image" and len(self.shape) != 3:
            fail(o, "image channel must have shape (C, H, W)")
        if not isinstance(self.units, Sentinel):
            req_str(o, "units", self.units)
        if not isinstance(self.frame, Sentinel):
            req_str(o, "frame", self.frame)
        self._set("rate_hz", req_finite(o, "rate_hz", self.rate_hz,
                                        allow=(UNKNOWN, INAPPLICABLE)))
        if self.neutral is not UNKNOWN:
            probs = self.value_problems(self.neutral)
            if probs:
                fail(o, f"neutral: {probs[0]}")
        self._set("payload", req_mapping(o, "payload", self.payload))

    def value_problems(self, value) -> list:
        """Why `value` cannot be this channel's reading ([] if it can).
        ABSENT and UNKNOWN are always acceptable readings: a missing value
        is reported, never fabricated."""
        if value is ABSENT or value is UNKNOWN:
            return []
        if isinstance(value, Sentinel):
            return [f"{value!r} is not a reading"]
        a = np.asarray(value)
        if a.shape != self.shape:
            return [f"shape {a.shape} != declared {self.shape}"]
        if a.dtype != np.dtype(self.dtype):
            return [f"dtype {a.dtype} != declared {self.dtype}"]
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            return ["non-finite reading"]
        return []


@register_record
@dataclass(frozen=True, eq=False)
class ObservationSpec(Record):
    SCHEMA = "observation_spec"
    VERSION = 1

    environment: str
    channels: Tuple[ChannelSpec, ...]
    payload: Dict[str, Any] = field(default_factory=dict)

    def _validate(self):
        o = "ObservationSpec"
        req_str(o, "environment", self.environment)
        self._set("channels", req_tuple(o, "channels", self.channels,
                                        elem=ChannelSpec, nonempty=True))
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            fail(o, f"duplicate channel names in {names}")
        self._set("payload", req_mapping(o, "payload", self.payload))

    def channel(self, name: str) -> ChannelSpec:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)

    def names(self):
        return tuple(c.name for c in self.channels)

    def policy_channels(self):
        """Names learning may consume. Never contains an evaluator channel."""
        return tuple(c.name for c in self.channels
                     if c.policy_visible and c.provenance != "evaluator")

    def evaluator_channels(self):
        return tuple(c.name for c in self.channels if c.provenance == "evaluator")

    def frames(self):
        """Declared frame names. Empty for an environment with no geometry."""
        return tuple(sorted({c.frame for c in self.channels
                             if isinstance(c.frame, str)}))


assert set(CHANNEL_PROVENANCES) <= set(PROVENANCES)
