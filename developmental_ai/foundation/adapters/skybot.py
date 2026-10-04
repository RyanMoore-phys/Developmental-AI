"""SkyBot's existing data, carried as records — no Minecraft required.

Plan Stage 2 gate: "existing SkyBot data can travel through the
interfaces." Two sources exist today and both are covered:

  * a live sensor-bus context (the `ctx` dict the loop builds per step)
    -> `observations_from_ctx`: one Observation per enabled sensor
  * a stored replay row (the bus's flat transport vector)
    -> `observations_from_transport`: one Observation per policy sensor

and the reverse, `transport_from_observations`, rebuilds the exact
transport vector `SensorBus.read_policy` would have produced, so a round
trip through records is bit-identical (contract-tested).

RED STAYS RED. The bus already keeps RED (oracle) sensors out of the
transport structurally. The records keep it that way: a RED sensor becomes
an `evaluator`-provenance channel that is never policy_visible, its
readings carry provenance "evaluator", `policy_observations` drops them,
Evidence refuses to mix them into sensor evidence, transition_valid refuses
them as transitions, and `transport_from_observations` refuses outright to
place one in the policy vector — judged by the CHANNEL's declaration, so
relabelling a RED reading's provenance does not smuggle it through.

ABSENT IS NOT NEUTRAL. `Sensor.read` substitutes the neutral value when a
transducer has no reading; that is right for the transport (a row must
exist) and wrong for a record (the absence becomes a claim). The record
path therefore reads the raw transducer and reports ABSENT; neutral is
restored only when rebuilding the transport. This reads `Sensor._read`, the
one private attribute used, because the bus offers no public raw read and
this layer must not modify `developmental_ai/sensors`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..contracts import (ABSENT, UNKNOWN, ChannelSpec, ContractError,
                         Observation, ObservationSpec)

_RED = "red"


class SkyBotRecordAdapter:
    def __init__(self, bus, environment: str = "minecraft",
                 stream: str = "skybot-0"):
        self.bus = bus
        self.environment = environment
        self.stream = stream
        sensors = list(bus.policy_sensors()) + list(bus.oracle_sensors())
        self._sensors = {s.name: s for s in sensors}
        chans = []
        for s in sensors:
            red = s.classification == _RED
            shape = tuple(s.shape) if s.kind == "image" else (s.width,)
            chans.append(ChannelSpec(
                s.name, "image" if s.kind == "image" else "vector", shape,
                "float32", "evaluator" if red else "sensor", not red,
                required=True, units=UNKNOWN, frame=UNKNOWN,
                neutral=s.neutral().reshape(shape),
                payload={"classification": s.classification,
                         "sensor_version": int(s.version)}))
        self._spec = ObservationSpec(environment, tuple(chans), payload={
            "layout_hash": bus.layout_hash(), "transport_width": int(bus.width)})

    def observation_spec(self) -> ObservationSpec:
        return self._spec

    # ---- live context -> records ----------------------------------------
    def _raw(self, s, ctx):
        try:
            v = s._read(ctx)
        except Exception:
            return ABSENT
        if v is None:
            return ABSENT
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] != s.width or not np.all(np.isfinite(v)):
            return ABSENT
        return v.reshape(self._spec.channel(s.name).shape)

    def _obs(self, name, value, episode, seq, t_env, t_wall, origin):
        c = self._spec.channel(name)
        return Observation(self.environment, self.stream, str(episode), int(seq),
                           t_env, t_wall, name, c.provenance,
                           {"value": value, "origin": origin})

    def observations_from_ctx(self, ctx: Dict, episode, seq: int, t_env: float,
                              t_wall: float) -> List[Observation]:
        """Every enabled sensor (policy AND oracle), as records."""
        return [self._obs(c.name, self._raw(self._sensors[c.name], ctx),
                          episode, seq, t_env, t_wall, "sensor_bus")
                for c in self._spec.channels]

    # ---- stored transport row <-> records --------------------------------
    def _transport_layout(self):
        out = [(n, off, w) for n, off, w in self.bus.layout()]
        base = self.bus.vector_width
        out += [(n, base + off, w) for n, off, w, _shp in self.bus.image_layout()]
        return out

    def observations_from_transport(self, vec, episode, seq: int, t_env: float,
                                    t_wall: float,
                                    layout_hash: Optional[str] = None,
                                    origin: str = "replay_transport") -> List[Observation]:
        """A transport row -> one Observation per policy sensor.

        The row is a replay-buffer sensor row (origin "replay_transport") or
        the live `info["sensors"]` vector the shadow recorder reads
        (runtime/shadow.py passes origin "live_transport"); the bytes are the
        same transport either way. A row cannot distinguish "neutral because
        absent" from a real neutral reading, so values are carried as stored.
        A mismatched layout hash raises rather than reading fields at the
        wrong offsets.
        """
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.bus.width:
            raise ContractError(f"transport row has {vec.shape[0]} values, bus "
                                f"width is {self.bus.width}")
        if layout_hash is not None and layout_hash != self.bus.layout_hash():
            raise ContractError(f"transport layout {layout_hash} != bus "
                                f"{self.bus.layout_hash()}")
        return [self._obs(n, vec[off:off + w].reshape(self._spec.channel(n).shape),
                          episode, seq, t_env, t_wall, origin)
                for n, off, w in self._transport_layout()]

    def transport_from_observations(self, observations) -> np.ndarray:
        """Rebuild `bus.read_policy` output from records. Refuses evaluator
        data, duplicate or missing policy channels; ABSENT -> neutral."""
        by = {}
        for o in observations:
            c = self._spec.channel(o.channel)
            if c.provenance == "evaluator" or o.provenance == "evaluator" \
                    or not c.policy_visible:
                raise ContractError(
                    f"refusing to place evaluator channel {o.channel!r} in the "
                    f"policy transport")
            if o.channel in by:
                raise ContractError(f"duplicate channel {o.channel!r}")
            by[o.channel] = o
        parts = []
        for n, _off, w in self._transport_layout():
            if n not in by:
                raise ContractError(f"policy channel {n!r} missing")
            v = by[n].value
            if v is ABSENT or v is UNKNOWN:
                v = self._sensors[n].neutral()
            parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
        if not parts:
            return np.zeros(0, np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)
