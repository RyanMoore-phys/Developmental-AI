"""Environment variations that change ONE factor each (plan Stage 12.2).

    factor                     wrapper            what changes
    appearance                 AppearanceChange   one channel's values:
                                                  v' = scale * v[perm] + offset
    dynamics                   DynamicsShift      the adapter's own declared
                                                  dynamics parameters
    controls                   ControlRemap       which command does what
                                                  (discrete permutation, box
                                                  dimension permutation/sign)
    observation availability   ChannelDropout     readings become ABSENT with
                                                  probability p (never 0.0)
    embodiment                 AddActuators       action dimensionality: extra
                                                  actuators the body ignores

Each wrapper is itself an EnvironmentAdapter and passes
check_adapter_conformance (tests/_foundation_transfer_smoke.py), and each
is tested to change only its own factor: the others are bit-identical
under a matched rollout. Scope, episode ids and seq are the inner
adapter's, so stale-action rejection stays the inner adapter's job; spec
and command validation happen HERE, before any mapping, so an out-of-spec
command can never be mapped into an in-spec one.

info["executed_action"] is the action the AGENT issued (in the agent's
command space), stamped with the inner completion time — the agent must
never be shown the remapped command, or a control change would leak.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..contracts import (ABSENT, UNKNOWN, Action, ActionSpec, ContractError,
                         Observation, ObservationSpec, Sentinel)

FACTORS = ("appearance", "dynamics", "controls", "observation_availability",
           "embodiment")


class _Wrapper:
    factor = ""

    def __init__(self, inner):
        self.inner = inner

    # contract -------------------------------------------------------------
    def observation_spec(self) -> ObservationSpec:
        return self.inner.observation_spec()

    def action_spec(self) -> ActionSpec:
        return self.inner.action_spec()

    def reset(self, seed: Optional[int] = None) -> List[Observation]:
        return [self._obs(o) for o in self.inner.reset(seed=seed)]

    def step(self, action: Action):
        if not isinstance(action, Action):
            raise ContractError("step() takes an Action record")
        cmd = self.action_spec().check_action(action)     # validate FIRST
        inner_spec = self.inner.action_spec()
        inner_act = action.replace(spec_id=inner_spec.spec_id,
                                   command=self._command(cmd))
        obs, term, trunc, info = self.inner.step(inner_act)
        info = dict(info)
        ex = info.get("executed_action")
        t_done = ex.t_complete if isinstance(ex, Action) else UNKNOWN
        info["executed_action"] = action.replace(t_complete=t_done)
        return [self._obs(o) for o in obs], term, trunc, info

    # hooks ----------------------------------------------------------------
    def _command(self, cmd):
        return cmd

    def _obs(self, o: Observation) -> Observation:
        return o

    def describe(self) -> Dict[str, Any]:
        return {"factor": self.factor}

    # pass-through for stacked dynamics wrappers
    def dynamics_params(self):
        return self.inner.dynamics_params()

    def set_dynamics(self, **kw):
        return self.inner.set_dynamics(**kw)


class AppearanceChange(_Wrapper):
    factor = "appearance"

    def __init__(self, inner, channel: str, permutation: Optional[Sequence[int]] = None,
                 scale: float = 1.0, offset: float = 0.0):
        super().__init__(inner)
        spec = inner.observation_spec()
        c = spec.channel(channel)
        if np.dtype(c.dtype).kind != "f":
            raise ValueError(f"appearance change needs a float channel; "
                             f"{channel!r} is {c.dtype}")
        if c.provenance == "evaluator":
            raise ValueError("changing an evaluator channel changes no appearance "
                             "the learner sees")
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("scale must be finite and non-zero (invertible)")
        self.channel, self.scale, self.offset = channel, float(scale), float(offset)
        self.perm = None
        if permutation is not None:
            p = np.asarray(permutation, dtype=np.int64)
            if c.kind != "vector" or p.shape != (c.shape[0],) or \
                    sorted(p.tolist()) != list(range(c.shape[0])):
                raise ValueError(f"permutation must permute the {c.shape} vector")
            self.perm = p
        self._dtype = np.dtype(c.dtype)
        neutral = c.neutral if isinstance(c.neutral, Sentinel) else self.transform(c.neutral)
        new_c = c.replace(neutral=neutral,
                          payload=dict(c.payload, variation="appearance"))
        self._spec = spec.replace(channels=tuple(
            new_c if x.name == channel else x for x in spec.channels))

    def transform(self, v):
        a = np.asarray(v)
        if self.perm is not None:
            a = a[self.perm]
        return (self.scale * a + self.offset).astype(self._dtype)

    def invert(self, v):
        a = (np.asarray(v, dtype=np.float64) - self.offset) / self.scale
        if self.perm is not None:
            out = np.empty_like(a)
            out[self.perm] = a
            a = out
        return a.astype(self._dtype)

    def observation_spec(self):
        return self._spec

    def _obs(self, o):
        if o.channel != self.channel or isinstance(o.value, Sentinel):
            return o
        return o.replace(payload=dict(o.payload, value=self.transform(o.value)))

    def describe(self):
        return {"factor": self.factor, "channel": self.channel, "scale": self.scale,
                "offset": self.offset,
                "permutation": None if self.perm is None else self.perm.tolist()}


class DynamicsShift(_Wrapper):
    factor = "dynamics"

    def __init__(self, inner, **params):
        super().__init__(inner)
        if not hasattr(inner, "dynamics_params") or not hasattr(inner, "set_dynamics"):
            raise TypeError(f"{type(inner).__name__} declares no dynamics "
                            f"parameters (dynamics_params/set_dynamics)")
        known = inner.dynamics_params()
        bad = sorted(set(params) - set(known))
        if bad or not params:
            raise ValueError(f"unknown/empty dynamics params {bad or params}; "
                             f"declared: {sorted(known)}")
        self.params = dict(params)
        self.original = dict(known)
        inner.set_dynamics(**self.params)

    def reset(self, seed=None):
        self.inner.set_dynamics(**self.params)
        return super().reset(seed)

    def describe(self):
        return {"factor": self.factor, "from": self.original, "to": self.params}


class ControlRemap(_Wrapper):
    factor = "controls"

    def __init__(self, inner, permutation: Sequence[int],
                 signs: Optional[Sequence[float]] = None):
        super().__init__(inner)
        a = inner.action_spec()
        p = np.asarray(permutation, dtype=np.int64)
        size = a.n if a.kind == "discrete" else (
            int(np.prod(a.shape)) if a.kind == "box" else None)
        if size is None:
            raise ValueError("ControlRemap supports discrete and box actions")
        if p.shape != (size,) or sorted(p.tolist()) != list(range(size)):
            raise ValueError(f"permutation must permute {size} controls")
        self.perm = p
        self.signs = None
        if signs is not None:
            if a.kind != "box":
                raise ValueError("signs apply to box actions only")
            s = np.asarray(signs, dtype=np.float64)
            if s.shape != (size,) or not np.isin(s, (-1.0, 1.0)).all():
                raise ValueError("signs must be +-1 per dimension")
            if not (np.allclose(a.low, -a.high) or (s > 0).all()):
                raise ValueError("sign flips need a symmetric box")
            self.signs = s
        if a.kind == "box" and not (np.allclose(a.low, a.low[self.perm]) and
                                    np.allclose(a.high, a.high[self.perm])):
            raise ValueError("box permutation must map equal bounds onto each other")

    def _command(self, cmd):
        if isinstance(cmd, int):
            return int(self.perm[cmd])
        out = np.asarray(cmd, dtype=np.float64)[self.perm]
        return out * self.signs if self.signs is not None else out

    def inverse_command(self, inner_cmd):
        """The agent command that the inner adapter sees as `inner_cmd`."""
        if isinstance(inner_cmd, (int, np.integer)):
            return int(np.argsort(self.perm)[int(inner_cmd)])
        a = np.asarray(inner_cmd, dtype=np.float64)
        if self.signs is not None:
            a = a * self.signs
        out = np.empty_like(a)
        out[self.perm] = a
        return out

    def describe(self):
        return {"factor": self.factor, "permutation": self.perm.tolist(),
                "signs": None if self.signs is None else self.signs.tolist()}


class ChannelDropout(_Wrapper):
    factor = "observation_availability"

    def __init__(self, inner, channels: Sequence[str], p: float = 1.0, seed: int = 0):
        super().__init__(inner)
        spec = inner.observation_spec()
        for c in channels:
            if spec.channel(c).provenance == "evaluator":
                raise ValueError(f"{c!r} is an evaluator channel; dropping it "
                                 f"changes nothing the learner sees")
        if not (0.0 <= float(p) <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.channels, self.p = tuple(channels), float(p)
        self._rng = np.random.default_rng(seed)    # its own stream

    def _obs(self, o):
        if o.channel not in self.channels or o.value is ABSENT:
            return o
        if self.p >= 1.0 or self._rng.random() < self.p:
            return o.replace(payload=dict(o.payload, value=ABSENT))
        return o

    def describe(self):
        return {"factor": self.factor, "channels": list(self.channels), "p": self.p}


class AddActuators(_Wrapper):
    factor = "embodiment"

    def __init__(self, inner, extra: Sequence[int] = (2,)):
        """discrete n -> multi_discrete (n, *extra); box (d,) -> box (d+k,)
        where k = len(extra) and the extra dims span [-1, 1]. The added
        actuators exist and are validated, but move nothing."""
        super().__init__(inner)
        a = inner.action_spec()
        extra = tuple(int(e) for e in extra)
        if not extra:
            raise ValueError("extra must name at least one actuator")
        self.k = len(extra)
        if a.kind == "discrete":
            self._spec = ActionSpec.multi_discrete(
                f"{a.spec_id}+act{self.k}", (a.n,) + extra,
                units=a.units + ("index",) * self.k, duration_mode=a.duration_mode,
                tick_seconds=a.tick_seconds)
        elif a.kind == "box" and len(a.shape) == 1:
            self._spec = ActionSpec.box(
                f"{a.spec_id}+act{self.k}",
                np.concatenate([a.low, -np.ones(self.k)]),
                np.concatenate([a.high, np.ones(self.k)]),
                a.units + ("unitless",) * self.k, a.duration_mode, a.tick_seconds)
        else:
            raise ValueError("AddActuators supports discrete and 1-D box actions")
        self._inner_kind = a.kind
        self._d = 1 if a.kind == "discrete" else a.shape[0]

    def action_spec(self):
        return self._spec

    def _command(self, cmd):
        if self._inner_kind == "discrete":
            return int(cmd[0])
        return np.asarray(cmd, dtype=np.float64)[: self._d]

    def describe(self):
        return {"factor": self.factor, "extra_actuators": self.k,
                "action_spec": self._spec.spec_id}
