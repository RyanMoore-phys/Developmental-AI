"""A minimal NONSPATIAL environment that satisfies the adapter contract.

Plan Stage 2 completion gate: "a minimal nonspatial adapter can satisfy the
same contract without fabricated spatial variables." This is it: a lever
box with no positions, no frames, no geometry of any kind.

    hidden state   counter c >= 0, and a regime r in {0, 1} chosen at reset
    discrete       0 = rest, 1 = lever A, 2 = lever B; the lever matching r
                   increments c, the other decrements it (floored at 0)
    box            one force in [-1, 1]; c += round(2u) under r = 0 and
                   c -= round(2u) under r = 1
    channels       "count"   scalar int64, sensor, required; with probability
                             `dropout` the display fails and reads ABSENT
                   "chime"   scalar bool, sensor, OPTIONAL — emitted only on
                             steps where c lands on a multiple of 3
                   "regime"  scalar int64, EVALUATOR — the hidden truth, for
                             measuring a learner, never for feeding one
    ends           terminated at c >= target; truncated at max_steps;
                   client_recovery every `recovery_every` steps if set

Every channel declares frame=INAPPLICABLE, which is the honest statement:
the question "in which frame?" has no answer here.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import (ABSENT, INAPPLICABLE, Action, ActionSpec, ChannelSpec,
                         ContractError, Observation, ObservationSpec,
                         RecordValidationError)


class NonspatialAdapter:
    ENVIRONMENT = "nonspatial-lever"

    def __init__(self, action_kind: str = "discrete", stream: str = "lever-0",
                 target: int = 5, max_steps: int = 30, dropout: float = 0.1,
                 recovery_every: Optional[int] = None, seed: int = 0,
                 clock: Callable[[], float] = time.time):
        if action_kind not in ("discrete", "box"):
            raise ValueError(f"action_kind must be 'discrete' or 'box', got {action_kind!r}")
        self.stream = stream
        self.target, self.max_steps = int(target), int(max_steps)
        self.dropout = float(dropout)
        self.recovery_every = recovery_every
        self._clock = clock
        self._rng = np.random.default_rng(seed)
        self._episodes = 0
        self._total_steps = 0
        self._episode: Optional[str] = None
        self._seq = 0
        self._done = True
        self._c = 0
        self._r = 0
        if action_kind == "discrete":
            self._aspec = ActionSpec.discrete("lever.discrete", 3,
                                              payload={"labels": ["rest", "A", "B"]})
        else:
            self._aspec = ActionSpec.box("lever.force", [-1.0], [1.0], ("force",))
        na = dict(frame=INAPPLICABLE)
        self._ospec = ObservationSpec(self.ENVIRONMENT, (
            ChannelSpec("count", "scalar", (), "int64", "sensor", True,
                        required=True, units="count", **na),
            ChannelSpec("chime", "scalar", (), "bool", "sensor", True,
                        required=False, units="event", **na),
            ChannelSpec("regime", "scalar", (), "int64", "evaluator", False,
                        required=True, units="index", **na),
        ))

    # ---- contract ------------------------------------------------------
    def observation_spec(self) -> ObservationSpec:
        return self._ospec

    def action_spec(self) -> ActionSpec:
        return self._aspec

    def reset(self, seed: Optional[int] = None) -> List[Observation]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._begin_episode()
        return self._observe()

    def step(self, action: Action) -> Tuple[List[Observation], bool, bool, Dict]:
        if self._done:
            raise ContractError("step() before reset() or after an episode ended")
        if not isinstance(action, Action):
            raise RecordValidationError("step() takes an Action record")
        cmd = self._aspec.check_action(action)      # raises on out-of-spec
        if (action.environment, action.stream, action.episode, action.seq) != (
                self.ENVIRONMENT, self.stream, self._episode, self._seq):
            raise RecordValidationError(
                f"stale action for {action.episode!r}@{action.seq}; current "
                f"is {self._episode!r}@{self._seq}")
        self._total_steps += 1
        if self.recovery_every and self._total_steps % self.recovery_every == 0:
            executed = action.replace(t_complete=self._clock())
            self._begin_episode()
            return self._observe(), False, False, {
                "client_recovery": True, "executed_action": executed}
        if self._aspec.kind == "discrete":
            delta = 0 if cmd == 0 else (1 if cmd - 1 == self._r else -1)
        else:
            delta = int(np.rint(2.0 * float(cmd[0]))) * (1 if self._r == 0 else -1)
        self._c = max(0, self._c + delta)
        self._seq += 1
        terminated = self._c >= self.target
        truncated = (not terminated) and self._seq >= self.max_steps
        self._done = terminated or truncated
        executed = action.replace(t_complete=self._clock())
        return self._observe(), terminated, truncated, {
            "client_recovery": False, "executed_action": executed}

    # ---- internals -----------------------------------------------------
    def _begin_episode(self):
        self._episodes += 1
        self._episode = f"{self.stream}/ep{self._episodes}"
        self._seq = 0
        self._done = False
        self._c = 0
        self._r = int(self._rng.integers(2))

    def _observe(self) -> List[Observation]:
        t_env, t_wall = float(self._seq), float(self._clock())
        base = (self.ENVIRONMENT, self.stream, self._episode, self._seq, t_env, t_wall)
        count = (ABSENT if self._rng.random() < self.dropout
                 else np.int64(self._c))
        out = [Observation(*base, "count", "sensor", {"value": count})]
        if self._seq > 0 and self._c % 3 == 0:
            out.append(Observation(*base, "chime", "sensor", {"value": np.bool_(True)}))
        out.append(Observation(*base, "regime", "evaluator", {"value": np.int64(self._r)}))
        return out
