"""A second, DIFFERENT nonspatial adapter: cue -> response (plan Stage 12.1).

The lever box (adapters/nonspatial.py) has scalar int channels, an
optional event channel and a hidden binary regime. This one differs on the
axes a learning core could accidentally depend on:

    observation  "cue"     vector float32 (K,), one-hot, sensor, required
                 "reward"  scalar float64, sensor, required; ABSENT at seq 0
                           (no action has been taken yet — absent, not zero)
                 "rule"    scalar int64, EVALUATOR: the hidden shift
    control      discrete K ("respond with symbol k")
    dynamics     reward 1 iff response == (cue + shift) mod K, plus optional
                 Gaussian reward noise; `shift` is the dynamics parameter
                 that DynamicsShift changes (dynamics_params/set_dynamics)
    ends         never terminates; truncated at episode_len

Random streams are separated so a variation of one factor cannot move
another: the cue sequence draws only from the cue stream (seeded at reset),
the reward noise only from the noise stream. Frames are INAPPLICABLE.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import (ABSENT, INAPPLICABLE, Action, ActionSpec, ChannelSpec,
                         ContractError, Observation, ObservationSpec,
                         RecordValidationError)


class CueResponseAdapter:
    ENVIRONMENT = "nonspatial-cue-response"

    def __init__(self, n_cues: int = 6, shift: int = 1, episode_len: int = 20,
                 reward_noise: float = 0.0, stream: str = "cue-0",
                 seed: int = 0, clock: Callable[[], float] = time.time):
        if isinstance(n_cues, bool) or not isinstance(n_cues, int) or n_cues < 2:
            raise ValueError(f"n_cues must be an int >= 2, got {n_cues!r}")
        if episode_len < 1:
            raise ValueError("episode_len must be >= 1")
        if reward_noise < 0:
            raise ValueError("reward_noise must be >= 0")
        self.n_cues, self.episode_len = int(n_cues), int(episode_len)
        self.reward_noise = float(reward_noise)
        self.stream = stream
        self._clock = clock
        self.set_dynamics(shift=shift)
        self._cue_rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng(seed + 1_000_003)
        self._episodes = 0
        self._episode: Optional[str] = None
        self._seq = 0
        self._done = True
        self._cue = 0
        self._last_reward = ABSENT
        self._aspec = ActionSpec.discrete("cue.respond", self.n_cues,
                                          units=("symbol",))
        na = dict(frame=INAPPLICABLE)
        self._ospec = ObservationSpec(self.ENVIRONMENT, (
            ChannelSpec("cue", "vector", (self.n_cues,), "float32", "sensor",
                        True, required=True, units="one-hot", **na),
            ChannelSpec("reward", "scalar", (), "float64", "sensor", True,
                        required=True, units="reward", **na),
            ChannelSpec("rule", "scalar", (), "int64", "evaluator", False,
                        required=True, units="index", **na),
        ))

    # ---- dynamics parameters (consumed by variations.DynamicsShift) -----
    def dynamics_params(self) -> Dict[str, int]:
        return {"shift": self.shift}

    def set_dynamics(self, shift: int) -> None:
        if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
            raise ValueError(f"shift must be an int, got {shift!r}")
        self.shift = int(shift) % self.n_cues

    # ---- contract -------------------------------------------------------
    def observation_spec(self) -> ObservationSpec:
        return self._ospec

    def action_spec(self) -> ActionSpec:
        return self._aspec

    def reset(self, seed: Optional[int] = None) -> List[Observation]:
        if seed is not None:
            self._cue_rng = np.random.default_rng(seed)
            self._noise_rng = np.random.default_rng(seed + 1_000_003)
        self._episodes += 1
        self._episode = f"{self.stream}/ep{self._episodes}"
        self._seq = 0
        self._done = False
        self._last_reward = ABSENT
        self._cue = int(self._cue_rng.integers(self.n_cues))
        return self._observe()

    def step(self, action: Action) -> Tuple[List[Observation], bool, bool, Dict]:
        if self._done:
            raise ContractError("step() before reset() or after an episode ended")
        if not isinstance(action, Action):
            raise RecordValidationError("step() takes an Action record")
        cmd = self._aspec.check_action(action)
        if (action.environment, action.stream, action.episode, action.seq) != (
                self.ENVIRONMENT, self.stream, self._episode, self._seq):
            raise RecordValidationError(
                f"stale action for {action.episode!r}@{action.seq}; current "
                f"is {self._episode!r}@{self._seq}")
        r = 1.0 if cmd == (self._cue + self.shift) % self.n_cues else 0.0
        if self.reward_noise > 0:
            r += float(self._noise_rng.normal(scale=self.reward_noise))
        self._last_reward = np.float64(r)
        self._seq += 1
        self._cue = int(self._cue_rng.integers(self.n_cues))
        truncated = self._seq >= self.episode_len
        self._done = truncated
        return self._observe(), False, truncated, {
            "client_recovery": False,
            "executed_action": action.replace(t_complete=max(
                float(self._clock()), action.t_dispatch))}

    def _observe(self) -> List[Observation]:
        base = (self.ENVIRONMENT, self.stream, self._episode, self._seq,
                float(self._seq), float(self._clock()))
        cue = np.zeros(self.n_cues, dtype=np.float32)
        cue[self._cue] = 1.0
        return [Observation(*base, "cue", "sensor", {"value": cue}),
                Observation(*base, "reward", "sensor", {"value": self._last_reward}),
                Observation(*base, "rule", "evaluator",
                            {"value": np.int64(self.shift)})]
