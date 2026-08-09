"""Crafter → Gymnasium adapter (NEXT_OBJECTIVES.md #1 — the richer env).

Crafter (Hafner, 2021): 2D open-ended survival with a 22-achievement tech
tree. Obs IS the game render — (64, 64, 3) uint8 HWC — and the action space
is Discrete(17). It is the natural next world for this project: the crafting
tree is a family of discoverable laws (wood+table→planks, ...), i.e. exactly
what the Rung-9 law-induction machinery and the curiosity/skill stack were
built for.

WHY AN ADAPTER (verified against crafter 1.8.3): crafter uses the LEGACY gym
API — `reset()` returns a bare obs, `step()` returns a 4-tuple with a single
`done`, its spaces are crafter's own DiscreteSpace/BoxSpace (not gymnasium),
and its env ids register with old `gym`, not gymnasium. This whole project is
built on gymnasium 5-tuples and gymnasium spaces (DevelopmentalEnvWrapper
unpacks them directly), so raw crafter crashes at reset.

The adapter also splits crafter's conflated `done` honestly:
  terminated = done AND info["discount"] == 0   (the player DIED)
  truncated  = done AND discount != 0           (the length cap fired)
which keeps the H4 eval semantics and GAE bootstrapping meaningful.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class CrafterEnvAdapter(gym.Env):
    """Gymnasium-API wrapper around ``crafter.Env``.

    Obs: (64, 64, 3) uint8 HWC in [0, 255] (convert downstream — see
    NativePixelObsWrapper). Actions: Discrete(17). Reward: +1 per first-time
    achievement per episode plus a small health delta (CrafterReward), or
    constant 0 (CrafterNoReward — the curiosity-only variant).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        reward: bool = True,
        length: int = 10000,
        size: int = 64,
        seed: Optional[int] = None,
    ):
        super().__init__()
        import crafter  # deferred so the package is optional

        self._crafter = crafter
        self._reward = bool(reward)
        self._length = int(length)
        self._size = int(size)
        self._seed = seed
        self._env = crafter.Env(
            size=(self._size, self._size), reward=self._reward,
            length=self._length, seed=seed)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self._size, self._size, 3),
            dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(17)

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None and seed != self._seed:
            # crafter takes its seed at construction; rebuild for a
            # reproducible episode sequence (world gen happens at reset
            # anyway, so this costs nothing extra).
            self._seed = seed
            self._env = self._crafter.Env(
                size=(self._size, self._size), reward=self._reward,
                length=self._length, seed=seed)
        obs = self._env.reset()
        return np.asarray(obs, dtype=np.uint8), {}

    def step(self, action: int
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(int(action))
        # Honest termination split (see module docstring).
        terminated = bool(done) and float(info.get("discount", 1.0)) == 0.0
        truncated = bool(done) and not terminated
        return (np.asarray(obs, dtype=np.uint8), float(reward),
                terminated, truncated, dict(info))

    def render(self):
        return self._env.render()

    def close(self):
        self._env = None
