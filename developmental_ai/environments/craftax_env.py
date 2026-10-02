"""Craftax → Gymnasium adapter (NEXT_OBJECTIVES.md — the deeper env).

Craftax (Matthews et al., 2024): a JAX reimplementation and EXPANSION of
Crafter — 9 procedurally-generated dungeon floors, 60+ achievements, 43
actions, ladders/bosses/potions. The genuine content step-up from Crafter for
the cumulative-learning + law-induction goal (a far deeper tech/achievement
tree, and a richer family of discoverable rules).

WHY AN ADAPTER: Craftax is JAX-FUNCTIONAL, not OO-gym. Its env is a pure
function pair:
    obs, state            = env.reset(key, params)
    obs, state, r, done, i = env.step(key, state, action, params)
with an explicit PRNG key and an explicitly-threaded immutable `state`. This
project's loop is a stateful `import gymnasium as gym` OO loop over numpy. The
adapter holds (key, state, params) internally, jits `step` for speed, and
converts JAX arrays → numpy at the boundary, presenting the gymnasium API.

REPRESENTATION: we use the SYMBOLIC variant (`Craftax-Symbolic-v1`): obs is a
flat (8268,) float32 [0,1] FACTORED state vector (tile map + inventory + mobs
+ status), which (a) suits this project's structured/law-induction goal better
than raw pixels, (b) avoids the full pixel variant's non-square 130x110 frame
(our CNN path is square-only) and its aggressive downsampling, and (c) reuses
the well-worn MLP (vector) world-model path. Set pixel_obs=false in config.

DEVICE NOTE: jaxlib on the training host is CPU-only, so Craftax steps on the CPU while
torch trains on the GPU — a clean split, no GPU contention, ~300+ steps/s.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class CraftaxEnvAdapter(gym.Env):
    """Gymnasium-API wrapper around a Craftax JAX env.

    obs: flat float32 [0,1] factored-state vector (symbolic variant).
    actions: Discrete(43). Reward: +1 per first-time achievement (60+) plus
    small health/intrinsic terms. Episodes are long (procedural, multi-floor).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_name: str = "Craftax-Symbolic-v1",
        seed: int = 0,
    ):
        super().__init__()
        import jax
        from craftax.craftax_env import make_craftax_env_from_name

        self._jax = jax
        self._env = make_craftax_env_from_name(env_name, auto_reset=False)
        self._params = self._env.default_params
        # Achievement NAMES: Craftax keeps achievements in the JAX STATE (a
        # (67,) bool array), NOT in info like Crafter — without this mapping
        # the goal channel and any achievement measurement are silently blind
        # (caught 2026-07-15: a 200k run measured breadth=0). Resolve the
        # enum for the env family; fall back to index names.
        try:
            if "Classic" in env_name:
                from craftax.craftax_classic.constants import Achievement
            else:
                from craftax.craftax.constants import Achievement
            self._achievement_names = [e.name.lower() for e in Achievement]
        except Exception:
            self._achievement_names = None
        # jit the step for throughput (reset is called rarely).
        self._step_fn = jax.jit(self._env.step)
        self._reset_fn = jax.jit(self._env.reset)
        self._key = jax.random.PRNGKey(int(seed))
        self._state = None

        obs_space = self._env.observation_space(self._params)
        act_space = self._env.action_space(self._params)
        obs_shape = tuple(int(x) for x in obs_space.shape)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(int(act_space.n))

    def _split(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._key = self._jax.random.PRNGKey(int(seed))
        obs, self._state = self._reset_fn(self._split(), self._params)
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action: int
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, self._state, reward, done, info = self._step_fn(
            self._split(), self._state, int(action), self._params)
        done = bool(done)
        # Craftax (like Crafter) reports a single `done`. Its info carries a
        # discount (0.0 on true termination/death); when present, use it to
        # split terminated vs truncated so GAE bootstrapping stays correct.
        disc = info.get("discount", None)
        if disc is not None:
            terminated = done and float(disc) == 0.0
        else:
            terminated = done
        truncated = done and not terminated
        # Convert JAX scalars/arrays in info to python/numpy so downstream
        # numpy code never touches a jax array.
        clean_info = {k: (np.asarray(v) if hasattr(v, "shape") else v)
                      for k, v in info.items()}
        # Surface achievements from the JAX state as a Crafter-style
        # name->count dict (per-episode bools -> 0/1 counts), so the goal
        # channel + measurement taps work identically on both worlds.
        try:
            ach = np.asarray(self._state.achievements).astype(int)
            if self._achievement_names and len(self._achievement_names) == len(ach):
                clean_info["achievements"] = {
                    self._achievement_names[i]: int(ach[i])
                    for i in range(len(ach))}
            else:
                clean_info["achievements"] = {
                    f"ach_{i}": int(v) for i, v in enumerate(ach)}
        except Exception:
            pass
        return (np.asarray(obs, dtype=np.float32), float(reward),
                terminated, truncated, clean_info)

    def close(self):
        self._state = None
