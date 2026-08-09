"""
Noisy-TV gridworld — the Rung 3 (autotelic / intrinsic-motivation) DECISIVE test.

WHY THIS EXISTS
---------------
Rung 3 asks whether the agent invents its own goals and grows its competence
frontier with NO external reward, AND — critically — whether its curiosity tracks
LEARNING PROGRESS rather than raw novelty. Raw-novelty curiosity (prediction
error = reward, e.g. ICM) has a famous failure mode: the "noisy TV." An
uncontrollable source of randomness produces high prediction error FOREVER, so a
novelty-driven agent gets mesmerized and stops exploring. A learning-progress
agent ignores it (you never get better at predicting noise) and keeps expanding
the part of the world it can actually master.

THE ENVIRONMENT
---------------
An empty room (no goal, reward always 0 — pure intrinsic motivation) plus a
"noisy TV": a region around a fixed cell where the agent's observation is
CORRUPTED WITH FRESH RANDOM VALUES every step. The corruption is:
  * uncontrollable + unlearnable  -> permanent high prediction error (novelty),
    but ZERO learning progress (the world model never improves on noise);
  * local — only when the agent is within `tv_zone_radius` of the TV cell, so the
    agent must CHOOSE to "watch the TV" to receive the novelty.

The learnable structure is simply the room itself: the egocentric symbolic view
is predictable everywhere except the TV zone, and the frontier the agent can grow
is spatial COVERAGE — the set of distinct cells it visits.

METRICS (tracked here, persisted across episode resets within a run)
  * coverage      — number of DISTINCT interior cells ever visited (the frontier).
  * tv_steps      — steps spent in the TV zone (the noisy-TV trap).
  * coverage_curve— periodic (total_steps, coverage, tv_steps) snapshots.

PREDICTIONS
  * novelty (ICM): mesmerized by the TV -> high tv_steps, coverage stalls.
  * learning-progress: ignores the TV -> low tv_steps, coverage keeps growing.
PASS = LP coverage >> novelty coverage AND LP tv-fraction << novelty tv-fraction.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball
from minigrid.minigrid_env import MiniGridEnv


class NoisyTVEnv(MiniGridEnv):
    """Reward-free room with a local 'noisy TV' that corrupts the observation.
    See module docstring. Exposes coverage / tv_steps metrics that persist across
    resets within a single run (a fresh env per run starts them at zero)."""

    def __init__(
        self,
        size: int = 8,
        tv_zone_radius: int = 1,
        noise_fraction: float = 0.7,
        max_steps: int = 200,
        coverage_sample_every: int = 500,
        **kwargs,
    ):
        self.tv_zone_radius = int(tv_zone_radius)
        self.noise_fraction = float(noise_fraction)
        self.coverage_sample_every = int(coverage_sample_every)
        self.tv_pos = None
        self._in_tv_zone = False

        # --- run-level metrics (NOT reset on episode reset) ---
        self._visited = set()       # distinct interior cells ever occupied
        self._tv_steps = 0
        self._total_steps = 0
        self.coverage_curve = []    # [(total_steps, coverage, tv_steps), ...]
        self._noise_cells = None    # fixed set of (row,col) image cells to corrupt
        self._rng_noise = np.random.RandomState(0)

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "explore the room (there is no goal; ignore the noisy TV)"

    def reset_metrics(self):
        self._visited = set()
        self._tv_steps = 0
        self._total_steps = 0
        self.coverage_curve = []

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        # The TV: a fixed landmark in the top-right interior corner. (No Goal,
        # so there is no task reward anywhere.)
        self.tv_pos = (width - 2, 1)
        self.grid.set(*self.tv_pos, Ball("red"))   # visual landmark for the TV
        self.place_agent()
        self.mission = self._gen_mission()

    def _agent_in_tv_zone(self) -> bool:
        if self.tv_pos is None:
            return False
        ax, ay = self.agent_pos
        tx, ty = self.tv_pos
        return abs(ax - tx) + abs(ay - ty) <= self.tv_zone_radius

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Reward-free: this env carries NO external reward signal.
        reward = 0.0
        terminated = False

        self._total_steps += 1
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        self._visited.add((ax, ay))
        self._in_tv_zone = self._agent_in_tv_zone()
        if self._in_tv_zone:
            self._tv_steps += 1

        if self._total_steps % self.coverage_sample_every == 0:
            self.coverage_curve.append(
                (self._total_steps, len(self._visited), self._tv_steps))

        info["in_tv_zone"] = self._in_tv_zone
        info["coverage"] = len(self._visited)
        info["tv_steps"] = self._tv_steps
        info["total_env_steps"] = self._total_steps
        return obs, reward, terminated, truncated, info

    def gen_obs(self):
        """Standard egocentric symbolic obs, but CORRUPTED with fresh random
        values whenever the agent is in the TV zone — the noisy TV."""
        obs = super().gen_obs()
        if self._agent_in_tv_zone():
            img = obs["image"]
            h, w = img.shape[0], img.shape[1]
            if self._noise_cells is None:
                ncells = h * w
                k = int(round(self.noise_fraction * ncells))
                idx = np.arange(ncells)[:k]
                self._noise_cells = [(int(i // w), int(i % w)) for i in idx]
            # Fresh random VALID-range encodings each step -> unlearnable novelty.
            for (r, c) in self._noise_cells:
                img[r, c, 0] = self._rng_noise.randint(0, 11)   # object idx
                img[r, c, 1] = self._rng_noise.randint(0, 6)    # color idx
                img[r, c, 2] = self._rng_noise.randint(0, 3)    # state idx
            obs["image"] = img
        return obs


def _register():
    entry = "developmental_ai.environments.noisy_tv_gridworld:NoisyTVEnv"
    for size in (8, 10, 12, 16, 20, 24):
        env_id = f"MiniGrid-NoisyTV-{size}x{size}-v0"
        if env_id in gymnasium.registry:
            continue
        gymnasium.register(id=env_id, entry_point=entry, kwargs={"size": size})


_register()


if __name__ == "__main__":
    # ---- self-test: noise is real, local, reward-free, coverage countable ----
    env = NoisyTVEnv(size=8)
    obs, _ = env.reset(seed=0)
    interior = (env.grid.width - 2) * (env.grid.height - 2)
    print(f"[info] size={env.grid.width} interior_cells={interior} tv_pos={env.tv_pos}")

    # (1) Outside the TV zone the obs is STABLE across steps with a no-op-ish turn;
    #     inside the zone it CHANGES every step (noise). Drive the agent to the TV.
    import numpy as np
    # Force the agent next to the TV, facing it.
    tx, ty = env.tv_pos
    env.agent_pos = (tx - 1, ty)
    env.agent_dir = 0
    assert env._agent_in_tv_zone(), "agent should be in TV zone"
    imgs = []
    for _ in range(5):
        o = env.gen_obs()
        imgs.append(o["image"].copy())
    diffs = [not np.array_equal(imgs[0], imgs[i]) for i in range(1, len(imgs))]
    assert all(diffs), "TV-zone obs should change every step (noise)"
    print("[ok] in TV zone: observation is fresh noise every step")

    # (2) Far from the TV, repeated identical actions give a STABLE obs (no noise).
    env.agent_pos = (1, env.grid.height - 2)   # opposite corner
    env.agent_dir = 0
    assert not env._agent_in_tv_zone()
    o1 = env.gen_obs()["image"].copy()
    o2 = env.gen_obs()["image"].copy()
    assert np.array_equal(o1, o2), "non-TV obs must be stable (no noise)"
    print("[ok] outside TV zone: observation is clean/stable")

    # (3) Reward is always 0; episodes never terminate early (only truncate).
    env.reset(seed=1)
    rewards, terms = [], []
    for _ in range(50):
        _, r, term, trunc, info = env.step(env.action_space.sample())
        rewards.append(r); terms.append(term)
        if trunc:
            env.reset()
    assert all(r == 0.0 for r in rewards), "env must be reward-free"
    assert not any(terms), "env must not terminate (reward-free exploration)"
    print("[ok] reward-free, non-terminating")

    # (4) Coverage + tv metrics accumulate and are exposed.
    env.reset_metrics()
    env.reset(seed=2)
    for _ in range(400):
        _, _, _, trunc, info = env.step(env.action_space.sample())
        if trunc:
            env.reset()
    print(f"[ok] random policy: coverage={info['coverage']}/{interior} "
          f"tv_steps={info['tv_steps']}/{info['total_env_steps']} "
          f"curve_pts={len(env.coverage_curve)}")
    assert info["coverage"] >= 2
    print("\nNOISY-TV ENV SELF-TEST OK")
