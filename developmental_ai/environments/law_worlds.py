"""Law-Worlds — a family of environments for the law-induction program
(Rung 9; CODE_AUDIT_2026-07.md follow-up, "scientific method" stages 1-5).

THE IDEA (multi-environment invariance pressure)
------------------------------------------------
A *law* is what survives across environments while everything on the surface
varies. Each WORLD (fixed by `world_seed`) draws hidden BINDINGS for a set of
abstract law schemas; each EPISODE (reset seed) redraws the layout. Within a
world the bindings are constant; across worlds they permute. So:

  * a policy that memorizes surface facts ("grab the green key", "avoid blue
    tiles") from training worlds FAILS in a new world;
  * the invariant, transferable knowledge is the SCHEMA — "exactly one floor
    color is hazardous", "keys open doors via a fixed color permutation" —
    plus the cheap experiments that bind it in a new world.

THE LAW SCHEMAS (abstract, world-invariant)
-------------------------------------------
  OPENS(k -> d): the door of color d opens on toggle iff the agent carries the
      key of color binding[d] (a per-world permutation over KEY_COLORS). A
      wrong-key toggle does nothing (cheap, safe experiment).
  HAZARD(h): exactly one floor color (per world) is hazardous — stepping on it
      terminates the episode with `hazard_reward` (a COSTLY experiment; one
      trial per episode, mirroring the Rung-6 costly-probe principle).

Layout per episode: walled room, vertical wall with one locked BindingDoor,
goal behind it, the correct key AND a decoy key on the agent's side, and
colored floor patches (hazard color + safe decoys) scattered off the critical
path (a hazard-free route always exists; verified at generation).

GROUND TRUTH (`binding`, `hazard_color`) is exposed as attributes for GRADING
AND LESION CONTROLS ONLY — never encoded in the observation beyond ordinary
object colors. Same convention as rung7's grader and rung6's regime.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Floor, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

KEY_COLORS = ["red", "green", "blue", "yellow"]
FLOOR_COLORS = ["red", "green", "blue", "yellow", "purple"]


class BindingDoor(Door):
    """A locked door that opens iff the agent carries the key whose color the
    WORLD's binding assigns to this door color — not necessarily the same
    color. Encoding is the ordinary locked-door encoding; the binding is
    invisible."""

    def encode(self):
        state = 0 if self.is_open else 2
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def toggle(self, env, pos):
        if self.is_open:
            self.is_open = False
            return True
        needed = env.binding.get(self.color)
        carrying = getattr(env, "carrying", None)
        if (
            carrying is not None
            and carrying.type == "key"
            and carrying.color == needed
        ):
            self.is_open = True
            self.is_locked = False
            return True
        return False  # wrong/no key: nothing happens (cheap experiment)


class LawWorldEnv(MiniGridEnv):
    """One WORLD of the law family. `world_seed` fixes the hidden bindings;
    `reset(seed=...)` varies the layout. See module docstring."""

    def __init__(
        self,
        world_seed: int,
        size: int = 8,
        n_floor_patches: int = 6,
        hazard_reward: float = -0.5,
        wrong_key_reward: float = -0.2,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        self.world_seed = int(world_seed)
        self.size = int(size)
        self.n_floor_patches = int(n_floor_patches)
        self.hazard_reward = float(hazard_reward)
        self.wrong_key_reward = float(wrong_key_reward)
        # Optional per-episode door-color override (set by harnesses BEFORE
        # reset to cycle door colors — gives the binding schema systematic
        # cross-door exposure instead of random repeats). None = random.
        self.force_door_color: Optional[str] = None

        # ---- Per-world hidden bindings (constant across episodes) ----
        wrng = np.random.RandomState(self.world_seed)
        perm = wrng.permutation(len(KEY_COLORS))
        self.binding: Dict[str, str] = {
            d: KEY_COLORS[perm[i]] for i, d in enumerate(KEY_COLORS)
        }
        self.hazard_color: str = FLOOR_COLORS[wrng.randint(len(FLOOR_COLORS))]

        if max_steps is None:
            max_steps = 8 * size**2
        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "discover this world's laws, open the door, reach the goal"

    # ---- layout generation -------------------------------------------------
    def _gen_grid(self, width, height):
        for _attempt in range(50):
            if self._try_gen_grid(width, height):
                return
        raise RuntimeError("could not generate a valid law-world layout")

    def _try_gen_grid(self, width, height) -> bool:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        split = width - 3
        self.grid.vert_wall(split, 0)
        door_y = self._rand_int(1, height - 2)
        # Door COLOR varies per EPISODE (the binding permutation is only
        # learnable as a schema because different doors appear across
        # episodes of the same world).
        door_color = (self.force_door_color
                      or KEY_COLORS[self._rand_int(0, len(KEY_COLORS))])
        self.door = BindingDoor(door_color, is_locked=True)
        self.put_obj(self.door, split, door_y)
        self.put_obj(Goal(), width - 2, height - 2)

        # A colored BAND the agent MUST cross to reach the door: the column
        # adjacent to the wall, split into segments of distinct floor colors
        # (one may be the world's hazard color). Forced choice = the hazard
        # law actually gets exercised (a cautious router would otherwise
        # never touch colored floor and never learn the law).
        band_x = split - 1
        n_seg = 3
        colors = [self.hazard_color] + [
            c for c in FLOOR_COLORS if c != self.hazard_color
        ]
        seg_colors = [colors[i % len(colors)] for i in range(n_seg)]
        # shuffle segment order per episode
        for i in range(n_seg - 1, 0, -1):
            j = self._rand_int(0, i + 1)
            seg_colors[i], seg_colors[j] = seg_colors[j], seg_colors[i]
        self._patch_cells = []
        rows = list(range(1, height - 1))
        seg_len = max(1, len(rows) // n_seg)
        for idx, y in enumerate(rows):
            color = seg_colors[min(idx // seg_len, n_seg - 1)]
            self.grid.set(band_x, y, Floor(color))
            self._patch_cells.append(((band_x, y), color))

        # ALL key colors present (real search space for the binding law);
        # keys strictly left of the band.
        left = (band_x, height)
        for color in KEY_COLORS:
            self.place_obj(Key(color), top=(0, 0), size=left)

        self.place_agent(top=(0, 0), size=left)

        return self._hazard_free_path_exists(split, door_y)

    def _hazard_free_path_exists(self, split, door_y) -> bool:
        """BFS from agent to door treating hazard patches as blocked — the
        world must always be solvable without stepping on a hazard."""
        from collections import deque

        blocked = {
            tuple(p)
            for p, c in self._patch_cells
            if c == self.hazard_color
        }
        start = tuple(self.agent_pos)
        target = (split, door_y)
        q, seen = deque([start]), {start}
        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in seen or (nx, ny) in blocked:
                    continue
                if (nx, ny) == target:
                    return True
                cell = self.grid.get(nx, ny)
                if cell is None or cell.can_overlap() or cell.type == "key":
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return False

    # ---- law enforcement ---------------------------------------------------
    def step(self, action):
        # Wrong-key toggle is a TRAP (rung-6 costly-probe principle): the
        # binding experiment is informative but expensive, so schema-driven
        # pruning of key candidates is decisive rather than a step tax.
        toggling_door = (
            action == self.actions.toggle
            and self.grid.get(*self.front_pos) is self.door
            and not self.door.is_open
        )
        obs, reward, terminated, truncated, info = super().step(action)
        if toggling_door and not self.door.is_open and not (
                terminated or truncated):
            if self.carrying is not None and self.carrying.type == "key":
                reward = self.wrong_key_reward
                terminated = True
                info["wrong_key"] = True
        if not (terminated or truncated):
            cell = self.grid.get(*self.agent_pos)
            if (
                cell is not None
                and cell.type == "floor"
                and cell.color == self.hazard_color
            ):
                reward = self.hazard_reward
                terminated = True
                info["hazard"] = True
        return obs, reward, terminated, truncated, info
