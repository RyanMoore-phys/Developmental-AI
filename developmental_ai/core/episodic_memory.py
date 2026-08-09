"""
Episodic Working-Memory broadcast — the Rung 6 (Global Workspace) channel.

WHY THIS EXISTS
---------------
The original Rung 6 broadcast (a GNN mean-pool over the knowledge graph) was a
single *global* vector recomputed only every 50 episodes, so it was constant
within an episode and could carry no decision-relevant information. The learned
KnowledgeConditioner gate correctly stayed shut (resting sigmoid ~0.047 after
training, unchanged from init across all 8 seeds) — there was nothing worth
admitting. The ablation was therefore null *by construction*.

This module replaces that channel with information the broadcast can actually
make load-bearing: a within-episode working memory of WHERE the agent last saw
the task-relevant objects (key / door / goal). A feed-forward, partially
observing agent forgets these the instant they leave its 7x7 view; broadcasting
the remembered locations gives it something the raw observation does not have.
If integration is real, the gate should open and zeroing the broadcast (the
Rung 6 lesion) should measurably degrade behaviour.

WHAT IT IS ALLOWED TO USE
-------------------------
Only information the agent legitimately possesses:
  * proprioception — its own world position, heading, and inventory
    (env.unwrapped.agent_pos / agent_dir / carrying), and
  * its PARTIAL view — env.unwrapped.gen_obs()["image"], the same 7x7x3 masked
    egocentric grid the policy receives (occluded cells read as "unseen").
It NEVER reads env.grid (the full map) — that would be cheating (full
observability), which is exactly the thing this test is about.

ENCODING (DIM = 14)
-------------------
Per tracked object {key, door, goal}: [seen, ahead, right, recency]
  * seen     — 1.0 once the object has ever been observed this episode
  * ahead    — homing component along the agent's current facing (+ = in front)
  * right    — homing component to the agent's current right (+ = to the right)
               (homing vector = agent's world pos when it last saw the object,
                minus its current pos, rotated into the agent's own frame and
                normalised by grid size — so it is directly actionable as
                turn/forward commands)
  * recency  — exp(-age/20) decay since last sighting
Plus two scalar flags: carrying_key, door_open.

The module is env-agnostic: on a non-MiniGrid env (no agent_pos) update() is a
no-op and feature() returns zeros, so the policy input shape is preserved.
"""

from __future__ import annotations

import numpy as np


# MiniGrid object indices (minigrid.core.constants.OBJECT_TO_IDX) — stable.
_OBJ_DOOR = 4
_OBJ_KEY = 5
_OBJ_GOAL = 8
# Door state channel (minigrid.core.constants.STATE_TO_IDX): open=0, closed=1, locked=2
_STATE_OPEN = 0
# Heading -> unit forward vector (minigrid DIR_TO_VEC): 0=E,1=S,2=W,3=N
_DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class EpisodicWorkingMemory:
    """Within-episode memory of last-seen key/door/goal locations, emitted as a
    fixed-width broadcast vector for the policy's KnowledgeConditioner gate."""

    OBJECTS = (_OBJ_KEY, _OBJ_DOOR, _OBJ_GOAL)
    # per object: [seen, ahead, right, recency] (4); + carrying_key + door_open (2)
    DIM = len(OBJECTS) * 4 + 2

    def __init__(self) -> None:
        self._norm = 8.0  # grid-size normaliser; refreshed from env each update
        self.reset()

    def reset(self) -> None:
        self._seen = {o: False for o in self.OBJECTS}
        self._last_pos = {o: (0.0, 0.0) for o in self.OBJECTS}
        self._last_step = {o: 0 for o in self.OBJECTS}
        self._cur_pos = (0.0, 0.0)
        self._dir = 0
        self._step = 0
        self._carrying_key = 0.0
        self._door_open = 0.0

    @staticmethod
    def _base(env):
        return getattr(env, "unwrapped", env)

    def update(self, env) -> None:
        """Fold the agent's current pose + partial view into memory. Call once
        per environment interaction (after reset and after each step)."""
        base = self._base(env)
        pos = getattr(base, "agent_pos", None)
        if pos is None:
            return  # non-MiniGrid env: stay at zeros

        self._cur_pos = (float(pos[0]), float(pos[1]))
        self._dir = int(getattr(base, "agent_dir", 0))
        w = getattr(base, "width", None)
        h = getattr(base, "height", None)
        if w:
            self._norm = float(max(w, h or w))
        self._step += 1

        # Inventory (proprioception): is the agent holding a key?
        carrying = getattr(base, "carrying", None)
        self._carrying_key = (
            1.0 if (carrying is not None and getattr(carrying, "type", "") == "key") else 0.0
        )

        # Partial view ONLY — gen_obs() masks cells the agent cannot see.
        try:
            view = np.asarray(base.gen_obs()["image"])
        except Exception:
            return
        if view.ndim != 3 or view.shape[2] < 3:
            return
        obj_layer = view[:, :, 0]
        state_layer = view[:, :, 2]

        for o in self.OBJECTS:
            mask = obj_layer == o
            if mask.any():
                self._seen[o] = True
                self._last_pos[o] = self._cur_pos
                self._last_step[o] = self._step
                if o == _OBJ_DOOR:
                    self._door_open = 1.0 if (state_layer[mask] == _STATE_OPEN).any() else 0.0

    def feature(self) -> np.ndarray:
        """Build the broadcast vector in the agent's egocentric frame."""
        v = np.zeros(self.DIM, dtype=np.float32)
        fx, fy = _DIR_TO_VEC[self._dir % 4]
        rx, ry = -fy, fx  # right = forward rotated +90 deg
        cx, cy = self._cur_pos
        i = 0
        for o in self.OBJECTS:
            if self._seen[o]:
                lx, ly = self._last_pos[o]
                dx = (lx - cx) / self._norm
                dy = (ly - cy) / self._norm
                v[i] = 1.0
                v[i + 1] = dx * fx + dy * fy   # ahead (+) / behind (-)
                v[i + 2] = dx * rx + dy * ry   # right (+) / left (-)
                age = self._step - self._last_step[o]
                v[i + 3] = float(np.exp(-age / 20.0))
            i += 4
        v[i] = self._carrying_key
        v[i + 1] = self._door_open
        return v
