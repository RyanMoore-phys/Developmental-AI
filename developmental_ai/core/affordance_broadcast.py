"""
Symbolic Affordance broadcast — the Rung 6 (Global Workspace) *knowledge* channel.

WHY THIS EXISTS
---------------
Rung 6 asks whether a symbolic broadcast is *functionally integrated* into
behaviour (information flows symbol -> policy) or merely decorative. Two earlier
channels failed to give a testable answer:

  * the original KG GNN mean-pool was a single GLOBAL vector recomputed every 50
    episodes -> constant within an episode -> no decision-relevant content; the
    learned gate correctly stayed shut. Null by construction.
  * a within-episode *spatial memory* (last-seen object locations) would have
    worked, but it manufactures a MEMORY bottleneck — and this system already
    computes a recurrent memory (the RSSM) that the policy simply never consumes.
    Testing memory is the wrong question for a neurosymbolic-*knowledge* claim.

This channel instead carries abstract TASK-STRUCTURE KNOWLEDGE — the affordance
rule "to pass a locked door you must be carrying a key" — distilled to a compact
symbolic vector. It encodes WHAT the task state is and WHICH subgoal applies, not
WHERE anything is. Because the rule is grid-size / layout invariant, it is the
natural unit of CROSS-TASK TRANSFER: a policy that learned to act on the phase
signal on an easy env can be dropped onto a harder env it could never crack from
sparse reward alone, and the phase tells it what to do. Lesioning the broadcast
(zeroing its content) then measures whether that transferred knowledge was
load-bearing.

WHAT IT IS ALLOWED TO USE
-------------------------
Only legitimately-held information:
  * proprioception — the agent's own inventory (env.unwrapped.carrying), and
  * its PARTIAL view — env.unwrapped.gen_obs()["image"] (the same masked
    egocentric grid the policy sees), used ONLY to read the door's state.
It NEVER reads env.grid (full observability) and stores NO object locations.

ENCODING (DIM = 5)
------------------
  [has_key, door_open, phase==NEED_KEY, phase==NEED_OPEN_DOOR, phase==GO_TO_GOAL]
    * has_key   — 1.0 if currently carrying a key (proprioception)
    * door_open — 1.0 once the door has been observed open (monotonic task state)
    * phase     — one-hot of the rule-derived subgoal:
                    door_open            -> GO_TO_GOAL
                    has_key, !door_open  -> NEED_OPEN_DOOR   (use the key)
                    else                 -> NEED_KEY         (fetch the key)

Env-agnostic: on a non-MiniGrid env (no agent/carrying) update() is a no-op and
feature() returns zeros, so the policy input shape is preserved.
"""

from __future__ import annotations

import numpy as np


# MiniGrid object indices (minigrid.core.constants.OBJECT_TO_IDX) — stable.
_OBJ_DOOR = 4
# Door state channel (minigrid.core.constants.STATE_TO_IDX): open=0, closed=1, locked=2
_STATE_OPEN = 0


class AffordanceBroadcast:
    """Compact symbolic task-phase / affordance vector for the policy's
    knowledge channel. Carries the 'locked-door => needs-key' rule, not memory."""

    # [has_key, door_open, NEED_KEY, NEED_OPEN_DOOR, GO_TO_GOAL]
    DIM = 5

    # Canonical, self-consistent feature for each phase (index = phase id).
    # These are exactly the vectors feature() emits, used by the Rung-6
    # "scramble"/"constant" controls so a lesioned broadcast stays IN
    # DISTRIBUTION (a valid affordance vector) while carrying wrong / no info.
    #   0 NEED_KEY       : no key, door closed
    #   1 NEED_OPEN_DOOR : holding key, door closed
    #   2 GO_TO_GOAL     : door open (key consumed by opening)
    PHASE_VECTORS = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active = False        # set once we see a MiniGrid agent
        self._has_key = 0.0
        self._door_open = 0.0       # monotonic: latches to 1 once seen open

    @staticmethod
    def _base(env):
        return getattr(env, "unwrapped", env)

    def update(self, env) -> None:
        """Fold the agent's current inventory + partial view into the affordance
        state. Call once per environment interaction (after reset and each step)."""
        base = self._base(env)
        # Proprioception gate: only meaningful on a MiniGrid-style agent.
        if not hasattr(base, "carrying"):
            return
        self._active = True

        carrying = getattr(base, "carrying", None)
        self._has_key = (
            1.0 if (carrying is not None and getattr(carrying, "type", "") == "key") else 0.0
        )

        # Partial view ONLY — read the door's state if it is currently visible.
        try:
            view = np.asarray(base.gen_obs()["image"])
        except Exception:
            return
        if view.ndim != 3 or view.shape[2] < 3:
            return
        door_mask = view[:, :, 0] == _OBJ_DOOR
        if door_mask.any():
            if (view[:, :, 2][door_mask] == _STATE_OPEN).any():
                self._door_open = 1.0  # latch (task progress is monotonic)

    def feature(self) -> np.ndarray:
        """Build the broadcast vector: raw state bits + rule-derived phase."""
        v = np.zeros(self.DIM, dtype=np.float32)
        if not self._active:
            return v  # non-MiniGrid / pre-reset: true no-op (null feature)
        v[0] = self._has_key
        v[1] = self._door_open
        # Rule: locked-door => needs-key. Derive the active subgoal phase.
        if self._door_open:
            v[4] = 1.0          # GO_TO_GOAL
        elif self._has_key:
            v[3] = 1.0          # NEED_OPEN_DOOR (carry the key to the door)
        else:
            v[2] = 1.0          # NEED_KEY
        return v

    # ---- Rung-6 lesion controls (content vs presence) ----------------------
    # These produce IN-DISTRIBUTION but information-destroyed broadcasts, so a
    # behavioural drop vs intact cannot be dismissed as the policy merely
    # breaking on a never-seen all-zeros input (the OOD confound that a plain
    # zero-lesion leaves open).

    def _true_phase(self) -> int:
        """Current rule-derived phase id (0/1/2) from latched state."""
        if self._door_open:
            return 2
        if self._has_key:
            return 1
        return 0

    def scrambled_feature(self, rng: np.random.RandomState) -> np.ndarray:
        """A VALID-but-WRONG affordance vector: a canonical phase vector for a
        phase OTHER than the true one. Same format/statistics as a real
        broadcast, but it does not describe the current state — so if intact >
        scramble, behaviour tracks the broadcast's CONTENT, not its presence."""
        if not self._active:
            return np.zeros(self.DIM, dtype=np.float32)
        t = self._true_phase()
        wrong = [p for p in (0, 1, 2) if p != t]
        return self.PHASE_VECTORS[int(rng.choice(wrong))].copy()

    def constant_feature(self) -> np.ndarray:
        """A FIXED valid affordance vector (always NEED_KEY) regardless of
        state: in-distribution, but carries no information because it never
        varies. Tests whether the policy needs the broadcast's VARIATION."""
        if not self._active:
            return np.zeros(self.DIM, dtype=np.float32)
        return self.PHASE_VECTORS[0].copy()
