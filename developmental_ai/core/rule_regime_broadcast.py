"""
Rule-Regime broadcast — the symbolic knowledge channel for the Rung-6 DECISIVE
test (see environments/rule_regime_doorkey.py).

It carries the one piece of task-structure KNOWLEDGE the observation provably
lacks: the hidden affordance REGIME for this episode (does the locked-looking
door need a key, or does it open directly?), plus the rule-derived subgoal phase.
The regime is read from the env's privileged `regime` attribute — exactly as the
affordance broadcast reads `carrying` — and is never present in the observation.
Because the signal is supplied every step (nothing to remember), this tests
GLOBAL INTEGRATION (Global Workspace), not memory.

ENCODING (DIM = 7)
------------------
  [is_KEY, is_NOKEY, has_key, door_open, GET_KEY, GO_TO_DOOR, GO_TO_GOAL]
    * is_KEY / is_NOKEY — one-hot of the active affordance rule.
    * has_key           — proprioception (carrying a key).
    * door_open         — latched once the door is seen open (partial view).
    * phase one-hot     — rule-derived subgoal:
        door_open                 -> GO_TO_GOAL
        KEY  regime & !has_key     -> GET_KEY
        KEY  regime &  has_key     -> GO_TO_DOOR  (carry the key to the door)
        NOKEY regime               -> GO_TO_DOOR  (toggle directly; never get key)

CONTROLS (train-time arms; each in-distribution for its own training, so NO OOD
confound — the whole point of the aliasing redesign):
    * zero      — handled in developmental_loop (all-zeros): presence severed.
    * constant  — a fixed valid vector (present, no variation).
    * noise     — a random valid vector each step, UNCORRELATED with the regime
                  (present, varies, but carries no information). constant AND noise
                  both capping at the 0.5 aliasing ceiling, while intact exceeds it,
                  shows it is specifically the INFORMATION (not mere presence or
                  variation) that is load-bearing.
"""

from __future__ import annotations

import numpy as np

_OBJ_DOOR = 4       # OBJECT_TO_IDX["door"]
_STATE_OPEN = 0     # STATE_TO_IDX open

KEY_REGIME = "KEY"
NOKEY_REGIME = "NOKEY"


class RuleRegimeBroadcast:
    """Compact symbolic vector carrying the active affordance RULE + subgoal."""

    DIM = 7

    # Canonical, self-consistent valid vectors (the only ones feature() emits).
    # Used by the constant/noise controls so a lesioned broadcast stays a VALID
    # affordance vector rather than going out of distribution.
    #   idx: [is_KEY, is_NOKEY, has_key, door_open, GET_KEY, GO_TO_DOOR, GO_TO_GOAL]
    PHASE_VECTORS = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   # 0 KEY  / GET_KEY
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],   # 1 KEY  / GO_TO_DOOR (has key)
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],   # 2 KEY  / GO_TO_GOAL
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # 3 NOKEY/ GO_TO_DOOR
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],   # 4 NOKEY/ GO_TO_GOAL
        ],
        dtype=np.float32,
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._regime = None
        self._has_key = 0.0
        self._door_open = 0.0   # monotonic latch

    @staticmethod
    def _base(env):
        return getattr(env, "unwrapped", env)

    def update(self, env) -> None:
        base = self._base(env)
        if not hasattr(base, "carrying"):
            return  # non-MiniGrid: stay inactive (null feature, shape preserved)
        self._active = True
        self._regime = getattr(base, "regime", None)

        carrying = getattr(base, "carrying", None)
        self._has_key = (
            1.0 if (carrying is not None and getattr(carrying, "type", "") == "key") else 0.0
        )

        # Latch door_open from the partial egocentric view only.
        try:
            view = np.asarray(base.gen_obs()["image"])
        except Exception:
            return
        if view.ndim != 3 or view.shape[2] < 3:
            return
        door_mask = view[:, :, 0] == _OBJ_DOOR
        if door_mask.any() and (view[:, :, 2][door_mask] == _STATE_OPEN).any():
            self._door_open = 1.0

    def _true_index(self) -> int:
        """Index into PHASE_VECTORS for the current true (regime, phase)."""
        is_key = self._regime != NOKEY_REGIME  # default/None -> treat as KEY
        if is_key:
            if self._door_open:
                return 2
            if self._has_key:
                return 1
            return 0
        # NOKEY
        return 4 if self._door_open else 3

    def feature(self) -> np.ndarray:
        if not self._active:
            return np.zeros(self.DIM, dtype=np.float32)
        return self.PHASE_VECTORS[self._true_index()].copy()

    # ---- Rung-6 controls (in-distribution, information-destroyed) -------------

    def constant_feature(self) -> np.ndarray:
        """Fixed valid vector (KEY/GET_KEY) regardless of state: present, no
        variation -> tests whether the policy needs the signal to VARY."""
        if not self._active:
            return np.zeros(self.DIM, dtype=np.float32)
        return self.PHASE_VECTORS[0].copy()

    def noise_feature(self, rng: np.random.RandomState) -> np.ndarray:
        """A random VALID vector each step, drawn independently of the true
        regime: present and varying, but carrying no information."""
        if not self._active:
            return np.zeros(self.DIM, dtype=np.float32)
        return self.PHASE_VECTORS[int(rng.randint(len(self.PHASE_VECTORS)))].copy()

    def scrambled_feature(self, rng: np.random.RandomState) -> np.ndarray:
        """Interface-compatible 'wrong vector' control. NOTE: a *deterministic*
        regime flip is bijective with the truth and a from-scratch policy would
        just learn the inverse, so the train-time arms use `noise` instead. Kept
        for harness compatibility; emits a valid vector for a DIFFERENT phase."""
        if not self._active:
            return np.zeros(self.DIM, dtype=np.float32)
        t = self._true_index()
        wrong = [i for i in range(len(self.PHASE_VECTORS)) if i != t]
        return self.PHASE_VECTORS[int(rng.choice(wrong))].copy()
