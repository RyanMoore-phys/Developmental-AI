"""
SeqGoalEnv — a minimal compositional gridworld for Rung 2 (skill composition).

Three modes share an IDENTICAL observation/action space, so a policy trained on
one mode can be reused on another:

  mode "A"  : reward for reaching corner A           -> primitive skill goto_A
  mode "B"  : reward for reaching corner B           -> primitive skill goto_B
  mode "AB" : reward ONLY for reaching A *then* B     -> the composite task

The composite is sparse and sequential: touching B before A does nothing. A small
one-time bonus for reaching A makes it learnable-but-slow from scratch, while the
bulk of the reward requires completing the chain. The observation exposes a
`reached_A` flag, which a composite executor uses to switch goto_A -> goto_B at
exactly the right moment (no episode-midpoint guessing).

Registered ids: SeqGoal-A-v0, SeqGoal-B-v0, SeqGoal-AB-v0.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # N, S, W, E


class SeqGoalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(self, mode="AB", size=5, max_steps=40, render_mode=None,
                 force_reached_A=False, a_pos=None, b_pos=None):
        super().__init__()
        assert mode in ("A", "B", "AB")
        self.mode = mode
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode
        # When True the reached_A flag is pinned on at reset. Used to train the
        # second-leg primitive (goto_B) with flag=1 — the exact distribution it
        # sees inside the composite — so it is never fed an OOD flag on reuse.
        self.force_reached_A = force_reached_A
        # Goal positions default to opposite corners (diagonal). a_pos/b_pos let
        # callers place them ASYMMETRICALLY (breaks the x<->y symmetry — needed so
        # an autonomous cross-domain correspondence is unique).
        self.A = np.array(a_pos if a_pos is not None else [0, 0])
        self.B = np.array(b_pos if b_pos is not None else [size - 1, size - 1])
        # obs: [agent_x, agent_y, A_x, A_y, B_x, B_y, reached_A] normalized. The
        # flag IS observable, so the from-scratch baselines can learn the phase;
        # primitives avoid OOD via force_reached_A (above) rather than by hiding it.
        self.observation_space = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.pos = np.array([size // 2, size // 2])
        self.reached_A = False
        self.t = 0

    def _obs(self):
        s = max(self.size - 1, 1)
        return np.array([
            self.pos[0] / s, self.pos[1] / s,
            self.A[0] / s, self.A[1] / s,
            self.B[0] / s, self.B[1] / s,
            1.0 if self.reached_A else 0.0,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            self.pos = self.np_random.integers(0, self.size, size=2)
            if not (np.array_equal(self.pos, self.A) or np.array_equal(self.pos, self.B)):
                break
        self.reached_A = bool(self.force_reached_A)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        dx, dy = _MOVES[int(action)]
        self.pos = np.clip(self.pos + np.array([dx, dy]), 0, self.size - 1)
        at_A = np.array_equal(self.pos, self.A)
        at_B = np.array_equal(self.pos, self.B)

        reward, done, success = -0.01, False, False
        if self.mode == "A":
            if at_A:
                reward, done, success = 1.0, True, True
        elif self.mode == "B":
            if at_B:
                reward, done, success = 1.0, True, True
        else:  # AB — sequential composite
            if at_A and not self.reached_A:
                self.reached_A = True
                reward = 0.2                       # small one-time subgoal bonus
            elif at_B and self.reached_A:
                reward, done, success = 1.0, True, True

        truncated = self.t >= self.max_steps and not done
        info = {"success": success, "reached_A": self.reached_A}
        return self._obs(), reward, done, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        cell = 40
        img = np.full((self.size * cell, self.size * cell, 3), 255, np.uint8)
        def paint(xy, color):
            x, y = xy
            img[x * cell:(x + 1) * cell, y * cell:(y + 1) * cell] = color
        paint(self.A, (80, 200, 120) if self.reached_A else (60, 160, 90))
        paint(self.B, (90, 130, 240))
        paint(self.pos, (230, 120, 60))
        return img


def register_seq_goal():
    """Register the SeqGoal modes with Gymnasium (idempotent).

    SeqGoal-B1-v0 is mode B with the reached_A flag pinned on — the env the
    second-leg primitive (goto_B) is trained in, matching the composite's
    flag=1 distribution.
    """
    specs = [
        ("SeqGoal-A-v0", {"mode": "A"}, 40),
        ("SeqGoal-B-v0", {"mode": "B"}, 40),
        ("SeqGoal-B1-v0", {"mode": "B", "force_reached_A": True}, 40),
        ("SeqGoal-AB-v0", {"mode": "AB"}, 40),
        # Larger AB target for cumulative-transfer (Rung 1): same scale-invariant
        # obs, but a 9x9 grid is much harder to solve from scratch.
        ("SeqGoal-AB-L-v0", {"mode": "AB", "size": 9, "max_steps": 80}, 80),
        # Asymmetric-goal 9x9 (B off the diagonal) for the AUTONOMOUS Rung 7 test:
        # breaks the x<->y symmetry so the cross-domain correspondence is unique.
        ("SeqGoal-AB-Asym-v0",
         {"mode": "AB", "size": 9, "max_steps": 80, "b_pos": [8, 2]}, 80),
    ]
    for env_id, kwargs, max_steps in specs:
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="developmental_ai.environments.seq_goal:SeqGoalEnv",
            kwargs=kwargs,
            max_episode_steps=max_steps,
        )


register_seq_goal()
