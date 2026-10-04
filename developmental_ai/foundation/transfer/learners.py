"""A tiny tabular learner and the record-level learning loop (fixtures).

TabularQLearner is deliberately the smallest learner that can carry state
from one environment to another: a dict from observed-state key to action
values. It reads ONLY `policy_observations` (never an evaluator channel),
and it keys missing readings by their sentinel name, so ABSENT is its own
state rather than being collapsed into zero.

run_learning drives any EnvironmentAdapter with it and returns the per-step
reward curve; rewards that are ABSENT are NaN in the curve (unknown, not 0).
"""

from __future__ import annotations

import copy
import itertools
import time
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from ..adapters import policy_observations
from ..contracts import UNKNOWN, Action, ActionSpec, ObservationSpec, Sentinel


def state_key(observations, spec: ObservationSpec,
              exclude: Sequence[str] = ("reward",), decimals: int = 3) -> Tuple:
    parts = []
    for o in sorted(policy_observations(observations, spec), key=lambda x: x.channel):
        if o.channel in exclude:
            continue
        v = o.value
        if isinstance(v, Sentinel):
            parts.append((o.channel, v.name))
        else:
            a = np.round(np.asarray(v, dtype=np.float64).ravel(), decimals)
            parts.append((o.channel, tuple(float(x) + 0.0 for x in a)))
    return tuple(parts)


def reading(observations, channel: str) -> Optional[float]:
    for o in observations:
        if o.channel == channel:
            v = o.value
            return None if isinstance(v, Sentinel) else float(v)
    return None


class TabularQLearner:
    def __init__(self, action_spec: ActionSpec, lr: float = 0.5, gamma: float = 0.0,
                 epsilon: float = 0.1, seed: int = 0):
        if action_spec.kind == "discrete":
            self.commands: List[Any] = list(range(action_spec.n))
        elif action_spec.kind == "multi_discrete":
            self.commands = list(itertools.product(*[range(n) for n in action_spec.nvec]))
        else:
            raise ValueError("TabularQLearner needs a discrete or multi_discrete "
                             "action spec, not a box")
        if not (0 < lr <= 1) or not (0 <= gamma < 1) or not (0 <= epsilon <= 1):
            raise ValueError("need 0<lr<=1, 0<=gamma<1, 0<=epsilon<=1")
        self.spec_signature = (action_spec.kind, action_spec.shape,
                               action_spec.n if action_spec.kind == "discrete"
                               else tuple(action_spec.nvec))
        self.lr, self.gamma, self.epsilon = float(lr), float(gamma), float(epsilon)
        self.q: Dict[Hashable, np.ndarray] = {}
        self.updates = 0
        self.r_min, self.r_max = np.inf, -np.inf
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def compatible(self, action_spec: ActionSpec) -> bool:
        try:
            return TabularQLearner(action_spec).spec_signature == self.spec_signature
        except ValueError:
            return False

    def clone(self) -> "TabularQLearner":
        return copy.deepcopy(self)

    def knows(self, s) -> bool:
        return s in self.q

    def values(self, s) -> np.ndarray:
        v = self.q.get(s)
        return v if v is not None else np.zeros(len(self.commands))

    def act(self, s, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.commands)))
        v = self.values(s)
        best = np.flatnonzero(v == v.max())
        return int(best[self.rng.integers(best.size)])

    def td_error(self, s, a: int, r: float, s2, terminal: bool) -> float:
        boot = 0.0 if terminal else self.gamma * float(self.values(s2).max())
        return r + boot - float(self.values(s)[a])

    def update(self, s, a: int, r: float, s2, terminal: bool) -> float:
        td = self.td_error(s, a, r, s2, terminal)
        if s not in self.q:
            self.q[s] = np.zeros(len(self.commands))
        self.q[s][a] += self.lr * td
        self.updates += 1
        self.r_min, self.r_max = min(self.r_min, r), max(self.r_max, r)
        return td


def run_learning(learner: TabularQLearner, env, steps: int, seed: int,
                 learn: bool = True, reward_channel: str = "reward",
                 exclude: Sequence[str] = ("reward",)) -> Dict[str, Any]:
    """Drive `env` for `steps` interactions. learn=False is a frozen greedy
    probe (no exploration, no updates). Returns rewards (NaN where ABSENT),
    whether each state was already known, TD errors, and update count."""
    ospec, aspec = env.observation_spec(), env.action_spec()
    if not learner.compatible(aspec):
        raise ValueError("learner action space does not match the environment")
    obs = env.reset(seed=seed)
    s = state_key(obs, ospec, exclude)
    rewards, known, tds = [], [], []
    u0 = learner.updates
    for _ in range(int(steps)):
        a = learner.act(s, explore=learn)
        head = obs[0]
        act = Action(head.environment, head.stream, head.episode, head.seq,
                     aspec.spec_id, learner.commands[a], UNKNOWN, time.time())
        obs2, term, trunc, info = env.step(act)
        if info.get("client_recovery"):
            rewards.append(np.nan)
            known.append(learner.knows(s))
            obs, s = obs2, state_key(obs2, ospec, exclude)
            continue
        r = reading(obs2, reward_channel)
        s2 = state_key(obs2, ospec, exclude)
        known.append(learner.knows(s))
        if r is None:
            rewards.append(np.nan)
        else:
            rewards.append(r)
            if learn:
                tds.append(learner.update(s, a, r, s2, term))
            else:
                tds.append(learner.td_error(s, a, r, s2, term))
        if term or trunc:
            obs = env.reset()
            s = state_key(obs, ospec, exclude)
        else:
            obs, s = obs2, s2
    return {"rewards": np.asarray(rewards, dtype=np.float64),
            "known": np.asarray(known, dtype=bool),
            "td_errors": np.asarray(tds, dtype=np.float64),
            "updates": learner.updates - u0}
