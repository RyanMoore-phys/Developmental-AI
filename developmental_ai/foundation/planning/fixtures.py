"""Closed-loop fixtures for Stage 11: a point mass to steer, and a parametric
mechanism that can be fitted from its transitions — or set deliberately
wrong.

    PointMassAdapter     an EnvironmentAdapter (adapters.protocol contract):
                         channels "pos", "vel", "goal" (sensor, policy
                         visible, frame "world"), control "discrete" (5
                         pushes: none, +x, -x, +y, -y) or "box" (force in
                         [-1, 1]^2). Dynamics:
                             vel' = 0.8 vel + GAIN * force * dt + noise
                             pos' = clip(pos + vel' dt, -L, L)
                         `dynamics_params`/`set_dynamics(gain=...)` and
                         transfer.ControlRemap apply variations.
    PointMassMechanism   a BaseMechanism with the same functional form,
                         fitted by least squares: vel'_j on [vel_j, f dt].
                         `set_params` makes a hand-set (e.g. wrong but
                         confident) model; `to_record`/`from_record` carry
                         it through contracts.Mechanism with id + version.
    ProportionalController   the stand-in "existing learned controller"
                         used as fallback in the closed-loop fixtures; it is
                         deliberately noisy (it is a baseline, not an oracle).
    state_from_obs, policy_vector, make_action, run_episode, GoalReward,
    DISCRETE_FORCES, encode_discrete

Fixture code: offline only.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import (INAPPLICABLE, UNKNOWN, Action, ActionSpec, ChannelSpec,
                         ContractError, Mechanism, Observation, ObservationSpec,
                         RecordValidationError)
from ..mechanisms.api import BaseMechanism, EntityBatch, MixedOutcome, TransitionBatch

DISCRETE_FORCES = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], np.float64)
DT = 0.1


def encode_discrete(idx) -> np.ndarray:
    """(H, C) indices -> (H, C, 1, 2) forces."""
    i = np.asarray(idx).astype(np.int64)
    return DISCRETE_FORCES[i][..., None, :]


class PointMassAdapter:
    ENVIRONMENT = "fixture-point-mass"

    def __init__(self, control: str = "discrete", gain: float = 4.0, noise: float = 0.01,
                 episode_len: int = 30, limit: float = 3.0, goal_radius: float = 1.5,
                 stream: str = "pm-0", seed: int = 0,
                 clock: Callable[[], float] = time.time):
        if control not in ("discrete", "box"):
            raise ValueError("control must be 'discrete' or 'box'")
        self.control, self.noise = control, float(noise)
        self.episode_len, self.limit, self.goal_radius = int(episode_len), float(limit), \
            float(goal_radius)
        self.stream, self._clock = stream, clock
        self.set_dynamics(gain=gain)
        self._rng = np.random.default_rng(seed)
        self._episodes, self._episode, self._seq, self._done = 0, None, 0, True
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.goal = np.zeros(2)
        self.start_override: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if control == "discrete":
            self._aspec = ActionSpec.discrete("pm.push", 5, units=("index",),
                                              tick_seconds=DT)
        else:
            self._aspec = ActionSpec.box("pm.force", [-1.0, -1.0], [1.0, 1.0],
                                         ("N", "N"), tick_seconds=DT)
        mk = lambda n, u: ChannelSpec(n, "vector", (2,), "float64", "sensor", True,
                                      required=True, units=u, frame="world")
        self._ospec = ObservationSpec(self.ENVIRONMENT, (mk("pos", "m"), mk("vel", "m/s"),
                                                         mk("goal", "m")))

    def dynamics_params(self) -> Dict[str, float]:
        return {"gain": self.gain}

    def set_dynamics(self, gain: float) -> None:
        self.gain = float(gain)

    def observation_spec(self) -> ObservationSpec:
        return self._ospec

    def action_spec(self) -> ActionSpec:
        return self._aspec

    def reset(self, seed: Optional[int] = None) -> List[Observation]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._episodes += 1
        self._episode = f"{self.stream}/ep{self._episodes}"
        self._seq, self._done = 0, False
        r = self.goal_radius
        self.goal = self._rng.uniform(-r, r, 2)
        if self.start_override is not None:
            self.pos, self.vel = (np.array(x, np.float64) for x in self.start_override)
        else:
            self.pos = self._rng.uniform(-r, r, 2)
            self.vel = np.zeros(2)
        return self._observe()

    def force_of(self, cmd) -> np.ndarray:
        if self.control == "discrete":
            return DISCRETE_FORCES[int(cmd)]
        return np.asarray(cmd, np.float64)

    def step(self, action: Action):
        if self._done:
            raise ContractError("step() before reset() or after an episode ended")
        if not isinstance(action, Action):
            raise RecordValidationError("step() takes an Action record")
        cmd = self._aspec.check_action(action)
        if (action.environment, action.stream, action.episode, action.seq) != (
                self.ENVIRONMENT, self.stream, self._episode, self._seq):
            raise RecordValidationError(f"stale action for {action.episode!r}@{action.seq}")
        f = self.force_of(cmd)
        self.vel = 0.8 * self.vel + self.gain * f * DT + self._rng.normal(0, self.noise, 2)
        p = self.pos + self.vel * DT
        hit = np.abs(p) > self.limit
        self.pos = np.clip(p, -self.limit, self.limit)
        self.vel = np.where(hit, 0.0, self.vel)
        self._seq += 1
        trunc = self._seq >= self.episode_len
        self._done = trunc
        return self._observe(), False, trunc, {
            "client_recovery": False,
            "executed_action": action.replace(t_complete=max(float(self._clock()),
                                                             action.t_dispatch))}

    def _observe(self) -> List[Observation]:
        base = (self.ENVIRONMENT, self.stream, self._episode, self._seq,
                self._seq * DT, float(self._clock()))
        return [Observation(*base, "pos", "sensor", {"value": self.pos.copy()}),
                Observation(*base, "vel", "sensor", {"value": self.vel.copy()}),
                Observation(*base, "goal", "sensor", {"value": self.goal.copy()})]


# ---- record helpers ----------------------------------------------------------

def _by_channel(obs: List[Observation]) -> Dict[str, Observation]:
    return {o.channel: o for o in obs}


def state_from_obs(obs: List[Observation]) -> Tuple[EntityBatch, np.ndarray, str]:
    """-> (EntityBatch B=1 N=1 D=2 A=1, goal (2,), provenance). Refuses a
    batch that mixes provenances: the result must be one kind of fact."""
    ch = _by_channel(obs)
    provs = {ch[c].provenance for c in ("pos", "vel")}
    if len(provs) != 1:
        raise ContractError(f"mixed provenance {provs}")
    s = EntityBatch(np.asarray(ch["pos"].value, np.float64).reshape(1, 1, 2),
                    np.asarray(ch["vel"].value, np.float64).reshape(1, 1, 2),
                    np.ones((1, 1, 1)))
    return s, np.asarray(ch["goal"].value, np.float64), provs.pop()


def policy_vector(obs: List[Observation]) -> np.ndarray:
    """(4,) float32 [pos - goal, vel]: the controller's input."""
    ch = _by_channel(obs)
    return np.concatenate([ch["pos"].value - ch["goal"].value,
                           ch["vel"].value]).astype(np.float32)


def make_action(obs: List[Observation], spec: ActionSpec, cmd) -> Action:
    h = obs[0]
    if spec.kind == "discrete":
        cmd = int(cmd)
    return Action(h.environment, h.stream, h.episode, h.seq, spec.spec_id, cmd,
                  UNKNOWN, time.time())


class GoalReward:
    """reward_fn for BoundedPlanner: -||pos - goal|| (goal set per episode)."""

    def __init__(self):
        self.goal = np.zeros(2)

    def __call__(self, next_cont, action):
        return -np.linalg.norm(np.asarray(next_cont)[:, :2] - self.goal[None], axis=1)


def run_episode(env, choose: Callable, seed: int, on_step: Optional[Callable] = None,
                ) -> Dict[str, Any]:
    """choose(obs) -> command. on_step(obs, cmd, next_obs). Real outcomes:
    final and mean distance to the goal."""
    obs = env.reset(seed=seed)
    spec = env.action_spec()
    dists = []
    while True:
        cmd = choose(obs)
        nxt, term, trunc, _info = env.step(make_action(obs, spec, cmd))
        if on_step is not None:
            on_step(obs, cmd, nxt)
        ch = _by_channel(nxt)
        dists.append(float(np.linalg.norm(ch["pos"].value - ch["goal"].value)))
        obs = nxt
        if term or trunc:
            break
    return {"final_dist": dists[-1], "mean_dist": float(np.mean(dists)),
            "late_dist": float(np.mean(dists[len(dists) // 2:])), "steps": len(dists)}


class ProportionalController:
    """Noisy PD controller on policy_vector: the stand-in learned controller."""

    def __init__(self, kp: float = 1.0, kd: float = 0.3, noise: float = 0.5,
                 discrete: bool = True, seed: int = 0):
        self.kp, self.kd, self.noise, self.discrete = kp, kd, noise, discrete
        self.rng = np.random.default_rng(seed)

    def force(self, v) -> np.ndarray:
        v = np.asarray(v, np.float64)
        f = -self.kp * v[:2] - self.kd * v[2:] + self.rng.normal(0, self.noise, 2)
        return np.clip(f, -1, 1)

    def __call__(self, v):
        f = self.force(v)
        if not self.discrete:
            return f
        return int(np.argmax(DISCRETE_FORCES @ f - 0.25 * (DISCRETE_FORCES ** 2).sum(1)))


# ---- the mechanism -------------------------------------------------------------

class PointMassMechanism(BaseMechanism):
    """vel'_j = d_j vel_j + (G f)_j dt ; pos' = pos + vel' dt (one entity)."""

    def __init__(self, mechanism_id: str = "pm.linear"):
        super().__init__(mechanism_id, 1, 2, 1, 2, ("free", "wall"),
                         {"form": "linear_damped_force", "uses_absolute_position": False})
        self.damping = np.full(2, 0.5)
        self.gain = np.zeros((2, 2))
        self.var_vel = np.full(2, 1.0)
        self.var_pos = np.full(2, 1e-2)

    def set_params(self, damping, gain, var_vel, var_pos) -> None:
        self.damping = np.asarray(damping, np.float64).reshape(2)
        self.gain = np.asarray(gain, np.float64).reshape(2, 2)
        self.var_vel = np.asarray(var_vel, np.float64).reshape(2)
        self.var_pos = np.asarray(var_pos, np.float64).reshape(2)
        if np.any(self.var_vel <= 0) or np.any(self.var_pos <= 0):
            raise ContractError("variances must be > 0")
        self.version += 1

    def _means(self, state, action, dt):
        vel = state.vel[:, 0, :]
        f = action[:, 0, :]
        v2 = self.damping[None] * vel + (f @ self.gain.T) * dt[:, None]
        p2 = state.pos[:, 0, :] + v2 * dt[:, None]
        return p2, v2

    def _predict(self, state, action, dt) -> MixedOutcome:
        p2, v2 = self._means(state, action, dt)
        B = state.B
        mean = np.concatenate([p2, v2], 1)
        var = np.broadcast_to(np.concatenate([self.var_pos, self.var_vel]), (B, 4)).copy()
        probs = np.broadcast_to(np.array([1 - 1e-3, 1e-3]), (B, 1, 2))   # events not modelled
        return MixedOutcome.gaussian_categorical(self.layout, mean, var, probs)

    def _fit(self, batch: TransitionBatch, **kw):
        vel, f, dt = batch.state.vel[:, 0], batch.action[:, 0], batch.dt
        v2 = batch.next_state.vel[:, 0]
        for j in range(2):
            X = np.stack([vel[:, j], f[:, 0] * dt, f[:, 1] * dt], 1)
            coef, *_ = np.linalg.lstsq(X, v2[:, j], rcond=None)
            self.damping[j], self.gain[j] = coef[0], coef[1:]
        p2, vp = self._means(batch.state, batch.action, dt)
        self.var_vel = np.maximum(((v2 - vp) ** 2).mean(0), 1e-6)
        self.var_pos = np.maximum(((batch.next_state.pos[:, 0] - p2) ** 2).mean(0), 1e-8)
        return {"damping": self.damping.copy(), "gain": self.gain.copy()}

    def to_record(self, evidence_refs=()) -> Mechanism:
        if self.version < 1:
            raise ContractError("an unfitted mechanism has no record")
        return Mechanism(self.mechanism_id, self.version, ("pos", "vel", "force", "dt"),
                         ("next_pos", "next_vel"),
                         {"kind": "support_box", "fitted": self._lo is not None},
                         {"damping": self.damping.copy(), "gain": self.gain.copy(),
                          "var_vel": self.var_vel.copy(), "var_pos": self.var_pos.copy()},
                         evidence_refs=tuple(evidence_refs),
                         payload={"form": self.assumptions["form"]})

    @classmethod
    def from_record(cls, rec: Mechanism) -> "PointMassMechanism":
        m = cls(rec.mechanism_id)
        p = rec.parameters
        m.set_params(p["damping"], p["gain"], p["var_vel"], p["var_pos"])
        m.version = rec.version
        return m


def transitions(rows) -> TransitionBatch:
    """rows: [(EntityBatch s, model action (1,1,2), dt, EntityBatch s')]."""
    s = EntityBatch.concat([r[0] for r in rows])
    a = np.concatenate([np.asarray(r[1]).reshape(1, 1, 2) for r in rows])
    dt = np.array([r[2] for r in rows], np.float64)
    s2 = EntityBatch.concat([r[3] for r in rows])
    return TransitionBatch(s, a, dt, s2)
