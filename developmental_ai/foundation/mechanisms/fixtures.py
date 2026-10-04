"""A synthetic but physically structured fixture: balls in a 2-D box.

NOT a substitute for Minecraft evidence (plan §1). It exists so mechanism
models can be trained, scored and falsified on data whose generating law is
KNOWN, which no SkyBot log offers.

    world        axis-aligned box [0, size]^2 in frame "box_world" (metres,
                 seconds, kg), gravity (0, -g) fixed to the world
    balls        N discs, radius r, mass m_i; ball 0 is ACTUATED: the action
                 is a force (N) on it, held for the step; other balls get 0
    walls        4 STATIC ENTITIES (pos = wall midpoint, attrs carry the
                 inward normal). Making the walls entities is what lets a
                 relational model be translation-invariant honestly: shift
                 every position, walls included, and the physics is unchanged
                 (gravity is a direction, not a place)
    contacts     elastic (restitution e, default 1) ball-wall and ball-ball
    dt           variable per step, uniform in dt_range; the simulator
                 integrates internally with substeps <= max_substep, so the
                 outcome depends on dt the way physics does, not linearly
    events       per entity per step: 0 none, 1 wall contact, 2 ball contact
                 (ball wins if both); walls report 0
    noise        optional Gaussian observation noise on ball pos/vel ONLY
                 (walls are static scenery observed exactly — declared here)

attrs = [radius, mass, is_wall, normal_x, normal_y]  (A = 5)
action = (N_entities, 2) force; only row 0 is ever non-zero.

Records: `trajectory_records` emits Observation/Action records (provenance
"sensor"; the noise-free state is NOT emitted — nothing evaluator-side can
leak into training). Episodes are split train/held-out by
runtime.splits (episode is the unit; plan §7.2.2).

Laws as geometry constraints (law-violation diagnostics):
    box_containment_constraint   hard: no ball centre beyond wall - radius
    free_flight_energy_constraint  soft: KE + PE of an UNACTUATED ball is
        conserved over a step with NO contact; INAPPLICABLE when it touched
        something or was pushed (the law's applicability, plan §2.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..contracts import ABSENT, Action, Observation
from ..geometry.constraints import Constraint
from ..runtime.splits import partition
from .data import EntityStream

EVENT_CLASSES = ("none", "wall", "ball")
FRAME = "box_world"
ATTR_NAMES = ("radius", "mass", "is_wall", "normal_x", "normal_y")


@dataclass
class BoxWorld:
    n_balls: int = 3
    size: float = 4.0
    radius: float = 0.25
    gravity: float = 9.81
    restitution: float = 1.0
    dt_range: Tuple[float, float] = (0.02, 0.1)
    force_scale: float = 6.0
    speed: float = 3.0
    obs_noise: float = 0.0
    max_substep: float = 0.004
    mass_range: Tuple[float, float] = (0.5, 2.0)

    @property
    def n_entities(self) -> int:
        return self.n_balls + 4

    def wall_attrs(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.size
        pos = np.array([[0.0, s / 2], [s, s / 2], [s / 2, 0.0], [s / 2, s]])
        nrm = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        att = np.concatenate([np.zeros((4, 2)), np.ones((4, 1)), nrm], 1)
        return pos, att

    # ---- simulation ----------------------------------------------------------
    def initial(self, rng, speed_scale: float = 1.0):
        nb, r, s = self.n_balls, self.radius, self.size
        pos = np.zeros((nb, 2))
        for i in range(nb):                       # rejection-sample, no overlap
            for _ in range(1000):
                p = rng.uniform(r + 0.05, s - r - 0.05, 2)
                if all(np.linalg.norm(p - pos[j]) > 2 * r + 0.05 for j in range(i)):
                    break
            pos[i] = p
        vel = rng.normal(0, self.speed * speed_scale / np.sqrt(2), (nb, 2))
        mass = rng.uniform(*self.mass_range, nb)
        return pos, vel, mass

    def step(self, pos, vel, mass, force, dt):
        """Integrate one step of length dt; returns (pos, vel, events)."""
        nb, r, s, e = self.n_balls, self.radius, self.size, self.restitution
        n_sub = max(1, int(np.ceil(dt / self.max_substep)))
        h = dt / n_sub
        pos, vel = pos.copy(), vel.copy()
        ev = np.zeros(nb, np.int64)
        acc = np.zeros((nb, 2))
        acc[:, 1] = -self.gravity
        acc[0] += force / mass[0]
        for _ in range(n_sub):
            vel += acc * h
            pos += vel * h
            for d in range(2):
                lo = pos[:, d] < r
                hi = pos[:, d] > s - r
                pos[lo, d] = 2 * r - pos[lo, d]
                pos[hi, d] = 2 * (s - r) - pos[hi, d]
                vel[lo, d] = np.abs(vel[lo, d]) * e
                vel[hi, d] = -np.abs(vel[hi, d]) * e
                ev[(lo | hi) & (ev == 0)] = 1
            dpa = pos[None] - pos[:, None]
            dd = np.sqrt((dpa ** 2).sum(-1))
            close = np.argwhere(np.triu((dd < 2 * r) & (dd > 0), 1))
            for i, j in close:
                dp = pos[j] - pos[i]
                dist = np.linalg.norm(dp)
                if 0 < dist < 2 * r:
                    n = dp / dist
                    rel = float((vel[i] - vel[j]) @ n)
                    if rel > 0:               # approaching
                        mi, mj = mass[i], mass[j]
                        J = (1 + e) * rel / (1 / mi + 1 / mj)
                        vel[i] -= J / mi * n
                        vel[j] += J / mj * n
                    corr = 0.5 * (2 * r - dist) * n
                    pos[i] -= corr
                    pos[j] += corr
                    ev[i] = ev[j] = 2
        return pos, vel, ev

    def simulate(self, seed: int, T: int, speed_scale: float = 1.0,
                 gravity: Optional[float] = None) -> Dict[str, np.ndarray]:
        """One trajectory of T steps. Returns noise-free and observed arrays
        over ALL entities: pos/vel (T+1, Ne, 2), attrs (Ne, 5), actions
        (T, Ne, 2), dts (T,), events (T, Ne)."""
        rng = np.random.default_rng(seed)
        g0 = self.gravity
        if gravity is not None:
            self.gravity = float(gravity)
        try:
            pos, vel, mass = self.initial(rng, speed_scale)
            nb, ne = self.n_balls, self.n_entities
            wpos, watt = self.wall_attrs()
            attrs = np.concatenate([np.stack([np.full(nb, self.radius), mass,
                                              np.zeros(nb), np.zeros(nb), np.zeros(nb)], 1),
                                    watt], 0)
            P = np.zeros((T + 1, ne, 2))
            V = np.zeros((T + 1, ne, 2))
            P[:, nb:] = wpos
            P[0, :nb], V[0, :nb] = pos, vel
            acts = np.zeros((T, ne, 2))
            dts = rng.uniform(*self.dt_range, T)
            evs = np.zeros((T, ne), np.int64)
            for t in range(T):
                f = rng.uniform(-self.force_scale, self.force_scale, 2)
                acts[t, 0] = f
                pos, vel, ev = self.step(pos, vel, mass, f, dts[t])
                P[t + 1, :nb], V[t + 1, :nb], evs[t, :nb] = pos, vel, ev
            Po, Vo = P.copy(), V.copy()
            if self.obs_noise > 0:
                Po[:, :nb] += rng.normal(0, self.obs_noise, Po[:, :nb].shape)
                Vo[:, :nb] += rng.normal(0, self.obs_noise, Vo[:, :nb].shape)
        finally:
            self.gravity = g0
        return {"pos": Po, "vel": Vo, "pos_true": P, "vel_true": V, "attrs": attrs,
                "actions": acts, "dts": dts, "events": evs}

    # ---- indices -----------------------------------------------------------
    def ball_cont_index(self) -> np.ndarray:
        """Continuous outcome dims belonging to balls (walls excluded)."""
        return np.arange(self.n_balls * 4)

    def ball_disc_index(self) -> np.ndarray:
        return np.arange(self.n_balls)


def trajectory_records(traj: Mapping[str, np.ndarray], *, episode: str,
                       stream: str = "s0", environment: str = "box2d",
                       t_wall0: float = 0.0) -> Tuple[List[Observation], List[Action]]:
    T = len(traj["dts"])
    t_env = np.concatenate([[0.0], np.cumsum(traj["dts"])])
    obs, acts = [], []
    for t in range(T + 1):
        obs.append(Observation(environment, stream, episode, t, float(t_env[t]),
                               float(t_wall0 + t_env[t]), "entities", "sensor",
                               {"pos": traj["pos"][t], "vel": traj["vel"][t],
                                "attrs": traj["attrs"],
                                "events": traj["events"][t - 1] if t > 0 else ABSENT}))
        if t < T:
            acts.append(Action(environment, stream, episode, t, "box2d.force",
                               traj["actions"][t], float(traj["dts"][t]),
                               float(t_wall0 + t_env[t]), float(t_wall0 + t_env[t + 1])))
    return obs, acts


def make_streams(world: BoxWorld, n_traj: int, T: int, seed: int = 0,
                 heldout_fraction: float = 0.25, stream: str = "s0",
                 **sim_kw) -> Dict[str, Any]:
    """Simulate n_traj trajectories, split them BY EPISODE with runtime.splits,
    and concatenate each side back to back (as a live buffer would hold
    them). Returns {"train": EntityStream, "heldout": EntityStream,
    "train_episodes", "heldout_episodes", "trajectories"}."""
    trajs = {f"ep{seed}_{k}": world.simulate(seed * 10007 + k, T, **sim_kw)
             for k in range(n_traj)}
    scope = f"box2d:{stream}"
    dev, held = partition([(scope, e) for e in trajs], heldout_fraction)
    if not dev or not held:                        # tiny n: force one held out
        keys = [(scope, e) for e in trajs]
        dev, held = keys[:-1], keys[-1:]
    out = {"trajectories": trajs, "train_episodes": [e for _, e in dev],
           "heldout_episodes": [e for _, e in held]}
    for name, keys in (("train", dev), ("heldout", held)):
        O, A, tw = [], [], 0.0
        for _, e in keys:
            o, a = trajectory_records(trajs[e], episode=e, stream=stream, t_wall0=tw)
            O += o
            A += a
            tw = o[-1].t_wall + 1.0
        out[name] = EntityStream(O, A, frame=FRAME)
    return out


# ======================================================================
# Laws (geometry constraints) for law-violation diagnostics
# ======================================================================

def box_containment_constraint(size: float, tolerance: float = 0.05) -> Constraint:
    """state = {"boundary": "box", "pos": (N, 2), "attrs": (N, 5)}."""
    def applicable(st):
        if st.get("boundary") != "box":
            return False, "no declared box boundary"
        if not np.any(np.asarray(st["attrs"])[:, 2] == 0):
            return False, "no balls"
        return True, ""

    def residual(st):
        p, a = np.asarray(st["pos"]), np.asarray(st["attrs"])
        b = a[:, 2] == 0
        r = a[b, 0:1]
        pen = np.maximum(np.maximum(r - p[b], p[b] - (size - r)), 0.0)
        return float(pen.max())

    return Constraint("box_containment", ("boundary", "box"), applicable, residual,
                      tolerance, "hard", evidence=("fixture: declared walls",))


def free_flight_energy_constraint(gravity: float, tolerance: float = 0.25) -> Constraint:
    """state = {"before": {"pos","vel"}, "after": {"pos","vel"}, "attrs",
    "events": (N,) actual events, "action": (N, 2)}. Residual = max |dE| (J)
    over unactuated balls that touched nothing during the step."""
    def _free(st):
        a = np.asarray(st["attrs"])
        ev = np.asarray(st["events"])
        act = np.abs(np.asarray(st["action"])).sum(1)
        return (a[:, 2] == 0) & (ev == 0) & (act == 0)

    def applicable(st):
        if np.any(np.asarray(st["events"]) < 0):
            return False, "contact events not observed"
        if not np.any(_free(st)):
            return False, "every ball contacted something or was actuated"
        return True, ""

    def residual(st):
        f = _free(st)
        m = np.asarray(st["attrs"])[f, 1]
        E = lambda s: 0.5 * m * (np.asarray(s["vel"])[f] ** 2).sum(1) + \
            m * gravity * np.asarray(s["pos"])[f, 1]
        return float(np.max(np.abs(E(st["after"]) - E(st["before"]))))

    return Constraint("free_flight_energy", ("law", "energy"), applicable, residual,
                      tolerance, "soft", evidence=("fixture: elastic, conservative",))
