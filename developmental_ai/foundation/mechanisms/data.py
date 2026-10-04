"""Records -> training/evaluation arrays, with reset exclusion (plan §7.4).

`EntityStream` takes Observation and Action RECORDS in arrival order — the
way a live buffer holds them, episodes back to back, streams interleaved —
and decides which (obs_t, action_t, obs_t+1) triples are real transitions.
It does not re-implement the rule: every pair goes through
`contracts.streams.transition_problems` (same environment/stream/episode,
adjacent seq, time monotone, trainable provenance), and on top of that:

    an Action with the obs's scope and seq must exist, with a KNOWN duration
    (dt is a model input; an UNKNOWN duration is not silently 0 or 1)
    both observations must carry pos/vel (a missing payload is ABSENT, not
    zeros); a missing `events` makes the event MISSING, not class 0

A multi-step window is valid only if every pair inside it is. So a window
that would splice two episodes is REFUSED rather than averaged in — the
counts of refused windows are reported so a test can prove the rule bit.

Observation payload (channel `channel`, default "entities"):
    pos (N, D), vel (N, D), attrs (N, A), events (N,) int — the contacts
    during the step that ENDED at this observation (ABSENT at seq 0).
Action command: (N, Ad) numeric; Action.duration: dt in seconds.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..contracts import UNKNOWN, Action, ContractError, Observation, is_missing
from ..contracts.streams import transition_problems
from .api import MISSING, EntityBatch, TransitionBatch


class EntityStream:
    def __init__(self, observations: Sequence[Observation], actions: Sequence[Action],
                 channel: str = "entities", frame: str = "world"):
        obs = [o for o in observations if isinstance(o, Observation) and o.channel == channel]
        if len(obs) < 2:
            raise ContractError(f"EntityStream needs >= 2 {channel!r} observations")
        self.channel, self.frame = channel, frame
        self.observations = obs
        amap: Dict[Tuple, Action] = {}
        for a in actions:
            if not isinstance(a, Action):
                raise ContractError("actions must be Action records")
            key = (a.scope, a.seq)
            if key in amap:
                raise ContractError(f"duplicate action for {key}")
            amap[key] = a
        shape = None
        self._pos, self._vel, self._attrs, self._events = [], [], [], []
        self._has_state = []
        for o in obs:
            p = o.payload
            ok = all(k in p and not is_missing(p[k]) for k in ("pos", "vel", "attrs"))
            self._has_state.append(ok)
            if ok:
                pos, vel, att = (np.asarray(p[k], np.float64) for k in ("pos", "vel", "attrs"))
                if shape is None:
                    shape = (pos.shape, att.shape)
                if (pos.shape, att.shape) != shape or vel.shape != pos.shape:
                    raise ContractError(f"entity shapes change inside the stream at "
                                        f"{o.scope}/{o.seq}: {pos.shape}/{att.shape}")
            else:
                pos = vel = att = None
            ev = p.get("events", UNKNOWN)
            self._pos.append(pos)
            self._vel.append(vel)
            self._attrs.append(att)
            self._events.append(None if is_missing(ev) else np.asarray(ev, np.int64))
        if shape is None:
            raise ContractError("no observation carries an entity state")
        (self.N, self.D), self.A = shape[0], shape[1][1]
        n = len(obs)
        self.pair_valid = np.zeros(n - 1, bool)
        self.pair_problems: List[List[str]] = []
        self._act = [None] * (n - 1)
        self._dt = np.full(n - 1, np.nan)
        for i in range(n - 1):
            pr = transition_problems(obs[i], obs[i + 1])
            a = amap.get((obs[i].scope, obs[i].seq))
            if a is None:
                pr.append("no action recorded for this step")
            elif a.duration is UNKNOWN:
                pr.append("action duration UNKNOWN (dt is a model input)")
            else:
                cmd = np.asarray(a.command, np.float64)
                if cmd.ndim != 2 or cmd.shape[0] != self.N:
                    pr.append(f"action command shape {cmd.shape} is not (N={self.N}, Ad)")
                elif a.duration <= 0:
                    pr.append("action duration must be > 0")
                else:
                    self._act[i] = cmd
                    self._dt[i] = float(a.duration)
            if not (self._has_state[i] and self._has_state[i + 1]):
                pr.append("entity state ABSENT at one end")
            self.pair_problems.append(pr)
            self.pair_valid[i] = not pr
        ads = {self._act[i].shape[1] for i in np.flatnonzero(self.pair_valid)}
        if len(ads) > 1:
            raise ContractError(f"action dims vary inside the stream: {sorted(ads)}")
        self.Ad = ads.pop() if ads else 0

    def __len__(self):
        return len(self.observations)

    # ---- windows --------------------------------------------------------------
    def windows(self, horizon: int) -> np.ndarray:
        """Start indices whose `horizon` consecutive pairs are ALL valid."""
        H = int(horizon)
        if H < 1:
            raise ContractError("horizon must be >= 1")
        v = self.pair_valid.astype(int)
        if len(v) < H:
            return np.zeros(0, int)
        c = np.concatenate([[0], np.cumsum(v)])
        return np.flatnonzero(c[H:] - c[:-H] == H)

    def refused_windows(self, horizon: int) -> int:
        """Windows a naive index-based slicer would have produced but the
        transition rule refuses (they cross a reset, a stream or a gap)."""
        return max(0, len(self.pair_valid) - int(horizon) + 1) - len(self.windows(horizon))

    def _state(self, idx) -> EntityBatch:
        return EntityBatch(np.stack([self._pos[i] for i in idx]),
                           np.stack([self._vel[i] for i in idx]),
                           np.stack([self._attrs[i] for i in idx]), self.frame)

    def _ev(self, i):
        e = self._events[i]
        return np.full(self.N, MISSING, np.int64) if e is None else e

    def transitions(self, starts=None) -> TransitionBatch:
        """One-step transitions at `starts` (default: every valid pair)."""
        idx = np.flatnonzero(self.pair_valid) if starts is None else np.asarray(starts, int)
        if len(idx) == 0:
            raise ContractError("no valid transitions")
        if not np.all(self.pair_valid[idx]):
            raise ContractError("requested a transition the stream rule refuses: "
                                f"{self.pair_problems[int(idx[~self.pair_valid[idx]][0])]}")
        return TransitionBatch(self._state(idx), np.stack([self._act[i] for i in idx]),
                               self._dt[idx], self._state(idx + 1),
                               np.stack([self._ev(i + 1) for i in idx]))

    def window_batch(self, starts, horizon: int, allow_invalid: bool = False):
        """(state0, actions (H,B,N,Ad), dts (H,B), targets [TransitionBatch-
        style dicts per h]) for windows at `starts`. `allow_invalid=True`
        exists ONLY so a test can measure what a splice would have cost;
        it fills a missing action with zeros and dt with the median."""
        st = np.asarray(starts, int)
        H = int(horizon)
        valid = set(self.windows(H).tolist())
        bad = [s for s in st.tolist() if s not in valid]
        if bad and not allow_invalid:
            raise ContractError(f"{len(bad)} window(s) cross a refused pair "
                                f"(e.g. start {bad[0]}: "
                                f"{[p for p in self.pair_problems[bad[0]:bad[0] + H] if p][:1]})")
        med = float(np.nanmedian(self._dt))
        acts = np.stack([[self._act[s + h] if self._act[s + h] is not None
                          else np.zeros((self.N, max(self.Ad, 1))) for s in st]
                         for h in range(H)])
        dts = np.stack([[self._dt[s + h] if np.isfinite(self._dt[s + h]) else med
                         for s in st] for h in range(H)])
        targets = [{"continuous": self._state(st + h + 1).continuous(),
                    "discrete": np.stack([self._ev(s + h + 1) for s in st])}
                   for h in range(H)]
        return self._state(st), acts, dts, targets

    # ---- episodes ------------------------------------------------------------
    def episode_of(self, i: int) -> Tuple[str, str]:
        o = self.observations[i]
        return (f"{o.environment}:{o.stream}", o.episode)
