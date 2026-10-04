"""AdapterRecorder — drives any EnvironmentAdapter and files what happens.

The store's reference PRODUCER (plan §11: no interface without one). Per
step, in this order, which is the order the store enforces:

    1. log every observation of seq t (evaluator channels land in the
       evaluator partition by provenance, not by the recorder's choice)
    2. predict the tracked channel at t+1 for the chosen command and
       log_prediction(...) — BEFORE the environment is stepped
    3. step; log the EXECUTED action (actual t_complete from the adapter)
    4. client_recovery -> log_episode_end(old, "client_recovery"); the
       prediction stays unresolved (its outcome is ABSENT, not invented)
       otherwise -> log observations of t+1, log_outcome(Evidence[obs t,
       action t, obs t+1] linking the prediction), score it, and file the
       evaluator channels as separate evaluator Evidence
    5. terminated / truncated -> log_episode_end with that reason; reset

`predictor(obs_t, command) -> (mean, std)` is any model; `snapshot` is the
runtime.snapshots.Snapshot (or a provenance dict) whose weights made the
prediction, recorded as snapshot_id + payload["snapshots"].
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..contracts import ABSENT, UNKNOWN, Action, Evidence, Prediction, is_missing
from .scoring import score_point_prediction


def _provenance(snapshot) -> Dict[str, Any]:
    if snapshot is None:
        raise ValueError("a prediction must name the snapshot that made it")
    if hasattr(snapshot, "provenance"):
        return snapshot.provenance()
    return dict(snapshot)


class AdapterRecorder:
    def __init__(self, store, adapter, channel: str,
                 predictor: Callable, snapshot, clock: Callable[[], float],
                 policy: Optional[Callable[[np.random.Generator], Any]] = None,
                 seed: int = 0, commanded_duration: Any = UNKNOWN,
                 score_interpreter: str = "rmse-gaussian"):
        self.store, self.adapter, self.channel = store, adapter, channel
        self.predictor, self.clock = predictor, clock
        self.prov = _provenance(snapshot)
        self.snapshot_id = f"{self.prov['name']}#{self.prov['snapshot_id']}"
        self.spec = adapter.action_spec()
        self.rng = np.random.default_rng(seed)
        self.policy = policy or (lambda rng: int(rng.integers(0, self.spec.n)))
        self.commanded_duration = commanded_duration
        self.interp = score_interpreter
        self.outcomes: List[str] = []
        self.predictions: List[str] = []
        self.unresolved: List[str] = []
        self._n = 0
        self._obs = None

    def _log_all(self, obs) -> Dict[str, Any]:
        for o in obs:
            self.store.log_observation(o)
        return {o.channel: o for o in obs}

    def run(self, n_steps: int) -> "AdapterRecorder":
        st = self.store
        if self._obs is None:
            self._obs = self._log_all(self.adapter.reset())
        for _ in range(int(n_steps)):
            cur = self._obs
            ctx = cur[self.channel]
            cmd = self.policy(self.rng)
            self._n += 1
            mean, std = self.predictor(ctx, cmd)
            pid = f"{ctx.stream}:{ctx.episode}:{ctx.seq}:p{self._n}"
            pred = Prediction(
                pid, f"belief:{ctx.episode}:{ctx.seq}",
                {"predictor": self.prov["content_hash"][:16]},
                self.snapshot_id, self.spec.spec_id, (cmd,), 1,
                {"kind": "gaussian", "mean": float(mean), "std": float(std),
                 "channel": self.channel},
                float(self.clock()), payload={"snapshots": {
                    self.prov["name"]: dict(self.prov)}})
            pref = st.log_prediction(pred, ctx.ref())
            self.predictions.append(pref)
            act = Action(ctx.environment, ctx.stream, ctx.episode, ctx.seq,
                         self.spec.spec_id, cmd, self.commanded_duration,
                         float(self.clock()))
            new, term, trunc, info = self.adapter.step(act)
            executed = info["executed_action"]
            st.log_action(executed)
            if info.get("client_recovery"):
                st.log_episode_end(ctx.scope, "client_recovery", ctx.seq,
                                   float(self.clock()), "client rebuilt")
                self.unresolved.append(pref)
                self._obs = self._log_all(new)
                continue
            nxt = self._log_all(new)
            sensor_now = tuple(o for o in (ctx, nxt[self.channel]))
            missing = any(is_missing(o.value) for o in sensor_now)
            ev = Evidence(f"ev:{pid}", sensor_now, (executed,), (pid,),
                          "partial" if missing else "valid", "sensor",
                          "reading ABSENT" if missing else "")
            oref = st.log_outcome(ev)
            self.outcomes.append(oref)
            if not missing:
                err, contra = score_point_prediction(pred, ev, self.channel,
                                                     ctx.seq)
                st.score(oref, pid, err, contra, self.interp, 1)
            ev_obs = tuple(o for o in new if o.provenance == "evaluator")
            if ev_obs:
                st.log_outcome(Evidence(f"evaluator:{pid}", ev_obs, (), (),
                                        "valid", "evaluator"))
            if term or trunc:
                st.log_episode_end(nxt[self.channel].scope,
                                   "terminated" if term else "truncated",
                                   nxt[self.channel].seq, float(self.clock()))
                self._obs = self._log_all(self.adapter.reset())
            else:
                self._obs = nxt
        return self
