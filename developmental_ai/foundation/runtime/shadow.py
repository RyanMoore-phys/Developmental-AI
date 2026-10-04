"""ShadowRecorder — live SkyBot steps, filed as foundation records (plan §8 step 2).

WHAT IT IS FOR
    Plan §8's progression is Offline -> SHADOW -> limited integration ->
    default. Shadow means "compute predictions during a run without
    controlling actions or rewards". This is that stage for the evidence
    infrastructure, and it closes two gates at once:

      * Stage 2: "existing SkyBot data can travel through the interfaces" —
        the LIVE sensor transport, executed actions and episode boundaries
        become Observation / Action / Evidence records;
      * Stage 5: "save predictions before outcomes arrive, including source
        model versions" — the world model's one-step prior is logged BEFORE
        env.step, so the store's write order proves it was not made with
        the answer in hand.

    It is OFF by default (`foundation.shadow.enabled: false`) and
    `from_config` returns None when off, so the disabled loop runs exactly
    the code it ran before plus one `is not None` test per call site.

WHAT IT MUST NEVER DO (tests/_foundation_shadow_smoke.py contract C)
    Change the active learner. Concretely, it:
      * reads only — it copies the numbers it keeps (transport vectors,
        oracle readings, scalar rewards) and never writes into obs, info,
        reward, replay, rollout or RSSM state objects;
      * consumes NO random numbers. The world-model prediction is the
        RSSM prior's PROBABILITIES at t+1, computed with the same modules
        `RSSM.imagine_step` uses, but without its categorical sample — so
        no Gumbel noise is drawn. RSSMPredictor (adapters/
        skybot_world_model.py) instead forks the CPU RNG around a sampling
        imagine_step; that is exact single-threaded, but with
        `async_wm.enabled: true` (the live config) the trainer thread draws
        from the same global generator, and fork_rng's restore-on-exit
        would REWIND whatever the trainer drew meanwhile. Not drawing at all
        is the only shadow-safe option here. The two paths agree in eval
        mode (contract-tested), so this is not a second model;
      * never toggles train/eval mode, never takes gradients, and holds
        the loop's `_wm_param_lock` while it reads weights so an async
        optimizer step cannot be observed half-applied;
      * never raises into the loop. Any exception, and any wall-time budget
        overrun, disables the recorder for the rest of the run with ONE
        logged warning.

    ESCAPE (CLAUDE.md §4.1 — a guard needs a reachable re-opener): the
    self-disable is per process. Fix the cause (raise max_ms_per_step, lower
    every_n_steps, free disk) and relaunch with foundation.shadow.enabled:
    true. Nothing persistent records the disable, so nothing can latch it.

RECORD LAYOUT (scope = environment / stream / episode)
    stream   "stream-<i>", one per parallel env; streams are never mixed.
    episode  "<run_id>-e<k>": a fresh id after every boundary the LOOP acts
             on (any done, a client rebuild, the episodic fleet re-reset),
             so a reset never splices two worlds into one scope.
    seq      the agent-step index within that episode. Observation seq t is
             the transport the env returned with the previous step (what
             the agent had sensed when it chose action t); action t is what
             it then executed; seq t+1 is the consequence.
    GREEN    every policy-visible sensor of the bus transport, provenance
             "sensor", split dev/held-out by the store's episode split.
    RED      every key of info["oracle"], provenance "evaluator", which the
             store files in the EVALUATOR partition only (a learner's view
             cannot reach it). Its outcome is separate evaluator Evidence.
    action   t_dispatch = the loop's own env-step dispatch clock (_t_env0);
             t_complete = the moment the loop held that env's result (an
             upper bound on the true completion: the fleet is collected in
             index order). duration UNKNOWN (ticks are in the payload).
    ending   terminated / truncated / client_recovery, as the store defines
             them; a client rebuild leaves the step's prediction unresolved
             (its outcome is ABSENT, never invented).

SAMPLING AND BOUNDS
    every_n_steps   record one step in N (all streams on that step). A
                    sampled step logs obs t, the prediction, action t,
                    obs t+1 and the evidence that links them; unsampled
                    steps cost a counter increment and one vector copy per
                    stream.
    max_bytes       the EvidenceStore's own retention budget (oldest
                    evidence evicted first; on-disk bytes never exceed it).
    max_ms_per_step amortised shadow wall time per loop step, averaged over
                    `budget_window` steps; exceeding it disables the
                    recorder. `max_ms_single_step` catches one pathological
                    stall (an fsync on a dying disk) without waiting for
                    the window.
"""

from __future__ import annotations

import collections
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULTS = {
    "enabled": False,
    "root": "runlogs/foundation_shadow",
    "max_bytes": 1 << 30,
    "chunk_bytes": 1 << 20,
    "every_n_steps": 16,
    "max_ms_per_step": 5.0,
    "budget_window": 256,
    "max_ms_single_step": 2000.0,
    "predict": True,
    "fsync": False,
}

GREEN_ORIGIN = "live_transport"


def shadow_config(cfg: Optional[Dict]) -> Dict[str, Any]:
    """The resolved `foundation.shadow` block (defaults filled in)."""
    raw = (((cfg or {}).get("foundation") or {}).get("shadow") or {})
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"foundation.shadow: unknown keys {sorted(unknown)} "
                         f"(known: {sorted(DEFAULTS)})")
    out = dict(DEFAULTS)
    out.update(raw)
    return out


class _Stream:
    __slots__ = ("idx", "name", "ep_n", "episode", "seq", "vec", "oracle",
                 "t_wall", "logged_seq", "any_logged", "ctx", "pid",
                 "adapter", "green")

    def __init__(self, idx: int):
        self.idx = idx
        self.name = f"stream-{idx}"
        self.ep_n = 0
        self.episode = ""
        self.seq = 0
        self.vec = None            # transport at `seq` (a copy), or None
        self.oracle = None         # {name: array copy} at `seq`, or None
        self.t_wall = 0.0
        self.logged_seq = -1       # seq whose observations are in the store
        self.any_logged = False
        self.ctx = None            # {channel: Observation} logged at seq
        self.pid = None            # prediction id made at seq this step
        self.adapter = None
        self.green = None          # the GREEN records logged at logged_seq


class ShadowRecorder:
    """See module docstring. Constructed only by `from_config`."""

    # ------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: Optional[Dict], *, bus=None,
                    action_dim: Optional[int] = None, is_discrete: bool = True,
                    num_streams: int = 1, environment: Optional[str] = None,
                    layout_hash: Optional[str] = None,
                    action_repeat: int = 1) -> Optional["ShadowRecorder"]:
        """None when `foundation.shadow.enabled` is false (the default) —
        and also None, with one warning, when it is on but cannot start:
        a shadow component may never stop the learner from booting."""
        try:
            sc = shadow_config(cfg)
        except Exception as e:
            logger.warning("foundation.shadow: bad config (%s); running "
                           "WITHOUT shadow recording", e)
            return None
        if not bool(sc["enabled"]):
            return None
        try:
            env = environment or str(((cfg or {}).get("environment") or {})
                                     .get("name") or "env")
            return cls(sc, bus=bus, action_dim=action_dim,
                       is_discrete=is_discrete, num_streams=num_streams,
                       environment=env, layout_hash=layout_hash,
                       action_repeat=action_repeat)
        except Exception as e:
            logger.warning("foundation.shadow: could not start (%s: %s); "
                           "running WITHOUT shadow recording. Fix the cause "
                           "and relaunch to re-enable.", type(e).__name__, e)
            return None

    def __init__(self, sc: Dict[str, Any], *, bus, action_dim, is_discrete,
                 num_streams, environment, layout_hash, action_repeat):
        from ..experience import EvidenceStore
        self.cfg = dict(sc)
        self.every = max(1, int(sc["every_n_steps"]))
        self.max_ms = float(sc["max_ms_per_step"])
        self.max_ms_single = float(sc["max_ms_single_step"])
        self.window = max(1, int(sc["budget_window"]))
        self.do_predict = bool(sc["predict"])
        self.bus = bus
        self.layout_hash = layout_hash if bus is not None else None
        self.action_dim = None if action_dim is None else int(action_dim)
        self.is_discrete = bool(is_discrete)
        self.environment = str(environment)
        self.action_repeat = int(action_repeat or 1)
        self.spec_id = (f"{self.environment}:"
                        f"{'discrete' if self.is_discrete else 'box'}"
                        f"{self.action_dim}")
        self.run_id = (time.strftime("%Y%m%dT%H%M%S", time.gmtime())
                       + f"-{os.getpid()}")
        self.store = EvidenceStore(sc["root"], max_bytes=int(sc["max_bytes"]),
                                   chunk_bytes=int(sc["chunk_bytes"]),
                                   fsync=bool(sc["fsync"]))
        self.streams: List[_Stream] = []
        self._ensure_streams(int(num_streams))
        self.enabled = True
        self.disabled_reason: Optional[str] = None
        self.warnings = 0
        self._tick = 0
        self._sampled = False
        self._step_ms = 0.0
        self._ms = collections.deque(maxlen=self.window)
        self.n = collections.Counter()
        self.ms_hist: List[float] = []      # per-step shadow ms (bounded)
        logger.info("foundation.shadow ON: root=%s every_n_steps=%d "
                    "max_bytes=%d max_ms_per_step=%.2f predict=%s run=%s",
                    self.store.root, self.every, self.store.max_bytes,
                    self.max_ms, self.do_predict, self.run_id)

    def _ensure_streams(self, n: int) -> None:
        while len(self.streams) < n:
            st = _Stream(len(self.streams))
            self._new_episode(st)
            if self.bus is not None:
                from ..adapters import SkyBotRecordAdapter
                st.adapter = SkyBotRecordAdapter(self.bus, self.environment,
                                                 stream=st.name)
            self.streams.append(st)

    def _new_episode(self, st: _Stream) -> None:
        st.episode = f"{self.run_id}-e{st.ep_n}"
        st.ep_n += 1
        st.seq = 0
        st.vec = st.oracle = st.ctx = st.pid = st.green = None
        st.logged_seq = -1
        st.any_logged = False

    # ---------------------------------------------------------- failure
    def _fail(self, why: str) -> None:
        """Self-disable for the rest of the run. Exactly one warning."""
        if not self.enabled:
            return
        self.enabled = False
        self.disabled_reason = why
        self.warnings += 1
        logger.warning(
            "foundation.shadow DISABLED for the rest of this run: %s. The "
            "live learner is unaffected (shadow only reads). To re-enable, "
            "fix the cause and relaunch with foundation.shadow.enabled: "
            "true.", why)
        try:
            self.store.close()
        except Exception:
            pass

    def _charge(self, t0: float) -> None:
        self._step_ms += (time.perf_counter() - t0) * 1000.0

    def _end_step_budget(self) -> None:
        ms, self._step_ms = self._step_ms, 0.0
        self._ms.append(ms)
        if len(self.ms_hist) < 100000:
            self.ms_hist.append(ms)
        if ms > self.max_ms_single:
            self._fail(f"one step cost {ms:.1f} ms > max_ms_single_step "
                       f"{self.max_ms_single:.1f}")
        elif len(self._ms) >= self.window:
            mean = float(np.mean(self._ms))
            if mean > self.max_ms:
                self._fail(f"mean shadow cost {mean:.3f} ms/step over the "
                           f"last {len(self._ms)} steps > max_ms_per_step "
                           f"{self.max_ms:.3f}")

    # ------------------------------------------------------- records
    def _green(self, st: _Stream, vec, seq: int, t_wall: float):
        from ..contracts import Observation
        if st.adapter is not None:
            return st.adapter.observations_from_transport(
                vec, st.episode, seq, float(seq), t_wall,
                layout_hash=self.layout_hash, origin=GREEN_ORIGIN)
        return [Observation(self.environment, st.name, st.episode, seq,
                            float(seq), t_wall, "transport", "sensor",
                            {"value": vec, "origin": GREEN_ORIGIN})]

    def _red(self, st: _Stream, oracle, seq: int, t_wall: float):
        from ..contracts import ABSENT, Observation
        out = []
        for name in sorted(oracle or {}):
            v = oracle[name]
            if v is None or not np.all(np.isfinite(v)):
                v = ABSENT
            out.append(Observation(self.environment, st.name, st.episode, seq,
                                   float(seq), t_wall, str(name), "evaluator",
                                   {"value": v, "origin": "info.oracle"}))
        return out

    def _log_obs(self, st: _Stream, vec, oracle, seq: int, t_wall: float):
        green = self._green(st, vec, seq, t_wall)
        for o in green:
            self.store.log_observation(o)
        red = self._red(st, oracle, seq, t_wall)
        for o in red:
            self.store.log_observation(o)
        st.any_logged = True
        st.logged_seq = seq
        st.green = green
        self.n["observations"] += len(green)
        self.n["evaluator_observations"] += len(red)
        return green, red

    @staticmethod
    def _copy_oracle(info) -> Optional[Dict[str, np.ndarray]]:
        orc = info.get("oracle")
        if not isinstance(orc, dict) or not orc:
            return None
        out = {}
        for k, v in orc.items():
            try:
                out[str(k)] = np.array(v, dtype=np.float32, copy=True).reshape(-1)
            except Exception:
                out[str(k)] = None
        return out

    def _transport_of(self, info) -> Optional[np.ndarray]:
        v = info.get("sensors")
        if v is None and self.bus is None:
            v = info.get("proprio")
        if v is None:
            return None
        return np.array(v, dtype=np.float32, copy=True).reshape(-1)

    def _evidence_obs(self, st: _Stream, obs):
        """Evidence cites the VECTOR channels; image channels (fovea, 4K
        floats each) stay individually logged and are reachable from the
        interaction's context observations, rather than being written twice
        per sampled step. Falls back to everything if no vector exists."""
        if st.adapter is None:
            return obs
        spec = st.adapter.observation_spec()
        vec = tuple(o for o in obs if spec.channel(o.channel).kind != "image")
        return vec or obs

    def _command(self, a):
        if self.is_discrete:
            return int(a)
        return np.array(a, dtype=np.float32, copy=True).reshape(-1)

    # -------------------------------------------------- the two calls
    def before_step(self, rssm_state, actions, lock=None,
                    world_model=None) -> None:
        """Called after action selection, BEFORE env.step. On a sampled step
        logs every stream's observation t and the world model's prediction
        for the action about to be executed. Never raises."""
        if not self.enabled:
            return
        t0 = time.perf_counter()
        try:
            self._ensure_streams(len(actions))
            self._tick += 1
            self._sampled = ((self._tick - 1) % self.every) == 0
            for st in self.streams:
                st.pid = None
                st.ctx = None
            if not self._sampled:
                return
            ready = []
            for st in self.streams[:len(actions)]:
                if st.vec is None:
                    self.n["skipped_no_context"] += 1
                    continue
                if st.logged_seq != st.seq:
                    self._log_obs(st, st.vec, st.oracle, st.seq, st.t_wall)
                # the records AS LOGGED: evidence must cite them byte-identical
                st.ctx = {o.channel: o for o in st.green}
                ready.append(st)
            if ready and self.do_predict and world_model is not None:
                self._predict(ready, rssm_state, actions, lock, world_model)
        except Exception as e:
            self._fail(f"before_step raised {type(e).__name__}: {e}")
        finally:
            self._charge(t0)

    def _model_version(self, world_model) -> str:
        try:
            opt = getattr(world_model, "optimizer", None)
            p0 = next(world_model.parameters())
            stp = opt.state.get(p0, {}).get("step") if opt is not None else None
            return f"adam_step={int(stp)}" if stp is not None else "adam_step=0"
        except Exception:
            return "unknown"

    def _predict(self, ready, rssm_state, actions, lock, world_model) -> None:
        import contextlib
        import torch
        from ..contracts import Prediction
        r = world_model.rssm
        S, C = int(r.stochastic_size), int(r.stochastic_classes)
        ad = int(r.action_dim)
        rows, cmds = [], []
        for st in ready:
            cmd = self._command(actions[st.idx])
            if self.is_discrete and not (0 <= cmd < ad):
                self.n["skipped_bad_action"] += 1
                continue
            rows.append(st)
            cmds.append(cmd)
        if not rows:
            return
        a = np.zeros((len(rows), ad), np.float32)
        for i, c in enumerate(cmds):
            if self.is_discrete:
                a[i, c] = 1.0
            else:
                a[i, :] = np.asarray(c, np.float32).reshape(-1)[:ad]
        with (lock if lock is not None else contextlib.nullcontext()):
            with torch.no_grad():
                h0 = rssm_state["h"]
                idx = torch.as_tensor([st.idx for st in rows], device=h0.device,
                                      dtype=torch.long)
                probs = rssm_prior_probs(r, h0.index_select(0, idx),
                                         rssm_state["z"].index_select(0, idx),
                                         a)
            version = self._model_version(world_model)
        snap = f"{self.run_id}:world_model.rssm@{version}"
        t_pred = time.time()
        for i, st in enumerate(rows):
            ctx = next(iter(st.ctx.values()))
            pid = f"{st.name}:{st.episode}:{st.seq}:p"
            pred = Prediction(
                pid, f"rssm:{st.name}:{st.episode}:{st.seq}",
                {"world_model.rssm": version}, snap, self.spec_id,
                (cmds[i],), 1,
                {"kind": "rssm_prior_categorical",
                 "probs": probs[i:i + 1].astype(np.float32)},
                t_pred,
                {"sampling": "none: prior probabilities, no RNG draw",
                 "knowledge": "not conditioned",
                 "representation": (f"rssm_latent[h{r.deterministic_size},"
                                    f"z{S}x{C}]")})
            self.store.log_prediction(pred, ctx.ref())
            st.pid = pid
            self.n["predictions"] += 1

    def after_step(self, step_infos, actions, rewards, dones, restarted,
                   ends, t_dispatch, prim_extrinsic=None, intrinsic=None,
                   fleet_reset=False, dream=False) -> None:
        """Called once per step AFTER reward assembly and BEFORE the loop
        advances/resets streams. `ends[e]` = (terminated, truncated,
        t_result) per env. Logs action t, obs t+1 and the evidence on a
        sampled step; ends episodes on the boundaries the loop acts on;
        caches obs t+1 (a copy) for the next step. Never raises."""
        if not self.enabled:
            return
        t0 = time.perf_counter()
        try:
            self._after(step_infos, actions, rewards, dones, restarted, ends,
                        t_dispatch, prim_extrinsic, intrinsic, fleet_reset,
                        dream)
        except Exception as e:
            self._fail(f"after_step raised {type(e).__name__}: {e}")
        finally:
            self._charge(t0)
            if self.enabled:
                self._end_step_budget()

    def _after(self, step_infos, actions, rewards, dones, restarted, ends,
               t_dispatch, prim_extrinsic, intrinsic, fleet_reset, dream):
        from ..contracts import UNKNOWN, Action, Evidence
        n = len(actions)
        self._ensure_streams(n)
        self.n["steps"] += 1
        if self._sampled:
            self.n["sampled_steps"] += 1
        for e in range(n):
            st = self.streams[e]
            info = step_infos[e] if isinstance(step_infos[e], dict) else {}
            term, trunc, t_res = ends[e]
            t_res = max(float(t_res), float(t_dispatch))
            recovery = bool(restarted[e]) or bool(info.get("env_restarted"))
            boundary = bool(dones[e]) or recovery or bool(fleet_reset)
            nvec = None if recovery else self._transport_of(info)
            norc = None if recovery else self._copy_oracle(info)
            if self._sampled and st.ctx is not None and st.logged_seq == st.seq:
                act = Action(
                    self.environment, st.name, st.episode, st.seq,
                    self.spec_id, self._command(actions[e]), UNKNOWN,
                    float(t_dispatch), t_res,
                    {"actor": "dream" if dream else "policy",
                     "action_repeat_ticks": self.action_repeat,
                     "t_complete_basis": "loop held the env result"})
                self.store.log_action(act)
                self.n["actions"] += 1
                if recovery:
                    if st.pid is not None:
                        self.n["unresolved_predictions"] += 1
                elif nvec is not None:
                    green, red = self._log_obs(st, nvec, norc, st.seq + 1,
                                               t_res)
                    ctx_obs = self._evidence_obs(st, tuple(st.ctx.values()))
                    green = self._evidence_obs(st, tuple(green))
                    pay = {"reward_env": float(rewards[e]),
                           "reward_replay": float(
                               prim_extrinsic if (e == 0 and prim_extrinsic
                                                  is not None)
                               else rewards[e]),
                           "done": bool(dones[e]), "terminated": bool(term),
                           "truncated": bool(trunc)}
                    if intrinsic is not None:
                        pay["intrinsic"] = float(intrinsic[e])
                    eid = f"ev:{st.name}:{st.episode}:{st.seq}"
                    self.store.log_outcome(Evidence(
                        eid, ctx_obs + tuple(green), (act,),
                        (st.pid,) if st.pid is not None else (),
                        "valid", "sensor", "", pay))
                    self.n["outcomes"] += 1
                    if red:
                        self.store.log_outcome(Evidence(
                            f"evaluator:{eid}", tuple(red), (), (), "valid",
                            "evaluator"))
                        self.n["evaluator_outcomes"] += 1
            if boundary:
                if st.any_logged:
                    reason = ("client_recovery" if recovery else
                              "terminated" if term else "truncated")
                    detail = ("episodic fleet re-reset (primary stream ended)"
                              if (fleet_reset and not dones[e]
                                  and not recovery) else "")
                    self.store.log_episode_end(
                        _scope(self.environment, st), reason,
                        st.seq + (0 if recovery else 1), t_res, detail)
                    self.n[f"end_{reason}"] += 1
                self._new_episode(st)
            else:
                st.seq += 1
                st.vec, st.oracle, st.t_wall = nvec, norc, t_res
            st.pid = None
            st.ctx = None

    # ------------------------------------------------------ reporting
    def stats(self) -> Dict[str, Any]:
        ms = np.asarray(self.ms_hist, np.float64)
        return {"enabled": self.enabled, "disabled_reason": self.disabled_reason,
                "warnings": self.warnings, "counts": dict(self.n),
                "ms_per_step_mean": float(ms.mean()) if ms.size else 0.0,
                "ms_per_step_p95": (float(np.percentile(ms, 95))
                                    if ms.size else 0.0),
                "store_bytes": self.store.disk_bytes(),
                "root": self.store.root, "run_id": self.run_id}

    def close(self) -> None:
        try:
            if self.enabled:
                self.store.close()
            logger.info("foundation.shadow summary: %s", self.stats())
        except Exception:
            pass


def rssm_prior_probs(rssm, h, z, actions) -> np.ndarray:
    """(k, S, C) prior probabilities of z_{t+1} given (h_t, z_t, a_t).

    `RSSM.imagine_step` up to — and NOT including — its categorical sample,
    which is the only random draw in it: h_{t+1} is computed before the
    sample and the prior depends only on h_{t+1}. So these probabilities
    equal RSSMPredictor's horizon-1 outcome["probs"] in train AND eval mode
    (contract-tested against it, so an edit to imagine_step that this
    stops mirroring fails a test instead of silently forking the model).
    Inputs are never modified; gradients are off; no RNG is consumed.
    """
    import torch
    with torch.no_grad():
        dev = h.device
        at = torch.as_tensor(np.asarray(actions, np.float32), device=dev)
        if getattr(rssm, "film", None) is not None:
            base = rssm.film(at, z)
        else:
            base = torch.cat([z, at], dim=-1)
        h1 = rssm.gru_norm(rssm.gru(rssm.gru_input_proj(base), h))
        S, C = int(rssm.stochastic_size), int(rssm.stochastic_classes)
        logits = rssm.prior_net(h1).view(h.shape[0], S, C)
        return torch.softmax(logits.float(), -1).cpu().numpy()


def _scope(environment: str, st: _Stream):
    from ..contracts import Scope
    return Scope(environment, st.name, st.episode)
