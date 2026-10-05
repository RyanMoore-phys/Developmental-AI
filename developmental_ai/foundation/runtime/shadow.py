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
      * never raises into the loop. Any exception in the core recorder
        disables it for the rest of the run with ONE logged warning. A
        wall-time budget overrun DEGRADES IN STAGES (2026-10-05), one warning
        per stage, a fresh budget window after each: "full" -> observable
        forecasts off ("categorical") -> categorical predictions off
        ("records") -> recorder disabled. stats()["stage"] says which.
      * isolates the OPTIONAL observable forecast (observable_predict): its
        exceptions are counted (counts["observable_errors"]) and never reach
        the recorder-wide disable; `observable_max_failures` consecutive
        failures at one site (forecast / persistence / score) turn ONLY the
        observable forecast off, with one warning.

    ESCAPE (CLAUDE.md §4.1 — a guard needs a reachable re-opener): the
    self-disable and every degrade stage are per process. Fix the cause (raise max_ms_per_step, lower
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
             payload["source"] (2026-10-04, OPTIONAL — stores written before
             it simply lack the key; readers must treat it as unrecorded):
             WHO chose the executed command — "policy" (the shared policy
             decided this step), "option_slot:<n>" (an option is running in
             slot n; payload["skill"] = the skill_id of the frame that
             actually resolved the primitive, payload["option_depth"] its
             nesting depth, payload["option_root_skill"] the invoked root
             when nested, payload["scripted"] when a scripted slot), or
             "option_fallback" (an option closed inside act() and emitted a
             fallback primitive), "dream_actor" (the dream actor drove the
             step). Read from the option executor's state AFTER act() and
             BEFORE env.step — a pure read (see action_sources).
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
                    `budget_window` steps; exceeding it degrades the
                    recorder one stage (observable -> categorical ->
                    disabled), see "never raises" above. `max_ms_single_step` catches one pathological
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
    "observable_predict": False,
    # Consecutive failures at ONE observable site (forecast / persistence /
    # score) before the observable forecast alone is switched off.
    "observable_max_failures": 3,
    "fsync": False,
    # Max wait for the world model's param lock before SKIPPING one
    # prediction (2026-10-04). The async WM trainer holds that lock while it
    # updates; a blocking acquire made the ACTING thread wait on every
    # sampled step and tripped the budget live (5.13 ms/step vs 0.7 on CPU).
    "predict_lock_timeout_ms": 1.0,
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
    # PER-KEY reads, not out.update(raw) (2026-10-04): dict.update copies at
    # C level and bypasses TrackedConfig.__getitem__, so the live config
    # echo reported every foundation.shadow.* key as "never read" while the
    # recorder was running on exactly those values.
    for k in list(raw.keys()):
        out[k] = raw[k]
    return out


class _Stream:
    __slots__ = ("idx", "name", "ep_n", "episode", "seq", "vec", "oracle",
                 "t_wall", "logged_seq", "any_logged", "ctx", "pid",
                 "adapter", "green", "source", "observable")

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
        self.observable = None
        self.source = None         # who chose action `seq` (action_sources)


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
        self.do_observable = bool(sc["observable_predict"])
        self.obs_fail_limit = max(1, int(sc["observable_max_failures"]))
        self._obs_fail_run = collections.Counter()   # site -> consecutive
        self.observable_off_reason: Optional[str] = None
        self.degraded: List[Dict[str, str]] = []     # budget stages taken
        self.lock_timeout_s = max(0.0, float(sc["predict_lock_timeout_ms"])) / 1000.0
        self.t = collections.defaultdict(float)   # ms: lock_wait, predict, write
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
                    "max_bytes=%d max_ms_per_step=%.2f predict=%s "
                    "observable_predict=%s run=%s",
                    self.store.root, self.every, self.store.max_bytes,
                    self.max_ms, self.do_predict, self.do_observable,
                    self.run_id)

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
        st.observable = None
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
            if mean <= self.max_ms:
                return
            # DEGRADE BEFORE DYING, IN STAGES (2026-10-04, staged 2026-10-05).
            # Most expensive first: the observable forecast (a full decoder
            # forward inside the WM lock), then the categorical prediction
            # (lock + GRU forward). Observations, actions and outcomes are
            # cheap and are the evidence itself. Each stage starts a fresh
            # window; only a recorder STILL over budget with both off dies.
            totals = (f"totals ms: lock_wait {self.t['lock_wait']:.0f}, "
                      f"predict {self.t['predict']:.0f}, observable "
                      f"{self.t['observable']:.0f}, write "
                      f"{self.t['write']:.0f}")
            if self.do_predict and self.do_observable:
                self._ms.clear()
                self.degraded.append({"stage": "categorical",
                                      "reason": f"mean {mean:.3f} ms/step"})
                self._observable_off(
                    f"mean {mean:.3f} ms/step over {self.window} steps > "
                    f"max_ms_per_step {self.max_ms:.3f}; {totals}")
            elif self.do_predict:
                self.do_predict = False
                self.warnings += 1
                self._ms.clear()
                self.degraded.append({"stage": "records",
                                      "reason": f"mean {mean:.3f} ms/step"})
                logger.warning(
                    "foundation.shadow: predictions OFF for the rest of this "
                    "run (mean %.3f ms/step over %d steps > max_ms_per_step "
                    "%.3f; %s; %d predictions skipped on a busy lock). Still "
                    "recording observations, actions and outcomes.",
                    mean, self.window, self.max_ms, totals,
                    self.n["skipped_lock_busy"])
            else:
                self._fail(f"mean shadow cost {mean:.3f} ms/step over the "
                           f"last {len(self._ms)} steps > max_ms_per_step "
                           f"{self.max_ms:.3f} (predictions already off; "
                           f"{totals})")

    # ------------------------------------------- observable isolation
    def _observable_off(self, why: str) -> None:
        """Observable forecast off for the rest of the run. ONE warning.
        Per process (relaunch to re-enable), like every other stage."""
        if not self.do_observable:
            return
        self.do_observable = False
        self.observable_off_reason = why
        self.warnings += 1
        logger.warning(
            "foundation.shadow: observable forecasts OFF for the rest of this "
            "run (%s). Still recording categorical predictions (if on), "
            "observations, actions and outcomes.", why)

    def _observable_error(self, site: str, e: BaseException) -> None:
        self.n["observable_errors"] += 1
        self.n[f"observable_errors:{site}"] += 1
        self._obs_fail_run[site] += 1
        logger.debug("foundation.shadow observable %s failed: %s: %s",
                     site, type(e).__name__, e)
        if self._obs_fail_run[site] >= self.obs_fail_limit:
            self._observable_off(
                f"{self._obs_fail_run[site]} consecutive {site} failures, "
                f"last {type(e).__name__}: {e}")

    def _observable_ok(self, site: str) -> None:
        self._obs_fail_run[site] = 0

    def _rows_of(self, observations, idx: List[int]) -> np.ndarray:
        """Host float32 copies of rows `idx` of the loop's observations.

        Preferred input: the numpy obs the loop already has (an ndarray or a
        list of per-stream arrays) — no device traffic at all. A torch tensor
        is accepted; a NON-CPU one forces a device->host copy (and a sync
        with queued kernels) and is counted as observable_device_copies."""
        if hasattr(observations, "detach"):          # torch.Tensor
            if observations.device.type != "cpu":
                self.n["observable_device_copies"] += 1
            sel = observations[list(idx)]
            return sel.detach().to("cpu").float().numpy().reshape(len(idx), -1)
        if isinstance(observations, (list, tuple)):
            return np.stack([np.asarray(observations[i], np.float32).reshape(-1)
                             for i in idx])
        x = np.asarray(observations, dtype=np.float32)
        return x[np.asarray(idx, dtype=np.int64)].reshape(len(idx), -1)

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
                    world_model=None, executor=None, observations=None) -> None:
        """Called after action selection, BEFORE env.step. On a sampled step
        logs every stream's observation t and the world model's prediction
        for the action about to be executed, and notes which actor chose
        each stream's action (`executor` = the loop's option executor or
        None; read, never called into). Never raises."""
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
                st.source = None
                st.observable = None
            if not self._sampled:
                return
            srcs = action_sources(executor, len(actions))
            for st in self.streams[:len(actions)]:
                st.source = srcs[st.idx]
                if srcs[st.idx].get("source") == "unknown":
                    self.n["source_unknown"] += 1
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
                self._predict(ready, rssm_state, actions, lock, world_model, observations)
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

    def _predict(self, ready, rssm_state, actions, lock, world_model, observations=None) -> None:
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
        # NEVER BLOCK THE ACTING THREAD ON THE TRAINER'S LOCK. Bounded wait,
        # then skip this one prediction (obs/action/outcome still recorded).
        tw = time.perf_counter()
        got = True
        if lock is not None:
            got = (lock.acquire(timeout=self.lock_timeout_s)
                   if self.lock_timeout_s > 0 else lock.acquire(blocking=False))
        tp = time.perf_counter()
        self.t["lock_wait"] += (tp - tw) * 1000.0
        if not got:
            self.n["skipped_lock_busy"] += 1
            for st in rows:
                source = (st.source or {}).get("source", "unknown")
                self.n[f"skipped_lock_busy:{st.name}:{source}"] += 1
            return
        want_obs = self.do_observable and observations is not None
        decoded = None
        obs_ms = 0.0
        try:
            with torch.no_grad():
                h0 = rssm_state["h"]
                idx = torch.as_tensor([st.idx for st in rows], device=h0.device,
                                      dtype=torch.long)
                h1, logits = _prior_step(r, h0.index_select(0, idx),
                                         rssm_state["z"].index_select(0, idx),
                                         a)
                probs = _probs_np(logits, len(rows), S, C)
                if want_obs:
                    # Only WEIGHT READS live inside the lock: the decoder
                    # forward on the h1/logits just computed (no second GRU
                    # transition), one host copy. Pooling happens below.
                    to = time.perf_counter()
                    try:
                        from .observable import decode_prior_mode
                        decoded = (decode_prior_mode(world_model, h1, logits)
                                   .float().cpu().numpy())
                        self._observable_ok("forecast")
                    except Exception as e:          # noqa: BLE001 — isolated
                        decoded = None
                        self._observable_error("forecast", e)
                    obs_ms = (time.perf_counter() - to) * 1000.0
                    self.t["observable"] += obs_ms
            version = self._model_version(world_model)
        finally:
            if lock is not None:
                lock.release()
        tw2 = time.perf_counter()
        # "predict" = the categorical part; the decoder forward is in
        # "observable" (so the totals in a degrade warning do not overlap)
        self.t["predict"] += (tw2 - tp) * 1000.0 - obs_ms
        if decoded is not None:
            self._forecast(rows, decoded, world_model, observations)
            tw2 = time.perf_counter()
        snap = f"{self.run_id}:world_model.rssm@{version}"
        t_pred = time.time()
        for i, st in enumerate(rows):
            ctx = next(iter(st.ctx.values()))
            pid = f"{st.name}:{st.episode}:{st.seq}:p"
            outcome = {"kind": "rssm_prior_categorical",
                       "probs": probs[i:i + 1].astype(np.float32)}
            if st.observable is not None:
                # OPTIONAL key: absent when the feature is off, so stores
                # without it read as "unrecorded", exactly like old ones.
                outcome["observable"] = st.observable
            pred = Prediction(
                pid, f"rssm:{st.name}:{st.episode}:{st.seq}",
                {"world_model.rssm": version}, snap, self.spec_id,
                (cmds[i],), 1,
                outcome,
                t_pred,
                {"sampling": "none: prior probabilities, no RNG draw",
                 "knowledge": "not conditioned",
                 "representation": (f"rssm_latent[h{r.deterministic_size},"
                                    f"z{S}x{C}]")})
            self.store.log_prediction(pred, ctx.ref())
            st.pid = pid
            self.n["predictions"] += 1
        self.t["write"] += (time.perf_counter() - tw2) * 1000.0

    def _forecast(self, rows, decoded, world_model, observations) -> None:
        """Pool forecast + persistence baseline on the HOST, outside the
        lock. Failures are isolated (see _observable_error)."""
        from .observable import PROBE_SIDE, pool_observation, probe_meta
        to = time.perf_counter()
        try:
            meta = probe_meta(world_model)
            predicted = pool_observation(decoded, **meta)
            persistence = pool_observation(
                self._rows_of(observations, [st.idx for st in rows]), **meta)
            if predicted.shape != persistence.shape:
                raise ValueError(f"forecast {predicted.shape} vs observation "
                                 f"{persistence.shape}")
            target = (f"pooled-pov-{PROBE_SIDE}x{PROBE_SIDE}"
                      if meta["pixel"] else "observation")
            for j, st in enumerate(rows):
                st.observable = {
                    "prediction": predicted[j].copy(),
                    "persistence": persistence[j].copy(),
                    "pixel": meta["pixel"], "channels": meta["channels"],
                    "side": meta["size"], "probe_side": PROBE_SIDE,
                    "horizon": 1, "units": "observation units",
                    "decoder": "categorical-mode", "target": target}
            self._observable_ok("persistence")
        except Exception as e:                      # noqa: BLE001 — isolated
            for st in rows:
                st.observable = None
            self._observable_error("persistence", e)
        finally:
            self.t["observable"] += (time.perf_counter() - to) * 1000.0

    def _score(self, forecast, boundary, observations, e) -> Dict[str, Any]:
        """Score a filed forecast against obs t+1, pooled in numpy on the
        host obs (no device traffic when the loop passes numpy)."""
        scored = {"status": ("invalid-boundary" if boundary
                             else "missing-outcome"),
                  "horizon": 1, "target": forecast["target"]}
        if boundary or observations is None:
            return scored
        if not self.do_observable:
            scored["status"] = "observable-off"
            return scored
        from .observable import pool_observation
        to = time.perf_counter()
        try:
            target = pool_observation(
                self._rows_of(observations, [e]), forecast["pixel"],
                forecast["channels"], forecast["side"],
                forecast.get("probe_side", 8))[0]
            pred, base = forecast["prediction"], forecast["persistence"]
            if target.shape != pred.shape:
                raise ValueError(f"outcome {target.shape} vs forecast "
                                 f"{pred.shape}")
            if np.isfinite(target).all() and np.isfinite(pred).all():
                scored.update(
                    status="scored", target_value=target,
                    mse=float(np.mean((pred - target) ** 2)),
                    persistence_mse=float(np.mean((base - target) ** 2)))
                self.n["observable_scored"] += 1
            else:
                scored["status"] = "non-finite"
            self._observable_ok("score")
        except Exception as ex:                     # noqa: BLE001 — isolated
            scored["status"] = "score-error"
            self._observable_error("score", ex)
        finally:
            self.t["observable"] += (time.perf_counter() - to) * 1000.0
        return scored

    def after_step(self, step_infos, actions, rewards, dones, restarted,
                   ends, t_dispatch, prim_extrinsic=None, intrinsic=None,
                   fleet_reset=False, dream=False, observations=None) -> None:
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
                        dream, observations)
        except Exception as e:
            self._fail(f"after_step raised {type(e).__name__}: {e}")
        finally:
            self._charge(t0)
            if self.enabled:
                self._end_step_budget()

    def _after(self, step_infos, actions, rewards, dones, restarted, ends,
               t_dispatch, prim_extrinsic, intrinsic, fleet_reset, dream, observations=None):
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
                apay = {"actor": "dream" if dream else "policy",
                        "action_repeat_ticks": self.action_repeat,
                        "t_complete_basis": "loop held the env result"}
                if dream:
                    apay["source"] = "dream_actor"
                elif st.source is not None:
                    apay.update(st.source)
                act = Action(
                    self.environment, st.name, st.episode, st.seq,
                    self.spec_id, self._command(actions[e]), UNKNOWN,
                    float(t_dispatch), t_res, apay)
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
                    if st.observable is not None:
                        pay["observable_score"] = self._score(
                            st.observable, boundary, observations, e)
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
                "predicting": self.do_predict,
                "observable_predicting": self.do_observable,
                "observable_off_reason": self.observable_off_reason,
                "stage": self.stage(), "degraded": list(self.degraded),
                "ms_totals": {k: round(v, 1) for k, v in self.t.items()},
                "ms_per_step_mean": float(ms.mean()) if ms.size else 0.0,
                "ms_per_step_p95": (float(np.percentile(ms, 95))
                                    if ms.size else 0.0),
                "store_bytes": self.store.disk_bytes(),
                "root": self.store.root, "run_id": self.run_id}

    def stage(self) -> str:
        """full (categorical + observable) / categorical / records /
        disabled. "categorical" also when observable_predict was never on."""
        if not self.enabled:
            return "disabled"
        if not self.do_predict:
            return "records"
        return "full" if self.do_observable else "categorical"

    def close(self) -> None:
        try:
            if self.enabled:
                self.store.close()
            logger.info("foundation.shadow summary: %s", self.stats())
        except Exception:
            pass


def action_sources(executor, n: int) -> List[Dict[str, Any]]:
    """Per stream, WHO chose the action just resolved: a payload fragment
    {"source": ...} (see the module docstring's action entry).

    Called after OptionExecutor.act() and before env.step, when the
    executor's state describes exactly this step: a stream whose root
    runtime is active is being driven by that option (act() opens options
    and resolves through the frozen skill stack; nothing closes them again
    until the post-step observe), the top nested frame being the skill that
    emitted the primitive. An inactive runtime with a decision record
    (`_primary_primitive` for stream 0, `_scout_primitive[e]` for scouts) is
    the shared policy; inactive WITHOUT one means an option was closed
    inside act() (unbound slot / rssm skill without a latent) and returned a
    fallback primitive.

    READ-ONLY: attribute reads and `bank.is_scripted` (a set membership
    test) — no executor method with side effects, no RNG. Never raises:
    anything unexpected yields "unknown" for that stream.
    """
    if executor is None:
        return [{"source": "policy"} for _ in range(n)]
    out: List[Dict[str, Any]] = []
    for e in range(n):
        try:
            rt = executor.runtimes[e]
            if rt.slot is not None:
                stack = list(executor.substacks[e]) if hasattr(
                    executor, "substacks") else []
                top = stack[-1] if stack else rt
                d = {"source": f"option_slot:{int(rt.slot)}",
                     "skill": str(top.skill_id),
                     "option_depth": 1 + len(stack)}
                if stack:
                    d["option_root_skill"] = str(rt.skill_id)
                bank = getattr(executor, "bank", None)
                if bank is not None and bank.is_scripted(rt.slot):
                    d["scripted"] = True
                out.append(d)
                continue
            # getattr: OptionExecutor creates _primary_primitive in act(),
            # not __init__ (act() always writes it for an idle stream 0)
            rec = (getattr(executor, "_primary_primitive", None) if e == 0
                   else executor._scout_primitive[e])
            out.append({"source": "policy" if rec is not None
                        else "option_fallback"})
        except Exception:                       # noqa: BLE001 — observer
            out.append({"source": "unknown"})
    return out


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
        _, logits = _prior_step(rssm, h, z, actions)
        return _probs_np(logits, h.shape[0], int(rssm.stochastic_size),
                         int(rssm.stochastic_classes))


def _prior_step(rssm, h, z, actions):
    """(h_{t+1}, prior logits) — the one transition both the categorical
    prediction and the observable forecast read (observable.prior_step)."""
    import torch
    from .observable import prior_step
    at = torch.as_tensor(np.asarray(actions, np.float32), device=h.device)
    return prior_step(rssm, {"h": h, "z": z}, at)


def _probs_np(logits, k: int, S: int, C: int) -> np.ndarray:
    import torch
    return torch.softmax(logits.view(k, S, C).float(), -1).cpu().numpy()


def _scope(environment: str, st: _Stream):
    from ..contracts import Scope
    return Scope(environment, st.name, st.episode)
