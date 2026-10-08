"""LearningTelemetry — what the agent learned and what paid it, per segment.

MEASUREMENT ONLY (2026-10-07). Nothing here feeds reward, actions, replay,
training or normalisation. The loop hands it values it has ALREADY computed;
it never draws a random number, never mutates an argument and never calls a
method with side effects on a learner (the pop_*_stats readers are the
producers' own consume-once counters, which exist for this file alone).
`tests/_learning_telemetry_smoke.py` proves the learner is byte-identical with
it on or off.

WHY THIS EXISTS. metrics.jsonl carries the ledger's SHARES (unsigned) and
stream 0's channel means. It cannot answer the three questions the
scoreboard (CLAUDE.md §9) keeps asking: which source actually paid the agent
(signed, per stream, including the terms the ledger never books —
icm_base, empowerment, approach, gui_dwell, and every multiplier that
shrank the drive), what it DID for that pay (actions, who chose them, menu
time, standing still, breaks and logs this segment), and whether the
learners were healthy while it happened.

RECONCILIATION — THE ONE CLAIM THAT MAKES THE REWARD SECTION TRUSTWORTHY.
The primary stream's reward is assembled by a chain of additive terms and
multiplicative damps on two running values: `intrinsic[0]` (I) and
`prim_extrinsic` (E). The loop calls `mark(label, I, E)` after each stage;
the CHANGE since the previous mark is booked to that label (a damp books a
negative change). The step is closed by `commit()` with the mixer's actual
output, and each part is converted to mixed units with the same weights and
the same return-ratio intrinsic scale the mixer applied (re-derived here from
the mixer's own state, not inferred from the output). So
`sum(reward[stream].values()) == reward_mix[stream].mixed_sum` is a genuine
check of the decomposition, and `reconcile_residual_sum` is reported rather
than folded in. A residual that is not ~0 means a stage was missed.

DEFENSIVE CONTRACT (as infra/metrics_sink.py): never raises, bounded memory,
records its own errors. Every public method swallows and counts.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA = "skybot.learning"
VERSION = 1
_MAX_KEYS = 256            # per-stream histogram / source cap (bounded memory)


def _f(x) -> Optional[float]:
    """Plain float or None (tensors via .item(); non-finite -> None)."""
    try:
        if hasattr(x, "item"):
            x = x.item()
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _scalar0(x) -> float:
    """intrinsic[0] as a float (tensor row, array or plain float)."""
    try:
        if hasattr(x, "__getitem__") and not isinstance(x, (float, int)):
            x = x[0]
        if hasattr(x, "item"):
            x = x.item()
        return float(x)
    except Exception:
        return float("nan")


def intrinsic_scale(mixer, intrinsic: float) -> float:
    """The return-ratio scale RewardMixer.mix applied to `intrinsic`.

    Re-derived from the mixer's state AFTER the call (the EMAs are advanced
    before scaling inside mix(), so the post-call state is the in-call state).
    Mirrors policy/actor_critic.py RewardMixer.mix exactly; if the mixer is a
    different shape this returns 1.0 and the residual will say so.
    """
    try:
        tr = float(getattr(mixer, "target_ratio", 0.0) or 0.0)
        ie = float(getattr(mixer, "_int_ema", 0.0) or 0.0)
        ee = float(getattr(mixer, "_ext_ema", 0.0) or 0.0)
        if tr > 0.0 and ee > 1e-8 and ie > 1e-8:
            ratio = ie / ee
            if ratio > tr:
                return max(float(getattr(mixer, "min_intrinsic_scale", 0.1)),
                           tr / ratio)
    except Exception:
        pass
    return 1.0


class _StreamSeg:
    """One stream's per-segment accumulators."""

    def __init__(self):
        self.reward: Dict[str, float] = {}
        self.mix = {"intrinsic_raw_sum": 0.0, "intrinsic_after_damp": 0.0,
                    "intrinsic_after_gui_zero": 0.0, "extrinsic_sum": 0.0,
                    "mixed_sum": 0.0, "reconcile_residual_sum": 0.0,
                    "int_scale_sum": 0.0, "n": 0}
        self.mult_sum: Dict[str, float] = {}
        self.mult_n: Dict[str, int] = {}
        self.actions: Dict[str, int] = {}
        self.sources: Dict[str, int] = {}
        self.steps = 0
        self.gui = 0
        self.gui_run = 0
        self.gui_longest = 0
        self.still = 0
        self.still_n = 0
        self.breaks: Dict[str, int] = {}
        self.ach: List[str] = []


def _bump(d: Dict[str, Any], k: str, v=1) -> None:
    if k in d or len(d) < _MAX_KEYS:
        d[k] = d.get(k, 0) + v


class LearningTelemetry:
    """Per-segment learning record -> runlogs/learning.jsonl. Never raises."""

    def __init__(self, cfg: Optional[dict] = None, run_id: Optional[str] = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.run_id = run_id or os.environ.get("SKYBOT_RUN_ID") or time.strftime(
            "%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
        self.errors = 0
        self.last_error: Optional[str] = None
        self.sink = None
        if self.enabled:
            try:
                from developmental_ai.infra.metrics_sink import MetricsSink
                self.sink = MetricsSink({
                    "enabled": True,
                    "path": str(cfg.get("path", "runlogs/learning.jsonl")),
                    "max_bytes": int(cfg.get("max_bytes", 64 * 1024 * 1024)),
                    "keep": int(cfg.get("keep", 5)),
                    "fsync": bool(cfg.get("fsync", True))})
            except Exception as exc:                   # pragma: no cover
                self._err("sink", exc)
        self._seg: Dict[int, _StreamSeg] = {}
        self._run_breaks: Dict[int, int] = {}
        self._run_logs: Dict[int, int] = {}
        self._seen_types: Dict[int, set] = {}
        self._episode: Dict[int, int] = {}
        # primary step book (pending until commit)
        self._p_i: Dict[str, float] = {}
        self._p_e: Dict[str, float] = {}
        self._p_mult: Dict[str, float] = {}
        self._prev_i = 0.0
        self._prev_e = 0.0
        self._raw_i = 0.0
        self._pending = False
        self._pre_src: List[str] = []
        self._pre_act: List[str] = []
        # WM block accumulator (written from the async trainer thread)
        self._wm_lock = threading.Lock()
        self._wm_sum: Dict[str, float] = {}
        self._wm_n = 0
        self._wm_samples = 0
        self._gn_ema: Optional[float] = None
        self._gn_seen = 0
        # health
        self.nan_steps = 0
        self.grad_spikes = 0
        # timing
        self._t_last = time.time()
        self._phase_prev: Dict[str, float] = {}
        self._ewn_prev = 0
        self._ews_prev = 0.0
        self._timing: Dict[str, Any] = {}
        self.cost_s = 0.0
        self.last_cost_ms = 0.0
        self._last_ppo_id = None

    # ------------------------------------------------------------- internals
    def _err(self, where: str, exc: BaseException) -> None:
        self.errors += 1
        self.last_error = f"{where}: {exc!r}"
        if self.errors == 1:
            logger.warning("learning telemetry %s failed: %r (further errors "
                           "counted in health.hook_errors)", where, exc)

    def _s(self, e: int) -> _StreamSeg:
        s = self._seg.get(e)
        if s is None:
            s = self._seg[e] = _StreamSeg()
        return s

    def episode_id(self, e: int) -> str:
        return f"{self.run_id}:{int(e)}:{self._episode.get(int(e), 0)}"

    # ------------------------------------------------- primary reward book
    def begin(self, intrinsic) -> None:
        """After the base curiosity scale: the step's first stage."""
        t0 = time.perf_counter()
        try:
            i0 = _scalar0(intrinsic)
            self._p_i = {"icm_base": i0}
            self._p_e = {}
            self._p_mult = {}
            self._prev_i, self._prev_e = i0, 0.0
            self._raw_i = i0
            self._pending = True
        except Exception as exc:
            self._err("begin", exc)
        self.cost_s += time.perf_counter() - t0

    def mark(self, label: str, intrinsic, extrinsic, ext_label: Optional[str]
             = None, mult_name: Optional[str] = None, mult=None) -> None:
        """Book the change of I and E since the last mark to `label`
        (E's change to `ext_label` when given). A multiplicative stage
        passes its factor as (mult_name, mult) for the mean-multiplier."""
        if not self._pending:
            return
        t0 = time.perf_counter()
        try:
            i = _scalar0(intrinsic)
            fe = _f(extrinsic)
            ex = self._prev_e if fe is None else fe
            di, de = i - self._prev_i, ex - self._prev_e
            if di != 0.0:
                self._p_i[label] = self._p_i.get(label, 0.0) + di
                if di > 0.0 and mult_name is None:
                    self._raw_i += di
            if de != 0.0:
                k = ext_label or label
                self._p_e[k] = self._p_e.get(k, 0.0) + de
            self._prev_i, self._prev_e = i, ex
            if mult_name is not None:
                m = _f(mult)
                if m is not None:
                    self._p_mult[mult_name] = m
        except Exception as exc:
            self._err("mark", exc)
        self.cost_s += time.perf_counter() - t0

    def commit(self, stream: int, mixed, intrinsic, extrinsic, mixer,
               after_damp: Optional[float] = None) -> None:
        """Close the primary step with the reward the body actually used."""
        if not self._pending:
            return
        t0 = time.perf_counter()
        try:
            self.mark("unattributed", intrinsic, extrinsic)
            i = self._prev_i
            if after_damp is None:      # I before the GUI zeroing
                after_damp = i - self._p_i.get("gui_zero", 0.0)
            self._book(stream, self._p_i, self._p_e, mixed, i, self._prev_e,
                       mixer, self._raw_i, after_damp, self._p_mult)
        except Exception as exc:
            self._err("commit", exc)
        self._pending = False
        self.cost_s += time.perf_counter() - t0

    def book_parts(self, stream: int, parts_i: Dict[str, float],
                   parts_e: Dict[str, float], mixed, mixer,
                   after_damp: Optional[float] = None,
                   mults: Optional[Dict[str, float]] = None) -> None:
        """One-shot form (scouts): the parts are known in one place."""
        t0 = time.perf_counter()
        try:
            i = sum(parts_i.values())
            raw = sum(v for k, v in parts_i.items() if v > 0.0
                      and not k.startswith("damp_") and k != "gui_zero")
            self._book(stream, parts_i, parts_e, mixed, i,
                       sum(parts_e.values()), mixer, raw,
                       i if after_damp is None else after_damp, mults or {})
        except Exception as exc:
            self._err("book_parts", exc)
        self.cost_s += time.perf_counter() - t0

    def _book(self, stream, parts_i, parts_e, mixed, i_final, e_final, mixer,
              raw_i, after_damp, mults) -> None:
        mixed = _f(mixed)
        s = self._s(int(stream))
        if mixed is None or not math.isfinite(i_final) or \
                not math.isfinite(e_final):
            self.nan_steps += 1
            return
        wi = float(getattr(mixer, "intrinsic_weight", 1.0))
        we = float(getattr(mixer, "extrinsic_weight", 0.0))
        k = intrinsic_scale(mixer, i_final)
        ci = wi * k
        for lab, v in parts_i.items():
            _bump(s.reward, lab, ci * v)
        for lab, v in parts_e.items():
            _bump(s.reward, lab, we * v)
        m = s.mix
        m["intrinsic_raw_sum"] += raw_i
        m["intrinsic_after_damp"] += after_damp
        m["intrinsic_after_gui_zero"] += i_final
        m["extrinsic_sum"] += e_final
        m["mixed_sum"] += mixed
        m["reconcile_residual_sum"] += mixed - (ci * i_final + we * e_final)
        m["int_scale_sum"] += k
        m["n"] += 1
        m["w_intrinsic"], m["w_extrinsic"] = wi, we
        for name, v in (mults or {}).items():
            s.mult_sum[name] = s.mult_sum.get(name, 0.0) + float(v)
            s.mult_n[name] = s.mult_n.get(name, 0) + 1

    # ---------------------------------------------------- behaviour/outcomes
    def pre_step(self, env_actions, sources: List[str]) -> None:
        """After act(), before env.step: who chose what, per stream."""
        t0 = time.perf_counter()
        try:
            acts = []
            for a in list(env_actions or []):
                try:
                    acts.append(str(int(a)))
                except Exception:
                    acts.append("continuous")
            self._pre_act = acts
            self._pre_src = list(sources)
        except Exception as exc:
            self._err("pre_step", exc)
        self.cost_s += time.perf_counter() - t0

    def post_step(self, step_infos, dones, n: int) -> None:
        """After env.step: what the step did, per stream."""
        t0 = time.perf_counter()
        try:
            for e in range(int(n)):
                si = (step_infos[e] if step_infos and e < len(step_infos)
                      else None) or {}
                s = self._s(e)
                s.steps += 1
                if e < len(self._pre_act):
                    _bump(s.actions, self._pre_act[e])
                if e < len(self._pre_src):
                    _bump(s.sources, self._pre_src[e])
                if bool(si.get("gui_open")):
                    s.gui += 1
                    s.gui_run += 1
                    s.gui_longest = max(s.gui_longest, s.gui_run)
                else:
                    s.gui_run = 0
                mv = (si.get("world") or {}).get("moved")
                if mv is not None:
                    s.still_n += 1
                    if float(mv) < 0.05:
                        s.still += 1
                seen = self._seen_types.setdefault(e, set())
                for ev in si.get("events") or []:
                    try:
                        kind, sub = ev[0], str(ev[1])
                    except Exception:
                        continue
                    if kind == "break":
                        _bump(s.breaks, sub)
                        self._run_breaks[e] = self._run_breaks.get(e, 0) + 1
                        if "log" in sub:
                            self._run_logs[e] = self._run_logs.get(e, 0) + 1
                    if kind in ("break", "craft", "place", "pickup"):
                        key = f"{kind}:{sub}"
                        if key not in seen and len(seen) < 4096:
                            seen.add(key)
                            if len(s.ach) < 64:
                                s.ach.append(key)
                try:
                    if dones is not None and bool(dones[e]):
                        self._episode[e] = self._episode.get(e, 0) + 1
                except Exception:
                    pass
        except Exception as exc:
            self._err("post_step", exc)
        self.cost_s += time.perf_counter() - t0

    # ------------------------------------------------------------ WM steps
    def wm_step(self, stats, batch=None) -> None:
        """Once per WM gradient step (may run on the async trainer thread)."""
        t0 = time.perf_counter()
        try:
            st = dict(stats or {})
            nb = 0
            try:
                sh = batch["observations"].shape
                nb = int(sh[0]) * int(sh[1])
            except Exception:
                pass
            with self._wm_lock:
                bad = False
                for k, v in st.items():
                    fv = _f(v)
                    if fv is None:
                        bad = True
                        continue
                    self._wm_sum[k] = self._wm_sum.get(k, 0.0) + fv
                self._wm_n += 1
                self._wm_samples += nb
                if bad:
                    self.nan_steps += 1
                gn = _f(st.get("grad_norm"))
                if gn is not None:
                    if (self._gn_ema is not None and self._gn_seen >= 20
                            and gn > 10.0 * self._gn_ema):
                        self.grad_spikes += 1
                    self._gn_ema = gn if self._gn_ema is None else \
                        0.98 * self._gn_ema + 0.02 * gn
                    self._gn_seen += 1
        except Exception as exc:
            self._err("wm_step", exc)
        self.cost_s += time.perf_counter() - t0

    # -------------------------------------------------------------- timing
    def snapshot_timing(self, phase_acc, env_wait_n, env_wait_sum) -> None:
        """Called BEFORE _log_progress resets the phase accumulators. Diffs
        against the previous snapshot when nothing reset them (verbose 0)."""
        t0 = time.perf_counter()
        try:
            acc = dict(phase_acc or {})
            ewn = int(env_wait_n or 0)
            ews = float(env_wait_sum or 0.0)
            if ewn >= self._ewn_prev:         # no reset since last snapshot
                dn = ewn - self._ewn_prev
                ds = ews - self._ews_prev
                d = {k: v - self._phase_prev.get(k, 0.0)
                     for k, v in acc.items()}
            else:
                dn, ds, d = ewn, ews, acc
            self._phase_prev, self._ewn_prev, self._ews_prev = acc, ewn, ews
            dn = max(1, dn)
            self._timing = {f"{k}_ms": round(1000.0 * v / dn, 3)
                            for k, v in d.items()}
            self._timing["env_wait_ms"] = round(1000.0 * ds / dn, 3)
            self._timing["timed_steps"] = int(dn)
        except Exception as exc:
            self._err("timing", exc)
        self.cost_s += time.perf_counter() - t0

    # -------------------------------------------------------------- record
    def _stream_sections(self) -> Dict[str, Dict[str, Any]]:
        out = {"reward": {}, "reward_mix": {}, "outcomes": {},
               "behaviour": {}}
        for e in sorted(self._seg):
            s = self._seg[e]
            key = f"stream-{e}"
            out["reward"][key] = dict(s.reward)
            m = dict(s.mix)
            n = max(1, m.pop("n"))
            m["int_scale_mean"] = m.pop("int_scale_sum") / n
            m["steps_booked"] = int(s.mix["n"])
            for name in ("farm_damp", "habituation", "boring_view",
                         "gui_zero"):
                # A damp not applied this step was a factor of 1.0.
                tot = s.mix["n"]
                if tot:
                    m[f"{name}_mean"] = (s.mult_sum.get(name, 0.0)
                                         + (tot - s.mult_n.get(name, 0))) / tot
                else:
                    m[f"{name}_mean"] = None
            out["reward_mix"][key] = m
            logs = sum(v for k, v in s.breaks.items() if "log" in k)
            out["outcomes"][key] = {
                "breaks_by_type": dict(s.breaks),
                "breaks_segment": int(sum(s.breaks.values())),
                "logs_segment": int(logs),
                "breaks_run": int(self._run_breaks.get(e, 0)),
                "logs_run": int(self._run_logs.get(e, 0)),
                "achievements_new": list(s.ach)}
            tot = sum(s.actions.values())
            ent = 0.0
            for v in s.actions.values():
                p = v / max(1, tot)
                if p > 0:
                    ent -= p * math.log2(p)
            out["behaviour"][key] = {
                "action_hist": dict(s.actions),
                "action_entropy_bits": ent if tot else None,
                "source_hist": dict(s.sources),
                "gui_frac": (s.gui / s.steps) if s.steps else None,
                "gui_longest_run": int(s.gui_longest),
                "still_frac": (s.still / s.still_n) if s.still_n else None,
                "steps": int(s.steps)}
        return out

    def record(self, total_timesteps: int, sections: Dict[str, Any],
               extra_dropped: int = 0) -> Optional[Dict[str, Any]]:
        """Build and write one record; reset the segment accumulators.
        `sections` supplies exploration/ppo/curiosity/replay/ledger pieces
        the loop gathered. Returns the record (for tests) or None."""
        t0 = time.perf_counter()
        rec = None
        try:
            now = time.time()
            ss = self._stream_sections()
            seg_steps = int(max([b.get("steps", 0) for b in
                                 ss["behaviour"].values()] or [0]))
            with self._wm_lock:
                wn = self._wm_n
                wm = {k: v / wn for k, v in self._wm_sum.items()} if wn else {}
                wm["steps"] = int(wn)
                wm["replay_ratio"] = (self._wm_samples / max(1, sum(
                    b.get("steps", 0) for b in ss["behaviour"].values()))
                    if wn else 0.0)
                self._wm_sum, self._wm_n, self._wm_samples = {}, 0, 0
            dt = max(1e-6, now - self._t_last)
            timing = dict(self._timing)
            timing["segment_wall_s"] = round(dt, 3)
            timing["steps_per_s"] = round(seg_steps / dt, 3)
            timing["telemetry_ms"] = round(1000.0 * self.cost_s, 3)
            self.last_cost_ms = 1000.0 * self.cost_s
            self._t_last = now
            ppo = sections.get("ppo")
            sink_dropped = int(extra_dropped) + int(
                getattr(self.sink, "dropped", 0) or 0)
            rec = {
                "schema": SCHEMA, "v": VERSION, "run_id": self.run_id,
                "total_timesteps": int(total_timesteps),
                "segment_steps": seg_steps,
                "streams": len(ss["behaviour"]),
                "reward": ss["reward"], "reward_mix": ss["reward_mix"],
                "outcomes": ss["outcomes"], "behaviour": ss["behaviour"],
                "exploration": sections.get("exploration"),
                "ppo": ppo, "wm": wm,
                "curiosity": sections.get("curiosity"),
                "replay": sections.get("replay"),
                "ledger_signed": sections.get("ledger_signed"),
                "timing": timing,
                "health": {"nan_steps": int(self.nan_steps),
                           "grad_spikes": int(self.grad_spikes),
                           "sink_dropped": sink_dropped,
                           "hook_errors": int(self.errors)},
            }
            if self.sink is not None:
                self.sink.write(rec)
            self.last_record = rec
        except Exception as exc:
            self._err("record", exc)
        self._seg = {}
        self.nan_steps = 0
        self.grad_spikes = 0
        self.cost_s = time.perf_counter() - t0   # record's own cost counts
        return rec
