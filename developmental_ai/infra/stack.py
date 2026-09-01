"""InfraStack — the one facade the organism talks to for general infrastructure.

WHY A FACADE. The 2026-08 infrastructure wave adds ~a dozen domain-agnostic
monitors (gates, heartbeats, reward ledger, farm detector, signal health,
behaviour drift, stuck escalation, episodic memory, affordance map,
empowerment, decision traces). Wiring each of them into a 7,800-line loop
individually would repeat the exact duplicated-call-site hazard this project
has been bitten by five times. Instead the loop makes a HANDFUL of calls
(`on_step`, `beat`, `signal_observe`, `segment`) and this facade fans out.

DESIGN RULES (all measured lessons, none aesthetic):
  * Monitoring must never take the run down: every leaf module is constructed
    defensively — a broken import or a raising monitor disables ITSELF with a
    logged warning and the organism keeps running.
  * Everything is DOMAIN-AGNOSTIC: the stack speaks percepts/actions/events
    ("kind:subtype" strings), never game nouns. The environment adapter owns
    the taxonomy (the event-centric contract, GENERAL_INFRASTRUCTURE.md #44).
  * Monitors read; only the STUCK ESCALATOR ever recommends action, and even
    then it only *returns* recommendations — the loop applies them, so the
    chain of authority stays visible in one place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import zlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


def _stable_cell_id(cell: Any) -> int:
    """Process-stable integer id for a state cluster key. Python's hash() is
    salted per process, which would silently reset the farm detector's memory
    on every restart-and-compare workflow — crc32 is stable and cheap."""
    return zlib.crc32(str(cell).encode("utf-8", "replace")) & 0x7FFFFFFF


class InfraStack:
    """Bundle of general monitors with graceful per-piece degradation."""

    def __init__(self, cfg: Optional[dict], action_dim: int,
                 log_dir: str = "podlogs"):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.log_dir = str(cfg.get("log_dir", log_dir))
        self._cfg = cfg
        # every leaf is Optional: None means "unavailable or disabled", and
        # every public method checks — so a partially-built stack is safe.
        self.heartbeat = None
        self.gates = None
        self.invariants = None
        self.ledger = None
        self.farm = None
        self.signals = None
        self.drift = None
        self.stuck = None
        self.trace = None
        self.episodic = None
        self.affordance = None
        self._empowerment_fn = None
        self._empowerment_pot = None
        self.empowerment_weight = float(cfg.get("empowerment_weight", 0.0))
        self.empowerment_interval = int(cfg.get("empowerment_interval", 25))
        self._emp_phi_prev: Optional[float] = None
        self._emp_last_step = -1
        self._emp_last_score = 0.0
        # fovea-category <-> event association (the LEARNED replacement for
        # the hand-written category->block-name map; see category_boringness)
        self._assoc: Dict[str, Dict[str, float]] = {}
        self._assoc_beta = float(cfg.get("assoc_beta", 0.05))
        # cached degenerate-signal set (recomputed per segment — computing
        # entropy over rolling windows per STEP would be pure waste)
        self._degenerate_cache: Set[str] = set()
        # last-known lifetime event counts (from the env's persisted memory)
        self._event_counts: Dict[str, int] = {}
        self._seg_events = 0
        self._sight_throttle: Dict[str, int] = {}
        self._help_count = 0
        self._last_stuck_level = 0
        self._help_gap = 0
        self._boost_active = False

        if not self.enabled:
            logger.info("infra: DISABLED by config")
            return

        def _build(name: str, fn: Callable[[], Any]):
            try:
                return fn()
            except Exception as e:      # a monitor must never kill the run
                logger.warning("infra: %s unavailable (%s) — running without",
                               name, e)
                return None

        if cfg.get("heartbeats", True):
            self.heartbeat = _build("heartbeat", self._mk_heartbeat)
        if cfg.get("gates", True):
            self.gates = _build("gates", self._mk_gates)
        if cfg.get("invariants", True):
            self.invariants = _build("invariants", self._mk_invariants)
        if cfg.get("ledger", True):
            self.ledger = _build("ledger", self._mk_ledger)
        if cfg.get("farm_detector", True):
            self.farm = _build("farm_detector", self._mk_farm)
        if cfg.get("signal_health", True):
            self.signals = _build("signal_health", self._mk_signals)
        if cfg.get("behavior_drift", True):
            self.drift = _build("behavior_drift",
                                lambda: self._mk_drift(action_dim))
        if cfg.get("stuck_monitor", True):
            self.stuck = _build("stuck_monitor", self._mk_stuck)
        if cfg.get("traces", True):
            self.trace = _build("traces", self._mk_trace)
        if cfg.get("episodic_memory", True):
            self.episodic = _build("episodic_memory", self._mk_episodic)
        if cfg.get("affordance", True):
            self.affordance = _build("affordance", self._mk_affordance)
        if self.empowerment_weight > 0.0:
            _build("empowerment", self._mk_empowerment)

        logger.info(
            "infra: stack up (%s)",
            ", ".join(n for n, v in [
                ("heartbeats", self.heartbeat), ("gates", self.gates),
                ("invariants", self.invariants), ("ledger", self.ledger),
                ("farm", self.farm), ("signals", self.signals),
                ("drift", self.drift), ("stuck", self.stuck),
                ("traces", self.trace), ("episodic", self.episodic),
                ("affordance", self.affordance),
                ("empowerment", self._empowerment_fn)] if v is not None))

    # ---- leaf constructors (isolated so one bad import kills one piece) ----

    def _mk_heartbeat(self):
        from developmental_ai.infra.lifecycle import Heartbeat
        return Heartbeat()

    def _mk_gates(self):
        from developmental_ai.infra.gate import GateRegistry
        return GateRegistry()

    def _mk_invariants(self):
        from developmental_ai.infra.lifecycle import InvariantSet
        inv = InvariantSet()
        # a few concrete, always-true-or-broken assertions. ctx keys are
        # supplied by segment(); a missing key reads as "cannot evaluate" and
        # the check passes (never punish a caller for partial context).
        inv.add("intrinsic_finite", lambda c: (
            None if c.get("intrinsic_per_step") is None
            or abs(float(c["intrinsic_per_step"])) < 1e6
            else f"intrinsic/step exploded: {c['intrinsic_per_step']}"))
        inv.add("grounding_alive", lambda c: (
            None if c.get("labels") is None or c.get("step", 0) < 20000
            or int(c["labels"]) > 0
            else "0 teacher labels after 20k steps — grounding is dead"))
        inv.add("segment_reward_sane", lambda c: (
            None if c.get("seg_extrinsic") is None
            or float(c["seg_extrinsic"]) > -1e4
            else f"segment extrinsic absurd: {c['seg_extrinsic']}"))
        return inv

    def _mk_ledger(self):
        from developmental_ai.infra.ledger import RewardLedger
        return RewardLedger(
            share_alarm=float(self._cfg.get("ledger_share_alarm", 0.8)),
            consecutive=int(self._cfg.get("ledger_consecutive", 3)))

    def _mk_farm(self):
        from developmental_ai.infra.ledger import FarmDetector
        return FarmDetector(
            revisit_horizon=int(self._cfg.get("farm_horizon", 400)),
            min_loops=int(self._cfg.get("farm_min_loops", 6)),
            income_alarm=float(self._cfg.get("farm_income_alarm", 0.02)))

    def _mk_signals(self):
        from developmental_ai.infra.signal_health import SignalMonitor
        return SignalMonitor()

    def _mk_drift(self, action_dim: int):
        from developmental_ai.infra.monitors import BehaviorDrift
        return BehaviorDrift(n_bins=int(action_dim))

    def _mk_stuck(self):
        from developmental_ai.infra.monitors import StuckMonitor
        floors = dict(self._cfg.get("stuck_floors", {
            # all three at/below floor for `patience` segments = stuck:
            # earning nothing, discovering nothing, causing nothing.
            "seg_extrinsic": 0.0,
            "cells_delta": 0.0,
            "events": 0.0,
        }))
        return StuckMonitor(floors,
                            patience=int(self._cfg.get("stuck_patience", 3)),
                            cooldown=int(self._cfg.get("stuck_cooldown", 6)))

    def _mk_trace(self):
        from developmental_ai.infra.monitors import DecisionTrace
        return DecisionTrace()

    def _mk_episodic(self):
        from developmental_ai.infra.episodic import EpisodicEventMemory
        return EpisodicEventMemory()

    def _mk_affordance(self):
        from developmental_ai.infra.affordance import AffordanceMap
        return AffordanceMap()

    def _mk_empowerment(self):
        from developmental_ai.infra.empowerment import (
            EmpowermentPotential, empowerment_score)
        self._empowerment_fn = empowerment_score
        self._empowerment_pot = EmpowermentPotential()
        return self._empowerment_fn

    # ---- hot-path hooks (each guarded, each contained) --------------------

    def beat(self, name: str, step: int) -> None:
        if self.heartbeat is not None:
            try:
                self.heartbeat.beat(name, step)
            except Exception:
                pass

    def register_heartbeat(self, name: str, expected_every: int,
                           step: int) -> None:
        if self.heartbeat is not None:
            try:
                self.heartbeat.register(name, expected_every, step)
            except Exception:
                pass

    def signal_observe(self, probs: Optional[Dict[str, float]]) -> None:
        if self.signals is None or not probs:
            return
        try:
            for k, v in probs.items():
                self.signals.observe(k, float(v))
        except Exception:
            pass

    def degenerate_signals(self) -> Set[str]:
        """Cached per segment — see _degenerate_cache note above."""
        return self._degenerate_cache

    def on_step(self, step: int, action: int,
                events: Sequence[Tuple[str, str]],
                cell: Any, position: Optional[Tuple[float, float, float]],
                pitch: Optional[float],
                fovea_probs: Optional[Dict[str, float]],
                fovea_counts: Optional[Dict[str, int]],
                step_reward: float,
                event_counts: Optional[Dict[str, int]] = None) -> None:
        """One call per primary waking step. Everything inside is O(1)-ish
        and contained; `events` is the adapter's typed effect stream."""
        if not self.enabled:
            return
        try:
            if event_counts:
                self._event_counts = event_counts
            if self.drift is not None:
                self.drift.update(int(action))
            if self.farm is not None and cell is not None:
                self.farm.step(_stable_cell_id(cell), int(action),
                               float(step_reward), int(step))
            if self.affordance is not None:
                # the affordance map is CAUSED-effect statistics only:
                # observed_change is by definition NOT this body's doing,
                # and death is something that happens TO it — crediting
                # either to the current action would poison the
                # action->effect evidence (e.g. "noop produces effects")
                self.affordance.observe(
                    int(action),
                    [e for e in events
                     if e[0] not in ("observed_change", "death")],
                    int(step))
            for kind, sub in events:
                _caused = kind not in ("observed_change", "death")
                if _caused:
                    # only CAUSED effects are a sign of life for the stuck
                    # monitor — passive scenery motion must never read as
                    # "the agent is doing things" (review 2026-08-09)
                    self._seg_events += 1
                if self.episodic is not None:
                    # non-caused changes are throttled landmarks, not a
                    # firehose: an unthrottled storm (rain, water) would
                    # evict the sighting/achievement landmarks the bearing
                    # sense navigates by
                    if _caused or (step - self._sight_throttle.get(
                            "__obs_change__", -10**9) >= 100):
                        if not _caused:
                            self._sight_throttle["__obs_change__"] = step
                        self.episodic.record(kind, sub, int(step), position)
                # LEARNED category<->event association: what did the fovea
                # contain when this event fired? This is the measured
                # replacement for the hand-written category->name map the
                # boring-view discount used to carry (a general agent cannot
                # ship a block taxonomy; it can remember what it was looking
                # at when things happened). CAUSED events only, and never
                # the social category: mastering log-breaks must not make
                # the TEACHER boring (review 2026-08-09) — habituating to
                # people because things happen near them is backwards.
                if fovea_probs and _caused:
                    key = f"{kind}:{sub}"
                    b = self._assoc_beta
                    for cat, p in fovea_probs.items():
                        if cat == "player_visible":
                            continue
                        if fovea_counts and fovea_counts.get(cat, 0) < 5:
                            continue
                        d = self._assoc.setdefault(cat, {})
                        d[key] = (1.0 - b) * d.get(key, 0.0) + b * float(p)
            # goal-object sighting -> episodic landmark (throttled: a
            # sighting STREAK is one sighting, not five hundred)
            if (self.episodic is not None and fovea_probs and position):
                for cat, p in fovea_probs.items():
                    if float(p) >= 0.5 and (
                            not fovea_counts or fovea_counts.get(cat, 0) >= 5):
                        if step - self._sight_throttle.get(cat, -10**9) >= 50:
                            self._sight_throttle[cat] = step
                            self.episodic.record("sighting", cat, int(step),
                                                 position,
                                                 salience=float(p))
            if self.trace is not None:
                self.trace.push({"t": int(step), "a": int(action),
                                 "ev": [f"{k}:{s}" for k, s in events],
                                 "cell": str(cell),
                                 "r": round(float(step_reward), 4),
                                 "pitch": (None if pitch is None
                                           else round(float(pitch), 1))})
        except Exception:
            pass

    def record_reward(self, source: str, amount: float) -> None:
        if self.ledger is not None:
            try:
                self.ledger.record(source, float(amount))
            except Exception:
                pass

    def category_boringness(self, habituation_scale: float
                            ) -> Dict[str, float]:
        """cat -> [0,1]: how strongly this perceptual category is associated
        with events the agent has fully habituated to. max over event types of
        assoc(cat,event) * familiarity(event). Cold start = {} = no discount,
        which is the safe direction (novelty is only ever taken away once the
        association is EARNED)."""
        if habituation_scale <= 0.0 or not self._assoc:
            return {}
        out: Dict[str, float] = {}
        for cat, evs in self._assoc.items():
            best = 0.0
            for key, a in evs.items():
                n = self._event_counts.get(key, 0)
                fam = 1.0 - 1.0 / (1.0 + n / float(habituation_scale))
                best = max(best, float(a) * fam)
            if best > 0.05:
                out[cat] = min(1.0, best)
        return out

    def empowerment_shaping(self, world_model, rssm_state, action_dim: int,
                            step: int, gamma: float, wm_lock=None) -> float:
        """Telescoping empowerment shaping (potential on normalized reachable-
        future diversity). Returns the shaping delta for THIS step (0.0 when
        off/not due). Trapped states (pits, towers, corners) score low with
        zero domain knowledge — this is the general mechanism that replaces
        hand-written geometry rules over time."""
        if (self._empowerment_fn is None or self.empowerment_weight <= 0.0
                or step - self._emp_last_step < self.empowerment_interval):
            return 0.0
        self._emp_last_step = step
        try:
            start = {"h": rssm_state["h"][0:1].detach(),
                     "z": rssm_state["z"][0:1].detach()}
            if wm_lock is not None:
                # NON-BLOCKING, deliberately: this runs in the ACTING loop,
                # and the async WM trainer can hold the lock for a whole
                # training block. Waiting here would stall acting — the exact
                # freeze-and-get-kicked failure the acting-while-learning
                # design exists to prevent. A skipped reading costs nothing
                # (the potential simply doesn't move this tick).
                if not wm_lock.acquire(blocking=False):
                    return 0.0
                try:
                    score = self._empowerment_fn(world_model, start,
                                                 action_dim)
                finally:
                    wm_lock.release()
            else:
                score = self._empowerment_fn(world_model, start, action_dim)
            self._emp_last_score = float(score)
            phi = self._empowerment_pot.update(float(score))
            if self._emp_phi_prev is None:
                self._emp_phi_prev = phi
                return 0.0
            f = self.empowerment_weight * (gamma * phi - self._emp_phi_prev)
            self._emp_phi_prev = phi
            self.record_reward("empowerment", f)
            return float(f)
        except Exception:
            return 0.0

    # ---- segment-cadence evaluation ---------------------------------------

    def gate_state(self, name: str, closed: bool, step: int, reason: str,
                   reopen: str, max_closed: int) -> None:
        if self.gates is not None:
            try:
                self.gates.set_state(name, closed, step, reason=reason,
                                     reopen=reopen,
                                     max_closed_steps=max_closed)
            except Exception:
                pass

    def segment(self, ctx: dict) -> Tuple[List[str], Dict[str, Any]]:
        """Evaluate everything that runs at segment cadence. Returns
        (printable lines, actions) where actions may contain
        {"stuck_level": int, "stuck_reason": str}. The LOOP applies actions —
        the stack only ever recommends."""
        lines: List[str] = []
        actions: Dict[str, Any] = {}
        if not self.enabled:
            return lines, actions
        step = int(ctx.get("step", 0))
        try:
            if self.signals is not None:
                # COLD-START PATIENCE, MEASURED IN POSITIVES (revised
                # 2026-08-11 from live evidence). A signal may only be called
                # DEGENERATE once the teacher has actually observed it TRUE
                # enough times — i.e. once it has had a genuine OPPORTUNITY
                # TO VARY. Anything else measures the world, not the head.
                #
                # The first cut of this gate counted total labels, which was
                # useless: one VLM query labels EVERY predicate at once, so
                # label_counts[p] just tracks elapsed queries and all 29
                # predicates crossed a 25-label threshold within minutes.
                # Live consequence: `tree_visible` was flagged DEGENERATE
                # while the agent stood in a treeless spot (fovea tree=0.007,
                # "sighting never"), and since this set is the magnet's
                # exclusion list, the magnet was forbidden from ever steering
                # toward trees — w=0.0000, target=None. A constant "no tree"
                # where there are no trees is CORRECT perception, and
                # punishing it disabled the one drive that finds trees.
                #
                # ctx["signal_evidence"] = POSITIVE observation counts.
                # Absent evidence keeps the old strict behaviour.
                _ev_min = int(self._cfg.get(
                    "degenerate_min_positives",
                    self._cfg.get("degenerate_min_labels", 10)))
                _evid = dict(ctx.get("signal_evidence") or {})
                _flagged = {
                    s for s in self.signals.degenerate()
                    if not _evid or _evid.get(s, 0) >= _ev_min}
                # STARVED vs BROKEN (2026-08-16, the 5th guard-becomes-
                # latch). The evidence gate above fixed the COLD-START case;
                # live it latched anyway: a head with plenty of historical
                # positives (35 logs felled) went constant-LOW because the
                # agent stared at the ground, was flagged, and — since this
                # set is the magnet's exclusion list — the one drive that
                # would have made it look up was forbidden from steering.
                # Constant-low WITH proven positives is honest absence, and
                # absence is precisely what seeking exists to cure: leave it
                # steerable. Constant-HIGH (asserts presence regardless of
                # scene) or positive-free stays excluded — steering on a
                # stuck-high head is the original tree_visible waste. A
                # stuck-low BROKEN head slips through this split, but fails
                # INERT (its seek potential never pays), not as a farm.
                _means = self.signals.means()
                _starved = ({s for s in _flagged
                             if _evid.get(s, 0) >= _ev_min
                             and _means.get(s, 1.0) < 0.5}
                            if _evid else set())
                self._degenerate_cache = _flagged - _starved
                # Mass starvation is a GAZE verdict, not a sensor verdict:
                # when most of what the agent can name reads constant-absent
                # at once, the world did not empty — the view did. Surface it
                # as an action so the loop can right the gaze (and the
                # advisor can say so) instead of demoting perception.
                _frac = float(self._cfg.get("gaze_starved_frac", 0.6))
                if (_means and _starved
                        and len(_starved) / max(1, len(_means)) >= _frac):
                    actions["gaze_starved"] = sorted(_starved)
                    lines.append(
                        f"  infra/signals: GAZE-STARVED — "
                        f"{len(_starved)}/{len(_means)} predicates constant-"
                        f"absent with proven positives; treating as a gaze "
                        f"problem (steering exclusion NOT applied)")
                rep = self.signals.report()
                if rep:
                    lines.append("  infra/signals: " + rep
                                 + (f" | DEGENERATE={sorted(self._degenerate_cache)}"
                                    if self._degenerate_cache else "")
                                 + (f" | STARVED={sorted(_starved)}"
                                    if _starved else ""))
            if self.ledger is not None and ctx.get("ledger_sources"):
                for src, amt in dict(ctx["ledger_sources"]).items():
                    self.ledger.record(str(src), float(amt))
                seg = self.ledger.segment()
                shares = ", ".join(
                    f"{k}={v:.0%}" for k, v in sorted(
                        seg.get("shares", {}).items(),
                        key=lambda kv: -kv[1])[:5])
                lines.append(f"  infra/ledger: total={seg.get('total', 0.0):+.2f}"
                             f" hhi={seg.get('hhi', 0.0):.2f} | {shares}")
                for a in seg.get("alarms", []):
                    lines.append("  infra/ledger ALARM: " + a)
            if self.farm is not None:
                for s in self.farm.segment_report():
                    lines.append("  infra/farm: " + s)
            if self.drift is not None:
                d = self.drift.segment_report()
                if d:
                    lines.append("  infra/drift: " + d)
                    if self.trace is not None:
                        self.trace.dump(os.path.join(
                            self.log_dir, "decision_traces.jsonl"),
                            "behavior drift: " + d)
            if self.heartbeat is not None:
                table, overdue = self.heartbeat.report(step)
                if overdue:
                    lines.append("  infra/heartbeat OVERDUE: "
                                 + ", ".join(overdue))
            if self.gates is not None:
                for a in self.gates.alarms(step):
                    lines.append("  infra/gate ALARM: " + a)
            if self.invariants is not None:
                for v in self.invariants.evaluate(ctx):
                    lines.append("  infra/" + v)
            if self.affordance is not None:
                need = list(ctx.get("required_effects") or [])
                for u in self.affordance.unproduced(need, min_steps=30000,
                                                    now=step):
                    lines.append("  infra/affordance: " + u)
                dead = self.affordance.actions_without_effect(min_uses=3000)
                if dead:
                    lines.append(f"  infra/affordance: no observed effect "
                                 f"from actions {dead}")
            if self.episodic is not None:
                s = self.episodic.summary(step, kinds=["sighting", "break"])
                if s:
                    lines.append("  infra/episodic: " + s)
            if self._empowerment_fn is not None:
                lines.append(f"  infra/empowerment: last={self._emp_last_score:.4f}"
                             f" phi_prev={self._emp_phi_prev if self._emp_phi_prev is None else round(self._emp_phi_prev, 3)}")
            if self.stuck is not None and bool(ctx.get("stuck_eligible",
                                                       True)):
                metrics = {k: float(ctx.get(k, 0.0)) for k in
                           ("seg_extrinsic", "cells_delta", "events")}
                level, reason = self.stuck.update(metrics)
                if level > 0:
                    lines.append(f"  infra/stuck: LEVEL {level} — {reason}")
                    actions["stuck_level"] = level
                    actions["stuck_reason"] = reason
                    # help fires on the TRANSITION into L3 and then at most
                    # once per help_min_gap segments while it persists —
                    # asking every segment is spam, not communication
                    # (measured: L3 re-emitted per segment on the probe).
                    if level >= 3:
                        self._help_gap = getattr(self, "_help_gap", 0) + 1
                        if (self._last_stuck_level < 3
                                or self._help_gap >= int(self._cfg.get(
                                    "help_min_gap", 20))):
                            self._help_gap = 0
                            self._emit_help(ctx, reason)
                self._last_stuck_level = level
        except Exception as e:
            lines.append(f"  infra: segment evaluation error contained: {e!r}")
        self._seg_events = 0
        return lines, actions

    def pop_segment_events(self) -> int:
        """Events observed this segment (read in ctx assembly BEFORE segment()
        resets it)."""
        return self._seg_events

    # ---- help requests (#51): the agent telling a human it is in trouble --

    def _emit_help(self, ctx: dict, reason: str) -> None:
        """Structured request to the human/oracle channel. Days of this
        project's debugging were spent inferring states the agent could simply
        have reported; this is the reporting. It is a REQUEST, not a
        dependency — the run continues regardless of any answer."""
        try:
            payload = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step": ctx.get("step"),
                "situation": reason,
                "position": ctx.get("position"),
                "segment_extrinsic": ctx.get("seg_extrinsic"),
                "cells_delta": ctx.get("cells_delta"),
                "events": ctx.get("events"),
                "episodic": (self.episodic.summary(int(ctx.get("step", 0)))
                             if self.episodic is not None else None),
                "hypothesis": ("no income, no new territory and no caused "
                               "events for several segments — likely stuck, "
                               "trapped, or every reachable activity is "
                               "habituated; consider relocating me or "
                               "checking my affordances"),
            }
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, str(self._cfg.get(
                "help_file", "help_requests.jsonl")))
            with open(path, "a") as f:
                f.write(json.dumps(payload, default=str) + "\n")
            self._help_count += 1
            logger.warning("infra: HELP REQUEST #%d written to %s — %s",
                           self._help_count, path, reason)
            if self.trace is not None:
                self.trace.dump(os.path.join(self.log_dir,
                                             "decision_traces.jsonl"),
                                "help request: " + reason)
        except Exception:
            pass

    @staticmethod
    def provenance_line(config_path: Optional[str], config_obj: dict) -> str:
        """One-line run provenance (#57 light): config digest + timestamp so a
        log can always be tied back to the exact configuration that produced
        it, even with no version control on the host."""
        try:
            blob = json.dumps(config_obj, sort_keys=True, default=str)
            digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
        except Exception:
            digest = "unhashable"
        return (f"provenance: config={config_path or '<inline>'} "
                f"sha={digest} started={time.strftime('%Y-%m-%dT%H:%M:%S')}")
