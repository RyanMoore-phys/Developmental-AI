"""Run-health monitors — stuck detection, behavioural drift, decision trace.

Why this module exists (each class answers a measured failure, not a
hypothetical):

  * StuckMonitor — the agent once stood on a single spot for SIX HOURS and
    nothing in the stack noticed: every subsystem kept ticking, every metric
    kept logging, and only a human reading the logs the next morning saw that
    all of them had been pinned to their floor the whole time.  The general
    answer is not "detect standing still" (domain-specific, and the next
    stall will be a menu dwell or a spin loop instead) but "detect that EVERY
    health metric the caller nominated has been at its floor for a sustained
    period", with STAGED escalation so the response can start cheap (log it)
    and get progressively more drastic (perturb, reset) only if the cheap
    response did not work.

  * BehaviorDrift — every reward-hack we have caught so far was first visible
    as a behavioural regime change: one action's share of the policy jumping
    from ~3% to ~22% over a few thousand steps, which a human happened to
    spot in a histogram days later.  Jensen-Shannon divergence between a
    recent action histogram and a slow-EMA baseline automates exactly that
    observation, and names the largest-moving bin so the alarm is actionable
    ("bin 7 share 3.1%->22.4%") rather than a bare scalar.

  * DecisionTrace — "why did it do that?" currently costs hours of log
    forensics because the interesting context (what the agent believed and
    chose in the seconds BEFORE the anomaly) is interleaved with megabytes of
    routine output.  A small ring buffer of structured decision records,
    dumped to jsonl only when something (a stuck escalation, a drift alarm)
    asks for it, answers the question in seconds and costs nothing when
    nothing goes wrong.

Design constraints honoured throughout:

  * DOMAIN-AGNOSTIC — metrics are named by the caller, behaviour bins are
    opaque indices, decision records are opaque dicts.  Nothing in here knows
    what the agent is or does.
  * NEVER-CRASH — these classes run inside multi-day loops; a monitor that
    can throw is worse than no monitor, because it kills the run it was
    supposed to protect.  Every public method catches broad exceptions and
    degrades to inert behaviour (current level, ``None`` report, ``None``
    dump path); errors surface as data, never as control flow.
  * NO NEW DEPENDENCIES — stdlib + numpy only.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np


class StuckMonitor:
    """Staged stuck-detection over caller-nominated health metrics.

    The caller declares, once, which metrics constitute "signs of life" and
    what floor each must stay above (``floors``: metric-name -> floor).  Each
    segment it feeds the current values to :meth:`update` and gets back an
    escalation level:

        0  fine — at least one floored metric is above its floor
        1  first escalation (intended response: cheap, e.g. log + dump trace)
        2  second escalation (e.g. inject exploration / perturb)
        3  final escalation (e.g. hard reset) — the ceiling; never exceeded

    Escalation only begins after ``patience`` CONSECUTIVE segments in which
    EVERY floored metric sat at/below its floor — a single bad segment is
    noise, not a stall.  After an escalation, ``cooldown`` segments must pass
    (in addition to another ``patience`` window of continued stuckness)
    before the next step, because the level-N intervention needs time to show
    an effect before we conclude it failed and go harsher.  Crucially the
    cooldown gates RE-ESCALATION ONLY: the moment any single metric recovers
    above its floor the level snaps back to 0 with no delay, because a
    recovered run must never be perturbed by a stale escalation.  (This
    asymmetry is deliberate — the codebase has hit the "guard becomes latch"
    anti-pattern repeatedly, where a protective state sticks past the
    condition that justified it.)

    Missing or non-numeric metric values are IGNORED for that segment (they
    neither prove stuckness nor prove recovery): a caller that briefly stops
    reporting one metric must not be able to trigger — or mask — an
    escalation by omission.  A segment in which nothing judgeable was
    reported leaves the state exactly as it was.
    """

    def __init__(self, floors: Dict[str, float], patience: int = 3,
                 cooldown: int = 6):
        # Copy defensively: the caller may mutate its dict later, and a
        # monitor whose contract silently changes mid-run is undebuggable.
        self.floors: Dict[str, float] = {str(k): float(v)
                                         for k, v in dict(floors).items()}
        self.patience = max(1, int(patience))
        self.cooldown = max(0, int(cooldown))
        self._level = 0            # current escalation level (0..3)
        self._streak = 0           # consecutive all-at/below-floor segments
        self._since_esc = 0        # segments since the last escalation step
        self._last_reason = ""

    def update(self, metrics: Dict[str, float]) -> Tuple[int, str]:
        """One segment. Returns (level, reason); reason is "" at level 0."""
        try:
            return self._update(metrics)
        except Exception as exc:                       # pragma: no cover
            # Containment: a malformed metrics payload must not kill the run.
            # Hold the current level and expose the failure as data.
            return (self._level, f"monitor_error: {exc!r}")

    # -- internals --------------------------------------------------------

    def _update(self, metrics: Dict[str, float]) -> Tuple[int, str]:
        stuck: Dict[str, float] = {}
        judged = 0
        for name, floor in self.floors.items():
            if metrics is None or name not in metrics:
                continue
            try:
                value = float(metrics[name])
            except (TypeError, ValueError):
                continue
            judged += 1
            if value <= floor:
                stuck[name] = value
            else:
                # INSTANT reset on any single recovery — see class docstring.
                self._level = 0
                self._streak = 0
                self._since_esc = 0
                self._last_reason = ""
                return (0, "")
        if judged == 0:
            # No evidence either way this segment: hold state unchanged.
            return (self._level, self._last_reason if self._level else "")

        # Every judged metric is at/below its floor.
        self._streak += 1
        if self._level == 0:
            if self._streak >= self.patience:
                self._level = 1
                self._since_esc = 0
        elif self._level < 3:
            self._since_esc += 1
            # Re-escalation needs BOTH another patience window of continued
            # stuckness AND the cooldown to have elapsed; the counters run
            # together while stuck, so the gate is simply the max.
            if self._since_esc >= max(self.patience, self.cooldown):
                self._level += 1
                self._since_esc = 0

        reason = ""
        if self._level:
            parts = [f"{k}={v:.4g}<=floor({self.floors[k]:.4g})"
                     for k, v in stuck.items()]
            reason = "stuck " + ", ".join(parts)
        self._last_reason = reason
        return (self._level, reason)


class BehaviorDrift:
    """Regime-change alarm over a stream of categorical behaviour choices.

    Feed one ``bin_index`` per step (e.g. the action taken); once per segment
    ask :meth:`segment_report`.  The report compares the RECENT histogram
    against a LONG-EMA baseline using Jensen-Shannon divergence (base-2, so
    the value is in bits and bounded by 1.0 — a stable, interpretable alarm
    threshold, unlike KL which is unbounded and asymmetric).  When JS crosses
    ``alarm_js`` the report names the single largest-moving bin with its
    before/after share, because "bin 7 share 3.1%->22.4%" is what a human
    needs to start investigating and a bare divergence number is not.

    Two-timescale design: the baseline moves at ``baseline_beta`` PER
    OBSERVATION (folded in per-segment via the closed form
    ``1-(1-beta)^n``), so genuine long-term policy improvement is slowly
    absorbed and stops alarming, while a fast regime change — the signature
    of a reward hack — stands out against it.

    Rate-of-data guards (both learned from real monitor failure modes):

      * A report is only computed once the recent window holds at least
        ``recent_window/4`` observations — JS on a near-empty histogram is
        pure sampling noise and would fire false alarms after every reset.
      * When there is NOT enough data the recent window is deliberately NOT
        folded/reset, so a caller that polls faster than data arrives keeps
        accumulating instead of being starved into permanent silence.

    On a sufficient-data call the recent window IS folded into the baseline
    and restarted, whether or not it alarmed — the segment has been judged
    and its evidence belongs to history.
    """

    def __init__(self, n_bins: int, recent_window: int = 2048,
                 baseline_beta: float = 0.0005, alarm_js: float = 0.12):
        self.n_bins = max(1, int(n_bins))
        self.recent_window = max(4, int(recent_window))
        self.baseline_beta = float(baseline_beta)
        self.alarm_js = float(alarm_js)
        self._recent = np.zeros(self.n_bins, dtype=np.float64)
        self._recent_n = 0
        self._baseline: Optional[np.ndarray] = None  # prob dist, set on 1st fold

    def update(self, bin_index: int) -> None:
        """Record one observation. Out-of-range/non-int indices are dropped
        silently — a bad index must not crash the run, and inventing a bin
        for it would corrupt the histogram it is supposed to protect."""
        try:
            i = int(bin_index)
        except (TypeError, ValueError):
            return
        if 0 <= i < self.n_bins:
            self._recent[i] += 1.0
            self._recent_n += 1

    def segment_report(self) -> Optional[str]:
        """None when quiet (below alarm / not enough data); else one line
        naming the largest-moving bin. See class docstring for fold rules."""
        try:
            return self._segment_report()
        except Exception:                              # pragma: no cover
            return None

    # -- internals --------------------------------------------------------

    def _segment_report(self) -> Optional[str]:
        if self._recent_n < self.recent_window / 4:
            return None                      # too little data; keep counting
        recent = self._recent / float(self._recent_n)
        if self._baseline is None:
            # Cold start: the first full window IS the baseline. No JS is
            # computable against nothing, so no alarm on the first segment.
            self._baseline = recent.copy()
            self._reset_recent()
            return None
        before = self._baseline.copy()       # compare against PRE-fold state
        js = self._js_bits(recent, before)
        self._fold(recent)
        self._reset_recent()
        if js < self.alarm_js:
            return None
        i = int(np.argmax(np.abs(recent - before)))
        return (f"bin {i} share {before[i] * 100.0:.1f}%->"
                f"{recent[i] * 100.0:.1f}% (JS={js:.3f})")

    def _fold(self, recent: np.ndarray) -> None:
        # Per-observation EMA applied in one batch step: n observations at
        # rate beta are exactly one step at rate 1-(1-beta)^n. This keeps the
        # baseline's timescale defined in OBSERVATIONS, independent of how
        # often the caller happens to poll segment_report().
        n = self._recent_n
        eff = 1.0 - (1.0 - self.baseline_beta) ** n
        assert self._baseline is not None
        self._baseline = (1.0 - eff) * self._baseline + eff * recent
        total = float(self._baseline.sum())
        if total > 0.0:                      # guard float drift; stay a dist
            self._baseline /= total

    def _reset_recent(self) -> None:
        self._recent = np.zeros(self.n_bins, dtype=np.float64)
        self._recent_n = 0

    @staticmethod
    def _js_bits(p: np.ndarray, q: np.ndarray) -> float:
        # Jensen-Shannon divergence, log base 2 (bits). Zero-mass bins
        # contribute zero by the KL convention 0*log(0/x) = 0; np.where
        # keeps that exact without epsilon-fudging the distributions.
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pm = np.where(p > 0.0, p * np.log2(p / m), 0.0)
            kl_qm = np.where(q > 0.0, q * np.log2(q / m), 0.0)
        return float(0.5 * kl_pm.sum() + 0.5 * kl_qm.sum())


class DecisionTrace:
    """Rolling buffer of decision records, dumped to jsonl on demand.

    push() is called on the hot path every decision, so it is a bare deque
    append — no serialisation, no I/O, no locks.  All cost is deferred to
    dump(), which only runs when a monitor (or a human) asks "what just
    happened?".  The buffer is a ring (``maxlen``): the interesting context
    is always the RECENT past, and an unbounded trace is a slow memory leak
    on a multi-day run.

    dump() safety properties, each preventing a specific way tracing could
    hurt the run it observes:

      * RATE-LIMITED — a monitor stuck in an alarm loop (e.g. StuckMonitor at
        level 3 every segment) must not write gigabytes of near-identical
        dumps; fewer than ``min_pushes_between_dumps`` new pushes since the
        last successful dump returns ``None`` without touching the disk.
        The first-ever dump is always allowed.
      * I/O-CONTAINED — a full disk or bad path returns ``None``; it never
        raises into the caller's loop.
      * SERIALISATION-PROOF — records are opaque caller dicts and WILL
        eventually contain a numpy scalar, a tensor, or a live object.
        ``json.dumps(..., default=str)`` stringifies anything json cannot
        encode, and a record that defeats even that is replaced by an
        ``{"__unserialisable__": ...}`` stub so one poisoned record cannot
        void the whole dump.

    Dumps APPEND, each chunk prefixed by a ``{"__dump__": reason, "n": N}``
    header line, so successive alarms build one forensic timeline in a single
    file and each chunk states why it exists.  The buffer is NOT cleared by a
    dump: it is a view of the recent past, and the next alarm deserves its
    own complete window.
    """

    def __init__(self, maxlen: int = 2048,
                 min_pushes_between_dumps: int = 1000):
        self._buf: deque = deque(maxlen=max(1, int(maxlen)))
        self._min_pushes = max(0, int(min_pushes_between_dumps))
        # None = never dumped: the first dump must always be allowed, since
        # requiring N pushes before the FIRST forensic snapshot would blind
        # us exactly when an early-run failure needs it most.
        self._pushes_since_dump: Optional[int] = None

    def push(self, record: dict) -> None:
        try:
            self._buf.append(record)
            if self._pushes_since_dump is not None:
                self._pushes_since_dump += 1
        except Exception:                              # pragma: no cover
            pass

    def dump(self, path: str, reason: str) -> Optional[str]:
        try:
            if (self._pushes_since_dump is not None
                    and self._pushes_since_dump < self._min_pushes):
                return None                             # rate-limited
            records = list(self._buf)
            lines = [json.dumps({"__dump__": str(reason), "n": len(records)},
                                default=str)]
            for rec in records:
                try:
                    lines.append(json.dumps(rec, default=str))
                except Exception as exc:
                    lines.append(json.dumps(
                        {"__unserialisable__": repr(exc)}))
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            # Reset only AFTER a successful write: a failed dump must not
            # consume the rate-limit budget of the retry that follows it.
            self._pushes_since_dump = 0
            return path
        except Exception:
            return None
