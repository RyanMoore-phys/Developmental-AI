"""Reward income statement + behavioural-loop farm detector (domain-agnostic).

WHY THIS MODULE EXISTS. Reward hacking has been this project's dominant
failure mode: five distinct reward farms were found in one 48-hour window,
and every single one of them announced itself the same two ways long before
anyone diagnosed it by reading policy weights:

  1. ON THE INCOME STATEMENT — one reward source quietly grew to dominate
     total income (the pickup-reward farm was >95% of income for hours while
     the agent learned nothing), and
  2. IN THE TRAJECTORY — a short behavioural cycle that the agent repeated
     with POSITIVE NET INCOME per lap.

Both signatures are cheap to compute online and neither requires knowing
anything about the environment, so this module watches for exactly those two
things and nothing else:

  * ``RewardLedger`` is the income statement: per-source accumulation within
    a segment, share-of-income + concentration (HHI) at segment close, and a
    persistence alarm when one source dominates for several segments in a
    row (one dominant segment is often legitimate — a burst of achievement
    reward looks dominant for a moment; a farm dominates FOREVER).

  * ``FarmDetector`` is the cycle auditor. It keys on the theorem that makes
    the audit sound: potential-based shaping telescopes to ~0 over any closed
    loop, so sustained positive income on a revisited state is definitionally
    either real achievement or a farm — shaping cannot produce it. (The
    zero-reward-stall fix moved all shaping to potential-only form precisely
    so this invariant holds; this detector is the monitor that collects on
    that investment.)

DEFENSIVE CONTRACT. This is monitoring code running inside multi-day
unattended loops. It must never be the thing that kills the run: every public
entry point catches and contains its own failures and surfaces them as DATA
(alarm/report strings), never as exceptions. Losing a ledger datapoint is
recoverable; crashing the developmental loop at hour 40 is not.

Domain-agnostic on purpose: "source" is any string the caller uses to name a
reward stream, "cell" is any int the caller uses to discretise state (a
position hash, an abstraction-layer cluster id, ...), "action" is any int
index. Nothing here knows what the agent is or what world it lives in.

Pure stdlib; no numpy needed at these data volumes.
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from typing import Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# RewardLedger
# ---------------------------------------------------------------------------

# A ledger keyed by caller-supplied strings can be grown without bound by a
# buggy caller that bakes an id into the source name ("goal_1832_reward").
# That exact class of bug (per-instance keys where a per-kind key was meant)
# produced the junk-goal factory's 48 ungrounded slots, so we refuse to let
# it become a memory leak here: past this many distinct sources, new ones
# fold into a single overflow bucket. The fold is itself a signal — if
# "(other)" ever dominates the statement, the caller's naming is broken.
_MAX_SOURCES = 512

# Zero-total detection threshold. Exact ==0.0 would be fine for the common
# "nothing recorded" case, but shaping terms can leave true-zero segments
# with 1e-17 float dust; treat those as zero too rather than emitting a
# shares dict of meaningless dust fractions.
_EPS = 1e-12


class RewardLedger:
    """Per-source income statement with a dominance-persistence alarm.

    Shares are computed over the sum of ABSOLUTE amounts per source. This is
    deliberate: with signed sums, a farm paying +10/segment sitting next to a
    shaping term paying -9/segment nets to +1 and the farm's signed share
    explodes past 1.0 (or, mirrored, shrinks to invisibility). Every farm we
    actually caught was partially masked by negative shaping in exactly this
    way, so the statement books gross activity, not net — a source moving a
    lot of reward in either direction is a source worth looking at.
    """

    def __init__(self, share_alarm: float = 0.8, consecutive: int = 3):
        self.share_alarm = float(share_alarm)
        self.consecutive = max(1, int(consecutive))
        # Per-segment accumulators, cleared by segment().
        self._signed: Dict[str, float] = {}
        self._abs: Dict[str, float] = {}
        # Dominance streaks persist ACROSS segment() calls — that persistence
        # is the whole alarm design (one hot segment is noise, N in a row is
        # a farm signature).
        self._streaks: Dict[str, int] = {}
        # Contained-error memory, surfaced as an alarm line at segment close.
        self._err_count: int = 0
        self._last_err: str = ""

    # -- recording ---------------------------------------------------------

    def record(self, source: str, amount: float) -> None:
        """Accumulate one reward event into the current segment.

        Never raises. Non-finite amounts are dropped and counted as errors —
        a single NaN admitted here would propagate through every future
        total/share and silently blind the alarm for the rest of the run.
        """
        try:
            src = str(source)
            amt = float(amount)
            if not math.isfinite(amt):
                self._note_error(f"non-finite amount from {src!r}")
                return
            if src not in self._signed and len(self._signed) >= _MAX_SOURCES:
                src = "(other)"
            self._signed[src] = self._signed.get(src, 0.0) + amt
            self._abs[src] = self._abs.get(src, 0.0) + abs(amt)
        except Exception as exc:  # noqa: BLE001 — containment is the contract
            self._note_error(repr(exc))

    # -- segment close -----------------------------------------------------

    def segment(self) -> dict:
        """Close the current segment and return its income statement.

        Returns ``{"total", "shares", "hhi", "alarms"}`` and RESETS the
        per-segment accumulators. ``total`` is the signed net; ``shares`` are
        fractions of the abs-sum gross; ``hhi`` is the Herfindahl index
        (sum of squared shares — 1.0 means a monopoly, 1/n means n equal
        sources), which gives a single scalar to plot even when no alarm has
        fired yet.

        Zero-total segments (nothing recorded, or only zero amounts) return
        empty shares AND clear every dominance streak: an idle stretch is
        positive evidence the "dominant" source is not a runaway farm, since
        a farm by definition never stops paying.
        """
        try:
            signed, abs_ = self._signed, self._abs
            self._signed, self._abs = {}, {}

            total = float(sum(signed.values()))
            abs_total = float(sum(abs_.values()))
            alarms: List[str] = []

            if abs_total <= _EPS:
                self._streaks.clear()
                out = {"total": total, "shares": {}, "hhi": 0.0,
                       "alarms": alarms}
            else:
                shares = {s: v / abs_total for s, v in abs_.items()}
                hhi = float(sum(f * f for f in shares.values()))
                # Streaks: only sources dominant THIS segment carry a streak
                # forward; everything else (sub-threshold or absent) resets.
                new_streaks: Dict[str, int] = {}
                for s, f in shares.items():
                    if f > self.share_alarm:
                        st = self._streaks.get(s, 0) + 1
                        new_streaks[s] = st
                        if st >= self.consecutive:
                            alarms.append(
                                f"DOMINANT source={s} share={f:.2f} "
                                f"segments={st}")
                self._streaks = new_streaks
                out = {"total": total, "shares": shares, "hhi": hhi,
                       "alarms": alarms}

            if self._err_count:
                out["alarms"].append(
                    f"LEDGER-ERROR count={self._err_count} "
                    f"last={self._last_err}")
                self._err_count, self._last_err = 0, ""
            return out
        except Exception as exc:  # noqa: BLE001
            # Even a broken close must hand the caller a well-shaped dict.
            return {"total": 0.0, "shares": {}, "hhi": 0.0,
                    "alarms": [f"LEDGER-ERROR count=1 last={exc!r}"]}

    # -- internals ---------------------------------------------------------

    def _note_error(self, msg: str) -> None:
        self._err_count += 1
        self._last_err = msg[:200]  # bound memory even if msg embeds a frame


# ---------------------------------------------------------------------------
# FarmDetector
# ---------------------------------------------------------------------------

# EMA smoothing for a cell's per-loop income rate. 0.2 keeps ~5 loops of
# memory: fast enough to notice a farm within its first dozen laps, slow
# enough that one lucky lap (achievement reward happening to land mid-cycle)
# does not by itself push a cell over the alarm line.
_EMA_BETA = 0.2

# Per-cell action histograms exist so the report can say not just WHERE the
# loop is but roughly WHAT the agent does on it (the pickup farm was
# instantly recognisable from its two-action signature). They are capped
# because the action space in this project has already widened once mid-run
# (10 -> 12) and may again; on overflow the smallest count is evicted, which
# preserves the dominant signature the report actually prints.
_MAX_HIST = 16


class FarmDetector:
    """Flags revisited state cells that pay sustained positive net income.

    The soundness argument, once more, because it is the entire design: with
    potential-based shaping, cumulative reward over any closed loop
    telescopes to (terminal achievement + intrinsic) only — the shaping terms
    cancel. So a cell the agent keeps returning to, whose loop income rate
    stays positive after smoothing, is either a place where real progress
    keeps happening (fine — a human glancing at the report will recognise
    it) or a reward farm (the thing that has burned this project five times).
    Either way it deserves a line in the report; the detector only ranks, it
    never intervenes.
    """

    def __init__(self, revisit_horizon: int = 400, min_loops: int = 6,
                 income_alarm: float = 0.02, max_cells: int = 20000):
        self.revisit_horizon = max(1, int(revisit_horizon))
        self.min_loops = max(1, int(min_loops))
        self.income_alarm = float(income_alarm)
        self.max_cells = max(1, int(max_cells))
        # Running cumulative reward: loop income is a difference of two
        # readings of this counter, so per-cell storage stays O(1) no matter
        # how long the loop is.
        self._cum: float = 0.0
        # Per-cell state, ordered by recency of visit (OrderedDict as LRU).
        # rec = [last_step, cum_at_last_step, loops, ema_or_None, hist_dict]
        self._cells: "OrderedDict[int, list]" = OrderedDict()
        # Rolling (step, action) window for histogram attribution. Bounded by
        # the horizon: anything older can never fall inside a valid loop.
        self._recent: Deque[Tuple[int, int]] = deque(
            maxlen=self.revisit_horizon)
        self._err_count: int = 0
        self._last_err: str = ""

    # -- online update -----------------------------------------------------

    def step(self, cell: int, action: int, step_reward: float,
             step: int) -> None:
        """Ingest one transition. Never raises.

        The current step's reward is booked into the cumulative counter
        BEFORE the revisit check, so the reward for arriving back at a cell
        counts toward that loop's income — farms typically pay exactly on
        the closing transition, and excluding it would let a one-step-payout
        farm read as zero-income forever.
        """
        try:
            cell = int(cell)
            action = int(action)
            step = int(step)
            r = float(step_reward)
            if math.isfinite(r):
                self._cum += r
            else:
                self._note_error(f"non-finite reward at step {step}")

            self._recent.append((step, action))

            rec = self._cells.get(cell)
            if rec is not None:
                then, cum_then = rec[0], rec[1]
                span = step - then
                # span<=0 guards duplicate calls and timeline resets (an env
                # restart that rewinds the caller's step counter must not
                # close a "loop" of negative length).
                if 0 < span <= self.revisit_horizon:
                    rate = (self._cum - cum_then) / max(1, span)
                    rec[2] += 1
                    rec[3] = (rate if rec[3] is None
                              else (1.0 - _EMA_BETA) * rec[3]
                              + _EMA_BETA * rate)
                    self._absorb_actions(rec[4], then, step)
                rec[0], rec[1] = step, self._cum
                self._cells.move_to_end(cell)
            else:
                self._cells[cell] = [step, self._cum, 0, None, {}]
                # LRU eviction: the cell not visited for longest goes first.
                # Farms revisit constantly, so a live farm can never be the
                # eviction victim even at the 20k-cell bound.
                while len(self._cells) > self.max_cells:
                    self._cells.popitem(last=False)
        except Exception as exc:  # noqa: BLE001
            self._note_error(repr(exc))

    # -- reporting ---------------------------------------------------------

    def segment_report(self) -> List[str]:
        """Lines for cells looping >= min_loops with EMA income > alarm.

        Best-first (highest EMA rate, then most loops, then lowest cell id
        for determinism), capped at 5 lines — the report is for a human
        scanning a log, and if more than 5 cells qualify the top 5 are
        already the story. Never raises; internal failures come back as a
        single error line instead of a traceback.
        """
        try:
            hits = []
            for cid, rec in self._cells.items():
                if (rec[2] >= self.min_loops and rec[3] is not None
                        and rec[3] > self.income_alarm):
                    hits.append((rec[3], rec[2], cid, rec[4]))
            hits.sort(key=lambda t: (-t[0], -t[1], t[2]))
            lines: List[str] = []
            for ema, loops, cid, hist in hits[:5]:
                top = [a for a, _ in sorted(
                    hist.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
                lines.append(f"FARM? cell={cid} loops={loops} "
                             f"+{ema:.3f}/step actions={top}")
            if self._err_count:
                lines.append(f"DETECTOR-ERROR count={self._err_count} "
                             f"last={self._last_err}")
                self._err_count, self._last_err = 0, ""
            return lines
        except Exception as exc:  # noqa: BLE001
            return [f"DETECTOR-ERROR count=1 last={exc!r}"]

    # -- internals ---------------------------------------------------------

    def _absorb_actions(self, hist: Dict[int, int], then: int,
                        now: int) -> None:
        """Fold the actions taken on the (then, now] window into ``hist``."""
        for s, a in self._recent:
            if then < s <= now:
                if a not in hist and len(hist) >= _MAX_HIST:
                    # Evict the rarest to admit the new action: keeps the
                    # dominant top-3 signature intact under any churn.
                    victim = min(hist.items(), key=lambda kv: kv[1])[0]
                    del hist[victim]
                hist[a] = hist.get(a, 0) + 1

    def _note_error(self, msg: str) -> None:
        self._err_count += 1
        self._last_err = msg[:200]
