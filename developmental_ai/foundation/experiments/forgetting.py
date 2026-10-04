"""Retention probes and the forgetting/instability tracker (plan Stage 10
item 7; Stage 11 item 7).

THE FAILURE IT EXISTS FOR. A learner that forgets and relearns the same
material shows a falling loss on every relearning pass. A progress signal
built on "loss went down" pays for every pass, forever: the agent can
manufacture unlimited "discovery" by oscillating (CLAUDE.md §9: most
historical progress was the agent finding a way to get paid for nothing).

THE FAILURE IT MUST NOT CAUSE (verifier finding B2, stage9_11/RESULTS.md).
The first version moved a probe between unknown/known/forgotten on SINGLE
evaluations and discounted gains whenever the probe was not known while
charging regressions in full. Stationary probe noise that straddles the
hysteresis band then flipped the state every few evaluations: 16 probes of
i.i.d. N(0.5, 0.2) for 2000 evaluations netted -3011..-3057 "progress"
(telescoped truth: -1.4..+0.6) and flagged 16/16 probes as forgetting
loops. A usefulness term built on that would punish noisy-but-stable
competence, and the flag could not clear while the noise lasted (§4.1).

WHAT IT DOES NOW. `RetentionTracker.observe(t, losses)` takes the current
loss of every retention probe (dev probes; see progress.py).

  STATE CHANGES ARE STATISTICAL, NOT SINGLE CROSSINGS. Per probe:
    noise floor  sigma = median over the last `noise_history` evaluations of
                 the rolling `window`-sample variance, unbiased by the chi^2
                 median (then max with `min_noise`). A median of short-window
                 variances measures within-regime scatter and ignores the
                 few windows that straddle a genuine level shift, so a
                 sustained phase is not mistaken for noise and noise is not
                 mistaken for phases.
    known        mean of the last `window` losses + z * sigma/sqrt(window)
                 < known_below
    forgotten    (from known only) mean - z * sigma/sqrt(window)
                 > forgotten_above
  so a transition needs a SUSTAINED level that is significantly beyond the
  threshold given the probe's own measured noise. No transition before
  `min_history` evaluations (a noise floor needs data).

  ACCOUNTING IS SYMMETRIC WITHIN A STATE. Per evaluation:
    gain_p       = max(0, prev_p - now_p)            improvement
    forgetting_p = max(0, now_p - prev_p)            regression
    weight_p     = relearn_discount ** learned_p  while the probe is
                   FORGOTTEN (learned_p = times it was already known),
                   else 1
    progress     = sum_p weight_p * (gain_p - forgetting_p)          (NET)

  * inside ANY state the weight multiplies both directions, so progress
    telescopes: a probe whose loss merely fluctuates — in whatever state —
    has E[progress] = 0 and total progress bounded by its range;
  * the only non-telescoping terms come from GENUINE state changes: the
    fall that makes a known probe forgotten is charged at weight 1 (it is
    still known when it falls), the climb back is credited at the discount.
    An endless learn/forget loop therefore nets about
    -amplitude * (1 - discount) per cycle, and its relearning is reported
    as `relearning_progress`, never as discovery;
  * a probe that completes `loop_flag` forget->relearn cycles is FLAGGED
    (`flags`), with an `instability` fraction per evaluation.

  The price: a forget/relearn alternation FASTER than `window` evaluations
  is indistinguishable from noise of that amplitude (a two-point i.i.d.
  process has the same marginal and the same short-window variance), so it
  is treated as noise — net 0, unflagged — not as a loop. Its progress is
  still NET (telescoping), so it cannot farm credit either.

RECOVERY (§4.1). The flag is a guard; what re-opens it:
  * retention: each `retention_needed(pid)` consecutive evaluations in
    which a flagged probe stays known removes one loop — `stable_evals`, or
    twice the longest known stretch the probe has already forgotten after,
    whichever is longer (a loop that holds each phase for 12 evaluations
    must not clear its own flag every cycle). That number is finite and
    known from the probe's own history. Leaving the known state needs
    significant deterioration, so a stationary noisy probe that is known
    accrues stability (the B2 latch is gone);
  * change: `reset_probe(pid, reason)` — what a change detector
    (inference.PredictiveCUSUM) calls when the rule behind a probe has
    really changed — clears loops and the discount, so relearning after a
    genuine change is new learning again.
A flagged probe that is currently forgotten stays flagged until it is
relearned and retained, or reset: both reachable by the agent's own
evidence; neither needs a human.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional

import numpy as np
from scipy.stats import chi2

from ..contracts import ContractError

UNKNOWN_STATE, KNOWN, FORGOTTEN = "unknown", "known", "forgotten"


class RetentionTracker:
    def __init__(self, known_below: float, forgotten_above: float,
                 relearn_discount: float = 0.25, loop_flag: int = 2,
                 stable_evals: int = 5, max_probes: int = 4096,
                 window: int = 4, z: float = 3.0, noise_history: int = 32,
                 min_history: Optional[int] = None, min_noise: float = 0.0):
        if not (math.isfinite(known_below) and math.isfinite(forgotten_above)) \
                or not known_below < forgotten_above:
            raise ContractError("need finite known_below < forgotten_above (hysteresis)")
        if not 0.0 <= relearn_discount < 1.0:
            raise ContractError("relearn_discount must be in [0, 1)")
        if int(loop_flag) < 1 or int(stable_evals) < 1:
            raise ContractError("loop_flag and stable_evals must be >= 1")
        if isinstance(window, bool) or int(window) < 2:
            raise ContractError("window must be >= 2 (a level needs repeated evaluations)")
        if not (math.isfinite(z) and z >= 0):
            raise ContractError("z must be finite and >= 0")
        if int(noise_history) < int(window) + 1:
            raise ContractError("noise_history must exceed window")
        mh = 2 * int(window) if min_history is None else int(min_history)
        if not int(window) <= mh <= int(noise_history):
            raise ContractError("need window <= min_history <= noise_history")
        if not (math.isfinite(min_noise) and min_noise >= 0):
            raise ContractError("min_noise must be finite and >= 0")
        self.known_below, self.forgotten_above = float(known_below), float(forgotten_above)
        self.relearn_discount = float(relearn_discount)
        self.loop_flag, self.stable_evals = int(loop_flag), int(stable_evals)
        self.max_probes = int(max_probes)
        self.window, self.z = int(window), float(z)
        self.noise_history, self.min_history = int(noise_history), mh
        self.min_noise = float(min_noise)
        # median of chi^2_{w-1}/(w-1): unbiases a median of sample variances
        self._var_median = float(chi2.median(self.window - 1) / (self.window - 1))
        self.state: Dict[str, str] = {}
        self.last: Dict[str, float] = {}
        self.learned: Dict[str, int] = {}
        self.loops: Dict[str, int] = {}
        self.stable: Dict[str, int] = {}
        self.run: Dict[str, int] = {}         # consecutive evaluations in KNOWN
        self.held_max: Dict[str, int] = {}    # longest KNOWN run that ended in forgetting
        self.hist: Dict[str, Deque[float]] = {}
        self.resets: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.totals = {"raw_gain": 0.0, "progress": 0.0, "relearning_progress": 0.0,
                       "forgetting": 0.0, "forget_events": 0, "relearn_events": 0}
        self._t = None

    def reset_probe(self, probe_id: str, reason: str) -> None:
        if probe_id not in self.state:
            raise ContractError(f"unknown probe {probe_id!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ContractError("a reset needs a stated reason (e.g. the change "
                                "detector's alarm)")
        self.resets.append({"probe": probe_id, "reason": reason,
                            "learned": self.learned[probe_id], "t": self._t})
        self.state[probe_id] = UNKNOWN_STATE
        self.learned[probe_id] = 0
        self.loops[probe_id] = 0
        self.stable[probe_id] = 0
        self.run[probe_id] = self.held_max[probe_id] = 0
        self.hist[probe_id].clear()          # the old regime's levels are not evidence

    def retention_needed(self, probe_id: str) -> int:
        """Consecutive known evaluations that remove one loop: stable_evals,
        or twice the longest known stretch this probe has already forgotten
        after, whichever is longer — retention must outlast the loop's own
        known phase, or a slow loop would clear its own flag every cycle."""
        return max(self.stable_evals, 2 * self.held_max.get(probe_id, 0))

    def noise_floor(self, probe_id: str) -> Optional[float]:
        """Measured per-probe noise sd (None until min_history evaluations)."""
        h = self.hist.get(probe_id)
        if h is None or len(h) < self.min_history:
            return None
        x = np.asarray(h, np.float64)
        win = np.lib.stride_tricks.sliding_window_view(x, self.window)
        s2 = float(np.median(win.var(axis=1, ddof=1))) / self._var_median
        return max(math.sqrt(max(s2, 0.0)), self.min_noise)

    def _level(self, pid: str):
        """(mean of the last `window` losses, its z * standard error) or None."""
        sd = self.noise_floor(pid)
        if sd is None:
            return None
        h = self.hist[pid]
        recent = [h[-k] for k in range(1, self.window + 1)]
        return sum(recent) / self.window, self.z * sd / math.sqrt(self.window)

    def observe(self, t: int, losses: Mapping[str, float]) -> Dict[str, Any]:
        if self._t is not None and t <= self._t:
            raise ContractError(f"evaluation time {t} does not advance past {self._t}")
        if not losses:
            raise ContractError("no probe losses given")
        raw = prog = relearn = forget = 0.0
        forgot_now = 0
        for pid, v in losses.items():
            v = float(v)
            if not math.isfinite(v):
                raise ContractError(f"probe {pid!r} loss {v} is not finite")
            if pid not in self.state:
                if len(self.state) >= self.max_probes:
                    raise ContractError("max_probes exceeded")
                self.state[pid], self.learned[pid] = UNKNOWN_STATE, 0
                self.loops[pid], self.stable[pid] = 0, 0
                self.run[pid] = self.held_max[pid] = 0
                self.hist[pid] = deque(maxlen=self.noise_history)
            prev = self.last.get(pid)
            st = self.state[pid]
            if prev is not None:
                g, f = max(0.0, prev - v), max(0.0, v - prev)
                w = self.relearn_discount ** self.learned[pid] if st == FORGOTTEN else 1.0
                raw += g
                prog += w * (g - f)
                relearn += (1.0 - w) * g
                forget += f
            self.hist[pid].append(v)
            self.last[pid] = v
            # state machine: only a sustained, significant level moves it
            if st != KNOWN:
                h = self.hist[pid]
                cheap = len(h) >= self.window and \
                    sum(h[-k] for k in range(1, self.window + 1)) / self.window < self.known_below
                lv = self._level(pid) if cheap else None
                if lv is not None and lv[0] + lv[1] < self.known_below:
                    if st == FORGOTTEN:
                        self.loops[pid] += 1
                        self.totals["relearn_events"] += 1
                    self.state[pid] = KNOWN
                    self.learned[pid] += 1
                    self.stable[pid] = self.run[pid] = 0
            else:
                h = self.hist[pid]
                cheap = sum(h[-k] for k in range(1, self.window + 1)) / self.window \
                    > self.forgotten_above if len(h) >= self.window else False
                lv = self._level(pid) if cheap else None
                if lv is not None and lv[0] - lv[1] > self.forgotten_above:
                    self.state[pid] = FORGOTTEN
                    self.held_max[pid] = max(self.held_max[pid], self.run[pid])
                    self.stable[pid] = 0
                    forgot_now += 1
                    self.totals["forget_events"] += 1
                else:
                    self.stable[pid] += 1
                    self.run[pid] += 1
                    if self.stable[pid] >= self.retention_needed(pid) and self.loops[pid] > 0:
                        self.loops[pid] -= 1                 # recovery: retained
                        self.stable[pid] = 0
        self._t = t
        for k, x in (("raw_gain", raw), ("progress", prog),
                     ("relearning_progress", relearn), ("forgetting", forget)):
            self.totals[k] += x
        out = {"t": t, "raw_gain": raw, "progress": prog,
               "relearning_progress": relearn, "forgetting": forget,
               "instability": forgot_now / float(len(losses)),
               "flags": self.flags()}
        self.history.append(out)
        return out

    def flags(self) -> List[str]:
        return sorted(p for p, n in self.loops.items() if n >= self.loop_flag)
