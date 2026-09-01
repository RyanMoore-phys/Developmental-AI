"""Signal information-content monitor: catches faithfully-learned constants.

WHY THIS EXISTS
---------------
The single most damaging perception bug this project has suffered was not a
crash — it was silent: a learned predicate sat pinned at ~1.0 for weeks
because its teacher answered "true" to every query, and the student learned
that constant perfectly.  Every dashboard showed ~0.98 student/teacher
agreement, which looked like health.  But agreement only measures
student-matches-teacher; a constant matches its constant teacher flawlessly
while carrying ZERO bits of information, so nothing downstream (goal
selection, shaping potentials, attention) could ever be steered by it.  The
one metric that would have flagged it on day one is information content, so
that is what this module measures: Shannon entropy of the binarised stream
plus raw variance, over a rolling window, per signal.

WHY BOTH ENTROPY *AND* VARIANCE MUST BE LOW TO DEMOTE
-----------------------------------------------------
Binarised entropy alone is blind to sub-threshold analog variation: a signal
hovering just above 0.5 with genuine spread binarises to a constant (0 bits)
yet still carries usable analog gradient for a shaping potential.  Demoting
it on entropy alone would kill a live signal.  Conversely, variance alone
cannot distinguish "informative" from "jittery around a constant decision".
Only a signal that is flat under BOTH lenses is a true dead constant, so
``degenerate()`` requires both floors to be violated.

WHY A ROLLING WINDOW
--------------------
Demotion must be reversible without operator intervention.  The stuck
predicate eventually *was* fixed upstream (the teacher was replaced), and a
lifetime-aggregate monitor would have kept it demoted for days on stale
history.  A ``deque(maxlen=window)`` forgets the constant era at exactly the
rate new evidence arrives, so recovery is automatic.

DEFENSIVE CONTRACT
------------------
This is monitoring code running inside multi-day training loops: it must
NEVER be able to crash the run.  Every public method catches and contains
all exceptions and returns a safe empty value instead.  Containment is not
silence, though — silently eating problems is exactly how the constant
predicate survived — so contained failures are counted in ``_errors`` and
rejected non-finite observations in ``_dropped``, both inspectable as data.

Pure Python + stdlib; no third-party imports, so the module stays importable
from the slimmest probe script or an emergency REPL on a headless box.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Set, Tuple


class SignalMonitor:
    """Rolling per-signal information-content tracker.

    Feed every scalar probability the perception stack emits through
    ``observe()``; poll ``degenerate()`` at report cadence to find signals
    that have collapsed into constants and should be demoted from steering
    duty, and ``report()`` for a one-line log summary.

    Parameters
    ----------
    window:
        Rolling window length per signal.  Sized so that a fixed constant
        era ages out within one window of healthy observations (see module
        docstring on reversibility).
    min_obs:
        Minimum observations before a signal is judged at all.  A young
        signal is legitimately constant-ish while its teacher warms up;
        demoting it early would be the guard-becomes-latch anti-pattern —
        so below this count a signal is simply never listed anywhere.
    entropy_floor:
        Bits below which the binarised stream counts as flat.  0.08 bits
        corresponds to ~99% of observations on one side of 0.5 — far
        flatter than any usefully-steering predicate ever measured here.
    var_floor:
        Raw-probability variance below which the analog stream counts as
        flat.  1e-4 is a stddev of 0.01: visually a flat line.
    """

    def __init__(self, window: int = 1500, min_obs: int = 400,
                 entropy_floor: float = 0.08,
                 var_floor: float = 1e-4) -> None:
        self.window = int(window)
        self.min_obs = int(min_obs)
        self.entropy_floor = float(entropy_floor)
        self.var_floor = float(var_floor)
        self._streams: Dict[str, Deque[float]] = {}
        # Contained-failure ledger: monitoring may never raise, but it must
        # not hide its own breakage either (that is how the constant
        # predicate survived weeks).  Exposed as plain data for probes.
        self._errors: Dict[str, int] = {}
        self._dropped: int = 0  # non-finite observations rejected

    # ------------------------------------------------------------------ in
    def observe(self, name: str, p: float) -> None:
        """Record one probability observation for ``name``; never raises.

        Out-of-range values (including +/-inf, whose clamp direction is
        unambiguous) are clamped silently: an upstream layer emitting
        1.0000001 from float error must not spam logs or lose data.  NaN is
        DROPPED, not clamped: under Python's ``min``/``max`` a NaN silently
        coerces to whichever bound compares first, which would fabricate a
        confident observation out of garbage — the exact failure class this
        monitor exists to expose.
        """
        try:
            key = str(name)
            v = float(p)
            if v != v:  # NaN — the only float with no defensible clamp
                self._dropped += 1
                return
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            dq = self._streams.get(key)
            if dq is None:
                dq = deque(maxlen=self.window)
                self._streams[key] = dq
            dq.append(v)
        except Exception:
            self._errors["observe"] = self._errors.get("observe", 0) + 1

    # ----------------------------------------------------------------- out
    def bits(self) -> Dict[str, float]:
        """Shannon entropy (bits) of each signal's binarised-at-0.5 stream.

        Only signals with >= ``min_obs`` observations appear — a judgement
        on a warming-up signal is worse than no judgement (see ``min_obs``).
        Binarisation uses ``p >= 0.5`` so the metric matches how downstream
        consumers actually threshold the signal into decisions: what we
        care about is whether the *decision* stream carries information.
        """
        out: Dict[str, float] = {}
        try:
            for name, snap in self._snapshots().items():
                ones = sum(1 for v in snap if v >= 0.5)
                out[name] = self._binary_entropy(ones / len(snap))
        except Exception:
            self._errors["bits"] = self._errors.get("bits", 0) + 1
        return out

    def variance(self) -> Dict[str, float]:
        """Raw-probability (population) variance per signal over the window.

        Same ``min_obs`` gate as ``bits()`` so callers can zip the two dicts
        without key mismatches.  This is the lens that saves an analog
        signal hovering near 0.5 from a false degenerate verdict.
        """
        out: Dict[str, float] = {}
        try:
            for name, snap in self._snapshots().items():
                n = len(snap)
                mean = sum(snap) / n
                out[name] = sum((v - mean) ** 2 for v in snap) / n
        except Exception:
            self._errors["variance"] = self._errors.get("variance", 0) + 1
        return out

    def means(self) -> Dict[str, float]:
        """Window mean per signal, same ``min_obs`` gate as ``bits()``.

        Exists for one consumer question (stack.segment, 2026-08-16): a
        degenerate signal stuck LOW is a different animal from one stuck
        HIGH.  Constant-low with proven positives is usually honest absence
        — the thing is not in view — while constant-high asserts presence
        regardless of scene, which is the learned-constant pathology the
        gate exists for.  The split cannot be made from entropy or variance
        (both are side-agnostic), only from the mean."""
        out: Dict[str, float] = {}
        try:
            for name, snap in self._snapshots().items():
                out[name] = sum(snap) / len(snap)
        except Exception:
            self._errors["means"] = self._errors.get("means", 0) + 1
        return out

    def degenerate(self) -> Set[str]:
        """Signals proven flat under BOTH lenses: safe to demote.

        Requires >= ``min_obs`` observations AND entropy < ``entropy_floor``
        AND variance < ``var_floor``.  The conjunction is deliberate — see
        the module docstring: entropy alone would demote a live analog
        signal whose decisions happen to be one-sided.
        """
        out: Set[str] = set()
        try:
            var = self.variance()
            for name, ent in self.bits().items():
                if (ent < self.entropy_floor
                        and var.get(name, 0.0) < self.var_floor):
                    out.add(name)
        except Exception:
            self._errors["degenerate"] = self._errors.get("degenerate", 0) + 1
        return out

    def report(self, top: int = 6) -> str:
        """One-line summary of the lowest-entropy signals, for the run log.

        Shows the WORST signals on purpose: healthy signals need no ink,
        and the whole point is that a dying signal becomes visible in the
        default log stream *before* someone thinks to go looking for it.
        Returns "" when no signal has reached ``min_obs`` so callers can
        ``if line:`` without formatting noise during warm-up.
        """
        try:
            b = self.bits()
            if not b:
                return ""
            worst = sorted(b.items(), key=lambda kv: kv[1])[:max(1, int(top))]
            inner = ", ".join(f"{n}={e:.2f}b" for n, e in worst)
            return f"signal bits: worst [{inner}]"
        except Exception:
            self._errors["report"] = self._errors.get("report", 0) + 1
            return ""

    # ------------------------------------------------------------- helpers
    def _snapshots(self) -> Dict[str, Tuple[float, ...]]:
        """Freeze each mature stream into a tuple before analysis.

        ``observe()`` may run on a different thread (env loop) than the
        reporting cadence (trainer loop); iterating a deque mid-append
        raises RuntimeError in CPython.  Snapshotting per-signal inside its
        own try means one racy signal costs at most its own entry this
        tick, never the whole report — partial data beats no data in a
        monitor.
        """
        out: Dict[str, Tuple[float, ...]] = {}
        for name in list(self._streams.keys()):
            try:
                dq = self._streams.get(name)
                if dq is None:
                    continue
                snap = tuple(dq)
                if len(snap) >= self.min_obs:
                    out[name] = snap
            except Exception:
                self._errors["snapshot"] = (
                    self._errors.get("snapshot", 0) + 1)
        return out

    @staticmethod
    def _binary_entropy(q: float) -> float:
        """H(q) in bits with the 0·log0 := 0 convention (constants -> 0.0)."""
        if q <= 0.0 or q >= 1.0:
            return 0.0
        return -(q * math.log2(q) + (1.0 - q) * math.log2(1.0 - q))
