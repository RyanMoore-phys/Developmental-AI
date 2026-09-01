"""MonotoneCounter — one home for "external cumulative counter" semantics.

WHY THIS EXISTS. Twice this project mis-read an external counter's initial
synchronisation as a burst of real events, and both bugs were expensive
because the mistake happened at the *reader*, far from anything that looked
like counter logic:

  * A server-persistent lifetime total arrived as 0 -> 73 on connect. The
    reader interpreted that as 73 fresh events and triggered its "something
    is badly wrong" recovery — a rejoin — which re-synchronised the counter
    again, forever: a rejoin loop every 57 seconds.
  * Per-session statistics restart at 0 when the external process resets.
    Readers that kept the old baseline saw phantom negative deltas, or —
    worse — counted the climb back up to the old value as new events.

The lesson is that "first observation is baseline, big jumps are resync,
decreases are rebaseline" is a SEMANTICS, not a convention, and it must live
in exactly one place. Every consumer of an external cumulative counter goes
through this class; none of them re-implements the delta arithmetic.

Behavioural contract (see update()):

  * ``None``           -> no reading this tick; 0 events, state untouched.
  * first real reading -> adopted as baseline; 0 events. Whatever history the
                          external counter carries predates us and is not ours
                          to count.
  * small increase     -> that many events (rounded). The baseline advances
                          only when at least one whole event is emitted, so a
                          slow fractional creep accumulates instead of being
                          rounded away tick after tick (a 0.4/tick drift must
                          eventually produce its events, not vanish).
  * large increase     -> RESYNC. A jump beyond ``sanity_delta`` in one tick
                          is, on the measured evidence above, a counter
                          catching up — not that many simultaneous events.
                          0 events; the jump is recorded as data
                          (``last_resync``) for whoever wants to log it.
  * any decrease       -> REBASELINE. Cumulative counters do not go down;
                          a decrease means the external source reset. 0
                          events, adopt the new (lower) reading.

Design constraints honoured throughout:
  * DOMAIN-AGNOSTIC — this class knows nothing about what an "event" is.
  * MONITORING MUST NEVER KILL THE RUN — malformed readings (NaN, inf,
    non-numeric junk from a half-parsed log line) degrade to "no reading",
    never to an exception in a multi-day loop. They are tallied in
    ``bad_readings`` so a flood of garbage is visible as data.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


class MonotoneCounter:
    """Tracks one external monotone (cumulative) counter and converts its
    readings into a stream of event counts with resync/rebaseline semantics.

    Typical use: one instance per external statistic, ``update()`` called with
    every observation (``None`` when the stat was absent this tick), and the
    returned int fed to whatever consumes events. ``pop_resync()`` lets a
    logging layer surface "the counter jumped old->new and we deliberately
    ignored it" without the counting layer having to know about logging.
    """

    def __init__(self, sanity_delta: float = 2.5):
        # Largest single-tick increase we are willing to believe is real
        # events rather than a synchronisation burst. The default (2.5) says
        # "up to two events per observation is plausible, three is not" —
        # right for stats sampled faster than events can physically occur.
        # Callers with bursty-but-real sources should raise it; the 0->73
        # join burst is rejected at any sane setting.
        # Floor at 0.5: below that, a legitimate +1 event would itself be
        # classified as a resync and the counter could never count anything —
        # a config typo must not silently create a dead counter.
        try:
            self._sanity_delta: float = max(0.5, float(sanity_delta))
        except (TypeError, ValueError):
            self._sanity_delta = 2.5

        self._baseline: Optional[float] = None
        #: Most recent ignored jump as ``(old_baseline, new_baseline)``.
        #: Overwritten by a newer resync; cleared by ``pop_resync()``.
        self.last_resync: Optional[Tuple[float, float]] = None
        #: Count of malformed readings swallowed (NaN/inf/non-numeric).
        #: Exposed as data so garbage input is observable, never fatal.
        self.bad_readings: int = 0

    # ------------------------------------------------------------------ API

    @property
    def value(self) -> Optional[float]:
        """Current baseline (the last adopted reading); None before the
        first real observation."""
        return self._baseline

    def update(self, value: Optional[float]) -> int:
        """Feed one observation; return the number of NEW events it implies.

        Never raises: any reading that cannot be understood as a finite float
        is treated as "no reading" (0 events) and tallied in bad_readings.
        """
        v = self._coerce(value)
        if v is None:
            return 0                             # no reading this tick

        if self._baseline is None:
            # First contact. The external counter's pre-existing total is
            # history, not events — adopting it silently is the entire fix
            # for the 0->73 join burst.
            self._baseline = v
            return 0

        delta = v - self._baseline
        if delta < 0.0:
            # Cumulative counters don't decrease; the source reset under us.
            # Keeping the old baseline here is exactly the phantom-delta bug:
            # the climb back up would be double-counted as fresh events.
            self._baseline = v
            return 0

        if delta > self._sanity_delta:
            # Synchronisation burst: adopt, don't count, but leave evidence.
            self.last_resync = (self._baseline, v)
            self._baseline = v
            return 0

        events = int(round(delta))
        if events > 0:
            # Only advance the baseline once whole events are emitted; a
            # fractional creep (+0.4 per tick) must accumulate against the
            # held baseline until it amounts to a real event, instead of
            # being rounded to zero and discarded every single tick.
            self._baseline = v
        return events

    def pop_resync(self) -> Optional[Tuple[float, float]]:
        """Return the pending ``(old, new)`` resync record and clear it,
        or None if no resync happened since the last pop. Read-and-clear so
        a periodic logger reports each burst exactly once."""
        r = self.last_resync
        self.last_resync = None
        return r

    # ------------------------------------------------------------- helpers

    def _coerce(self, value: object) -> Optional[float]:
        """Best-effort finite-float conversion; junk becomes None + a tally.

        External stats arrive from log parsers, RCON strings and half-updated
        dicts; a monitoring path must absorb whatever those produce."""
        if value is None:
            return None
        try:
            v = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            self.bad_readings += 1
            return None
        if math.isnan(v) or math.isinf(v):
            self.bad_readings += 1
            return None
        return v
