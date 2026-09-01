"""Proof-of-life heartbeats + executable invariants.

WHY THIS EXISTS.  The most expensive bug class in this project has been
SILENT INERTNESS: a subsystem that is configured, logged as "active" at
startup, and then never actually executes.  Five confirmed instances to
date (one dead for weeks) shared the same shape — the code path that was
supposed to fire had a guard, a stale flag, or a dead wire in front of
it, and nothing in the run ever noticed because absence of output looks
identical to healthy silence.  ``Heartbeat`` inverts that default: every
subsystem must continuously prove it runs, and a subsystem that stops
beating (or NEVER beats) becomes loud data instead of quiet nothing.

``InvariantSet`` is the companion for a subtler failure: properties we
*know* must hold ("no gate stays closed forever", "no single reward
source dominates") but that live only in heads and postmortems.  Each
such property becomes a callable checked every reporting segment, so a
drifting run is flagged while it is still cheap to stop.

DEFENSIVE CONTRACT.  This module is monitoring; monitoring must never be
able to take down a multi-day run.  Every externally-supplied callable
is executed inside a containment boundary and every failure is converted
into *data* (a report line), never an exception that propagates to the
training loop.  The formatting paths are likewise wrapped: a malformed
registration should produce an ugly line, not a crash.

Domain-agnostic by design: names are opaque strings, steps are opaque
monotone integers, contexts are plain dicts.  Nothing here knows what
the subsystems do.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

# Grace factor: a subsystem is only flagged after missing TWO full
# expected periods.  One period of slack is deliberate — many subsystems
# beat on best-effort cadences (queue drains, throttled flushes) and a
# single late beat is normal jitter, not death.  Two consecutive missed
# periods has, empirically, always meant "actually dead".
_GRACE = 2


class Heartbeat:
    """Registry of "I ran" pulses with overdue detection.

    Subsystems call :meth:`beat` at their natural execution point (the
    line that would only run if the subsystem is genuinely alive — put
    the beat AFTER the guard, inside the work, never before it, or the
    beat itself becomes the next silent-inertness lie).  A periodic
    :meth:`report` turns the registry into human-readable table lines
    plus a machine-readable list of overdue names.
    """

    def __init__(self) -> None:
        # name -> (expected_every, registered_at_step, last_beat or None)
        # Insertion order is preserved so report tables are stable across
        # segments — a table whose rows reshuffle is unreadable in logs.
        self._entries: Dict[str, List[Optional[int]]] = {}

    def register(self, name: str, expected_every: int, step: int) -> None:
        """Declare a subsystem and its expected beat cadence.

        ``expected_every`` is the step interval between beats; ``0``
        means "may legitimately be silent" — the subsystem is listed
        (visibility) but never flagged overdue (no false alarms for
        opportunistic paths like error handlers or rare-event hooks).

        Registering declares the EXPECTATION even if the subsystem never
        runs — that is the whole point: never-fired is detectable only
        because registration happened somewhere else than the beat.
        Re-registering an existing name updates its cadence but keeps
        its beat history (a cadence tune-up must not amnesty a corpse).
        """
        if name in self._entries:
            self._entries[name][0] = int(expected_every)
            return
        self._entries[name] = [int(expected_every), int(step), None]

    def beat(self, name: str, step: int) -> None:
        """Record a proof-of-life pulse.

        Beating an unregistered name auto-registers it with
        ``expected_every=0``: visibility without ceremony.  A subsystem
        that was never formally declared still shows up in the table
        (so we can SEE it runs), but carries no cadence expectation
        (we never claimed to know one, so we must not invent alarms).
        """
        entry = self._entries.get(name)
        if entry is None:
            self._entries[name] = [0, int(step), int(step)]
            return
        entry[2] = int(step)

    def _is_overdue(self, expected: int, registered: int,
                    last: Optional[int], step: int) -> bool:
        """OVERDUE = more than _GRACE expected periods of silence.

        never-fired uses registration as the reference point: a
        subsystem that has existed for 2 periods and never beaten is
        exactly as dead as one that beat once and then stopped — this
        is the case the five historical inertness bugs all fell into.
        """
        if expected <= 0:
            return False                     # "may be silent" contract
        ref = registered if last is None else last
        return (step - ref) > _GRACE * expected

    def report(self, step: int) -> Tuple[List[str], List[str]]:
        """Render the registry: ``(table_lines, overdue_names)``.

        The table is for humans grepping a log; the overdue list is for
        code (escalation, counters, kill-switches).  Both are derived
        from the same per-entry judgment so they can never disagree.
        Formatting is contained per-row: one corrupt entry costs one
        ugly line, never the whole report.
        """
        lines: List[str] = []
        overdue: List[str] = []
        for name, entry in self._entries.items():
            try:
                expected, registered, last = entry
                late = self._is_overdue(expected, registered, last, step)
                if last is None:
                    age = step - registered
                    line = (f"{name}: last=never "
                            f"({age} ago, every~{expected})")
                    if late:
                        line += " <-- NEVER FIRED"
                else:
                    age = step - last
                    line = (f"{name}: last={last} "
                            f"({age} ago, every~{expected})")
                    if late:
                        line += " <-- OVERDUE"
                lines.append(line)
                if late:
                    overdue.append(name)
            except Exception as e:  # noqa: BLE001 - monitoring never raises
                lines.append(f"{name}: <report error: {e!r}>")
        return lines, overdue


class InvariantSet:
    """Named, per-segment executable checks with violation streaks.

    Each invariant is a ``fn(ctx) -> Optional[str]``: ``None`` when the
    property holds, else a one-line description of HOW it is violated
    (the message should carry the measured numbers — "gate X closed for
    3400 steps" — because by the time a human reads it the live state
    is gone).

    Streaks exist because a single-segment violation is often transient
    (a gate closed across one segment boundary) while a GROWING streak
    is the signature of a latch — the recurring anti-pattern where a
    temporary guard becomes a permanent off-switch.  Callers escalate on
    ``streak(name)`` crossing a threshold rather than on any single hit.
    """

    def __init__(self) -> None:
        self._fns: Dict[str, Callable[[dict], Optional[str]]] = {}
        self._streaks: Dict[str, int] = {}

    def add(self, name: str, fn: Callable[[dict], Optional[str]]) -> None:
        """Register (or replace) an invariant.

        Replacing resets the streak: a rewritten check is a new claim
        about the world and inherits no debt from the old one.
        """
        self._fns[name] = fn
        self._streaks[name] = 0

    def evaluate(self, ctx: dict) -> List[str]:
        """Run every invariant against ``ctx``; return violation lines.

        A RAISING fn is contained and reported as
        ``INVARIANT <name> ERRORED: <e>`` — the daemon must never take
        the run down.  An errored check also INCREMENTS the streak: an
        invariant that cannot execute cannot vouch for the property it
        guards, and treating "my check crashed" as "all clear" is how a
        broken sensor hides a real fire for weeks.
        """
        out: List[str] = []
        for name, fn in self._fns.items():
            try:
                msg = fn(ctx)
            except Exception as e:  # noqa: BLE001 - contain, expose as data
                out.append(f"INVARIANT {name} ERRORED: {e}")
                self._streaks[name] = self._streaks.get(name, 0) + 1
                continue
            if msg is None:
                self._streaks[name] = 0
            else:
                # Coerce defensively: a fn returning a non-str truthy
                # value still yields a printable violation line rather
                # than a formatting crash three frames away.
                out.append(f"INVARIANT {name}: {msg}")
                self._streaks[name] = self._streaks.get(name, 0) + 1
        return out

    def streak(self, name: str) -> int:
        """Consecutive segments this invariant has failed (0 = healthy).

        Unknown names return 0 rather than raising — escalation code
        polling a not-yet-registered invariant is a wiring race, not a
        reason to kill a run.
        """
        return self._streaks.get(name, 0)
