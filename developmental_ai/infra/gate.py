"""Gate / GateRegistry — structural defence against "a guard becomes a latch".

WHY THIS EXISTS. Nine separate times in this project a protective mechanism —
something that suppresses a behaviour until conditions improve — silently
became an ABSORBING state: the condition that was supposed to reopen it could
no longer occur *because* the gate was closed. The worst instance cost a
19-hour zero-reward run (a cold-start budget latch); another froze every
learned skill at zero invocations forever (a competence threshold whose input
signal only moved while the gate was open). The failure is always the same
shape and it is never visible from inside the mechanism that has it: locally,
each gate looks like a sensible `if`.

The fix here is STRUCTURAL, not case-by-case. Any code that closes a gate must
declare, at construction time, (a) the human-readable condition under which it
reopens and (b) the maximum number of steps it is allowed to stay closed. The
framework then does the one thing the nine buggy gates never did: it watches
the promise. A gate closed past its budget produces an ALARM STRING — data,
surfaced through `alarms()`/`table()` into whatever log or dashboard the run
already has — naming the gate, how long it has been shut, and the reopen
condition someone once believed would fire. A latch can still happen, but it
can no longer happen *quietly*.

Design constraints honoured throughout:
  * DOMAIN-AGNOSTIC — nothing here knows what behaviour a gate suppresses.
  * MONITORING MUST NEVER KILL THE RUN — every public method is written so a
    bad argument or internal surprise degrades to conservative state or an
    error string in the output, never an exception into a multi-day loop.
  * PASSIVE — this module only observes and reports. It never reopens a gate
    itself, because auto-reopening would be yet another mechanism with its own
    latch modes; the alarm is the product.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Sentinel defaults for the registry's create-or-get path. A caller that
# passes only a name (the common case at every close/open site after the one
# declaration site) must not silently overwrite the real declaration with
# these placeholders — see GateRegistry.gate().
_DEFAULT_REOPEN = "unspecified"
_DEFAULT_MAX_CLOSED = 50000


def _as_step(step: object) -> int:
    """Coerce a caller-supplied step to int WITHOUT raising.

    Monitoring code taking down the run it monitors is the one outcome worse
    than a latch, so a garbage step (None, NaN, a string from a corrupted
    counter) collapses to 0 rather than raising. The consequence is benign in
    both directions: a gate closed "at step 0" merely alarms EARLY, which is
    exactly the failure mode we want to be loud.
    """
    try:
        return int(step)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0


class Gate:
    """One suppression mechanism, with its reopen promise attached.

    The object is deliberately dumb: it records WHEN it closed and WHY, and
    can say whether the closure has outlived its declared budget. All policy
    (what closes it, what reopens it) stays in the caller — a gate that could
    act on its own state would just be latch-risk relocated.
    """

    def __init__(self, name: str, reopen: str, max_closed_steps: int) -> None:
        self.name = str(name)
        # The reopen string is the whole point of the exercise: when the
        # alarm fires at 3am on day two, the operator must be able to read
        # WHICH condition was supposed to fire and go check why it did not,
        # without archaeology through the closing code.
        self.reopen = str(reopen)
        # A non-positive budget would make every closed gate instantly
        # overdue forever; clamp to 1 so a config typo yields a loud-but-
        # sane gate instead of an alarm firehose that trains people to
        # ignore the channel (alarm fatigue is how latch #9 survived).
        self.max_closed_steps = max(1, _as_step(max_closed_steps))
        self._closed_since: Optional[int] = None
        self._reason: str = ""

    # -- state transitions -------------------------------------------------

    def close(self, step: int, reason: str = "") -> None:
        """Close the gate. Idempotent: re-closing while already closed keeps
        the ORIGINAL closed-since step and first reason.

        This idempotency is load-bearing, not a convenience. Condition-fed
        gates get "closed" every step their condition holds; if each call
        reset the clock, a permanently-stuck gate would report itself as
        freshly closed forever and the overdue alarm — the entire purpose of
        this module — could never fire. (That is precisely the self-masking
        shape of the historical latches.)
        """
        if self._closed_since is None:
            self._closed_since = _as_step(step)
            self._reason = str(reason)

    def open(self, step: int) -> None:
        """Open the gate. Idempotent; clears the closure record so the next
        close starts a fresh clock with a fresh reason."""
        # `step` is accepted (symmetry with close, and so call sites never
        # need to special-case) but not stored: an open gate has nothing to
        # be overdue about, and keeping stale timing state around is how
        # "cleared" flags come back from the dead.
        del step
        self._closed_since = None
        self._reason = ""

    # -- inspection --------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        return self._closed_since is not None

    def steps_closed(self, step: int) -> int:
        """How long the gate has been closed as of `step` (0 while open).

        Clamped at 0 so a stale/backwards step from the caller (seen in
        practice when two loops share a gate but tick different counters)
        reads as "just closed" rather than producing a negative age that
        arithmetic downstream would mishandle.
        """
        if self._closed_since is None:
            return 0
        return max(0, _as_step(step) - self._closed_since)

    def overdue(self, step: int) -> Optional[str]:
        """None while open or within budget; else a one-line alarm.

        The alarm carries everything an operator needs in one grep-able
        line: gate name, age vs budget, the reason it closed, and the
        declared reopen condition — because the diagnosis of every latch so
        far has been "the reopen condition can no longer occur", and you can
        only check that if the line tells you what the condition was.
        """
        if self._closed_since is None:
            return None
        n = self.steps_closed(step)
        if n <= self.max_closed_steps:
            return None
        return (
            "GATE OVERDUE: '%s' closed for %d steps (max %d, since step %d) "
            "reason='%s' reopens when: %s"
            % (self.name, n, self.max_closed_steps, self._closed_since,
               self._reason, self.reopen)
        )


class GateRegistry:
    """The run-wide roster of every declared gate.

    A registry exists (rather than loose Gate objects) so that ONE monitoring
    call site — the main loop's periodic health dump — can enumerate every
    suppression mechanism in the system without knowing any of them by name.
    Latches historically survived because each gate was private to its
    subsystem and nobody's job was to look at all of them.
    """

    def __init__(self) -> None:
        # Insertion-ordered (plain dict): table()/alarms() output order is
        # declaration order, which is stable across a run — important because
        # these lines get diffed between health dumps to spot state changes.
        self._gates: Dict[str, Gate] = {}

    def gate(self, name: str, reopen: str = _DEFAULT_REOPEN,
             max_closed_steps: int = _DEFAULT_MAX_CLOSED) -> Gate:
        """Create-or-get by name. Later calls may pass only the name.

        Upgrade-not-clobber rule: if the gate already exists, arguments that
        are still at their sentinel defaults are ignored (so the many
        `registry.gate("x").close(...)` call sites cannot erase the one real
        declaration), while explicitly-passed non-default values REPLACE the
        stored ones (so a gate first touched by a generic code path can be
        given its real reopen promise later, e.g. by set_state()).
        """
        name = str(name)
        g = self._gates.get(name)
        if g is None:
            g = Gate(name, reopen, max_closed_steps)
            self._gates[name] = g
        else:
            if reopen != _DEFAULT_REOPEN:
                g.reopen = str(reopen)
            if max_closed_steps != _DEFAULT_MAX_CLOSED:
                g.max_closed_steps = max(1, _as_step(max_closed_steps))
        return g

    def set_state(self, name: str, closed: bool, step: int, reason: str = "",
                  reopen: str = _DEFAULT_REOPEN,
                  max_closed_steps: int = _DEFAULT_MAX_CLOSED) -> Gate:
        """Convenience for CONDITION-FED gates: mirror a boolean every step.

        Most real gates are not event-driven — they are `suppressed = cond`
        recomputed every tick. This lets that call site stay a single line
        while still getting edge detection: the open->closed edge records
        step+reason, the closed->open edge clears, and repeated same-state
        calls are no-ops (delegating to close/open idempotency, so the
        overdue clock measures the FULL closure, not the last tick of it).
        """
        g = self.gate(name, reopen=reopen, max_closed_steps=max_closed_steps)
        if closed:
            g.close(step, reason)
        else:
            g.open(step)
        return g

    # -- monitoring surface ------------------------------------------------

    def alarms(self, step: int) -> List[str]:
        """Every overdue message, in declaration order.

        Per-gate try/except: one corrupted gate must not hide the alarms of
        the healthy ones (and must not raise into the caller's loop) — its
        failure is itself reported as an alarm line, because a monitor that
        cannot monitor is exactly the news this channel exists to carry.
        """
        out: List[str] = []
        for name, g in list(self._gates.items()):
            try:
                msg = g.overdue(step)
                if msg is not None:
                    out.append(msg)
            except Exception as exc:  # noqa: BLE001 — contain, expose as data
                out.append("GATE MONITOR ERROR: '%s' overdue() raised %s: %s"
                           % (name, type(exc).__name__, exc))
        return out

    def table(self, step: int) -> List[str]:
        """One line per gate for the periodic health dump.

        Format is intentionally rigid ("name: OPEN" / "name: CLOSED
        since=<s> (<n> steps, max=<m>) reason=<r>") so downstream log
        scrapers can key on it; change it and every dashboard regex breaks.
        """
        out: List[str] = []
        for name, g in list(self._gates.items()):
            try:
                if not g.is_closed:
                    out.append("%s: OPEN" % name)
                else:
                    out.append(
                        "%s: CLOSED since=%d (%d steps, max=%d) reason=%s"
                        % (name, g._closed_since, g.steps_closed(step),
                           g.max_closed_steps, g._reason))
            except Exception as exc:  # noqa: BLE001 — contain, expose as data
                out.append("%s: <monitor error %s: %s>"
                           % (name, type(exc).__name__, exc))
        return out
