"""AffordanceMap — the agent's empirical record of what its body can DO.

WHY THIS EXISTS. This project lost weeks debugging "motivation" — reward
shaping, curiosity schedules, goal selection — when the real problem was a
BODY problem: the action space contained no way to perform a prerequisite
manipulation, so an entire goal class was structurally unreachable, and no
instrument anywhere in the stack said so. Every motivation-side diagnostic
reads the same for "unwilling" and "unable": zero reward, zero progress,
flat learning curves. The two cases have opposite fixes (tune incentives vs
grow the action space), so conflating them sends debugging in exactly the
wrong direction for as long as nobody thinks to ask the other question.

An affordance map answers the OTHER question with data instead of intuition.
It watches the raw (action, resulting-events) stream and maintains three
views that motivation diagnostics cannot provide:

  * effects_of(action)  — what has this action EVER caused? An action with
    thousands of uses and an empty effect set is a strong sign the action is
    a no-op in this embodiment (or its effect is invisible to the event
    stream, which is the same emergency wearing a different hat).
  * producers(effect)   — which actions can cause this? An effect the
    curriculum REQUIRES that has an empty producer list after real
    experience is the "unable" verdict, stated as a measurement.
  * unproduced(required, ...) — the standing alarm form of the above: the
    required effects that have NEVER occurred, reported only once enough
    steps have elapsed that absence is evidence rather than earliness.

Vocabulary. An event is a ``(kind, subtype)`` pair of strings — e.g. a
category of outcome and the specific thing it happened to. Queries accept
either the bare kind (``"acquire"`` = "did we ever acquire ANYTHING") or
the full ``"kind:subtype"`` key (``"acquire:tool_x"`` = "did we ever
acquire that SPECIFIC thing"). Both granularities matter: the bare kind
distinguishes "this verb works at all" from the full key's "this verb works
on the thing the goal needs".

Design constraints honoured throughout:
  * DOMAIN-AGNOSTIC — actions are opaque ints, events are opaque string
    pairs. The map knows nothing about any particular environment.
  * PURELY OBSERVATIONAL — it never selects actions, never shapes reward.
    It is an instrument. Instruments that also steer become the thing they
    were meant to measure (see the guard-becomes-latch pattern this project
    has now hit four times).
  * MONITORING MUST NEVER KILL THE RUN — malformed input (a non-int action
    from a half-migrated caller, a junk event tuple from a parser) degrades
    to "not counted", tallied in ``errors``/``last_error`` as data. A
    multi-day run must survive its own diagnostics.
  * BOUNDED MEMORY — an adversarial or buggy event source that mints
    unlimited distinct subtypes cannot grow the map without limit; past a
    generous cap, novel keys are dropped and tallied (``dropped``), because
    a diagnostic that OOMs the run it watches has failed its one job.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

# Caps chosen far above any sane vocabulary (a real embodiment has tens of
# actions and hundreds of event types) but low enough that a runaway
# generator of unique subtypes stays a few MB, not a leak that kills day 3.
_MAX_ACTIONS = 10_000
_MAX_EFFECT_KEYS = 50_000


class AffordanceMap:
    """Empirical action -> effect map with never-produced alarms.

    All public methods are safe to call at any time, including on a
    completely empty map, and never raise on malformed input.
    """

    def __init__(self) -> None:
        # Per-action bookkeeping. Plain dicts keyed by the opaque action int;
        # no assumption that actions are dense, small, or even non-negative.
        self._uses: Dict[int, int] = {}
        self._effects: Dict[int, Dict[str, int]] = {}  # action -> full-key -> n
        # Per-effect bookkeeping at BOTH granularities. "last step produced"
        # is tracked (not just a bool) so a future consumer can distinguish
        # "never" from "not lately" without a schema change — the never-vs-
        # not-lately line is exactly where regression bugs hide.
        self._kind_last: Dict[str, int] = {}
        self._kind_producers: Dict[str, Set[int]] = {}
        self._full_last: Dict[str, int] = {}
        self._full_producers: Dict[str, Set[int]] = {}
        # Fallback step for malformed `step` args: losing the effect credit
        # because the TIMESTAMP was junk would punish the wrong field.
        self._last_step: int = 0
        # Errors are data, never exceptions.
        self.errors: int = 0
        self.last_error: Optional[str] = None
        self.dropped: int = 0  # events discarded by the memory caps

    # ------------------------------------------------------------- ingest

    def observe(self, action: int, events: List[Tuple[str, str]],
                step: int) -> None:
        """Record one USE of `action` and credit each resulting event to it.

        `events` is the (possibly empty) list of ``(kind, subtype)`` pairs
        the environment attributed to this action at this step. An empty
        list is meaningful data — it is exactly the accumulating evidence
        behind actions_without_effect().
        """
        try:
            try:
                a = int(action)
            except (TypeError, ValueError):
                # Cannot attribute anything without an action identity; the
                # whole observation is unattributable, not just the events.
                self.errors += 1
                self.last_error = f"unusable action {action!r}"
                return
            try:
                s = int(step)
                self._last_step = s
            except (TypeError, ValueError):
                # Bad timestamp, good payload: keep the payload. The map's
                # verdicts ("ever produced", producer sets) do not depend on
                # WHEN; only recency does, and stale recency beats losing
                # the only evidence an action works.
                s = self._last_step
                self.errors += 1
                self.last_error = f"unusable step {step!r}"
            if a not in self._uses and len(self._uses) >= _MAX_ACTIONS:
                self.dropped += 1
                return
            self._uses[a] = self._uses.get(a, 0) + 1
            if events is None:
                return
            for ev in events:
                try:
                    kind, subtype = ev
                    key = f"{kind}:{subtype}"
                    kind = str(kind)
                    key = str(key)
                except (TypeError, ValueError):
                    self.errors += 1
                    self.last_error = f"unusable event {ev!r}"
                    continue
                if (key not in self._full_last
                        and len(self._full_last) >= _MAX_EFFECT_KEYS):
                    self.dropped += 1
                    continue
                per = self._effects.setdefault(a, {})
                per[key] = per.get(key, 0) + 1
                self._full_last[key] = s
                self._full_producers.setdefault(key, set()).add(a)
                self._kind_last[kind] = s
                self._kind_producers.setdefault(kind, set()).add(a)
        except Exception as e:  # pragma: no cover — belt and braces
            # No observation is worth the run. Anything not anticipated
            # above still becomes data instead of a crash.
            self.errors += 1
            self.last_error = f"observe failed: {e!r}"

    # ------------------------------------------------------------ queries

    def effects_of(self, action: int) -> Dict[str, int]:
        """Everything this action has ever caused: ``"kind:subtype" -> count``.

        Returns a copy — callers tabulating or mutating the result must not
        be able to corrupt the evidence base.
        """
        try:
            return dict(self._effects.get(int(action), {}))
        except (TypeError, ValueError):
            return {}

    def producers(self, effect: str) -> List[int]:
        """Actions that have EVER produced `effect`, sorted.

        Accepts a bare kind (``"open"``) or a full ``"kind:subtype"`` key.
        An empty list for a goal-required effect is the map's core verdict:
        no known body movement causes this — the goal is unreachable until
        the action space (or the event sensor) changes.
        """
        try:
            e = str(effect)
            table = self._full_producers if ":" in e else self._kind_producers
            return sorted(table.get(e, ()))
        except Exception:
            return []

    def actions_without_effect(self, min_uses: int = 500) -> List[int]:
        """Actions used at least `min_uses` times that never caused ANY event.

        The threshold exists because absence of effect is only evidence
        after real repetition — an action tried 3 times in a corner tells
        you nothing. 500 well-spread uses with zero events is the profile
        of a structural no-op (or an unsensed effect: equally alarming).
        """
        try:
            n = int(min_uses)
        except (TypeError, ValueError):
            n = 500
        return sorted(a for a, u in self._uses.items()
                      if u >= n and not self._effects.get(a))

    def unproduced(self, required: List[str], min_steps: int = 20000,
                   now: int = 0) -> List[str]:
        """Required effects (kind or kind:subtype) NEVER produced, once
        `now` >= `min_steps`.

        Silent before min_steps by design: early in a run "never happened
        yet" is the normal condition of everything, and an alarm that cries
        at step 100 trains its readers to ignore it by step 100k. After the
        threshold, absence IS the finding — formatted so the line can go
        straight into a health report and name the missing capability.
        """
        out: List[str] = []
        try:
            if now < min_steps:
                return out
            for eff in (required or []):
                try:
                    e = str(eff)
                except Exception:
                    continue
                seen = (e in self._full_last) if ":" in e \
                    else (e in self._kind_last)
                if not seen:
                    out.append(f"{e}: NEVER PRODUCED in {now} steps")
        except Exception as e:
            self.errors += 1
            self.last_error = f"unproduced failed: {e!r}"
        return out

    # ------------------------------------------------------------- report

    def report(self, top: int = 8) -> List[str]:
        """Human-readable snapshot: the `top` most-used actions with their
        effect profiles, plus a no-observed-effect line if any action has
        earned one.

        Effect entries are ordered by count (desc) so the line's head shows
        what the action mostly does; ties break alphabetically for stable
        output across runs (diffs of successive reports should show change,
        not dict-ordering noise).
        """
        lines: List[str] = []
        try:
            ranked = sorted(self._uses.items(),
                            key=lambda kv: (-kv[1], kv[0]))[:max(0, int(top))]
            for a, uses in ranked:
                eff = self._effects.get(a, {})
                inner = ", ".join(
                    f"{k} {v}" for k, v in
                    sorted(eff.items(), key=lambda kv: (-kv[1], kv[0])))
                lines.append(f"action {a}: {uses} uses -> {{{inner}}}")
            dead = self.actions_without_effect()
            if dead:
                lines.append(f"no observed effect: actions {dead}")
            if self.errors:
                # Surfacing our own ingestion trouble in the same channel:
                # a silently degraded instrument is worse than no instrument.
                lines.append(f"affordance-map ingest errors: {self.errors} "
                             f"(last: {self.last_error})")
            if self.dropped:
                lines.append(f"affordance-map dropped (memory cap): "
                             f"{self.dropped}")
        except Exception as e:
            self.errors += 1
            self.last_error = f"report failed: {e!r}"
            lines.append(f"report degraded: {e!r}")
        return lines
