"""Discrete variables, relations, timestamps and the sentinels they need.

Not everything the agent represents is a continuous vector in Euclidean
space (plan §2.2). A block type is a symbol from a finite domain; "on top of"
is a relation; "the moment the log dropped" is a time on a particular clock.
These types make those structures first-class AND make "we do not know"
expressible without collapsing it to a default.

SENTINELS. `UNKNOWN` and `INAPPLICABLE` are distinct singletons. Both refuse
`bool()` — `if diag.violated:` on an inapplicable constraint raises instead of
silently reading as "not violated", which is the exact confusion plan §2.3
warns about. (These are the `foundation.contracts.sentinels` singletons —
re-exported, so identity holds across packages.)

CLOCKS. Two timestamps are only comparable on the same clock AND the same
epoch. MineRL's env clock restarts every episode (CLAUDE.md §7: stats restart
at 0 on every reset), so an env time from episode 3 compared against one from
episode 4 is meaningless — the epoch field makes that an error, not a
negative duration.
"""

from __future__ import annotations

from typing import Any, FrozenSet, Hashable, Iterable, Optional, Sequence, Tuple

from .errors import ClockMismatchError, DomainError, UnitError


# Canonical process-wide sentinels (foundation.contracts), so `x is UNKNOWN`
# holds across packages. sentinels.py is dependency-free.
from ..contracts.sentinels import ABSENT, INAPPLICABLE, UNKNOWN, Sentinel  # noqa: E402


def is_unknown(x: Any) -> bool:
    return x is UNKNOWN


# --------------------------------------------------------------- discrete

class DiscreteVar:
    """A variable over a finite, declared domain.

    `value` is a domain member or `UNKNOWN`. When unknown, `support` is the
    set of values still possible (defaults to the whole domain), so partial
    knowledge ("it is a log, but which wood?") is expressible without
    inventing a probability.
    """

    __slots__ = ("name", "domain", "value", "support")

    def __init__(self, name: str, domain: Sequence[Hashable],
                 value: Any = UNKNOWN,
                 support: Optional[Iterable[Hashable]] = None):
        dom = tuple(domain)
        if not dom:
            raise DomainError(f"{name}: domain must be non-empty")
        if len(set(dom)) != len(dom):
            raise DomainError(f"{name}: domain has duplicate members")
        if any(isinstance(d, Sentinel) for d in dom):
            raise DomainError(
                f"{name}: sentinels cannot be domain members; UNKNOWN is a "
                f"state of knowledge, not a value of the variable")
        if value is not UNKNOWN and value not in dom:
            raise DomainError(f"{name}: {value!r} not in domain {dom}")
        if value is UNKNOWN:
            sup = frozenset(dom) if support is None else frozenset(support)
            if not sup:
                raise DomainError(f"{name}: empty support is a contradiction")
            if not sup <= frozenset(dom):
                raise DomainError(f"{name}: support {set(sup)} outside domain")
        else:
            if support is not None and frozenset(support) != {value}:
                raise DomainError(f"{name}: support disagrees with value")
            sup = frozenset([value])
        self.name = name
        self.domain: Tuple[Hashable, ...] = dom
        self.value = value
        self.support: FrozenSet[Hashable] = sup

    @property
    def is_known(self) -> bool:
        return self.value is not UNKNOWN

    def with_value(self, value: Any) -> "DiscreteVar":
        return DiscreteVar(self.name, self.domain, value)

    def restrict(self, allowed: Iterable[Hashable]) -> "DiscreteVar":
        """Narrow the support; collapses to a known value when one remains."""
        sup = self.support & frozenset(allowed)
        if not sup:
            raise DomainError(
                f"{self.name}: restriction to {set(allowed)} leaves no "
                f"possible value (support was {set(self.support)})")
        if len(sup) == 1:
            return DiscreteVar(self.name, self.domain, next(iter(sup)))
        return DiscreteVar(self.name, self.domain, UNKNOWN, sup)

    def __repr__(self) -> str:
        if self.is_known:
            return f"DiscreteVar({self.name}={self.value!r})"
        return f"DiscreteVar({self.name}=UNKNOWN, support={sorted(map(repr, self.support))})"


# --------------------------------------------------------------- relations

_TRUTH = (True, False)


class RelationType:
    """A named relation of fixed arity, optionally with per-argument types."""

    __slots__ = ("name", "arity", "arg_types")

    def __init__(self, name: str, arity: int,
                 arg_types: Optional[Sequence[type]] = None):
        if int(arity) < 1:
            raise DomainError(f"relation {name}: arity must be >= 1")
        if arg_types is not None and len(arg_types) != int(arity):
            raise DomainError(
                f"relation {name}: {len(arg_types)} arg types for arity {arity}")
        self.name = name
        self.arity = int(arity)
        self.arg_types = tuple(arg_types) if arg_types is not None else None

    def __call__(self, *args, truth: Any = True,
                 evidence: Sequence[str] = ()) -> "Relation":
        return Relation(self, args, truth=truth, evidence=evidence)

    def __repr__(self) -> str:
        return f"RelationType({self.name}/{self.arity})"


class Relation:
    """One relation instance. `truth` is True, False or UNKNOWN."""

    __slots__ = ("type", "args", "truth", "evidence")

    def __init__(self, rtype: RelationType, args: Sequence[Hashable],
                 truth: Any = True, evidence: Sequence[str] = ()):
        args = tuple(args)
        if len(args) != rtype.arity:
            raise DomainError(
                f"relation {rtype.name} has arity {rtype.arity}, got "
                f"{len(args)} args {args}")
        if rtype.arg_types is not None:
            for i, (a, t) in enumerate(zip(args, rtype.arg_types)):
                if not isinstance(a, t):
                    raise DomainError(
                        f"relation {rtype.name} arg {i}: expected "
                        f"{t.__name__}, got {type(a).__name__}")
        if not (truth is UNKNOWN or (isinstance(truth, bool) and truth in _TRUTH)):
            raise DomainError(
                f"relation truth must be True, False or UNKNOWN, got {truth!r}")
        self.type = rtype
        self.args = args
        self.truth = truth
        self.evidence = tuple(evidence)

    @property
    def key(self) -> Tuple[str, Tuple[Hashable, ...]]:
        return (self.type.name, self.args)

    def __repr__(self) -> str:
        return f"{self.type.name}{self.args}={self.truth!r}"


# --------------------------------------------------------------- time

class Timestamp:
    """A time on a NAMED clock, optionally within an epoch (e.g. an episode).

    `clock` is e.g. "env" (simulation ticks/seconds, restarts per episode)
    or "wall" (host monotonic seconds). `unit` is a time Unit (seconds by
    default; an adapter may declare "tick"). Differences come back as a
    time Quantity whose FRAME is the clock, so an env duration cannot be
    added to a wall duration either.
    """

    __slots__ = ("value", "clock", "epoch", "unit")

    def __init__(self, value: float, clock: str, epoch: Optional[Hashable] = None,
                 unit=None):
        from .units import S, TIME
        u = S if unit is None else unit
        if u.dims != TIME:
            raise UnitError(f"timestamp unit must be a time unit, got {u}")
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            raise DomainError(f"timestamp value must be finite, got {value}")
        if not clock or not isinstance(clock, str):
            raise ClockMismatchError("timestamp requires a named clock")
        self.value = v
        self.clock = clock
        self.epoch = epoch
        self.unit = u

    @property
    def time_frame(self) -> str:
        return f"clock:{self.clock}" + ("" if self.epoch is None else f"/{self.epoch}")

    def _check(self, other: "Timestamp") -> None:
        if not isinstance(other, Timestamp):
            raise TypeError(f"expected Timestamp, got {type(other).__name__}")
        if other.clock != self.clock:
            raise ClockMismatchError(
                f"cannot mix clock {self.clock!r} with {other.clock!r}; env "
                f"time and wall time drift apart whenever the server lags")
        if other.epoch != self.epoch:
            raise ClockMismatchError(
                f"cannot mix epochs {self.epoch!r} and {other.epoch!r} of clock "
                f"{self.clock!r}; the env clock restarts every episode")

    def _si(self) -> float:
        return self.value * self.unit.factor

    def __sub__(self, other: "Timestamp"):
        from .units import Quantity, S
        if not isinstance(other, Timestamp):
            raise TypeError("subtract a Timestamp to get a duration; to move a "
                            "timestamp use `ts + duration`")
        self._check(other)
        return Quantity((self._si() - other._si()) / self.unit.factor,
                        self.unit, frame=self.time_frame)

    def duration(self, dt: float, unit=None):
        """A duration Quantity tagged with this timestamp's clock frame."""
        from .units import Quantity
        return Quantity(float(dt), self.unit if unit is None else unit,
                        frame=self.time_frame)

    def __add__(self, dur):
        from .units import Quantity, TIME
        from .errors import FrameMismatchError
        if not isinstance(dur, Quantity):
            raise TypeError("add a duration Quantity (use ts.duration(dt))")
        if dur.unit.dims != TIME:
            raise UnitError(f"cannot add {dur.unit} to a timestamp")
        if dur.frame != self.time_frame:
            raise FrameMismatchError(
                f"duration measured on {dur.frame!r} cannot move a timestamp "
                f"on {self.time_frame!r}")
        if dur.value.shape != ():
            raise DomainError("timestamp offset must be a scalar duration")
        return Timestamp(self.value + float(dur.value) * dur.unit.factor
                         / self.unit.factor, self.clock, self.epoch, self.unit)

    def __lt__(self, other):
        self._check(other)
        return self._si() < other._si()

    def __le__(self, other):
        self._check(other)
        return self._si() <= other._si()

    def __gt__(self, other):
        self._check(other)
        return self._si() > other._si()

    def __ge__(self, other):
        self._check(other)
        return self._si() >= other._si()

    def __eq__(self, other):
        if not isinstance(other, Timestamp):
            return NotImplemented
        self._check(other)
        return self._si() == other._si()

    def __hash__(self):
        return hash((self.clock, self.epoch, self._si()))

    def __repr__(self) -> str:
        return f"Timestamp({self.value} {self.unit.name} @ {self.time_frame})"
