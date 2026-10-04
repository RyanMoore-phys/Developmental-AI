"""UNKNOWN, ABSENT and INAPPLICABLE — three states that are not zero.

Plan §5: "Unknown, absent, zero, and inapplicable must be separate states."
The sensor bus already shows why: a sensor with no reading reports NEUTRAL,
and neutral is a value — the absence becomes a claim the instant it is
written as 0.0.

    UNKNOWN       a value exists but nobody has measured it (an unmeasured
                  tick length, a parameter not yet fitted)
    ABSENT        the source was asked and produced nothing this time (a
                  frame that failed to render, a client mid-rebuild)
    INAPPLICABLE  the question has no answer here (a spatial frame in a
                  nonspatial environment)

Each is a process-wide singleton: identity survives copy, deepcopy, pickle
and the JSON codec, so `x is UNKNOWN` is always the right test.

TRUTHINESS RAISES. `if reading:` on a sentinel is almost always a bug —
it silently collapses "unknown" into "false" or "zero", which is the exact
conflation these exist to prevent. Use `is_missing(x)` or an identity test.
"""

from __future__ import annotations

from typing import Any, Dict


class Sentinel:
    """One named missingness state. Do not instantiate; use the module
    constants."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        object.__setattr__(self, "name", name)

    def __setattr__(self, key, value):
        raise AttributeError("sentinels are immutable")

    def __repr__(self) -> str:
        return self.name

    def __bool__(self):
        raise TypeError(
            f"{self.name} has no truth value; test identity "
            f"(`x is {self.name}`) or use is_missing(x) instead")

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(("foundation.sentinel", self.name))

    def __reduce__(self):
        return (sentinel_by_name, (self.name,))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


UNKNOWN = Sentinel("UNKNOWN")
ABSENT = Sentinel("ABSENT")
INAPPLICABLE = Sentinel("INAPPLICABLE")

_BY_NAME: Dict[str, Sentinel] = {s.name: s for s in (UNKNOWN, ABSENT, INAPPLICABLE)}
SENTINELS = tuple(_BY_NAME.values())


def sentinel_by_name(name: str) -> Sentinel:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"no sentinel named {name!r}; "
                         f"known: {sorted(_BY_NAME)}") from None


def is_missing(x: Any) -> bool:
    """True for any of the three sentinels. NOT true for 0, None, NaN or an
    empty array — those are values, and saying otherwise is the bug."""
    return isinstance(x, Sentinel)
