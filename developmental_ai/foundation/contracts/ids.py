"""Scoped identifiers.

Plan §5: "Entity IDs and frame IDs need explicit scope; do not assume
identities persist across unrelated resets or environments."

This fork of MineRL makes a fresh world on every reset (CLAUDE.md §7), so
"entity 7" in episode 3 and "entity 7" in episode 4 are unrelated things that
happen to share a number. The ID types here therefore carry their scope, and
equality compares it: the same local name under two scopes is NOT equal, and
there is no implicit cross-scope lookup. Anything that legitimately persists
across resets must earn that through evidence, not through an ID collision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import RecordValidationError


def _nonempty_str(owner: str, field: str, v) -> None:
    if not isinstance(v, str) or not v:
        raise RecordValidationError(
            f"{owner}.{field} must be a non-empty str, got {v!r}")


@dataclass(frozen=True)
class Scope:
    """Where an identity is valid: one episode of one stream of one
    environment."""

    environment: str
    stream: str
    episode: str

    def __post_init__(self):
        for f in ("environment", "stream", "episode"):
            _nonempty_str("Scope", f, getattr(self, f))


@dataclass(frozen=True)
class EntityId:
    scope: Scope
    local: str

    def __post_init__(self):
        if not isinstance(self.scope, Scope):
            raise RecordValidationError(
                f"EntityId.scope must be a Scope, got {type(self.scope).__name__}")
        _nonempty_str("EntityId", "local", self.local)


@dataclass(frozen=True)
class FrameId:
    scope: Scope
    local: str

    def __post_init__(self):
        if not isinstance(self.scope, Scope):
            raise RecordValidationError(
                f"FrameId.scope must be a Scope, got {type(self.scope).__name__}")
        _nonempty_str("FrameId", "local", self.local)


@dataclass(frozen=True)
class ObsRef:
    """A pointer to one Observation: (scope, seq, channel) is its key."""

    scope: Scope
    seq: int
    channel: str

    def __post_init__(self):
        if not isinstance(self.scope, Scope):
            raise RecordValidationError("ObsRef.scope must be a Scope")
        if (not isinstance(self.seq, int) or isinstance(self.seq, bool)
                or self.seq < 0):
            raise RecordValidationError(
                f"ObsRef.seq must be an int >= 0, got {self.seq!r}")
        _nonempty_str("ObsRef", "channel", self.channel)


def same_scope(a, b) -> bool:
    """True iff two scoped things (IDs, Scopes, Observations) share a scope.
    Anything without a scope is never in the same scope as anything."""
    sa = a if isinstance(a, Scope) else getattr(a, "scope", None)
    sb = b if isinstance(b, Scope) else getattr(b, "scope", None)
    return isinstance(sa, Scope) and sa == sb
