"""Explicit error types for the typed-state / geometry library.

Every invalid operation raises one of these rather than returning a plausible
number. The hierarchy exists so callers can catch "any geometry misuse"
(`GeometryError`) or one specific misuse, and so tests can assert that the
RIGHT check fired rather than any exception at all.
"""


class GeometryError(Exception):
    """Base class for every typed-state / geometry misuse."""


class UnitError(GeometryError, TypeError):
    """Incompatible physical dimensions (e.g. metres + seconds)."""


class ScaleStatusError(GeometryError):
    """An up-to-scale or unknown-scale value was combined with, or used as,
    a metric value. Monocular estimates must not silently become metric."""


class FrameError(GeometryError):
    """Base for coordinate-frame misuse."""


class UnknownFrameError(FrameError, KeyError):
    """A frame name that was never registered."""


class DisconnectedFramesError(FrameError):
    """Two registered frames with no transform path between them."""


class FrameMismatchError(FrameError):
    """Values expressed in different frames were combined without conversion."""


class ClockMismatchError(GeometryError):
    """Timestamps from different clocks (env vs wall) or epochs were mixed."""


class DegenerateError(GeometryError, ValueError):
    """A degenerate input (zero-norm quaternion, zero axis with a nonzero
    angle, non-finite values) that has no well-defined meaning."""


class DomainError(GeometryError, ValueError):
    """A discrete value outside its declared domain, or a relation arity
    mismatch."""


class ConstraintError(GeometryError):
    """Misconfigured constraint (duplicate name, bad kind, bad tolerance)."""
