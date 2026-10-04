"""Errors raised by the evidence store.

All subclass ExperienceError (a ContractError, hence a ValueError), so a
caller can catch exactly the store's refusals. Each names the rule it
enforces; none is raised for a condition the caller cannot correct.
"""

from ..contracts.errors import ContractError


class ExperienceError(ContractError):
    """Base class for every evidence-store refusal."""


class ProvenanceError(ExperienceError):
    """A record's provenance forbids what was asked: an imagined
    observation or imagined evidence filed as real (plan §7.4, "a model
    cannot certify its own imagined outcomes as real evidence"), or a
    prediction conditioned on evaluator data."""


class OrderingError(ExperienceError):
    """Write order broke a causal rule: an outcome linked to a prediction
    that was written after the outcome's observations arrived, an action
    logged before the observation it answers, a sequence running backwards,
    or a record written into an episode that has already ended."""


class DuplicateRecordError(ExperienceError):
    """Raw records are immutable: the same key cannot be written twice."""


class PrivacyError(ExperienceError):
    """An evaluator-partition record was requested through a path that is
    not the evaluator view."""


class RecordEvicted(ExperienceError, LookupError):
    """The record existed but its chunk was evicted by the retention rule.
    Distinct from never-existed (KeyError)."""


class StoreBudgetError(ExperienceError):
    """The disk budget cannot be met without deleting pinned evidence, or a
    pin would push pinned data past its declared share of the budget. The
    escape is always named in the message (unpin, or raise max_bytes)."""


class StoreCorruption(ExperienceError):
    """A record failed its content hash on READ (it passed on open, so the
    bytes changed underneath a running store)."""
