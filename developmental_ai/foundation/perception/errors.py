"""Errors raised by foundation.perception.

All subclass ContractError (and so ValueError): a perception failure is a
contract failure — a belief built on the wrong representation version, a
privileged input reaching the tracker, a scale claim nobody earned.
"""

from ..contracts.errors import ContractError


class PerceptionError(ContractError):
    """Base class for every foundation.perception violation."""


class StaleDependentError(PerceptionError):
    """A cache/mechanism/skill bound to a representation version that has
    since changed semantics was used without being translated or rebuilt."""


class PrivilegedInputError(PerceptionError):
    """Evaluator (or imagined) data was offered to a learning/inference path,
    or a non-evaluator input was offered to the oracle diagnostics."""


class UndeclaredFamilyError(PerceptionError):
    """A state family was used without an explicit declaration."""
