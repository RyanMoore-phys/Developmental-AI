"""Errors raised by the contract layer.

All subclass ValueError so existing `except ValueError` call sites keep
working, and all subclass ContractError so a caller can catch exactly the
contract failures and nothing else.
"""


class ContractError(ValueError):
    """Base class for every contract violation in foundation.contracts."""


class RecordValidationError(ContractError):
    """A record was built (or decoded) with a missing, malformed, or
    inconsistent field. Raised at construction, never later."""


class SchemaVersionError(ContractError):
    """A serialized record carries a schema name or version this code cannot
    read and no registered migration reaches the current version. Raised
    instead of guessing: a silently half-read record is worse than none."""


class AdapterConformanceError(ContractError):
    """An EnvironmentAdapter broke the adapter contract. `problems` lists
    every violation found, not just the first."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("adapter contract violated:\n  - "
                         + "\n  - ".join(self.problems))
