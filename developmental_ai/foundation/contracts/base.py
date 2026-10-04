"""The Record base class and the schema registry.

Every record is a frozen dataclass with two class constants, SCHEMA (a
stable name) and VERSION (an int bumped on any incompatible field change),
and validates itself in __post_init__. Common fields are typed and checked;
anything domain-specific (Minecraft block names, button labels, ...) goes in
the record's `payload` dict, which the contract layer carries but never
interprets. That split is plan §6 Stage 2 item 3.

Equality is structural and array-aware (see values.deep_equal), so records
holding numpy arrays compare sanely. Records are unhashable: their payloads
are dicts.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Dict, Type

from .errors import SchemaVersionError
from .values import deep_equal

RECORD_TYPES: Dict[str, Type["Record"]] = {}


def register_record(cls):
    """Class decorator: add a Record subclass to the schema registry."""
    schema = getattr(cls, "SCHEMA", None)
    version = getattr(cls, "VERSION", None)
    if not isinstance(schema, str) or not schema:
        raise TypeError(f"{cls.__name__} needs a SCHEMA name")
    if not isinstance(version, int) or version < 1:
        raise TypeError(f"{cls.__name__} needs an integer VERSION >= 1")
    if schema in RECORD_TYPES and RECORD_TYPES[schema] is not cls:
        raise TypeError(f"schema {schema!r} already registered by "
                        f"{RECORD_TYPES[schema].__name__}")
    RECORD_TYPES[schema] = cls
    return cls


class Record:
    SCHEMA: ClassVar[str] = ""
    VERSION: ClassVar[int] = 0

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:                  # pragma: no cover
        raise NotImplementedError

    def _set(self, name: str, value) -> None:
        """Normalise a field during validation (frozen dataclass escape)."""
        object.__setattr__(self, name, value)

    def field_names(self):
        return tuple(f.name for f in dataclasses.fields(self))

    def replace(self, **changes):
        """A validated copy with some fields changed."""
        return dataclasses.replace(self, **changes)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return all(deep_equal(getattr(self, n), getattr(other, n))
                   for n in self.field_names())

    __hash__ = None


def record_class(schema: str) -> Type[Record]:
    try:
        return RECORD_TYPES[schema]
    except KeyError:
        raise SchemaVersionError(
            f"unknown record schema {schema!r}; known: "
            f"{sorted(RECORD_TYPES)}") from None
