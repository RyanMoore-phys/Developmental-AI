"""Versioned serialization for every record.

WIRE FORMAT
    {"schema": "<SCHEMA>", "version": <int>, "fields": {<name>: <value>}}

    Values use plain JSON where JSON is exact, and a one-key tagged object
    where it is not:

        {"__nd__": {"dtype": "<f4", "shape": [..], "b64": ".."}}   ndarray
        {"__np__": {"dtype": "<i8", "b64": ".."}}                   numpy scalar
        {"__tuple__": [..]}                                         tuple
        {"__sentinel__": "UNKNOWN"}                                 sentinel
        {"__float__": "nan" | "inf" | "-inf"}                       non-finite
        {"__id__": "Scope" | "EntityId" | "FrameId" | "ObsRef", ..} scoped ID
        {"__record__": {<nested wire record>}}                      record

    Arrays travel as raw little/big-endian bytes with their exact dtype
    string and shape, so a float32 sensor reading comes back bit-identical,
    NaNs included. Record payload keys may not start with "__" (enforced at
    construction), so a tag can never collide with user data.

VERSIONS (plan §6 Stage 2 item 5)
    A record whose version equals the class VERSION decodes directly. An
    OLDER version decodes only through registered migrations, one version
    step at a time (register_migration(schema, v, v+1, fn)); a missing step
    raises SchemaVersionError naming it. A NEWER version always raises
    SchemaVersionError: guessing at fields written by future code is how a
    restored brain silently starts empty (CLAUDE.md §6, the podlogs rename).
    Decoding never writes anything back; converting stored data is an
    explicit tool's job, into a new location (plan §8).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import base as _base
from .errors import RecordValidationError, SchemaVersionError
from .ids import EntityId, FrameId, ObsRef, Scope
from .sentinels import Sentinel, sentinel_by_name

_ID_CLASSES = {c.__name__: c for c in (Scope, EntityId, FrameId, ObsRef)}
_TAGS = ("__nd__", "__np__", "__tuple__", "__sentinel__", "__float__",
         "__id__", "__record__")


# --------------------------------------------------------------- migrations

class MigrationRegistry:
    """schema -> {from_version: fn(fields) -> fields} for single steps.

    `fn` receives the DECODED field dict of a version-`from_v` record and
    returns the field dict of version `from_v + 1`. It must not mutate its
    input."""

    def __init__(self):
        self._steps: Dict[str, Dict[int, Callable]] = {}

    def register(self, schema: str, from_v: int, to_v: int, fn: Callable) -> None:
        if not isinstance(from_v, int) or not isinstance(to_v, int):
            raise TypeError("migration versions must be ints")
        if to_v != from_v + 1:
            raise ValueError(f"migrations are single steps; got {from_v}->{to_v}")
        steps = self._steps.setdefault(schema, {})
        if from_v in steps:
            raise ValueError(f"migration {schema} v{from_v}->v{to_v} "
                             f"already registered")
        steps[from_v] = fn

    def migrate(self, schema: str, version: int, target: int,
                fields: Dict[str, Any]) -> Dict[str, Any]:
        steps = self._steps.get(schema, {})
        v = version
        while v < target:
            fn = steps.get(v)
            if fn is None:
                raise SchemaVersionError(
                    f"{schema} v{version} is older than the supported "
                    f"v{target} and no migration v{v}->v{v + 1} is "
                    f"registered")
            fields = fn(dict(fields))
            if not isinstance(fields, dict):
                raise SchemaVersionError(
                    f"migration {schema} v{v}->v{v + 1} returned "
                    f"{type(fields).__name__}, not a dict")
            v += 1
        return fields


MIGRATIONS = MigrationRegistry()


def register_migration(schema: str, from_v: int, to_v: int, fn: Callable) -> None:
    MIGRATIONS.register(schema, from_v, to_v, fn)


# ------------------------------------------------------------------ values

def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def encode_value(v):
    # numpy first: np.float64 subclasses float and would lose its type.
    if isinstance(v, np.ndarray):
        return {"__nd__": {"dtype": v.dtype.str, "shape": list(v.shape),
                           "b64": _b64(v)}}
    if isinstance(v, np.generic):
        return {"__np__": {"dtype": v.dtype.str, "b64": _b64(np.asarray(v))}}
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, Sentinel):
        return {"__sentinel__": v.name}
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else {"__float__": repr(v)}
    if isinstance(v, tuple):
        return {"__tuple__": [encode_value(e) for e in v]}
    if isinstance(v, list):
        return [encode_value(e) for e in v]
    if isinstance(v, dict):
        out = {}
        for k, e in v.items():
            if not isinstance(k, str) or k.startswith("__"):
                raise RecordValidationError(f"cannot encode dict key {k!r}")
            out[k] = encode_value(e)
        return out
    if isinstance(v, (Scope, EntityId, FrameId, ObsRef)):
        d = {"__id__": type(v).__name__}
        for f in dataclasses.fields(v):
            d[f.name] = encode_value(getattr(v, f.name))
        return d
    if isinstance(v, _base.Record):
        return {"__record__": to_dict(v)}
    raise RecordValidationError(f"cannot encode value of type {type(v).__name__}")


def _array_from(spec: dict, scalar: bool):
    try:
        dt = np.dtype(spec["dtype"])
        raw = base64.b64decode(spec["b64"].encode("ascii"), validate=True)
        shape = () if scalar else tuple(int(x) for x in spec["shape"])
    except (KeyError, TypeError, ValueError) as e:
        raise RecordValidationError(f"malformed array encoding: {e}") from None
    if dt.kind not in "biufc":
        raise RecordValidationError(f"array dtype {dt} not allowed")
    n = int(np.prod(shape)) if shape else 1
    if len(raw) != n * dt.itemsize:
        raise RecordValidationError(
            f"array payload has {len(raw)} bytes, expected {n * dt.itemsize} "
            f"for {dt} {shape}")
    a = np.frombuffer(raw, dtype=dt).reshape(shape).copy()
    return a[()] if scalar else a


def decode_value(v, registry: Optional[MigrationRegistry] = None):
    if isinstance(v, list):
        return [decode_value(e, registry) for e in v]
    if not isinstance(v, dict):
        return v
    tags = [k for k in v if k in _TAGS]
    if not tags:
        return {k: decode_value(e, registry) for k, e in v.items()}
    tag = tags[0]
    body = v[tag]
    if tag != "__id__" and len(v) != 1:
        raise RecordValidationError(f"tagged value {tag} has extra keys {sorted(v)}")
    if tag == "__nd__":
        return _array_from(body, scalar=False)
    if tag == "__np__":
        return _array_from(body, scalar=True)
    if tag == "__tuple__":
        if not isinstance(body, list):
            raise RecordValidationError("__tuple__ body must be a list")
        return tuple(decode_value(e, registry) for e in body)
    if tag == "__sentinel__":
        try:
            return sentinel_by_name(body)
        except (ValueError, TypeError) as e:
            raise RecordValidationError(str(e)) from None
    if tag == "__float__":
        if body not in ("nan", "inf", "-inf"):
            raise RecordValidationError(f"bad __float__ {body!r}")
        return float(body)
    if tag == "__id__":
        cls = _ID_CLASSES.get(body)
        if cls is None:
            raise RecordValidationError(f"unknown id type {body!r}")
        names = [f.name for f in dataclasses.fields(cls)]
        got = set(v) - {"__id__"}
        if got != set(names):
            raise RecordValidationError(
                f"{body} expects fields {names}, got {sorted(got)}")
        return cls(**{n: decode_value(v[n], registry) for n in names})
    return from_dict(body, registry=registry)


# ----------------------------------------------------------------- records

def to_dict(record) -> Dict[str, Any]:
    if not isinstance(record, _base.Record):
        raise RecordValidationError(f"not a record: {type(record).__name__}")
    return {"schema": record.SCHEMA, "version": record.VERSION,
            "fields": {f.name: encode_value(getattr(record, f.name))
                       for f in dataclasses.fields(record)}}


def from_dict(d, registry: Optional[MigrationRegistry] = None):
    """Decode a wire dict. `registry` defaults to the global MIGRATIONS."""
    registry = MIGRATIONS if registry is None else registry
    if not isinstance(d, dict):
        raise RecordValidationError(f"record must be a dict, got {type(d).__name__}")
    missing = [k for k in ("schema", "version", "fields") if k not in d]
    if missing:
        raise RecordValidationError(f"record envelope missing {missing}")
    extra = set(d) - {"schema", "version", "fields"}
    if extra:
        raise RecordValidationError(f"record envelope has unknown keys {sorted(extra)}")
    schema, version, raw = d["schema"], d["version"], d["fields"]
    if not isinstance(schema, str):
        raise RecordValidationError("schema must be a str")
    if isinstance(version, bool) or not isinstance(version, int):
        raise SchemaVersionError(f"{schema}: version must be an int, got {version!r}")
    if not isinstance(raw, dict):
        raise RecordValidationError(f"{schema}: fields must be a dict")
    cls = _base.record_class(schema)
    if version > cls.VERSION:
        raise SchemaVersionError(
            f"{schema} v{version} was written by newer code; this code reads "
            f"up to v{cls.VERSION}. Refusing to guess at its fields.")
    fields = {k: decode_value(e, registry) for k, e in raw.items()}
    if version < cls.VERSION:
        fields = registry.migrate(schema, version, cls.VERSION, fields)
    names = [f.name for f in dataclasses.fields(cls)]
    miss = [n for n in names if n not in fields]
    unknown = sorted(set(fields) - set(names))
    if miss or unknown:
        raise RecordValidationError(
            f"{schema} v{cls.VERSION}: missing fields {miss}, unknown fields "
            f"{unknown}")
    try:
        return cls(**{n: fields[n] for n in names})
    except RecordValidationError:
        raise
    except (TypeError, ValueError) as e:
        raise RecordValidationError(f"{schema}: {e}") from None


def to_json(record, **kw) -> str:
    kw.setdefault("sort_keys", True)
    return json.dumps(to_dict(record), allow_nan=False, **kw)


def from_json(text: str, registry: Optional[MigrationRegistry] = None):
    try:
        d = json.loads(text)
    except (TypeError, ValueError) as e:
        raise RecordValidationError(f"not valid JSON: {e}") from None
    return from_dict(d, registry=registry)
