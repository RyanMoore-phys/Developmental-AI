"""Field-level validators and the value grammar records may carry.

A record value is one of: None, bool, int, float, str, a Sentinel, a numpy
array of a numeric/bool dtype (or a numpy scalar), a tuple or list of
values, a dict with str keys of values, a scoped ID, or another Record.
Anything else is rejected AT CONSTRUCTION, so a record that exists is a
record the codec can write. Object arrays are refused: they would pickle
arbitrary code into what is meant to be data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

from .errors import RecordValidationError
from .ids import EntityId, FrameId, ObsRef, Scope
from .sentinels import Sentinel

PROVENANCES = ("sensor", "inferred", "imagined", "evaluator")
ID_TYPES = (Scope, EntityId, FrameId, ObsRef)
_ARRAY_KINDS = "biufc"          # bool, int, uint, float, complex


def fail(owner: str, msg: str):
    raise RecordValidationError(f"{owner}: {msg}")


def req_str(owner: str, name: str, v, allow_empty: bool = False) -> str:
    if not isinstance(v, str):
        fail(owner, f"{name} must be str, got {type(v).__name__}")
    if not v and not allow_empty:
        fail(owner, f"{name} must be non-empty")
    return v


def req_int(owner: str, name: str, v, minimum: int = None) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        fail(owner, f"{name} must be int, got {type(v).__name__}")
    v = int(v)
    if minimum is not None and v < minimum:
        fail(owner, f"{name} must be >= {minimum}, got {v}")
    return v


def req_bool(owner: str, name: str, v) -> bool:
    if not isinstance(v, (bool, np.bool_)):
        fail(owner, f"{name} must be bool, got {type(v).__name__}")
    return bool(v)


def req_finite(owner: str, name: str, v, allow: Tuple = ()) -> Any:
    """A finite real number, or one of the sentinels in `allow`."""
    if isinstance(v, Sentinel):
        if v in allow:
            return v
        fail(owner, f"{name} may not be {v!r} (allowed: {allow or 'none'})")
    if isinstance(v, (bool, np.bool_)) or not isinstance(
            v, (int, float, np.integer, np.floating)):
        fail(owner, f"{name} must be a real number, got {type(v).__name__}")
    v = float(v)
    if not math.isfinite(v):
        fail(owner, f"{name} must be finite, got {v}")
    return v


def req_enum(owner: str, name: str, v, choices: Iterable[str]) -> str:
    choices = tuple(choices)
    if v not in choices:
        fail(owner, f"{name} must be one of {choices}, got {v!r}")
    return v


def req_mapping(owner: str, name: str, v) -> Dict[str, Any]:
    if not isinstance(v, Mapping):
        fail(owner, f"{name} must be a mapping, got {type(v).__name__}")
    out = dict(v)
    for k, val in out.items():
        if not isinstance(k, str) or not k:
            fail(owner, f"{name} keys must be non-empty str, got {k!r}")
        if k.startswith("__"):
            fail(owner, f"{name} key {k!r} is reserved for the codec")
        check_value(owner, f"{name}[{k!r}]", val)
    return out


def req_tuple(owner: str, name: str, v, elem=None, nonempty=False) -> tuple:
    if isinstance(v, (str, bytes)) or not isinstance(v, (tuple, list)):
        fail(owner, f"{name} must be a tuple/list, got {type(v).__name__}")
    out = tuple(v)
    if nonempty and not out:
        fail(owner, f"{name} must be non-empty")
    for i, e in enumerate(out):
        if elem is not None and not isinstance(e, elem):
            fail(owner, f"{name}[{i}] must be {getattr(elem, '__name__', elem)}, "
                        f"got {type(e).__name__}")
        check_value(owner, f"{name}[{i}]", e)
    return out


def req_str_tuple(owner: str, name: str, v, nonempty=False, unique=False):
    out = req_tuple(owner, name, v, elem=str, nonempty=nonempty)
    if any(not s for s in out):
        fail(owner, f"{name} entries must be non-empty")
    if unique and len(set(out)) != len(out):
        fail(owner, f"{name} entries must be unique, got {out}")
    return out


def check_value(owner: str, path: str, v) -> None:
    """Raise unless `v` is inside the record value grammar."""
    from .base import Record                     # local: base imports us
    if v is None or isinstance(v, (bool, str, Sentinel) + ID_TYPES):
        return
    if isinstance(v, (int, float)):
        return
    if isinstance(v, (np.ndarray, np.generic)):
        if np.asarray(v).dtype.kind not in _ARRAY_KINDS:
            fail(owner, f"{path}: numpy dtype {np.asarray(v).dtype} is not "
                        f"allowed (numeric/bool only)")
        return
    if isinstance(v, (tuple, list)):
        for i, e in enumerate(v):
            check_value(owner, f"{path}[{i}]", e)
        return
    if isinstance(v, Mapping):
        for k, e in v.items():
            if not isinstance(k, str):
                fail(owner, f"{path}: dict keys must be str, got {k!r}")
            if k.startswith("__"):
                fail(owner, f"{path}: key {k!r} is reserved for the codec")
            check_value(owner, f"{path}[{k!r}]", e)
        return
    if isinstance(v, Record):
        return
    fail(owner, f"{path}: unsupported value type {type(v).__name__}")


def deep_equal(a, b) -> bool:
    """Structural equality that understands arrays (dtype + shape + values,
    NaN == NaN), sentinels (identity) and tuple-vs-list (different)."""
    if isinstance(a, (np.ndarray, np.generic)) or isinstance(b, (np.ndarray, np.generic)):
        if type(a) is not type(b):
            return False
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, Sentinel) or isinstance(b, Sentinel):
        return a is b
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b
