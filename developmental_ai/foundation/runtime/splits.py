"""Deterministic dev / held-out assignment (plan Stage 1 item 4, §7.2.2).

WHAT IT CLAIMS
    1. The split of a unit is a pure function of (salt, scope, unit id): the
       same unit lands on the same side on every machine, in every process,
       forever (no Python `hash()`, which is salted per process).
    2. The unit of assignment is the EPISODE (or scenario/world), never the
       frame. A frame's side is computed from its episode key alone, so two
       adjacent frames of one episode cannot straddle the split — the leak
       §7.2.2 forbids ("Adjacent frames from one episode must not leak across
       training and evaluation").
    3. `assert_no_leak` reduces any id (episode key OR frame key) to its
       episode key before comparing, so a frame-level leak is caught even
       when the two id lists never share an exact element.

WHAT AN "EPISODE" IS HERE
    A reset-bounded stretch of experience in one scope. For SkyBot's lifelong
    stream that is the span between resets (`reset_on_death_only: true`), NOT
    a 1024-step segment — consecutive segments of one stream ARE adjacent
    frames, and splitting by segment would leak exactly as frames do. Scope
    names where ids are valid (e.g. "minerl:paper-server:stream0"); ids do not
    persist across unrelated resets or environments (plan §5).

The salt is fixed and recorded in every run manifest (seeds.split_salt).
Changing it re-deals every unit — do that only with a new SPLIT_VERSION.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, NamedTuple, Sequence, Tuple

SPLIT_VERSION = 1
SPLIT_SALT = "skybot-foundation-split-v1"
DEV = "dev"
HELDOUT = "heldout"
DEFAULT_HELDOUT_FRACTION = 0.2


class SplitLeakError(AssertionError):
    """The same episode appears on both sides of a split."""


class EpisodeKey(NamedTuple):
    scope: str
    episode_id: str


def _check_scope(scope: Any) -> str:
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(f"split scope must be a non-empty string, got "
                         f"{scope!r}; an unscoped id is ambiguous across "
                         f"environments and resets")
    if "\x1f" in scope:
        raise ValueError("scope may not contain the separator \\x1f")
    return scope


def _check_id(eid: Any) -> str:
    if isinstance(eid, bool) or not isinstance(eid, (int, str)):
        raise TypeError(f"episode id must be int or str, got "
                        f"{type(eid).__name__}")
    s = str(eid)
    if not s or "\x1f" in s:
        raise ValueError(f"bad episode id {eid!r}")
    return s


def episode_key(scope: str, episode_id: Any) -> EpisodeKey:
    return EpisodeKey(_check_scope(scope), _check_id(episode_id))


def unit_fraction(scope: str, episode_id: Any, salt: str = SPLIT_SALT) -> float:
    """Uniform in [0, 1), a pure function of (salt, scope, id)."""
    k = episode_key(scope, episode_id)
    h = hashlib.sha256(
        f"{salt}\x1f{k.scope}\x1f{k.episode_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def assign_split(scope: str, episode_id: Any,
                 heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
                 salt: str = SPLIT_SALT) -> str:
    if not (0.0 <= float(heldout_fraction) <= 1.0):
        raise ValueError(f"heldout_fraction {heldout_fraction} outside [0,1]")
    return HELDOUT if unit_fraction(scope, episode_id, salt) < float(
        heldout_fraction) else DEV


def assign_frame(scope: str, episode_id: Any, t: int,
                 heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
                 salt: str = SPLIT_SALT) -> str:
    """A frame's side. `t` is validated and then deliberately IGNORED."""
    if isinstance(t, bool) or not isinstance(t, int) or t < 0:
        raise ValueError(f"frame index must be a non-negative int, got {t!r}")
    return assign_split(scope, episode_id, heldout_fraction, salt)


def _to_episode_key(x: Any) -> EpisodeKey:
    """Accept EpisodeKey, (scope, id) or (scope, id, t) frame keys."""
    if isinstance(x, EpisodeKey):
        return x
    if isinstance(x, (tuple, list)) and len(x) in (2, 3):
        return episode_key(x[0], x[1])
    raise TypeError(f"expected (scope, episode_id[, t]), got {x!r}")


def assert_no_leak(train_ids: Iterable[Any], eval_ids: Iterable[Any]) -> None:
    """Raise SplitLeakError if any episode contributes to both sides."""
    a = {_to_episode_key(x) for x in train_ids}
    b = {_to_episode_key(x) for x in eval_ids}
    both = sorted(a & b)
    if both:
        shown = ", ".join(f"{k.scope}/{k.episode_id}" for k in both[:5])
        raise SplitLeakError(
            f"{len(both)} episode(s) appear in BOTH train and eval "
            f"(e.g. {shown}); adjacent frames of one episode must stay on "
            f"one side (plan §7.2.2)")


def partition(ids: Sequence[Any],
              heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
              salt: str = SPLIT_SALT) -> Tuple[List[Any], List[Any]]:
    """Split episode or frame ids into (dev, heldout), by episode key."""
    dev, held = [], []
    for x in ids:
        k = _to_episode_key(x)
        (held if assign_split(k.scope, k.episode_id, heldout_fraction, salt)
         == HELDOUT else dev).append(x)
    assert_no_leak(dev, held)
    return dev, held


def split_spec(heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
               salt: str = SPLIT_SALT) -> Dict[str, Any]:
    """The record a manifest stores so a split can be re-derived."""
    return {"version": SPLIT_VERSION, "salt": salt,
            "heldout_fraction": float(heldout_fraction),
            "unit": "episode (reset-bounded), never frame or segment",
            "hash": "sha256(salt\\x1fscope\\x1fepisode_id)[:8] / 2**64"}
