"""Immutable, versioned model snapshots (plan Stage 3 item 4, §5).

WHAT IT CLAIMS
    1. `publish` deep-copies a module's (or a state dict's) tensors into a
       snapshot that NOTHING can mutate afterwards — not the trainer stepping
       the live module, not a later publish, not a reader. Tensors are stored
       as read-only numpy arrays (writeable=False); consumers get either those
       read-only views or fresh copies (`to_state_dict`, `load_into`).
    2. Snapshot ids increase strictly monotonically across the registry, and
       each snapshot carries a content hash over (key, dtype, shape, bytes),
       so a prediction that records `provenance()` names exactly the weights
       that produced it, and `verify()` proves they are still those weights.
    3. Publication is atomic with respect to readers: `current`/`pin` return
       a complete snapshot, never a half-built one, and a publish while a
       reader holds a snapshot leaves the held one bit-identical.

WHY (repository incidents this mirrors)
    `_save_checkpoint` takes `_wm_param_lock` because a state_dict() walk
    mid-optimizer-step persisted TORN weights (mixed pre/post-update tensors).
    The async WM trainer writes the very module the policy reads latents
    from. `publish(..., lock=...)` takes the caller's lock for the copy, so a
    background learner publishes at an explicit boundary and inference reads
    a fixed version — the plan's "Background learners publish model
    snapshots at explicit boundaries. Planners record which snapshots
    produced their predictions."

    bfloat16 has no numpy dtype; such tensors are stored as float32 with the
    original dtype recorded and restored by `to_state_dict`. The hash covers
    the stored bytes plus the recorded dtype.

Shadow-only: nothing in the live loop publishes or reads these yet.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import threading
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 1


class SnapshotCorruption(RuntimeError):
    """A snapshot's content no longer matches its recorded hash."""


def _freeze_value(v: Any) -> Tuple[Any, str]:
    """-> (frozen value, dtype tag)."""
    try:
        import torch
    except ImportError:                                # pragma: no cover
        torch = None
    if torch is not None and isinstance(v, torch.Tensor):
        t = v.detach().to("cpu")
        tag = str(t.dtype).replace("torch.", "")
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        a = np.array(t.numpy(), copy=True)
    elif isinstance(v, np.ndarray):
        a = np.array(v, copy=True)
        tag = str(a.dtype)
    elif isinstance(v, (bool, int, float, str)) or v is None:
        return v, type(v).__name__
    else:
        raise TypeError(f"cannot snapshot value of type {type(v).__name__}; "
                        f"only tensors, arrays and scalars")
    if a.dtype == object:
        raise TypeError("object arrays cannot be snapshotted")
    # Read-only VIEW of a read-only base: a holder cannot simply flip the
    # flag back (numpy refuses WRITEABLE=True on a view of a read-only
    # base). Deliberate circumvention via `.base` is what verify() catches.
    a.setflags(write=False)
    v = a.view()
    v.setflags(write=False)
    return v, tag


def _content_hash(items: Mapping[str, Any], tags: Mapping[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(items):
        v = items[k]
        h.update(k.encode("utf-8") + b"\0" + tags[k].encode() + b"\0")
        if isinstance(v, np.ndarray):
            h.update(str(v.shape).encode() + b"\0")
            h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(repr(v).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class Snapshot:
    """One immutable published version. Construct only via the registry."""

    __slots__ = ("name", "snapshot_id", "content_hash", "_items", "_tags",
                 "meta")

    def __init__(self, name: str, snapshot_id: int, items: Dict[str, Any],
                 tags: Dict[str, str], meta: Optional[Dict[str, Any]] = None):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "snapshot_id", int(snapshot_id))
        object.__setattr__(self, "_items", MappingProxyType(dict(items)))
        object.__setattr__(self, "_tags", MappingProxyType(dict(tags)))
        object.__setattr__(self, "content_hash", _content_hash(items, tags))
        object.__setattr__(self, "meta", MappingProxyType(dict(meta or {})))

    def __setattr__(self, k: str, v: Any) -> None:
        raise AttributeError(f"Snapshot is immutable (tried to set {k!r})")

    @property
    def arrays(self) -> Mapping[str, Any]:
        """Read-only mapping of read-only arrays / scalars (zero copy)."""
        return self._items

    def keys(self):
        return self._items.keys()

    def provenance(self) -> Dict[str, Any]:
        """What a prediction records about the weights that produced it."""
        return {"name": self.name, "snapshot_id": self.snapshot_id,
                "content_hash": self.content_hash,
                "schema_version": SCHEMA_VERSION}

    def verify(self) -> None:
        h = _content_hash(self._items, self._tags)
        if h != self.content_hash:
            raise SnapshotCorruption(
                f"snapshot {self.name}#{self.snapshot_id} content changed "
                f"after publication ({self.content_hash[:12]} -> {h[:12]})")

    def to_state_dict(self, device: Any = None) -> Dict[str, Any]:
        """Fresh, writable torch copies — mutating them cannot reach us."""
        import torch
        out: Dict[str, Any] = {}
        for k, v in self._items.items():
            if isinstance(v, np.ndarray):
                t = torch.tensor(np.array(v, copy=True))
                tag = self._tags[k]
                if tag == "bfloat16":
                    t = t.to(torch.bfloat16)
                out[k] = t.to(device) if device is not None else t
            else:
                out[k] = v
        return out

    def load_into(self, module: Any, strict: bool = True) -> Any:
        return module.load_state_dict(self.to_state_dict(), strict=strict)

    def __repr__(self) -> str:
        return (f"Snapshot({self.name!r}, id={self.snapshot_id}, "
                f"hash={self.content_hash[:12]}, n={len(self._items)})")


class SnapshotRegistry:
    """Named, versioned, immutable snapshots with atomic publication."""

    def __init__(self, keep: int = 4):
        if int(keep) < 1:
            raise ValueError("keep must be >= 1")
        self.keep = int(keep)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._history: Dict[str, list] = {}

    def publish(self, name: str, module_or_state: Any,
                lock: Any = None, meta: Optional[Dict[str, Any]] = None
                ) -> Snapshot:
        """Copy `module_or_state` into a new immutable snapshot.

        `lock` (e.g. the loop's `_wm_param_lock`) is held while the source is
        COPIED, so a concurrently stepping optimizer cannot tear it. The
        registry's own lock is held only to assign the id and swap the
        pointer, so readers are never blocked by a copy.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("snapshot name must be a non-empty string")
        cm = lock if lock is not None else contextlib.nullcontext()
        with cm:
            sd = (module_or_state.state_dict()
                  if hasattr(module_or_state, "state_dict")
                  else module_or_state)
            if not isinstance(sd, Mapping):
                raise TypeError("publish needs a module or a mapping")
            items, tags = {}, {}
            for k, v in sd.items():
                items[str(k)], tags[str(k)] = _freeze_value(v)
        with self._lock:
            snap = Snapshot(name, next(self._ids), items, tags, meta)
            hist = self._history.setdefault(name, [])
            hist.append(snap)
            del hist[:-self.keep]
        return snap

    def current(self, name: str) -> Snapshot:
        with self._lock:
            hist = self._history.get(name)
            if not hist:
                raise LookupError(f"no snapshot published under {name!r}")
            return hist[-1]

    def get(self, name: str, snapshot_id: int) -> Snapshot:
        with self._lock:
            for s in self._history.get(name, ()):
                if s.snapshot_id == int(snapshot_id):
                    return s
        raise LookupError(f"{name}#{snapshot_id} not retained "
                          f"(keep={self.keep})")

    def pin(self, *names: str) -> Dict[str, Snapshot]:
        """A consistent cut: the current snapshot of each name, atomically."""
        with self._lock:
            out = {}
            for n in names:
                hist = self._history.get(n)
                if not hist:
                    raise LookupError(f"no snapshot published under {n!r}")
                out[n] = hist[-1]
            return out

    def names(self):
        with self._lock:
            return sorted(self._history)


def stamp_prediction(payload: Dict[str, Any],
                     snapshots: Mapping[str, Snapshot]) -> Dict[str, Any]:
    """Return a copy of `payload` recording which snapshots produced it."""
    if "snapshots" in payload:
        raise ValueError("payload already carries snapshot provenance")
    out = dict(payload)
    out["snapshots"] = {n: s.provenance() for n, s in snapshots.items()}
    return out
