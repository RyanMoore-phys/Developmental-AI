"""Structured per-segment metrics, written durably to disk.

WHY THIS EXISTS
    Everything this system measures already goes somewhere — the reward
    ledger's income statement, break counts by block type, crafts, entropy,
    option activity — but all of it lands in a human-readable log. There is no
    way to plot it, and when the first live run died after four hours the only
    evidence of what it had ACHIEVED was prose in a 18 KB text file.

    This module does not measure anything new. It serialises what the loop has
    already computed into one JSON object per segment, so a dashboard can read
    it and a pod death cannot take it with it.

WHY fsync, NOT JUST flush
    `flush()` only pushes bytes out of Python's buffer into the OS page cache.
    A process crash survives that; a HOST death does not. The first live run
    was killed by a deterministic crash after its last checkpoint, and the
    interesting question — what was it doing in the segment that killed it? —
    is exactly the record most at risk. fsync costs a few ms per segment (a
    segment is ~1024 env steps, i.e. minutes) and buys the guarantee that a
    written record is on disk when `write()` returns.

WHY ROTATION IS NOT OPTIONAL
    `podlogs/ollama.log` reached 29 MB unattended in this project, and the VLM
    chatter that filled it needed a dedicated capper script. A per-segment
    record on a multi-day run does the same more slowly. Rotating here means
    the tracker cannot be the thing that fills the pod's disk.

DEFENSIVE CONTRACT (identical to infra/consequence.py and infra/anticipation.py)
    never raises, bounded memory, records its own errors for inspection. A
    monitoring system that can crash the thing it monitors is worse than no
    monitoring system. Every public method swallows and notes.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

# Bumped when the RECORD SHAPE changes in a way a reader must know about.
# The ingester stores it per row, so a dashboard can tell "this field is
# missing because it is old" from "this field is missing because it broke".
SCHEMA_VERSION = 1


def _jsonable(obj: Any) -> Any:
    """Best-effort coercion to something json.dumps will accept.

    Metric values arrive from numpy, torch and plain python interchangeably;
    a numpy float32 raises `TypeError: Object of type float32 is not JSON
    serializable` and would otherwise lose the whole record. Coerce rather
    than refuse — a metric written as a string beats a segment written as
    nothing.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    # numpy / torch scalars and arrays, without importing either
    for attr in ("item", "tolist"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:
                pass
    return str(obj)


class MetricsSink:
    """Append-only JSONL of per-segment records. Never raises."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.path = str(cfg.get("path", "podlogs/metrics.jsonl"))
        # 64 MB default: ~100k segments at typical record size, and small
        # enough that the rsync backfill of a rotated file stays quick.
        self.max_bytes = int(cfg.get("max_bytes", 64 * 1024 * 1024))
        self.keep = int(cfg.get("keep", 5))
        # fsync is the durability guarantee; exposed so a throughput-bound
        # run can trade it away deliberately rather than by accident.
        self.fsync = bool(cfg.get("fsync", True))

        self.seq = 0
        self.written = 0
        self.dropped = 0
        self.last_error: Optional[str] = None
        self.errors: List[str] = []

        # SEQ MUST NEVER GO BACKWARDS ACROSS A RESTART.
        # `seq` is the collector's dedupe key (INSERT OR IGNORE on a UNIQUE
        # column), so a restart that reuses numbers makes the ingester
        # silently discard every new record as "already seen" — the tracker
        # would look alive and store nothing.
        #
        # Counting lines in the current file is NOT sufficient, and the first
        # version of this did exactly that: after a rotation the live file is
        # short (1 line) while the rotated files hold seq 1..N, so the next
        # run restarts at 1 and collides with everything already ingested.
        # Re-reading the rotated files would mean up to keep*max_bytes of I/O
        # at every startup (320 MB at the defaults), so instead a sidecar
        # holds the high-water mark and rotation never touches it.
        self._seq_path = self.path + ".seq"
        _hw = 0
        try:
            if os.path.exists(self._seq_path):
                with open(self._seq_path) as f:
                    _hw = int((f.read() or "0").strip() or 0)
        except Exception as exc:
            self._note(f"seq sidecar read failed: {exc!r}")
        _lines = 0
        try:
            if os.path.exists(self.path):
                with open(self.path, "rb") as f:
                    _lines = sum(1 for _ in f)
        except Exception as exc:
            self._note(f"seq resume failed: {exc!r}")
        # max(): the sidecar is authoritative, but a file written before the
        # sidecar existed (or a lost sidecar) must still not rewind.
        self.seq = max(_hw, _lines)

    # ------------------------------------------------------------------ #
    def _note(self, msg: str) -> None:
        self.last_error = str(msg)
        if len(self.errors) < 64:
            self.errors.append(str(msg))

    def _persist_seq(self) -> None:
        """Record the high-water seq so a restart cannot rewind it.

        Written atomically (tmp + os.replace) so a crash mid-write leaves
        either the old value or the new one, never a truncated number that
        would parse as a smaller seq. Deliberately NOT fsync'd: losing it
        costs at most a fallback to the line count, which the constructor's
        max() already handles, and it is written on every segment.
        """
        try:
            tmp = self._seq_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(str(int(self.seq)))
            os.replace(tmp, self._seq_path)
        except Exception as exc:
            self._note(f"seq persist failed: {exc!r}")

    def _rotate_if_needed(self) -> None:
        try:
            if self.max_bytes <= 0 or not os.path.exists(self.path):
                return
            if os.path.getsize(self.path) < self.max_bytes:
                return
            # metrics.jsonl -> .1, .1 -> .2, ... oldest falls off the end.
            for i in range(self.keep - 1, 0, -1):
                src, dst = f"{self.path}.{i}", f"{self.path}.{i + 1}"
                if os.path.exists(src):
                    os.replace(src, dst)
            os.replace(self.path, f"{self.path}.1")
        except Exception as exc:
            self._note(f"rotate failed: {exc!r}")

    # ------------------------------------------------------------------ #
    def write(self, record: Dict[str, Any]) -> bool:
        """Append one record. Returns True if it reached disk.

        Adds `seq`, `wall_time` and `schema_version`; the caller supplies
        everything else. Returns False rather than raising on any failure —
        see the module docstring.
        """
        if not self.enabled:
            return False
        try:
            self.seq += 1
            row = {
                "schema_version": SCHEMA_VERSION,
                "seq": int(self.seq),
                "wall_time": time.time(),
            }
            row.update(_jsonable(dict(record or {})))
            line = json.dumps(row, separators=(",", ":"), default=str)

            self._rotate_if_needed()
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                if self.fsync:
                    # flush() only reaches the OS page cache; fsync() is what
                    # survives a host death. See the module docstring.
                    os.fsync(f.fileno())
            self._persist_seq()
            self.written += 1
            return True
        except Exception as exc:                       # pragma: no cover
            self.dropped += 1
            self._note(f"write failed: {exc!r}")
            return False

    # ------------------------------------------------------------------ #
    def state(self) -> Dict[str, Any]:
        """Forensics for the segment log: is the tracker actually tracking?"""
        return {
            "path": self.path,
            "seq": int(self.seq),
            "written": int(self.written),
            "dropped": int(self.dropped),
            "last_error": self.last_error,
        }
