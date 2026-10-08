"""ExplorationTracker — how far, how widely and how recently the agent moved.

MEASUREMENT ONLY. Nothing here feeds reward, actions, replay, training or
normalisation. The loop calls `observe()` once per stream per step with the
evaluator's position and `segment_summary()` once per segment; the summary
goes into learning.jsonl under "exploration" and the position trace goes to an
EVALUATOR-ONLY file (`runlogs/position_trace.jsonl`) that the agent never
reads.

WHY THIS EXISTS. The only movement number the run reported was the env's
`cells_seen`, a career visit table capped at 50,000 cells with FIFO eviction:
once at the cap its length never changes again, so the derived per-segment
cells delta read 0 forever — "no new territory" whether the agent was pinned
to a tree or walking across the map (CLAUDE.md §4.1, a guard become a latch).
This tracker keeps the unique-cell set UNBOUNDED by default (`max_cells=None`)
and compact (sorted int64 array + small pending set, 8 bytes/cell), and adds
the quantities that separate "stuck" from "exploring": path length, net
displacement, radius of gyration, stationary fraction and time since the last
new cell.

DEFINITIONS (per stream, per segment unless stated):
  path_blocks          sum of 3D step distances between consecutive
                       observations; a step > TELEPORT_BLOCKS (20) is a
                       teleport (counted in `teleports`, not in the path) and
                       no step is taken across an episode change.
  net_displacement     3D distance first -> last position of the segment.
  cells                (floor(x/cell_size), floor(z/cell_size)) packed in int64.
  unique_cells_segment distinct cells visited this segment.
  new_cells_segment    cells first visited EVER (since load) this segment.
  unique_cells_total   size of the career set (persists via state_dict).
  cells_per_hour       new_cells_segment / segment wall hours (None if 0 h).
  radius_of_gyration   RMS distance of the segment's positions from their mean.
  steps_since_new_cell observations since the last new cell; persists across
                       segments and restarts.
  seconds_since_new_cell wall seconds since the last new cell (persists).
  stationary_frac      fraction of steps moving < stationary_eps (steps =
                       observation pairs in one episode, teleports excluded);
                       None when there were no such steps.
  y_min / y_max        vertical range of the segment.
Undefined values are None (never a fake 0): ingest keeps numbers only.

Pure module: stdlib + numpy.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TELEPORT_BLOCKS = 20.0
_I32_MIN, _I32_MAX = -(2 ** 31), 2 ** 31 - 1


def pack_cell(cx: int, cz: int) -> int:
    """Pack two int32 cell coordinates into one signed int64."""
    cx = min(max(int(cx), _I32_MIN), _I32_MAX)
    cz = min(max(int(cz), _I32_MIN), _I32_MAX)
    return (cx << 32) | (cz & 0xFFFFFFFF)


def unpack_cell(k: int) -> tuple:
    k = int(k)
    cz = k & 0xFFFFFFFF
    if cz >= 2 ** 31:
        cz -= 2 ** 32
    return (k >> 32, cz)


class CompactCellSet:
    """An unbounded int64 set: sorted numpy array + a small pending set.

    8 bytes per cell once merged (a Python set costs ~70). `add` returns True
    when the key was new. `max_cells` (optional) stops GROWTH but never makes
    `add` lie: once full, unseen keys are reported new and counted in
    `overflow` instead of being silently absorbed.
    """

    MERGE_AT = 8192

    def __init__(self, max_cells: Optional[int] = None):
        self.max_cells = None if max_cells is None else int(max_cells)
        self._arr = np.zeros(0, dtype=np.int64)
        self._pending: set = set()
        self.overflow = 0

    def __len__(self) -> int:
        return int(self._arr.size) + len(self._pending)

    def __contains__(self, k: int) -> bool:
        k = int(k)
        if k in self._pending:
            return True
        a = self._arr
        if a.size == 0:
            return False
        i = int(np.searchsorted(a, k))
        return i < a.size and int(a[i]) == k

    def add(self, k: int) -> bool:
        k = int(k)
        if k in self:
            return False
        if self.max_cells is not None and len(self) >= self.max_cells:
            self.overflow += 1
            return True
        self._pending.add(k)
        if len(self._pending) >= self.MERGE_AT:
            self._merge()
        return True

    def _merge(self) -> None:
        if self._pending:
            p = np.fromiter(self._pending, dtype=np.int64,
                            count=len(self._pending))
            self._arr = np.union1d(self._arr, p).astype(np.int64)
            self._pending = set()

    def to_list(self) -> List[int]:
        self._merge()
        return [int(v) for v in self._arr]

    def load_list(self, keys) -> None:
        a = np.asarray(list(keys), dtype=np.int64)
        self._arr = np.unique(a).astype(np.int64)
        self._pending = set()


class _Stream:
    def __init__(self, max_cells):
        self.cells = CompactCellSet(max_cells)
        self.steps_since_new_cell = 0
        self.last_new_cell_t: Optional[float] = None
        self.total_obs = 0
        self.total_path = 0.0
        self.teleports_total = 0
        self.last_pos: Optional[tuple] = None
        self.last_episode: Any = None
        self.last_t: Optional[float] = None
        self.seg_end_t: Optional[float] = None   # previous segment's last obs
        self.reset_segment()

    def reset_segment(self):
        self.n_obs = 0
        self.path = 0.0
        self.teleports = 0
        self.n_moves = 0
        self.n_still = 0
        self.first_pos: Optional[tuple] = None
        self.seg_last_pos: Optional[tuple] = None
        self.seg_t0: Optional[float] = None
        self.sx = self.sy = self.sz = self.sq = 0.0   # shifted by first_pos
        self.y_min: Optional[float] = None
        self.y_max: Optional[float] = None
        self.seg_cells: set = set()
        self.new_cells = 0


class ExplorationTracker:
    """See module docstring. API is the M -> L contract in TELEMETRY_SPEC."""

    ROTATE_BYTES = 200 * 1024 * 1024
    ROTATE_KEEP = 3
    FLUSH_LINES = 4096           # safety flush between segments

    def __init__(self, cell_size: float = 1.0, trace_every_steps: int = 16,
                 trace_path: Optional[str] = "runlogs/position_trace.jsonl",
                 stationary_eps: float = 0.25,
                 max_cells: Optional[int] = None,
                 run_id: Optional[str] = None):
        if not (float(cell_size) > 0):
            raise ValueError(f"cell_size must be > 0, got {cell_size}")
        self.cell_size = float(cell_size)
        self.trace_every_steps = max(1, int(trace_every_steps))
        self.trace_path = trace_path
        self.stationary_eps = float(stationary_eps)
        self.max_cells = max_cells
        self.run_id = run_id if run_id is not None else os.environ.get(
            "SKYBOT_RUN_ID")
        self.rotate_bytes = self.ROTATE_BYTES
        self._streams: Dict[int, _Stream] = {}
        self._buf: List[str] = []
        self.rejected = 0
        self._warned = False

    # ------------------------------------------------------------------ io
    def _warn(self, msg, e):
        if not self._warned:
            self._warned = True
            logger.warning("exploration: %s: %s (further errors silent)",
                           msg, e)

    def _stream(self, i: int) -> _Stream:
        s = self._streams.get(i)
        if s is None:
            s = self._streams[i] = _Stream(self.max_cells)
        return s

    def _rotate(self, path: str) -> None:
        try:
            if os.path.getsize(path) < self.rotate_bytes:
                return
        except OSError:
            return
        for k in range(self.ROTATE_KEEP, 0, -1):
            src = path if k == 1 else f"{path}.{k - 1}"
            dst = f"{path}.{k}"
            if os.path.exists(src):
                os.replace(src, dst)

    def flush(self) -> None:
        if not self._buf:
            return
        lines, self._buf = self._buf, []
        if not self.trace_path:
            return
        try:
            d = os.path.dirname(self.trace_path)
            if d:
                os.makedirs(d, exist_ok=True)
            self._rotate(self.trace_path)
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write("".join(lines))
        except Exception as e:  # measurement must never raise into the loop
            self._warn("trace write failed", e)

    # ------------------------------------------------------------- observe
    def observe(self, stream: int, step: int, t_wall: float, pos,
                yaw: Optional[float] = None, pitch: Optional[float] = None,
                episode: Optional[str] = None) -> None:
        try:
            x, y, z = (float(pos[0]), float(pos[1]), float(pos[2]))
            t = float(t_wall)
        except Exception:
            self.rejected += 1
            return
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
                and math.isfinite(t)):
            self.rejected += 1
            return
        s = self._stream(int(stream))
        p = (x, y, z)

        # step from the previous observation, same episode only
        if s.last_pos is not None and episode == s.last_episode:
            d = math.sqrt((x - s.last_pos[0]) ** 2 + (y - s.last_pos[1]) ** 2
                          + (z - s.last_pos[2]) ** 2)
            if d > TELEPORT_BLOCKS:
                s.teleports += 1
                s.teleports_total += 1
            else:
                s.path += d
                s.total_path += d
                s.n_moves += 1
                if d < self.stationary_eps:
                    s.n_still += 1

        # segment geometry (sums shifted by the first point: no cancellation
        # at large world coordinates)
        if s.first_pos is None:
            s.first_pos = p
            s.seg_t0 = s.seg_end_t if s.seg_end_t is not None else t
        fx, fy, fz = s.first_pos
        dx, dy, dz = x - fx, y - fy, z - fz
        s.sx += dx
        s.sy += dy
        s.sz += dz
        s.sq += dx * dx + dy * dy + dz * dz
        s.y_min = y if s.y_min is None else min(s.y_min, y)
        s.y_max = y if s.y_max is None else max(s.y_max, y)

        # cells
        k = pack_cell(math.floor(x / self.cell_size),
                      math.floor(z / self.cell_size))
        s.seg_cells.add(k)
        if s.cells.add(k):
            s.new_cells += 1
            s.steps_since_new_cell = 0
            s.last_new_cell_t = t
        else:
            s.steps_since_new_cell += 1

        # trace (first observation of a stream, then every N)
        if s.total_obs % self.trace_every_steps == 0 and self.trace_path:
            self._buf.append(json.dumps({
                "schema": "skybot.position", "v": 1, "run_id": self.run_id,
                "stream": int(stream), "episode": episode, "step": int(step),
                "t_wall": round(t, 3), "x": round(x, 3), "y": round(y, 3),
                "z": round(z, 3),
                "yaw": None if yaw is None else round(float(yaw), 2),
                "pitch": None if pitch is None else round(float(pitch), 2),
            }) + "\n")
            if len(self._buf) >= self.FLUSH_LINES:
                self.flush()

        s.n_obs += 1
        s.total_obs += 1
        s.last_pos = p
        s.seg_last_pos = p
        s.last_episode = episode
        s.last_t = t

    # ------------------------------------------------------------- summary
    def _summ(self, s: _Stream) -> Dict[str, Any]:
        n = s.n_obs
        out: Dict[str, Any] = {
            "path_blocks": float(s.path),
            "net_displacement": None, "unique_cells_segment": len(s.seg_cells),
            "unique_cells_total": len(s.cells),
            "new_cells_segment": int(s.new_cells), "cells_per_hour": None,
            "radius_of_gyration": None,
            "steps_since_new_cell": int(s.steps_since_new_cell),
            "seconds_since_new_cell": None,
            "stationary_frac": (float(s.n_still) / s.n_moves
                                if s.n_moves else None),
            "y_min": s.y_min, "y_max": s.y_max, "n_obs": int(n),
            "teleports": int(s.teleports),
        }
        if n:
            a, b = s.first_pos, s.seg_last_pos
            out["net_displacement"] = float(math.sqrt(
                (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2))
            mean_sq = (s.sx * s.sx + s.sy * s.sy + s.sz * s.sz) / (n * n)
            out["radius_of_gyration"] = float(math.sqrt(
                max(0.0, s.sq / n - mean_sq)))
            hours = (s.last_t - s.seg_t0) / 3600.0
            if hours > 0:
                out["cells_per_hour"] = float(s.new_cells / hours)
        if s.last_new_cell_t is not None and s.last_t is not None:
            out["seconds_since_new_cell"] = float(
                max(0.0, s.last_t - s.last_new_cell_t))
        return out

    def segment_summary(self, reset: bool = True) -> Dict[str, Dict[str, Any]]:
        out = {}
        for i in sorted(self._streams):
            s = self._streams[i]
            try:
                out[f"stream-{i}"] = self._summ(s)
            except Exception as e:
                self._warn("summary failed", e)
            if reset:
                if s.last_t is not None:
                    s.seg_end_t = s.last_t
                s.reset_segment()
        self.flush()
        return out

    # --------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, Any]:
        return {
            "v": 1, "cell_size": self.cell_size,
            "streams": {str(i): {
                "cells": s.cells.to_list(),
                "steps_since_new_cell": int(s.steps_since_new_cell),
                "last_new_cell_t": s.last_new_cell_t,
                "total_obs": int(s.total_obs),
                "total_path": float(s.total_path),
                "teleports_total": int(s.teleports_total),
            } for i, s in self._streams.items()},
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        if not isinstance(d, dict) or int(d.get("v", 0)) != 1:
            raise ValueError("exploration state: unknown schema version")
        if float(d.get("cell_size", self.cell_size)) != self.cell_size:
            raise ValueError(
                f"exploration state: cell_size {d.get('cell_size')} != "
                f"{self.cell_size} (cells would mean different areas)")
        for k, sd in (d.get("streams") or {}).items():
            s = self._stream(int(k))
            s.cells.load_list(sd.get("cells") or [])
            s.steps_since_new_cell = int(sd.get("steps_since_new_cell", 0))
            t = sd.get("last_new_cell_t")
            s.last_new_cell_t = None if t is None else float(t)
            s.total_obs = int(sd.get("total_obs", 0))
            s.total_path = float(sd.get("total_path", 0.0))
            s.teleports_total = int(sd.get("teleports_total", 0))
