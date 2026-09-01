"""Episodic event memory — "what happened, where, and how long ago".

WHY THIS EXISTS: the agent already has gradient memory (a replay buffer), but
gradient memory is not RETRIEVABLE — nothing in the stack can answer "the last
time I saw my goal object, it was over there".  In practice this cost a
measured six-hour aimless wander: the goal had been in view, the sighting was
consumed as a scalar reward, and the position was gone.  This module is the
substrate that makes such facts queryable: a compact ring buffer of typed
event records with recency, count, proximity and bearing queries plus a
one-line human-readable report for logs/monitors.

Scope of this wave: RECORDING + QUERYING + REPORTING only.  Behavioural
integration (using a bearing to steer) is a later wave and lives elsewhere —
this module deliberately knows nothing about actions, rewards or any specific
environment vocabulary.

Design decisions, and the failure each prevents:
  * Ring buffer (deque with maxlen): a multi-day run must not grow memory
    without bound.  Eviction is oldest-first, which is the correct semantics
    for episodic recall — stale sightings are exactly the ones worth losing.
  * All queries are O(n) linear scans.  At the default capacity of 4000 this
    is microseconds; an index would add invalidation bugs for zero measured
    benefit.
  * DEFENSIVE THROUGHOUT: this is monitoring-adjacent code and must NEVER be
    able to crash a run.  record() coerces or drops malformed input instead
    of raising; every query catches internally and returns its empty value.
    Failures are not silent — they are counted in `error_count` and the last
    message is kept in `last_error`, so a monitor can surface "memory is
    being fed garbage" as DATA rather than as a dead process.
  * Horizontal-plane distance: near() and bearing_from() measure euclidean
    distance over (x, z) ONLY.  For ground navigation, height difference is
    not travel distance — a record 3 units below the caller on a slope is
    "right here", not "3 away" — and including y made proximity queries miss
    exactly the recalls they exist for.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


class EpisodicEventMemory:
    """Bounded store of (kind, subtype, step, position, salience) events.

    ``kind`` is a coarse category chosen by the caller (e.g. "sighting",
    "achieve"), ``subtype`` names the specific thing within it.  Both are
    plain strings — the memory imposes no vocabulary, keeping it usable by
    any domain that can name its events.
    """

    def __init__(self, capacity: int = 4000):
        # Guard the capacity itself: a config typo (0, -5, "4000") must not
        # take down the run, so fall back to the default instead of raising.
        try:
            capacity = int(capacity)
        except (TypeError, ValueError):
            capacity = 4000
        if capacity <= 0:
            capacity = 4000
        self._buf: Deque[Dict] = deque(maxlen=capacity)
        # Error telemetry: malformed input is contained, counted, and the
        # last message kept so monitors can report it as data.
        self.error_count: int = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # recording
    # ------------------------------------------------------------------ #

    def record(self, kind: str, subtype: str, step: int,
               position: Optional[Tuple[float, float, float]] = None,
               salience: float = 1.0) -> None:
        """Append one event.  Never raises.

        Malformed positions are stored as None (a positionless event is
        still worth counting — "it happened N times" survives even when
        "where" was garbage), and malformed salience falls back to 1.0.
        A record whose kind/step cannot even be coerced is dropped, but
        the drop is visible in `error_count`/`last_error`.
        """
        try:
            pos: Optional[Tuple[float, float, float]] = None
            if position is not None:
                try:
                    x, y, z = (float(position[0]), float(position[1]),
                               float(position[2]))
                    if math.isfinite(x) and math.isfinite(y) \
                            and math.isfinite(z):
                        pos = (x, y, z)
                    else:
                        self._note_error("non-finite position dropped")
                except (TypeError, ValueError, IndexError):
                    self._note_error("malformed position dropped")
            try:
                sal = float(salience)
                if not math.isfinite(sal):
                    sal = 1.0
            except (TypeError, ValueError):
                sal = 1.0
                self._note_error("malformed salience -> 1.0")
            self._buf.append({
                "kind": str(kind),
                "subtype": str(subtype),
                "step": int(step),
                "position": pos,
                "salience": sal,
            })
        except Exception as exc:                       # pragma: no cover
            # Absolute backstop: an unrecordable event costs one memory,
            # never the run.
            self._note_error(f"record dropped: {exc!r}")

    # ------------------------------------------------------------------ #
    # queries — each catches internally and returns its empty value
    # ------------------------------------------------------------------ #

    def last(self, kind: str,
             subtype: Optional[str] = None) -> Optional[Dict]:
        """Most recent record matching kind (and subtype, if given).

        Returns a COPY of the record dict, so a caller mutating the result
        (e.g. annotating it for a report) cannot corrupt the store.
        """
        try:
            for rec in reversed(self._buf):
                if rec["kind"] == kind and \
                        (subtype is None or rec["subtype"] == subtype):
                    return dict(rec)
            return None
        except Exception as exc:
            self._note_error(f"last() failed: {exc!r}")
            return None

    def count(self, kind: str, subtype: Optional[str] = None) -> int:
        """Number of retained records matching kind (and subtype, if given).

        NOTE: counts only what the ring buffer still holds — after eviction
        this is a recency-weighted count, not a lifetime total, which is the
        honest number for "how often has this been happening LATELY".
        """
        try:
            return sum(1 for rec in self._buf
                       if rec["kind"] == kind and
                       (subtype is None or rec["subtype"] == subtype))
        except Exception as exc:
            self._note_error(f"count() failed: {exc!r}")
            return 0

    def near(self, position: Tuple[float, float, float], radius: float,
             kind: Optional[str] = None) -> List[Dict]:
        """Records within `radius` of `position`, most recent first.

        Distance is euclidean over (x, z) ONLY — height is not distance for
        navigation purposes (see module docstring), so a record directly
        above or below the query point is at distance 0.  Records without a
        position can never be "near" and are skipped.  Results are copies.
        """
        try:
            qx, qz = float(position[0]), float(position[2])
            r2 = float(radius) ** 2
            out: List[Dict] = []
            for rec in reversed(self._buf):            # recent first
                if kind is not None and rec["kind"] != kind:
                    continue
                pos = rec["position"]
                if pos is None:
                    continue
                dx, dz = pos[0] - qx, pos[2] - qz
                if dx * dx + dz * dz <= r2:
                    out.append(dict(rec))
            return out
        except Exception as exc:
            self._note_error(f"near() failed: {exc!r}")
            return []

    def bearing_from(self, position: Tuple[float, float, float], kind: str,
                     subtype: Optional[str] = None
                     ) -> Optional[Tuple[float, float]]:
        """(distance, bearing_deg) to the most recent matching record that
        HAS a position; None when no positioned match exists.

        Distance is horizontal-plane (x, z) euclidean.  Bearing is
        ``math.degrees(math.atan2(dx, dz))`` with dx, dz = target minus
        origin: a target due +x is 90 deg, due +z is 0 deg, range
        (-180, 180].  This is a PURE geometric bearing — the CALLER owns any
        game-specific yaw convention (sign flips, offsets, wrap-around).
        Baking one engine's convention in here is exactly how a
        domain-agnostic module stops being one.

        Positionless records of the same kind/subtype are skipped, not
        treated as "no match": the newest event may have arrived without
        coordinates while an older sighting still knows where to go.
        """
        try:
            for rec in reversed(self._buf):
                if rec["kind"] != kind:
                    continue
                if subtype is not None and rec["subtype"] != subtype:
                    continue
                pos = rec["position"]
                if pos is None:
                    continue                       # keep looking further back
                dx = pos[0] - float(position[0])
                dz = pos[2] - float(position[2])
                return (math.hypot(dx, dz),
                        math.degrees(math.atan2(dx, dz)))
            return None
        except Exception as exc:
            self._note_error(f"bearing_from() failed: {exc!r}")
            return None

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #

    def summary(self, now_step: int,
                kinds: Optional[List[str]] = None) -> str:
        """Compact one-liner for logs/monitors, e.g.

            "sighting:goal @(12,-40) 340 steps ago | achieve:goal never"

        `kinds` entries may be a bare kind ("sighting") or a
        "kind:subtype" spec ("achieve:goal"); each contributes one segment
        describing its most recent record — position (x,z, rounded) when
        known, and age in steps.  A requested-but-unseen entry reports
        "never": absence is information (a monitor watching for the first
        achievement needs the explicit negative, not a missing segment).
        With kinds=None, one segment per distinct kind ever retained, in
        first-seen order.  Never raises — a formatting failure returns an
        error MARKER string so the report line itself carries the fault.
        """
        try:
            if kinds is None:
                seen: List[str] = []
                for rec in self._buf:                  # first-seen order
                    if rec["kind"] not in seen:
                        seen.append(rec["kind"])
                specs = seen
            else:
                specs = list(kinds)
            if not specs:
                return "(no episodic records)"
            parts: List[str] = []
            for spec in specs:
                kind, _, sub = str(spec).partition(":")
                rec = self.last(kind, sub if sub else None)
                if rec is None:
                    parts.append(f"{spec} never")
                    continue
                label = f"{rec['kind']}:{rec['subtype']}"
                seg = label
                if rec["position"] is not None:
                    x, _, z = rec["position"]
                    seg += f" @({x:.0f},{z:.0f})"
                seg += f" {int(now_step) - rec['step']} steps ago"
                parts.append(seg)
            return " | ".join(parts)
        except Exception as exc:
            self._note_error(f"summary() failed: {exc!r}")
            return f"(episodic summary error: {exc!r})"

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _note_error(self, msg: str) -> None:
        """Count and keep — errors surface as data, never as exceptions."""
        self.error_count += 1
        self.last_error = msg

    def __len__(self) -> int:
        return len(self._buf)
