"""Save-state checkpoints and automatic rewind for emulator games.

Born 2026-10-11 with the universal-games work (docs/UNIVERSAL_GAMES.md §4).
User decision: snapshot the GAME every 5 minutes, keep a ring of 3, and rewind
the game automatically when reward stagnation shows SkyBot is stuck. The brain
is never rewound — it keeps everything it learned, including being stuck.

Any adapter whose backend can save and load its own state (ViZDoom now,
PyBoy / stable-retro next) uses these two pieces:

    SnapshotRing   wall-clock interval, newest `keep` files, oldest deleted.
    StuckWatch     fed one reward per step; says when the stream is stuck.

Minecraft cannot use either: MineRL cannot save a world (CLAUDE.md §7) and
the Paper server is shared with other agents.

THIS IS A GUARD, SO IT MUST NOT LATCH (CLAUDE.md §4.1). Every refusal state
here has a stated, reachable exit:

    cooldown after a rewind      -> expires after one window
    rewinds that do not help     -> after `max_rewinds_without_recovery`, the
                                    next stuck call asks for a FRESH EPISODE
                                    instead (and the ring is cleared), so a
                                    rewind loop cannot become a trap
    recovery                     -> resets that count
    no snapshot / load failure   -> the adapter falls back to a fresh episode

and it pays nothing, ever: it writes no reward, it only moves the game.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

# What StuckWatch.observe() asks the adapter to do.
NONE = "none"
REWIND = "rewind"
FRESH = "fresh"


class SnapshotRing:
    """Newest-`keep` save states for one stream, written every `interval_s`.

    `save_fn(path)` and `load_fn(path)` are the backend's own save/load.
    Files are named with a monotonically increasing counter, so ordering never
    depends on mtime (which a restored backup or a clock step can scramble).
    """

    def __init__(self, directory: str, save_fn: Callable[[str], None],
                 load_fn: Callable[[str], None], interval_s: float = 300.0,
                 keep: int = 3, ext: str = "state",
                 clock: Callable[[], float] = time.monotonic):
        if keep < 1:
            raise ValueError("SnapshotRing.keep must be >= 1")
        self.dir = directory
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.interval_s = float(interval_s)
        self.keep = int(keep)
        self.ext = ext.lstrip(".")
        self.clock = clock
        os.makedirs(self.dir, exist_ok=True)
        # A previous process's ring belongs to a game state that no longer
        # exists in this process (the emulator restarted): restoring it would
        # be a different save than the one this run recorded. Start clean.
        for f in os.listdir(self.dir):
            if f.startswith("snap_") and f.endswith("." + self.ext):
                try:
                    os.remove(os.path.join(self.dir, f))
                except OSError:
                    pass
        self._files: Deque[Tuple[float, str]] = deque()
        self._n = 0
        self._last = None          # first maybe_save() saves immediately
        self.saves = 0
        self.save_failures = 0

    def __len__(self) -> int:
        return len(self._files)

    def due(self) -> bool:
        return self._last is None or (self.clock() - self._last) >= self.interval_s

    def maybe_save(self) -> bool:
        """Save if the interval has elapsed. Never raises."""
        if not self.due():
            return False
        path = os.path.join(self.dir, f"snap_{self._n:08d}.{self.ext}")
        self._last = self.clock()
        try:
            self.save_fn(path)
        except Exception as e:                      # backend refused / disk
            self.save_failures += 1
            logger.warning("snapshot save failed (%s): %s", path, e)
            return False
        self._n += 1
        self.saves += 1
        self._files.append((self._last, path))
        while len(self._files) > self.keep:
            _t, old = self._files.popleft()
            try:
                os.remove(old)
            except OSError:
                pass
        return True

    def oldest(self) -> Optional[str]:
        return self._files[0][1] if self._files else None

    def load_oldest(self) -> bool:
        """Restore the oldest snapshot. False (and logged) if none/failed."""
        path = self.oldest()
        if path is None or not os.path.exists(path):
            return False
        try:
            self.load_fn(path)
        except Exception as e:
            logger.warning("snapshot load failed (%s): %s", path, e)
            return False
        # The newer snapshots were taken DURING the slump this rewind escapes.
        # Kept, they would age into "oldest" and a later rewind would land
        # back inside the stuck period. The timeline after `path` is gone.
        while len(self._files) > 1:
            _t, p = self._files.pop()
            try:
                os.remove(p)
            except OSError:
                pass
        return True

    def clear(self) -> None:
        while self._files:
            _t, p = self._files.popleft()
            try:
                os.remove(p)
            except OSError:
                pass
        self._last = None          # next maybe_save() saves immediately

    def restart_interval(self) -> None:
        """Count the next interval from now (after a rewind, the restored
        state is already the 'snapshot'; saving it again at once would only
        evict an older, different one)."""
        self._last = self.clock()


class StuckWatch:
    """Reward-stagnation detector for one stream.

    Long-run baseline: an EMA of per-step reward with half-life
    `baseline_halflife_s` of wall clock (time-weighted, so a fast game and a
    slow game mean the same thing by "the last half hour").
    Short window: the mean reward over the last `window_s`.

    STUCK when, after `warmup_s`, the short-window mean has stayed at or below
    `ratio x baseline` for a full `window_s`. Relative on purpose: curiosity
    decays as the world model learns, so an absolute floor would eventually
    call every stream stuck.

    A non-positive baseline (nothing has ever paid on this stream) never
    reads as stuck: "nothing changed" is not evidence of a trap.
    """

    def __init__(self, window_s: float = 180.0, ratio: float = 0.25,
                 warmup_s: float = 600.0, baseline_halflife_s: float = 1800.0,
                 max_rewinds_without_recovery: int = 3,
                 clock: Callable[[], float] = time.monotonic):
        self.window_s = float(window_s)
        self.ratio = float(ratio)
        self.warmup_s = float(warmup_s)
        self.halflife = float(baseline_halflife_s)
        self.max_rewinds = int(max_rewinds_without_recovery)
        self.clock = clock
        self._t0 = clock()
        self._win: Deque[Tuple[float, float]] = deque()
        self._win_sum = 0.0
        self._baseline: Optional[float] = None
        self._t_last: Optional[float] = None
        self._low_since: Optional[float] = None
        self._cool_until = 0.0
        self._rewinds_since_recovery = 0
        self._high_since: Optional[float] = None
        # telemetry
        self.rewinds = 0
        self.fresh_starts = 0

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    def window_mean(self) -> Optional[float]:
        return (self._win_sum / len(self._win)) if self._win else None

    def observe(self, reward: float) -> str:
        """Feed one step's reward; returns NONE, REWIND or FRESH."""
        now = self.clock()
        r = float(reward)
        if not math.isfinite(r):
            return NONE
        # time-weighted EMA baseline
        if self._baseline is None:
            self._baseline = r
        else:
            dt = max(0.0, now - (self._t_last if self._t_last is not None else now))
            a = 1.0 - 0.5 ** (dt / self.halflife) if self.halflife > 0 else 1.0
            self._baseline += a * (r - self._baseline)
        self._t_last = now
        self._win.append((now, r))
        self._win_sum += r
        while self._win and now - self._win[0][0] > self.window_s:
            self._win_sum -= self._win.popleft()[1]

        if now - self._t0 < self.warmup_s or now < self._cool_until:
            return NONE
        base = self._baseline
        mean = self.window_mean()
        if base is None or mean is None or base <= 0.0:
            self._low_since = None
            return NONE
        if mean <= self.ratio * base:
            self._high_since = None
            if self._low_since is None:
                self._low_since = now
            if now - self._low_since >= self.window_s:
                return self._fire(now)
        else:
            self._low_since = None
            # RECOVERY: above the line for a whole window resets the count
            if self._high_since is None:
                self._high_since = now
            elif now - self._high_since >= self.window_s:
                self._rewinds_since_recovery = 0
        return NONE

    def _fire(self, now: float) -> str:
        self._low_since = None
        self._cool_until = now + self.window_s
        # the slump is history: judge the restored state on its own window
        self._win.clear()
        self._win_sum = 0.0
        if self._rewinds_since_recovery >= self.max_rewinds:
            self._rewinds_since_recovery = 0
            self.fresh_starts += 1
            return FRESH
        self._rewinds_since_recovery += 1
        self.rewinds += 1
        return REWIND

    def stats(self) -> dict:
        return {"rewinds": self.rewinds, "fresh_starts": self.fresh_starts,
                "baseline": self._baseline, "window_mean": self.window_mean(),
                "rewinds_since_recovery": self._rewinds_since_recovery}
