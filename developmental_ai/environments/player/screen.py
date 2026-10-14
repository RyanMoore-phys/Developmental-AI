"""Screen capture for the Player Adapter: the game's screen at its own rate.

docs/UNIVERSAL_GAMES.md §11. A `FramePump` grabs the seat's display at the
game's frame rate on its own thread — the game never waits for SkyBot, and
neither does the capture. It keeps the NEWEST frame for the next decision,
hands EVERY frame to the viewer sink (the game at its own speed), and tracks
when the picture last CHANGED: a game whose screen has not changed for many
seconds while SkyBot holds inputs has frozen or hung (health check).

Sources:
    MssScreen    MIT-SHM grab of an X display via `mss` (P0 measures its rate)
    FakeScreen   frames from a callable (tests)
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np


class ScreenSource:
    def grab(self) -> np.ndarray:            # HxWx3 uint8
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeScreen(ScreenSource):
    def __init__(self, make: Callable[[int], np.ndarray]):
        self._make = make
        self.n = 0

    def grab(self):
        self.n += 1
        return self._make(self.n)


class MssScreen(ScreenSource):
    def __init__(self, display: str, width: int, height: int):
        import mss
        self._sct = mss.mss(display=display)
        self._mon = {"left": 0, "top": 0, "width": int(width), "height": int(height)}

    def grab(self):
        a = np.asarray(self._sct.grab(self._mon))       # BGRA
        return np.ascontiguousarray(a[:, :, 2::-1])     # -> RGB

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass


def _signature(frame: np.ndarray) -> np.ndarray:
    """A cheap fingerprint for 'did the picture change' (8x8 luma means)."""
    h, w = frame.shape[:2]
    ys, xs = max(1, h // 8), max(1, w // 8)
    f = frame[:ys * 8, :xs * 8].astype(np.float32).mean(axis=2)
    return f.reshape(8, ys, 8, xs).mean(axis=(1, 3))


class FramePump:
    def __init__(self, source: ScreenSource, fps: float = 35.0,
                 change_threshold: float = 1.0):
        self.source = source
        self.fps = float(fps)
        self.change_threshold = float(change_threshold)
        self.cv = threading.Condition()
        self.count = 0
        self.latest: Optional[np.ndarray] = None
        self.last_change_t = time.monotonic()
        self.late = 0
        self.error: Optional[BaseException] = None
        self.sink: Optional[Callable[[np.ndarray, float], None]] = None
        self._sig = None
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self) -> None:
        self._t = threading.Thread(target=self._run, daemon=True, name="player-screen")
        self._t.start()

    def _run(self) -> None:
        period = 1.0 / self.fps
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                f = self.source.grab()
            except Exception as e:                    # display gone
                self.error = e
                with self.cv:
                    self.cv.notify_all()
                return
            sig = _signature(f)
            now = time.monotonic()
            if self._sig is None or np.abs(sig - self._sig).mean() > self.change_threshold:
                self.last_change_t = now
            self._sig = sig
            with self.cv:
                self.latest = f
                self.count += 1
                self.cv.notify_all()
            sink = self.sink
            if sink is not None:
                try:
                    sink(f, period)
                except Exception:
                    pass
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                self.late += 1
                if d < -0.5:
                    nxt = time.perf_counter()

    def wait_newer(self, seen: int, timeout: float) -> Tuple[Optional[np.ndarray], int]:
        """Block until a frame newer than `seen` exists; (frame, count)."""
        with self.cv:
            ok = self.cv.wait_for(lambda: self.count > seen or self.error is not None,
                                  timeout=timeout)
            if self.error is not None:
                raise RuntimeError(f"screen capture died: {self.error}")
            if not ok:
                raise RuntimeError(f"no new frame in {timeout:.0f}s")
            return self.latest, self.count

    def frozen_for(self) -> float:
        return time.monotonic() - self.last_change_t

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
        self.source.close()
