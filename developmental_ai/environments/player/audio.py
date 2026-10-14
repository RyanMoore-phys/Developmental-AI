"""Game sound for the Player Adapter: what comes out of the game's speakers.

docs/UNIVERSAL_GAMES.md §11. Each seat routes its game's audio to its own
PulseAudio null sink; `ParecAudio` records that sink's monitor (stereo int16)
on a reader thread. `read()` returns everything since the previous call, so
a decision hears exactly the sound of the time it spanned — the same contract
the real-time Doom adapter gives `universal.stereo_mel`.

Sources:
    ParecAudio   `parec` on <sink>.monitor (P0 measures latency / jitter)
    FakeAudio    samples from a callable (tests)
"""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np


class AudioSource:
    sample_rate = 22050

    def read(self) -> Optional[np.ndarray]:   # (N, 2) int16, or None
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeAudio(AudioSource):
    def __init__(self, make: Callable[[], Optional[np.ndarray]], sample_rate=22050):
        self._make = make
        self.sample_rate = int(sample_rate)

    def read(self):
        return self._make()


class ParecAudio(AudioSource):
    def __init__(self, sink: str, sample_rate: int = 22050, max_seconds: float = 2.0):
        self.sample_rate = int(sample_rate)
        self._cap = int(max_seconds * self.sample_rate) * 4      # bytes, stereo s16
        self._buf = deque()
        self._bytes = 0
        self._lock = threading.Lock()
        self._p = subprocess.Popen(
            ["parec", "-d", f"{sink}.monitor", "--format=s16le",
             f"--rate={self.sample_rate}", "--channels=2", "--latency-msec=20"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._t = threading.Thread(target=self._run, daemon=True, name="player-audio")
        self._t.start()

    def _run(self):
        out = self._p.stdout
        while True:
            chunk = out.read(4 * 512)
            if not chunk:
                return
            with self._lock:
                self._buf.append(chunk)
                self._bytes += len(chunk)
                while self._bytes > self._cap and self._buf:   # keep the newest 2 s
                    self._bytes -= len(self._buf.popleft())

    def read(self):
        with self._lock:
            if not self._buf:
                return None
            raw = b"".join(self._buf)
            self._buf.clear()
            self._bytes = 0
        n = len(raw) // 4
        if n == 0:
            return None
        return np.frombuffer(raw[:n * 4], dtype=np.int16).reshape(n, 2)

    def close(self):
        try:
            self._p.terminate()
            self._p.wait(timeout=3)
        except Exception:
            pass
