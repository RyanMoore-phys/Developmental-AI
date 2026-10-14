"""Held keyboard/mouse input for the Player Adapter.

docs/UNIVERSAL_GAMES.md §11. SkyBot's decision is a controller SLOT that it
HOLDS until its next decision. `HeldInputs` turns slot changes into exactly
the press/release events a person's hands would make (release what is no
longer held, press what newly is), and a 60 Hz `InputPump` integrates the
held mouse velocity into relative motion — fractional pixels carried, so
"turn at 400 px/s" turns 400 px per second whatever the decision rate.

Every event passes the GAME's WHITELIST (only keys/buttons named in its
YAML); anything else is dropped and counted, never sent. `release_all()`
runs on every stop, restart, rewind and at interpreter exit: a key left
held down in a game that keeps running is a stuck key forever.

Backends (the P0 spike decides which is the default on the host):
    XTestBackend   python-xlib XTest on the seat's own X display
    FakeInputBackend   records events (tests; no X needed)
"""

from __future__ import annotations

import threading
import time
from typing import FrozenSet, List, Optional, Sequence, Tuple

from developmental_ai.environments.player.controller import Slot, whitelist


class InputBackend:
    def key(self, name: str, down: bool) -> None:
        raise NotImplementedError

    def button(self, n: int, down: bool) -> None:
        raise NotImplementedError

    def move(self, dx: int, dy: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeInputBackend(InputBackend):
    def __init__(self):
        self.events: List[Tuple] = []
        self.lock = threading.Lock()

    def key(self, name, down):
        with self.lock:
            self.events.append(("key", name, bool(down)))

    def button(self, n, down):
        with self.lock:
            self.events.append(("button", int(n), bool(down)))

    def move(self, dx, dy):
        with self.lock:
            self.events.append(("move", int(dx), int(dy)))

    def moved(self) -> Tuple[int, int]:
        with self.lock:
            return (sum(e[1] for e in self.events if e[0] == "move"),
                    sum(e[2] for e in self.events if e[0] == "move"))


class XTestBackend(InputBackend):
    """Synthetic input on ONE X display (the seat's), via the XTest extension.
    Relative motion uses MotionNotify with detail=True — what pointer-grabbing
    games (mouselook) read. Whether that is enough for every game is a P0
    question; a per-seat uinput backend is the fallback."""

    def __init__(self, display: str):
        from Xlib import X, XK, display as xdisplay
        from Xlib.ext import xtest
        self._X, self._XK, self._xt = X, XK, xtest
        self.d = xdisplay.Display(display)
        self._lock = threading.Lock()
        self._codes = {}

    def _code(self, name: str) -> int:
        c = self._codes.get(name)
        if c is None:
            c = self.d.keysym_to_keycode(self._XK.string_to_keysym(name))
            if not c:
                raise ValueError(f"no keycode for keysym {name!r} on this display")
            self._codes[name] = c
        return c

    def key(self, name, down):
        with self._lock:
            self._xt.fake_input(self.d, self._X.KeyPress if down else self._X.KeyRelease,
                                self._code(name))
            self.d.sync()

    def button(self, n, down):
        with self._lock:
            self._xt.fake_input(self.d, self._X.ButtonPress if down else self._X.ButtonRelease,
                                int(n))
            self.d.sync()

    def move(self, dx, dy):
        with self._lock:
            self._xt.fake_input(self.d, self._X.MotionNotify, detail=True,
                                x=int(dx), y=int(dy))
            self.d.sync()

    def close(self):
        try:
            self.d.close()
        except Exception:
            pass


class HeldInputs:
    """The held-input state machine. Thread-safe: the actor sets slots, the
    pump ticks the mouse."""

    def __init__(self, backend: InputBackend, slots: Sequence[Slot]):
        self.backend = backend
        self.slots = list(slots)
        self.allowed_keys, self.allowed_buttons = whitelist(self.slots)
        self._lock = threading.Lock()
        self._keys: FrozenSet[str] = frozenset()
        self._buttons: FrozenSet[int] = frozenset()
        self._vx = self._vy = 0.0
        self._cx = self._cy = 0.0         # fractional motion carried
        self.dropped = 0                  # events refused by the whitelist
        self.slot = 0

    # -- guarded sends -------------------------------------------------------
    def _key(self, name, down):
        if name not in self.allowed_keys:
            self.dropped += 1
            return
        self.backend.key(name, down)

    def _button(self, n, down):
        if n not in self.allowed_buttons:
            self.dropped += 1
            return
        self.backend.button(n, down)

    # -- the decision ---------------------------------------------------------
    def set_slot(self, i: int) -> None:
        i = int(i)
        s = self.slots[i] if 0 <= i < len(self.slots) else self.slots[0]
        with self._lock:
            for k in sorted(self._keys - s.keys):
                self._key(k, False)
            for b in sorted(self._buttons - s.buttons):
                self._button(b, False)
            for k in sorted(s.keys - self._keys):
                self._key(k, True)
            for b in sorted(s.buttons - self._buttons):
                self._button(b, True)
            self._keys, self._buttons = s.keys, s.buttons
            if (s.mouse_vx, s.mouse_vy) != (self._vx, self._vy):
                self._cx = self._cy = 0.0
            self._vx, self._vy = s.mouse_vx, s.mouse_vy
            self.slot = i if 0 <= i < len(self.slots) else 0

    def tick(self, dt: float) -> None:
        """Integrate the held mouse velocity over dt seconds."""
        with self._lock:
            if self._vx == 0.0 and self._vy == 0.0:
                return
            self._cx += self._vx * dt
            self._cy += self._vy * dt
            mx, my = int(self._cx), int(self._cy)
            if mx or my:
                self._cx -= mx
                self._cy -= my
                self.backend.move(mx, my)

    def release_all(self) -> None:
        with self._lock:
            for k in sorted(self._keys):
                self._key(k, False)
            for b in sorted(self._buttons):
                self._button(b, False)
            self._keys, self._buttons = frozenset(), frozenset()
            self._vx = self._vy = self._cx = self._cy = 0.0
            self.slot = 0

    def tap(self, keys: Sequence[str], hold_s: float = 0.05) -> None:
        """An ENVIRONMENT-level key tap (quicksave / quickload) — not SkyBot's
        input, so it bypasses the controller whitelist on purpose; only the
        game YAML's `save` keys are ever passed here."""
        with self._lock:
            for k in keys:
                self.backend.key(k, True)
        time.sleep(hold_s)
        with self._lock:
            for k in reversed(list(keys)):
                self.backend.key(k, False)


class InputPump:
    """Applies held mouse velocity at `hz` on its own thread."""

    def __init__(self, held: HeldInputs, hz: float = 60.0):
        self.held = held
        self.hz = float(hz)
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def start(self) -> None:
        self._t = threading.Thread(target=self._run, daemon=True, name="player-input")
        self._t.start()

    def _run(self) -> None:
        period = 1.0 / self.hz
        last = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            try:
                self.held.tick(now - last)
            except Exception as e:            # display gone: report, stop
                self.error = e
                return
            last = now
            time.sleep(period)

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
