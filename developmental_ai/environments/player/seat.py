"""A seat: one virtual screen, one pair of speakers, one running game.

docs/UNIVERSAL_GAMES.md §11. Every Player Adapter stream gets its own seat:

    Xvfb :<display>          the game's screen (nothing else lives on it)
    PulseAudio null sink     the game's speakers (PULSE_SINK for the game)
    the game process         unmodified; optional `vglrun -d egl` (GPU GL,
                             as the MineRL clients use) and a CPU pin

ISOLATION IS THE SAFETY ARGUMENT for synthetic input: SkyBot's keyboard and
mouse act on the seat's own display, so they cannot reach the host desktop or
another stream. A seat therefore REFUSES a display number that is already in
use (a lock file it did not create) rather than share it.

Teardown is total and idempotent — game, window manager, Xvfb, audio sink —
and is also registered at interpreter exit, so a crashed trainer cannot leave
an orphaned game running (cf. the orphaned Java clients fixed in bebd8db).
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import time
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class SeatError(RuntimeError):
    pass


def display_in_use(n: int) -> bool:
    return os.path.exists(f"/tmp/.X{n}-lock") or os.path.exists(f"/tmp/.X11-unix/X{n}")


class Seat:
    def __init__(self, launch: Sequence[str], display: Optional[int] = None,
                 width: int = 640, height: int = 480,
                 audio_sink: Optional[str] = None, gl: bool = False,
                 cpu: Optional[int] = None, cwd: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None, boot_s: float = 3.0,
                 window_manager: bool = True):
        if not launch:
            raise SeatError("seat: empty launch command")
        self.launch = list(launch)
        self.display = display
        self.width, self.height = int(width), int(height)
        self.audio_sink = audio_sink
        self.gl, self.cpu, self.cwd = bool(gl), cpu, cwd
        self.boot_s = float(boot_s)
        self.window_manager = bool(window_manager)
        self.extra_env = dict(env or {})
        self._xvfb: Optional[subprocess.Popen] = None
        self._wm: Optional[subprocess.Popen] = None
        self.game: Optional[subprocess.Popen] = None
        self._sink_module: Optional[str] = None
        self.restarts = 0
        self._closed = False
        atexit.register(self.close)

    # ------------------------------------------------------------ env
    @property
    def display_name(self) -> Optional[str]:
        return None if self.display is None else f":{self.display}"

    def env(self) -> Dict[str, str]:
        e = dict(os.environ)
        if self.display is not None:
            e["DISPLAY"] = self.display_name
        if self.audio_sink:
            e["PULSE_SINK"] = self.audio_sink
            e.setdefault("SDL_AUDIODRIVER", "pulse")
        e.update(self.extra_env)
        return e

    # ------------------------------------------------------------ start
    def start(self) -> None:
        if self.display is not None:
            if display_in_use(self.display):
                raise SeatError(f"display :{self.display} is already in use — a seat "
                                f"never shares a screen (pick another display_base)")
            if shutil.which("Xvfb") is None:
                raise SeatError("Xvfb not installed (provision: xvfb)")
            self._xvfb = subprocess.Popen(
                ["Xvfb", self.display_name, "-screen", "0",
                 f"{self.width}x{self.height}x24", "-nolisten", "tcp"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(50):
                if display_in_use(self.display):
                    break
                time.sleep(0.1)
            if self.window_manager and shutil.which("openbox"):
                self._wm = subprocess.Popen(["openbox"], env=self.env(),
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
        if self.audio_sink:
            self._load_sink()
        self.start_game()

    def _load_sink(self) -> None:
        if shutil.which("pactl") is None:
            raise SeatError("pactl not installed (provision: pulseaudio)")
        subprocess.run(["pulseaudio", "--start", "--exit-idle-time=-1"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r = subprocess.run(["pactl", "load-module", "module-null-sink",
                            f"sink_name={self.audio_sink}"],
                           capture_output=True, text=True)
        self._sink_module = r.stdout.strip() or None
        if self._sink_module is None:
            logger.warning("seat %s: audio sink %s not created (%s) — SkyBot "
                           "will be deaf in this game", self.display_name,
                           self.audio_sink, r.stderr.strip())

    def start_game(self) -> None:
        argv = list(self.launch)
        if self.gl:
            argv = ["vglrun", "-d", "egl"] + argv
        if self.cpu is not None and shutil.which("taskset"):
            argv = ["taskset", "-c", str(int(self.cpu))] + argv
        self.game = subprocess.Popen(argv, env=self.env(), cwd=self.cwd,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     start_new_session=True)
        time.sleep(self.boot_s)

    # ------------------------------------------------------------ health
    def game_alive(self) -> bool:
        return self.game is not None and self.game.poll() is None

    def restart_game(self) -> None:
        """What a person does with a crashed or wedged game: quit, relaunch."""
        self._stop_game()
        self.restarts += 1
        self.start_game()

    # ------------------------------------------------------------ stop
    def _stop_game(self) -> None:
        g, self.game = self.game, None
        if g is None or g.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(g.pid), signal.SIGTERM)
            g.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(g.pid), signal.SIGKILL)
            except Exception:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_game()
        for p in (self._wm, self._xvfb):
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except Exception:
                    p.kill()
        if self._sink_module:
            subprocess.run(["pactl", "unload-module", self._sink_module],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
