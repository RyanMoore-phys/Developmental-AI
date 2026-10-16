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
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

SEAT_MARKER = "SKYBOT_SEAT"   # same name as launchers.SEAT_MARKER


class SeatError(RuntimeError):
    pass


X_TMP = "/tmp"                     # where X servers keep .X<n>-lock / .X11-unix
OWNER_MARKER = "SKYBOT_SEAT_OWNER"  # on a seat's Xvfb: the trainer pid that owns it


def display_in_use(n: int, tmp: Optional[str] = None) -> bool:
    tmp = tmp or X_TMP
    return os.path.exists(f"{tmp}/.X{n}-lock") or os.path.exists(f"{tmp}/.X11-unix/X{n}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_env(pid: int, proc: str = "/proc") -> Dict[str, str]:
    try:
        with open(f"{proc}/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep:
            out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


class Seat:
    def __init__(self, launch: Sequence[str], display: Optional[int] = None,
                 width: int = 640, height: int = 480,
                 audio_sink: Optional[str] = None, gl: bool = False,
                 cpu: Optional[int] = None, cwd: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None, boot_s: float = 3.0,
                 window_manager: bool = True,
                 pre_start: Optional[Callable[[], None]] = None):
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
        self.pre_start = pre_start
        self.refused: Optional[str] = None
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
                self.clear_stale_display()      # raises only on a real conflict
            if shutil.which("Xvfb") is None:
                raise SeatError("Xvfb not installed (provision: xvfb)")
            self._xvfb = subprocess.Popen(
                ["Xvfb", self.display_name, "-screen", "0",
                 f"{self.width}x{self.height}x24", "-nolisten", "tcp"],
                env=dict(os.environ, **{OWNER_MARKER: str(os.getpid())}),
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

    def clear_stale_display(self, tmp: Optional[str] = None, proc: str = "/proc",
                            wait_s: float = 10.0) -> str:
        """A display lock left by a DEAD seat must not refuse forever.

        WHY (2026-10-14, the universal-games sweep): a trainer that is
        SIGKILLed or OOM-killed leaves its Xvfb running (and its game, in its
        own session), so /tmp/.X<n>-lock stays. The old check refused every
        locked display, so after one hard crash both doom_player seats refused
        on every relaunch — a guard-becomes-latch (CLAUDE.md §4.1) with nothing
        that could ever re-open it. Each seat's Xvfb now carries
        SKYBOT_SEAT_OWNER=<trainer pid>, and a lock is cleared when:

          * its X server is dead              -> stale files, removed
          * its owner trainer is dead         -> orphan: game + Xvfb killed
          * its owner is THIS process         -> our own abandoned seat (a
                                                 rebuild racing the async
                                                 close): wait, then kill
        and still REFUSED when the owner is another LIVE trainer, or the X
        server is not a seat's at all (exit: stop it, or move display_base).
        Returns what it did.
        """
        tmp = tmp or X_TMP
        n = self.display
        lock, sock = f"{tmp}/.X{n}-lock", f"{tmp}/.X11-unix/X{n}"
        try:
            xpid = int(open(lock).read().strip())
        except (OSError, ValueError):
            xpid = 0

        def _remove():
            for f in (lock, sock):
                try:
                    os.unlink(f)
                except OSError:
                    pass

        if xpid <= 0 or not _pid_alive(xpid):
            _remove()
            logger.warning("seat :%s: removed a stale X lock (server gone)", n)
            return "stale"
        owner = _read_env(xpid, proc).get(OWNER_MARKER)
        if not owner or not owner.isdigit():
            raise SeatError(f"display :{n} is held by an X server SkyBot did not start "
                            f"(pid {xpid}) — stop it or pick another display_base")
        owner_pid = int(owner)
        if owner_pid != os.getpid() and _pid_alive(owner_pid):
            raise SeatError(f"display :{n} belongs to a running trainer (pid {owner_pid}) "
                            f"— a seat never shares a screen")
        if owner_pid == os.getpid():
            t0 = time.time()
            while display_in_use(n, tmp) and time.time() - t0 < wait_s:
                time.sleep(0.2)
            if not display_in_use(n, tmp):
                return "released"
        # an orphan (owner dead) or our own seat that would not close: its game
        # first (it carries the seat marker), then its X server
        self._kill_marked(proc=proc)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(xpid, sig)
            except OSError:
                pass
            t0 = time.time()
            while _pid_alive(xpid) and time.time() - t0 < 3.0:
                time.sleep(0.1)
            if not _pid_alive(xpid):
                break
        _remove()
        logger.warning("seat :%s: cleared an orphaned seat (Xvfb pid %d, owner %d)",
                       n, xpid, owner_pid)
        return "orphan"

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

    def game_env(self) -> Dict[str, str]:
        """The game's environment: the seat's, plus the marker every process
        it spawns inherits — how teardown finds what Steam/browsers re-parent."""
        e = self.env()
        if self.display is not None:
            e[SEAT_MARKER] = str(self.display)
        return e

    def start_game(self) -> None:
        # A refusal here does NOT raise: the game simply stays down, the
        # adapter sees a dead game and retries on its sliding restart budget,
        # and this check runs again each time — so closing the conflicting
        # program is the whole exit (CLAUDE.md §4.1). Raising would crash the
        # trainer and the supervisor would crash-loop on it.
        self.refused = None
        if self.pre_start is not None:
            try:
                self.pre_start()
            except RuntimeError as ex:
                self.refused = str(ex)
                self.game = None
                logger.error("seat %s: game not started: %s", self.display_name, ex)
                return
        argv = list(self.launch)
        if self.gl:
            argv = ["vglrun", "-d", "egl"] + argv
        if self.cpu is not None and shutil.which("taskset"):
            argv = ["taskset", "-c", str(int(self.cpu))] + argv
        self.game = subprocess.Popen(argv, env=self.game_env(), cwd=self.cwd,
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
        if g is not None and g.poll() is None:
            try:
                os.killpg(os.getpgid(g.pid), signal.SIGTERM)
                g.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(g.pid), signal.SIGKILL)
                except Exception:
                    pass
        # Steam and browsers re-parent their children (the actual game, the
        # renderers) out of the group above; without this a restart would
        # leave the old game running on this display behind the new one.
        self._kill_marked()

    def _kill_marked(self, proc: str = "/proc") -> int:
        """SIGTERM then SIGKILL every process of ours carrying this seat's marker."""
        if self.display is None or not os.path.isdir(proc):
            return 0
        mine = str(self.display)
        uid, me = os.getuid(), os.getpid()
        pids = []
        for pid in os.listdir(proc):
            if not pid.isdigit() or int(pid) == me:
                continue
            try:
                if os.stat(f"{proc}/{pid}").st_uid != uid:
                    continue
                with open(f"{proc}/{pid}/environ", "rb") as f:
                    env = f.read().split(b"\0")
            except OSError:
                continue
            if f"{SEAT_MARKER}={mine}".encode() in env:
                pids.append(int(pid))
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            if sig == signal.SIGTERM and pids:
                time.sleep(1.0)
        return len(pids)

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
