"""How a seat starts its game: a plain executable, a Steam game, or a web page.

docs/UNIVERSAL_GAMES.md §11.1. A Player Adapter game YAML says how to launch
the game with a `launcher:` block (or the older top-level `launch:` list,
which is still the `exec` launcher):

    launcher: {type: exec, argv: [chocolate-doom, -iwad, roms/doom/freedoom2.wad]}

    launcher: {type: steam, appid: 70}                     # through the Steam client
    launcher: {type: steam, appid: 70, exe: Half-Life/hl_linux}   # the game binary itself

    launcher: {type: browser, url: "https://play2048.co/"}  # kiosk app window

`{stream}`, `{display}` and `{name}` are filled per seat in every string.

WHAT EACH ONE COSTS, stated so nobody finds out on the host:

  * steam, CLIENT mode (no `exe`): the Steam client launches the game, so DRM,
    Proton (Windows games) and cloud saves all work. But Steam is ONE client
    per Linux user, and a second `steam -applaunch` is forwarded to the one
    already running — whose game would then open on THAT client's display,
    not this seat's. So client mode is refused for `streams` > 1, and a seat
    refuses to start while a Steam client this seat did not start is running
    (exit: close that client). Steam must already be logged in on the host
    once ("remember me"); nothing here can type a password or a Guard code.
  * steam, DIRECT mode (`exe`): runs the installed binary with SteamAppId set,
    no client. Many streams are fine. Works for DRM-free games; games that
    need the client running fail to start, and that shows as a dead game.
  * browser: Chrome/Chromium (preferred) or Firefox in kiosk app mode with a
    private profile PER SEAT, so streams never share cookies or storage.
    Browser shortcuts that leave the game (close, new window, devtools, print,
    back, address bar) are one-way doors (CLAUDE.md §4.5), so any Ctrl/Alt
    chord and any F-key in a browser game's controller is refused at compose
    time unless the YAML lists it under `launcher.allow_keys`.

Every game process gets SKYBOT_SEAT=<display> in its environment. Steam and
browsers re-parent their children out of the process group the seat started,
so the seat's teardown also kills every process carrying its marker (seat.py).
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

TYPES = ("exec", "steam", "browser")
CHROME_FAMILY = ("google-chrome-stable", "google-chrome", "chromium",
                 "chromium-browser")
FIREFOX = ("firefox",)
SEAT_MARKER = "SKYBOT_SEAT"

_CTRL_ALT = frozenset({"Control_L", "Control_R", "Alt_L", "Alt_R"})
_FKEYS = frozenset(f"F{i}" for i in range(1, 25))


class LauncherError(ValueError):
    pass


@dataclass
class LaunchSpec:
    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    kind: str = "exec"
    # called by the seat right before the game starts; raises SeatError-like
    # RuntimeError to refuse (e.g. a foreign Steam client is running)
    pre_start: Optional[Callable[[], None]] = None


# ------------------------------------------------------------------ parse
def launcher_of(game: Dict[str, Any]) -> Dict[str, Any]:
    """The game's launcher block, with the legacy `launch:` list mapped to exec."""
    lb = game.get("launcher")
    if lb is None:
        return {"type": "exec", "argv": list(game.get("launch") or [])}
    if not isinstance(lb, dict):
        raise LauncherError(f"launcher must be a mapping, got {type(lb).__name__}")
    lb = dict(lb)
    lb.setdefault("type", "exec")
    if lb["type"] == "exec" and "argv" not in lb and game.get("launch"):
        lb["argv"] = list(game["launch"])
    return lb


def validate_launcher(game: Dict[str, Any], slots=None) -> None:
    """Compose-time checks. `slots` = parsed controller slots (browser keys)."""
    lb = launcher_of(game)
    t = lb.get("type")
    if t not in TYPES:
        raise LauncherError(f"launcher.type {t!r} unknown (one of {list(TYPES)})")
    if game.get("launcher") is not None and game.get("launch"):
        raise LauncherError("give either `launcher:` or the old `launch:` list, not both")
    streams = int(game.get("streams", 1))
    if t == "exec":
        if not lb.get("argv"):
            raise LauncherError("exec launcher needs a non-empty argv (or `launch:`)")
    elif t == "steam":
        try:
            appid = int(lb.get("appid"))
        except (TypeError, ValueError):
            raise LauncherError("steam launcher needs a numeric `appid`")
        if appid <= 0:
            raise LauncherError("steam launcher needs a positive `appid`")
        if not lb.get("exe") and streams > 1:
            raise LauncherError(
                f"steam CLIENT mode runs one game per Linux user, but streams={streams}: "
                "a second seat's launch would be forwarded to the first seat's client "
                "and open on the wrong screen. Use streams: 1, or `exe:` (direct mode) "
                "for a DRM-free game")
    elif t == "browser":
        url = str(lb.get("url") or "")
        if not re.match(r"^(https?|file)://", url):
            raise LauncherError(f"browser launcher needs an http(s):// or file:// url, got {url!r}")
        b = lb.get("browser", "auto")
        if b not in ("auto", "chrome", "chromium", "firefox"):
            raise LauncherError(f"launcher.browser {b!r} unknown (auto|chrome|chromium|firefox)")
        if slots is not None:
            probs = browser_key_problems(slots, lb.get("allow_keys") or [])
            if probs:
                raise LauncherError("; ".join(probs))


def browser_key_problems(slots: Sequence[Any], allow: Sequence[str]) -> List[str]:
    """Browser shortcuts leave the game and nothing in a slot can come back."""
    allow = set(allow or [])
    out = []
    for i, s in enumerate(slots):
        keys = set(getattr(s, "keys", ()) or ())
        bad = sorted(k for k in keys & (_CTRL_ALT | _FKEYS) if k not in allow)
        if bad:
            out.append(f"controller slot {i}: {bad} in a browser game — Ctrl/Alt "
                       "chords and F-keys are browser shortcuts (close, new window, "
                       "devtools, back) that leave the game: a one-way door. List a "
                       "key under launcher.allow_keys only if the game needs it")
    return out


# ------------------------------------------------------------------ steam
def steam_libraries(home: Optional[str] = None) -> List[str]:
    """Every steamapps directory Steam knows about (libraryfolders.vdf)."""
    home = home or os.path.expanduser("~")
    roots = [os.path.join(home, ".steam", "steam"),
             os.path.join(home, ".local", "share", "Steam"),
             os.path.join(home, ".steam", "debian-installation")]
    libs: List[str] = []
    for r in roots:
        sa = os.path.join(r, "steamapps")
        if os.path.isdir(sa):
            libs.append(os.path.realpath(sa))
        vdf = os.path.join(sa, "libraryfolders.vdf")
        try:
            with open(vdf) as f:
                for m in re.finditer(r'"path"\s+"([^"]+)"', f.read()):
                    p = os.path.join(m.group(1), "steamapps")
                    if os.path.isdir(p):
                        libs.append(os.path.realpath(p))
        except OSError:
            pass
    seen, out = set(), []
    for p in libs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def steam_app_installed(appid: int, libs: Optional[List[str]] = None) -> Optional[str]:
    for lib in (libs if libs is not None else steam_libraries()):
        if os.path.isfile(os.path.join(lib, f"appmanifest_{int(appid)}.acf")):
            return lib
    return None


def _proc_env(pid: str, proc: str = "/proc") -> Dict[str, str]:
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


def foreign_steam_clients(display: int, proc: str = "/proc") -> List[str]:
    """Steam clients of this user NOT started by this seat (pid list)."""
    if not os.path.isdir(proc):
        return []
    uid, out = os.getuid(), []
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"{proc}/{pid}").st_uid != uid:
                continue
            with open(f"{proc}/{pid}/comm") as f:
                comm = f.read().strip()
        except OSError:
            continue
        if comm not in ("steam", "steam.sh"):
            continue
        if _proc_env(pid, proc).get(SEAT_MARKER) != str(display):
            out.append(pid)
    return out


def _steam_spec(lb, fmt, display, which, home) -> LaunchSpec:
    appid = int(lb["appid"])
    args = [fmt(a) for a in (lb.get("args") or [])]
    env = {"SteamAppId": str(appid), "SteamGameId": str(appid)}
    exe = lb.get("exe")
    if exe:
        exe = fmt(exe)
        if not os.path.isabs(exe):
            lib = steam_app_installed(appid, steam_libraries(home))
            base = os.path.join(lib, "common") if lib else os.path.join(
                home or os.path.expanduser("~"), ".steam", "steam", "steamapps", "common")
            exe = os.path.join(base, exe)
        return LaunchSpec([exe] + args, env=env, cwd=os.path.dirname(exe), kind="steam-direct")
    steam = lb.get("steam_bin") or which("steam") or "steam"

    def refuse_foreign():
        pids = foreign_steam_clients(display)
        if pids:
            raise RuntimeError(
                f"a Steam client this seat did not start is running (pid {', '.join(pids)}): "
                f"`-applaunch` would be forwarded to it and the game would open on its "
                f"screen, not :{display}. Close that Steam client (steam -shutdown) and "
                f"the seat starts on its next restart")
    return LaunchSpec([steam, "-silent", "-applaunch", str(appid)] + args,
                      env=env, kind="steam-client", pre_start=refuse_foreign)


# ---------------------------------------------------------------- browser
def find_browser(pref: str = "auto", which: Callable = shutil.which) -> Optional[str]:
    order = {"auto": CHROME_FAMILY + FIREFOX,
             "chrome": ("google-chrome-stable", "google-chrome"),
             "chromium": ("chromium", "chromium-browser"),
             "firefox": FIREFOX}[pref]
    for name in order:
        p = which(name)
        if p:
            return p
    return None


def _is_snap(path: str) -> Optional[str]:
    """Snap name if `path` is a snap launcher (snaps cannot see /workspace).

    Covers /snap/bin/<name> (a symlink to /usr/bin/snap) and Ubuntu's
    /usr/bin/chromium-browser, a small shell script that execs the snap.
    """
    rp = os.path.realpath(path)
    if path.startswith("/snap/") or rp.startswith("/snap/") or rp == "/usr/bin/snap":
        return os.path.basename(path).split(".")[0]
    try:
        if os.path.getsize(rp) < 16384:
            with open(rp, "rb") as f:
                m = re.search(rb"/snap/bin/([A-Za-z0-9_-]+)", f.read())
            if m:
                return m.group(1).decode()
    except OSError:
        pass
    return None


def browser_profile(game_name: str, stream: int, binary: str,
                    root: str = ROOT, home: Optional[str] = None) -> str:
    snap = _is_snap(binary)
    if snap:
        # A snap's home interface reaches only non-hidden dirs under ~/snap/<name>
        return os.path.join(home or os.path.expanduser("~"), "snap", snap,
                            "common", "skybot", f"{game_name}_{stream}")
    return os.path.join(root, "brain_universal", "browser_profiles",
                        f"{game_name}_{stream}")


def _browser_spec(game, lb, fmt, stream, which, root, home) -> LaunchSpec:
    binary = lb.get("browser_bin") or find_browser(lb.get("browser", "auto"), which)
    if not binary:
        raise LauncherError("browser game but no browser installed (google-chrome, "
                            "chromium or firefox) — run the game's install action")
    w = game.get("window") or {}
    W, H = int(w.get("width", 640)), int(w.get("height", 480))
    url = fmt(lb["url"])
    profile = browser_profile(str(game.get("name", "game")), stream, binary, root, home)
    extra = [fmt(a) for a in (lb.get("args") or [])]
    if os.path.basename(binary).startswith("firefox"):
        argv = [binary, "--new-instance", "--profile", profile, "--kiosk",
                "--width", str(W), "--height", str(H)] + extra + [url]
        return LaunchSpec(argv, kind="browser-firefox")
    argv = [binary, f"--user-data-dir={profile}", "--no-first-run",
            "--no-default-browser-check", "--disable-session-crashed-bubble",
            "--hide-crash-restore-bubble", "--disable-infobars",
            "--password-store=basic", "--autoplay-policy=no-user-gesture-required",
            # the seat's window is never "visible" to a compositor; without these
            # Chrome throttles timers and rendering of a backgrounded page
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--window-position=0,0", f"--window-size={W},{H}"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        # Chrome refuses to run as root without it (MAIN_USER may be root)
        argv.append("--no-sandbox")
    if not game.get("gl", False):
        # Xvfb has no GPU: WebGL games need the software rasteriser
        argv += ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    argv += extra + ["--kiosk", f"--app={url}"]
    return LaunchSpec(argv, kind="browser-chrome")


def prepare_profile(spec: LaunchSpec, game: Dict[str, Any]) -> None:
    """Create the per-seat profile dir; Firefox also gets autoplay allowed."""
    prof = None
    for i, a in enumerate(spec.argv):
        if a.startswith("--user-data-dir="):
            prof = a.split("=", 1)[1]
        elif a == "--profile" and i + 1 < len(spec.argv):
            prof = spec.argv[i + 1]
    if not prof:
        return
    os.makedirs(prof, exist_ok=True)
    if spec.kind == "browser-firefox":
        with open(os.path.join(prof, "user.js"), "w") as f:
            f.write('user_pref("media.autoplay.default", 0);\n'
                    'user_pref("browser.shell.checkDefaultBrowser", false);\n'
                    'user_pref("browser.sessionstore.resume_from_crash", false);\n'
                    'user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);\n')


# ------------------------------------------------------------------ resolve
def resolve_launch(game: Dict[str, Any], stream: int, display: int, name: str,
                   root: str = ROOT, which: Callable = shutil.which,
                   home: Optional[str] = None) -> LaunchSpec:
    lb = launcher_of(game)
    t = lb.get("type", "exec")

    def fmt(x: Any) -> str:
        return str(x).format(stream=stream, display=display, name=name)

    if t == "exec":
        spec = LaunchSpec([fmt(a) for a in lb.get("argv") or []], kind="exec")
    elif t == "steam":
        spec = _steam_spec(lb, fmt, display, which, home)
    elif t == "browser":
        spec = _browser_spec(game, lb, fmt, stream, which, root, home)
    else:
        raise LauncherError(f"launcher.type {t!r} unknown")
    if not spec.argv:
        raise LauncherError("empty launch command")
    if game.get("cwd") and spec.cwd is None:
        spec.cwd = str(game["cwd"])
    # SEAT_MARKER is added by Seat.start_game (the game only, never its WM)
    return spec


# ---------------------------------------------------------------- preflight
def preflight(game: Dict[str, Any], which: Callable = shutil.which,
              home: Optional[str] = None, root: str = ROOT) -> List[str]:
    """What is missing on THIS machine to start the game (empty = ready)."""
    miss: List[str] = []
    try:
        lb = launcher_of(game)
    except LauncherError as e:
        return [str(e)]
    t = lb.get("type", "exec")
    if t == "exec":
        argv = lb.get("argv") or ["?"]
        if which(str(argv[0])) is None:
            miss.append(str(argv[0]))
    elif t == "steam":
        appid = int(lb.get("appid") or 0)
        if lb.get("exe"):
            spec = _steam_spec(lb, str, 0, which, home)
            if not os.path.isfile(spec.argv[0]):
                miss.append(f"steam app {appid} binary {spec.argv[0]}")
        else:
            if not (lb.get("steam_bin") or which("steam")):
                miss.append("steam (the Steam client)")
            if not steam_app_installed(appid, steam_libraries(home)):
                miss.append(f"steam app {appid} not installed in any Steam library")
    elif t == "browser":
        if not (lb.get("browser_bin") or find_browser(lb.get("browser", "auto"), which)):
            miss.append("a browser (google-chrome, chromium or firefox)")
    for tool in ["Xvfb"] + (["pactl", "parec"]
                            if (game.get("audio") or {}).get("enabled", True) else []):
        if which(tool) is None:
            miss.append(tool)
    if game.get("gl") and which("vglrun") is None:
        miss.append("vglrun (gl: true)")
    # `requires:` — files the game cannot start without. A missing one is the
    # silent failure the sweep found (2026-10-14): chocolate-doom without its
    # WAD exits at boot, the adapter spends its restart budget, and SkyBot
    # trains on an empty black screen.
    for r in game.get("requires") or []:
        path = str(r) if os.path.isabs(str(r)) else os.path.join(root, str(r))
        if not os.path.exists(path):
            miss.append(f"file {r}")
    # game files the YAML declares must already be in place
    for f in (game.get("install") or {}).get("files") or []:
        dest = f.get("dest") if isinstance(f, dict) else None
        if dest and not os.path.exists(dest if os.path.isabs(dest) else os.path.join(root, dest)):
            miss.append(f"file {dest}")
    return miss
