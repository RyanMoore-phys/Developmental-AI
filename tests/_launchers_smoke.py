"""Player Adapter launchers: any game — an executable, a Steam game, a web page.

Born 2026-10-14 (docs/UNIVERSAL_GAMES.md §11.1). User decision: the Player
Adapter must launch Steam games and browser games as well as plain
executables. Each launcher has a failure mode that would be silent on the
host, so each is a contract here:

    A. EXEC IS UNCHANGED: the old top-level `launch:` list is the exec
       launcher, placeholders filled per seat; doom_player resolves to the
       exact argv it always had.
    B. STEAM CLIENT MODE CANNOT OPEN ON THE WRONG SCREEN: Steam is one client
       per Linux user and forwards a second -applaunch to the running one, so
       streams > 1 is refused at compose time, and a seat refuses to launch
       while a Steam client it did not start is running — WITHOUT raising
       (the game stays down, the adapter's sliding restart budget retries,
       and the refusal clears once that client is closed: no latch, §4.1).
    C. STEAM DIRECT MODE: the binary under the library that holds the app
       (libraryfolders.vdf honoured), SteamAppId set, cwd = its directory;
       any number of streams.
    D. BROWSER: kiosk app window on the page, a PRIVATE profile per seat
       (streams never share cookies/storage), SwiftShader only when there is
       no GPU, Firefox gets its own argv, a snap browser's profile lives where
       the snap can write; non-http urls refused.
    E. BROWSER SHORTCUTS ARE ONE-WAY DOORS (§4.5): any Ctrl/Alt chord or
       F-key in a browser game's controller is refused unless allow-listed.
    F. TEARDOWN FINDS RE-PARENTED CHILDREN: the game env carries the seat
       marker (the window manager's does not), and teardown kills every
       process carrying it — Steam and browsers move their children out of
       the process group the seat started.
    G. PREFLIGHT NAMES WHAT IS MISSING, per launcher type.
    H. COMPOSE: the shipped example YAMLs validate; a steam-client game with
       2 streams is refused by tools/compose_run_config.py.

Run: PYTHONPATH=. python tests/_launchers_smoke.py
"""
import copy
import os
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.environments.player import launchers as L       # noqa: E402
from developmental_ai.environments.player.controller import validate  # noqa: E402
from developmental_ai.environments.player.seat import Seat            # noqa: E402

NAME = "launchers"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def raises(fn, needle):
    try:
        fn()
    except (L.LauncherError, ValueError) as e:
        return needle in str(e)
    return False


def load(rel):
    return yaml.safe_load(open(os.path.join(ROOT, rel)))["game"]


def which_all(name):
    return "/usr/bin/" + name


def which_none(name):
    return None


def main():
    tmp = tempfile.mkdtemp(prefix="launchers_")
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{NAME}] ALL PASS")


def run(tmp):
    # ---------------------------------------------------------------- A
    dp = load("configs/games/doom_player.yaml")
    sp = L.resolve_launch(dp, 1, 101, "SkyBot1")
    check(sp.kind == "exec" and sp.argv == [str(x) for x in dp["launch"]],
          "A. doom_player's `launch:` list resolves to the same exec argv")
    g = {"name": "x", "launch": ["game", "--seat", "{display}", "--who", "{name}", "{stream}"]}
    check(L.resolve_launch(g, 2, 102, "SkyBot2").argv
          == ["game", "--seat", "102", "--who", "SkyBot2", "2"],
          "A. {display} {name} {stream} filled per seat")
    check(raises(lambda: L.validate_launcher({"launch": ["a"], "launcher": {"type": "exec", "argv": ["b"]}}),
                 "not both"), "A. `launch:` and `launcher:` together refused")

    # ---------------------------------------------------------------- B
    st = load("configs/games/examples/steam_game.yaml")
    sp = L.resolve_launch(st, 0, 120, "SkyBot", which=which_all)
    check(sp.argv[:4] == ["/usr/bin/steam", "-silent", "-applaunch", "70"]
          and sp.env.get("SteamAppId") == "70" and sp.kind == "steam-client",
          "B. client mode: steam -silent -applaunch <appid>, SteamAppId set")
    two = dict(st, streams=2)
    check(raises(lambda: L.validate_launcher(two), "one game per Linux user"),
          "B. client mode with streams 2 refused (would open on the wrong screen)")
    # fake /proc: one steam client of ours WITHOUT the seat marker
    proc = os.path.join(tmp, "proc")
    os.makedirs(os.path.join(proc, "4242"))
    open(os.path.join(proc, "4242", "comm"), "w").write("steam\n")
    open(os.path.join(proc, "4242", "environ"), "wb").write(b"DISPLAY=:0\0HOME=/x\0")
    check(L.foreign_steam_clients(120, proc=proc) == ["4242"],
          "B. a Steam client this seat did not start is detected")
    open(os.path.join(proc, "4242", "environ"), "wb").write(b"SKYBOT_SEAT=120\0")
    check(L.foreign_steam_clients(120, proc=proc) == [],
          "B. the seat's own Steam client (carries its marker) is not foreign")
    calls = {"n": 0}

    def refuse_then_clear():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("a Steam client this seat did not start is running")

    marker = os.path.join(tmp, "started")
    seat = Seat([sys.executable, "-c", f"open({marker!r}, 'w').write('x')"],
                display=None, boot_s=0.0, pre_start=refuse_then_clear)
    seat.start_game()
    check(seat.game is None and "Steam client" in (seat.refused or "")
          and not seat.game_alive(),
          "B. refusal does not raise: the game stays down and says why")
    seat.restart_game()
    if seat.game is not None:
        seat.game.wait(timeout=10)
    check(seat.refused is None and os.path.exists(marker),
          "B. no latch: the next restart re-checks and starts once it is clear")
    seat.close()

    # ---------------------------------------------------------------- C
    home = os.path.join(tmp, "home")
    lib1 = os.path.join(home, ".steam", "steam", "steamapps")
    lib2 = os.path.join(tmp, "bigdisk", "SteamLibrary", "steamapps")
    os.makedirs(lib1)
    os.makedirs(os.path.join(lib2, "common", "Half-Life"))
    open(os.path.join(lib1, "libraryfolders.vdf"), "w").write(
        '"libraryfolders"\n{\n "0" { "path" "%s" }\n "1" { "path" "%s" }\n}\n'
        % (os.path.dirname(lib1), os.path.dirname(lib2)))
    open(os.path.join(lib2, "appmanifest_70.acf"), "w").write('"AppState" {}')
    check(L.steam_app_installed(70, L.steam_libraries(home)) == os.path.realpath(lib2),
          "C. libraryfolders.vdf: the app is found in the second library")
    direct = copy.deepcopy(st)
    direct["launcher"]["exe"] = "Half-Life/hl_linux"
    direct["streams"] = 2
    direct_ok = not raises(lambda: L.validate_launcher(direct), "")
    sp = L.resolve_launch(direct, 1, 121, "SkyBot1", home=home)
    want = os.path.join(os.path.realpath(lib2), "common", "Half-Life", "hl_linux")
    check(sp.argv[0] == want and sp.cwd == os.path.dirname(want)
          and sp.env["SteamAppId"] == "70" and sp.kind == "steam-direct",
          "C. direct mode: the binary in its library, cwd its dir, SteamAppId set")
    check(direct_ok, "C. direct mode allows streams 2")

    # ---------------------------------------------------------------- D
    br = load("configs/games/examples/browser_2048.yaml")
    a = L.resolve_launch(br, 0, 110, "SkyBot", which=which_all, root=tmp)
    b = L.resolve_launch(br, 1, 111, "SkyBot1", which=which_all, root=tmp)
    pa = [x for x in a.argv if x.startswith("--user-data-dir=")]
    pb = [x for x in b.argv if x.startswith("--user-data-dir=")]
    check(a.argv[0] == "/usr/bin/google-chrome-stable" and "--kiosk" in a.argv
          and a.argv[-1] == "--app=https://play2048.co/",
          "D. chrome: kiosk app window on the game's url")
    check(pa and pb and pa != pb, "D. a private profile per seat (%s)" % pa[0].rsplit("/", 1)[-1])
    check("--enable-unsafe-swiftshader" in a.argv,
          "D. no GPU (gl: false): WebGL gets the software rasteriser")
    gpu = L.resolve_launch(dict(br, gl=True), 0, 110, "SkyBot", which=which_all, root=tmp)
    check(not any("swiftshader" in x for x in gpu.argv), "D. gl: true: no SwiftShader flags")
    ff = L.resolve_launch(br, 0, 110, "SkyBot",
                          which=lambda n: "/usr/bin/firefox" if n == "firefox" else None, root=tmp)
    check(ff.kind == "browser-firefox" and "--profile" in ff.argv and "--kiosk" in ff.argv
          and ff.argv[-1] == "https://play2048.co/", "D. firefox: its own argv")
    L.prepare_profile(ff, br)
    prof = ff.argv[ff.argv.index("--profile") + 1]
    check(os.path.isfile(os.path.join(prof, "user.js")),
          "D. firefox profile created with autoplay allowed (user.js)")
    snapdir = os.path.join(tmp, "snap", "bin")
    os.makedirs(snapdir)
    shim = os.path.join(tmp, "chromium-browser")
    open(shim, "w").write("#!/bin/sh\nexec /snap/bin/chromium \"$@\"\n")
    check(L.browser_profile("g", 0, shim, root=tmp, home=home)
          == os.path.join(home, "snap", "chromium", "common", "skybot", "g_0"),
          "D. snap chromium (Ubuntu's shim): profile under ~/snap/chromium/common")
    bad = copy.deepcopy(br)
    bad["launcher"]["url"] = "javascript:alert(1)"
    check(raises(lambda: L.validate_launcher(bad), "http(s)://"), "D. non-http(s)/file url refused")
    check(raises(lambda: L.resolve_launch(br, 0, 110, "S", which=which_none, root=tmp),
                 "no browser installed"), "D. no browser installed: named, not a crash later")

    # ---------------------------------------------------------------- E
    slots = validate(br, 48)
    check(not raises(lambda: L.validate_launcher(br, slots), ""),
          "E. the shipped browser example validates (arrow keys only)")
    for keys, what in ((["Control_L", "w"], "Ctrl+W"), (["Alt_L", "Left"], "Alt+Left"),
                       (["F12"], "F12")):
        g2 = copy.deepcopy(br)
        g2["controller"] = g2["controller"] + [{"keys": keys}]
        check(raises(lambda: L.validate_launcher(g2, validate(g2, 48)), "one-way door"),
              f"E. {what} in a browser game refused")
    g3 = copy.deepcopy(br)
    g3["controller"] = g3["controller"] + [{"keys": ["Control_L"]}]
    g3["launcher"]["allow_keys"] = ["Control_L"]
    check(not raises(lambda: L.validate_launcher(g3, validate(g3, 48)), ""),
          "E. an allow-listed key is accepted")

    # ---------------------------------------------------------------- F
    s2 = Seat(["true"], display=4321, boot_s=0.0)
    check(s2.game_env().get("SKYBOT_SEAT") == "4321" and "SKYBOT_SEAT" not in s2.env(),
          "F. the game env carries the seat marker; the window manager's does not")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    fproc = os.path.join(tmp, "proc2")
    os.makedirs(os.path.join(fproc, str(child.pid)))
    open(os.path.join(fproc, str(child.pid), "environ"), "wb").write(b"A=1\0SKYBOT_SEAT=4321\0")
    os.makedirs(os.path.join(fproc, "1"))
    open(os.path.join(fproc, "1", "environ"), "wb").write(b"SKYBOT_SEAT=9999\0")
    killed = s2._kill_marked(proc=fproc)
    t0 = time.time()
    while child.poll() is None and time.time() - t0 < 5:
        time.sleep(0.05)
    check(killed == 1 and child.poll() is not None,
          "F. teardown kills the re-parented process carrying its marker, and only it")
    s2._closed = True

    # ---------------------------------------------------------------- G
    m = L.preflight(st, which=which_none, home=home)
    check(any("Steam client" in x for x in m) and any("Xvfb" in x for x in m),
          "G. steam client mode: missing client and Xvfb named")
    m = L.preflight(st, which=which_all, home=home)
    check(not any("not installed in any Steam library" in x for x in m),
          "G. an app present in a library is not reported missing")
    m = L.preflight(br, which=which_none)
    check(any("browser" in x for x in m), "G. browser: no browser named")
    check(L.preflight(dp, which=which_all) == [], "G. exec: nothing missing when all tools exist")

    # ---------------------------------------------------------------- H
    from tools.compose_run_config import ComposeError, compose
    gdir = os.path.join(tmp, "games")
    shutil.copytree(os.path.join(ROOT, "configs", "games"), gdir,
                    ignore=shutil.ignore_patterns("examples"))
    for ex in ("browser_2048", "steam_game"):
        shutil.copy(os.path.join(ROOT, "configs/games/examples", ex + ".yaml"), gdir)
    base = os.path.join(ROOT, "configs/minecraft_skybot.yaml")
    uni = os.path.join(ROOT, "configs/universal.yaml")
    cfg = compose(["browser_2048", "steam_game"], base, uni, gdir)
    check([s["game"] for s in cfg["parallel_envs"]["streams"]]
          == ["browser_2048", "browser_2048", "steam_game"],
          "H. browser + steam examples compose (3 streams)")
    d = yaml.safe_load(open(os.path.join(gdir, "steam_game.yaml")))
    d["game"]["streams"] = 2
    yaml.safe_dump(d, open(os.path.join(gdir, "steam_game.yaml"), "w"))
    try:
        compose(["steam_game"], base, uni, gdir)
        ok = False
    except ComposeError as e:
        ok = "one game per Linux user" in str(e)
    check(ok, "H. steam client mode with 2 streams refused at compose time")


if __name__ == "__main__":
    main()
