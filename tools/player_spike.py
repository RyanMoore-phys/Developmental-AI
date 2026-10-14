#!/usr/bin/env python3
"""Player Adapter — stage P0 spike: measure before building (Linux host only).

Plan: ~/.claude/plans/typed-petting-nautilus.md, docs/UNIVERSAL_GAMES.md.
The Player Adapter runs an UNMODIFIED game in its own virtual display and
plays it with a virtual keyboard/mouse while capturing its screen and sound.
Every piece of that is an assumption until measured on the real host. This
script stands up ONE seat by hand and measures the five things the design
depends on, printing PASS/FAIL against the P0 gates:

    grab      screen capture rate of the seat display     gate >= 30 fps
    latency   key press -> visible pixel change            gate <= 50 ms (median)
    mouse     XTest relative motion turns the view         gate: view changes
    audio     the game's sound reaches a capture sink      gate: non-silent
    cost      CPU% and RSS of the game process             reported

Default game: Chocolate Doom on Freedoom (`apt install chocolate-doom`;
the WAD comes from roms/doom via scripts/fetch_roms.sh).

    ./venv_mc/bin/python tools/player_spike.py [--display 190] [--seconds 5]
        [--launch "chocolate-doom -iwad roms/doom/freedoom2.wad -window -geometry 640x480"]
        [--gl]            # run the game under `vglrun -d egl` (GPU GL)
        [--json out.json]

Needs: Xvfb, pulseaudio (pactl/parec), python-xlib, mss, psutil, numpy.
Touches nothing outside its own display (:190) and its own audio sink
(skybot_spike); everything is torn down on exit, even on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

GATES = {"grab_fps": 30.0, "latency_ms": 50.0}


def need(cmd: str) -> None:
    if shutil.which(cmd) is None:
        sys.exit(f"SPIKE CANNOT RUN: `{cmd}` not found (see the docstring's Needs)")


class Seat:
    """One display + one audio sink + one game process, torn down on exit."""

    def __init__(self, display: int, w: int, h: int):
        self.display, self.w, self.h = display, w, h
        self.procs = []
        self.sink_module = None
        self.env = dict(os.environ, DISPLAY=f":{display}",
                        PULSE_SINK="skybot_spike", SDL_AUDIODRIVER="pulse")

    def start_display(self):
        p = subprocess.Popen(["Xvfb", f":{self.display}", "-screen", "0",
                              f"{self.w}x{self.h}x24", "-nolisten", "tcp"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p)
        time.sleep(1.0)
        if shutil.which("openbox"):          # focus handling, as MineRL uses
            self.procs.append(subprocess.Popen(["openbox"], env=self.env,
                                               stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL))
            time.sleep(0.5)

    def start_audio(self):
        subprocess.run(["pulseaudio", "--start", "--exit-idle-time=-1"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = subprocess.run(["pactl", "load-module", "module-null-sink",
                              "sink_name=skybot_spike"],
                             capture_output=True, text=True)
        self.sink_module = out.stdout.strip() or None

    def launch(self, cmd, gl: bool):
        argv = shlex.split(cmd)
        if gl:
            argv = ["vglrun", "-d", "egl"] + argv
        p = subprocess.Popen(argv, env=self.env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        self.procs.append(p)
        self.game = p
        return p

    def close(self):
        for p in reversed(self.procs):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM) \
                    if p is getattr(self, "game", None) else p.terminate()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        if self.sink_module:
            subprocess.run(["pactl", "unload-module", self.sink_module],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def grab_fn(display: int, w: int, h: int):
    import mss
    sct = mss.mss(display=f":{display}")
    mon = {"left": 0, "top": 0, "width": w, "height": h}

    def grab():
        return np.asarray(sct.grab(mon))[:, :, :3]
    return grab


def xtest(display: int):
    from Xlib import X, XK, display as xdisplay
    from Xlib.ext import xtest as xt
    d = xdisplay.Display(f":{display}")

    def key(name: str, down: bool):
        code = d.keysym_to_keycode(XK.string_to_keysym(name))
        xt.fake_input(d, X.KeyPress if down else X.KeyRelease, code)
        d.sync()

    def rel_motion(dx: int, dy: int):
        # detail=True => RELATIVE motion (what pointer-grabbing games read)
        xt.fake_input(d, X.MotionNotify, detail=True, x=int(dx), y=int(dy))
        d.sync()
    return key, rel_motion


def measure(args) -> dict:
    for c in ("Xvfb", "pactl", "parec"):
        need(c)
    res = {"launch": args.launch, "gl": args.gl}
    seat = Seat(args.display, args.width, args.height)
    try:
        seat.start_display()
        seat.start_audio()
        game = seat.launch(args.launch, args.gl)
        time.sleep(args.boot)
        if game.poll() is not None:
            res["error"] = f"game exited at boot (rc {game.returncode})"
            return res
        grab = grab_fn(args.display, args.width, args.height)
        key, rel = xtest(args.display)
        # get past a title screen: Doom starts in its demo loop; Escape+Enter
        # starts a new game in Chocolate Doom's default menu.
        for k in ("Escape", "Return", "Return", "Return"):
            key(k, True); key(k, False); time.sleep(0.4)
        time.sleep(1.5)

        # ---- grab rate
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            grab(); n += 1
        res["grab_fps"] = round(n / (time.perf_counter() - t0), 1)

        # ---- key -> pixel latency (turn right, then left to undo)
        lat = []
        for i in range(12):
            base = grab().astype(np.int16)
            k = "Right" if i % 2 == 0 else "Left"
            t0 = time.perf_counter()
            key(k, True)
            seen = None
            while time.perf_counter() - t0 < 0.5:
                if np.abs(grab().astype(np.int16) - base).mean() > 2.0:
                    seen = time.perf_counter() - t0
                    break
            key(k, False)
            if seen is not None:
                lat.append(1000 * seen)
            time.sleep(0.25)
        res["latency_ms_median"] = round(float(np.median(lat)), 1) if lat else None
        res["latency_samples"] = len(lat)

        # ---- relative mouse turns the view
        base = grab().astype(np.int16)
        for _ in range(30):
            rel(25, 0); time.sleep(1 / 60)
        res["mouse_view_change"] = round(float(np.abs(grab().astype(np.int16) - base).mean()), 2)

        # ---- audio: capture while firing
        rec = subprocess.Popen(["parec", "-d", "skybot_spike.monitor", "--format=s16le",
                                "--rate=22050", "--channels=2"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 2.5:
            key("Control_L", True); time.sleep(0.15); key("Control_L", False); time.sleep(0.15)
        rec.terminate()
        raw = rec.communicate(timeout=5)[0]
        a = np.frombuffer(raw, dtype=np.int16)
        res["audio_samples"] = int(a.size)
        res["audio_peak"] = int(np.abs(a).max()) if a.size else 0

        # ---- cost
        try:
            import psutil
            pp = psutil.Process(game.pid)
            pp.cpu_percent(None); time.sleep(2.0)
            res["game_cpu_pct"] = round(pp.cpu_percent(None), 1)
            res["game_rss_mb"] = round(pp.memory_info().rss / 1e6, 1)
        except Exception as e:
            res["cost_error"] = repr(e)
    finally:
        seat.close()
    return res


def verdict(r: dict) -> int:
    rows = [
        ("grab", r.get("grab_fps"), lambda v: v is not None and v >= GATES["grab_fps"],
         f">= {GATES['grab_fps']:.0f} fps"),
        ("latency", r.get("latency_ms_median"),
         lambda v: v is not None and v <= GATES["latency_ms"], f"<= {GATES['latency_ms']:.0f} ms"),
        ("mouse", r.get("mouse_view_change"), lambda v: v is not None and v > 2.0,
         "view changes (mean |dpx| > 2)"),
        ("audio", r.get("audio_peak"), lambda v: v is not None and v > 0, "non-silent"),
    ]
    fails = 0
    print("\n=== PLAYER ADAPTER P0 SPIKE ===")
    if "error" in r:
        print("ERROR:", r["error"])
        return 1
    for name, v, ok, gate in rows:
        good = ok(v)
        fails += 0 if good else 1
        print(f"  {'PASS' if good else 'FAIL'}  {name:8s} {v!s:>10}   gate {gate}")
    print(f"  info  cost     cpu {r.get('game_cpu_pct')}%  rss {r.get('game_rss_mb')} MB")
    print("SPIKE-OK" if fails == 0 else f"SPIKE-FAILED ({fails} gate(s))")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Player Adapter P0 spike")
    ap.add_argument("--display", type=int, default=190)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--boot", type=float, default=4.0)
    ap.add_argument("--gl", action="store_true")
    ap.add_argument("--launch", default="chocolate-doom -iwad roms/doom/freedoom2.wad "
                    "-window -geometry 640x480 -nomusic")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    r = measure(a)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    return verdict(r)


if __name__ == "__main__":
    sys.exit(main())
