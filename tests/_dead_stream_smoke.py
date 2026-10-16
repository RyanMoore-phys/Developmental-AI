"""A dead game stream comes back; a dead seat's screen does not block the next.

Born 2026-10-14 from the universal-games sweep, two findings on the same
failure — a game engine that dies under a live run:

  1. The lifelong body (_collect_segment, the one SkyBot runs) treated every
     failed step as "the wrapper rebuilt its own world". Only MineRL's does.
     When ViZDoom's clock thread or the Player Adapter's screen capture died,
     every later step()/reset() raised, and the stream carried one frozen
     frame with a restart flag on every step for the rest of the run.
  2. A SIGKILLed / OOM-killed trainer leaves its seats' Xvfb running, and the
     seat refused any locked display — so after one hard crash both
     doom_player seats refused on every relaunch: guard-becomes-latch (§4.1).

Contracts:
    A. NO HEAL ON A BLIP: fewer than heal_after_failures consecutive raised
       steps rebuild nothing.
    B. HEAL: at the threshold the old env is closed off-thread, a new one is
       built and reset, its obs returned, the failure count cleared; slot 0's
       self.env and the viewer follow the new env.
    C. A RAISING RESET HEALS AT ONCE (force).
    D. BUDGETED, NOT LATCHED: past heal_max_per_hour the heal is refused;
       once the hour slides it is allowed again.
    E. A FAILED HEAL is not retried every step: it needs a fresh run of
       failed steps.
    F. WIRED INTO THE LIVE BODY: _collect_segment counts raised steps and
       heals in both the carry branch and the failed-reset branch; the
       episodic body heals at its boundary through _parallel_reset.
    G. STALE LOCK (X server dead) -> removed.
    H. AN X SERVER THAT IS NOT A SEAT'S -> refused, named.
    I. ANOTHER LIVE TRAINER'S SEAT -> refused (a seat never shares a screen).
    J. ORPHAN (owner trainer dead) -> its game and Xvfb are killed and the
       lock cleared, so the relaunch starts.
    K. OUR OWN ABANDONED SEAT (a rebuild racing the async close) -> waits for
       it to release.

Run: PYTHONPATH=. python tests/_dead_stream_smoke.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.core.developmental_loop import DevelopmentalAI   # noqa: E402
from developmental_ai.environments.player.seat import Seat, SeatError  # noqa: E402

NAME = "dead-stream"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


class Env:
    made = 0

    def __init__(self, tag, fail_reset=False):
        self.tag, self.fail_reset, self.closed = tag, fail_reset, threading.Event()
        Env.made += 1

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("doom clock died")
        return np.full(4, 7.0, np.float32), {}

    def close(self):
        self.closed.set()


class Loop:
    """Just what _maybe_heal_stream touches."""

    def __init__(self, envs, **pcfg):
        self.config = {"parallel_envs": dict(reset_timeout_s=5, **pcfg)}
        self._env_pool = ThreadPoolExecutor(2)
        self.envs = envs
        self.env = envs[0]
        self.build_fails = False

    def _make_one_env(self, i):
        if self.build_fails:
            raise RuntimeError("cannot start engine")
        return Env(f"new{i}")

    def heal(self, i, force=False):
        return DevelopmentalAI._maybe_heal_stream(self, self.envs, i, force=force)


def spawn_reaped(argv, env=None):
    """A child that is reaped the moment it dies (no zombie reads as alive)."""
    p = subprocess.Popen(argv, env=env)
    threading.Thread(target=p.wait, daemon=True).start()
    return p


def main():
    tmp = tempfile.mkdtemp(prefix="deadstream_")
    kids = []
    try:
        heal_contracts()
        seat_contracts(tmp, kids)
    finally:
        for k in kids:
            try:
                k.kill()
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{NAME}] ALL PASS")


def heal_contracts():
    # ---------------------------------------------------------------- A
    old0, old1 = Env("old0"), Env("old1")
    lp = Loop([old0, old1], heal_after_failures=3, heal_max_per_hour=2)
    lp._stream_fails = {1: 2}
    check(lp.heal(1) is None and lp.envs[1] is old1, "A. 2 failed steps: nothing rebuilt")

    # ---------------------------------------------------------------- B
    lp._stream_fails = {1: 3}
    o = lp.heal(1)
    check(o is not None and float(o[0]) == 7.0 and lp.envs[1].tag == "new1",
          "B. 3 failed steps: stream 1 rebuilt and reset, its obs returned")
    check(old1.closed.wait(5), "B. the dead env is closed (off-thread)")
    check(1 not in lp._stream_fails, "B. the failure count is cleared")
    lp._stream_fails = {0: 5}
    lp._viewer_env = old0
    lp.heal(0)
    check(lp.env is lp.envs[0] and lp.envs[0].tag == "new0" and lp._viewer_env is lp.envs[0],
          "B. slot 0: self.env and the viewer follow the new env")

    # ---------------------------------------------------------------- C
    lp2 = Loop([Env("a"), Env("b")])
    check(lp2.heal(1, force=True) is not None, "C. a raising reset heals at once (force)")

    # ---------------------------------------------------------------- D
    lp._stream_fails = {1: 9}
    check(lp.heal(1) is not None, "D. second heal this hour allowed (max 2)")
    lp._stream_fails = {1: 9}
    keep = lp.envs[1]
    check(lp.heal(1) is None and lp.envs[1] is keep, "D. third heal this hour refused")
    lp._stream_heals[1] = deque(t - 3700 for t in lp._stream_heals[1])
    check(lp.heal(1) is not None, "D. the hour slides: allowed again (no latch)")

    # ---------------------------------------------------------------- E
    lp3 = Loop([Env("a"), Env("b")])
    lp3.build_fails = True
    lp3._stream_fails = {1: 3}
    check(lp3.heal(1) is None and lp3._stream_fails[1] == 0,
          "E. a failed heal resets the count: no rebuild attempt every step")

    # ---------------------------------------------------------------- F
    src = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    cs = src[src.index("    def _collect_segment"):]
    cs = cs[:cs.index("\n    def ", 10)]
    check("_stream_fails" in cs and cs.count("self._maybe_heal_stream(") == 2,
          "F. the live body counts raised steps and heals in both branches")
    check("_bounded_rebuild(i)" in src and "def _parallel_reset" in src,
          "F. the episodic body heals at its boundary (_parallel_reset)")


def seat_contracts(tmp, kids):
    xt = os.path.join(tmp, "x")
    proc = os.path.join(tmp, "proc")
    os.makedirs(os.path.join(xt, ".X11-unix"))
    os.makedirs(proc)

    def lock(n, pid):
        open(os.path.join(xt, f".X{n}-lock"), "w").write(f"{pid:>10}\n")
        open(os.path.join(xt, ".X11-unix", f"X{n}"), "w").write("")

    def fake_env(pid, **env):
        d = os.path.join(proc, str(pid))
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "environ"), "wb").write(
            b"".join(f"{k}={v}".encode() + b"\0" for k, v in env.items()))

    def locked(n):
        return os.path.exists(os.path.join(xt, f".X{n}-lock"))

    sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]

    # ---------------------------------------------------------------- G
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock(201, dead.pid)
    s = Seat(["true"], display=201)
    s._closed = True
    check(s.clear_stale_display(tmp=xt, proc=proc) == "stale" and not locked(201),
          "G. lock of a dead X server: removed")

    # ---------------------------------------------------------------- H
    foreign = spawn_reaped(sleeper)
    kids.append(foreign)
    lock(202, foreign.pid)
    fake_env(foreign.pid, DISPLAY=":0")
    s = Seat(["true"], display=202)
    s._closed = True
    try:
        s.clear_stale_display(tmp=xt, proc=proc)
        msg = ""
    except SeatError as e:
        msg = str(e)
    check("did not start" in msg and locked(202), "H. an X server that is not a seat's: refused, named")

    # ---------------------------------------------------------------- I
    other_trainer = spawn_reaped(sleeper)
    xv = spawn_reaped(sleeper)
    kids += [other_trainer, xv]
    lock(203, xv.pid)
    fake_env(xv.pid, SKYBOT_SEAT_OWNER=str(other_trainer.pid))
    s = Seat(["true"], display=203)
    s._closed = True
    try:
        s.clear_stale_display(tmp=xt, proc=proc)
        msg = ""
    except SeatError as e:
        msg = str(e)
    check("running trainer" in msg and xv.poll() is None,
          "I. another live trainer's seat: refused, its Xvfb untouched")

    # ---------------------------------------------------------------- J
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    xv2 = spawn_reaped(sleeper)
    game = spawn_reaped(sleeper)
    kids += [xv2, game]
    lock(204, xv2.pid)
    fake_env(xv2.pid, SKYBOT_SEAT_OWNER=str(gone.pid))
    fake_env(game.pid, SKYBOT_SEAT="204")
    s = Seat(["true"], display=204)
    s._closed = True
    r = s.clear_stale_display(tmp=xt, proc=proc)
    t0 = time.time()
    while (xv2.poll() is None or game.poll() is None) and time.time() - t0 < 5:
        time.sleep(0.05)
    check(r == "orphan" and xv2.poll() is not None and game.poll() is not None
          and not locked(204),
          "J. orphan (owner dead): its game and Xvfb killed, the lock cleared")

    # ---------------------------------------------------------------- K
    xv3 = spawn_reaped(sleeper)
    kids.append(xv3)
    lock(205, xv3.pid)
    fake_env(xv3.pid, SKYBOT_SEAT_OWNER=str(os.getpid()))

    def async_close():
        time.sleep(0.5)
        xv3.kill()
        os.unlink(os.path.join(xt, ".X205-lock"))
        os.unlink(os.path.join(xt, ".X11-unix", "X205"))
    threading.Thread(target=async_close, daemon=True).start()
    s = Seat(["true"], display=205)
    s._closed = True
    check(s.clear_stale_display(tmp=xt, proc=proc, wait_s=5) == "released",
          "K. our own abandoned seat: waits for its async close to release it")


if __name__ == "__main__":
    main()
