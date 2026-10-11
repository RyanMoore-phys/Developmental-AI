"""No Minecraft client may outlive the env that owns it (2026-10-09).

LIVE INCIDENT: on `main` a num_envs=2 run owned FOUR Java clients. Socket
counters over 15 s: ports 9276/9277 received +99 MB / +82 MB (the two live
streams); ports 9273/9275 had received 443 KB in total and +0 -- dead, but
each holding ~1 GB of Java heap plus a ~200 MB process_watcher, ~2.4 GB on a
15.3 GB box whose single run already measures ~13 GB. Nothing would ever have
freed them before the trainer exited. Two separate leaks, both triggered by
reset hangs (that night: swap thrash from a duplicate run):

  9275 -- REBUILD AFTER CLOSE. _parallel_reset abandons a reset that exceeds
          reset_timeout_s and close()s the adapter on a daemon thread. The
          abandoned MineRLEnvAdapter.reset() is still in its 3-attempt loop on
          a pool thread; close() breaks its socket, the loop logs "attempt
          2/3: Bad file descriptor" and _rebuild() boots a NEW client on an
          adapter nothing references. "env 0: abandoned client closed" was
          logged and true -- of the OLD client.
  9273 -- OVERWRITE WITHOUT CLOSE. _bounded_rebuild did `envs[i] =
          self._make_one_env(i)` on each try; "env 0 rebuild try 1 failed ()"
          then dropped try 1's env, client still running.

This suite runs the REAL MineRLEnvAdapter.reset/_rebuild/close and the REAL
DevelopmentalAI._parallel_reset against fake clients (no MineRL, no Java).

Contracts:
    A. NO REBUILD AFTER CLOSE: a reset() hung on a client, then close()d from
       another thread (exactly what _reap does), exits with an error and
       boots ZERO new clients. FALSIFICATION: without the _closed check the
       hung reset rebuilds and A fails on the client count.
    B. CLOSE DURING BOOT: close() landing while _make_underlying is booting
       a replacement -> _rebuild closes that fresh client and raises; no
       client is left open.
    C. NOT A LATCH (CLAUDE.md 4.1): on an adapter that was never closed, a
       failed reset still rebuilds and returns the fresh client's obs. The
       guard only ever fires after close(), which is terminal by design.
    D. _parallel_reset: when the original reset AND both rebuild tries hang,
       the original and try 1 are both closed (try 1 is the 9273 leak); the
       final try stays open because it is still envs[i] and is reused.

Run: PYTHONPATH=. python tests/_client_leak_smoke.py
"""
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np

NAME = "client-leak"
HANG_S = 3.0          # a fake hung reset gives up on its own after this


class FakeClient:
    """A raw MineRL env: reset() blocks like a stuck socket until closed."""
    made = []

    def __init__(self, hang=True):
        self.hang = hang
        self.closed = False
        self._dead = threading.Event()
        FakeClient.made.append(self)

    def reset(self):
        if not self.hang:
            return {"pov": "FRESH"}
        self._dead.wait(HANG_S)
        raise OSError(9, "Bad file descriptor")    # what close() does to it

    def close(self):
        self.closed = True
        self._dead.set()


def wait_for(pred, timeout=10.0):
    t = time.time() + timeout
    while time.time() < t:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def make_adapter(first, factory):
    """A real MineRLEnvAdapter with only what reset/_rebuild/close touch."""
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter
    a = MineRLEnvAdapter.__new__(MineRLEnvAdapter)
    a._closed = False
    a._env = first
    a._make_underlying = factory
    a._validate_macros = lambda: None
    a._dr = types.SimpleNamespace(reset=lambda: None)
    a._count_logs = lambda o: 0
    a._mine_counts = lambda o: {}
    a._craft_counts = lambda o: {}
    a._pov_to_obs = lambda o: o["pov"]
    a._save_break_memory = lambda force=False: None
    return a


def main():
    # ---- A: hung reset + close() from another thread -> no new client ----
    FakeClient.made = []
    old = FakeClient()
    a = make_adapter(old, lambda: FakeClient())
    box = {}

    def _run():
        try:
            box["ret"] = a.reset()
        except Exception as ex:
            box["err"] = ex
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.3)                       # reset is now blocked on `old`
    a.close()                             # what _reap does
    t.join(HANG_S * 4)
    assert not t.is_alive(), "abandoned reset never returned"
    assert "err" in box, "reset on a closed adapter returned %r" % box.get("ret")
    assert len(FakeClient.made) == 1, \
        "closed adapter booted %d new client(s)" % (len(FakeClient.made) - 1)
    assert old.closed
    print("  A. hung reset closed mid-loop -> raises, 0 new clients (was: "
          "rebuild on an unreferenced adapter)")

    # ---- B: close() lands during the replacement's boot -------------------
    FakeClient.made = []
    old = FakeClient()
    b = None

    def _boot_racing_close():
        b.close()                         # arrives mid-boot
        return FakeClient()
    b = make_adapter(old, _boot_racing_close)
    try:
        b._rebuild()
        raise AssertionError("_rebuild returned after close()")
    except RuntimeError:
        pass
    assert len(FakeClient.made) == 2 and all(c.closed for c in FakeClient.made), \
        "open clients after close-during-boot: %s" % [c.closed for c in FakeClient.made]
    print("  B. close() during the ~90 s boot -> fresh client closed, raises")

    # ---- C: an open adapter still recovers (the guard is not a latch) -----
    FakeClient.made = []
    broken = FakeClient()
    broken._dead.set()                    # fails at once, like a dead socket
    c = make_adapter(broken, lambda: FakeClient(hang=False))
    obs, info = c.reset()
    assert obs == "FRESH" and len(FakeClient.made) == 2 and broken.closed, \
        "open adapter did not rebuild: obs=%r made=%d" % (obs, len(FakeClient.made))
    print("  C. never-closed adapter: failed reset still rebuilds and returns obs")

    # ---- D: _parallel_reset closes every env it replaces ------------------
    from developmental_ai.core.developmental_loop import DevelopmentalAI

    class FakeEnv:
        def __init__(self, hang):
            self.client = FakeClient(hang=hang)

        def reset(self):
            return self.client.reset(), {}

        def close(self):
            self.client.close()

    FakeClient.made = []
    built = []

    def _make_one_env(i):
        e = FakeEnv(hang=True)
        built.append(e)
        return e
    pool = ThreadPoolExecutor(max_workers=8)
    fake_self = types.SimpleNamespace(
        config={"parallel_envs": {"reset_timeout_s": 0.3}},
        _env_pool=pool, obs_dim=4, _make_one_env=_make_one_env)
    orig, good = FakeEnv(hang=True), FakeEnv(hang=False)
    good.reset = lambda: (np.ones(4), {})
    envs = [orig, good]
    obs = DevelopmentalAI._parallel_reset(fake_self, envs)
    assert len(built) == 2, "expected 2 rebuild tries, got %d" % len(built)
    try1, try2 = built
    assert envs[0] is try2, "envs[0] is not the last try"
    assert wait_for(lambda: orig.client.closed), "abandoned original never closed"
    assert wait_for(lambda: try1.client.closed), \
        "rebuild try 1 overwritten without close (the 9273 leak)"
    assert not try2.client.closed, "the env still in envs[0] was closed"
    assert np.all(obs[0] == 0) and np.all(obs[1] == 1)
    pool.shutdown(wait=False)
    print("  D. _parallel_reset: original + try 1 closed, live try kept, "
          "stream zero-filled")
    print("[%s] ALL PASS" % NAME)


if __name__ == "__main__":
    main()
