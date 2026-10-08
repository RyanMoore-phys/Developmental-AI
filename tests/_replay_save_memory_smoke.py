"""Replay-buffer SAVE memory smoke (2026-10-07).

THE LIVE INCIDENT THIS ENCODES

    16 GB host, memory near-full, swap climbing. The host went unresponsive
    seconds after

        replay buffer saved: 50000 transitions in 5.2s

    `ReplayBuffer.save()` gathered a FULL in-RAM copy of every column
    (`self.observations[order]` — ~1.2 GB per 25k-row stream of 3x128x128
    uint8) before `np.save`: a ~2.4 GB transient over two streams that the
    memory budget never saw. And it is not a shutdown-only path —
    `_save_checkpoint` -> `_save_replay_buffer` runs it on EVERY periodic
    checkpoint. It also held the buffer lock across the gather, so the
    acting thread's add() stalled behind it.

    The fix streams each column to its .npy in bounded chunks, taking the
    lock per chunk. That re-opens a race the old code did not have: add()
    keeps overwriting the OLDEST slot while the save is reading. The saved
    set is chronological and overwrites walk it from the oldest end, so the
    possibly-overwritten rows form a PREFIX; the manifest records its length
    as `start` and load() skips it.

Contracts:
    A. BYTES UNCHANGED. With no concurrent adds, every .npy is byte-identical
       to what the OLD gather + np.save wrote (fixed and block storage, with
       and without max_transitions), and the manifest differs only by
       `start == 0`. The old path is reimplemented here as the reference.
    B. CONCURRENT ADDS NEVER LEAK. A thread calling add() while save streams
       (buffer full and wrapping, also partly full, also max_transitions):
       every loaded row equals the pre-save snapshot row it claims to be
       (unique per-row ids in every column), and `start` equals the exact
       overwrite count when it is deterministic. The raw files are shown to
       CONTAIN overwritten rows in the prefix, so the check is not vacuous.
    C. MEMORY IS BOUNDED. Peak extra allocation during save of a ~200 MB
       buffer is < 25% of the buffer (old path: ~100%, measured alongside as
       the regression witness).
    D. OLD SAVES LOAD IDENTICALLY. A directory written by the old code (no
       `start` key) loads to the same buffer state as the old load did.
    E. THE LOCK IS NEVER HELD ACROSS A COLUMN. Lock acquisitions during save
       are at least one per chunk per column, and the longest hold is a small
       fraction of a single full-column gather.

Run: PYTHONPATH=. python tests/_replay_save_memory_smoke.py
"""
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import tracemalloc

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.world_model.replay_buffer import ReplayBuffer

GROW = {"enabled": True, "block_transitions": 1000,
        "max_ram_frac": 0.6, "hard_max_gb": 0.0}
COLS = ("observations", "actions", "rewards", "dones", "priorities",
        "restarts", "proprio")


# ---------------------------------------------------------------- reference
def old_save(buf, path, max_transitions=None):
    """The pre-2026-10-07 ReplayBuffer.save, verbatim in effect."""
    with buf._lock:
        order = buf._chronological()
        if max_transitions is not None and len(order) > int(max_transitions):
            order = order[-int(max_transitions):]
        n = int(len(order))
        cols = {"observations": buf.observations[order],
                "actions": buf.actions[order],
                "rewards": buf.rewards[order],
                "dones": buf.dones[order],
                "priorities": buf.priorities[order],
                "restarts": buf.restarts[order]}
        if buf.proprio is not None:
            cols["proprio"] = buf.proprio[order]
        meta = {"n": n, "obs_dim": int(buf.obs_dim),
                "action_dim": int(buf.action_dim),
                "obs_uint8": bool(buf._obs_uint8),
                "max_priority": float(buf.max_priority),
                "proprio_dim": int(buf.proprio_dim),
                "sensor_layout": str(buf.sensor_layout)}
    os.makedirs(path, exist_ok=True)
    for k, v in cols.items():
        np.save(os.path.join(path, k + ".npy"), v)
    with open(os.path.join(path, "manifest.json"), "w") as fh:
        json.dump(meta, fh)
    return n


def old_load_state(buf, path):
    """The pre-2026-10-07 load's resulting state, computed with full in-RAM
    arrays (only the paths the tests exercise: matching layout, fixed or
    growable capacity)."""
    with open(os.path.join(path, "manifest.json")) as fh:
        meta = json.load(fh)
    cols = {k: np.load(os.path.join(path, k + ".npy"))
            for k in COLS if os.path.isfile(os.path.join(path, k + ".npy"))}
    n = int(cols["observations"].shape[0])
    if "restarts" not in cols:
        cols["restarts"] = np.zeros(n, bool)
    while buf._growable and not buf._growth_capped and n > buf.capacity:
        buf._grow_pending = True
        if not buf.maybe_grow():
            break
    if n > buf.capacity:
        cols = {k: v[-buf.capacity:] for k, v in cols.items()}
        n = buf.capacity
    return {"n": n, "cols": cols,
            "max_priority": float(meta.get("max_priority", 1.0)) or 1.0}


def make(cap, od=24, ad=3, pd=4, uint8=True, growth=None):
    return ReplayBuffer(capacity=cap, obs_dim=od, action_dim=ad,
                        obs_uint8=uint8, growth=growth, proprio_dim=pd,
                        sensor_layout="L" if pd else "")


def fill(buf, n, base=0, rng=None):
    rng = rng or np.random.RandomState(0)
    for t in range(n):
        i = base + t
        obs = (np.full(buf.obs_dim, (i % 251) / 255.0, np.float32)
               if buf._obs_uint8 else np.full(buf.obs_dim, float(i), np.float32))
        act = np.full(buf.action_dim, float(i), np.float32)
        buf.add(obs, act, float(i), done=(i % 97 == 96),
                restart=(i % 389 == 388),
                proprio=np.full(buf.proprio_dim, float(i), np.float32))
        if t % 100 == 99:
            buf.maybe_grow()                 # the WM cadence, as live
    # non-trivial priorities so that column is distinguishable too
    with buf._lock:
        k = len(buf.priorities)
        buf.priorities[np.arange(k)] = rng.rand(k).astype(np.float32) + 0.1


def files_equal(a, b, names):
    for k in names:
        with open(os.path.join(a, k + ".npy"), "rb") as f1, \
                open(os.path.join(b, k + ".npy"), "rb") as f2:
            if f1.read() != f2.read():
                return k
    return None


def state_of(buf):
    n = buf.size
    order = buf._chronological()
    out = {k: np.asarray(getattr(buf, k)[order]) for k in COLS
           if getattr(buf, k) is not None}
    return n, out


# ---------------------------------------------------------------- contracts
def test_bytes_unchanged(tmp):
    print("A. no concurrent adds -> byte-identical to the old np.save path")
    cases = [("fixed, wrapped", make(1500, uint8=True), 2300, None),
             ("fixed, partial", make(1500, uint8=False), 900, None),
             ("fixed, max_t", make(1500, uint8=True), 2300, 700),
             ("block, grown", make(1000, growth=GROW), 2600, None),
             ("block, max_t", make(1000, growth=GROW), 2600, 1100),
             ("no proprio", make(1200, pd=0), 1700, None)]
    for name, buf, n_add, mt in cases:
        fill(buf, n_add)
        ref, new = os.path.join(tmp, "ref"), os.path.join(tmp, "new")
        for p in (ref, new):
            shutil.rmtree(p, ignore_errors=True)
        buf._SAVE_CHUNK_ROWS = 97            # odd chunk -> ragged tail
        n_old = old_save(buf, ref, mt)
        n_new = buf.save(new, mt)
        names = [k for k in COLS if os.path.isfile(os.path.join(ref, k + ".npy"))]
        bad = files_equal(ref, new, names)
        assert bad is None, f"{name}: {bad}.npy bytes differ from old np.save"
        assert n_old == n_new, (name, n_old, n_new)
        m_old = json.load(open(os.path.join(ref, "manifest.json")))
        m_new = json.load(open(os.path.join(new, "manifest.json")))
        assert m_new.pop("start") == 0, name
        assert m_old == m_new, (name, m_old, m_new)
        # and the new file round-trips to the same chronological state
        b2 = make(buf.capacity if not buf._growable else 1000,
                  od=buf.obs_dim, pd=buf.proprio_dim,
                  uint8=buf._obs_uint8,
                  growth=GROW if buf._growable else None)
        assert b2.load(new) == n_new
        _, s2 = state_of(b2)
        for k in names:
            ref_arr = np.load(os.path.join(ref, k + ".npy"))
            assert np.array_equal(s2[k], ref_arr), (name, k)
        print(f"  A. {name:15s} n={n_new:5d} files={len(names)} identical")


class _Adder(threading.Thread):
    def __init__(self, buf, base, k=None, stop=None):
        super().__init__(daemon=True)
        self.buf, self.base, self.k, self.stop = buf, base, k, stop
        self.go = threading.Event()
        self.n = 0

    def run(self):
        self.go.wait(10)
        i = self.base
        while True:
            if self.k is not None and self.n >= self.k:
                return
            if self.stop is not None and self.stop.is_set():
                return
            self.buf.add(np.full(self.buf.obs_dim, float(i), np.float32),
                         np.full(self.buf.action_dim, float(i), np.float32),
                         float(i), done=False,
                         proprio=np.full(self.buf.proprio_dim, float(i),
                                         np.float32))
            i += 1
            self.n += 1
            if self.k is None:
                time.sleep(0)


def _concurrent_case(tmp, cap, pre, k, mt, deterministic):
    buf = make(cap, od=16, uint8=False)        # float obs -> exact ids
    fill(buf, pre)
    with buf._lock:
        buf.priorities[np.arange(cap)] = 1.0 + np.arange(cap, dtype=np.float32)
    snap_n, snap = state_of(buf)
    if mt is not None and snap_n > mt:
        snap = {kk: v[-mt:] for kk, v in snap.items()}
    stop = threading.Event()
    adder = _Adder(buf, base=10 ** 6, k=k if deterministic else None,
                   stop=None if deterministic else stop)
    adder.start()
    buf._SAVE_CHUNK_ROWS = 16
    real = buf._stream_column
    calls = {"n": 0}

    def hooked(fpath, col, order):
        calls["n"] += 1
        if calls["n"] == 1:
            adder.go.set()                    # adds begin AFTER the snapshot
        elif calls["n"] == 2 and deterministic:
            adder.join(10)                    # all k adds land mid-save
        return real(fpath, col, order)

    buf._stream_column = hooked
    p = os.path.join(tmp, f"conc_{cap}_{pre}_{k}_{mt}_{deterministic}")
    n_ret = buf.save(p, mt)
    stop.set(); adder.join(10)
    meta = json.load(open(os.path.join(p, "manifest.json")))
    start = meta["start"]
    n_saved = meta["n"]
    # raw files: how many prefix rows really were overwritten?
    raw_r = np.load(os.path.join(p, "rewards.npy"))
    raw_bad = int((raw_r != snap["rewards"]).sum())
    assert (raw_r[start:] == snap["rewards"][start:]).all(), \
        "an overwritten row lies OUTSIDE the skipped prefix"
    b2 = make(cap, od=16, uint8=False)
    n_load = b2.load(p)
    assert n_load == n_saved - start == n_ret, (n_load, n_saved, start, n_ret)
    _, s2 = state_of(b2)
    for kk in COLS:
        exp = snap[kk][start:]
        assert s2[kk].shape == exp.shape, (kk, s2[kk].shape, exp.shape)
        assert np.array_equal(s2[kk], exp), \
            f"{kk}: loaded row differs from the pre-save snapshot"
    return start, n_saved, raw_bad, adder.n


def test_concurrent_adds(tmp):
    print("B. add() during save never leaks a newer row into the old prefix")
    # (cap, pre-filled adds, k concurrent adds, max_transitions, expected start)
    cases = [(3000, 4000, 800, None, 800),     # full + wrapping
             (3000, 2500, 800, None, 300),     # 500 free slots absorb 500
             (3000, 4000, 800, 2000, 0),       # overwrites miss the saved set
             (3000, 4000, 1500, 2000, 500)]
    for cap, pre, k, mt, exp in cases:
        start, n, raw_bad, added = _concurrent_case(tmp, cap, pre, k, mt, True)
        assert added == k
        assert start == exp, (cap, pre, k, mt, start, exp)
        if exp:
            assert raw_bad > 0, "test is vacuous: no overwritten row reached disk"
        print(f"  B. cap={cap} pre={pre} adds={k} max_t={mt}: start={start} "
              f"(expected {exp}), raw prefix rows overwritten={raw_bad}, "
              f"loaded {n - start} rows all == snapshot")
    # free-running adder racing the whole save (non-deterministic count)
    worst = 0
    for trial in range(3):
        start, n, raw_bad, added = _concurrent_case(
            tmp, 3000, 4000 + trial, None, None, False)
        assert added > 0 and start > 0, (added, start)
        assert start >= raw_bad
        worst = max(worst, added)
    print(f"  B. free-running adder (up to {worst} adds/save): every loaded "
          f"row == snapshot, start > 0")


def _measure(fn):
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return peak, dt


def _rss_peak(fn):
    try:
        import psutil
    except Exception:
        fn()
        return None
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    done = threading.Event()

    def sampler():
        while not done.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    fn()
    done.set(); th.join()
    return peak[0] - base


def _big_buffer():
    cap, od = 4000, 3 * 128 * 128               # 196.6 MB of uint8 pixels
    buf = make(cap, od=od, ad=12, pd=13, uint8=True)
    with buf._lock:
        buf.observations[:] = (np.arange(cap * od, dtype=np.int64)
                               .reshape(cap, od) % 251).astype(np.uint8)
        buf.size, buf.position = cap, 1234
    return buf


def test_memory_bounded(tmp):
    print("C. peak extra memory during save is bounded")
    buf = _big_buffer()
    total = sum(getattr(buf, k).nbytes for k in COLS)
    po, pn = os.path.join(tmp, "mem_old"), os.path.join(tmp, "mem_new")
    old_peak, old_dt = _measure(lambda: old_save(buf, po))
    new_peak, new_dt = _measure(lambda: buf.save(pn))
    shutil.rmtree(po); shutil.rmtree(pn)
    old_rss = _rss_peak(lambda: old_save(buf, po))
    new_rss = _rss_peak(lambda: buf.save(pn))
    assert files_equal(po, pn, COLS) is None
    mb = 1e6
    print(f"  C. buffer {total / mb:.1f} MB | tracemalloc peak: old "
          f"{old_peak / mb:.1f} MB ({100 * old_peak / total:.0f}%) in "
          f"{old_dt:.2f}s, new {new_peak / mb:.1f} MB "
          f"({100 * new_peak / total:.1f}%) in {new_dt:.2f}s")
    if old_rss is not None:
        print(f"  C. RSS delta: old {old_rss / mb:.1f} MB, new "
              f"{new_rss / mb:.1f} MB")
    assert old_peak > 0.8 * total, "witness: old path should copy ~all of it"
    assert new_peak < 0.25 * total, (new_peak, total)
    # boot: the old load np.load-ed every column fully, THEN copied it in
    b_old = make(buf.capacity, od=buf.obs_dim, ad=12, pd=13, uint8=True)
    b_new = make(buf.capacity, od=buf.obs_dim, ad=12, pd=13, uint8=True)
    lo_peak, _ = _measure(lambda: old_load_state(b_old, po))
    ln_peak, _ = _measure(lambda: b_new.load(pn))
    print(f"  C. load tracemalloc peak: old {lo_peak / mb:.1f} MB, new "
          f"{ln_peak / mb:.1f} MB")
    assert ln_peak < 0.25 * total, (ln_peak, total)
    del b_old, b_new
    shutil.rmtree(po); shutil.rmtree(pn)
    return buf


def test_old_format_loads(tmp):
    print("D. an old-format directory (no `start`) loads identically")
    for name, mk_src, mk_dst in [
            ("fixed->fixed", lambda: make(1500), lambda: make(1500)),
            ("fixed->smaller", lambda: make(1500), lambda: make(900)),
            ("fixed->block", lambda: make(2000), lambda: make(1000, growth=GROW))]:
        src = mk_src()
        fill(src, 2700)
        p = os.path.join(tmp, "oldfmt_" + name.replace(">", ""))
        shutil.rmtree(p, ignore_errors=True)
        old_save(src, p)
        assert "start" not in json.load(open(os.path.join(p, "manifest.json")))
        ref_buf, new_buf = mk_dst(), mk_dst()
        ref = old_load_state(ref_buf, p)
        n = new_buf.load(p)
        assert n == ref["n"], (name, n, ref["n"])
        assert new_buf.size == n and new_buf.position == n % new_buf.capacity
        assert new_buf.capacity == ref_buf.capacity, name
        assert new_buf.max_priority == ref["max_priority"]
        for k in COLS:
            got = np.asarray(getattr(new_buf, k)[np.arange(n)])
            assert np.array_equal(got, ref["cols"][k]), (name, k)
        print(f"  D. {name:15s} n={n} capacity={new_buf.capacity} identical")
    # pre-2026-09-01 directory: no restarts.npy, no proprio.npy
    src = make(1200)
    fill(src, 1500)
    p = os.path.join(tmp, "ancient")
    old_save(src, p)
    os.remove(os.path.join(p, "restarts.npy"))
    os.remove(os.path.join(p, "proprio.npy"))
    b = make(1200)
    with b._lock:                       # dirty the slots so zero-fill shows
        b.restarts[np.arange(1200)] = True
        b.proprio[np.arange(1200)] = 7.0
    n = b.load(p)
    assert not np.asarray(b.restarts[np.arange(n)]).any()
    assert not np.asarray(b.proprio[np.arange(n)]).any()
    print("  D. missing restarts/proprio columns restore as False / zeros")


class _TimedLock:
    def __init__(self):
        self._l = threading.Lock()
        self.holds = []
        self._t = 0.0

    def acquire(self, *a, **kw):
        r = self._l.acquire(*a, **kw)
        self._t = time.perf_counter()
        return r

    def release(self):
        self.holds.append(time.perf_counter() - self._t)
        self._l.release()

    __enter__ = acquire

    def __exit__(self, *exc):
        self.release()


def test_lock_holds(tmp, buf):
    print("E. the lock is held per chunk, never across a column")
    order = buf._chronological()
    t0 = time.perf_counter()
    _ = buf.observations[order]
    full_gather = time.perf_counter() - t0
    del _
    lk = _TimedLock()
    buf._lock = lk
    p = os.path.join(tmp, "locks")
    buf.save(p)
    buf._lock = threading.Lock()
    n = len(order)
    min_chunks = 0
    for k in COLS:
        col = getattr(buf, k)
        rb = col.dtype.itemsize * int(np.prod(col.shape[1:]))
        rows = max(1, min(buf._SAVE_CHUNK_ROWS, buf._SAVE_CHUNK_BYTES // rb))
        min_chunks += math.ceil(n / rows)
    mx = max(lk.holds)
    print(f"  E. {len(lk.holds)} lock holds (>= {min_chunks} chunks), "
          f"longest {mx * 1e3:.2f} ms vs one full-column gather "
          f"{full_gather * 1e3:.1f} ms")
    assert len(lk.holds) >= min_chunks
    assert mx < 0.25 * full_gather, (mx, full_gather)
    shutil.rmtree(p)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="rbsave_")
    try:
        test_bytes_unchanged(tmp)
        test_concurrent_adds(tmp)
        big = test_memory_bounded(tmp)
        test_old_format_loads(tmp)
        test_lock_holds(tmp, big)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[replay_save_memory_smoke] ALL PASS")
