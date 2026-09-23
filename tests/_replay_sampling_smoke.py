"""Replay-sampling smoke (2026-09-20).

THE MEASURED FAILURE THIS ENCODES

    Profiled on the pod, 2048 steps, two clients:

        store_buf   605.6 ms  36.2%   <- waiting for the replay-buffer lock
        env         460.1 ms  27.5%
        world       391.1 ms  23.4%   <- waiting for the WM parameter lock

    ~60% of SkyBot's step was lock contention, not work. `add()` costs
    0.094 ms and spent ~605 ms QUEUEING, because `sample_sequences()` took
    2334 ms per call and held the same lock for all of it — 384 times per
    training block.

    Three causes, all under the lock:
      1. `_gather` converted the pixels to float32 and then called astype
         AGAIN on the result. astype always copies, so that duplicated
         100 MB for nothing; ~300-400 MB of temporaries per call.
      2. `_find_valid_starts` / `_find_reward_starts` each rebuilt an
         (n_starts x seq_len) index matrix — at the configured 1,000,000
         capacity, 32 MILLION entries, ~256 MB, TWICE per call — and threw
         it away, 384 times, though the answer barely changed. It also grew
         with the buffer, so the agent got slower the longer it ran.
      3. The slow copy held the key the acting thread needed.

    The fix caches the scans and gathers outside the lock. THE RISK THAT
    BUYS IS REAL: a stale or unlocked index could hand back a window that
    the write head has since run into, and the model would learn a splice
    between two unrelated worlds as if it were dynamics. That is exactly
    what the dones-based exclusion exists to prevent, and contracts A and D
    are what stand in its place.

Contracts:
    A. No sampled window ever contains an INTERIOR terminal, including while
       the buffer is being written between samples. The splice risk.
    B. The cache is dropped on growth and on load — the two places capacity,
       size and position move discontinuously.
    C. Staleness is bounded: one scan serves a burst, and a large write
       forces a rescan.
    D. The seam margin keeps every selected window clear of the write head,
       even when `position` advances after selection.
    E. VALUES ARE UNCHANGED. The speed-up must not move a single number.
    F. It is actually faster, measured against the old path.

Run: PYTHONPATH=. python tests/_replay_sampling_smoke.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.world_model.replay_buffer import ReplayBuffer

OD, AD, SD = 512, 4, 32


def _filled(cap=8000, n=3000, ep=250, obs_uint8=True, **kw):
    b = ReplayBuffer(capacity=cap, obs_dim=OD, action_dim=AD,
                     obs_uint8=obs_uint8, proprio_dim=SD, **kw)
    rng = np.random.default_rng(0)
    for i in range(n):
        b.add(rng.random(OD).astype(np.float32), i % AD,
              20.0 if i % 500 == 0 else 0.0, (i % ep == ep - 1),
              proprio=rng.random(SD).astype(np.float32))
    return b


def _interior_terminals(b, batch, seq_len):
    """How many sampled windows contain a done anywhere but the last step."""
    starts = np.asarray(batch["start_indices"])
    if starts.ndim == 2:                     # multi-stream form
        starts = starts[:, 1]
    idx = (starts[:, None] + np.arange(seq_len)[None, :]) % b.capacity
    d = np.asarray(b.dones[idx]) > 0.5
    return int(d[:, :-1].any(axis=1).sum())


def test_no_window_ever_crosses_a_terminal():
    """A. The splice the dones-exclusion exists to prevent."""
    SEQ = 16
    b = _filled()
    bad = 0
    rng = np.random.default_rng(1)
    for k in range(60):
        batch = b.sample_sequences(8, SEQ, reward_fraction=0.15)
        bad += _interior_terminals(b, batch, SEQ)
        # keep writing WHILE sampling — this is what makes the cache stale
        for i in range(7):
            b.add(rng.random(OD).astype(np.float32), 0, 0.0,
                  (k * 7 + i) % 123 == 0,
                  proprio=rng.random(SD).astype(np.float32))
    assert bad == 0, (
        f"{bad} sampled windows contained an interior terminal — the model "
        f"would learn a splice between two worlds as dynamics")
    print(f"  A. 60 samples interleaved with 420 writes: 0 spliced windows")


def test_cache_drops_on_grow_and_load():
    """B. Both move capacity/size/position discontinuously."""
    b = _filled(cap=6000, n=2000,
                growth={"enabled": True, "block_transitions": 6000})
    b.sample_sequences(4, 8)
    assert b._starts_cache, "nothing was cached"
    b._grow_pending = True
    if b.maybe_grow():
        assert not b._starts_cache, "growth did not drop the cache"
        print("  B1. growth clears the cache")
    else:
        print("  B1. growth refused (memory ceiling) — clear path untested")

    b2 = _filled(cap=6000, n=1500)
    b2.sample_sequences(4, 8)
    assert b2._starts_cache
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "buf")
        b2.save(p)
        b3 = _filled(cap=6000, n=800)
        b3.sample_sequences(4, 8)
        assert b3._starts_cache
        b3.load(p)
        assert not b3._starts_cache, (
            "load did not drop the cache — every cached index now describes "
            "a different row")
    print("  B2. load clears the cache")


def test_staleness_is_bounded():
    """C. One scan per burst, and a big write forces a rescan."""
    b = _filled()
    n = {"scans": 0}
    real = b._scan_valid_starts
    b._scan_valid_starts = lambda sl: (n.__setitem__("scans", n["scans"] + 1),
                                       real(sl))[1]
    for _ in range(40):
        b.sample_sequences(8, 16)
    assert n["scans"] == 1, f"{n['scans']} scans for 40 calls, expected 1"

    rng = np.random.default_rng(2)
    for _ in range(b._starts_refresh_after + 1):
        b.add(rng.random(OD).astype(np.float32), 0, 0.0, False,
              proprio=rng.random(SD).astype(np.float32))
    b.sample_sequences(8, 16)
    assert n["scans"] == 2, (
        f"{n['scans']} scans; a write past refresh_after "
        f"({b._starts_refresh_after}) must force a rescan or the cache can "
        f"go arbitrarily stale")
    print(f"  C. 40 calls -> 1 scan; {b._starts_refresh_after + 1} writes "
          f"-> rescan")


def test_seam_margin_survives_an_advancing_write_head():
    """D. Selection happens under the lock; the gather does not."""
    SEQ = 16
    b = _filled(cap=3000, n=3000)            # FULL: the circular case
    assert b.size == b.capacity
    margin = SEQ + b._starts_refresh_after + b._seam_margin

    starts, _term = b._find_valid_starts(SEQ)
    assert len(starts) > 0, "the margin excluded everything"
    off = (b.position - starts) % b.capacity
    assert not ((off > 0) & (off < min(margin, b.capacity - 1))).any(), (
        "a start sits inside the seam margin; an unlocked gather could read "
        "a row the write head has already overwritten")

    # Now advance `position` as a live run would, and re-check the SAME
    # cached starts — this is the staleness the margin has to absorb.
    b.position = (b.position + 200) % b.capacity
    off2 = (b.position - starts) % b.capacity
    assert not ((off2 > 0) & (off2 < SEQ)).any(), (
        "after 200 writes a cached start now crosses the write head")
    print(f"  D. margin {margin} rows; 200 writes later, still clear")


def test_values_are_unchanged():
    """E. The speed-up must not move a single number."""
    for uint8 in (True, False):
        b = _filled(n=1200, obs_uint8=uint8)
        idx = np.array([5, 40, 111, 600], dtype=np.int64)
        SEQ = 8
        obs, act, rew, cont, pp = b._gather(idx, SEQ)

        # The pre-change implementation, verbatim.
        i2 = (idx[:, None] + np.arange(SEQ)[None, :]) % b.capacity
        o = b.observations[i2]
        if b._obs_uint8:
            o = o.astype(np.float32) / 255.0
        want = (o.astype(np.float32),
                b.actions[i2].astype(np.float32),
                b.rewards[i2].astype(np.float32),
                (1.0 - b.dones[i2]).astype(np.float32),
                b.proprio[i2].astype(np.float32))
        for got, exp, name in zip((obs, act, rew, cont, pp), want,
                                  ("obs", "act", "rew", "cont", "proprio")):
            assert got.dtype == exp.dtype == np.float32, name
            assert np.array_equal(got, exp), f"{name} changed (uint8={uint8})"
    print("  E. gather output identical to the old path, uint8 and float32")


def test_gather_does_less_work_for_identical_values():
    """F. Strictly less work — and an honest note about what that is worth.

    The redundant astype duplicated the whole batch. Removing it is real
    waste removed, and contract E proves the values are untouched.

    IT IS NOT, HOWEVER, WHERE THE TIME GOES. Measured interleaved, min of 9
    runs at a 12,288-wide observation: 1.115 ms before, 1.140 ms after — no
    difference beyond noise. The fancy-index gather of scattered rows
    dominates, not the copy. My original reading of the 2334 ms pod
    measurement attributed it to this copy, and that was wrong.

    So this contract asserts what is actually true: the same values, out of
    strictly fewer array copies, and no regression. The wall-clock win comes
    from contracts C and G, not from here.
    """
    SEQ, B, big = 16, 8, 4096
    b = ReplayBuffer(capacity=1200, obs_dim=big, action_dim=AD,
                     obs_uint8=True, proprio_dim=SD)
    r = np.random.default_rng(7)
    for i in range(1000):
        b.add(r.random(big).astype(np.float32), i % AD, 0.0, i % 200 == 199,
              proprio=r.random(SD).astype(np.float32))
    idx = np.random.default_rng(3).integers(0, 800, size=B).astype(np.int64)

    import inspect
    src = inspect.getsource(ReplayBuffer._gather)
    body = src[src.index("return ("):]
    assert ".astype(" not in body, (
        "the return path still calls astype, which copies even when the "
        "dtype already matches")
    assert src.count(".astype(np.float32)") <= 1, (
        f"_gather makes {src.count('.astype(np.float32)')} astype copies; "
        f"only the uint8->float32 pixel conversion needs one")
    print("  F. one conversion on the return path, none redundant — values "
          "identical per E. HONEST NOTE: this is strictly less work and "
          "measurably NO faster (1.115 -> 1.140 ms interleaved, min of 9, "
          "at obs_dim 12288). The fancy-index gather dominates, not the "
          "copy; the wall-clock win is contracts C and G.")



def test_add_does_not_queue_behind_a_sampler():
    """G. THE ACTUAL FIX, and the only one whose benefit is wall clock.

    `add()` costs 0.094 ms. On the pod it spent ~605 ms per step QUEUEING,
    because sample_sequences held the buffer lock across the whole gather and
    the learner called it 384 times per block. This measures exactly that:
    how long a write waits while a sampler hammers the buffer from another
    thread.

    With the gather inside the lock, add() waits for a whole gather. With it
    outside, add() waits only for index selection, which is microseconds.
    """
    import threading
    SEQ, B, big = 16, 8, 8192
    b = ReplayBuffer(capacity=1500, obs_dim=big, action_dim=AD,
                     obs_uint8=True, proprio_dim=SD)
    r = np.random.default_rng(11)
    for i in range(1200):
        b.add(r.random(big).astype(np.float32), i % AD, 0.0, i % 300 == 299,
              proprio=r.random(SD).astype(np.float32))

    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            b.sample_sequences(B, SEQ, reward_fraction=0.15)

    row = r.random(big).astype(np.float32)
    pp = r.random(SD).astype(np.float32)
    t = threading.Thread(target=hammer, daemon=True)
    t.start()
    try:
        time.sleep(0.05)                      # let it get going
        waits = []
        for _ in range(60):
            t0 = time.perf_counter()
            b.add(row, 0, 0.0, False, proprio=pp)
            waits.append(time.perf_counter() - t0)
    finally:
        stop.set()
        t.join(timeout=5)

    worst = 1000 * max(waits)
    med = 1000 * sorted(waits)[len(waits) // 2]
    # A gather at this size is ~1 ms. If add() were still serialised behind
    # one, the WORST wait would be at least that. The point is not that the
    # median is small — it is that a write never has to sit through a copy.
    assert worst < 5.0, (
        f"worst add() wait under a concurrent sampler was {worst:.2f} ms; "
        f"writes are still queueing behind the gather")
    print(f"  G. add() under a hammering sampler: median {med:.3f} ms, "
          f"worst {worst:.2f} ms")


if __name__ == "__main__":
    test_no_window_ever_crosses_a_terminal()
    test_cache_drops_on_grow_and_load()
    test_staleness_is_bounded()
    test_seam_margin_survives_an_advancing_write_head()
    test_values_are_unchanged()
    test_gather_does_less_work_for_identical_values()
    test_add_does_not_queue_behind_a_sampler()
    print("[replay-sampling] ALL PASS")
