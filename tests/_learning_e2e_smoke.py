"""END-TO-END: does the world model actually LEARN? (Mac-ok, ~1 min)

    PYTHONPATH=. ./venv/bin/python tests/_learning_e2e_smoke.py

WHY THIS EXISTS
---------------
Every other test in this repo checks a contract in isolation. None of them
asked the only question that matters: **when the real training loop runs
against the real prioritised replay buffer, does the loss go down?**

It did not, and nothing caught it. `BlockArray.__setitem__` ignored numpy
broadcasting, so `update_priorities` raised

    ValueError: shape mismatch: value array of shape (8,1) could not be
    broadcast to indexing result of shape (32,)

whenever a sampled window spanned two blocks. That raise escaped the
`for _ in range(train_iters)` loop in `_train_world_model` into its
`except ValueError`, which logged at DEBUG (invisible) and returned `{}`.

MEASURED CONSEQUENCES, all downstream of that one line:
  * every block ran ONE gradient step instead of train_iters = 384
  * metrics were discarded -> `WM loss: 0.0000 (n=0)` for 158,000 steps
  * the stage controller never saw a reconstruction error -> `WM-error=inf`
    was the ONLY value it ever held
  * `explore` forever -> IMAGINE unreachable -> the dream never ran
  * goal discovery starved -> `Skills learned: 2 (0 mastered)`

The unit test for BlockArray proves the shapes broadcast. It does NOT prove
the agent learns. This one drives the actual pipeline —
buffer -> sample_sequences(prioritized) -> train_step -> update_priorities —
over a learnable synthetic world and asserts the loss falls.

CONTRACTS
  0. The test PROVES it exercises a block boundary — an earlier version
     passed with the bug restored, which made it worthless.
  1. The full prioritised loop runs N iterations with ZERO exceptions (the
     live crash, reproduced end to end).
  2. EVERY iteration completes — not 1 of N. This is the 384x throttle.
  3. Metrics come back populated with the keys the loop and the stage
     controller actually read, all finite.
  4. THE LOSS GOES DOWN. A pipeline that runs without learning is the same
     failure wearing a green tick.
  5. Reconstruction error — the stage controller's spine — also falls, so
     the curriculum could actually advance.
  6. Priorities are genuinely updated (PER is doing work, not silently inert).
"""
import numpy as np
import torch

from developmental_ai.world_model.replay_buffer import MultiStreamReplayBuffer
from developmental_ai.world_model.rssm import WorldModel

OBS_DIM, ACT_DIM = 12, 4
# SEQ_LEN MATCHES PRODUCTION (world_model.sequence_length: 50). This is not a
# detail — at seq_len 8 a window straddles a 1000-transition block boundary
# only ~0.7% of the time, so an early version of this test PASSED WITH THE
# BUG RESTORED. It was measuring nothing. At 50 the chance is ~5%, so a run
# of this size crosses boundaries dozens of times, and
# `test_it_really_exercises_a_block_boundary` now asserts it happened rather
# than trusting the odds.
SEQ_LEN, BATCH = 50, 8
BLOCK = 1000                      # min allowed; the default 25,000 never spans
N_ADD = 4200                      # > 4 blocks


def _fill_buffer(buf, rng):
    """A LEARNABLE world: next obs is a fixed linear function of (obs, action).

    Deliberately learnable — if the dynamics were noise, a flat loss would be
    correct and contract 4 could not distinguish a working trainer from a
    broken one.
    """
    W = rng.standard_normal((OBS_DIM + ACT_DIM, OBS_DIM)).astype(np.float32)
    W *= 0.3
    obs = rng.standard_normal(OBS_DIM).astype(np.float32)
    for t in range(N_ADD):
        a = int(rng.integers(ACT_DIM))
        onehot = np.zeros(ACT_DIM, dtype=np.float32)
        onehot[a] = 1.0
        nxt = np.tanh(np.concatenate([obs, onehot]) @ W)
        done = (t % 200 == 199)
        buf.add(obs.copy(), a, float(nxt.sum() * 0.1), done, stream=0)
        # PRODUCTION CALLS THIS. `add` only sets `_grow_pending`; the actual
        # block allocation happens in `maybe_grow()`, which the real loop
        # invokes inside _train_world_model (developmental_loop.py:6899) so a
        # multi-hundred-MB allocation never stalls the agent mid-episode.
        # Omitting it kept this buffer pinned at ONE block, so no window could
        # ever span a boundary and the test was silently vacuous.
        if t % 200 == 0:
            buf.maybe_grow()
        obs = nxt if not done else rng.standard_normal(OBS_DIM).astype(np.float32)


def _build():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    # sample_sequences draws from the GLOBAL numpy RNG, so without this the
    # run-to-run crossing count varies and contract 0 becomes a coin flip.
    # Seeded, this file is reproducible: verified to FAIL (ValueError, shape
    # (8,1) vs (250,)) with the pre-fix __setitem__ restored, and to pass
    # with it. A test that has not been shown to fail proves nothing.
    np.random.seed(0)
    buf = MultiStreamReplayBuffer(
        num_streams=1, capacity=8000, obs_dim=OBS_DIM, action_dim=ACT_DIM,
        growth={"enabled": True, "block_transitions": BLOCK})
    _fill_buffer(buf, rng)
    wm = WorldModel(
        obs_dim=OBS_DIM, action_dim=ACT_DIM, stochastic_size=8,
        stochastic_classes=8, deterministic_size=64, hidden_dim=64,
        learning_rate=3e-4)
    return buf, wm


def _train(buf, wm, iters, collect=None):
    """The REAL loop: sample -> train_step -> update_priorities."""
    ok = 0
    for _ in range(iters):
        batch = buf.sample_sequences(
            batch_size=BATCH, seq_len=SEQ_LEN, device="cpu",
            prioritized=True)
        m, errs = wm.train_step(
            observations=batch["observations"], actions=batch["actions"],
            rewards=batch["rewards"], continues=batch["continues"],
            importance_weights=batch.get("weights"), return_per_sample=True)
        buf.update_priorities(batch["start_indices"], SEQ_LEN, errs)
        ok += 1
        if collect is not None:
            collect.append(m)
    return ok


def test_it_really_exercises_a_block_boundary():
    """Contract 0 — PROVE the test is not vacuous.

    A test that would pass with the bug present is worse than no test. An
    earlier version of this file did exactly that (seq_len 8 made a spanning
    window a ~0.7% event), so the crossing is now asserted, not assumed.
    """
    buf, _ = _build()
    s0 = buf.streams[0]
    spans = 0
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = buf.sample_sequences(batch_size=BATCH, seq_len=SEQ_LEN,
                                 device="cpu", prioritized=True)
        starts = np.asarray(b["start_indices"])[:, 1]
        idx = (starts[:, None] + np.arange(SEQ_LEN)[None, :]) % s0.capacity
        spans += int(((idx // BLOCK).min(axis=1)
                      != (idx // BLOCK).max(axis=1)).sum())
    assert spans > 0, (
        f"no sampled window crossed a block boundary in 200 batches, so this "
        f"file cannot detect the bug it exists for (seq_len={SEQ_LEN}, "
        f"block={BLOCK})")
    print(f"[learn-e2e] 0. {spans} sampled windows crossed a block boundary "
          f"— the buggy path IS exercised")


def test_the_full_prioritised_loop_runs_without_raising():
    """Contracts 1 + 2 — the live crash, end to end."""
    buf, wm = _build()
    iters = 120          # margin: boundary crossings are ~1 in 320 windows
    ran = _train(buf, wm, iters)
    assert ran == iters, (
        f"only {ran}/{iters} iterations completed. Live, exactly ONE ran per "
        f"block before update_priorities raised — a {iters}x throttle on the "
        f"world model that reported itself as silence.")
    print(f"[learn-e2e] 1+2. {ran}/{iters} prioritised iterations completed, "
          f"0 exceptions (block={BLOCK}, windows span boundaries)")


def test_metrics_are_populated_and_finite():
    """Contract 3 — the keys the loop and stage controller actually read."""
    buf, wm = _build()
    got = []
    _train(buf, wm, 5, collect=got)
    assert got and got[-1], "train_step returned no metrics"
    m = got[-1]
    for k in ("total", "reconstruction", "kl"):
        assert k in m, f"missing {k!r}; loop/stage-controller read this. {list(m)}"
        assert np.isfinite(m[k]), f"{k} is not finite: {m[k]}"
    print(f"[learn-e2e] 3. metrics populated + finite: "
          f"{ {k: round(float(m[k]), 3) for k in ('total', 'reconstruction', 'kl')} }")


def test_the_loss_actually_goes_down():
    """Contract 4 — the whole point. A green pipeline that learns nothing is
    the same failure with a nicer colour."""
    buf, wm = _build()
    got = []
    _train(buf, wm, 120, collect=got)
    tot = [float(m["total"]) for m in got]
    first, last = float(np.mean(tot[:10])), float(np.mean(tot[-10:]))
    assert last < first, (
        f"world-model loss did NOT fall: {first:.3f} -> {last:.3f}. The "
        f"pipeline runs but nothing is learned.")
    assert len(set(np.round(tot, 6))) > 10, (
        "loss is essentially constant — a pinned number is indistinguishable "
        "from a stale cache")
    print(f"[learn-e2e] 4. LOSS FALLS {first:.3f} -> {last:.3f} "
          f"({100 * (first - last) / abs(first):.0f}% over 120 iters, "
          f"{len(set(np.round(tot, 6)))} distinct values)")


def test_reconstruction_error_falls_so_the_curriculum_can_advance():
    """Contract 5 — the stage controller's spine.

    `WM-error` is the mean of recent reconstruction errors; the controller
    advances on its LEVEL and SLOPE. If this never falls, the agent stays in
    `explore` forever no matter how healthy the rest of the run looks.
    """
    buf, wm = _build()
    got = []
    _train(buf, wm, 120, collect=got)
    rec = [float(m["reconstruction"]) for m in got]
    first, last = float(np.mean(rec[:10])), float(np.mean(rec[-10:]))
    assert last < first, (
        f"reconstruction error did not fall: {first:.3f} -> {last:.3f}; the "
        f"stage controller could never leave `explore`")
    print(f"[learn-e2e] 5. reconstruction {first:.3f} -> {last:.3f} — the "
          f"stage controller now has a falling signal to act on")


def test_priorities_are_really_updated():
    """Contract 6 — PER must not be silently inert.

    This is what crashed. If it were quietly skipped instead, sampling would
    degrade to uniform and nothing would say so.
    """
    buf, wm = _build()
    s0 = buf.streams[0]
    before = np.array(s0.priorities[np.arange(min(2000, s0.size))],
                      dtype=np.float32).copy()
    _train(buf, wm, 30)
    after = np.array(s0.priorities[np.arange(min(2000, s0.size))],
                     dtype=np.float32)
    changed = int((before != after).sum())
    assert changed > 0, (
        "no priority changed across 30 iterations — prioritised replay is "
        "inert and sampling has silently degraded to uniform")
    assert np.all(np.isfinite(after)), "non-finite priority would poison sampling"
    print(f"[learn-e2e] 6. {changed} priorities updated across 30 iterations; "
          f"all finite")


if __name__ == "__main__":
    for fn in (test_it_really_exercises_a_block_boundary,
               test_the_full_prioritised_loop_runs_without_raising,
               test_metrics_are_populated_and_finite,
               test_the_loss_actually_goes_down,
               test_reconstruction_error_falls_so_the_curriculum_can_advance,
               test_priorities_are_really_updated):
        fn()
    print("[learn-e2e] ALL PASS")
