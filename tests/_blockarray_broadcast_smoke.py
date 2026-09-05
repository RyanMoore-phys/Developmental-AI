"""BlockArray broadcast smoke — the bug that throttled the world model 384x.

    PYTHONPATH=. ./venv/bin/python tests/_blockarray_broadcast_smoke.py

THE LIVE FAILURE (surfaced 2026-09-04, after the WARNING was raised above
DEBUG level; it had been silent for the life of the project)

    ValueError: shape mismatch: value array of shape (8,1) could not be
    broadcast to indexing result of shape (32,)
      replay_buffer.py:700 in update_priorities
        self.priorities[idx] = errors[:, None]

`update_priorities` does the textbook numpy thing:

    idx = starts[:, None] + arange(seq_len)      # (n_seq, seq_len)
    priorities[idx] = errors[:, None]            # (n_seq, 1) -> broadcast

`BlockArray` stands in for an ndarray but did not honour broadcasting. Its
`_split` flattens the index, and the MULTI-BLOCK branch then did `value[m]`
with a (n_seq, 1) value against a 1-D mask — shapes that cannot align. The
SINGLE-BLOCK fast path happened to work, which is exactly why this survived
review: it only fires when a sampled window SPANS TWO BLOCKS, so it appeared
when the replay buffer became block storage and never before.

WHAT IT COST. The raise propagated out of `_train_world_model`'s
`for _ in range(train_iters)` loop into its `except ValueError` (which logged
at DEBUG, so nothing printed) and returned `{}`. Consequences, all measured:

  * every block ran ONE gradient step instead of train_iters = **384**
  * the block's metrics were discarded, so `WM loss` read `0.0000 (n=0)`
  * the stage controller never got a reconstruction error -> `WM-error=inf`
    was the ONLY value in a 158,000-step log
  * `explore` forever -> IMAGINE unreachable -> the dream never ran
  * goal discovery starved -> `Skills learned: 2 (0 mastered)`

One shape bug in a container, thereby throttling the world model and blinding
the curriculum. The lesson is CLAUDE.md §5 in a new costume: the failure was
visible the whole time, at a log level nobody reads.

CONTRACTS
  1. The exact live shapes work: (n_seq, 1) written into an (n_seq, seq_len)
     index that SPANS BLOCKS.
  2. The values actually land — every element of the window gets its row's
     error, verified against a plain ndarray doing the same assignment.
  3. Scalar and full-shape writes still work (no regression on the paths that
     already worked).
  4. A genuinely incompatible shape still RAISES — the fix must not paper
     over real errors by silently reshaping.
  5. The single-block fast path is still exercised and still correct.
"""
import numpy as np

from developmental_ai.world_model.replay_buffer import BlockArray


def _mk(n, block, dtype=np.float32, row_shape=()):
    """A BlockArray of at least `n` slots, split into `block`-sized blocks."""
    return BlockArray(block=block, row_shape=row_shape, dtype=dtype,
                      initial_blocks=max(1, -(-n // block)))


def test_the_exact_live_shapes_across_a_block_boundary():
    """Contract 1 — (8,1) into a (8,4) index spanning two blocks."""
    n_seq, seq_len, block = 8, 4, 16
    ba = _mk(64, block)
    # start the windows so they straddle the boundary at 16
    starts = np.arange(n_seq) + 13          # 13..20 -> windows cross 16
    idx = (starts[:, None] + np.arange(seq_len)[None, :]) % 64
    assert idx.min() // block != idx.max() // block, (
        "precondition: this test is only meaningful if the index spans "
        "more than one block")
    errors = (np.arange(n_seq, dtype=np.float32) + 1.0)[:, None]  # (8,1)
    ba[idx] = errors                         # this is what crashed live
    print(f"[blockarray] 1. wrote {errors.shape} into an index of "
          f"{idx.shape} spanning blocks "
          f"{idx.min() // block}..{idx.max() // block} without raising")


def test_the_values_actually_land():
    """Contract 2 — not raising is not the same as being right."""
    n_seq, seq_len, block, n = 8, 4, 16, 64
    ba = _mk(n, block)
    # NON-OVERLAPPING windows. With overlapping ones (starts 13,14,15...) a
    # later row legitimately overwrites an earlier row's cells, so a
    # per-element "each window carries its own error" check would be testing
    # the wrong thing — it fails against a plain ndarray too.
    starts = 13 + seq_len * np.arange(n_seq)
    idx = (starts[:, None] + np.arange(seq_len)[None, :]) % n
    assert len(np.unique(idx)) == idx.size, "precondition: windows disjoint"
    errors = (np.arange(n_seq, dtype=np.float32) + 1.0)[:, None]
    ba[idx] = errors
    ref = np.zeros(n, dtype=np.float32)
    ref[idx] = errors                        # plain ndarray, same assignment
    got = ba[np.arange(n)]
    assert np.array_equal(got, ref), (
        f"BlockArray disagrees with ndarray broadcasting:\n"
        f"  got {got[10:26]}\n  ref {ref[10:26]}")
    # every element of every window carries ITS OWN row's error
    for r in range(n_seq):
        for c in range(seq_len):
            assert got[idx[r, c]] == errors[r, 0], (r, c)
    print(f"[blockarray] 2. values match a plain ndarray exactly; each "
          f"window carries its own row's error")


def test_scalar_and_full_shape_writes_still_work():
    """Contract 3 — the paths that already worked must keep working."""
    ba = _mk(64, 16)
    ba[5] = 2.5                              # scalar key
    assert ba[np.array([5])][0] == 2.5
    idx = np.array([[1, 2], [30, 31]])       # spans blocks
    ba[idx] = np.array([[1.0, 2.0], [3.0, 4.0]])   # exact-shape value
    got = ba[idx.reshape(-1)]
    assert np.array_equal(got, np.array([1.0, 2.0, 3.0, 4.0], np.float32)), got
    ba[np.array([7, 40])] = 9.0              # scalar broadcast to 1-D index
    assert np.array_equal(ba[np.array([7, 40])], np.array([9.0, 9.0], np.float32))
    print("[blockarray] 3. scalar keys, exact-shape values and scalar "
          "broadcast all still correct")


def test_incompatible_shapes_still_raise():
    """Contract 4 — do not paper over real errors."""
    ba = _mk(64, 16)
    idx = np.array([[1, 2, 3], [30, 31, 32]])        # (2,3), spans blocks
    try:
        ba[idx] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # (5,) — nonsense
    except ValueError:
        print("[blockarray] 4. a genuinely incompatible value still raises "
              "(the fix broadcasts, it does not reshape blindly)")
        return
    raise AssertionError(
        "an incompatible value was silently accepted — that would hide the "
        "next bug of this class instead of surfacing it")


def test_single_block_fast_path():
    """Contract 5 — the branch that already worked, still exercised."""
    block = 16
    ba = _mk(64, block)
    idx = np.array([[1, 2, 3], [4, 5, 6]])           # entirely in block 0
    assert idx.min() // block == idx.max() // block, "precondition"
    ba[idx] = np.array([[7.0], [8.0]])                # (2,1) broadcast
    got = ba[idx.reshape(-1)]
    assert np.array_equal(
        got, np.array([7.0, 7.0, 7.0, 8.0, 8.0, 8.0], np.float32)), got
    print("[blockarray] 5. single-block fast path still broadcasts correctly")


if __name__ == "__main__":
    for fn in (test_the_exact_live_shapes_across_a_block_boundary,
               test_the_values_actually_land,
               test_scalar_and_full_shape_writes_still_work,
               test_incompatible_shapes_still_raise,
               test_single_block_fast_path):
        fn()
    print("[blockarray] ALL PASS")
