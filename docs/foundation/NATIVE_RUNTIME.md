# Native runtime: what would justify leaving Python, and how it would be proven

Covers plan Stage 12, items 4–7. Code:

- `developmental_ai/foundation/transfer/profiling.py`
- `developmental_ai/foundation/transfer/parity.py`
- `tools/profile_loop.py`

Contracts: `tests/_foundation_transfer_smoke.py` (F, G).

## Status: no native port exists, and none is being started

**No Rust (or other native) code has been written.** No profile of the full
loop on the training host has yet shown a stable, CPU-bound bottleneck that a
port would remove. The only measured bottleneck so far argues *against* a
port:

- The replay-sampling profile (`tests/_replay_sampling_smoke.py`) showed about
  60% of the step spent in **lock contention**, not compute.
- That was fixed in Python, by caching scans and gathering outside the lock.

A Rust copy of the code before that fix would have been a faster way to wait
on the same lock.

The VLM (`fovea_interval`) and the GPU world-model update already run in
native code. CPU-side Python is not known to be the limit on the 16 GB `main`
host.

## Item 4: profile the full loop before porting

```bash
PYTHONPATH=. python tools/profile_loop.py MODULE:FUNCTION --iterations 3 --json prof.json
PYTHONPATH=. python tools/profile_loop.py --demo        # fixture loop
```

The report has three parts:

- **Top hotspots by own time.** cProfile inflates small calls, so use this for
  ranking only.
- **Python versus builtin/C call counts, per iteration.** A builtin count that
  grows with the data size is the signature of per-element work.
- **Named `Sections` timings.** Use these from an unprofiled run for absolute
  step rates.

A port decision needs this report from the **training host**, on the live
config (`configs/minecraft_skybot.yaml`, `num_envs: 2`), over at least one
full training block. The fixture demo is not evidence.

What the fixture demo does show: about 60% of its wall time is record
construction and validation (`Record.__post_init__`, measured cumulatively),
at about 230 builtin calls per interaction. If records ever enter the hot loop, the answer is
batching validation per step, not porting it.

## Item 5: selection criteria for one Rust component

A candidate must meet **all** of the following:

1. **Stable API.** Its signature and semantics have been unchanged for a
   training-host-proven run, and nothing in `NEXT_OBJECTIVES.md` plans to
   redesign it.
2. **CPU-bound and measured.** It takes at least 15% of the training-host step
   in the item-4 report, and the time is its own time, not waiting on a lock,
   the GPU, the network or the server.
3. **Batched.** It takes and returns whole arrays (a step or a segment), so a
   call crosses Python/Rust O(1) times per batch.
4. **Python reference retained.** The Python implementation stays in the tree
   as the reference and the fallback. A config switch selects between them,
   and the default stays Python until parity and speed are both shown on the
   training host.
5. **Parity tested.** `assert_native_parity` passes on recorded real inputs,
   including edge cases (empty batch, NaN, ABSENT-coded entries), at a stated
   `rtol`.
6. **Measured gain.** The training-host step rate improves by more than run-to-run
   noise, measured with `tools/ab_compare.py` on basis `equal_time`, with the
   Python reference as the baseline arm.

Likely first candidates, **if** a profile ever ranks them: replay-buffer
index scans and gathers (`world_model/replay_buffer.py`), and
episodic-memory nearest-neighbour search. Neither currently qualifies.

## Item 6: keep operations batched

Never expose a per-element native function. Per-element crossings appear in
the profiler as builtin calls that grow with N. Contract G shows a loop going
from 1,001 to 4,001 crossings on 4× the data, while the batched version stays
at 3. A native function that is called per element repeats the cost it was
meant to remove.

## Item 7: custom GPU kernels

Write a custom GPU kernel only after a profile shows a remaining GPU-side
need, such as a fused op that dominates the update. Nothing does today. VLM
frequency, not kernel speed, is the measured GPU trade-off (CLAUDE.md §6:
`fovea_interval: 12` cost 28% of the step rate).

## Parity test template

```python
from developmental_ai.foundation.transfer.parity import assert_native_parity

def test_native_gather_parity():
    from developmental_ai.world_model.replay_buffer import _gather_py as ref
    native = pytest_importorskip_equivalent("skybot_native").gather   # or skip
    inputs = load_recorded_cases("runlogs/parity/gather/*.npz")       # real data
    rep = assert_native_parity(ref, native, inputs, rtol=1e-6, atol=0.0)
    print(f"parity over {rep.n_cases} cases, max rel err {rep.max_rel_err:.1e}, "
          f"speedup x{rep.speedup:.1f}")
```

`assert_native_parity(py_fn, native_fn, inputs, rtol, atol=0.0)` gives each
function its own deep copy of each case. It then checks four things:

- output structure, shape and dtype kind;
- values within tolerance, with NaN and inf in the same positions;
- the arguments **after** the call, so in-place mutation must match;
- that both functions raise the same exception type, when either raises.

It returns the timings of both functions. Contract F checks that the helper
catches a 1e-4 perturbation, a moved NaN, a change of dtype kind, a different
mutation, and a mismatched exception.
