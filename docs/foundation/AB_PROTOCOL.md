# A/B protocol: how a learning comparison is run here

Implements plan §7.2. Code: `developmental_ai/foundation/experiments/ab.py`.
CLI: `tools/ab_compare.py`. Contracts: `tests/_foundation_ab_smoke.py`,
`tests/unit/test_foundation_ab_unit.py`.

The harness exists because of CLAUDE.md §9: the reward number has been wrong
every time it was read after the fact. A comparison here is an argument
written down **before** the data exists, and the harness refuses a verdict
otherwise.

## 1. Preregister, then freeze

```python
from developmental_ai.foundation.experiments import Preregistration, make_split

split = make_split("minerl:paper:stream0", episode_ids, heldout_fraction=0.2)
prereg = Preregistration(
    name="magnet-v2-vs-v1",
    hypothesis="v2 magnet raises logs per 10k steps",
    primary_metric="logs_per_10k", direction="higher", delta=0.5,
    baseline_arm="v1", candidate_arm="v2", ablation_arm="v2_no_fovea",
    seeds=(0, 1, 2, 3, 4), basis="equal_interactions",
    resource_budget={"max_wall_seconds": 3600, "max_interactions": 200_000},
    failure_conditions=("NaN loss", "client never boots"),
    split_fingerprints=(split.fingerprint,),
).freeze()
```

| Field | Meaning |
|---|---|
| `primary_metric`, `direction` | the one number that decides; `higher` or `lower` is better |
| `delta` | the smallest improvement worth having, in metric units |
| `basis` | `equal_interactions` (default), `equal_time`, or `final` |
| `seeds` | fixed here; `run_ab` refuses any other set |
| `ablation_arm` | the candidate minus the new component (plan §7.2.8) |
| `alternative_arm` | a simple alternative, where feasible |
| `resource_budget` | `max_wall_seconds`, `max_py_alloc_mb`, `max_interactions` per run |
| `max_candidate_failure_fraction` | how often the candidate may fail before it is rejected (default 0) |
| `split_fingerprints` | optional: pins the exact data |

`freeze()` stamps a sha256 of every field plus the freeze time.

**No verdict is computed** (`PreregistrationViolation`) when the prereg was
edited after freezing, the results were recorded under a different prereg
hash, or the prereg was frozen after the first recorded result. That last
check catches the edit-then-refreeze move, because the hash it produces is
self-consistent. A run also refuses an unfrozen prereg and any change to the
seeds.

## 2. Arms

`arm(seed, split) -> dict`. It must contain the primary metric as a finite
number. It may also contain:

- `curve`: a list of `{"interactions", "seconds", "value"}` points, both axes
  non-decreasing.
- `interactions` and `model_updates`.
- Any other JSON-able field.

Every arm receives the **same frozen `DataSplit`** for a given (seed, split).
A split is made per episode by `runtime.splits`, so adjacent frames never
straddle it. The fingerprint of the split is recorded per run, and a pair
whose fingerprints differ voids the comparison.

An arm that raises, returns malformed or non-finite metrics, or overruns the
budget is **recorded** (status `failed` / `budget_exceeded`, error text kept).
It is never skipped or retried.

## 3. The decision

For each paired (seed, split) unit, `d = sign × (candidate − baseline)` on the
preregistered basis. Equal-interaction and equal-time values are read off the
curves at the pair's common budget, using a step function. All three bases
are reported, and a sign disagreement between them is noted.

The CI is a paired percentile bootstrap, widened by
`t(n−1, .975) / 1.96 · sqrt(n/(n−1))`. At n = 3 the raw percentile bootstrap
false-accepts a null difference about 15% of the time; the widened interval
does so about 3% of the time. The smoke test measures both.

| Condition | Verdict |
|---|---|
| fewer than 3 preregistered seeds | **inconclusive (screening only)** |
| candidate failures above the allowance | **reject** |
| fewer than 3 usable pairs | inconclusive |
| CI low > 0 and mean ≥ delta | **accept** |
| CI high < delta | **reject** (the effect is ruled out) |
| otherwise | inconclusive |

The ablation and alternative comparisons are reported, but they never decide
the verdict. Any verdict other than accept keeps the baseline. Three seeds do
not guarantee reliability (§7.2.6); expand the seeds when the uncertainty
could change the decision.

## 4. Resources

Each run records:

- **wall seconds**
- **tracemalloc peak**: Python allocations, with the peak reset at the start
  of each run.
- **process RSS high-water mark**: from `getrusage`. It is monotone across the
  process, so it bounds a run rather than measuring it, and the report labels
  it that way.

GPU memory is not measured. Add it to the arm's metrics when it matters.

Equal-time values come from the arms' own clocks. They vary with machine load,
so a verdict on basis `equal_time` is not bit-reproducible.

## 5. Reports: negative results are kept

`run_ab(..., out_dir=D)` writes these files under a unique name that is never
overwritten:

- `D/<name>-<hash10>-<ns>.json`
- `D/<name>-<hash10>-<ns>.md`

It also appends a line to `D/index.jsonl`, whatever the verdict.
`load_report(path)` + `decide()` re-derives the verdict from the stored file.

```bash
PYTHONPATH=. python tools/ab_compare.py \
    developmental_ai.foundation.experiments.ab:demo_experiment --out runlogs/ab
```

The exit code is 0 for accept, 1 for reject, 2 for inconclusive and 3 for a
refused verdict. The CLI deliberately has no seed flag.

## 6. Shared worlds

Plan §7.2 notes that agents on one Paper server are not independent runs. An
arm that uses the shared server must report this in its metrics, for example
`{"shared_world": true, "peers": 2}`. Any wording that claims independence
has to come from isolated instances.
