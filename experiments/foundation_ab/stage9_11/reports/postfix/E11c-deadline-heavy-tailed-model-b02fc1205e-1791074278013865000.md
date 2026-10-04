# A/B report: E11c-deadline-heavy-tailed-model

**Verdict: reject**

- CI upper bound 0.001556 < delta 0.01: the hypothesised effect is ruled out
- note: criterion unmet: baseline 'constant' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: with a model whose call latency is heavy-tailed (2% of calls take 30 ms) under a 10 ms deadline, >= 1% more decisions take over TWICE the deadline than with a constant-latency model of equal mean: the 'hard' deadline is bounded only by one chunk's worst case
- primary metric: `overrun_fraction` (higher is better)
- acceptance rule: accept iff heavy_tail beats constant on overrun_fraction (higher is better, basis=final) by mean >= 0.01 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.01 or candidate failures exceed 0; else inconclusive
- arms: baseline `constant`, candidate `heavy_tail`, ablation `heavy_tail_chunk8`, alternative `None`
- seeds: [0, 1, 2, 3, 4]; splits: ['e11:deadline#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071311533965000 ns, hash `b02fc1205e18649b4ab7d6b1e735b81520ef7b1860d3f8371cefdc9be3f31db1`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| constant | 5/5 | 0 | 0 | 0.002667 | 0.003651 | 750 | 0 | 6.020 | - |
| heavy_tail | 5/5 | 0 | 0 | 0 | 0 | 750 | 0 | 3.930 | - |
| heavy_tail_chunk8 | 5/5 | 0 | 0 | 0 | 0 | 750 | 0 | 5.881 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | -0.002667 | -0.006889 | 0.001556 | 0 |
| equal_interactions | 5 | -0.002667 | -0.006889 | 0.001556 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| constant | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0.006667 | 150 | 1.091 | - | 103.6 |  |
| heavy_tail | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 0.739 | - | 103.6 |  |
| heavy_tail_chunk8 | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 0.019 | - | 103.6 |  |
| heavy_tail | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 0.800 | - | 103.6 |  |
| heavy_tail_chunk8 | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 1.457 | - | 103.6 |  |
| constant | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 1.224 | - | 103.6 |  |
| heavy_tail_chunk8 | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 1.480 | - | 103.6 |  |
| constant | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0.006667 | 150 | 1.162 | - | 103.6 |  |
| heavy_tail | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 0.931 | - | 103.6 |  |
| constant | 3 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 1.194 | - | 103.6 |  |
| heavy_tail | 3 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 0.673 | - | 103.6 |  |
| heavy_tail_chunk8 | 3 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 1.469 | - | 103.6 |  |
| heavy_tail | 4 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 0.787 | - | 103.6 |  |
| heavy_tail_chunk8 | 4 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 1.456 | - | 103.6 |  |
| constant | 4 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 1.349 | - | 103.6 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
