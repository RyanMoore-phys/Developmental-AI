# A/B report: E11c-deadline-heavy-tailed-model

**Verdict: accept**

- mean improvement 0.06667 >= delta 0.01, CI [0.04344, 0.09411] excludes 0
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
| constant | 5/5 | 0 | 0 | 0.01867 | 0.01282 | 750 | 0 | 5.341 | - |
| heavy_tail | 5/5 | 0 | 0 | 0.08533 | 0.01193 | 750 | 0 | 3.369 | - |
| heavy_tail_chunk8 | 5/5 | 0 | 0 | 0.1547 | 0.008692 | 750 | 0 | 5.636 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.06667 | 0.04344 | 0.09411 | 0 |
| equal_interactions | 5 | 0.06667 | 0.04344 | 0.09411 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | -0.06933 | -0.09467 | -0.044 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| constant | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0.02667 | 150 | 0.911 | - | 363.4 |  |
| heavy_tail | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0.07333 | 150 | 0.602 | - | 363.4 |  |
| heavy_tail_chunk8 | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0.16 | 150 | 1.143 | - | 363.4 |  |
| heavy_tail | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0.09333 | 150 | 0.760 | - | 363.4 |  |
| heavy_tail_chunk8 | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0.1467 | 150 | 1.159 | - | 363.4 |  |
| constant | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0.03333 | 150 | 1.011 | - | 363.4 |  |
| heavy_tail_chunk8 | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0.1467 | 150 | 1.133 | - | 363.4 |  |
| constant | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 150 | 1.312 | - | 363.4 |  |
| heavy_tail | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0.1 | 150 | 0.736 | - | 363.4 |  |
| constant | 3 | e11:deadline#skybot-foundation-split-v1 | ok | 0.02 | 150 | 0.987 | - | 363.4 |  |
| heavy_tail | 3 | e11:deadline#skybot-foundation-split-v1 | ok | 0.07333 | 150 | 0.608 | - | 363.4 |  |
| heavy_tail_chunk8 | 3 | e11:deadline#skybot-foundation-split-v1 | ok | 0.1667 | 150 | 1.098 | - | 363.4 |  |
| heavy_tail | 4 | e11:deadline#skybot-foundation-split-v1 | ok | 0.08667 | 150 | 0.663 | - | 363.4 |  |
| heavy_tail_chunk8 | 4 | e11:deadline#skybot-foundation-split-v1 | ok | 0.1533 | 150 | 1.102 | - | 363.4 |  |
| constant | 4 | e11:deadline#skybot-foundation-split-v1 | ok | 0.01333 | 150 | 1.119 | - | 363.4 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
