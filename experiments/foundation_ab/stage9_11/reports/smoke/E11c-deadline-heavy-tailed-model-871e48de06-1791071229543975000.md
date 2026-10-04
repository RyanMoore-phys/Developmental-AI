# A/B report: E11c-deadline-heavy-tailed-model

**Verdict: inconclusive**

- CI [-0.02037, 0.1813] with mean 0.09167 neither meets nor rules out delta 0.01
- note: criterion unmet: baseline 'constant' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: with a model whose call latency is heavy-tailed (2% of calls take 30 ms) under a 10 ms deadline, >= 1% more decisions take over TWICE the deadline than with a constant-latency model of equal mean: the 'hard' deadline is bounded only by one chunk's worst case
- primary metric: `overrun_fraction` (higher is better)
- acceptance rule: accept iff heavy_tail beats constant on overrun_fraction (higher is better, basis=final) by mean >= 0.01 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.01 or candidate failures exceed 0; else inconclusive
- arms: baseline `constant`, candidate `heavy_tail`, ablation `heavy_tail_chunk8`, alternative `None`
- seeds: [0, 1, 2]; splits: ['e11:deadline#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071221513998000 ns, hash `871e48de0634b5f1e293209f87a0c5ec1394828245365571cd8a7c151f6de155`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| constant | 3/3 | 0 | 0 | 0 | 0 | 120 | 0 | 0.863 | - |
| heavy_tail | 3/3 | 0 | 0 | 0.09167 | 0.03819 | 120 | 0 | 0.603 | - |
| heavy_tail_chunk8 | 3/3 | 0 | 0 | 0.15 | 0 | 120 | 0 | 1.030 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 0.09167 | -0.02037 | 0.1813 | 0 |
| equal_interactions | 3 | 0.09167 | -0.02037 | 0.1813 | 0 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | -0.05833 | -0.1704 | 0.0313 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| constant | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 40 | 0.230 | - | 317.5 |  |
| heavy_tail | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0.125 | 40 | 0.242 | - | 317.5 |  |
| heavy_tail_chunk8 | 0 | e11:deadline#skybot-foundation-split-v1 | ok | 0.15 | 40 | 0.329 | - | 317.5 |  |
| heavy_tail | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0.05 | 40 | 0.146 | - | 317.5 |  |
| heavy_tail_chunk8 | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0.15 | 40 | 0.344 | - | 317.5 |  |
| constant | 1 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 40 | 0.387 | - | 317.5 |  |
| heavy_tail_chunk8 | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0.15 | 40 | 0.356 | - | 317.5 |  |
| constant | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0 | 40 | 0.247 | - | 317.5 |  |
| heavy_tail | 2 | e11:deadline#skybot-foundation-split-v1 | ok | 0.1 | 40 | 0.215 | - | 317.5 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
