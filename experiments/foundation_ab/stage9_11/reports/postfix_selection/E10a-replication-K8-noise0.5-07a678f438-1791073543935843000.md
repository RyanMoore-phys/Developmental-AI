# A/B report: E10a-replication-K8-noise0.5

**Verdict: accept**

- mean improvement 17.66 >= delta 1, CI [12.64, 22.95] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- WARNING: exploratory arms not in the prereg (reported, never decisive): ['lp']

## Preregistration

- hypothesis: through the library's own fixture path with NEW parameters (K=8, reward noise 0.5, budget 80), info-gain identifies the hidden shift in >= 1 fewer real interactions than random
- primary metric: `interactions_to_identify` (lower is better)
- acceptance rule: accept iff info_gain beats random on interactions_to_identify (lower is better, basis=final) by mean >= 1 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 1 or candidate failures exceed 0; else inconclusive
- arms: baseline `random`, candidate `info_gain`, ablation `entropy`, alternative `icm`
- seeds: [0, 1, 2, 3, 4]; splits: ['e10:lib-k8#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises', 'censored at budget+1=81, counted']
- frozen: 1791071277577572000 ns, hash `07a678f4385cdb705626922e1c458fec7af7530ea5288435d063ac307f52e016`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| info_gain | 5/5 | 0 | 0 | 11.26 | 1.032 | 1013 | 0 | 9.291 | - |
| random | 5/5 | 0 | 0 | 28.91 | 4.621 | 2600 | 0 | 8.060 | - |
| icm | 5/5 | 0 | 0 | 28.86 | 6.388 | 2596 | 0 | 8.261 | - |
| entropy | 5/5 | 0 | 0 | 81 | 0 | 7200 | 0 | 34.949 | - |
| lp | 5/5 | 0 | 0 | 37.87 | 12.85 | 3392 | 0 | 8.522 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 17.66 | 12.64 | 22.95 | 0 |
| equal_interactions | 0 | - | - | - | 5 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 69.74 | 68.46 | 71.03 | 0 |
| vs alternative_arm | 5 | 17.6 | 8.943 | 23.32 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| info_gain | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 11.22 | 202 | 2.246 | - | 82.86 |  |
| random | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 29 | 522 | 1.974 | - | 82.86 |  |
| icm | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 31.61 | 568 | 2.331 | - | 82.86 |  |
| entropy | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 7.947 | - | 82.86 |  |
| lp | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 40.67 | 727 | 1.154 | - | 82.86 |  |
| random | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 36.28 | 652 | 1.619 | - | 82.86 |  |
| icm | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 33.78 | 608 | 1.667 | - | 82.86 |  |
| entropy | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 6.732 | - | 82.86 |  |
| lp | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 49 | 876 | 2.484 | - | 82.86 |  |
| info_gain | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 12.39 | 223 | 1.865 | - | 82.86 |  |
| icm | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 29.72 | 535 | 1.515 | - | 82.86 |  |
| entropy | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 7.059 | - | 82.86 |  |
| lp | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 30.83 | 554 | 1.173 | - | 82.86 |  |
| info_gain | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 12.17 | 219 | 1.854 | - | 82.86 |  |
| random | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 25.44 | 458 | 1.408 | - | 82.86 |  |
| entropy | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 6.712 | - | 82.86 |  |
| lp | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 49.5 | 887 | 2.416 | - | 82.86 |  |
| info_gain | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 10 | 180 | 1.725 | - | 82.86 |  |
| random | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 29.28 | 526 | 1.682 | - | 82.86 |  |
| icm | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 31.44 | 566 | 1.758 | - | 82.86 |  |
| lp | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 19.33 | 348 | 1.296 | - | 82.86 |  |
| info_gain | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 10.5 | 189 | 1.600 | - | 82.86 |  |
| random | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 24.56 | 442 | 1.377 | - | 82.86 |  |
| icm | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 17.72 | 319 | 0.990 | - | 82.86 |  |
| entropy | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 6.498 | - | 82.86 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
