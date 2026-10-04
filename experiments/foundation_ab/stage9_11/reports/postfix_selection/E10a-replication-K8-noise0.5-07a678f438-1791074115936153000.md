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
| info_gain | 5/5 | 0 | 0 | 11.26 | 1.032 | 1013 | 0 | 2.082 | - |
| random | 5/5 | 0 | 0 | 28.91 | 4.621 | 2600 | 0 | 2.104 | - |
| icm | 5/5 | 0 | 0 | 28.86 | 6.388 | 2596 | 0 | 2.834 | - |
| entropy | 5/5 | 0 | 0 | 81 | 0 | 7200 | 0 | 10.583 | - |
| lp | 5/5 | 0 | 0 | 37.87 | 12.85 | 3392 | 0 | 3.099 | - |

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
| info_gain | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 11.22 | 202 | 0.391 | - | 84.27 |  |
| random | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 29 | 522 | 0.462 | - | 84.27 |  |
| icm | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 31.61 | 568 | 0.526 | - | 84.27 |  |
| entropy | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 2.222 | - | 84.27 |  |
| lp | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 40.67 | 727 | 0.623 | - | 84.27 |  |
| random | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 36.28 | 652 | 0.694 | - | 84.27 |  |
| icm | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 33.78 | 608 | 1.118 | - | 84.27 |  |
| entropy | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 3.522 | - | 84.27 |  |
| lp | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 49 | 876 | 1.306 | - | 84.3 |  |
| info_gain | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 12.39 | 223 | 0.770 | - | 84.3 |  |
| icm | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 29.72 | 535 | 0.629 | - | 84.31 |  |
| entropy | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 1.863 | - | 84.31 |  |
| lp | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 30.83 | 554 | 0.364 | - | 84.33 |  |
| info_gain | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 12.17 | 219 | 0.442 | - | 84.33 |  |
| random | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 25.44 | 458 | 0.380 | - | 84.33 |  |
| entropy | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 1.533 | - | 84.33 |  |
| lp | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 49.5 | 887 | 0.591 | - | 84.34 |  |
| info_gain | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 10 | 180 | 0.234 | - | 84.34 |  |
| random | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 29.28 | 526 | 0.304 | - | 84.34 |  |
| icm | 3 | e10:lib-k8#skybot-foundation-split-v1 | ok | 31.44 | 566 | 0.348 | - | 84.34 |  |
| lp | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 19.33 | 348 | 0.215 | - | 84.34 |  |
| info_gain | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 10.5 | 189 | 0.245 | - | 84.34 |  |
| random | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 24.56 | 442 | 0.264 | - | 84.34 |  |
| icm | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 17.72 | 319 | 0.214 | - | 84.34 |  |
| entropy | 4 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 1440 | 1.444 | - | 84.39 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
