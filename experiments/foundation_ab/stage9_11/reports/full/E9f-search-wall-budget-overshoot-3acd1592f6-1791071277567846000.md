# A/B report: E9f-search-wall-budget-overshoot

**Verdict: accept**

- mean improvement 17.33 >= delta 0.5, CI [16.15, 18.43] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: search() under wall_s = 0.25 s overshoots its budget by > 50% (elapsed/wall_s - small-table ratio >= 0.5) on a large table (150 variables x 60000 rows) — work outside the guarded fits is unbudgeted
- primary metric: `overshoot_ratio` (higher is better)
- acceptance rule: accept iff large_table beats small_table on overshoot_ratio (higher is better, basis=final) by mean >= 0.5 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.5 or candidate failures exceed 0; else inconclusive
- arms: baseline `small_table`, candidate `large_table`, ablation `hung_fitter`, alternative `None`
- seeds: [0, 1, 2, 3, 4]; splits: ['e9:budget#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071242885185000 ns, hash `3acd1592f6c9c76f03c6e37c9268c30c00d31ff06f1c49cb1773896f93a5a5a1`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| small_table | 5/5 | 0 | 0 | 0.4589 | 0.1243 | 0 | 0 | 0.581 | - |
| large_table | 5/5 | 0 | 0 | 17.79 | 0.9535 | 0 | 0 | 23.491 | - |
| hung_fitter | 5/5 | 0 | 0 | 1.041 | 0.001297 | 0 | 0 | 1.310 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 17.33 | 16.15 | 18.43 | 0 |
| equal_interactions | 0 | - | - | - | 5 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 16.75 | 15.44 | 17.85 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| small_table | 0 | e9:budget#skybot-foundation-split-v1 | ok | 0.4653 | - | 0.118 | - | 67.66 |  |
| large_table | 0 | e9:budget#skybot-foundation-split-v1 | ok | 17.24 | - | 4.555 | - | 229.6 |  |
| hung_fitter | 0 | e9:budget#skybot-foundation-split-v1 | ok | 1.041 | - | 0.262 | - | 229.6 |  |
| large_table | 1 | e9:budget#skybot-foundation-split-v1 | ok | 18.39 | - | 4.830 | - | 271.9 |  |
| hung_fitter | 1 | e9:budget#skybot-foundation-split-v1 | ok | 1.041 | - | 0.262 | - | 271.9 |  |
| small_table | 1 | e9:budget#skybot-foundation-split-v1 | ok | 0.3665 | - | 0.093 | - | 271.9 |  |
| hung_fitter | 2 | e9:budget#skybot-foundation-split-v1 | ok | 1.044 | - | 0.262 | - | 271.9 |  |
| small_table | 2 | e9:budget#skybot-foundation-split-v1 | ok | 0.4429 | - | 0.112 | - | 271.9 |  |
| large_table | 2 | e9:budget#skybot-foundation-split-v1 | ok | 18.72 | - | 4.921 | - | 271.9 |  |
| small_table | 3 | e9:budget#skybot-foundation-split-v1 | ok | 0.6645 | - | 0.168 | - | 271.9 |  |
| large_table | 3 | e9:budget#skybot-foundation-split-v1 | ok | 18.2 | - | 4.862 | - | 310.4 |  |
| hung_fitter | 3 | e9:budget#skybot-foundation-split-v1 | ok | 1.041 | - | 0.262 | - | 310.5 |  |
| large_table | 4 | e9:budget#skybot-foundation-split-v1 | ok | 16.4 | - | 4.323 | - | 362.9 |  |
| hung_fitter | 4 | e9:budget#skybot-foundation-split-v1 | ok | 1.041 | - | 0.262 | - | 362.9 |  |
| small_table | 4 | e9:budget#skybot-foundation-split-v1 | ok | 0.3554 | - | 0.090 | - | 363.2 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
