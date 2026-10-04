# A/B report: E9f-search-wall-budget-overshoot

**Verdict: accept**

- mean improvement 17.63 >= delta 0.5, CI [16.07, 19.4] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: search() under wall_s = 0.25 s overshoots its budget by > 50% (elapsed/wall_s - small-table ratio >= 0.5) on a large table (150 variables x 60000 rows) — work outside the guarded fits is unbudgeted
- primary metric: `overshoot_ratio` (higher is better)
- acceptance rule: accept iff large_table beats small_table on overshoot_ratio (higher is better, basis=final) by mean >= 0.5 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.5 or candidate failures exceed 0; else inconclusive
- arms: baseline `small_table`, candidate `large_table`, ablation `hung_fitter`, alternative `None`
- seeds: [0, 1, 2]; splits: ['e9:budget#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071193521481000 ns, hash `80010a907a6f033dc07239e80ebb44dfa5c0a4f2b59cee7d9e1df1c1391c8398`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| small_table | 3/3 | 0 | 0 | 0.4578 | 0.007381 | 0 | 0 | 0.348 | - |
| large_table | 3/3 | 0 | 0 | 18.09 | 0.6157 | 0 | 0 | 14.300 | - |
| hung_fitter | 3/3 | 0 | 0 | 1.017 | 0.00493 | 0 | 0 | 0.768 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 17.63 | 16.07 | 19.4 | 0 |
| equal_interactions | 0 | - | - | - | 3 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 17.07 | 15.54 | 18.83 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| small_table | 0 | e9:budget#skybot-foundation-split-v1 | ok | 0.4662 | - | 0.118 | - | 67.2 |  |
| large_table | 0 | e9:budget#skybot-foundation-split-v1 | ok | 17.51 | - | 4.621 | - | 227.8 |  |
| hung_fitter | 0 | e9:budget#skybot-foundation-split-v1 | ok | 1.012 | - | 0.255 | - | 227.8 |  |
| large_table | 1 | e9:budget#skybot-foundation-split-v1 | ok | 18.74 | - | 4.934 | - | 308 |  |
| hung_fitter | 1 | e9:budget#skybot-foundation-split-v1 | ok | 1.016 | - | 0.256 | - | 308.1 |  |
| small_table | 1 | e9:budget#skybot-foundation-split-v1 | ok | 0.4523 | - | 0.115 | - | 308.2 |  |
| hung_fitter | 2 | e9:budget#skybot-foundation-split-v1 | ok | 1.022 | - | 0.257 | - | 308.3 |  |
| small_table | 2 | e9:budget#skybot-foundation-split-v1 | ok | 0.4549 | - | 0.116 | - | 308.7 |  |
| large_table | 2 | e9:budget#skybot-foundation-split-v1 | ok | 18 | - | 4.746 | - | 317.5 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
