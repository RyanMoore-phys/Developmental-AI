# A/B report: E9f-search-wall-budget-overshoot

**Verdict: accept**

- mean improvement 0.8218 >= delta 0.5, CI [0.6606, 0.9298] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: search() under wall_s = 0.25 s overshoots its budget by > 50% (elapsed/wall_s - small-table ratio >= 0.5) on a large table (150 variables x 60000 rows) — work outside the guarded fits is unbudgeted
- primary metric: `overshoot_ratio` (higher is better)
- acceptance rule: accept iff large_table beats small_table on overshoot_ratio (higher is better, basis=final) by mean >= 0.5 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.5 or candidate failures exceed 0; else inconclusive
- arms: baseline `small_table`, candidate `large_table`, ablation `hung_fitter`, alternative `None`
- seeds: [0, 1, 2, 3, 4]; splits: ['e9:budget#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791074172879183000 ns, hash `5dfe90cce71e5e1a80067a961cdb3a65ebd55e03949b85d7309e7da51d87c867`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| small_table | 5/5 | 0 | 0 | 0.2287 | 0.09172 | 0 | 0 | 0.290 | - |
| large_table | 5/5 | 0 | 0 | 1.051 | 0.02784 | 0 | 0 | 2.084 | - |
| hung_fitter | 5/5 | 0 | 0 | 1.024 | 0.01526 | 0 | 0 | 1.285 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.8218 | 0.6606 | 0.9298 | 0 |
| equal_interactions | 0 | - | - | - | 5 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0.0266 | -0.01287 | 0.05779 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| small_table | 0 | e9:budget#skybot-foundation-split-v1 | ok | 0.1545 | - | 0.039 | - | 87.23 |  |
| large_table | 0 | e9:budget#skybot-foundation-split-v1 | ok | 1.068 | - | 0.390 | - | 244.8 |  |
| hung_fitter | 0 | e9:budget#skybot-foundation-split-v1 | ok | 1.019 | - | 0.256 | - | 244.8 |  |
| large_table | 1 | e9:budget#skybot-foundation-split-v1 | ok | 1.047 | - | 0.393 | - | 258.9 |  |
| hung_fitter | 1 | e9:budget#skybot-foundation-split-v1 | ok | 1.002 | - | 0.251 | - | 258.9 |  |
| small_table | 1 | e9:budget#skybot-foundation-split-v1 | ok | 0.2139 | - | 0.054 | - | 259.2 |  |
| hung_fitter | 2 | e9:budget#skybot-foundation-split-v1 | ok | 1.04 | - | 0.261 | - | 259.2 |  |
| small_table | 2 | e9:budget#skybot-foundation-split-v1 | ok | 0.1837 | - | 0.047 | - | 259.4 |  |
| large_table | 2 | e9:budget#skybot-foundation-split-v1 | ok | 1.052 | - | 0.379 | - | 292.7 |  |
| small_table | 3 | e9:budget#skybot-foundation-split-v1 | ok | 0.3877 | - | 0.098 | - | 292.8 |  |
| large_table | 3 | e9:budget#skybot-foundation-split-v1 | ok | 1.007 | - | 0.502 | - | 377.7 |  |
| hung_fitter | 3 | e9:budget#skybot-foundation-split-v1 | ok | 1.023 | - | 0.257 | - | 377.7 |  |
| large_table | 4 | e9:budget#skybot-foundation-split-v1 | ok | 1.079 | - | 0.420 | - | 377.8 |  |
| hung_fitter | 4 | e9:budget#skybot-foundation-split-v1 | ok | 1.036 | - | 0.260 | - | 378 |  |
| small_table | 4 | e9:budget#skybot-foundation-split-v1 | ok | 0.2039 | - | 0.052 | - | 378.2 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
