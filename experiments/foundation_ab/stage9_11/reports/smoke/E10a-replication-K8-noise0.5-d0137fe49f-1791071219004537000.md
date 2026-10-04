# A/B report: E10a-replication-K8-noise0.5

**Verdict: accept**

- mean improvement 16.17 >= delta 1, CI [0.4816, 39.47] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- WARNING: exploratory arms not in the prereg (reported, never decisive): ['lp']

## Preregistration

- hypothesis: through the library's own fixture path with NEW parameters (K=8, reward noise 0.5, budget 80), info-gain identifies the hidden shift in >= 1 fewer real interactions than random
- primary metric: `interactions_to_identify` (lower is better)
- acceptance rule: accept iff info_gain beats random on interactions_to_identify (lower is better, basis=final) by mean >= 1 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 1 or candidate failures exceed 0; else inconclusive
- arms: baseline `random`, candidate `info_gain`, ablation `entropy`, alternative `icm`
- seeds: [0, 1, 2]; splits: ['e10:lib-k8#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises', 'censored at budget+1=81, counted']
- frozen: 1791071214608765000 ns, hash `d0137fe49f43f49f86992f4c7861d98f2f6ad4e25165b5c4ca369a2758bb0c55`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| info_gain | 3/3 | 0 | 0 | 13.39 | 6.259 | 241 | 0 | 0.304 | - |
| random | 3/3 | 0 | 0 | 29.56 | 5.412 | 532 | 0 | 0.690 | - |
| icm | 3/3 | 0 | 0 | 30.72 | 2.761 | 553 | 0 | 0.735 | - |
| entropy | 3/3 | 0 | 0 | 81 | 0 | 1440 | 0 | 1.709 | - |
| lp | 3/3 | 0 | 0 | 38.67 | 16.5 | 691 | 0 | 0.956 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 16.17 | 0.4816 | 39.47 | 0 |
| equal_interactions | 0 | - | - | - | 3 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 67.61 | 50.28 | 83.89 | 0 |
| vs alternative_arm | 3 | 17.33 | 0.752 | 40.64 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| info_gain | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 7.333 | 44 | 0.069 | - | 317.5 |  |
| random | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 32.17 | 193 | 0.260 | - | 317.5 |  |
| icm | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 33.33 | 200 | 0.279 | - | 317.5 |  |
| entropy | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 480 | 0.613 | - | 317.5 |  |
| lp | 0 | e10:lib-k8#skybot-foundation-split-v1 | ok | 38.67 | 230 | 0.394 | - | 317.5 |  |
| random | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 33.17 | 199 | 0.268 | - | 317.5 |  |
| icm | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 31 | 186 | 0.257 | - | 317.5 |  |
| entropy | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 480 | 0.567 | - | 317.5 |  |
| lp | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 55.17 | 328 | 0.408 | - | 317.5 |  |
| info_gain | 1 | e10:lib-k8#skybot-foundation-split-v1 | ok | 19.83 | 119 | 0.141 | - | 317.5 |  |
| icm | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 27.83 | 167 | 0.199 | - | 317.5 |  |
| entropy | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 81 | 480 | 0.529 | - | 317.5 |  |
| lp | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 22.17 | 133 | 0.154 | - | 317.5 |  |
| info_gain | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 13 | 78 | 0.094 | - | 317.5 |  |
| random | 2 | e10:lib-k8#skybot-foundation-split-v1 | ok | 23.33 | 140 | 0.162 | - | 317.5 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
