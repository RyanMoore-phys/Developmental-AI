# A/B report: E10c-learnable-useless-distractor

**Verdict: accept**

- mean improvement 5.475 >= delta 1, CI [4.128, 7.177] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: with a deterministic 4-bit lamp the task does not need (joint shift x lamp hypotheses), info-gain on the JOINT set wastes interactions on the lamp: task-marginal info-gain identifies the shift in >= 1 fewer interactions
- primary metric: `interactions_to_identify` (lower is better)
- acceptance rule: accept iff info_marginal beats info_joint on interactions_to_identify (lower is better, basis=final) by mean >= 1 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 1 or candidate failures exceed 0; else inconclusive
- arms: baseline `info_joint`, candidate `info_marginal`, ablation `random`, alternative `icm`
- seeds: [0, 1, 2, 3, 4]; splits: ['e10:lamp#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071277578320000 ns, hash `9d74dc992b8552d50c6d864778f9a492e46f540f9ba9064de2b55889f08f4e11`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| info_joint | 5/5 | 0 | 0 | 10.6 | 1.251 | 424 | 0 | 0.364 | - |
| info_marginal | 5/5 | 0 | 0 | 5.125 | 1.296 | 205 | 0 | 0.452 | - |
| random | 5/5 | 0 | 0 | 19.7 | 3.328 | 788 | 0 | 0.200 | - |
| icm | 5/5 | 0 | 0 | 17.12 | 3.292 | 685 | 0 | 0.203 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 5.475 | 4.128 | 7.177 | 0 |
| equal_interactions | 0 | - | - | - | 5 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 14.57 | 8.637 | 19.56 | 0 |
| vs alternative_arm | 5 | 12 | 8.234 | 15.64 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| info_joint | 0 | e10:lamp#skybot-foundation-split-v1 | ok | 11.12 | 89 | 0.080 | - | 86 |  |
| info_marginal | 0 | e10:lamp#skybot-foundation-split-v1 | ok | 6.75 | 54 | 0.116 | - | 86 |  |
| random | 0 | e10:lamp#skybot-foundation-split-v1 | ok | 15.5 | 124 | 0.035 | - | 86 |  |
| icm | 0 | e10:lamp#skybot-foundation-split-v1 | ok | 21.88 | 175 | 0.048 | - | 86 |  |
| info_marginal | 1 | e10:lamp#skybot-foundation-split-v1 | ok | 5.375 | 43 | 0.095 | - | 86 |  |
| random | 1 | e10:lamp#skybot-foundation-split-v1 | ok | 23 | 184 | 0.045 | - | 86 |  |
| icm | 1 | e10:lamp#skybot-foundation-split-v1 | ok | 15.88 | 127 | 0.039 | - | 86 |  |
| info_joint | 1 | e10:lamp#skybot-foundation-split-v1 | ok | 10.88 | 87 | 0.074 | - | 86 |  |
| random | 2 | e10:lamp#skybot-foundation-split-v1 | ok | 23 | 184 | 0.044 | - | 86 |  |
| icm | 2 | e10:lamp#skybot-foundation-split-v1 | ok | 18.75 | 150 | 0.043 | - | 86 |  |
| info_joint | 2 | e10:lamp#skybot-foundation-split-v1 | ok | 12 | 96 | 0.079 | - | 86 |  |
| info_marginal | 2 | e10:lamp#skybot-foundation-split-v1 | ok | 4.375 | 35 | 0.079 | - | 86 |  |
| icm | 3 | e10:lamp#skybot-foundation-split-v1 | ok | 13.25 | 106 | 0.034 | - | 86 |  |
| info_joint | 3 | e10:lamp#skybot-foundation-split-v1 | ok | 10.38 | 83 | 0.070 | - | 86 |  |
| info_marginal | 3 | e10:lamp#skybot-foundation-split-v1 | ok | 5.75 | 46 | 0.099 | - | 86 |  |
| random | 3 | e10:lamp#skybot-foundation-split-v1 | ok | 17.5 | 140 | 0.037 | - | 86 |  |
| info_joint | 4 | e10:lamp#skybot-foundation-split-v1 | ok | 8.625 | 69 | 0.061 | - | 86 |  |
| info_marginal | 4 | e10:lamp#skybot-foundation-split-v1 | ok | 3.375 | 27 | 0.064 | - | 86 |  |
| random | 4 | e10:lamp#skybot-foundation-split-v1 | ok | 19.5 | 156 | 0.040 | - | 86 |  |
| icm | 4 | e10:lamp#skybot-foundation-split-v1 | ok | 15.88 | 127 | 0.039 | - | 86 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
