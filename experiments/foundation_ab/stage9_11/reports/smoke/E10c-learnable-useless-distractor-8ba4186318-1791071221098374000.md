# A/B report: E10c-learnable-useless-distractor

**Verdict: accept**

- mean improvement 3.667 >= delta 1, CI [1.874, 4.563] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: with a deterministic 4-bit lamp the task does not need (joint shift x lamp hypotheses), info-gain on the JOINT set wastes interactions on the lamp: task-marginal info-gain identifies the shift in >= 1 fewer interactions
- primary metric: `interactions_to_identify` (lower is better)
- acceptance rule: accept iff info_marginal beats info_joint on interactions_to_identify (lower is better, basis=final) by mean >= 1 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 1 or candidate failures exceed 0; else inconclusive
- arms: baseline `info_joint`, candidate `info_marginal`, ablation `random`, alternative `icm`
- seeds: [0, 1, 2]; splits: ['e10:lamp#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071214609369000 ns, hash `8ba4186318bc26808af23bbe5a1e61d1ab5fbba6a7e43a96df920a383da8421e`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| info_joint | 3/3 | 0 | 0 | 9.667 | 1.764 | 87 | 0 | 0.172 | - |
| info_marginal | 3/3 | 0 | 0 | 6 | 2.333 | 54 | 0 | 0.156 | - |
| random | 3/3 | 0 | 0 | 22.44 | 5.059 | 202 | 0 | 0.097 | - |
| icm | 3/3 | 0 | 0 | 16.33 | 6.566 | 147 | 0 | 0.093 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 3.667 | 1.874 | 4.563 | 0 |
| equal_interactions | 0 | - | - | - | 3 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 16.44 | 4.494 | 23.32 | 0 |
| vs alternative_arm | 3 | 10.33 | -6.696 | 20.19 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| info_joint | 0 | e10:lamp#skybot-foundation-split-v1 | ok | 8.333 | 25 | 0.047 | - | 317.5 |  |
| info_marginal | 0 | e10:lamp#skybot-foundation-split-v1 | ok | 4.333 | 13 | 0.038 | - | 317.5 |  |
| random | 0 | e10:lamp#skybot-foundation-split-v1 | ok | 23.33 | 70 | 0.033 | - | 317.5 |  |
| icm | 0 | e10:lamp#skybot-foundation-split-v1 | ok | 18.33 | 55 | 0.031 | - | 317.5 |  |
| info_marginal | 1 | e10:lamp#skybot-foundation-split-v1 | ok | 8.667 | 26 | 0.067 | - | 317.5 |  |
| random | 1 | e10:lamp#skybot-foundation-split-v1 | ok | 27 | 81 | 0.037 | - | 317.5 |  |
| icm | 1 | e10:lamp#skybot-foundation-split-v1 | ok | 21.67 | 65 | 0.035 | - | 317.5 |  |
| info_joint | 1 | e10:lamp#skybot-foundation-split-v1 | ok | 11.67 | 35 | 0.062 | - | 317.5 |  |
| random | 2 | e10:lamp#skybot-foundation-split-v1 | ok | 17 | 51 | 0.027 | - | 317.5 |  |
| icm | 2 | e10:lamp#skybot-foundation-split-v1 | ok | 9 | 27 | 0.026 | - | 317.5 |  |
| info_joint | 2 | e10:lamp#skybot-foundation-split-v1 | ok | 9 | 27 | 0.063 | - | 317.5 |  |
| info_marginal | 2 | e10:lamp#skybot-foundation-split-v1 | ok | 5 | 15 | 0.052 | - | 317.5 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
