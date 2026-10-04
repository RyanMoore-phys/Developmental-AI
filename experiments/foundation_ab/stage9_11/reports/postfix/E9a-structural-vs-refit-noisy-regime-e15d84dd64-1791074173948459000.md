# A/B report: E9a-structural-vs-refit-noisy-regime

**Verdict: accept**

- mean improvement 0.7923 >= delta 0.1, CI [0.7055, 0.879] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: after a structural rule change (threshold 0.3 on temp), on NEW noisier (sd 0.5), smaller (16x20 post rows) data with 7 distractors, the full proposal language yields lower held-out NLL than REFIT-only under the same SearchBudget (implementers report 1.82 nats/row on their fixture)
- primary metric: `nll_robust` (lower is better)
- acceptance rule: accept iff full beats refit_only on nll_robust (lower is better, basis=final) by mean >= 0.1 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.1 or candidate failures exceed 0; else inconclusive
- arms: baseline `refit_only`, candidate `full`, ablation `full_no_split`, alternative `no_update`
- seeds: [0, 1, 2, 3, 4]; splits: ['e9:nregime#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 120.0}
- failure conditions: ['any arm raises or exceeds 120 s', 'held-out NLL non-finite', 'delta 0.1 nats/row is the minimum effect worth a structure']
- frozen: 1791074172876537000 ns, hash `e15d84dd64c6683d5fd39dbc2032994192d549ae21fbfbeeed414731a07793ab`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 5/5 | 0 | 0 | 1.652 | 0.03075 | 1600 | 0 | 0.095 | - |
| full | 5/5 | 0 | 0 | 0.8596 | 0.08581 | 1600 | 0 | 0.735 | - |
| full_no_split | 5/5 | 0 | 0 | 0.8597 | 0.08588 | 1600 | 0 | 0.207 | - |
| no_update | 5/5 | 0 | 0 | 4.397 | 0.3078 | 0 | 0 | 0.014 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.7923 | 0.7055 | 0.879 | 0 |
| equal_interactions | 5 | 0.7923 | 0.7055 | 0.879 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 5.924e-05 | -3.457e-05 | 0.0002468 | 0 |
| vs alternative_arm | 5 | 3.537 | 3.099 | 3.862 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | e9:nregime#skybot-foundation-split-v1 | ok | 1.626 | 320 | 0.020 | - | 83.77 |  |
| full | 0 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8448 | 320 | 0.165 | - | 84.14 |  |
| full_no_split | 0 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8448 | 320 | 0.049 | - | 84.16 |  |
| no_update | 0 | e9:nregime#skybot-foundation-split-v1 | ok | 3.875 | - | 0.003 | - | 84.16 |  |
| full | 1 | e9:nregime#skybot-foundation-split-v1 | ok | 0.9398 | 320 | 0.087 | - | 84.23 |  |
| full_no_split | 1 | e9:nregime#skybot-foundation-split-v1 | ok | 0.9401 | 320 | 0.039 | - | 84.38 |  |
| no_update | 1 | e9:nregime#skybot-foundation-split-v1 | ok | 4.647 | - | 0.003 | - | 84.38 |  |
| refit_only | 1 | e9:nregime#skybot-foundation-split-v1 | ok | 1.664 | 320 | 0.019 | - | 84.42 |  |
| full_no_split | 2 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8437 | 320 | 0.040 | - | 84.42 |  |
| no_update | 2 | e9:nregime#skybot-foundation-split-v1 | ok | 4.58 | - | 0.003 | - | 84.44 |  |
| refit_only | 2 | e9:nregime#skybot-foundation-split-v1 | ok | 1.688 | 320 | 0.019 | - | 84.53 |  |
| full | 2 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8437 | 320 | 0.161 | - | 84.66 |  |
| no_update | 3 | e9:nregime#skybot-foundation-split-v1 | ok | 4.497 | - | 0.003 | - | 84.66 |  |
| refit_only | 3 | e9:nregime#skybot-foundation-split-v1 | ok | 1.614 | 320 | 0.019 | - | 84.66 |  |
| full | 3 | e9:nregime#skybot-foundation-split-v1 | ok | 0.7317 | 320 | 0.161 | - | 84.69 |  |
| full_no_split | 3 | e9:nregime#skybot-foundation-split-v1 | ok | 0.7317 | 320 | 0.040 | - | 84.69 |  |
| refit_only | 4 | e9:nregime#skybot-foundation-split-v1 | ok | 1.667 | 320 | 0.019 | - | 84.69 |  |
| full | 4 | e9:nregime#skybot-foundation-split-v1 | ok | 0.9383 | 320 | 0.161 | - | 84.7 |  |
| full_no_split | 4 | e9:nregime#skybot-foundation-split-v1 | ok | 0.9383 | 320 | 0.040 | - | 84.72 |  |
| no_update | 4 | e9:nregime#skybot-foundation-split-v1 | ok | 4.384 | - | 0.003 | - | 84.72 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
