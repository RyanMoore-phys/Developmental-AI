# A/B report: E9b-control-slope-change-structure-hurts

**Verdict: reject**

- CI upper bound 0 < delta 0.02: the hypothesised effect is ruled out
- note: criterion unmet: baseline 'full' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- WARNING: exploratory arms not in the prereg (reported, never decisive): ['full_slope3']

## Preregistration

- hypothesis: CONTROL (falsification direction): after a slope-only change (2 -> 4), with noise and 7 distractors, REFIT-only beats the full language on held-out NLL by >= 0.02 nats/row, i.e. structure search invents structure that costs prediction
- primary metric: `nll_robust` (lower is better)
- acceptance rule: accept iff refit_only beats full on nll_robust (lower is better, basis=final) by mean >= 0.02 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.02 or candidate failures exceed 0; else inconclusive
- arms: baseline `full`, candidate `refit_only`, ablation `no_update`, alternative `None`
- seeds: [0, 1, 2]; splits: ['e9:nparam#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 120.0}
- failure conditions: ['any arm raises']
- frozen: 1791071193515201000 ns, hash `4beccc7d096429610788cea3c75a73f72a0197cbe55d6e38c57c36c65969b0f7`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| full | 3/3 | 0 | 0 | 0.8349 | 0.09092 | 960 | 0 | 0.402 | - |
| refit_only | 3/3 | 0 | 0 | 0.8349 | 0.09092 | 960 | 0 | 0.081 | - |
| no_update | 3/3 | 0 | 0 | 3.147 | 0.1688 | 0 | 0 | 0.021 | - |
| full_slope3 | 3/3 | 0 | 0 | 1.24 | 0.448 | 960 | 0 | 0.264 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 0 | 0 | 0 | 0 |
| equal_interactions | 3 | 0 | 0 | 0 | 0 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 2.312 | 1.743 | 3.002 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| full | 0 | e9:nparam#skybot-foundation-split-v1 | ok | 0.8661 | 320 | 0.128 | - | 66.53 |  |
| refit_only | 0 | e9:nparam#skybot-foundation-split-v1 | ok | 0.8661 | 320 | 0.026 | - | 66.53 |  |
| no_update | 0 | e9:nparam#skybot-foundation-split-v1 | ok | 2.967 | - | 0.007 | - | 66.53 |  |
| full_slope3 | 0 | e9:nparam#skybot-foundation-split-v1 | ok | 1.412 | 320 | 0.021 | - | 66.53 |  |
| refit_only | 1 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7325 | 320 | 0.025 | - | 66.53 |  |
| no_update | 1 | e9:nparam#skybot-foundation-split-v1 | ok | 3.301 | - | 0.007 | - | 66.53 |  |
| full_slope3 | 1 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7314 | 320 | 0.116 | - | 66.53 |  |
| full | 1 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7325 | 320 | 0.139 | - | 66.53 |  |
| no_update | 2 | e9:nparam#skybot-foundation-split-v1 | ok | 3.173 | - | 0.007 | - | 66.53 |  |
| full_slope3 | 2 | e9:nparam#skybot-foundation-split-v1 | ok | 1.576 | 320 | 0.127 | - | 66.53 |  |
| full | 2 | e9:nparam#skybot-foundation-split-v1 | ok | 0.9061 | 320 | 0.134 | - | 66.53 |  |
| refit_only | 2 | e9:nparam#skybot-foundation-split-v1 | ok | 0.9061 | 320 | 0.029 | - | 66.53 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
