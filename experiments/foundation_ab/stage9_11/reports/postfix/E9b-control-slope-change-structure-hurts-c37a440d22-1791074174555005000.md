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
- seeds: [0, 1, 2, 3, 4]; splits: ['e9:nparam#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 120.0}
- failure conditions: ['any arm raises']
- frozen: 1791074172876748000 ns, hash `c37a440d22acf9429074a7f3749e898f57608bf93764cd5c65913881cca6b717`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| full | 5/5 | 0 | 0 | 0.8001 | 0.05931 | 1600 | 0 | 0.249 | - |
| refit_only | 5/5 | 0 | 0 | 0.8001 | 0.05931 | 1600 | 0 | 0.093 | - |
| no_update | 5/5 | 0 | 0 | 3.013 | 0.2667 | 0 | 0 | 0.013 | - |
| full_slope3 | 5/5 | 0 | 0 | 1.259 | 0.3047 | 1600 | 0 | 0.248 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0 | 0 | 0 | 0 |
| equal_interactions | 5 | 0 | 0 | 0 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 2.213 | 1.927 | 2.58 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| full | 0 | e9:nparam#skybot-foundation-split-v1 | ok | 0.8321 | 320 | 0.050 | - | 86.38 |  |
| refit_only | 0 | e9:nparam#skybot-foundation-split-v1 | ok | 0.8321 | 320 | 0.019 | - | 86.38 |  |
| no_update | 0 | e9:nparam#skybot-foundation-split-v1 | ok | 2.973 | - | 0.003 | - | 86.38 |  |
| full_slope3 | 0 | e9:nparam#skybot-foundation-split-v1 | ok | 1.41 | 320 | 0.050 | - | 86.38 |  |
| refit_only | 1 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7485 | 320 | 0.019 | - | 86.38 |  |
| no_update | 1 | e9:nparam#skybot-foundation-split-v1 | ok | 3.385 | - | 0.003 | - | 86.38 |  |
| full_slope3 | 1 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7474 | 320 | 0.050 | - | 86.38 |  |
| full | 1 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7485 | 320 | 0.049 | - | 86.38 |  |
| no_update | 2 | e9:nparam#skybot-foundation-split-v1 | ok | 3.163 | - | 0.003 | - | 86.38 |  |
| full_slope3 | 2 | e9:nparam#skybot-foundation-split-v1 | ok | 1.541 | 320 | 0.050 | - | 86.38 |  |
| full | 2 | e9:nparam#skybot-foundation-split-v1 | ok | 0.8858 | 320 | 0.049 | - | 86.38 |  |
| refit_only | 2 | e9:nparam#skybot-foundation-split-v1 | ok | 0.8858 | 320 | 0.019 | - | 86.38 |  |
| full_slope3 | 3 | e9:nparam#skybot-foundation-split-v1 | ok | 1.254 | 320 | 0.049 | - | 86.38 |  |
| full | 3 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7464 | 320 | 0.050 | - | 86.38 |  |
| refit_only | 3 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7464 | 320 | 0.018 | - | 86.38 |  |
| no_update | 3 | e9:nparam#skybot-foundation-split-v1 | ok | 2.717 | - | 0.003 | - | 86.38 |  |
| full | 4 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7876 | 320 | 0.050 | - | 86.38 |  |
| refit_only | 4 | e9:nparam#skybot-foundation-split-v1 | ok | 0.7876 | 320 | 0.019 | - | 86.38 |  |
| no_update | 4 | e9:nparam#skybot-foundation-split-v1 | ok | 2.826 | - | 0.003 | - | 86.38 |  |
| full_slope3 | 4 | e9:nparam#skybot-foundation-split-v1 | ok | 1.344 | 320 | 0.049 | - | 86.38 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
