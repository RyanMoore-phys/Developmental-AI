# A/B report: E9c-false-structure-on-pure-noise

**Verdict: reject**

- CI upper bound 0 < delta 0.2: the hypothesised effect is ruled out
- note: criterion unmet: baseline 'refit_only' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: on pure noise with 12 distractors and a forced search (10x20 rows), the full language gets a STRUCTURAL mechanism promoted through the registry in >= 20% of runs more than REFIT-only (which cannot)
- primary metric: `false_struct_promotions` (higher is better)
- acceptance rule: accept iff full beats refit_only on false_struct_promotions (higher is better, basis=final) by mean >= 0.2 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.2 or candidate failures exceed 0; else inconclusive
- arms: baseline `refit_only`, candidate `full`, ablation `full_no_split`, alternative `None`
- seeds: [0, 1, 2]; splits: ['e9:noise#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071193515581000 ns, hash `57a65c18de3fd0cef4a688083f3e1865fee32d68ec6897813d0e15cb1545467c`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 3/3 | 0 | 0 | 0 | 0 | 1320 | 0 | 0.037 | - |
| full | 3/3 | 0 | 0 | 0 | 0 | 1320 | 0 | 0.370 | - |
| full_no_split | 3/3 | 0 | 0 | 0 | 0 | 1320 | 0 | 0.165 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 0 | 0 | 0 | 0 |
| equal_interactions | 3 | 0 | 0 | 0 | 0 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.014 | - | 66.53 |  |
| full | 0 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.134 | - | 66.53 |  |
| full_no_split | 0 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.055 | - | 66.53 |  |
| full | 1 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.120 | - | 66.53 |  |
| full_no_split | 1 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.056 | - | 66.53 |  |
| refit_only | 1 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.012 | - | 66.53 |  |
| full_no_split | 2 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.055 | - | 66.53 |  |
| refit_only | 2 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.011 | - | 66.53 |  |
| full | 2 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.116 | - | 66.53 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
