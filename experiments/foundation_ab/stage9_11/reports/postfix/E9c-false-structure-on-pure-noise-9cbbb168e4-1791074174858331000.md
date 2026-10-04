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
- seeds: [0, 1, 2, 3, 4]; splits: ['e9:noise#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791074172876894000 ns, hash `9cbbb168e480f5778aba7338f60b0352b5e6c3e30f20f38ef631532bfa0605b6`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 5/5 | 0 | 0 | 0 | 0 | 2200 | 0 | 0.018 | - |
| full | 5/5 | 0 | 0 | 0 | 0 | 2200 | 0 | 0.195 | - |
| full_no_split | 5/5 | 0 | 0 | 0 | 0 | 2200 | 0 | 0.088 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0 | 0 | 0 | 0 |
| equal_interactions | 5 | 0 | 0 | 0 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.004 | - | 86.38 |  |
| full | 0 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.039 | - | 86.38 |  |
| full_no_split | 0 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.018 | - | 86.38 |  |
| full | 1 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.039 | - | 86.38 |  |
| full_no_split | 1 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.018 | - | 86.38 |  |
| refit_only | 1 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.003 | - | 86.38 |  |
| full_no_split | 2 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.017 | - | 86.38 |  |
| refit_only | 2 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.004 | - | 86.38 |  |
| full | 2 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.038 | - | 86.38 |  |
| refit_only | 3 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.004 | - | 86.38 |  |
| full | 3 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.040 | - | 86.38 |  |
| full_no_split | 3 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.018 | - | 86.38 |  |
| full | 4 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.039 | - | 86.38 |  |
| full_no_split | 4 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.017 | - | 86.38 |  |
| refit_only | 4 | e9:noise#skybot-foundation-split-v1 | ok | 0 | 440 | 0.003 | - | 86.38 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
