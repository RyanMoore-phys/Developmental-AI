# A/B report: E9a-structural-vs-refit-noisy-regime

**Verdict: accept**

- mean improvement 0.8195 >= delta 0.1, CI [0.72, 0.8889] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: after a structural rule change (threshold 0.3 on temp), on NEW noisier (sd 0.5), smaller (16x20 post rows) data with 7 distractors, the full proposal language yields lower held-out NLL than REFIT-only under the same SearchBudget (implementers report 1.82 nats/row on their fixture)
- primary metric: `nll_robust` (lower is better)
- acceptance rule: accept iff full beats refit_only on nll_robust (lower is better, basis=final) by mean >= 0.1 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.1 or candidate failures exceed 0; else inconclusive
- arms: baseline `refit_only`, candidate `full`, ablation `full_no_split`, alternative `no_update`
- seeds: [0, 1, 2]; splits: ['e9:nregime#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 120.0}
- failure conditions: ['any arm raises or exceeds 120 s', 'held-out NLL non-finite', 'delta 0.1 nats/row is the minimum effect worth a structure']
- frozen: 1791071193514707000 ns, hash `f2934b473b1c8d568d4b0d5edbb091fd51b04a62c5e3d848457956e6e92879d3`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 3/3 | 0 | 0 | 1.656 | 0.05455 | 960 | 0 | 0.079 | - |
| full | 3/3 | 0 | 0 | 0.837 | 0.03333 | 960 | 0 | 1.087 | - |
| full_no_split | 3/3 | 0 | 0 | 0.8325 | 0.03738 | 960 | 0 | 0.282 | - |
| no_update | 3/3 | 0 | 0 | 4.285 | 0.2985 | 0 | 0 | 0.021 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 0.8195 | 0.72 | 0.8889 | 0 |
| equal_interactions | 3 | 0.8195 | 0.72 | 0.8889 | 0 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | -0.004455 | -0.02842 | 0.007525 | 0 |
| vs alternative_arm | 3 | 3.448 | 2.589 | 3.922 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | e9:nregime#skybot-foundation-split-v1 | ok | 1.598 | 320 | 0.028 | - | 64.52 |  |
| full | 0 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8155 | 320 | 0.430 | - | 64.91 |  |
| full_no_split | 0 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8155 | 320 | 0.115 | - | 64.97 |  |
| no_update | 0 | e9:nregime#skybot-foundation-split-v1 | ok | 3.944 | - | 0.007 | - | 64.97 |  |
| full | 1 | e9:nregime#skybot-foundation-split-v1 | ok | 0.82 | 320 | 0.229 | - | 65.06 |  |
| full_no_split | 1 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8067 | 320 | 0.084 | - | 65.23 |  |
| no_update | 1 | e9:nregime#skybot-foundation-split-v1 | ok | 4.411 | - | 0.007 | - | 65.23 |  |
| refit_only | 1 | e9:nregime#skybot-foundation-split-v1 | ok | 1.665 | 320 | 0.025 | - | 65.27 |  |
| full_no_split | 2 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8754 | 320 | 0.083 | - | 65.27 |  |
| no_update | 2 | e9:nregime#skybot-foundation-split-v1 | ok | 4.5 | - | 0.007 | - | 65.27 |  |
| refit_only | 2 | e9:nregime#skybot-foundation-split-v1 | ok | 1.706 | 320 | 0.025 | - | 65.33 |  |
| full | 2 | e9:nregime#skybot-foundation-split-v1 | ok | 0.8754 | 320 | 0.427 | - | 65.41 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
