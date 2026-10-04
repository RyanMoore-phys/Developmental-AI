# A/B report: E9d-rare-misfit-f0.03

**Verdict: accept**

- mean improvement 4.243 >= delta 0.05, CI [2.559, 5.96] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: a change touching only 3% of rows (y += 3 when r == 1) is found by the full language: held-out GAUSSIAN NLL beats REFIT-only by >= 0.05 nats/row (if the 1% outlier component hides the misfit, this fails while the eps~0 ablation succeeds)
- primary metric: `nll_gauss` (lower is better)
- acceptance rule: accept iff full_robust beats refit_only on nll_gauss (lower is better, basis=final) by mean >= 0.05 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.05 or candidate failures exceed 0; else inconclusive
- arms: baseline `refit_only`, candidate `full_robust`, ablation `full_eps0`, alternative `None`
- seeds: [0, 1, 2, 3, 4]; splits: ['e9:rare0.03#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791074172877046000 ns, hash `cc30a3fd358e347a0a7df71c8657624d09b1afb7249976e803589e639832e1fd`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 5/5 | 0 | 0 | 4.07 | 1.445 | 3000 | 0 | 0.064 | - |
| full_robust | 5/5 | 0 | 0 | -0.1733 | 0.04673 | 3000 | 0 | 0.383 | - |
| full_eps0 | 5/5 | 0 | 0 | -0.1733 | 0.04673 | 3000 | 0 | 0.344 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 4.243 | 2.559 | 5.96 | 0 |
| equal_interactions | 5 | 4.243 | 2.559 | 5.96 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | e9:rare0.03#skybot-foundation-split-v1 | ok | 5.139 | 600 | 0.013 | - | 86.38 |  |
| full_robust | 0 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1734 | 600 | 0.068 | - | 86.38 |  |
| full_eps0 | 0 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1734 | 600 | 0.067 | - | 86.38 |  |
| full_robust | 1 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.2369 | 600 | 0.085 | - | 86.38 |  |
| full_eps0 | 1 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.2369 | 600 | 0.082 | - | 86.41 |  |
| refit_only | 1 | e9:rare0.03#skybot-foundation-split-v1 | ok | 2.506 | 600 | 0.013 | - | 86.41 |  |
| full_eps0 | 2 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1949 | 600 | 0.056 | - | 86.47 |  |
| refit_only | 2 | e9:rare0.03#skybot-foundation-split-v1 | ok | 3.638 | 600 | 0.013 | - | 86.47 |  |
| full_robust | 2 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1949 | 600 | 0.057 | - | 86.5 |  |
| refit_only | 3 | e9:rare0.03#skybot-foundation-split-v1 | ok | 3.092 | 600 | 0.013 | - | 86.5 |  |
| full_robust | 3 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1476 | 600 | 0.091 | - | 86.61 |  |
| full_eps0 | 3 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1476 | 600 | 0.059 | - | 86.64 |  |
| full_robust | 4 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1136 | 600 | 0.081 | - | 86.75 |  |
| full_eps0 | 4 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1136 | 600 | 0.080 | - | 86.75 |  |
| refit_only | 4 | e9:rare0.03#skybot-foundation-split-v1 | ok | 5.973 | 600 | 0.013 | - | 86.75 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
