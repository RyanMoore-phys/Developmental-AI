# A/B report: E9e-rare-misfit-f0.01

**Verdict: accept**

- mean improvement 0.7821 >= delta 0.05, CI [0.1395, 1.371] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: a change touching only 1% of rows (y += 3 when r == 1) is found by the full language: held-out GAUSSIAN NLL beats REFIT-only by >= 0.05 nats/row (if the 1% outlier component hides the misfit, this fails while the eps~0 ablation succeeds)
- primary metric: `nll_gauss` (lower is better)
- acceptance rule: accept iff full_robust beats refit_only on nll_gauss (lower is better, basis=final) by mean >= 0.05 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.05 or candidate failures exceed 0; else inconclusive
- arms: baseline `refit_only`, candidate `full_robust`, ablation `full_eps0`, alternative `None`
- seeds: [0, 1, 2, 3, 4]; splits: ['e9:rare0.01#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791074172877187000 ns, hash `9fc4d26fd743b33a131577801f1366b2ea0c2e5fec75a414fc65fd396dfe12a5`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 5/5 | 0 | 0 | 0.5916 | 0.5485 | 3000 | 0 | 0.067 | - |
| full_robust | 5/5 | 0 | 0 | -0.1905 | 0.04186 | 3000 | 0 | 0.323 | - |
| full_eps0 | 5/5 | 0 | 0 | -0.1905 | 0.04186 | 3000 | 0 | 0.363 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.7821 | 0.1395 | 1.371 | 0 |
| equal_interactions | 5 | 0.7821 | 0.1395 | 1.371 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | e9:rare0.01#skybot-foundation-split-v1 | ok | 0.5066 | 600 | 0.013 | - | 86.75 |  |
| full_robust | 0 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.1805 | 600 | 0.050 | - | 86.75 |  |
| full_eps0 | 0 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.1805 | 600 | 0.050 | - | 86.75 |  |
| full_robust | 1 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.176 | 600 | 0.050 | - | 86.8 |  |
| full_eps0 | 1 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.176 | 600 | 0.090 | - | 86.84 |  |
| refit_only | 1 | e9:rare0.01#skybot-foundation-split-v1 | ok | 1.116 | 600 | 0.013 | - | 86.88 |  |
| full_eps0 | 2 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.2334 | 600 | 0.091 | - | 86.88 |  |
| refit_only | 2 | e9:rare0.01#skybot-foundation-split-v1 | ok | 0.4964 | 600 | 0.014 | - | 86.89 |  |
| full_robust | 2 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.2334 | 600 | 0.092 | - | 86.89 |  |
| refit_only | 3 | e9:rare0.01#skybot-foundation-split-v1 | ok | 1.073 | 600 | 0.013 | - | 86.91 |  |
| full_robust | 3 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.1329 | 600 | 0.080 | - | 86.91 |  |
| full_eps0 | 3 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.1329 | 600 | 0.080 | - | 86.91 |  |
| full_robust | 4 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.2297 | 600 | 0.052 | - | 86.91 |  |
| full_eps0 | 4 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.2297 | 600 | 0.051 | - | 86.92 |  |
| refit_only | 4 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.2335 | 600 | 0.014 | - | 86.94 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
