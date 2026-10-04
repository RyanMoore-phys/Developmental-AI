# A/B report: E9e-rare-misfit-f0.01

**Verdict: accept**

- mean improvement 0.9029 >= delta 0.05, CI [0.3225, 1.949] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: a change touching only 1% of rows (y += 3 when r == 1) is found by the full language: held-out GAUSSIAN NLL beats REFIT-only by >= 0.05 nats/row (if the 1% outlier component hides the misfit, this fails while the eps~0 ablation succeeds)
- primary metric: `nll_gauss` (lower is better)
- acceptance rule: accept iff full_robust beats refit_only on nll_gauss (lower is better, basis=final) by mean >= 0.05 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.05 or candidate failures exceed 0; else inconclusive
- arms: baseline `refit_only`, candidate `full_robust`, ablation `full_eps0`, alternative `None`
- seeds: [0, 1, 2]; splits: ['e9:rare0.01#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071193516331000 ns, hash `1a0c1c6a001d0d143fc80eb0d839d8cbd719716f091aa377a5366d245c2f9b63`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 3/3 | 0 | 0 | 0.7063 | 0.3548 | 1800 | 0 | 0.085 | - |
| full_robust | 3/3 | 0 | 0 | -0.1966 | 0.03193 | 1800 | 0 | 0.568 | - |
| full_eps0 | 3/3 | 0 | 0 | -0.1966 | 0.03193 | 1800 | 0 | 0.656 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 0.9029 | 0.3225 | 1.949 | 0 |
| equal_interactions | 3 | 0.9029 | 0.3225 | 1.949 | 0 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | e9:rare0.01#skybot-foundation-split-v1 | ok | 0.5066 | 600 | 0.030 | - | 67 |  |
| full_robust | 0 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.1805 | 600 | 0.155 | - | 67 |  |
| full_eps0 | 0 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.1805 | 600 | 0.138 | - | 67.02 |  |
| full_robust | 1 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.176 | 600 | 0.139 | - | 67.02 |  |
| full_eps0 | 1 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.176 | 600 | 0.256 | - | 67.02 |  |
| refit_only | 1 | e9:rare0.01#skybot-foundation-split-v1 | ok | 1.116 | 600 | 0.027 | - | 67.02 |  |
| full_eps0 | 2 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.2334 | 600 | 0.262 | - | 67.02 |  |
| refit_only | 2 | e9:rare0.01#skybot-foundation-split-v1 | ok | 0.4964 | 600 | 0.028 | - | 67.03 |  |
| full_robust | 2 | e9:rare0.01#skybot-foundation-split-v1 | ok | -0.2334 | 600 | 0.274 | - | 67.03 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
