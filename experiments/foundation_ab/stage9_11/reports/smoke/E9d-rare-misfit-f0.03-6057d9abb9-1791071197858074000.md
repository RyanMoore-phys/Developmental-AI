# A/B report: E9d-rare-misfit-f0.03

**Verdict: accept**

- mean improvement 3.963 >= delta 0.05, CI [0.6837, 7.591] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: a change touching only 3% of rows (y += 3 when r == 1) is found by the full language: held-out GAUSSIAN NLL beats REFIT-only by >= 0.05 nats/row (if the 1% outlier component hides the misfit, this fails while the eps~0 ablation succeeds)
- primary metric: `nll_gauss` (lower is better)
- acceptance rule: accept iff full_robust beats refit_only on nll_gauss (lower is better, basis=final) by mean >= 0.05 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.05 or candidate failures exceed 0; else inconclusive
- arms: baseline `refit_only`, candidate `full_robust`, ablation `full_eps0`, alternative `None`
- seeds: [0, 1, 2]; splits: ['e9:rare0.03#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises']
- frozen: 1791071193515965000 ns, hash `6057d9abb9e5176dbd5b4a7a65967593416e4c1b803ee585425c6454589b4875`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 3/3 | 0 | 0 | 3.761 | 1.32 | 1800 | 0 | 0.085 | - |
| full_robust | 3/3 | 0 | 0 | -0.2018 | 0.0323 | 1800 | 0 | 0.737 | - |
| full_eps0 | 3/3 | 0 | 0 | -0.2018 | 0.0323 | 1800 | 0 | 0.648 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 3.963 | 0.6837 | 7.591 | 0 |
| equal_interactions | 3 | 3.963 | 0.6837 | 7.591 | 0 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | e9:rare0.03#skybot-foundation-split-v1 | ok | 5.139 | 600 | 0.028 | - | 66.53 |  |
| full_robust | 0 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1734 | 600 | 0.275 | - | 66.53 |  |
| full_eps0 | 0 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1734 | 600 | 0.214 | - | 66.53 |  |
| full_robust | 1 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.2369 | 600 | 0.289 | - | 66.53 |  |
| full_eps0 | 1 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.2369 | 600 | 0.266 | - | 66.53 |  |
| refit_only | 1 | e9:rare0.03#skybot-foundation-split-v1 | ok | 2.506 | 600 | 0.029 | - | 66.53 |  |
| full_eps0 | 2 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1949 | 600 | 0.168 | - | 66.53 |  |
| refit_only | 2 | e9:rare0.03#skybot-foundation-split-v1 | ok | 3.638 | 600 | 0.028 | - | 66.64 |  |
| full_robust | 2 | e9:rare0.03#skybot-foundation-split-v1 | ok | -0.1949 | 600 | 0.173 | - | 66.73 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
