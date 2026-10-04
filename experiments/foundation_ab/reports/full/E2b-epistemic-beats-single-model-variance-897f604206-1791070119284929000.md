# A/B report: E2b-epistemic-beats-single-model-variance

**Verdict: reject**

- CI upper bound -0.19 < delta 0.05: the hypothesised effect is ruled out
- note: criterion unmet: baseline 'k1' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- WARNING: exploratory arms not in the prereg (reported, never decisive): ['ens5_iid']

## Preregistration

- hypothesis: ens5 epistemic variance predicts error better (Spearman, pooled) than a single model's own predictive variance (K=1)
- primary metric: `unc_err_spearman` (higher is better)
- acceptance rule: accept iff ens5 beats k1 on unc_err_spearman (higher is better, basis=final) by mean >= 0.05 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.05 or candidate failures exceed 0; else inconclusive
- arms: baseline `k1`, candidate `ens5`, ablation `ens5_noboot`, alternative `null_perm`
- seeds: [21, 22, 23, 24, 25]; splits: ['box2d:e2#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 300.0}
- failure conditions: ['any arm raises', 'non-finite metric']
- frozen: 1791069894376400000 ns, hash `897f604206bc4e9b7ae2dca0c1e47a6c8729b5b71fc280cfd3aba4ddda9d92a9`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.247 | 0.06118 | 9000 | 0 | 0.000 | - |
| k1 | 5/5 | 0 | 0 | 0.5162 | 0.04489 | 9000 | 0 | 0.000 | - |
| null_perm | 5/5 | 0 | 0 | 0.0006922 | 0.02754 | 9000 | 0 | 0.000 | - |
| ens5_noboot | 5/5 | 0 | 0 | 0.0337 | 0.04364 | 9000 | 0 | 0.000 | - |
| ens5_iid | 5/5 | 0 | 0 | 0.2777 | 0.07609 | 9000 | 0 | 0.000 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | -0.2692 | -0.3431 | -0.19 | 0 |
| equal_interactions | 5 | -0.2692 | -0.3431 | -0.19 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0.2133 | 0.09885 | 0.3277 | 0 |
| vs alternative_arm | 5 | 0.2463 | 0.1551 | 0.3368 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| ens5 | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.3163 | 1800 | 0.000 | - | 593.7 |  |
| k1 | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5445 | 1800 | 0.000 | - | 593.7 |  |
| null_perm | 21 | box2d:e2#skybot-foundation-split-v1 | ok | -0.02828 | 1800 | 0.000 | - | 593.7 |  |
| ens5_noboot | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.01507 | 1800 | 0.000 | - | 593.7 |  |
| ens5_iid | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.2894 | 1800 | 0.000 | - | 593.7 |  |
| k1 | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5706 | 1800 | 0.000 | - | 593.7 |  |
| null_perm | 22 | box2d:e2#skybot-foundation-split-v1 | ok | -0.01902 | 1800 | 0.000 | - | 593.7 |  |
| ens5_noboot | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.03356 | 1800 | 0.000 | - | 593.7 |  |
| ens5_iid | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.2812 | 1800 | 0.000 | - | 593.7 |  |
| ens5 | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.2345 | 1800 | 0.000 | - | 593.7 |  |
| null_perm | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.003762 | 1800 | 0.000 | - | 593.7 |  |
| ens5_noboot | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.08146 | 1800 | 0.000 | - | 593.7 |  |
| ens5_iid | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.1492 | 1800 | 0.000 | - | 593.7 |  |
| ens5 | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.2467 | 1800 | 0.000 | - | 593.7 |  |
| k1 | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5248 | 1800 | 0.000 | - | 593.7 |  |
| ens5_noboot | 24 | box2d:e2#skybot-foundation-split-v1 | ok | -0.02859 | 1800 | 0.000 | - | 593.7 |  |
| ens5_iid | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.3284 | 1800 | 0.000 | - | 593.7 |  |
| ens5 | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.2835 | 1800 | 0.000 | - | 593.7 |  |
| k1 | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.4655 | 1800 | 0.000 | - | 593.7 |  |
| null_perm | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.04296 | 1800 | 0.000 | - | 593.7 |  |
| ens5_iid | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.3403 | 1800 | 0.000 | - | 593.7 |  |
| ens5 | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.1539 | 1800 | 0.000 | - | 593.7 |  |
| k1 | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.4756 | 1800 | 0.000 | - | 593.7 |  |
| null_perm | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.004035 | 1800 | 0.000 | - | 593.7 |  |
| ens5_noboot | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.06702 | 1800 | 0.000 | - | 593.7 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
