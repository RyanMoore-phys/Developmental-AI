# A/B report: E2a-epistemic-has-predictive-value

**Verdict: accept**

- mean improvement 0.5852 >= delta 0.1, CI [0.5466, 0.6197] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- WARNING: exploratory arms not in the prereg (reported, never decisive): ['ens5_iid']

## Preregistration

- hypothesis: ens5 epistemic variance rank-correlates with held-out squared error (pooled id + gravity/box shifts) by >= 0.1 more than a permuted (information-free) signal
- primary metric: `unc_err_spearman` (higher is better)
- acceptance rule: accept iff ens5 beats null_perm on unc_err_spearman (higher is better, basis=final) by mean >= 0.1 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.1 or candidate failures exceed 0; else inconclusive
- arms: baseline `null_perm`, candidate `ens5`, ablation `k1`, alternative `ens5_noboot`
- seeds: [21, 22, 23, 24, 25]; splits: ['box2d:e2#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 300.0}
- failure conditions: ['any arm raises', 'non-finite metric']
- frozen: 1791072057369336000 ns, hash `7d07b24a616e0ed2117f4a5e617b078b01b1a02e1d420bde561bba5511d82a0f`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.5885 | 0.0304 | 9000 | 0 | 54.136 | - |
| k1 | 5/5 | 0 | 0 | 0.6454 | 0.0262 | 9000 | 0 | 21.096 | - |
| null_perm | 5/5 | 0 | 0 | 0.003273 | 0.007567 | 9000 | 0 | 55.448 | - |
| ens5_noboot | 5/5 | 0 | 0 | 0.5995 | 0.04142 | 9000 | 0 | 51.849 | - |
| ens5_iid | 5/5 | 0 | 0 | 0.572 | 0.02707 | 9000 | 0 | 48.552 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.5852 | 0.5466 | 0.6197 | 0 |
| equal_interactions | 5 | 0.5852 | 0.5466 | 0.6197 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | -0.05688 | -0.1039 | -0.01693 | 0 |
| vs alternative_arm | 5 | -0.01108 | -0.04321 | 0.01836 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| ens5 | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5756 | 1800 | 11.088 | - | 627.6 |  |
| k1 | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6426 | 1800 | 4.250 | - | 627.6 |  |
| null_perm | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.004008 | 1800 | 11.897 | - | 627.6 |  |
| ens5_noboot | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6016 | 1800 | 11.360 | - | 627.6 |  |
| ens5_iid | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5626 | 1800 | 9.588 | - | 627.6 |  |
| k1 | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6823 | 1800 | 3.906 | - | 627.6 |  |
| null_perm | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.01062 | 1800 | 13.082 | - | 627.6 |  |
| ens5_noboot | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6309 | 1800 | 9.569 | - | 627.6 |  |
| ens5_iid | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5631 | 1800 | 10.067 | - | 627.6 |  |
| ens5 | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6294 | 1800 | 9.806 | - | 627.6 |  |
| null_perm | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.003857 | 1800 | 10.147 | - | 627.6 |  |
| ens5_noboot | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5387 | 1800 | 9.576 | - | 627.6 |  |
| ens5_iid | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5763 | 1800 | 9.675 | - | 627.6 |  |
| ens5 | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5465 | 1800 | 11.433 | - | 627.6 |  |
| k1 | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6596 | 1800 | 4.183 | - | 627.6 |  |
| ens5_noboot | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6433 | 1800 | 9.535 | - | 627.6 |  |
| ens5_iid | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6153 | 1800 | 9.656 | - | 627.6 |  |
| ens5 | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5958 | 1800 | 10.146 | - | 627.6 |  |
| k1 | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6227 | 1800 | 3.802 | - | 627.6 |  |
| null_perm | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.007206 | 1800 | 9.575 | - | 627.6 |  |
| ens5_iid | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5426 | 1800 | 9.566 | - | 627.6 |  |
| ens5 | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.595 | 1800 | 11.663 | - | 627.6 |  |
| k1 | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6196 | 1800 | 4.956 | - | 627.6 |  |
| null_perm | 25 | box2d:e2#skybot-foundation-split-v1 | ok | -0.009326 | 1800 | 10.746 | - | 627.6 |  |
| ens5_noboot | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5834 | 1800 | 11.809 | - | 627.6 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
