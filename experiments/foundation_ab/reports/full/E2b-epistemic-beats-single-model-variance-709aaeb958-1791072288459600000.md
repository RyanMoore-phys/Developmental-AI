# A/B report: E2b-epistemic-beats-single-model-variance

**Verdict: reject**

- CI upper bound -0.01618 < delta 0.05: the hypothesised effect is ruled out
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
- frozen: 1791072057369387000 ns, hash `709aaeb95837ba31c00a863e0fb7e281bd5a5d6813e2bbf81953ac9f9c6d143f`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.5885 | 0.0304 | 9000 | 0 | 0.000 | - |
| k1 | 5/5 | 0 | 0 | 0.6454 | 0.0262 | 9000 | 0 | 0.000 | - |
| null_perm | 5/5 | 0 | 0 | 0.003273 | 0.007567 | 9000 | 0 | 0.000 | - |
| ens5_noboot | 5/5 | 0 | 0 | 0.5995 | 0.04142 | 9000 | 0 | 0.000 | - |
| ens5_iid | 5/5 | 0 | 0 | 0.572 | 0.02707 | 9000 | 0 | 0.000 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | -0.05688 | -0.1032 | -0.01618 | 0 |
| equal_interactions | 5 | -0.05688 | -0.1032 | -0.01618 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | -0.01108 | -0.04321 | 0.01836 | 0 |
| vs alternative_arm | 5 | 0.5852 | 0.5469 | 0.6197 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| ens5 | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5756 | 1800 | 0.000 | - | 627.6 |  |
| k1 | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6426 | 1800 | 0.000 | - | 627.6 |  |
| null_perm | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.004008 | 1800 | 0.000 | - | 627.6 |  |
| ens5_noboot | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6016 | 1800 | 0.000 | - | 627.6 |  |
| ens5_iid | 21 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5626 | 1800 | 0.000 | - | 627.6 |  |
| k1 | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6823 | 1800 | 0.000 | - | 627.6 |  |
| null_perm | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.01062 | 1800 | 0.000 | - | 627.6 |  |
| ens5_noboot | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6309 | 1800 | 0.000 | - | 627.6 |  |
| ens5_iid | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5631 | 1800 | 0.000 | - | 627.6 |  |
| ens5 | 22 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6294 | 1800 | 0.000 | - | 627.6 |  |
| null_perm | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.003857 | 1800 | 0.000 | - | 627.6 |  |
| ens5_noboot | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5387 | 1800 | 0.000 | - | 627.6 |  |
| ens5_iid | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5763 | 1800 | 0.000 | - | 627.6 |  |
| ens5 | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5465 | 1800 | 0.000 | - | 627.6 |  |
| k1 | 23 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6596 | 1800 | 0.000 | - | 627.6 |  |
| ens5_noboot | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6433 | 1800 | 0.000 | - | 627.6 |  |
| ens5_iid | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6153 | 1800 | 0.000 | - | 627.6 |  |
| ens5 | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5958 | 1800 | 0.000 | - | 627.6 |  |
| k1 | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6227 | 1800 | 0.000 | - | 627.6 |  |
| null_perm | 24 | box2d:e2#skybot-foundation-split-v1 | ok | 0.007206 | 1800 | 0.000 | - | 627.6 |  |
| ens5_iid | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5426 | 1800 | 0.000 | - | 627.6 |  |
| ens5 | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.595 | 1800 | 0.000 | - | 627.6 |  |
| k1 | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.6196 | 1800 | 0.000 | - | 627.6 |  |
| null_perm | 25 | box2d:e2#skybot-foundation-split-v1 | ok | -0.009326 | 1800 | 0.000 | - | 627.6 |  |
| ens5_noboot | 25 | box2d:e2#skybot-foundation-split-v1 | ok | 0.5834 | 1800 | 0.000 | - | 627.6 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
