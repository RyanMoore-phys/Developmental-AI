# A/B report: E1a-IN-vs-flopmatched-MLP-nll1

**Verdict: inconclusive**

- CI [-0.01801, 0.8076] with mean 0.326 neither meets nor rules out delta 0.05
- note: criterion unmet: baseline 'mlp_flops' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- WARNING: exploratory arms not in the prereg (reported, never decisive): ['in_norel']

## Preregistration

- hypothesis: At equal FLOPs/sample, equal data and equal epochs, the relational IN has lower held-out 1-step Gaussian NLL (ball dims) than an MLP on the same structured inputs
- primary metric: `nll1` (lower is better)
- acceptance rule: accept iff in beats mlp_flops on nll1 (lower is better, basis=final) by mean >= 0.05 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.05 or candidate failures exceed 0; else inconclusive
- arms: baseline `mlp_flops`, candidate `in`, ablation `in_noedges`, alternative `mlp128`
- seeds: [11, 12, 13, 14, 15]; splits: ['box2d:e1#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 300.0}
- failure conditions: ['any arm raises', 'non-finite metric', 'loss does not fall']
- frozen: 1791069632293082000 ns, hash `c128bac46c3533216d4593b5adeb51e1285bf07cde7a1a66ee4cccb8c91d9848`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| in | 5/5 | 0 | 0 | 0.1753 | 0.0997 | 8800 | 8400 | 43.474 | - |
| mlp_flops | 5/5 | 0 | 0 | 0.5012 | 0.4295 | 8800 | 8400 | 34.100 | - |
| mlp128 | 5/5 | 0 | 0 | 0.6229 | 0.2486 | 8800 | 8400 | 118.176 | - |
| in_noedges | 5/5 | 0 | 0 | 0.2132 | 0.08562 | 8800 | 8400 | 22.288 | - |
| in_norel | 5/5 | 0 | 0 | 0.1622 | 0.09168 | 8800 | 8400 | 44.007 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.326 | -0.01801 | 0.8076 | 0 |
| equal_interactions | 5 | 0.326 | -0.01801 | 0.8076 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0.03789 | -0.02156 | 0.09733 | 0 |
| vs alternative_arm | 5 | 0.4476 | 0.2477 | 0.659 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| in | 11 | box2d:e1#skybot-foundation-split-v1 | ok | 0.1906 | 1760 | 8.289 | - | 593.7 |  |
| mlp_flops | 11 | box2d:e1#skybot-foundation-split-v1 | ok | 0.2839 | 1760 | 6.628 | - | 593.7 |  |
| mlp128 | 11 | box2d:e1#skybot-foundation-split-v1 | ok | 0.5351 | 1760 | 22.202 | - | 593.7 |  |
| in_noedges | 11 | box2d:e1#skybot-foundation-split-v1 | ok | 0.1789 | 1760 | 4.273 | - | 593.7 |  |
| in_norel | 11 | box2d:e1#skybot-foundation-split-v1 | ok | 0.1416 | 1760 | 9.139 | - | 593.7 |  |
| mlp_flops | 12 | box2d:e1#skybot-foundation-split-v1 | ok | 0.2711 | 1760 | 6.272 | - | 593.7 |  |
| mlp128 | 12 | box2d:e1#skybot-foundation-split-v1 | ok | 0.4505 | 1760 | 22.696 | - | 593.7 |  |
| in_noedges | 12 | box2d:e1#skybot-foundation-split-v1 | ok | 0.1396 | 1760 | 4.180 | - | 593.7 |  |
| in_norel | 12 | box2d:e1#skybot-foundation-split-v1 | ok | 0.09296 | 1760 | 7.964 | - | 593.7 |  |
| in | 12 | box2d:e1#skybot-foundation-split-v1 | ok | 0.05061 | 1760 | 8.010 | - | 593.7 |  |
| mlp128 | 13 | box2d:e1#skybot-foundation-split-v1 | ok | 0.4191 | 1760 | 22.332 | - | 593.7 |  |
| in_noedges | 13 | box2d:e1#skybot-foundation-split-v1 | ok | 0.1598 | 1760 | 4.218 | - | 593.7 |  |
| in_norel | 13 | box2d:e1#skybot-foundation-split-v1 | ok | 0.07098 | 1760 | 8.065 | - | 593.7 |  |
| in | 13 | box2d:e1#skybot-foundation-split-v1 | ok | 0.1601 | 1760 | 8.080 | - | 593.7 |  |
| mlp_flops | 13 | box2d:e1#skybot-foundation-split-v1 | ok | 0.2087 | 1760 | 6.125 | - | 593.7 |  |
| in_noedges | 14 | box2d:e1#skybot-foundation-split-v1 | ok | 0.3525 | 1760 | 4.301 | - | 593.7 |  |
| in_norel | 14 | box2d:e1#skybot-foundation-split-v1 | ok | 0.2956 | 1760 | 8.847 | - | 593.7 |  |
| in | 14 | box2d:e1#skybot-foundation-split-v1 | ok | 0.327 | 1760 | 9.174 | - | 593.7 |  |
| mlp_flops | 14 | box2d:e1#skybot-foundation-split-v1 | ok | 1.244 | 1760 | 6.881 | - | 593.7 |  |
| mlp128 | 14 | box2d:e1#skybot-foundation-split-v1 | ok | 1.029 | 1760 | 24.002 | - | 593.7 |  |
| in_norel | 15 | box2d:e1#skybot-foundation-split-v1 | ok | 0.21 | 1760 | 9.993 | - | 593.7 |  |
| in | 15 | box2d:e1#skybot-foundation-split-v1 | ok | 0.1482 | 1760 | 9.921 | - | 593.7 |  |
| mlp_flops | 15 | box2d:e1#skybot-foundation-split-v1 | ok | 0.4983 | 1760 | 8.193 | - | 593.7 |  |
| mlp128 | 15 | box2d:e1#skybot-foundation-split-v1 | ok | 0.6809 | 1760 | 26.944 | - | 593.7 |  |
| in_noedges | 15 | box2d:e1#skybot-foundation-split-v1 | ok | 0.2351 | 1760 | 5.316 | - | 593.7 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
