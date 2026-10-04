# A/B report: E11a-gate-vs-ungated-rare-region

**Verdict: inconclusive**

- CI [-0.2036, 0.6138] with mean 0.1035 neither meets nor rules out delta 0.02
- note: criterion unmet: baseline 'ungated' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: with a model wrong only inside a rarely visited disc (claims boost, truth is mud), the reliability gate lowers real mean distance to the goal versus the ungated planner by >= 0.02 m
- primary metric: `mean_dist` (lower is better)
- acceptance rule: accept iff gated beats ungated on mean_dist (lower is better, basis=final) by mean >= 0.02 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.02 or candidate failures exceed 0; else inconclusive
- arms: baseline `ungated`, candidate `gated`, ablation `fallback`, alternative `gated_true`
- seeds: [0, 1, 2]; splits: ['e11:mud#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 120.0}
- failure conditions: ['any arm raises']
- frozen: 1791071221513619000 ns, hash `e1d72c942e2b2f7bd261eb91b06336f360314756c4d1114cf38bf01427d6c4b0`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| fallback | 3/3 | 0 | 0 | 1.219 | 0.4273 | 270 | 0 | 0.063 | - |
| gated | 3/3 | 0 | 0 | 1.187 | 0.4787 | 270 | 0 | 1.071 | - |
| ungated | 3/3 | 0 | 0 | 1.291 | 0.4663 | 270 | 0 | 2.044 | - |
| gated_true | 3/3 | 0 | 0 | 0.9939 | 0.361 | 270 | 0 | 2.342 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 0.1035 | -0.2036 | 0.6138 | 0 |
| equal_interactions | 3 | 0.1035 | -0.2036 | 0.6138 | 0 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 0.03234 | -0.2551 | 0.3819 | 0 |
| vs alternative_arm | 3 | -0.1933 | -0.464 | 0.2606 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| fallback | 0 | e11:mud#skybot-foundation-split-v1 | ok | 1.04 | 90 | 0.021 | - | 317.5 |  |
| gated | 0 | e11:mud#skybot-foundation-split-v1 | ok | 1.114 | 90 | 0.460 | - | 317.5 |  |
| ungated | 0 | e11:mud#skybot-foundation-split-v1 | ok | 1.407 | 90 | 0.688 | - | 317.5 |  |
| gated_true | 0 | e11:mud#skybot-foundation-split-v1 | ok | 0.8528 | 90 | 0.753 | - | 317.5 |  |
| gated | 1 | e11:mud#skybot-foundation-split-v1 | ok | 0.7491 | 90 | 0.515 | - | 317.5 |  |
| ungated | 1 | e11:mud#skybot-foundation-split-v1 | ok | 0.777 | 90 | 0.664 | - | 317.5 |  |
| gated_true | 1 | e11:mud#skybot-foundation-split-v1 | ok | 0.7246 | 90 | 0.772 | - | 317.5 |  |
| fallback | 1 | e11:mud#skybot-foundation-split-v1 | ok | 0.9115 | 90 | 0.021 | - | 317.5 |  |
| ungated | 2 | e11:mud#skybot-foundation-split-v1 | ok | 1.687 | 90 | 0.692 | - | 317.5 |  |
| gated_true | 2 | e11:mud#skybot-foundation-split-v1 | ok | 1.404 | 90 | 0.816 | - | 317.5 |  |
| fallback | 2 | e11:mud#skybot-foundation-split-v1 | ok | 1.707 | 90 | 0.022 | - | 317.5 |  |
| gated | 2 | e11:mud#skybot-foundation-split-v1 | ok | 1.698 | 90 | 0.096 | - | 317.5 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
