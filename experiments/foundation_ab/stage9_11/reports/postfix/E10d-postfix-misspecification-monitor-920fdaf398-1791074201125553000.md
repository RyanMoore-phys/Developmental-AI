# A/B report: E10d-postfix-misspecification-monitor

**Verdict: accept**

- mean improvement 0.8 >= delta 0.5, CI [0.656, 0.9152] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: on the verifier's E10d misspecified scenarios (truth not a shift; budget 60), the library misspecification monitor (HypothesisSet.update_categorical -> misspecified) lowers the rate of confident-wrong-UNFLAGGED scenarios by >= 0.5 against the same runs without it; well-specified false flags reported as the alternative arm
- primary metric: `confident_wrong_unflagged` (lower is better)
- acceptance rule: accept iff info_misspec_mon beats info_misspec_nomon on confident_wrong_unflagged (lower is better, basis=final) by mean >= 0.5 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.5 or candidate failures exceed 0; else inconclusive
- arms: baseline `info_misspec_nomon`, candidate `info_misspec_mon`, ablation `random_misspec_mon`, alternative `info_wellspec_mon`
- seeds: [0, 1, 2, 3, 4]; splits: ['e10:misspec#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 120.0}
- failure conditions: ['any arm raises', 'well-specified arm flags in > 5% of scenarios']
- frozen: 1791074195728371000 ns, hash `920fdaf3984754145ffb0f92eade759c9d35e026044e5a8ca0652c2da73bc4cb`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| info_misspec_nomon | 5/5 | 0 | 0 | 1 | 0 | 0 | 0 | 1.429 | - |
| info_misspec_mon | 5/5 | 0 | 0 | 0.2 | 0.09959 | 0 | 0 | 1.650 | - |
| random_misspec_mon | 5/5 | 0 | 0 | 0.9091 | 0.06428 | 0 | 0 | 0.676 | - |
| info_wellspec_mon | 5/5 | 0 | 0 | 0.01818 | 0.04066 | 0 | 0 | 1.641 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.8 | 0.656 | 0.9152 | 0 |
| equal_interactions | 0 | - | - | - | 5 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0.7091 | 0.6227 | 0.7955 | 0 |
| vs alternative_arm | 5 | -0.1818 | -0.3258 | -0.06665 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| info_misspec_nomon | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | - | 0.272 | - | 378.2 |  |
| info_misspec_mon | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 0.3636 | - | 0.322 | - | 378.2 |  |
| random_misspec_mon | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | - | 0.156 | - | 378.2 |  |
| info_wellspec_mon | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | - | 0.332 | - | 378.2 |  |
| info_misspec_mon | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 0.1818 | - | 0.338 | - | 378.2 |  |
| random_misspec_mon | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 0.9091 | - | 0.131 | - | 378.2 |  |
| info_wellspec_mon | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 0.09091 | - | 0.338 | - | 378.2 |  |
| info_misspec_nomon | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | - | 0.287 | - | 378.2 |  |
| random_misspec_mon | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 0.8182 | - | 0.131 | - | 378.2 |  |
| info_wellspec_mon | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | - | 0.333 | - | 378.2 |  |
| info_misspec_nomon | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | - | 0.284 | - | 378.2 |  |
| info_misspec_mon | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 0.1818 | - | 0.328 | - | 378.2 |  |
| info_wellspec_mon | 3 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | - | 0.317 | - | 378.2 |  |
| info_misspec_nomon | 3 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | - | 0.313 | - | 378.2 |  |
| info_misspec_mon | 3 | e10:misspec#skybot-foundation-split-v1 | ok | 0.1818 | - | 0.332 | - | 378.2 |  |
| random_misspec_mon | 3 | e10:misspec#skybot-foundation-split-v1 | ok | 0.9091 | - | 0.129 | - | 378.2 |  |
| info_misspec_nomon | 4 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | - | 0.272 | - | 378.2 |  |
| info_misspec_mon | 4 | e10:misspec#skybot-foundation-split-v1 | ok | 0.09091 | - | 0.330 | - | 378.2 |  |
| random_misspec_mon | 4 | e10:misspec#skybot-foundation-split-v1 | ok | 0.9091 | - | 0.130 | - | 378.2 |  |
| info_wellspec_mon | 4 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | - | 0.321 | - | 378.2 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
