# A/B report: E10d-misspecified-hypothesis-set

**Verdict: accept**

- mean improvement 0.9818 >= delta 0.3, CI [0.9242, 1.011] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: when the truth is NOT in the hypothesis set, info-gain selection reaches > 0.9 posterior on a (necessarily wrong) shift with no flag in >= 30 percentage points more scenarios than it reaches confident-wrong when the set is well specified
- primary metric: `confident_wrong_unflagged` (higher is better)
- acceptance rule: accept iff info_misspec beats info_wellspec on confident_wrong_unflagged (higher is better, basis=final) by mean >= 0.3 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.3 or candidate failures exceed 0; else inconclusive
- arms: baseline `info_wellspec`, candidate `info_misspec`, ablation `random_misspec`, alternative `info_ppc_misspec`
- seeds: [0, 1, 2, 3, 4]; splits: ['e10:misspec#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises', "the alternative arm's PPC flag is the verifier's, not the library's; it is reported to show feasibility only"]
- frozen: 1791071277578660000 ns, hash `2fd51cf6b351919430828c445b9a353ad2eb27a7334860cd7a056abbcf89411d`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| info_wellspec | 5/5 | 0 | 0 | 0.01818 | 0.04066 | 285 | 0 | 0.134 | - |
| info_misspec | 5/5 | 0 | 0 | 1 | 0 | 701 | 0 | 0.314 | - |
| random_misspec | 5/5 | 0 | 0 | 1 | 0 | 768 | 0 | 0.109 | - |
| info_ppc_misspec | 5/5 | 0 | 0 | 0.9273 | 0.04066 | 701 | 0 | 0.318 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.9818 | 0.9242 | 1.011 | 0 |
| equal_interactions | 0 | - | - | - | 5 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | 0 | 0 | 0 | 0 |
| vs alternative_arm | 5 | 0.07273 | 0.01514 | 0.1015 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| info_wellspec | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | 67 | 0.032 | - | 86 |  |
| info_misspec | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 164 | 0.074 | - | 86 |  |
| random_misspec | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 166 | 0.024 | - | 86 |  |
| info_ppc_misspec | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 0.9091 | 164 | 0.075 | - | 86 |  |
| info_misspec | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 72 | 0.034 | - | 86 |  |
| random_misspec | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 150 | 0.022 | - | 86 |  |
| info_ppc_misspec | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 0.9091 | 72 | 0.034 | - | 86 |  |
| info_wellspec | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 0.09091 | 56 | 0.027 | - | 86 |  |
| random_misspec | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 111 | 0.017 | - | 86 |  |
| info_ppc_misspec | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 0.9091 | 183 | 0.082 | - | 86 |  |
| info_wellspec | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | 61 | 0.029 | - | 86 |  |
| info_misspec | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 183 | 0.080 | - | 86 |  |
| info_ppc_misspec | 3 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 158 | 0.069 | - | 86 |  |
| info_wellspec | 3 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | 52 | 0.024 | - | 86 |  |
| info_misspec | 3 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 158 | 0.069 | - | 86 |  |
| random_misspec | 3 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 139 | 0.019 | - | 86 |  |
| info_wellspec | 4 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | 49 | 0.023 | - | 86 |  |
| info_misspec | 4 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 124 | 0.057 | - | 86 |  |
| random_misspec | 4 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 202 | 0.028 | - | 86 |  |
| info_ppc_misspec | 4 | e10:misspec#skybot-foundation-split-v1 | ok | 0.9091 | 124 | 0.058 | - | 86 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
