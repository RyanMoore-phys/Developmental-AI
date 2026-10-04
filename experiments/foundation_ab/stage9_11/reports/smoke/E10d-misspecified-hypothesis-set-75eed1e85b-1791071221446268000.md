# A/B report: E10d-misspecified-hypothesis-set

**Verdict: accept**

- mean improvement 1 >= delta 0.3, CI [1, 1] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

## Preregistration

- hypothesis: when the truth is NOT in the hypothesis set, info-gain selection reaches > 0.9 posterior on a (necessarily wrong) shift with no flag in >= 30 percentage points more scenarios than it reaches confident-wrong when the set is well specified
- primary metric: `confident_wrong_unflagged` (higher is better)
- acceptance rule: accept iff info_misspec beats info_wellspec on confident_wrong_unflagged (higher is better, basis=final) by mean >= 0.3 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.3 or candidate failures exceed 0; else inconclusive
- arms: baseline `info_wellspec`, candidate `info_misspec`, ablation `random_misspec`, alternative `info_ppc_misspec`
- seeds: [0, 1, 2]; splits: ['e10:misspec#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 60.0}
- failure conditions: ['any arm raises', "the alternative arm's PPC flag is the verifier's, not the library's; it is reported to show feasibility only"]
- frozen: 1791071214609608000 ns, hash `75eed1e85ba4fa8a62443cdbe14307938e55d62375d94cababe9e199b98eca6d`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| info_wellspec | 3/3 | 0 | 0 | 0 | 0 | 68 | 0 | 0.081 | - |
| info_misspec | 3/3 | 0 | 0 | 1 | 0 | 95 | 0 | 0.109 | - |
| random_misspec | 3/3 | 0 | 0 | 1 | 0 | 119 | 0 | 0.043 | - |
| info_ppc_misspec | 3/3 | 0 | 0 | 1 | 0 | 95 | 0 | 0.106 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 1 | 1 | 1 | 0 |
| equal_interactions | 0 | - | - | - | 3 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 0 | 0 | 0 | 0 |
| vs alternative_arm | 3 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| info_wellspec | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | 20 | 0.026 | - | 317.5 |  |
| info_misspec | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 27 | 0.035 | - | 317.5 |  |
| random_misspec | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 40 | 0.016 | - | 317.5 |  |
| info_ppc_misspec | 0 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 27 | 0.033 | - | 317.5 |  |
| info_misspec | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 17 | 0.020 | - | 317.5 |  |
| random_misspec | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 37 | 0.014 | - | 317.5 |  |
| info_ppc_misspec | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 17 | 0.021 | - | 317.5 |  |
| info_wellspec | 1 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | 18 | 0.020 | - | 317.5 |  |
| random_misspec | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 42 | 0.013 | - | 317.5 |  |
| info_ppc_misspec | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 51 | 0.053 | - | 317.5 |  |
| info_wellspec | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 0 | 30 | 0.035 | - | 317.5 |  |
| info_misspec | 2 | e10:misspec#skybot-foundation-split-v1 | ok | 1 | 51 | 0.054 | - | 317.5 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
