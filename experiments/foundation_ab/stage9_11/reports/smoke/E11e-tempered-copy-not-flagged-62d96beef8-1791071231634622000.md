# A/B report: E11e-tempered-copy-not-flagged

**Verdict: accept**

- mean improvement 1 >= delta 0.5, CI [1, 1] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- WARNING: exploratory arms not in the prereg (reported, never decisive): ['drift_0.001', 'drift_0.01', 'drift_0.1', 'offprobe']

## Preregistration

- hypothesis: a copy with logits x3 (identical argmax on every state, so identical deterministic behaviour) is flagged at a rate >= 0.5 BELOW a byte copy
- primary metric: `flag_rate` (lower is better)
- acceptance rule: accept iff tempered beats name_only on flag_rate (lower is better, basis=final) by mean >= 0.5 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.5 or candidate failures exceed 0; else inconclusive
- arms: baseline `name_only`, candidate `tempered`, ablation `samename_diff`, alternative `None`
- seeds: [0, 1, 2]; splits: ['e11:skills#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 30.0}
- failure conditions: ['any arm raises']
- frozen: 1791071221514298000 ns, hash `62d96beef8b6690cfcaecfc465995dbea46c82e85431956ed7bd03910a172176`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| name_only | 3/3 | 0 | 0 | 1 | 0 | 18 | 0 | 0.000 | - |
| samename_diff | 3/3 | 0 | 0 | 0 | 0 | 18 | 0 | 0.000 | - |
| offprobe | 3/3 | 0 | 0 | 0.8889 | 0.1925 | 18 | 0 | 0.000 | - |
| tempered | 3/3 | 0 | 0 | 0 | 0 | 18 | 0 | 0.000 | - |
| drift_0.001 | 3/3 | 0 | 0 | 1 | 0 | 18 | 0 | 0.000 | - |
| drift_0.01 | 3/3 | 0 | 0 | 1 | 0 | 18 | 0 | 0.000 | - |
| drift_0.1 | 3/3 | 0 | 0 | 0 | 0 | 18 | 0 | 0.000 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 3 | 1 | 1 | 1 | 0 |
| equal_interactions | 3 | 1 | 1 | 1 | 0 |
| equal_time | 0 | - | - | - | 3 |
| vs ablation_arm | 3 | 0 | 0 | 0 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| name_only | 0 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| samename_diff | 0 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |
| offprobe | 0 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| tempered | 0 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |
| drift_0.001 | 0 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| drift_0.01 | 0 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| drift_0.1 | 0 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |
| samename_diff | 1 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |
| offprobe | 1 | e11:skills#skybot-foundation-split-v1 | ok | 0.6667 | 6 | 0.000 | - | 317.5 |  |
| tempered | 1 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |
| drift_0.001 | 1 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| drift_0.01 | 1 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| drift_0.1 | 1 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |
| name_only | 1 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| offprobe | 2 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| tempered | 2 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |
| drift_0.001 | 2 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| drift_0.01 | 2 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| drift_0.1 | 2 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |
| name_only | 2 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 6 | 0.000 | - | 317.5 |  |
| samename_diff | 2 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 6 | 0.000 | - | 317.5 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
