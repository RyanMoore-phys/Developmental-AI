# A/B report: E11d-offprobe-false-duplicate

**Verdict: accept**

- mean improvement 0.92 >= delta 0.5, CI [0.825, 0.9992] excludes 0
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- WARNING: exploratory arms not in the prereg (reported, never decisive): ['drift_0.001', 'drift_0.01', 'drift_0.1', 'tempered']

## Preregistration

- hypothesis: two skills that differ only outside the probe distribution (argmax permuted when any |s_i| > 7) are flagged duplicates at a rate >= 0.5 above that of an unrelated same-name policy
- primary metric: `flag_rate` (higher is better)
- acceptance rule: accept iff offprobe beats samename_diff on flag_rate (higher is better, basis=final) by mean >= 0.5 with the paired bootstrap 95% CI (small-sample expanded) excluding 0, across >= 3 seeds; reject iff the CI upper bound < 0.5 or candidate failures exceed 0; else inconclusive
- arms: baseline `samename_diff`, candidate `offprobe`, ablation `name_only`, alternative `None`
- seeds: [0, 1, 2, 3, 4]; splits: ['e11:skills#skybot-foundation-split-v1']
- resource budget: {'max_wall_seconds': 30.0}
- failure conditions: ['any arm raises']
- frozen: 1791071311534130000 ns, hash `7bdc16f4e89a950c9701d36ff01a999938b3078fcc3245aafcaa0c2203d3e7a5`

## Arms

| arm | ok/runs | failed | budget | mean | std | interactions | updates | wall s | py peak MB |
|---|---|---|---|---|---|---|---|---|---|
| name_only | 5/5 | 0 | 0 | 1 | 0 | 100 | 0 | 0.906 | - |
| samename_diff | 5/5 | 0 | 0 | 0 | 0 | 100 | 0 | 1.101 | - |
| offprobe | 5/5 | 0 | 0 | 0.92 | 0.07583 | 100 | 0 | 1.026 | - |
| tempered | 5/5 | 0 | 0 | 0 | 0 | 100 | 0 | 0.805 | - |
| drift_0.001 | 5/5 | 0 | 0 | 1 | 0 | 100 | 0 | 0.846 | - |
| drift_0.01 | 5/5 | 0 | 0 | 1 | 0 | 100 | 0 | 0.975 | - |
| drift_0.1 | 5/5 | 0 | 0 | 0 | 0 | 100 | 0 | 0.852 | - |

## Comparisons (candidate minus baseline, sign-adjusted)

| basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|
| final (decisive) | 5 | 0.92 | 0.825 | 0.9992 | 0 |
| equal_interactions | 5 | 0.92 | 0.825 | 0.9992 | 0 |
| equal_time | 0 | - | - | - | 5 |
| vs ablation_arm | 5 | -0.08 | -0.175 | -0.0008235 | 0 |

## Per-run outcomes

| arm | seed | split | status | primary | interactions | wall s | py peak MB | rss hwm MB | error |
|---|---|---|---|---|---|---|---|---|---|
| name_only | 0 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.147 | - | 103.6 |  |
| samename_diff | 0 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.202 | - | 103.6 |  |
| offprobe | 0 | e11:skills#skybot-foundation-split-v1 | ok | 0.95 | 20 | 0.242 | - | 103.6 |  |
| tempered | 0 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.164 | - | 103.6 |  |
| drift_0.001 | 0 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.153 | - | 103.6 |  |
| drift_0.01 | 0 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.151 | - | 103.6 |  |
| drift_0.1 | 0 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.151 | - | 103.6 |  |
| samename_diff | 1 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.149 | - | 103.6 |  |
| offprobe | 1 | e11:skills#skybot-foundation-split-v1 | ok | 0.8 | 20 | 0.160 | - | 103.6 |  |
| tempered | 1 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.143 | - | 103.6 |  |
| drift_0.001 | 1 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.142 | - | 103.6 |  |
| drift_0.01 | 1 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.141 | - | 103.6 |  |
| drift_0.1 | 1 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.151 | - | 103.6 |  |
| name_only | 1 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.159 | - | 103.6 |  |
| offprobe | 2 | e11:skills#skybot-foundation-split-v1 | ok | 0.9 | 20 | 0.174 | - | 103.6 |  |
| tempered | 2 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.151 | - | 103.6 |  |
| drift_0.001 | 2 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.152 | - | 103.6 |  |
| drift_0.01 | 2 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.159 | - | 103.6 |  |
| drift_0.1 | 2 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.164 | - | 103.6 |  |
| name_only | 2 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.164 | - | 103.6 |  |
| samename_diff | 2 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.164 | - | 103.6 |  |
| tempered | 3 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.164 | - | 103.6 |  |
| drift_0.001 | 3 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.167 | - | 103.6 |  |
| drift_0.01 | 3 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.165 | - | 103.6 |  |
| drift_0.1 | 3 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.171 | - | 103.6 |  |
| name_only | 3 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.225 | - | 103.6 |  |
| samename_diff | 3 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.187 | - | 103.6 |  |
| offprobe | 3 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.209 | - | 103.6 |  |
| drift_0.001 | 4 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.232 | - | 103.6 |  |
| drift_0.01 | 4 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.359 | - | 103.6 |  |
| drift_0.1 | 4 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.215 | - | 103.6 |  |
| name_only | 4 | e11:skills#skybot-foundation-split-v1 | ok | 1 | 20 | 0.210 | - | 103.6 |  |
| samename_diff | 4 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.399 | - | 103.6 |  |
| offprobe | 4 | e11:skills#skybot-foundation-split-v1 | ok | 0.95 | 20 | 0.240 | - | 103.6 |  |
| tempered | 4 | e11:skills#skybot-foundation-split-v1 | ok | 0 | 20 | 0.184 | - | 103.6 |  |

rss hwm = process RSS high-water mark (monotone across runs); py peak = tracemalloc peak within the run.
