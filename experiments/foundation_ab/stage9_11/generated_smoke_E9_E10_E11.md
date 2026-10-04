## E9 (21s wall)

### E9a-structural-vs-refit-noisy-regime

- **verdict: accept** — mean improvement 0.8195 >= delta 0.1, CI [0.72, 0.8889] excludes 0
- hypothesis: after a structural rule change (threshold 0.3 on temp), on NEW noisier (sd 0.5), smaller (16x20 post rows) data with 7 distractors, the full proposal language yields lower held-out NLL than REFIT-only under the same SearchBudget (implementers report 1.82 nats/row on their fixture)
- primary `nll_robust` (lower better), delta 0.1, basis `final`; baseline `refit_only`, candidate `full`, ablation `full_no_split`, alternative `no_update`; seeds [0, 1, 2]
- prereg hash `f2934b473b1c`, frozen 1791071193514707000 ns; first result 1791071193549858000 ns
- report: `reports/smoke/E9a-structural-vs-refit-noisy-regime-f2934b473b-1791071195028556000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| full vs refit_only | final (decisive) | 3 | 0.8195 | 0.72 | 0.8889 |
| full vs refit_only | equal_interactions | 3 | 0.8195 | 0.72 | 0.8889 |
| full vs refit_only | equal_time | 0 | - | - | - |
| full vs full_no_split (ablation_arm) | final | 3 | -0.004455 | -0.02842 | 0.007525 |
| full vs no_update (alternative_arm) | final | 3 | 3.448 | 2.589 | 3.922 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| refit_only | 3/3 | 0 | 0 | 1.656 | 0.05455 |
| full | 3/3 | 0 | 0 | 0.837 | 0.03333 |
| full_no_split | 3/3 | 0 | 0 | 0.8325 | 0.03738 |
| no_update | 3/3 | 0 | 0 | 4.285 | 0.2985 |

Per-run outcomes (E9a):

| arm | seed | status | wall s | nll_robust | nll_gauss | incumbent_op | deps | n_distractor_deps | alarms | searches | candidates_evaluated | stopped | search_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0 | ok | 0.43 | 0.8155 | 0.8749 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 150 | candidates | 0.406 |
| full | 1 | ok | 0.23 | 0.82 | 0.8208 | SPLIT_APPLICABILITY | temp,x | 0 | 15 | 1 | 150 | candidates | 0.2054 |
| full | 2 | ok | 0.43 | 0.8754 | 0.8803 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 150 | candidates | 0.4027 |
| full_no_split | 0 | ok | 0.11 | 0.8155 | 0.8749 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 42 | exhausted | 0.09119 |
| full_no_split | 1 | ok | 0.08 | 0.8067 | 0.8062 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 27 | exhausted | 0.06046 |
| full_no_split | 2 | ok | 0.08 | 0.8754 | 0.8803 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 27 | exhausted | 0.05954 |
| no_update | 0 | ok | 0.01 | 3.944 | 6.964 | - | x | 0 | - | - | - | - | - |
| no_update | 1 | ok | 0.01 | 4.411 | 8.058 | - | x | 0 | - | - | - | - | - |
| no_update | 2 | ok | 0.01 | 4.5 | 8.781 | - | x | 0 | - | - | - | - | - |
| refit_only | 0 | ok | 0.03 | 1.598 | 1.598 | REFIT | x | 0 | 15 | 1 | 1 | exhausted | 0.003278 |
| refit_only | 1 | ok | 0.03 | 1.665 | 1.666 | REFIT | x | 0 | 15 | 1 | 1 | exhausted | 0.002579 |
| refit_only | 2 | ok | 0.03 | 1.706 | 1.707 | REFIT | x | 0 | 15 | 1 | 1 | exhausted | 0.002743 |

Per-arm means:

| arm | nll_robust | nll_gauss | incumbent_op | deps | n_distractor_deps | alarms | searches | candidates_evaluated | stopped | search_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| refit_only | 1.656 | 1.657 | - | - | 0 | 15 | 1 | 1 | - | 0.002867 |
| full | 0.837 | 0.8587 | - | - | 0 | 15 | 1 | 150 | - | 0.3381 |
| full_no_split | 0.8325 | 0.8538 | - | - | 0 | 15 | 1 | 32 | - | 0.0704 |
| no_update | 4.285 | 7.934 | - | - | 0 | - | - | - | - | - |

### E9b-control-slope-change-structure-hurts

- **verdict: reject** — CI upper bound 0 < delta 0.02: the hypothesised effect is ruled out
- hypothesis: CONTROL (falsification direction): after a slope-only change (2 -> 4), with noise and 7 distractors, REFIT-only beats the full language on held-out NLL by >= 0.02 nats/row, i.e. structure search invents structure that costs prediction
- primary `nll_robust` (lower better), delta 0.02, basis `final`; baseline `full`, candidate `refit_only`, ablation `no_update`, alternative `None`; seeds [0, 1, 2]
- prereg hash `4beccc7d0964`, frozen 1791071193515201000 ns; first result 1791071195164447000 ns
- report: `reports/smoke/E9b-control-slope-change-structure-hurts-4beccc7d09-1791071195804265000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| refit_only vs full | final (decisive) | 3 | 0 | 0 | 0 |
| refit_only vs full | equal_interactions | 3 | 0 | 0 | 0 |
| refit_only vs full | equal_time | 0 | - | - | - |
| refit_only vs no_update (ablation_arm) | final | 3 | 2.312 | 1.743 | 3.002 |

- note: criterion unmet: baseline 'full' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['full_slope3']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| full | 3/3 | 0 | 0 | 0.8349 | 0.09092 |
| refit_only | 3/3 | 0 | 0 | 0.8349 | 0.09092 |
| no_update | 3/3 | 0 | 0 | 3.147 | 0.1688 |
| full_slope3 | 3/3 | 0 | 0 | 1.24 | 0.448 |

Per-run outcomes (E9b):

| arm | seed | status | wall s | nll_robust | nll_gauss | incumbent_op | deps | alarms | searches | candidates_evaluated | stopped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0 | ok | 0.13 | 0.8661 | 0.8779 | REFIT | x | 15 | 1 | 74 | exhausted |
| full | 1 | ok | 0.14 | 0.7325 | 0.7265 | REFIT | x | 15 | 1 | 74 | exhausted |
| full | 2 | ok | 0.13 | 0.9061 | 0.9023 | REFIT | x | 14 | 1 | 74 | exhausted |
| full_slope3 | 0 | ok | 0.02 | 1.412 | 1.433 | root | x | 1 | 0 | 0 | - |
| full_slope3 | 1 | ok | 0.12 | 0.7314 | 0.7265 | REFIT | x | 6 | 1 | 74 | exhausted |
| full_slope3 | 2 | ok | 0.13 | 1.576 | 1.619 | root | x | 3 | 1 | 74 | exhausted |
| no_update | 0 | ok | 0.01 | 2.967 | 3.347 | - | x | - | - | - | - |
| no_update | 1 | ok | 0.01 | 3.301 | 4.079 | - | x | - | - | - | - |
| no_update | 2 | ok | 0.01 | 3.173 | 3.744 | - | x | - | - | - | - |
| refit_only | 0 | ok | 0.03 | 0.8661 | 0.8779 | REFIT | x | 15 | 1 | 1 | exhausted |
| refit_only | 1 | ok | 0.03 | 0.7325 | 0.7265 | REFIT | x | 15 | 1 | 1 | exhausted |
| refit_only | 2 | ok | 0.03 | 0.9061 | 0.9023 | REFIT | x | 14 | 1 | 1 | exhausted |

Per-arm means:

| arm | nll_robust | nll_gauss | incumbent_op | deps | alarms | searches | candidates_evaluated | stopped |
|---|---|---|---|---|---|---|---|---|
| full | 0.8349 | 0.8356 | - | - | 14.67 | 1 | 74 | - |
| refit_only | 0.8349 | 0.8356 | - | - | 14.67 | 1 | 1 | - |
| no_update | 3.147 | 3.724 | - | - | - | - | - | - |
| full_slope3 | 1.24 | 1.259 | - | - | 3.333 | 0.6667 | 49.33 | - |

### E9c-false-structure-on-pure-noise

- **verdict: reject** — CI upper bound 0 < delta 0.2: the hypothesised effect is ruled out
- hypothesis: on pure noise with 12 distractors and a forced search (10x20 rows), the full language gets a STRUCTURAL mechanism promoted through the registry in >= 20% of runs more than REFIT-only (which cannot)
- primary `false_struct_promotions` (higher better), delta 0.2, basis `final`; baseline `refit_only`, candidate `full`, ablation `full_no_split`, alternative `None`; seeds [0, 1, 2]
- prereg hash `57a65c18de3f`, frozen 1791071193515581000 ns; first result 1791071195823487000 ns
- report: `reports/smoke/E9c-false-structure-on-pure-noise-57a65c18de-1791071196382414000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| full vs refit_only | final (decisive) | 3 | 0 | 0 | 0 |
| full vs refit_only | equal_interactions | 3 | 0 | 0 | 0 |
| full vs refit_only | equal_time | 0 | - | - | - |
| full vs full_no_split (ablation_arm) | final | 3 | 0 | 0 | 0 |

- note: criterion unmet: baseline 'refit_only' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| refit_only | 3/3 | 0 | 0 | 0 | 0 |
| full | 3/3 | 0 | 0 | 0 | 0 |
| full_no_split | 3/3 | 0 | 0 | 0 | 0 |

Per-run outcomes (E9c):

| arm | seed | status | wall s | false_struct_promotions | struct_accepted | struct_accepted_ops | admitted | promotions | incumbent_op | deps | nll_gauss | evaluated | val_episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0 | ok | 0.13 | 0 | 0 | - | 0 | 0 | root |  | 1.375 | 97 | 3 |
| full | 1 | ok | 0.12 | 0 | 0 | - | 0 | 0 | root |  | 1.433 | 97 | 3 |
| full | 2 | ok | 0.12 | 0 | 0 | - | 0 | 0 | root |  | 1.366 | 97 | 3 |
| full_no_split | 0 | ok | 0.05 | 0 | 0 | - | 0 | 0 | root |  | 1.375 | 25 | 3 |
| full_no_split | 1 | ok | 0.06 | 0 | 0 | - | 0 | 0 | root |  | 1.433 | 25 | 3 |
| full_no_split | 2 | ok | 0.05 | 0 | 0 | - | 0 | 0 | root |  | 1.366 | 25 | 3 |
| refit_only | 0 | ok | 0.01 | 0 | 0 | - | 0 | 0 | root |  | 1.375 | 1 | 3 |
| refit_only | 1 | ok | 0.01 | 0 | 0 | - | 0 | 0 | root |  | 1.433 | 1 | 3 |
| refit_only | 2 | ok | 0.01 | 0 | 0 | - | 0 | 0 | root |  | 1.366 | 1 | 3 |

Per-arm means:

| arm | false_struct_promotions | struct_accepted | struct_accepted_ops | admitted | promotions | incumbent_op | deps | nll_gauss | evaluated | val_episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | 0 | - | 0 | 0 | - | - | 1.391 | 1 | 3 |
| full | 0 | 0 | - | 0 | 0 | - | - | 1.391 | 97 | 3 |
| full_no_split | 0 | 0 | - | 0 | 0 | - | - | 1.391 | 25 | 3 |

### E9d-rare-misfit-f0.03

- **verdict: accept** — mean improvement 3.963 >= delta 0.05, CI [0.6837, 7.591] excludes 0
- hypothesis: a change touching only 3% of rows (y += 3 when r == 1) is found by the full language: held-out GAUSSIAN NLL beats REFIT-only by >= 0.05 nats/row (if the 1% outlier component hides the misfit, this fails while the eps~0 ablation succeeds)
- primary `nll_gauss` (lower better), delta 0.05, basis `final`; baseline `refit_only`, candidate `full_robust`, ablation `full_eps0`, alternative `None`; seeds [0, 1, 2]
- prereg hash `6057d9abb9e5`, frozen 1791071193515965000 ns; first result 1791071196415118000 ns
- report: `reports/smoke/E9d-rare-misfit-f0.03-6057d9abb9-1791071197858074000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| full_robust vs refit_only | final (decisive) | 3 | 3.963 | 0.6837 | 7.591 |
| full_robust vs refit_only | equal_interactions | 3 | 3.963 | 0.6837 | 7.591 |
| full_robust vs refit_only | equal_time | 0 | - | - | - |
| full_robust vs full_eps0 (ablation_arm) | final | 3 | 0 | 0 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| refit_only | 3/3 | 0 | 0 | 3.761 | 1.32 |
| full_robust | 3/3 | 0 | 0 | -0.2018 | 0.0323 |
| full_eps0 | 3/3 | 0 | 0 | -0.2018 | 0.0323 |

Per-run outcomes (E9d):

| arm | seed | status | wall s | nll_gauss | nll_robust | nll_gauss_rare_rows | captured_r | best_op | deps | rare_rows_search | cusum_alarms_on_change | evaluated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_eps0 | 0 | ok | 0.21 | -0.1734 | -0.1678 | -0.2448 | True | SPLIT_APPLICABILITY | r,x | 19 | 19 | 123 |
| full_eps0 | 1 | ok | 0.27 | -0.2369 | -0.2301 | 0.3382 | True | SPLIT_APPLICABILITY | r,x | 19 | 19 | 126 |
| full_eps0 | 2 | ok | 0.17 | -0.1949 | -0.1888 | -0.1146 | True | REPLACE_TRANSITION | r,x | 19 | 19 | 92 |
| full_robust | 0 | ok | 0.27 | -0.1734 | -0.1678 | -0.2448 | True | SPLIT_APPLICABILITY | r,x | 19 | 19 | 123 |
| full_robust | 1 | ok | 0.29 | -0.2369 | -0.2301 | 0.3382 | True | SPLIT_APPLICABILITY | r,x | 19 | 19 | 126 |
| full_robust | 2 | ok | 0.17 | -0.1949 | -0.1888 | -0.1146 | True | REPLACE_TRANSITION | r,x | 19 | 19 | 92 |
| refit_only | 0 | ok | 0.03 | 5.139 | 0.2395 | 119.5 | False | none | x | 19 | 19 | 1 |
| refit_only | 1 | ok | 0.03 | 2.506 | -0.05152 | 124.1 | False | none | x | 19 | 19 | 1 |
| refit_only | 2 | ok | 0.03 | 3.638 | 0.1842 | 93.68 | False | none | x | 19 | 19 | 1 |

Per-arm means:

| arm | nll_gauss | nll_robust | nll_gauss_rare_rows | captured_r | best_op | deps | rare_rows_search | cusum_alarms_on_change | evaluated |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 3.761 | 0.1241 | 112.4 | 0 | - | - | 19 | 19 | 1 |
| full_robust | -0.2018 | -0.1956 | -0.007057 | 1 | - | - | 19 | 19 | 113.7 |
| full_eps0 | -0.2018 | -0.1956 | -0.007057 | 1 | - | - | 19 | 19 | 113.7 |

### E9e-rare-misfit-f0.01

- **verdict: accept** — mean improvement 0.9029 >= delta 0.05, CI [0.3225, 1.949] excludes 0
- hypothesis: a change touching only 1% of rows (y += 3 when r == 1) is found by the full language: held-out GAUSSIAN NLL beats REFIT-only by >= 0.05 nats/row (if the 1% outlier component hides the misfit, this fails while the eps~0 ablation succeeds)
- primary `nll_gauss` (lower better), delta 0.05, basis `final`; baseline `refit_only`, candidate `full_robust`, ablation `full_eps0`, alternative `None`; seeds [0, 1, 2]
- prereg hash `1a0c1c6a001d`, frozen 1791071193516331000 ns; first result 1791071197893961000 ns
- report: `reports/smoke/E9e-rare-misfit-f0.01-1a0c1c6a00-1791071199174233000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| full_robust vs refit_only | final (decisive) | 3 | 0.9029 | 0.3225 | 1.949 |
| full_robust vs refit_only | equal_interactions | 3 | 0.9029 | 0.3225 | 1.949 |
| full_robust vs refit_only | equal_time | 0 | - | - | - |
| full_robust vs full_eps0 (ablation_arm) | final | 3 | 0 | 0 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| refit_only | 3/3 | 0 | 0 | 0.7063 | 0.3548 |
| full_robust | 3/3 | 0 | 0 | -0.1966 | 0.03193 |
| full_eps0 | 3/3 | 0 | 0 | -0.1966 | 0.03193 |

Per-run outcomes (E9e):

| arm | seed | status | wall s | nll_gauss | nll_robust | nll_gauss_rare_rows | captured_r | best_op | deps | rare_rows_search | cusum_alarms_on_change | evaluated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_eps0 | 0 | ok | 0.14 | -0.1805 | -0.176 | -0.4031 | True | ADD_DEPENDENCY | r,x | 10 | 10 | 65 |
| full_eps0 | 1 | ok | 0.26 | -0.176 | -0.1722 | -0.5343 | True | ADD_DEPENDENCY | r,x | 5 | 5 | 131 |
| full_eps0 | 2 | ok | 0.26 | -0.2334 | -0.2276 | -0.5666 | True | ADD_DEPENDENCY | r,x | 4 | 4 | 129 |
| full_robust | 0 | ok | 0.16 | -0.1805 | -0.176 | -0.4031 | True | ADD_DEPENDENCY | r,x | 10 | 10 | 65 |
| full_robust | 1 | ok | 0.14 | -0.176 | -0.1722 | -0.5343 | True | ADD_DEPENDENCY | r,x | 5 | 5 | 65 |
| full_robust | 2 | ok | 0.27 | -0.2334 | -0.2276 | -0.5666 | True | ADD_DEPENDENCY | r,x | 4 | 4 | 129 |
| refit_only | 0 | ok | 0.03 | 0.5066 | -0.1231 | 113.7 | False | none | x | 10 | 10 | 1 |
| refit_only | 1 | ok | 0.03 | 1.116 | -0.06661 | 106.9 | False | none | x | 5 | 5 | 1 |
| refit_only | 2 | ok | 0.03 | 0.4964 | -0.1682 | 119.8 | False | none | x | 4 | 4 | 1 |

Per-arm means:

| arm | nll_gauss | nll_robust | nll_gauss_rare_rows | captured_r | best_op | deps | rare_rows_search | cusum_alarms_on_change | evaluated |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0.7063 | -0.1193 | 113.5 | 0 | - | - | 6.333 | 6.333 | 1 |
| full_robust | -0.1966 | -0.1919 | -0.5013 | 1 | - | - | 6.333 | 6.333 | 86.33 |
| full_eps0 | -0.1966 | -0.1919 | -0.5013 | 1 | - | - | 6.333 | 6.333 | 108.3 |

### E9f-search-wall-budget-overshoot

- **verdict: accept** — mean improvement 17.63 >= delta 0.5, CI [16.07, 19.4] excludes 0
- hypothesis: search() under wall_s = 0.25 s overshoots its budget by > 50% (elapsed/wall_s - small-table ratio >= 0.5) on a large table (150 variables x 60000 rows) — work outside the guarded fits is unbudgeted
- primary `overshoot_ratio` (higher better), delta 0.5, basis `final`; baseline `small_table`, candidate `large_table`, ablation `hung_fitter`, alternative `None`; seeds [0, 1, 2]
- prereg hash `80010a907a6f`, frozen 1791071193521481000 ns; first result 1791071199299241000 ns
- report: `reports/smoke/E9f-search-wall-budget-overshoot-80010a907a-1791071214598742000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| large_table vs small_table | final (decisive) | 3 | 17.63 | 16.07 | 19.4 |
| large_table vs small_table | equal_interactions | 0 | - | - | - |
| large_table vs small_table | equal_time | 0 | - | - | - |
| large_table vs hung_fitter (ablation_arm) | final | 3 | 17.07 | 15.54 | 18.83 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| small_table | 3/3 | 0 | 0 | 0.4578 | 0.007381 |
| large_table | 3/3 | 0 | 0 | 18.09 | 0.6157 |
| hung_fitter | 3/3 | 0 | 0 | 1.017 | 0.00493 |

Per-run outcomes (E9f):

| arm | seed | status | wall s | overshoot_ratio | elapsed | internal_elapsed | rows | vars | evaluated | stopped | leaked_threads |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hung_fitter | 0 | ok | 0.25 | 1.012 | 0.2531 | 0.2531 | 2000 | 5 | 1 | wall | 1 |
| hung_fitter | 1 | ok | 0.26 | 1.016 | 0.254 | 0.2539 | 2000 | 5 | 1 | wall | 1 |
| hung_fitter | 2 | ok | 0.26 | 1.022 | 0.2556 | 0.2555 | 2000 | 5 | 1 | wall | 1 |
| large_table | 0 | ok | 4.62 | 17.51 | 4.379 | 4.374 | 60000 | 150 | 1 | wall | 0 |
| large_table | 1 | ok | 4.93 | 18.74 | 4.685 | 4.678 | 60000 | 150 | 1 | wall | -1 |
| large_table | 2 | ok | 4.75 | 18 | 4.501 | 4.496 | 60000 | 150 | 1 | wall | -2 |
| small_table | 0 | ok | 0.12 | 0.4662 | 0.1165 | 0.1164 | 2000 | 5 | 51 | exhausted | 0 |
| small_table | 1 | ok | 0.11 | 0.4523 | 0.1131 | 0.1129 | 2000 | 5 | 51 | exhausted | 0 |
| small_table | 2 | ok | 0.12 | 0.4549 | 0.1137 | 0.1136 | 2000 | 5 | 51 | exhausted | 0 |

Per-arm means:

| arm | overshoot_ratio | elapsed | internal_elapsed | rows | vars | evaluated | stopped | leaked_threads |
|---|---|---|---|---|---|---|---|---|
| small_table | 0.4578 | 0.1144 | 0.1143 | 2000 | 5 | 51 | - | 0 |
| large_table | 18.09 | 4.521 | 4.516 | 6e+04 | 150 | 1 | - | -1 |
| hung_fitter | 1.017 | 0.2542 | 0.2542 | 2000 | 5 | 1 | - | 1 |

## E10 (7s wall)

### E10a-replication-K8-noise0.5

- **verdict: accept** — mean improvement 16.17 >= delta 1, CI [0.4816, 39.47] excludes 0
- hypothesis: through the library's own fixture path with NEW parameters (K=8, reward noise 0.5, budget 80), info-gain identifies the hidden shift in >= 1 fewer real interactions than random
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `random`, candidate `info_gain`, ablation `entropy`, alternative `icm`; seeds [0, 1, 2]
- prereg hash `d0137fe49f43`, frozen 1791071214608765000 ns; first result 1791071214678487000 ns
- report: `reports/smoke/E10a-replication-K8-noise0.5-d0137fe49f-1791071219004537000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_gain vs random | final (decisive) | 3 | 16.17 | 0.4816 | 39.47 |
| info_gain vs random | equal_interactions | 0 | - | - | - |
| info_gain vs random | equal_time | 0 | - | - | - |
| info_gain vs entropy (ablation_arm) | final | 3 | 67.61 | 50.28 | 83.89 |
| info_gain vs icm (alternative_arm) | final | 3 | 17.33 | 0.752 | 40.64 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['lp']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_gain | 3/3 | 0 | 0 | 13.39 | 6.259 |
| random | 3/3 | 0 | 0 | 29.56 | 5.412 |
| icm | 3/3 | 0 | 0 | 30.72 | 2.761 |
| entropy | 3/3 | 0 | 0 | 81 | 0 |
| lp | 3/3 | 0 | 0 | 38.67 | 16.5 |

Per-run outcomes (E10a):

| arm | seed | status | wall s | interactions_to_identify | censored | tv_fraction | map_correct | realized_nats_per_interaction |
|---|---|---|---|---|---|---|---|---|
| entropy | 0 | ok | 0.61 | 81 | 6 | 0.9542 | 0.3333 | 0.004729 |
| entropy | 1 | ok | 0.57 | 81 | 6 | 0.9646 | 0.1667 | 0.004129 |
| entropy | 2 | ok | 0.53 | 81 | 6 | 0.9542 | 0.1667 | 0.005191 |
| icm | 0 | ok | 0.28 | 33.33 | 0 | 0.1564 | 1 | 0.0764 |
| icm | 1 | ok | 0.26 | 31 | 0 | 0.1067 | 1 | 0.08371 |
| icm | 2 | ok | 0.20 | 27.83 | 0 | 0.07257 | 1 | 0.07518 |
| info_gain | 0 | ok | 0.07 | 7.333 | 0 | 0 | 1 | 0.1477 |
| info_gain | 1 | ok | 0.14 | 19.83 | 0 | 0 | 1 | 0.1696 |
| info_gain | 2 | ok | 0.09 | 13 | 0 | 0 | 1 | 0.174 |
| lp | 0 | ok | 0.39 | 38.67 | 2 | 0.222 | 0.8333 | 0.07834 |
| lp | 1 | ok | 0.41 | 55.17 | 3 | 0.2934 | 0.8333 | 0.06288 |
| lp | 2 | ok | 0.15 | 22.17 | 0 | 0.09194 | 1 | 0.08307 |
| random | 0 | ok | 0.26 | 32.17 | 0 | 0.1312 | 1 | 0.07299 |
| random | 1 | ok | 0.27 | 33.17 | 0 | 0.09615 | 1 | 0.09135 |
| random | 2 | ok | 0.16 | 23.33 | 0 | 0.1365 | 1 | 0.08293 |

Per-arm means:

| arm | interactions_to_identify | censored | tv_fraction | map_correct | realized_nats_per_interaction |
|---|---|---|---|---|---|
| info_gain | 13.39 | 0 | 0 | 1 | 0.1638 |
| random | 29.56 | 0 | 0.1213 | 1 | 0.08242 |
| icm | 30.72 | 0 | 0.1119 | 1 | 0.07843 |
| entropy | 81 | 6 | 0.9576 | 0.2222 | 0.004683 |
| lp | 38.67 | 1.667 | 0.2024 | 0.8889 | 0.07476 |

### E10b-four-noisy-TVs

- **verdict: inconclusive** — CI [-5.495, 26.23] with mean 8.667 neither meets nor rules out delta 1
- hypothesis: with FOUR coin-flip TVs (K=6, noise 0.35, budget 60), info-gain identifies the shift in >= 1 fewer interactions than random
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `random`, candidate `info_joint`, ablation `entropy`, alternative `icm`; seeds [0, 1, 2]
- prereg hash `dffe2f028454`, frozen 1791071214609095000 ns; first result 1791071219046915000 ns
- report: `reports/smoke/E10b-four-noisy-TVs-dffe2f0284-1791071220574669000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_joint vs random | final (decisive) | 3 | 8.667 | -5.495 | 26.23 |
| info_joint vs random | equal_interactions | 0 | - | - | - |
| info_joint vs random | equal_time | 0 | - | - | - |
| info_joint vs entropy (ablation_arm) | final | 3 | 54.8 | 49.96 | 58.03 |
| info_joint vs icm (alternative_arm) | final | 3 | 11 | 0.2445 | 18.53 |

- note: criterion unmet: baseline 'random' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['lp']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_joint | 3/3 | 0 | 0 | 6.2 | 1.587 |
| random | 3/3 | 0 | 0 | 14.87 | 4.688 |
| icm | 3/3 | 0 | 0 | 17.2 | 4.045 |
| entropy | 3/3 | 0 | 0 | 61 | 0 |
| lp | 3/3 | 0 | 0 | 21 | 14.23 |

Per-run outcomes (E10b):

| arm | seed | status | wall s | interactions_to_identify | censored | tv_fraction |
|---|---|---|---|---|---|---|
| entropy | 0 | ok | 0.36 | 61 | 1 | 0.97 |
| entropy | 1 | ok | 0.40 | 61 | 1 | 0.98 |
| entropy | 2 | ok | 0.36 | 61 | 1 | 0.97 |
| icm | 0 | ok | 0.02 | 12.6 | 0 | 0.3326 |
| icm | 1 | ok | 0.04 | 18.8 | 0 | 0.4905 |
| icm | 2 | ok | 0.04 | 20.2 | 0 | 0.4771 |
| info_joint | 0 | ok | 0.04 | 5.6 | 0 | 0 |
| info_joint | 1 | ok | 0.04 | 5 | 0 | 0.02857 |
| info_joint | 2 | ok | 0.05 | 8 | 0 | 0.01333 |
| lp | 0 | ok | 0.03 | 13.6 | 0 | 0.3268 |
| lp | 1 | ok | 0.03 | 12 | 0 | 0.2943 |
| lp | 2 | ok | 0.07 | 37.4 | 0.4 | 0.5876 |
| random | 0 | ok | 0.02 | 13 | 0 | 0.3539 |
| random | 1 | ok | 0.04 | 20.2 | 0 | 0.4589 |
| random | 2 | ok | 0.02 | 11.4 | 0 | 0.3449 |

Per-arm means:

| arm | interactions_to_identify | censored | tv_fraction |
|---|---|---|---|
| info_joint | 6.2 | 0 | 0.01397 |
| random | 14.87 | 0 | 0.3859 |
| icm | 17.2 | 0 | 0.4334 |
| entropy | 61 | 1 | 0.9733 |
| lp | 21 | 0.1333 | 0.4029 |

### E10c-learnable-useless-distractor

- **verdict: accept** — mean improvement 3.667 >= delta 1, CI [1.874, 4.563] excludes 0
- hypothesis: with a deterministic 4-bit lamp the task does not need (joint shift x lamp hypotheses), info-gain on the JOINT set wastes interactions on the lamp: task-marginal info-gain identifies the shift in >= 1 fewer interactions
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `info_joint`, candidate `info_marginal`, ablation `random`, alternative `icm`; seeds [0, 1, 2]
- prereg hash `8ba4186318bc`, frozen 1791071214609369000 ns; first result 1791071220627117000 ns
- report: `reports/smoke/E10c-learnable-useless-distractor-8ba4186318-1791071221098374000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_marginal vs info_joint | final (decisive) | 3 | 3.667 | 1.874 | 4.563 |
| info_marginal vs info_joint | equal_interactions | 0 | - | - | - |
| info_marginal vs info_joint | equal_time | 0 | - | - | - |
| info_marginal vs random (ablation_arm) | final | 3 | 16.44 | 4.494 | 23.32 |
| info_marginal vs icm (alternative_arm) | final | 3 | 10.33 | -6.696 | 20.19 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_joint | 3/3 | 0 | 0 | 9.667 | 1.764 |
| info_marginal | 3/3 | 0 | 0 | 6 | 2.333 |
| random | 3/3 | 0 | 0 | 22.44 | 5.059 |
| icm | 3/3 | 0 | 0 | 16.33 | 6.566 |

Per-run outcomes (E10c):

| arm | seed | status | wall s | interactions_to_identify | censored | lamp_fraction | tv_fraction |
|---|---|---|---|---|---|---|---|
| icm | 0 | ok | 0.03 | 18.33 | 0 | 0.2917 | 0.1389 |
| icm | 1 | ok | 0.04 | 21.67 | 0 | 0.2873 | 0.119 |
| icm | 2 | ok | 0.03 | 9 | 0 | 0.2992 | 0.04167 |
| info_joint | 0 | ok | 0.05 | 8.333 | 0 | 0.5404 | 0 |
| info_joint | 1 | ok | 0.06 | 11.67 | 0 | 0.3805 | 0 |
| info_joint | 2 | ok | 0.06 | 9 | 0 | 0.4598 | 0 |
| info_marginal | 0 | ok | 0.04 | 4.333 | 0 | 0.08333 | 0 |
| info_marginal | 1 | ok | 0.07 | 8.667 | 0 | 0 | 0 |
| info_marginal | 2 | ok | 0.05 | 5 | 0 | 0.04762 | 0 |
| random | 0 | ok | 0.03 | 23.33 | 0 | 0.3773 | 0.1499 |
| random | 1 | ok | 0.04 | 27 | 0 | 0.3651 | 0.08995 |
| random | 2 | ok | 0.03 | 17 | 0 | 0.3389 | 0.06944 |

Per-arm means:

| arm | interactions_to_identify | censored | lamp_fraction | tv_fraction |
|---|---|---|---|---|
| info_joint | 9.667 | 0 | 0.4602 | 0 |
| info_marginal | 6 | 0 | 0.04365 | 0 |
| random | 22.44 | 0 | 0.3604 | 0.1031 |
| icm | 16.33 | 0 | 0.2927 | 0.09987 |

### E10d-misspecified-hypothesis-set

- **verdict: accept** — mean improvement 1 >= delta 0.3, CI [1, 1] excludes 0
- hypothesis: when the truth is NOT in the hypothesis set, info-gain selection reaches > 0.9 posterior on a (necessarily wrong) shift with no flag in >= 30 percentage points more scenarios than it reaches confident-wrong when the set is well specified
- primary `confident_wrong_unflagged` (higher better), delta 0.3, basis `final`; baseline `info_wellspec`, candidate `info_misspec`, ablation `random_misspec`, alternative `info_ppc_misspec`; seeds [0, 1, 2]
- prereg hash `75eed1e85ba4`, frozen 1791071214609608000 ns; first result 1791071221132033000 ns
- report: `reports/smoke/E10d-misspecified-hypothesis-set-75eed1e85b-1791071221446268000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_misspec vs info_wellspec | final (decisive) | 3 | 1 | 1 | 1 |
| info_misspec vs info_wellspec | equal_interactions | 0 | - | - | - |
| info_misspec vs info_wellspec | equal_time | 0 | - | - | - |
| info_misspec vs random_misspec (ablation_arm) | final | 3 | 0 | 0 | 0 |
| info_misspec vs info_ppc_misspec (alternative_arm) | final | 3 | 0 | 0 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_wellspec | 3/3 | 0 | 0 | 0 | 0 |
| info_misspec | 3/3 | 0 | 0 | 1 | 0 |
| random_misspec | 3/3 | 0 | 0 | 1 | 0 |
| info_ppc_misspec | 3/3 | 0 | 0 | 1 | 0 |

Per-run outcomes (E10d):

| arm | seed | status | wall s | confident_wrong_unflagged | confident | time_to_confidence | ppc_flagged | interactions_to_identify |
|---|---|---|---|---|---|---|---|---|
| info_misspec | 0 | ok | 0.03 | 1 | 1 | 9 | 0 | 61 |
| info_misspec | 1 | ok | 0.02 | 1 | 1 | 5.667 | 0 | 61 |
| info_misspec | 2 | ok | 0.05 | 1 | 1 | 17 | 0 | 61 |
| info_ppc_misspec | 0 | ok | 0.03 | 1 | 1 | 9 | 0 | 61 |
| info_ppc_misspec | 1 | ok | 0.02 | 1 | 1 | 5.667 | 0 | 61 |
| info_ppc_misspec | 2 | ok | 0.05 | 1 | 1 | 17 | 0 | 61 |
| info_wellspec | 0 | ok | 0.03 | 0 | 1 | 6.667 | 0 | 6.667 |
| info_wellspec | 1 | ok | 0.02 | 0 | 1 | 6 | 0 | 6 |
| info_wellspec | 2 | ok | 0.04 | 0 | 1 | 10 | 0 | 10 |
| random_misspec | 0 | ok | 0.02 | 1 | 1 | 13.33 | 0 | 61 |
| random_misspec | 1 | ok | 0.01 | 1 | 1 | 12.33 | 0.3333 | 61 |
| random_misspec | 2 | ok | 0.01 | 1 | 1 | 14 | 0.3333 | 61 |

Per-arm means:

| arm | confident_wrong_unflagged | confident | time_to_confidence | ppc_flagged | interactions_to_identify |
|---|---|---|---|---|---|
| info_wellspec | 0 | 1 | 7.556 | 0 | 7.556 |
| info_misspec | 1 | 1 | 10.56 | 0 | 61 |
| random_misspec | 1 | 1 | 13.22 | 0.2222 | 61 |
| info_ppc_misspec | 1 | 1 | 10.56 | 0 | 61 |

## E10e idle / stationary-probe retention tracker (measurement, not an A/B)

RetentionTracker(known_below=0.45, forgotten_above=0.55); 16 probes with i.i.d. N(center, sd) losses, 300 evaluations, seeds [0].

| config | seed | net progress | telescoped first-last | raw (clipped LP) gain | relearn events | flagged probes /16 |
|---|---|---|---|---|---|---|
| straddle 0.5+-0.2 | 0 | -443.2 | -1.27 | 532.8 | 969 | 16 |
| wide 0.3+-0.2 | 0 | -189.5 | -1.27 | 532.8 | 442 | 8 |
| far 0.2+-0.05 | 0 | -0.3 | -0.32 | 133.2 | 0 | 0 |
| straddle annealed | 0 | -886.3 | -2.97 | 1069.4 | 1057 | 16 |

## E11 (10s wall)

### E11a-gate-vs-ungated-rare-region

- **verdict: inconclusive** — CI [-0.2036, 0.6138] with mean 0.1035 neither meets nor rules out delta 0.02
- hypothesis: with a model wrong only inside a rarely visited disc (claims boost, truth is mud), the reliability gate lowers real mean distance to the goal versus the ungated planner by >= 0.02 m
- primary `mean_dist` (lower better), delta 0.02, basis `final`; baseline `ungated`, candidate `gated`, ablation `fallback`, alternative `gated_true`; seeds [0, 1, 2]
- prereg hash `e1d72c942e2b`, frozen 1791071221513619000 ns; first result 1791071221535548000 ns
- report: `reports/smoke/E11a-gate-vs-ungated-rare-region-e1d72c942e-1791071227035463000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| gated vs ungated | final (decisive) | 3 | 0.1035 | -0.2036 | 0.6138 |
| gated vs ungated | equal_interactions | 3 | 0.1035 | -0.2036 | 0.6138 |
| gated vs ungated | equal_time | 0 | - | - | - |
| gated vs fallback (ablation_arm) | final | 3 | 0.03234 | -0.2551 | 0.3819 |
| gated vs gated_true (alternative_arm) | final | 3 | -0.1933 | -0.464 | 0.2606 |

- note: criterion unmet: baseline 'ungated' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| fallback | 3/3 | 0 | 0 | 1.219 | 0.4273 |
| gated | 3/3 | 0 | 0 | 1.187 | 0.4787 |
| ungated | 3/3 | 0 | 0 | 1.291 | 0.4663 |
| gated_true | 3/3 | 0 | 0 | 0.9939 | 0.361 |

Per-run outcomes (E11a):

| arm | seed | status | wall s | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|---|---|---|
| fallback | 0 | ok | 0.02 | 1.04 | 0.05556 | 0 | 0 | 0 | 0 |
| fallback | 1 | ok | 0.02 | 0.9115 | 0.2 | 0 | 0 | 0 | 0 |
| fallback | 2 | ok | 0.02 | 1.707 | 0.7778 | 0 | 0 | 0 | 0 |
| gated | 0 | ok | 0.46 | 1.114 | 0.2667 | 1 | 1 | 1 | 0.6111 |
| gated | 1 | ok | 0.51 | 0.7491 | 0.2333 | 1 | 1 | 1 | 0.7556 |
| gated | 2 | ok | 0.10 | 1.698 | 0.7889 | 1 | 1 | 1 | 0.05556 |
| gated_true | 0 | ok | 0.75 | 0.8528 | 0 | 0 | 0 | 0 | 0.9667 |
| gated_true | 1 | ok | 0.77 | 0.7246 | 0.2222 | 19 | 1 | 0 | 0.9667 |
| gated_true | 2 | ok | 0.82 | 1.404 | 0.5222 | 45 | 2 | 0 | 0.9667 |
| ungated | 0 | ok | 0.69 | 1.407 | 0.5556 | 48 | 2 | 0 | 0.9889 |
| ungated | 1 | ok | 0.66 | 0.777 | 0.2444 | 21 | 1 | 0 | 0.9889 |
| ungated | 2 | ok | 0.69 | 1.687 | 0.7889 | 68 | 3 | 0 | 0.9889 |

Per-arm means:

| arm | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|
| fallback | 1.219 | 0.3444 | 0 | 0 | 0 | 0 |
| gated | 1.187 | 0.4296 | 1 | 1 | 1 | 0.4741 |
| ungated | 1.291 | 0.5296 | 45.67 | 2 | 0 | 0.9889 |
| gated_true | 0.9939 | 0.2481 | 21.33 | 1 | 0 | 0.9667 |

### E11b-exploitation-leaks-through-gate

- **verdict: inconclusive** — CI [-0.3819, 0.2551] with mean -0.03234 neither meets nor rules out delta 0.01
- hypothesis: FALSIFICATION of 'the gate prevents exploitation': the planner keeps re-entering the region the model is wrong about after each gate re-opening, so the fallback ALONE achieves lower real mean distance than the gated planner by >= 0.01 m
- primary `mean_dist` (lower better), delta 0.01, basis `final`; baseline `gated`, candidate `fallback`, ablation `ungated`, alternative `None`; seeds [0, 1, 2]
- prereg hash `19ce1737491e`, frozen 1791071221513793000 ns; first result 1791071227041643000 ns
- report: `reports/smoke/E11b-exploitation-leaks-through-gate-19ce173749-1791071227042157000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| fallback vs gated | final (decisive) | 3 | -0.03234 | -0.3819 | 0.2551 |
| fallback vs gated | equal_interactions | 3 | -0.03234 | -0.3819 | 0.2551 |
| fallback vs gated | equal_time | 0 | - | - | - |
| fallback vs ungated (ablation_arm) | final | 3 | 0.07114 | -0.4817 | 0.8689 |

- note: criterion unmet: baseline 'gated' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['gated_true']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| fallback | 3/3 | 0 | 0 | 1.219 | 0.4273 |
| gated | 3/3 | 0 | 0 | 1.187 | 0.4787 |
| ungated | 3/3 | 0 | 0 | 1.291 | 0.4663 |
| gated_true | 3/3 | 0 | 0 | 0.9939 | 0.361 |

Per-run outcomes (E11b):

| arm | seed | status | wall s | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|---|---|---|
| fallback | 0 | ok | 0.00 | 1.04 | 0.05556 | 0 | 0 | 0 | 0 |
| fallback | 1 | ok | 0.00 | 0.9115 | 0.2 | 0 | 0 | 0 | 0 |
| fallback | 2 | ok | 0.00 | 1.707 | 0.7778 | 0 | 0 | 0 | 0 |
| gated | 0 | ok | 0.00 | 1.114 | 0.2667 | 1 | 1 | 1 | 0.6111 |
| gated | 1 | ok | 0.00 | 0.7491 | 0.2333 | 1 | 1 | 1 | 0.7556 |
| gated | 2 | ok | 0.00 | 1.698 | 0.7889 | 1 | 1 | 1 | 0.05556 |
| gated_true | 0 | ok | 0.00 | 0.8528 | 0 | 0 | 0 | 0 | 0.9667 |
| gated_true | 1 | ok | 0.00 | 0.7246 | 0.2222 | 19 | 1 | 0 | 0.9667 |
| gated_true | 2 | ok | 0.00 | 1.404 | 0.5222 | 45 | 2 | 0 | 0.9667 |
| ungated | 0 | ok | 0.00 | 1.407 | 0.5556 | 48 | 2 | 0 | 0.9889 |
| ungated | 1 | ok | 0.00 | 0.777 | 0.2444 | 21 | 1 | 0 | 0.9889 |
| ungated | 2 | ok | 0.00 | 1.687 | 0.7889 | 68 | 3 | 0 | 0.9889 |

Per-arm means:

| arm | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|
| fallback | 1.219 | 0.3444 | 0 | 0 | 0 | 0 |
| gated | 1.187 | 0.4296 | 1 | 1 | 1 | 0.4741 |
| ungated | 1.291 | 0.5296 | 45.67 | 2 | 0 | 0.9889 |
| gated_true | 0.9939 | 0.2481 | 21.33 | 1 | 0 | 0.9667 |

### E11c-deadline-heavy-tailed-model

- **verdict: inconclusive** — CI [-0.02037, 0.1813] with mean 0.09167 neither meets nor rules out delta 0.01
- hypothesis: with a model whose call latency is heavy-tailed (2% of calls take 30 ms) under a 10 ms deadline, >= 1% more decisions take over TWICE the deadline than with a constant-latency model of equal mean: the 'hard' deadline is bounded only by one chunk's worst case
- primary `overrun_fraction` (higher better), delta 0.01, basis `final`; baseline `constant`, candidate `heavy_tail`, ablation `heavy_tail_chunk8`, alternative `None`; seeds [0, 1, 2]
- prereg hash `871e48de0634`, frozen 1791071221513998000 ns; first result 1791071227277324000 ns
- report: `reports/smoke/E11c-deadline-heavy-tailed-model-871e48de06-1791071229543975000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| heavy_tail vs constant | final (decisive) | 3 | 0.09167 | -0.02037 | 0.1813 |
| heavy_tail vs constant | equal_interactions | 3 | 0.09167 | -0.02037 | 0.1813 |
| heavy_tail vs constant | equal_time | 0 | - | - | - |
| heavy_tail vs heavy_tail_chunk8 (ablation_arm) | final | 3 | -0.05833 | -0.1704 | 0.0313 |

- note: criterion unmet: baseline 'constant' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| constant | 3/3 | 0 | 0 | 0 | 0 |
| heavy_tail | 3/3 | 0 | 0 | 0.09167 | 0.03819 |
| heavy_tail_chunk8 | 3/3 | 0 | 0 | 0.15 | 0 |

Per-run outcomes (E11c):

| arm | seed | status | wall s | overrun_fraction | late_fraction | max_latency_over_deadline | p99_latency_ms | median_latency_ms | planner_fraction | timeout_fraction | probes | late_chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constant | 0 | ok | 0.23 | 0 | 0.075 | 1.025 | 10.16 | 5.282 | 1 | 0 | 0 | 1 |
| constant | 1 | ok | 0.39 | 0 | 0 | 0.9982 | 9.981 | 9.742 | 1 | 0 | 0 | 0 |
| constant | 2 | ok | 0.25 | 0 | 0 | 0.8395 | 8.144 | 5.929 | 1 | 0 | 0 | 0 |
| heavy_tail | 0 | ok | 0.24 | 0.125 | 0.125 | 4.292 | 42.76 | 0.1389 | 0.325 | 0.675 | 4 | 5 |
| heavy_tail | 1 | ok | 0.15 | 0.05 | 0.05 | 3.588 | 35.1 | 2.515 | 0.7 | 0.3 | 2 | 2 |
| heavy_tail | 2 | ok | 0.22 | 0.1 | 0.1 | 4.375 | 43.15 | 1.489 | 0.425 | 0.575 | 4 | 4 |
| heavy_tail_chunk8 | 0 | ok | 0.33 | 0.15 | 0.175 | 6.747 | 60.45 | 0.1052 | 0.2 | 0.8 | 6 | 6 |
| heavy_tail_chunk8 | 1 | ok | 0.34 | 0.15 | 0.175 | 4.547 | 45.22 | 0.1254 | 0.325 | 0.675 | 5 | 6 |
| heavy_tail_chunk8 | 2 | ok | 0.36 | 0.15 | 0.2 | 7.563 | 65.81 | 0.1415 | 0.3 | 0.7 | 5 | 7 |

Per-arm means:

| arm | overrun_fraction | late_fraction | max_latency_over_deadline | p99_latency_ms | median_latency_ms | planner_fraction | timeout_fraction | probes | late_chunks |
|---|---|---|---|---|---|---|---|---|---|
| constant | 0 | 0.025 | 0.9543 | 9.429 | 6.984 | 1 | 0 | 0 | 0.3333 |
| heavy_tail | 0.09167 | 0.09167 | 4.085 | 40.34 | 1.381 | 0.4833 | 0.5167 | 3.333 | 3.667 |
| heavy_tail_chunk8 | 0.15 | 0.1833 | 6.285 | 57.16 | 0.124 | 0.275 | 0.725 | 5.333 | 6.333 |

### E11d-offprobe-false-duplicate

- **verdict: accept** — mean improvement 0.8889 >= delta 0.5, CI [0.2914, 1.188] excludes 0
- hypothesis: two skills that differ only outside the probe distribution (argmax permuted when any |s_i| > 7) are flagged duplicates at a rate >= 0.5 above that of an unrelated same-name policy
- primary `flag_rate` (higher better), delta 0.5, basis `final`; baseline `samename_diff`, candidate `offprobe`, ablation `name_only`, alternative `None`; seeds [0, 1, 2]
- prereg hash `67081e559060`, frozen 1791071221514202000 ns; first result 1791071229640897000 ns
- report: `reports/smoke/E11d-offprobe-false-duplicate-67081e5590-1791071231627503000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| offprobe vs samename_diff | final (decisive) | 3 | 0.8889 | 0.2914 | 1.188 |
| offprobe vs samename_diff | equal_interactions | 3 | 0.8889 | 0.2914 | 1.188 |
| offprobe vs samename_diff | equal_time | 0 | - | - | - |
| offprobe vs name_only (ablation_arm) | final | 3 | -0.1111 | -0.7086 | 0.1877 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['drift_0.001', 'drift_0.01', 'drift_0.1', 'tempered']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| name_only | 3/3 | 0 | 0 | 1 | 0 |
| samename_diff | 3/3 | 0 | 0 | 0 | 0 |
| offprobe | 3/3 | 0 | 0 | 0.8889 | 0.1925 |
| tempered | 3/3 | 0 | 0 | 0 | 0 |
| drift_0.001 | 3/3 | 0 | 0 | 1 | 0 |
| drift_0.01 | 3/3 | 0 | 0 | 1 | 0 |
| drift_0.1 | 3/3 | 0 | 0 | 0 | 0 |

Per-run outcomes (E11d):

| arm | seed | status | wall s | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|---|---|---|
| drift_0.001 | 0 | ok | 0.09 | 1 | 3.107e-07 | 4.349e-07 | 0.0008333 | 0 |
| drift_0.001 | 1 | ok | 0.09 | 1 | 3.018e-07 | 4.404e-07 | 0.002083 | 0.3333 |
| drift_0.001 | 2 | ok | 0.10 | 1 | 5.096e-07 | 8.661e-07 | 0.0004167 | 0 |
| drift_0.01 | 0 | ok | 0.09 | 1 | 3.108e-05 | 4.353e-05 | 0.008333 | 0 |
| drift_0.01 | 1 | ok | 0.09 | 1 | 3.02e-05 | 4.447e-05 | 0.009167 | 0.3333 |
| drift_0.01 | 2 | ok | 0.10 | 1 | 5.128e-05 | 8.746e-05 | 0.00875 | 0 |
| drift_0.1 | 0 | ok | 0.10 | 0 | 0.003101 | 0.004346 | 0.08167 | 0 |
| drift_0.1 | 1 | ok | 0.09 | 0 | 0.003042 | 0.004784 | 0.08208 | 0.3333 |
| drift_0.1 | 2 | ok | 0.14 | 0 | 0.005401 | 0.009433 | 0.0975 | 0 |
| name_only | 0 | ok | 0.09 | 1 | 0 | 0 | 0 | 0 |
| name_only | 1 | ok | 0.09 | 1 | 0 | 0 | 0 | 0.3333 |
| name_only | 2 | ok | 0.10 | 1 | 0 | 0 | 0 | 0 |
| offprobe | 0 | ok | 0.10 | 1 | 0 | 0 | 0.3004 | 0 |
| offprobe | 1 | ok | 0.11 | 0.6667 | 0.00546 | 0.019 | 0.2896 | 0.3333 |
| offprobe | 2 | ok | 0.10 | 1 | 0 | 0 | 0.3033 | 0 |
| samename_diff | 0 | ok | 0.14 | 0 | 0.2929 | 0.4124 | 0.7496 | 0 |
| samename_diff | 1 | ok | 0.09 | 0 | 0.2881 | 0.3757 | 0.6804 | 0.3333 |
| samename_diff | 2 | ok | 0.10 | 0 | 0.3174 | 0.3917 | 0.805 | 0 |
| tempered | 0 | ok | 0.09 | 0 | 0.09824 | 0.114 | 0 | 0 |
| tempered | 1 | ok | 0.10 | 0 | 0.103 | 0.1075 | 0 | 0.3333 |
| tempered | 2 | ok | 0.09 | 0 | 0.1043 | 0.1212 | 0 | 0 |

Per-arm means:

| arm | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|
| name_only | 1 | 0 | 0 | 0 | 0.1111 |
| samename_diff | 0 | 0.2995 | 0.3933 | 0.745 | 0.1111 |
| offprobe | 0.8889 | 0.00182 | 0.006333 | 0.2978 | 0.1111 |
| tempered | 0 | 0.1019 | 0.1142 | 0 | 0.1111 |
| drift_0.001 | 1 | 3.74e-07 | 5.805e-07 | 0.001111 | 0.1111 |
| drift_0.01 | 1 | 3.752e-05 | 5.849e-05 | 0.00875 | 0.1111 |
| drift_0.1 | 0 | 0.003848 | 0.006188 | 0.08708 | 0.1111 |

### E11e-tempered-copy-not-flagged

- **verdict: accept** — mean improvement 1 >= delta 0.5, CI [1, 1] excludes 0
- hypothesis: a copy with logits x3 (identical argmax on every state, so identical deterministic behaviour) is flagged at a rate >= 0.5 BELOW a byte copy
- primary `flag_rate` (lower better), delta 0.5, basis `final`; baseline `name_only`, candidate `tempered`, ablation `samename_diff`, alternative `None`; seeds [0, 1, 2]
- prereg hash `62d96beef8b6`, frozen 1791071221514298000 ns; first result 1791071231634043000 ns
- report: `reports/smoke/E11e-tempered-copy-not-flagged-62d96beef8-1791071231634622000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| tempered vs name_only | final (decisive) | 3 | 1 | 1 | 1 |
| tempered vs name_only | equal_interactions | 3 | 1 | 1 | 1 |
| tempered vs name_only | equal_time | 0 | - | - | - |
| tempered vs samename_diff (ablation_arm) | final | 3 | 0 | 0 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['drift_0.001', 'drift_0.01', 'drift_0.1', 'offprobe']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| name_only | 3/3 | 0 | 0 | 1 | 0 |
| samename_diff | 3/3 | 0 | 0 | 0 | 0 |
| offprobe | 3/3 | 0 | 0 | 0.8889 | 0.1925 |
| tempered | 3/3 | 0 | 0 | 0 | 0 |
| drift_0.001 | 3/3 | 0 | 0 | 1 | 0 |
| drift_0.01 | 3/3 | 0 | 0 | 1 | 0 |
| drift_0.1 | 3/3 | 0 | 0 | 0 | 0 |

Per-run outcomes (E11e):

| arm | seed | status | wall s | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|---|---|---|
| drift_0.001 | 0 | ok | 0.00 | 1 | 3.107e-07 | 4.349e-07 | 0.0008333 | 0 |
| drift_0.001 | 1 | ok | 0.00 | 1 | 3.018e-07 | 4.404e-07 | 0.002083 | 0.3333 |
| drift_0.001 | 2 | ok | 0.00 | 1 | 5.096e-07 | 8.661e-07 | 0.0004167 | 0 |
| drift_0.01 | 0 | ok | 0.00 | 1 | 3.108e-05 | 4.353e-05 | 0.008333 | 0 |
| drift_0.01 | 1 | ok | 0.00 | 1 | 3.02e-05 | 4.447e-05 | 0.009167 | 0.3333 |
| drift_0.01 | 2 | ok | 0.00 | 1 | 5.128e-05 | 8.746e-05 | 0.00875 | 0 |
| drift_0.1 | 0 | ok | 0.00 | 0 | 0.003101 | 0.004346 | 0.08167 | 0 |
| drift_0.1 | 1 | ok | 0.00 | 0 | 0.003042 | 0.004784 | 0.08208 | 0.3333 |
| drift_0.1 | 2 | ok | 0.00 | 0 | 0.005401 | 0.009433 | 0.0975 | 0 |
| name_only | 0 | ok | 0.00 | 1 | 0 | 0 | 0 | 0 |
| name_only | 1 | ok | 0.00 | 1 | 0 | 0 | 0 | 0.3333 |
| name_only | 2 | ok | 0.00 | 1 | 0 | 0 | 0 | 0 |
| offprobe | 0 | ok | 0.00 | 1 | 0 | 0 | 0.3004 | 0 |
| offprobe | 1 | ok | 0.00 | 0.6667 | 0.00546 | 0.019 | 0.2896 | 0.3333 |
| offprobe | 2 | ok | 0.00 | 1 | 0 | 0 | 0.3033 | 0 |
| samename_diff | 0 | ok | 0.00 | 0 | 0.2929 | 0.4124 | 0.7496 | 0 |
| samename_diff | 1 | ok | 0.00 | 0 | 0.2881 | 0.3757 | 0.6804 | 0.3333 |
| samename_diff | 2 | ok | 0.00 | 0 | 0.3174 | 0.3917 | 0.805 | 0 |
| tempered | 0 | ok | 0.00 | 0 | 0.09824 | 0.114 | 0 | 0 |
| tempered | 1 | ok | 0.00 | 0 | 0.103 | 0.1075 | 0 | 0.3333 |
| tempered | 2 | ok | 0.00 | 0 | 0.1043 | 0.1212 | 0 | 0 |

Per-arm means:

| arm | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|
| name_only | 1 | 0 | 0 | 0 | 0.1111 |
| samename_diff | 0 | 0.2995 | 0.3933 | 0.745 | 0.1111 |
| offprobe | 0.8889 | 0.00182 | 0.006333 | 0.2978 | 0.1111 |
| tempered | 0 | 0.1019 | 0.1142 | 0 | 0.1111 |
| drift_0.001 | 1 | 3.74e-07 | 5.805e-07 | 0.001111 | 0.1111 |
| drift_0.01 | 1 | 3.752e-05 | 5.849e-05 | 0.00875 | 0.1111 |
| drift_0.1 | 0 | 0.003848 | 0.006188 | 0.08708 | 0.1111 |


Total wall 38s
