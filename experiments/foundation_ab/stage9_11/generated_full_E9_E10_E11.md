## E9 (35s wall)

### E9a-structural-vs-refit-noisy-regime

- **verdict: accept** — mean improvement 0.7923 >= delta 0.1, CI [0.7067, 0.879] excludes 0
- hypothesis: after a structural rule change (threshold 0.3 on temp), on NEW noisier (sd 0.5), smaller (16x20 post rows) data with 7 distractors, the full proposal language yields lower held-out NLL than REFIT-only under the same SearchBudget (implementers report 1.82 nats/row on their fixture)
- primary `nll_robust` (lower better), delta 0.1, basis `final`; baseline `refit_only`, candidate `full`, ablation `full_no_split`, alternative `no_update`; seeds [0, 1, 2, 3, 4]
- prereg hash `2c627817b956`, frozen 1791071242878311000 ns; first result 1791071242913831000 ns
- report: `reports/full/E9a-structural-vs-refit-noisy-regime-2c627817b9-1791071245625191000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| full vs refit_only | final (decisive) | 5 | 0.7923 | 0.7067 | 0.879 |
| full vs refit_only | equal_interactions | 5 | 0.7923 | 0.7067 | 0.879 |
| full vs refit_only | equal_time | 0 | - | - | - |
| full vs full_no_split (ablation_arm) | final | 5 | 5.924e-05 | -3.457e-05 | 0.0002468 |
| full vs no_update (alternative_arm) | final | 5 | 3.537 | 3.099 | 3.862 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| refit_only | 5/5 | 0 | 0 | 1.652 | 0.03075 |
| full | 5/5 | 0 | 0 | 0.8596 | 0.08581 |
| full_no_split | 5/5 | 0 | 0 | 0.8597 | 0.08588 |
| no_update | 5/5 | 0 | 0 | 4.397 | 0.3078 |

Per-run outcomes (E9a):

| arm | seed | status | wall s | nll_robust | nll_gauss | incumbent_op | deps | n_distractor_deps | alarms | searches | candidates_evaluated | stopped | search_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0 | ok | 0.47 | 0.8448 | 0.8931 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 150 | candidates | 0.439 |
| full | 1 | ok | 0.24 | 0.9398 | 0.9775 | SPLIT_APPLICABILITY | temp,x | 0 | 15 | 1 | 150 | candidates | 0.2109 |
| full | 2 | ok | 0.45 | 0.8437 | 0.8499 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 150 | candidates | 0.4242 |
| full | 3 | ok | 0.45 | 0.7317 | 0.8268 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 150 | candidates | 0.4234 |
| full | 4 | ok | 0.45 | 0.9383 | 1.192 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 150 | candidates | 0.4233 |
| full_no_split | 0 | ok | 0.12 | 0.8448 | 0.8931 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 42 | exhausted | 0.09483 |
| full_no_split | 1 | ok | 0.09 | 0.9401 | 1.03 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 27 | exhausted | 0.06413 |
| full_no_split | 2 | ok | 0.09 | 0.8437 | 0.8499 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 27 | exhausted | 0.06623 |
| full_no_split | 3 | ok | 0.09 | 0.7317 | 0.8268 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 27 | exhausted | 0.06614 |
| full_no_split | 4 | ok | 0.08 | 0.9383 | 1.192 | REPLACE_TRANSITION | temp,x | 0 | 15 | 1 | 27 | exhausted | 0.06031 |
| no_update | 0 | ok | 0.01 | 3.875 | 6.809 | - | x | 0 | - | - | - | - | - |
| no_update | 1 | ok | 0.01 | 4.647 | 8.493 | - | x | 0 | - | - | - | - | - |
| no_update | 2 | ok | 0.01 | 4.58 | 9.15 | - | x | 0 | - | - | - | - | - |
| no_update | 3 | ok | 0.01 | 4.497 | 10.25 | - | x | 0 | - | - | - | - | - |
| no_update | 4 | ok | 0.01 | 4.384 | 9.012 | - | x | 0 | - | - | - | - | - |
| refit_only | 0 | ok | 0.03 | 1.626 | 1.626 | REFIT | x | 0 | 15 | 1 | 1 | exhausted | 0.003253 |
| refit_only | 1 | ok | 0.03 | 1.664 | 1.665 | REFIT | x | 0 | 15 | 1 | 1 | exhausted | 0.002771 |
| refit_only | 2 | ok | 0.03 | 1.688 | 1.689 | REFIT | x | 0 | 15 | 1 | 1 | exhausted | 0.002918 |
| refit_only | 3 | ok | 0.03 | 1.614 | 1.615 | REFIT | x | 0 | 15 | 1 | 1 | exhausted | 0.002708 |
| refit_only | 4 | ok | 0.03 | 1.667 | 1.667 | REFIT | x | 0 | 15 | 1 | 1 | exhausted | 0.00277 |

Per-arm means:

| arm | nll_robust | nll_gauss | incumbent_op | deps | n_distractor_deps | alarms | searches | candidates_evaluated | stopped | search_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| refit_only | 1.652 | 1.652 | - | - | 0 | 15 | 1 | 1 | - | 0.002884 |
| full | 0.8596 | 0.9479 | - | - | 0 | 15 | 1 | 150 | - | 0.3842 |
| full_no_split | 0.8597 | 0.9584 | - | - | 0 | 15 | 1 | 30 | - | 0.07033 |
| no_update | 4.397 | 8.743 | - | - | 0 | - | - | - | - | - |

### E9b-control-slope-change-structure-hurts

- **verdict: reject** — CI upper bound 0 < delta 0.02: the hypothesised effect is ruled out
- hypothesis: CONTROL (falsification direction): after a slope-only change (2 -> 4), with noise and 7 distractors, REFIT-only beats the full language on held-out NLL by >= 0.02 nats/row, i.e. structure search invents structure that costs prediction
- primary `nll_robust` (lower better), delta 0.02, basis `final`; baseline `full`, candidate `refit_only`, ablation `no_update`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `290c087734d1`, frozen 1791071242878878000 ns; first result 1791071245748807000 ns
- report: `reports/full/E9b-control-slope-change-structure-hurts-290c087734-1791071246808992000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| refit_only vs full | final (decisive) | 5 | 0 | 0 | 0 |
| refit_only vs full | equal_interactions | 5 | 0 | 0 | 0 |
| refit_only vs full | equal_time | 0 | - | - | - |
| refit_only vs no_update (ablation_arm) | final | 5 | 2.213 | 1.926 | 2.559 |

- note: criterion unmet: baseline 'full' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['full_slope3']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| full | 5/5 | 0 | 0 | 0.8001 | 0.05931 |
| refit_only | 5/5 | 0 | 0 | 0.8001 | 0.05931 |
| no_update | 5/5 | 0 | 0 | 3.013 | 0.2667 |
| full_slope3 | 5/5 | 0 | 0 | 1.259 | 0.3047 |

Per-run outcomes (E9b):

| arm | seed | status | wall s | nll_robust | nll_gauss | incumbent_op | deps | alarms | searches | candidates_evaluated | stopped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0 | ok | 0.11 | 0.8321 | 0.8412 | REFIT | x | 15 | 1 | 74 | exhausted |
| full | 1 | ok | 0.12 | 0.7485 | 0.7428 | REFIT | x | 15 | 1 | 74 | exhausted |
| full | 2 | ok | 0.11 | 0.8858 | 0.8817 | REFIT | x | 14 | 1 | 74 | exhausted |
| full | 3 | ok | 0.12 | 0.7464 | 0.7469 | REFIT | x | 15 | 1 | 73 | exhausted |
| full | 4 | ok | 0.12 | 0.7876 | 0.7921 | REFIT | x | 15 | 1 | 74 | exhausted |
| full_slope3 | 0 | ok | 0.02 | 1.41 | 1.43 | root | x | 1 | 0 | 0 | - |
| full_slope3 | 1 | ok | 0.12 | 0.7474 | 0.7428 | REFIT | x | 6 | 1 | 74 | exhausted |
| full_slope3 | 2 | ok | 0.11 | 1.541 | 1.578 | root | x | 3 | 1 | 74 | exhausted |
| full_slope3 | 3 | ok | 0.12 | 1.254 | 1.272 | root | x | 1 | 1 | 74 | exhausted |
| full_slope3 | 4 | ok | 0.03 | 1.344 | 1.388 | root | x | 0 | 0 | 0 | - |
| no_update | 0 | ok | 0.01 | 2.973 | 3.371 | - | x | - | - | - | - |
| no_update | 1 | ok | 0.01 | 3.385 | 4.277 | - | x | - | - | - | - |
| no_update | 2 | ok | 0.01 | 3.163 | 3.687 | - | x | - | - | - | - |
| no_update | 3 | ok | 0.01 | 2.717 | 3.055 | - | x | - | - | - | - |
| no_update | 4 | ok | 0.01 | 2.826 | 3.32 | - | x | - | - | - | - |
| refit_only | 0 | ok | 0.03 | 0.8321 | 0.8412 | REFIT | x | 15 | 1 | 1 | exhausted |
| refit_only | 1 | ok | 0.03 | 0.7485 | 0.7428 | REFIT | x | 15 | 1 | 1 | exhausted |
| refit_only | 2 | ok | 0.03 | 0.8858 | 0.8817 | REFIT | x | 14 | 1 | 1 | exhausted |
| refit_only | 3 | ok | 0.03 | 0.7464 | 0.7469 | REFIT | x | 15 | 1 | 1 | exhausted |
| refit_only | 4 | ok | 0.03 | 0.7876 | 0.7921 | REFIT | x | 15 | 1 | 1 | exhausted |

Per-arm means:

| arm | nll_robust | nll_gauss | incumbent_op | deps | alarms | searches | candidates_evaluated | stopped |
|---|---|---|---|---|---|---|---|---|
| full | 0.8001 | 0.8009 | - | - | 14.8 | 1 | 73.8 | - |
| refit_only | 0.8001 | 0.8009 | - | - | 14.8 | 1 | 1 | - |
| no_update | 3.013 | 3.542 | - | - | - | - | - | - |
| full_slope3 | 1.259 | 1.282 | - | - | 2.2 | 0.6 | 44.4 | - |

### E9c-false-structure-on-pure-noise

- **verdict: reject** — CI upper bound 0 < delta 0.2: the hypothesised effect is ruled out
- hypothesis: on pure noise with 12 distractors and a forced search (10x20 rows), the full language gets a STRUCTURAL mechanism promoted through the registry in >= 20% of runs more than REFIT-only (which cannot)
- primary `false_struct_promotions` (higher better), delta 0.2, basis `final`; baseline `refit_only`, candidate `full`, ablation `full_no_split`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `485697446257`, frozen 1791071242879275000 ns; first result 1791071246828685000 ns
- report: `reports/full/E9c-false-structure-on-pure-noise-4856974462-1791071247934708000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| full vs refit_only | final (decisive) | 5 | 0 | 0 | 0 |
| full vs refit_only | equal_interactions | 5 | 0 | 0 | 0 |
| full vs refit_only | equal_time | 0 | - | - | - |
| full vs full_no_split (ablation_arm) | final | 5 | 0 | 0 | 0 |

- note: criterion unmet: baseline 'refit_only' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| refit_only | 5/5 | 0 | 0 | 0 | 0 |
| full | 5/5 | 0 | 0 | 0 | 0 |
| full_no_split | 5/5 | 0 | 0 | 0 | 0 |

Per-run outcomes (E9c):

| arm | seed | status | wall s | false_struct_promotions | struct_accepted | struct_accepted_ops | admitted | promotions | incumbent_op | deps | nll_gauss | evaluated | val_episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0 | ok | 0.24 | 0 | 0 | - | 0 | 0 | root |  | 1.375 | 97 | 3 |
| full | 1 | ok | 0.14 | 0 | 0 | - | 0 | 0 | root |  | 1.433 | 97 | 3 |
| full | 2 | ok | 0.13 | 0 | 0 | - | 0 | 0 | root |  | 1.366 | 97 | 3 |
| full | 3 | ok | 0.13 | 0 | 0 | - | 1 | 0 | root |  | 1.433 | 97 | 3 |
| full | 4 | ok | 0.13 | 0 | 0 | - | 0 | 0 | root |  | 1.483 | 97 | 3 |
| full_no_split | 0 | ok | 0.07 | 0 | 0 | - | 0 | 0 | root |  | 1.375 | 25 | 3 |
| full_no_split | 1 | ok | 0.06 | 0 | 0 | - | 0 | 0 | root |  | 1.433 | 25 | 3 |
| full_no_split | 2 | ok | 0.06 | 0 | 0 | - | 0 | 0 | root |  | 1.366 | 25 | 3 |
| full_no_split | 3 | ok | 0.06 | 0 | 0 | - | 1 | 0 | root |  | 1.433 | 25 | 3 |
| full_no_split | 4 | ok | 0.06 | 0 | 0 | - | 0 | 0 | root |  | 1.483 | 25 | 3 |
| refit_only | 0 | ok | 0.01 | 0 | 0 | - | 0 | 0 | root |  | 1.375 | 1 | 3 |
| refit_only | 1 | ok | 0.01 | 0 | 0 | - | 0 | 0 | root |  | 1.433 | 1 | 3 |
| refit_only | 2 | ok | 0.01 | 0 | 0 | - | 0 | 0 | root |  | 1.366 | 1 | 3 |
| refit_only | 3 | ok | 0.01 | 0 | 0 | - | 1 | 0 | root |  | 1.433 | 1 | 3 |
| refit_only | 4 | ok | 0.01 | 0 | 0 | - | 0 | 0 | root |  | 1.483 | 1 | 3 |

Per-arm means:

| arm | false_struct_promotions | struct_accepted | struct_accepted_ops | admitted | promotions | incumbent_op | deps | nll_gauss | evaluated | val_episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0 | 0 | - | 0.2 | 0 | - | - | 1.418 | 1 | 3 |
| full | 0 | 0 | - | 0.2 | 0 | - | - | 1.418 | 97 | 3 |
| full_no_split | 0 | 0 | - | 0.2 | 0 | - | - | 1.418 | 25 | 3 |

### E9d-rare-misfit-f0.03

- **verdict: accept** — mean improvement 4.243 >= delta 0.05, CI [2.527, 6.016] excludes 0
- hypothesis: a change touching only 3% of rows (y += 3 when r == 1) is found by the full language: held-out GAUSSIAN NLL beats REFIT-only by >= 0.05 nats/row (if the 1% outlier component hides the misfit, this fails while the eps~0 ablation succeeds)
- primary `nll_gauss` (lower better), delta 0.05, basis `final`; baseline `refit_only`, candidate `full_robust`, ablation `full_eps0`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `e273ed0227d1`, frozen 1791071242879671000 ns; first result 1791071247969762000 ns
- report: `reports/full/E9d-rare-misfit-f0.03-e273ed0227-1791071250093687000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| full_robust vs refit_only | final (decisive) | 5 | 4.243 | 2.527 | 6.016 |
| full_robust vs refit_only | equal_interactions | 5 | 4.243 | 2.527 | 6.016 |
| full_robust vs refit_only | equal_time | 0 | - | - | - |
| full_robust vs full_eps0 (ablation_arm) | final | 5 | 0 | 0 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| refit_only | 5/5 | 0 | 0 | 4.07 | 1.445 |
| full_robust | 5/5 | 0 | 0 | -0.1733 | 0.04673 |
| full_eps0 | 5/5 | 0 | 0 | -0.1733 | 0.04673 |

Per-run outcomes (E9d):

| arm | seed | status | wall s | nll_gauss | nll_robust | nll_gauss_rare_rows | captured_r | best_op | deps | rare_rows_search | cusum_alarms_on_change | evaluated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_eps0 | 0 | ok | 0.20 | -0.1734 | -0.1678 | -0.2448 | True | SPLIT_APPLICABILITY | r,x | 19 | 19 | 123 |
| full_eps0 | 1 | ok | 0.24 | -0.2369 | -0.2301 | 0.3382 | True | SPLIT_APPLICABILITY | r,x | 19 | 19 | 126 |
| full_eps0 | 2 | ok | 0.18 | -0.1949 | -0.1888 | -0.1146 | True | REPLACE_TRANSITION | r,x | 19 | 19 | 92 |
| full_eps0 | 3 | ok | 0.17 | -0.1476 | -0.1426 | -0.2923 | True | ADD_DEPENDENCY | r,x | 19 | 19 | 92 |
| full_eps0 | 4 | ok | 0.23 | -0.1136 | -0.1079 | -0.0007466 | True | ADD_DEPENDENCY | r,x | 24 | 24 | 126 |
| full_robust | 0 | ok | 0.20 | -0.1734 | -0.1678 | -0.2448 | True | SPLIT_APPLICABILITY | r,x | 19 | 19 | 123 |
| full_robust | 1 | ok | 0.24 | -0.2369 | -0.2301 | 0.3382 | True | SPLIT_APPLICABILITY | r,x | 19 | 19 | 126 |
| full_robust | 2 | ok | 0.16 | -0.1949 | -0.1888 | -0.1146 | True | REPLACE_TRANSITION | r,x | 19 | 19 | 92 |
| full_robust | 3 | ok | 0.16 | -0.1476 | -0.1426 | -0.2923 | True | ADD_DEPENDENCY | r,x | 19 | 19 | 92 |
| full_robust | 4 | ok | 0.23 | -0.1136 | -0.1079 | -0.0007466 | True | ADD_DEPENDENCY | r,x | 24 | 24 | 126 |
| refit_only | 0 | ok | 0.03 | 5.139 | 0.2395 | 119.5 | False | none | x | 19 | 19 | 1 |
| refit_only | 1 | ok | 0.03 | 2.506 | -0.05152 | 124.1 | False | none | x | 19 | 19 | 1 |
| refit_only | 2 | ok | 0.03 | 3.638 | 0.1842 | 93.68 | False | none | x | 19 | 19 | 1 |
| refit_only | 3 | ok | 0.03 | 3.092 | 0.1256 | 109.1 | False | none | x | 19 | 19 | 1 |
| refit_only | 4 | ok | 0.03 | 5.973 | 0.4684 | 96.59 | False | none | x | 24 | 24 | 1 |

Per-arm means:

| arm | nll_gauss | nll_robust | nll_gauss_rare_rows | captured_r | best_op | deps | rare_rows_search | cusum_alarms_on_change | evaluated |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 4.07 | 0.1933 | 108.6 | 0 | - | - | 20 | 20 | 1 |
| full_robust | -0.1733 | -0.1674 | -0.06284 | 1 | - | - | 20 | 20 | 111.8 |
| full_eps0 | -0.1733 | -0.1674 | -0.06284 | 1 | - | - | 20 | 20 | 111.8 |

### E9e-rare-misfit-f0.01

- **verdict: accept** — mean improvement 0.7821 >= delta 0.05, CI [0.153, 1.371] excludes 0
- hypothesis: a change touching only 1% of rows (y += 3 when r == 1) is found by the full language: held-out GAUSSIAN NLL beats REFIT-only by >= 0.05 nats/row (if the 1% outlier component hides the misfit, this fails while the eps~0 ablation succeeds)
- primary `nll_gauss` (lower better), delta 0.05, basis `final`; baseline `refit_only`, candidate `full_robust`, ablation `full_eps0`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `05e07b9d78cc`, frozen 1791071242880045000 ns; first result 1791071250127431000 ns
- report: `reports/full/E9e-rare-misfit-f0.01-05e07b9d78-1791071252175062000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| full_robust vs refit_only | final (decisive) | 5 | 0.7821 | 0.153 | 1.371 |
| full_robust vs refit_only | equal_interactions | 5 | 0.7821 | 0.153 | 1.371 |
| full_robust vs refit_only | equal_time | 0 | - | - | - |
| full_robust vs full_eps0 (ablation_arm) | final | 5 | 0 | 0 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| refit_only | 5/5 | 0 | 0 | 0.5916 | 0.5485 |
| full_robust | 5/5 | 0 | 0 | -0.1905 | 0.04186 |
| full_eps0 | 5/5 | 0 | 0 | -0.1905 | 0.04186 |

Per-run outcomes (E9e):

| arm | seed | status | wall s | nll_gauss | nll_robust | nll_gauss_rare_rows | captured_r | best_op | deps | rare_rows_search | cusum_alarms_on_change | evaluated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_eps0 | 0 | ok | 0.13 | -0.1805 | -0.176 | -0.4031 | True | ADD_DEPENDENCY | r,x | 10 | 10 | 65 |
| full_eps0 | 1 | ok | 0.25 | -0.176 | -0.1722 | -0.5343 | True | ADD_DEPENDENCY | r,x | 5 | 5 | 131 |
| full_eps0 | 2 | ok | 0.26 | -0.2334 | -0.2276 | -0.5666 | True | ADD_DEPENDENCY | r,x | 4 | 4 | 129 |
| full_eps0 | 3 | ok | 0.22 | -0.1329 | -0.1316 | 1.564 | True | SPLIT_APPLICABILITY | r,x | 7 | 7 | 126 |
| full_eps0 | 4 | ok | 0.14 | -0.2297 | -0.2233 | 0 | True | ADD_DEPENDENCY | r,x | 8 | 8 | 65 |
| full_robust | 0 | ok | 0.14 | -0.1805 | -0.176 | -0.4031 | True | ADD_DEPENDENCY | r,x | 10 | 10 | 65 |
| full_robust | 1 | ok | 0.13 | -0.176 | -0.1722 | -0.5343 | True | ADD_DEPENDENCY | r,x | 5 | 5 | 65 |
| full_robust | 2 | ok | 0.23 | -0.2334 | -0.2276 | -0.5666 | True | ADD_DEPENDENCY | r,x | 4 | 4 | 129 |
| full_robust | 3 | ok | 0.23 | -0.1329 | -0.1316 | 1.564 | True | SPLIT_APPLICABILITY | r,x | 7 | 7 | 126 |
| full_robust | 4 | ok | 0.20 | -0.2297 | -0.2233 | 0 | True | ADD_DEPENDENCY | r,x | 8 | 8 | 65 |
| refit_only | 0 | ok | 0.03 | 0.5066 | -0.1231 | 113.7 | False | none | x | 10 | 10 | 1 |
| refit_only | 1 | ok | 0.03 | 1.116 | -0.06661 | 106.9 | False | none | x | 5 | 5 | 1 |
| refit_only | 2 | ok | 0.03 | 0.4964 | -0.1682 | 119.8 | False | none | x | 4 | 4 | 1 |
| refit_only | 3 | ok | 0.03 | 1.073 | -0.04199 | 101.4 | False | none | x | 7 | 7 | 1 |
| refit_only | 4 | ok | 0.03 | -0.2335 | -0.2281 | 0 | False | none | x | 8 | 8 | 1 |

Per-arm means:

| arm | nll_gauss | nll_robust | nll_gauss_rare_rows | captured_r | best_op | deps | rare_rows_search | cusum_alarms_on_change | evaluated |
|---|---|---|---|---|---|---|---|---|---|
| refit_only | 0.5916 | -0.1256 | 88.36 | 0 | - | - | 6.8 | 6.8 | 1 |
| full_robust | -0.1905 | -0.1861 | 0.01199 | 1 | - | - | 6.8 | 6.8 | 90 |
| full_eps0 | -0.1905 | -0.1861 | 0.01199 | 1 | - | - | 6.8 | 6.8 | 103.2 |

### E9f-search-wall-budget-overshoot

- **verdict: accept** — mean improvement 17.33 >= delta 0.5, CI [16.15, 18.43] excludes 0
- hypothesis: search() under wall_s = 0.25 s overshoots its budget by > 50% (elapsed/wall_s - small-table ratio >= 0.5) on a large table (150 variables x 60000 rows) — work outside the guarded fits is unbudgeted
- primary `overshoot_ratio` (higher better), delta 0.5, basis `final`; baseline `small_table`, candidate `large_table`, ablation `hung_fitter`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `3acd1592f6c9`, frozen 1791071242885185000 ns; first result 1791071252301990000 ns
- report: `reports/full/E9f-search-wall-budget-overshoot-3acd1592f6-1791071277567846000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| large_table vs small_table | final (decisive) | 5 | 17.33 | 16.15 | 18.43 |
| large_table vs small_table | equal_interactions | 0 | - | - | - |
| large_table vs small_table | equal_time | 0 | - | - | - |
| large_table vs hung_fitter (ablation_arm) | final | 5 | 16.75 | 15.44 | 17.85 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| small_table | 5/5 | 0 | 0 | 0.4589 | 0.1243 |
| large_table | 5/5 | 0 | 0 | 17.79 | 0.9535 |
| hung_fitter | 5/5 | 0 | 0 | 1.041 | 0.001297 |

Per-run outcomes (E9f):

| arm | seed | status | wall s | overshoot_ratio | elapsed | internal_elapsed | rows | vars | evaluated | stopped | leaked_threads |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hung_fitter | 0 | ok | 0.26 | 1.041 | 0.2603 | 0.2602 | 2000 | 5 | 1 | wall | 1 |
| hung_fitter | 1 | ok | 0.26 | 1.041 | 0.2602 | 0.2601 | 2000 | 5 | 1 | wall | 1 |
| hung_fitter | 2 | ok | 0.26 | 1.044 | 0.2609 | 0.2609 | 2000 | 5 | 1 | wall | 1 |
| hung_fitter | 3 | ok | 0.26 | 1.041 | 0.2602 | 0.2602 | 2000 | 5 | 1 | wall | 1 |
| hung_fitter | 4 | ok | 0.26 | 1.041 | 0.2602 | 0.2602 | 2000 | 5 | 1 | wall | 1 |
| large_table | 0 | ok | 4.55 | 17.24 | 4.309 | 4.303 | 60000 | 150 | 1 | wall | 0 |
| large_table | 1 | ok | 4.83 | 18.39 | 4.598 | 4.592 | 60000 | 150 | 1 | wall | 0 |
| large_table | 2 | ok | 4.92 | 18.72 | 4.68 | 4.672 | 60000 | 150 | 1 | wall | -2 |
| large_table | 3 | ok | 4.86 | 18.2 | 4.55 | 4.546 | 60000 | 150 | 1 | wall | 0 |
| large_table | 4 | ok | 4.32 | 16.4 | 4.1 | 4.095 | 60000 | 150 | 1 | wall | 0 |
| small_table | 0 | ok | 0.12 | 0.4653 | 0.1163 | 0.1162 | 2000 | 5 | 51 | exhausted | 0 |
| small_table | 1 | ok | 0.09 | 0.3665 | 0.09162 | 0.09153 | 2000 | 5 | 51 | exhausted | 0 |
| small_table | 2 | ok | 0.11 | 0.4429 | 0.1107 | 0.1106 | 2000 | 5 | 51 | exhausted | 0 |
| small_table | 3 | ok | 0.17 | 0.6645 | 0.1661 | 0.166 | 2000 | 5 | 51 | exhausted | 0 |
| small_table | 4 | ok | 0.09 | 0.3554 | 0.08884 | 0.08877 | 2000 | 5 | 51 | exhausted | 0 |

Per-arm means:

| arm | overshoot_ratio | elapsed | internal_elapsed | rows | vars | evaluated | stopped | leaked_threads |
|---|---|---|---|---|---|---|---|---|
| small_table | 0.4589 | 0.1147 | 0.1146 | 2000 | 5 | 51 | - | 0 |
| large_table | 17.79 | 4.447 | 4.442 | 6e+04 | 150 | 1 | - | -0.4 |
| hung_fitter | 1.041 | 0.2604 | 0.2603 | 2000 | 5 | 1 | - | 1 |

## E10 (32s wall)

### E10a-replication-K8-noise0.5

- **verdict: accept** — mean improvement 17.66 >= delta 1, CI [12.64, 22.95] excludes 0
- hypothesis: through the library's own fixture path with NEW parameters (K=8, reward noise 0.5, budget 80), info-gain identifies the hidden shift in >= 1 fewer real interactions than random
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `random`, candidate `info_gain`, ablation `entropy`, alternative `icm`; seeds [0, 1, 2, 3, 4]
- prereg hash `07a678f4385c`, frozen 1791071277577572000 ns; first result 1791071277844781000 ns
- report: `reports/full/E10a-replication-K8-noise0.5-07a678f438-1791071299431152000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_gain vs random | final (decisive) | 5 | 17.66 | 12.64 | 22.95 |
| info_gain vs random | equal_interactions | 0 | - | - | - |
| info_gain vs random | equal_time | 0 | - | - | - |
| info_gain vs entropy (ablation_arm) | final | 5 | 69.74 | 68.46 | 71.03 |
| info_gain vs icm (alternative_arm) | final | 5 | 17.6 | 8.943 | 23.32 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['lp']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_gain | 5/5 | 0 | 0 | 11.26 | 1.032 |
| random | 5/5 | 0 | 0 | 28.91 | 4.621 |
| icm | 5/5 | 0 | 0 | 28.86 | 6.388 |
| entropy | 5/5 | 0 | 0 | 81 | 0 |
| lp | 5/5 | 0 | 0 | 37.87 | 12.85 |

Per-run outcomes (E10a):

| arm | seed | status | wall s | interactions_to_identify | censored | tv_fraction | map_correct | realized_nats_per_interaction |
|---|---|---|---|---|---|---|---|---|
| entropy | 0 | ok | 1.92 | 81 | 18 | 0.9549 | 0.1667 | 0.004808 |
| entropy | 1 | ok | 1.72 | 81 | 18 | 0.9479 | 0.4444 | 0.005588 |
| entropy | 2 | ok | 1.72 | 81 | 18 | 0.9604 | 0.1667 | 0.003959 |
| entropy | 3 | ok | 1.74 | 81 | 18 | 0.9528 | 0.1667 | 0.004946 |
| entropy | 4 | ok | 1.90 | 81 | 18 | 0.9444 | 0.5556 | 0.005352 |
| icm | 0 | ok | 0.71 | 31.61 | 1 | 0.1858 | 1 | 0.06971 |
| icm | 1 | ok | 0.99 | 33.78 | 0 | 0.1346 | 1 | 0.07294 |
| icm | 2 | ok | 0.74 | 29.72 | 0 | 0.08385 | 1 | 0.07495 |
| icm | 3 | ok | 0.79 | 31.44 | 0 | 0.1817 | 1 | 0.06858 |
| icm | 4 | ok | 0.41 | 17.72 | 0 | 0.1189 | 1 | 0.08606 |
| info_gain | 0 | ok | 0.27 | 11.22 | 0 | 0.003968 | 1 | 0.1683 |
| info_gain | 1 | ok | 0.29 | 12.39 | 0 | 0 | 1 | 0.1669 |
| info_gain | 2 | ok | 0.29 | 12.17 | 0 | 0 | 1 | 0.1665 |
| info_gain | 3 | ok | 0.25 | 10 | 0 | 0.002778 | 1 | 0.1637 |
| info_gain | 4 | ok | 0.25 | 10.5 | 0 | 0.005556 | 1 | 0.1549 |
| lp | 0 | ok | 0.94 | 40.67 | 5 | 0.2377 | 0.8889 | 0.06959 |
| lp | 1 | ok | 1.10 | 49 | 6 | 0.2473 | 0.9444 | 0.06279 |
| lp | 2 | ok | 0.70 | 30.83 | 1 | 0.119 | 1 | 0.07381 |
| lp | 3 | ok | 1.08 | 49.5 | 4 | 0.2788 | 1 | 0.05721 |
| lp | 4 | ok | 0.47 | 19.33 | 0 | 0.1035 | 1 | 0.08295 |
| random | 0 | ok | 0.63 | 29 | 0 | 0.1112 | 1 | 0.07144 |
| random | 1 | ok | 0.91 | 36.28 | 1 | 0.114 | 1 | 0.07945 |
| random | 2 | ok | 0.55 | 25.44 | 0 | 0.1101 | 1 | 0.08511 |
| random | 3 | ok | 0.94 | 29.28 | 1 | 0.1226 | 1 | 0.07424 |
| random | 4 | ok | 0.55 | 24.56 | 0 | 0.1104 | 1 | 0.07724 |

Per-arm means:

| arm | interactions_to_identify | censored | tv_fraction | map_correct | realized_nats_per_interaction |
|---|---|---|---|---|---|
| info_gain | 11.26 | 0 | 0.00246 | 1 | 0.1641 |
| random | 28.91 | 0.4 | 0.1137 | 1 | 0.0775 |
| icm | 28.86 | 0.2 | 0.141 | 1 | 0.07445 |
| entropy | 81 | 18 | 0.9521 | 0.3 | 0.004931 |
| lp | 37.87 | 3.2 | 0.1972 | 0.9667 | 0.06927 |

### E10b-four-noisy-TVs

- **verdict: accept** — mean improvement 9.983 >= delta 1, CI [7.212, 13.1] excludes 0
- hypothesis: with FOUR coin-flip TVs (K=6, noise 0.35, budget 60), info-gain identifies the shift in >= 1 fewer interactions than random
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `random`, candidate `info_joint`, ablation `entropy`, alternative `icm`; seeds [0, 1, 2, 3, 4]
- prereg hash `d559e8a8bb87`, frozen 1791071277577970000 ns; first result 1791071299539784000 ns
- report: `reports/full/E10b-four-noisy-TVs-d559e8a8bb-1791071305539054000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_joint vs random | final (decisive) | 5 | 9.983 | 7.212 | 13.1 |
| info_joint vs random | equal_interactions | 0 | - | - | - |
| info_joint vs random | equal_time | 0 | - | - | - |
| info_joint vs entropy (ablation_arm) | final | 5 | 55.18 | 54.68 | 55.76 |
| info_joint vs icm (alternative_arm) | final | 5 | 14.27 | 11.73 | 17.43 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['lp']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_joint | 5/5 | 0 | 0 | 5.817 | 0.4425 |
| random | 5/5 | 0 | 0 | 15.8 | 2.239 |
| icm | 5/5 | 0 | 0 | 20.08 | 2.492 |
| entropy | 5/5 | 0 | 0 | 61 | 0 |
| lp | 5/5 | 0 | 0 | 21.53 | 5.104 |

Per-run outcomes (E10b):

| arm | seed | status | wall s | interactions_to_identify | censored | tv_fraction |
|---|---|---|---|---|---|---|
| entropy | 0 | ok | 0.95 | 61 | 1 | 0.9722 |
| entropy | 1 | ok | 0.83 | 61 | 1 | 0.9736 |
| entropy | 2 | ok | 0.82 | 61 | 1 | 0.9722 |
| entropy | 3 | ok | 0.92 | 61 | 1 | 0.9611 |
| entropy | 4 | ok | 0.84 | 61 | 1 | 0.9681 |
| icm | 0 | ok | 0.11 | 20.17 | 0 | 0.4343 |
| icm | 1 | ok | 0.09 | 18.67 | 0 | 0.4538 |
| icm | 2 | ok | 0.09 | 20.42 | 0.08333 | 0.3762 |
| icm | 3 | ok | 0.08 | 17.25 | 0 | 0.4037 |
| icm | 4 | ok | 0.10 | 23.92 | 0 | 0.4726 |
| info_joint | 0 | ok | 0.10 | 5.667 | 0 | 0 |
| info_joint | 1 | ok | 0.09 | 6.083 | 0 | 0.05873 |
| info_joint | 2 | ok | 0.10 | 6.333 | 0 | 0.005556 |
| info_joint | 3 | ok | 0.08 | 5.167 | 0 | 0.0119 |
| info_joint | 4 | ok | 0.09 | 5.833 | 0 | 0.00641 |
| lp | 0 | ok | 0.11 | 25.42 | 0.1667 | 0.4611 |
| lp | 1 | ok | 0.08 | 19.08 | 0.08333 | 0.4045 |
| lp | 2 | ok | 0.12 | 28.08 | 0.25 | 0.4234 |
| lp | 3 | ok | 0.07 | 15.5 | 0 | 0.3446 |
| lp | 4 | ok | 0.09 | 19.58 | 0 | 0.3739 |
| random | 0 | ok | 0.07 | 14.25 | 0 | 0.3522 |
| random | 1 | ok | 0.06 | 13.75 | 0 | 0.3042 |
| random | 2 | ok | 0.06 | 15 | 0 | 0.4142 |
| random | 3 | ok | 0.06 | 16.75 | 0 | 0.4087 |
| random | 4 | ok | 0.07 | 19.25 | 0 | 0.3717 |

Per-arm means:

| arm | interactions_to_identify | censored | tv_fraction |
|---|---|---|---|
| info_joint | 5.817 | 0 | 0.01652 |
| random | 15.8 | 0 | 0.3702 |
| icm | 20.08 | 0.01667 | 0.4281 |
| entropy | 61 | 1 | 0.9694 |
| lp | 21.53 | 0.1 | 0.4015 |

### E10c-learnable-useless-distractor

- **verdict: accept** — mean improvement 5.475 >= delta 1, CI [4.128, 7.177] excludes 0
- hypothesis: with a deterministic 4-bit lamp the task does not need (joint shift x lamp hypotheses), info-gain on the JOINT set wastes interactions on the lamp: task-marginal info-gain identifies the shift in >= 1 fewer interactions
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `info_joint`, candidate `info_marginal`, ablation `random`, alternative `icm`; seeds [0, 1, 2, 3, 4]
- prereg hash `9d74dc992b85`, frozen 1791071277578320000 ns; first result 1791071305693144000 ns
- report: `reports/full/E10c-learnable-useless-distractor-9d74dc992b-1791071307579603000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_marginal vs info_joint | final (decisive) | 5 | 5.475 | 4.128 | 7.177 |
| info_marginal vs info_joint | equal_interactions | 0 | - | - | - |
| info_marginal vs info_joint | equal_time | 0 | - | - | - |
| info_marginal vs random (ablation_arm) | final | 5 | 14.57 | 8.637 | 19.56 |
| info_marginal vs icm (alternative_arm) | final | 5 | 12 | 8.234 | 15.64 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_joint | 5/5 | 0 | 0 | 10.6 | 1.251 |
| info_marginal | 5/5 | 0 | 0 | 5.125 | 1.296 |
| random | 5/5 | 0 | 0 | 19.7 | 3.328 |
| icm | 5/5 | 0 | 0 | 17.12 | 3.292 |

Per-run outcomes (E10c):

| arm | seed | status | wall s | interactions_to_identify | censored | lamp_fraction | tv_fraction |
|---|---|---|---|---|---|---|---|
| icm | 0 | ok | 0.10 | 21.88 | 0 | 0.2819 | 0.1602 |
| icm | 1 | ok | 0.07 | 15.88 | 0 | 0.279 | 0.1013 |
| icm | 2 | ok | 0.08 | 18.75 | 0 | 0.3126 | 0.08639 |
| icm | 3 | ok | 0.06 | 13.25 | 0 | 0.2483 | 0.1386 |
| icm | 4 | ok | 0.07 | 15.88 | 0 | 0.3098 | 0.1562 |
| info_joint | 0 | ok | 0.15 | 11.12 | 0 | 0.441 | 0.01042 |
| info_joint | 1 | ok | 0.14 | 10.88 | 0 | 0.3973 | 0 |
| info_joint | 2 | ok | 0.15 | 12 | 0 | 0.4409 | 0 |
| info_joint | 3 | ok | 0.13 | 10.38 | 0 | 0.4595 | 0 |
| info_joint | 4 | ok | 0.12 | 8.625 | 0 | 0.4714 | 0.01562 |
| info_marginal | 0 | ok | 0.14 | 6.75 | 0 | 0.04688 | 0.01562 |
| info_marginal | 1 | ok | 0.12 | 5.375 | 0 | 0.01786 | 0 |
| info_marginal | 2 | ok | 0.10 | 4.375 | 0 | 0.01786 | 0 |
| info_marginal | 3 | ok | 0.13 | 5.75 | 0 | 0.01786 | 0 |
| info_marginal | 4 | ok | 0.08 | 3.375 | 0 | 0.05952 | 0.03125 |
| random | 0 | ok | 0.07 | 15.5 | 0 | 0.3738 | 0.08052 |
| random | 1 | ok | 0.09 | 23 | 0 | 0.278 | 0.08029 |
| random | 2 | ok | 0.08 | 23 | 0 | 0.4129 | 0.07033 |
| random | 3 | ok | 0.07 | 17.5 | 0 | 0.4106 | 0.07765 |
| random | 4 | ok | 0.08 | 19.5 | 0 | 0.2567 | 0.08543 |

Per-arm means:

| arm | interactions_to_identify | censored | lamp_fraction | tv_fraction |
|---|---|---|---|---|
| info_joint | 10.6 | 0 | 0.442 | 0.005208 |
| info_marginal | 5.125 | 0 | 0.03199 | 0.009375 |
| random | 19.7 | 0 | 0.3464 | 0.07884 |
| icm | 17.12 | 0 | 0.2863 | 0.1285 |

### E10d-misspecified-hypothesis-set

- **verdict: accept** — mean improvement 0.9818 >= delta 0.3, CI [0.9242, 1.011] excludes 0
- hypothesis: when the truth is NOT in the hypothesis set, info-gain selection reaches > 0.9 posterior on a (necessarily wrong) shift with no flag in >= 30 percentage points more scenarios than it reaches confident-wrong when the set is well specified
- primary `confident_wrong_unflagged` (higher better), delta 0.3, basis `final`; baseline `info_wellspec`, candidate `info_misspec`, ablation `random_misspec`, alternative `info_ppc_misspec`; seeds [0, 1, 2, 3, 4]
- prereg hash `2fd51cf6b351`, frozen 1791071277578660000 ns; first result 1791071307648028000 ns
- report: `reports/full/E10d-misspecified-hypothesis-set-2fd51cf6b3-1791071309501221000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_misspec vs info_wellspec | final (decisive) | 5 | 0.9818 | 0.9242 | 1.011 |
| info_misspec vs info_wellspec | equal_interactions | 0 | - | - | - |
| info_misspec vs info_wellspec | equal_time | 0 | - | - | - |
| info_misspec vs random_misspec (ablation_arm) | final | 5 | 0 | 0 | 0 |
| info_misspec vs info_ppc_misspec (alternative_arm) | final | 5 | 0.07273 | 0.01514 | 0.1015 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_wellspec | 5/5 | 0 | 0 | 0.01818 | 0.04066 |
| info_misspec | 5/5 | 0 | 0 | 1 | 0 |
| random_misspec | 5/5 | 0 | 0 | 1 | 0 |
| info_ppc_misspec | 5/5 | 0 | 0 | 0.9273 | 0.04066 |

Per-run outcomes (E10d):

| arm | seed | status | wall s | confident_wrong_unflagged | confident | time_to_confidence | ppc_flagged | interactions_to_identify |
|---|---|---|---|---|---|---|---|---|
| info_misspec | 0 | ok | 0.14 | 1 | 1 | 14.91 | 0.09091 | 61 |
| info_misspec | 1 | ok | 0.07 | 1 | 1 | 6.545 | 0.09091 | 61 |
| info_misspec | 2 | ok | 0.18 | 1 | 1 | 16.64 | 0.09091 | 61 |
| info_misspec | 3 | ok | 0.15 | 1 | 1 | 14.36 | 0 | 61 |
| info_misspec | 4 | ok | 0.13 | 1 | 1 | 11.27 | 0.09091 | 61 |
| info_ppc_misspec | 0 | ok | 0.14 | 0.9091 | 1 | 14.91 | 0.09091 | 61 |
| info_ppc_misspec | 1 | ok | 0.07 | 0.9091 | 1 | 6.545 | 0.09091 | 61 |
| info_ppc_misspec | 2 | ok | 0.17 | 0.9091 | 1 | 16.64 | 0.09091 | 61 |
| info_ppc_misspec | 3 | ok | 0.16 | 1 | 1 | 14.36 | 0 | 61 |
| info_ppc_misspec | 4 | ok | 0.15 | 0.9091 | 1 | 11.27 | 0.09091 | 61 |
| info_wellspec | 0 | ok | 0.06 | 0 | 1 | 6.091 | 0 | 6.091 |
| info_wellspec | 1 | ok | 0.05 | 0.09091 | 1 | 4.636 | 0 | 5.091 |
| info_wellspec | 2 | ok | 0.07 | 0 | 1 | 5.545 | 0 | 5.545 |
| info_wellspec | 3 | ok | 0.06 | 0 | 1 | 4.727 | 0 | 4.727 |
| info_wellspec | 4 | ok | 0.05 | 0 | 1 | 4.455 | 0 | 4.455 |
| random_misspec | 0 | ok | 0.04 | 1 | 1 | 15.09 | 0 | 61 |
| random_misspec | 1 | ok | 0.08 | 1 | 1 | 13.64 | 0.1818 | 61 |
| random_misspec | 2 | ok | 0.03 | 1 | 1 | 10.09 | 0.09091 | 61 |
| random_misspec | 3 | ok | 0.05 | 1 | 1 | 12.64 | 0 | 61 |
| random_misspec | 4 | ok | 0.07 | 1 | 1 | 18.36 | 0.09091 | 61 |

Per-arm means:

| arm | confident_wrong_unflagged | confident | time_to_confidence | ppc_flagged | interactions_to_identify |
|---|---|---|---|---|---|
| info_wellspec | 0.01818 | 1 | 5.091 | 0 | 5.182 |
| info_misspec | 1 | 1 | 12.75 | 0.07273 | 61 |
| random_misspec | 1 | 1 | 13.96 | 0.07273 | 61 |
| info_ppc_misspec | 0.9273 | 1 | 12.75 | 0.07273 | 61 |

## E10e idle / stationary-probe retention tracker (measurement, not an A/B)

RetentionTracker(known_below=0.45, forgotten_above=0.55); 16 probes with i.i.d. N(center, sd) losses, 2000 evaluations, seeds [0, 1, 2, 3, 4].

| config | seed | net progress | telescoped first-last | raw (clipped LP) gain | relearn events | flagged probes /16 |
|---|---|---|---|---|---|---|
| straddle 0.5+-0.2 | 0 | -3011.7 | -0.65 | 3584.9 | 6395 | 16 |
| straddle 0.5+-0.2 | 1 | -3011.9 | -0.22 | 3580.9 | 6410 | 16 |
| straddle 0.5+-0.2 | 2 | -3031.7 | -0.87 | 3593.4 | 6393 | 16 |
| straddle 0.5+-0.2 | 3 | -3056.6 | -1.37 | 3599.1 | 6411 | 16 |
| straddle 0.5+-0.2 | 4 | -3041.2 | 0.62 | 3607.3 | 6374 | 16 |
| wide 0.3+-0.2 | 0 | -1269.9 | -0.65 | 3584.9 | 2953 | 10 |
| wide 0.3+-0.2 | 1 | -1230.9 | -0.22 | 3580.9 | 2838 | 7 |
| wide 0.3+-0.2 | 2 | -1289.2 | -0.87 | 3593.4 | 2976 | 9 |
| wide 0.3+-0.2 | 3 | -1302.3 | -1.37 | 3599.1 | 3015 | 12 |
| wide 0.3+-0.2 | 4 | -1289.5 | 0.62 | 3607.3 | 3037 | 12 |
| far 0.2+-0.05 | 0 | -0.2 | -0.16 | 896.2 | 0 | 0 |
| far 0.2+-0.05 | 1 | -0.1 | -0.06 | 895.2 | 0 | 0 |
| far 0.2+-0.05 | 2 | -0.2 | -0.22 | 898.4 | 0 | 0 |
| far 0.2+-0.05 | 3 | -0.3 | -0.34 | 899.8 | 0 | 0 |
| far 0.2+-0.05 | 4 | 0.2 | 0.15 | 901.8 | 0 | 0 |
| straddle annealed | 0 | -6071.3 | -2.36 | 7165.5 | 7102 | 16 |
| straddle annealed | 1 | -6076.1 | 0.59 | 7170.4 | 7085 | 16 |
| straddle annealed | 2 | -6107.4 | -0.93 | 7184.4 | 7096 | 16 |
| straddle annealed | 3 | -6163.4 | -2.83 | 7193.3 | 7103 | 16 |
| straddle annealed | 4 | -6158.7 | 1.18 | 7216.5 | 7100 | 16 |

## E11 (57s wall)

### E11a-gate-vs-ungated-rare-region

- **verdict: inconclusive** — CI [-0.0321, 0.05984] with mean 0.01575 neither meets nor rules out delta 0.02
- hypothesis: with a model wrong only inside a rarely visited disc (claims boost, truth is mud), the reliability gate lowers real mean distance to the goal versus the ungated planner by >= 0.02 m
- primary `mean_dist` (lower better), delta 0.02, basis `final`; baseline `ungated`, candidate `gated`, ablation `fallback`, alternative `gated_true`; seeds [0, 1, 2, 3, 4]
- prereg hash `2508ca667846`, frozen 1791071311533654000 ns; first result 1791071311637617000 ns
- report: `reports/full/E11a-gate-vs-ungated-rare-region-2508ca6678-1791071347375644000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| gated vs ungated | final (decisive) | 5 | 0.01575 | -0.0321 | 0.05984 |
| gated vs ungated | equal_interactions | 5 | 0.01575 | -0.0321 | 0.05984 |
| gated vs ungated | equal_time | 0 | - | - | - |
| gated vs fallback (ablation_arm) | final | 5 | 0.04094 | -0.0004376 | 0.08231 |
| gated vs gated_true (alternative_arm) | final | 5 | -0.1907 | -0.2871 | -0.1285 |

- note: criterion unmet: baseline 'ungated' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| fallback | 5/5 | 0 | 0 | 0.904 | 0.1018 |
| gated | 5/5 | 0 | 0 | 0.863 | 0.09994 |
| ungated | 5/5 | 0 | 0 | 0.8788 | 0.0794 |
| gated_true | 5/5 | 0 | 0 | 0.6724 | 0.09728 |

Per-run outcomes (E11a):

| arm | seed | status | wall s | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|---|---|---|
| fallback | 0 | ok | 0.10 | 0.7467 | 0.2375 | 0 | 0 | 0 | 0 |
| fallback | 1 | ok | 0.06 | 0.9167 | 0.4167 | 0 | 0 | 0 | 0 |
| fallback | 2 | ok | 0.06 | 0.9728 | 0.3042 | 0 | 0 | 0 | 0 |
| fallback | 3 | ok | 0.07 | 0.8749 | 0.3875 | 0 | 0 | 0 | 0 |
| fallback | 4 | ok | 0.07 | 1.009 | 0.4396 | 0 | 0 | 0 | 0 |
| gated | 0 | ok | 2.05 | 0.707 | 0.3146 | 6 | 3 | 6 | 0.5542 |
| gated | 1 | ok | 1.22 | 0.8373 | 0.4146 | 7 | 5 | 7 | 0.4521 |
| gated | 2 | ok | 1.23 | 0.957 | 0.4021 | 7 | 5 | 6 | 0.475 |
| gated | 3 | ok | 1.29 | 0.8736 | 0.4667 | 5 | 4 | 5 | 0.4146 |
| gated | 4 | ok | 1.60 | 0.9404 | 0.4479 | 7 | 7 | 7 | 0.4021 |
| gated_true | 0 | ok | 2.82 | 0.5309 | 0.1417 | 69 | 1 | 0 | 0.9938 |
| gated_true | 1 | ok | 2.43 | 0.6868 | 0.2708 | 127 | 4 | 0 | 0.9938 |
| gated_true | 2 | ok | 2.44 | 0.7842 | 0.2583 | 120 | 4 | 0 | 0.9938 |
| gated_true | 3 | ok | 2.68 | 0.73 | 0.3729 | 174 | 5 | 0 | 0.9938 |
| gated_true | 4 | ok | 3.16 | 0.6299 | 0.1354 | 61 | 4 | 0 | 0.9938 |
| ungated | 0 | ok | 3.58 | 0.7512 | 0.3896 | 183 | 7 | 0 | 0.9979 |
| ungated | 1 | ok | 2.36 | 0.8541 | 0.4167 | 194 | 7 | 0 | 0.9979 |
| ungated | 2 | ok | 2.37 | 0.9126 | 0.3833 | 177 | 7 | 0 | 0.9979 |
| ungated | 3 | ok | 2.84 | 0.9294 | 0.5375 | 251 | 9 | 0 | 0.9979 |
| ungated | 4 | ok | 3.37 | 0.9467 | 0.4583 | 211 | 8 | 0 | 0.9979 |

Per-arm means:

| arm | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|
| fallback | 0.904 | 0.3571 | 0 | 0 | 0 | 0 |
| gated | 0.863 | 0.4092 | 6.4 | 4.8 | 6.2 | 0.4596 |
| ungated | 0.8788 | 0.4371 | 203.2 | 7.6 | 0 | 0.9979 |
| gated_true | 0.6724 | 0.2358 | 110.2 | 3.6 | 0 | 0.9938 |

### E11b-exploitation-leaks-through-gate

- **verdict: reject** — CI upper bound 0.0004376 < delta 0.01: the hypothesised effect is ruled out
- hypothesis: FALSIFICATION of 'the gate prevents exploitation': the planner keeps re-entering the region the model is wrong about after each gate re-opening, so the fallback ALONE achieves lower real mean distance than the gated planner by >= 0.01 m
- primary `mean_dist` (lower better), delta 0.01, basis `final`; baseline `gated`, candidate `fallback`, ablation `ungated`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `e47d881a7340`, frozen 1791071311533792000 ns; first result 1791071347381609000 ns
- report: `reports/full/E11b-exploitation-leaks-through-gate-e47d881a73-1791071347382158000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| fallback vs gated | final (decisive) | 5 | -0.04094 | -0.08231 | 0.0004376 |
| fallback vs gated | equal_interactions | 5 | -0.04094 | -0.08231 | 0.0004376 |
| fallback vs gated | equal_time | 0 | - | - | - |
| fallback vs ungated (ablation_arm) | final | 5 | -0.02519 | -0.08337 | 0.04823 |

- note: criterion unmet: baseline 'gated' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['gated_true']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| fallback | 5/5 | 0 | 0 | 0.904 | 0.1018 |
| gated | 5/5 | 0 | 0 | 0.863 | 0.09994 |
| ungated | 5/5 | 0 | 0 | 0.8788 | 0.0794 |
| gated_true | 5/5 | 0 | 0 | 0.6724 | 0.09728 |

Per-run outcomes (E11b):

| arm | seed | status | wall s | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|---|---|---|
| fallback | 0 | ok | 0.00 | 0.7467 | 0.2375 | 0 | 0 | 0 | 0 |
| fallback | 1 | ok | 0.00 | 0.9167 | 0.4167 | 0 | 0 | 0 | 0 |
| fallback | 2 | ok | 0.00 | 0.9728 | 0.3042 | 0 | 0 | 0 | 0 |
| fallback | 3 | ok | 0.00 | 0.8749 | 0.3875 | 0 | 0 | 0 | 0 |
| fallback | 4 | ok | 0.00 | 1.009 | 0.4396 | 0 | 0 | 0 | 0 |
| gated | 0 | ok | 0.00 | 0.707 | 0.3146 | 6 | 3 | 6 | 0.5542 |
| gated | 1 | ok | 0.00 | 0.8373 | 0.4146 | 7 | 5 | 7 | 0.4521 |
| gated | 2 | ok | 0.00 | 0.957 | 0.4021 | 7 | 5 | 6 | 0.475 |
| gated | 3 | ok | 0.00 | 0.8736 | 0.4667 | 5 | 4 | 5 | 0.4146 |
| gated | 4 | ok | 0.00 | 0.9404 | 0.4479 | 7 | 7 | 7 | 0.4021 |
| gated_true | 0 | ok | 0.00 | 0.5309 | 0.1417 | 69 | 1 | 0 | 0.9938 |
| gated_true | 1 | ok | 0.00 | 0.6868 | 0.2708 | 127 | 4 | 0 | 0.9938 |
| gated_true | 2 | ok | 0.00 | 0.7842 | 0.2583 | 120 | 4 | 0 | 0.9938 |
| gated_true | 3 | ok | 0.00 | 0.73 | 0.3729 | 174 | 5 | 0 | 0.9938 |
| gated_true | 4 | ok | 0.00 | 0.6299 | 0.1354 | 61 | 4 | 0 | 0.9938 |
| ungated | 0 | ok | 0.00 | 0.7512 | 0.3896 | 183 | 7 | 0 | 0.9979 |
| ungated | 1 | ok | 0.00 | 0.8541 | 0.4167 | 194 | 7 | 0 | 0.9979 |
| ungated | 2 | ok | 0.00 | 0.9126 | 0.3833 | 177 | 7 | 0 | 0.9979 |
| ungated | 3 | ok | 0.00 | 0.9294 | 0.5375 | 251 | 9 | 0 | 0.9979 |
| ungated | 4 | ok | 0.00 | 0.9467 | 0.4583 | 211 | 8 | 0 | 0.9979 |

Per-arm means:

| arm | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|
| fallback | 0.904 | 0.3571 | 0 | 0 | 0 | 0 |
| gated | 0.863 | 0.4092 | 6.4 | 4.8 | 6.2 | 0.4596 |
| ungated | 0.8788 | 0.4371 | 203.2 | 7.6 | 0 | 0.9979 |
| gated_true | 0.6724 | 0.2358 | 110.2 | 3.6 | 0 | 0.9938 |

### E11c-deadline-heavy-tailed-model

- **verdict: accept** — mean improvement 0.06667 >= delta 0.01, CI [0.04344, 0.09411] excludes 0
- hypothesis: with a model whose call latency is heavy-tailed (2% of calls take 30 ms) under a 10 ms deadline, >= 1% more decisions take over TWICE the deadline than with a constant-latency model of equal mean: the 'hard' deadline is bounded only by one chunk's worst case
- primary `overrun_fraction` (higher better), delta 0.01, basis `final`; baseline `constant`, candidate `heavy_tail`, ablation `heavy_tail_chunk8`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `b02fc1205e18`, frozen 1791071311533965000 ns; first result 1791071348297519000 ns
- report: `reports/full/E11c-deadline-heavy-tailed-model-b02fc1205e-1791071361733501000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| heavy_tail vs constant | final (decisive) | 5 | 0.06667 | 0.04344 | 0.09411 |
| heavy_tail vs constant | equal_interactions | 5 | 0.06667 | 0.04344 | 0.09411 |
| heavy_tail vs constant | equal_time | 0 | - | - | - |
| heavy_tail vs heavy_tail_chunk8 (ablation_arm) | final | 5 | -0.06933 | -0.09467 | -0.044 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| constant | 5/5 | 0 | 0 | 0.01867 | 0.01282 |
| heavy_tail | 5/5 | 0 | 0 | 0.08533 | 0.01193 |
| heavy_tail_chunk8 | 5/5 | 0 | 0 | 0.1547 | 0.008692 |

Per-run outcomes (E11c):

| arm | seed | status | wall s | overrun_fraction | late_fraction | max_latency_over_deadline | p99_latency_ms | median_latency_ms | planner_fraction | timeout_fraction | probes | late_chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constant | 0 | ok | 0.91 | 0.02667 | 0.1 | 6.406 | 26.49 | 6.124 | 0.56 | 0.44 | 11 | 14 |
| constant | 1 | ok | 1.01 | 0.03333 | 0.08 | 6.027 | 37.79 | 5.851 | 0.6533 | 0.3467 | 9 | 11 |
| constant | 2 | ok | 1.31 | 0 | 0.02 | 1.012 | 10.07 | 9.291 | 1 | 0 | 0 | 1 |
| constant | 3 | ok | 0.99 | 0.02 | 0.08 | 2.481 | 22.8 | 9.309 | 0.6733 | 0.3267 | 8 | 11 |
| constant | 4 | ok | 1.12 | 0.01333 | 0.04 | 2.757 | 20.54 | 8.691 | 0.8333 | 0.1667 | 4 | 6 |
| heavy_tail | 0 | ok | 0.60 | 0.07333 | 0.07333 | 4.266 | 42 | 1.748 | 0.5933 | 0.4067 | 11 | 11 |
| heavy_tail | 1 | ok | 0.76 | 0.09333 | 0.1 | 5.705 | 42.61 | 1.733 | 0.5133 | 0.4867 | 13 | 15 |
| heavy_tail | 2 | ok | 0.74 | 0.1 | 0.1 | 4.246 | 41.83 | 2.005 | 0.42 | 0.58 | 14 | 15 |
| heavy_tail | 3 | ok | 0.61 | 0.07333 | 0.08 | 4.914 | 44.2 | 1.727 | 0.5667 | 0.4333 | 12 | 12 |
| heavy_tail | 4 | ok | 0.66 | 0.08667 | 0.08667 | 8.101 | 44.59 | 1.317 | 0.52 | 0.48 | 12 | 13 |
| heavy_tail_chunk8 | 0 | ok | 1.14 | 0.16 | 0.1667 | 8.549 | 49.44 | 0.06927 | 0.2 | 0.8 | 23 | 24 |
| heavy_tail_chunk8 | 1 | ok | 1.16 | 0.1467 | 0.18 | 4.961 | 49.07 | 0.09675 | 0.26 | 0.74 | 21 | 24 |
| heavy_tail_chunk8 | 2 | ok | 1.13 | 0.1467 | 0.1533 | 4.993 | 49.29 | 0.09348 | 0.2467 | 0.7533 | 21 | 22 |
| heavy_tail_chunk8 | 3 | ok | 1.10 | 0.1667 | 0.1667 | 4.989 | 46.23 | 0.06594 | 0.1867 | 0.8133 | 24 | 25 |
| heavy_tail_chunk8 | 4 | ok | 1.10 | 0.1533 | 0.16 | 5.063 | 47.96 | 0.07027 | 0.2467 | 0.7533 | 22 | 23 |

Per-arm means:

| arm | overrun_fraction | late_fraction | max_latency_over_deadline | p99_latency_ms | median_latency_ms | planner_fraction | timeout_fraction | probes | late_chunks |
|---|---|---|---|---|---|---|---|---|---|
| constant | 0.01867 | 0.064 | 3.737 | 23.54 | 7.853 | 0.744 | 0.256 | 6.4 | 8.6 |
| heavy_tail | 0.08533 | 0.088 | 5.446 | 43.05 | 1.706 | 0.5227 | 0.4773 | 12.4 | 13.2 |
| heavy_tail_chunk8 | 0.1547 | 0.1653 | 5.711 | 48.4 | 0.07914 | 0.228 | 0.772 | 22.2 | 23.6 |

### E11d-offprobe-false-duplicate

- **verdict: accept** — mean improvement 0.92 >= delta 0.5, CI [0.825, 0.9992] excludes 0
- hypothesis: two skills that differ only outside the probe distribution (argmax permuted when any |s_i| > 7) are flagged duplicates at a rate >= 0.5 above that of an unrelated same-name policy
- primary `flag_rate` (higher better), delta 0.5, basis `final`; baseline `samename_diff`, candidate `offprobe`, ablation `name_only`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `7bdc16f4e89a`, frozen 1791071311534130000 ns; first result 1791071362028919000 ns
- report: `reports/full/E11d-offprobe-false-duplicate-7bdc16f4e8-1791071368648067000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| offprobe vs samename_diff | final (decisive) | 5 | 0.92 | 0.825 | 0.9992 |
| offprobe vs samename_diff | equal_interactions | 5 | 0.92 | 0.825 | 0.9992 |
| offprobe vs samename_diff | equal_time | 0 | - | - | - |
| offprobe vs name_only (ablation_arm) | final | 5 | -0.08 | -0.175 | -0.0008235 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['drift_0.001', 'drift_0.01', 'drift_0.1', 'tempered']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| name_only | 5/5 | 0 | 0 | 1 | 0 |
| samename_diff | 5/5 | 0 | 0 | 0 | 0 |
| offprobe | 5/5 | 0 | 0 | 0.92 | 0.07583 |
| tempered | 5/5 | 0 | 0 | 0 | 0 |
| drift_0.001 | 5/5 | 0 | 0 | 1 | 0 |
| drift_0.01 | 5/5 | 0 | 0 | 1 | 0 |
| drift_0.1 | 5/5 | 0 | 0 | 0 | 0 |

Per-run outcomes (E11d):

| arm | seed | status | wall s | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|---|---|---|
| drift_0.001 | 0 | ok | 0.27 | 1 | 3.084e-07 | 4.657e-07 | 0.000375 | 0.05 |
| drift_0.001 | 1 | ok | 0.20 | 1 | 3.016e-07 | 6.09e-07 | 0.000875 | 0.2 |
| drift_0.001 | 2 | ok | 0.19 | 1 | 3.937e-07 | 8.661e-07 | 0.0005 | 0.1 |
| drift_0.001 | 3 | ok | 0.19 | 1 | 4.063e-07 | 7.55e-07 | 0.000625 | 0 |
| drift_0.001 | 4 | ok | 0.17 | 1 | 4.408e-07 | 8.914e-07 | 0.001375 | 0.05 |
| drift_0.01 | 0 | ok | 0.22 | 1 | 3.078e-05 | 4.619e-05 | 0.007875 | 0.05 |
| drift_0.01 | 1 | ok | 0.20 | 1 | 3.013e-05 | 5.976e-05 | 0.009 | 0.2 |
| drift_0.01 | 2 | ok | 0.22 | 1 | 3.945e-05 | 8.746e-05 | 0.009375 | 0.1 |
| drift_0.01 | 3 | ok | 0.19 | 1 | 4.075e-05 | 7.648e-05 | 0.006125 | 0 |
| drift_0.01 | 4 | ok | 0.16 | 1 | 4.409e-05 | 8.883e-05 | 0.00975 | 0.05 |
| drift_0.1 | 0 | ok | 0.22 | 0 | 0.003007 | 0.004346 | 0.07425 | 0.05 |
| drift_0.1 | 1 | ok | 0.19 | 0 | 0.002984 | 0.004903 | 0.07938 | 0.2 |
| drift_0.1 | 2 | ok | 0.19 | 0 | 0.004007 | 0.009433 | 0.09375 | 0.1 |
| drift_0.1 | 3 | ok | 0.19 | 0 | 0.00413 | 0.008217 | 0.09112 | 0 |
| drift_0.1 | 4 | ok | 0.15 | 0 | 0.004346 | 0.008447 | 0.09963 | 0.05 |
| name_only | 0 | ok | 0.29 | 1 | 0 | 0 | 0 | 0.05 |
| name_only | 1 | ok | 0.19 | 1 | 0 | 0 | 0 | 0.2 |
| name_only | 2 | ok | 0.19 | 1 | 0 | 0 | 0 | 0.1 |
| name_only | 3 | ok | 0.18 | 1 | 0 | 0 | 0 | 0 |
| name_only | 4 | ok | 0.15 | 1 | 0 | 0 | 0 | 0.05 |
| offprobe | 0 | ok | 0.25 | 0.95 | 0.0008436 | 0.01687 | 0.2886 | 0.05 |
| offprobe | 1 | ok | 0.21 | 0.8 | 0.00298 | 0.01987 | 0.2904 | 0.2 |
| offprobe | 2 | ok | 0.21 | 0.9 | 0.001906 | 0.02671 | 0.2863 | 0.1 |
| offprobe | 3 | ok | 0.20 | 1 | 0 | 0 | 0.2879 | 0 |
| offprobe | 4 | ok | 0.19 | 0.95 | 0.0008918 | 0.01784 | 0.2814 | 0.05 |
| samename_diff | 0 | ok | 0.22 | 0 | 0.3449 | 0.4825 | 0.7771 | 0.05 |
| samename_diff | 1 | ok | 0.22 | 0 | 0.3215 | 0.4402 | 0.762 | 0.2 |
| samename_diff | 2 | ok | 0.19 | 0 | 0.319 | 0.4412 | 0.788 | 0.1 |
| samename_diff | 3 | ok | 0.19 | 0 | 0.3274 | 0.5139 | 0.7897 | 0 |
| samename_diff | 4 | ok | 0.15 | 0 | 0.3334 | 0.4434 | 0.7886 | 0.05 |
| tempered | 0 | ok | 0.22 | 0 | 0.1009 | 0.1237 | 0 | 0.05 |
| tempered | 1 | ok | 0.21 | 0 | 0.1025 | 0.1208 | 0 | 0.2 |
| tempered | 2 | ok | 0.19 | 0 | 0.1052 | 0.1293 | 0 | 0.1 |
| tempered | 3 | ok | 0.19 | 0 | 0.09949 | 0.1136 | 0 | 0 |
| tempered | 4 | ok | 0.15 | 0 | 0.1025 | 0.1202 | 0 | 0.05 |

Per-arm means:

| arm | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|
| name_only | 1 | 0 | 0 | 0 | 0.08 |
| samename_diff | 0 | 0.3292 | 0.4642 | 0.7811 | 0.08 |
| offprobe | 0.92 | 0.001324 | 0.01626 | 0.2869 | 0.08 |
| tempered | 0 | 0.1021 | 0.1215 | 0 | 0.08 |
| drift_0.001 | 1 | 3.702e-07 | 7.174e-07 | 0.00075 | 0.08 |
| drift_0.01 | 1 | 3.704e-05 | 7.175e-05 | 0.008425 | 0.08 |
| drift_0.1 | 0 | 0.003695 | 0.007069 | 0.08763 | 0.08 |

### E11e-tempered-copy-not-flagged

- **verdict: accept** — mean improvement 1 >= delta 0.5, CI [1, 1] excludes 0
- hypothesis: a copy with logits x3 (identical argmax on every state, so identical deterministic behaviour) is flagged at a rate >= 0.5 BELOW a byte copy
- primary `flag_rate` (lower better), delta 0.5, basis `final`; baseline `name_only`, candidate `tempered`, ablation `samename_diff`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `68ea5467a4c2`, frozen 1791071311534210000 ns; first result 1791071368652144000 ns
- report: `reports/full/E11e-tempered-copy-not-flagged-68ea5467a4-1791071368652595000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| tempered vs name_only | final (decisive) | 5 | 1 | 1 | 1 |
| tempered vs name_only | equal_interactions | 5 | 1 | 1 | 1 |
| tempered vs name_only | equal_time | 0 | - | - | - |
| tempered vs samename_diff (ablation_arm) | final | 5 | 0 | 0 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['drift_0.001', 'drift_0.01', 'drift_0.1', 'offprobe']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| name_only | 5/5 | 0 | 0 | 1 | 0 |
| samename_diff | 5/5 | 0 | 0 | 0 | 0 |
| offprobe | 5/5 | 0 | 0 | 0.92 | 0.07583 |
| tempered | 5/5 | 0 | 0 | 0 | 0 |
| drift_0.001 | 5/5 | 0 | 0 | 1 | 0 |
| drift_0.01 | 5/5 | 0 | 0 | 1 | 0 |
| drift_0.1 | 5/5 | 0 | 0 | 0 | 0 |

Per-run outcomes (E11e):

| arm | seed | status | wall s | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|---|---|---|
| drift_0.001 | 0 | ok | 0.00 | 1 | 3.084e-07 | 4.657e-07 | 0.000375 | 0.05 |
| drift_0.001 | 1 | ok | 0.00 | 1 | 3.016e-07 | 6.09e-07 | 0.000875 | 0.2 |
| drift_0.001 | 2 | ok | 0.00 | 1 | 3.937e-07 | 8.661e-07 | 0.0005 | 0.1 |
| drift_0.001 | 3 | ok | 0.00 | 1 | 4.063e-07 | 7.55e-07 | 0.000625 | 0 |
| drift_0.001 | 4 | ok | 0.00 | 1 | 4.408e-07 | 8.914e-07 | 0.001375 | 0.05 |
| drift_0.01 | 0 | ok | 0.00 | 1 | 3.078e-05 | 4.619e-05 | 0.007875 | 0.05 |
| drift_0.01 | 1 | ok | 0.00 | 1 | 3.013e-05 | 5.976e-05 | 0.009 | 0.2 |
| drift_0.01 | 2 | ok | 0.00 | 1 | 3.945e-05 | 8.746e-05 | 0.009375 | 0.1 |
| drift_0.01 | 3 | ok | 0.00 | 1 | 4.075e-05 | 7.648e-05 | 0.006125 | 0 |
| drift_0.01 | 4 | ok | 0.00 | 1 | 4.409e-05 | 8.883e-05 | 0.00975 | 0.05 |
| drift_0.1 | 0 | ok | 0.00 | 0 | 0.003007 | 0.004346 | 0.07425 | 0.05 |
| drift_0.1 | 1 | ok | 0.00 | 0 | 0.002984 | 0.004903 | 0.07938 | 0.2 |
| drift_0.1 | 2 | ok | 0.00 | 0 | 0.004007 | 0.009433 | 0.09375 | 0.1 |
| drift_0.1 | 3 | ok | 0.00 | 0 | 0.00413 | 0.008217 | 0.09112 | 0 |
| drift_0.1 | 4 | ok | 0.00 | 0 | 0.004346 | 0.008447 | 0.09963 | 0.05 |
| name_only | 0 | ok | 0.00 | 1 | 0 | 0 | 0 | 0.05 |
| name_only | 1 | ok | 0.00 | 1 | 0 | 0 | 0 | 0.2 |
| name_only | 2 | ok | 0.00 | 1 | 0 | 0 | 0 | 0.1 |
| name_only | 3 | ok | 0.00 | 1 | 0 | 0 | 0 | 0 |
| name_only | 4 | ok | 0.00 | 1 | 0 | 0 | 0 | 0.05 |
| offprobe | 0 | ok | 0.00 | 0.95 | 0.0008436 | 0.01687 | 0.2886 | 0.05 |
| offprobe | 1 | ok | 0.00 | 0.8 | 0.00298 | 0.01987 | 0.2904 | 0.2 |
| offprobe | 2 | ok | 0.00 | 0.9 | 0.001906 | 0.02671 | 0.2863 | 0.1 |
| offprobe | 3 | ok | 0.00 | 1 | 0 | 0 | 0.2879 | 0 |
| offprobe | 4 | ok | 0.00 | 0.95 | 0.0008918 | 0.01784 | 0.2814 | 0.05 |
| samename_diff | 0 | ok | 0.00 | 0 | 0.3449 | 0.4825 | 0.7771 | 0.05 |
| samename_diff | 1 | ok | 0.00 | 0 | 0.3215 | 0.4402 | 0.762 | 0.2 |
| samename_diff | 2 | ok | 0.00 | 0 | 0.319 | 0.4412 | 0.788 | 0.1 |
| samename_diff | 3 | ok | 0.00 | 0 | 0.3274 | 0.5139 | 0.7897 | 0 |
| samename_diff | 4 | ok | 0.00 | 0 | 0.3334 | 0.4434 | 0.7886 | 0.05 |
| tempered | 0 | ok | 0.00 | 0 | 0.1009 | 0.1237 | 0 | 0.05 |
| tempered | 1 | ok | 0.00 | 0 | 0.1025 | 0.1208 | 0 | 0.2 |
| tempered | 2 | ok | 0.00 | 0 | 0.1052 | 0.1293 | 0 | 0.1 |
| tempered | 3 | ok | 0.00 | 0 | 0.09949 | 0.1136 | 0 | 0 |
| tempered | 4 | ok | 0.00 | 0 | 0.1025 | 0.1202 | 0 | 0.05 |

Per-arm means:

| arm | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|
| name_only | 1 | 0 | 0 | 0 | 0.08 |
| samename_diff | 0 | 0.3292 | 0.4642 | 0.7811 | 0.08 |
| offprobe | 0.92 | 0.001324 | 0.01626 | 0.2869 | 0.08 |
| tempered | 0 | 0.1021 | 0.1215 | 0 | 0.08 |
| drift_0.001 | 1 | 3.702e-07 | 7.174e-07 | 0.00075 | 0.08 |
| drift_0.01 | 1 | 3.704e-05 | 7.175e-05 | 0.008425 | 0.08 |
| drift_0.1 | 0 | 0.003695 | 0.007069 | 0.08763 | 0.08 |


Total wall 126s
