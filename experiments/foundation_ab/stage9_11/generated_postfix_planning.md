## E11a

### E11a-gate-vs-ungated-rare-region

- **verdict: accept** — mean improvement 0.08356 >= delta 0.02, CI [0.002304, 0.1648] excludes 0
- hypothesis: with a model wrong only inside a rarely visited disc (claims boost, truth is mud), the reliability gate lowers real mean distance to the goal versus the ungated planner by >= 0.02 m
- primary `mean_dist` (lower better), delta 0.02, basis `final`; baseline `ungated`, candidate `gated`, ablation `fallback`, alternative `gated_true`; seeds [0, 1, 2, 3, 4]
- prereg hash `2508ca667846`, frozen 1791071311533654000 ns; first result 1791074222618397000 ns
- report: `reports/postfix/E11a-gate-vs-ungated-rare-region-2508ca6678-1791074262171760000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| gated vs ungated | final (decisive) | 5 | 0.08356 | 0.002304 | 0.1648 |
| gated vs ungated | equal_interactions | 5 | 0.08356 | 0.002304 | 0.1648 |
| gated vs ungated | equal_time | 0 | - | - | - |
| gated vs fallback (ablation_arm) | final | 5 | 0.1087 | 0.06867 | 0.1499 |
| gated vs gated_true (alternative_arm) | final | 5 | -0.1229 | -0.2719 | -0.00994 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['gated_global']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| fallback | 5/5 | 0 | 0 | 0.904 | 0.1018 |
| gated | 5/5 | 0 | 0 | 0.7952 | 0.1173 |
| ungated | 5/5 | 0 | 0 | 0.8788 | 0.0794 |
| gated_true | 5/5 | 0 | 0 | 0.6724 | 0.09728 |
| gated_global | 5/5 | 0 | 0 | 0.863 | 0.09994 |

Per-run outcomes (E11a):

| arm | seed | status | wall s | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|---|---|---|
| fallback | 0 | ok | 0.04 | 0.7467 | 0.2375 | 0 | 0 | 0 | 0 |
| fallback | 1 | ok | 0.04 | 0.9167 | 0.4167 | 0 | 0 | 0 | 0 |
| fallback | 2 | ok | 0.03 | 0.9728 | 0.3042 | 0 | 0 | 0 | 0 |
| fallback | 3 | ok | 0.05 | 0.8749 | 0.3875 | 0 | 0 | 0 | 0 |
| fallback | 4 | ok | 0.04 | 1.009 | 0.4396 | 0 | 0 | 0 | 0 |
| gated | 0 | ok | 3.53 | 0.6422 | 0.2104 | 1 | 1 | 1 | 0.6979 |
| gated | 1 | ok | 3.21 | 0.7609 | 0.2271 | 1 | 1 | 1 | 0.7 |
| gated | 2 | ok | 3.34 | 0.8956 | 0.2354 | 3 | 2 | 4 | 0.6646 |
| gated | 3 | ok | 3.35 | 0.7478 | 0.2687 | 1 | 1 | 1 | 0.6708 |
| gated | 4 | ok | 5.57 | 0.9297 | 0.2958 | 2 | 2 | 2 | 0.6396 |
| gated_global | 0 | ok | 0.86 | 0.707 | 0.3146 | 6 | 3 | 6 | 0.5542 |
| gated_global | 1 | ok | 0.72 | 0.8373 | 0.4146 | 7 | 5 | 7 | 0.4521 |
| gated_global | 2 | ok | 0.79 | 0.957 | 0.4021 | 7 | 5 | 6 | 0.475 |
| gated_global | 3 | ok | 0.67 | 0.8736 | 0.4667 | 5 | 4 | 5 | 0.4146 |
| gated_global | 4 | ok | 0.71 | 0.9404 | 0.4479 | 7 | 7 | 7 | 0.4021 |
| gated_true | 0 | ok | 1.48 | 0.5309 | 0.1417 | 69 | 1 | 0 | 0.9938 |
| gated_true | 1 | ok | 1.52 | 0.6868 | 0.2708 | 127 | 4 | 0 | 0.9938 |
| gated_true | 2 | ok | 1.49 | 0.7842 | 0.2583 | 120 | 4 | 0 | 0.9938 |
| gated_true | 3 | ok | 1.69 | 0.73 | 0.3729 | 174 | 5 | 0 | 0.9938 |
| gated_true | 4 | ok | 1.76 | 0.6299 | 0.1354 | 61 | 4 | 0 | 0.9938 |
| ungated | 0 | ok | 1.66 | 0.7512 | 0.3896 | 183 | 7 | 0 | 0.9979 |
| ungated | 1 | ok | 1.67 | 0.8541 | 0.4167 | 194 | 7 | 0 | 0.9979 |
| ungated | 2 | ok | 1.49 | 0.9126 | 0.3833 | 177 | 7 | 0 | 0.9979 |
| ungated | 3 | ok | 1.85 | 0.9294 | 0.5375 | 251 | 9 | 0 | 0.9979 |
| ungated | 4 | ok | 2.02 | 0.9467 | 0.4583 | 211 | 8 | 0 | 0.9979 |

Per-arm means:

| arm | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|
| fallback | 0.904 | 0.3571 | 0 | 0 | 0 | 0 |
| gated | 0.7952 | 0.2475 | 1.6 | 1.4 | 1.8 | 0.6746 |
| ungated | 0.8788 | 0.4371 | 203.2 | 7.6 | 0 | 0.9979 |
| gated_true | 0.6724 | 0.2358 | 110.2 | 3.6 | 0 | 0.9938 |
| gated_global | 0.863 | 0.4092 | 6.4 | 4.8 | 6.2 | 0.4596 |

## E11b

### E11b-exploitation-leaks-through-gate

- **verdict: reject** — CI upper bound -0.06867 < delta 0.01: the hypothesised effect is ruled out
- hypothesis: FALSIFICATION of 'the gate prevents exploitation': the planner keeps re-entering the region the model is wrong about after each gate re-opening, so the fallback ALONE achieves lower real mean distance than the gated planner by >= 0.01 m
- primary `mean_dist` (lower better), delta 0.01, basis `final`; baseline `gated`, candidate `fallback`, ablation `ungated`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `e47d881a7340`, frozen 1791071311533792000 ns; first result 1791074262177821000 ns
- report: `reports/postfix/E11b-exploitation-leaks-through-gate-e47d881a73-1791074262178245000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| fallback vs gated | final (decisive) | 5 | -0.1087 | -0.1499 | -0.06867 |
| fallback vs gated | equal_interactions | 5 | -0.1087 | -0.1499 | -0.06867 |
| fallback vs gated | equal_time | 0 | - | - | - |
| fallback vs ungated (ablation_arm) | final | 5 | -0.02519 | -0.08337 | 0.04823 |

- note: criterion unmet: baseline 'gated' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['gated_global', 'gated_true']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| fallback | 5/5 | 0 | 0 | 0.904 | 0.1018 |
| gated | 5/5 | 0 | 0 | 0.7952 | 0.1173 |
| ungated | 5/5 | 0 | 0 | 0.8788 | 0.0794 |
| gated_true | 5/5 | 0 | 0 | 0.6724 | 0.09728 |
| gated_global | 5/5 | 0 | 0 | 0.863 | 0.09994 |

Per-run outcomes (E11b):

| arm | seed | status | wall s | mean_dist | frac_steps_in_mud | planner_steps_in_mud | planner_entries_into_mud | gate_closes | planner_share |
|---|---|---|---|---|---|---|---|---|---|
| fallback | 0 | ok | 0.00 | 0.7467 | 0.2375 | 0 | 0 | 0 | 0 |
| fallback | 1 | ok | 0.00 | 0.9167 | 0.4167 | 0 | 0 | 0 | 0 |
| fallback | 2 | ok | 0.00 | 0.9728 | 0.3042 | 0 | 0 | 0 | 0 |
| fallback | 3 | ok | 0.00 | 0.8749 | 0.3875 | 0 | 0 | 0 | 0 |
| fallback | 4 | ok | 0.00 | 1.009 | 0.4396 | 0 | 0 | 0 | 0 |
| gated | 0 | ok | 0.00 | 0.6422 | 0.2104 | 1 | 1 | 1 | 0.6979 |
| gated | 1 | ok | 0.00 | 0.7609 | 0.2271 | 1 | 1 | 1 | 0.7 |
| gated | 2 | ok | 0.00 | 0.8956 | 0.2354 | 3 | 2 | 4 | 0.6646 |
| gated | 3 | ok | 0.00 | 0.7478 | 0.2687 | 1 | 1 | 1 | 0.6708 |
| gated | 4 | ok | 0.00 | 0.9297 | 0.2958 | 2 | 2 | 2 | 0.6396 |
| gated_global | 0 | ok | 0.00 | 0.707 | 0.3146 | 6 | 3 | 6 | 0.5542 |
| gated_global | 1 | ok | 0.00 | 0.8373 | 0.4146 | 7 | 5 | 7 | 0.4521 |
| gated_global | 2 | ok | 0.00 | 0.957 | 0.4021 | 7 | 5 | 6 | 0.475 |
| gated_global | 3 | ok | 0.00 | 0.8736 | 0.4667 | 5 | 4 | 5 | 0.4146 |
| gated_global | 4 | ok | 0.00 | 0.9404 | 0.4479 | 7 | 7 | 7 | 0.4021 |
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
| gated | 0.7952 | 0.2475 | 1.6 | 1.4 | 1.8 | 0.6746 |
| ungated | 0.8788 | 0.4371 | 203.2 | 7.6 | 0 | 0.9979 |
| gated_true | 0.6724 | 0.2358 | 110.2 | 3.6 | 0 | 0.9938 |
| gated_global | 0.863 | 0.4092 | 6.4 | 4.8 | 6.2 | 0.4596 |

## E11c

### E11c-deadline-heavy-tailed-model

- **verdict: reject** — CI upper bound 0.001556 < delta 0.01: the hypothesised effect is ruled out
- hypothesis: with a model whose call latency is heavy-tailed (2% of calls take 30 ms) under a 10 ms deadline, >= 1% more decisions take over TWICE the deadline than with a constant-latency model of equal mean: the 'hard' deadline is bounded only by one chunk's worst case
- primary `overrun_fraction` (higher better), delta 0.01, basis `final`; baseline `constant`, candidate `heavy_tail`, ablation `heavy_tail_chunk8`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `b02fc1205e18`, frozen 1791071311533965000 ns; first result 1791074263272672000 ns
- report: `reports/postfix/E11c-deadline-heavy-tailed-model-b02fc1205e-1791074278013865000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| heavy_tail vs constant | final (decisive) | 5 | -0.002667 | -0.006889 | 0.001556 |
| heavy_tail vs constant | equal_interactions | 5 | -0.002667 | -0.006889 | 0.001556 |
| heavy_tail vs constant | equal_time | 0 | - | - | - |
| heavy_tail vs heavy_tail_chunk8 (ablation_arm) | final | 5 | 0 | 0 | 0 |

- note: criterion unmet: baseline 'constant' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| constant | 5/5 | 0 | 0 | 0.002667 | 0.003651 |
| heavy_tail | 5/5 | 0 | 0 | 0 | 0 |
| heavy_tail_chunk8 | 5/5 | 0 | 0 | 0 | 0 |

Per-run outcomes (E11c):

| arm | seed | status | wall s | overrun_fraction | late_fraction | max_latency_over_deadline | p99_latency_ms | median_latency_ms | planner_fraction | timeout_fraction | probes | late_chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| constant | 0 | ok | 1.09 | 0.006667 | 0.08 | 4.131 | 11.17 | 6.177 | 0.9733 | 0.02667 | 0 | 5 |
| constant | 1 | ok | 1.22 | 0 | 0.06667 | 1.957 | 16.03 | 8.54 | 0.9067 | 0.09333 | 1 | 7 |
| constant | 2 | ok | 1.16 | 0.006667 | 0.08 | 2.456 | 15.47 | 8.623 | 0.9467 | 0.05333 | 0 | 10 |
| constant | 3 | ok | 1.19 | 0 | 0.09333 | 1.641 | 14.8 | 8.742 | 0.92 | 0.08 | 0 | 12 |
| constant | 4 | ok | 1.35 | 0 | 0.01333 | 1.009 | 9.974 | 8.946 | 1 | 0 | 0 | 2 |
| heavy_tail | 0 | ok | 0.74 | 0 | 0.1133 | 1.077 | 10.43 | 2.032 | 0.7 | 0.3 | 0 | 15 |
| heavy_tail | 1 | ok | 0.80 | 0 | 0.1133 | 1.041 | 10.35 | 2.514 | 0.6667 | 0.3333 | 0 | 17 |
| heavy_tail | 2 | ok | 0.93 | 0 | 0.18 | 1.177 | 11.34 | 7.314 | 0.6467 | 0.3533 | 0 | 23 |
| heavy_tail | 3 | ok | 0.67 | 0 | 0.1 | 1.026 | 10.24 | 1.473 | 0.7267 | 0.2733 | 0 | 15 |
| heavy_tail | 4 | ok | 0.79 | 0 | 0.1333 | 1.314 | 10.81 | 1.552 | 0.6533 | 0.3467 | 0 | 17 |
| heavy_tail_chunk8 | 0 | ok | 0.02 | 0 | 0.006667 | 1.033 | 0.08367 | 0.05715 | 0 | 1 | 0 | 1 |
| heavy_tail_chunk8 | 1 | ok | 1.46 | 0 | 0.2533 | 1.066 | 10.58 | 9.617 | 0.3933 | 0.6067 | 0 | 36 |
| heavy_tail_chunk8 | 2 | ok | 1.48 | 0 | 0.28 | 1.676 | 12.06 | 9.74 | 0.4267 | 0.5733 | 1 | 37 |
| heavy_tail_chunk8 | 3 | ok | 1.47 | 0 | 0.2533 | 1.035 | 10.34 | 9.689 | 0.3733 | 0.6267 | 0 | 35 |
| heavy_tail_chunk8 | 4 | ok | 1.46 | 0 | 0.2467 | 1.818 | 10.94 | 9.619 | 0.3933 | 0.6067 | 0 | 35 |

Per-arm means:

| arm | overrun_fraction | late_fraction | max_latency_over_deadline | p99_latency_ms | median_latency_ms | planner_fraction | timeout_fraction | probes | late_chunks |
|---|---|---|---|---|---|---|---|---|---|
| constant | 0.002667 | 0.06667 | 2.239 | 13.49 | 8.206 | 0.9493 | 0.05067 | 0.2 | 7.2 |
| heavy_tail | 0 | 0.128 | 1.127 | 10.63 | 2.977 | 0.6787 | 0.3213 | 0 | 17.4 |
| heavy_tail_chunk8 | 0 | 0.208 | 1.326 | 8.799 | 7.744 | 0.3173 | 0.6827 | 0.2 | 28.8 |

## E11d

### E11d-offprobe-false-duplicate

- **verdict: accept** — mean improvement 0.92 >= delta 0.5, CI [0.825, 0.9992] excludes 0
- hypothesis: two skills that differ only outside the probe distribution (argmax permuted when any |s_i| > 7) are flagged duplicates at a rate >= 0.5 above that of an unrelated same-name policy
- primary `flag_rate` (higher better), delta 0.5, basis `final`; baseline `samename_diff`, candidate `offprobe`, ablation `name_only`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `7bdc16f4e89a`, frozen 1791071311534130000 ns; first result 1791074278165309000 ns
- report: `reports/postfix/E11d-offprobe-false-duplicate-7bdc16f4e8-1791074284530391000.json`

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
| drift_0.001 | 0 | ok | 0.15 | 1 | 3.084e-07 | 4.657e-07 | 0.000375 | 0.05 |
| drift_0.001 | 1 | ok | 0.14 | 1 | 3.016e-07 | 6.09e-07 | 0.000875 | 0.2 |
| drift_0.001 | 2 | ok | 0.15 | 1 | 3.937e-07 | 8.661e-07 | 0.0005 | 0.1 |
| drift_0.001 | 3 | ok | 0.17 | 1 | 4.063e-07 | 7.55e-07 | 0.000625 | 0 |
| drift_0.001 | 4 | ok | 0.23 | 1 | 4.408e-07 | 8.914e-07 | 0.001375 | 0.05 |
| drift_0.01 | 0 | ok | 0.15 | 1 | 3.078e-05 | 4.619e-05 | 0.007875 | 0.05 |
| drift_0.01 | 1 | ok | 0.14 | 1 | 3.013e-05 | 5.976e-05 | 0.009 | 0.2 |
| drift_0.01 | 2 | ok | 0.16 | 1 | 3.945e-05 | 8.746e-05 | 0.009375 | 0.1 |
| drift_0.01 | 3 | ok | 0.16 | 1 | 4.075e-05 | 7.648e-05 | 0.006125 | 0 |
| drift_0.01 | 4 | ok | 0.36 | 1 | 4.409e-05 | 8.883e-05 | 0.00975 | 0.05 |
| drift_0.1 | 0 | ok | 0.15 | 0 | 0.003007 | 0.004346 | 0.07425 | 0.05 |
| drift_0.1 | 1 | ok | 0.15 | 0 | 0.002984 | 0.004903 | 0.07938 | 0.2 |
| drift_0.1 | 2 | ok | 0.16 | 0 | 0.004007 | 0.009433 | 0.09375 | 0.1 |
| drift_0.1 | 3 | ok | 0.17 | 0 | 0.00413 | 0.008217 | 0.09112 | 0 |
| drift_0.1 | 4 | ok | 0.22 | 0 | 0.004346 | 0.008447 | 0.09963 | 0.05 |
| name_only | 0 | ok | 0.15 | 1 | 0 | 0 | 0 | 0.05 |
| name_only | 1 | ok | 0.16 | 1 | 0 | 0 | 0 | 0.2 |
| name_only | 2 | ok | 0.16 | 1 | 0 | 0 | 0 | 0.1 |
| name_only | 3 | ok | 0.23 | 1 | 0 | 0 | 0 | 0 |
| name_only | 4 | ok | 0.21 | 1 | 0 | 0 | 0 | 0.05 |
| offprobe | 0 | ok | 0.24 | 0.95 | 0.0008436 | 0.01687 | 0.2886 | 0.05 |
| offprobe | 1 | ok | 0.16 | 0.8 | 0.00298 | 0.01987 | 0.2904 | 0.2 |
| offprobe | 2 | ok | 0.17 | 0.9 | 0.001906 | 0.02671 | 0.2863 | 0.1 |
| offprobe | 3 | ok | 0.21 | 1 | 0 | 0 | 0.2879 | 0 |
| offprobe | 4 | ok | 0.24 | 0.95 | 0.0008918 | 0.01784 | 0.2814 | 0.05 |
| samename_diff | 0 | ok | 0.20 | 0 | 0.3449 | 0.4825 | 0.7771 | 0.05 |
| samename_diff | 1 | ok | 0.15 | 0 | 0.3215 | 0.4402 | 0.762 | 0.2 |
| samename_diff | 2 | ok | 0.16 | 0 | 0.319 | 0.4412 | 0.788 | 0.1 |
| samename_diff | 3 | ok | 0.19 | 0 | 0.3274 | 0.5139 | 0.7897 | 0 |
| samename_diff | 4 | ok | 0.40 | 0 | 0.3334 | 0.4434 | 0.7886 | 0.05 |
| tempered | 0 | ok | 0.16 | 0 | 0.1009 | 0.1237 | 0 | 0.05 |
| tempered | 1 | ok | 0.14 | 0 | 0.1025 | 0.1208 | 0 | 0.2 |
| tempered | 2 | ok | 0.15 | 0 | 0.1052 | 0.1293 | 0 | 0.1 |
| tempered | 3 | ok | 0.16 | 0 | 0.09949 | 0.1136 | 0 | 0 |
| tempered | 4 | ok | 0.18 | 0 | 0.1025 | 0.1202 | 0 | 0.05 |

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

## E11e

### E11e-tempered-copy-not-flagged

- **verdict: accept** — mean improvement 1 >= delta 0.5, CI [1, 1] excludes 0
- hypothesis: a copy with logits x3 (identical argmax on every state, so identical deterministic behaviour) is flagged at a rate >= 0.5 BELOW a byte copy
- primary `flag_rate` (lower better), delta 0.5, basis `final`; baseline `name_only`, candidate `tempered`, ablation `samename_diff`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `68ea5467a4c2`, frozen 1791071311534210000 ns; first result 1791074284536433000 ns
- report: `reports/postfix/E11e-tempered-copy-not-flagged-68ea5467a4-1791074284536997000.json`

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

## E11d+evidence

### E11d-offprobe-false-duplicate

- **verdict: reject** — CI upper bound 0 < delta 0.5: the hypothesised effect is ruled out
- hypothesis: two skills that differ only outside the probe distribution (argmax permuted when any |s_i| > 7) are flagged duplicates at a rate >= 0.5 above that of an unrelated same-name policy
- primary `flag_rate` (higher better), delta 0.5, basis `final`; baseline `samename_diff`, candidate `offprobe`, ablation `name_only`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `7bdc16f4e89a`, frozen 1791071311534130000 ns; first result 1791074284807479000 ns
- report: `reports/postfix/E11d-offprobe-false-duplicate-7bdc16f4e8-1791074292689299000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| offprobe vs samename_diff | final (decisive) | 5 | 0 | 0 | 0 |
| offprobe vs samename_diff | equal_interactions | 5 | 0 | 0 | 0 |
| offprobe vs samename_diff | equal_time | 0 | - | - | - |
| offprobe vs name_only (ablation_arm) | final | 5 | -1 | -1 | -1 |

- note: criterion unmet: baseline 'samename_diff' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['drift_0.001', 'drift_0.01', 'drift_0.1', 'offprobe_inside', 'tempered']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| name_only | 5/5 | 0 | 0 | 1 | 0 |
| samename_diff | 5/5 | 0 | 0 | 0 | 0 |
| offprobe | 5/5 | 0 | 0 | 0 | 0 |
| tempered | 5/5 | 0 | 0 | 0 | 0 |
| drift_0.001 | 5/5 | 0 | 0 | 1 | 0 |
| drift_0.01 | 5/5 | 0 | 0 | 1 | 0 |
| drift_0.1 | 5/5 | 0 | 0 | 0 | 0 |
| offprobe_inside | 5/5 | 0 | 0 | 0.69 | 0.08944 |

Per-run outcomes (E11d):

| arm | seed | status | wall s | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|---|---|---|
| drift_0.001 | 0 | ok | 0.20 | 1 | 4.908e-07 | 7.53e-07 | 0.000375 | 1 |
| drift_0.001 | 1 | ok | 0.25 | 1 | 4.953e-07 | 7.842e-07 | 0.000875 | 1 |
| drift_0.001 | 2 | ok | 0.19 | 1 | 5.744e-07 | 1.159e-06 | 0.0005 | 1 |
| drift_0.001 | 3 | ok | 0.22 | 1 | 6.994e-07 | 1.636e-06 | 0.000625 | 1 |
| drift_0.001 | 4 | ok | 0.19 | 1 | 6.242e-07 | 9.37e-07 | 0.001375 | 1 |
| drift_0.01 | 0 | ok | 0.20 | 1 | 4.908e-05 | 7.607e-05 | 0.007875 | 1 |
| drift_0.01 | 1 | ok | 0.22 | 1 | 4.956e-05 | 7.823e-05 | 0.009 | 1 |
| drift_0.01 | 2 | ok | 0.19 | 1 | 5.757e-05 | 0.0001169 | 0.009375 | 1 |
| drift_0.01 | 3 | ok | 0.20 | 1 | 6.989e-05 | 0.0001608 | 0.006125 | 1 |
| drift_0.01 | 4 | ok | 0.19 | 1 | 6.233e-05 | 9.395e-05 | 0.00975 | 1 |
| drift_0.1 | 0 | ok | 0.20 | 0 | 0.004732 | 0.007144 | 0.07425 | 1 |
| drift_0.1 | 1 | ok | 0.20 | 0 | 0.004838 | 0.007025 | 0.07938 | 1 |
| drift_0.1 | 2 | ok | 0.19 | 0 | 0.005732 | 0.01232 | 0.09375 | 1 |
| drift_0.1 | 3 | ok | 0.19 | 0 | 0.006672 | 0.01256 | 0.09112 | 1 |
| drift_0.1 | 4 | ok | 0.19 | 0 | 0.005939 | 0.01026 | 0.09963 | 1 |
| name_only | 0 | ok | 0.27 | 1 | 0 | 0 | 0 | 1 |
| name_only | 1 | ok | 0.20 | 1 | 0 | 0 | 0 | 1 |
| name_only | 2 | ok | 0.19 | 1 | 0 | 0 | 0 | 1 |
| name_only | 3 | ok | 0.19 | 1 | 0 | 0 | 0 | 1 |
| name_only | 4 | ok | 0.18 | 1 | 0 | 0 | 0 | 1 |
| offprobe | 0 | ok | 0.24 | 0 | 0.1223 | 0.1642 | 0.2886 | 1 |
| offprobe | 1 | ok | 0.22 | 0 | 0.1089 | 0.1711 | 0.2904 | 1 |
| offprobe | 2 | ok | 0.22 | 0 | 0.1009 | 0.1497 | 0.2863 | 1 |
| offprobe | 3 | ok | 0.21 | 0 | 0.1204 | 0.1788 | 0.2879 | 1 |
| offprobe | 4 | ok | 0.20 | 0 | 0.1128 | 0.1556 | 0.2814 | 1 |
| offprobe_inside | 0 | ok | 0.21 | 0.75 | 0.0006655 | 0.003375 | 0.2886 | 1 |
| offprobe_inside | 1 | ok | 0.22 | 0.6 | 0.001336 | 0.00543 | 0.2904 | 1 |
| offprobe_inside | 2 | ok | 0.21 | 0.8 | 0.0006418 | 0.005342 | 0.2863 | 1 |
| offprobe_inside | 3 | ok | 0.21 | 0.6 | 0.001611 | 0.008145 | 0.2879 | 1 |
| offprobe_inside | 4 | ok | 0.21 | 0.7 | 0.001409 | 0.007556 | 0.2814 | 1 |
| samename_diff | 0 | ok | 0.24 | 0 | 0.3929 | 0.5271 | 0.7771 | 1 |
| samename_diff | 1 | ok | 0.19 | 0 | 0.3671 | 0.4989 | 0.762 | 1 |
| samename_diff | 2 | ok | 0.19 | 0 | 0.355 | 0.4948 | 0.788 | 1 |
| samename_diff | 3 | ok | 0.19 | 0 | 0.3631 | 0.5001 | 0.7897 | 1 |
| samename_diff | 4 | ok | 0.18 | 0 | 0.3703 | 0.4892 | 0.7886 | 1 |
| tempered | 0 | ok | 0.21 | 0 | 0.09537 | 0.1102 | 0 | 1 |
| tempered | 1 | ok | 0.21 | 0 | 0.09635 | 0.12 | 0 | 1 |
| tempered | 2 | ok | 0.19 | 0 | 0.1012 | 0.1172 | 0 | 1 |
| tempered | 3 | ok | 0.19 | 0 | 0.09739 | 0.1118 | 0 | 1 |
| tempered | 4 | ok | 0.18 | 0 | 0.09956 | 0.1147 | 0 | 1 |

Per-arm means:

| arm | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|
| name_only | 1 | 0 | 0 | 0 | 1 |
| samename_diff | 0 | 0.3697 | 0.502 | 0.7811 | 1 |
| offprobe | 0 | 0.1131 | 0.1639 | 0.2869 | 1 |
| tempered | 0 | 0.09798 | 0.1148 | 0 | 1 |
| drift_0.001 | 1 | 5.768e-07 | 1.054e-06 | 0.00075 | 1 |
| drift_0.01 | 1 | 5.769e-05 | 0.0001052 | 0.008425 | 1 |
| drift_0.1 | 0 | 0.005583 | 0.009862 | 0.08763 | 1 |
| offprobe_inside | 0.69 | 0.001133 | 0.005969 | 0.2869 | 1 |

## E11e+evidence

### E11e-tempered-copy-not-flagged

- **verdict: accept** — mean improvement 1 >= delta 0.5, CI [1, 1] excludes 0
- hypothesis: a copy with logits x3 (identical argmax on every state, so identical deterministic behaviour) is flagged at a rate >= 0.5 BELOW a byte copy
- primary `flag_rate` (lower better), delta 0.5, basis `final`; baseline `name_only`, candidate `tempered`, ablation `samename_diff`, alternative `None`; seeds [0, 1, 2, 3, 4]
- prereg hash `68ea5467a4c2`, frozen 1791071311534210000 ns; first result 1791074292693623000 ns
- report: `reports/postfix/E11e-tempered-copy-not-flagged-68ea5467a4-1791074292694101000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| tempered vs name_only | final (decisive) | 5 | 1 | 1 | 1 |
| tempered vs name_only | equal_interactions | 5 | 1 | 1 | 1 |
| tempered vs name_only | equal_time | 0 | - | - | - |
| tempered vs samename_diff (ablation_arm) | final | 5 | 0 | 0 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['drift_0.001', 'drift_0.01', 'drift_0.1', 'offprobe', 'offprobe_inside']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| name_only | 5/5 | 0 | 0 | 1 | 0 |
| samename_diff | 5/5 | 0 | 0 | 0 | 0 |
| offprobe | 5/5 | 0 | 0 | 0 | 0 |
| tempered | 5/5 | 0 | 0 | 0 | 0 |
| drift_0.001 | 5/5 | 0 | 0 | 1 | 0 |
| drift_0.01 | 5/5 | 0 | 0 | 1 | 0 |
| drift_0.1 | 5/5 | 0 | 0 | 0 | 0 |
| offprobe_inside | 5/5 | 0 | 0 | 0.69 | 0.08944 |

Per-run outcomes (E11e):

| arm | seed | status | wall s | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|---|---|---|
| drift_0.001 | 0 | ok | 0.00 | 1 | 4.908e-07 | 7.53e-07 | 0.000375 | 1 |
| drift_0.001 | 1 | ok | 0.00 | 1 | 4.953e-07 | 7.842e-07 | 0.000875 | 1 |
| drift_0.001 | 2 | ok | 0.00 | 1 | 5.744e-07 | 1.159e-06 | 0.0005 | 1 |
| drift_0.001 | 3 | ok | 0.00 | 1 | 6.994e-07 | 1.636e-06 | 0.000625 | 1 |
| drift_0.001 | 4 | ok | 0.00 | 1 | 6.242e-07 | 9.37e-07 | 0.001375 | 1 |
| drift_0.01 | 0 | ok | 0.00 | 1 | 4.908e-05 | 7.607e-05 | 0.007875 | 1 |
| drift_0.01 | 1 | ok | 0.00 | 1 | 4.956e-05 | 7.823e-05 | 0.009 | 1 |
| drift_0.01 | 2 | ok | 0.00 | 1 | 5.757e-05 | 0.0001169 | 0.009375 | 1 |
| drift_0.01 | 3 | ok | 0.00 | 1 | 6.989e-05 | 0.0001608 | 0.006125 | 1 |
| drift_0.01 | 4 | ok | 0.00 | 1 | 6.233e-05 | 9.395e-05 | 0.00975 | 1 |
| drift_0.1 | 0 | ok | 0.00 | 0 | 0.004732 | 0.007144 | 0.07425 | 1 |
| drift_0.1 | 1 | ok | 0.00 | 0 | 0.004838 | 0.007025 | 0.07938 | 1 |
| drift_0.1 | 2 | ok | 0.00 | 0 | 0.005732 | 0.01232 | 0.09375 | 1 |
| drift_0.1 | 3 | ok | 0.00 | 0 | 0.006672 | 0.01256 | 0.09112 | 1 |
| drift_0.1 | 4 | ok | 0.00 | 0 | 0.005939 | 0.01026 | 0.09963 | 1 |
| name_only | 0 | ok | 0.00 | 1 | 0 | 0 | 0 | 1 |
| name_only | 1 | ok | 0.00 | 1 | 0 | 0 | 0 | 1 |
| name_only | 2 | ok | 0.00 | 1 | 0 | 0 | 0 | 1 |
| name_only | 3 | ok | 0.00 | 1 | 0 | 0 | 0 | 1 |
| name_only | 4 | ok | 0.00 | 1 | 0 | 0 | 0 | 1 |
| offprobe | 0 | ok | 0.00 | 0 | 0.1223 | 0.1642 | 0.2886 | 1 |
| offprobe | 1 | ok | 0.00 | 0 | 0.1089 | 0.1711 | 0.2904 | 1 |
| offprobe | 2 | ok | 0.00 | 0 | 0.1009 | 0.1497 | 0.2863 | 1 |
| offprobe | 3 | ok | 0.00 | 0 | 0.1204 | 0.1788 | 0.2879 | 1 |
| offprobe | 4 | ok | 0.00 | 0 | 0.1128 | 0.1556 | 0.2814 | 1 |
| offprobe_inside | 0 | ok | 0.00 | 0.75 | 0.0006655 | 0.003375 | 0.2886 | 1 |
| offprobe_inside | 1 | ok | 0.00 | 0.6 | 0.001336 | 0.00543 | 0.2904 | 1 |
| offprobe_inside | 2 | ok | 0.00 | 0.8 | 0.0006418 | 0.005342 | 0.2863 | 1 |
| offprobe_inside | 3 | ok | 0.00 | 0.6 | 0.001611 | 0.008145 | 0.2879 | 1 |
| offprobe_inside | 4 | ok | 0.00 | 0.7 | 0.001409 | 0.007556 | 0.2814 | 1 |
| samename_diff | 0 | ok | 0.00 | 0 | 0.3929 | 0.5271 | 0.7771 | 1 |
| samename_diff | 1 | ok | 0.00 | 0 | 0.3671 | 0.4989 | 0.762 | 1 |
| samename_diff | 2 | ok | 0.00 | 0 | 0.355 | 0.4948 | 0.788 | 1 |
| samename_diff | 3 | ok | 0.00 | 0 | 0.3631 | 0.5001 | 0.7897 | 1 |
| samename_diff | 4 | ok | 0.00 | 0 | 0.3703 | 0.4892 | 0.7886 | 1 |
| tempered | 0 | ok | 0.00 | 0 | 0.09537 | 0.1102 | 0 | 1 |
| tempered | 1 | ok | 0.00 | 0 | 0.09635 | 0.12 | 0 | 1 |
| tempered | 2 | ok | 0.00 | 0 | 0.1012 | 0.1172 | 0 | 1 |
| tempered | 3 | ok | 0.00 | 0 | 0.09739 | 0.1118 | 0 | 1 |
| tempered | 4 | ok | 0.00 | 0 | 0.09956 | 0.1147 | 0 | 1 |

Per-arm means:

| arm | flag_rate | js_bits_mean | js_bits_max | deploy_argmax_disagreement | probe_sets_touching_region |
|---|---|---|---|---|---|
| name_only | 1 | 0 | 0 | 0 | 1 |
| samename_diff | 0 | 0.3697 | 0.502 | 0.7811 | 1 |
| offprobe | 0 | 0.1131 | 0.1639 | 0.2869 | 1 |
| tempered | 0 | 0.09798 | 0.1148 | 0 | 1 |
| drift_0.001 | 1 | 5.768e-07 | 1.054e-06 | 0.00075 | 1 |
| drift_0.01 | 1 | 5.769e-05 | 0.0001052 | 0.008425 | 1 |
| drift_0.1 | 0 | 0.005583 | 0.009862 | 0.08763 | 1 |
| offprobe_inside | 0.69 | 0.001133 | 0.005969 | 0.2869 | 1 |

## E11c latency distribution (decide() latency / 10 ms deadline; 5 seeds x 150 decisions; measurement, not an A/B)

| model | > 1x | > 1.25x | > 2x | p50 | p99 | max | planner | fallback_timeout |
|---|---|---|---|---|---|---|---|---|
| constant | 4.7% | 0.4% | 0.3% | 0.61 | 1.02 | 3.51 | 99.5% | 0.5% |
| heavy_tail | 12.5% | 0.0% | 0.0% | 0.27 | 1.04 | 1.06 | 67.5% | 32.5% |
| heavy_tail_chunk8 | 22.8% | 0.3% | 0.0% | 0.95 | 1.06 | 1.67 | 31.7% | 68.3% |


Total wall 85s
