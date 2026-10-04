## E1 (447s wall)

### E1a-IN-vs-flopmatched-MLP-nll1

- **verdict: accept** — mean improvement 2.757 >= delta 0.05, CI [1.708, 3.714] excludes 0
- hypothesis: At equal FLOPs/sample, equal data and equal epochs, the relational IN has lower held-out 1-step Gaussian NLL (ball dims) than an MLP on the same structured inputs
- primary `nll1` (lower better), delta 0.05, basis `final`; baseline `mlp_flops`, candidate `in`, ablation `in_noedges`, alternative `mlp128`; seeds [11, 12, 13, 14, 15]
- prereg hash `6b41c0914428`, frozen 1791071610496847000 ns; first result 1791071628751179000 ns
- report: `reports/full/E1a-IN-vs-flopmatched-MLP-nll1-6b41c09144-1791072057353761000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| in vs mlp_flops | final (decisive) | 5 | 2.757 | 1.708 | 3.714 | 0 |
| in vs mlp_flops | equal_interactions | 5 | 2.757 | 1.708 | 3.714 | 0 |
| in vs mlp_flops | equal_time | 0 | - | - | - | 5 |
| in vs in_noedges (ablation_arm) | final | 5 | 0.2685 | -0.343 | 0.8049 | 0 |
| in vs mlp128 (alternative_arm) | final | 5 | 3.042 | 1.361 | 4.722 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['in_norel']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| in | 5/5 | 0 | 0 | -0.02328 | 0.4986 |
| mlp_flops | 5/5 | 0 | 0 | 2.734 | 1.177 |
| mlp128 | 5/5 | 0 | 0 | 3.018 | 1.5 |
| in_noedges | 5/5 | 0 | 0 | 0.2452 | 0.1829 |
| in_norel | 5/5 | 0 | 0 | 1.074 | 1.082 |

### E1b-IN-vs-flopmatched-MLP-rmse10

- **verdict: inconclusive** — CI [-0.07053, 0.08366] with mean 0.002253 neither meets nor rules out delta 0.05
- hypothesis: At equal FLOPs/sample, the IN has lower 10-step mean-rollout RMSE (ball dims) than the FLOP-matched MLP
- primary `rmse10` (lower better), delta 0.05, basis `final`; baseline `mlp_flops`, candidate `in`, ablation `in_noedges`, alternative `mlp128`; seeds [11, 12, 13, 14, 15]
- prereg hash `d129b7e963f9`, frozen 1791071610496944000 ns; first result 1791072057359278000 ns
- report: `reports/full/E1b-IN-vs-flopmatched-MLP-rmse10-d129b7e963-1791072057360142000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| in vs mlp_flops | final (decisive) | 5 | 0.002253 | -0.07053 | 0.08366 | 0 |
| in vs mlp_flops | equal_interactions | 5 | 0.002253 | -0.07053 | 0.08366 | 0 |
| in vs mlp_flops | equal_time | 0 | - | - | - | 5 |
| in vs in_noedges (ablation_arm) | final | 5 | 0.2276 | 0.0624 | 0.4064 | 0 |
| in vs mlp128 (alternative_arm) | final | 5 | 0.3543 | 0.2552 | 0.4632 | 0 |

- note: criterion unmet: baseline 'mlp_flops' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['in_norel']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| in | 5/5 | 0 | 0 | 2.669 | 0.2337 |
| mlp_flops | 5/5 | 0 | 0 | 2.671 | 0.1899 |
| mlp128 | 5/5 | 0 | 0 | 3.023 | 0.1907 |
| in_noedges | 5/5 | 0 | 0 | 2.896 | 0.1285 |
| in_norel | 5/5 | 0 | 0 | 3.064 | 0.236 |

### E1c-IN-vs-MLP128-equal-time-nll1

- **verdict: inconclusive** — CI [-3.393, 133.5] with mean 45.3 neither meets nor rules out delta 0.05
- hypothesis: At equal TRAINING WALL-CLOCK, the IN has lower held-out 1-step NLL than the mechanisms agent's MLP-128 given as many epochs as fit
- primary `nll1` (lower better), delta 0.05, basis `equal_time`; baseline `mlp128`, candidate `in`, ablation `in_noedges`, alternative `mlp_flops`; seeds [11, 12, 13, 14, 15]
- prereg hash `2b811318574e`, frozen 1791071610497008000 ns; first result 1791072057364103000 ns
- report: `reports/full/E1c-IN-vs-MLP128-equal-time-nll1-2b81131857-1791072057364929000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| in vs mlp128 | final | 5 | 3.042 | 1.361 | 4.813 | 0 |
| in vs mlp128 | equal_interactions | 5 | 3.042 | 1.361 | 4.813 | 0 |
| in vs mlp128 | equal_time (decisive) | 5 | 45.3 | -3.393 | 133.5 | 0 |
| in vs in_noedges (ablation_arm) | equal_time | 0 | - | - | - | 5 |
| in vs mlp_flops (alternative_arm) | equal_time | 0 | - | - | - | 5 |

- note: criterion unmet: baseline 'mlp128' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['in_norel']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| in | 5/5 | 0 | 0 | -0.02328 | 0.4986 |
| mlp_flops | 5/5 | 0 | 0 | 2.734 | 1.177 |
| mlp128 | 5/5 | 0 | 0 | 3.018 | 1.5 |
| in_noedges | 5/5 | 0 | 0 | 0.2452 | 0.1829 |
| in_norel | 5/5 | 0 | 0 | 1.074 | 1.082 |

### E1 per-run outcomes (shared by every E1 prereg)

| arm | seed | status | data_fp | nll1 | rmse1 | cov90_1 | nll_event | rmse10 | nll10 | cov90_10 | rmse1_contact | rmse1_free | flops_per_sample | params | train_seconds | predict_ms_per64 | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in | 11 | ok | 48e64930ceb56c2f | -0.5353 | 1.397 | 0.9334 | 0.3565 | 2.3 | 31.25 | 0.9225 | 2.332 | 0.3926 | 1076544 | 20507 | 13.54 | 3 | 60 |
| in | 12 | ok | 3665a578c80e2c5d | -0.3568 | 1.364 | 0.8638 | 0.3301 | 2.744 | 57.36 | 0.8492 | 2.272 | 0.3875 | 1076544 | 20507 | 16.12 | 3.752 | 60 |
| in | 13 | ok | 26026ec2a9554a35 | 0.4964 | 1.356 | 0.9426 | 0.4307 | 2.591 | 27.57 | 0.9183 | 2.194 | 0.4159 | 1076544 | 20507 | 7.393 | 2.204 | 60 |
| in | 14 | ok | 05adfad483698ec7 | 0.5256 | 1.57 | 0.9024 | 0.3758 | 2.827 | 82.12 | 0.8583 | 2.489 | 0.4412 | 1076544 | 20507 | 6.927 | 1.96 | 60 |
| in | 15 | ok | c9b331240ce9067f | -0.2463 | 1.569 | 0.8746 | 0.3459 | 2.882 | 31.68 | 0.9004 | 2.563 | 0.2783 | 1076544 | 20507 | 7.824 | 2.506 | 60 |
| in_noedges | 11 | ok | 48e64930ceb56c2f | 0.309 | 1.56 | 0.9321 | 0.5081 | 2.699 | 363.8 | 0.7588 | 2.567 | 0.5409 | 149856 | 20507 | 6.556 | 2.419 | 60 |
| in_noedges | 12 | ok | 3665a578c80e2c5d | 0.02484 | 1.552 | 0.9463 | 0.4974 | 2.936 | 202 | 0.7462 | 2.475 | 0.6972 | 149856 | 20507 | 3.459 | 0.8542 | 60 |
| in_noedges | 13 | ok | 26026ec2a9554a35 | 0.144 | 1.507 | 0.9315 | 0.5184 | 2.925 | 262.3 | 0.7887 | 2.434 | 0.4761 | 149856 | 20507 | 3.465 | 1.024 | 60 |
| in_noedges | 14 | ok | 05adfad483698ec7 | 0.5115 | 1.762 | 0.9325 | 0.5639 | 2.869 | 310.9 | 0.7033 | 2.61 | 0.9205 | 149856 | 20507 | 2.999 | 0.991 | 60 |
| in_noedges | 15 | ok | c9b331240ce9067f | 0.2368 | 1.659 | 0.9399 | 0.5439 | 3.051 | 171.2 | 0.7929 | 2.59 | 0.6783 | 149856 | 20507 | 2.517 | 0.9538 | 60 |
| in_norel | 11 | ok | 48e64930ceb56c2f | 0.2793 | 1.563 | 0.9413 | 0.4995 | 2.656 | 215.3 | 0.7908 | 2.541 | 0.6118 | 1076544 | 20507 | 19.04 | 5.83 | 60 |
| in_norel | 12 | ok | 3665a578c80e2c5d | 0.3258 | 1.561 | 0.926 | 0.5035 | 3.15 | 249.7 | 0.7717 | 2.535 | 0.611 | 1076544 | 20507 | 10.03 | 2.765 | 60 |
| in_norel | 13 | ok | 26026ec2a9554a35 | 0.9364 | 1.51 | 0.8864 | 0.5279 | 3.086 | 219.9 | 0.795 | 2.408 | 0.5594 | 1076544 | 20507 | 10.56 | 1.994 | 60 |
| in_norel | 14 | ok | 05adfad483698ec7 | 2.928 | 1.751 | 0.9109 | 0.57 | 3.178 | 143.9 | 0.7783 | 2.729 | 0.63 | 1076544 | 20507 | 7.289 | 2.067 | 60 |
| in_norel | 15 | ok | c9b331240ce9067f | 0.8991 | 1.673 | 0.9318 | 0.5485 | 3.251 | 120.4 | 0.8179 | 2.601 | 0.7058 | 1076544 | 20507 | 8.572 | 2.629 | 60 |
| mlp128 | 11 | ok | 48e64930ceb56c2f | 3.636 | 1.639 | 0.8495 | 0.4278 | 2.79 | 57.84 | 0.8367 | 2.798 | 0.1868 | 72448 | 36557 | 2.243 | 0.9852 | 60 |
| mlp128 | 12 | ok | 3665a578c80e2c5d | 2.102 | 1.616 | 0.8653 | 0.4324 | 3.114 | 35.45 | 0.8579 | 2.751 | 0.2103 | 72448 | 36557 | 4.16 | 0.5265 | 60 |
| mlp128 | 13 | ok | 26026ec2a9554a35 | 2.157 | 1.576 | 0.8632 | 0.4544 | 2.862 | 56.72 | 0.8542 | 2.616 | 0.2027 | 72448 | 36557 | 2.702 | 0.5681 | 60 |
| mlp128 | 14 | ok | 05adfad483698ec7 | 5.383 | 1.818 | 0.8565 | 0.4765 | 3.102 | 46.98 | 0.8817 | 2.941 | 0.2174 | 72448 | 36557 | 1.476 | 0.479 | 60 |
| mlp128 | 15 | ok | c9b331240ce9067f | 1.814 | 1.736 | 0.8878 | 0.4572 | 3.248 | 14.57 | 0.9113 | 2.852 | 0.2035 | 72448 | 36557 | 1.445 | 0.36 | 60 |
| mlp_flops | 11 | ok | 48e64930ceb56c2f | 2.638 | 1.527 | 0.841 | 0.4379 | 2.395 | 19.21 | 0.8846 | 2.602 | 0.2105 | 1078752 | 540775 | 7.196 | 1.122 | 60 |
| mlp_flops | 12 | ok | 3665a578c80e2c5d | 1.385 | 1.456 | 0.9 | 0.4436 | 2.677 | 15.11 | 0.8883 | 2.457 | 0.3015 | 1078752 | 540775 | 14.15 | 2.156 | 60 |
| mlp_flops | 13 | ok | 26026ec2a9554a35 | 4.16 | 1.474 | 0.8524 | 0.469 | 2.591 | 33.69 | 0.8408 | 2.442 | 0.2293 | 1078752 | 540775 | 5.125 | 0.7268 | 60 |
| mlp_flops | 14 | ok | 05adfad483698ec7 | 3.659 | 1.654 | 0.8709 | 0.5404 | 2.834 | 24.43 | 0.8862 | 2.665 | 0.2718 | 1078752 | 540775 | 5.042 | 0.7124 | 60 |
| mlp_flops | 15 | ok | c9b331240ce9067f | 1.828 | 1.586 | 0.8857 | 0.5078 | 2.858 | 19.97 | 0.8983 | 2.591 | 0.2826 | 1078752 | 540775 | 5.377 | 0.7082 | 60 |

E1 per-arm means over seeds:

| arm | nll1 | rmse1 | cov90_1 | nll_event | rmse10 | nll10 | cov90_10 | rmse1_contact | rmse1_free | flops_per_sample | params | train_seconds | predict_ms_per64 | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in | -0.02328 | 1.451 | 0.9033 | 0.3678 | 2.669 | 46 | 0.8898 | 2.37 | 0.3831 | 1.077e+06 | 2.051e+04 | 10.36 | 2.684 | 60 |
| mlp_flops | 2.734 | 1.539 | 0.87 | 0.4798 | 2.671 | 22.48 | 0.8797 | 2.552 | 0.2591 | 1.079e+06 | 5.408e+05 | 7.377 | 1.085 | 60 |
| mlp128 | 3.018 | 1.677 | 0.8644 | 0.4497 | 3.023 | 42.31 | 0.8683 | 2.792 | 0.2041 | 7.245e+04 | 3.656e+04 | 2.405 | 0.5838 | 60 |
| in_noedges | 0.2452 | 1.608 | 0.9365 | 0.5264 | 2.896 | 262 | 0.758 | 2.535 | 0.6626 | 1.499e+05 | 2.051e+04 | 3.799 | 1.248 | 60 |
| in_norel | 1.074 | 1.612 | 0.9193 | 0.5299 | 3.064 | 189.9 | 0.7907 | 2.563 | 0.6236 | 1.077e+06 | 2.051e+04 | 11.1 | 3.057 | 60 |

## E2 (231s wall)

### E2a-epistemic-has-predictive-value

- **verdict: accept** — mean improvement 0.5852 >= delta 0.1, CI [0.5466, 0.6197] excludes 0
- hypothesis: ens5 epistemic variance rank-correlates with held-out squared error (pooled id + gravity/box shifts) by >= 0.1 more than a permuted (information-free) signal
- primary `unc_err_spearman` (higher better), delta 0.1, basis `final`; baseline `null_perm`, candidate `ens5`, ablation `k1`, alternative `ens5_noboot`; seeds [21, 22, 23, 24, 25]
- prereg hash `7d07b24a616e`, frozen 1791072057369336000 ns; first result 1791072068457825000 ns
- report: `reports/full/E2a-epistemic-has-predictive-value-7d07b24a61-1791072288453204000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| ens5 vs null_perm | final (decisive) | 5 | 0.5852 | 0.5466 | 0.6197 | 0 |
| ens5 vs null_perm | equal_interactions | 5 | 0.5852 | 0.5466 | 0.6197 | 0 |
| ens5 vs null_perm | equal_time | 0 | - | - | - | 5 |
| ens5 vs k1 (ablation_arm) | final | 5 | -0.05688 | -0.1039 | -0.01693 | 0 |
| ens5 vs ens5_noboot (alternative_arm) | final | 5 | -0.01108 | -0.04321 | 0.01836 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['ens5_iid']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.5885 | 0.0304 |
| k1 | 5/5 | 0 | 0 | 0.6454 | 0.0262 |
| null_perm | 5/5 | 0 | 0 | 0.003273 | 0.007567 |
| ens5_noboot | 5/5 | 0 | 0 | 0.5995 | 0.04142 |
| ens5_iid | 5/5 | 0 | 0 | 0.572 | 0.02707 |

### E2b-epistemic-beats-single-model-variance

- **verdict: reject** — CI upper bound -0.01618 < delta 0.05: the hypothesised effect is ruled out
- hypothesis: ens5 epistemic variance predicts error better (Spearman, pooled) than a single model's own predictive variance (K=1)
- primary `unc_err_spearman` (higher better), delta 0.05, basis `final`; baseline `k1`, candidate `ens5`, ablation `ens5_noboot`, alternative `null_perm`; seeds [21, 22, 23, 24, 25]
- prereg hash `709aaeb95837`, frozen 1791072057369387000 ns; first result 1791072288458684000 ns
- report: `reports/full/E2b-epistemic-beats-single-model-variance-709aaeb958-1791072288459600000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| ens5 vs k1 | final (decisive) | 5 | -0.05688 | -0.1032 | -0.01618 | 0 |
| ens5 vs k1 | equal_interactions | 5 | -0.05688 | -0.1032 | -0.01618 | 0 |
| ens5 vs k1 | equal_time | 0 | - | - | - | 5 |
| ens5 vs ens5_noboot (ablation_arm) | final | 5 | -0.01108 | -0.04321 | 0.01836 | 0 |
| ens5 vs null_perm (alternative_arm) | final | 5 | 0.5852 | 0.5469 | 0.6197 | 0 |

- note: criterion unmet: baseline 'k1' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['ens5_iid']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.5885 | 0.0304 |
| k1 | 5/5 | 0 | 0 | 0.6454 | 0.0262 |
| null_perm | 5/5 | 0 | 0 | 0.003273 | 0.007567 |
| ens5_noboot | 5/5 | 0 | 0 | 0.5995 | 0.04142 |
| ens5_iid | 5/5 | 0 | 0 | 0.572 | 0.02707 |

### E2c-coverage-under-shift

- **verdict: accept** — mean improvement 0.02183 >= delta 0.02, CI [0.01157, 0.03267] excludes 0
- hypothesis: the K=5 mixture's 90% intervals stay closer to nominal under shift (mean |cov90-0.9| over g4, g20, box6) than K=1's by >= 0.02
- primary `cov90_shift_abs_err` (lower better), delta 0.02, basis `final`; baseline `k1`, candidate `ens5`, ablation `ens5_noboot`, alternative `ens5_iid`; seeds [21, 22, 23, 24, 25]
- prereg hash `ab100e42ba2c`, frozen 1791072057369427000 ns; first result 1791072288464463000 ns
- report: `reports/full/E2c-coverage-under-shift-ab100e42ba-1791072288465371000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| ens5 vs k1 | final (decisive) | 5 | 0.02183 | 0.01157 | 0.03267 | 0 |
| ens5 vs k1 | equal_interactions | 5 | 0.02183 | 0.01157 | 0.03267 | 0 |
| ens5 vs k1 | equal_time | 0 | - | - | - | 5 |
| ens5 vs ens5_noboot (ablation_arm) | final | 5 | 0.001667 | -0.002292 | 0.006432 | 0 |
| ens5 vs ens5_iid (alternative_arm) | final | 5 | 0.007611 | 0.0003533 | 0.01469 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['null_perm']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.04231 | 0.00729 |
| k1 | 5/5 | 0 | 0 | 0.06415 | 0.01079 |
| null_perm | 5/5 | 0 | 0 | 0.04231 | 0.00729 |
| ens5_noboot | 5/5 | 0 | 0 | 0.04398 | 0.006768 |
| ens5_iid | 5/5 | 0 | 0 | 0.04993 | 0.00582 |

### E2 per-run outcomes (shared by every E2 prereg)

| arm | seed | status | data_fp | unc_err_spearman | spearman_id | spearman_g4 | spearman_g20 | spearman_box6 | cov90_id | cov90_g4 | cov90_g20 | cov90_box6 | cov90_shift_abs_err | nll_id | nll_g4 | nll_g20 | nll_box6 | signal_ratio_g4 | signal_ratio_g20 | signal_ratio_box6 | conf_wrong_frac_id | conf_wrong_frac_g4 | conf_wrong_frac_g20 | conf_wrong_frac_box6 | outside3sd_frac_id | outside3sd_frac_g4 | outside3sd_frac_g20 | outside3sd_frac_box6 | train_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ens5 | 21 | ok | 64680c6639b6d52d | 0.5756 | 0.7104 | 0.5429 | 0.6867 | 0.7288 | 0.9196 | 0.8826 | 0.8501 | 0.9411 | 0.03611 | 1.359 | 0.9918 | 1.696 | 0.4141 | 0.6666 | 1.587 | 1.347 | 0.001667 | 0 | 0 | 0.001667 | 0.05 | 0.05333 | 0.07333 | 0.03167 | 7.108 |
| ens5 | 22 | ok | ce5ea45a545dc1fb | 0.6294 | 0.7091 | 0.641 | 0.6508 | 0.7324 | 0.9243 | 0.8729 | 0.854 | 0.9496 | 0.04088 | 1.555 | 1.975 | 1.731 | 0.397 | 0.6749 | 1.646 | 1.312 | 0 | 0 | 0.001667 | 0 | 0.03833 | 0.05333 | 0.055 | 0.015 | 6.345 |
| ens5 | 23 | ok | 9cac146ced565994 | 0.5465 | 0.697 | 0.5375 | 0.666 | 0.6617 | 0.9199 | 0.8796 | 0.8485 | 0.9499 | 0.0406 | 0.3941 | 0.5791 | 1.587 | 0.1324 | 0.7871 | 1.473 | 1.561 | 0 | 0.001667 | 0 | 0 | 0.045 | 0.03333 | 0.05167 | 0.025 | 7.426 |
| ens5 | 24 | ok | 6601da99128cbf97 | 0.5958 | 0.7411 | 0.5433 | 0.6705 | 0.7575 | 0.9299 | 0.879 | 0.8653 | 0.9615 | 0.03907 | 0.3072 | 0.9446 | 1.077 | 0.162 | 0.6445 | 1.683 | 1.487 | 0 | 0 | 0.001667 | 0 | 0.03667 | 0.03167 | 0.045 | 0.008333 | 6.826 |
| ens5 | 25 | ok | db446a4e0430a090 | 0.595 | 0.6535 | 0.5222 | 0.6822 | 0.7626 | 0.921 | 0.8597 | 0.8274 | 0.9518 | 0.05491 | 0.7226 | 2.372 | 2.662 | 0.2025 | 0.5603 | 1.587 | 1.66 | 0 | 0 | 0.001667 | 0 | 0.04667 | 0.07 | 0.08 | 0.02 | 8.069 |
| ens5_iid | 21 | ok | 64680c6639b6d52d | 0.5626 | 0.7176 | 0.4891 | 0.6601 | 0.71 | 0.914 | 0.869 | 0.8331 | 0.9403 | 0.04606 | 0.9005 | 1.979 | 1.844 | 0.2088 | 0.6393 | 1.641 | 1.114 | 0 | 0 | 0 | 0.001667 | 0.04833 | 0.06333 | 0.07833 | 0.02167 | 6.273 |
| ens5_iid | 22 | ok | ce5ea45a545dc1fb | 0.5631 | 0.6846 | 0.5856 | 0.6285 | 0.7133 | 0.9164 | 0.8479 | 0.8318 | 0.9375 | 0.05259 | 3.329 | 7.173 | 2.453 | 0.5162 | 0.6255 | 1.612 | 1.534 | 0 | 0 | 0 | 0 | 0.04333 | 0.08 | 0.08 | 0.03167 | 6.319 |
| ens5_iid | 23 | ok | 9cac146ced565994 | 0.5763 | 0.7169 | 0.5401 | 0.6587 | 0.7526 | 0.9115 | 0.8597 | 0.8246 | 0.9467 | 0.05412 | 0.7092 | 1 | 2.193 | 0.2471 | 0.6438 | 1.476 | 1.32 | 0 | 0.001667 | 0.003333 | 0 | 0.06167 | 0.045 | 0.08333 | 0.025 | 6.322 |
| ens5_iid | 24 | ok | 6601da99128cbf97 | 0.6153 | 0.7496 | 0.5021 | 0.6769 | 0.8074 | 0.9289 | 0.8815 | 0.8514 | 0.9579 | 0.04167 | 0.4243 | 0.9571 | 1.078 | 0.2841 | 0.6253 | 1.864 | 2.025 | 0.003333 | 0 | 0.001667 | 0 | 0.03333 | 0.04167 | 0.045 | 0.01333 | 6.32 |
| ens5_iid | 25 | ok | db446a4e0430a090 | 0.5426 | 0.6444 | 0.5149 | 0.6767 | 0.7251 | 0.9142 | 0.8593 | 0.8286 | 0.9535 | 0.05519 | 0.829 | 1.819 | 2.992 | 0.2127 | 0.5259 | 1.722 | 1.616 | 0 | 0 | 0.001667 | 0 | 0.05167 | 0.05833 | 0.07833 | 0.01667 | 6.142 |
| ens5_noboot | 21 | ok | 64680c6639b6d52d | 0.6016 | 0.7418 | 0.5025 | 0.6941 | 0.7528 | 0.92 | 0.8729 | 0.8415 | 0.9442 | 0.04324 | 1.546 | 1.184 | 2.096 | 0.1494 | 0.6131 | 1.622 | 1.549 | 0 | 0 | 0 | 0.003333 | 0.05167 | 0.05667 | 0.08 | 0.025 | 7.14 |
| ens5_noboot | 22 | ok | ce5ea45a545dc1fb | 0.6309 | 0.7401 | 0.626 | 0.645 | 0.7662 | 0.919 | 0.8708 | 0.8493 | 0.9482 | 0.04269 | 1.545 | 5.063 | 1.691 | 0.4311 | 0.639 | 1.867 | 1.711 | 0.001667 | 0 | 0 | 0 | 0.04333 | 0.06333 | 0.06833 | 0.02167 | 6.253 |
| ens5_noboot | 23 | ok | 9cac146ced565994 | 0.5387 | 0.7075 | 0.5434 | 0.6597 | 0.7219 | 0.9214 | 0.8793 | 0.8462 | 0.9529 | 0.04245 | 0.4252 | 0.667 | 1.673 | 0.1779 | 0.8039 | 1.742 | 2.054 | 0 | 0.001667 | 0.003333 | 0 | 0.04833 | 0.03333 | 0.06833 | 0.02167 | 6.207 |
| ens5_noboot | 24 | ok | 6601da99128cbf97 | 0.6433 | 0.7777 | 0.5435 | 0.7095 | 0.8039 | 0.9367 | 0.8844 | 0.8679 | 0.9618 | 0.03648 | 0.3162 | 0.584 | 1.157 | 0.6032 | 0.7028 | 1.735 | 1.7 | 0 | 0 | 0 | 0 | 0.025 | 0.03167 | 0.04667 | 0.01167 | 6.148 |
| ens5_noboot | 25 | ok | db446a4e0430a090 | 0.5834 | 0.6675 | 0.5729 | 0.7015 | 0.7904 | 0.92 | 0.8562 | 0.8292 | 0.9506 | 0.05505 | 1.031 | 1.82 | 3.697 | 0.4741 | 0.5621 | 1.705 | 1.883 | 0 | 0 | 0.005 | 0 | 0.05167 | 0.07167 | 0.09167 | 0.01833 | 7.768 |
| k1 | 21 | ok | 64680c6639b6d52d | 0.6426 | 0.6964 | 0.5871 | 0.6899 | 0.709 | 0.8953 | 0.8403 | 0.8029 | 0.8781 | 0.05958 | 3.229 | 4.479 | 5.6 | 2.393 | 0.5798 | 1.912 | 1.178 | 0 | 0 | 0.001667 | 0 | 0.06167 | 0.09167 | 0.1267 | 0.05833 | 1.421 |
| k1 | 22 | ok | ce5ea45a545dc1fb | 0.6823 | 0.7096 | 0.6233 | 0.6226 | 0.7465 | 0.8783 | 0.825 | 0.8144 | 0.8781 | 0.06083 | 10.67 | 13.6 | 3.964 | 3.752 | 0.5888 | 2.279 | 1.491 | 0.001667 | 0.001667 | 0 | 0 | 0.06333 | 0.09167 | 0.1233 | 0.04 | 1.245 |
| k1 | 23 | ok | 9cac146ced565994 | 0.6596 | 0.7321 | 0.6108 | 0.6746 | 0.7214 | 0.8892 | 0.8347 | 0.7819 | 0.8592 | 0.07472 | 1.623 | 1.719 | 5.749 | 1.916 | 0.6636 | 1.469 | 1.564 | 0 | 0 | 0.001667 | 0 | 0.06333 | 0.06667 | 0.1667 | 0.04167 | 1.244 |
| k1 | 24 | ok | 6601da99128cbf97 | 0.6227 | 0.7542 | 0.6223 | 0.6657 | 0.6527 | 0.9069 | 0.8372 | 0.8263 | 0.914 | 0.05019 | 0.9572 | 1.453 | 2.417 | 1.043 | 0.5426 | 2.086 | 2.816 | 0 | 0 | 0 | 0 | 0.03333 | 0.04833 | 0.09667 | 0.01833 | 1.253 |
| k1 | 25 | ok | db446a4e0430a090 | 0.6196 | 0.6742 | 0.6284 | 0.69 | 0.7124 | 0.8939 | 0.8147 | 0.7858 | 0.8732 | 0.07542 | 2.683 | 4.277 | 9.65 | 3.831 | 0.3666 | 1.839 | 1.627 | 0 | 0 | 0.003333 | 0 | 0.065 | 0.1017 | 0.1633 | 0.04833 | 1.655 |
| null_perm | 21 | ok | 64680c6639b6d52d | 0.004008 | 0.02354 | 0.01654 | -0.008214 | 0.003641 | 0.9196 | 0.8826 | 0.8501 | 0.9411 | 0.03611 | 1.359 | 0.9918 | 1.696 | 0.4141 | 0.9393 | 0.9344 | 0.9728 | 0.006667 | 0 | 0.055 | 0.008333 | 0.05 | 0.05333 | 0.07333 | 0.03167 | 8.253 |
| null_perm | 22 | ok | ce5ea45a545dc1fb | 0.01062 | -0.04365 | -0.023 | 0.0005441 | 0.0211 | 0.9243 | 0.8729 | 0.854 | 0.9496 | 0.04088 | 1.555 | 1.975 | 1.731 | 0.397 | 1.063 | 1.129 | 1.014 | 0.006667 | 0.003333 | 0.035 | 0.01667 | 0.03833 | 0.05333 | 0.055 | 0.015 | 8.906 |
| null_perm | 23 | ok | 9cac146ced565994 | 0.003857 | -0.03033 | -0.06683 | 0.04185 | 0.03837 | 0.9199 | 0.8796 | 0.8485 | 0.9499 | 0.0406 | 0.3941 | 0.5791 | 1.587 | 0.1324 | 0.9951 | 1.037 | 0.9864 | 0.005 | 0.005 | 0.02167 | 0.001667 | 0.045 | 0.03333 | 0.05167 | 0.025 | 6.666 |
| null_perm | 24 | ok | 6601da99128cbf97 | 0.007206 | 0.06848 | 0.007113 | -0.01113 | -0.0541 | 0.9299 | 0.879 | 0.8653 | 0.9615 | 0.03907 | 0.3072 | 0.9446 | 1.077 | 0.162 | 0.9954 | 0.9382 | 0.9071 | 0.006667 | 0 | 0.045 | 0.01 | 0.03667 | 0.03167 | 0.045 | 0.008333 | 6.209 |
| null_perm | 25 | ok | db446a4e0430a090 | -0.009326 | -0.01779 | -0.1034 | 0.02237 | 0.002348 | 0.921 | 0.8597 | 0.8274 | 0.9518 | 0.05491 | 0.7226 | 2.372 | 2.662 | 0.2025 | 1.143 | 1.16 | 1.085 | 0.005 | 0 | 0.05 | 0.01333 | 0.04667 | 0.07 | 0.08 | 0.02 | 6.823 |

E2 per-arm means over seeds:

| arm | unc_err_spearman | spearman_id | spearman_g4 | spearman_g20 | spearman_box6 | cov90_id | cov90_g4 | cov90_g20 | cov90_box6 | cov90_shift_abs_err | nll_id | nll_g4 | nll_g20 | nll_box6 | signal_ratio_g4 | signal_ratio_g20 | signal_ratio_box6 | conf_wrong_frac_id | conf_wrong_frac_g4 | conf_wrong_frac_g20 | conf_wrong_frac_box6 | outside3sd_frac_id | outside3sd_frac_g4 | outside3sd_frac_g20 | outside3sd_frac_box6 | train_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ens5 | 0.5885 | 0.7022 | 0.5574 | 0.6712 | 0.7286 | 0.9229 | 0.8748 | 0.8491 | 0.9508 | 0.04231 | 0.8677 | 1.372 | 1.751 | 0.2616 | 0.6667 | 1.595 | 1.473 | 0.0003333 | 0.0003333 | 0.001 | 0.0003333 | 0.04333 | 0.04833 | 0.061 | 0.02 | 7.155 |
| k1 | 0.6454 | 0.7133 | 0.6144 | 0.6686 | 0.7084 | 0.8927 | 0.8304 | 0.8023 | 0.8805 | 0.06415 | 3.832 | 5.106 | 5.476 | 2.587 | 0.5483 | 1.917 | 1.735 | 0.0003333 | 0.0003333 | 0.001333 | 0 | 0.05733 | 0.08 | 0.1353 | 0.04133 | 1.364 |
| null_perm | 0.003273 | 5.099e-05 | -0.03392 | 0.009085 | 0.002272 | 0.9229 | 0.8748 | 0.8491 | 0.9508 | 0.04231 | 0.8677 | 1.372 | 1.751 | 0.2616 | 1.027 | 1.04 | 0.9931 | 0.006 | 0.001667 | 0.04133 | 0.01 | 0.04333 | 0.04833 | 0.061 | 0.02 | 7.371 |
| ens5_noboot | 0.5995 | 0.7269 | 0.5576 | 0.682 | 0.767 | 0.9234 | 0.8727 | 0.8468 | 0.9515 | 0.04398 | 0.9726 | 1.864 | 2.063 | 0.3671 | 0.6642 | 1.734 | 1.78 | 0.0003333 | 0.0003333 | 0.001667 | 0.0006667 | 0.044 | 0.05133 | 0.071 | 0.01967 | 6.703 |
| ens5_iid | 0.572 | 0.7026 | 0.5263 | 0.6602 | 0.7417 | 0.917 | 0.8635 | 0.8339 | 0.9472 | 0.04993 | 1.238 | 2.586 | 2.112 | 0.2938 | 0.612 | 1.663 | 1.522 | 0.0006667 | 0.0003333 | 0.001333 | 0.0003333 | 0.04767 | 0.05767 | 0.073 | 0.02167 | 6.275 |
