## E10a — original arms (original prereg `07a678f4385c` from `reports/full/E10a-replication-K8-noise0.5-07a678f438-1791071299431152000.json`)

### E10a-replication-K8-noise0.5

- **verdict: accept** — mean improvement 17.66 >= delta 1, CI [12.64, 22.95] excludes 0
- hypothesis: through the library's own fixture path with NEW parameters (K=8, reward noise 0.5, budget 80), info-gain identifies the hidden shift in >= 1 fewer real interactions than random
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `random`, candidate `info_gain`, ablation `entropy`, alternative `icm`; seeds [0, 1, 2, 3, 4]
- prereg hash `07a678f4385c`, frozen 1791071277577572000 ns; first result 1791074095598688000 ns
- report: `reports/postfix_selection/E10a-replication-K8-noise0.5-07a678f438-1791074115936153000.json`

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
| entropy | 0 | ok | 2.22 | 81 | 18 | 0.9549 | 0.1667 | 0.004808 |
| entropy | 1 | ok | 3.52 | 81 | 18 | 0.9479 | 0.4444 | 0.005588 |
| entropy | 2 | ok | 1.86 | 81 | 18 | 0.9604 | 0.1667 | 0.003959 |
| entropy | 3 | ok | 1.53 | 81 | 18 | 0.9528 | 0.1667 | 0.004946 |
| entropy | 4 | ok | 1.44 | 81 | 18 | 0.9444 | 0.5556 | 0.005352 |
| icm | 0 | ok | 0.53 | 31.61 | 1 | 0.1858 | 1 | 0.06971 |
| icm | 1 | ok | 1.12 | 33.78 | 0 | 0.1346 | 1 | 0.07294 |
| icm | 2 | ok | 0.63 | 29.72 | 0 | 0.08385 | 1 | 0.07495 |
| icm | 3 | ok | 0.35 | 31.44 | 0 | 0.1817 | 1 | 0.06858 |
| icm | 4 | ok | 0.21 | 17.72 | 0 | 0.1189 | 1 | 0.08606 |
| info_gain | 0 | ok | 0.39 | 11.22 | 0 | 0.003968 | 1 | 0.1683 |
| info_gain | 1 | ok | 0.77 | 12.39 | 0 | 0 | 1 | 0.1669 |
| info_gain | 2 | ok | 0.44 | 12.17 | 0 | 0 | 1 | 0.1665 |
| info_gain | 3 | ok | 0.23 | 10 | 0 | 0.002778 | 1 | 0.1637 |
| info_gain | 4 | ok | 0.24 | 10.5 | 0 | 0.005556 | 1 | 0.1549 |
| lp | 0 | ok | 0.62 | 40.67 | 5 | 0.2377 | 0.8889 | 0.06959 |
| lp | 1 | ok | 1.31 | 49 | 6 | 0.2473 | 0.9444 | 0.06279 |
| lp | 2 | ok | 0.36 | 30.83 | 1 | 0.119 | 1 | 0.07381 |
| lp | 3 | ok | 0.59 | 49.5 | 4 | 0.2788 | 1 | 0.05721 |
| lp | 4 | ok | 0.21 | 19.33 | 0 | 0.1035 | 1 | 0.08295 |
| random | 0 | ok | 0.46 | 29 | 0 | 0.1112 | 1 | 0.07144 |
| random | 1 | ok | 0.69 | 36.28 | 1 | 0.114 | 1 | 0.07945 |
| random | 2 | ok | 0.38 | 25.44 | 0 | 0.1101 | 1 | 0.08511 |
| random | 3 | ok | 0.30 | 29.28 | 1 | 0.1226 | 1 | 0.07424 |
| random | 4 | ok | 0.26 | 24.56 | 0 | 0.1104 | 1 | 0.07724 |

Per-arm means:

| arm | interactions_to_identify | censored | tv_fraction | map_correct | realized_nats_per_interaction |
|---|---|---|---|---|---|
| info_gain | 11.26 | 0 | 0.00246 | 1 | 0.1641 |
| random | 28.91 | 0.4 | 0.1137 | 1 | 0.0775 |
| icm | 28.86 | 0.2 | 0.141 | 1 | 0.07445 |
| entropy | 81 | 18 | 0.9521 | 0.3 | 0.004931 |
| lp | 37.87 | 3.2 | 0.1972 | 0.9667 | 0.06927 |

## E10b — original arms (original prereg `d559e8a8bb87` from `reports/full/E10b-four-noisy-TVs-d559e8a8bb-1791071305539054000.json`)

### E10b-four-noisy-TVs

- **verdict: accept** — mean improvement 9.983 >= delta 1, CI [7.212, 13.1] excludes 0
- hypothesis: with FOUR coin-flip TVs (K=6, noise 0.35, budget 60), info-gain identifies the shift in >= 1 fewer interactions than random
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `random`, candidate `info_joint`, ablation `entropy`, alternative `icm`; seeds [0, 1, 2, 3, 4]
- prereg hash `d559e8a8bb87`, frozen 1791071277577970000 ns; first result 1791074115996478000 ns
- report: `reports/postfix_selection/E10b-four-noisy-TVs-d559e8a8bb-1791074120113924000.json`

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
| entropy | 0 | ok | 0.54 | 61 | 1 | 0.9722 |
| entropy | 1 | ok | 0.62 | 61 | 1 | 0.9736 |
| entropy | 2 | ok | 0.58 | 61 | 1 | 0.9722 |
| entropy | 3 | ok | 0.55 | 61 | 1 | 0.9611 |
| entropy | 4 | ok | 0.77 | 61 | 1 | 0.9681 |
| icm | 0 | ok | 0.06 | 20.17 | 0 | 0.4343 |
| icm | 1 | ok | 0.05 | 18.67 | 0 | 0.4538 |
| icm | 2 | ok | 0.06 | 20.42 | 0.08333 | 0.3762 |
| icm | 3 | ok | 0.05 | 17.25 | 0 | 0.4037 |
| icm | 4 | ok | 0.07 | 23.92 | 0 | 0.4726 |
| info_joint | 0 | ok | 0.05 | 5.667 | 0 | 0 |
| info_joint | 1 | ok | 0.06 | 6.083 | 0 | 0.05873 |
| info_joint | 2 | ok | 0.06 | 6.333 | 0 | 0.005556 |
| info_joint | 3 | ok | 0.05 | 5.167 | 0 | 0.0119 |
| info_joint | 4 | ok | 0.06 | 5.833 | 0 | 0.00641 |
| lp | 0 | ok | 0.07 | 25.42 | 0.1667 | 0.4611 |
| lp | 1 | ok | 0.06 | 19.08 | 0.08333 | 0.4045 |
| lp | 2 | ok | 0.08 | 28.08 | 0.25 | 0.4234 |
| lp | 3 | ok | 0.05 | 15.5 | 0 | 0.3446 |
| lp | 4 | ok | 0.06 | 19.58 | 0 | 0.3739 |
| random | 0 | ok | 0.04 | 14.25 | 0 | 0.3522 |
| random | 1 | ok | 0.03 | 13.75 | 0 | 0.3042 |
| random | 2 | ok | 0.04 | 15 | 0 | 0.4142 |
| random | 3 | ok | 0.04 | 16.75 | 0 | 0.4087 |
| random | 4 | ok | 0.05 | 19.25 | 0 | 0.3717 |

Per-arm means:

| arm | interactions_to_identify | censored | tv_fraction |
|---|---|---|---|
| info_joint | 5.817 | 0 | 0.01652 |
| random | 15.8 | 0 | 0.3702 |
| icm | 20.08 | 0.01667 | 0.4281 |
| entropy | 61 | 1 | 0.9694 |
| lp | 21.53 | 0.1 | 0.4015 |

## E10b — library arms (original prereg `d559e8a8bb87` from `reports/full/E10b-four-noisy-TVs-d559e8a8bb-1791071305539054000.json`)

### E10b-four-noisy-TVs

- **verdict: accept** — mean improvement 9.983 >= delta 1, CI [7.212, 13.1] excludes 0
- hypothesis: with FOUR coin-flip TVs (K=6, noise 0.35, budget 60), info-gain identifies the shift in >= 1 fewer interactions than random
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `random`, candidate `info_joint`, ablation `entropy`, alternative `icm`; seeds [0, 1, 2, 3, 4]
- prereg hash `d559e8a8bb87`, frozen 1791071277577970000 ns; first result 1791074120211578000 ns
- report: `reports/postfix_selection/E10b-four-noisy-TVs-d559e8a8bb-1791074124539013000.json`

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
| entropy | 0 | ok | 0.59 | 61 | 1 | 0.9722 |
| entropy | 1 | ok | 0.56 | 61 | 1 | 0.9736 |
| entropy | 2 | ok | 0.75 | 61 | 1 | 0.9722 |
| entropy | 3 | ok | 0.61 | 61 | 1 | 0.9611 |
| entropy | 4 | ok | 0.60 | 61 | 1 | 0.9681 |
| icm | 0 | ok | 0.06 | 20.17 | 0 | 0.4343 |
| icm | 1 | ok | 0.06 | 18.67 | 0 | 0.4538 |
| icm | 2 | ok | 0.06 | 20.42 | 0.08333 | 0.3762 |
| icm | 3 | ok | 0.05 | 17.25 | 0 | 0.4037 |
| icm | 4 | ok | 0.07 | 23.92 | 0 | 0.4726 |
| info_joint | 0 | ok | 0.09 | 5.667 | 0 | 0 |
| info_joint | 1 | ok | 0.09 | 6.083 | 0 | 0.05873 |
| info_joint | 2 | ok | 0.12 | 6.333 | 0 | 0.005556 |
| info_joint | 3 | ok | 0.08 | 5.167 | 0 | 0.0119 |
| info_joint | 4 | ok | 0.09 | 5.833 | 0 | 0.00641 |
| lp | 0 | ok | 0.07 | 25.42 | 0.1667 | 0.4611 |
| lp | 1 | ok | 0.06 | 19.08 | 0.08333 | 0.4045 |
| lp | 2 | ok | 0.10 | 28.08 | 0.25 | 0.4234 |
| lp | 3 | ok | 0.05 | 15.5 | 0 | 0.3446 |
| lp | 4 | ok | 0.06 | 19.58 | 0 | 0.3739 |
| random | 0 | ok | 0.04 | 14.25 | 0 | 0.3522 |
| random | 1 | ok | 0.03 | 13.75 | 0 | 0.3042 |
| random | 2 | ok | 0.05 | 15 | 0 | 0.4142 |
| random | 3 | ok | 0.04 | 16.75 | 0 | 0.4087 |
| random | 4 | ok | 0.05 | 19.25 | 0 | 0.3717 |

Per-arm means:

| arm | interactions_to_identify | censored | tv_fraction |
|---|---|---|---|
| info_joint | 5.817 | 0 | 0.01652 |
| random | 15.8 | 0 | 0.3702 |
| icm | 20.08 | 0.01667 | 0.4281 |
| entropy | 61 | 1 | 0.9694 |
| lp | 21.53 | 0.1 | 0.4015 |

## E10c — original arms (original prereg `9d74dc992b85` from `reports/full/E10c-learnable-useless-distractor-9d74dc992b-1791071307579603000.json`)

### E10c-learnable-useless-distractor

- **verdict: accept** — mean improvement 5.475 >= delta 1, CI [4.128, 7.177] excludes 0
- hypothesis: with a deterministic 4-bit lamp the task does not need (joint shift x lamp hypotheses), info-gain on the JOINT set wastes interactions on the lamp: task-marginal info-gain identifies the shift in >= 1 fewer interactions
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `info_joint`, candidate `info_marginal`, ablation `random`, alternative `icm`; seeds [0, 1, 2, 3, 4]
- prereg hash `9d74dc992b85`, frozen 1791071277578320000 ns; first result 1791074124651179000 ns
- report: `reports/postfix_selection/E10c-learnable-useless-distractor-9d74dc992b-1791074126655439000.json`

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
| icm | 1 | ok | 0.10 | 15.88 | 0 | 0.279 | 0.1013 |
| icm | 2 | ok | 0.08 | 18.75 | 0 | 0.3126 | 0.08639 |
| icm | 3 | ok | 0.07 | 13.25 | 0 | 0.2483 | 0.1386 |
| icm | 4 | ok | 0.06 | 15.88 | 0 | 0.3098 | 0.1562 |
| info_joint | 0 | ok | 0.11 | 11.12 | 0 | 0.441 | 0.01042 |
| info_joint | 1 | ok | 0.16 | 10.88 | 0 | 0.3973 | 0 |
| info_joint | 2 | ok | 0.18 | 12 | 0 | 0.4409 | 0 |
| info_joint | 3 | ok | 0.16 | 10.38 | 0 | 0.4595 | 0 |
| info_joint | 4 | ok | 0.15 | 8.625 | 0 | 0.4714 | 0.01562 |
| info_marginal | 0 | ok | 0.10 | 6.75 | 0 | 0.04688 | 0.01562 |
| info_marginal | 1 | ok | 0.14 | 5.375 | 0 | 0.01786 | 0 |
| info_marginal | 2 | ok | 0.13 | 4.375 | 0 | 0.01786 | 0 |
| info_marginal | 3 | ok | 0.13 | 5.75 | 0 | 0.01786 | 0 |
| info_marginal | 4 | ok | 0.07 | 3.375 | 0 | 0.05952 | 0.03125 |
| random | 0 | ok | 0.06 | 15.5 | 0 | 0.3738 | 0.08052 |
| random | 1 | ok | 0.08 | 23 | 0 | 0.278 | 0.08029 |
| random | 2 | ok | 0.10 | 23 | 0 | 0.4129 | 0.07033 |
| random | 3 | ok | 0.08 | 17.5 | 0 | 0.4106 | 0.07765 |
| random | 4 | ok | 0.06 | 19.5 | 0 | 0.2567 | 0.08543 |

Per-arm means:

| arm | interactions_to_identify | censored | lamp_fraction | tv_fraction |
|---|---|---|---|---|
| info_joint | 10.6 | 0 | 0.442 | 0.005208 |
| info_marginal | 5.125 | 0 | 0.03199 | 0.009375 |
| random | 19.7 | 0 | 0.3464 | 0.07884 |
| icm | 17.12 | 0 | 0.2863 | 0.1285 |

## E10c — library arms (original prereg `9d74dc992b85` from `reports/full/E10c-learnable-useless-distractor-9d74dc992b-1791071307579603000.json`)

### E10c-learnable-useless-distractor

- **verdict: accept** — mean improvement 5.475 >= delta 1, CI [4.128, 7.177] excludes 0
- hypothesis: with a deterministic 4-bit lamp the task does not need (joint shift x lamp hypotheses), info-gain on the JOINT set wastes interactions on the lamp: task-marginal info-gain identifies the shift in >= 1 fewer interactions
- primary `interactions_to_identify` (lower better), delta 1, basis `final`; baseline `info_joint`, candidate `info_marginal`, ablation `random`, alternative `icm`; seeds [0, 1, 2, 3, 4]
- prereg hash `9d74dc992b85`, frozen 1791071277578320000 ns; first result 1791074126780917000 ns
- report: `reports/postfix_selection/E10c-learnable-useless-distractor-9d74dc992b-1791074129140800000.json`

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
| icm | 0 | ok | 0.08 | 21.88 | 0 | 0.2819 | 0.1602 |
| icm | 1 | ok | 0.11 | 15.88 | 0 | 0.279 | 0.1013 |
| icm | 2 | ok | 0.10 | 18.75 | 0 | 0.3126 | 0.08639 |
| icm | 3 | ok | 0.10 | 13.25 | 0 | 0.2483 | 0.1386 |
| icm | 4 | ok | 0.06 | 15.88 | 0 | 0.3098 | 0.1562 |
| info_joint | 0 | ok | 0.12 | 11.12 | 0 | 0.441 | 0.01042 |
| info_joint | 1 | ok | 0.18 | 10.88 | 0 | 0.3973 | 0 |
| info_joint | 2 | ok | 0.24 | 12 | 0 | 0.4409 | 0 |
| info_joint | 3 | ok | 0.20 | 10.38 | 0 | 0.4595 | 0 |
| info_joint | 4 | ok | 0.13 | 8.625 | 0 | 0.4714 | 0.01562 |
| info_marginal | 0 | ok | 0.13 | 6.75 | 0 | 0.04688 | 0.01562 |
| info_marginal | 1 | ok | 0.11 | 5.375 | 0 | 0.01786 | 0 |
| info_marginal | 2 | ok | 0.16 | 4.375 | 0 | 0.01786 | 0 |
| info_marginal | 3 | ok | 0.20 | 5.75 | 0 | 0.01786 | 0 |
| info_marginal | 4 | ok | 0.09 | 3.375 | 0 | 0.05952 | 0.03125 |
| random | 0 | ok | 0.05 | 15.5 | 0 | 0.3738 | 0.08052 |
| random | 1 | ok | 0.08 | 23 | 0 | 0.278 | 0.08029 |
| random | 2 | ok | 0.17 | 23 | 0 | 0.4129 | 0.07033 |
| random | 3 | ok | 0.09 | 17.5 | 0 | 0.4106 | 0.07765 |
| random | 4 | ok | 0.07 | 19.5 | 0 | 0.2567 | 0.08543 |

Per-arm means:

| arm | interactions_to_identify | censored | lamp_fraction | tv_fraction |
|---|---|---|---|---|
| info_joint | 10.6 | 0 | 0.442 | 0.005208 |
| info_marginal | 5.125 | 0 | 0.03199 | 0.009375 |
| random | 19.7 | 0 | 0.3464 | 0.07884 |
| icm | 17.12 | 0 | 0.2863 | 0.1285 |

## E10d — original arms (original prereg `2fd51cf6b351` from `reports/full/E10d-misspecified-hypothesis-set-2fd51cf6b3-1791071309501221000.json`)

### E10d-misspecified-hypothesis-set

- **verdict: accept** — mean improvement 0.9818 >= delta 0.3, CI [0.9242, 1.011] excludes 0
- hypothesis: when the truth is NOT in the hypothesis set, info-gain selection reaches > 0.9 posterior on a (necessarily wrong) shift with no flag in >= 30 percentage points more scenarios than it reaches confident-wrong when the set is well specified
- primary `confident_wrong_unflagged` (higher better), delta 0.3, basis `final`; baseline `info_wellspec`, candidate `info_misspec`, ablation `random_misspec`, alternative `info_ppc_misspec`; seeds [0, 1, 2, 3, 4]
- prereg hash `2fd51cf6b351`, frozen 1791071277578660000 ns; first result 1791074129201046000 ns
- report: `reports/postfix_selection/E10d-misspecified-hypothesis-set-2fd51cf6b3-1791074130421037000.json`

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

| arm | seed | status | wall s | confident_wrong_unflagged | confident | time_to_confidence | ppc_flagged | interactions_to_identify | posterior_confident_wrong | flag_before_confidence | interactions_to_resolved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| info_misspec | 0 | ok | 0.13 | 1 | 1 | 14.91 | 0.09091 | 61 | - | - | - |
| info_misspec | 1 | ok | 0.06 | 1 | 1 | 6.545 | 0.09091 | 61 | - | - | - |
| info_misspec | 2 | ok | 0.11 | 1 | 1 | 16.64 | 0.09091 | 61 | - | - | - |
| info_misspec | 3 | ok | 0.09 | 1 | 1 | 14.36 | 0 | 61 | - | - | - |
| info_misspec | 4 | ok | 0.07 | 1 | 1 | 11.27 | 0.09091 | 61 | - | - | - |
| info_ppc_misspec | 0 | ok | 0.12 | 0.9091 | 1 | 14.91 | 0.09091 | 61 | - | - | - |
| info_ppc_misspec | 1 | ok | 0.05 | 0.9091 | 1 | 6.545 | 0.09091 | 61 | - | - | - |
| info_ppc_misspec | 2 | ok | 0.13 | 0.9091 | 1 | 16.64 | 0.09091 | 61 | - | - | - |
| info_ppc_misspec | 3 | ok | 0.09 | 1 | 1 | 14.36 | 0 | 61 | - | - | - |
| info_ppc_misspec | 4 | ok | 0.06 | 0.9091 | 1 | 11.27 | 0.09091 | 61 | - | - | - |
| info_wellspec | 0 | ok | 0.05 | 0 | 1 | 6.091 | 0 | 6.091 | - | - | - |
| info_wellspec | 1 | ok | 0.04 | 0.09091 | 1 | 4.636 | 0 | 5.091 | - | - | - |
| info_wellspec | 2 | ok | 0.04 | 0 | 1 | 5.545 | 0 | 5.545 | - | - | - |
| info_wellspec | 3 | ok | 0.03 | 0 | 1 | 4.727 | 0 | 4.727 | - | - | - |
| info_wellspec | 4 | ok | 0.03 | 0 | 1 | 4.455 | 0 | 4.455 | - | - | - |
| random_misspec | 0 | ok | 0.04 | 1 | 1 | 15.09 | 0 | 61 | - | - | - |
| random_misspec | 1 | ok | 0.04 | 1 | 1 | 13.64 | 0.1818 | 61 | - | - | - |
| random_misspec | 2 | ok | 0.03 | 1 | 1 | 10.09 | 0.09091 | 61 | - | - | - |
| random_misspec | 3 | ok | 0.02 | 1 | 1 | 12.64 | 0 | 61 | - | - | - |
| random_misspec | 4 | ok | 0.03 | 1 | 1 | 18.36 | 0.09091 | 61 | - | - | - |

Per-arm means:

| arm | confident_wrong_unflagged | confident | time_to_confidence | ppc_flagged | interactions_to_identify | posterior_confident_wrong | flag_before_confidence | interactions_to_resolved |
|---|---|---|---|---|---|---|---|---|
| info_wellspec | 0.01818 | 1 | 5.091 | 0 | 5.182 | - | - | - |
| info_misspec | 1 | 1 | 12.75 | 0.07273 | 61 | - | - | - |
| random_misspec | 1 | 1 | 13.96 | 0.07273 | 61 | - | - | - |
| info_ppc_misspec | 0.9273 | 1 | 12.75 | 0.07273 | 61 | - | - | - |

## E10d — library arms (original prereg `2fd51cf6b351` from `reports/full/E10d-misspecified-hypothesis-set-2fd51cf6b3-1791071309501221000.json`)

### E10d-misspecified-hypothesis-set

- **verdict: reject** — CI upper bound 0.07576 < delta 0.3: the hypothesised effect is ruled out
- hypothesis: when the truth is NOT in the hypothesis set, info-gain selection reaches > 0.9 posterior on a (necessarily wrong) shift with no flag in >= 30 percentage points more scenarios than it reaches confident-wrong when the set is well specified
- primary `confident_wrong_unflagged` (higher better), delta 0.3, basis `final`; baseline `info_wellspec`, candidate `info_misspec`, ablation `random_misspec`, alternative `info_ppc_misspec`; seeds [0, 1, 2, 3, 4]
- prereg hash `2fd51cf6b351`, frozen 1791071277578660000 ns; first result 1791074130560427000 ns
- report: `reports/postfix_selection/E10d-misspecified-hypothesis-set-2fd51cf6b3-1791074134780530000.json`

| comparison (+ve = candidate better) | basis | pairs | mean diff | CI low | CI high |
|---|---|---|---|---|---|
| info_misspec vs info_wellspec | final (decisive) | 5 | 0.01818 | -0.01061 | 0.07576 |
| info_misspec vs info_wellspec | equal_interactions | 0 | - | - | - |
| info_misspec vs info_wellspec | equal_time | 0 | - | - | - |
| info_misspec vs random_misspec (ablation_arm) | final | 5 | 0 | -0.08637 | 0.08637 |
| info_misspec vs info_ppc_misspec (alternative_arm) | final | 5 | -0.9091 | -0.9955 | -0.8227 |

- note: criterion unmet: baseline 'info_wellspec' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| info_wellspec | 5/5 | 0 | 0 | 0 | 0 |
| info_misspec | 5/5 | 0 | 0 | 0.01818 | 0.04066 |
| random_misspec | 5/5 | 0 | 0 | 0.01818 | 0.04066 |
| info_ppc_misspec | 5/5 | 0 | 0 | 0.9273 | 0.04066 |

Per-run outcomes (E10d):

| arm | seed | status | wall s | confident_wrong_unflagged | confident | time_to_confidence | ppc_flagged | interactions_to_identify | posterior_confident_wrong | flag_before_confidence | interactions_to_resolved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| info_misspec | 0 | ok | 0.50 | 0 | 1 | 14.91 | 1 | 61 | 1 | 0 | 61 |
| info_misspec | 1 | ok | 0.48 | 0 | 1 | 6.545 | 1 | 61 | 1 | 0 | 61 |
| info_misspec | 2 | ok | 0.49 | 0 | 1 | 16.64 | 1 | 61 | 1 | 0.09091 | 61 |
| info_misspec | 3 | ok | 0.49 | 0 | 1 | 14.36 | 1 | 61 | 1 | 0.09091 | 61 |
| info_misspec | 4 | ok | 0.44 | 0.09091 | 1 | 11.27 | 0.9091 | 61 | 1 | 0 | 56.36 |
| info_ppc_misspec | 0 | ok | 0.08 | 0.9091 | 1 | 14.91 | 0.09091 | 61 | - | - | - |
| info_ppc_misspec | 1 | ok | 0.03 | 0.9091 | 1 | 6.545 | 0.09091 | 61 | - | - | - |
| info_ppc_misspec | 2 | ok | 0.09 | 0.9091 | 1 | 16.64 | 0.09091 | 61 | - | - | - |
| info_ppc_misspec | 3 | ok | 0.07 | 1 | 1 | 14.36 | 0 | 61 | - | - | - |
| info_ppc_misspec | 4 | ok | 0.06 | 0.9091 | 1 | 11.27 | 0.09091 | 61 | - | - | - |
| info_wellspec | 0 | ok | 0.14 | 0 | 1 | 6.091 | 0 | 6.091 | 0 | 0 | 14.64 |
| info_wellspec | 1 | ok | 0.12 | 0 | 1 | 4.636 | 0 | 5.091 | 0.09091 | 0 | 15.27 |
| info_wellspec | 2 | ok | 0.12 | 0 | 1 | 5.545 | 0 | 5.545 | 0 | 0 | 13.27 |
| info_wellspec | 3 | ok | 0.10 | 0 | 1 | 4.727 | 0 | 4.727 | 0 | 0 | 11.91 |
| info_wellspec | 4 | ok | 0.10 | 0 | 1 | 4.455 | 0 | 4.455 | 0 | 0 | 11.82 |
| random_misspec | 0 | ok | 0.21 | 0 | 1 | 15.09 | 0.3636 | 61 | 1 | 0 | 61 |
| random_misspec | 1 | ok | 0.20 | 0 | 1 | 13.64 | 0.6364 | 61 | 1 | 0 | 61 |
| random_misspec | 2 | ok | 0.25 | 0 | 1 | 10.09 | 0.3636 | 61 | 1 | 0 | 61 |
| random_misspec | 3 | ok | 0.19 | 0.09091 | 1 | 12.64 | 0.4545 | 61 | 1 | 0 | 59 |
| random_misspec | 4 | ok | 0.20 | 0 | 1 | 18.36 | 0.7273 | 61 | 1 | 0 | 61 |

Per-arm means:

| arm | confident_wrong_unflagged | confident | time_to_confidence | ppc_flagged | interactions_to_identify | posterior_confident_wrong | flag_before_confidence | interactions_to_resolved |
|---|---|---|---|---|---|---|---|---|
| info_wellspec | 0 | 1 | 5.091 | 0 | 5.182 | 0.01818 | 0 | 13.38 |
| info_misspec | 0.01818 | 1 | 12.75 | 0.9818 | 61 | 1 | 0.03636 | 60.07 |
| random_misspec | 0.01818 | 1 | 13.96 | 0.5091 | 61 | 1 | 0 | 60.6 |
| info_ppc_misspec | 0.9273 | 1 | 12.75 | 0.07273 | 61 | - | - | - |

## E10e (post-fix tracker)

## E10e idle / stationary-probe retention tracker (measurement, not an A/B)

RetentionTracker(known_below=0.45, forgotten_above=0.55); 16 probes with i.i.d. N(center, sd) losses, 2000 evaluations, seeds [0, 1, 2, 3, 4].

| config | seed | net progress | telescoped first-last | raw (clipped LP) gain | relearn events | flagged probes /16 |
|---|---|---|---|---|---|---|
| straddle 0.5+-0.2 | 0 | -2.8 | -0.65 | 3584.9 | 1 | 0 |
| straddle 0.5+-0.2 | 1 | -1.6 | -0.22 | 3580.9 | 1 | 0 |
| straddle 0.5+-0.2 | 2 | -2.4 | -0.87 | 3593.4 | 2 | 0 |
| straddle 0.5+-0.2 | 3 | -2.4 | -1.37 | 3599.1 | 1 | 0 |
| straddle 0.5+-0.2 | 4 | -0.6 | 0.62 | 3607.3 | 0 | 0 |
| wide 0.3+-0.2 | 0 | -0.7 | -0.65 | 3584.9 | 0 | 0 |
| wide 0.3+-0.2 | 1 | -0.2 | -0.22 | 3580.9 | 0 | 0 |
| wide 0.3+-0.2 | 2 | -0.9 | -0.87 | 3593.4 | 0 | 0 |
| wide 0.3+-0.2 | 3 | -1.4 | -1.37 | 3599.1 | 0 | 0 |
| wide 0.3+-0.2 | 4 | 0.6 | 0.62 | 3607.3 | 0 | 0 |
| far 0.2+-0.05 | 0 | -0.2 | -0.16 | 896.2 | 0 | 0 |
| far 0.2+-0.05 | 1 | -0.1 | -0.06 | 895.2 | 0 | 0 |
| far 0.2+-0.05 | 2 | -0.2 | -0.22 | 898.4 | 0 | 0 |
| far 0.2+-0.05 | 3 | -0.3 | -0.34 | 899.8 | 0 | 0 |
| far 0.2+-0.05 | 4 | 0.2 | 0.15 | 901.8 | 0 | 0 |
| straddle annealed | 0 | -13.0 | -2.36 | 7165.5 | 6 | 0 |
| straddle annealed | 1 | -5.2 | 0.59 | 7170.4 | 2 | 0 |
| straddle annealed | 2 | -12.1 | -0.93 | 7184.4 | 7 | 0 |
| straddle annealed | 3 | -6.5 | -2.83 | 7193.3 | 3 | 0 |
| straddle annealed | 4 | -7.2 | 1.18 | 7216.5 | 3 | 0 |


Total wall 46s
