# Stage 6–8 verification A/Bs: results

These comparisons were run by an independent verifier who did not write the code under test.
All of them use the plan §7.2 harness (`developmental_ai/foundation/experiments/ab.py`).
Every preregistration was frozen inside `experiments()` before any arm ran, and the
harness checked that order for each report. Results come from 5 seeds per experiment,
split by episode, with identical arrays for every arm (the smoke test checks the
`data_fp` hash). Each experiment also has an ablation arm. Raw reports are in
`reports/full/` (JSON and md, never overwritten, plus `index.jsonl`).

To reproduce: `PYTHONPATH=. python -m experiments.foundation_ab.run full`. The run took
about 8.7 min of wall-clock on 8-core macOS with torch 4 threads (E1 262 s, E2 225 s,
E3 32 s). The fast check is `tests/_foundation_ab_verify_smoke.py` (about 20 s). It
asserts only the protocol, never a winner.

How the runs were set up:
- **Caching.** Within one experiment, an arm's result is computed once and shared by all
  of that experiment's preregs. The harness's per-run `wall s` is therefore about 0 for
  every prereg after the first. Compute costs come from the arm's own `train_seconds`,
  `ms_per_frame` and `flops_per_sample`.
- **Memory.** tracemalloc was off, so that timing is not distorted. Only the RSS
  high-water mark is recorded.

## Verdicts

| prereg | verdict | key numbers (5 seeds) |
|---|---|---|
| **E1a**: IN beats a FLOP-matched MLP on 1-step NLL | **inconclusive** | NLL: IN 0.175, MLP-661 0.501. Mean gain 0.326, CI [−0.018, 0.81]. On seed 13 the gain was only 0.049. |
| **E1b**: IN beats a FLOP-matched MLP on 10-step RMSE | **reject** | RMSE10: IN 2.86, MLP-661 2.63, MLP-128 2.55. The IN is worse by 0.23, CI [−0.45, −0.03]. |
| **E1c**: IN beats MLP-128 at equal training time (NLL) | **inconclusive** | The IN won on all 5 seeds (+0.8 to +16), CI [−0.13, 14.4]. The gain comes from MLP-128 **overfitting** when trained longer: its NLL rose from 0.62 at 60 epochs to tens or hundreds at 480. No early stopping was used. |
| **E2a**: ensemble epistemic variance predicts error better than a null signal | **accept** | Spearman 0.247 vs 0.001 (permuted), CI [0.155, 0.337]. |
| **E2b**: epistemic variance predicts error better than a single model's own variance | **reject** | 0.247 vs **0.516** for K=1. The ensemble is worse by 0.27, CI [−0.34, −0.19]. |
| **E2c**: K=5 coverage holds under shift better than K=1 | **inconclusive** | Mean \|cov90 − 0.9\|: 0.023 vs 0.029, CI [−0.008, 0.021]. |
| **E3a**: tracker beats per-frame NN-velocity (standard fixture) | **accept** | OSPA: tracker 0.0401, nnvel 0.0559, static 0.0510. This reproduces the perception agent's 0.0399 vs 0.0506. |
| **E3b**: tracker beats per-frame NN-velocity (hard: N=10, clutter 2, occluder, 3 shared looks) | **accept** | OSPA: tracker 0.0518, nnvel 0.0662, static 0.0642. Oracle: 0.040 vs 0.060. |

## What this means for the gates

**Stage 6 is not met as an architectural claim.**
- **The NLL gain is not significant at matched FLOPs.** At equal FLOPs (1.08 M per
  sample; the IN is 15× MLP-128) the IN's 1-step NLL advantage does not reach significance.
- **The ablations say the relational part is not what helps:**
  - Removing **relative geometry** from the edges (`in_norel`) gave a **better** NLL than
    the full IN (0.162 vs 0.175).
  - Removing **edges** entirely (`in_noedges`, 7× fewer FLOPs) cost only 0.038 nats, with
    a CI that includes 0 (CI [−0.022, 0.097]).
  - So whatever NLL edge exists comes from the per-entity factorised heads, not from
    relations.
- **Long-horizon error is worse.** The IN is significantly worse on 10-step error than
  both MLPs.
- **Point error is unchanged.** 1-step RMSE is identical across all arms (1.65–1.67), so
  any gain is in variance calibration, not in the prediction itself.
- **The IN is slower at equal FLOPs:** predict 1.98 vs 0.68 ms per 64, train 6.1 vs 4.6 s.

**Stage 7 is partly met.**
- **Some predictive value, but less than a single model.** Epistemic variance does carry
  information about error (E2a). It carries less than a single model's own predicted
  variance (E2b).
- **It does not "increase appropriately" when the rule changes:**
  - Gravity 9.81→4: mean epistemic variance went **down** (×0.95).
  - Gravity 9.81→20: it rose only ×1.25, while RMSE rose from 1.63 to 2.50.
  - Box 4→6 (inputs out of range): it rose ×2.45.
- **Agreement did coincide with confident errors.** "Confidently wrong" here means
  epistemic variance at or below the in-distribution median *and* error at or above the
  in-distribution 99th percentile. On g20 that was 0.67% of transitions (4 per seed),
  against 0.03% in distribution: 20 times more. Points outside the 3-sd interval rose from
  1.2% to 3.5%.
- **The bootstrap does the work:**
  - Without resampling (`ens5_noboot`) the correlation collapses to 0.034.
  - Episode-level and i.i.d. bootstrap are about equal (0.247 vs 0.278).

**Stage 8: the downstream gain replicates on new seeds and on a harder fixture**, but
with these caveats:
1. `IdentityTracker` is a hand-built Kalman and Hungarian tracker, not a *learned* state.
   This verifies persistent tracked state, not the plan's "learned representation".
2. Removing the appearance term (κ = 1e-6) changes OSPA by only 0.0002 (standard) and
   0.0009 (hard). Almost all of the gain comes from the motion model and the persistence
   of tracks, not from appearance-based identity.
3. The preregistered baseline (nnvel) was the *weaker* of the two per-frame arms. The
   tracker also beats the stronger one (static) by 0.011 and 0.012.
4. The tracker predicts fewer points (5.4 vs 6.4) because it never emits clutter, which
   OSPA on detections partly rewards. No baseline with a clutter filter was preregistered.
5. The tracker costs about 46–75 times more per frame than the per-frame baselines (0.8–1.5 ms vs 0.02 ms).

## Bugs and design issues found (not fixed; developmental_ai/ untouched)

1. **The mechanism training loss is dominated by static walls.**
   - **Where:** `mechanisms/_nets.py:204` (target = change of *every* entity,
     walls included), `:75` (NLL summed over all dimensions), `:28` (`VAR_FLOOR = 1e-4`
     in standardised units).
   - **Mechanism:** the 16 wall dimensions have an exactly-zero change, so the model can
     push them to the variance floor. They then contribute about −66 of a training NLL
     whose floor is −73.7. On the mechanisms agent's own data (seed 7, 60 epochs) the
     scored ball dimensions contribute −8.4 for the IN and −1.8 for MLP-128.
   - **Effect:** training loss is not monotone (6 epochs with a jump of more than +5 for
     the IN). On small data it diverges outright.
   - **Repro:** `make_split("box2d:e1", ep0..15, 0.3)`, seed 11, T=40,
     `AblatedIN(mode="norel")`, 8 epochs at lr 1e-3. Per-epoch loss:
     `[9.5, 4.2, -2.1, -10.5, -15.9, 35.8, 283.1, 332.1]`.
2. **`BaseMechanism.fit` cannot resume training.** Every call re-creates Adam
   (`_nets.py:97`), so chunked or continual fitting resets the optimiser. The
   FLOP-matched MLP's loss rose across those resets (9.3 → 310 at toy scale).
3. **The harness's `equal_interactions` basis gives a misleading figure on fixed-data
   curves.**
   - **Where:** `experiments/ab.py:518`/`:540`.
   - **Mechanism:** when every curve point has the same interaction count, this basis
     takes the **last** curve point, not the reported primary value.
   - **Effect:** in E1c it printed a +118 "equal_interactions" difference. That is the
     480-epoch overfit MLP-128, not the 60-epoch one. The verdict was not affected.
4. **On basis `equal_time`, the ablation and alternative comparisons cannot pair**
   when those arms have a single curve point (`ab.py:540`, common budget = the shorter
   run). E1c's comparisons against `in_noedges` and `mlp_flops` therefore have 0 pairs.

## Pilot and deviations

- **Learning-rate pilot.** A pilot on seed 99, which is not a final seed, checked
  lr 3e-3 against 1e-3 for the IN, MLP-661 and MLP-128. 3e-3 was at least as good for all
  three, so the mechanisms smoke's lr was kept.
- **Training was changed before the final run.** An early draft trained in resumed
  chunks; that was dropped in favour of single fits (see bug 2). The toy-scale smoke uses
  lr 5e-4 so that every arm passes the "loss must fall" condition. The full runs use 3e-3.
- **E2 setting.** E2 uses a box of size 4 with contacts and noise 0.05. The inference
  agent used size 40, free flight only and noise 0.2, so this is a harder setting than
  theirs.

---

# Generated tables (per prereg, per run)

## E1 (262s wall)

### E1a-IN-vs-flopmatched-MLP-nll1

- **verdict: inconclusive** — CI [-0.01801, 0.8076] with mean 0.326 neither meets nor rules out delta 0.05
- hypothesis: At equal FLOPs/sample, equal data and equal epochs, the relational IN has lower held-out 1-step Gaussian NLL (ball dims) than an MLP on the same structured inputs
- primary `nll1` (lower better), delta 0.05, basis `final`; baseline `mlp_flops`, candidate `in`, ablation `in_noedges`, alternative `mlp128`; seeds [11, 12, 13, 14, 15]
- prereg hash `c128bac46c35`, frozen 1791069632293082000 ns; first result 1791069640582148000 ns
- report: `reports/full/E1a-IN-vs-flopmatched-MLP-nll1-c128bac46c-1791069894356621000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| in vs mlp_flops | final (decisive) | 5 | 0.326 | -0.01801 | 0.8076 | 0 |
| in vs mlp_flops | equal_interactions | 5 | 0.326 | -0.01801 | 0.8076 | 0 |
| in vs mlp_flops | equal_time | 0 | - | - | - | 5 |
| in vs in_noedges (ablation_arm) | final | 5 | 0.03789 | -0.02156 | 0.09733 | 0 |
| in vs mlp128 (alternative_arm) | final | 5 | 0.4476 | 0.2477 | 0.659 | 0 |

- note: criterion unmet: baseline 'mlp_flops' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['in_norel']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| in | 5/5 | 0 | 0 | 0.1753 | 0.0997 |
| mlp_flops | 5/5 | 0 | 0 | 0.5012 | 0.4295 |
| mlp128 | 5/5 | 0 | 0 | 0.6229 | 0.2486 |
| in_noedges | 5/5 | 0 | 0 | 0.2132 | 0.08562 |
| in_norel | 5/5 | 0 | 0 | 0.1622 | 0.09168 |

### E1b-IN-vs-flopmatched-MLP-rmse10

- **verdict: reject** — CI upper bound -0.03486 < delta 0.05: the hypothesised effect is ruled out
- hypothesis: At equal FLOPs/sample, the IN has lower 10-step mean-rollout RMSE (ball dims) than the FLOP-matched MLP
- primary `rmse10` (lower better), delta 0.05, basis `final`; baseline `mlp_flops`, candidate `in`, ablation `in_noedges`, alternative `mlp128`; seeds [11, 12, 13, 14, 15]
- prereg hash `76ae0468eeb0`, frozen 1791069632293147000 ns; first result 1791069894362922000 ns
- report: `reports/full/E1b-IN-vs-flopmatched-MLP-rmse10-76ae0468ee-1791069894363996000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| in vs mlp_flops | final (decisive) | 5 | -0.2307 | -0.4494 | -0.03486 | 0 |
| in vs mlp_flops | equal_interactions | 5 | -0.2307 | -0.4494 | -0.03486 | 0 |
| in vs mlp_flops | equal_time | 0 | - | - | - | 5 |
| in vs in_noedges (ablation_arm) | final | 5 | -0.07011 | -0.1547 | 0.00618 | 0 |
| in vs mlp128 (alternative_arm) | final | 5 | -0.3103 | -0.4486 | -0.159 | 0 |

- note: criterion unmet: baseline 'mlp_flops' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['in_norel']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| in | 5/5 | 0 | 0 | 2.861 | 0.1834 |
| mlp_flops | 5/5 | 0 | 0 | 2.63 | 0.212 |
| mlp128 | 5/5 | 0 | 0 | 2.55 | 0.177 |
| in_noedges | 5/5 | 0 | 0 | 2.791 | 0.1456 |
| in_norel | 5/5 | 0 | 0 | 2.801 | 0.1347 |

### E1c-IN-vs-MLP128-equal-time-nll1

- **verdict: inconclusive** — CI [-0.1323, 14.45] with mean 6.429 neither meets nor rules out delta 0.05
- hypothesis: At equal TRAINING WALL-CLOCK, the IN has lower held-out 1-step NLL than the mechanisms agent's MLP-128 given as many epochs as fit
- primary `nll1` (lower better), delta 0.05, basis `equal_time`; baseline `mlp128`, candidate `in`, ablation `in_noedges`, alternative `mlp_flops`; seeds [11, 12, 13, 14, 15]
- prereg hash `ec183801b8d9`, frozen 1791069632293189000 ns; first result 1791069894369627000 ns
- report: `reports/full/E1c-IN-vs-MLP128-equal-time-nll1-ec183801b8-1791069894370688000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| in vs mlp128 | final | 5 | 0.4476 | 0.2477 | 0.659 | 0 |
| in vs mlp128 | equal_interactions | 5 | 118.2 | 58.63 | 182.9 | 0 |
| in vs mlp128 | equal_time (decisive) | 5 | 6.429 | -0.1323 | 14.45 | 0 |
| in vs in_noedges (ablation_arm) | equal_time | 0 | - | - | - | 5 |
| in vs mlp_flops (alternative_arm) | equal_time | 0 | - | - | - | 5 |

- note: criterion unmet: baseline 'mlp128' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['in_norel']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| in | 5/5 | 0 | 0 | 0.1753 | 0.0997 |
| mlp_flops | 5/5 | 0 | 0 | 0.5012 | 0.4295 |
| mlp128 | 5/5 | 0 | 0 | 0.6229 | 0.2486 |
| in_noedges | 5/5 | 0 | 0 | 0.2132 | 0.08562 |
| in_norel | 5/5 | 0 | 0 | 0.1622 | 0.09168 |

### E1 per-run outcomes (shared by every E1 prereg)

| arm | seed | status | data_fp | nll1 | rmse1 | cov90_1 | nll_event | rmse10 | nll10 | cov90_10 | rmse1_contact | rmse1_free | flops_per_sample | params | train_seconds | predict_ms_per64 | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in | 11 | ok | 48e64930ceb56c2f | 0.1906 | 1.626 | 0.9468 | 0.4692 | 2.639 | 24.59 | 0.8862 | 2.754 | 0.3137 | 1076544 | 20507 | 5.96 | 1.524 | 60 |
| in | 12 | ok | 3665a578c80e2c5d | 0.05061 | 1.597 | 0.9462 | 0.4728 | 2.985 | 11.75 | 0.9237 | 2.686 | 0.3677 | 1076544 | 20507 | 5.621 | 1.668 | 60 |
| in | 13 | ok | 26026ec2a9554a35 | 0.1601 | 1.562 | 0.9247 | 0.4894 | 2.735 | 68.75 | 0.8058 | 2.578 | 0.2898 | 1076544 | 20507 | 5.8 | 1.727 | 60 |
| in | 14 | ok | 05adfad483698ec7 | 0.327 | 1.792 | 0.9489 | 0.5541 | 2.852 | 21.48 | 0.8871 | 2.866 | 0.4042 | 1076544 | 20507 | 6.676 | 1.729 | 60 |
| in | 15 | ok | c9b331240ce9067f | 0.1482 | 1.729 | 0.9321 | 0.4961 | 3.093 | 59.08 | 0.8192 | 2.826 | 0.3056 | 1076544 | 20507 | 6.493 | 3.235 | 60 |
| in_noedges | 11 | ok | 48e64930ceb56c2f | 0.1789 | 1.642 | 0.9365 | 0.5021 | 2.629 | 81.57 | 0.8046 | 2.771 | 0.3588 | 149856 | 20507 | 2.353 | 0.6341 | 60 |
| in_noedges | 12 | ok | 3665a578c80e2c5d | 0.1396 | 1.609 | 0.9379 | 0.5 | 2.823 | 80.39 | 0.795 | 2.674 | 0.4772 | 149856 | 20507 | 2.224 | 0.7658 | 60 |
| in_noedges | 13 | ok | 26026ec2a9554a35 | 0.1598 | 1.56 | 0.9326 | 0.5249 | 2.669 | 145.8 | 0.7717 | 2.548 | 0.4037 | 149856 | 20507 | 2.31 | 0.7249 | 60 |
| in_noedges | 14 | ok | 05adfad483698ec7 | 0.3525 | 1.816 | 0.9338 | 0.5639 | 2.84 | 88.52 | 0.8042 | 2.881 | 0.5031 | 149856 | 20507 | 2.393 | 0.6679 | 60 |
| in_noedges | 15 | ok | c9b331240ce9067f | 0.2351 | 1.726 | 0.9349 | 0.5439 | 2.992 | 85.78 | 0.8187 | 2.79 | 0.4393 | 149856 | 20507 | 2.765 | 1.029 | 60 |
| in_norel | 11 | ok | 48e64930ceb56c2f | 0.1416 | 1.647 | 0.9437 | 0.496 | 2.62 | 68.22 | 0.815 | 2.788 | 0.3202 | 1076544 | 20507 | 6.645 | 1.563 | 60 |
| in_norel | 12 | ok | 3665a578c80e2c5d | 0.09296 | 1.599 | 0.9408 | 0.4932 | 2.828 | 59.53 | 0.825 | 2.663 | 0.4612 | 1076544 | 20507 | 5.635 | 1.682 | 60 |
| in_norel | 13 | ok | 26026ec2a9554a35 | 0.07098 | 1.567 | 0.9356 | 0.5137 | 2.711 | 183.9 | 0.7517 | 2.569 | 0.3636 | 1076544 | 20507 | 5.674 | 1.775 | 60 |
| in_norel | 14 | ok | 05adfad483698ec7 | 0.2956 | 1.809 | 0.9393 | 0.5618 | 2.947 | 52.73 | 0.8221 | 2.869 | 0.503 | 1076544 | 20507 | 6.481 | 1.632 | 60 |
| in_norel | 15 | ok | c9b331240ce9067f | 0.21 | 1.714 | 0.9526 | 0.5408 | 2.9 | 37.47 | 0.8433 | 2.753 | 0.4996 | 1076544 | 20507 | 7.083 | 2.063 | 60 |
| mlp128 | 11 | ok | 48e64930ceb56c2f | 0.5351 | 1.598 | 0.9065 | 0.4889 | 2.339 | 47.72 | 0.8087 | 2.662 | 0.464 | 72448 | 36557 | 1.354 | 0.3445 | 60 |
| mlp128 | 12 | ok | 3665a578c80e2c5d | 0.4505 | 1.582 | 0.9342 | 0.4872 | 2.528 | 25.22 | 0.8612 | 2.59 | 0.5727 | 72448 | 36557 | 1.41 | 0.3204 | 60 |
| mlp128 | 13 | ok | 26026ec2a9554a35 | 0.4191 | 1.555 | 0.9385 | 0.5092 | 2.426 | 27.9 | 0.8758 | 2.519 | 0.4703 | 72448 | 36557 | 1.336 | 0.3502 | 60 |
| mlp128 | 14 | ok | 05adfad483698ec7 | 1.029 | 1.787 | 0.9045 | 0.5531 | 2.712 | 30.44 | 0.8187 | 2.824 | 0.5279 | 72448 | 36557 | 1.347 | 0.4023 | 60 |
| mlp128 | 15 | ok | c9b331240ce9067f | 0.6809 | 1.715 | 0.9243 | 0.5263 | 2.746 | 31.17 | 0.8529 | 2.766 | 0.4583 | 72448 | 36557 | 1.483 | 0.3932 | 60 |
| mlp_flops | 11 | ok | 48e64930ceb56c2f | 0.2839 | 1.613 | 0.9182 | 0.4656 | 2.415 | 47.56 | 0.8221 | 2.73 | 0.3188 | 1078752 | 540775 | 4.435 | 0.6737 | 60 |
| mlp_flops | 12 | ok | 3665a578c80e2c5d | 0.2711 | 1.571 | 0.9442 | 0.4673 | 2.514 | 24.63 | 0.8625 | 2.581 | 0.5464 | 1078752 | 540775 | 4.338 | 0.6689 | 60 |
| mlp_flops | 13 | ok | 26026ec2a9554a35 | 0.2087 | 1.56 | 0.9314 | 0.4807 | 2.508 | 18.78 | 0.8875 | 2.551 | 0.3909 | 1078752 | 540775 | 4.212 | 0.6498 | 60 |
| mlp_flops | 14 | ok | 05adfad483698ec7 | 1.244 | 1.808 | 0.899 | 0.5323 | 2.823 | 59.38 | 0.8217 | 2.884 | 0.4378 | 1078752 | 540775 | 4.759 | 0.6467 | 60 |
| mlp_flops | 15 | ok | c9b331240ce9067f | 0.4983 | 1.721 | 0.9269 | 0.5144 | 2.89 | 29.97 | 0.8404 | 2.776 | 0.4563 | 1078752 | 540775 | 5.147 | 0.7805 | 60 |

E1 per-arm means over seeds:

| arm | nll1 | rmse1 | cov90_1 | nll_event | rmse10 | nll10 | cov90_10 | rmse1_contact | rmse1_free | flops_per_sample | params | train_seconds | predict_ms_per64 | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in | 0.1753 | 1.661 | 0.9397 | 0.4963 | 2.861 | 37.13 | 0.8644 | 2.742 | 0.3362 | 1.077e+06 | 2.051e+04 | 6.11 | 1.977 | 60 |
| mlp_flops | 0.5012 | 1.655 | 0.9239 | 0.492 | 2.63 | 36.06 | 0.8468 | 2.705 | 0.43 | 1.079e+06 | 5.408e+05 | 4.578 | 0.6839 | 60 |
| mlp128 | 0.6229 | 1.647 | 0.9216 | 0.5129 | 2.55 | 32.49 | 0.8435 | 2.672 | 0.4986 | 7.245e+04 | 3.656e+04 | 1.386 | 0.3621 | 60 |
| in_noedges | 0.2132 | 1.671 | 0.9351 | 0.527 | 2.791 | 96.42 | 0.7988 | 2.733 | 0.4365 | 1.499e+05 | 2.051e+04 | 2.409 | 0.7644 | 60 |
| in_norel | 0.1622 | 1.667 | 0.9424 | 0.5211 | 2.801 | 80.37 | 0.8114 | 2.728 | 0.4295 | 1.077e+06 | 2.051e+04 | 6.303 | 1.743 | 60 |

## E2 (225s wall)

### E2a-epistemic-has-predictive-value

- **verdict: accept** — mean improvement 0.2463 >= delta 0.1, CI [0.1551, 0.3368] excludes 0
- hypothesis: ens5 epistemic variance rank-correlates with held-out squared error (pooled id + gravity/box shifts) by >= 0.1 more than a permuted (information-free) signal
- primary `unc_err_spearman` (higher better), delta 0.1, basis `final`; baseline `null_perm`, candidate `ens5`, ablation `k1`, alternative `ens5_noboot`; seeds [21, 22, 23, 24, 25]
- prereg hash `b2c9e6849ab1`, frozen 1791069894376333000 ns; first result 1791069904688219000 ns
- report: `reports/full/E2a-epistemic-has-predictive-value-b2c9e6849a-1791070119278045000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| ens5 vs null_perm | final (decisive) | 5 | 0.2463 | 0.1551 | 0.3368 | 0 |
| ens5 vs null_perm | equal_interactions | 5 | 0.2463 | 0.1551 | 0.3368 | 0 |
| ens5 vs null_perm | equal_time | 0 | - | - | - | 5 |
| ens5 vs k1 (ablation_arm) | final | 5 | -0.2692 | -0.3477 | -0.1908 | 0 |
| ens5 vs ens5_noboot (alternative_arm) | final | 5 | 0.2133 | 0.09885 | 0.3277 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['ens5_iid']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.247 | 0.06118 |
| k1 | 5/5 | 0 | 0 | 0.5162 | 0.04489 |
| null_perm | 5/5 | 0 | 0 | 0.0006922 | 0.02754 |
| ens5_noboot | 5/5 | 0 | 0 | 0.0337 | 0.04364 |
| ens5_iid | 5/5 | 0 | 0 | 0.2777 | 0.07609 |

### E2b-epistemic-beats-single-model-variance

- **verdict: reject** — CI upper bound -0.19 < delta 0.05: the hypothesised effect is ruled out
- hypothesis: ens5 epistemic variance predicts error better (Spearman, pooled) than a single model's own predictive variance (K=1)
- primary `unc_err_spearman` (higher better), delta 0.05, basis `final`; baseline `k1`, candidate `ens5`, ablation `ens5_noboot`, alternative `null_perm`; seeds [21, 22, 23, 24, 25]
- prereg hash `897f604206bc`, frozen 1791069894376400000 ns; first result 1791070119283921000 ns
- report: `reports/full/E2b-epistemic-beats-single-model-variance-897f604206-1791070119284929000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| ens5 vs k1 | final (decisive) | 5 | -0.2692 | -0.3431 | -0.19 | 0 |
| ens5 vs k1 | equal_interactions | 5 | -0.2692 | -0.3431 | -0.19 | 0 |
| ens5 vs k1 | equal_time | 0 | - | - | - | 5 |
| ens5 vs ens5_noboot (ablation_arm) | final | 5 | 0.2133 | 0.09885 | 0.3277 | 0 |
| ens5 vs null_perm (alternative_arm) | final | 5 | 0.2463 | 0.1551 | 0.3368 | 0 |

- note: criterion unmet: baseline 'k1' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['ens5_iid']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.247 | 0.06118 |
| k1 | 5/5 | 0 | 0 | 0.5162 | 0.04489 |
| null_perm | 5/5 | 0 | 0 | 0.0006922 | 0.02754 |
| ens5_noboot | 5/5 | 0 | 0 | 0.0337 | 0.04364 |
| ens5_iid | 5/5 | 0 | 0 | 0.2777 | 0.07609 |

### E2c-coverage-under-shift

- **verdict: inconclusive** — CI [-0.008297, 0.0207] with mean 0.006204 neither meets nor rules out delta 0.02
- hypothesis: the K=5 mixture's 90% intervals stay closer to nominal under shift (mean |cov90-0.9| over g4, g20, box6) than K=1's by >= 0.02
- primary `cov90_shift_abs_err` (lower better), delta 0.02, basis `final`; baseline `k1`, candidate `ens5`, ablation `ens5_noboot`, alternative `ens5_iid`; seeds [21, 22, 23, 24, 25]
- prereg hash `92394e58fa5d`, frozen 1791069894376449000 ns; first result 1791070119290196000 ns
- report: `reports/full/E2c-coverage-under-shift-92394e58fa-1791070119291193000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| ens5 vs k1 | final (decisive) | 5 | 0.006204 | -0.008297 | 0.0207 | 0 |
| ens5 vs k1 | equal_interactions | 5 | 0.006204 | -0.008297 | 0.0207 | 0 |
| ens5 vs k1 | equal_time | 0 | - | - | - | 5 |
| ens5 vs ens5_noboot (ablation_arm) | final | 5 | 0.001269 | -0.001063 | 0.003277 | 0 |
| ens5 vs ens5_iid (alternative_arm) | final | 5 | 0.001019 | -0.0009902 | 0.003027 | 0 |

- note: criterion unmet: baseline 'k1' is preserved
- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)
- harness warning: exploratory arms not in the prereg (reported, never decisive): ['null_perm']

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| ens5 | 5/5 | 0 | 0 | 0.02325 | 0.004685 |
| k1 | 5/5 | 0 | 0 | 0.02945 | 0.01 |
| null_perm | 5/5 | 0 | 0 | 0.02325 | 0.004685 |
| ens5_noboot | 5/5 | 0 | 0 | 0.02452 | 0.005289 |
| ens5_iid | 5/5 | 0 | 0 | 0.02427 | 0.005626 |

### E2 per-run outcomes (shared by every E2 prereg)

| arm | seed | status | data_fp | unc_err_spearman | spearman_id | spearman_g4 | spearman_g20 | spearman_box6 | cov90_id | cov90_g4 | cov90_g20 | cov90_box6 | cov90_shift_abs_err | nll_id | nll_g4 | nll_g20 | nll_box6 | signal_ratio_g4 | signal_ratio_g20 | signal_ratio_box6 | conf_wrong_frac_id | conf_wrong_frac_g4 | conf_wrong_frac_g20 | conf_wrong_frac_box6 | outside3sd_frac_id | outside3sd_frac_g4 | outside3sd_frac_g20 | outside3sd_frac_box6 | train_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ens5 | 21 | ok | 64680c6639b6d52d | 0.3163 | 0.4384 | 0.4519 | 0.5007 | 0.5519 | 0.909 | 0.9225 | 0.8856 | 0.9117 | 0.0162 | 0.6486 | 0.4508 | 1.034 | 0.5556 | 0.7127 | 1.065 | 1.995 | 0 | 0 | 0.01333 | 0 | 0.01833 | 0.01333 | 0.035 | 0.01 | 6.85 |
| ens5 | 22 | ok | ce5ea45a545dc1fb | 0.2345 | 0.4308 | 0.3627 | 0.5011 | 0.3451 | 0.9033 | 0.9099 | 0.8529 | 0.8915 | 0.02181 | 0.6765 | 0.5901 | 1.23 | 0.6478 | 1.052 | 1.332 | 3.331 | 0 | 0.001667 | 0.001667 | 0 | 0.01833 | 0.01833 | 0.06833 | 0.02333 | 7.881 |
| ens5 | 23 | ok | 9cac146ced565994 | 0.2467 | 0.4486 | 0.3352 | 0.3949 | 0.1626 | 0.9189 | 0.9404 | 0.8932 | 0.9389 | 0.0287 | 0.7091 | 0.5247 | 0.9544 | 0.5544 | 0.9772 | 1.27 | 2.07 | 0 | 0 | 0.005 | 0 | 0.008333 | 0.006667 | 0.02 | 0.001667 | 6.669 |
| ens5 | 24 | ok | 6601da99128cbf97 | 0.2835 | 0.394 | 0.1864 | 0.4031 | 0.3497 | 0.9358 | 0.9424 | 0.8969 | 0.9261 | 0.02384 | 0.5465 | 0.3908 | 1.006 | 0.5573 | 1.035 | 1.352 | 2.001 | 0 | 0 | 0.003333 | 0 | 0 | 0 | 0.025 | 0.001667 | 6.244 |
| ens5 | 25 | ok | db446a4e0430a090 | 0.1539 | 0.2609 | 0.08411 | 0.4111 | 0.2644 | 0.925 | 0.9335 | 0.8615 | 0.8949 | 0.02569 | 0.7157 | 0.4505 | 1.148 | 0.7079 | 0.9871 | 1.233 | 2.845 | 0.001667 | 0 | 0.01 | 0 | 0.01333 | 0.003333 | 0.02833 | 0.02 | 6.684 |
| ens5_iid | 21 | ok | 64680c6639b6d52d | 0.2894 | 0.3883 | 0.4174 | 0.4413 | 0.3655 | 0.904 | 0.9169 | 0.884 | 0.9169 | 0.01662 | 0.6679 | 0.4613 | 1.05 | 0.5625 | 0.8773 | 1.256 | 2.293 | 0.001667 | 0 | 0.008333 | 0 | 0.01167 | 0.01333 | 0.03167 | 0.008333 | 8.163 |
| ens5_iid | 22 | ok | ce5ea45a545dc1fb | 0.2812 | 0.4568 | 0.5449 | 0.4812 | 0.4236 | 0.9031 | 0.9121 | 0.8518 | 0.8968 | 0.02116 | 0.7022 | 0.605 | 1.272 | 0.7351 | 0.8843 | 1.276 | 2.788 | 0 | 0 | 0.01 | 0 | 0.02333 | 0.01667 | 0.07333 | 0.01333 | 6.798 |
| ens5_iid | 23 | ok | 9cac146ced565994 | 0.1492 | 0.3 | 0.3328 | 0.2939 | 0.01962 | 0.9207 | 0.9426 | 0.8893 | 0.941 | 0.03144 | 0.7119 | 0.5429 | 0.9821 | 0.5783 | 0.8744 | 1.064 | 2.623 | 0.001667 | 0.003333 | 0.006667 | 0 | 0.006667 | 0.005 | 0.02 | 0.001667 | 6.564 |
| ens5_iid | 24 | ok | 6601da99128cbf97 | 0.3284 | 0.5517 | 0.313 | 0.5625 | 0.2286 | 0.939 | 0.9499 | 0.8989 | 0.9289 | 0.02662 | 0.5753 | 0.4316 | 1.031 | 0.5798 | 0.9526 | 1.246 | 2.484 | 0.001667 | 0 | 0.003333 | 0 | 0 | 0 | 0.02667 | 0.001667 | 7.625 |
| ens5_iid | 25 | ok | db446a4e0430a090 | 0.3403 | 0.4067 | 0.4241 | 0.4844 | 0.3858 | 0.9222 | 0.9349 | 0.8604 | 0.9021 | 0.02551 | 0.6702 | 0.4181 | 1.129 | 0.6387 | 0.7989 | 1.308 | 2.374 | 0 | 0 | 0.008333 | 0 | 0.01 | 0.005 | 0.02667 | 0.01667 | 6.125 |
| ens5_noboot | 21 | ok | 64680c6639b6d52d | 0.01507 | 0.1224 | 0.05332 | 0.2419 | 0.07297 | 0.9103 | 0.9219 | 0.8839 | 0.9149 | 0.01764 | 0.6811 | 0.4829 | 1.09 | 0.5863 | 0.9936 | 0.9968 | 3.218 | 0 | 0 | 0.02 | 0 | 0.008333 | 0.008333 | 0.03167 | 0.006667 | 6.471 |
| ens5_noboot | 22 | ok | ce5ea45a545dc1fb | 0.03356 | 0.02842 | 0.08613 | 0.2982 | 0.133 | 0.9053 | 0.9129 | 0.8543 | 0.8974 | 0.02042 | 0.7719 | 0.7186 | 1.339 | 0.7546 | 1.044 | 1.168 | 6.432 | 0 | 0.003333 | 0.01167 | 0 | 0.01833 | 0.015 | 0.06167 | 0.01167 | 7.688 |
| ens5_noboot | 23 | ok | 9cac146ced565994 | 0.08146 | 0.273 | 0.2737 | 0.2528 | 0.03335 | 0.9206 | 0.9417 | 0.8876 | 0.9372 | 0.03042 | 0.7629 | 0.5716 | 1.026 | 0.6032 | 1.045 | 1.124 | 4.338 | 0.001667 | 0.005 | 0.006667 | 0 | 0.008333 | 0.003333 | 0.02 | 0.001667 | 6.128 |
| ens5_noboot | 24 | ok | 6601da99128cbf97 | -0.02859 | 0.141 | -0.2048 | 0.1336 | -0.05631 | 0.9394 | 0.9506 | 0.9007 | 0.9303 | 0.02718 | 0.6095 | 0.4672 | 1.095 | 0.6196 | 1.255 | 1.128 | 4.526 | 0.005 | 0 | 0.01667 | 0 | 0 | 0 | 0.02333 | 0 | 6.128 |
| ens5_noboot | 25 | ok | db446a4e0430a090 | 0.06702 | 0.04198 | 0.07093 | 0.2193 | 0.3225 | 0.9208 | 0.9344 | 0.8586 | 0.895 | 0.02694 | 0.7331 | 0.4567 | 1.187 | 0.7228 | 1.022 | 1.158 | 3.352 | 0.006667 | 0 | 0.01667 | 0 | 0.01167 | 0.003333 | 0.03 | 0.02 | 6.455 |
| k1 | 21 | ok | 64680c6639b6d52d | 0.5445 | 0.4719 | 0.3341 | 0.6106 | 0.6493 | 0.9085 | 0.9197 | 0.8788 | 0.8471 | 0.0313 | 0.7277 | 0.5147 | 1.181 | 0.7977 | 0.7702 | 1.11 | 0.661 | 0.001667 | 0 | 0.02 | 0.005 | 0.008333 | 0.006667 | 0.03 | 0.015 | 1.356 |
| k1 | 22 | ok | ce5ea45a545dc1fb | 0.5706 | 0.5971 | 0.3856 | 0.6102 | 0.5742 | 0.9033 | 0.9153 | 0.8544 | 0.8683 | 0.03083 | 0.8049 | 0.7471 | 1.406 | 0.8062 | 0.8608 | 1.143 | 0.8224 | 0 | 0.001667 | 0.01333 | 0.001667 | 0.01833 | 0.01333 | 0.06833 | 0.015 | 1.77 |
| k1 | 23 | ok | 9cac146ced565994 | 0.5248 | 0.5879 | 0.335 | 0.6377 | 0.6004 | 0.9207 | 0.9401 | 0.8876 | 0.8886 | 0.0213 | 0.7933 | 0.585 | 1.072 | 0.735 | 0.9437 | 1.029 | 0.9679 | 0.001667 | 0.003333 | 0.006667 | 0.001667 | 0.01 | 0.005 | 0.01667 | 0.001667 | 1.23 |
| k1 | 24 | ok | 6601da99128cbf97 | 0.4655 | 0.4927 | 0.3645 | 0.5359 | 0.4675 | 0.9392 | 0.9515 | 0.8981 | 0.8954 | 0.01935 | 0.6213 | 0.479 | 1.157 | 0.6924 | 0.9551 | 1.112 | 0.9181 | 0.001667 | 0 | 0.01333 | 0.003333 | 0 | 0 | 0.02167 | 0.001667 | 1.229 |
| k1 | 25 | ok | db446a4e0430a090 | 0.4756 | 0.4642 | 0.3207 | 0.6491 | 0.5053 | 0.9154 | 0.9319 | 0.8569 | 0.8415 | 0.04449 | 0.8016 | 0.5095 | 1.31 | 0.9773 | 0.8008 | 1.015 | 0.729 | 0 | 0 | 0.01167 | 0.01333 | 0.01333 | 0.006667 | 0.03833 | 0.035 | 2.707 |
| null_perm | 21 | ok | 64680c6639b6d52d | -0.02828 | -0.02641 | -0.02811 | -0.06124 | 0.01071 | 0.909 | 0.9225 | 0.8856 | 0.9117 | 0.0162 | 0.6486 | 0.4508 | 1.034 | 0.5556 | 0.9439 | 0.9422 | 0.9428 | 0.005 | 0 | 0.05667 | 0.015 | 0.01833 | 0.01333 | 0.035 | 0.01 | 6.663 |
| null_perm | 22 | ok | ce5ea45a545dc1fb | -0.01902 | -0.04071 | -0.06055 | -0.06708 | 0.01879 | 0.9033 | 0.9099 | 0.8529 | 0.8915 | 0.02181 | 0.6765 | 0.5901 | 1.23 | 0.6478 | 0.9691 | 1.015 | 0.9234 | 0.01 | 0.003333 | 0.04333 | 0.01667 | 0.01833 | 0.01833 | 0.06833 | 0.02333 | 6.287 |
| null_perm | 23 | ok | 9cac146ced565994 | 0.003762 | -0.03684 | -0.0179 | 0.05225 | 0.02005 | 0.9189 | 0.9404 | 0.8932 | 0.9389 | 0.0287 | 0.7091 | 0.5247 | 0.9544 | 0.5544 | 1.035 | 1.013 | 0.9984 | 0.001667 | 0.005 | 0.03167 | 0.005 | 0.008333 | 0.006667 | 0.02 | 0.001667 | 6.934 |
| null_perm | 24 | ok | 6601da99128cbf97 | 0.04296 | 0.04918 | 0.01223 | -0.005161 | 0.008359 | 0.9358 | 0.9424 | 0.8969 | 0.9261 | 0.02384 | 0.5465 | 0.3908 | 1.006 | 0.5573 | 0.9491 | 1.021 | 0.9677 | 0.005 | 0 | 0.05333 | 0.01 | 0 | 0 | 0.025 | 0.001667 | 6.228 |
| null_perm | 25 | ok | db446a4e0430a090 | 0.004035 | 0.05866 | -0.08621 | 0.04286 | -0.01447 | 0.925 | 0.9335 | 0.8615 | 0.8949 | 0.02569 | 0.7157 | 0.4505 | 1.148 | 0.7079 | 1.008 | 1.022 | 1.02 | 0.005 | 0 | 0.04167 | 0.01833 | 0.01333 | 0.003333 | 0.02833 | 0.02 | 7.075 |

E2 per-arm means over seeds:

| arm | unc_err_spearman | spearman_id | spearman_g4 | spearman_g20 | spearman_box6 | cov90_id | cov90_g4 | cov90_g20 | cov90_box6 | cov90_shift_abs_err | nll_id | nll_g4 | nll_g20 | nll_box6 | signal_ratio_g4 | signal_ratio_g20 | signal_ratio_box6 | conf_wrong_frac_id | conf_wrong_frac_g4 | conf_wrong_frac_g20 | conf_wrong_frac_box6 | outside3sd_frac_id | outside3sd_frac_g4 | outside3sd_frac_g20 | outside3sd_frac_box6 | train_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ens5 | 0.247 | 0.3946 | 0.284 | 0.4422 | 0.3347 | 0.9184 | 0.9297 | 0.878 | 0.9126 | 0.02325 | 0.6593 | 0.4814 | 1.075 | 0.6046 | 0.9528 | 1.251 | 2.448 | 0.0003333 | 0.0003333 | 0.006667 | 0 | 0.01167 | 0.008333 | 0.03533 | 0.01133 | 6.866 |
| k1 | 0.5162 | 0.5227 | 0.348 | 0.6087 | 0.5594 | 0.9174 | 0.9317 | 0.8752 | 0.8682 | 0.02945 | 0.7498 | 0.5671 | 1.225 | 0.8017 | 0.8661 | 1.082 | 0.8197 | 0.001 | 0.001 | 0.013 | 0.005 | 0.01 | 0.006333 | 0.035 | 0.01367 | 1.659 |
| null_perm | 0.0006922 | 0.0007767 | -0.03611 | -0.007674 | 0.008687 | 0.9184 | 0.9297 | 0.878 | 0.9126 | 0.02325 | 0.6593 | 0.4814 | 1.075 | 0.6046 | 0.981 | 1.003 | 0.9705 | 0.005333 | 0.001667 | 0.04533 | 0.013 | 0.01167 | 0.008333 | 0.03533 | 0.01133 | 6.637 |
| ens5_noboot | 0.0337 | 0.1214 | 0.05586 | 0.2292 | 0.1011 | 0.9193 | 0.9323 | 0.877 | 0.9149 | 0.02452 | 0.7117 | 0.5394 | 1.147 | 0.6573 | 1.072 | 1.115 | 4.373 | 0.002667 | 0.001667 | 0.01433 | 0 | 0.009333 | 0.006 | 0.03333 | 0.008 | 6.574 |
| ens5_iid | 0.2777 | 0.4207 | 0.4065 | 0.4527 | 0.2846 | 0.9178 | 0.9313 | 0.8769 | 0.9171 | 0.02427 | 0.6655 | 0.4918 | 1.093 | 0.6189 | 0.8775 | 1.23 | 2.512 | 0.001 | 0.0006667 | 0.007333 | 0 | 0.01033 | 0.008 | 0.03567 | 0.008333 | 7.055 |

## E3 (32s wall)

### E3a-tracker-vs-perframe-std

- **verdict: accept** — mean improvement 0.0158 >= delta 0.005, CI [0.01552, 0.01615] excludes 0
- hypothesis: [std] tracker state predicts next-frame detections with OSPA (c=0.1) lower by >= 0.005 than per-frame NN-velocity features
- primary `ospa_obs` (lower better), delta 0.005, basis `final`; baseline `nnvel`, candidate `tracker`, ablation `tracker_noapp`, alternative `static`; seeds [31, 32, 33, 34, 35]
- prereg hash `1b4fe12a84bc`, frozen 1791070119296826000 ns; first result 1791070120246214000 ns
- report: `reports/full/E3a-tracker-vs-perframe-std-1b4fe12a84-1791070130082713000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| tracker vs nnvel | final (decisive) | 5 | 0.0158 | 0.01552 | 0.01615 | 0 |
| tracker vs nnvel | equal_interactions | 5 | 0.0158 | 0.01552 | 0.01615 | 0 |
| tracker vs nnvel | equal_time | 0 | - | - | - | 5 |
| tracker vs tracker_noapp (ablation_arm) | final | 5 | 0.0001987 | 0.0001177 | 0.0002927 | 0 |
| tracker vs static (alternative_arm) | final | 5 | 0.01084 | 0.01001 | 0.01158 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| tracker | 5/5 | 0 | 0 | 0.04012 | 0.0002217 |
| nnvel | 5/5 | 0 | 0 | 0.05591 | 0.0002139 |
| static | 5/5 | 0 | 0 | 0.05096 | 0.0007365 |
| tracker_noapp | 5/5 | 0 | 0 | 0.04032 | 0.0002441 |

### E3b-tracker-vs-perframe-hard

- **verdict: accept** — mean improvement 0.01437 >= delta 0.005, CI [0.01397, 0.01475] excludes 0
- hypothesis: [hard] tracker state predicts next-frame detections with OSPA (c=0.1) lower by >= 0.005 than per-frame NN-velocity features
- primary `ospa_obs` (lower better), delta 0.005, basis `final`; baseline `nnvel`, candidate `tracker`, ablation `tracker_noapp`, alternative `static`; seeds [31, 32, 33, 34, 35]
- prereg hash `3d5961e2e0ac`, frozen 1791070119296891000 ns; first result 1791070131952041000 ns
- report: `reports/full/E3b-tracker-vs-perframe-hard-3d5961e2e0-1791070151682353000.json`

| comparison (sign-adjusted, +ve = candidate better) | basis | pairs | mean diff | CI low | CI high | skipped |
|---|---|---|---|---|---|---|
| tracker vs nnvel | final (decisive) | 5 | 0.01437 | 0.01397 | 0.01475 | 0 |
| tracker vs nnvel | equal_interactions | 5 | 0.01437 | 0.01397 | 0.01475 | 0 |
| tracker vs nnvel | equal_time | 0 | - | - | - | 5 |
| tracker vs tracker_noapp (ablation_arm) | final | 5 | 0.0008915 | 0.0007551 | 0.001014 | 0 |
| tracker vs static (alternative_arm) | final | 5 | 0.0124 | 0.01195 | 0.01276 | 0 |

- note: few seeds: three seeds are not a guarantee of statistical reliability (plan §7.2.6)

| arm | ok/runs | failed | budget | mean primary | std |
|---|---|---|---|---|---|
| tracker | 5/5 | 0 | 0 | 0.05183 | 0.0005391 |
| nnvel | 5/5 | 0 | 0 | 0.0662 | 0.0004421 |
| static | 5/5 | 0 | 0 | 0.06423 | 0.0003829 |
| tracker_noapp | 5/5 | 0 | 0 | 0.05272 | 0.0005312 |

### E3 per-run outcomes (shared by every E3 prereg)

| arm | seed | status | data_fp | ospa_obs | ospa_oracle_PRIVILEGED | ospa_obs_p90 | mean_predicted_count | ms_per_frame | frames_scored |
|---|---|---|---|---|---|---|---|---|---|
| nnvel | 31 | ok | 8e86f8e509eae52e | 0.05563 | 0.05041 | 0.07166 | 6.424 | 0.01603 | 1032 |
| nnvel | 32 | ok | 97b81cd59989c5ef | 0.05586 | 0.05079 | 0.0722 | 6.353 | 0.01711 | 1032 |
| nnvel | 33 | ok | dd49f241f48f1830 | 0.05615 | 0.05107 | 0.07281 | 6.374 | 0.01952 | 1032 |
| nnvel | 34 | ok | a6944ef2e8260d75 | 0.05582 | 0.05092 | 0.07157 | 6.393 | 0.01735 | 1032 |
| nnvel | 35 | ok | 61304cff38bc7fec | 0.0561 | 0.05111 | 0.07205 | 6.398 | 0.01602 | 1032 |
| static | 31 | ok | 8e86f8e509eae52e | 0.05128 | 0.04584 | 0.06499 | 6.424 | 0.004862 | 1032 |
| static | 32 | ok | 97b81cd59989c5ef | 0.05047 | 0.04509 | 0.06566 | 6.353 | 0.0005502 | 1032 |
| static | 33 | ok | dd49f241f48f1830 | 0.04992 | 0.04423 | 0.06449 | 6.374 | 0.0005372 | 1032 |
| static | 34 | ok | a6944ef2e8260d75 | 0.05145 | 0.0462 | 0.06458 | 6.393 | 0.0005487 | 1032 |
| static | 35 | ok | 61304cff38bc7fec | 0.05167 | 0.04644 | 0.06449 | 6.398 | 0.0005246 | 1032 |
| tracker | 31 | ok | 8e86f8e509eae52e | 0.04006 | 0.02874 | 0.0575 | 5.404 | 0.7441 | 1032 |
| tracker | 32 | ok | 97b81cd59989c5ef | 0.04014 | 0.02934 | 0.05678 | 5.373 | 0.7059 | 1032 |
| tracker | 33 | ok | dd49f241f48f1830 | 0.03991 | 0.0285 | 0.05666 | 5.371 | 0.9709 | 1032 |
| tracker | 34 | ok | a6944ef2e8260d75 | 0.03999 | 0.02887 | 0.05567 | 5.401 | 0.8677 | 1032 |
| tracker | 35 | ok | 61304cff38bc7fec | 0.04049 | 0.02925 | 0.05658 | 5.371 | 0.68 | 1032 |
| tracker_noapp | 31 | ok | 8e86f8e509eae52e | 0.04024 | 0.02891 | 0.05769 | 5.394 | 0.7884 | 1032 |
| tracker_noapp | 32 | ok | 97b81cd59989c5ef | 0.04025 | 0.02945 | 0.0568 | 5.367 | 0.6838 | 1032 |
| tracker_noapp | 33 | ok | dd49f241f48f1830 | 0.04008 | 0.02869 | 0.05701 | 5.354 | 0.6822 | 1032 |
| tracker_noapp | 34 | ok | a6944ef2e8260d75 | 0.04029 | 0.02921 | 0.05607 | 5.391 | 0.8964 | 1032 |
| tracker_noapp | 35 | ok | 61304cff38bc7fec | 0.04073 | 0.02951 | 0.05684 | 5.348 | 0.7291 | 1032 |

E3 per-arm means over seeds:

| arm | ospa_obs | ospa_oracle_PRIVILEGED | ospa_obs_p90 | mean_predicted_count | ms_per_frame | frames_scored |
|---|---|---|---|---|---|---|
| tracker | 0.04012 | 0.02894 | 0.05664 | 5.384 | 0.7937 | 1032 |
| nnvel | 0.05591 | 0.05086 | 0.07206 | 6.389 | 0.01721 | 1032 |
| static | 0.05096 | 0.04556 | 0.06484 | 6.389 | 0.001405 | 1032 |
| tracker_noapp | 0.04032 | 0.02915 | 0.05688 | 5.371 | 0.756 | 1032 |

---

# Post-fix re-run (2026-10-03)

Everything above this line is the original verification and is unchanged. This section
re-runs E1 and E2 after bugs 1–3 were fixed in `developmental_ai/`. E3 does not use the
changed code and was not re-run.

## What was fixed

1. **Static dims (bug 1)**, in `mechanisms/_nets.py` (also `residual.py` and the RSSM
   baseline). A continuous outcome dim is *static* when its change is exactly zero over
   the training data (`|next − current| ≤ 1e-9`), or when the caller declares it with
   `declare_static`. Static dims stay as inputs. They are masked out of the NLL and
   predicted as the identity (mean = current value exactly, variance =
   `VAR_FLOOR·sd²`). The mask can only shrink: a dim that later moves is learned from
   then on. If declared-static data moves, `fit` raises `ContractError`, and
   `undeclare_static` lifts it. With this change the verifier's divergence repro trains
   monotonically: the IN's per-epoch loss rose by at most −0.27 nats, where it used to
   jump +16 to +76.
2. **Resumable training (bug 2).** Each mechanism now keeps one Adam optimiser across
   fits; each fit's lr and weight decay are set in place. `training_state_dict()` /
   `load_training_state_dict()` save and restore the weights, optimiser moments,
   version, support and static mask (on the IN, MLP, residual and RSSM mechanisms and
   on `BootstrapEnsemble`). A restored model then continues bit-for-bit.
   `state_dict()` (snapshot weights) is unchanged. **E1 and E2 use single fresh fits,
   so this fix does not change their numbers.**
3. **`equal_interactions` (bug 3)**, in `experiments/ab.py` `_at_interactions`. A run
   that used exactly the common interaction budget now contributes its reported
   primary. Before, it contributed the last curve point at that count. In E1c the
   equal_interactions line now reads +3.04, the same as `final`. The old figure of +118
   came from the 480-epoch overfit MLP.

Bug 4 (equal_time ablation/alternative pairs with single-point curves) is unchanged.

## Protocol

- **Preregistrations: identical.** They are the same `experiments()` definitions,
  re-frozen before any arm ran. A JSON diff against the original reports shows only
  `frozen_at_ns` and `frozen_hash` differ. The hash covers the freeze time, so it
  changes on every run. The hash does not cover code, so nothing was refused and no
  `*_postfix` prereg was needed.
- **Seeds, splits, metrics, arms, epochs, lr and acceptance rules: unchanged.**
  `data_fp` per seed matches the original run (for example seed 11 `48e64930…` and
  seed 21 `64680c66…`).
- **Command:** `PYTHONPATH=. python -m experiments.foundation_ab.run full E1 E2`.
  Tables are in `generated_full_E1_E2.md`; reports are appended under
  `reports/full/`.
- **Wall-clock caveat.** Other agents were loading the machine during this run (load
  average about 4 on 8 cores). Train seconds went up for every arm (IN 10.4 s vs 6.1 s
  originally). The equal_time basis in E1c depends on that.

## Verdicts: original vs post-fix

| prereg | original | **post-fix** | key numbers (post-fix, 5 seeds) |
|---|---|---|---|
| E1a IN vs FLOP-matched MLP, nll1 | inconclusive | **accept** | NLL: IN −0.023 (was 0.175), MLP-661 2.73 (was 0.50). Gain 2.76, CI [1.71, 3.71]. |
| E1b IN vs FLOP-matched MLP, rmse10 | reject | **inconclusive** | RMSE10: IN 2.669 (was 2.861), MLP-661 2.671 (was 2.63). Diff +0.002, CI [−0.071, 0.084]. |
| E1c IN vs MLP-128, equal time, nll1 | inconclusive | **inconclusive** | Equal-time gain 45.3, CI [−3.4, 133.5]. equal_interactions now +3.04 (was a spurious +118). |
| E2a epistemic beats null signal | accept | **accept** | Spearman 0.589 vs 0.003, CI [0.547, 0.620] (was 0.247). |
| E2b epistemic beats K=1 own variance | reject | **reject** | 0.589 vs 0.645, diff −0.057, CI [−0.103, −0.016] (was −0.27). |
| E2c K=5 coverage under shift beats K=1 | inconclusive | **accept** | \|cov90−0.9\|: 0.042 vs 0.064, gain 0.0218 ≥ δ 0.02, CI [0.0116, 0.0327]. This is barely past δ. |

## What changed, and what to be careful about

**E1: the fix changed which arm wins, mostly through calibration and not through the mean.**
- **Point error moved, and the IN improved most:**

  | arm | rmse1 | rmse10 |
  |---|---|---|
  | IN | 1.661 → 1.451 | 2.861 → 2.669 |
  | MLP-661 | 1.655 → 1.539 | 2.63 → 2.671 |
  | MLP-128 | 1.647 → 1.677 | 2.55 → 3.023 |

  Before the fix, 1-step RMSE was the same for every arm. It no longer is.
- **The MLPs' held-out NLL went much worse:** MLP-661 0.50 → 2.73 and MLP-128
  0.62 → 3.02, while their cov90 dropped to about 0.87. The ball dims now get all of
  the gradient, and the 60-epoch, lr 3e-3 recipe overfits them into overconfidence.
  The mechanisms smoke measures the same thing on its data: held-out NLL is best near
  10 epochs, not 60. The old loss's wall term was acting as an accidental brake.
  **The recipe was frozen against the broken loss.** The E1a accept therefore says
  "the IN stays calibrated under this recipe and the MLPs do not". It does not show a
  better mean.
- **The relational part is now partly supported:**
  - On rmse10 the IN beats the no-edges ablation by 0.23, CI [0.06, 0.41]. That was
    not significant before.
  - On nll1 the ablation gap is still not significant: 0.27, CI [−0.34, 0.80].
  - `in_norel` is now clearly worse than the IN (nll1 1.07, rmse10 3.06). Before, it
    beat the IN.
- **Not re-checked:** a re-tuned recipe (fewer epochs or early stopping) for the new
  loss. That would be a new preregistration, not this re-run.

**E2: the "bootstrap does the work" finding was an artifact of bug 1.**
- **Without resampling, the ensemble now matches the bootstrap one:** `ens5_noboot`
  Spearman 0.600 vs ens5 0.589 (it was 0.034 vs 0.247). Seed diversity alone now gives
  members that disagree usefully.
- **K=1's own variance still predicts error better** (0.645), but the gap fell from
  0.27 to 0.057.
- **K=1 is now badly overconfident:** nll_id 3.83 vs 0.87 for ens5, and outside-3sd on
  g20 13.5% vs 6.1%. This is the same over-training effect as in E1. It is what makes
  E2c an accept.
- **"Increase appropriately" is still not met:** mean epistemic variance changes by
  ×0.67 under g4 (it went *down*, as before), ×1.60 under g20 and ×1.47 under box6.
- **Confident errors went down a lot:** on g20, confidently wrong is now 0.10%, against
  0.67% before. The in-distribution rate is 0.03%.

## Tests added (each fails on the pre-fix code)

- `tests/_foundation_mechanisms_smoke.py` **K**:
  - divergence: the old code jumped +16 to +76 nats; this is shown by a probe, because
    the old `fit` has no `loss_hist`
  - exact wall mask for IN, MLP and RSSM
  - bit-exact identity on walls
  - the escape path (a moving wall dim is re-learned)
  - refusal and undeclare for a contradicted declaration
- **C** (the gravity probe) was rebuilt. It used to select steps by their *outcome*
  (no contact event). The model predicts E[dv | state], and near walls and balls that
  expectation includes the chance of a bounce. That biased g_hat about 25% low on
  every seed and on both torch stacks. C now selects (state, ball) pairs where no
  contact is possible, judged from the input state alone. On those pairs the true
  dv gives exactly 9.81, and the model reads 9.03 (torch 2.10) and 8.59 (torch 2.6).
  The thresholds are unchanged and C uses a single seed. To fit the suite budget,
  torch threads are pinned to 2 and the RSSM and G models train fewer epochs.
- `tests/unit/test_foundation_mechanisms_unit.py` adds four cases:
  - static-mask rules and the head_loss mask
  - bit-exact pass-through
  - Adam step count across fits (10 = 4 + 6)
  - training-state round trip that resumes bit-exact; restoring weights without the
    optimiser moments diverges
- `tests/unit/test_foundation_inference_unit.py` adds an ensemble training-state round
  trip.
- `tests/unit/test_foundation_ab_unit.py` adds a check that equal_interactions on a
  fixed-data curve reads the primary. The old code gave 120.0 instead of 0.62.
