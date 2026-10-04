# Stage 9–11 verification A/Bs: results

An independent verifier ran these comparisons. The verifier did not write the code
under test and edited nothing under `developmental_ai/`. Every comparison uses the
plan §7.2 harness (`developmental_ai/foundation/experiments/ab.py`).

**Protocol**
- Each module's preregistrations were frozen inside `experiments()` before any arm ran.
  Arms shared by several preregs are memoised, and each memoised computation is stamped
  after the last freeze. The harness re-checked the order for every report.
- 5 seeds (0–4) per prereg, split by episode or scenario.
- Every prereg has an ablation arm. Most also have an alternative arm.
- Arms that share a world consumed identical arrays: `data_fp` is the same for every
  (seed, split). The smoke test checks this.
- Every run finished with status `ok`. No failures, no budget overruns.
- Negative results are kept.

**Pilot.** One pilot ran on seed 100 (not a final seed) to check feasibility. It changed
two things before the freeze, both reported below:
- The E9b control was moved from slope 2→3 to slope 2→4, because 2→3 raised zero change
  alarms at noise 0.5, so every arm stayed on the root and the control tested nothing.
  2→3 is kept as an exploratory arm.
- The E9f large table was raised to 150 variables.

**Reproduce**

```
PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.run full       # ~2 min
PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.run followup   # post-hoc E9c sweep
PYTHONPATH=. python tests/_foundation_ab_verify9_11_smoke.py              # ~25 s, protocol only
```

**Where things are**
- Raw reports: `reports/full/` (JSON and md, never overwritten, plus `index.jsonl`).
- Full per-run tables with every metric: `generated_full_E9_E10_E11.md`.
- The appendix below gives the primary metric per seed.

**Compute**
- Full run: 126 s wall, 111 s CPU (8-core macOS, Python 3.12, numpy 1.26).
- Follow-up: 2 s. Smoke: 24 s. Pilots: about 40 s.
- Total: under 4 CPU-minutes.
- Per-arm wall times are in the appendix. The second prereg of a shared arm set shows
  about 0 s because of the memo.
- E11c measures wall-clock latency using `sleep`, so its numbers depend on the
  machine's timer.

## Verdicts

| prereg | verdict | key numbers (5 seeds) | what it means for the claim |
|---|---|---|---|
| **E9a**: full language beats REFIT-only after a structural change, on new noisy data with 7 distractors | **accept** | Held-out NLL (robust): full 0.860, REFIT 1.652, no update 4.397. Gain 0.79 nats/row, CI [0.71, 0.88]. | **Holds** on harder data. The gain is less than half the reported 1.82, which is expected with 2.5× the noise. The no-split ablation is identical (Δ 6e-5): **SPLIT_APPLICABILITY contributes nothing here**. REPLACE_TRANSITION(piecewise, temp) does the work. The full search hits `max_candidates` = 150 on every seed for the same answer that no-split finds in 27–42. |
| **E9b**: control, slope 2→4: does REFIT-only beat full (does structure invent harm)? | **reject** | Both arms 0.800 exactly on all 5 seeds. | **Holds.** No structure was invented. **But** the exploratory arm with the implementers' own slope 2→3 at noise 0.5 kept the stale root on 4/5 seeds (mean NLL 1.26; the one seed that adapted reached 0.75). Adaptation **fails silently** there (see finding F5). |
| **E9c**: pure noise, 12 distractors, forced search: are structural mechanisms promoted? | **reject** | 0/5 promotions, 0 structural acceptances. Post-hoc, 60 more seeds: 2/60 searches accepted one structural candidate, 0/60 promoted. | **Holds.** The registry's fresh-window test filters the 3% search-level false positives. |
| **E9d**: misfit confined to 3% of rows | **accept** | Gaussian NLL: full −0.173, REFIT 4.07. r captured on 5/5 seeds. | **The 1% outlier component does not hide this misfit**: eps≈0 gives identical results. However, the **robust metric** puts REFIT at 0.19 against a Gaussian 4.07. The implementers' gate metric (robust NLL) compresses large errors, so their 1.82 figure is in compressed units. |
| **E9e**: misfit confined to 1% of rows | **accept** | Full −0.191, REFIT 0.592 (Gaussian). r captured on 5/5 seeds, through ADD_DEPENDENCY. | Holds. |
| **E9f**: does `search()` overshoot `wall_s` = 0.25 s on a large table? | **accept** (claim falsified) | 150 vars × 60k rows: **4.1–4.7 s (16–19× the budget)** with only 1 candidate evaluated. Small table: 0.46×. Hung fitter: 1.04×. | **"BUDGETS (all explicit, all honoured)" is false.** Proposal enumeration runs outside the deadline guard. Bug B1. |
| **E10a**: replication through the library's own fixture, new parameters (K=8, noise 0.5, budget 80) | **accept** | Info-gain 11.3, random 28.9, ICM 28.9, LP 37.9, entropy 81 (censored) interactions. Gain 17.7, CI [12.6, 23.0]. | **Holds** and replicates. |
| **E10b**: four noisy TVs | **accept** | Info-gain 5.8, random 15.8, ICM 20.1 (43% of picks on TVs), entropy 61 (97% on TVs). | Holds. |
| **E10c**: learnable but useless distractor (a 4-bit deterministic lamp) | **accept** (weakness confirmed) | Joint info-gain 10.6 with **44% of interactions spent on the lamp**. Task-marginal info-gain 5.1 (3% on the lamp). Gain 5.5, CI [4.1, 7.2]. Joint still beats random (19.7) and ICM (17.1). | **Info-gain wastes interactions on any uncertainty in the hypothesis set, relevant or not.** A deterministic distractor pays log 2 per query, against about 0.25 nats for a noisy response, so it is chosen first. `usefulness` and `task` default to weight 0, so nothing steers it back to the question asked. |
| **E10d**: misspecified hypothesis set (truth not contained) | **accept** (weakness confirmed) | Confident (>0.9) on a wrong shift with no flag: **100%** of scenarios, in 12.8 interactions on average. Well-specified: 1.8%. Random: also 100%. The verifier's simple posterior-predictive flag fired first in only 7%. | **Nothing detects or flags misspecification.** There is no API for it. The posterior floor does not prevent a confident wrong answer. |
| **E11a**: rare-region wrong model: is the gated planner better than the ungated one? | **inconclusive** | Mean distance: gated 0.863, ungated 0.879, fallback 0.904, gated with the true model 0.672. Δ 0.016, CI [−0.032, 0.060]. | The gate cut planner-controlled steps in the wrong region by 97% (6.4 against 203 per seed). It did not significantly improve the real outcome. |
| **E11b**: does exploitation leak through the gate (fallback beats gated)? | **reject** | Fallback is worse than gated by 0.041, CI [−0.082, 0.0004]. | **The gate is not net-harmful**, so the claim survives in that sense. **But** the planner re-entered the wrong region 4.8 times per seed (20 episodes): each time the window flushed on fallback steps outside the region, the gate re-opened, with no memory of where the model failed (F7). Gated runs spent *more* steps in the mud than the fallback alone (41% against 36%), and the planner held control only 46% of the time. |
| **E11c**: heavy-tailed model latency against a 10 ms deadline | **accept** (claim falsified as "hard") | Decisions over 2× the deadline: heavy tail **8.5%** (worst 5.4×), constant-latency model 1.9% (worst 3.7×). | The deadline holds only up to one synchronous chunk, as documented ("Overrun is bounded by ONE chunk"). That chunk is unbounded: smaller chunks (8) made it **worse** (15.5%) through more reprobes. Even a constant-latency model breaks the deadline. |
| **E11d**: skills that differ only outside the probe distribution | **accept** (weakness confirmed) | Flagged as duplicates in **92%** of trials, yet they disagree on **29%** of deployment states (N(0,4²)). An unrelated same-name policy: flagged 0%. | Duplicate detection is only as good as probe coverage, and nothing checks that coverage. |
| **E11e**: a copy with logits ×3 (identical argmax everywhere) | **accept** (weakness confirmed) | Flagged in **0%** of trials (JS 0.10 bits). Deterministic behaviour is identical on 100% of states. | The JS-on-distributions signature calls two deterministically identical skills distinct. `ActorCriticFallback` defaults to `deterministic=True` (planner.py:619). Weight drift 0.1 is not flagged (8.8% argmax disagreement); drift 0.01 is flagged (0.8%). |

The claims named in the brief, as tested:

- **Name versus behaviour.** Confirmed: a byte copy under another name is flagged 100%,
  and a same-name, different policy is flagged 0%. The bridge keys identity on
  `skill_id`, not on the name.
- **Idle and no-op actions accrue no persistent progress.** Confirmed in the direction
  that matters: no positive net progress (E10e). But see B2: the tracker invents
  forgetting instead.

## Findings and bugs (file:line, minimal repro)

**B1. Discovery search budget is not honoured. Severity: medium.**
- `discovery/search.py:358` calls `enumerate_proposals(source, fit_t, ...)` outside
  `guarded()`. So do `p.apply` and `structure_key` (around :363).
- `enumerate_proposals` (`search.py:160`) calls `_best_split` (an argsort) and
  `table.levels` once per variable. `table.levels` (`table.py:172`) is an `np.unique`
  sort and is recomputed by `is_discrete` (:176) about three times per variable.
- Profile of 150 vars × 60k rows: 2.6 s of 2.7 s is inside `enumerate_proposals`. Of
  that, 1.4 s is sorting from `np.unique`.
- Repro: `search(initial_mechanism("y"), EvidenceTable({v0..v149, y: noise}, 60k rows,
  600 episodes), SearchBudget(wall_s=0.25))` returns after 4.1–4.7 s, with
  `stopped="wall"` and `evaluated=1`.
- Also: `_run_with_timeout` (`search.py:139`) abandons a timed-out fit thread,
  which keeps burning CPU. One thread leaked per hung fit (E9f).
- `max_bytes` bounds only the serialized size of retained candidates, not working memory.

**B2. Retention tracker invents forgetting on stationary noise. Severity: medium.**
- Location: `experiments/forgetting.py:110-113`. Gains are discounted (`w =
  relearn_discount ** learned`) whenever the state is not KNOWN, but regressions are
  always charged in full.
- So a probe whose stationary loss straddles the hysteresis band has a strongly negative
  E[progress]. That contradicts the docstring (:28): "a stationary model whose probe losses
  merely fluctuate has E[progress] = 0".
- Repro: `RetentionTracker(0.45, 0.55)`, 16 probes, i.i.d. N(0.5, 0.2) losses for 2000
  evaluations.
  - Net progress −3011 to −3057 over 5 seeds. The telescoped first-minus-last total is
    between −1.4 and +0.6.
  - **16/16 probes flagged as forgetting loops.**
  - With N(0.3, 0.2): −1270 net, 7–12 of 16 flagged.
  - With N(0.2, 0.05), which never crosses the thresholds: about 0, the correct value.
- The flag clears only after 5 consecutive KNOWN evaluations (:131), which straddling
  noise never produces. While the noise persists, that is a latch (CLAUDE.md §4.1).
- A usefulness term built on this would penalise noisy but stable competences.

**B3. Planner deadline is not hard. Severity: low; documented.**
- Location: `planning/planner.py:404-421`. A chunk is synchronous, so the overrun is one
  chunk's worst case.
- 8.5% of decisions take more than 2× the deadline under a heavy-tailed model, and 1.9%
  under a constant-latency model.
- The reprobe escape (:405) makes smaller chunks worse (15.5%).

**F4. The robust NLL hides misfit in the reported metric. Severity: reporting.**
- `forms._robust_ll` (`forms.py:299`, ROBUST_EPS = 0.01) caps each row's cost.
- The implementers' gate numbers use `mech.loglik`, which is robust: REFIT on the 3%
  rare-row world scores 0.19 robust against 4.07 Gaussian.
- It does **not** stop search from finding the misfit (E9d/e, ablation identical). It only
  compresses reported gaps.

**F5. Silent non-adaptation for variance-inflating parameter changes. Severity: medium.**
- Location: `inference/change.py:43,73-76`. The dispersion CUSUM drifts upward only when
  E[z²] > 1 + √2 · k_disp = 2.41 (k_disp = 1).
- At slope 2→3 with noise 0.5, the residual variance grows 2.33×, so the dispersion
  statistic drifts downward. Alarms come only from chance excursions (0–6 per run).
- The reviser therefore keeps the stale root on 4/5 seeds: mean NLL 1.26, against 0.75 on the one seed that
  refitted (E9b, `full_slope3`).

**F6. Experiment selection has no notion of relevance or of misspecification. Severity:
design gap.**
- See E10c (44% of interactions on a useless deterministic lamp) and E10d (100% confident
  wrong, no flag).
- `hypothesis_information` (`information.py`) scores I(H;O) over the whole set.

**F7. The reliability gate has no memory of where the model failed. Severity: design
gap.**
- Location: `planner.py:161-236`. `ReliabilityMonitor` is a sliding window of recent
  z² values.
- Validations of fallback steps outside the bad region refill the window, and the gate
  re-opens. The planner then returns to the region it was wrong about: 4.8 entries per
  20 episodes.
- The fix belongs in the model (fit on the failures), or in a region- or applicability-
  aware gate. The gate alone cannot recover the 0.19 m lost against the true model.

**F8. Duplicate detection depends on probes and on the stochastic policy. Severity:
design gap.**
- Location: `planning/skills.py:336-371`. Same-name collisions are handled correctly
  through `skill_id`.
- The probe set is the caller's. Nothing checks that it covers deployment states, and
  divergence is computed on action distributions rather than on executed behaviour
  (E11d/e).

**Smaller observations**
- **E9a validation power.** A search on 8 post-change episodes has only 4 validation
  episodes. In the pilot (seed 100) the correct piecewise(temp) structure was rejected
  as inconclusive (t = 0.28), because a threshold misplaced by 0.06 cost one validation
  episode −0.7 nats/row.
- **E11c.** Even the constant-latency planner ceded about 26% of decisions to
  `fallback_timeout` at 10 ms (E11c).

## What this means for the gates

- **Stage 9.** The completion gate replicates on harder, new data (+0.79 nats/row, CI
  excludes 0). No hallucinated structure on pure noise (0/65 promotions) or on the slope
  control.
  - The budget claim is false for wide or large tables (B1).
  - The implementers' own slope 2→3 control is not detected at noise 0.5 (F5), so
    "adaptation" can fail silently.
  - SPLIT_APPLICABILITY added nothing measurable in E9a.
- **Stage 10.** Replicates strongly (E10a/b). Noisy TVs are handled exactly, the
  I(H;O) = 0 case.
  - Wastes interactions on learnable but irrelevant structure (E10c).
  - Converges confidently on misspecified sets without a flag (E10d).
  - The forgetting tracker produces false forgetting on stationary noise (B2).
- **Stage 11.** The gate prevents *sustained* exploitation (−97% planner steps in the
  wrong region). It does not prevent *repeated* exploitation (F7), and it gave no
  significant real-outcome gain over the ungated planner.
  - The deadline is soft (B3).
  - Behavioural duplicate detection works for the name cases in the brief but fails off
    the probes and under tempering (E11d/e).
  - The completion gate ("improved real behaviour") is **not shown** by these fixtures:
    gated is better than fallback by only 0.041 m, CI [−0.0004, 0.082].

## Appendix: per-seed primary metric, every arm, every prereg

(The second prereg of a shared arm set shows wall about 0 s because of the memo.)

**E10a-replication-K8-noise0.5** — `interactions_to_identify` (lower better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| info_gain (candidate) | 11.22 | 12.39 | 12.17 | 10 | 10.5 | 11.26 | 1.34 |
| random (baseline) | 29 | 36.28 | 25.44 | 29.28 | 24.56 | 28.91 | 3.58 |
| icm (alternative) | 31.61 | 33.78 | 29.72 | 31.44 | 17.72 | 28.86 | 3.65 |
| entropy (ablation) | 81 | 81 | 81 | 81 | 81 | 81 | 8.99 |
| lp (exploratory) | 40.67 | 49 | 30.83 | 49.5 | 19.33 | 37.87 | 4.29 |

Decisive paired diff (+ = candidate better): 17.66, 95% CI [12.64, 22.95]; prereg hash `07a678f4385c`.

**E10b-four-noisy-TVs** — `interactions_to_identify` (lower better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| info_joint (candidate) | 5.667 | 6.083 | 6.333 | 5.167 | 5.833 | 5.817 | 0.46 |
| random (baseline) | 14.25 | 13.75 | 15 | 16.75 | 19.25 | 15.8 | 0.32 |
| icm (alternative) | 20.17 | 18.67 | 20.42 | 17.25 | 23.92 | 20.08 | 0.47 |
| entropy (ablation) | 61 | 61 | 61 | 61 | 61 | 61 | 4.37 |
| lp (exploratory) | 25.42 | 19.08 | 28.08 | 15.5 | 19.58 | 21.53 | 0.47 |

Decisive paired diff (+ = candidate better): 9.983, 95% CI [7.212, 13.1]; prereg hash `d559e8a8bb87`.

**E10c-learnable-useless-distractor** — `interactions_to_identify` (lower better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| info_joint (baseline) | 11.12 | 10.88 | 12 | 10.38 | 8.625 | 10.6 | 0.69 |
| info_marginal (candidate) | 6.75 | 5.375 | 4.375 | 5.75 | 3.375 | 5.125 | 0.57 |
| random (ablation) | 15.5 | 23 | 23 | 17.5 | 19.5 | 19.7 | 0.38 |
| icm (alternative) | 21.88 | 15.88 | 18.75 | 13.25 | 15.88 | 17.12 | 0.38 |

Decisive paired diff (+ = candidate better): 5.475, 95% CI [4.128, 7.177]; prereg hash `9d74dc992b85`.

**E10d-misspecified-hypothesis-set** — `confident_wrong_unflagged` (higher better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| info_wellspec (baseline) | 0 | 0.09091 | 0 | 0 | 0 | 0.01818 | 0.29 |
| info_misspec (candidate) | 1 | 1 | 1 | 1 | 1 | 1 | 0.66 |
| random_misspec (ablation) | 1 | 1 | 1 | 1 | 1 | 1 | 0.27 |
| info_ppc_misspec (alternative) | 0.9091 | 0.9091 | 0.9091 | 1 | 0.9091 | 0.9273 | 0.68 |

Decisive paired diff (+ = candidate better): 0.9818, 95% CI [0.9242, 1.011]; prereg hash `2fd51cf6b351`.

**E11a-gate-vs-ungated-rare-region** — `mean_dist` (lower better), verdict **inconclusive**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| fallback (ablation) | 0.7467 | 0.9167 | 0.9728 | 0.8749 | 1.009 | 0.904 | 0.37 |
| gated (candidate) | 0.707 | 0.8373 | 0.957 | 0.8736 | 0.9404 | 0.863 | 7.40 |
| ungated (baseline) | 0.7512 | 0.8541 | 0.9126 | 0.9294 | 0.9467 | 0.8788 | 14.53 |
| gated_true (alternative) | 0.5309 | 0.6868 | 0.7842 | 0.73 | 0.6299 | 0.6724 | 13.54 |

Decisive paired diff (+ = candidate better): 0.01575, 95% CI [-0.0321, 0.05984]; prereg hash `2508ca667846`.

**E11b-exploitation-leaks-through-gate** — `mean_dist` (lower better), verdict **reject**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| fallback (candidate) | 0.7467 | 0.9167 | 0.9728 | 0.8749 | 1.009 | 0.904 | 0.00 |
| gated (baseline) | 0.707 | 0.8373 | 0.957 | 0.8736 | 0.9404 | 0.863 | 0.00 |
| ungated (ablation) | 0.7512 | 0.8541 | 0.9126 | 0.9294 | 0.9467 | 0.8788 | 0.00 |
| gated_true (exploratory) | 0.5309 | 0.6868 | 0.7842 | 0.73 | 0.6299 | 0.6724 | 0.00 |

Decisive paired diff (+ = candidate better): -0.04094, 95% CI [-0.08231, 0.0004376]; prereg hash `e47d881a7340`.

**E11c-deadline-heavy-tailed-model** — `overrun_fraction` (higher better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| constant (baseline) | 0.02667 | 0.03333 | 0 | 0.02 | 0.01333 | 0.01867 | 5.34 |
| heavy_tail (candidate) | 0.07333 | 0.09333 | 0.1 | 0.07333 | 0.08667 | 0.08533 | 3.37 |
| heavy_tail_chunk8 (ablation) | 0.16 | 0.1467 | 0.1467 | 0.1667 | 0.1533 | 0.1547 | 5.64 |

Decisive paired diff (+ = candidate better): 0.06667, 95% CI [0.04344, 0.09411]; prereg hash `b02fc1205e18`.

**E11d-offprobe-false-duplicate** — `flag_rate` (higher better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| name_only (ablation) | 1 | 1 | 1 | 1 | 1 | 1 | 1.01 |
| samename_diff (baseline) | 0 | 0 | 0 | 0 | 0 | 0 | 0.97 |
| offprobe (candidate) | 0.95 | 0.8 | 0.9 | 1 | 0.95 | 0.92 | 1.06 |
| tempered (exploratory) | 0 | 0 | 0 | 0 | 0 | 0 | 0.95 |
| drift_0.001 (exploratory) | 1 | 1 | 1 | 1 | 1 | 1 | 1.00 |
| drift_0.01 (exploratory) | 1 | 1 | 1 | 1 | 1 | 1 | 0.99 |
| drift_0.1 (exploratory) | 0 | 0 | 0 | 0 | 0 | 0 | 0.93 |

Decisive paired diff (+ = candidate better): 0.92, 95% CI [0.825, 0.9992]; prereg hash `7bdc16f4e89a`.

**E11e-tempered-copy-not-flagged** — `flag_rate` (lower better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| name_only (baseline) | 1 | 1 | 1 | 1 | 1 | 1 | 0.00 |
| samename_diff (ablation) | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| offprobe (exploratory) | 0.95 | 0.8 | 0.9 | 1 | 0.95 | 0.92 | 0.00 |
| tempered (candidate) | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| drift_0.001 (exploratory) | 1 | 1 | 1 | 1 | 1 | 1 | 0.00 |
| drift_0.01 (exploratory) | 1 | 1 | 1 | 1 | 1 | 1 | 0.00 |
| drift_0.1 (exploratory) | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |

Decisive paired diff (+ = candidate better): 1, 95% CI [1, 1]; prereg hash `68ea5467a4c2`.

**E9a-structural-vs-refit-noisy-regime** — `nll_robust` (lower better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| refit_only (baseline) | 1.626 | 1.664 | 1.688 | 1.614 | 1.667 | 1.652 | 0.14 |
| full (candidate) | 0.8448 | 0.9398 | 0.8437 | 0.7317 | 0.9383 | 0.8596 | 2.05 |
| full_no_split (ablation) | 0.8448 | 0.9401 | 0.8437 | 0.7317 | 0.9383 | 0.8597 | 0.48 |
| no_update (alternative) | 3.875 | 4.647 | 4.58 | 4.497 | 4.384 | 4.397 | 0.04 |

Decisive paired diff (+ = candidate better): 0.7923, 95% CI [0.7067, 0.879]; prereg hash `2c627817b956`.

**E9b-control-slope-change-structure-hurts** — `nll_robust` (lower better), verdict **reject**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| full (baseline) | 0.8321 | 0.7485 | 0.8858 | 0.7464 | 0.7876 | 0.8001 | 0.59 |
| refit_only (candidate) | 0.8321 | 0.7485 | 0.8858 | 0.7464 | 0.7876 | 0.8001 | 0.14 |
| no_update (ablation) | 2.973 | 3.385 | 3.163 | 2.717 | 2.826 | 3.013 | 0.04 |
| full_slope3 (exploratory) | 1.41 | 0.7474 | 1.541 | 1.254 | 1.344 | 1.259 | 0.40 |

Decisive paired diff (+ = candidate better): 0, 95% CI [0, 0]; prereg hash `290c087734d1`.

**E9c-false-structure-on-pure-noise** — `false_struct_promotions` (higher better), verdict **reject**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| refit_only (baseline) | 0 | 0 | 0 | 0 | 0 | 0 | 0.06 |
| full (candidate) | 0 | 0 | 0 | 0 | 0 | 0 | 0.76 |
| full_no_split (ablation) | 0 | 0 | 0 | 0 | 0 | 0 | 0.30 |

Decisive paired diff (+ = candidate better): 0, 95% CI [0, 0]; prereg hash `485697446257`.

**E9d-rare-misfit-f0.03** — `nll_gauss` (lower better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| refit_only (baseline) | 5.139 | 2.506 | 3.638 | 3.092 | 5.973 | 4.07 | 0.14 |
| full_robust (candidate) | -0.1734 | -0.2369 | -0.1949 | -0.1476 | -0.1136 | -0.1733 | 1.00 |
| full_eps0 (ablation) | -0.1734 | -0.2369 | -0.1949 | -0.1476 | -0.1136 | -0.1733 | 1.01 |

Decisive paired diff (+ = candidate better): 4.243, 95% CI [2.527, 6.016]; prereg hash `e273ed0227d1`.

**E9e-rare-misfit-f0.01** — `nll_gauss` (lower better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| refit_only (baseline) | 0.5066 | 1.116 | 0.4964 | 1.073 | -0.2335 | 0.5916 | 0.14 |
| full_robust (candidate) | -0.1805 | -0.176 | -0.2334 | -0.1329 | -0.2297 | -0.1905 | 0.93 |
| full_eps0 (ablation) | -0.1805 | -0.176 | -0.2334 | -0.1329 | -0.2297 | -0.1905 | 1.00 |

Decisive paired diff (+ = candidate better): 0.7821, 95% CI [0.153, 1.371]; prereg hash `05e07b9d78cc`.

**E9f-search-wall-budget-overshoot** — `overshoot_ratio` (higher better), verdict **accept**

| arm | s0 | s1 | s2 | s3 | s4 | mean | wall s (sum) |
|---|---|---|---|---|---|---|---|
| small_table (baseline) | 0.4653 | 0.3665 | 0.4429 | 0.6645 | 0.3554 | 0.4589 | 0.58 |
| large_table (candidate) | 17.24 | 18.39 | 18.72 | 18.2 | 16.4 | 17.79 | 23.49 |
| hung_fitter (ablation) | 1.041 | 1.041 | 1.044 | 1.041 | 1.041 | 1.041 | 1.31 |

Decisive paired diff (+ = candidate better): 17.33, 95% CI [16.15, 18.43]; prereg hash `3acd1592f6c9`.

