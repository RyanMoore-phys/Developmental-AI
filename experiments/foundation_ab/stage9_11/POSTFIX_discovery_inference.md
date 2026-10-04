# Post-fix re-run: discovery and inference (B1, F5, F6)

Written by the fixer, not the verifier. `RESULTS.md` and the verifier's
`generated_*` files and `reports/full/` are untouched.

**Reproduce**

```
PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.postfix_discovery_inference   # ~30 s
```

- Reports: `reports/postfix/` (JSON and md).
- Tables: `generated_postfix_discovery_inference.md`.

**Preregs**

- E9 and E10 were re-run through the verifier's own `experiments()` and arms, with
  `run.run_module` and a different output directory.
- The frozen hash covers `frozen_at_ns`, so every freeze gets a new hash. I compared
  every re-run prereg with the verifier's saved JSON one field at a time. The only
  differences are `frozen_at_ns` and `frozen_hash`, so no `*_postfix` copies were
  needed.
- One new prereg, **E10d-postfix-misspecification-monitor**, was frozen before its
  arms ran. The original E10d arms cannot read a flag, because there was no API.

## B1. The search budget is now honoured

**Fix**
- `discovery/search.py`: `iter_proposals` enumerates proposals lazily and checks the
  deadline before every data-touching step. The search loop checks the deadline
  before each `apply` and `structure_key`. With `rng` set, enumeration also stops at
  the deadline.
- `discovery/table.py`: `levels()` and `is_discrete()` are cached per column. A
  non-integer column is rejected in O(n), without a sort.
- `_cap_rows` uses one `np.unique` instead of a scan per episode.
- The proposal order is unchanged. The eager and lazy lists are byte-identical on a
  mixed discrete/continuous fixture, and gate J is unchanged at 1.82 nats/row.

| E9f arm (5 seeds, `wall_s` 0.25) | before: overshoot (elapsed / wall_s) | after |
|---|---|---|
| large_table (150 vars × 60k rows) | **17.79×** (4.1–4.7 s), 1 candidate evaluated | **1.05×** (0.25–0.27 s), 4 evaluated |
| small_table | 0.46× | 0.23× |
| hung_fitter | 1.04× | 1.02× |

**Verdict.** The verdict label is still "accept", because the prereg measures
large − small ≥ 0.5 and the small table now finishes its search at 0.23×. The
hypothesis itself ("overshoots by > 50%") is now false in absolute terms: elapsed is
1.05× the budget.

**Still open**
- An abandoned fit thread cannot be killed in Python. `leaked_threads` is still 1 per
  hung fit.
- `max_bytes` still bounds only retained candidates.
- The up-front fit/validation split is O(rows) and is not budgeted. It costs about
  0.06 s at 60k rows.

**Regression test.** `tests/_foundation_discovery_smoke.py` K covers 150 vars × 60k
rows, with the default fitter and with a 0.05 s/fit fitter.
- It must return within 1.5 × `wall_s`.
- Before the fix it failed at 10.5× with 1 candidate evaluated.
- Now: 1.03× and 1.01×.

## F5. The change detector now sees variance-inflating changes

**Fix** (`inference/change.py`)
- The dispersion CUSUM increment was `q − k_disp` with `k_disp = 1`, so it drifted up
  only when E[z²] > 2.41.
- It is now Page's log-likelihood-ratio increment for "variance is `disp_ratio` × the
  model's", with `disp_ratio` 3 and `h_disp` 16 nats. Its zero-drift point is E[z²] =
  1.65.
- z² is clipped at 16 per dimension, and the mean statistic at ±4 per step. One
  outlier is never an alarm on its own. Before, a single |z| > 6.5 fired the
  dispersion alarm.
- Constructor change: `k_disp` is replaced by `disp_ratio`, `disp_clip` and
  `mean_clip`. No caller passed `k_disp`.

**Detector, measured directly.** The residual is x + N(0, 0.5²), scored at sd 0.5,
which is the 2→3 slope change. 400 runs.

| | alarm within 80 rows | within 160 | median first alarm | null dispersion alarms, 250 × 4000 steps, D = 1 / 2 / 6 |
|---|---|---|---|---|
| before | 49% | 76% | 83 rows | 1 / 0 / 0 |
| after | 74% | 99% | 61 rows | **0 / 0 / 0** |

- Mean-shift null alarms are unchanged: 9–15 per 10⁶ steps before and after.
- Model variance understated 1.2×: 0 alarms per 1000 steps both before and after.
  At 1.4×: 0.08 (was 0.03).
- A variance 16× too large is now caught in a median 6 steps (was 2), because of the
  clip.
- I rejected the more sensitive setting (ratio 2.5, h 12; median 46 rows). It fired
  once on the stage 7 ensemble's in-distribution predictive, which broke inference
  smoke G. At 3/16 that stream peaks at 13.3 of 16.

**E9b `full_slope3` (exploratory arm, original prereg split)**

| | alarms per seed | searches per seed | root replaced | NLL |
|---|---|---|---|---|
| before | 1, 6, 3, 1, 0 | 0, 1, 1, 1, 0 | 1/5 | 1.259 |
| after | 4, 7, 4, 2, 3 | **1, 1, 1, 1, 1** | 1/5 | 1.259 |

The detector now triggers revision on 5/5 seeds, but the held-out NLL does not move.
Each alarm is followed by 8 episodes of search evidence and then 2 judging windows of
4 episodes each. That fits in 16 post-change episodes only if the alarm comes in the
first post episode, and with noise 0.5 that is rare (2% within 20 rows).

Ad hoc measurement, not preregistered: with 32 post-change episodes (64-id split),
- the fixed detector replaces the root on **5/5** seeds, NLL 0.748;
- the pre-fix detector replaces it on 4/5, NLL 0.871;
- leaving the root in place gives NLL 1.28–1.55.

The reconstructed pre-fix detector reproduces the verifier's alarm counts exactly
(1, 6, 3, 1, 0). Every other E9 arm is numerically identical to the verifier's run
(E9a 0.8596, E9b control 0.8001, E9c 0/5, E9d −0.1733, E9e −0.1905).

**Regression test.** `tests/_foundation_inference_smoke.py` I.
- Requires ≥ 95% of runs to alarm within 160 rows and a median ≤ 80 rows; the old
  detector got 76% and 88. Now: 99% and 62.
- 0 dispersion alarms in 50 × 4000 steps of null noise.
- One 8-sd outlier gives 0 alarms.

## F6. Misspecification is now flagged

**Fix** (`inference/hypotheses.py`)
- New `MisspecificationMonitor`: Page's CUSUM of log q1(o) − log q0(o).
  - q0 is the set's posterior predictive *before* the observation, optionally widened
    by `tolerance`.
  - q1 = (1 − 0.5)·q0 + 0.5·wide, where wide is uniform over the outcomes, or a 4×
    variance for a Gaussian.
  - q1 is a proper distribution, so the false-alarm run length is ≥ eᵇ for **any**
    predictive. Default h = 8, about 3000 observations.
- `HypothesisSet.update_categorical(table, outcome)` gives the same posterior as
  `update` and feeds the monitor.
- It is read through `hs.misspecified` and `hs.misspecification()`. The report
  includes statistic, threshold, evidence count, LLR, flags, `flagged_at`, and the
  number of unmonitored updates.
- Flag lifecycle (§4.1):
  - The flag clears when the statistic returns to 0.
  - `add_hypothesis` resets it, because a changed set is a new question. That is the
    escape path.
- I discarded an earlier draft that standardised the score deficit. On the skewed
  scores of a confident binary set, it flagged 20–45% of well-specified runs.

**Wiring into discovery** (`discovery/revision.py`)
- `ChangeTriggeredReviser` runs a monitor with Gaussian scores (tolerance 0.1, h 14)
  on the incumbent's predictive.
- When it is flagged, it opens a revision exactly like a CUSUM alarm: event
  `"misspecified"` with the evidence count, and a contradiction recorded.
- It is reset after the search runs and whenever the incumbent changes. A still-wrong
  incumbent must earn the flag again before the next bounded search. It is
  level-triggered, so it is not a latch.
- `rv.misspecification()` exposes the report.

**E10d** (5 seeds × 11 held-out scenarios, budget 60, the verifier's split)

| arm | confident wrong | flagged | confident wrong **and unflagged** |
|---|---|---|---|
| original `info_misspec` (re-run, unchanged) | 100% | — | **100%** |
| postfix `info_misspec_nomon` (run to budget, no monitor) | 100% | 0% | 100% |
| postfix `info_misspec_mon` (candidate) | 100% | 80% | **20%** |
| postfix `random_misspec_mon` (ablation) | 100% | 9% | 91% |
| postfix `info_wellspec_mon` (well-specified) | 1.8% | **0%** | 1.8% |

**Verdict: accept**, Δ 0.80, CI [0.66, 0.92].

**Limits, kept honest**
- The flag never comes **before** confidence. Confidence arrives at about 13
  interactions and the flag at a median of about 45. A consumer gets a flag on an
  already-confident set, not a prevention.
- Random querying produces too little contradicting evidence within 60 interactions
  (9%).
- Gaussian scores in the reviser:
  - On the slope/misfit fixtures, the fixed CUSUM usually fires first.
  - In smoke L, with the detector live, the monitor opened 0/8 revisions. It is a
    second line there.
  - It proves its wiring when the detector is blind: 8/8 seeds open a revision and
    end up using h, against 0/8 with the monitor off too.

**Regression tests**
- `tests/_foundation_inference_smoke.py` J
  - Misspecified set: 37/40 flagged within 80 interactions (requirement ≥ 36).
  - Well-specified: 1/200 flagged in 200 interactions. The bound is ≤ 1; it was
    measured at 1/300 over 300 seeds. A sequential test cannot promise zero.
  - Adding a hypothesis resets the flag.
  - Before the fix there was no API, so the test fails.
- `tests/_foundation_discovery_smoke.py` L: the reviser wiring and a null with 0
  triggers and 0 searches.
- Unit tests: `tests/unit/test_foundation_inference_unit.py` (`misspec_monitor_parts`,
  `hypothesis_update_categorical`, `cusum_dispersion_params`) and
  `tests/unit/test_foundation_discovery_unit.py`
  (`table_level_cache_and_lazy_enumeration`).

## Suites after the change

All of these pass:
- unit and smoke for discovery, inference, mechanisms, selection and planning
  (planning uses `PredictiveCUSUM`);
- `_foundation_ab_verify9_11_smoke`;
- `_foundation_ab_smoke`;
- the ab unit tests.

The inference smoke went from 20 s to about 31 s of CPU, because contracts I and J
were added.
