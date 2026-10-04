# Post-fix: Stage 10 selection findings B2, F6/E10c, F6/E10d

Date: 2026-10-03. Scope: `developmental_ai/foundation/experiments/` only (not `ab.py`).
`RESULTS.md` is unchanged. Everything here is offline. No live reward, action, replay or
normalisation path is touched.

How to reproduce:

- `PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.postfix_selection` (about 70 s).
  It writes `generated_postfix_selection.md` and its reports go to `reports/postfix_selection/`.
- The new runner `postfix_selection.py` is the only file added in this directory. The verifier's
  modules are imported, not edited.

**Preregistrations.** All four E10 preregs are the **original frozen ones**, loaded from the
verifier's `reports/full/E10*.json`. Each passes `prereg.verify()` against its original hash
(`07a678f4385c`, `d559e8a8bb87`, `9d74dc992b85`, `2fd51cf6b351`) and was frozen before every
result here. No `*_postfix` prereg was needed.

The Stage 10 A/B in `tests/_foundation_selection_smoke.py` builds its prereg inline. Its content
is unchanged, so it was re-run as is.

## 1. B2: retention tracker invented forgetting on stationary noise

**Cause.** The tracker changed state on a *single* evaluation crossing a threshold. It discounted
gains while a probe was not known but charged regressions in full.

**Fix** (`forgetting.py`), in two parts.

*State changes now need a sustained, statistically significant level.* Two quantities are
compared:

- the mean of the last `window` = 4 losses;
- `z` = 3 standard errors, using a measured per-probe noise floor. The floor is the median of
  rolling 4-sample variances over the last 32 evaluations, unbiased by the χ² median.

A median of short-window variances ignores the few windows that straddle a real level shift.
So a sustained phase reads as signal and stationary scatter reads as noise.

*The weight multiplies gains and regressions alike within a state.* Inside any state, progress
therefore telescopes. Only genuine state changes are non-telescoping: a fall from known is charged
at full weight, and the climb back is credited at the relearning discount.

**Guard and its recovery (§4.1).** A flag is removed when either of these happens:

- The probe stays known for `max(stable_evals, 2 × longest known stretch it already forgot
  after)` evaluations. This is finite and known from the probe's own history.
- `reset_probe(reason)` is called, for example on a change-detector alarm.

Leaving the known state needs significant deterioration. So a noisy but retained probe
accumulates stability, which removes the B2 latch.

**Known limit.** A loop that alternates faster than `window` evaluations cannot be told apart from
two-point noise of the same amplitude. It is treated as noise: net progress 0, no flag, no credit
farmed. For that reason smoke contract G now holds each phase for 8 evaluations.

**E10e**: verifier probe, 16 probes, 2000 evaluations, 5 seeds:

| config | net progress, before | net progress, after | telescoped first−last | flagged /16, before → after | relearn events, before → after |
|---|---|---|---|---|---|
| straddle N(0.5, 0.2) | −3011 … −3057 | −2.8 … −0.6 | −1.4 … +0.6 | 16 → **0** | ~6400 → 0–2 |
| wide N(0.3, 0.2) | −1231 … −1302 | −1.4 … +0.6 (= telescoped) | same | 7–12 → **0** | ~2950 → 0 |
| far N(0.2, 0.05) | −0.3 … +0.2 | unchanged | same | 0 → 0 | 0 → 0 |
| straddle, sd annealed 0.6→0.2 | −6071 … −6163 | −13.0 … −5.2 | −2.8 … +1.2 | 16 → **0** | ~7100 → 2–7 |

**False-flag rate.** It is 0 out of 320 probe-runs here (16 probes × 4 configs × 5 seeds), and 0
out of 96 in the smoke. A non-stationary (annealed) noise level still produces a few spurious
forget/relearn events: residual bias of at most 13 against a raw clipped gain of 7200.

A genuine loop is still caught. In the smoke, a noisy loop (0.9↔0.1 every 12 evaluations, sd
0.05) flags 16/16 probes and nets −99, with 153 reported as discounted relearning. The sustained
loop in contract G flags 4/4 and nets −29.1 on a raw gain of 38.0, of which 32.9 is relearning.
A genuine one-time discovery still nets its full 38.0.

## 2. F6/E10c: a learnable but useless distractor took 44% of interactions

**Fix.** `information.targeted_information(probs, table, question)` computes I(Q;O), where Q
(the question variable) is one answer label per hypothesis. Nuisance factors are marginalised.

- I(H;O|Q) is returned separately as `nuisance` and is never part of `info_gain`.
- If knowing a nuisance factor would improve competence, that claim belongs to the separately
  weighted and logged `usefulness` term.

`question.ExperimentQuestion(hset, answer_of)` binds a hypothesis set to the question. The
fixture's `info_gain` arm now uses it, and `info_joint` (plain I(H;O)) is kept as the ablation
witness.

On the verifier's own fixture (E10c, original prereg `9d74dc992b85`, 5 seeds):

| arm | interactions to identify | lamp fraction |
|---|---|---|
| joint I(H;O), the pre-fix behaviour (verifier arm, reproduced) | 10.60 | **44.2%** |
| **library targeted I(shift;O)** (`ExperimentQuestion`) | **5.125** | **3.2%** |
| random | 19.70 | 34.6% |
| ICM | 17.12 | 28.6% |

- Verdict under the original prereg: **accept**. Gain 5.475, CI [4.13, 7.18].
- The library arm is decision-for-decision identical to the verifier's independently written
  `info_marginal` arm: same numbers to 4 digits, same `data_fp`. This cross-checks the
  implementation against an independent one.
- Library fixture, smoke N, 16 scenarios: targeted 2.3% lamp and 4.6 interactions, versus joint
  52.8% and 8.2.

**Not worse on the original noisy-TV fixture.** With a shift-only set, Q = H, so targeted equals
joint. The smoke checks 12 out of 12 scenarios with identical choices.

- Stage 10 A/B, smoke contract M, same prereg content: info_gain **5.48** vs random **10.50**,
  verdict **accept**, mean diff 5.02, CI [2.99, 7.08]. This is identical to before the fix. ICM
  13.22, entropy 59.26, LP 12.24 are also unchanged.
- E10a (library path, K=8, noise 0.5): info_gain 11.26 vs random 28.91, verdict accept, gain 17.66,
  CI [12.64, 22.95]. Identical to RESULTS.md.
- E10b (four TVs), library arm: 5.82 vs random 15.8, verdict accept. Identical.

## 3. F6/E10d: misspecified set, 100% confident-wrong, no flag

**Fix** (`question.py`), in three parts.

*Model check.* `MisspecificationMonitor` is a per-hypothesis anytime-valid e-detector.

- The bet is `((1−λ)T_h(o) + λ·U(o)) / T_h(o)`, with λ ∈ {0.25, 0.5, 1}. It has expectation 1
  under h.
- It runs as a CUSUM capped at 2× its threshold.
- A hypothesis is rejected at `max_λ ≥ |λ|/α` (α = 0.01).
- The set is flagged when **every** hypothesis is rejected. A noisy TV bets nothing.

*Consuming the inference flag.* inference/hypotheses.py now has `misspecified`, a
posterior-predictive score-deficit CUSUM. `ExperimentQuestion.observe` updates the set through
`update_categorical` whenever it exists, so that monitor sees every outcome, and ORs its flag in.
The hook is `question._inference_flag`, which reads the flag defensively.

*An honest conclusion.*

- A posterior above 0.9 makes the conclusion **provisional**, not resolved. While provisional,
  selection plans on the posterior mixed half-and-half with uniform. That makes the most
  informative experiment the one that could falsify the leading answer.
- The conclusion becomes **resolved** only after passing tests worth `confirm` = 5 expected
  failures. Each passed test is credited with its risk: the probability it would have failed had
  another alternative been true. A failed test resets the count and demotes even a resolved
  conclusion.
- When flagged, the state is `misspecified`, `reliable` is False and `request_revision` is True.
  `ExperimentQuestion.conclusion(exp)` writes the unreliable verdict into the Experiment record.
  `planning_probs()` re-opens targeted information so probing continues.
- Every `Selection` carries the question status in its log.

**Recovery (§4.1).** The flag clears in any of three ways:

- automatically, when some hypothesis fits again. The capped CUSUM bounds how many fitting
  outcomes this takes (about 45 sharp outcomes);
- when set membership changes, i.e. a structural revision restarts the check;
- through `reset(reason)`.

E10d, original prereg `2fd51cf6b351`, 5 seeds × 11 scenarios, budget 60:

| arm | conclusion reported wrong, unflagged (primary) | posterior > 0.9 on a wrong shift | flagged | resolved at |
|---|---|---|---|---|
| verifier: info, misspecified (pre-fix) | **100%** | 100% | — (own PPC: 7%) | — |
| **library: info, misspecified** | **1.8%** | 100% | **98.2%** | not resolved (budget) |
| library: random, misspecified | 1.8% | 100% | 50.9% | not resolved |
| library: info, well specified | 0% | 1.8% | **0%** (false flag) | 13.4 interactions |

- Verdict under the original prereg is now **reject**: mean diff 0.018, CI [−0.011, 0.076] < 0.3.
  The weakness the prereg asserted no longer holds.
- Bayes over a set without the truth still goes confident; the posterior is not changed. What
  changed is that the confidence is not reported as identification, and the set is flagged.
- The flag mostly fires **after** the posterior reaches 0.9: it fires first in only 3.6% of
  scenarios. That is the reason for the provisional/confirm stage.

Smoke contract O (library fixture, 20 + 20 scenarios):

- misspecified: reported-wrong 0%, flagged 100%;
- well specified: 100% resolved, 0 wrong, 0% false flags.

Larger sweep, 60 + 60 scenarios, info arm:

| confirm | reported wrong (misspecified) | flagged (misspecified) | resolved at (well specified) |
|---|---|---|---|
| 4 | 10% | 90% | 12.2 |
| 5 (default) | 3.3% | 96.7% | 14.9 |
| 6 | 1.7% | 98.3% | 16.5 |

These sweep numbers were measured before the inference flag was OR-ed in. The confirm = 5 row
was re-measured with it.

**The price.** On a well-specified set the evaluator-side identification time does not change
(5.18 on E10d, 5.48 on the Stage 10 A/B). Reporting a *resolved* conclusion takes about 8 more
interactions (13.4 on E10d, 14.9 in the library fixture).

## Tests

| suite | before the fix | after |
|---|---|---|
| `tests/_foundation_selection_smoke.py` (26 contracts) | the new G, G2, N, O contracts **FAIL** on the original code | ALL PASS |
| `tests/unit/test_foundation_selection_unit.py` | — | 22/22 |
| `tests/_foundation_ab_smoke.py`, `tests/unit/test_foundation_ab_unit.py` | — | ALL PASS |
| `tests/_foundation_ab_verify9_11_smoke.py` | — | ALL PASS |
| `tests/_metrics_sink_smoke.py` (contract 7: `InfraSink` keeps a private ledger) | — | ALL PASS |

Failure details on the original code:

- G2 flagged 74/96 probes.
- G: 0 flags, because the old recovery cleared a sustained loop's flag every cycle.
- N and O: the API was missing.
