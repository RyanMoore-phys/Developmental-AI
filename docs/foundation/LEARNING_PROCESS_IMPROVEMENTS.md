# Learning process improvements

## Implemented scope

These changes strengthen the existing curiosity, world-model diagnostics and
foundation comparison workflow. PyTorch and gradient training remain in place.
No claim of improved live exploration follows from passing software checks.

### 1. Curiosity lifecycle

`LearningProgressCuriosity` receives the configured significance threshold,
autocorrelation ceiling and visit-collapse switch. Visits use explicit stream
identities. Real resets close visits. On an environment restart the splice row
(it spans two worlds) is excluded from curiosity reward and training, and the
visit open before it — real pre-crash experience — is closed as valid, not
discarded. Collection segment boundaries alone do not reset visits. Evaluation
uses its own stream id (`("eval", 0)`), so it can never continue or close the
live stream 0's visit; its final open visit is discarded.

`curiosity_visits.pkl` preserves completed evidence and normalization alongside
the existing neural checkpoint. Restoring drops open visits because a restarted
environment is a new visit. Older checkpoints without this sidecar still load.

**Sidecar robustness (2026-10-05 review).** Both sidecars are written atomically
(temp file, fsync, `os.replace`). A missing sidecar is silent. An unreadable or
rejected one — truncated write, or prototypes whose size no longer matches
`lp_pixel_pool`/`image_channels` — gives that component alone a cold start, one
warning, and the file is renamed to `<name>.corrupt-<timestamp>` (never
deleted). Previously either case raised inside `load_checkpoint`, which the
supervisor would relaunch into forever. Loaders validate before mutating.
Both files are in the brain mirror's explicit file list.

### 2. Observable progress

`ProbeSetProgress` compares the same development probes across optimizer versions
using a fixed deterministic observation-prediction objective. Replacement probes
establish new baselines. Missing evaluations cannot lower an aggregate merely by
removing difficult probes. Objective changes establish fresh baselines.

Events record probe identities, prior versions, improvement, forgetting and
standard error. Improvement is measured against each probe's best recorded loss;
recovering previously achieved performance does not count as new learning.
The error adjustment is a diagnostic heuristic, not a calibrated statistical
guarantee for correlated replay data. `progress_probes.pkl` persists evidence
only: probes, identities, per-probe baselines/best losses, the normaliser and
event serials. The evaluation clock (`_last_eval_step`), version cursor and
every config value (`eval_every`, capacity, `significance_z`) are NOT restored —
the step counter restarts at 0 in a new process, and a restored clock kept
probes from being re-evaluated until the new run overtook the old count.

**Reward behavior changes:** positive `infra.progress_weight` now enables
measurement only. The previous standing per-step progress payment is removed.
Existing learning-progress curiosity remains active. `EvidenceCredit` is a
bounded, expiring, single-use API for explicitly attributed events and is not
connected to PPO. A future integration needs evidence that collection caused
useful learning, plus a controlled reward ablation.

### 3. Scorable shadow forecasts

**Live default: `observable_predict: false`** until its lock/GPU-sync fixes are
measured in a live window (on 2026-10-05 predictions degraded at 5.009 ms/step
against a 5.0 ms budget — margin ~0).

With `foundation.shadow.observable_predict`, the recorder saves a decoded
one-step forecast before the action and scores it against the subsequent
observation, alongside a persistence baseline. Pixel targets are pooled to a
fixed 8 × 8 grid. Decoder output uses the categorical latent mode; it is not the
mean of the stochastic predictive distribution.

Boundary and missing-outcome cases are labeled instead of treated as successes.
Lock skips include stream and source context. Existing sampling, locking and
budget controls apply. The report separates observable error from categorical
entropy; confidence alone is not prediction accuracy. These checks do not yet
establish calibrated uncertainty, useful multi-step prediction or causal control.

### 4. Shared GUI costs

The primary and scout paths share a pure, itemized GUI cost calculation. Existing
potential-difference and dwell-step costs are retained. The user's pending GUI
and report changes are preserved.

### 5. Validation and comparison

Neural mechanisms accept a separate validation batch, patience and minimum
improvement. Early stopping restores both selected weights and optimizer state.
The shadow comparison reserves development episodes for validation; a single
development episode uses a chronological split with a one-transition gap and
must be interpreted as exploratory. Held-out data is used only for final scoring.

Persistence, ridge, MLP, ensemble and action ablation share the same fit rows.
The new frozen protocol records epoch and ensemble budgets; reports include
actual and selected epochs. Ensemble and single-model compute are still unequal,
so this comparison cannot establish a compute-matched architectural advantage.
Historical foundation experiment protocols and their published results are
unchanged; optional validation support does not retroactively validate them.

## Validation coverage and next evidence gate

Regression coverage includes stream identities, invalid recovery transitions,
checkpoint restoration, probe replacement/failure, repeated model versions,
forgetting, duplicate reward claims, deterministic observable targets and
early-stopping restoration. Existing GUI and shadow checks cover path parity,
budget handling and learner noninterference.

Before enabling a new reward or planner, collect sufficient live shadow data,
check per-stream coverage and observable error against persistence, and freeze a
new experiment with episode-separated validation and held-out environments.
Compare structured models with ordinary baselines under explicit interaction
and compute budgets over multiple seeds. Require repeatable held-out gains and
acceptable runtime cost before a separately controlled live integration.
