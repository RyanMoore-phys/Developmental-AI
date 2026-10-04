# Foundation milestones (improvement plan §10)

Status as of 2026-10-03. Everything below is **Offline or Shadow** (plan §8).
Nothing here controls SkyBot's actions or rewards. No result below comes from
Minecraft. Every "gate" is a synthetic-fixture result unless it says otherwise.
Read CLAUDE.md §9 before treating any of it as progress: the scoreboard (one
log, ever) has not moved.

Tests: `PYTHONPATH=. python tests/run_all.py all`. Verified green on two
stacks: py3.10 / torch 2.6.0 / numpy 2.2.6 (the CI and host mirror) and
py3.12 / torch 2.10 / numpy 1.26. CI now also runs the unit and contract tiers.

| Stage | Package | Suites (unit / contract) | Evidence on fixtures | Open: needs the training host |
|---|---|---|---|---|
| 1 Baseline | `runtime/{manifest,assumptions,splits,baseline}`, `tools/run_manifest.py`, `tools/baseline_report.py`, `ASSUMPTIONS.md`, `BASELINE_AUDIT.md` | runtime / baseline | Manifest is deterministic. The assumption register is checked against the live sensor registry. The split cannot leak. | Repeated baseline runs, and WM error by horizon (not logged today) |
| 2 Contracts | `contracts/`, `adapters/` | contracts / contracts | SkyBot sensor-bus data travels through the records, with RED kept evaluator-only. The nonspatial adapter passes the same conformance check. | None |
| 3 Orchestration | `runtime/collection_path.py` (wired at boot), `runtime/snapshots.py` | runtime / collection_path | `num_envs: 1` on a SkyBot config now **raises**. The escape is `parallel_envs.allow_legacy_single_env: true`. The live config's path is unchanged (lifelong_segment). | Recorded-input parity. The full stage extraction from the loop is **not done**. |
| 4 Geometry | `geometry/` | geometry / geometry | Round trips, covariance, rotor == quaternion, gradcheck. Up-to-scale quantities never silently become metric. | None |
| 5 Evidence | `experience/`, `EXPERIENCE_RECONCILIATION.md` | experience / experience | Crash recovery works. Predictions are stored before outcomes. A stored interaction can be rebuilt with its prediction and snapshot. | Live growth and recovery time on the host disk |
| Shadow | `runtime/shadow.py`, both live bodies, **`foundation.shadow.enabled: true`** (user decision, 2026-10-03) | shadow | The learner is byte-identical with shadow off and on. About 0.7 ms/step at every_n_steps 16, capped at 1 GiB, and it self-disables over 5 ms/step. | Confirm `Phase timing: shadow` and disk growth on the first live run |
| 6 Mechanisms | `mechanisms/` | mechanisms / mechanisms | **Gate mixed.** After the bug fixes: E1a accept (NLL, mostly from MLP overconfidence), E1b inconclusive, E1c inconclusive. The IN costs about 15× the MLP's FLOPs. See `experiments/foundation_ab/RESULTS_stage6_8.md`. | Comparison on real recorded SkyBot data |
| 7 Inference | `inference/` | inference / inference | **Partial.** Epistemic variance predicts error (accept), but no better than a single model's own variance (reject). It does not rise under rule change, so the change detector carries that job. | Calibration on real data |
| 8 Perception | `perception/` | perception / perception | **Accept**, reproduced by an independent verifier: the tracker beats per-frame features on downstream prediction. The tracker is fixed (Kalman/Hungarian), not learned. | Pixel-based slots on Minecraft frames |
| 9 Discovery | `discovery/` | discovery / discovery | **Accept**: beats refit-only by 0.79 nats/row on the verifier's fixtures (the author claimed 1.82). 0/65 false promotions on noise. The search budget now holds (1.05×). | Real mechanisms |
| 10 Selection | `experiments/{candidates,information,question,selection,forgetting}` | selection / selection | **Accept**: 5.48 vs 10.50 interactions against random, with a noisy TV present. Targeted info gain cut the useless-distractor share from 44% to 3%. Misspecification is flagged, and confident-wrong reports fell from 100% to 1.8%. | Closed-loop comparison against ICM / LP / imagination curiosity |
| 11 Planning/skills | `planning/` | planning / planning | The deadline holds (0% over 1.25×). The region gate gives E11a accept, but the region design was explored on the prereg seeds, so it is not clean. Duplicate-skill detection goes by behaviour, never by name. | Real behaviour. "Planning success in imagination alone does not qualify." |
| 12 Transfer | `transfer/`, `experiments/ab.py`, `tools/ab_compare.py`, `tools/profile_loop.py`, `NATIVE_RUNTIME.md` | ab / ab, transfer | Two adapters pass conformance. Negative transfer is detected and the learner abstains. The A/B harness refuses a tampered preregistration. | **No Rust port**, by design: no host profile justifies one yet |

## Live-reward fixes (2026-10-03, user-authorised)
- **Stream-0 GUI-dwell cost now applies without the vision scaffold.** It
  used to sit inside `vision_scaffold is not None`, and `llm.vision.enabled`
  is false, so stream 0 paid nothing in a menu. It now runs in both live
  bodies on `info["gui_open"]`, as a plain difference into prim_extrinsic:
  entering costs -0.05, each pinned step pays exactly 0, leaving refunds
  +0.05. `tests/_gui_farm_smoke.py` D.
- **The episodic body re-adopts cost potentials at episode start.**
  `_run_episode_parallel` never reset `_gui_dwell_phi` / `_pitch_level_phi`,
  so a reset taken inside a menu paid a phantom refund. `_gui_farm_smoke.py` E.
- **`proprio.depth`** (from the engine's true y) stays as is. It is
  registered as supplied structure (ASSUMPTIONS.md AM5); this was a user
  decision, to protect the trained world model's inputs.
- **Reward tiers keyed by block names** (AM7) are the task's definition of
  extrinsic reward, not an observation. They stay as is.

## Known limits
Fixed 2026-10-03 (each has a regression test that failed on the old code):
- ~~`discovery/search.py`: an abandoned fit thread keeps running.~~ Each
  evaluation gets a `CancelToken` (a float deadline that can be cancelled).
  The library forms, the causal-annotation refits (which used to run with no
  deadline) and the scoring steps all poll it. An abandoned cooperative fit
  now exits within one iteration: discovery smoke P1 measured 5-28 ms,
  against 0.28-0.53 s before. At most `MAX_ABANDONED_FITS` (4)
  non-cooperative fit threads can be alive at once. Past that cap, search
  returns `budget_exhausted`. A slot frees itself when its fitter returns
  (P2).
- ~~`planning/planner.py`: a model call that never returns leaves the
  fallback in control for good.~~ The watchdog marks a call older than
  `hang_after_s` (default max(1 s, 20 × deadline)) as hung. It poisons that
  worker, spawns a fresh one and forces a re-probe. At most `max_respawns`
  (3) poisoned workers can be alive, and respawns back off exponentially.
  `observe` waits at most `hang_after_s`; before the fix it waited without
  limit. Planning smoke Q: a model that hangs once gives control back to
  the planner after 1.0 s (never, before). A model that always hangs gets
  2 respawns, at most +3 threads, and the fallback keeps every step.
- ~~E11e: a tempered copy is reported as distinct.~~ `find_duplicates`
  now gives one of four verdicts: `duplicate`, `near_duplicate`, `distinct`
  or `indistinguishable_on_probes`. A pair is `near_duplicate` when JS is
  at or above the threshold but the modal action agrees on every probe
  where 64 executions would tell the two skills' modes apart (95%
  multinomial bound). Smoke R, 20 trials: the logits×3 copy is
  near_duplicate 20/20, a byte copy duplicate 20/20, and a same-name
  different policy distinct 20/20.

Still open:
- An abandoned fit or model call that never checks its token cannot be
  killed. It is only capped, and its slot stays held until it returns.
- A poisoned model call can still be running while the fresh worker or
  `observe` uses the same model, so model calls have to be thread-safe.
- `near_duplicate` depends on `noise_samples` (64). Weight drift 0.1 (8.8%
  argmax disagreement, all at near-ties) grades near_duplicate in 19/20
  trials. Continuous signatures have no near_duplicate verdict.
- GC and OS pauses are not bounded.

## Rollback
Set `foundation.shadow.enabled: false` to stop the recorder. The other live
changes are the GUI-dwell fixes above (revert in `developmental_loop.py`) and
the collection-path guard; to bypass it, set
`parallel_envs.allow_legacy_single_env: true`. Nothing in `foundation/`
writes to brain state, replay, or `skill_bank_mc_curiosity/`.
