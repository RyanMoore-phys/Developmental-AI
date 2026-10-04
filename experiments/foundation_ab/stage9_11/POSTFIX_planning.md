# Stage 11 (planning) after the fixes for B3, F7 and F8

This note covers the three planning findings in `RESULTS.md`: B3 (the deadline is not hard), F7 (the reliability gate has no memory of where the model failed) and F8 (duplicate detection depends on the probe set). `RESULTS.md` itself is unchanged. The code changes are confined to `developmental_ai/foundation/planning/` and its tests.

**Reproduce**

```
PYTHONPATH=. python -m experiments.foundation_ab.stage9_11.postfix_planning   # ~90 s
PYTHONPATH=. python tests/unit/test_foundation_planning_unit.py               # ~1 s
PYTHONPATH=. python tests/_foundation_planning_smoke.py                       # ~18 s
```

**Where things are**
- Raw reports: `reports/postfix/`.
- Full tables: `generated_postfix_planning.md`.
- E11c latency distribution: `reports/postfix/e11c_latency.json`.

**Preregistrations.** The re-run used the verifier's original frozen preregs, loaded from `reports/full/E11*.json` with the same content, hash and freeze time. `run_ab` re-verified every hash: E11a `2508ca667846`, E11b `e47d881a7340`, E11c `b02fc1205e18`, E11d `7bdc16f4e89a`, E11e `68ea5467a4c2`. None was refused, so no `*_postfix` prereg was created.

**Arms.** Wherever a fix applies through the default API, the verifier's arm code in `e11_planning.py` runs unchanged. The new runner adds three things, all labelled in the tables:
- An exploratory arm, `gated_global`, which is the pre-fix global gate.
- An E11c latency distribution, measured by a line-for-line copy of `deadline_arm` that keeps every latency.
- A second run of E11d/e with evidence-built probe sets (see F8).

## Summary

| finding | metric | before | after |
|---|---|---|---|
| B3 deadline | decisions over 1.25× the deadline, heavy-tailed model | 8.5% | **0.0%** |
| B3 deadline | worst decision, heavy-tailed model | 7.9× | **1.06×** |
| B3 deadline | E11c verdict ("the deadline is not hard") | accept | **reject** |
| F7 region | planner entries into the wrong region, per seed | 4.8 | **1.4** |
| F7 region | planner steps in the wrong region, per seed | 6.4 | **1.6** |
| F7 region | real mean distance, gated (lower is better) | 0.863 | **0.795** |
| F7 region | E11a verdict (gate vs ungated) | inconclusive | **accept** |
| F8 duplicates | off-probe pair called a duplicate | 92% | **0%** |
| F8 duplicates | byte copy under another name called a duplicate | 100% | 100% |
| F8 duplicates | same name, different policy, called a duplicate | 0% | 0% |

For F7, the outcome gain is smaller on seeds that were not used for tuning, and the defaults were not tuned blind. See the tuning disclosure under F7. For F8, "after" is the duplicate verdict on an evidence-built probe set.

## B3: the deadline is now hard, up to thread wake-up latency

**Change** (`planner.py`):
- **Model calls run on one worker thread.** With the real clock (the default), the planner waits for a call only until the deadline. Python cannot kill a running call, so an abandoned call keeps the worker busy until it returns. While it does, the model is unavailable: the next plan waits only as long as its own budget allows, then times out. At most one call is ever in flight, and no thread leaks per timeout.
- **Deadline checks run more often.** The planner checks the deadline before each batch of candidates. Inside a batch, a cooperative cancel is polled between horizon steps.
- **On timeout it returns the best plan so far.** If no CEM iteration completed, it returns `timed_out` and the controller falls back.
- **The cost estimate is a running median of the last 16 chunk costs.** It replaces the EMA. Abandoned calls enter at their censored waited time. One spike no longer shuts the planner down for several calls.
- **Long waits are sliced.** On macOS a single 20 ms timed wait overshoots to 22–30 ms because of timer coalescing. That alone breaks a 1.25× deadline. The wait now repeatedly sleeps for half the remaining time, down to 0.5 ms slices.
- **An injected clock keeps the old behaviour.** Unit tests that inject a clock get the old synchronous chunks, which stay deterministic.

**Regression tests.** Both failed on the old code and pass now.
- `deadline_holds_through_one_slow_model_call`: a single 60 ms call used to give a 64 ms plan.
- `deadline_returns_best_so_far`: a stall in the second iteration used to give a 92 ms plan.

Smoke contract M covers the statistical claim. The heavy-tailed fixture must stay under 2% of decisions over 1.25×. As a witness, the same fixture run synchronously must exceed it on at least 3% (it measured 14%).

**E11c latency distribution.** `decide()` latency divided by the 10 ms deadline, over 5 seeds × 150 decisions:

| model | over 1.25×, before | over 1.25×, after | over 2×, before | over 2×, after | p99, before | p99, after | max, before | max, after | planner share, before | planner share, after |
|---|---|---|---|---|---|---|---|---|---|---|
| heavy tail (2% × 30 ms) | 8.5% | **0.0%** | 8.5% | **0.0%** | 4.14 | 1.04 | 7.87 | 1.06 | 52.9% | 67.5% |
| constant 0.6 ms | 0.27% | 0.4% | 0.27% | 0.3% | 0.97 | 1.02 | 3.38 | 3.51 | 98.3% | 99.5% |
| heavy tail, chunk 8 | 15.9% | **0.3%** | 15.9% | **0.0%** | 4.74 | 1.06 | 7.30 | 1.67 | 21.1% | 31.7% |

- The "before" column is my reproduction on the pre-fix code, using the same copy of the arm.
- My pre-fix reproduction matched the verifier's numbers: 8.5% heavy tail (verifier 8.5%), 15.9% with chunk 8 (verifier 15.5%).

**Prereg verdict.** E11c went from **accept** (claim falsified) to **reject**.
- Paired difference in decisions over 2×, heavy tail minus constant: −0.0027, CI [−0.0069, 0.0016].
- The heavy-tailed arm now has 0 such decisions per seed. Before the fix it had 0.085.

**What is not bounded**
1. **Pauses of the main thread itself.** These include the garbage collector and OS scheduling. The constant-latency arm had 2 of its 750 decisions above 2×, with a maximum of 3.5×. The model's latency cannot explain those.
2. **A model that spins in pure Python.** It holds the GIL, so the wake-up can be late by up to the interpreter switch interval of 5 ms.
3. **A model call that never returns.** It keeps the planner unavailable permanently and the fallback controls. This is the honest outcome for a hung model, and control comes back the moment the call returns.

`PlanningController.observe` waits for any in-flight call first, so the model is never used from two threads at once.

**Cost.** The default thread isolation adds CPU time: the planning smoke's system time went up, though wall time did not. Running numpy on the worker thread costs a thread handoff per chunk.

## F7: region-aware reliability

**Change** (`planner.py`, `RegionMemory` and `ReliabilityMonitor`):
- **Every real validation is recorded.** Each record keeps the input state, pass or fail, the model version and the time, in bounded FIFO memory: 512 failures and 4096 passes.
- **Each failure keeps its local evidence.** For every failure record, the memory keeps kernel-weighted sums of failures and passes around it, updated incrementally (no rescan on the hot path). A failure counts as "bad" while failures make up more than 25% of the evidence around it.
- **A state is unreliable when bad failures surround it.** The test is a bad-failure weight of at least 0.5, using a Gaussian kernel with bandwidth 0.25 in standardised RMS units.
- **The controller acts on the region verdict in two ways.**
  - It falls back while the current state is unreliable.
  - The planner cuts each imagined rollout at the first step whose input state is unreliable, and fills the rest with the reward floor, exactly like the epistemic cut.
- **Explained failures stay local.** A failure whose state was already in a marked region does not close the global window. It still updates the region memory and the CUSUM.
- **Regions are keyed on entity positions by default.** This is `PlanningController(region_key="position")`. The alternatives are `"state"` or a callable.

**Why positions.** On the mud fixture, failures are recorded at the low speeds the mud forces. The planner enters the mud at high speed.
- With full-state keys at bandwidth 0.25, failures did not generalise across speed: 4.2 re-entries per seed.
- At bandwidth 0.5 the region instead blurred 1–2 m past the mud: 683 of 708 region refusals happened outside it.
- Position keys over-block a failure that depends on velocity. That is the safe side, and the same three ways back (below) apply.

**Ways back (CLAUDE.md §4.1).** These are in the docstring and tested in `region_memory_marks_failures_and_reopens`:
1. **The current model passes again in that region.** Fallback steps are validated wherever they go.
2. **A refit (a model version bump).** Older-version records are weighted by `region_stale_weight`, which defaults to 0: a refit model is judged on its own record, like the global window. Smoke D motivated the default. At weight 0.25, the correctly refitted model was kept out of every place the broken model had failed, and the real late distance after the refit went from 0.155 to 0.815. The cost of weight 0 is that a refit that did not fix a region gets re-entered once there. With a weight between 0 and 1, the region stays closed until the new model's own passes there outweigh the old failures. The unit test covers both weights.
3. **Decay.** Records halve every 1000 validations, so a region nobody revisits returns to unknown and defers to the global gate.

Smoke contract N exercises the second way on the mud fixture. After a refit to the true dynamics, the planner controlled 127 steps inside the mud and the region was no longer marked.

**Regression tests.** Both failed on the old code: the monitor had no region API.
- `region_memory_marks_failures_and_reopens`
- `planner_cuts_rollouts_through_failed_regions`

Smoke contract N, over 30 episodes: the global gate entered the mud 8 times (the witness requires at least 3), and the region gate entered it 2 times.

**E11a/b results** (5 seeds × 20 episodes, the verifier's `region_arm` unchanged):

| arm | real mean distance | planner entries into mud per seed | planner steps in mud per seed | share of steps in mud | planner share |
|---|---|---|---|---|---|
| gated before the fix (global) | 0.863 | 4.8 | 6.4 | 0.409 | 0.460 |
| **gated after the fix (region)** | **0.795** | **1.4** | **1.6** | **0.248** | **0.675** |
| gated_global (exploratory: `regions=False`) | 0.863 | 4.8 | 6.4 | 0.409 | 0.460 |
| ungated | 0.879 | 7.6 | 203.2 | 0.437 | 0.998 |
| fallback only | 0.904 | 0 | 0 | 0.357 | 0 |
| gated with the true model | 0.672 | 3.6 | 110.2 | 0.236 | 0.994 |

- `gated_global` reproduces the pre-fix gated arm exactly. That is the sanity check.
- Region gated beats global gated on all 5 seeds: 0.642 vs 0.707, 0.761 vs 0.837, 0.896 vs 0.957, 0.748 vs 0.874, and 0.930 vs 0.940.

**Prereg verdicts**

| prereg | before | after |
|---|---|---|
| E11a (gated beats ungated by ≥ 0.02) | **inconclusive**: Δ 0.016, CI [−0.032, 0.060] | **accept**: Δ 0.084, CI [0.002, 0.165] |
| E11b (fallback beats gated) | **reject**: Δ −0.041, CI [−0.082, 0.0004] | **reject**: Δ −0.109, CI [−0.150, −0.069] |

After the fix, gated is clearly better than the fallback alone.

**Tuning disclosure.** This matters for how much weight E11a deserves.
- The region design was explored on the mud fixture **with seeds 0–4, which are the prereg seeds**. That covered bandwidth, state versus position keys, the explained-failure rule and the stale weight.
- I then checked the choice on seeds 100–104 before freezing the defaults.
- So the E11a "accept" is **not an untouched confirmation**. The seeds-100–104 numbers are the cleaner estimate, and the gain on real outcome there is small:

| configuration on seeds 100–104 (final rule set) | real mean distance | entries per seed |
|---|---|---|
| global gate | 0.842 | 5.0 |
| region, bandwidth 0.15 | 0.786 | 3.0 |
| **region, bandwidth 0.25 (default)** | **0.820** | **2.0** |
| region, bandwidth 0.35 | 0.883 | 1.4 |
| region, bandwidth 0.5 | 0.903 | 1.0 |

- Re-entries fall reliably, by roughly 2.5–3.5×.
- Real outcome is a trade-off. Wider regions mean fewer re-entries but more steps handed to the noisy fallback.
- 0.25 is a compromise, not an optimum. The 0.786 at bandwidth 0.15 was better on distance but re-entered more.

The verifier's caveat still stands: the gate cannot recover what only a better model provides. The gap to the true model remains (0.795 vs 0.672).

## F8: a duplicate verdict now needs evidence coverage

**Change** (`skills.py`):
- **`build_probe_set(evidence, shared)`** builds the probe set from each skill's own evidenced states, up to 64 per skill by farthest-point sampling, plus shared probes. It records each skill's coverage: the fraction of its evidence within radius 0.5 (standardised RMS) of a probe.
- **`LearnedSkill.evidence_states()`** returns where a skill was invoked and where it went, from real attempts only.
- **`find_duplicates` returns `DuplicateFinding` records.** Each carries a verdict and a statement such as "indistinguishable on 152 probes covering 100% of A's 24 evidenced states and …".
  - The verdict is **"duplicate"** only when the probe set covers at least 90% of at least 8 evidenced states for both skills.
  - Otherwise the verdict is **"indistinguishable"**. Bare probe arrays always get this verdict, because nothing says they cover either skill.
- **Findings are deliberately not tuples.** Code that unpacked `(a, b, d)` and treated every pair as a duplicate now has to read the verdict.
- **Identity stays keyed on skill ids.** Names are never consulted.

**Regression test.** `duplicates_require_evidence_coverage` failed on the old code because the API was missing. Smoke contract F now takes its verdict on an evidence-built probe set. New smoke contract O covers the off-probe case:
- Bare probes call 9 of 10 off-probe pairs "indistinguishable". That is the witness.
- With evidence-built probe sets, the off-probe pair is a duplicate in 0 of 10 trials, a byte copy in 10 of 10, and a same-name different policy in 0 of 10.

**E11d/e results**

| arm | (i) verifier's arm, unchanged: flag rate | (ii) evidence-built probe set: duplicate-verdict rate |
|---|---|---|
| offprobe (prereg candidate) | 0.92 | **0.00** |
| name_only (byte copy) | 1.00 | 1.00 |
| samename_diff | 0.00 | 0.00 |
| tempered (logits ×3) | 0.00 | 0.00 |
| drift 1e-3 | 1.00 | 1.00 |
| drift 1e-2 | 1.00 | 1.00 |
| drift 0.1 | 0.00 | 0.00 |
| offprobe_inside (exploratory) | – | 0.69 |

- **Run (i)** uses the verifier's arm unchanged. It counts `len(find_duplicates(...))` on the bare 32-probe set. The count is unchanged by construction, but every finding is now labelled "indistinguishable … covering an UNKNOWN fraction". **Verdict: E11d accept.**
- **Run (ii)** uses the same pairs and the same 32 shared probes, under the same frozen prereg. Each skill's evidence is 64 states drawn from the deployment distribution N(0, 4²), where the verifier measured 29% argmax disagreement. Minimum coverage was 1.0. **Verdict: E11d reject**: paired difference 0.00, CI [0, 0].
- **`offprobe_inside`** draws evidence from N(0, 2²) only: two skills only ever used where they agree. There a duplicate verdict is the honest answer, because coverage is relative to the evidence. The 31% not flagged are, I believe, trials whose evidence or shared probes happened to reach some |s_i| > 7. I estimate that happens in about 26% of trials at this sample size; I did not count it directly.

**Not addressed.** E11e is unchanged and still accepted: a tempered copy is called distinct. Signatures still compare action distributions, not executed deterministic behaviour (the second half of F8). That was outside this brief.

## API and state changes

- **Region gate is on by default.** `ReliabilityMonitor(..., regions=True, region_*)` is the default; `validate(..., where=)` takes the input state; there are new `region_unreliable` and `reliable_at` methods. Its state schema is now 2.
- **`PlanningController` takes `region_key` (default `"position"`)**, and `observe` passes `where`.
- **`BoundedPlanner` changes**
  - New `isolate_calls` parameter, defaulting to on with the real clock.
  - `plan(..., region_check=)` takes the region test.
  - New `busy()` and `wait_idle()` methods.
  - Planner and controller state schema is now 2 (the cost history is saved).
- **New exports:** `find_duplicates` now returns `DuplicateFinding` records; `ProbeSet`, `build_probe_set` and `RegionMemory` are new.
- **Nothing is wired into the live loop.** All of this is offline/shadow only, per plan §8.
