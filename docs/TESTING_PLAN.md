# Testing plan — proving the new machinery works, and keeps working

*2026-09-19. Covers Phases 1–7. Written to be executed on the training host, in order,
over days rather than in one pass.*

---

## 0. What this plan is for

Everything in `docs/PLURALITY_ROADMAP.md` is **offline-verified and
live-unverified**. 54 unit cases and 5 contract suites pass on synthetic
data. None of it has seen a Minecraft frame.

The gap matters because this project's failure mode is not code that crashes
— it is **code that runs and quietly does nothing**, or code that pays the
agent for standing still. Both look like success from a green test suite.
So this plan is organised around *what could be true and undetected*, not
around coverage.

**The standing rule:** a stage does not begin until the stage before it has
produced its numbers. A green test is not a result.

---

## 1. Test tiers

| Tier | What it proves | Cost | Where |
|---|---|---|---|
| **unit** (54 cases, 3 files) | Parts: shapes, bounds, neutrals, degenerate inputs | ~3 s | anywhere |
| **contract** (5 suites) | Design: the arguments each wave was built on | ~7 s | anywhere |
| **legacy** (6 suites) | Regression surface predating this work | ~60 s | needs gymnasium ⇒ host |
| **integration** (§4) | The stack boots and steps against real MineRL | minutes | host |
| **soak** (§5) | Behaviour over hours: does anything drift, latch or farm? | hours–days | host |

```bash
PYTHONPATH=. python tests/run_all.py unit
PYTHONPATH=. python tests/run_all.py contract
PYTHONPATH=. python tests/run_all.py legacy
PYTHONPATH=. python tests/run_all.py all      # exit code = failing suites
```

---

## 2. Stage 0 — environment feasibility (BLOCKING)

**Nothing else runs until these four answers exist.** Each is a fact about
MineRL that no amount of reading settles.

| # | Question | How | If NO |
|---|---|---|---|
| 0.1 | Does MineRL build and step at all on this training host? | `scripts/host_stage0.py --boot` | **Stop.** Report. |
| 0.2 | Does the action space contain `hotbar.1-9`, `left`, `right`, `sprint`, `sneak`? | `_validate_macros` crashes at boot if not; `--dump-space` prints it | Remove those macros (append-only, so indices 0–20 survive) and re-run. Report the reduced set. |
| 0.3 | Does POV include the HUD overlay? | `--dump-frame` writes a PNG; inspect the bottom 12% | A2 stays disabled permanently; A9 keeps the keys but the agent cannot see selection. Record it. |
| 0.4 | Is there any audio handler? | `--probe-audio` lists observation handlers | Expected NO. M1 stays disabled; the Python half waits. |

Stage 0 is the point at which *"if MineRL does not have the capabilities,
let me know and stop"* is decided. It is deliberately first and cheap.

---

## 3. Stage 1 — offline suites on the training host

Same code, real hardware, plus the six legacy suites that need the full
stack and therefore have never run on a dev Mac.

```bash
PYTHONPATH=. python tests/run_all.py all
```

**Pass condition:** every suite PASS. **Any legacy failure is a regression
caused by this work** — it is the whole reason that tier exists.

> **It earned its keep on the first run.** Two legacy suites failed, both on
> hotbar keys, and one was an explicit `assert not any("hotbar" in k ...)`
> with the reason written beside it: *minimal exploration burden*. The
> response was to **remove the feature, not weaken the test** — and Stage
> 0.3's HUD measurement showed the original decision had been right all
> along. A contract that encodes a decision is how a project argues with its
> future self.

---

## 4. Stage 2 — integration, 500 steps

One MineRL client, `num_envs: 1`, random policy. Not learning; proving the
plumbing carries real data.

| # | Assertion | Why it is not covered offline |
|---|---|---|
| 2.1 | `info["sensors"]` width == the bus's declared width, every step | Offline the bus is fed a synthetic context |
| 2.2 | `info["sensor_layout"]` is constant across the run | A hash that changes mid-run means a sensor is reshaping |
| 2.3 | **Every sensor varies.** `std > 0` over 500 steps | **The most important check in this stage.** A sensor that is wired but always neutral is indistinguishable from a working one in any unit test. |
| 2.4 | No NaN anywhere in the transport | Real frames include failures a synthetic one never produces |
| 2.5 | `true_position` appears in `info["oracle"]` and **never** in `info["sensors"]` | Contract asserts the code path; this asserts the live dict |
| 2.6 | Every macro index produces a distinct action dict at the socket | The silent-drop failure class |
| 2.7 | Step rate ≥ 90% of the pre-change baseline | A1 adds a crop and A5 adds macros; neither may cost throughput |
| 2.8 | Occupancy map is non-zero after 100 steps of walking | Proves the probe→scan→map chain carries a real signal |

Script: `scripts/host_stage2_integration.py`.

---

## 4b. Stage 3 — the FULL LOOP, 300 steps

Stage 2 drives the bare adapter. Five things only exist once the
developmental loop is running, and each is silent when wrong:

| # | Assertion | Why it is silent |
|---|---|---|
| 3.1 | loop's transport width == world model's `proprio_dim` | A mismatch raises, but only on the first batch — hours in |
| 3.2 | policy's declared `proprio_dim` == what `_augment_proprio` writes | **Silent.** Shifts every map into the wrong slot; the agent trains on garbage that looks like data |
| 3.4 | `flow_loss`, `horizon_loss`, `reconstruction_error` recorded and finite | A metric that is never recorded looks identical to one that is zero |
| 3.5–3.6 | spatial maps advance, and the confidence clamp holds | An unclamped map compounds to "walled in everywhere" over hours |
| 3.7 | oracle measured drift, and reached nothing else | — |
| 3.8 | replay buffer carries the sensor column at the right width | A column written at the wrong width restores as neutral forever |

```bash
PYTHONPATH=. python scripts/host_stage3_loop.py --steps 300
```

Runs with **one client and no external server** — MineRL generates its own
world, and `remote_server` only matters for the social experiment.

---

## 5. Stage 4 — the flow go/no-go (gates Phase 5 and `flow_mode: depth`)

The one measurement everything downstream is waiting on.

**Run:** 2 clients, 6–12 hours, current config (`flow_mode: raw`,
`slots.enabled: false`).

| Metric | Pass | Meaning |
|---|---|---|
| `flow_loss` | falls, monotone over hours | The warp objective learns something from Minecraft frames |
| `flow_mask` | settles in 0.6–0.95 | Auto-masking is dropping occlusions, **not starving the loss** |
| `horizon_loss` | falls | The 15-step dream horizon is actually being supervised |
| `oracle_dr_drift` | reported, bounded | Dead reckoning is usable; the number itself is the result |
| `reward pool: N windows` | small (tens) | The 33× dilution is gone |

**If `flow_loss` does not fall:** Phase 5 stays off, `flow_mode` stays `raw`,
and the question becomes whether the flow head is wrong or the data is too
thin. Do **not** enable slots to "see if it helps".

---

## 6. Stage 5 — the falsifiers

These are not pass/fail. They are the plots that catch the failure this
project keeps having: **being paid for doing nothing.**

1. **Income vs `moved`.** Non-zero flow-derived income at `moved ≈ 0` means
   another wage for standing still.
2. **Income vs `pitch`.** Correlation with looking up means the sky was
   rebuilt.
3. **`flow_mask` over time.** A retained fraction trending to 0 means the
   photometric loss is starving itself.
4. **Occupancy max over time.** Trending to 1.0 everywhere means the
   confidence clamp is being defeated and the agent believes it is walled in.
5. **Walkability entropy.** Collapsing to all-zero means it has concluded
   nothing is passable — the belief that stops an agent trying.
6. **Per-sensor variance, hourly.** A sensor that goes constant has died.

Script: `scripts/host_falsifiers.py`, run against the metrics dump Stage 3
and the soak both write.

---

## 7. Stage 6 — soak, and the scoreboard

Only after Stages 0–5. Multi-day, 2 clients.

**The scoreboard is the point, and it has not moved in this project's
history.** Watch it, not the reward number:

- blocks broken → **logs** (history: 13,305 → 35)
- longest attack run, and what was under the crosshair when it ended
  *(the oracle can answer this; nothing else ever could)*
- distinct predicates that ever fire
- skills minted / invoked

**Ablations, one at a time, 6 h each.** Run *after* a baseline exists:

| Toggle | Question |
|---|---|
| `sensors.enabled` −`fovea_native` | Does seeing the crack overlay change break→log? |
| `flow_strides: [1,2,4]` | Does the measured retreat reproduce live? |
| `spatial.enabled: false` | Does spatial memory change approach behaviour? |
| `latent_horizons: []` | Does the dream horizon matter to the policy? |
| `goal_replay_threshold: 1e-3` | How much did the 33× dilution cost? |

---

## 8. Continuous regime

Once a soak run is stable:

- **Every commit:** `tests/run_all.py all` on the training host. Exit code gates.
- **Hourly:** per-sensor variance + the six falsifier plots.
- **Daily:** scoreboard delta; `rsync` the brain training host→Mac (CLAUDE.md §6 — a
  host has already died with ~11 days of unbacked state).
- **Weekly:** re-run one ablation to confirm the conclusion still holds.

**When something breaks, encode it as a test.** That is why every contract
docstring in this repo names a live incident and its measured numbers, and
it is why the seven bugs in `PLURALITY_ROADMAP.md` §11 each have one.

---

## 9. Honest limits of this plan

- **Nothing here proves the agent learns better.** It proves the machinery
  carries real signal and does not pay for stillness. Whether structure per
  dimension helps is Stage 5's question and will take days.
- **Audio is untestable** until the Java work exists.
- **A2/HUD is undecidable** from here; Stage 0.3 settles it in one frame.
- **Slots have no contract tests at all**, deliberately — every claim worth
  asserting needs a working flow field first.
