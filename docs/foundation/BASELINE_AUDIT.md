# Baseline audit (Stage 1, item 1) — 2026-10-03

This audit was done by reading code only. Line numbers refer to
`developmental_ai/core/developmental_loop.py` (the "loop") as it stands after
this stage's edit, unless a different file is named. No training-host run
evidence was available: the local `runlogs/` holds only the CI ledger.
Anything that needs the training host is listed at the end.

## 1. Active configuration and collection path

- Launch: `scripts/launch_skybot.sh:111` runs `run_minecraft.py --config
  configs/minecraft_skybot.yaml --seed 0` with no `--forever`.
- Config resolution: `run_minecraft.py` does `yaml.safe_load`, then sets
  `cfg["seed"]`, then applies the `--forever` lifelong override. **No
  defaults are merged anywhere.** Keys that are absent take in-code `.get()`
  defaults. `foundation/runtime/manifest.resolve_config` reproduces this
  exactly.
- Values: `lifelong.enabled: true`, `parallel_envs: {enabled: true,
  num_envs: 2}`, `environment.remote_server_scope: all`.
- Dispatch in `train()` (3093-3099):
  - `lifelong` → `_collect_segment` (5492). **This is the live body.**
  - `parallel` → `_run_episode_parallel` (4268).
  - otherwise → `_run_episode` (3677).
- Inside `_collect_segment`, `use_dream_actor` is hard-coded `False` (5510).
- **Changed in this stage:** `select_collection_path` runs at 307, before
  the first `make_env`.
  - It refuses the single-env body for Minecraft configs, and for any config
    that requests parallel but has `num_envs <= 1`.
  - The escape is `parallel_envs.allow_legacy_single_env: true`.
  - Lifelong without parallel is refused with no escape, because no
    single-env lifelong body exists. The older check at 721 is kept.
  - Non-Minecraft configs that never asked for parallel keep the historical
    single-env default. About 20 crafter, minigrid and cartpole smokes use
    that path.
  - `tests/_collection_path_smoke.py` checks every config in `configs/`.
  - A cross-check at about 701 raises if `_use_parallel_envs` disagrees with
    the decision.

## 2. Observation sources

- Pixels: a 128 px POV. Body: `info["proprio"]` (13 fields,
  `minerl_env.py:980`).
- Sensor bus: `info["sensors"]` is `SensorBus.read_policy`
  (`minerl_env.py` ~2405). Enabled sensors: proprio, screen_fx,
  fovea_native, fovea_delta, dead_reckon, motion, light, sky, true_position.
- RED `true_position` goes only to `info["oracle"]` (`minerl_env.py` ~2419).
  Its only consumer is `_oracle_observe` (10470, called at 5886).
- Replay: a single `replay_buffer.add` per env per step (6821). It stores:
  - pixels, action and reward label, `done` and `restart`, `stream=e`;
  - `proprio` = the previous step's `info["sensors"]`.
  `info["oracle"]` is **not** stored.
- See `ASSUMPTIONS.md` for the two supplied-structure flags: AM5
  (`proprio.depth` is derived from the engine's true y) and AM7 (reward
  tiers are keyed by block name).

## 3. Reward channels (live path `_collect_segment`)

- **Env extrinsic:** `rewards.append(float(rew))` (5701). A hung client
  gives 0.0.
- **`intrinsic` (all streams):** ICM via `curiosity.compute_intrinsic_reward`
  (5895), scaled by `icm_base_scale`. Stream 0 gets a boring-view discount
  with a floor. This term still pays a small amount while idle, because
  prediction error on a drifting sky is not zero.
- **Further `intrinsic[0]` terms (waking):**
  - imagination curiosity;
  - symbol novelty;
  - symbol-centring potential (γΦ′−Φ);
  - view novelty;
  - reach potential (γΦ′−Φ);
  - persistence (Φ=0 when not attacking);
  - pitch-level (plain difference, so it pays 0 while pinned);
  - gaze bucket;
  - coverage (pays 0 when still);
  - habituation multiplier;
  - empowerment via `_infra_step`.

  These sit at roughly 6050-6640. Their line numbers come from a delegated
  read and were not re-verified one by one.
- **GUI zeroing:** `intrinsic[0] *= 0` while `gui_open` (6882), immediately
  before `mixed = reward_mixer.mix(intrinsic[0], prim_extrinsic)` (6883).
  This erases every intrinsic term during occlusion, as CLAUDE.md §4.4
  describes.
- **`prim_extrinsic`** starts as `rewards[0]` (6509). The magnet shaping
  (6571), the infra approach reward (6588) and the **stream-0 GUI-dwell
  cost** (6621) are all inside `if not use_dream_actor and
  self.vision_scaffold is not None:` (6526).
- **FINDING: the stream-0 GUI-dwell cost is dead on the live config.**
  - The scaffold is only built when `llm.vision.enabled` is true (2145), and
    `configs/minecraft_skybot.yaml` sets `llm.vision.enabled: false`. So
    `self.vision_scaffold is None`.
  - With the scaffold off, the primary stream pays **no** `gui_dwell_weight`
    cost, even though `curiosity.gui_dwell_weight: 0.05` is configured. The
    yaml comment at ~171 says "a menu now pays strictly <= 0", and that
    rests on this cost.
  - Only the scouts are charged, via `_scout_mixed_reward` (11102, called at
    7027).
  - A menu still earns 0 intrinsic, so this is not a farm. But the gradient
    toward the GUI exit that §4.4 relies on is absent for stream 0. The
    same gating exists in `_run_episode_parallel` (4939).
  - **Not fixed here**, because fixing it changes live reward. It needs its
    own test and a training-host run.
- **Mixer:** `policy/actor_critic.py:130` returns `w_i·scaled_int +
  w_e·ext`, with adaptive gating and return-ratio normalisation that only
  ever shrinks intrinsic.
- **Replay reward label:** `prim_extrinsic` for stream 0 and the raw
  `rewards[e]` for scouts (6823). No normalisation is applied at write
  time.

## 4. Model-update ownership

- **PPO:**
  - Stream 0 always trains it.
  - The scouts also train it when `_scouts_in_ppo` is set (522:
    `lifelong.enabled and policy.scouts_in_ppo`, default true). Their
    rewards come from `_scout_mixed_reward`.
  - The update runs at segment end with a bootstrap value per stream.
- **World model:**
  - Trained on the `lifelong.wm_train_every` cadence (250; ticket at 7192).
  - Runs in the background thread `wm-trainer` (7696), which calls
    `_train_world_model` (7448).
  - Samples the shared multi-stream buffer, so it learns from **all
    streams**.
  - Parameter reads and writes go through `_wm_param_lock`.
- **Curiosity / ICM:** `_curiosity_train` on the full n-env batch every step
  (5949).
- **Dream:** never inside `_collect_segment`. It runs in `run()` after each
  segment (augment mode). Dream control is off under lifelong.

## 5. Measurement surface

- `runlogs/metrics.jsonl` holds one record per segment (`_emit_metrics`).
  `runlogs/heartbeat.jsonl` holds a record every 15 s (`_emit_heartbeat`:
  steps/s, breaks, logs, gpu_mem_mb, income_now).
- **Not logged:**
  - per-horizon WM prediction error (only an aggregate `horizon_loss` exists
    in `training_metrics`, and it is not emitted);
  - process RSS;
  - VRAM beyond `torch.cuda.memory_allocated`.
- `tools/baseline_report.py` reads whatever is present and reports
  everything else as "unknown", with the reason.

## Unknowns and checks that need the training host

1. A real baseline: run `tools/run_manifest.py build --out
   runlogs/manifest.json` at launch, then `tools/baseline_report.py
   runlogs` on a completed run. **No fresh baseline numbers exist.** The
   README's historical numbers are not measurements.
2. Variation between runs: needs at least 2 runs from identical manifests.
   The shared Paper server is persistent state, so runs are not independent
   (§7.2).
3. Counter validation: `breaks_by_type` semantics across resets and break
   memory (CLAUDE.md §5, §7).
4. Checkpoint restoration parity: requires brain state, which exists only
   on the host.
5. Whether `option_executor` and `imagination_curiosity` are active at
   runtime. The config suggests yes; this was not verified.
6. Integration contract E of `_collection_path_smoke.py`
   (`DevelopmentalAI` refuses before `make_env`):
   - It needs gymnasium, so it is SKIPPED on the dev Mac.
   - It passed locally under a stubbed gymnasium.
   - It runs for real on CI and on the host.
7. A per-horizon WM error probe: it has to be added before Stage 6
   comparisons mean anything.
8. Process RSS and VRAM: use the host's node exporter or `ps`, because the
   run does not log them.
