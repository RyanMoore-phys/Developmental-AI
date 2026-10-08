# Telemetry inventory and learning-metric catalog — 2026-10-07

Read-only research. Nothing in the tree was changed to produce this. Line
numbers are from the working tree on branch `Improv` on 2026-10-07
(`developmental_ai/core/developmental_loop.py` is 13,065 lines; it moves, so
grep the quoted symbol before you trust a number).

Abbreviations: **DL** = `developmental_ai/core/developmental_loop.py`,
**ENV** = `developmental_ai/environments/minerl_env.py`, **AC** =
`developmental_ai/policy/actor_critic.py`, **RSSM** =
`developmental_ai/world_model/rssm.py`, **LP** =
`developmental_ai/curiosity/learning_progress.py`.

---

## 0. Constraints any change here has to keep

These are enforced by tests. A telemetry change that breaks one of them is
wrong even if the numbers look right.

| Contract | Where | What it means for telemetry |
|---|---|---|
| One emission site | `tests/_metrics_sink_smoke.py:134-144` counts `self._emit_metrics(episode_metrics)` == 1 (call at DL:3657) | Every new segment-level field goes **inside** `_emit_metrics` (DL:8745). Do not add a second call per stepping body (CLAUDE.md §4.2). |
| One `ledger.segment()` consumer | `_metrics_sink_smoke.py:147-187`; the one call is `infra/stack.py:646`, stashed at :654 | Signed provenance has to come out of that one call, through `last_ledger_segment`, or from a separate accumulator. A second `segment()` call empties the statement. |
| One oracle consumer | `tests/_oracle_isolation_smoke.py:139-162`: the AST string constant `"oracle"` must appear **once** in DL. The **text** of `_oracle_observe` must not contain `self.policy`, `world_model`, `replay_buffer`, `curiosity` or `reward` | The position trace and the stuckness stats have to be computed inside `_oracle_observe` (DL:10705). Keep the single `.get("oracle")` and loop over streams around it. **That ban covers comments and variable names too.** A comment saying "reward" inside the method fails the test. |
| Sink never raises, seq never rewinds | `_metrics_sink_smoke.py:44-131`, `infra/metrics_sink.py` | Reuse `MetricsSink` for any new JSONL stream. It already gives fsync, rotation and a monotone `seq` sidecar. |
| Ingest is replayable | `_metrics_sink_smoke.py:189-229` | New tables must be `INSERT OR IGNORE` on a UNIQUE key, and must rebuild identically from the JSONL. |

**Policy-visibility rule for everything below.** Every proposed metric is
**write-only telemetry**. No gate, reward term or policy input may read it.
This rule is already broken once: the `magnet_weight` gate reads
`last_ledger_segment["shares"]["magnet_seek"]` (DL:12754-12760). That makes
the ledger's output partly a control input. Any new provenance field
therefore lives in a **separate key**, and nothing may read it back.

---

## 1. What is persisted today

### 1.1 `runlogs/metrics.jsonl`: one record per segment (`_emit_metrics`, DL:8745-8926)

The sink adds `schema_version` (=1, `metrics_sink.py:44`), `seq` and
`wall_time` (`metrics_sink.py:168-172`). A segment is `lifelong.segment_len:
1024` loop steps × `num_envs: 2`, about 7 min (`configs/minecraft_skybot.yaml:3006`, `:1092`).

| Field | Source (line) | What it actually is |
|---|---|---|
| `total_timesteps`, `episode`, `uptime_s` | DL:8784-8788 | Fleet frames. `episode` = segment count in lifelong mode. |
| `reward_total` | ledger stash, DL:8795 | **Signed net** of the *ledgered sources only* (see 1.4). It is not the PPO reward. |
| `reward_shares` | DL:8796 | **Abs-gross** shares per source. Sign is lost. |
| `reward_hhi`, `reward_alarms` | DL:8797-8798 | Concentration; DOMINANT and LEDGER-ERROR strings. |
| `episode_reward` | `_collect_segment` return, DL:7464 | Σ raw env reward, **stream 0 only**. |
| `intrinsic_reward` | DL:7466 | Mean of `intrinsic[0]` per step, **after** GUI zeroing (DL:7005) and **before** the mixer weights. |
| `episode_length` | DL:7465 | Loop steps in the segment. |
| `curiosity`, `avg_episode_reward`, `elapsed_time_seconds` | DL:8801-8805 | **Never produced** by `_collect_segment`. They are always absent, so ingest's `curiosity` column is always NULL. |
| `policy_entropy` | tail of the `training_metrics` deque, DL:8814-8830 | Last PPO update's per-minibatch mean entropy. |
| `world_model_loss`, `kl_divergence`, `reconstruction_error`, `inverse_dynamics_loss`, `symbolic_decoder_loss/accuracy` | deque tail (`_record_wm_telemetry`, DL:7532-7578) | The **last gradient step of the latest finished async block**. It is not a block mean (DL:7679/7691 overwrite `metrics` every iteration), and its staleness is unknown. |
| `dream_distill_eff_weight`, `glue_mastery_score`, `glue_kg_density` | deque tail | |
| `imagination_bonus/boredom/probes` | `imagination_curiosity.stats`, DL:8835-8844 | |
| `breaks_by_type`, `breaks_total`, `logs`, `breaks_per_log` | `_last_env_info["breaks_by_type"]`, DL:8866-8877 | **Career, stream 0 only.** See §4. |
| `crafts_by_type`, `crafts_total` | DL:8879-8883 | Career, stream 0. |
| `attack_run_max/mean`, `attack_runs` | `runs_nobreak`, DL:8885-8890 | A **rolling 400-run deque** (ENV:472). It is not per segment. |
| `skills_bound`, `skills_refused`, `skill_disk_loads`, `cond_live`, `cond_load_failures`, `rssm_no_latent` | option bank, DL:8895-8905 | |
| `skills_minted/mastered/composite`, `avg_success_rate` | `skill_bank.get_stats()`, DL:8914-8923 | |

The same function also runs the memory-census hook (DL:8764-8776). It writes
its own output and adds nothing to the record.

### 1.2 `runlogs/heartbeat.jsonl`: every 15 s (`_emit_heartbeat`, DL:8669-8743)

It is called from `_infra_step` (DL:11113), so it is fed by stream 0's
per-step hook only. Fields: `total_timesteps`, `uptime_s`, `steps_per_s`,
`breaks_total`, `logs`, `attack_run`, `episodes` (DL:8700-8708);
`income_now` (top-12 **abs** shares from `_income_now`, DL:8722-8727); and
`gpu_mem_mb` (DL:8732-8736). `_income_now` (DL:8585-8616) includes `icm_base`
and `extrinsic`. The ledger lacks `icm_base`, so the two feeds disagree on
which sources exist.

### 1.3 Console segment block (stdout only; `_log_progress`, DL:11682-12720, plus `InfraStack.segment`, `infra/stack.py:563-725`)

All of this reaches nothing machine-readable unless someone greps the host log.

- **Reward / WM / losses:** `Reward (avg)` (11727), `Intrinsic reward`
  (11730), `WM loss … (n=, recon=)` (11736), `KL divergence` (11739),
  `Policy loss` (11740), `Curiosity loss` (11741), `Sym decoder` (11757),
  `Inverse dynamics` (11761), `Imagination` (11777), `Dream policy` /
  `Dream distill` (12660, 12692).
- **Behaviour:** `Body/behaviour: gui=…(runN) food hp depth tool carry hand`
  (11790-11808), `Curiosity split in-menu vs world` (11812), `Position:
  x y z cell` (12170), `Looking: pitch` (12190), `Actions:` full histogram
  (12195), `Reach (segment)` (12371), `Pitch (segment)` (12380), `Ticks-to-break`
  (12093), `Chop diagnosis` + `blocks broken:` (12619-12622), `Break sensor:
  mine_block present=` (12613).
- **Policy / PPO:** `Policy: entropy … % of max` (12219), `PPO trust region:
  approx_kl epochs updates` (12265), `PPO update rate` (12279), `PPO forensics:
  raw_adv_std |adv| obs_spread max_prob rows updates adv_clip V-CLAMPED` (12304).
- **Shaping terms:** `Coverage` (12203), `Magnet pay` (12359), `Shaping:
  persistence` (12474), `Gaze level` (12483), `Symbols` (12492), `Centring`
  (12504), `Reward census: ICM/LP base` (12520), `Action attribution`
  (12545), `LP internals: running_std median raw_pred_err` (12567), `Gaze`
  (12594), `Novelty(seen)` (12605), `Anticipation bonus` (11926).
- **Skills / options:** `Skills learned` (11818), `Mastery ledger` (11833),
  `Nested options` (11837), `Skill deltas` (12001), `Option invocations`
  (12006), `Option offers` / `Gate state` (12033-12058), `Chop budget` (12077).
- **Stage / lifelong:** `Dev stage (WM-error, n, slope)` (11862-11869),
  `Lifelong stream h_evolve ||h||` (11880), `Working set` (11887).
- **Timing:** `Loop timing` (12399), `Phase timing` across the 12 `_PHASES`
  (DL:9889) plus UNACCOUNTED (12425), `Env breakdown (in-thread)` (12444),
  `Feat drift` (12462); `AsyncWM replay-ratio … skip … ms/iter … reward pool`
  once per goal horizon (DL:7358).
- **Infra:** `infra/signals`, `infra/ledger` (top-5 abs shares),
  `infra/ledger ALARM`, `infra/farm FARM? cell loops +x/step actions`,
  `infra/drift`, `infra/heartbeat OVERDUE`, `infra/gate ALARM`,
  `infra/affordance`, `infra/episodic`, `infra/empowerment`, `infra/stuck:
  LEVEL n` (`stack.py:631-707`).
- `Mastery score` (12646), `KG density` (12647), and `ORACLE (evaluation only):
  dead-reckoning drift` via `logger.info` every 500 steps (DL:10753).

### 1.4 The reward ledger (`infra/ledger.py`)

- `record()` keeps **both** `_signed` and `_abs` per source (ledger.py:108-112).
- `segment()` swaps both out (ledger.py:139) and returns only `{"total":
  Σsigned, "shares": abs/abs_total, "hhi", "alarms"}` (:147, :164).
  **Per-source signed totals are discarded at line 139.** That is a one-key
  additive fix: `out["signed"] = signed`.
- **Inputs are a partial list.** At segment close it receives the
  `_ledger_snapshot` (DL:11692-11712): coverage, gaze, novelty, symbols,
  sym_center, persistence, reach, gaze_level, imagination, magnet_seek,
  extrinsic_env. It also receives per-step `record_reward` calls: memory_pull
  (11265), frontier (11291), anticipation (11326) and effort (11347).
- **Not ledgered, though each one moves the PPO reward on the live path
  `_collect_segment`:**
  - ICM/LP base (`_cen_base`, DL:6049);
  - empowerment shaping (`_infra_step` adds `_extra` from
    `empowerment_shaping`, DL:11189, with no `record_reward`);
  - approach reward `_ar` (DL:6705);
  - the GUI dwell cost (`_gui_reward`, DL:6733).
- **Multipliers are invisible:** boring-view (DL:6036), habituation (6639),
  farm damp (6755), GUI zeroing (7004-7005) and the mixer weights (7006).
- **Stream 0 only.** Scout rows, which train PPO under `lifelong.enabled`
  (DL:522; `_scout_mixed_reward` DL:11386, called at 7150), are never
  booked. On `num_envs: 2` that is about half the PPO rows with no
  provenance at all.

### 1.5 `docker/monitor/ingest.py` → SQLite on node1

- `segments` table: `seq` PK, `schema_version`, 34 scalar columns
  (ingest.py:40-64), and JSON columns `reward_shares`, `reward_alarms`,
  `breaks_by_type`, `crafts_by_type` (:65). Every other key lands verbatim in
  `extra` (:160).
- `heartbeat` table (:89-93, :191-196), including `income_now` as JSON.
- `server` table (:81-83).
- **Emitted but not promoted:** `reconstruction_error`, `imagination_*`,
  `skill_disk_loads`. They sit in `extra`.
- **Promoted but never produced:** `curiosity`.
- **Re-reads the entire live file on every 10 s poll** (:140-167) and never
  reads rotated files. That is fine at about 3 KB/segment. It will not scale
  to a position trace.
- The docstring says `crafts_by_item`; the key is actually `crafts_by_type`.
- Collector: `docker/monitor/collect_metrics.sh` (`tail -F` plus an rsync
  backfill of the remote `runlogs/metrics.jsonl`, and the heartbeat every
  30 s).
- Grafana (`docker/monitor/grafana/provisioning/dashboards/skybot.json`) has
  13 panels: live steps/s, breaks per log, HHI, players, live income shares,
  breaks vs logs, breaks by type, crafts, skills, learning signals (entropy,
  WM loss, KL, intrinsic), reward+attack, and uptime.

### 1.6 Other persisted artefacts

- **Break memory** `runlogs/breaks_by_type.json` (yaml:237). It holds career
  breaks, places, crafts, pickups and the coverage `cells` visit table
  (ENV:502-540; written on every break, ENV:2353).
- **Decision traces** `decision_traces.jsonl`, dumped on behaviour drift
  (`stack.py:671`).
- **Foundation shadow store** `runlogs/foundation_shadow` (yaml:3119-3134;
  **enabled live**, every 16 steps, 1 GB cap). It already files
  `info["oracle"]` readings (shadow.py:412-439) in the evaluator partition,
  for **all streams**, together with the action source
  (policy / option_slot / fallback). This is the closest thing to a position
  trace that exists today. It uses the EvidenceStore format, nothing on node1
  ingests it, and it is sampled with the shadow budget's degrade stages
  (shadow.py:38-46).
- Memory census (`infra/memory_census.py`, new, untracked).
- Checkpoints: `options_state.json`, skill bank, `knowledge_graph.json`, etc.

---

## 2. Computed but not persisted

"Cost" is the marginal cost of persisting a value that is already computed.
Unless stated, it is a dict copy into the segment record, which is
negligible next to the per-segment fsync.

| Value | Lives at | Persisted? | Cost to log |
|---|---|---|---|
| **PPO update dict**: `value_loss`, `approx_kl`, `epochs_run/requested/capped`, `n_updates`, `max_updates`, `kl_stopped`, `minibatched`, `raw_adv_std`, `raw_adv_absmean`, `obs_spread`, `max_prob`, `rows`, `rows_option/primitive`, `tau_mean`, `env_steps`, `adv_clipped_frac`, `value_clamped_frac`, `value_space` | Returned by AC:1649-1687 and kept as `self._last_ppo` (DL:7451) | Printed only. `policy_loss` reaches `training_metrics` but is **not emitted** | ~0: `rec["ppo"] = dict(self._last_ppo)` |
| PPO `nonfinite_skip`, `policy.last_update_forced` | AC (skip path), read at DL:12304-12318 | printed | ~0 |
| **PPO values never computed**: clip fraction (`ratio` outside 1±ε at AC:1544), explained variance (values vs `returns`, AC:1407), pre-clip grad norms (return value of `clip_grad_norm_` discarded at AC:1589-1590), post-normalisation advantage stats, value/return mean | — | no | One reduction per minibatch plus `.item()` syncs, about 10-50 µs each. Accumulate as tensors and sync once per update. |
| PPO learning rates (static Adam, AC:666, 1966-1968) | optimizer | no | Once per run, in the manifest |
| **WM loss heads** `reward`, `continue`, `flow_smooth` | `compute_loss` dict, RSSM:2369-2391 | **never recorded at all** (not in `_WM_TELEMETRY`, DL:7519-7530) | ~0 (add to the tuple) |
| WM `flow`, `flow_mask`, `horizon`, `slot` | `training_metrics` via `_WM_TELEMETRY` | recorded, **not emitted** | ~0 |
| WM block **mean** across `train_iters` steps | overwritten at DL:7679/7691 | no; only the last step survives | One float add per key per step in `_train_world_model` |
| WM pre-clip grad norm | discarded return value, RSSM:2443 | no | ~0 (one `.item()` per gradient step) |
| WM prior/posterior entropy, free-bits clamp hit fraction | inside `compute_kl_loss` (RSSM ~400-430) | no | Small: these tensors exist already |
| WM NaN skips | `_record_wm_telemetry` silently drops NaN (DL:7569) | **not counted** | One counter |
| Async WM: `_wm_cadence_hits`, `_wm_blocks_run` (skip rate), `_wm_block_ms`, `_wm_block_iters`, `_wm_reward_pool`, live `_wm_train_iters` | DL:7322-7370, 7692-7700 | printed once per goal horizon | ~0. **Snapshot them before the reset at DL:7403-7404.** |
| `_wm_skip_count`, `_wm_empty_count` | DL:7726, 7760 | logger warnings only | ~0 |
| ICM `icm_forward`, `icm_inverse` | `icm.py:364-367` (`_last_icm_metrics`, DL:10023-10040) | dropped; `icm_total` reaches `curiosity_loss`, **not emitted** | ~0 |
| `curiosity.stats` (`curiosity_mean/std`, `exploration_ratio`, history size) | `icm.py:425-431` | printed (`Exploration ratio`) | ~0 |
| LP `_running_std`, `_running_median`, `last_pred_error`, `last_action_attribution`, `last_flow_error` | LP:476-501, 552-562 | printed (`LP internals`) | ~0 |
| **LP bucket count** `len(_bucket_err)`, open visit runs `len(_lp_runs)` | LP:113, 283 | **never reported** | ~0 |
| **LP gate outcomes**: evaluated / drop≤0 / abs-floor reject / t-test reject / pass | branches of `_lp_from_history`, LP:381-425 | **not counted** | 5 int counters, O(1) per entry |
| Segment sums `_attr_sum/_attr_nz`, `_bv_sum`, `_gui_intr` (in-menu vs world raw surprise) | DL:12512-12560, 11809-11815 | printed | ~0 |
| `progress_curiosity.stats` (probes, last_mean_loss, last_progress, forgetting, eval_errors) | `infra/progress_curiosity.py:332`, stashed as `infra.progress_measurement` (DL:11232) | no | ~0 |
| Dream `dream_actor_loss`, `dream_critic_loss`, `dream_entropy`, `dream_returns_mean`, `dream_distill_loss`, `dream_distill_gate_frac` | AC:2112-2117; `training_metrics` (DL:3459-3485) | only `eff_weight` emitted | ~0 |
| Per-term shaping sums (signed): `_cov_sum`, `_gaze_sum`, `_nov_sum`, `_sym_sum`, `_symc_sum`, `_persist_sum`, `_reach_sum`, `_pitch_level_sum`, `_cen_imag`, `_cen_base`, `_mag_turn/_mag_other`, `_anticip_sum`, `_seg_extrinsic_sum` | DL:11692-11712, 8594-8609 | Fed to the ledger, where the sign is lost; `_cen_base` is never fed | ~0: read before the reset in `_log_progress` |
| Option executor `snapshot()`: `offered_vs_chosen` (decisions, with_offer, picked, probation, retry_after, with_learned, learned_offered_sum, scout_rows_dropped), `gating` (offers, restricted, competence_floor), `nested` | `policy/options.py:1874-1906` | printed (12033-12058) | Small: skip `recent`/`co` and take the counters |
| `bank.invoked` per slot, per-slot `delta_mag`, per-skill `asked_log`/`wm_fidelity` | options.py:144, 209; DL:11822-11833, 11994-12001 | printed top-N | small |
| Stage controller `stage`, `_level()`, `_slope()`, `len(error_history)`, `_stage_blind_segments` | DL:11857-11872 | printed | ~0 |
| Lifelong `h_liveness()`, `h_norm()` | DL:11880 | printed | ~0 |
| Phase accumulators `_phase_acc[12 phases]`, `_env_wait_sum/n`, `_env_parts` | DL:9954-9968, 9927-9952 | printed, then **reset** at DL:12450-12455 | ~0, but snapshot them before the reset. `_log_progress` runs before `_emit_metrics`, so `_log_progress` has to stash them. |
| Infra: stuck level/reason (`stack.py:701-721`), farm report (`:663-665`), gate alarms, heartbeat overdue, degenerate/starved sets, empowerment last score, affordance dead actions, `cells_delta` (DL:12718-12721) | infra stack | printed | ~0. Return them in `actions` or stash them like the ledger. |
| Oracle `oracle_dr_drift` and `_oracle_travel` | DL:10743-10760 | `training_metrics` + logger; **not emitted** | ~0 |
| Env info: `pickups_by_type`, `inventory`, `events`, `mine_block_present/entries/nonzero`, `gui_open`, `gui_frac`, `gui_run_max`, `action_hist`, `break_ticks`, `attack_run_max`, `effort`, `swing_failed`, `step_ms` | ENV:2426-2500, 2030-2038 | Partly printed for stream 0; **scouts' are never read** | Per-step dict reads are already done |
| Body: `_proprio_per_env[0]` food/hp/depth/tool/carry, `mainhand` | DL:11796-11804 | printed | ~0 |
| `MetricsSink.state()` (`written`, `dropped`, `last_error`) | metrics_sink.py:204-212 | **no**: the tracker never reports its own drops | ~0 |
| Replay buffer: `len`, capacity, `nbytes`, block count, growth refusals (`MEMORY_BUDGET.request`), PER priority stats, reward/terminal pool sizes | `world_model/replay_buffer.py` (`maybe_grow` :557, `_priority_probs` :873, `_scan_reward_starts` :1000) | no (`reward_pool` printed only) | small; PER stats take an O(N) pass, so sample at most once per segment |

---

## 3. The oracle (RED) channel

**What it provides.** It provides only `true_position = [x, y, z]`, float32,
read from `ctx["world"]` (`sensors/builtins.py:266-267, 406-411`). It is
classified RED and is returned by `SensorBus.read_oracle`
(`sensors/registry.py:271`). The env puts it in `info["oracle"]`
(ENV:2414-2419). **The oracle carries no yaw or pitch.** Yaw and pitch live
in `info["world"]` (ENV:1110-1117). That dict is the GREEN source the proprio
heading is built from, so reading yaw/pitch from `world` for telemetry
crosses no new boundary. The live config enables `true_position` (yaml:331).

**How `_oracle_observe` uses it now** (DL:10705-10760; called once per step
in both bodies, DL:4806 and 6000):

- Stream 0 only (`step_infos[0]`, DL:10726).
- It sets an origin at the first reading after `_oracle_reset` (episode
  start).
- It accumulates the **xz path length** `_oracle_travel` with no teleport
  filtering (DL:10743).
- It compares against dead reckoning `world["dr_x"], ["dr_z"]` to get the
  drift error (DL:10745-10748).
- Every `oracle_report_every` (500) steps it logs the mean drift and appends
  `oracle_dr_drift` to `training_metrics` (DL:10750-10760). Nothing emits it.
- It never records the position itself, the y coordinate, or anything about
  stuckness.

**Absolute position already exists outside the oracle, and some of it is
policy-visible.** This is not a leak through `info["oracle"]`, but it matters
for the catalog. `info["world"]` carries `xpos/ypos/zpos`, the alias `x/z`,
`cell = (x//8, z//8)`, `cells_seen` and `moved` (ENV:1087-1156). It feeds:

- the **coverage intrinsic** `1/sqrt(1+n)` per absolute cell (ENV:1103);
- the farm detector's cell, whose damp multiplies `intrinsic[0]` (DL:6755,
  11353);
- episodic positions and `_memory_pull_phi` (DL:11045-11063; only active
  with the vision scaffold, which is off live);
- the stuck monitor via `cells_delta` (DL:12718-12794).

So **the evaluator measures of position must be computed separately** from
these, in `_oracle_observe`. They should not reuse `cells_seen`, which is a
reward-coupled, career, capped counter (see §4).

**The shadow recorder is a second reader of `info["oracle"]`**
(`foundation/runtime/shadow.py:439`). It lives outside DL, so contract D does
not count it. Its writes go to the evaluator partition.

---

## 4. Validity of the key counters (CLAUDE.md §5: validate before building on them)

| Counter | Verdict | Evidence |
|---|---|---|
| `breaks_by_type` / `breaks_total` | **Career, stream 0 only, survives restarts.** It is not per run and not per segment. | Restored from break memory at adapter init (ENV:502-540). Incremented by per-step `mine_block` deltas (ENV:2332-2352). Those deltas handle §7's per-world reset correctly, because `_mine_prev` is re-baselined in `reset()` (ENV:1853-1867). Scouts get no break memory (DL:4150 "primary stream's biography"). |
| `logs` (metrics/heartbeat) | Career count of **broken blocks whose name contains "log"**. It is not logs obtained. | DL:8873, 8699. Breaking a log whose drop is never picked up still counts. |
| `info["logs"]` | **A different quantity with the same name**: logs in the inventory *now* | ENV:1852, 2458. Do not join it with the one above. |
| `breaks_per_log` | A career ratio, dominated by history | It cannot show a recent improvement. Report a per-segment and per-run ratio next to it. |
| `achievements["mine_*"]` | Per-world `mine_block` totals; restart at 0 each reset | ENV:2459-2461. Never baseline against these across episodes (§7). |
| `mine_block_present` | Sensor health. A zero break count with `present=False` means a blind sensor, not an idle agent. | ENV:2448-2455. **Not emitted.** |
| `crafts_by_type` | Career, stream 0, persisted | ENV:516-518, 2477 |
| `places_by_type` | **Known garbage** (CLAUDE.md §5: `iron_axe: 2281`) | Exclude it, or tag it `unvalidated`. |
| `action_hist`, `gui_frac`, `gui_run_max` | **Cumulative since the adapter was constructed** (process lifetime). Never reset. | ENV:394, 548-551, 2030-2038. The per-segment console `Actions:` and `gui=` lines are therefore *since process start*, though they print every segment. |
| `attack_run_*` | A trailing window of the last 400 swing runs | ENV:472 |
| `cells_seen` | Career visit table (8-block cells), persisted in break memory, **capped at 50,000 with FIFO eviction** | ENV:1093-1096, 525-530. **Once at the cap, `len()` stops growing, so `cells_delta` reads 0 even while the agent explores new ground.** The stuck monitor consumes `cells_delta` (DL:12794, `stack.py:703`). That is a latent guard-becomes-latch (§4.1). The cap is 3.2 km² of 8×8 cells. Check the live count on the host before acting. |
| `episode_reward` / `intrinsic_reward` | Stream 0 raw env reward / stream 0 intrinsic before weighting | DL:7304-7305, 7464-7466. Neither is the reward PPO trained on. |
| `reward_total` / `reward_shares` | Partial: missing sources, missing multipliers, stream 0 only (§1.4) | The shares are abs-based by design (ledger.py:75-84). |
| Deque-tail learning signals | The latest value, not a segment statistic. WM values may predate the segment. | DL:8814-8830 |
| `total_episodes` | Segment count in lifelong mode | DL:3155-3167 |

**Validation hook to build in.** Count per-segment breaks from the typed
event stream (`("break", type)` in `info["events"]`, ENV:2334) for every
stream. For stream 0, `Σ events` must equal the difference of its career
`breaks_by_type`. A mismatch is a broken counter, and this check turns §5's
lesson into an automatic one.

---

## 5. Metric catalog

Columns:

- **Exists?** names the place where the value is already emitted, or says
  `printed` (stdout only), `computed` (in memory, never output) or `NEW`.
- **Emitter** is where the value should be produced.
  - `EM` = `_emit_metrics` (DL:8745). It is the only segment-record writer.
  - `OO` = `_oracle_observe` (DL:10705). It is the only oracle reader.
  - `LP*` = `_log_progress`, which must stash its segment snapshot into
    `self._seg_stash` before the existing resets, for `EM` to read.
- **Policy-visible** is "no" for every row. That is a requirement. Nothing
  reads these values back.
- Frequency is per segment unless stated.

### A. Reward provenance (signed, per source, per stream)

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `rew.signed.<src>` | Σ signed amount per ledger source this segment | `RewardLedger.segment()`: add `"signed": signed` at ledger.py:164/147 | ~0 | ledger → stash → EM | **NEW** (computed, discarded at ledger.py:139) |
| `rew.abs.<src>` | Σ\|amount\| per source | same | ~0 | EM | computed (shares only) |
| `rew.missing_src` | Booked additionally: `icm_base` (`_cen_base`), `empowerment` (DL:11189), `approach` (`_ar`, DL:6705), `gui_dwell` (`_gui_reward`, DL:6733). **Book them into a separate `ProvenanceBook`, not into the ledger.** Adding sources would change the shares that the `magnet_weight` gate reads (DL:12754). | per-step float adds at those sites | ~1 µs/step | A new per-stream accumulator. The edits sit in **both** stepping bodies (§4.2: `replace_all`, then confirm the count is 2). | **NEW** |
| `rew.mult.<name>.mean` | Mean multiplier applied to `intrinsic[0]`: boring-view (6036), habituation (6639), farm damp (6755), GUI zero (7005) | per-step | ~1 µs/step | same accumulator | **NEW** (`_bv_sum` printed) |
| `rew.ppo.mean/std/min/max` per stream | Stats of the **mixed** reward actually stored in PPO rows (`mixed` at DL:7006; scouts' `_sc_mixed`, DL:7150) | per-step | ~1 µs/step | accumulator → EM | **NEW**: the number PPO optimises is not logged anywhere |
| `rew.mixer.w_int/w_ext` | `reward_mixer.weights` | AC:252 | ~0 | EM | printed (11844) |
| `rew.channel.int_pre_gui/int_post_gui/ext` per stream | Intrinsic before and after zeroing, and prim_extrinsic | per-step | ~0 | accumulator | partly (`intrinsic_reward` = post, stream 0) |
| `rew.hhi`, `rew.alarms`, `rew.total` | as today | ledger | — | EM | **exists** `reward_*` |
| `rew.farm.top` | Top 5 FarmDetector cells: loops, EMA income, top actions | `FarmDetector.segment_report` data (ledger.py:360) | small | stash → EM | printed |

### B. Task outcomes

| Name | Definition | Source | Freq | Cost | Emitter | Exists? |
|---|---|---|---|---|---|---|
| `task.breaks_seg.<type>` per stream | Breaks this segment, from typed events | `info["events"]` of every stream | per-step count → seg | ~1 µs/step | accumulator in both bodies → EM | **NEW** |
| `task.logs_seg` per stream | Subset of the above whose type contains "log" | same | seg | ~0 | EM | **NEW** |
| `task.breaks_career.*`, `task.logs_career` | Today's fields, renamed honestly | `_last_env_info` | seg | ~0 | EM | **exists** as `breaks_by_type` / `logs` (keep the old keys) |
| `task.breaks_run.*` | Career minus the value at process start | baseline at first segment | seg | ~0 | EM | **NEW** |
| `task.counter_check` | `Σ events − Δcareer` for stream 0; must be 0 | B rows above | seg | ~0 | EM | **NEW** (validation) |
| `task.crafts_seg.*`, `task.pickups_seg.*`, `task.deaths_seg` | Events by kind | `info["events"]` | seg | ~0 | EM | **NEW** (pickups never emitted) |
| `task.first_of_type` | Block/item types seen for the first time in career this segment | diff of career keys | seg | ~0 | EM | **NEW** |
| `task.inventory_end` per stream, `task.inventory_delta` | Item→count at segment end and change across the segment | `info["inventory"]` (ENV:2500) | seg | small | EM | **NEW** |
| `task.mine_block_present/entries/nonzero` | Break-sensor health | ENV:2448-2455 | seg | ~0 | EM | printed only |
| `task.ticks_to_break` (mean, p90) | `break_ticks` | ENV:2431 | seg | ~0 | EM | printed |

### C. Behaviour

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `beh.actions_seg.<name>` per stream | Primitive action counts this segment | Difference of cumulative `action_hist` per stream, or count `env_actions[e]` in the loop | ~0 | EM (diff) | **NEW** (only a cumulative histogram is printed) |
| `beh.action_entropy_emp` | Entropy of that empirical histogram (nats), plus `% of log(A)` | derived | ~0 | EM | **NEW** |
| `beh.policy_entropy` | as today | PPO | — | EM | **exists** |
| `beh.source_frac.{policy,option,fallback,dream}` | Share of steps by who chose the action | option executor state after `act()` (the shadow recorder already derives this) | ~1 µs/step | accumulator | **NEW** (shadow store only, sampled) |
| `beh.gui_frac_seg`, `beh.gui_run_max_seg`, `beh.gui_entries_seg` per stream | GUI-open share, longest run, open count **this segment** | `info["gui_open"]` per step | ~0 | accumulator | **NEW** (cumulative `gui_frac` printed) |
| `beh.idle_frac` | Steps with `moved < 0.05`, no event, and action ∈ {noop, camera} | `world["moved"]`, events | ~0 | accumulator | **NEW** |
| `beh.attack_frac`, `beh.futile_swing_runs` | Attack share; runs ending with no break (`swing_failed`) | info | ~0 | accumulator | partly (rolling `runs_nobreak`) |
| `beh.pitch_mean`, `beh.pitch_clamp_frac` | as printed | `_pitch_sum/_pitch_clamped` | ~0 | LP* | printed |
| `beh.reach_mean`, `beh.in_reach_frac` | | `_reachval_*` | ~0 | LP* | printed |
| `beh.body.{food,hp,depth,carry}`, `beh.mainhand` | stream 0 proprio at segment end | `_proprio_per_env` | ~0 | EM | printed |

### D. Exploration and stuckness (evaluator only; oracle-derived)

All of these are computed in **OO**, for **every stream**, from `true_position`.
OO writes a per-stream segment summary to `self._eval_seg[e]`. EM reads it
the way it reads the ledger stash, and OO resets it after EM has read. OO
also appends samples to the trace sink (§6.2). Raw samples come from every
loop step, which costs about 2 µs/stream/step. They are **never** fed to any
reward, gate or advisor.

| Name | Definition | Freq | Exists? |
|---|---|---|---|
| `expl.path_xz` | Σ hypot(Δx, Δz) between consecutive samples, **excluding teleports** (a single-step jump > 8 blocks, or a `death`/`env_restarted` step), counted separately | seg | partly (`_oracle_travel`: per episode, stream 0, no teleport filter) |
| `expl.path_3d` | Same, including Δy | seg | NEW |
| `expl.net_disp` | hypot(x_end − x_start, z_end − z_start) | seg | NEW |
| `expl.straightness` | net_disp / path_xz (0 = circling, 1 = straight) | seg | NEW |
| `expl.rg_xz` | Radius of gyration over the segment's samples: sqrt(mean\|p − p̄\|²) | seg | NEW |
| `expl.rg_trailing` | Same over a trailing window of the last 4096 samples (a ring buffer) | seg | NEW |
| `expl.cells4_new_seg`, `expl.cells16_new_seg` | Cells (4-block and 16-block grids) **first visited this run**. The run-local set is capped at 200k with an `overflow` count, and never evicts silently. | seg | NEW. The adapter's `cells_seen` is reward-coupled and capped; do not reuse it. |
| `expl.cells_per_hour` | `cells4_new` / wall hours | seg (Grafana can also derive it) | NEW |
| `expl.steps_since_new_cell` (end value and max in segment) | Loop steps since the last new 4-block cell | seg | NEW |
| `expl.top_cell_dwell_frac` | Share of samples in the most visited 4-block cell this segment | seg | NEW |
| `expl.y_min/max/mean` | Vertical range; catches pits and canopy perches (the 2026-08-02 incident) | seg | NEW |
| `expl.teleports`, `expl.deaths` | Count | seg | NEW |
| `expl.dr_drift_mean`, `expl.dr_drift_per_block` | Existing drift measure, now per stream and emitted | seg | computed (`oracle_dr_drift`) |
| `expl.stuck_eval` | Evaluator verdict `path_xz < 16 ∧ rg_xz < 4 ∧ steps_since_new_cell > 2048`. It is reported only, as an **independent check on** `infra/stuck`, never fed to it. | seg | NEW |
| `expl.dist_from_origin` | Distance from the episode-origin position (the dead-reckoning frame) and from world spawn (0,0) | seg | NEW |

### E. World model

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `wm.loss.{total,reconstruction,kl,reward,continue,inverse,flow,flow_smooth,horizon,slot}` | **Block mean** over the `train_iters` gradient steps | `compute_loss` dict (RSSM:2369); mean in `_train_world_model` | ~10 float adds per gradient step | `_record_wm_telemetry` → EM | partial: total/kl/recon/inverse emitted as last-step values; reward/continue/flow_smooth never recorded |
| `wm.flow_mask` | Mean retained fraction | same | ~0 | same | computed |
| `wm.grad_norm_pre_clip` (mean, max), `wm.clip_frac` | Return of `clip_grad_norm_` (RSSM:2443) and the share > 100 | per gradient step | 1 `.item()` | same | **NEW** |
| `wm.prior_entropy`, `wm.post_entropy`, `wm.kl_raw` vs `wm.kl_freebits` | Categorical entropies; KL before and after the free-nats clamp | `compute_kl_loss` (RSSM ~400) | small | same | **NEW** |
| `wm.blocks_seg`, `wm.last_block_age_s`, `wm.steps_seg` | Finished blocks this segment; wall age of the latest metrics | trainer thread | ~0 | EM | **NEW**: staleness is unknown today |
| `wm.stage`, `wm.stage_level`, `wm.stage_slope`, `wm.stage_n`, `wm.stage_blind_segs` | Developmental stage controller | DL:11857-11872 | ~0 | LP* | printed |
| `wm.symdec.{loss,acc,conf}` | as today | sd stats | ~0 | EM | exists (loss, acc) |
| `wm.lifelong.{h_evolve,h_norm}` | | DL:11880 | ~0 | EM | printed |

### F. Policy / PPO

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `ppo.{policy_loss,value_loss,entropy,approx_kl}` | Per-update means | `self._last_ppo` | ~0 | EM | entropy only (policy_loss in deque, not emitted) |
| `ppo.{epochs_run,epochs_requested,epochs_capped,n_updates,max_updates,kl_stopped,minibatched}` | Update-budget forensics | `_last_ppo` | ~0 | EM | printed |
| `ppo.{rows,rows_option,rows_primitive,tau_mean,env_steps}` + `ppo.rows_by_stream` | Rollout shape | `_last_ppo` (+ per-stream count) | ~0 | EM | printed (no per-stream) |
| `ppo.{raw_adv_std,raw_adv_absmean,adv_clipped_frac,obs_spread,max_prob,value_clamped_frac,nonfinite_skip,forced}` | Collapse and divergence forensics | `_last_ppo` | ~0 | EM | printed |
| `ppo.clip_frac` | mean(\|ratio − 1\| > clip_range) | AC:1541-1546 | per minibatch | AC returns → EM | **NEW** |
| `ppo.explained_var` | 1 − Var(R − V)/Var(R), on the update's rows | AC:1407 | one pass | same | **NEW** |
| `ppo.adv_norm.{mean,std,min,max}`, `ppo.value.{mean,std}`, `ppo.return.{mean,std}` | | AC train_step | one pass | same | **NEW** |
| `ppo.grad_norm.{actor,critic}` (mean, max) and `ppo.grad_clip_frac` | Pre-clip norms (AC:1589-1590) | per minibatch | `.item()` ×2 | same | **NEW** |
| `ppo.lr.{actor,critic}`, `ppo.entropy_coef`, `ppo.clip_range` | Static. Emit once per run and on change. | optimizer | ~0 | manifest + EM | **NEW** |
| `ppo.update_ms` | `ppo` phase time | `_phase_acc["ppo"]` | ~0 | LP* | printed |

### G. Curiosity internals

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `cur.icm.{total,forward,inverse}` | Segment mean of ICM train losses | `_last_icm_metrics` | ~0 (accumulate) | EM | `curiosity_loss` deque only, not emitted |
| `cur.pred_err.{mean,p50,p95}` per stream | Raw forward-model error | `last_pred_error` (LP:476) per step | ~0; p95 from a small reservoir | accumulator | printed (last value) |
| `cur.lp.gate.{evaluated,nonpositive,absfloor_rej,ttest_rej,passed}` | Per-segment counts of the `_lp_from_history` branches | LP:381-425 | 5 int adds | LP counters → EM | **NEW** |
| `cur.lp.pass_rate` | passed / evaluated (a frozen-model baseline is ≤ 2%, per LP docstring) | derived | ~0 | EM | **NEW** |
| `cur.lp.n_buckets`, `cur.lp.open_visits` | `len(_bucket_err)`, `len(_lp_runs)` | LP | ~0 | EM | **NEW** |
| `cur.lp.{running_median,running_std}`, `cur.lp.std_floored_frac` | Normaliser state; share of steps where std < 1e-3 (the floor bites) | LP:552-562 | ~0 | EM | printed |
| `cur.lp.nonzero_frac` | Share of steps with LP reward > 0 | per step | ~0 | accumulator | **NEW** |
| `cur.attr.{mean,nonzero_frac}` | Action attribution | `_attr_*` | ~0 | LP* | printed |
| `cur.flow_residual` | as recorded | `training_metrics["flow_residual"]` | ~0 | EM | computed |
| `cur.gui_split.{in_menu,world}` | Raw surprise per step in a menu vs in the world | `_gui_intr` | ~0 | LP* | printed |
| `cur.imagination.{bonus,boredom,probes}` | as today | `imagination_curiosity.stats` | — | EM | **exists** |
| `cur.progress.{probes,last_mean_loss,last_progress,forgetting,eval_errors}` | Paired-progress shadow | `infra.progress_measurement` | ~0 | EM | **NEW** (computed) |
| `cur.exploration_ratio`, `cur.mean`, `cur.std` | `curiosity.stats` | icm.py:425 | ~0 | EM | printed |
| `cur.anticipation.{mean,payouts,lifetime_paid}` | | DL:11926 | ~0 | LP* | printed |

### H. Replay / data

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `replay.size`, `replay.size_by_stream`, `replay.capacity`, `replay.nbytes`, `replay.blocks` | Occupancy | `ReplayBuffer` (`__len__`, `.nbytes` :109) | ~0 | EM | **NEW** |
| `replay.grow_refused` | `MEMORY_BUDGET.request` refusals (replay_buffer.py:278) | counter | ~0 | EM | **NEW** |
| `replay.added_seg` | Transitions added this segment | counter in `add` | ~0 | EM | **NEW** |
| `replay.ratio` | (blocks × train_iters × batch × seq) / added_seg: the true replay ratio | derived | ~0 | EM | **NEW** (config states only the nominal value) |
| `replay.reward_pool`, `replay.terminal_pool` | Goal and terminal window counts in the latest batch | `_wm_reward_pool` | ~0 | EM | printed |
| `replay.per.{mean,max,entropy}` | PER priority distribution | `_priority_probs` (:873) | O(N); once per segment | EM | **NEW** |
| `replay.sample_age_mean` | Mean (now − index) of sampled starts | sampler | small | trainer → EM | **NEW** |

### I. Skills / options

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `opt.bound/minted/mastered/composite/refused`, `opt.avg_success` | as today | bank / skill_bank | — | EM | **exists** |
| `opt.invoked_seg.<slot>` | Difference of `bank.invoked` | options.py:209 | ~0 | EM | **NEW** (cumulative top-4 printed) |
| `opt.ovc_seg.{decisions,with_offer,picked,probation,with_learned,learned_offered,scout_rows_dropped}` | Difference of `snapshot()["offered_vs_chosen"]` | options.py:1882-1891 | small | EM | printed |
| `opt.gating.{offers,restricted,competence_floor}` | | options.py:1892-1895 | ~0 | EM | **NEW** |
| `opt.nested.{pushes,cycle_blocks,depth_blocks,max_depth}` | | options.py:1896-1904 | ~0 | EM | printed |
| `opt.delta_mag.<slot>` | Individuation per slot | bank slots | ~0 | EM | printed top-8 |
| `opt.mastery.<skill>.{asked,hit,wm_fidelity}` | Per skill. Key by `slot_keys` and goal, **never by name** (§4.6). | skill_bank | small | EM | printed top-6 |
| `opt.duration.{mean,p90}` | Option length in env steps (`tau`) | option close events | ~0 | EM | partial (`tau_mean` of PPO rows) |

### J. Timing / throughput

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `time.step_ms`, `time.env_ms`, `time.own_ms`, `time.steps_per_s` | Loop timing | `_env_wait_*`, segment wall time | ~0 | LP* | printed (`steps_per_s` in heartbeat) |
| `time.phase_ms.<phase>` (12 phases + `unaccounted`) | Per-step ms | `_phase_acc` | ~0 | LP* (before reset at 12450) | printed |
| `time.env_parts_ms.<engine,obs,sense,adapter>`, `time.env_slowest_ms`, `time.pool_overhead_ms` | | `_env_parts` | ~0 | LP* | printed |
| `time.wm.{block_ms,iters,ms_per_iter,skip_rate,train_iters_live}` | Async trainer cadence | DL:7353-7402 | ~0 | snapshot before the reset at 7403 → EM | printed per goal horizon |
| `time.seg_wall_s`, `time.ckpt_ms`, `time.sink_write_ms` | | wall clocks | ~0 | EM | **NEW** |
| `time.shadow.{stage,ms_per_step}` | Shadow recorder budget state | `shadow.stats()` (shadow.py:801) | ~0 | EM | **NEW** |
| `time.vlm.{calls,ms}` | LLM / VLM calls | `llm_stats` (DL:12655) | ~0 | EM | printed |

### K. Health

| Name | Definition | Source | Cost | Emitter | Exists? |
|---|---|---|---|---|---|
| `health.nan.wm_skipped` | NaN losses dropped by `_record_wm_telemetry` (DL:7569) | counter | ~0 | EM | **NEW** |
| `health.nan.ppo_skipped` | `nonfinite_skip` updates | `_last_ppo` | ~0 | EM | printed |
| `health.nan.ledger` | LEDGER-ERROR count | ledger alarms | ~0 | EM | inside `reward_alarms` strings |
| `health.grad_spike.{wm,actor,critic}` | grad_norm > EMA + 4·EMSD (Welford over segments) | E/F grad norms | ~0 | EM | **NEW** |
| `health.value_clamped_frac` | Critic divergence | `_last_ppo` | ~0 | EM | printed |
| `health.wm.{skip_count,empty_count}` | Trainer exceptions and empty blocks | DL:7726, 7760 | ~0 | EM | logger only |
| `health.sink.{written,dropped,last_error}` | The tracker's own health | `MetricsSink.state()` | ~0 | EM (previous write's state) | **NEW** |
| `health.infra.{stuck_level,stuck_reason,gate_alarms,heartbeat_overdue,degenerate,starved}` | Infra verdicts | `stack.segment` lines/actions | ~0 | stash → EM | printed |
| `health.env.{resets,rebuilds,client_recoveries}` per stream | Client churn | env and loop counters | ~0 | EM | **NEW** |
| `health.mem.{gpu_alloc_mb,gpu_reserved_mb,rss_mb}` | | torch / `/proc` | ~0 | EM | gpu_alloc in heartbeat only |
| `health.feat_drift_max` | A2 self-check | `policy.last_feat_drift` | ~0 | LP* | printed |

---

## 6. Storage recommendation

### 6.1 Segment stream: `runlogs/metrics.jsonl`, schema_version 2

- **Same sink, same single emission site.** Bump `SCHEMA_VERSION` to 2
  (`metrics_sink.py:44`). Keep every v1 key unchanged, so that today's ingest
  SCALARS, Grafana panels and the `_metrics_sink_smoke` replay test keep
  working. Add the new metrics as **flat dotted keys**
  (`"ppo.clip_frac": 0.12`). Open-ended maps stay nested JSON
  (`"rew.signed": {...}`, `"task.breaks_seg": {"0": {...}, "1": {...}}`).
- Per-stream values go under the key `"s<e>"` inside each map, so streams
  are never averaged together.
- Add `run_id` (process start time plus host) and
  `window: {steps0, steps1, wall0, wall1}`, so per-segment deltas can be
  checked against cumulative fields.
- Expected size is about 6-10 KB/record at ~200 segments/day, so 2 MB/day.
  The existing 64 MB × 5 rotation lasts years.
- **Cost discipline.** Everything above is a read of a value that already
  exists, or an O(1) accumulator. The only new GPU syncs are the PPO and WM
  grad norms and the clip fraction. Batch them into one `.item()` per
  update.

### 6.2 Position trace: `runlogs/trace.jsonl` (new, schema_version 1)

- Writer: a second `MetricsSink` instance (like the heartbeat sink,
  DL:2355-2366) with `fsync: false`, `max_bytes` 32 MB, `keep` 8. **It is
  written only from `_oracle_observe`.** That keeps the single
  `.get("oracle")`; the method loops over `step_infos` inside it. Name the
  attribute something like `self._trace_sink`; the words `reward` and
  `curiosity` must not appear anywhere in the method body (contract D).
- Sampling: every `trace_every_steps` (default 16, matching the shadow
  recorder) **per stream**. Always also sample on an event (break, death,
  GUI open/close, teleport).
- Record:
  `{schema_version, seq, wall_time, run_id, t (total_timesteps), s (stream),
  ep (episode id), x, y, z, yaw, pitch, dr_x, dr_z, moved, gui, ev (event
  kinds this step, optional)}`. That is about 160 bytes.
  - Throughput: at ~5 loop steps/s × 2 streams / 16, about 0.6 lines/s,
    roughly 8 MB/day.
  - yaw and pitch come from `info["world"]`, which is GREEN and already read
    by the method (DL:10731).
- The per-stream segment summary (§5 D) is computed from **every** step, not
  from the samples, and handed to EM through `self._eval_seg`.

### 6.3 node1 ingest and Grafana

- **Incremental reads.** Replace the full re-read of each poll with a
  per-file `(inode, byte_offset)` checkpoint in a `_ingest_state` table.
  Keep `INSERT OR IGNORE` on `seq`, so the replay-rebuild contract still
  holds.
- **Long table for v2 scalars.** Add `segment_kv(seq INTEGER, key TEXT,
  value REAL, PRIMARY KEY(seq, key))`. Every numeric dotted key goes there
  automatically, so a new metric never needs a migration. Keep the wide
  `segments` table for the v1 panels. Nested maps stay JSON in `extra`, or in
  new JSON columns `rew_signed` and `breaks_seg`.
- **Trace table.** Add `trace(seq PRIMARY KEY, run_id, t, s, ep, wall_time,
  x, y, z, yaw, pitch, gui, ev)` with `INDEX(s, wall_time)`.
  - Retention: keep 7 days of raw rows on node1. Downsample older rows to one
    row per stream per minute in a `trace_1m` view.
  - The JSONL stays the system of record.
- **Collector.** Add `REMOTE_TRACE_PATH` to `collect_metrics.sh`, using the
  same `tail -F` + rsync pair.
- **New panels:**
  1. Signed reward by source, as stacked bars per segment, one panel per
     stream (positive above zero, costs below).
  2. Mixed PPO reward mean, against `rew.signed` total.
  3. Per-segment breaks and logs (not career), plus the `task.counter_check`
     tile, which should read 0.
  4. An XZ path scatter (last N hours, colour = time, one panel per stream)
     and a `y` timeseries.
  5. `expl.rg_xz`, `path_xz`, `steps_since_new_cell`, `cells_per_hour`, with
     `stuck_eval` and `infra/stuck` levels on the same axis.
  6. WM loss heads (block mean) and grad norm.
  7. PPO: approx_kl, clip_frac, explained_var, entropy as % of max, adv std,
     grad norms.
  8. LP gate pass rate and bucket count.
  9. Phase timing as a stacked area including `unaccounted`.
  10. Health tiles: NaN counters, sink drops, WM skip rate.

### 6.4 Tests to add with the change

- `_metrics_sink_smoke`:
  - schema v2 still has every v1 key;
  - the emission-site count is still 1;
  - `segment_kv` rebuilds identically.
- `_oracle_isolation_smoke` must pass unchanged: one `"oracle"` constant and
  the forbidden substrings absent.
- A new trace smoke:
  - the sink never raises;
  - a teleport is excluded from `path_xz`;
  - a closed square gives `net_disp` ≈ 0 and `straightness` ≈ 0;
  - samples are per stream.
- A new provenance smoke:
  - `Σ rew.signed` equals the ledger's `total`;
  - the new `ProvenanceBook` edits appear **twice** in DL (both bodies,
    §4.2);
  - `last_ledger_segment["shares"]` is unchanged by the new book (the
    `magnet_weight` gate's input is untouched).
- A counter-validation smoke: event-counted breaks equal the career delta
  across a simulated reset (CLAUDE.md §7).

---

## 7. The ten biggest gaps

1. **The reward PPO trains on is not logged.** The ledger drops signed totals
   per source (ledger.py:139). It omits icm_base, empowerment, approach and
   GUI dwell, and all the multipliers. It never sees the scout half of the
   PPO rows.
2. **Every break and log counter is career, stream 0 only.**
   `breaks_per_log` cannot show recent change, and scouts' breaks are
   invisible. No per-segment or per-run count exists, and nothing
   cross-checks one.
3. **There is no position trace and no stuckness measure from the
   evaluator.** The oracle is read for stream 0 only, its data is reduced to
   a drift mean, and that mean is never emitted. The adapter's `cells_seen`
   is reward-coupled and capped at 50k cells (ENV:1095), which can freeze
   `cells_delta` at 0.
4. **The PPO dict is computed and only printed** (`self._last_ppo`): value
   loss, approx_kl, adv stats, rows, clamp fractions.
5. **PPO clip fraction, explained variance, grad norms and LR are never
   computed.**
6. **WM telemetry is the last step of a block**, of unknown age. The
   `reward`, `continue` and `flow_smooth` heads are never recorded; flow,
   horizon and slot are recorded but not emitted; grad norm and latent
   entropies are missing.
7. **LP internals are invisible**: no gate pass/reject counts and no bucket
   count. The normaliser state is printed only.
8. **Behaviour counters are since process start**, not per segment
   (`action_hist`, `gui_frac`, ENV:394, 548). No per-segment action, GUI or
   idle histogram exists, and no action-source split outside the sampled
   shadow store.
9. **Timing, async-WM cadence, replay occupancy and replay ratio exist only
   in stdout** or not at all. The phase accumulators are reset before the
   sink could read them.
10. **Health is unmeasured.** NaN drops are uncounted (DL:7569), the sink
    never reports its own `dropped`, grad spikes and client rebuilds are not
    tracked, and ingest has a dead `curiosity` column while
    `reconstruction_error` sits in `extra`.
