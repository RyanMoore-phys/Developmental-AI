# Universal games: one SkyBot brain for every game

Status: **design + Stage U1 (Doom) implemented, not yet host-proven.**
Date: 2026-10-11. Owner decisions recorded in §2 are fixed; do not reopen them
without the user.

The thesis has always been *pixels, sound and buttons — meaning is earned*.
Minecraft was the first world, not the shape of the agent. This document makes
the loop game-shaped-agnostic: any game that can hand SkyBot what a human player
gets (screen, speakers, controller) can be added with a YAML file and, where no
backend exists yet, one adapter class. **One brain learns all of them at once.**

---

## 1. What "universal" means here

| A human player gets | SkyBot gets | Where |
|---|---|---|
| The screen | RGB frame, resized to the brain's one canonical `image_size` (128², 3 ch) | adapter → flat CHW float [0,1] |
| The speakers | Stereo mel spectrogram `(2, 32, 16)` per step, on the existing `audio` sensor | adapter → `info["audio"]` → sensor bus |
| A controller | **SkyBot's controller**: `universal.action_slots` (48) discrete slots | `UniversalControllerWrapper` |
| Their own body | The existing 13-field proprio (`MineRLEnvAdapter.PROPRIO_KEYS`) | adapter fills what the game has, neutral otherwise |
| A reason to play | Curiosity / learning progress — the reward systems already in the loop | unchanged |

What SkyBot does **not** get, in any game: memory reads, object lists, map
coordinates, score variables. Those are **RED** on the sensor bus — evaluation
only, structurally unreachable by learning (`sensors/registry.py`). This rules
out the reward definitions of stable-retro `data.json` and the Pokémon-RL
projects, which pay from RAM; we use their *emulators*, never their rewards.

## 2. Fixed decisions (user, 2026-10-11)

1. **One brain for all games. Non-negotiable.** No per-game world model,
   policy, curiosity or replay schema.
2. **Concurrent streams.** Each parallel stream plays a game; all streams feed
   the one shared world model and policy every step, so training batches always
   mix games (the strongest guard against forgetting one game while learning
   another).
3. **Rewards only as a player could earn them.** Curiosity-based, as today.
4. **Save-state checkpoints every 5 minutes, ring of 3**, with automatic
   rewind of the *game* when reward stagnation shows SkyBot is stuck.
5. **Fresh universal brain.** The Minecraft brain (`logs/checkpoints`,
   `skill_bank_mc_rssm`, `skill_bank_mc_curiosity`, `runlogs/*.json`) is
   archived **untouched**; the universal brain lives under `brain_universal/`.
6. **Doom first** (ViZDoom + Freedoom), then other games.
7. **ROMs stay out of the public repo**: `roms/<game>/`, gitignored. Free games
   are fetched by `scripts/fetch_roms.sh`; commercial ROMs are the user's own
   dumps and are never downloaded by tooling.
8. **A separate GitHub deploy action per game**, Minecraft included.

## 3. Architecture

```text
configs/minecraft_skybot.yaml   the tuned LEARNER (unchanged; still runs the legacy MC brain alone)
        +  configs/universal.yaml        brain contract: action_slots, image_size, sensors, brain_universal/ paths
        +  configs/games/<primary>.yaml  `primary:` overrides for whichever game holds stream 0
        +  configs/games/<g>.yaml × N    `streams:` — one entry per concurrent stream
        =  runlogs/run_config.yaml       written by tools/compose_run_config.py at every launch
```

Inside the loop, the only structural change is **per-stream construction**:
`parallel_envs.streams[i]` names the game and its env kwargs for slot `i`, and
all three build sites (`__init__` primary, `_ensure_parallel_envs` scouts,
`_make_one_env` rebuilds) go through one builder, `_build_stream_env(i)`.
Without `streams:` the legacy calls run unchanged (byte-identical), so
`minecraft_skybot.yaml` alone is still exactly the agent it was.

### 3.1 The brain contract (why every stream can share one brain)

The loop sizes the entire brain from **stream 0** (`obs_dim`, `action_dim`,
`PROPRIO_DIM`, sensor layout — loop ~377/496/383). Mixing games is safe only if
every stream presents the identical shape. `UniversalControllerWrapper`
guarantees it at construction and **refuses to start** otherwise:

| Quantity | Universal value | Enforced by |
|---|---|---|
| obs | `3 × 128 × 128` flat CHW float [0,1] | wrapper checks `observation_space.shape` |
| actions | `Discrete(universal.action_slots)` = 48 | wrapper pads; game mapping ≤ 48 |
| slot 0 | the game's **no-op**, in every game | game YAML validation; `curiosity.null_action: 0` relies on it |
| proprio | 13 fields, `PROPRIO_KEYS` order | adapters reuse `MineRLEnvAdapter._proprio` |
| sensors | one `sensors.enabled` list for the whole run | composer writes it once into `environment`, every stream inherits it |

48 slots: Minecraft uses 28, Doom 16, a Game Boy ~10, SNES/Genesis ~16. Growing
`action_slots` later changes the brain's shapes, so it is set generously now.

### 3.2 SkyBot's controller

A human uses the same hands on every game; the buttons mean different things in
each. SkyBot is the same: **slot k is a physical button on SkyBot's controller**,
and each game YAML maps slots to that game's inputs. The world model and policy
see the game through pixels and sound, so they can — and must — learn what each
slot does *in this game*. Nothing names a slot.

Two layers make an unmapped slot harmless:

1. **Safe by construction.** `UniversalControllerWrapper` executes slot 0 (the
   no-op) for any slot the game does not map. A missed mask costs one idle
   step, never a crash or a phantom input.
2. **Masked for efficiency.** The wrapper exposes `action_mask`; the option
   executor and the no-options select sites AND it into the existing
   per-decision mask (`options.py` `primitive_mask` / `mask`, stored per PPO
   row and re-applied in the update). The dream actor and ICM are not masked;
   by (1) that wastes a little imagination, nothing more.

### 3.3 Rewards

Unchanged in kind: curiosity (ICM + learning progress), the infra signals, and
whatever a game adapter returns as `r`. **Game adapters return `r = 0`.** A
player does not receive a number for walking forward; they receive new sights
and sounds, which is exactly what curiosity pays for. Minecraft keeps its
existing log economics (its adapter, its `primary:` block), because changing
the Minecraft reward is out of scope here.

Stream-0 shaping (gaze level, GUI dwell, persistence, coverage, …) reads
Minecraft `info` keys with defaults, so on a Doom stream 0 they are inert
(audited: every read is `.get(..., default)`). Scouts get `_scout_mixed_reward`.

### 3.4 Episodes, death and level progression (Doom)

* **Death** → `terminated=True`. The next `reset()` reloads the **current map**
  (what a player does after dying).
* **Level exit** → `terminated=True`; the next `reset()` starts the **next map**
  (MAP01 → MAP02 …, wrapping after the last). Progress is the player's.
* **Rewind** → `truncated=True`, `info["rewind"]=True` **and
  `info["env_restarted"]=True`**; the next `reset()` loads the oldest snapshot
  in the ring (§4). `env_restarted` is not optional: the lifelong body stores
  `done = terminated or restart` and deliberately ignores a plain truncation,
  so a rewind signalled by `truncated` alone would never reset the stream and
  the adapter would return "rewind pending" forever — a latch (§4.1).

All three are boundaries the loop already handles (`dones[e]` → `death_reset`,
bounded `reset()` through the env pool): **no transition is ever spliced
across them.**

### 3.5 Sound

ViZDoom exposes a stereo int16 buffer (`set_audio_buffer_enabled`). The adapter
converts the samples covering the last step into a log-mel spectrogram
`(2, 32, 16)` in [0,1] (left/right kept separate — direction is half the
information) and emits `info["audio"]`, which the existing GREEN `audio`
sensor already consumes. Minecraft emits nothing and the sensor reads neutral
zeros, exactly as it does today. Requires OpenAL at runtime (`libopenal1` on
the host; `brew install openal-soft` on a Mac). Without it the buffer is
silent — the adapter logs that once at boot rather than training deaf silently.

Enabling the audio sensor exposed a latent crash: `SensorImageEncoder` sized
its input from `h` alone, so any non-square image sensor (audio is 32×16)
failed on its first forward pass. Fixed in `rssm.py` (takes `h` and `w`).

## 4. Save-state checkpoints and automatic rewind

Implemented in `environments/rewind.py`, used by any adapter that can save and
load its emulator state (Doom now; PyBoy/stable-retro next). **Minecraft cannot
rewind**: MineRL cannot save a world (CLAUDE.md §7) and the Paper server is
shared with other agents.

**SnapshotRing.** Every `interval_s` (300) of wall clock, the adapter writes a
save state to `brain_universal/snapshots/stream_<i>/snap_<n>.<ext>` and keeps
the newest `keep` (3), deleting the oldest. Wall clock, not steps, because the
user's rule is time-based and step rates differ ~100× between games.

**StuckWatch.** Fed one number per step by the loop: the reward that stream
earned (`env r + intrinsic`). It keeps a long-run baseline (EMA, half-life
`baseline_halflife_s`) and a short window (`window_s`). The stream is **stuck**
when, after `warmup_s`, the short-window mean has stayed at or below
`ratio × baseline` for `window_s` seconds. Relative, because curiosity decays
as the world model learns — an absolute floor would eventually call every
stream stuck. **Detection latency is ~(2 − ratio) windows** after reward
collapses: the window *mean* must first fall to the line (~0.75 window of
zeros at ratio 0.25), then stay there one full window. Measured in
`tests/_rewind_smoke.py`: 107 s for a 60 s window — ≈ 5¼ min at the default
180 s, which is why the ring keeps 15 minutes.

**Rewind.** On stuck, the adapter ends the step with `truncated=True,
info["rewind"]=True, info["env_restarted"]=True` (§3.4), and `reset()` loads
the **oldest** snapshot (≥ 10 min
old with keep=3, i.e. before the slump began). **The brain is never rewound**:
it keeps everything it learned, including the experience of being stuck.

**Escape paths (CLAUDE.md §4.1 — this is a guard, so it must not latch):**

| Guard | What re-opens it |
|---|---|
| Rewind cooldown | Expires after `window_s`; the watch re-arms with a fresh window. |
| Repeated rewinds | After `max_rewinds_without_recovery` (3) rewinds with no recovery in between, the next one starts a **fresh episode** (current map, clean start) and clears the ring instead — a rewind loop cannot become a trap. A recovery (window mean back above the ratio for a full window) resets the count. |
| No snapshot yet | Rewind falls back to a fresh episode. |
| Snapshot load fails | Logged; falls back to a fresh episode. |

**What it pays while idle:** nothing. It writes no reward; it only moves the
game. A rewind is a boundary, so no potential crosses it.

## 5. Game YAML schema

`configs/games/<game>.yaml`:

```yaml
game:
  name: doom
  env_name: ViZDoom-Freedoom          # make_env dispatch key
  backend: vizdoom
  streams: 2                          # concurrent streams when this game is enabled
  rom: roms/doom/freedoom2.wad        # gitignored; adapter refuses to start if absent
  maps: [MAP01, MAP02, ...]
  render: RES_256X192                 # native; resized to the brain's image_size
  action_repeat: 4                    # tics per agent step
  audio: {enabled: true, sample_rate: 22050}
  controller:                         # slot -> inputs held for the step; slot 0 MUST be {}
    - {}
    - {MOVE_FORWARD: 1}
    ...
  checkpoint: {interval_s: 300, keep: 3}
  stuck: {window_s: 180, ratio: 0.25, warmup_s: 600, baseline_halflife_s: 1800,
          max_rewinds_without_recovery: 3}
  evaluator: [KILLCOUNT, ITEMCOUNT, SECRETCOUNT, HEALTH]   # RED: logged, never learned
primary:                              # deep-merged into the run config when this game is stream 0
  llm: {vision: {enabled: false}}
```

## 6. Operations

* **Enabled games** live on the host in `runlogs/games.enabled` (one name per
  line). `launch_skybot.sh` reads it: absent → the legacy Minecraft-only run,
  unchanged; present → compose `runlogs/run_config.yaml` for those games and
  run the universal brain. Stream order is fixed (`minecraft` first, then
  alphabetical) so slot 0 — the bare `SkyBot` player name on the Paper server —
  stays Minecraft's whenever Minecraft is enabled.
* **Minecraft-only preflight** (socat bridge, server ping, Xvfb, java) runs
  only when Minecraft is enabled.
* **Per-game GitHub actions**: `Game — Minecraft` and `Game — Doom`
  (`.github/workflows/game-*.yml`), each `workflow_dispatch` with
  `action: enable | disable | status` and a `restart` toggle. Enable/disable
  edits `games.enabled` and (optionally) restarts the one supervisor — there
  is still exactly one trainer and one brain. See `docs/OPERATIONS.md`.
* **Provisioning**: `provision_host.sh` STAGE 4d installs `libopenal1`,
  `vizdoom==1.3.1` and runs `fetch_roms.sh`. The Doom action also re-checks
  those idempotently, because provision is destructive and gated.
* **The process is still `run_minecraft.py`.** Seven stop/launch checks match
  that name; renaming it while a run is live would make every stop script miss
  the running process. It takes `--config`, so it is already game-agnostic;
  the rename is deferred to a quiet moment and must update all seven at once.
* **Backups**: `brain_universal/` is added to the node2 brain mirror
  (excluding its replay buffer, as for the legacy brain).

## 7. Throughput: the cost of lockstep

All streams step in lockstep through the env pool. A mixed fleet therefore
runs at the **slowest game's** pace: Doom alone measured ~4,800 steps/s
headless on the Mac, MineRL ~3–10 steps/s. Mixed with Minecraft, Doom streams
are throttled to Minecraft's rate — correct, but wasteful. A Doom-only run is
limited by the GPU's training cadence instead. Asynchronous per-game stepping
is the next infrastructure item if mixed fleets become the norm (§9).

RAM: a ViZDoom process measured ~31 MB. The 16 GB host's constraint stays the
MineRL clients (~1.2 GB each) and the trainer.

## 8. Stages

| Stage | Content | State |
|---|---|---|
| U0 | Design (this document), ROM layout, fetch script | done |
| U1 | Universal controller, per-stream construction, Doom adapter, audio, snapshots + rewind, composer, launcher, per-game actions, tests | **implemented; needs a live host run** |
| U2 | Live mixed run Minecraft+Doom on `main`; measure step rate, RAM, rewinds/hour, audio non-silence, prediction error per game | needs host |
| U3 | Game Boy via PyBoy (Pokémon — user-supplied ROM), Game Boy controller map, menu dwell audit (CLAUDE.md §4.4/§4.5 apply directly) | next |
| U4 | NES/SNES/Genesis via stable-retro emulator only (no `data.json` rewards) | later |
| U5 | Asynchronous per-game stepping | if U2 shows lockstep is the bottleneck |

## 10. Real time: the game does not wait, acting does not wait for learning

User decisions (2026-10-11): the game runs on its own timer; acting must be
independent of learning; SkyBot decides as fast as it can. Built in three
measured stages, universal-brain only (the legacy Minecraft run is unchanged).

**Why it was needed (measured on a Mac, Doom-only, CPU):** each decision took
4–15 s. The Doom engines were ~100% idle; the acting thread spent ~61% of its
time waiting on `_wm_param_lock`, held by the world-model trainer for every
gradient step (seconds on a CPU, and Python locks are not fair). Synchronous
Doom froze between decisions, so the viewer showed ~0.1 fps.

| Stage | What | Measured |
|---|---|---|
| **R1** game clock | `realtime: true` (Doom): a clock thread steps the engine one tic every 1/35 s with the HELD buttons; `step()` only sets buttons and returns the newest frame + how many tics passed. Every engine call (ticks, deaths, map changes, save/load) is on that thread. The viewer is fed by the clock itself. New GREEN sense **`tempo`** = game time since the last decision. | 34.9 fps, 0 late tics; 0.5 s of thinking = 18 tics in the next decision; fast brain = 1 tic per decision (35/s, never twice on a tic); rewind restores within 0.05 blocks |
| **R2** inference copy | `async_wm.inference_copy`: the trainer trains its own copy (`_wm_train`, with the optimizer) and publishes into `self.world_model` every `publish_every` (4) gradient steps, holding the lock only for the copy. | acting thread: 0 samples waiting on the lock (was ~61%); 400 decisions in ~30 s vs ~15 s per decision before |
| **R3** standby actor | Between segments (PPO update, dream training, reports, checkpoints) a standby thread plays every real-time stream with a frozen policy copy (refreshed when PPO lands) and the published world model; the loop takes the streams back with their newest frame and belief, and marks the replay gap. | `tests/_standby_actor_smoke.py`; live: handoff logged per boundary |

Consequences, stated plainly:

* **A slow decision is lag, not a pause.** The last buttons stay held while game
  time passes; `tempo` tells the world model how long.
* **SkyBot sees the newest frame when it decides,** not every frame between —
  the viewer shows all of them.
* **Standby play is not learned from.** It keeps SkyBot alive and moving; the
  gap is a sequence boundary in replay and resets the per-stream potentials
  (the same resets a death performs).
* **The standby actor plays primitives only** — option/skill execution needs
  learner-side bookkeeping. Within segments options work as before.
* **Still on the acting thread within a segment:** inference (encoder, RSSM
  step, policy), reward assembly, and the curiosity model's small per-step
  update (~3% of acting time on the Mac). Moving that last one off-thread
  changes how learning progress is measured, so it is left in place and named
  here rather than hidden.
* **Minecraft stays synchronous** — MineRL's client cannot run on a free clock.
* **Shutdown fixes found on the way:** the trainer now stops between gradient
  steps (it waited out 90 s and could exit inside torch: code 139); a Doom
  clock stops itself at interpreter exit (a crash path that skipped `close()`
  hung the process).

## 11. The Player Adapter — any game, the way a person plays it

User decisions (2026-10-13): one universal bridge for every game — the
unmodified game in its own virtual display, played through a screen,
speakers, keyboard and mouse. Proven on Doom first (Chocolate Doom, with the
ViZDoom stream as the known-good baseline), then a stock Minecraft client
(offline mode on the Paper server; the user owns a licence). Once host-proven
it **replaces MineRL**. Plan: `typed-petting-nautilus`.

```text
 SEAT i: Xvfb :(display_base+i) + PulseAudio null sink + the game process
     ▲ XTest keys/buttons/relative motion        │ MIT-SHM grab + parec
     │                                           ▼
 PlayerAdapter (real time, same contract as §10): HOLD slot a -> newest
 frame, the sound of that span, how much time passed (tempo); viewer fed by
 the capture thread at the game's own rate.
```

| Piece | File | What it guarantees |
|---|---|---|
| Controller | `environments/player/controller.py` | slot 0 releases everything; whitelist; OS keys + Alt+F-key / Ctrl+Alt refused; one-way doors refused (§4.5) |
| Held inputs | `environments/player/inputs.py` | exact press/release diffs; mouse = held VELOCITY integrated at 60 Hz (400 px/s is 400 px/s at any decision rate); release-all on every stop |
| Screen | `environments/player/screen.py` | capture at the game's rate on its own thread; newest frame for decisions; every frame to the viewer; frozen-screen detection |
| Audio | `environments/player/audio.py` | the seat's own sink, everything since the last decision |
| Seat | `environments/player/seat.py` | its own display (refuses one already in use), sink and process; total teardown, also at exit |
| Adapter | `environments/player/adapter.py` | real-time contract; dead/frozen game → restart (sliding hourly budget, no latch); stuck → quickload if the game has save keys, else restart |

**What a universal adapter does not have:** engine variables. No health,
position or episode signal — SkyBot reads its body off the HUD (proprio reads
neutral), play is continuous, and a respawn key must be in the controller.
Reward is 0.0; curiosity pays. The honest scoreboard for Minecraft moves to
the Paper server (RCON scoreboard, RED) at stage P3.

**Status:** P1 core built and tested with fake backends
(`tests/_player_adapter_smoke.py`, 34 contracts). **Unproven on the host.**
Stage P0 (`tools/player_spike.py`) must run on `main` first — it measures
grab rate (≥ 30 fps), key→pixel latency (≤ 50 ms), XTest relative mouse, and
audio — and decides whether XTest is enough or a per-seat uinput backend is
needed. `configs/games/doom_player.yaml` is the first game; its launch flags
are Chocolate Doom 3.x and are part of what P0 checks. Install with the
`game-doom_player` action (`action: install`) or provisioning STAGE 4e.

| Stage | State |
|---|---|
| P0 spike on main | tool ready; host offline |
| P1 core + Doom | built, fake-tested |
| P2 robustness on host | — |
| P3 stock Minecraft (portablemc offline, RCON scoreboard) | — |
| P4 replace MineRL | — (consequences listed in the plan: MineRL-only signals go inert) |

## 9. Known limits and open questions

* **Shared slots are deliberately meaningless across games.** Transfer between
  games happens through the latent (pixels/sound), not through slot identity.
  Whether one 48-way head is better than per-game disjoint ranges is an
  empirical question for U2; the masks make either layout possible later
  without changing `action_slots`.
* **Stream-0 shaping is still Minecraft-derived.** It is inert on other games,
  not wrong — but other games get only curiosity on stream 0. That is the
  intended reward; it is also the least-tested reward.
* **VLM symbolizer / vision scaffold** prompts are Minecraft-specific and stay
  off when Minecraft is not stream 0 (Doom's `primary:` block). Both are off in
  the live config already.
* **Lockstep** (§7).
* **The scoreboard.** None of this moves CLAUDE.md §9 by itself. What it does
  give: a second world, ~1000× faster, in which the same learner either learns
  something or visibly does not — the fastest available evidence about whether
  the log problem is the learner or MineRL.
