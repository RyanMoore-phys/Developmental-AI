# CLAUDE.md — working notes for agents on this repo

Read this before touching anything. `README.md` tells you where files are; this
tells you how to work here and what has already gone wrong. Nearly every rule
below was paid for by a multi-day run that produced nothing.

---

## 1. What this is

A curiosity-driven neurosymbolic RL agent ("SkyBot") that learns Minecraft
**from scratch — no demos, no videos, no recipes**. The stack:

| Piece | Where | Role |
|---|---|---|
| DreamerV3-style RSSM world model | `world_model/rssm.py` | predicts latents; the dream substrate |
| PPO actor-critic | `policy/actor_critic.py` | the shared policy all skills copy from |
| Options / SMDP | `policy/options.py` | skills as temporally-extended actions |
| ICM + learning-progress curiosity | `curiosity/` | the intrinsic drive |
| Vision magnet + fovea | `llm/vision_scaffold.py` | curiosity-ranked visual seeking |
| Local VLM (qwen2.5vl:7b via Ollama) | `llm/vlm_symbolizer.py` | symbol grounding, unstuck advice |
| Knowledge graph | `knowledge_graph/` | asserted/retracted facts |
| General infrastructure | `infra/` (wired by `infra/stack.py`) | gates, ledger, signal health, empowerment, episodic, affordance, advisor |
| The loop that owns all of it | `core/developmental_loop.py` (**9.4k lines**) | reward assembly, stepping, training |

**The standing principle:** meaning and skill are *earned from experience, never
declared*. The agent gets pixels and buttons and has to find out what they do.
This is enforced by `tests/_no_scripted_skills_smoke.py`, which fails the build
if a scripted option reappears in the config. If you think you need a scripted
macro — you don't; see §4.1.

---

## 2. Running things

**Always use the repo venv.** System `python3` lacks `yaml`, `torch`, everything.

```bash
cd "/Users/rimac/Desktop/Developmental AI"
./venv/bin/python tests/_gui_farm_smoke.py        # Python 3.9.6
```

Tests are **standalone `__main__` scripts**, not pytest. Each prints numbered
contract lines and ends with `[name] ALL PASS`. Since 2026-09-19 there IS a
runner, `tests/run_all.py`, in three tiers — but it only shells out to those
same scripts, so running one directly still works and is still the right move
when you are iterating:

```bash
PYTHONPATH=. python tests/run_all.py unit       # parts: shapes, bounds, ~3s
PYTHONPATH=. python tests/run_all.py contract   # design arguments, ~10s
PYTHONPATH=. python tests/run_all.py legacy     # needs gymnasium -> pod only
PYTHONPATH=. python tests/run_all.py all        # exit code = failing suites
```

**The two tiers mean different things.** A `unit` failure is a bug in a
function. A `contract` failure means a DESIGN CLAIM stopped holding — those
docstrings name the live incident and the measured numbers, and one going red
is a finding, not a chore. A `.pth` file in the venv makes `import developmental_ai`
work from any cwd, but prefix `PYTHONPATH=.` anyway — that's the documented form
in every test docstring, and it's what works on the pod.

Training does **not** run on the Mac. It runs on a rented GPU pod
(`scripts/launch_skybot.sh`, kept alive by `scripts/supervise_skybot.sh`).
See `host_repository/docs/RUNNING.md`.

### Stopping a run

`touch runlogs/STOP` — graceful, and the supervisor won't restart it. Do **not**
`kill` the python process: the supervisor treats that as a crash and relaunches.
Also note `pkill -f "tailscale nc <ip>"` will kill the **socat bridge** too
(socat's cmdline contains that string) — this has stranded a run before.

---

## 3. Standing user instructions

- **No git commits, no branches, no deployment ceremony** unless explicitly
  asked. Work and edit in place. (The repo has exactly one commit by design.)
- **Only commit pod-proven changes** — if it hasn't run live, it isn't proven.
- **Never wipe `skill_bank_mc_curiosity/`** on the pod. It is the agent's
  accumulated developmental memory.
- Peer learning between SkyBots is a *feature*, not contamination. Multiple
  agents on one server is a deliberate social-learning experiment.

---

## 4. The recurring bug classes

These have each bitten multiple times. Check for them **by name** in review.

### 4.1 Guard-becomes-latch (≥9 occurrences)

A guard added to stop a bad behaviour becomes a permanent trap because nothing
can ever satisfy it again.

- The competence gate froze at an inherited bias of −5.68 → **all 16 skills, 0
  invocations, forever**.
- `disable_macros: [10]` removed `inventory` — the *only* GUI exit — while
  `use` (which opens villager/chest screens) stayed enabled. SkyBot sat in a
  wandering villager's trade menu for **10,149 consecutive steps**.
- The magnet's cold-start BUDGET latch caused **19 hours of zero reward**.
- The degenerate-signal gate counted *labels* instead of *positives* and
  disabled the magnet live.

**Rule:** every guard needs an escape path that the agent can actually reach,
and you must state what re-opens it. If the answer is "a human notices," it's a
latch.

### 4.2 Duplicated-body drift

`developmental_loop.py` steps the environment in **two places**:

- `_run_episode_parallel` (~line 3382)
- `_collect_segment` (~line 4282) ← this is the one SkyBot actually runs

Reward logic is duplicated between them. An edit that lands in one and not the
other is silent — it just doesn't apply live. Use `replace_all: true`, then
**grep to confirm the count is 2**. Several tests assert exactly this, e.g.
`assert src.count("_gui_now2 = bool(") == 2`.

**There is a THIRD body, and it is unwired** (found 2026-09-22): `_run_episode`
(~line 3645), the single-env path. It has **none** of Waves 1–2 or Phases 1–7 —
no `info["sensors"]`, no `_augment_proprio`, no `_wm_proprio_batch`, no
`_flow_senses`, no `_spatial_step`, no `_phase_mark`. Reaching it is easy and
silent: `parallel_envs.num_envs <= 1` sets `_use_parallel_envs = False`, which
raises if `lifelong.enabled` is true and otherwise drops straight into it.
**So `num_envs: 1` is not "the same agent with one client"** — it is the
pre-Wave-1 agent. Keep `num_envs: 2`, or port the wiring first.

### 4.3 The γ-discounting defect in state costs

A potential written the textbook way, `F = w(γΦ′ − Φ)`, pays **`−w(1−γ)Φ` every
step Φ is held constant**. For a *cost* potential (Φ < 0) that flips the sign:
sitting still becomes a **wage**.

This shipped once and the log showed it: `Gaze level: +0.00014/step` — the agent
was being *paid* to stare at the pitch clamp.

**Rule:** telescoping potentials are for *progress*. A **state cost** must use
the **plain difference** (`γ = 1`), so it charges on entry, refunds on exit, and
pays exactly **0** while pinned. `tests/_gui_farm_smoke.py` encodes both forms —
the flawed one is kept as a regression witness.

Corollary: a telescoping potential **structurally cannot discourage dwelling**.
If you want dwelling to hurt, you need a genuine per-step cost, not a potential.

### 4.4 Reward-channel confusion

`intrinsic` is **zeroed while a GUI is open**. Any penalty you put there is
erased exactly when you need it. GUI/dwell costs must land in `prim_extrinsic`,
*before* the zeroing line. Meanwhile the magnet's shaping was added to
`prim_extrinsic` and so kept paying through an occluded camera — that leak was
77% of all income during the villager incident.

**Before adding any reward term, answer:** which channel, is it gated, does it
survive occlusion, and what does it pay while the agent does nothing?

### 4.5 One-way doors in the action space

Never offer an action that *opens* a state without the action that *closes* it.
Enforced by `test_no_one_way_doors` in `_gui_farm_smoke.py`.

### 4.6 Skills are copies, not memories

A minted skill is a **byte copy of the shared policy** (measured cosine
0.869–1.000; three pairs bit-identical; slot signal attenuated 20×). So:

- **Never key effects on a skill's `name`** — the name carries no weight.
- Merging the skill bank **does not stick**; skills re-derive from goal
  `slot_keys`.
- "Mastery" is currently unmeasurable (3 of its 4 ingredients don't exist).

---

## 5. Debugging discipline

**Measure before theorizing.** This project has burned days on plausible
theories. Two examples worth internalizing:

- The 19-hour zero-reward stall "was curiosity exhaustion." It was a cold-start
  budget latch.
- The `places` counter looked like it proved wasted no-ops. It lists
  `iron_axe: 2281` and `acacia_door: 3004` — the heuristic is **garbage**. Don't
  build an argument on a counter you haven't validated.

**Falsify your own fix.** The gaze-level potential above was caught by a test
written for a *different* bug. Write the test that would catch you being wrong,
not the one that confirms you're right.

**When you find a real failure, encode it as a test.** The convention here is a
long docstring naming the live incident, the measured numbers, and the contracts
— see `_gui_farm_smoke.py` and `_no_scripted_skills_smoke.py`. That's why
reversing a principle requires deleting a test that explains itself, rather than
quietly appending four lines of YAML.

**Don't trust `STATE.md`.** It has been stale and sent a session chasing a
terminated pod while the live one ran elsewhere. Verify against the running
system.

---

## 6. Live infrastructure

- **Training host (since 2026-09-22): the user's own main computer**, not a
  rented pod. Ubuntu Server at **192.168.1.10**, on the **LAN and the tailnet
  at once**. RTX 5050 (8 GB), Ryzen 5 5500 (6c/12t), **16 GB RAM**, 256 GB NVMe
  — far tighter than the pod it replaced (16 GB VRAM, 128 cores, 125 GB RAM),
  and several config values were cut to fit (see below). Everything still
  installs at `/workspace/devai` and deploys as `root@`, deliberately: the box
  is shaped like a pod so almost nothing needed changing. **One thing did, and
  it cost a failed deploy (2026-09-22): `/workspace` itself.** On RunPod that
  was the platform-mounted network volume, present before any script ran —
  nothing in this repo has ever created it, and `provision_host.sh` opens with
  `cd /workspace/devai || exit 1`, so it cannot be what makes it either. On an
  owned box nobody mounts it, and `rsync` creates only the LAST path component,
  so the deploy died on `mkdir "/workspace/devai" failed: No such file or
  directory (2)` — **exit code 11, which reads like a permissions problem and
  is not one.** `deploy_skybot.sh` now creates it and prints the filesystem and
  free space, because a silently-created `/workspace` on the root disk is also
  how you would learn far too late that a data volume failed to mount.
  The old vast.ai pod is off. IPs in `host_repository/docs/RUNNING.md` are
  **stale** (older RunPod box).
- **Everything targets the host via the runner's `.env`**, never a value in the
  tree: `MAIN_HOST`, `MAIN_SSH_PORT`, `MAIN_SSH_KEYFILE`, `MAIN_USER`,
  `MC_SERVER_TS_IP`. To
  move hosts again, edit `.env` — not the repo. See `docs/CI_SETUP.md`.
  **These were `POD_*` until 2026-09-22** (138 occurrences across 14 files).
  Renamed because the name had stopped describing the thing: there is no pod,
  the target is the owned box `main`, and the port no longer rotates. There is
  deliberately **no `POD_*` fallback** — every consumer guards with `:?` and
  names the missing variable, so a stale `.env` fails loudly at the preflight
  instead of expanding to `root@` and reporting `Could not resolve hostname`.
- **`MAIN_USER` (added 2026-09-22) — the login account, was hardcoded `root`.**
  RunPod only ever gave you root; `main` is a normal Ubuntu box, so it is
  `skybot`. Default stays `root`, so a root host is unchanged. **A non-root
  `MAIN_USER` REQUIRES PASSWORDLESS SUDO on the target** — provisioning installs
  apt packages and creates `/workspace` under `/`. `provision_host.sh` now routes
  every privileged command through `$SUDO` (empty when already root) and aborts
  at the top with a `sudoers.d` recipe if `sudo -n true` fails; it runs detached
  under `nohup` with no tty, so a password prompt would not block visibly — it
  would just fail stages silently amid `set -x` noise.
- **Keep `MAIN_HOST` an IP, not the name `main`.** The runner on `node1` is
  **dockerized**, and `docker/runner/docker-compose.yml` sets no `extra_hosts`
  and no `dns`. A container does not inherit `node1`'s `/etc/hosts`, and mDNS
  (`main.local`) needs an avahi client it does not have — so a name that
  resolves perfectly in your shell fails inside the container as `Could not
  resolve hostname`. Use `skybot@main` freely in hand-typed commands; leave
  the `.env` on the address. To use the name there too, add
  `extra_hosts: ["main:192.168.1.10"]` to the runner service first.
- **`podlogs/` is now `runlogs/`, and every `pod_*` file is `host_*` (2026-09-22).**
  267 references across 61 files; `pod.yml` became `host.yml` (safe — `deploy.yml`
  triggers on `workflows: ["ci"]`, never on this one). **THE TRAP IS BRAIN STATE,
  NOT CODE.** `break_memory_path` lives under that directory, so any backup taken
  before this rename carries `podlogs/` paths: restoring one onto a renamed tree
  silently starts the break memory EMPTY rather than erroring, and the agent
  re-opens every mastered block tier at full worth. On restore, `mv podlogs
  runlogs` FIRST. This was safe to do now only because `main` was unprovisioned
  and the old pod was already off — there was no live state to orphan.
- **The runner lives on `node1`, a DIFFERENT machine from the training host.**
  `node1` runs the self-hosted Actions runner and holds the only `.env`; `main`
  (192.168.1.10) is the box everything deploys to. Its `hostname` really does
  print `main`, which is the quickest way to tell which one you are sitting on.
- **Game server:** the user's own Paper server, on **another LAN machine**,
  reached through `socat` → `127.0.0.1:25565` exactly as before. The bridge now
  has **two far sides** and picks by address family: RFC1918 goes over plain
  TCP on the LAN, everything else (tailnet 100.x, MagicDNS names) keeps
  `tailscale nc` untouched. **Tailscale is fully retained** — it is the path
  again as soon as ethernet is back, via one `.env` value. The endpoint the
  agent sees is `127.0.0.1:25565` either way, which is why
  `environment.remote_server` never changes. The *server* is the client-count
  bottleneck, not the host: 4 MineRL clients produced 0 segments in 17 minutes.
  2 is the proven number, and `remote_server_scope: all` puts **both** streams
  on the Paper server (no stream runs a local generated world).
- **What the 16 GB box forced (2026-09-22).** `buffer_growth` was capped at
  **2 GB** (`hard_max_gb` 16 → 2, `max_ram_frac` 0.6 → 0.15) and
  `block_transitions` 25000 → **5000**. That last one is not cosmetic: the
  **initial block is allocated per stream WITHOUT a budget request** —
  `ReplayBuffer.__init__` allocates outright and only `maybe_grow()` calls
  `MEMORY_BUDGET.request()` — so real RSS exceeds `hard_max_gb` by
  `num_streams × block_transitions` always. At 25k blocks a "2 GB" ceiling
  measured **3.70 GB**. Revisit `block_transitions` whenever `hard_max_gb`
  moves. `OMP_NUM_THREADS` is 6, not 16.
- **Ollama** serves the VLM on the same GPU. `fovea_interval: 12` cost **28% of
  the step rate**; 30 is the tuned value. VLM frequency trades directly against
  training throughput.
- **Backups:** brain state (`world_model.pt`, `symbolizer.pt`, `familiarity.pt`,
  `magnet.pt`, `knowledge_graph.json`, `options_state.json`, break memory, skill
  bank) lives **only on the training host**. A pod died with ~11 days of
  unbacked state. Moving to owned hardware removes the *pod-terminated* risk
  and replaces it with *disk failure* — 256 GB, one NVMe, no redundancy — so
  the habit still stands: **rsync the brain host→Mac periodically during long
  runs.** Code always flows Mac→host, so code is never at risk.
  **`pull_brain.sh` does not exist** despite `deploy_host.sh`'s header citing
  it; nothing automates this yet.

---

## 7. MineRL / environment gotchas

- `mine_block` stats **restart at 0 on every reset** — a new world per mission.
  Use per-episode marks; never baseline against a running total.
- `FlatInventoryObservation` gives item→count with **no slot indices**. There is
  no equip action and `equipped_items` is dead in MineRL 1.0 — mainhand must be
  derived from evidence (see `_last_placed_item` in `minerl_env.py`).
- World persistence is **impossible in this fork** (measured). Each episode is a
  fresh world; anything that must persist has to live in the brain, not the map.
- Multi-agent usernames go through `handlers.MultiplayerUsername` on a
  **per-instance** spec. Slot 0 keeps the bare name `SkyBot` (the offline UUID
  owns that player's data).
- Boot hangs are solved by **VirtualGL EGL + per-core `taskset`** (an NVIDIA/AMD
  memcpy race); this is baked into `provision_host.sh`.

---

## 8. Where to look

| Question | File |
|---|---|
| Why isn't it acting? | `docs/ACTION_STALL_AUDIT.md` — 30+ ranked issues |
| Domain-independent redesign | `docs/GENERAL_INFRASTRUCTURE.md` — 57 changes |
| Open findings | `AUDIT_FINDINGS.md` |
| Depth/flow from motion | `docs/PERSPECTIVE_LEARNING.md` — waves 1-2 |
| Sensors, slots, action space | `docs/PLURALITY_ROADMAP.md` — phases 1-7 |
| How any of it gets proven | `docs/TESTING_PLAN.md` — stages 0-5 |
| What's next | `NEXT_OBJECTIVES.md`, `ROADMAP.md` |
| Pod ops | `host_repository/docs/{RUNNING,PROVISIONING,TRANSFER,RECREATE}.md` |
| Live config | `configs/minecraft_skybot.yaml` (the only one that matters) |

---

## 9. The honest scoreboard

Keep this in view; it's the point of the project.

- **399 blocks broken in pod history → exactly 1 log, ever.** Later runs reached
  13,305 breaks with **35 logs**; all log skills sit at **0/20**.
- Chopping a tree — the original objective — **has never been learned**.
- Most historical "progress" was the agent finding a way to get paid for doing
  nothing: staring at the sky (96% of drive), sitting in a menu (77% of income),
  holding attack against an unreachable trunk (96% of option activity, 0 logs).

When you evaluate a change, ask what it does to *that* scoreboard — not to the
reward number. The reward number has been wrong every single time.
