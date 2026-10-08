# Telemetry inventory and ops design: crash bundles, off-box evidence, "main went silent"

Written 2026-10-07 after the 2026-10-06 incident. This is a read-only inventory
plus a design. **Nothing in the Design section is implemented yet.** Each
inventory claim cites the file and line it comes from. Claims I could not
verify from the repo are marked **UNVERIFIED** and listed in section 3.

Hosts are named, never addressed: `main` (training host), `node1` (runner +
monitor stack), `node2` (brain mirror), `paper-server`.

## 0. The incident this answers

- 2026-10-05: both runs ended in `terminate called without an active exception`
  at exit. That is a C++ abort, so SIGABRT (`run_minecraft.py` `_close_bounded`
  docstring, uncommitted diff).
- 2026-10-06 03:07:58 UTC: `main` stopped logging mid-stream, 22 s after the
  agent aborted at exit (`scripts/host_hardening.sh:4-6`). This followed a
  *clean* `runlogs/STOP`.
- No OOM kill, panic or core dump was recorded (`scripts/host_hardening.sh:25-29`).
  Memory was near-full with swap climbing (`:37-41`).
- After ~41 h a human powered the host down by hand.

Everything `main` knew about the freeze stayed on `main`, and `main` was the
thing that froze. The question this doc answers: **what evidence exists off
the box, and what is the smallest change that makes the next freeze
explainable?**

---

## 1. Inventory: what exists today

### 1.1 How crashes are recorded on `main`

| Artifact | Writer | What it captures | When |
|---|---|---|---|
| `runlogs/supervisor.log` | the supervisor's own stdout (`scripts/supervise_skybot.sh:27-28`) | start time and budget (`:39`); each `launching` (`:60`); the launcher's full output, echoed (`:63`); refusals (`:70`); `watching pid=` (`:80`); clean stop (`:86`); `same fault 3x` (`:110-113`); restart and backoff (`:115`) | continuous |
| `runlogs/crashes.log` | supervisor, only on an exit **without** `runlogs/STOP` (`:85-97`) | an `=== agent exited <date> ===` header. Then `tail -30` of the run log, filtered through `grep -A22 Traceback \| head -30` (`:94-95`) | after the exit is noticed (see the 20 s poll below) |
| `runlogs/minecraft_skybot_run.log` (+ `.1`..`.3`) | agent stdout/stderr, unbuffered (`scripts/launch_skybot.sh:128-133`); rotated on each launch (`:122-127`) | everything the agent prints | continuous |
| `runlogs/skybot_run.pid` | launcher (`launch_skybot.sh:136`) | PID only | at launch |
| `host.yml` `status` action | human-triggered (`.github/workflows/host.yml:118-146`) | `tail -12 crashes.log`; `dmesg` OOM grep (`tail -5`); top RSS consumers (`ps --sort=-rss \| head -8`); `tail -15` run log; `tail -25` supervisor.log | on demand, **only while main is up** |

Weaknesses, all visible in the code:

- **No exit code and no signal are ever recorded.** The agent is started with
  `setsid ... &` and then `disown` (`launch_skybot.sh:130-135`), so the
  supervisor is not its parent and cannot `wait` for it. All it can see is
  `kill -0` stop succeeding (`supervise_skybot.sh:81-83`). It cannot tell
  SIGABRT, SIGKILL (the OOM killer or earlyoom), SIGSEGV and a normal exit 0
  apart.
- **A clean STOP hides an abnormal exit.** If `runlogs/STOP` is present, the
  supervisor prints `clean stop` and exits 0 without writing anything
  (`supervise_skybot.sh:85-88`). The 2026-10-05/06 aborts all happened under a
  STOP, so **crashes.log has no entry for the exit that preceded the freeze**.
- **A crash without a Python traceback records only the header line.**
  SIGKILL, SIGABRT from C++ and segfaults print no `Traceback`, so the
  `grep -A22 Traceback` at `:95` keeps nothing.
- **The deterministic-fault stop never fires for non-Python deaths.** `SIG` is
  the first `*Error|*Exception` line (`:101-102`). The repeat test needs it to
  be non-empty (`:103`). With an empty `SIG`, `SAME` resets to 1 every time
  (`:105-106`). Three OOM kills in a row therefore loop forever, backing off
  to a 300 s cap (`:33-34`, `:116-118`).
- **Exit time has 20 s resolution.** The watch loop polls every 20 s
  (`:81-83`).
- **`supervisor.log` is truncated on every launch.** Both launch paths use `>`:
  `host.yml:169` and `scripts/deploy_skybot.sh:216`. The previous
  supervision's history is erased exactly when someone relaunches after an
  incident.
- **No host state is captured at exit**: no free memory, swap, PSI, GPU,
  kernel log or disk. The only snapshot of host state is the interactive
  `status` action, which needs a live host and a human.
- **Core dumps are off for the agent** (`launch_skybot.sh:109-115`, uncommitted).
  This is the right call on a 16 GB / one-NVMe box, but it removes the last
  post-mortem artifact for a native crash. The exit code is now the only
  record of *which* signal killed it, and nothing records it.

### 1.2 What a host freeze destroys or strands

The table below covers a 2026-10-06-style freeze: the kernel is alive but
userspace and the disk are starved, and the box ends in a manual power-off.

| Evidence | Lives on | Survives the freeze? | Off-box copy today |
|---|---|---|---|
| run log, last seconds to minutes | `main` NVMe | partly. Unsynced page cache is lost at power-off | **none** |
| `supervisor.log`, `crashes.log` | `main` | yes (if not truncated by the next launch, 1.1) | **none** |
| `metrics.jsonl` (per segment, fsync'd) | `main` | yes | **yes**: live `tail -F` plus a 300 s rsync to node1 (1.3) |
| `heartbeat.jsonl` (~15 s, **not** fsync'd: `configs/minecraft_skybot.yaml:3055-3060`) | `main` | last seconds lost | **yes**: rsync every ~30-40 s to node1 |
| `memory_census.jsonl` (every 1800 s: `minecraft_skybot.yaml:3072-3074`) | `main` | yes | **none** |
| `runlogs/foundation_shadow/` | `main` | yes | only if `INCLUDE_SHADOW=1` on node2 (default `0`: `scripts/brain_mirror/brain_mirror.py:137`, `brain_mirror.env.example:35`) |
| journald (kernel, sshd, earlyoom) | `main` NVMe | partly. journald syncs non-critical messages only every `SyncIntervalSec` (default 5 min), and a stalled disk syncs nothing | **none** |
| kernel ring buffer (`dmesg`) | RAM | **no**. Gone at power-off | **none** |
| pstore / kdump vmcore | `main` | only on a **panic**. A thrash does not panic | none |
| host vitals (RAM, swap, PSI, vmstat, disk, NVMe temp, GPU) | — | — | **yes**: Prometheus on node1 scrapes `main` every 10 s, up to the last scrape that answered (1.3) |
| brain checkpoints | `main` | yes | **yes**: node2 every 5 min (1.4) |

**Conclusion: node1 already holds the host-metric trajectory up to the freeze.**
What it lacks is every *log*: the supervisor's, the crash log, the run-log
tail, and the kernel's.

### 1.3 What node1 already collects (`docker/monitor/`)

**Collector** (`collect_metrics.sh`, container `skybot-collector`,
`docker-compose.yml:28-51`):

- Logs into `main` as `MAIN_USER` with `BatchMode`, `ConnectTimeout=20` and
  `ServerAlive 15x4` (`collect_metrics.sh:43-46`).
- Pulls **only two files**:
  - `runlogs/metrics.jsonl`: a live `ssh tail -F -n 0` (`:120-121`) plus an
    rsync backfill every `RSYNC_EVERY=300` s (`:33`, `:134-137`), deduplicated
    on `seq` (`:75-80`).
  - `runlogs/heartbeat.jsonl`: rsync every `HB_EVERY=30` s (`:36`, `:128-133`),
    checked on a 10 s loop (`:126`), so up to ~40 s old.
- When the tail drops, it backfills and reconnects every 10 s (`:140-145`).
- It warns (rate-limited to once per 300 s) when the metrics file cannot be
  fetched (`:64-70`). The heartbeat pull fails **silently** (`:97`,
  `|| return 0`).
- It does **not** pull `supervisor.log`, `crashes.log`, the run log,
  `memory_census.jsonl`, the journal or dmesg.

**server_poll** (`server_poll.py`, `docker-compose.yml:54-67`) writes
`server.jsonl` every `POLL_SECONDS=30` (`monitor.env.example:37`). It is an
independent players-online falsifier (`docs/MONITORING.md:76-90`).

**ingest.py → `/data/skybot.db`** (WAL mode, `ingest.py:72-75`) has three
tables:

- `segments` (`:77-79`): `seq` PK, `schema_version`, about 38 scalar columns
  (`:40-63`), JSON columns `reward_shares`, `reward_alarms`, `breaks_by_type`
  and `crafts_by_type` (`:64-65`), and `extra`.
- `server` (`:81-83`): `wall_time` PK, `players_online`, `max_players`,
  `version`.
- `heartbeat` (`:89-93`): `seq` PK, `wall_time`, `total_timesteps`,
  `uptime_s`, `steps_per_s`, `breaks_total`, `logs`, `attack_run`, `episodes`,
  `gpu_mem_mb`, `income_now`.

Migration is add-only (`:96-127`). **There is no table for incidents, host
state or exits.**

**Prometheus** (`docker-compose.yml:158-175`, `prometheus.yml`):

- `scrape_interval: 10s` (`prometheus.yml:8`).
- Job `node` (`:11-31`): `node-exporter:9100` (node1 itself), `node2:9100`,
  `paper-server:9100`, `main:9100`.
- Job `gpu` (`:39-42`): `main:9835` (`nvidia_gpu_exporter`).
- On `main`, both exporters are systemd binaries, LAN-bound, with
  `CPUQuota=10%` and `MemoryMax=128M` (`scripts/install_host_exporters.sh:66-89`).
  `node_exporter` v1.12.1 runs with **no extra flags** (`:93-95`), so it has
  **no textfile collector on main**.
- **Retention: the default, 15 days.** The compose service sets no `command:`
  and no `--storage.tsdb.retention.*` flag (`docker-compose.yml:158-175`). Data
  from 2026-10-06 ages out around **2026-10-21**.
- **No `rule_files`, no Alertmanager.**

**Grafana** (`docker-compose.yml:89-108`, `grafana:11.1.0`):

- Datasources: `skybot-sqlite` (default, `provisioning/datasources/sqlite.yml`)
  and `rack-prometheus` (`provisioning/datasources/prometheus.yml`).
- Provider: file, `allowUiUpdates: true` (`provisioning/dashboards/dashboards.yml`).
- **One provisioned dashboard**, `skybot.json` ("SkyBot — live", uid
  `skybot-live`, refresh 30 s, last 24 h). It has 13 panels, **all on the
  SQLite datasource**:

  | id | type | title (`skybot.json` line) |
  |---|---|---|
  | 20 | timeseries | LIVE (~15s): steps/s, breaks, logs (:14) |
  | 21 | stat | Steps/s (live) (:40) |
  | 1 | stat | BREAKS PER LOG (the scoreboard — history is 399:1) (:62) |
  | 2 | stat | Reward concentration (HHI) (:106) |
  | 3 | stat | Players online (expect num_envs) (:152) |
  | 4 | timeseries | Where the reward comes from (live, ~15s) (:175) |
  | 5 | timeseries | Breaks vs logs (cumulative) (:201) |
  | 6 | table | Blocks broken by type (latest) (:226) |
  | 7 | table | Crafts by item (latest) (:248) |
  | 8 | timeseries | Skills: minted vs bound vs refused (:270) |
  | 9 | timeseries | Learning signals (entropy, WM loss, KL, intrinsic) (:296) |
  | 10 | timeseries | Reward + attack behaviour (:322) |
  | 40 | timeseries | Uptime — this run, and lifetime steps (:348) |

- **No host-metrics dashboard is committed**, although the Prometheus
  datasource is. Any host panels were built in the UI and live only in the
  `grafana_data` volume (**UNVERIFIED**).
- **No alerting provisioning exists**: there is no
  `provisioning/alerting/` directory.

### 1.4 node2 (`scripts/brain_mirror/`)

- A user systemd timer runs every 5 min (`install_node2.sh:101`).
- It pulls checkpoints, break, consequence and anticipation state, and the
  skill banks (`brain_mirror.py:111-125`, README :9-22).
- It explicitly does not copy "logs, metrics" (README :20-22).
- It reads from `main` at `nice 19 ionice -c3` (`brain_mirror.py:139`).
- When `main` is down it exits 11 each run (README :104-105). That makes node2
  a second, independent liveness witness, but **nothing alerts on it**.

### 1.5 `main`'s own black boxes

- **journald:** nothing in the repo configures it. `host_hardening.sh:4-5` reads
  the previous boot's journal ("ordinary sshd lines, then nothing"), so it is
  persistent in practice (the Ubuntu default with `/var/log/journal`). It sits
  on the same NVMe, and `SyncIntervalSec` is presumably the default 5 min
  (**UNVERIFIED**). The final minutes are exactly what it does not guarantee.
- **kdump** is installed and active, and saves a vmcore under `/var/crash` on
  a panic (`host_hardening.sh:27`, `:79`). It saved nothing on 2026-10-06
  because nothing panicked.
- **pstore**, **panic→reboot**, the **hardware watchdog**, **printk 7**,
  optional **hung_task_panic**, **NVMe APST off** and **earlyoom** are all
  offered by `scripts/host_hardening.sh` (`:71-119`). It is dry-run by default
  (`:47-52`), so whether any of it has been applied is **UNVERIFIED**. earlyoom
  logs to the journal (`:45`), which is on the same disk.
- **Core dumps** are off for the agent (`launch_skybot.sh:115`).

### 1.6 Gaps, ranked

1. **G1. No exit code or signal, and a clean STOP suppresses all crash
   recording.** The very exit before the freeze left no record (1.1).
2. **G2. No log ever leaves `main`.** supervisor.log, crashes.log, the
   run-log tail, memory_census and the journal are all strandable (1.2).
3. **G3. No kernel-message channel survives a disk or userspace stall.**
   dmesg dies with RAM. The journal and pstore need a disk or a panic.
4. **G4. No alerting at all.** No rules, no contact point. A 41 h silence was
   ended by a human noticing, which is the CLAUDE.md §4.1 definition of a latch.
5. **G5. Host state at exit is never snapshotted.**
6. **G6. Prometheus retention is 15 d.** The only off-box trajectory of the
   2026-10-06 memory climb expires around 2026-10-21.
7. **G7. No per-process RSS (agent / java / ollama), SMART media errors or
   per-process GPU memory** in Prometheus (section 2.3).
8. **G8. Non-Python crash loops never trip the 3x deterministic stop.**
9. **G9. supervisor.log is truncated on relaunch.**
10. **G10. No committed host dashboard.**

**Do this now, before any code (G6).** Export `main`'s series for
2026-10-05 22:00 to 2026-10-06 03:30 UTC from Prometheus on node1, before they
age out:

```bash
# run on node1; 9090 is published by docker-compose.yml:162-163
M='instance="main:9100"'
for q in "node_memory_MemAvailable_bytes{$M}" "node_memory_SwapFree_bytes{$M}" \
         "rate(node_vmstat_pswpin{$M}[1m])" "rate(node_vmstat_pswpout{$M}[1m])" \
         "rate(node_vmstat_pgmajfault{$M}[1m])" \
         "rate(node_pressure_memory_stalled_seconds_total{$M}[1m])" \
         "rate(node_pressure_io_stalled_seconds_total{$M}[1m])" "up{$M}"; do
  curl -sG localhost:9090/api/v1/query_range --data-urlencode "query=$q" \
    --data-urlencode start=2026-10-05T22:00:00Z \
    --data-urlencode end=2026-10-06T03:30:00Z --data-urlencode step=10s
  echo
done > main_2026-10-06_freeze.jsonl
```

Also read the last
records of `/data/heartbeat.jsonl` and `/data/metrics.jsonl` in the
`metrics_data` volume. These are the only off-box agent records from before
the freeze.

---

## 2. Design

The two halves cover different failures:

- **The crash bundle** covers *the agent died, the host lives*. That is the
  common case.
- **The off-box stream** covers *the host died*. A bundle cannot help there,
  because the bundle is written on the dying host.

### 2.1 Per-incident crash bundle

**Prerequisite: capture the exit status (fixes G1).** In
`launch_skybot.sh:128-133`, run python inside a thin wrapper so something that
outlives it records `$?`:

```bash
setsid bash -c '
  ./venv_mc/bin/python run_minecraft.py --config configs/minecraft_skybot.yaml \
      --timesteps "$1" --seed 0 --out minecraft_skybot_results
  rc=$?; printf "%s %s\n" "$rc" "$(date -Is)" > runlogs/agent_exit.status.tmp
  mv -f runlogs/agent_exit.status.tmp runlogs/agent_exit.status' _ "$TS" \
  > runlogs/minecraft_skybot_run.log 2>&1 < /dev/null &
```

- `$!` becomes the wrapper's PID. The wrapper lives exactly as long as python
  plus microseconds, so `skybot_run.pid`, the supervisor's `kill -0` and
  `host_stop_wait.sh` keep working.
- `pgrep -f "run_minecraft[.]py"` (`:19`) matches the wrapper too, which is
  harmless.
- `rc > 128` means signal `rc-128`: 134 is ABRT, 137 is KILL (the OOM killer
  or earlyoom), 139 is SEGV, 143 is TERM.
- `rm -f runlogs/agent_exit.status` at launch, so a stale status is never read.
- Also build the identity once, at launch, when the host is healthy:
  `./venv_mc/bin/python tools/run_manifest.py build --no-hash-checkpoints --out runlogs/run_manifest.json`.
  See `tools/run_manifest.py:1-10` and `developmental_ai/foundation/runtime/manifest.py:349-389`.
  - `--no-hash-checkpoints` matters: hashing a multi-GB skill bank is the
    wrong workload for a stressed box.
  - The host tree has no `.git` (`manifest.py:11-16`), so `git_revision` is
    `unknown` there. `source.tree_sha256` plus `config.file_sha256` is the
    identity.

**Supervisor classification**, after the watch loop at `supervise_skybot.sh:81-83`:

| STOP present | rc | kind | bundle |
|---|---|---|---|
| yes | 0 | `clean` | short summary |
| yes | ≠0 or missing | `stop-abnormal` (**the 2026-10-05/06 class**) | full |
| no | any | `crash` | full |

The supervisor then does four things:

1. Runs `timeout 90 nice -n 10 bash scripts/incident_bundle.sh <kind> <rc>`,
   **before** it exits or restarts.
2. Appends one line to `crashes.log` for *every* non-clean kind:
   `=== agent exited <ts> kind=<k> rc=<rc> sig=<NAME> bundle=runlogs/incidents/<ts>-<k> ===`.
   The traceback grep stays as it is.
3. Falls back to `SIG="rc=$rc"` when there is no Python exception line, which
   fixes G8. Three identical SIGKILLs then stop loudly, like any other
   deterministic fault.
4. Records `LAUNCH_EPOCH` at `:60`, so the bundle can say "since start".

A bundle failure must **never** block the restart. The script is bounded and
its exit code is ignored.

**`runlogs/incidents/<UTC-ts>-<kind>/` contents.** Every command is wrapped in
`timeout 10`. Root-only reads use `sudo -n` (`MAIN_USER` already has
passwordless sudo: CLAUDE.md §6). The bundle is written to `.partial/` and
`mv`'d into place, so a puller never sees half a bundle.

| file | source | cap |
|---|---|---|
| `meta.json` | kind, rc, signal name, exit ts (from `agent_exit.status`), `LAUNCH_EPOCH`, agent runtime, `uptime`, `/proc/sys/kernel/random/boot_id`, restart count this supervision, `source.tree_sha256` and `config.file_sha256` from `run_manifest.json`, `sha256sum` of the config now (catches an edit mid-run) | tiny |
| `runlog_tail.txt` | `tail -n 400 runlogs/minecraft_skybot_run.log` | ~100 KB |
| `crashes_entry.txt`, `supervisor_tail.txt` | the line just written; `tail -n 100 runlogs/supervisor.log` | small |
| `mem.txt` | `free -h`; `/proc/meminfo`; `swapon --show`; `/proc/pressure/{memory,io,cpu}`; `grep -E '^(pswpin\|pswpout\|pgmajfault\|oom_kill) ' /proc/vmstat` | small |
| `procs.txt` | `ps -eo pid,ppid,rss,vsz,etime,stat,comm,args --sort=-rss \| head -25` | small |
| `gpu.txt` | `nvidia-smi`; `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` | small |
| `kernel.txt` | `sudo -n journalctl -k --since @$LAUNCH_EPOCH --no-pager \| tail -n 400`, falling back to `sudo -n dmesg -T \| tail -n 400` | ~100 KB |
| `oom.txt` | `sudo -n journalctl -u earlyoom --since @$LAUNCH_EPOCH --no-pager \| tail -n 50`; `grep -iE 'oom-kill\|killed process' kernel.txt` | small |
| `disk.txt` | `df -h / /workspace`; `timeout 10 du -sh runlogs logs/checkpoints` | small |
| `nvme.txt` | `sudo -n smartctl -H -A /dev/nvme0`, or `nvme smart-log`, if installed | small |
| `agent_feeds.txt` | `tail -n 5` of heartbeat.jsonl and metrics.jsonl; `tail -n 1` of memory_census.jsonl | small |
| `run_manifest.json` | copied | ~50 KB |

The `clean` summary keeps only `meta.json`, `tail -n 50` of the run log,
`free -h` and `df -h`.

Every bundle also appends one JSON line to `runlogs/incidents/index.jsonl`
(kind, ts, rc, sig, MemAvailable, SwapFree, bundle path). That line is what
node1 counts and ingests.

**Retention on `main`:** keep the newest 50 full bundles and the newest 20
`clean` summaries, delete anything older than 30 days, and enforce a hard cap
of 100 MB total, oldest first. A bundle is roughly 0.2-0.5 MB, so the cap is
generous. The retention pass runs at the end of `incident_bundle.sh` and never
deletes `index.jsonl`. node1 keeps its own copy under its own retention
(2.2). On `main` the bundle is a convenience, not the system of record.

No Python runs in the bundle script. Everything that needs Python, the
manifest, was done at launch. A bundle written while memory is short must not
itself import torch.

### 2.2 Off-box durability: options compared

| Option | Covers | Runs on `main` | Survives a userspace/disk stall | New moving parts | Verdict |
|---|---|---|---|---|---|
| **A. Extend `collect_metrics.sh` pull** | files: supervisor/crashes logs, incident bundles, run-log tail, memory_census, `agent_exit.status`, `run_manifest.json` | nothing new (sshd + rsync, already used) | to the last successful pull (≤60 s) | one function in an existing container | **yes** |
| B. `systemd-journal-upload` → `systemd-journal-remote` on node1 | journald only. The supervisor and agent write *files*, not the journal | a new userspace daemon | no: starves with sshd | packages on both ends, a listening port, TLS or plain HTTP, a new container | no |
| C. Prometheus (already present) | host metrics, 10 s, already off-box | node_exporter (present) | to the last answered scrape | none, plus a textfile collector (2.3) | **yes, it *is* the host-stats stream** |
| D. **netconsole** (kernel printk → UDP → node1) | kernel messages: hung-task, NVMe timeouts, OOM, soft lockup | in-kernel, no userspace, no disk | **yes**. The only option that does | one module option on `main`; a ~5-line UDP listener on node1 | **yes** |
| E. rsyslog forwarding | journald/syslog | userspace daemon | no | config on both | no (B's weakness, plus another daemon) |
| F. A separate host-stats JSONL stream (pulled per minute) | host vitals | a timer | to the last pull | a writer, a puller, an ingester | **no: duplicates C** |

**Recommendation: A + C + D.** No new daemon on `main` except the kernel's own
netconsole.

**A, concretely.** Add `backfill_ops()` to `collect_metrics.sh`, on
`OPS_EVERY=60` s, using the same `SSH_OPTS` (`:43-46`):

- One `rsync -a --timeout=30` of the following into `/data/ops/` (no
  `--delete`, so a bundle pruned on `main` survives on node1):
  - `runlogs/supervisor.log`, `runlogs/crashes.log`
  - `runlogs/agent_exit.status`, `runlogs/run_manifest.json`
  - `runlogs/memory_census.jsonl`, `runlogs/incidents/`
  - `runlogs/STOP` (if present)
- One `ssh ... tail -n 300 runlogs/minecraft_skybot_run.log` into
  `/data/ops/runlog_tail.txt`, written via tmp + mv.
- Call it in **both** the inner loop (`:125-138`) and the reconnect path
  (`:140-145`). Otherwise it pauses exactly while `main` is flapping.
- Touch `/data/ops/.last_ok` on success. That file is what tells you the
  monitor itself has gone blind.
- Warn (rate-limited like `:64-70`) on failure, not `|| return 0`.
- node1 retention for `/data/ops/incidents/`: 180 days, with an age prune in
  the same function. The run-log tail is overwritten every minute. Every
  incident's own copy lives in its bundle.
- Optional, later: `ingest.py` gets an `incidents` table from `index.jsonl`,
  add-only like `:96-127`. A Grafana annotation query on it would mark every
  exit on every panel.

**C, concretely.**

- Add `--storage.tsdb.retention.time=60d --storage.tsdb.retention.size=8GB` to
  the Prometheus service, which means writing `command:` out in full,
  including `--config.file` and `--storage.tsdb.path`. Rough sizing: 4 targets
  × ~1.5k series × 8640 samples/day × ~1.3 B ≈ 70 MB/day. node1's disk is
  ~98 GB (`collect_metrics.sh:63`). **UNVERIFIED** sizing. Check
  `prometheus_tsdb_storage_blocks_bytes` after a week.
- Add the textfile collector on `main` (2.3).
- Commit a "main host" dashboard (G10). Suggested panels:
  - MemAvailable and SwapFree
  - PSI memory/io full
  - pswpin/pswpout
  - pgmajfault
  - per-process RSS
  - VRAM per process
  - NVMe temp and media errors
  - `up` for both `main` jobs

**D, concretely.**

- On `main`, set `netconsole` with `netconsole=@/enp5s0,6666@<NODE1_ADDR>/`.
  Put it in `/etc/modprobe.d/` plus `/etc/modules-load.d/`, with the address
  kept only on the host, like every other address (CLAUDE.md §6).
- `kernel.printk = 7 4 1 7` is already set by `host_hardening.sh:78`, so
  warnings reach the console, and netconsole *is* a console.
- On node1, a tiny container appends `socat -u UDP-RECV:6666 STDOUT` to
  `/data/ops/netconsole.log`, with logrotate or a size cap.
- This is the one channel that would have answered **thrash versus NVMe stall
  versus NIC** for 2026-10-06:
  - hung-task warnings naming `nvme` or `jbd2` → storage
  - nothing at all while the ping in 2.4 still succeeds → userspace thrash
  - nothing, and no ping → NIC or kernel
- Caveat: netconsole needs a NIC driver with netpoll support. The wired NIC
  (`enp5s0`, `host_hardening.sh:22-23`) is probably `r8169`, which has it
  (**UNVERIFIED**). It is UDP and best-effort, and it carries kernel messages
  only.

**Also, locally on `main` (cheap):**

- Set `Storage=persistent` and `SyncIntervalSec=30s` in
  `/etc/systemd/journald.conf.d/`, so the on-disk journal loses ≤30 s rather
  than ≤5 min whenever the disk still takes writes.
- Make both launchers **append** to supervisor.log (`>>` at `host.yml:169` and
  `deploy_skybot.sh:216`) with a size cap, or rotate it like the run log
  (`launch_skybot.sh:122-127`). This fixes G9.

**Why node1 and not node2.** node1 already holds the key, the collector, the
`metrics_data` volume, Prometheus and Grafana. node2's mirror can optionally
add `runlogs/incidents/` as a second copy (its `BRAIN_PATHS` override), but
that is redundancy, not the design.

### 2.3 Host stats to add

"Already" means `node_exporter` v1.12.1 on `main` collects it with default
collectors, and Prometheus on node1 already stores it at 10 s.

| Stat | Already in Prometheus? | Metric / how to add |
|---|---|---|
| Memory PSI `/proc/pressure/memory` | **yes** (pressure collector, default on) | `node_pressure_memory_waiting_seconds_total` (some), `..._stalled_seconds_total` (full) |
| IO PSI | **yes** | `node_pressure_io_waiting_seconds_total`, `node_pressure_io_stalled_seconds_total` |
| CPU PSI | **yes** | `node_pressure_cpu_waiting_seconds_total` |
| Swap in/out rates | **yes** (vmstat collector default fields `oom_kill\|pgpg\|pswp\|pg.*fault`) | `rate(node_vmstat_pswpin[1m])`, `rate(node_vmstat_pswpout[1m])` |
| Major page faults | **yes** | `rate(node_vmstat_pgmajfault[1m])` |
| Kernel OOM kills | **yes** | `node_vmstat_oom_kill` (counter) |
| MemAvailable, SwapFree | **yes** | `node_memory_MemAvailable_bytes`, `node_memory_SwapFree_bytes` |
| Disk latency / saturation | **yes** | `node_disk_io_time_weighted_seconds_total`, `node_disk_io_time_seconds_total` |
| NVMe temperature | **probably yes** (hwmon) | `node_hwmon_temp_celsius{chip=~"nvme.*"}`. **UNVERIFIED** chip label on `main` |
| NVMe media errors, error-log entries, critical warning, unsafe shutdowns, % used | **no** | root timer every 5 min: `smartctl -j -A /dev/nvme0` (or `nvme smart-log -o json`) → textfile `nvme.prom` |
| GPU totals (VRAM used/total, util, temp, power) | **yes** (job `gpu`, `main:9835`) | `nvidia_smi_memory_used_bytes` etc. |
| GPU memory **per process** | **UNVERIFIED**. `prometheus.yml:33-35` says "per-process", but `nvidia_gpu_exporter` wraps `nvidia-smi --query-gpu`, which is per-GPU. Check with `curl main:9835/metrics \| grep -i proc` | timer: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` → `skybot_gpu_proc_bytes{comm=...}` |
| Agent (python) RSS / swap | **no** (node_exporter has no per-process metrics) | 60 s timer → textfile `procs.prom`: `skybot_proc_rss_bytes{role="agent\|java\|ollama\|llama-server"}` summed from `ps -eo rss,comm,args`. Add `VmSwap` from `/proc/<pid>/status` for the agent |
| Java client RSS (each, and count) | **no** | same timer: `role="java"`, plus `skybot_java_clients` (`host.yml` status checks "want 2") |
| Textfile freshness | — | each writer emits `skybot_textfile_written_unixtime`, so a dead timer is visible, not silent |

To enable the textfile collector on `main`:

- Pass `--collector.textfile.directory=/var/lib/node_exporter/textfile` to the
  `node_exporter` `install_one` call (`install_host_exporters.sh:93-95`). It is
  extra args through `$*`, `:77`.
- Create that directory world-readable, because the unit uses `DynamicUser`
  (`:76`).
- Writers emit `*.prom.tmp` then `mv`.
- Two units: a `skybot-procstats.timer` (60 s, unprivileged) and a
  `skybot-nvme.timer` (5 min, root).

During a thrash these timers stall like everything else. Their job is the
trend *into* the stall, and PSI plus swap rates are already scraped
independently.

### 2.4 "main went silent" alert

**Mechanism:**

- Grafana-managed alert rules, provisioned from a new
  `docker/monitor/grafana/provisioning/alerting/` (rules + contact point),
  querying the existing `rack-prometheus` datasource. **No new container:**
  Grafana 11 evaluates and notifies on its own, without Prometheus rules or an
  Alertmanager.
- Contact point: a webhook (ntfy, Discord or similar), with the URL kept in
  `docker/monitor/.env` and referenced as `$__env{...}`.

**node1-side signals.** The collector writes them as a textfile into the
existing `node_textfile` volume (`docker-compose.yml:128-134`, `:183-184`). The
collector must mount that volume, and it uses its own filename,
`skybot_ops.prom`, not the `gpu.prom` that `gpu-metrics` writes.

- `skybot_heartbeat_age_seconds`: now minus the last `wall_time` in
  `/data/heartbeat.jsonl`
- `skybot_ops_pull_age_seconds`: now minus the `/data/ops/.last_ok` mtime
- `skybot_stop_present`: 0 or 1, from the ops pull
- `skybot_main_ping_ok`: `ping -c1 -W2 main`. busybox ping in the alpine
  image; NET_RAW is a default docker capability
- `skybot_incidents_total`: line count of `/data/ops/incidents/index.jsonl`

**Rules:**

| Rule | Condition (PromQL) | For | Meaning |
|---|---|---|---|
| **MainHostSilent** | `up{job="node",instance="main:9100"} == 0`, with **NoData → Alerting** (an absent series is silence too) | 2m | The box stopped answering. The `ping_ok` label splits it: ping OK means kernel alive, userspace or disk starved (the 2026-10-06 signature); ping failing means NIC or kernel dead or powered off |
| **MainThrashing** (precursor) | `rate(node_pressure_memory_stalled_seconds_total{instance="main:9100"}[1m]) > 0.10` **or** `(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.05 and node_memory_SwapFree_bytes / node_memory_SwapTotal_bytes < 0.25)` | 2m | Would have fired *before* the 2026-10-06 freeze. Thresholds are a starting point. Calibrate against the 1.6 export |
| **AgentSilent** | `skybot_heartbeat_age_seconds > 900 and on() skybot_stop_present == 0 and on() up{instance="main:9100"} == 1` | 1m | Host up, agent not stepping, no deliberate stop |
| **AgentCrashLoop** | `increase(skybot_incidents_total[1h]) >= 3` | 0 | Restarting repeatedly. This catches G8 even before the supervisor fix |
| **MonitorBlind** | `skybot_ops_pull_age_seconds > 300 and on() up{instance="main:9100"} == 1` | 5m | The collector is broken, so silence elsewhere must not be read as health |

**Why 900 s for AgentSilent.** The heartbeat is emitted on the per-step path
(`developmental_loop.py:8669-8690`, "This runs on the per-step path"). It
pauses whenever stepping pauses: synchronous PPO updates (`async_wm` only moves
the world model off-thread: `minecraft_skybot.yaml:3076-3079`), and the
~8.5 min replay-buffer save on stop (`docs/OPERATIONS.md:35-40`). 15 min sits
above both. During a stop, `stop_present` suppresses the rule anyway.

### 2.5 Guard check (CLAUDE.md §4.1): what re-opens each guard

- **Bundle script:** bounded by `timeout 90`, exit code ignored. The restart
  never waits on it. It cannot latch the supervisor.
- **3x same `rc` → supervisor stops:** this is the existing deliberate
  behaviour (`docs/OPERATIONS.md:85-91`), now also applied to signal deaths.
  Re-opened by `host.yml` `launch`. This is a human action by design: three
  identical deaths are a diagnosis, not a transient.
- **earlyoom kill → relaunch → 3x stop:** a memory config that cannot fit
  ends in the loud stop above, not an infinite 300 s loop.
- **AgentSilent suppressed by STOP:** re-opened automatically, because `launch`
  removes STOP (`host.yml:167`).
- **Every alert auto-resolves** when its condition clears. None needs a human
  to reset.
- **Retention prunes** never delete `index.jsonl`, the newest bundle, or
  anything on node1 younger than 180 d.

**Cost while nothing happens:**

- the bundle: 0
- the ops pull: one ssh plus a small rsync per minute
- the textfile timers: milliseconds per minute
- netconsole: 0 bytes until the kernel prints
- the alerts: Grafana evaluation only

### 2.6 Rollout order

1. **Now:** run the Prometheus export in 1.6 (it ages out around 2026-10-21),
   and raise retention (2.2 C).
2. Capture the exit status in the launcher, classify in the supervisor, write
   the bundle, append to supervisor.log (2.1, G9). Write a contract test in
   the house style (`tests/_*_smoke.py` with an incident docstring). The
   classification table and the `rc`→signal mapping are the claims it should
   encode.
3. `backfill_ops` in the collector (2.2 A).
4. Alert rules and the contact point (2.4). Send a test notification, then
   `touch runlogs/STOP` and confirm AgentSilent stays quiet.
5. Textfile collector and timers on `main`, plus the host dashboard (2.3).
6. netconsole (2.2 D). Verify with `echo "<4>netconsole test" | sudo tee /dev/kmsg`
   and look for the line on node1.

Per CLAUDE.md §3, none of this is "proven" until it has run live on `main`.
Step 2 is proven only by an actual non-clean exit producing a bundle that
node1 then holds.

---

## 3. UNVERIFIED (check on the hosts)

- journald `Storage=` and `SyncIntervalSec` on `main`: `systemd-analyze cat-config systemd/journald.conf`.
- Which `host_hardening.sh` options were applied: `sudo bash scripts/host_hardening.sh`
  (the dry run prints current state).
- Whether `nvidia_gpu_exporter` exposes per-process memory, despite
  `prometheus.yml:35`.
- The hwmon chip label for the NVMe on `main`.
- Whether the Grafana `grafana_data` volume holds UI-built host dashboards.
- netconsole support in `main`'s NIC driver (`ethtool -i enp5s0`).
- Prometheus on-disk growth rate before choosing the retention size.
