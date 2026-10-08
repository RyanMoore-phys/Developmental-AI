# Monitoring

Two stacks that share a Grafana and nothing else. Knowing which is which saves
a lot of confusion.

| stack | answers | path |
|---|---|---|
| **Prometheus + exporters** | *How are the machines?* CPU, RAM, disk, VRAM, temperature — for every host. | exporters on each host → Prometheus (controller) → Grafana |
| **SQLite + collector** | *How is the agent?* Steps/s, breaks, logs, reward shares, world-model loss. | agent writes JSONL → collector rsyncs → `ingest.py` → SQLite → Grafana |

They are separate because they answer different questions at different
cadences, and because the raw JSONL is the agent's system of record. Delete
`skybot.db` and `ingest.py` rebuilds it identically from the log.

---

## The trap worth knowing first

The `node-exporter` and `gpu-metrics` containers in `docker/monitor/` run on
**the controller** and report **the controller**:

```yaml
node-exporter:
  hostname: ${MONITOR_HOSTNAME:-node1}
  volumes: [ /:/host:ro,rslave ]     # the CONTROLLER's filesystem

gpu-metrics:
  devices: [ /dev/dri:/dev/dri ]      # an INTEL iGPU, via intel_gpu_top
```

Neither has ever seen the training host's NVIDIA GPU. That is
`nvidia_gpu_exporter` on the training host, installed by
`scripts/install_host_exporters.sh` and scraped as a separate Prometheus job.

Confusing the two costs a day: you watch a GPU graph, conclude the training GPU
is healthy, and it is a different machine's integrated graphics.

---

## Adding a host

```bash
# on the new host
bash scripts/install_host_exporters.sh
```

`node_exporter` on :9100 always; `nvidia_gpu_exporter` on :9835 only where
`nvidia-smi` exists. Both are systemd units with `Restart=always`, bound to the
LAN address, capped at 10% CPU and 128 MB.

Then add it to `docker/monitor/prometheus.yml` and restart Prometheus.

**If a target shows DOWN with `context deadline exceeded`**, that is a timeout,
not a refusal — the packets are being dropped. A closed port refuses
*immediately*. That distinction points straight at a firewall:

```bash
sudo ufw allow from <MONITORING_SUBNET> to any port 9100 proto tcp
sudo ufw allow from <MONITORING_SUBNET> to any port 9835 proto tcp
```

If traffic reaches the host over a VPN interface rather than the LAN, the
source address will not match a LAN-scoped rule. Allow on the interface
instead.

---

## The agent feed

Three files on the training host, pulled by the collector:

| file | cadence | holds |
|---|---|---|
| `runlogs/metrics.jsonl` | per segment (~7 min) | the authoritative closed observation: losses, break counts, reward shares |
| `runlogs/heartbeat.jsonl` | ~15 s | running totals, live income shares — for "what is it doing *right now*" |
| `server.jsonl` | ~30 s | written by the controller: how many players the game server actually sees |

`server.jsonl` is a **falsifier**, not a second source. Its value is
*contradicting* the agent's self-report: the agent believing it is playing and
the server seeing nobody is a real failure mode, and only the server can tell
you.

### Reading `players_online: -1`

`-1` is not a bug. It is recorded deliberately when the server is unreachable —
a down server is a data point, and skipping the write would leave a gap
indistinguishable from the poller itself dying. **The reason is in the
`version` column**, which reads `unreachable: <ExceptionName>`.

A sustained `-1` almost always means `MC_HOST` in the monitor `.env` is stale.

---

## Reward shares

The panel shows each income source as a fraction of gross reward, plus an
**`other`** residual.

**`other` is the point.** It is `1 − sum(named sources)`, so a non-zero value
means the agent is being paid by something the dashboard cannot name. The
earlier version of this panel hardcoded six keys and silently dropped the rest,
which is how a channel reached 69% of all income while being invisible.

If you add a reward source, add it to the panel. The residual will tell you if
you forget.

---

## Idempotency and schema

`ingest.py` is `INSERT OR IGNORE` keyed on a monotonic `seq`, so re-reading the
whole JSONL is safe and is how you rebuild after a bad ingest. Nested maps
(`reward_shares`, `breaks_by_type`, `income_now`) are stored as JSON columns so
a new source needs no migration.

Scalar columns do need one, and it is **add-only** — every column is nullable
and nothing is ever dropped or renamed. A table created before a column exists
otherwise fails every insert with `no such column`, which presents as frozen
panels and healthy containers, with nothing in any log.

---

## Deploying monitor changes

The compose files pin their project names (`name: skybot-monitor`,
`name: skybot-runner`). Compose otherwise derives the project from the
*directory*, and named volumes are `<project>_<volume>` — so running the same
file from a different path would create **empty** volumes and look exactly like
total data loss. The pin is what makes the directory irrelevant.

Check your volumes survived a move:

```bash
docker volume ls | grep skybot-monitor
```

You want the existing volumes reattached, not new ones.

## Learning, exploration and incident telemetry (2026-10-07)

The collector's evidence mirror drops `learning.jsonl*`, `position_trace.jsonl*`,
`memory_census.jsonl*` and `incidents/` flat into `/data`; `ingest.py` reads them
beside `metrics.jsonl` (override with `DATA_DIR` or the per-feed `*_PATH` vars).

| Table | From | Key (INSERT OR IGNORE) |
|---|---|---|
| `learning_kv(run_id, seq, wall_time, key, value)` | `learning.jsonl` (`skybot.learning` v1), every numeric leaf as a dotted key, e.g. `reward.stream-0.gui_dwell`, `exploration.stream-1.steps_since_new_cell`, `ppo.approx_kl` | run_id, seq, key |
| `positions(run_id, stream, step, t_wall, x, y, z, yaw, pitch)` | `position_trace.jsonl` (`skybot.position` v1) — **evaluator-only** | run_id, stream, step, t_wall |
| `incidents(dir, class, exit_code, signal, t_start, t_end, run_id, git_rev, config_hash)` | `incidents/<ts>-<class>/summary.json` + `index.jsonl` (keeps pruned bundles); times are **epoch s** | dir |
| `memory_census(wall_time, key, value)` | `memory_census.jsonl`, numeric leaves except the unbounded `objects` / `deep_tensors` | wall_time, key |

Same rule as `segments`: the JSONL is the record. `rm skybot.db` and one pass
rebuilds identical rows; rotated `.N` siblings are read too. Booleans become
0/1; lists, strings and `null` (a missing producer) produce **no row** — a gap,
not a zero. A torn last line waits for its newline; a corrupt line or an
unknown `(schema, v)` is skipped and counted in the ingest log line
(`ingest telemetry skipped ...`). Key names are the producer's; there is no
mapping layer to drift.

**Dashboards** (`grafana/provisioning/dashboards/`): *SkyBot Learning*
(scoreboard, signed reward by source, mix, behaviour), *SkyBot Exploration*
(X/Z scatter, Y over time, steps/seconds since new cell, cells/hour, path,
displacement, radius), *SkyBot ML Health* (PPO, WM, curiosity gate, replay,
timing, health), *SkyBot Host* (memory, swap, PSI, GPU, census, incidents).

**Alerts** (`grafana/provisioning/alerting/skybot_alerts.yml`, folder SkyBot):
main down (`up == 0` 5 m), memory thrash (swap-in > 500 pages/s or PSI-full >
10 %, 5 m), heartbeat stale (> 15 min), crash loop (≥ 3 `crash` incidents in
30 min), collector blind (newest `learning_kv` row > 15 min old), stuck
(`steps_since_new_cell` > 5000 for 1 h, per stream). All auto-resolve. A
deliberate STOP fires *heartbeat stale* and *collector blind* until relaunch.
Delivery goes to the default policy; `contact_points.yml` is a **placeholder**
(`skybot-oncall`, a dead localhost webhook) — put the real receiver in a local,
uncommitted copy on node1 and route the default policy to it.

**Offline:** `PYTHONPATH=. python tools/learning_report.py --learning
runlogs/learning.jsonl --positions runlogs/position_trace.jsonl --png` writes
`runlogs/learning_report/report.md` (+ PNGs) from the raw JSONL alone.

`tests/_telemetry_ingest_smoke.py` runs **every** dashboard and alert query
against a synthetic producer-shaped ingest: a renamed key fails CI instead of
showing an empty panel.
