# Operations

Day-to-day running of a live agent. For first-time setup see
[`REPLICATION.md`](REPLICATION.md).

---

## The three actions

Everything routes through the `host` workflow (Actions → host), which needs no
SSH from you.

| action | what it does |
|---|---|
| `status` | read-only. Run, bridge, GPU, host health, stream health, crash reasons, supervisor log. |
| `connect` | (re)establishes the socat bridge to the game server. |
| `launch` | clears the stop-file and starts the supervisor. **Moves no code.** |
| `stop` | writes `runlogs/STOP` and *waits* for a graceful exit. |
| `provision` | self-seeding: rsyncs the repo and runs the full build. For a cold host. |

## The ordering that matters

```
code change  →  stop  →  deploy  →  launch
```

`launch` starts whatever is already on the host. `deploy` is the only step that
syncs code, and it **refuses while training is live** — deliberately, so a push
can never interrupt a run. Skipping `deploy` is the single most common way to
spend an hour wondering why a fix had no effect.

`deploy` runs with `DEPLOY_LAUNCH=0`: it syncs and smoke-tests, and never
starts training. Starting is always a deliberate act.

## Stopping takes about 13 minutes

A healthy graceful stop is ~5 minutes to finish the current segment plus ~8.5
minutes to save the replay buffer. The wait budget is 20 minutes and it
**fails** if it expires rather than reporting success — an earlier version
waited 5 minutes, gave up, and exited 0, so "I stopped waiting and it is still
training" showed as a green check.

A stop that returns *instantly* means there was no live process: the agent had
already died. Check `runlogs/crashes.log`.

---

## Reading `status`

```
== run ==
training LIVE (pid 27261)        supervisor: 1 proc       STOP file: absent
== bridge ==
socat bridge UP                  tailscaled up
== gpu ==
NVIDIA GeForce RTX 5050, 5632 MiB, 8151 MiB, 30 %
== host ==
    RAM  9959 used / 15341 total, 5381 avail
    SWAP 1863 / 4095
    DISK 81G used of 232G (37%)
    java clients: 2  (want 2)
```

What each line is actually telling you:

- **`supervisor: 2 proc`** — you launched twice. The second one cannot start a
  competing agent (the `REFUSED: run alive` guard) but it will retry every 60
  seconds forever and race the first at the next restart.
- **`java clients: 3`** — a client leaked. A hung reset is abandoned and
  rebuilt; the abandoned process is reaped on a daemon thread, and a count
  above `num_envs` means that failed.
- **GPU at 3%** — usually means training has not started yet (the buffer is
  filling), *or* you sampled between training blocks. GPU use here is bursty by
  design; a single `nvidia-smi` reading is a point sample of a spiky workload.
  Use the Grafana time series instead.
- **`zero-filled stream events`** — the number that should always be 0. A
  stream whose rebuild is exhausted is zero-filled and the run **continues**:
  steps accumulate and the dashboard looks healthy while half the world model's
  input is blank frames.

---

## When it crashes

The supervisor restarts on crash with backoff, and **stops relaunching**
after the same fault three times — that message is a diagnosis, not a failure:

```
[supervisor] STOPPING RELAUNCHES: kind=crash sig=SIGABRT 3x within 1800s — this is
[supervisor]   deterministic and will not fix itself; wrote runlogs/CRASHLOOP
```

### Exit classes, incident bundles, the crash-loop latch (2026-10-07)

The agent runs under a wrapper (`launch_skybot.sh`) that writes
`runlogs/skybot_run.exit` (`exit_code`, `signal`, `t_end`); `skybot_run.pid`
is still the agent's own pid. Every exit is classified:

| STOP | rc / log tail | class |
|---|---|---|
| yes | 0, no `terminate called` / `Traceback` | `clean` |
| yes | anything else, or no `.exit` | `abnormal_under_stop` (the 2026-10-06 class) |
| no | any | `crash` |

rc > 128 is a signal: 134 ABRT, 137 KILL (OOM / earlyoom), 139 SEGV, 143 TERM.
Each exit writes `runlogs/incidents/<UTC>-<class>/` (`scripts/incident_bundle.sh`:
`summary.json`, run-log tail, memory/PSI, top RSS, GPU, kernel journal,
earlyoom, disk; `clean` gets a short one) and a line in
`runlogs/incidents/index.jsonl`. Kept: newest 50 / 100 MB on `main`; node1's
collector mirrors them every 60 s and keeps 180 d.

**Crash-loop latch.** Same class + signal (or `rc=N`) 3x within 30 min, or the
same Python exception 3x in a row, writes `runlogs/CRASHLOOP` and the
supervisor stops relaunching (it stays alive, idle). **To re-open** — any one:

- `rm runlogs/CRASHLOOP` — the idle supervisor resumes within 60 s;
- run `bash scripts/launch_skybot.sh` by hand — it clears the latch and the
  supervisor adopts that agent;
- `host.yml` → `launch` (starting a supervisor clears it);
- `touch runlogs/STOP` ends the idle supervisor cleanly.

`runlogs/supervisor.log` is still truncated by every launcher (`>`); the
supervisor keeps the last 5 sessions as `supervisor.log.1..5`.

**Kernel log off-box.** `sudo bash scripts/host_hardening.sh --apply
--netconsole node1` streams kernel messages as UDP to a LAN receiver; on node1
run `nc -u -l 6666 >> netconsole.log`.

Start here:

```bash
grep -hoE "[A-Za-z_.]*(Error|Exception)" runlogs/crashes.log | sort | uniq -c | sort -rn
```

Common classes and what they mean:

| exception | meaning |
|---|---|
| `TimeoutError` (MineRL socket) | a client hung. Expected occasionally; the rebuild path absorbs it. |
| `torch.OutOfMemoryError` | VRAM. Reduce `world_model.batch_size`, or set `grad_checkpoint: true`. |
| OOM-killer in `dmesg` | system RAM. Check what is resident — a local VLM is usually the largest consumer. |
| `NameError` / `AttributeError` | a code bug, usually on a rarely-taken path. |

**Memory diagnosis is two separate questions.** A system-RAM OOM shows as swap
climbing then a process dying. A CUDA OOM shows as nothing at all in system
memory — the process dies and releases its VRAM. If the host memory graph is
flat across a crash, the cause is not host memory.

---

## Backups

Brain state lives **only on the training host**, on one disk, with no
redundancy:

```
world_model.pt   symbolizer.pt   familiarity.pt   magnet.pt
knowledge_graph.json   options_state.json   skill_bank_*/   runlogs/breaks_by_type.json
```

Nothing automates copying these. Pull them to a second machine periodically
during long runs. Code always flows controller → host, so code is never at
risk; the brain is the irreplaceable part.

**On restore, match the directory layout first.** `break_memory_path` lives
under `runlogs/`. Restoring a backup whose paths differ silently starts the
break memory *empty* rather than erroring — and an agent with empty break
memory re-opens every mastered block tier at full reward.

---

## Routine checks

| frequency | what |
|---|---|
| per run | `status`: restart count stable, `java clients` correct, bridge UP |
| daily | disk headroom; the replay buffer and logs grow |
| weekly | pull brain state to a second machine |
| after any config change | confirm it reached the host — `grep` the value in `/workspace/devai/configs/` |

That last one is worth the habit. A config change that never deployed is
indistinguishable from a config change that did not work.
