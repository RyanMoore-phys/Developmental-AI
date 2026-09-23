# Running a training run

> **HOST CHANGED 2026-09-22.** Training now runs on the user's own main
> computer — Ubuntu Server, **192.168.1.10**, on the LAN and the tailnet at
> once — not a rented pod. It is installed at `/workspace/devai` and deployed
> to as `root@` deliberately, so every command in this file works unchanged;
> only `POD_HOST` and `POD_PORT` below take different values (`22`, fixed —
> the rotating-port problem is gone). CI reads them from the runner's `.env`,
> never from here: see `docs/CI_SETUP.md`. The 16 GB of RAM forced several
> config cuts — CLAUDE.md §6 lists them.

All commands assume the connection shorthand. **No literal host/port here
(2026-09-02):** on a rented pod both changed on every rebuild, so a written-down
address was stale immediately — and a stale address reads exactly like a dead
run. The habit is kept now that the address is fixed, because the runner `.env`
is the one place that should name a host.

```bash
export POD_HOST=<host-ip>             # 192.168.1.10, or the pod's Connect IP
export POD_PORT=<ssh-port>            # 22 on the main computer
SSH="ssh root@$POD_HOST -p $POD_PORT -i ~/.ssh/skybot_ed25519 \
     -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
# NOTE: zsh will NOT run a command stored in a variable ($SSH ...). Either paste
# the full ssh line, or wrap it in a shell function. (Known gotcha.)
```

Once the pod is on the tailnet, prefer the stable name and skip the key:
`tailscale ssh root@<pod-magicdns-name>`.

## 1. Sync code from the Mac

From the parent repo dir:

```bash
rsync -rlptz -e "ssh -p $POD_PORT -i ~/.ssh/skybot_ed25519 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  developmental_ai configs scripts run_minecraft.py \
  root@"$POD_HOST":/workspace/devai/
```

## 2. Launch

`scripts/launch_lifelong.sh <timesteps>` is idempotent and safe:

```bash
$SSH 'cd /workspace/devai && bash scripts/launch_lifelong.sh 1000000'
```

What it does: **refuses** if a run or stale java is alive; ensures `Xvfb :77`,
`openbox`, and `ollama serve`; `rm -rf podlogs/brain`; then launches DETACHED via
`setsid` with `DISPLAY=:77 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MINERL_HEADLESS=1`,
logging to `podlogs/minecraft_lifelong_run.log`, pid → `podlogs/lifelong_run.pid`.

First Minecraft boot takes **~90 s** (4 java clients spawn). Episode 1 appears a
few minutes later (~5–6 min/segment). It does NOT wipe skill_bank or broadcaster
state — it continues the lifelong memory; only `podlogs/brain` is regenerated.

## 3. Monitor — the signals that matter

```bash
$SSH 'cd /workspace/devai && grep -aE "Episode [0-9]+ \| Timestep|this ep:" podlogs/minecraft_lifelong_run.log | tail'
```

- **`Reward (avg): X (this ep: Y)`** — the *cumulative mean* is `~sum/N` and
  decays like `8.46/N` after one big early episode; it is NOT a decline. Read the
  `(this ep: Y)` RAW value for recent reality.
- **`h_evolve=…`** — cross-segment RSSM evolution. ~0 = frozen (bug); >0 = the
  lifelong carry is live (was verified at 0.49–0.64). `||h||` is LayerNorm-pinned
  at √512≈22.6 and is BLIND to evolution — ignore it for liveness.
- **Curiosity magnet line** — `tree=…` prob and `target=…`. `tree=0.000, target=
  stone_visible` means the fleet wandered off-tree (the seek drive should recover
  it).
- **Liveness:** `ps -o %cpu,etime -p $(cat podlogs/lifelong_run.pid)` — ~100% CPU
  = advancing; ~0% = hung.
- **Log-breaking:** it shows as GOAL unlocks in `broadcaster_state.json`
  (`slot_uid` + `unlock_log`), NOT as skill "behaviours" (skills need ~20 eps of
  mastery) and NOT as `grep log run.log` (achievements aren't printed).

Quick health one-liner:

```bash
$SSH 'cd /workspace/devai && echo "cpu:"; ps -o %cpu,etime -p $(cat podlogs/lifelong_run.pid); \
  grep -aicE "Traceback|CRITICAL|Fatal" podlogs/minecraft_lifelong_run.log; \
  grep -aoE "h_evolve=[0-9.na]+" podlogs/minecraft_lifelong_run.log | tail -3'
```

## 4. Stop (gracefully)

```bash
$SSH 'cd /workspace/devai && kill $(cat podlogs/lifelong_run.pid); sleep 6; \
  pkill -x java; sleep 3; \
  echo "python left: $(pgrep -fc run_minecraft[.]py) java left: $(pgrep -xc java)"'
```

Archive the log first if you want to keep it: `cp podlogs/minecraft_lifelong_run.log
podlogs/minecraft_lifelong_run.$(some-tag).log`. The launch script's `>` redirect
truncates the live log on the next start.

## Compute note

The organism needs a GPU realistically (RSSM + CNN-ICM + llava:7b). **On a CPU pod
it will run but be very slow**, and llava on CPU is impractically slow — expect
the VLM grounding cadence to crawl. Use CPU pods for data rescue / light work, GPU
pods for training.

## Gotchas

- Detached launch: python stdout is block-buffered under nohup → `PYTHONUNBUFFERED=1`
  (the launch script sets it). Episode-boundary progress is also visible from
  video-file mtimes.
- Background watchers run under zsh → a `$VAR`-as-command silently fails; use a
  shell function in watcher scripts.
- Ollama reliability: launch it with `OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1
  OLLAMA_NUM_PARALLEL=1` and `keep_alive=-1` in the generate call (the "model
  runner unexpectedly stopped" crash is reload-under-pressure).
