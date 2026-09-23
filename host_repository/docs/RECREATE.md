# Recreating the RunPod from scratch — full guide

End-to-end steps to stand up a fresh RunPod GPU pod running the Developmental-AI
organism, in either of two environment modes:

- **MineRL mode** (default, most stable) — local generated Treechop worlds.
- **External-server mode** — env 0 joins your own Minecraft server over a
  private Tailscale tunnel. See also `SERVER_CONNECTION.md`.

Everything here has been proven on an RTX A4000 / AMD EPYC pod, Ubuntu 22.04.

---

## 0. Prerequisites (once)

- A RunPod account + an SSH key. **Use `~/.ssh/skybot_ed25519`** — the key the
  RunPod dashboard shows as "id_ed25519" does not exist locally under that name.
- The brain backup (`pod_repository/data/skill_bank_mc_curiosity/`, the byte-exact
  3.5 GB skill bank) and this repo, on your Mac.
- For external-server mode only: a Tailscale account, and a Minecraft server you
  control (see `SERVER_CONNECTION.md`).

## 1. Create the pod

Create a GPU pod (RTX A4000 16 GB is enough; the box is a 64-core/128-thread
EPYC). Choose an **Ubuntu 22.04** template (24.04 also works). Attach a
**persistent volume mounted at `/workspace`** — this is a MooseFS network volume
that survives pod stop/restart and holds everything.

Note the pod's SSH host, **port** (changes every pod), and that the IP is often
reused. All SSH/scp in this guide uses:

```bash
SSH="ssh root@<IP> -p <PORT> -i ~/.ssh/skybot_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
```

The `UserKnownHostsFile=/dev/null` flag is **required** (reused IPs otherwise trip
host-key mismatch and the connection is refused).

## 2. Sync the repo to the pod

From your Mac, inside the repo:

```bash
rsync -rlptz -e "ssh -p <PORT> -i ~/.ssh/skybot_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  ./ root@<IP>:/workspace/devai/ \
  --exclude '.git' --exclude 'pod_results' --exclude '__pycache__'
```

Use `-rlptz`, **not** `-az` — the network volume forbids chown, so `-a` exits 23
(files still copy, but the non-zero exit is confusing). The cached VirtualGL
`.deb` under `pod_repository/data/minerl_build/` must be included (provisioning
prefers it over a live download).

## 3. Provision (~40–60 min)

Run detached and watch the log — the MineRL build alone runs a full gradle build:

```bash
$SSH 'cd /workspace/devai && nohup bash scripts/provision_pod.sh > podlogs/provision.log 2>&1 &'
$SSH 'tail -f /workspace/devai/podlogs/provision.log'   # Ctrl-C to stop watching
```

`scripts/provision_pod.sh` stages (see `PROVISIONING.md` for the deep why of each):

| Stage | What |
|---|---|
| 1  | apt: openjdk-8, xvfb, openbox, xdotool, python3.10 |
| 1b | **VirtualGL 3.1.1** (GPU headless GL via EGL — fixes the boot hang) |
| 1c | **Tailscale + socat** (optional; external-server mode only) |
| 2  | venv_mc + torch **cu124** (default cu130 → `cuda False` on driver 550) |
| 3–4 | MineRL v1.0.0 MCP-Reborn manual build (mixin removed, assets unpacked, MouseHelper focus patch) |
| 4c | **launchClient.sh GPU-GL wrap**: `vglrun -d egl` + `taskset -c` per-core pin |
| 5–7 | options.txt, ollama+llava, Xvfb :77 + openbox |

Success = the final log line is `=== PROVISION-COMPLETE ===` with `errors: 0`.
Hard-fail markers are `PROVISION-FAILED: <what>`.

**Two load-bearing fixes baked into stage 1b/4c** (the MineRL boot hang):
1. Software GL (llvmpipe under Xvfb) hangs the client forever at "Reloading
   ResourceManager" → VirtualGL EGL renders on the GPU instead.
2. The NVIDIA driver's CPU-dispatched memcpy stack-corrupts in a multithread
   race on AMD EPYC (glibc "futex facility" abort at GL upload) → each client is
   pinned to a single distinct core (`taskset -c $((port % $(nproc)))`), which
   serializes the race away while keeping the N envs parallel across cores.

## 4. Restore the brain (byte-exact)

**NEVER wipe `skill_bank_mc_curiosity`** — it is the organism's accumulated
lifelong memory (goals + minted skills). Upload the backup as a BACKGROUND task
(a foreground command is killed by the 2-min tool timeout mid-transfer):

```bash
rsync -rlptz --partial -e "ssh -p <PORT> -i ~/.ssh/skybot_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  pod_repository/data/skill_bank_mc_curiosity/ \
  root@<IP>:/workspace/devai/skill_bank_mc_curiosity/
```

Then clean any rsync partials and verify (see `TRANSFER.md` for the exact
byte-verify — `du`/`awk`/`printf %d` all give WRONG answers on this volume; the
authoritative check is `find ... -exec stat -c %s {} + | awk '{s+=$1}END{printf "%.0f",s}'`).
A quick load-check:

```bash
$SSH 'cd /workspace/devai && ./venv_mc/bin/python -c "import json; b=json.load(open(\"skill_bank_mc_curiosity/broadcaster_state.json\")); r=json.load(open(\"skill_bank_mc_curiosity/registry.json\")); print(\"working-set goals:\", sum(1 for n in b[\"slot_names\"] if n), \"| skills:\", len(r))"'
```

## 5. Launch

### MineRL mode (default)

```bash
$SSH 'cd /workspace/devai && bash scripts/launch_lifelong.sh 1000000'
```

4 local envs (env 0 = the continuous lifelong stream + 3 scouts), config
`configs/minecraft_lifelong.yaml`, log `podlogs/minecraft_lifelong_run.log`.

### External-server mode

First bring up the tunnel (see `SERVER_CONNECTION.md` for the full both-sides
setup), then:

```bash
$SSH 'cd /workspace/devai && bash scripts/connect_server.sh login'     # approve the URL in a browser
$SSH 'cd /workspace/devai && bash scripts/connect_server.sh bridge <server-tailscale-ip>'
$SSH 'cd /workspace/devai && bash scripts/launch_skybot.sh 1000000'    # pre-flight refuses if tunnel down
```

`launch_skybot.sh` uses `configs/minecraft_skybot.yaml` (env 0 joins the server,
1 local scout); it will not start unless tailscaled + the bridge + the server
ping all pass.

## 6. Verify a healthy run

```bash
$SSH 'cd /workspace/devai
  echo "java clients: $(pgrep -x java | wc -l)  (want 4 MineRL / 2 SkyBot)"
  echo "each pinned 1 core:"; for p in $(pgrep -x java); do taskset -cp $p; done
  echo "futex aborts: $(cat logs/mc_*.log 2>/dev/null | grep -ac "futex facility")  (want 0)"
  echo "GPU:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  echo "async WM: $(grep -ac "async WM trainer started" podlogs/*run.log)"
  echo "prospection: $(grep -ac "Prospection re-rank" podlogs/*run.log) decisions"'
```

Boot takes ~2 min for the GPU clients. Healthy = 4 (or 2) java each pinned to
its own core, GPU climbing, **0 futex aborts**, per-episode reward lines, and
(with the features on) `async WM trainer started` + periodic `Prospection`
lines.

## 7. Stopping / restarting

- **Graceful (lifelong forever-mode only):** `touch podlogs/STOP`. Budgeted runs
  (`forever: false`) need SIGTERM; if it doesn't exit in ~3 min, SIGKILL is safe
  — the skill bank writes atomically (tmp+fsync+rename), so a hard kill loses
  only in-memory WM/policy state (which is from-scratch each run anyway; the
  brain persists).
- **Restart** = fresh WM + policy, **same brain**. `logs/checkpoints/` holds the
  WM/policy snapshots but there is currently no auto-resume — the brain (skill
  bank) is the durable state.

## 8. Config features (both modes)

`configs/minecraft_lifelong.yaml` and `minecraft_skybot.yaml` both enable:

```yaml
async_wm:            # acting-while-learning: WM consolidation on a bg thread
  enabled: true      #   (removes the 20-50s per-cadence freeze; PPO stays sync)
prospection:         # imagine-before-committing at goal selection
  enabled: true
  weight: 0.5        #   blend vs the retrospective frontier score
  top_m: 4           #   re-rank the frontier's top-M candidates
  rollouts: 16       #   K imagined rollouts per candidate
  horizon: 15        #   H latent steps each
```

Both default-OFF when the block is absent, so every other config is unaffected.
See `../../developmental_ai/core/developmental_loop.py` (`_prospective_goal_score`,
`_ensure_wm_trainer`) and the memory notes for the design + the 11 adversarial-
review fixes.
