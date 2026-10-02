# Replication guide

How to stand this up from bare machines. Every step ends with a check, because
most of the ways this breaks are silent.

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first if you want to know what you
are building; this document assumes you already want to build it.

---

## 0. What you need

Three roles. They can be three machines or fewer — the only hard requirement is
that the **training host** and the **Minecraft server** are not the same box.

| role | what it does | requirements |
|---|---|---|
| **training host** | runs the agent and the Minecraft clients | NVIDIA GPU, Ubuntu Server, passwordless `sudo`, SSH |
| **game server** | a vanilla/Paper Minecraft server the agent plays on | Java, reachable from the training host |
| **controller** | CI runner + monitoring | Docker, reachable from both |

### Sizing, from measurement rather than guesswork

The reference deployment is deliberately modest hardware, and every number
below was paid for:

| resource | reference | what happens if you are below it |
|---|---|---|
| **VRAM** | 8 GB | The world model alone peaks near 6.7 GB in fp32. Below 8 GB you must reduce `world_model.batch_size`, and the local VLM will not fit at all. |
| **System RAM** | 16 GB | Two Minecraft clients (~1 GB each), torch (~4 GB) and the replay buffer. At 16 GB the VLM had to be disabled — it held 8.3 GB and was OOM-killed four times. |
| **Disk** | 256 GB | ~35 GB for the build, the rest is replay buffer and logs. **All brain state lives here and nothing backs it up.** |
| **CPU** | 6 cores | `OMP_NUM_THREADS` is set to 6; the Java clients compete for the same cores. |

**Throughput is the real constraint, not capability.** The reference setup
realises ~0.6 environment steps/second against a theoretical ~10/s. At that
rate the configured 4M-step budget takes ~77 days. If you have better hardware,
spend it on throughput before anything else.

---

## 1. Training host

```bash
# on the training host, as a user with passwordless sudo
sudo mkdir -p /workspace/devai
sudo chown -R "$USER:$USER" /workspace
```

`/workspace` is not special — it is simply where everything installs. **Nothing
in this repo creates it**, and `provision_host.sh` begins with
`cd /workspace/devai || exit 1`, so it cannot be what makes it either. Creating
it is step one, and forgetting it produces `rsync` exit code 11, which reads
like a permissions error and is not one.

Copy the repo there, then:

```bash
cd /workspace/devai
nohup bash scripts/provision_host.sh > runlogs/provision.log 2>&1 &
tail -f runlogs/provision.log
```

This takes **40–60 minutes** — it builds MineRL from source, including a full
Gradle build of a patched Minecraft client.

### Check

Two markers decide whether to keep waiting:

- **STAGE 2**, ~5 minutes in, prints your GPU's compute capability, the torch
  wheel it chose, and runs a **real matmul**. You want `kernel OK`. A mismatched
  wheel passes `torch.cuda.is_available()` and then fails on first use, which is
  why the probe does arithmetic rather than asking a boolean.
- **`PROVISION-COMPLETE`** at the end. Any `PROVISION-FAILED` names its stage.

---

## 2. Game server

Run a Paper or vanilla server in **offline mode** (the agent has no Mojang
account). Note its address and port.

The agent always connects to `127.0.0.1:25565` on the training host; a `socat`
bridge forwards that to the real server. This indirection exists so the agent's
config never changes when the server moves.

```bash
# on the training host
bash scripts/connect_server.sh bridge <SERVER_ADDRESS>
```

The bridge picks its transport by address family: **RFC1918** addresses
(`192.168.*`, `10.*`, `172.16–31.*`) go over plain TCP on the LAN; anything
else is treated as a Tailscale address and uses `tailscale nc`.

### Check

```bash
ss -tln | grep 127.0.0.1:25565                       # bridge listening
python3 scripts/mc_ping.py 127.0.0.1 25565 754       # server answers through it
```

Both must pass. The launcher runs the same ping in its preflight and refuses to
start otherwise — a run launched against a dead bridge strands its primary
stream in a reconnect loop.

---

## 3. Controller — CI runner

The runner deploys code to the training host, so it holds an SSH key for it.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/skybot_ed25519 -N ""
ssh-copy-id -i ~/.ssh/skybot_ed25519.pub <HOST_USER>@<TRAINING_HOST>
ssh -i ~/.ssh/skybot_ed25519 -o IdentitiesOnly=yes <HOST_USER>@<TRAINING_HOST> 'sudo -n true && echo ok'
```

That last command must print `ok`. **Passwordless sudo on the training host is
required** when `MAIN_USER` is not root: provisioning installs apt packages and
writes under `/`. The provisioner runs detached with no tty, so a password
prompt would not block visibly — it would fail stages silently.

Then bring up the runner (see [`CI_SETUP.md`](CI_SETUP.md) for registration):

```bash
cd docker/runner
cp runner.env.example .env      # edit it
docker compose up -d
```

**All connection values live in that `.env`, never in the repo.** It is
gitignored. Every consumer guards with `:?`, so a missing variable fails loudly
at the preflight instead of expanding to `root@` and reporting an unresolvable
hostname.

---

## 4. Controller — monitoring

Two stacks, deliberately separate. See [`MONITORING.md`](MONITORING.md) for why.

```bash
cd docker/monitor
cp monitor.env.example .env     # edit it; CHANGE GRAFANA_PASSWORD
docker compose up -d --build
```

Then install host exporters on **every** machine you want graphed, including
the training host:

```bash
bash scripts/install_host_exporters.sh
```

It installs `node_exporter` (:9100) and, where an NVIDIA GPU is present,
`nvidia_gpu_exporter` (:9835), as systemd units bound to the LAN address. On a
machine without `nvidia-smi` it installs only the first.

Add each machine to `docker/monitor/prometheus.yml`, then restart Prometheus.

### Check

- `http://<controller>:9090/targets` — every target **UP**
- `http://<controller>:3000` — Grafana; the SkyBot dashboard is provisioned
- If a target times out rather than refusing the connection, that is a
  **firewall**, not a wrong address. Allow 9100/9835 from your monitoring host.

---

## 5. First run

```
Actions → host → action: status      # confirm bridge UP, nothing running
Actions → host → action: launch
```

**Order matters, and it is not obvious:** `launch` starts what is already on
the host — it moves **no code**. `deploy` is the only step that syncs, and it
*refuses* while training is live. So the cycle for any code change is:

```
stop  →  deploy  →  launch
```

### Check

`action: status` should show `training LIVE`, `supervisor: 1 proc`,
`java clients: 2`, and `socat bridge UP`. The GPU will look idle for the first
30–60 minutes: the replay buffer is filling and the world model has not started
training yet.

---

## Things that will bite you

- **`workflow_run` runs the workflow file from the DEFAULT branch.** Editing
  `deploy.yml` on a feature branch does nothing until it reaches the default
  branch, while `actions/checkout` still pins the *code* to your commit. You
  get new code under an old workflow.
- **Hand-copied deployment directories drift.** If you `cp` the compose files
  somewhere instead of checking the repo out, every future change needs copying
  by hand, and the failure mode is a variable name mismatch that looks like a
  network problem.
- **A missing remote file is not an error to `rsync`.** The metrics collector
  logs loudly when its source path is gone, because it previously failed
  silently for days.
- **Deleting `skill_bank_*/` destroys accumulated learning.** It is not a cache.
