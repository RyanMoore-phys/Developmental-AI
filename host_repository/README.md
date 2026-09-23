# Developmental-AI — Pod Repository & Operations Reference

Single source of truth for **building, updating, running, and transferring** the
Developmental-AI Minecraft organism on a RunPod. Created 2026-07-22 while
rescuing the brain off a GPU pod that was swapped for a CPU pod.

The **goal** of the system: an agent that **fully learns Minecraft (MineRL
Treechop first) with NO videos/demos** — pure developmental learning from a
curiosity engine + a curiosity-ranked vision "magnet" + discovered goals →
minted skills + a DreamerV3-style world model + PPO + local-VLM symbolic
grounding, running as a **lifelong continuous stream** (no episodic resets).

## What's in here

```
host_repository/
├── README.md            ← you are here (index + quickstart + connection)
├── docs/
│   ├── RECREATE.md        ★ full end-to-end pod recreation (start here)
│   ├── SERVER_CONNECTION.md ★ external Minecraft server mode (tailscale, both sides)
│   ├── ARCHITECTURE.md   the organism: every subsystem and how they wire
│   ├── PROVISIONING.md   build a brand-new pod from scratch (the fragile parts)
│   ├── RUNNING.md        launch / monitor / stop a training run
│   ├── TRANSFER.md       move to a new pod + restore the brain (this rescue)
│   └── STATE.md          current brain state + audit-fix status + what's next
└── data/                 ← RESCUED, IRREPLACEABLE brain (the "never wipe" data)
    ├── skill_bank_mc_curiosity/   3.5 GB: goals + skills + policy weights
    ├── registry.json              skill index
    ├── broadcaster_state.json     goal index
    ├── configs/                   minecraft_lifelong.yaml, minecraft_skybot.yaml
    ├── minerl_build/              cached VirtualGL .deb (reproducible provisioning)
    ├── scripts/                   provision_host.sh, launch_lifelong.sh,
    │                              launch_skybot.sh, connect_server.sh,
    │                              mc_ping.py, rcon.py
    └── logs/                      run diagnostic logs
```

**Two environment modes** (2026-07-24): MineRL local worlds (`launch_lifelong.sh`,
default, most stable) or an external Minecraft server you own over a Tailscale
tunnel (`connect_server.sh` + `launch_skybot.sh`). Both run the same organism +
the acting-while-learning (`async_wm`) and imagine-before-committing
(`prospection`) features. **New pod? Follow `docs/RECREATE.md`.**

The **live code** (`developmental_ai/`, `configs/`, `scripts/`, smoke tests,
`AUDIT_FINDINGS.md`) lives in the parent repo `../` and is byte-synced to the
pod. This folder holds the pod-side DATA + the operational knowledge.

## Connection

**Addresses are NOT written down here any more (2026-09-02).** Both the host
and the port change on every pod rebuild, so any literal in this file is stale
the moment it is written — and a stale address reads exactly like a dead run
(see the port note below). Set them in your shell, or read them from the CI
variables `MAIN_HOST` / `MAIN_SSH_PORT`, which are the single source of truth:

```bash
export MAIN_HOST=<current-pod-ip>      # RunPod dashboard -> Connect
export POD_PORT=<current-ssh-port>
ssh root@"$MAIN_HOST" -p "$POD_PORT" -i ~/.ssh/skybot_ed25519 \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
```

Once the pod is on the tailnet you can skip host/port entirely — the MagicDNS
name is stable across rebuilds and Tailscale SSH needs no key:

```bash
tailscale ssh root@<pod-magicdns-name>     # e.g. devai-pod-2
```

- **Key:** `~/.ssh/skybot_ed25519` (the `~/.ssh/id_ed25519` path does NOT exist
  on this Mac, despite what the dashboard command prints).
- **`UserKnownHostsFile=/dev/null` is REQUIRED:** RunPod reuses IPs across
  pods, so the host key changes and plain SSH refuses
  with "REMOTE HOST IDENTIFICATION HAS CHANGED" — this flag bypasses it.
- **Port changes every pod restart** (22655 → 22681 → 34276 → …). Get the
  current port from the RunPod dashboard → Connect → "SSH over exposed TCP",
  or read it off a live watcher without leaving the shell:
  `pgrep -fl ssh | grep -o '\-p [0-9]*' | head -1`.
  **Every hard-coded port in this file is stale the moment the pod restarts.**
  This bites in a specific, dangerous way: the training pid is a POD pid, so a
  local `ps`/`grep` for it finds NOTHING and reads exactly like "the run died".
  Confirm you are on the right host and port before concluding a run is dead.
- Working dir on the pod: `/workspace/devai` (a **persistent MooseFS network
  volume** — survives pod stop/start, which is why the brain was recoverable).
- App venv: `./venv_mc` (Python 3.10 + torch). **Never use conda/base.**

## 60-second quickstart on a healthy pod

```bash
# 1. sync code from the Mac (run from the parent repo dir)
#    MAIN_HOST/POD_PORT as exported above — no literals, they go stale.
rsync -rlptz -e "ssh -p $POD_PORT -i ~/.ssh/skybot_ed25519 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  developmental_ai configs scripts run_minecraft.py \
  root@"$MAIN_HOST":/workspace/devai/

#    or just: MAIN_HOST=<magicdns-name> DEPLOY_TRANSPORT=tailscale \
#               bash scripts/deploy_skybot.sh

# 2. launch (idempotent; refuses if a run or stale java is alive)
ssh <conn> 'cd /workspace/devai && bash scripts/launch_lifelong.sh 1000000'

# 3. watch
ssh <conn> 'tail -f /workspace/devai/runlogs/minecraft_lifelong_run.log'
```

See `docs/RUNNING.md` for the full launch/monitor/stop workflow and the exact
signals to watch, and `docs/PROVISIONING.md` if the pod is fresh (no venv_mc /
no built MineRL jar).

## The one rule

**NEVER wipe `skill_bank_mc_curiosity/`** — it is the agent's accumulated
developmental memory (discovered goals, minted skills, policy weights,
competence, unlock history). There is no way to regenerate it. `data/` here is
its backup as of 2026-07-22.
