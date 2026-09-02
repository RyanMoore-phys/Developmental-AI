#!/bin/bash
# Sync this Mac's code to the pod and (re)launch SkyBot — one command.
#
# WHY THIS EXISTS: the pod's SSH PORT ROTATES on every restart (22655 -> 22681
# -> 34276 -> 19473 -> ...), so the rsync+launch recipe in docs/RUNNING.md goes
# stale the moment the pod bounces and has to be retyped from two places.
# Pass the current host/port once and everything downstream is derived.
#
# Usage:
#   bash scripts/deploy_skybot.sh <PORT> [HOST] [TIMESTEPS]
#   POD_HOST=1.2.3.4 bash scripts/deploy_skybot.sh 34276
#   bash scripts/deploy_skybot.sh 34276 1.2.3.4 4000000
#
#   # CI: over the tailnet, no key, no port, and DO NOT start training
#   DEPLOY_TRANSPORT=tailscale DEPLOY_LAUNCH=0 POD_HOST=devai-pod-2 \
#     bash scripts/deploy_skybot.sh
#
# It refuses to launch if the pre-flight fails (dead tailscale/socat bridge or
# an unreachable Minecraft server) — launch_skybot.sh owns that check, and a
# run started against a dead bridge strands the primary stream in a
# connect-fail rebuild loop.
set -euo pipefail

# TRANSPORT (added 2026-09-02 for CI). Two ways in:
#   ssh       (default) — public endpoint + key. PORT is required and rotates.
#   tailscale — `tailscale ssh root@<magicdns-name>`. PORT and KEY are unused;
#               HOST is the stable MagicDNS name, so the rotating-port problem
#               this script was written to work around simply disappears.
# The pod has no /dev/net/tun, so tailscaled is userspace-only and raw inbound
# TCP over the tailnet is NOT routable — Tailscale SSH is the only tailnet way
# in. See scripts/connect_server.sh for the matching --ssh flag and ACL note.
TRANSPORT="${DEPLOY_TRANSPORT:-ssh}"
# LAUNCH CONTROL: DEPLOY_LAUNCH=0 syncs + smoke-tests and STOPS. CI uses this
# so a push can never restart live training; a human launching stays the
# default. The stop/sync/smoke steps run either way.
LAUNCH="${DEPLOY_LAUNCH:-1}"

if [ "$TRANSPORT" = "tailscale" ]; then
  PORT="${1:-n/a}"
  HOST="${2:-${POD_HOST:?set POD_HOST (MagicDNS name) or pass HOST}}"
  TS="${3:-4000000}"
  SSH_CMD="tailscale ssh"
  RSH="tailscale ssh"
else
  PORT="${1:?usage: deploy_skybot.sh <PORT> [HOST] [TIMESTEPS]}"
  # No hardcoded IP default any more — it went stale on every pod rebuild and
  # put a real endpoint in the tree. POD_HOST is the single source of truth.
  HOST="${2:-${POD_HOST:?set POD_HOST or pass HOST as arg 2}}"
  TS="${3:-4000000}"
  KEY="${POD_SSH_KEY:-$HOME/.ssh/skybot_ed25519}"
  # UserKnownHostsFile=/dev/null is REQUIRED: RunPod reuses IPs across pods, so
  # the host key changes and plain ssh refuses with a HOST KEY CHANGED error.
  SSH_OPTS="-p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
  SSH_CMD="ssh $SSH_OPTS"
  RSH="ssh $SSH_OPTS"
fi

echo "==> target root@$HOST:$PORT via $TRANSPORT (launch=$LAUNCH)"
$SSH_CMD "root@$HOST" 'echo "    pod reachable: $(hostname)"'

echo "==> stopping any live run (a second run would fight for the clients)"
# STOP-FILE FIRST so the supervisor treats this as deliberate, not a crash,
# and does not immediately relaunch the old code underneath us.
# BY PID, NEVER `pkill -f run_minecraft` — a pattern match on that string also
# matches the cmdline of THIS SSH INVOCATION, so the remote shell kills itself
# and, under `set -e`, the deploy silently aborts leaving the old run alive.
# supervise_skybot.sh documents the same self-match trap for pgrep. The
# launcher writes podlogs/skybot_run.pid; that PID is the only safe handle.
$SSH_CMD "root@$HOST" '
  cd /workspace/devai || exit 0
  touch podlogs/STOP 2>/dev/null || true       # deliberate stop, not a crash
  # Supervisor first, or it relaunches the old code underneath us.
  # THE PATTERN IS SPLIT ON PURPOSE. Writing it literally puts the string in
  # THIS shell own cmdline, so `pkill -f` matches the shell, kills it, and the
  # deploy dies right here with the old run still up (it did, twice).
  # Concatenating keeps the literal out of /proc/self/cmdline.
  SUP="supervise""_skybot"
  pkill -f "$SUP" 2>/dev/null || true
  sleep 1
  PID=$(cat podlogs/skybot_run.pid 2>/dev/null || true)
  if [ -n "$PID" ] && ps -p "$PID" -o args= 2>/dev/null | grep -q minecraft; then
    echo "    stopping agent pid=$PID (SIGTERM: it finishes the segment)"
    kill "$PID" 2>/dev/null || true
    for i in $(seq 1 30); do ps -p "$PID" >/dev/null 2>&1 || break; sleep 2; done
    ps -p "$PID" >/dev/null 2>&1 && { echo "    still up -> SIGKILL";       kill -9 "$PID" 2>/dev/null || true; sleep 3; }
  else
    echo "    no live agent recorded in podlogs/skybot_run.pid"
  fi
  pkill -9 -x java 2>/dev/null || true          # -x: exact name, cannot self-match
  sleep 3
  LEFT=$(ps -eo args | grep -c "[r]un_minecraft")
  echo "    stopped (agent procs left=$LEFT java=$(pgrep -cx java))"
  [ "$LEFT" -eq 0 ] || { echo "    REFUSED: an agent process survived"; exit 1; }'

echo "==> syncing code (-rlptz, NOT -az: the network volume forbids chown)"
# --stats, NOT --info=stats1: macOS ships rsync 2.6.9, which predates --info=
rsync -rlptz --stats \
  -e "$RSH" \
  developmental_ai configs scripts tests run_minecraft.py \
  "root@$HOST:/workspace/devai/"

echo "==> smoke-testing ON THE POD (the only tree that matters)"
$SSH_CMD "root@$HOST" '
  cd /workspace/devai
  ./venv_mc/bin/python tests/_reward_fixes_smoke.py \
    && ./venv_mc/bin/python tests/_centering_drive_smoke.py \
    && ./venv_mc/bin/python tests/_resume_smoke.py \
    && ./venv_mc/bin/python tests/_stall_fixes_smoke.py' \
  || { echo "REFUSED: pod-side smoke test failed — not launching"; exit 1; }

if [ "$LAUNCH" != "1" ]; then
  # DELIBERATE STOP-SHORT (DEPLOY_LAUNCH=0). The code is synced and the
  # pod-side smokes passed, but nothing is started. Note the STOP-file is
  # left in place from the stop step above, so the supervisor stays down and
  # the pod sits idle until someone launches on purpose.
  echo "==> DEPLOY_LAUNCH=0 — synced and smoke-tested, NOT launching."
  echo "    podlogs/STOP is still set; the supervisor will not come back by itself."
  echo "    launch with: DEPLOY_LAUNCH=1 bash scripts/deploy_skybot.sh $PORT $HOST $TS"
  exit 0
fi

echo "==> clearing the stop-file and launching"
$SSH_CMD "root@$HOST" "
  cd /workspace/devai
  rm -f podlogs/STOP
  setsid nohup bash scripts/supervise_skybot.sh $TS \
    > podlogs/supervisor.log 2>&1 < /dev/null &
  sleep 25
  echo '    --- supervisor ---'; tail -5 podlogs/supervisor.log
  echo \"    run pid: \$(cat podlogs/skybot_run.pid 2>/dev/null || echo none)\""

echo "==> done. Watch it with:"
echo "    $SSH_CMD root@$HOST 'tail -f /workspace/devai/podlogs/minecraft_skybot_run.log'"
