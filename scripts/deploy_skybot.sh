#!/bin/bash
# Sync this Mac's code to the training host and (re)launch SkyBot — one command.
#
# WHY THIS EXISTS: the host's SSH PORT ROTATES on every restart (22655 -> 22681
# -> 34276 -> 19473 -> ...), so the rsync+launch recipe in docs/RUNNING.md goes
# stale the moment the training host bounces and has to be retyped from two places.
# Pass the current host/port once and everything downstream is derived.
#
# Usage:
#   bash scripts/deploy_skybot.sh <PORT> [HOST] [TIMESTEPS]
#   MAIN_HOST=1.2.3.4 bash scripts/deploy_skybot.sh 34276
#   bash scripts/deploy_skybot.sh 34276 1.2.3.4 4000000
#
#   # CI: over the tailnet, no key, no port, and DO NOT start training
#   DEPLOY_TRANSPORT=tailscale DEPLOY_LAUNCH=0 MAIN_HOST=devai-training host-2 \
#     bash scripts/deploy_skybot.sh
#
# It refuses to launch if the pre-flight fails (dead tailscale/socat bridge or
# an unreachable Minecraft server) — launch_skybot.sh owns that check, and a
# run started against a dead bridge strands the primary stream in a
# connect-fail rebuild loop.
set -euo pipefail

# ---- WHO WE LOG IN AS (added 2026-09-22) ----------------------------------
# Was hardcoded to root, because a rented GPU host only ever gave you root. The owned box
# `main` is a normal Ubuntu machine with a normal account, so the user is now a
# variable. Default stays `root` so nothing that used to work stops working.
# NOTE: a non-root MAIN_USER needs passwordless sudo on the target -- provision
# installs apt packages, and /workspace is created under / which it cannot
# write. provision_host.sh picks up sudo automatically (see SUDO= in it).
MAIN_USER="${MAIN_USER:-root}"

# TRANSPORT (added 2026-09-02 for CI). Two ways in:
#   ssh       (default) — public endpoint + key. PORT is required and rotates.
#   tailscale — `tailscale ssh <user>@<magicdns-name>`. PORT and KEY are unused;
#               HOST is the stable MagicDNS name, so the rotating-port problem
#               this script was written to work around simply disappears.
# The training host has no /dev/net/tun, so tailscaled is userspace-only and raw inbound
# TCP over the tailnet is NOT routable — Tailscale SSH is the only tailnet way
# in. See scripts/connect_server.sh for the matching --ssh flag and ACL note.
TRANSPORT="${DEPLOY_TRANSPORT:-ssh}"
# LAUNCH CONTROL: DEPLOY_LAUNCH=0 syncs + smoke-tests and STOPS. CI uses this
# so a push can never restart live training; a human launching stays the
# default. The stop/sync/smoke steps run either way.
LAUNCH="${DEPLOY_LAUNCH:-1}"

if [ "$TRANSPORT" = "tailscale" ]; then
  PORT="${1:-n/a}"
  HOST="${2:-${MAIN_HOST:?set MAIN_HOST (MagicDNS name) or pass HOST}}"
  TS="${3:-4000000}"
  SSH_CMD="tailscale ssh"
  RSH="tailscale ssh"
else
  PORT="${1:?usage: deploy_skybot.sh <PORT> [HOST] [TIMESTEPS]}"
  # No hardcoded IP default any more — it went stale on every host rebuild and
  # put a real endpoint in the tree. MAIN_HOST is the single source of truth.
  HOST="${2:-${MAIN_HOST:?set MAIN_HOST or pass HOST as arg 2}}"
  TS="${3:-4000000}"
  # MAIN_SSH_KEYFILE is the spelling the runner .env, both workflows,
  # host_exec.sh and setup_runner.sh all use; this script alone read
  # MAIN_SSH_KEY. Both defaulted to the same path, so nothing showed --
  # until a host with a different key, where this would silently ignore
  # the .env and fail with a bare "Permission denied (publickey)".
  KEY="${MAIN_SSH_KEYFILE:-${MAIN_SSH_KEY:-$HOME/.ssh/id_ed25519}}"
  # UserKnownHostsFile=/dev/null is REQUIRED: a rented GPU host reuses IPs across hosts, so
  # the host key changes and plain ssh refuses with a HOST KEY CHANGED error.
  # BatchMode=yes IS LOAD-BEARING IN CI (added 2026-09-02).
  # ConnectTimeout only bounds the TCP connect, NOT authentication. Without
  # BatchMode, a rejected key makes ssh fall back to keyboard-interactive and
  # BLOCK forever on a password prompt no CI job can answer — the job dies at
  # its timeout-minutes with the step still showing "in progress" and no error
  # anywhere. With it, ssh exits immediately: "Permission denied (publickey)".
  # ServerAlive* covers the other half: the host-side smoke tests run for
  # minutes over one connection, and a silently dropped link would hang just
  # as long.
  SSH_OPTS="-p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
  SSH_CMD="ssh $SSH_OPTS"
  RSH="ssh $SSH_OPTS"
fi

echo "==> target ${MAIN_USER}@$HOST:$PORT via $TRANSPORT (launch=$LAUNCH)"
$SSH_CMD "${MAIN_USER}@$HOST" 'echo "    host reachable: $(hostname)"'

# NO-LAUNCH MUST ALSO MEAN NO-STOP (2026-09-02). The stop below runs
# UNCONDITIONALLY and the DEPLOY_LAUNCH=0 early-exit is ~50 lines further
# down, so a sync-only deploy would STOP live training and then never restart
# it — strictly worse than restarting, and it would have fired automatically
# on every push to main once deploy.yml was wired up. DEPLOY_LAUNCH=0 means
# "I am not starting anything", and killing a run is never part of that
# intent. Refuse instead, and make the human decide.
if [ "$LAUNCH" != "1" ]; then
  _live=$($SSH_CMD "${MAIN_USER}@$HOST" 'pgrep -f "run_min""ecraft[.]py" | wc -l' 2>/dev/null | tr -d '[:space:]')
  if [ "${_live:-0}" != "0" ]; then
    if [ "${DEPLOY_ALLOW_STOP_LIVE:-0}" = "1" ]; then
      echo "==> WARNING: training is LIVE and DEPLOY_ALLOW_STOP_LIVE=1 — stopping it and NOT restarting"
    else
      echo "REFUSED: training is LIVE on $HOST and DEPLOY_LAUNCH=0."
      echo "  Syncing now would stop the run and leave it stopped."
      echo "  Stop it deliberately first (host.yml action=stop, or touch runlogs/STOP),"
      echo "  or re-run with DEPLOY_ALLOW_STOP_LIVE=1 if that is really what you want."
      exit 1
    fi
  fi
fi

echo "==> stopping any live run (a second run would fight for the clients)"
# STOP-FILE FIRST so the supervisor treats this as deliberate, not a crash,
# and does not immediately relaunch the old code underneath us.
# BY PID, NEVER `pkill -f run_minecraft` — a pattern match on that string also
# matches the cmdline of THIS SSH INVOCATION, so the remote shell kills itself
# and, under `set -e`, the deploy silently aborts leaving the old run alive.
# supervise_skybot.sh documents the same self-match trap for pgrep. The
# launcher writes runlogs/skybot_run.pid; that PID is the only safe handle.
$SSH_CMD "${MAIN_USER}@$HOST" '
  cd /workspace/devai || exit 0
  touch runlogs/STOP 2>/dev/null || true       # deliberate stop, not a crash
  # Supervisor first, or it relaunches the old code underneath us.
  # THE PATTERN IS SPLIT ON PURPOSE. Writing it literally puts the string in
  # THIS shell own cmdline, so `pkill -f` matches the shell, kills it, and the
  # deploy dies right here with the old run still up (it did, twice).
  # Concatenating keeps the literal out of /proc/self/cmdline.
  SUP="supervise""_skybot"
  pkill -f "$SUP" 2>/dev/null || true
  sleep 1
  PID=$(cat runlogs/skybot_run.pid 2>/dev/null || true)
  if [ -n "$PID" ] && ps -p "$PID" -o args= 2>/dev/null | grep -q minecraft; then
    echo "    stopping agent pid=$PID (SIGTERM: it finishes the segment)"
    kill "$PID" 2>/dev/null || true
    for i in $(seq 1 30); do ps -p "$PID" >/dev/null 2>&1 || break; sleep 2; done
    ps -p "$PID" >/dev/null 2>&1 && { echo "    still up -> SIGKILL";       kill -9 "$PID" 2>/dev/null || true; sleep 3; }
  else
    echo "    no live agent recorded in runlogs/skybot_run.pid"
  fi
  pkill -9 -x java 2>/dev/null || true          # -x: exact name, cannot self-match
  sleep 3
  LEFT=$(ps -eo args | grep -c "[r]un_minecraft")
  echo "    stopped (agent procs left=$LEFT java=$(pgrep -cx java))"
  [ "$LEFT" -eq 0 ] || { echo "    REFUSED: an agent process survived"; exit 1; }'

# ---- ENSURE THE DESTINATION EXISTS (added 2026-09-22) ----------------------
# rsync creates the LAST path component only, so a missing /workspace fails
# with a bare `mkdir "/workspace/devai" failed: No such file or directory (2)`
# and exit code 11 -- which reads like a permissions problem and is not one.
# On a rented GPU host /workspace was the platform-mounted network volume and always
# existed; on the OWNED Ubuntu host `main` (since 2026-09-22) NOTHING
# creates it -- provision_host.sh itself opens with `cd /workspace/devai ||
# exit 1`, so it cannot be what makes the directory either. Create it here and
# SAY SO, because a silently-created /workspace on the root filesystem is also
# how you would discover far too late that a data disk failed to mount.
echo "==> checking the destination tree"
$SSH_CMD "${MAIN_USER}@$HOST" '
  # SUDO, because / is not writable by a normal account and MAIN_USER is no
  # longer always root. Empty when we already are root -> unchanged old path.
  if [ "$(id -u)" -eq 0 ]; then S=""; else S="sudo"; fi
  if [ -d /workspace/devai ]; then
    echo "    /workspace/devai: present"
  else
    echo "    /workspace/devai: MISSING -> creating"
    $S mkdir -p /workspace/devai || { echo "    FAILED to create it"; exit 1; }
    # CHOWN IS LOAD-BEARING: rsync runs as MAIN_USER and would otherwise get
    # "Permission denied" writing into a root-owned directory it just made.
    [ -n "$S" ] && $S chown -R "$(id -u):$(id -g)" /workspace/devai
    echo "    created and chowned to $(whoami)."
    echo "    If /workspace was meant to be a SEPARATE DISK, it is not"
    echo "    mounted and this just made a directory on the OS disk."
  fi
  [ -w /workspace/devai ] || { echo "    NOT WRITABLE by $(whoami) -- rsync will fail"; exit 1; }
  df -h /workspace/devai | tail -1 | awk "{print \"    filesystem: \"\$1\"  size=\"\$2\"  avail=\"\$4}"
  [ -x /workspace/devai/venv_mc/bin/python ] \
    && echo "    venv_mc: present" || echo "    venv_mc: ABSENT (host not provisioned)"'

echo "==> syncing code (-rlptz, NOT -az: the network volume forbids chown)"
# --stats, NOT --info=stats1: macOS ships rsync 2.6.9, which predates --info=
# tools/ + experiments/ (2026-10-03): tests/_foundation_{baseline,ab_verify,
# ab_verify9_11}_smoke import or exec them; without them `run_all.py` on the
# host FAILS those suites on ModuleNotFoundError, which reads like a code bug.
rsync -rlptz --stats \
  -e "$RSH" \
  developmental_ai configs scripts tests tools experiments run_minecraft.py \
  "${MAIN_USER}@$HOST:/workspace/devai/"

echo "==> smoke-testing ON THE TRAINING HOST (the only tree that matters)"
$SSH_CMD "${MAIN_USER}@$HOST" '
  cd /workspace/devai
  # A MISSING INTERPRETER IS NOT A FAILING TEST. Without this the next line
  # dies "No such file or directory" and the caller reports "smoke test
  # failed", sending you to read test code on a box that has no venv at all.
  [ -x ./venv_mc/bin/python ] || {
    echo "    NOT PROVISIONED: /workspace/devai/venv_mc/bin/python is absent."
    echo "    This host has never been set up. Run, on the host:"
    echo "      cd /workspace/devai && nohup bash scripts/provision_host.sh \\"
    echo "        > runlogs/provision.log 2>&1 &   # 40-60 min, builds MineRL"
    exit 2; }
  ./venv_mc/bin/python tests/_reward_fixes_smoke.py \
    && ./venv_mc/bin/python tests/_centering_drive_smoke.py \
    && ./venv_mc/bin/python tests/_resume_smoke.py \
    && ./venv_mc/bin/python tests/_stall_fixes_smoke.py' \
  || { echo "REFUSED: host-side smoke test failed — not launching"; exit 1; }

if [ "$LAUNCH" != "1" ]; then
  # DELIBERATE STOP-SHORT (DEPLOY_LAUNCH=0). The code is synced and the
  # host-side smokes passed, but nothing is started. Note the STOP-file is
  # left in place from the stop step above, so the supervisor stays down and
  # the training host sits idle until someone launches on purpose.
  echo "==> DEPLOY_LAUNCH=0 — synced and smoke-tested, NOT launching."
  echo "    runlogs/STOP is still set; the supervisor will not come back by itself."
  echo "    launch with: DEPLOY_LAUNCH=1 bash scripts/deploy_skybot.sh $PORT $HOST $TS"
  exit 0
fi

echo "==> clearing the stop-file and launching"
$SSH_CMD "${MAIN_USER}@$HOST" "
  cd /workspace/devai
  rm -f runlogs/STOP
  setsid nohup bash scripts/supervise_skybot.sh $TS \
    > runlogs/supervisor.log 2>&1 < /dev/null &
  sleep 25
  echo '    --- supervisor ---'; tail -5 runlogs/supervisor.log
  echo \"    run pid: \$(cat runlogs/skybot_run.pid 2>/dev/null || echo none)\""

echo "==> done. Watch it with:"
echo "    $SSH_CMD ${MAIN_USER}@$HOST 'tail -f /workspace/devai/runlogs/minecraft_skybot_run.log'"
