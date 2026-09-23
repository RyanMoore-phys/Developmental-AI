#!/bin/bash
# Run a command on the pod over whichever transport is configured.
#
# WHY THIS EXISTS: the same six-line if/else was about to be copy-pasted into
# five workflow steps, which is exactly how the two bodies in
# developmental_loop.py drifted apart (CLAUDE.md §4.2). One place to change.
#
#   MAIN_HOST           required. Pod IP (ssh) or MagicDNS name (tailscale).
#   DEPLOY_TRANSPORT   ssh (default) | tailscale
#   MAIN_SSH_PORT       required for ssh transport; ROTATES on every pod restart
#   MAIN_SSH_KEYFILE    defaults to ~/.ssh/skybot_ed25519
#
# ssh is the default on purpose: it needs only a key the maintainer's machine
# already has, whereas the tailscale path additionally needs a tailnet ACL
# rule. See docs/CI_SETUP.md.
#
# Usage:  bash scripts/host_exec.sh 'cd /workspace/devai && ls'
set -euo pipefail

# ---- WHO WE LOG IN AS (added 2026-09-22) ----------------------------------
# Was hardcoded to root, because RunPod only ever gave you root. The owned box
# `main` is a normal Ubuntu machine with a normal account, so the user is now a
# variable. Default stays `root` so nothing that used to work stops working.
# NOTE: a non-root MAIN_USER needs passwordless sudo on the target -- provision
# installs apt packages, and /workspace is created under / which it cannot
# write. provision_host.sh picks up sudo automatically (see SUDO= in it).
MAIN_USER="${MAIN_USER:-root}"

: "${MAIN_HOST:?MAIN_HOST not set (put it in the runner .env — see docs/CI_SETUP.md)}"
TRANSPORT="${DEPLOY_TRANSPORT:-ssh}"

if [ "$TRANSPORT" = "tailscale" ]; then
  command -v tailscale >/dev/null \
    || { echo "host_exec: tailscale CLI not found on this host" >&2; exit 1; }
  exec tailscale ssh "${MAIN_USER}@${MAIN_HOST}" "$@"
fi

# NOTE: no apostrophes in these :? messages — bash parses the word inside
# ${var:?word} and a lone ' starts a quoted section, giving "unexpected EOF".
: "${MAIN_SSH_PORT:?MAIN_SSH_PORT not set (22 on the owned host `main`)}"
KEY="${MAIN_SSH_KEYFILE:-$HOME/.ssh/skybot_ed25519}"
[ -f "$KEY" ] || { echo "host_exec: no SSH key at $KEY" >&2; exit 1; }
# UserKnownHostsFile=/dev/null is REQUIRED: providers reuse IPs across pods, so
# the host key changes and plain ssh refuses with HOST KEY CHANGED.
exec ssh -p "$MAIN_SSH_PORT" -i "$KEY" \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 \
  -o BatchMode=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  "${MAIN_USER}@${MAIN_HOST}" "$@"
