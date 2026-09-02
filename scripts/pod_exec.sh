#!/bin/bash
# Run a command on the pod over whichever transport is configured.
#
# WHY THIS EXISTS: the same six-line if/else was about to be copy-pasted into
# five workflow steps, which is exactly how the two bodies in
# developmental_loop.py drifted apart (CLAUDE.md §4.2). One place to change.
#
#   POD_HOST           required. Pod IP (ssh) or MagicDNS name (tailscale).
#   DEPLOY_TRANSPORT   ssh (default) | tailscale
#   POD_SSH_PORT       required for ssh transport; ROTATES on every pod restart
#   POD_SSH_KEYFILE    defaults to ~/.ssh/skybot_ed25519
#
# ssh is the default on purpose: it needs only a key the maintainer's machine
# already has, whereas the tailscale path additionally needs a tailnet ACL
# rule. See docs/CI_SETUP.md.
#
# Usage:  bash scripts/pod_exec.sh 'cd /workspace/devai && ls'
set -euo pipefail

: "${POD_HOST:?POD_HOST not set (put it in the runner .env — see docs/CI_SETUP.md)}"
TRANSPORT="${DEPLOY_TRANSPORT:-ssh}"

if [ "$TRANSPORT" = "tailscale" ]; then
  command -v tailscale >/dev/null \
    || { echo "pod_exec: tailscale CLI not found on this host" >&2; exit 1; }
  exec tailscale ssh "root@${POD_HOST}" "$@"
fi

# NOTE: no apostrophes in these :? messages — bash parses the word inside
# ${var:?word} and a lone ' starts a quoted section, giving "unexpected EOF".
: "${POD_SSH_PORT:?POD_SSH_PORT not set - the pod SSH port rotates on every restart}"
KEY="${POD_SSH_KEYFILE:-$HOME/.ssh/skybot_ed25519}"
[ -f "$KEY" ] || { echo "pod_exec: no SSH key at $KEY" >&2; exit 1; }
# UserKnownHostsFile=/dev/null is REQUIRED: providers reuse IPs across pods, so
# the host key changes and plain ssh refuses with HOST KEY CHANGED.
exec ssh -p "$POD_SSH_PORT" -i "$KEY" \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 "root@${POD_HOST}" "$@"
