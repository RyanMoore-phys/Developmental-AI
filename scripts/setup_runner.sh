#!/bin/bash
# Install the `skybot` self-hosted GitHub Actions runner on this machine.
#
# Run it from anywhere:   bash scripts/setup_runner.sh
#
# It will prompt for a REGISTRATION TOKEN, which you get here:
#   https://github.com/<your-org>/<your-repo>/settings/actions/runners/new
# Copy only the token itself — the long string after `--token` in the snippet
# GitHub shows. It expires after ~1 hour and is single-use; if this script
# fails partway, just fetch a fresh one.
#
# The token is read with `read -s` rather than taken as an argument, so it
# never lands in your shell history.
#
# INSTALLS OUTSIDE THE REPO (~/actions-runner) on purpose: the runner's
# working directories must never be committable. .gitignore blocks
# actions-runner/ and _work/ as a second line of defence, but not being inside
# the repo at all is the real fix.
set -euo pipefail

REPO_URL="https://github.com/<your-org>/<your-repo>"
RUNNER_VERSION="2.337.0"
RUNNER_DIR="$HOME/actions-runner"
RUNNER_NAME="${RUNNER_NAME:-skybot-$(hostname -s)}"
# LABEL IS LOAD-BEARING: .github/workflows/{deploy,training host}.yml target
# `runs-on: [self-hosted, skybot]`. Without `skybot` the jobs queue forever
# with no error — GitHub simply never finds a matching runner.
LABELS="skybot"

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)   ARCH="osx-arm64" ;;
  Darwin/x86_64)  ARCH="osx-x64" ;;
  Linux/x86_64)   ARCH="linux-x64" ;;
  Linux/aarch64)  ARCH="linux-arm64" ;;
  *) echo "unsupported platform: $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac
TARBALL="actions-runner-${ARCH}-${RUNNER_VERSION}.tar.gz"
URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${TARBALL}"

echo "==> runner ${RUNNER_VERSION} (${ARCH}) -> ${RUNNER_DIR}"

if [ -d "$RUNNER_DIR" ] && [ -f "$RUNNER_DIR/config.sh" ]; then
  echo "    ${RUNNER_DIR} already exists."
  echo "    To reconfigure: cd ${RUNNER_DIR} && ./svc.sh stop && ./svc.sh uninstall && ./config.sh remove"
  echo "    Then re-run this script."
  exit 1
fi

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if [ ! -f "$TARBALL" ]; then
  echo "==> downloading"
  curl -fL -o "$TARBALL" "$URL"
fi
echo "==> extracting"
tar xzf "$TARBALL"

echo ""
echo "==> paste the REGISTRATION TOKEN (input hidden)"
echo "    get it from: ${REPO_URL}/settings/actions/runners/new"
read -r -s -p "    token: " TOKEN
echo ""
[ -n "$TOKEN" ] || { echo "no token given"; exit 1; }

echo "==> configuring as '${RUNNER_NAME}' with labels: self-hosted,${ARCH},${LABELS}"
./config.sh \
  --url "$REPO_URL" \
  --token "$TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "$LABELS" \
  --work _work \
  --unattended \
  --replace
unset TOKEN

# ---- the .env the workflows read ------------------------------------------
# Values are NOT secrets, but they live here rather than in the repo so a training host
# rebuild is a one-file edit and nothing connection-specific is ever committed.
if [ -f "$RUNNER_DIR/.env" ]; then
  echo "==> .env already exists, leaving it alone"
else
  echo "==> writing .env template"
  cat > "$RUNNER_DIR/.env" <<EOF
# Read by the runner at SERVICE START and applied to every job.
# After editing:  cd ~/actions-runner && ./svc.sh stop && ./svc.sh start

# --- required ---
# THE TRAINING HOST. Since 2026-09-22 this is the owned Ubuntu box `main` on
# the LAN, NOT a rented host -- which is why these are MAIN_* and not POD_*.
# The port no longer rotates (that was a rented GPU host remapping 22 on every restart);
# it is a fixed sshd on a fixed address, so this file should now be stable.
MAIN_HOST=CHANGE_ME   # the training host's LAN IP. Never commit the real value.
MAIN_SSH_PORT=22
# The login account on `main`. A NON-ROOT user needs PASSWORDLESS SUDO there:
# provisioning installs apt packages and creates /workspace under /.
MAIN_USER=skybot
MAIN_SSH_KEYFILE=$HOME/.ssh/id_ed25519

# --- required for: host.yml action=connect ---
MC_SERVER_TS_IP=100.64.0.11

# --- optional ---
# Use Tailscale SSH instead of plain ssh. Needs the tailnet ACL rule in
# docs/CI_SETUP.md section 4, and MAIN_HOST becomes the MagicDNS name.
# Worth it only to stop maintaining MAIN_SSH_PORT above.
#DEPLOY_TRANSPORT=tailscale
#TS_AUTHKEY=tskey-auth-...
EOF
  chmod 600 "$RUNNER_DIR/.env"
fi

echo "==> installing as a login service"
./svc.sh install
./svc.sh start
sleep 3
./svc.sh status || true

cat <<EOF

============================================================
Runner installed.

  dir:    ${RUNNER_DIR}
  name:   ${RUNNER_NAME}
  labels: self-hosted, ${ARCH}, ${LABELS}
  env:    ${RUNNER_DIR}/.env   (MAIN_HOST / MAIN_SSH_PORT / MAIN_SSH_KEYFILE)

CHECK IT:
  1. ${REPO_URL}/settings/actions/runners  -> should show "Idle"
  2. bash scripts/host_exec.sh 'hostname'   -> should print the training host hostname
     (run with the same env: set -a; . ~/actions-runner/.env; set +a)
  3. Actions -> host -> Run workflow -> action=status

macOS CAVEAT: svc.sh installs a LaunchAgent, so the runner only runs while
you are LOGGED IN and the Mac is AWAKE. If it sleeps, queued jobs simply wait.
To keep it available:  System Settings -> Lock Screen / Energy -> prevent
sleep, or run \`caffeinate -s\` while you expect jobs.

MANAGE:
  cd ${RUNNER_DIR}
  ./svc.sh status | stop | start
  ./svc.sh uninstall && ./config.sh remove   # full removal
============================================================
EOF
