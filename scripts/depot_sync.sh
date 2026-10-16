#!/bin/bash
# Sync the LAN game depot between the RUNNER (node1) and the training host.
#
# WHY (2026-10-14): a game install on `main` downloaded at ~127 kB/s and then
# died on four packages from esm.ubuntu.com. Game files, .debs and wheels only
# ever need to cross the internet ONCE; after that they should come from a
# machine on the LAN. The runner already holds the only SSH key to the host,
# so the depot lives next to it and needs no new credentials anywhere.
#
#   bash scripts/depot_sync.sh to-host     runner depot -> <host>/workspace/devai/depot
#   bash scripts/depot_sync.sh from-host   host depot -> runner depot (harvest what
#                                          an install just downloaded)
#   bash scripts/depot_sync.sh status      sizes on both sides
#
#   GAME_DEPOT   the runner-side directory. Default /home/runner/_work/depot:
#                inside the runner's existing persistent volume, so it works
#                with ZERO setup and survives restarts. To use a folder you can
#                drop files into by hand (a Chrome/Steam .deb, a copied game
#                library, or a disk on another machine mounted on node1), bind
#                it at /home/runner/depot (docker/runner/docker-compose.yml)
#                and set GAME_DEPOT=/home/runner/depot in docker/runner/.env.
#   MAIN_HOST / MAIN_USER / MAIN_SSH_PORT / MAIN_SSH_KEYFILE / DEPLOY_TRANSPORT
#                exactly as scripts/host_exec.sh
#
# NEVER BLOCKS AN INSTALL. No runner depot -> a notice and exit 0: the host's
# own depot and the internet still work (tools/game_install.py). Never deletes
# on either side (no --delete): the depot only grows, and pruning is a human
# decision. Layout and what reads it: tools/game_install.py.
set -uo pipefail

MODE="${1:?usage: depot_sync.sh <to-host|from-host|status>}"
DEPOT="${GAME_DEPOT:-/home/runner/_work/depot}"
MAIN_USER="${MAIN_USER:-root}"
: "${MAIN_HOST:?MAIN_HOST not set (runner .env)}"
REMOTE="${SKYBOT_REMOTE_ROOT:-/workspace/devai}/depot"

mkdir -p "$DEPOT" 2>/dev/null || true
if [ ! -d "$DEPOT" ]; then
  echo "depot: no runner depot at $DEPOT and it cannot be created — skipping"
  echo "  (installs still use the host's own depot and the internet)."
  echo "  See docs/OPERATIONS.md, 'Game depot'."
  exit 0
fi

if [ "${DEPLOY_TRANSPORT:-ssh}" = "tailscale" ]; then
  RSH="tailscale ssh"
  SSH=(tailscale ssh)
else
  : "${MAIN_SSH_PORT:?MAIN_SSH_PORT not set}"
  KEY="${MAIN_SSH_KEYFILE:-$HOME/.ssh/skybot_ed25519}"
  OPTS="-p $MAIN_SSH_PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
  RSH="ssh $OPTS"
  # shellcheck disable=SC2206
  SSH=(ssh $OPTS)
fi
DEST="${MAIN_USER}@${MAIN_HOST}"
# apt's in-flight downloads and its lock are not depot content
EXCL=(--exclude 'apt/partial/' --exclude 'apt/lock')

size() { du -sh "$1" 2>/dev/null | cut -f1; }

case "$MODE" in
  to-host)
    echo "depot: runner $DEPOT ($(size "$DEPOT")) -> host $REMOTE"
    "${SSH[@]}" "$DEST" "mkdir -p '$REMOTE'" || { echo "depot: cannot reach the host"; exit 1; }
    # --ignore-existing both ways: depot files never change once written, and
    # apt leaves its .debs root-owned on the host, which this login could not
    # overwrite anyway.
    rsync -rlt --partial --ignore-existing --stats "${EXCL[@]}" -e "$RSH" "$DEPOT/" "$DEST:$REMOTE/"
    ;;
  from-host)
    if [ ! -w "$DEPOT" ]; then
      echo "depot: $DEPOT is not writable by $(id -un) (uid $(id -u)) — nothing harvested."
      echo "  On node1: sudo chown -R $(id -u):$(id -g) <the folder bound at $DEPOT>"
      exit 1
    fi
    echo "depot: host $REMOTE -> runner $DEPOT"
    # --ignore-existing: files in the depot never change once there (a .deb or
    # wheel name carries its version; game files are re-added under a new path),
    # so this only ever ADDS, and a half-written host file cannot clobber a
    # good copy.
    rsync -rlt --partial --ignore-existing --stats "${EXCL[@]}" -e "$RSH" \
      "$DEST:$REMOTE/" "$DEPOT/"
    echo "depot: runner side now $(size "$DEPOT")"
    ;;
  status)
    echo "runner: $DEPOT $(size "$DEPOT")"
    "${SSH[@]}" "$DEST" "du -sh '$REMOTE' 2>/dev/null || echo 'host: no depot yet'"
    ;;
  *)
    echo "usage: depot_sync.sh <to-host|from-host|status>"; exit 2 ;;
esac
