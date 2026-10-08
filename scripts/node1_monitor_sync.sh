#!/usr/bin/env bash
# Keep node1's monitor stack current without a manual `git pull` (2026-10-08).
#
# WHY A LOCAL JOB, NOT THE RUNNER: the Actions runner on node1 is a container
# with NO access to the host's Docker, deliberately -- on a public repo,
# mounting docker.sock into it would hand root on node1 to any workflow that
# reaches it. The runner never needs this anyway: every job does its own
# actions/checkout. Only the LONG-LIVED monitor stack (docker/monitor:
# collector, ingest.py, Grafana dashboards/alerts) runs from a static
# checkout, so only it goes stale.
#
# Each run: fetch; if origin/<branch> moved, fast-forward ONLY (never merges,
# never discards local work -- a dirty or diverged checkout is reported and
# left alone); if anything under docker/monitor changed, rebuild/restart the
# stack. Idempotent, quiet when nothing changed, one lock against overlap.
#
# Install on node1 (macOS, launchd; runs every 10 min, also at login):
#   bash scripts/node1_monitor_sync.sh --install
# Run once by hand:
#   bash scripts/node1_monitor_sync.sh
# Remove:
#   bash scripts/node1_monitor_sync.sh --uninstall
set -u
BRANCH="${MONITOR_SYNC_BRANCH:-main}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.skybot.monitor-sync"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$REPO/runlogs/monitor_sync.log"

if [ "${1:-}" = "--install" ]; then
  mkdir -p "$HOME/Library/LaunchAgents" "$REPO/runlogs"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string><string>$REPO/scripts/node1_monitor_sync.sh</string>
  </array>
  <key>StartInterval</key><integer>600</integer>
  <key>RunAtLoad</key><true/>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
EOF
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST" && echo "installed: $PLIST (every 10 min, log: $LOG)"
  exit 0
fi
if [ "${1:-}" = "--uninstall" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST" && echo "removed $PLIST"
  exit 0
fi

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$REPO" || exit 1
LOCK="$REPO/runlogs/.monitor_sync.lock"
mkdir -p "$REPO/runlogs"
if ! mkdir "$LOCK" 2>/dev/null; then
  # a lock older than 30 min belongs to a dead run, not a live one (§4.1)
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then
    rmdir "$LOCK" 2>/dev/null; mkdir "$LOCK" || exit 0
  else
    exit 0
  fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

git fetch -q origin "$BRANCH" || { echo "$(ts) fetch failed"; exit 1; }
CUR="$(git rev-parse HEAD)"; NEW="$(git rev-parse "origin/$BRANCH")"
[ "$CUR" = "$NEW" ] && exit 0                      # nothing new: stay quiet

if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
  echo "$(ts) checkout is on $(git rev-parse --abbrev-ref HEAD), not $BRANCH -- not touching it"
  exit 0
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "$(ts) local modifications in tracked files -- not pulling (commit or stash them)"
  exit 0
fi
CHANGED="$(git diff --name-only "$CUR" "$NEW" -- docker/monitor)"
if ! git merge --ff-only -q "origin/$BRANCH"; then
  echo "$(ts) $BRANCH has diverged from origin -- fast-forward impossible, left alone"
  exit 0
fi
echo "$(ts) updated ${CUR:0:7} -> ${NEW:0:7}"
if [ -n "$CHANGED" ]; then
  echo "$(ts) docker/monitor changed:"; echo "$CHANGED" | sed 's/^/  /'
  ( cd docker/monitor && docker compose up -d --build ) \
    && echo "$(ts) monitor stack restarted" \
    || echo "$(ts) docker compose up FAILED -- stack left on its previous version"
fi
