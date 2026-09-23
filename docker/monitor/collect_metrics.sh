#!/bin/bash
# Pull SkyBot's per-segment metrics from the pod to this host, durably.
#
# TWO MECHANISMS, BECAUSE NEITHER ALONE IS SUFFICIENT:
#   tail -F   near-instant, but loses everything written while the ssh link
#             was down — and a rotating pod file plus a flaky tailnet means
#             that WILL happen on a multi-day run.
#   rsync     complete, but only as fresh as its interval.
# Running both makes the local copy live AND complete. Duplicates are fine:
# every record carries a monotonic `seq` and the ingester is idempotent on it
# (INSERT OR IGNORE), so overlap costs nothing and gaps cost data.
#
# THE LOCAL JSONL IS THE SYSTEM OF RECORD. The SQLite the dashboard reads is a
# rebuildable view — delete it and re-run the ingester and you get it back.
#
#   MAIN_HOST, MAIN_SSH_PORT, MAIN_SSH_KEYFILE   as in the runner .env
#   OUT_DIR      where to write (default /data)
#   REMOTE_PATH  pod-side metrics file
#   RSYNC_EVERY  seconds between backfills (default 300)
set -uo pipefail

# ---- WHO WE LOG IN AS (added 2026-09-22) ----------------------------------
# Was hardcoded to root, because RunPod only ever gave you root. The owned box
# `main` is a normal Ubuntu machine with a normal account, so the user is now a
# variable. Default stays `root` so nothing that used to work stops working.
MAIN_USER="${MAIN_USER:-root}"

: "${MAIN_HOST:?set MAIN_HOST}"
: "${MAIN_SSH_PORT:?set MAIN_SSH_PORT (22 on the owned host `main`)}"
KEY="${MAIN_SSH_KEYFILE:-$HOME/.ssh/skybot_ed25519}"
OUT_DIR="${OUT_DIR:-/data}"
REMOTE_PATH="${REMOTE_PATH:-/workspace/devai/runlogs/metrics.jsonl}"
RSYNC_EVERY="${RSYNC_EVERY:-300}"
# How often to pull the fast heartbeat feed. 30s keeps the
# dashboard live without a second persistent ssh connection.
HB_EVERY="${HB_EVERY:-30}"
OUT="$OUT_DIR/metrics.jsonl"

# BatchMode=yes is load-bearing: without it a rejected key drops ssh into an
# interactive password prompt and this loop blocks forever instead of
# retrying. ServerAlive* kills a silently dropped link in ~60s rather than
# leaving a tail that never returns.
SSH_OPTS=(-p "$MAIN_SSH_PORT" -i "$KEY"
          -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o BatchMode=yes -o ConnectTimeout=20
          -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

mkdir -p "$OUT_DIR"
touch "$OUT"

backfill() {
  # -a without -z: these are small JSON lines and the pod link is local-ish;
  # compression costs more CPU on a 2-core mini than it saves.
  rsync -a -e "ssh ${SSH_OPTS[*]}" \
    "${MAIN_USER}@${MAIN_HOST}:${REMOTE_PATH}" "$OUT_DIR/.backfill.jsonl" 2>/dev/null || return 0
  # Append only what we do not already have, keyed on seq. Sorting by seq and
  # de-duplicating keeps the file replayable from the top.
  cat "$OUT" "$OUT_DIR/.backfill.jsonl" 2>/dev/null \
    | awk 'match($0, /"seq":[0-9]+/) {
             s = substr($0, RSTART+6, RLENGTH-6)
             if (!(s in seen)) { seen[s] = 1; print }
           }' \
    | sort -t: -k2 -n > "$OUT_DIR/.merged.jsonl" 2>/dev/null
  if [ -s "$OUT_DIR/.merged.jsonl" ]; then
    mv "$OUT_DIR/.merged.jsonl" "$OUT"      # atomic within the same fs
  fi
  rm -f "$OUT_DIR/.backfill.jsonl"
}

# The HEARTBEAT is a second, faster feed (~15s vs ~7min for segments). It is
# pulled by rsync only, not tailed: at 15s intervals a 5-minute backfill is
# already fine granularity for a dashboard, and a second persistent ssh tail
# would double the connection count for very little gain.
HB_REMOTE="${HB_REMOTE_PATH:-/workspace/devai/runlogs/heartbeat.jsonl}"
HB_OUT="$OUT_DIR/heartbeat.jsonl"
touch "$HB_OUT"

backfill_hb() {
  rsync -a -e "ssh ${SSH_OPTS[*]}" \
    "${MAIN_USER}@${MAIN_HOST}:${HB_REMOTE}" "$OUT_DIR/.hb.jsonl" 2>/dev/null || return 0
  cat "$HB_OUT" "$OUT_DIR/.hb.jsonl" 2>/dev/null \
    | awk 'match($0, /"seq":[0-9]+/) {
             s = substr($0, RSTART+6, RLENGTH-6)
             if (!(s in seen)) { seen[s] = 1; print }
           }' \
    | sort -t: -k2 -n > "$OUT_DIR/.hb_merged.jsonl" 2>/dev/null
  [ -s "$OUT_DIR/.hb_merged.jsonl" ] && mv "$OUT_DIR/.hb_merged.jsonl" "$HB_OUT"
  rm -f "$OUT_DIR/.hb.jsonl"
}

echo "collector: pod=${MAIN_HOST}:${MAIN_SSH_PORT} -> ${OUT} (+ heartbeat)"
backfill
backfill_hb
LAST_RSYNC=$(date +%s)
# SEPARATE clock from LAST_RSYNC. Sharing it made the heartbeat condition
# true on every 10s pass for the whole 300s segment-backfill window, so it
# rsynced 3x more often than configured.
LAST_HB=$(date +%s)

while true; do
  # `tail -F` (capital) follows across ROTATION — the pod rotates this file at
  # 64 MB, and lowercase -f would silently keep reading the old inode forever.
  ssh "${SSH_OPTS[@]}" "${MAIN_USER}@${MAIN_HOST}" \
      "tail -F -n 0 '${REMOTE_PATH}' 2>/dev/null" >> "$OUT" &
  TAIL_PID=$!

  # Periodic backfill while the tail runs; also the liveness check.
  while kill -0 "$TAIL_PID" 2>/dev/null; do
    sleep 10
    NOW=$(date +%s)
    if [ $((NOW - LAST_HB)) -ge "$HB_EVERY" ]; then
      # Heartbeat pulled on a SHORTER interval than the segment backfill —
      # it is the live feed, and its whole point is granularity.
      backfill_hb
      LAST_HB=$NOW
    fi
    if [ $((NOW - LAST_RSYNC)) -ge "$RSYNC_EVERY" ]; then
      backfill
      LAST_RSYNC=$NOW
    fi
  done

  wait "$TAIL_PID" 2>/dev/null
  echo "collector: tail exited, backfilling then reconnecting in 10s"
  backfill
  backfill_hb
  LAST_RSYNC=$(date +%s); LAST_HB=$LAST_RSYNC
  sleep 10
done
