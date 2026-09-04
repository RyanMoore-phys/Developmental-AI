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
#   POD_HOST, POD_SSH_PORT, POD_SSH_KEYFILE   as in the runner .env
#   OUT_DIR      where to write (default /data)
#   REMOTE_PATH  pod-side metrics file
#   RSYNC_EVERY  seconds between backfills (default 300)
set -uo pipefail

: "${POD_HOST:?set POD_HOST}"
: "${POD_SSH_PORT:?set POD_SSH_PORT - it rotates on every pod restart}"
KEY="${POD_SSH_KEYFILE:-$HOME/.ssh/skybot_ed25519}"
OUT_DIR="${OUT_DIR:-/data}"
REMOTE_PATH="${REMOTE_PATH:-/workspace/devai/podlogs/metrics.jsonl}"
RSYNC_EVERY="${RSYNC_EVERY:-300}"
OUT="$OUT_DIR/metrics.jsonl"

# BatchMode=yes is load-bearing: without it a rejected key drops ssh into an
# interactive password prompt and this loop blocks forever instead of
# retrying. ServerAlive* kills a silently dropped link in ~60s rather than
# leaving a tail that never returns.
SSH_OPTS=(-p "$POD_SSH_PORT" -i "$KEY"
          -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o BatchMode=yes -o ConnectTimeout=20
          -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

mkdir -p "$OUT_DIR"
touch "$OUT"

backfill() {
  # -a without -z: these are small JSON lines and the pod link is local-ish;
  # compression costs more CPU on a 2-core mini than it saves.
  rsync -a -e "ssh ${SSH_OPTS[*]}" \
    "root@${POD_HOST}:${REMOTE_PATH}" "$OUT_DIR/.backfill.jsonl" 2>/dev/null || return 0
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

echo "collector: pod=${POD_HOST}:${POD_SSH_PORT} -> ${OUT}"
backfill
LAST_RSYNC=$(date +%s)

while true; do
  # `tail -F` (capital) follows across ROTATION — the pod rotates this file at
  # 64 MB, and lowercase -f would silently keep reading the old inode forever.
  ssh "${SSH_OPTS[@]}" "root@${POD_HOST}" \
      "tail -F -n 0 '${REMOTE_PATH}' 2>/dev/null" >> "$OUT" &
  TAIL_PID=$!

  # Periodic backfill while the tail runs; also the liveness check.
  while kill -0 "$TAIL_PID" 2>/dev/null; do
    sleep 10
    NOW=$(date +%s)
    if [ $((NOW - LAST_RSYNC)) -ge "$RSYNC_EVERY" ]; then
      backfill
      LAST_RSYNC=$NOW
    fi
  done

  wait "$TAIL_PID" 2>/dev/null
  echo "collector: tail exited, backfilling then reconnecting in 10s"
  backfill
  LAST_RSYNC=$(date +%s)
  sleep 10
done
