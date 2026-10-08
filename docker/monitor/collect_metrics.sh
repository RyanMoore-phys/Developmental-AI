#!/bin/bash
# Pull SkyBot's per-segment metrics from the training host to this host, durably.
#
# TWO MECHANISMS, BECAUSE NEITHER ALONE IS SUFFICIENT:
#   tail -F   near-instant, but loses everything written while the ssh link
#             was down — and a rotating host file plus a flaky tailnet means
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
#   REMOTE_PATH  host-side metrics file
#   RSYNC_EVERY  seconds between backfills (default 300)
#   PULL_EVERY   seconds between evidence-mirror pulls (default 60; see below)
set -uo pipefail

# ---- WHO WE LOG IN AS (added 2026-09-22) ----------------------------------
# Was hardcoded to root, because a rented GPU host only ever gave you root. The owned box
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
  # -a without -z: these are small JSON lines and the training host link is local-ish;
  # compression costs more CPU on a 2-core mini than it saves.
  if ! rsync -a -e "ssh ${SSH_OPTS[*]}" \
       "${MAIN_USER}@${MAIN_HOST}:${REMOTE_PATH}" \
       "$OUT_DIR/.backfill.jsonl" 2>/dev/null; then
    # SAY SO. This was `|| return 0`: a missing remote file returned SUCCESS,
    # so when podlogs/ became runlogs/ (2026-09-22) the collector went on
    # cheerfully fetching a path that no longer existed and the dashboard
    # simply stopped updating. Nothing anywhere said why. A collector that
    # cannot find its source is the one thing it must not be quiet about.
    # Rate-limited so a long outage does not fill the log it is competing
    # with for a 98 GB disk.
    _now=$(date +%s)
    if [ $(( _now - ${_LAST_WARN:-0} )) -ge 300 ]; then
      echo "collector: CANNOT FETCH ${REMOTE_PATH} from ${MAIN_HOST}" \
           "— wrong path, or the agent is not writing it." \
           "NOTE: podlogs/ was renamed to runlogs/ on 2026-09-22."
      _LAST_WARN=$_now
    fi
    return 0
  fi
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

# ---- EVIDENCE MIRROR (2026-10-07) -----------------------------------------
# The 2026-10-06 freeze left its evidence ON main: the run log, crashes.log,
# the kernel log -- none of it reachable while the box was down, and some of
# it rotated away by the next launch. Every PULL_EVERY s (default 60) this
# mirrors the rest of the telemetry flat into $OUT_DIR, where ingest.py reads:
#   learning.jsonl*  position_trace.jsonl*  memory_census.jsonl*
#   crashes.log  supervisor.log*  incidents/   (bundles, index.jsonl)
# metrics.jsonl and heartbeat.jsonl are NOT in this list: they keep their
# seq-merged copies above, which a raw mirror would clobber.
# WHY plain `rsync -a` and not --append: the feeds ROTATE on main (the
# metrics sink's 64 MB rotation; supervisor.log is truncated by every
# launcher). --append skips a file whose remote copy is now SHORTER than the
# local one, so after the first rotation it would silently stop updating --
# the same quiet failure as the podlogs rename above. Instead:
#   * the rotated siblings (`*.jsonl.1` ...) are pulled under their own
#     names, so a rotation loses nothing as long as main does not rotate the
#     same feed twice inside one PULL_EVERY;
#   * rsync writes to a temp file and renames, so ingest never reads half a
#     file (`--partial-dir` keeps an interrupted big transfer resumable);
#   * NO --delete: what main prunes (bundle retention, 3-deep logs) stays here.
# BOUNDED: `timeout PULL_TIMEOUT` (50 s) per pull, --max-size, and node1-side
# retention of incident bundles older than INCIDENT_KEEP_DAYS (180).
PULL_EVERY="${PULL_EVERY:-60}"
PULL_TIMEOUT="${PULL_TIMEOUT:-50}"
REMOTE_RUNLOGS="${REMOTE_RUNLOGS:-/workspace/devai/runlogs}"
INCIDENT_KEEP_DAYS="${INCIDENT_KEEP_DAYS:-180}"
mirror_pull() {
  timeout "$PULL_TIMEOUT" rsync -a --partial-dir=.rsync-partial --max-size=512M \
      -e "ssh ${SSH_OPTS[*]}" \
      --include='learning.jsonl*' --include='position_trace.jsonl*' \
      --include='memory_census.jsonl*' --include='crashes.log' \
      --include='supervisor.log*' --include='CRASHLOOP' \
      --exclude='incidents/.partial-*' \
      --include='incidents/' --include='incidents/**' --exclude='*' \
      "${MAIN_USER}@${MAIN_HOST}:${REMOTE_RUNLOGS}/" "$OUT_DIR/" 2>/dev/null
  _rc=$?
  # rc 23/24 = some listed file does not exist yet (a fresh host) -- normal.
  if [ "$_rc" -ne 0 ] && [ "$_rc" -ne 23 ] && [ "$_rc" -ne 24 ]; then
    _now=$(date +%s)
    if [ $(( _now - ${_LAST_MWARN:-0} )) -ge 300 ]; then
      echo "collector: evidence mirror FAILED rc=$_rc from ${MAIN_HOST}:${REMOTE_RUNLOGS}" \
           "(124 = timed out after ${PULL_TIMEOUT}s)"
      _LAST_MWARN=$_now
    fi
  fi
  # CRASHLOOP is a flag, not a log: mirror its ABSENCE too, or a cleared latch
  # would stay "on" here forever (rsync without --delete never removes it).
  if [ "$_rc" -eq 0 ] || [ "$_rc" -eq 23 ] || [ "$_rc" -eq 24 ]; then
    ssh "${SSH_OPTS[@]}" "${MAIN_USER}@${MAIN_HOST}" \
        "test -f '${REMOTE_RUNLOGS}/CRASHLOOP'" 2>/dev/null
    [ $? -eq 1 ] && rm -f "$OUT_DIR/CRASHLOOP"
  fi
  [ -d "$OUT_DIR/incidents" ] && find "$OUT_DIR/incidents" -mindepth 1 -maxdepth 1 \
      -type d -name '20*' -mtime +"$INCIDENT_KEEP_DAYS" -exec rm -rf {} + 2>/dev/null
  return 0
}

echo "collector: training host=${MAIN_HOST}:${MAIN_SSH_PORT} -> ${OUT} (+ heartbeat, evidence mirror every ${PULL_EVERY}s)"
backfill
backfill_hb
mirror_pull
LAST_PULL=$(date +%s)
LAST_RSYNC=$(date +%s)
# SEPARATE clock from LAST_RSYNC. Sharing it made the heartbeat condition
# true on every 10s pass for the whole 300s segment-backfill window, so it
# rsynced 3x more often than configured.
LAST_HB=$(date +%s)

while true; do
  # `tail -F` (capital) follows across ROTATION — the training host rotates this file at
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
    if [ $((NOW - LAST_PULL)) -ge "$PULL_EVERY" ]; then
      mirror_pull
      LAST_PULL=$NOW
    fi
  done

  wait "$TAIL_PID" 2>/dev/null
  echo "collector: tail exited, backfilling then reconnecting in 10s"
  backfill
  backfill_hb
  mirror_pull
  LAST_RSYNC=$(date +%s); LAST_HB=$LAST_RSYNC; LAST_PULL=$LAST_RSYNC
  sleep 10
done
