#!/usr/bin/env bash
# Graceful stop that WAITS FOR THE PROCESS TO ACTUALLY EXIT, and fails if it
# does not.
#
# WHY THIS EXISTS (2026-09-04). The stop step used to end like this:
#
#     for i in $(seq 1 30); do                 # 30 x 10s = 5 minutes
#       pgrep -f "$R[.]py" >/dev/null || { echo "stopped"; exit 0; }
#       sleep 10
#     done
#     echo "still running after 5 min — it may be mid-segment; check status."
#
# That last `echo` exits 0, so "I gave up waiting and it is STILL TRAINING"
# was reported as a GREEN CHECK. The tick meant "STOP file written", not
# "training stopped".
#
# And five minutes was never enough. Measured on the live training host:
#   * finishing the current segment: 2048 steps at ~6.8 steps/s  ~= 5 min
#   * writing the replay buffer:     325,096 transitions in 513.5s ~= 8.5 min
# so an honest graceful shutdown takes ~13 minutes. The old budget expired
# during a NORMAL stop, every time, and then reported success.
#
# The user-visible consequence was a loop nobody could explain: stop (green,
# still live) -> deploy -> "REFUSED: training is LIVE" -> stop again (now
# enough wall-clock has passed, so it works) -> deploy with
# --allow-stop-live. Two real bugs, read as flaky tooling.
#
# NEVER `kill`: the supervisor treats a killed process as a crash and
# relaunches it (CLAUDE.md §2). `touch runlogs/STOP` is the only stop that
# stays stopped.
#
# Usage: host_stop_wait.sh [timeout_seconds]   (default 1200 = 20 min)
set -euo pipefail

TIMEOUT="${1:-1200}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The pattern is SPLIT so this command line cannot match itself in pgrep —
# the same self-match trap documented in supervise_skybot.sh.
bash "$HERE/host_exec.sh" "
  set -eu
  cd /workspace/devai
  R=\"run_min\"\"ecraft\"
  touch runlogs/STOP
  echo 'STOP written; the agent finishes its segment, saves the brain, exits.'
  echo 'A full graceful stop is ~13 min (segment ~5, replay buffer ~8.5).'
  DEADLINE=\$(( \$(date +%s) + ${TIMEOUT} ))
  while [ \"\$(date +%s)\" -lt \"\$DEADLINE\" ]; do
    if ! pgrep -f \"\$R[.]py\" >/dev/null; then
      echo \"stopped cleanly after \$(( ${TIMEOUT} - (DEADLINE - \$(date +%s)) ))s\"
      exit 0
    fi
    # Show progress so a long wait is legible rather than looking hung.
    echo \"  still running (\$(( DEADLINE - \$(date +%s) ))s of budget left) — \$(tail -1 runlogs/minecraft_skybot_run.log 2>/dev/null | cut -c1-90)\"
    sleep 15
  done
  echo \"TIMED OUT after ${TIMEOUT}s with training still alive (pid \$(cat runlogs/skybot_run.pid 2>/dev/null || echo unknown)).\"
  echo \"The STOP file is in place, so it should still exit on its own; re-run stop to keep waiting.\"
  exit 1"
