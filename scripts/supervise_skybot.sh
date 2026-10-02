#!/bin/bash
# SUPERVISOR: keep SkyBot alive across PROCESS-level death.
#
# WHY THIS EXISTS
#   The adapter already auto-rejoins the SERVER: a kick or a death leaves the
#   client on a static screen, the frozen-POV watchdog notices, and the client
#   is rebuilt into a fresh mission (which is also the respawn path). That
#   handles everything that goes wrong INSIDE a running agent.
#
#   It cannot handle the agent itself dying. Twice on 2026-08-02 a python
#   exception took the whole process down, both Minecraft clients with it, and
#   from the server side that is indistinguishable from a kick — SkyBot simply
#   vanishes and never comes back, because nothing is left running to notice.
#   A multi-day lifelong run cannot depend on someone watching for that.
#
# WHAT IT DOES
#   Relaunches whenever the agent exits, with a backoff, and keeps a crash log
#   so a repeating fault is visible rather than hidden by the restart. The
#   launcher's own pre-flight still refuses to start on a dead tunnel, so a
#   broken bridge produces a clean refusal loop rather than a stranded agent.
#
#   STOP-FILE AWARE: `touch runlogs/STOP` ends the supervision loop cleanly —
#   the same file lifelong.stop_file already uses, so a deliberate stop is
#   never mistaken for a crash and restarted.
#
# Usage:
#   setsid nohup bash scripts/supervise_skybot.sh 4000000 \
#       > runlogs/supervisor.log 2>&1 < /dev/null &
set -u
cd /workspace/devai

TS=${1:-4000000}
BACKOFF=15          # seconds after a crash; doubles, capped
MAX_BACKOFF=300
STOP=runlogs/STOP
CRASHLOG=runlogs/crashes.log
mkdir -p runlogs

echo "[supervisor] started $(date -Is) budget=$TS"

while true; do
    if [ -f "$STOP" ]; then
        echo "[supervisor] $STOP present — stopping (deliberate, not a crash)"
        exit 0
    fi

    # Don't stack runs: if one is already alive, just watch it.
    # PID-BASED, NOT pgrep -f. A pattern match on the agent's command line
    # also matches ANY process whose cmdline merely CONTAINS that string —
    # including the monitoring/ssh commands used to check on the run. That
    # made the supervisor believe a dead agent was alive and sit in its wait
    # loop indefinitely, which is exactly the self-match that already caused
    # a spurious "REFUSED: run alive" from the launcher. Track the PID we
    # actually started and ask about THAT process.
    if [ -n "${AGENT_PID:-}" ] && kill -0 "$AGENT_PID" 2>/dev/null; then
        sleep 30
        continue
    fi

    echo "[supervisor] launching $(date -Is)"
    OUT=$(bash scripts/launch_skybot.sh "$TS" 2>&1)
    RC=$?
    echo "$OUT"
    # launch_skybot.sh prints "LAUNCHED skybot pid=NNNN"; that PID is the
    # only reliable handle on the agent we started.
    AGENT_PID=$(printf '%s' "$OUT" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | tail -1)
    if [ $RC -ne 0 ]; then
        # Pre-flight refused (dead tunnel/bridge/server). Not a crash — wait
        # and retry rather than hammering a server that is not there.
        echo "[supervisor] launcher refused (rc=$RC) — retrying in 60s"
        sleep 60
        continue
    fi

    if [ -z "$AGENT_PID" ]; then
        echo "[supervisor] could not read the launched PID — retrying in 60s"
        sleep 60
        continue
    fi
    echo "[supervisor] watching pid=$AGENT_PID"
    while kill -0 "$AGENT_PID" 2>/dev/null; do
        sleep 20
    done

    if [ -f "$STOP" ]; then
        echo "[supervisor] agent exited with $STOP present — clean stop"
        exit 0
    fi

    # It died on its own. Record WHY: a supervisor that silently restarts a
    # crash loop hides the very fault it is papering over.
    {
        echo "=== agent exited $(date -Is) ==="
        tail -30 runlogs/minecraft_skybot_run.log 2>/dev/null \
            | grep -A22 "Traceback" | head -30
        echo
    } >> "$CRASHLOG"
    # A DETERMINISTIC fault (one that fires on the first segment every time)
    # cannot resolve itself, and retrying it just walks the backoff to its cap
    # and idles the training host while looking like progress. Stop loudly instead.
    SIG=$(tail -30 runlogs/minecraft_skybot_run.log 2>/dev/null \
          | grep -m1 -E "^[A-Za-z_.]*(Error|Exception)" | cut -c1-120)
    if [ -n "$SIG" ] && [ "$SIG" = "${LAST_SIG:-}" ]; then
        SAME=$(( ${SAME:-1} + 1 ))
    else
        SAME=1
    fi
    LAST_SIG="$SIG"
    if [ "$SAME" -ge 3 ]; then
        echo "[supervisor] STOPPING: same fault 3x in a row — this is"
        echo "[supervisor]   deterministic and will not fix itself:"
        echo "[supervisor]   $SIG"
        exit 1
    fi
    echo "[supervisor] agent exited — see $CRASHLOG; restarting in ${BACKOFF}s"
    sleep "$BACKOFF"
    BACKOFF=$(( BACKOFF * 2 ))
    [ "$BACKOFF" -gt "$MAX_BACKOFF" ] && BACKOFF=$MAX_BACKOFF
done
