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
# EXIT CLASSIFICATION + INCIDENT BUNDLES (2026-10-07)
#   On 2026-10-05/06 the agent aborted AT EXIT under a deliberate STOP and the
#   host froze 22 s later; this script logged "clean stop", because all it
#   could see was "pid gone + STOP present". launch_skybot.sh now runs the
#   agent under a wrapper that writes runlogs/skybot_run.exit (exit_code,
#   signal, t_end). Every exit is classified:
#
#     STOP present, rc 0, no fatal marker in log tail -> clean
#     STOP present, rc != 0 / no .exit / fatal marker -> abnormal_under_stop
#     no STOP (any rc)                                 -> crash
#
#   Fatal markers: "terminate called" (C++ abort) and "Traceback".
#   Each exit gets `scripts/incident_bundle.sh <class> <run-start-epoch>`
#   (short summary for clean), bounded by `timeout 90`, rc ignored: a bundle
#   can never hold up a restart. Non-clean exits get one crashes.log line
#   naming kind, rc, signal and bundle. STOP present -> exit 0, never relaunch.
#
# CRASH-LOOP GUARD (CLAUDE.md 4.1 -- the escape paths are the point)
#   Same class + signal (or rc=N when not signalled) 3x within 30 min, or the
#   same python exception line 3x in a row (the original guard), writes
#   runlogs/CRASHLOOP and STOPS RELAUNCHING. The supervisor stays alive and
#   idles (polls every 60 s) so the latch can be lifted without a redeploy.
#   WHAT RE-OPENS IT, all reachable without editing anything:
#     * `rm runlogs/CRASHLOOP`  -> the idle supervisor resumes on its next poll
#     * a MANUAL `bash scripts/launch_skybot.sh` (no SKYBOT_SUPERVISED) clears
#       it; the idle supervisor adopts that agent's pid and watches it
#     * starting a supervisor (host.yml `launch`, deploy_skybot.sh) clears it:
#       those are deliberate human launches
#     * `touch runlogs/STOP` ends the idle supervisor cleanly
#
# LOG ROTATION (2026-10-07)
#   Every caller starts this as `... > runlogs/supervisor.log` (deploy_skybot.sh,
#   .github/workflows/host.yml, pipeline.yml), and that `>` truncates the
#   previous supervisor's log BEFORE this script runs, so it cannot be rotated
#   from in here. Instead every line is ALSO appended to supervisor.log.1, and
#   at start .1..4 shift to .2..5: the live log keeps its old behaviour and
#   the last 5 sessions survive. (Callers switching to `>>` would make the
#   live file accumulate too; the copies stay correct either way.)
#
# Usage:
#   setsid nohup bash scripts/supervise_skybot.sh 4000000 \
#       > runlogs/supervisor.log 2>&1 < /dev/null &
# SKYBOT_ROOT and the SKYBOT_SUP_* timings exist for
# tests/_incident_bundle_smoke.py; the defaults are the production values.
set -u
cd "${SKYBOT_ROOT:-/workspace/devai}"

TS=${1:-4000000}
BACKOFF=${SKYBOT_SUP_BACKOFF:-15}       # seconds after a crash; doubles, capped
BACKOFF0=$BACKOFF
MAX_BACKOFF=${SKYBOT_SUP_MAX_BACKOFF:-300}
WATCH_POLL=${SKYBOT_SUP_POLL:-20}
REFUSE_WAIT=${SKYBOT_SUP_REFUSE_WAIT:-60}
EXIT_WAIT=${SKYBOT_SUP_EXIT_WAIT:-30}   # wait this long for the .exit file
LOOP_WINDOW=${SKYBOT_SUP_LOOP_WINDOW:-1800}
LOOP_N=3
CL_POLL=${SKYBOT_SUP_CRASHLOOP_POLL:-60}
STOP=runlogs/STOP
CRASHLOG=runlogs/crashes.log
CRASHLOOP=runlogs/CRASHLOOP
HIST=runlogs/crash_history
EXITF=runlogs/skybot_run.exit
RUNLOG=runlogs/minecraft_skybot_run.log
mkdir -p runlogs

for i in 4 3 2 1; do
    [ -f "runlogs/supervisor.log.$i" ] && \
        mv -f "runlogs/supervisor.log.$i" "runlogs/supervisor.log.$((i + 1))"
done
KEEPLOG=runlogs/supervisor.log.1
: > "$KEEPLOG"
say() { printf '%s\n' "$*"; printf '%s\n' "$*" >> "$KEEPLOG"; }
now_iso() { date -Is 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ; }
TO90=""
command -v timeout >/dev/null 2>&1 && TO90="timeout 90"

say "[supervisor] started $(now_iso) budget=$TS"
if [ -f "$CRASHLOOP" ]; then
    say "[supervisor] clearing $CRASHLOOP: starting a supervisor is a deliberate launch"
    say "[supervisor]   was: $(head -1 "$CRASHLOOP" 2>/dev/null)"
    mv -f "$CRASHLOOP" "$CRASHLOOP.cleared" 2>/dev/null || rm -f "$CRASHLOOP"
    rm -f "$HIST"
fi

IDLE=""
while true; do
    if [ -f "$STOP" ]; then
        say "[supervisor] $STOP present — stopping (deliberate, not a crash)"
        exit 0
    fi

    if [ -f "$CRASHLOOP" ]; then
        if [ -z "$IDLE" ]; then
            say "[supervisor] $CRASHLOOP present — NOT relaunching. Re-open by:"
            say "[supervisor]   rm $CRASHLOOP   or a manual scripts/launch_skybot.sh"
            IDLE=1
        fi
        sleep "$CL_POLL"
        continue
    fi
    if [ -n "$IDLE" ]; then
        IDLE=""
        BACKOFF=$BACKOFF0; SAME=0; LAST_SIG=""
        say "[supervisor] $CRASHLOOP cleared $(now_iso) — resuming"
        # A manual launch cleared it: adopt that agent instead of launching a
        # second one into "REFUSED: run alive".
        P=$(cat runlogs/skybot_run.pid 2>/dev/null || true)
        if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then
            AGENT_PID=$P
            LAUNCH_EPOCH=$(date +%s)
            say "[supervisor] adopting live agent pid=$P"
        fi
    fi

    # Don't stack runs: if one is already alive, just watch it.
    # PID-BASED, NOT pgrep -f. A pattern match on the agent's command line
    # also matches ANY process whose cmdline merely CONTAINS that string —
    # including the monitoring/ssh commands used to check on the run. That
    # made the supervisor believe a dead agent was alive and sit in its wait
    # loop indefinitely, which is exactly the self-match that already caused
    # a spurious "REFUSED: run alive" from the launcher. Track the PID we
    # actually started and ask about THAT process.
    if [ -z "${AGENT_PID:-}" ] || ! kill -0 "$AGENT_PID" 2>/dev/null; then
        say "[supervisor] launching $(now_iso)"
        LAUNCH_EPOCH=$(date +%s)
        OUT=$(SKYBOT_SUPERVISED=1 bash scripts/launch_skybot.sh "$TS" 2>&1)
        RC=$?
        say "$OUT"
        # launch_skybot.sh prints "LAUNCHED skybot pid=NNNN"; that PID is the
        # only reliable handle on the agent we started.
        AGENT_PID=$(printf '%s' "$OUT" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | tail -1)
        if [ $RC -ne 0 ]; then
            # Pre-flight refused (dead tunnel/bridge/server). Not a crash — wait
            # and retry rather than hammering a server that is not there.
            say "[supervisor] launcher refused (rc=$RC) — retrying in ${REFUSE_WAIT}s"
            AGENT_PID=""
            sleep "$REFUSE_WAIT"
            continue
        fi

        if [ -z "$AGENT_PID" ]; then
            say "[supervisor] could not read the launched PID — retrying in ${REFUSE_WAIT}s"
            sleep "$REFUSE_WAIT"
            continue
        fi
    fi
    say "[supervisor] watching pid=$AGENT_PID"
    while kill -0 "$AGENT_PID" 2>/dev/null; do
        sleep "$WATCH_POLL"
    done

    # ---- classify the exit ------------------------------------------------
    # The wrapper writes .exit microseconds after python dies; allow for a
    # loaded box, bounded. Absent (wrapper killed too) -> rc unknown.
    _w=0
    while [ ! -f "$EXITF" ] && [ "$_w" -lt "$EXIT_WAIT" ]; do
        sleep 1; _w=$((_w + 1))
    done
    XRC=$(sed -n 's/^exit_code=//p' "$EXITF" 2>/dev/null | head -1)
    XSIG=$(sed -n 's/^signal=//p' "$EXITF" 2>/dev/null | head -1)
    XT0=$(sed -n 's/^t_start_epoch=//p' "$EXITF" 2>/dev/null | head -1)
    [ -n "$XRC" ] || XRC=none
    [ -n "$XSIG" ] || XSIG=none
    FATAL=""
    tail -n 60 "$RUNLOG" 2>/dev/null | grep -q "terminate called" && FATAL="terminate called"
    [ -z "$FATAL" ] && tail -n 60 "$RUNLOG" 2>/dev/null | grep -q "Traceback" && FATAL="Traceback"
    if [ -f "$STOP" ]; then
        if [ "$XRC" = "0" ] && [ -z "$FATAL" ]; then KIND=clean; else KIND=abnormal_under_stop; fi
    else
        KIND=crash
    fi
    say "[supervisor] agent pid=$AGENT_PID exited $(now_iso) kind=$KIND rc=$XRC sig=$XSIG${FATAL:+ marker=\"$FATAL\"}"
    BOUT=$($TO90 nice -n 10 bash scripts/incident_bundle.sh "$KIND" "${XT0:-${LAUNCH_EPOCH:-0}}" 2>&1 | tail -3)
    say "[supervisor] $BOUT"
    BUNDLE=$(printf '%s' "$BOUT" | sed -n 's/^incident bundle: //p' | tail -1)

    if [ "$KIND" != "clean" ]; then
        # Record WHY: a supervisor that silently restarts a crash loop hides
        # the very fault it is papering over.
        {
            echo "=== agent exited $(now_iso) kind=$KIND rc=$XRC sig=$XSIG bundle=${BUNDLE:-none} ==="
            tail -30 "$RUNLOG" 2>/dev/null | grep -A22 -E "Traceback|terminate called" | head -30
            echo
        } >> "$CRASHLOG"
    fi
    if [ -f "$STOP" ]; then
        say "[supervisor] agent exited with $STOP present — $KIND stop, not relaunching"
        exit 0
    fi

    # ---- crash-loop guards ------------------------------------------------
    # A DETERMINISTIC fault (one that fires on the first segment every time)
    # cannot resolve itself, and retrying it just walks the backoff to its cap
    # and idles the host while looking like progress. Latch loudly instead --
    # with the escape paths in the header.
    SIG=$(tail -30 "$RUNLOG" 2>/dev/null \
          | grep -m1 -E "^[A-Za-z_.]*(Error|Exception)" | cut -c1-120)
    if [ -n "$SIG" ] && [ "$SIG" = "${LAST_SIG:-}" ]; then
        SAME=$(( ${SAME:-1} + 1 ))
    else
        SAME=1
    fi
    LAST_SIG="$SIG"
    if [ "$XSIG" != "none" ]; then KEY=$XSIG; else KEY="rc=$XRC"; fi
    NOW=$(date +%s)
    echo "$NOW $KIND $KEY" >> "$HIST"
    tail -n 50 "$HIST" > "$HIST.tmp" 2>/dev/null && mv -f "$HIST.tmp" "$HIST"
    NREC=$(awk -v t=$((NOW - LOOP_WINDOW)) -v k="$KIND" -v s="$KEY" \
           '$1 >= t && $2 == k && $3 == s' "$HIST" | wc -l | tr -d ' ')
    WHY=""
    if [ "$NREC" -ge "$LOOP_N" ]; then
        WHY="kind=$KIND sig=$KEY ${NREC}x within ${LOOP_WINDOW}s"
    elif [ "$SAME" -ge 3 ]; then
        WHY="same fault 3x in a row: $SIG"
    fi
    if [ -n "$WHY" ]; then
        {
            echo "CRASHLOOP $(now_iso) $WHY"
            echo "last bundle: ${BUNDLE:-none}"
            echo "Re-open: rm runlogs/CRASHLOOP (the idle supervisor resumes), or"
            echo "run scripts/launch_skybot.sh by hand, or start a new supervisor."
        } > "$CRASHLOOP"
        echo "=== CRASHLOOP $(now_iso) $WHY ===" >> "$CRASHLOG"
        say "[supervisor] STOPPING RELAUNCHES: $WHY — this is"
        say "[supervisor]   deterministic and will not fix itself; wrote $CRASHLOOP"
        AGENT_PID=""
        continue
    fi
    say "[supervisor] agent exited — see $CRASHLOG; restarting in ${BACKOFF}s"
    sleep "$BACKOFF"
    BACKOFF=$(( BACKOFF * 2 ))
    [ "$BACKOFF" -gt "$MAX_BACKOFF" ] && BACKOFF=$MAX_BACKOFF
done
