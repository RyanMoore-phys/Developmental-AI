#!/bin/bash
# INCIDENT BUNDLE: freeze the evidence of one agent exit, on `main`, at the
# moment it happens, into runlogs/incidents/<UTC-ts>-<class>/.
#
# WHY THIS EXISTS (2026-10-07). On 2026-10-05/06 the agent aborted at exit
# under a deliberate STOP (`terminate called without an active exception`)
# and the host froze 22 s later. What was left to diagnose it: a run log the
# next launch rotated, a crashes.log line with no exit code, and a kernel log
# nobody had copied off the box. The supervisor could not even say whether
# the exit was clean -- it only ever saw "pid gone". The design is
# docs/foundation/TELEMETRY_INVENTORY_OPS.md section 2.1.
#
# Usage: incident_bundle.sh <class> <run-start-epoch>
#   class: clean | abnormal_under_stop | crash   (decided by supervise_skybot.sh)
#   run-start-epoch: when the supervisor launched this run (journal --since)
# Exit code / signal / t_end come from runlogs/skybot_run.exit, written by the
# exit wrapper in launch_skybot.sh. Missing -> null, never an error.
#
# CONTRACTS (tests/_incident_bundle_smoke.py)
#   * summary.json {"schema":"skybot.incident","v":1,"class","exit_code",
#     "signal","t_start","t_end","run_id","git_rev","config_hash", ...}.
#   * written to .partial-* and mv'd into place: a puller (node1's
#     collect_metrics.sh) never sees half a bundle.
#   * `clean` keeps only a short summary (run-log tail 50, free, df, uptime).
#   * one JSON line per bundle appended to runlogs/incidents/index.jsonl.
#   * retention: newest INCIDENT_KEEP (50) bundles and <= INCIDENT_MAX_KB
#     (100 MB) total, oldest first; never index.jsonl, never the bundle just
#     written.
#   * NO PYTHON, NO SUDO. A bundle taken while memory is short must not
#     import torch, and must not block on a password prompt. Every probe is
#     wrapped in `timeout 10` where available, and a missing tool (no GPU, no
#     journal access, macOS in the test) degrades to one "unavailable" line.
#   * Always exits 0. The supervisor ignores the rc anyway and bounds the whole
#     script with `timeout 90`; a bundle can never hold up a restart.
set -u
cd "${SKYBOT_ROOT:-/workspace/devai}" 2>/dev/null || exit 0

CLASS=$(printf '%s' "${1:-unknown}" | tr -c 'a-z_' '_' | cut -c1-40)
T_START=$(printf '%s' "${2:-0}" | tr -dc '0-9')
[ -n "$T_START" ] || T_START=0
KEEP=${INCIDENT_KEEP:-50}
MAX_KB=${INCIDENT_MAX_KB:-102400}
ROOT=runlogs/incidents
RUNLOG=runlogs/minecraft_skybot_run.log
EXITF=runlogs/skybot_run.exit
mkdir -p "$ROOT" 2>/dev/null || exit 0

TO=""
command -v timeout >/dev/null 2>&1 && TO="timeout 10"

TS=$(date -u +%Y%m%dT%H%M%SZ)
NAME="$TS-$CLASS"
[ -e "$ROOT/$NAME" ] && NAME="$TS-$CLASS-$$"
PART="$ROOT/.partial-$NAME"
rm -rf "$PART"
mkdir -p "$PART" 2>/dev/null || exit 0

# run <file> <shell command>: append "$ cmd", its output and rc to <file>.
run() {
    _f=$1; shift
    {
        echo "\$ $*"
        $TO bash -c "$*" 2>&1
        echo "(rc=$?)"
        echo
    } >> "$PART/$_f" 2>&1
}
# have <tool> <file>: true if installed, else record that it is not.
have() {
    command -v "$1" >/dev/null 2>&1 && return 0
    echo "unavailable: $1 not installed" >> "$PART/$2"
    return 1
}
iso_of() {
    [ "${1:-0}" -gt 0 ] 2>/dev/null || { echo ""; return; }
    date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
        || date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo ""
}
kv() { sed -n "s/^$1=//p" "$EXITF" 2>/dev/null | head -1; }
# JSON helpers: a string or null; an integer or null.
js() {
    if [ -z "${1:-}" ]; then echo null; return; fi
    printf '"%s"' "$(printf '%s' "$1" | tr -d '\000-\037' | sed 's/\\/\\\\/g; s/"/\\"/g')"
}
jn() { case "${1:-}" in ''|*[!0-9-]*) echo null ;; *) echo "$1" ;; esac; }
sha256_of() {
    { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1" 2>/dev/null; } | awk '{print $1}'
}

# ---- identity + exit status ------------------------------------------------
EXIT_CODE=$(kv exit_code)
SIGNAL=$(kv signal); [ "$SIGNAL" = "none" ] && SIGNAL=""
T_END_EPOCH=$(kv t_end_epoch)
[ -n "$T_END_EPOCH" ] || T_END_EPOCH=$(date +%s)
GIT_REV=$(git rev-parse HEAD 2>/dev/null || echo unknown)
CONFIG_HASH=$(sha256_of configs/minecraft_skybot.yaml)
# run_id: the agent stamps it on every feed line; take the newest one seen.
RUN_ID=""
for _feed in runlogs/learning.jsonl runlogs/heartbeat.jsonl runlogs/metrics.jsonl; do
    [ -f "$_feed" ] || continue
    RUN_ID=$(tail -n 5 "$_feed" 2>/dev/null | sed -n 's/.*"run_id": *"\([^"]*\)".*/\1/p' | tail -1)
    [ -n "$RUN_ID" ] && break
done
MEM_AVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
SWAP_FREE_KB=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo 2>/dev/null)

# ---- evidence ---------------------------------------------------------------
if [ "$CLASS" = "clean" ]; then
    tail -n 50 "$RUNLOG" > "$PART/runlog_tail.txt" 2>/dev/null || true
    have free mem.txt && run mem.txt 'free -h'
    run disk.txt 'df -h'
    run uptime.txt 'uptime'
else
    tail -n 300 "$RUNLOG" > "$PART/runlog_tail.txt" 2>/dev/null || true
    tail -n 100 runlogs/crashes.log > "$PART/crashes_tail.txt" 2>/dev/null || true
    tail -n 100 runlogs/supervisor.log > "$PART/supervisor_tail.txt" 2>/dev/null || true
    cp "$EXITF" "$PART/skybot_run.exit" 2>/dev/null || true
    have free mem.txt && run mem.txt 'free -h'
    have swapon mem.txt && run mem.txt 'swapon --show'
    for _p in /proc/pressure/memory /proc/pressure/io /proc/pressure/cpu; do
        if [ -r "$_p" ]; then run mem.txt "cat $_p"; else echo "unavailable: $_p" >> "$PART/mem.txt"; fi
    done
    [ -r /proc/vmstat ] && run mem.txt "grep -E '^(pswpin|pswpout|pgmajfault|oom_kill) ' /proc/vmstat"
    # top-15 by RSS; GNU ps first, BSD (the test on macOS) second.
    run procs.txt 'if ps -eo pid --sort=-rss >/dev/null 2>&1; then ps -eo pid,ppid,rss,vsz,etime,stat,comm,args --sort=-rss | head -16 | cut -c1-200; else ps -axo pid,ppid,rss,vsz,etime,stat,comm -m | head -16; fi'
    if have nvidia-smi gpu.txt; then
        run gpu.txt 'nvidia-smi'
        run gpu.txt 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
    fi
    if have journalctl kernel.txt; then
        # No sudo: on Ubuntu the login user reads the journal via group adm /
        # systemd-journal. Without it this records the refusal, which is
        # itself the finding (add the user to systemd-journal).
        if [ "$T_START" -gt 0 ]; then _since="@$T_START"; else _since="-2h"; fi
        run kernel.txt "journalctl -k --since '$_since' --no-pager | tail -n 400"
        run oom.txt "journalctl -u earlyoom --since '$_since' --no-pager | tail -n 50"
    fi
    run oom.txt "grep -iE 'earlyoom|oom-kill|out of memory|killed process' '$PART/kernel.txt' | tail -n 50"
    run disk.txt 'df -h'
    run uptime.txt 'uptime'
    [ -r /proc/sys/kernel/random/boot_id ] && run uptime.txt 'cat /proc/sys/kernel/random/boot_id'
    run identity.txt "echo git_rev=$GIT_REV; echo config_sha256=$CONFIG_HASH"
fi

T_START_ISO=$(iso_of "$T_START")
T_END_ISO=$(iso_of "$T_END_EPOCH")
cat > "$PART/summary.json" <<EOF
{"schema": "skybot.incident", "v": 1, "class": $(js "$CLASS"), "exit_code": $(jn "$EXIT_CODE"), "signal": $(js "$SIGNAL"), "t_start": $(js "$T_START_ISO"), "t_end": $(js "$T_END_ISO"), "t_start_epoch": $(jn "$T_START"), "t_end_epoch": $(jn "$T_END_EPOCH"), "run_id": $(js "$RUN_ID"), "git_rev": $(js "$GIT_REV"), "config_hash": $(js "$CONFIG_HASH"), "exit_file_present": $([ -f "$EXITF" ] && echo true || echo false), "mem_available_kb": $(jn "$MEM_AVAIL_KB"), "swap_free_kb": $(jn "$SWAP_FREE_KB"), "bundle": $(js "$ROOT/$NAME")}
EOF

mv "$PART" "$ROOT/$NAME" 2>/dev/null || { rm -rf "$PART"; exit 0; }
tr -d '\n' < "$ROOT/$NAME/summary.json" >> "$ROOT/index.jsonl" 2>/dev/null && echo >> "$ROOT/index.jsonl"
echo "incident bundle: $ROOT/$NAME"

# ---- retention ---------------------------------------------------------------
# Bundle names start with a UTC timestamp, so a name sort IS a time sort.
# Leftover .partial-* dirs (a bundle killed by the 90 s timeout) go too.
find "$ROOT" -maxdepth 1 -name '.partial-*' -type d -mmin +5 -exec rm -rf {} + 2>/dev/null
# The bundle just written is excluded up front, so it can never be pruned
# (two bundles in one second would otherwise sort by class name).
_list() { ls -1 "$ROOT" 2>/dev/null | grep -E '^[0-9]{8}T[0-9]{6}Z-' | grep -vxF "$NAME" | sort; }
_n=$(_list | wc -l | tr -d ' ')
if [ "$_n" -ge "$KEEP" ]; then
    _list | head -n $((_n - KEEP + 1)) | while read -r _d; do
        rm -rf "${ROOT:?}/$_d"
    done
fi
while :; do
    _kb=$(du -sk "$ROOT" 2>/dev/null | awk '{print $1}')
    [ -n "$_kb" ] && [ "$_kb" -gt "$MAX_KB" ] || break
    _old=$(_list | head -1)
    [ -n "$_old" ] || break
    rm -rf "${ROOT:?}/$_old"
done
exit 0
