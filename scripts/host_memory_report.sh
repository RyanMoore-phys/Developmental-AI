#!/usr/bin/env bash
# host_memory_report.sh — who holds RAM, swap and VRAM on the training host.
#
# Run ON the host (no sudo needed for your own processes; run with sudo to see
# PSS/swap of other users' processes, e.g. ollama's). Read-only.
#
#   bash scripts/host_memory_report.sh [repo_dir]
#
# Sections: free -h; per-process RSS/PSS/Swap (smaps_rollup + status) grouped
# into agent / java clients / ollama / docker / other; top swap users; GPU
# per process (nvidia-smi — the java clients render through VirtualGL, so
# check whether they show up here); then the latest in-process breakdown from
# runlogs/memory_census.jsonl via tools/memory_report.py, if present.
#
# MEASUREMENT ONLY. Changes nothing, signals nothing. (Never `kill` the agent:
# the supervisor treats it as a crash — use `touch runlogs/STOP`.)

set -u
REPO="${1:-/workspace/devai}"

hr() { printf '\n== %s ==\n' "$1"; }

hr "free -h"
free -h 2>/dev/null || echo "free not available"

hr "per-process memory (MB), sorted by PSS"
# One line per process: pid group rss pss swap name cmd
TMPF="$(mktemp 2>/dev/null || echo /tmp/host_mem_report.$$)"
for d in /proc/[0-9]*; do
    pid="${d#/proc/}"
    [ -r "$d/status" ] || continue
    name="$(awk '/^Name:/{print $2; exit}' "$d/status" 2>/dev/null)"
    rss_kb="$(awk '/^VmRSS:/{print $2; exit}' "$d/status" 2>/dev/null)"
    swap_kb="$(awk '/^VmSwap:/{print $2; exit}' "$d/status" 2>/dev/null)"
    [ -n "${rss_kb:-}" ] || continue        # kernel threads have no VmRSS
    pss_kb=""
    if [ -r "$d/smaps_rollup" ]; then
        pss_kb="$(awk '/^Pss:/{print $2; exit}' "$d/smaps_rollup" 2>/dev/null)"
    fi
    cmd="$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null | cut -c1-90)"
    case "$cmd" in
        *run_minecraft.py*|*developmental_ai*) grp="agent" ;;
        *java*|*minecraft*|*Malmo*|*mcp*)       grp="java"  ;;
        *ollama*)                               grp="ollama" ;;
        *dockerd*|*containerd*|*docker*)        grp="docker" ;;
        *)                                      grp="other" ;;
    esac
    [ "$name" = "java" ] && grp="java"
    printf '%s %s %s %s %s %s %s\n' "$pid" "$grp" "${rss_kb:-0}" \
        "${pss_kb:--1}" "${swap_kb:-0}" "${name:-?}" "${cmd:-}" >> "$TMPF"
done

printf '%-8s %-7s %9s %9s %9s  %-16s %s\n' PID GROUP RSS PSS SWAP NAME CMD
sort -k4,4nr -k3,3nr "$TMPF" | head -40 | awk '{
    pss = ($4 < 0) ? "n/a(sudo)" : sprintf("%.0f", $4/1024);
    cmd = ""; for (i = 7; i <= NF; i++) cmd = cmd " " $i;
    printf "%-8s %-7s %9.0f %9s %9.0f  %-16s%s\n", $1, $2, $3/1024, pss, $5/1024, $6, cmd
}'

hr "totals by group (MB) — PSS is the fair share; RSS double-counts shared pages"
awk '{
    g = $2; rss[g] += $3; sw[g] += $5; n[g]++;
    if ($4 >= 0) pss[g] += $4; else miss[g]++;
} END {
    printf "%-8s %6s %10s %10s %10s\n", "GROUP", "PROCS", "RSS", "PSS", "SWAP";
    for (g in rss) printf "%-8s %6d %10.0f %10.0f %10.0f%s\n", g, n[g],
        rss[g]/1024, pss[g]/1024, sw[g]/1024,
        (miss[g] ? "  (" miss[g] " PSS unreadable: rerun with sudo)" : "");
}' "$TMPF" | sort -k3,3nr

hr "top swap users (MB)"
sort -k5,5nr "$TMPF" | awk '$5 > 0' | head -15 | \
    awk '{printf "%-8s %-7s swap %8.0f  %s\n", $1, $2, $5/1024, $6}'
rm -f "$TMPF"

hr "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv 2>/dev/null
    echo
    echo "compute apps (CUDA contexts):"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
        2>/dev/null || echo "  (query failed)"
    echo
    # Graphics contexts (VirtualGL EGL from the java clients) are NOT listed
    # by --query-compute-apps; the plain table shows both C and G types.
    echo "all GPU processes (type C = compute, G = graphics; java via VirtualGL shows here if at all):"
    nvidia-smi 2>/dev/null | awk '/Processes:/{p=1} p' | sed -n '1,40p'
else
    echo "nvidia-smi not found"
fi

hr "in-process census (runlogs/memory_census.jsonl)"
CENSUS="$REPO/runlogs/memory_census.jsonl"
if [ -f "$CENSUS" ]; then
    PY="$REPO/venv/bin/python"
    [ -x "$PY" ] || PY="python3"
    (cd "$REPO" && PYTHONPATH=. "$PY" tools/memory_report.py "$CENSUS" --trend 12) \
        || echo "memory_report.py failed"
else
    echo "no census yet at $CENSUS"
    echo "request one: touch $REPO/runlogs/MEMCENSUS  (written at the next segment end)"
fi
