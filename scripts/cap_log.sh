#!/usr/bin/env bash
# Size-cap a growing log: once it exceeds MAX_KB, keep the newest KEEP bytes
# in <file>.1 and truncate the live file.
#
# du (allocated blocks), NOT stat -c%s: a writer that opened the file with a
# truncate-mode `>` redirect keeps its byte offset after truncation, leaving
# a sparse file whose APPARENT size never shrinks — allocated blocks are
# what actually fill the disk, so blocks are what this guard measures.
# Truncating under an append-mode (`>>`) writer is fully safe: its next
# write lands at the new EOF.
#
# Born 2026-08-16: runlogs/ollama.log hit 581 MB in 5 days (llama-server at
# log-verbosity 4). Same lesson as the 18 GB java log, generalised into a
# reusable guard instead of a third one-off.
FILE=${1:?usage: cap_log.sh <file> [max_kb] [keep_bytes] [interval_s]}
MAX_KB=${2:-102400}      # cap: 100 MB
KEEP=${3:-10485760}      # rotate away keeping the newest 10 MB
EVERY=${4:-600}          # check every 10 min
while sleep "$EVERY"; do
    [ -f "$FILE" ] || continue
    kb=$(du -k "$FILE" 2>/dev/null | cut -f1)
    if [ "${kb:-0}" -gt "$MAX_KB" ]; then
        tail -c "$KEEP" "$FILE" > "$FILE.1" 2>/dev/null || true
        : > "$FILE"
        echo "$(date -u +%FT%TZ) capped $FILE (${kb} KB; newest $KEEP bytes kept in $FILE.1)"
    fi
done
