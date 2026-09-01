#!/bin/bash
set -e
cd /workspace/devai
TS=${1:-1000000}
if pgrep -f "run_minecraft[.]py" >/dev/null; then echo "REFUSED: run alive"; exit 1; fi
# ---- REAP ORPHANED MINECRAFT CLIENTS (2026-07-27) --------------------
# Measured: 6 java processes alive for a 4-client run — clients from earlier
# restarts survive the trainer's death and are never reaped (~1.4GB each, so
# a few restart cycles silently eat the box).
#
# ORDER IS LOAD-BEARING. This MUST sit above the "REFUSED: stale java" guard
# below: a first version was placed after it and was therefore unreachable —
# the guard exited first, every relaunch refused, and the run stayed down.
# No trainer is alive at this point (the check above just proved it), so any
# java still running is by definition an orphan and safe to kill.
_stale=$(pgrep -x java | wc -l | tr -d " ")
if [ "${_stale:-0}" -gt 0 ]; then
  echo "[launch] reaping $_stale orphaned java client(s) — no trainer alive"
  pkill -KILL -x java 2>/dev/null || true
  sleep 3
fi
if pgrep -x java >/dev/null; then echo "REFUSED: stale java survived reap"; exit 1; fi
pgrep -f "Xvfb [:]77" >/dev/null || { nohup Xvfb :77 -screen 0 800x600x24 >/dev/null 2>&1 < /dev/null & sleep 2; }
DISPLAY=:77 sh -c "pgrep -x openbox >/dev/null" || { DISPLAY=:77 nohup openbox >/dev/null 2>&1 < /dev/null & sleep 1; }
pgrep -x ollama >/dev/null || { OLLAMA_DEBUG=0 nohup ollama serve >> podlogs/ollama.log 2>&1 < /dev/null & sleep 5; }
# append-mode above + capper below: 581 MB of VLM-server chatter in 5 days
# (measured 2026-08-16) would eat the disk on a long lifelong run
pgrep -f "cap_log[.]sh podlogs/ollama[.]log" >/dev/null || \
  { nohup bash scripts/cap_log.sh podlogs/ollama.log >> podlogs/cap_log.log 2>&1 < /dev/null & }
rm -rf podlogs/brain
DISPLAY=:77 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MINERL_HEADLESS=1 \
  setsid ./venv_mc/bin/python run_minecraft.py \
    --config configs/minecraft_skybot.yaml \
    --timesteps "$TS" --seed 0 --out minecraft_lifelong_results \
    > podlogs/minecraft_lifelong_run.log 2>&1 < /dev/null &
PY=$!
disown
echo "$PY" > podlogs/lifelong_run.pid
echo "LAUNCHED lifelong pid=$PY"
