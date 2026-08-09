#!/bin/bash
# Launch the curiosity-redesign Minecraft training run (16 clients).
# Pre-flight checks the display/ollama infra, refuses to double-launch,
# and records the python PID for watchers.
#
#   bash scripts/launch_curiosity_run.sh [TIMESTEPS] [SEED]
set -e
cd /workspace/devai
TS=${1:-400000}
SEED=${2:-0}
LOG=podlogs/minecraft_curiosity_run.log

# refuse to double-launch
if pgrep -f "run_minecraft[.]py" >/dev/null; then
    echo "REFUSED: a run_minecraft.py is already alive"; exit 1
fi
# stale JVMs from probes would fight for ports
if pgrep -x java >/dev/null; then
    echo "REFUSED: stale java clients present — clean up first"; exit 1
fi
# infra
pgrep -f "Xvfb [:]77" >/dev/null || { nohup Xvfb :77 -screen 0 800x600x24 >/dev/null 2>&1 < /dev/null & sleep 2; }
DISPLAY=:77 sh -c 'pgrep -x openbox >/dev/null' || { DISPLAY=:77 nohup openbox >/dev/null 2>&1 < /dev/null & sleep 1; }
pgrep -x ollama >/dev/null || { nohup ollama serve > podlogs/ollama.log 2>&1 < /dev/null & sleep 5; }
ollama list | grep -q llava || { echo "REFUSED: llava model missing"; exit 1; }

DISPLAY=:77 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MINERL_HEADLESS=1 \
  nohup ./venv_mc/bin/python run_minecraft.py \
    --config configs/minecraft_curiosity.yaml \
    --timesteps "$TS" --seed "$SEED" \
    --out minecraft_curiosity_results > "$LOG" 2>&1 < /dev/null &
PY=$!
disown
echo "$PY" > podlogs/curiosity_run.pid
echo "LAUNCHED python_pid=$PY log=$LOG"
