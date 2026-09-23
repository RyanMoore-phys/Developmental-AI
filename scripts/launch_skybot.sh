#!/bin/bash
# Launch the lifelong organism against the user's external Paper server
# ("SkyBot"): EVERY stream joins the server via the host-local socat bridge
# on 127.0.0.1:25565 -- `remote_server_scope: all` has been set since
# 2026-08-17, so no stream runs a local generated world any more. (This
# header said "the scout stays in a local generated world" until 2026-09-22;
# it was stale, and it is the first thing anyone reads when asking whether
# the run is really on the external server.)
#
# The bridge's far side is EITHER the tailnet or the LAN -- see
# scripts/connect_server.sh; this endpoint is identical either way, which is
# why nothing here needs to know which.
# Pre-flight refuses to start unless the bridge + tailnet are actually up —
# a run launched with a dead bridge would strand the primary in a
# connect-fail rebuild loop.
set -e
cd /workspace/devai
TS=${1:-1000000}
if pgrep -f "run_minecraft[.]py" >/dev/null; then echo "REFUSED: run alive"; exit 1; fi
# ---- STALE JAVA: CLEAR BUILD RESIDUE, REFUSE ONLY ON A REAL CLIENT --------
# This was a bare `pgrep -x java -> REFUSED: stale java` (2026-09-23), and it
# was a LATCH in the CLAUDE.md 4.1 sense: STAGE 3b of provisioning runs a
# gradle build, gradle leaves a DAEMON alive for hours, nothing in this repo
# ever stops it -- so the very act of provisioning a fresh host made the next
# launch impossible, forever, until a human noticed. The supervisor just
# retried every 60s printing four words that named no process.
# A gradle daemon is BUILD RESIDUE and never a competing Minecraft client, so
# it is safe to stop. Anything else still refuses -- that guard is real, a
# second client would fight for the server slots -- but now it PRINTS what it
# found, which is the difference between a diagnosis and a guess.
# Pattern SPLIT so pkill -f cannot match this script's own cmdline.
_GD="Gradle""Daemon"
if pgrep -f "$_GD" >/dev/null 2>&1; then
  echo "  clearing gradle daemon(s) left by the MineRL build"
  pkill -f "$_GD" 2>/dev/null || true
  sleep 3
fi
if pgrep -x java >/dev/null; then
  echo "REFUSED: stale java — not gradle residue. Running java processes:"
  ps -eo pid,etime,args | grep "[j]ava" | head -5 | sed 's/^/    /'
  echo "  If these are dead MineRL clients, clear them with: pkill -9 -x java"
  exit 1
fi
# ---- UNLOAD ANY RESIDENT OLLAMA MODEL (2026-09-23) ------------------------
# ollama serve is a SEPARATE long-running service and the VLM is requested with
# keep_alive=-1 (pin forever, deliberately: it is called every ~30-50 steps and
# reloading each time would cost more than it saves). The consequence is that
# RESTARTING THE AGENT DOES NOT RELOAD THE MODEL -- so a change to
# symbolic_grounding.num_ctx has NO EFFECT until the model is unloaded, because
# the KV cache was sized when it was first loaded, possibly days earlier.
# That is exactly how the num_ctx fix looked like it had failed: llama-server
# still at 7.8 GB after a restart, holding a cache for a context the new config
# had already reduced.
# Unloading here makes the next generate call re-read the options. Cost: one
# model load (~seconds) per launch. Non-fatal throughout -- a box with no
# ollama, or a dead one, must still be able to start training.
if command -v ollama >/dev/null 2>&1; then
  _LOADED=$(ollama ps 2>/dev/null | awk "NR>1 {print \$1}")
  for _m in $_LOADED; do
    echo "  unloading resident model: $_m (so num_ctx is re-read)"
    ollama stop "$_m" >/dev/null 2>&1 || true
  done
  [ -n "$_LOADED" ] && sleep 2
fi
pgrep -x tailscaled >/dev/null || { echo "REFUSED: tailscaled down"; exit 1; }
ss -tln | grep -q "127.0.0.1:25565" || { echo "REFUSED: socat bridge down"; exit 1; }
timeout 12 python3 scripts/mc_ping.py 127.0.0.1 25565 754 >/dev/null 2>&1 \
    || { echo "REFUSED: server unreachable through bridge"; exit 1; }
pgrep -f "Xvfb [:]77" >/dev/null || { nohup Xvfb :77 -screen 0 800x600x24 >/dev/null 2>&1 < /dev/null & sleep 2; }
DISPLAY=:77 sh -c "pgrep -x openbox >/dev/null" || { DISPLAY=:77 nohup openbox >/dev/null 2>&1 < /dev/null & sleep 1; }
pgrep -x ollama >/dev/null || { OLLAMA_DEBUG=0 nohup ollama serve >> runlogs/ollama.log 2>&1 < /dev/null & sleep 5; }
# append-mode above + capper below: 581 MB of VLM-server chatter in 5 days
# (measured 2026-08-16) would eat the disk on a long lifelong run
pgrep -f "cap_log[.]sh runlogs/ollama[.]log" >/dev/null || \
  { nohup bash scripts/cap_log.sh runlogs/ollama.log >> runlogs/cap_log.log 2>&1 < /dev/null & }
rm -rf runlogs/brain
# OMP_NUM_THREADS 16 -> 6 (2026-09-22): the pod had 128 cores, this box
# has 6/12 and two of them are pinned to Minecraft clients by the
# launchClient taskset wrap. Oversubscribing torch against that costs
# throughput rather than buying it.
# PYTORCH_CUDA_ALLOC_CONF (2026-09-23). The agent died with:
#   CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total
#   capacity of 7.56 GiB of which 68.88 MiB is free.
# 64 MiB failing on a card with 3+ GiB held by this process is FRAGMENTATION,
# not exhaustion, and torch names the fix in the error text itself.
# expandable_segments lets the allocator grow a segment instead of needing a
# contiguous block. This does NOT create memory: it is worth ~a few hundred MB
# of headroom, and the real fix for this box is the VLM KV-cache cap in
# configs (symbolic_grounding.num_ctx). Keep both.
DISPLAY=:77 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 MINERL_HEADLESS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid ./venv_mc/bin/python run_minecraft.py \
    --config configs/minecraft_skybot.yaml \
    --timesteps "$TS" --seed 0 --out minecraft_skybot_results \
    > runlogs/minecraft_skybot_run.log 2>&1 < /dev/null &
PY=$!
disown
echo "$PY" > runlogs/skybot_run.pid
echo "LAUNCHED skybot pid=$PY"
