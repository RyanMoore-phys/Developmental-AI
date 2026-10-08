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
# SKYBOT_ROOT exists for tests/_incident_bundle_smoke.py, which runs this
# script in a temp copy; on the host it is always /workspace/devai.
cd "${SKYBOT_ROOT:-/workspace/devai}"
TS=${1:-1000000}
# A MANUAL LAUNCH CLEARS THE CRASH-LOOP LATCH (2026-10-07). The supervisor
# writes runlogs/CRASHLOOP and stops relaunching after the same exit class +
# signal 3x in 30 min (supervise_skybot.sh). It calls this script with
# SKYBOT_SUPERVISED=1; anything else is a human (or host.yml) deciding to run
# again, which is exactly the escape that guard needs (CLAUDE.md 4.1).
if [ -z "${SKYBOT_SUPERVISED:-}" ] && [ -f runlogs/CRASHLOOP ]; then
  echo "  manual launch: clearing runlogs/CRASHLOOP ($(head -1 runlogs/CRASHLOOP 2>/dev/null))"
  rm -f runlogs/CRASHLOOP runlogs/crash_history
fi
# ---- PREFLIGHT BEGIN (tests/_incident_bundle_smoke.py strips to END) ------
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

  # ---- IF THE VLM IS OFF, OLLAMA HAS NO REASON TO RUN (2026-09-24) --------
  # Unloading the MODEL is not the same as stopping the SERVICE, and the
  # service will happily reload 8 GB the moment anything asks. With
  # symbolic_grounding disabled nothing should ask -- but provisioning starts
  # `ollama serve` unconditionally (STAGE 6) and the installer leaves a
  # systemd unit behind, so it comes back on every boot and reprovision.
  # On a 15.3 GB box that is 54% of RAM held for a sensor we deliberately
  # turned off. Read the CONFIG, not a flag here, so this can never disagree
  # with what the agent is actually doing.
  _VLM_ON=$(./venv_mc/bin/python -c "import yaml,sys; c=yaml.safe_load(open('configs/minecraft_skybot.yaml')); print('1' if (c.get('symbolic_grounding') or {}).get('enabled') else '0')" 2>/dev/null || echo 1)
  if [ "$_VLM_ON" = "0" ]; then
    echo "  symbolic_grounding disabled -> stopping ollama entirely"
    sudo systemctl disable --now ollama 2>/dev/null || true
    pkill -x ollama 2>/dev/null || true
    pkill -f llama-server 2>/dev/null || true
    sleep 2
    echo "  ollama procs left: $(pgrep -cx ollama 2>/dev/null || echo 0)"
  fi
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
# ---- PREFLIGHT END ---------------------------------------------------------
rm -rf runlogs/brain
# OMP_NUM_THREADS 16 -> 6 (2026-09-22): the training host had 128 cores, this box
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
# NO CORE DUMPS FOR THE AGENT (2026-10-07). An abort or segfault of this
# process dumps ~8-10 GB of RAM plus CUDA mappings to disk on a 16 GB host
# with one NVMe. On 2026-10-06 the host froze 22 s after the agent aborted
# at exit (`terminate called without an active exception`; fixed in
# run_minecraft.py). A dump of this process has never been used to debug
# anything; the run log and runlogs/crashes.log carry the traceback.
ulimit -c 0

# KEEP THE PREVIOUS RUNS' LOGS (2026-10-04). The `>` below truncates, so
# every launch -- including each supervisor relaunch after a crash -- erased
# the last run's log, and there was no earlier `Loop timing` to compare a
# slow run against. Rotate the last 3 instead. The supervisor reads the
# crashed log's tail BEFORE it calls this script, so it still sees it.
for i in 2 1; do
  [ -f "runlogs/minecraft_skybot_run.log.$i" ] && \
    mv -f "runlogs/minecraft_skybot_run.log.$i" "runlogs/minecraft_skybot_run.log.$((i + 1))"
done
[ -f runlogs/minecraft_skybot_run.log ] && \
  mv -f runlogs/minecraft_skybot_run.log runlogs/minecraft_skybot_run.log.1
# EXIT CAPTURE (2026-10-07). The agent's exit code used to die with it: the
# supervisor only ever saw "pid gone", so a clean STOP, an abort at exit
# (rc 134, the 2026-10-06 freeze) and an OOM kill (137) were indistinguishable
# (docs/foundation/TELEMETRY_INVENTORY_OPS.md 2.1, gap G1). A tiny bash
# wrapper now owns the python process and outlives it by microseconds:
#   * `sh -c 'echo $$ > pid; exec python'` -- the pid it writes IS the
#     agent's pid (exec keeps it), so runlogs/skybot_run.pid, the
#     supervisor's kill -0, deploy_skybot.sh's `ps -p PID -o args= | grep
#     minecraft` + SIGTERM, and host_stop_wait.sh all address the AGENT as
#     before. ($BASHPID would do it in bash 4, not in the 3.2 the test runs.)
#   * python runs in the FOREGROUND of the wrapper, so it keeps default
#     SIGINT/SIGQUIT handling (a background job of a non-interactive shell
#     would have them ignored).
#   * the wrapper traps TERM/INT/HUP, so a signal to the whole process group
#     still lets it write runlogs/skybot_run.exit after python dies.
# skybot_run.exit is key=value lines: exit_code, signal (NAME or "none"),
# pid, t_start_epoch, t_end_epoch, t_end (ISO UTC). rc>128 = signal rc-128
# (134 ABRT, 137 KILL = OOM/earlyoom, 139 SEGV, 143 TERM). It is written
# .tmp + mv, so a reader never sees half of it, and removed here first, so a
# stale one is never read for a new run.
# `pgrep -f "run_minecraft[.]py"` also matches the wrapper -- harmless: it
# is alive exactly when the agent is.
rm -f runlogs/skybot_run.exit runlogs/skybot_run.pid
_WRAP=$(cat <<'WRAPEOF'
pidf=$1; exitf=$2; shift 2
trap 'GOT_SIG=1' TERM INT HUP
t0=$(date +%s)
sh -c 'echo $$ > "$0.tmp" && mv -f "$0.tmp" "$0"; exec "$@"' "$pidf" "$@"
rc=$?
pid=$(cat "$pidf" 2>/dev/null || echo unknown)
sig=none
if [ "$rc" -gt 128 ] && [ "$rc" -lt 160 ]; then
  sig=$(kill -l $((rc - 128)) 2>/dev/null || echo "$((rc - 128))")
  sig=SIG${sig#SIG}
fi
{
  echo "exit_code=$rc"
  echo "signal=$sig"
  echo "pid=$pid"
  echo "t_start_epoch=$t0"
  echo "t_end_epoch=$(date +%s)"
  echo "t_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$exitf.tmp" && mv -f "$exitf.tmp" "$exitf"
exit "$rc"
WRAPEOF
)
# setsid is always present on the host; the guard is for the test on macOS.
_SETSID=$(command -v setsid || true)
DISPLAY=:77 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 MINERL_HEADLESS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $_SETSID bash -c "$_WRAP" skybot-exit-wrapper \
    runlogs/skybot_run.pid runlogs/skybot_run.exit \
    ./venv_mc/bin/python run_minecraft.py \
    --config configs/minecraft_skybot.yaml \
    --timesteps "$TS" --seed 0 --out minecraft_skybot_results \
    > runlogs/minecraft_skybot_run.log 2>&1 < /dev/null &
WRAPPER=$!
disown
# The wrapper writes the AGENT pid within milliseconds; wait (bounded) for it.
# Fallback: the wrapper's own pid, which lives exactly as long as the agent.
PY=""
for _i in $(seq 1 50); do
  PY=$(cat runlogs/skybot_run.pid 2>/dev/null || true)
  [ -n "$PY" ] && break
  sleep 0.2
done
if [ -z "$PY" ]; then
  PY=$WRAPPER
  echo "$PY" > runlogs/skybot_run.pid
fi
echo "LAUNCHED skybot pid=$PY"
