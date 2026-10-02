#!/usr/bin/env bash
# Deploy arch v3 to the training host: stop cleanly, sync, verify, relaunch, PROVE it.
#
# LESSONS BAKED IN (both cost a wasted deploy earlier today):
#   * the stop file is `runlogs/STOP` — NOT `STOP_LIFELONG`, which does not
#     exist and which the loop therefore ignored while I reported success.
#   * a stopped-and-relaunched run must be proven by a CHANGED PID, not by
#     `pgrep -c` returning 1: the old process survived and kept writing to
#     the renamed log (an open fd follows a rename), so every symptom of a
#     healthy new run was present while nothing had actually restarted.
set -uo pipefail
cd /workspace/devai || exit 1

OLD=$(pgrep -f "run_minecraft[.]py" | head -1)
echo "=== old PID=${OLD:-none} ==="

echo "=== graceful stop (runlogs/STOP) ==="
touch runlogs/STOP
for i in $(seq 1 24); do
  sleep 10
  pgrep -f "run_minecraft[.]py" >/dev/null || { echo "  exited after ~$((i*10))s"; break; }
done
pgrep -f "run_minecraft[.]py" >/dev/null && { echo "  SIGTERM"; pkill -TERM -f "run_minecraft[.]py"; sleep 20; }
pgrep -f "run_minecraft[.]py" >/dev/null && { echo "  SIGKILL"; pkill -KILL -f "run_minecraft[.]py"; sleep 8; }
pkill -KILL -x java 2>/dev/null; sleep 5
rm -f runlogs/STOP
echo "  stopped: runs=$(pgrep -fc "run_minecraft[.]py") java=$(pgrep -xc java || echo 0)"

echo "=== skill bank MUST be intact (never wiped) ==="
./venv_mc/bin/python -c "
import json; r=json.load(open('skill_bank_mc_curiosity/registry.json'))
sk=r.get('skills',[])
print('  skills:',len(sk))
print('  arch mix:', {a: sum(1 for s in sk if (s.get('arch') or 'flat')==a) for a in ('flat','conv')})
assert len(sk)>0, 'SKILL BANK EMPTY — ABORT'
" || exit 1

echo "=== contract + arch-v3 smokes on the training host ==="
# A MISSING OPTIONAL DEPENDENCY IS NOT A FAILING TEST. The training host has no
# `crafter` package, so _options_smoke (which builds a Crafter env for its
# integration half) can never pass there — and gating the relaunch on it
# stopped the run and refused to bring it back, for a reason unrelated to
# the code being deployed. Distinguish "broken" from "not installed here".
fail=0
for t in _no_scripted_skills_smoke _conv_encoder_smoke _mastery_wiring_smoke \
         _nested_options_smoke _action_widening_smoke _options_smoke; do
  out=$(PYTHONPATH=. ./venv_mc/bin/python tests/$t.py 2>&1)
  if [ $? -eq 0 ]; then echo "  ok   $t"
  elif echo "$out" | grep -q "ModuleNotFoundError"; then
    echo "  skip $t ($(echo "$out" | grep -o "No module named .*" | head -1))"
  else
    echo "  FAIL $t"; echo "$out" | tail -3 | sed "s/^/        /"; fail=$((fail+1))
  fi
done
[ "$fail" -gt 0 ] && { echo "*** $fail host smoke failure(s) — NOT relaunching ***"; exit 1; }

echo "=== relaunch ==="
mv -f runlogs/minecraft_lifelong_run.log runlogs/run_pre_archv3.log 2>/dev/null
nohup bash scripts/launch_lifelong.sh 1000000 > runlogs/launch.out 2>&1 &
sleep 120

NEW=$(pgrep -f "run_minecraft[.]py" | head -1)
echo "=== PROOF OF A NEW PROCESS ==="
echo "  old=${OLD:-none}  new=${NEW:-none}"
if [ -z "$NEW" ]; then echo "  *** NOTHING RUNNING ***"; tail -20 runlogs/launch.out; exit 1; fi
[ "$NEW" = "${OLD:-x}" ] && { echo "  *** PID UNCHANGED — relaunch did not take ***"; exit 1; }
ps -eo pid,etime,args | grep "run_minecraft[.]py" | grep -v grep | head -1

L=runlogs/minecraft_lifelong_run.log
echo "=== arch v3 LIVE? ==="
echo "  action_dim/arch: $(grep -a 'action_dim=' "$L" 2>/dev/null | head -1)"
echo "  conv policy:     $(grep -aic 'arch=conv\|CONV kdim' "$L" 2>/dev/null) mentions"
echo "  scripted slots:  $(grep -ac '<- SCRIPTED' "$L" 2>/dev/null) (must be 0)"
echo "  warm-start:      $(grep -a 'Warm-start' "$L" 2>/dev/null | tail -1 | cut -c1-110)"
echo "  errors:          $(grep -aiE 'Traceback|RuntimeError' "$L" 2>/dev/null | head -1 | cut -c1-90)"
echo "  alive: runs=$(pgrep -fc "run_minecraft[.]py") java=$(pgrep -xc java || echo 0)"
