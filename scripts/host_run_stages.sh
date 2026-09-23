#!/usr/bin/env bash
# Run the staged test sequence IN ORDER, stopping at the first gate failure.
#
# WHY A SCRIPT AND NOT A CHECKLIST. The order is load-bearing: Stage 0 decides
# whether MineRL can support this work at all, and running Stage 2 first would
# burn a client boot to learn something Stage 0 answers in a frame dump. The
# gates are the point, so they are enforced rather than remembered.
#
#   bash scripts/pod_run_stages.sh              # 0 -> 1 -> 2 -> 3
#   bash scripts/pod_run_stages.sh 0            # just the blocking one
#   STEPS2=200 STEPS3=150 bash scripts/pod_run_stages.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
ONLY="${1:-all}"
STEPS2="${STEPS2:-500}"
STEPS3="${STEPS3:-300}"
PY="${PY:-python3}"
LOG=podlogs/stages
mkdir -p "$LOG"

hdr() { echo; echo "############ $* ############"; }
fail() { echo; echo "!!!! GATE FAILED: $* — stopping here, as designed."; exit 1; }

if [ "$ONLY" = "all" ] || [ "$ONLY" = "0" ]; then
  hdr "STAGE 0 — MineRL feasibility (BLOCKING)"
  $PY scripts/pod_stage0.py --all 2>&1 | tee "$LOG/stage0.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "Stage 0"
fi
[ "$ONLY" = "0" ] && exit 0

if [ "$ONLY" = "all" ] || [ "$ONLY" = "1" ]; then
  hdr "STAGE 1 — every offline suite on real hardware"
  $PY tests/run_all.py all 2>&1 | tee "$LOG/stage1.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "Stage 1 (a LEGACY failure is a regression from this work)"
fi
[ "$ONLY" = "1" ] && exit 0

if [ "$ONLY" = "all" ] || [ "$ONLY" = "2" ]; then
  hdr "STAGE 2 — ${STEPS2} steps on the bare adapter"
  $PY scripts/pod_stage2_integration.py --steps "$STEPS2" 2>&1 | tee "$LOG/stage2.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "Stage 2"
fi
[ "$ONLY" = "2" ] && exit 0

if [ "$ONLY" = "all" ] || [ "$ONLY" = "3" ]; then
  hdr "STAGE 3 — ${STEPS3} steps of the full loop"
  $PY scripts/pod_stage3_loop.py --steps "$STEPS3" 2>&1 | tee "$LOG/stage3.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "Stage 3"
  hdr "FALSIFIERS over Stage 3's metrics"
  $PY scripts/pod_falsifiers.py --metrics podlogs/stage3/metrics.json 2>&1 | tee "$LOG/falsifiers.txt"
fi

echo
echo "ALL STAGES PASSED. Next is Stage 4 (the flow go/no-go), which needs"
echo "hours, not minutes — see docs/TESTING_PLAN.md."
