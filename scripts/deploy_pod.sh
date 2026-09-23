#!/usr/bin/env bash
# Push this tree to a RunPod box and set it up. Idempotent; safe to re-run.
#
# NEVER TOUCHES skill_bank_mc_curiosity/ OR podlogs/ (CLAUDE.md 3 and 6):
# the skill bank is the agent's accumulated developmental memory and podlogs
# is the only record of what happened. Code flows Mac -> pod; brain state
# flows pod -> Mac, and only via pull_brain.sh.
#
#   scripts/deploy_pod.sh <ssh-target> [ssh-port]
#   scripts/deploy_pod.sh <user>@<ssh-host>
#   scripts/deploy_pod.sh root@1.2.3.4 56569
set -euo pipefail

TARGET="${1:?usage: deploy_pod.sh <user@host> [port]}"
PORT="${2:-22}"
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE="${REMOTE_DIR:-/workspace/skybot}"
SSH="ssh -p ${PORT} -i ${KEY} -o StrictHostKeyChecking=no"

echo "==> target ${TARGET}:${PORT}  ->  ${REMOTE}"
${SSH} "${TARGET}" "mkdir -p ${REMOTE}/podlogs"

echo "==> rsync code"
rsync -az --delete -e "${SSH}" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'venv' --exclude '.venv' \
  --exclude 'podlogs' --exclude 'skill_bank_mc_curiosity' \
  --exclude '*.pt' --exclude '*.pth' --exclude 'assets' \
  ./ "${TARGET}:${REMOTE}/"

echo "==> environment"
${SSH} "${TARGET}" "cd ${REMOTE} && bash scripts/setup_pod_env.sh"

echo "==> offline suites (no MineRL needed)"
${SSH} "${TARGET}" "cd ${REMOTE} && PYTHONPATH=. python3 tests/run_all.py unit && PYTHONPATH=. python3 tests/run_all.py contract"

cat <<EOF

==> deployed. Next, IN ORDER (docs/TESTING_PLAN.md):

  ${SSH} ${TARGET}
  cd ${REMOTE}

  # Stage 0 — BLOCKING feasibility. Do not skip.
  PYTHONPATH=. python3 scripts/pod_stage0.py --all

  # Stage 1 — every suite on real hardware, incl. legacy
  PYTHONPATH=. python3 tests/run_all.py all

  # Stage 2 — 500 steps against real MineRL
  PYTHONPATH=. python3 scripts/pod_stage2_integration.py --steps 500
EOF
