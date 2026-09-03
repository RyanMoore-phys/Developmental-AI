#!/bin/bash
# Register at start, DE-register at stop.
#
# The de-registration trap is the important half. Without it every container
# restart leaves an extra dead runner listed in GitHub, and jobs can be queued
# onto a runner that no longer exists — they then sit "Queued" forever with no
# error, which is the single most confusing self-hosted failure mode.
set -euo pipefail

: "${GITHUB_URL:?set GITHUB_URL (e.g. https://github.com/OWNER/REPO)}"
: "${RUNNER_TOKEN:?set RUNNER_TOKEN — a REGISTRATION token from Settings -> Actions -> Runners -> New self-hosted runner. It expires in ~1 hour and is single-use; fetch a fresh one per (re)registration.}"
RUNNER_NAME="${RUNNER_NAME:-skybot-$(hostname)}"
# `skybot` costs nothing and lets a second runner be pinned later without
# touching any workflow. The workflows currently target bare `self-hosted`.
RUNNER_LABELS="${RUNNER_LABELS:-skybot,linux}"

cd /home/runner

cleanup() {
  echo "entrypoint: de-registering ${RUNNER_NAME}"
  # Best-effort: a token may have expired by shutdown time. Never fail the
  # stop path over it — a container that refuses to die is worse.
  ./config.sh remove --token "${RUNNER_TOKEN}" 2>/dev/null || \
    echo "entrypoint: de-register failed (token likely expired) — remove it in the GitHub UI"
  exit 0
}
trap cleanup SIGTERM SIGINT SIGQUIT

if [ ! -f .runner ]; then
  echo "entrypoint: registering ${RUNNER_NAME} (labels: ${RUNNER_LABELS})"
  ./config.sh \
    --url "${GITHUB_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME}" \
    --labels "${RUNNER_LABELS}" \
    --work _work \
    --unattended \
    --replace
else
  echo "entrypoint: already configured (persisted _work volume), reusing"
fi

# `& wait` NOT a bare exec: exec would replace this shell and the trap above
# would never fire, so the container would stop without de-registering.
./run.sh &
wait $!
