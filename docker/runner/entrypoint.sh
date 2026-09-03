#!/bin/bash
# Register at start, DE-register at stop.
#
# The de-registration trap is the important half. Without it every container
# restart leaves an extra dead runner listed in GitHub, and jobs can be queued
# onto a runner that no longer exists — they then sit "Queued" forever with no
# error, which is the single most confusing self-hosted failure mode.
set -euo pipefail

: "${GITHUB_URL:?set GITHUB_URL (e.g. https://github.com/OWNER/REPO)}"

# ---- SELF-MINTING REGISTRATION TOKENS (added 2026-09-02) -----------------
# A registration token is SINGLE-USE and expires in ~1 hour, and every
# `down -v` forces a fresh registration — so the manual flow breaks on every
# reboot, rebuild or power cut, failing with an unhelpful
#     api.github.com/actions/runner-registration : 404
# which reads like a wrong URL rather than a dead token.
#
# With GITHUB_PAT set, mint one on the fly instead. THE TRADE-OFF IS REAL and
# worth stating: this replaces a short-lived single-use token with a
# long-lived credential stored on this host. Use a FINE-GRAINED PAT limited to
# THIS repository with exactly one permission — Administration: Read and write
# — which is far narrower than a classic `repo` token. Do not use a classic
# token here.
#
# Without GITHUB_PAT the original manual path still works unchanged.
if [ -n "${GITHUB_PAT:-}" ]; then
  OWNER_REPO="${GITHUB_URL#https://github.com/}"
  OWNER_REPO="${OWNER_REPO%/}"
  OWNER_REPO="${OWNER_REPO%.git}"
  echo "entrypoint: minting a registration token for ${OWNER_REPO}"
  _resp=$(curl -sS -X POST \
    -H "Authorization: Bearer ${GITHUB_PAT}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${OWNER_REPO}/actions/runners/registration-token" \
    2>&1) || true
  RUNNER_TOKEN=$(printf '%s' "$_resp" | jq -r '.token // empty')
  if [ -z "$RUNNER_TOKEN" ]; then
    echo "entrypoint: could not mint a token. GitHub said:" >&2
    printf '%s\n' "$_resp" | head -5 >&2
    echo "  -> check GITHUB_PAT is a fine-grained PAT for ${OWNER_REPO}" >&2
    echo "     with Administration: Read and write, and is not expired." >&2
    exit 1
  fi
  echo "entrypoint: token minted OK"
fi

: "${RUNNER_TOKEN:?set RUNNER_TOKEN (a registration token from Settings -> Actions -> Runners -> New self-hosted runner; single-use, ~1h) OR set GITHUB_PAT to mint one automatically}"
RUNNER_NAME="${RUNNER_NAME:-skybot-$(hostname)}"
# `skybot` costs nothing and lets a second runner be pinned later without
# touching any workflow. The workflows currently target bare `self-hosted`.
RUNNER_LABELS="${RUNNER_LABELS:-skybot,linux}"

cd /home/runner

cleanup() {
  echo "entrypoint: de-registering ${RUNNER_NAME}"
  # De-registration needs a REMOVE token, which is a different (and equally
  # short-lived) token from the registration one. With a PAT we can mint a
  # fresh one at shutdown; without, the registration token is almost certainly
  # expired by now and removal will fail — hence best-effort.
  _rm="${RUNNER_TOKEN}"
  if [ -n "${GITHUB_PAT:-}" ]; then
    _rm=$(curl -sS -X POST \
      -H "Authorization: Bearer ${GITHUB_PAT}" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/${OWNER_REPO}/actions/runners/remove-token" \
      2>/dev/null | jq -r '.token // empty') || _rm="${RUNNER_TOKEN}"
    [ -n "$_rm" ] || _rm="${RUNNER_TOKEN}"
  fi
  # Never fail the stop path over this — a container that refuses to die is
  # worse than a stale entry in the runners list.
  ./config.sh remove --token "${_rm}" 2>/dev/null || \
    echo "entrypoint: de-register failed (token expired?) — remove it in the GitHub UI"
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
