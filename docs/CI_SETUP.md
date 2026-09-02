# CI setup — self-hosted runner, no GitHub secrets, no branch protection

**What this design gives you:** no credential is committed, none is stored in
GitHub Secrets, and nothing depends on a repo setting your plan may not offer.
This is a single-maintainer private repo and the pipeline is built for that,
not for a team.

Audited 2026-09-02: the content of **every blob in this repo's history** was
scanned and returned **zero** credential hits.

---

## Why there are no secrets to store

A GitHub-hosted runner is a blank box that must be *given* credentials. A
runner on your own machine already has them:

| Needed | Hosted runner | Self-hosted (this design) |
|---|---|---|
| SSH to the pod | private key uploaded to GitHub Secrets | `~/.ssh/skybot_ed25519`, already on disk |
| Reach the pod | tailnet OAuth client | the machine can already reach it |
| Which pod / server | repo variables | runner `.env`, a local file |

`deploy.yml` and `pod.yml` reference **zero** `secrets.*` — verified.

## Why there is no branch protection

Review-before-merge needs a second person. There isn't one. The production
gates live in the workflows instead, and work on any plan:

- automatic deploys require `ci` to have passed
  (`workflow_run.conclusion == 'success'`); `workflow_dispatch` bypasses it on
  purpose — you are the reviewer;
- the deploy checkout is **pinned to the exact commit CI validated**
  (`workflow_run.head_sha`), so a commit landing mid-CI cannot slip out
  untested;
- a live training run is never implicitly stopped or restarted.

## ⚠️ Precondition: PRIVATE repositories only

A self-hosted runner must never serve a public repo — anyone could fork, open
a PR, and execute arbitrary code as your user on your machine. If this repo
ever goes public, deregister the runner first.

---

## 1. Install the runner

GitHub → repo → **Settings → Actions → Runners → New self-hosted runner**.
Install it **outside** the repo (e.g. `~/actions-runner`) so its working
directories can never be committed; `.gitignore` also blocks `actions-runner/`
and `_work/` as a second line of defence.

**Give it the label `skybot`.** Both workflows target `[self-hosted, skybot]`,
so the label is what routes jobs. When a node joins later, label it `skybot`
too and jobs follow with no workflow edit.

Run it as a service so it survives reboots:

```bash
cd ~/actions-runner
./svc.sh install && ./svc.sh start
```

## 2. Put connection values in the runner's `.env`

The runner reads `.env` from its own root and applies it to every job. **This
file is never committed — it is not in the repo at all.**

```bash
cat > ~/actions-runner/.env <<'EOF'
# --- required ---
POD_HOST=<redacted-host>          # pod IP (ssh transport, the default)
POD_SSH_PORT=19983               # ROTATES on every pod restart — see below
POD_SSH_KEYFILE=/Users/rimac/.ssh/skybot_ed25519

# --- required for `pod.yml action=connect` ---
MC_SERVER_TS_IP=100.64.0.11    # the Paper server's tailnet IP

# --- optional ---
# DEPLOY_TRANSPORT=tailscale     # use Tailscale SSH instead of ssh; needs the
                                 # ACL rule in section 4, and POD_HOST becomes
                                 # the MagicDNS name (e.g. devai-pod-2)
# TS_AUTHKEY=tskey-auth-...      # lets `provision` join a fresh pod to the
                                 # tailnet without a browser click
EOF
chmod 600 ~/actions-runner/.env
```

Restart the runner after editing (`./svc.sh stop && ./svc.sh start`) — `.env`
is read at service start.

### The one thing you must keep updated

**`POD_SSH_PORT` changes every time the pod restarts** (22655 → 22681 → 34276
→ 19983 → …). When a job fails at the preflight step with an ssh error, this
is almost always why. Update `.env`, restart the runner, re-run.

This is the single reason to consider the tailscale transport later: a MagicDNS
name is stable across rebuilds, so nothing needs updating. It costs one ACL
rule (section 4). The ssh default was chosen because it works **today**, with
zero extra configuration.

## 3. Verify, in this order

```bash
# 1. runner shows "Idle" under Settings -> Actions -> Runners
# 2. transport works, straight from the runner host:
POD_HOST=... POD_SSH_PORT=... bash scripts/pod_exec.sh 'hostname'
# 3. read-only workflow:  Actions -> pod -> Run workflow -> action=status
# 4. then deploy, then connect, then launch.
```

Do **not** test `action=provision` against a working pod: it does
`rm -rf mc-build` and rebuilds MineRL (40–60 min). It is guarded behind typing
`PROVISION`, which is a guard on your *pod*, not a branch policy.

## 4. Optional — Tailscale SSH transport

Only if you want to stop updating `POD_SSH_PORT`. On the pod, once:

```bash
tailscale set --ssh=true
```

Then in the Tailscale admin console:

```json
"ssh": [
  { "action": "accept", "src": ["autogroup:member"], "dst": ["tag:devai"], "users": ["root"] }
]
```

Use **`accept`**, not `check` — `check` requires interactive browser re-auth,
which in CI does not fail, it **hangs** until timeout.

Verified 2026-09-02: with `--ssh` on but no ACL rule, an inbound attempt
reaches the pod and tailscaled replies *"tailnet policy does not permit you to
SSH to this node"*. That message means the transport works and only policy is
missing — the expected state before adding the rule.

---

## Keeping CI honest

`ci.yml` pins versions **read off the running pod**, not guessed:

```
python 3.10 · torch 2.6.0 · numpy 2.2.6 · gymnasium 1.3.0
```

This matters: the first version installed unpinned latest and 6 of 12 suites
failed while all 12 passed locally — one stack mismatch, not six bugs. The
second attempt pinned `numpy<2` on the assumption numpy 2 was the culprit;
the pod actually runs numpy **2.2.6**, and the failing suites were then re-run
*on the pod* against it and all passed. Guessing was wrong twice; reading the
versions off production was right.

**When you upgrade the pod, update these pins in the same change.** Re-read
them with:

```bash
bash scripts/pod_exec.sh 'cd /workspace/devai && ./venv_mc/bin/pip list'
```

A CI that tests a stack the pod does not run is worse than no CI — it reports
green for a configuration nobody deploys.

---

## What is stored where

| Value | Where | In git? | In GitHub? |
|---|---|---|---|
| `POD_HOST`, `POD_SSH_PORT` | runner `.env` | no | no |
| SSH private key | `~/.ssh/` on the runner host | no | no |
| `MC_SERVER_TS_IP` | runner `.env` | no | no |
| `TS_AUTHKEY` (optional) | runner `.env` | no | no |
| `CI_RUNS_ON` (optional) | repo variable | no | yes — non-secret |
