# CI setup — self-hosted runner, zero secrets in GitHub

**The property this design buys:** no credential is ever committed, and no
credential is ever stored in GitHub Secrets either. Every value the pipeline
needs lives in one file on the runner host, outside the repository.

Audited 2026-09-02: the content of **every blob in this repo's history** was
scanned and returned **zero** credential hits. The setup below is what keeps
that true.

---

## Why self-hosted removes the secrets, rather than just moving them

A GitHub-hosted runner is a fresh box that knows nothing, so it must be *given*
credentials — a tailnet OAuth client to reach the pod, and an SSH key to
bootstrap a bare one. Both would have to be uploaded to GitHub Secrets.

A self-hosted runner on a machine you already use has both already:

| Needed | GitHub-hosted | Self-hosted |
|---|---|---|
| Reach the pod over the tailnet | `TS_OAUTH_CLIENT_ID` + `TS_OAUTH_SECRET` | host is already on the tailnet — **nothing** |
| SSH into a bare pod to provision | `POD_SSH_KEY` (private key upload) | `~/.ssh/skybot_ed25519` already on disk — **nothing** |
| Know which pod / server | repo variables | runner `.env` — local file |

So the only secret-shaped value left is `TS_AUTHKEY`, used once per pod
rebuild, and it lives in the runner's local `.env` too.

---

## ⚠️ Precondition: PRIVATE repositories only

A self-hosted runner must never serve a public repo. On a public repo anyone
can fork it, open a pull request, and have your runner execute their code **as
your user, on your machine**. This repo is private; if that ever changes,
deregister the runner *first*.

---

## 1. Install the runner

GitHub → repo → **Settings → Actions → Runners → New self-hosted runner**, pick
your OS, and follow the generated commands. Install it **outside** the repo
(e.g. `~/actions-runner`) so its working directories can never be committed —
`.gitignore` also blocks `actions-runner/` and `_work/` as a second line of
defence.

When it asks for labels, add **`skybot`**. The workflows target
`[self-hosted, skybot]`, so the label is what routes jobs to the right box.
That matters later when a node joins: label the node `skybot` too and jobs
follow it with no workflow edit.

Run it as a service so it survives reboots:

```bash
cd ~/actions-runner
./svc.sh install     # macOS/Linux
./svc.sh start
```

## 2. Put the connection values in the runner's `.env`

The runner reads `.env` from its own root directory and applies it to every
job. **This file is never committed — it is not in the repo at all.**

```bash
cat > ~/actions-runner/.env <<'EOF'
# Pod on the tailnet — stable across rebuilds, unlike an IP.
POD_HOST=devai-pod-2

# The Paper server's tailnet IP, for the socat bridge.
MC_SERVER_TS_IP=100.64.0.11

# Bootstrap-only: reaching a BARE pod that is not on the tailnet yet.
# Both change on every pod rebuild — this file is the one place to update.
POD_SSH_HOST=<current-pod-ip>
POD_SSH_PORT=<current-ssh-port>
POD_SSH_KEYFILE=/Users/<you>/.ssh/skybot_ed25519

# Ephemeral, pre-authorised, tag:devai auth key. Used once per pod rebuild so
# provisioning can join the tailnet without a browser click.
TS_AUTHKEY=tskey-auth-...
EOF
chmod 600 ~/actions-runner/.env
```

Restart the runner after editing (`./svc.sh stop && ./svc.sh start`) — `.env`
is read at service start.

**Updating after a pod rebuild is a one-file edit:** `POD_SSH_HOST`,
`POD_SSH_PORT`, and a fresh `TS_AUTHKEY`. Nothing in the repo changes.

## 3. Tailnet ACL (one-time, in the Tailscale admin console)

```json
"tagOwners": {
  "tag:devai": ["autogroup:admin"]
},
"ssh": [
  {
    "action": "accept",
    "src":    ["autogroup:member"],
    "dst":    ["tag:devai"],
    "users":  ["root"]
  }
]
```

Use **`accept`**, not `check`. `check` requires interactive browser re-auth,
which in CI does not fail — it **hangs** until the job times out.

`autogroup:member` covers the runner host because it is your own device. If
you later run the runner on a tagged machine, add that tag to `src`.

Enable Tailscale SSH on the pod (once per pod):

```bash
tailscale set --ssh=true
```

Verified 2026-09-02: with `--ssh` on but no ACL rule, an inbound attempt
reaches the pod and tailscaled replies *"tailnet policy does not permit you to
SSH to this node"*. That message means the transport works and only policy is
missing — it is the expected state before step 3, not a failure.

## 4. Turn on GitHub secret scanning

Settings → **Code security** → enable **Secret scanning** and **Push
protection**. Push protection blocks a credential at `git push` instead of
after it is already in history. Nothing to migrate — history is clean.

---

## Verify, in this order

```bash
# 1. runner is online:  Settings -> Actions -> Runners shows "Idle"
# 2. transport works, from the runner host:
tailscale ssh root@$POD_HOST 'hostname'
# 3. read-only workflow first:
#    Actions -> pod -> Run workflow -> action=status
# 4. then connect, then launch.
```

Do **not** test `action=provision` against a working pod: it does
`rm -rf mc-build` and rebuilds MineRL (40–60 min). It is guarded behind typing
`PROVISION`, and that guard exists for this reason.

---

## What is still stored where

| Value | Where | In git? | In GitHub? |
|---|---|---|---|
| `POD_HOST`, `MC_SERVER_TS_IP` | runner `.env` | no | no |
| `POD_SSH_HOST`, `POD_SSH_PORT` | runner `.env` | no | no |
| SSH private key | `~/.ssh/` on the runner host | no | no |
| `TS_AUTHKEY` | runner `.env` | no | no |
| `CI_RUNS_ON` (optional) | repo variable | no | yes — non-secret |

The only thing this design puts in GitHub is an optional non-secret variable
choosing which runner executes the test gate.
