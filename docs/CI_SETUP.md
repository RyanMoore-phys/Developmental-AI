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
| SSH to the training host | private key uploaded to GitHub Secrets | `~/.ssh/skybot_ed25519`, already on disk |
| Reach the training host | tailnet OAuth client | the machine can already reach it |
| Which host / server | repo variables | runner `.env`, a local file |

`deploy.yml` and `host.yml` reference **zero** `secrets.*` — verified.

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

**This file is the ONLY place the training host is named.** Nothing in the
tree hardcodes it, which is what made moving off the rented host an `.env` edit
rather than a refactor. To move hosts again, edit this — not the repo.

```bash
cat > ~/actions-runner/.env <<'EOF'
# --- required ---
MAIN_HOST=192.168.1.10           # the main computer, on the LAN
MAIN_SSH_PORT=22                  # a fixed box: this no longer rotates
MAIN_SSH_KEYFILE=/Users/rimac/.ssh/id_ed25519

# --- required for `host.yml action=connect` ---
MC_SERVER_TS_IP=192.168.1.XX     # the Paper server. LAN IP or tailnet IP —
                                 # connect_server.sh picks the transport from
                                 # the address itself (see below).

# --- optional ---
# DEPLOY_TRANSPORT=tailscale     # use Tailscale SSH instead of ssh; needs the
                                 # ACL rule in section 4, and MAIN_HOST becomes
                                 # the MagicDNS name (e.g. devai-training host-2)
# TS_AUTHKEY=tskey-auth-...      # lets `provision` join a fresh box to the
                                 # tailnet without a browser click
EOF
chmod 600 ~/actions-runner/.env
```

### `MC_SERVER_TS_IP` is no longer tailnet-only (2026-09-22)

The name is kept for compatibility with both workflows; the **value** may now
be either. `connect_server.sh bridge` chooses by address family:

| `MC_SERVER_TS_IP` | far side of the bridge |
|---|---|
| `192.168.*`, `10.*`, `172.16–31.*` | plain `socat` TCP over the LAN |
| `100.x` tailnet, MagicDNS names, anything else | `tailscale nc` — **unchanged** |

Unrecognised input defaults to **tailscale**, deliberately: every value that
worked before still works, and a MagicDNS name is not pattern-matchable as a
tailnet address. `CONNECT_MODE=lan|tailscale` forces it. Either way the bridge
presents `127.0.0.1:25565`, so `environment.remote_server` never changes.

**Switching back to the tailnet when ethernet returns** is one value: set
`MC_SERVER_TS_IP` to the Paper server's `100.x` address and re-run
`host.yml action=connect`. Nothing else moves.

Restart the runner after editing (`./svc.sh stop && ./svc.sh start`) — `.env`
is read at service start.

### The one thing you must keep updated

**On a rented training host, `MAIN_SSH_PORT` changes every restart** (22655 → 22681 →
34276 → 19983 → …). When a job fails at the preflight step with an ssh error,
that is almost always why: update `.env`, restart the runner, re-run.

**On the main computer this no longer applies** — port 22 is fixed. The
equivalent failure there is the box being asleep or off the LAN.

This is the single reason to consider the tailscale transport later: a MagicDNS
name is stable across rebuilds, so nothing needs updating. It costs one ACL
rule (section 4). The ssh default was chosen because it works **today**, with
zero extra configuration.

## 3. Verify, in this order

```bash
# 1. runner shows "Idle" under Settings -> Actions -> Runners
# 2. transport works, straight from the runner host:
MAIN_HOST=... MAIN_SSH_PORT=... bash scripts/host_exec.sh 'hostname'
# 3. read-only workflow:  Actions -> host -> Run workflow -> action=status
# 4. then deploy, then connect, then launch.
```

### Bootstrapping a BARE host — provision before you push

**`provision` is self-seeding; `deploy` is not.** `host.yml action=provision`
rsyncs the tree itself before running `provision_host.sh`, so it works against
an empty box. `deploy_skybot.sh` runs its smoke tests with
`./venv_mc/bin/python`, which **provisioning is what creates** — so a deploy
against a bare host fails at the smoke step, having already synced.

`deploy.yml` also fires automatically whenever `ci` passes. So on a new host,
pushing before `.env` is updated aims a deploy at the *old* target.

Order for a fresh box:

1. Root SSH reachable (`PermitRootLogin prohibit-password`, key in
   `/root/.ssh/authorized_keys`). `provision` creates `/workspace/devai`.
2. `.env` above → **restart the runner**.
3. `host.yml action=status` — read-only; proves the transport.
4. `host.yml action=provision confirm_provision=PROVISION` (40–60 min).
   Picks the torch wheel from the card's compute capability — **cu128 on
   Blackwell (RTX 50xx)**, cu124 otherwise — and hard-fails if CUDA is
   visible-but-unusable rather than falling silently back to CPU.
5. `host.yml action=connect` — expect `bridge mode: lan` then
   `SERVER REACHABLE`.
6. Only now push, or run `deploy.yml` with `launch=false` to sync and
   smoke-test without starting training.

Do **not** test `action=provision` against a working training host: it does
`rm -rf mc-build` and rebuilds MineRL (40–60 min). It is guarded behind typing
`PROVISION`, which is a guard on your *training host*, not a branch policy.

## 4. Optional — Tailscale SSH transport

Only if you want to stop updating `MAIN_SSH_PORT`. On the training host, once:

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
reaches the training host and tailscaled replies *"tailnet policy does not permit you to
SSH to this node"*. That message means the transport works and only policy is
missing — the expected state before adding the rule.

---

## Keeping CI honest

`ci.yml` pins versions **read off the running training host**, not guessed:

```
python 3.10 · torch 2.6.0 · numpy 2.2.6 · gymnasium 1.3.0
```

This matters: the first version installed unpinned latest and 6 of 12 suites
failed while all 12 passed locally — one stack mismatch, not six bugs. The
second attempt pinned `numpy<2` on the assumption numpy 2 was the culprit;
the training host actually runs numpy **2.2.6**, and the failing suites were then re-run
*on the training host* against it and all passed. Guessing was wrong twice; reading the
versions off production was right.

**When you upgrade the training host, update these pins in the same change.** Re-read
them with:

```bash
bash scripts/host_exec.sh 'cd /workspace/devai && ./venv_mc/bin/pip list'
```

A CI that tests a stack the training host does not run is worse than no CI — it reports
green for a configuration nobody deploys.

---

## What is stored where

| Value | Where | In git? | In GitHub? |
|---|---|---|---|
| `MAIN_HOST`, `MAIN_SSH_PORT` | runner `.env` | no | no |
| SSH private key | `~/.ssh/` on the runner host | no | no |
| `MC_SERVER_TS_IP` | runner `.env` | no | no |
| `TS_AUTHKEY` (optional) | runner `.env` | no | no |
| `CI_RUNS_ON` (optional) | repo variable | no | yes — non-secret |
