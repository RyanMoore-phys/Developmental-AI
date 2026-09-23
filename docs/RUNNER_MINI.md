# CI runner on the headless Linux Mac mini

Moves the self-hosted runner off the dev Mac and onto a mini running Linux,
**containerised with resource caps** so the box keeps room for Grafana and
storage, starting **automatically on boot**, on a machine that **powers itself
back on after an outage**.

No application code changes. `deploy.yml`, `host.yml` and `pipeline.yml` already
say `runs-on: self-hosted`, which matches any self-hosted runner on any OS.
`ci.yml` runs the test suite on GitHub-hosted `ubuntu-latest` and never touches
this machine.

---

## What this runner does, and does not

| | Where it runs |
|---|---|
| `ci.yml` — the 12-suite test gate | GitHub-hosted `ubuntu-latest` |
| `deploy.yml`, `host.yml`, `pipeline.yml` | **this runner** |

So the container needs `openssh-client` and `rsync` (it ssh/rsyncs to the
training pod) and **not** a Python/ML stack. It is close to idle in steady
state; the CPU/memory caps exist to bound a runaway, not to size a workload.

---

## 1. Docker on the mini

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker        # `enable` is the auto-start-on-boot half
sudo usermod -aG docker "$USER"           # log out/in for this to take effect
```

`docker-compose-v2` matters: the `deploy.resources.limits` block is honoured by
Compose **v2**. Under the old v1 client it is silently ignored — if you end up
on v1, use the commented `cpus:` / `mem_limit:` keys in the compose file
instead, and confirm with `docker stats` rather than assuming.

## 2. Put the pod SSH key on the mini

The container mounts it read-only; it is never baked into the image and never
goes near GitHub.

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
# copy skybot_ed25519 across from the dev Mac, then:
chmod 600 ~/.ssh/skybot_ed25519
ssh -i ~/.ssh/skybot_ed25519 -p <MAIN_SSH_PORT> root@<MAIN_HOST> hostname
```

Do that last check **before** building anything — it isolates "key/network is
wrong" from "container is wrong", which otherwise present identically.

## 3. Configure and start

```bash
git clone https://github.com/<your-org>/Developmental-AI.git
cd Developmental-AI/docker/runner
# THE NAME MUST BE `.env` — compose interpolates ${MAIN_SSH_KEY_HOST_PATH}
# at parse time and only reads the shell or a file called exactly `.env`.
cp runner.env.example .env && chmod 600 .env
# edit .env: RUNNER_TOKEN, MAIN_SSH_KEY_HOST_PATH, MAIN_HOST, MAIN_SSH_PORT
docker compose up -d --build
docker compose logs -f          # expect "Listening for Jobs"
```

The registration token comes from **Settings → Actions → Runners → New
self-hosted runner** — copy only the string after `--token`. It expires in
about an hour and is single-use. You need it at first start and again only if
you wipe the volumes.

## 4. Verify before trusting it

```bash
# 1. GitHub -> Settings -> Actions -> Runners shows it Idle
# 2. the mounted key and the network path both work, from INSIDE the container:
docker compose exec runner bash -c 'ssh -i "$MAIN_SSH_KEYFILE" -p "$MAIN_SSH_PORT" root@"$MAIN_HOST" hostname'
# 3. resource caps are actually applied:
docker stats --no-stream skybot-runner
# 4. Actions -> pod -> Run workflow -> action=status   (read-only)
# 5. reboot the mini; the runner should return to Idle with no login
sudo reboot
```

Step 5 is the one people skip and then discover during an outage.

## 5. Auto power-on after a power cut

**This is firmware, not Linux.** It lives in the Mac's SMC/NVRAM and cannot be
set from Linux userspace.

On an Intel Mac mini, boot macOS (or Recovery → Utilities → Terminal) once and:

```bash
sudo pmset -a autorestart 1
pmset -g | grep autorestart          # expect: autorestart 1
```

It persists in firmware and applies regardless of which OS boots afterwards.

**Unverified caveat:** I have not confirmed a way to set this from Linux on
this hardware — do not assume a Linux `nvram` write does it. If the mini has
no macOS partition or recovery left, the dependable fallback is a smart plug
configured to restore power state.

With that set plus `restart: unless-stopped`, an outage recovers unattended:
power → mini boots → docker starts → runner registers → jobs flow.

## 6. Retire the dev-Mac runner (do this LAST)

Bring the mini up and confirm it **Idle** first, or there is a window with no
runner and jobs queue silently.

```bash
cd "/Users/rimac/Desktop/Developmental AI/actions-runner"
./svc.sh stop && ./svc.sh uninstall && ./config.sh remove
```

De-register rather than deleting the folder: a deleted-but-registered runner
lingers in GitHub's list, and jobs queued onto a runner that no longer exists
sit "Queued" forever with **no error** — the most confusing self-hosted failure
mode there is.

## 7. Day-to-day

| Task | Command |
|---|---|
| Logs | `docker compose logs -f` |
| Restart | `docker compose restart` |
| Update `MAIN_SSH_PORT` after a pod restart | edit `.env`, then `docker compose up -d` |
| Upgrade the runner | bump `RUNNER_VERSION` in the Dockerfile, `docker compose up -d --build` |
| Full reset | `docker compose down -v` (wipes registration — needs a fresh token) |

`docker compose restart` does **not** re-read `.env`; `up -d` recreates
the container and does. Editing the file and restarting looks like the edit
did nothing.
