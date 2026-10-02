#!/bin/bash
# Provision a fresh a rented GPU host for Developmental-AI Minecraft training.
# Idempotent-ish; logs stage markers to stdout (redirect to provision.log).
# Takes ~40-60 min (MineRL pip install runs a full gradle build, then we
# patch MouseHelper and rebuild).
#
# Usage (from /workspace/devai, after rsyncing the repo there):
#   nohup bash scripts/provision_host.sh > runlogs/provision.log 2>&1 &
set -x
# ---- ROOT vs SUDO (added 2026-09-22) --------------------------------------
# On a rented GPU host this always ran as root, so apt/dpkg/installers were bare. The
# owned box `main` logs in as a normal account, so every privileged command
# now goes through $SUDO -- EMPTY when we are already root, so the old path is
# byte-for-byte unchanged and this cannot regress a root host.
# A non-root MAIN_USER therefore REQUIRES PASSWORDLESS SUDO on the target:
# this script runs detached under nohup with no tty, so a password prompt does
# not block visibly, it just fails the stage and keeps going with -x noise.
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else
  SUDO="sudo"
  sudo -n true 2>/dev/null || {
    echo "FATAL: $(whoami) has no PASSWORDLESS sudo on $(hostname)."
    echo "  Provisioning installs apt packages and writes under /. Fix with:"
    echo "    echo \"$(whoami) ALL=(ALL) NOPASSWD:ALL\" | sudo tee /etc/sudoers.d/90-$(whoami)"
    exit 1; }
fi
cd /workspace/devai || exit 1
mkdir -p runlogs

echo "=== STAGE 1: apt packages ==="
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -q
# PYTHON 3.10 ON UBUNTU >=24.04 (2026-08-11, Vast RTX 5060 Ti box).
# 24.04 (noble) ships python3.12 and carries NO python3.10 packages, but the
# whole MineRL chain is pinned to 3.10 (legacy gym + setuptools==65.5.1; 3.12
# also removed stdlib distutils, which those builds still expect). Rather than
# re-qualify the entire pinned stack on 3.12, pull 3.10 from deadsnakes —
# verified to publish python3.10 for noble. 22.04 keeps its native path
# untouched, so the proven a rented GPU host recipe is unchanged.
. /etc/os-release
if ! apt-cache policy python3.10-venv 2>/dev/null | grep -q "Candidate: [0-9]"; then
  echo "  python3.10 absent on ${PRETTY_NAME:-this release} -> adding deadsnakes"
  $SUDO apt-get install -y -q software-properties-common
  $SUDO add-apt-repository -y ppa:deadsnakes/ppa \
    || { echo "PROVISION-FAILED: deadsnakes-ppa"; exit 1; }
  $SUDO apt-get update -q
fi
# socat MOVED HERE from STAGE 1c (2026-09-23). It was `apt-get install socat`
# with NO sudo and a `|| echo WARN` swallow, so on a non-root host it failed
# silently, provisioning still said COMPLETE, and `action: connect` died with
# `setsid: failed to execute socat`. It is NOT optional: remote_server_scope
# `all` routes BOTH streams through the socat bridge.
$SUDO apt-get install -y -q socat openjdk-8-jdk-headless xvfb openbox xdotool psmisc \
    python3.10-venv python3.10-dev
java -version || { echo "PROVISION-FAILED: java"; exit 1; }
python3.10 --version || { echo "PROVISION-FAILED: python3.10"; exit 1; }

echo "=== STAGE 1b: VirtualGL (GPU headless GL via EGL) ==="
# Root cause (2026-07-23): under Xvfb SOFTWARE GL (llvmpipe) the MineRL client
# hangs forever at "Reloading ResourceManager" (GPU idle, render thread starves
# the game loop). VirtualGL's EGL back-end (`vglrun -d egl`) routes GL to the
# NVIDIA GPU with no X server. Proven on the A4000 GPU host: glxinfo renderer flips
# from "llvmpipe" to "NVIDIA RTX A4000". The client boots to DORMANT and runs
# missions. Xvfb (:77) is still required as the dummy 2D/window server for the
# blit target — VGL renders on the GPU and the app reads pixels off-screen.
if ! which vglrun >/dev/null 2>&1; then
  VGL_DEB=/tmp/virtualgl_3.1.1_amd64.deb
  # SOURCEFORGE WENT 404 (measured 2026-09-22) and took provisioning with it:
  #   wget -qO ... sourceforge.net/projects/virtualgl/files/3.1.1/... -> 404
  # GitHub releases serves the same artifact and answers 200, so it is now the
  # PRIMARY and SourceForge the fallback -- the reverse of before. The cached
  # copy is kept first for reproducibility, but note it is almost never present:
  # deploy_skybot.sh rsyncs only developmental_ai/configs/scripts/tests, so
  # host_repository/ does NOT reach the training host. Treat it as a local-run
  # convenience, not a mirror you can rely on.
  # -L IS LOAD-BEARING on both: each redirects to a CDN, and without it you get
  # a 0-byte file and a dpkg error instead of a download error.
  VGL_URLS="https://github.com/VirtualGL/virtualgl/releases/download/3.1.1/virtualgl_3.1.1_amd64.deb
https://sourceforge.net/projects/virtualgl/files/3.1.1/virtualgl_3.1.1_amd64.deb/download"
  if [ -f host_repository/data/minerl_build/virtualgl_3.1.1_amd64.deb ]; then
    cp host_repository/data/minerl_build/virtualgl_3.1.1_amd64.deb "$VGL_DEB"
  else
    _got=0
    for _u in $VGL_URLS; do
      echo "  trying $_u"
      if curl -fsSL --max-time 180 -o "$VGL_DEB" "$_u" && [ -s "$VGL_DEB" ]; then
        # A CDN error page is a 200 with HTML in it. Verify it is really a .deb
        # (ar archive, magic "!<arch>") or we fail later inside dpkg instead.
        if head -c 7 "$VGL_DEB" | grep -q "!<arch>"; then _got=1; break; fi
        echo "  -> not a .deb ($(head -c 60 "$VGL_DEB" | tr -d '\0' | tr '\n' ' '))"
      fi
    done
    [ "$_got" = "1" ] || { echo "PROVISION-FAILED: virtualgl download (all mirrors)"; exit 1; }
  fi
  $SUDO apt-get install -y -q "$VGL_DEB" || $SUDO dpkg -i "$VGL_DEB" \
    || { echo "PROVISION-FAILED: virtualgl install"; exit 1; }
fi
which vglrun || { echo "PROVISION-FAILED: virtualgl missing"; exit 1; }

echo "=== STAGE 1c: Tailscale + socat (OPTIONAL — external-server play) ==="
# Only needed to point env 0 at an EXTERNAL Minecraft server over a private
# tailnet (configs/minecraft_skybot.yaml). Harmless for pure-MineRL runs.
# tailscale runs USERSPACE (a rented GPU host containers expose no /dev/net/tun), so a
# socat bridge makes the tunnel transparent to the java client. The `tailscale
# up` LOGIN is interactive (device approval) and CANNOT be scripted — it is a
# manual step, see scripts/connect_server.sh + docs/SERVER_CONNECTION.md.
if ! which tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | $SUDO sh || \
    echo "WARN: tailscale install failed (external-server play unavailable)"
fi
which socat >/dev/null 2>&1 || $SUDO apt-get install -y -q socat || true
# PROVISION-FAILED, NOT "WARN". The old warning was invisible: host.yml polls
# only `^=== STAGE|PROVISION-`, so a swallowed socat failure reported COMPLETE
# and surfaced an hour later as a broken bridge. Fail where it is cheap to fix.
which socat >/dev/null 2>&1 || { echo "PROVISION-FAILED: socat absent (needed by the server bridge)"; exit 1; }
echo "tailscale: $(which tailscale 2>/dev/null || echo ABSENT) | socat: $(which socat 2>/dev/null || echo ABSENT)"
# NON-INTERACTIVE TAILNET JOIN (2026-09-02). With TS_AUTHKEY exported, join
# here so a CI provision produces a training host that is already reachable — otherwise
# provisioning finishes and then blocks forever on a human clicking a URL,
# which is exactly what made this stage un-automatable before.
# Deliberately NOT fatal: a pure-MineRL run (no external server) does not need
# the tailnet, and provisioning should not die for an optional capability.
if [ -n "${TS_AUTHKEY:-}" ]; then
  echo "  TS_AUTHKEY present -> non-interactive tailnet join (+Tailscale SSH)"
  bash scripts/connect_server.sh login --authkey "$TS_AUTHKEY" \
    || echo "WARN: non-interactive tailnet join failed (key expired/untagged?)"
else
  echo "  no TS_AUTHKEY -> tailnet join is a MANUAL step:"
  echo "     bash scripts/connect_server.sh login"
fi

echo "=== STAGE 2: venv_mc (python3.10) + torch ==="
if [ ! -d venv_mc ]; then python3.10 -m venv venv_mc; fi
./venv_mc/bin/pip install -q --upgrade pip wheel setuptools
# TORCH CUDA WHEEL INDEX — CHOSEN FROM THE GPU, NOT HARDCODED (2026-08-11).
# Default PyPI torch bundles CUDA 13 (cu130), which a 550.x driver cannot
# initialise -> torch.cuda.is_available()==False (silent CPU-only training).
# That is why this was pinned to cu124. But cu124 wheels carry no kernels
# NEWER than sm_90, so on Blackwell (RTX 50xx, sm_120) every kernel launch
# dies with "no kernel image is available for execution on the device" —
# and is_available() still returns True, so the old check waved it through.
# Pick the index from the card's actual compute capability.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
     | head -1 | tr -d ' .')
if [ "${CC:-0}" -ge 120 ] 2>/dev/null; then
  TORCH_IDX="${TORCH_CUDA_INDEX:-cu128}"   # Blackwell REQUIRES cu128+
else
  TORCH_IDX="${TORCH_CUDA_INDEX:-cu124}"   # the proven Ampere/Ada path
fi
echo "  compute_cap=${CC:-unknown} -> torch wheel index ${TORCH_IDX}"
./venv_mc/bin/pip install -q --force-reinstall torch \
    --index-url "https://download.pytorch.org/whl/${TORCH_IDX}"
# matplotlib + pillow are the LIVE VIEWER's dependencies. Omitting them
# did not fail the run — the viewer caught its own ImportError and logged
# "viewer disabled (init failed): No module named 'matplotlib'" once, at
# INFO, on every boot. The run looked entirely healthy and the operator had
# no window into it for days. A dependency of an OBSERVABILITY tool is
# exactly the kind that goes unnoticed, because nothing downstream breaks.
./venv_mc/bin/pip install -q numpy imageio imageio-ffmpeg pyyaml \
    gymnasium ollama psutil matplotlib pillow
# HARD-FAIL if CUDA isn't merely VISIBLE but actually USABLE. is_available()
# alone passed on a cu124/sm_120 mismatch that then failed on first matmul, so
# the check now launches a real kernel and reads the result back.
./venv_mc/bin/python -c "
import torch
assert torch.cuda.is_available(), 'cuda not available (driver/toolkit mismatch)'
d = torch.device('cuda')
x = torch.randn(64, 64, device=d) @ torch.randn(64, 64, device=d)
torch.cuda.synchronize()
assert torch.isfinite(x).all(), 'kernel produced garbage'
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      '|', torch.cuda.get_device_name(0),
      'sm_%d%d' % torch.cuda.get_device_capability(0), '| kernel OK')
" || { echo "PROVISION-FAILED: torch-cuda"; exit 1; }

echo "=== STAGE 3a: gradle forge-maven mirror (dead-URL fix) ==="
# files.minecraftforge.net only 308-redirects now; MineRL 1.0's gradle uses
# it as a hard repo and fails 'downloadAssets'. A GLOBAL init.gradle rewrites
# it to the live maven.minecraftforge.net for EVERY gradle invocation (incl.
# pip's build-isolated one only if we build non-isolated — see 3b).
mkdir -p ~/.gradle
cat > ~/.gradle/init.gradle <<'GEOF'
allprojects {
    buildscript { repositories {
        all { ArtifactRepository repo -> if (repo instanceof MavenArtifactRepository &&
            repo.url.toString().contains("files.minecraftforge.net/maven")) { remove repo } }
        maven { url "https://maven.minecraftforge.net/" }; mavenCentral() } }
    repositories {
        all { ArtifactRepository repo -> if (repo instanceof MavenArtifactRepository &&
            repo.url.toString().contains("files.minecraftforge.net/maven")) { remove repo } }
        maven { url "https://maven.minecraftforge.net/" }; mavenCentral() }
}
GEOF

echo "=== STAGE 3b: MineRL v1.0.0 — MANUAL build (NON-isolated) ==="
# @v1.0.0 = MCP-Reborn/VPT line (@master is the old 0.4 tree; import crashes
# on modern gym). `pip install git+...` builds in ISOLATION and ignores our
# init.gradle -> downloadAssets fails on the dead forge URL. So: clone, run
# setup_mcp.sh to materialize MCP-Reborn, build with our gradle config, apply
# the DEVAI focus patches, shadowJar, THEN `pip install --no-build-isolation`.
MCP=mc-build/minerl/MCP-Reborn
BG="$MCP/build.gradle"
# IDEMPOTENT + de-mix BEFORE any gradle task. mixin/mixingradle crashes gradle
# CONFIGURATION (MixinExtension.groovy:1038 universalJar), which EVERY task
# (downloadAssets included) must pass — so the mixin MUST be gone before the
# first gradlew call (maintainer's own ead4986 "drop unused mixin" fix; keep
# shadow). The clone + ~10min MCP decompile + removal are SKIPPED if mc-build
# is already set up + de-mixed, so a re-run after a later-stage failure doesn't
# redo the slow decompile.
if [ -f "$BG" ] && grep -qE "DEVAI:.*(plugin|block) (dropped|removed)" "$BG"; then
    echo "mc-build already set up + de-mixed — skipping clone/decompile"
    chmod +x "$MCP/gradlew"
else
    rm -rf mc-build
    # NO git clone HERE — GitHub 401s git-upload-pack FROM TRAINING HOST IPs.
    # Measured 2026-09-02 on an A4000 GPU host, with clean git
    # config (no credential helper, no .netrc, no proxy, no insteadOf):
    #     GET  /minerllabs/minerl.git/info/refs      -> 200   (ls-remote works)
    #     POST /minerllabs/minerl.git/git-upload-pack -> 401
    #                        www-authenticate: Basic realm="GitHub"
    # So `git ls-remote` succeeds and `git clone` cannot, which produces the
    # thoroughly misleading pair:
    #     fatal: could not read Username for 'https://github.com'
    #     fatal: the remote end hung up unexpectedly
    # That reads as auth-on-a-private-repo. The repo is public; this is
    # GitHub's abuse throttle on a shared cloud IP. Nothing in our config
    # fixes it — the TRANSPORT has to change. Two dead ends already paid
    # for, do not retry them: forcing protocol.version=0 (fixes ls-remote,
    # NOT clone), and blaming the ref (`--branch v1.0.0` is correct; v1.0.0
    # is a BRANCH, refs/heads/v1.0.0 — there is no v1.0.0 tag, which is why
    # a .../tags/v1.0.0 tarball 404s).
    #
    # codeload.github.com serves plain tarballs over GET and is NOT throttled
    # here, so both clones become fetches. Safe because setup_mcp.sh itself
    # does `rm -rf .git` immediately after checkout — the git metadata is
    # disposable and a tarball at the right ref is a faithful substitute.
    MINERL_TGZ=/tmp/minerl_v1.0.0.tgz
    MCPR_TGZ=/tmp/mcp_reborn_1.16.5-20210115.tgz
    curl -fsSL --retry 3 --max-time 600 -o "$MINERL_TGZ" \
      "https://codeload.github.com/minerllabs/minerl/tar.gz/refs/heads/v1.0.0" \
      || { echo "PROVISION-FAILED: minerl tarball fetch"; exit 1; }
    mkdir -p /tmp/_mlx && rm -rf /tmp/_mlx/* && tar xzf "$MINERL_TGZ" -C /tmp/_mlx
    mv /tmp/_mlx/minerl-1.0.0 mc-build \
      || { echo "PROVISION-FAILED: minerl tarball layout"; exit 1; }
    # MCP-Reborn is the SECOND clone, inside minerl's own setup_mcp.sh.
    # NOTE the ref kind differs from minerl's: here 1.16.5-20210115 is a TAG.
    curl -fsSL --retry 3 --max-time 600 -o "$MCPR_TGZ" \
      "https://codeload.github.com/Hexeption/MCP-Reborn/tar.gz/refs/tags/1.16.5-20210115" \
      || { echo "PROVISION-FAILED: MCP-Reborn tarball fetch"; exit 1; }
    mkdir -p /tmp/_mcx && rm -rf /tmp/_mcx/* && tar xzf "$MCPR_TGZ" -C /tmp/_mcx
    mkdir -p mc-build/minerl
    mv /tmp/_mcx/MCP-Reborn-1.16.5-20210115 mc-build/minerl/MCP-Reborn \
      || { echo "PROVISION-FAILED: MCP-Reborn tarball layout"; exit 1; }
    test -f mc-build/minerl/MCP-Reborn/gradlew \
      || { echo "PROVISION-FAILED: MCP-Reborn missing gradlew"; exit 1; }
    # Neutralise the three lines that would clone (and the rm that would
    # delete what we just staged). Everything after them — chmod, ./gradlew
    # setup, the cleanup rm — must still run, so this is a surgical edit and
    # NOT a rewrite of setup_mcp.sh.
    SM=mc-build/scripts/setup_mcp.sh
    sed -i -E 's|^[[:space:]]*rm -rf MCP-Reborn[[:space:]]*$|: # DEVAI: pre-staged from tarball|' "$SM"
    sed -i -E 's|^[[:space:]]*git clone .*$|: # DEVAI: pre-staged from tarball|'                    "$SM"
    sed -i -E 's|^[[:space:]]*git checkout .*$|: # DEVAI: tarball already at the ref|'              "$SM"
    grep -qE "^[[:space:]]*git (clone|checkout)" mc-build/scripts/setup_mcp.sh \
      && { echo "PROVISION-FAILED: setup_mcp.sh still contains a git clone/checkout"; exit 1; }
    (cd mc-build && bash scripts/setup_mcp.sh && bash scripts/patch_mcp.sh) \
        || { echo "PROVISION-FAILED: setup_mcp"; exit 1; }
    chmod +x "$MCP/gradlew"
    # Replacement comments deliberately AVOID the words mixin/mixingradle/refmap
    # so the verification below can't false-positive on our OWN comments.
    python3 - "$BG" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r"(?m)^.*mixingradle.*$", "        // DEVAI: sponge-gradle classpath dropped (ead4986)", s)
s = re.sub(r"(?m)^.*apply plugin:\s*'org\.spongepowered\.mixin'.*$", "// DEVAI: sponge plugin dropped", s)
s = re.sub(r"(?s)mixin\s*\{[^}]*refmap[^}]*\}", "// DEVAI: sponge block dropped", s)
open(p, "w").write(s)
PYEOF
    # fail loudly ONLY on ACTIVE (non-comment) mixin refs — strip // lines first
    # (both our comments and the file's pre-existing commented mixin ref).
    if grep -vE "^[[:space:]]*//" "$BG" | grep -qE "org\.spongepowered[.:]mixin|mixingradle|modid\.refmap"; then
        echo "PROVISION-FAILED: active mixin refs remain"; grep -nE "mixin|refmap" "$BG"; exit 1
    fi
fi

(cd "$MCP" && ./gradlew downloadAssets --stacktrace) \
    || { echo "PROVISION-FAILED: downloadAssets"; exit 1; }

echo "=== STAGE 4: DEVAI headless focus patches + jar build ==="
# Root cause (2026-07-16): MouseHelper gates camera-velocity accumulation and
# grabMouse() on isGameFocused(); headless windows never focus, so synthetic
# camera + attack are dropped. Same patch the devs applied to updatePlayerLook.
MH="$MCP/src/main/java/net/minecraft/client/MouseHelper.java"
cp "$MH" "$MH.orig"
sed -i "s|if (this.minecraft.isGameFocused()) {|if (true \|\| this.minecraft.isGameFocused()) { // DEVAI: headless focus patch|" "$MH"
sed -i "s|if (this.isMouseGrabbed() \&\& this.minecraft.isGameFocused()) {|if (true \|\| (this.isMouseGrabbed() \&\& this.minecraft.isGameFocused())) { // DEVAI: headless focus patch|" "$MH"
# >=2 not ==2: the gate count varies by MCP-Reborn revision (old host had 2
# gates, the v1.0.0 clone has 3). All matched gates are legitimately disabled.
[ "$(grep -c DEVAI "$MH")" -ge 2 ] || { echo "PROVISION-FAILED: patch count"; exit 1; }
# unpack_assets: setup.py copies downloadAssets' hashed objects into
# src/main/resources so shadowJar BUNDLES the game assets into the jar.
# Bypassing prep_mcp skips this -> the client crashes at runtime with
# FileNotFoundException icons/icon_16x16.png. Replicate it here.
python3 - "$MCP" <<'PYEOF'
import sys, os, json, shutil
mcp = sys.argv[1]
ad = os.path.join(os.path.expanduser('~'), '.gradle', 'caches', 'forge_gradle', 'assets')
out = os.path.join(mcp, 'src', 'main', 'resources')
idx = json.load(open(os.path.join(ad, 'indexes', '1.16.json')))
for k, v in idx['objects'].items():
    h = v['hash']
    dst = os.path.join(out, k); os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(ad, 'objects', h[:2], h), dst)
print("unpacked", len(idx['objects']), "assets")
PYEOF
(cd "$MCP" && ./gradlew clean build shadowJar -x test) || { echo "PROVISION-FAILED: gradle"; exit 1; }

echo "=== STAGE 4a2: pin old setuptools (MineRL needs legacy gym) ==="
# MineRL depends on old `gym`, whose setup.py uses a loose extras_require that
# setuptools>=66 rejects ("'extras_require' must be a dictionary ...").
./venv_mc/bin/pip install -q "setuptools==65.5.1" "wheel==0.38.4" "pip==23.3.2"

echo "=== STAGE 4b: install python-only + drop in the built tree ==="
# A plain `pip install .` re-runs setup.py's prep_mcp -> re-applies patch_mcp
# (re-adding the broken mixin) AND `gradlew clean` WIPES our jar, then fails.
# READTHEDOCS=1 makes setup.py skip prep_mcp entirely (python code only); we
# then copy our fully-built MCP-Reborn (jar + patches + options) into place.
(cd mc-build && READTHEDOCS=1 /workspace/devai/venv_mc/bin/pip install \
    --no-build-isolation .) || { echo "PROVISION-FAILED: pip-install"; exit 1; }
SP=$(./venv_mc/bin/python -c "import minerl, os; print(os.path.dirname(minerl.__file__))")
rm -rf "$SP/MCP-Reborn"
cp -r "$MCP" "$SP/MCP-Reborn"
ls "$SP/MCP-Reborn/build/libs/"*.jar || { echo "PROVISION-FAILED: no jar"; exit 1; }
./venv_mc/bin/python -c "import minerl, minerl.herobraine.envs; print('minerl OK')" \
    || { echo "PROVISION-FAILED: import"; exit 1; }
MCP="$SP/MCP-Reborn"

echo "=== STAGE 4b2: stable in-game username (external-server play) ==="
# MCP-Reborn defaults the player name to "Player" + (millis % 1000)
# (Main.java:64), and launchClient.sh never forwards --username. On a real
# server that means a DIFFERENT player every launch — and because the client
# is offline-mode, the server derives the UUID from the name
# (Main.java:123, PlayerEntity.getOfflineUUID), so a random name is also a
# random identity: no persistent inventory, no whitelist entry, no perms,
# and a fresh stranger after every auto-rejoin.
# The flag already exists; it just was not passed. MC_USERNAME overrides.
LC="$MCP/launchClient.sh"
if [ -f "$LC" ] && ! grep -q -- "--username" "$LC"; then
  cp -n "$LC" "$LC.orig"
  sed -i 's|java -Xmx\$maxMem -jar \$fatjar --envPort=\$port|java -Xmx$maxMem -jar $fatjar --envPort=$port --username ${MC_USERNAME:-SkyBot}|' "$LC"
  grep -q -- "--username" "$LC" && echo "  username patch OK -> ${MC_USERNAME:-SkyBot}" \
    || echo "  WARN: username patch did NOT apply (launch line changed upstream?)"
fi
# THE NAME THAT ACTUALLY REACHES THE SERVER is the mission XML's AgentSection
# <Name>, NOT the --username launch flag above. Verified live: with only the
# launchClient patch applied the bot still joined as "MineRLAgent0", because
# the client takes its identity from the mission handshake. The template
# hardcodes <Name>MineRLAgent{{ agent_index }}</Name>; single-agent specs
# always render index 0, and only the PRIMARY stream joins the external
# server (the scout is in a local world), so a bare name cannot collide.
# Identity matters beyond cosmetics: offline-mode servers derive the UUID
# from the name, so a changing name means no persistent inventory,
# whitelist entry or permissions across rejoins.
MJ=$(./venv_mc/bin/python -c "import minerl,os;print(os.path.join(os.path.dirname(minerl.__file__),'herobraine','hero','mission.xml.j2'))" 2>/dev/null)
if [ -f "$MJ" ] && grep -q "MineRLAgent" "$MJ"; then
  cp -n "$MJ" "$MJ.orig"
  # PER-CLIENT IDENTITY (2026-08-17). This used to hard-code a single name,
  # which is invisible with one remote client and fatal with several: an
  # offline server kicks the incumbent on a duplicate login, so N clients
  # evict each other forever. The spec sets `agent_username` per instance
  # (MineRLEnvAdapter), and `| default` keeps every other env spec — which
  # never sets it — rendering exactly as before.
  sed -i "s|<Name>MineRLAgent{{ agent_index }}</Name>|<Name>{{ agent_username \| default(\"${MC_USERNAME:-SkyBot}\") }}</Name>|" "$MJ"
  # idempotent re-patch when a previous provision baked in the fixed name
  sed -i "s|<Name>${MC_USERNAME:-SkyBot}</Name>|<Name>{{ agent_username \| default(\"${MC_USERNAME:-SkyBot}\") }}</Name>|" "$MJ"
  grep -q "${MC_USERNAME:-SkyBot}" "$MJ" && echo "  agent-name patch OK -> ${MC_USERNAME:-SkyBot}" \
    || echo "  WARN: agent-name patch did NOT apply (template changed upstream?)"
fi

echo "=== STAGE 4c: GPU-GL launch wrap (VGL EGL + per-core pin) ==="
# Two coupled fixes on the java client launch (proven on the A4000 / EPYC 7702):
#  1. `vglrun -d egl` renders on the NVIDIA GPU (software llvmpipe hangs at boot).
#  2. `taskset -c $((port % nproc))` pins each client to ONE distinct core.
# Why the pin is mandatory: with GPU GL active, the NVIDIA driver's CPU-dispatched
# memcpy (AVX path selected off the "AuthenticAMD" CPUID) stack-corrupts in a
# multithread race during GL resource upload -> glibc "futex facility returned an
# unexpected error code" + SIGABRT right after ResourceManager. Serializing each
# client to a single core dodges the race (confirmed: strace/single-core boots
# clean, full-parallel aborts in ~2s). Distinct core per port keeps the N envs
# parallel across cores. GLIBC_TUNABLES/__GL_THREADED_OPTIMIZATIONS are belt-and-
# suspenders (fewer driver threads); the taskset pin is the load-bearing fix.
LC="$MCP/launchClient.sh"
python3 - "$LC" <<'PYEOF'
import sys
p = sys.argv[1]
lines = open(p).read().splitlines()
new = ("GLIBC_TUNABLES=glibc.pthread.rseq=0 __GL_THREADED_OPTIMIZATIONS=0 "
       "taskset -c $((port % $(nproc))) vglrun -d egl "
       "java -Xmx$maxMem -jar $fatjar --envPort=$port")
hit = False
for i, l in enumerate(lines):
    if l.strip().startswith("java -Xmx") and "--envPort" in l:
        lines[i] = new; hit = True
open(p, "w").write("\n".join(lines) + "\n")
print("launchClient patched" if hit else "NO-MATCH")
PYEOF
grep -q "vglrun -d egl" "$LC" || { echo "PROVISION-FAILED: launchClient wrap"; exit 1; }

echo "=== STAGE 5: tutorial toast off ==="
printf "tutorialStep:none\npauseOnLostFocus:false\n" > "$MCP/run/options.txt"

echo "=== STAGE 6: ollama + the VLM THE CONFIG ASKS FOR ==="
# MODEL NAME READ FROM THE CONFIG, NOT HARDCODED (2026-08-11). This stage
# pulled `llava:7b` while configs/minecraft_skybot.yaml had moved to
# qwen2.5vl:7b — and llava:7b was THE BROKEN SENSOR (it answered
# tree_visible=true on every patch, which is what made the vision magnet
# useless for weeks; qwen2.5vl:7b scored 8/8 on the same probe battery).
# A provisioner that installs a different model than the run requests is a
# silent, expensive drift, so derive the name from the config itself.
which ollama || (curl -fsSL https://ollama.com/install.sh | $SUDO sh)
pgrep -x ollama >/dev/null || (OLLAMA_DEBUG=0 nohup ollama serve >> runlogs/ollama.log 2>&1 < /dev/null & sleep 5)
# append-mode above + capper below: 581 MB of VLM-server chatter in 5 days
# (measured 2026-08-16) would eat the disk on a long lifelong run
pgrep -f "cap_log[.]sh runlogs/ollama[.]log" >/dev/null || \
  (nohup bash scripts/cap_log.sh runlogs/ollama.log >> runlogs/cap_log.log 2>&1 < /dev/null &)
# NOTE the key is symbolic_grounding.model — NOT llm.model, which names the
# (disabled) text model llama3.1:8b. Pulling that one would waste 5 GB and
# still leave the grounding head with no teacher.
VLM_MODEL=$(./venv_mc/bin/python -c "
import yaml
c = yaml.safe_load(open('configs/minecraft_skybot.yaml'))
print((c.get('symbolic_grounding') or {}).get('model') or '')
" 2>/dev/null | tr -d '[:space:]')
[ -n "$VLM_MODEL" ] || VLM_MODEL="qwen2.5vl:7b"
echo "  config asks for VLM: $VLM_MODEL"
# A `-q<N>` SUFFIX IS LOCALLY CREATED, NOT PULLABLE (fixed 2026-09-02).
# The registry has qwen2.5vl:3b and :7b but NOT :3b-q4 — measured, manifest
# HTTP 200 / 200 / 404 respectively. `ollama pull qwen2.5vl:3b-q4` therefore
# dies with "pull model manifest: file does not exist", which reads like a
# network problem and is actually a naming one. The config comment beside
# `symbolic_grounding.model` already documents the real recipe (pull the base,
# then `ollama create ... -q q4_K_M`); this stage just never implemented it,
# so setting the config to a quantized tag broke provisioning at the last
# stage with every earlier stage green.
case "$VLM_MODEL" in
  *-q[0-9]*)
    VLM_BASE="${VLM_MODEL%-q*}"          # qwen2.5vl:3b-q4 -> qwen2.5vl:3b
    VLM_QUANT="q${VLM_MODEL##*-q}"       # ...            -> q4
    # ollama wants a full method name; q4 is shorthand for q4_K_M.
    case "$VLM_QUANT" in
      q4) VLM_QUANT=q4_K_M ;;
      q8) VLM_QUANT=q8_0 ;;
    esac
    echo "  '$VLM_MODEL' is a LOCAL quantization of '$VLM_BASE' ($VLM_QUANT)"
    if ollama list 2>/dev/null | grep -q "^${VLM_MODEL}[[:space:]]"; then
      echo "  already present, skipping"
    else
      ollama pull "$VLM_BASE" \
        || { echo "PROVISION-FAILED: vlm base pull ($VLM_BASE)"; exit 1; }
      printf 'FROM %s\n' "$VLM_BASE" > /tmp/Modelfile.vlm
      if ! ollama create "$VLM_MODEL" -f /tmp/Modelfile.vlm -q "$VLM_QUANT"; then
        # ALREADY-QUANTIZED BASES ARE THE COMMON CASE (measured 2026-09-02).
        # ollama refuses with "quantization is only supported for F16, BF16
        # and F32 models" because the registry tag is ALREADY Q4_K_M — e.g.
        # qwen2.5vl:3b reports Q4_K_M at 3.2 GB. There is nothing to convert,
        # so alias the requested name onto the base rather than failing the
        # whole provision at the last stage over a naming convention.
        echo "  '$VLM_BASE' cannot be re-quantized (already quantized?) —"
        echo "  aliasing $VLM_MODEL -> $VLM_BASE instead"
        ollama cp "$VLM_BASE" "$VLM_MODEL" \
          || { echo "PROVISION-FAILED: vlm alias ($VLM_MODEL from $VLM_BASE)"; exit 1; }
      fi
    fi
    ;;
  *)
    ollama pull "$VLM_MODEL" || { echo "PROVISION-FAILED: vlm pull ($VLM_MODEL)"; exit 1; }
    ;;
esac
# Match the EXACT tag, not just the family. The old check grepped
# "${VLM_MODEL%%:*}" (i.e. just "qwen2.5vl"), so any qwen2.5vl variant
# satisfied it — including the wrong one, which is the same silent-drift
# failure this stage's own header warns about.
ollama list 2>/dev/null | grep -q "^${VLM_MODEL}[[:space:]]" \
  || { echo "PROVISION-FAILED: vlm '$VLM_MODEL' missing after install"; \
       echo "  have:"; ollama list; exit 1; }
echo "  VLM ready: $VLM_MODEL"

echo "=== STAGE 7: focused headless display (Xvfb :77 + openbox) ==="
pgrep -f "Xvfb [:]77" >/dev/null || (nohup Xvfb :77 -screen 0 800x600x24 >/dev/null 2>&1 < /dev/null & sleep 2)
pgrep -x openbox >/dev/null || (DISPLAY=:77 nohup openbox >/dev/null 2>&1 < /dev/null & sleep 1)

echo "=== PROVISION-COMPLETE ==="

echo "=== STAGE 8: Prometheus exporters (host visibility) ==="
# LAST, and NON-FATAL. Training must never fail to provision because a metrics
# exporter could not download. But it belongs IN provisioning: this box had no
# CPU/RAM/disk/VRAM graphs at all, and everything that went wrong on it in the
# first 24 hours was found by hand over ssh as a result.
bash scripts/install_host_exporters.sh \
  || echo "WARN: exporters not installed — main will not appear in Grafana"
