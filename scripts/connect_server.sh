#!/bin/bash
# Bring up the pod->external-Minecraft-server tunnel, in two phases (the
# tailscale device-approval step is interactive and cannot be scripted).
#
#   Phase 1 (login):   bash scripts/connect_server.sh login
#     -> starts tailscaled in USERSPACE mode and prints an approval URL.
#        Open it in a browser signed into YOUR tailnet, approve "devai-pod",
#        then (recommended) tag it tag:devai in the admin console so the ACL
#        cages it to only the game port. See docs/SERVER_CONNECTION.md.
#
#   Phase 1, NON-INTERACTIVE (CI):
#        bash scripts/connect_server.sh login --authkey tskey-auth-...
#        TS_AUTHKEY=tskey-auth-... bash scripts/connect_server.sh login
#     -> same thing with no browser step, and enables Tailscale SSH so CI can
#        get back IN. Use an ephemeral, pre-authorised, tag:devai auth key.
#
#   Phase 2 (bridge):  bash scripts/connect_server.sh bridge <server-tailscale-ip>
#     -> starts a socat bridge so the java client's 127.0.0.1:25565 is
#        forwarded over the tailnet to <server-tailscale-ip>:25565, then
#        Minecraft-pings it to confirm the server answers.
#
# After both phases succeed, launch with scripts/launch_skybot.sh (its
# pre-flight refuses to start unless this bridge + tailnet are up).
set -e
cd /workspace/devai
mkdir -p podlogs /workspace/tailscale-state

case "${1:-}" in
  login)
    # AUTHKEY MODE (added 2026-09-02 for CI): `login --authkey <key>`, or set
    # TS_AUTHKEY in the environment. This is the ONLY way GitHub Actions can
    # bring a pod onto the tailnet — the interactive branch below prints a URL
    # a human must click, which in CI hangs the job until it times out.
    #
    # --ssh IS LOAD-BEARING, not a nicety. This pod has no /dev/net/tun, so
    # tailscaled runs --tun=userspace-networking, and in that mode INBOUND raw
    # TCP to the tailnet IP is not routable — sshd listens on 0.0.0.0:22 but
    # nothing on the tailnet can reach it. Tailscale SSH is terminated inside
    # tailscaled itself, so it is the one inbound path that works here. Drop
    # --ssh and CI loses its only way in.
    #
    # The matching tailnet ACL must grant tag:ci -> tag:devai as root with
    # action "accept". Action "check" demands interactive browser re-auth and
    # will hang a CI job rather than fail it.
    AUTHKEY=""
    if [ "${2:-}" = "--authkey" ]; then AUTHKEY="${3:?--authkey needs a value}"
    elif [ -n "${TS_AUTHKEY:-}" ]; then AUTHKEY="$TS_AUTHKEY"; fi

    pgrep -x tailscaled >/dev/null || {
      nohup tailscaled --tun=userspace-networking \
        --statedir=/workspace/tailscale-state \
        > podlogs/tailscaled.log 2>&1 < /dev/null &
      sleep 3
    }
    pgrep -x tailscaled >/dev/null || { echo "tailscaled FAILED"; tail -5 podlogs/tailscaled.log; exit 1; }

    if [ -n "$AUTHKEY" ]; then
      echo "tailscaled up (userspace). Non-interactive login (authkey)..."
      # --reset so a re-run cannot inherit stale flags from a previous `up`.
      tailscale up --authkey="$AUTHKEY" --hostname=devai-pod --ssh \
        --advertise-tags=tag:devai --reset \
        || { echo "tailscale up FAILED (authkey expired/not tag-authorised?)"; exit 1; }
      tailscale status >/dev/null 2>&1 \
        || { echo "logged in but status unavailable"; exit 1; }
      echo "ON TAILNET: $(tailscale ip -4 2>/dev/null | head -1) ($(tailscale status --json 2>/dev/null | grep -o '\"DNSName\":\"[^\"]*\"' | head -1))"
      echo "next: bash scripts/connect_server.sh bridge <server-tailscale-ip>"
      exit 0
    fi

    echo "tailscaled up (userspace). Requesting login..."
    tailscale up --hostname=devai-pod 2>&1 | tee /tmp/ts_up.log &
    sleep 8
    echo ""
    echo ">>> APPROVE THIS DEVICE (open in a browser on your tailnet):"
    grep -oE "https://login.tailscale.com/[a-zA-Z0-9/]+" /tmp/ts_up.log | head -1 \
      || echo "    (no URL yet — run: tailscale up --hostname=devai-pod)"
    echo ">>> Then tag it tag:devai in the admin console (ACL cage), then run:"
    echo ">>>   bash scripts/connect_server.sh bridge <server-tailscale-ip>"
    ;;
  bridge)
    IP="${2:?usage: connect_server.sh bridge <server-tailscale-ip>}"
    tailscale status >/dev/null 2>&1 || { echo "not logged in — run: connect_server.sh login"; exit 1; }
    pkill -x socat 2>/dev/null || true; sleep 1
    setsid socat "TCP-LISTEN:25565,bind=127.0.0.1,fork,reuseaddr" \
      "EXEC:tailscale nc $IP 25565" > podlogs/socat_mc.log 2>&1 < /dev/null &
    sleep 2
    ss -tln 2>/dev/null | grep -q "127.0.0.1:25565" || { echo "bridge FAILED"; cat podlogs/socat_mc.log; exit 1; }
    echo "bridge up: 127.0.0.1:25565 -> $IP:25565"
    if timeout 12 python3 scripts/mc_ping.py 127.0.0.1 25565 754 2>/dev/null | grep -E "version|players"; then
      echo "SERVER REACHABLE — ready. Launch: bash scripts/launch_skybot.sh 1000000"
    else
      echo "WARN: bridge up but server did not answer the MC ping."
      echo "  Check: server online, ViaVersion+ViaBackwards loaded (accepts protocol 754),"
      echo "  its tailscale IP correct, and the tag:devai ACL allows $IP:25565."
    fi
    ;;
  *)
    echo "usage: connect_server.sh login [--authkey <key>] | bridge <server-tailscale-ip>"; exit 1;;
esac
