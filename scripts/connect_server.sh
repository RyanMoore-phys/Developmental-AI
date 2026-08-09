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
    pgrep -x tailscaled >/dev/null || {
      nohup tailscaled --tun=userspace-networking \
        --statedir=/workspace/tailscale-state \
        > podlogs/tailscaled.log 2>&1 < /dev/null &
      sleep 3
    }
    pgrep -x tailscaled >/dev/null || { echo "tailscaled FAILED"; tail -5 podlogs/tailscaled.log; exit 1; }
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
    echo "usage: connect_server.sh login | bridge <server-tailscale-ip>"; exit 1;;
esac
