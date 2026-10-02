#!/bin/bash
# Bring up the training host->external-Minecraft-server tunnel, in two phases (the
# tailscale device-approval step is interactive and cannot be scripted).
#
#   Phase 1 (login):   bash scripts/connect_server.sh login
#     -> starts tailscaled in USERSPACE mode and prints an approval URL.
#        Open it in a browser signed into YOUR tailnet, approve "devai-training host",
#        then (recommended) tag it tag:devai in the admin console so the ACL
#        cages it to only the game port. See docs/SERVER_CONNECTION.md.
#
#   Phase 1, NON-INTERACTIVE (CI):
#        bash scripts/connect_server.sh login --authkey tskey-auth-...
#        TS_AUTHKEY=tskey-auth-... bash scripts/connect_server.sh login
#     -> same thing with no browser step, and enables Tailscale SSH so CI can
#        get back IN. Use an ephemeral, pre-authorised, tag:devai auth key.
#
#   Phase 2 (bridge):  bash scripts/connect_server.sh bridge <server-address>
#     -> starts a socat bridge so the java client's 127.0.0.1:25565 is
#        forwarded to <server-address>:25565, then Minecraft-pings it to
#        confirm the server answers.
#
#        TWO FAR SIDES, ONE ENDPOINT (2026-09-22). An RFC1918 address
#        (192.168./10./172.16-31.) is reached over PLAIN TCP on the LAN;
#        anything else — tailnet 100.x, MagicDNS names — goes over the tailnet
#        via `tailscale nc`, exactly as before. Phase 1 is still required for
#        the tailnet path and is untouched. CONNECT_MODE=lan|tailscale forces
#        the choice.
#
# After both phases succeed, launch with scripts/launch_skybot.sh (its
# pre-flight refuses to start unless this bridge + tailnet are up).
set -e
cd /workspace/devai
mkdir -p runlogs /workspace/tailscale-state

case "${1:-}" in
  login)
    # AUTHKEY MODE (added 2026-09-02 for CI): `login --authkey <key>`, or set
    # TS_AUTHKEY in the environment. This is the ONLY way GitHub Actions can
    # bring a training host onto the tailnet — the interactive branch below prints a URL
    # a human must click, which in CI hangs the job until it times out.
    #
    # --ssh IS LOAD-BEARING, not a nicety. This host has no /dev/net/tun, so
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
        > runlogs/tailscaled.log 2>&1 < /dev/null &
      sleep 3
    }
    pgrep -x tailscaled >/dev/null || { echo "tailscaled FAILED"; tail -5 runlogs/tailscaled.log; exit 1; }

    if [ -n "$AUTHKEY" ]; then
      echo "tailscaled up (userspace). Non-interactive login (authkey)..."
      # --reset so a re-run cannot inherit stale flags from a previous `up`.
      tailscale up --authkey="$AUTHKEY" --hostname=devai-host --ssh \
        --advertise-tags=tag:devai --reset \
        || { echo "tailscale up FAILED (authkey expired/not tag-authorised?)"; exit 1; }
      tailscale status >/dev/null 2>&1 \
        || { echo "logged in but status unavailable"; exit 1; }
      echo "ON TAILNET: $(tailscale ip -4 2>/dev/null | head -1) ($(tailscale status --json 2>/dev/null | grep -o '\"DNSName\":\"[^\"]*\"' | head -1))"
      echo "next: bash scripts/connect_server.sh bridge <server-tailscale-ip>"
      exit 0
    fi

    echo "tailscaled up (userspace). Requesting login..."
    tailscale up --hostname=devai-host 2>&1 | tee /tmp/ts_up.log &
    sleep 8
    echo ""
    echo ">>> APPROVE THIS DEVICE (open in a browser on your tailnet):"
    grep -oE "https://login.tailscale.com/[a-zA-Z0-9/]+" /tmp/ts_up.log | head -1 \
      || echo "    (no URL yet — run: tailscale up --hostname=devai-training host)"
    echo ">>> Then tag it tag:devai in the admin console (ACL cage), then run:"
    echo ">>>   bash scripts/connect_server.sh bridge <server-tailscale-ip>"
    ;;
  bridge)
    IP="${2:?usage: connect_server.sh bridge <server-ip-or-magicdns-name>}"
    # ---- WHICH TRANSPORT (2026-09-22) ---------------------------------------
    # THE BRIDGE ALWAYS PRESENTS THE SAME ENDPOINT to the game client —
    # 127.0.0.1:25565 — so nothing downstream has to know which way the packets
    # actually leave this box: not `environment.remote_server` in
    # configs/minecraft_skybot.yaml, not launch_skybot.sh's pre-flight, not
    # host.yml's `connect` action. Only the FAR side of the socat changes, which
    # is why moving from a rented host to a LAN machine needs no config edit and
    # no workflow edit — just a different address in the runner .env.
    #
    # RFC1918 ADDRESSES GO OVER PLAIN TCP. `tailscale nc` speaks only to the
    # tailnet, so it silently cannot carry 192.168.1.x. EVERYTHING ELSE KEEPS
    # `tailscale nc` untouched: 100.x CGNAT tailnet addresses, MagicDNS names,
    # and anything unrecognised. Defaulting the UNKNOWN case to tailscale
    # rather than to LAN is deliberate — every input that works today keeps
    # working, and a MagicDNS name is not pattern-matchable as a tailnet
    # address, so guessing "LAN" on unknown input would break the proven path.
    #
    # CONNECT_MODE=lan|tailscale overrides the guess when it is wrong.
    case "${CONNECT_MODE:-auto}" in
      lan|tailscale) MODE="$CONNECT_MODE" ;;
      *)
        case "$IP" in
          127.*|localhost)
            echo "REFUSED: $IP is this machine. A bridge from 127.0.0.1:25565"
            echo "  to itself would loop. If the Paper server runs on THIS box,"
            echo "  no bridge is needed — 127.0.0.1:25565 already reaches it."
            exit 1 ;;
          192.168.*|10.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) MODE=lan ;;
          *) MODE=tailscale ;;
        esac ;;
    esac
    if [ "$MODE" = "tailscale" ]; then
      tailscale status >/dev/null 2>&1 || { echo "not logged in — run: connect_server.sh login"; exit 1; }
      FAR="EXEC:tailscale nc $IP 25565"
    else
      # NO `tailscale status` GATE ON THIS BRANCH (CLAUDE.md 4.1). On the LAN
      # path it is a precondition nothing on that path can satisfy, and a guard
      # whose only escape is "a human edits the script" is a latch. The MC ping
      # below is the check that actually carries value here, and it is
      # transport-independent — it fails for exactly the same reasons either
      # way, and it can be satisfied by fixing the thing it names.
      FAR="TCP:$IP:25565"
    fi
    echo "bridge mode: $MODE (far side $IP:25565)"
    pkill -x socat 2>/dev/null || true; sleep 1
    setsid socat "TCP-LISTEN:25565,bind=127.0.0.1,fork,reuseaddr" \
      "$FAR" > runlogs/socat_mc.log 2>&1 < /dev/null &
    sleep 2
    ss -tln 2>/dev/null | grep -q "127.0.0.1:25565" || { echo "bridge FAILED"; cat runlogs/socat_mc.log; exit 1; }
    echo "bridge up ($MODE): 127.0.0.1:25565 -> $IP:25565"
    if timeout 12 python3 scripts/mc_ping.py 127.0.0.1 25565 754 2>/dev/null | grep -E "version|players"; then
      echo "SERVER REACHABLE — ready. Launch: bash scripts/launch_skybot.sh 1000000"
    else
      echo "WARN: bridge up but server did not answer the MC ping."
      echo "  Check: server online, ViaVersion+ViaBackwards loaded (accepts protocol 754),"
      if [ "$MODE" = "tailscale" ]; then
        echo "  its tailscale IP correct, and the tag:devai ACL allows $IP:25565."
      else
        echo "  its LAN IP correct, server-ip= in server.properties NOT bound to"
        echo "  127.0.0.1 (it must listen on the LAN), and no firewall on 25565."
      fi
    fi
    ;;
  *)
    echo "usage: connect_server.sh login [--authkey <key>] | bridge <server-ip-or-magicdns-name>"
    echo "       bridge auto-selects plain TCP for RFC1918 (192.168./10./172.16-31.)"
    echo "       and tailscale nc for everything else; CONNECT_MODE=lan|tailscale overrides."
    exit 1;;
esac
