#!/usr/bin/env bash
# Make the training host RECOVER FROM A HANG by itself (2026-10-07).
#
# LIVE INCIDENT: on 2026-10-06 03:07:58 UTC the host stopped logging mid-
# stream (ordinary sshd lines, then nothing) 22 s after the agent aborted at
# exit, and stayed dead ~41 h until a human rebooted it. No OOM, no panic, no
# shutdown made it to the journal. Two things were missing: (1) nothing turns
# a kernel hang into a reboot, and (2) nothing keeps the panic text across
# the reboot, so the next incident would be just as unexplained.
#
# This script, run ON the host:
#   * kernel.panic=10, panic_on_oops=1, softlockup_panic=1  -> a panic or a
#     soft lockup reboots in 10 s instead of hanging forever. hung_task_panic
#     is OPT-IN (--panic-on-hung-io, 180 s): a legitimately slow flush could
#     trip it, but on this host a > 180 s I/O block is itself the incident.
#   * hardware watchdog (AMD sp5100_tco) fed by systemd: a HARD hang (kernel
#     can no longer run) reboots after 60 s. Skipped if no watchdog device.
#   * systemd-pstore: the panic log is saved to /var/lib/systemd/pstore and
#     survives the reboot.
#   * optional --disable-wifi: the Intel iwlwifi card crashed its firmware on
#     2026-10-05 22:36 ("Microcode SW error ... Device error - SW reset") and
#     its Bluetooth half fails to load firmware on every boot. The host is
#     wired (enp5s0). Refused if the default route uses a wireless interface.
#
# UPDATE 2026-10-07 (evidence from the host): NOT a core dump (apport wrote
# nothing new), NOT a kernel panic (kdump is installed and saved nothing),
# NOT an OOM kill (none logged). The next boot reports a SOFTWARE reset, so
# the kernel was alive 41 h later, yet nothing reached disk and no SSH login
# succeeded: most consistent with a storage (NVMe) or NIC stall. Hence:
#   * console loglevel 7 -- warnings print on the host's screen even when
#     the disk cannot take them (the screen becomes the black box);
#   * optional --panic-on-hung-io -- tasks blocked on I/O > 180 s panic the
#     kernel, so kdump saves a vmcore of WHAT was stuck, then it reboots;
#   * optional --nvme-no-apst -- disables NVMe autonomous power-state
#     transitions (a known controller-hang cause); needs a reboot.
#
# UPDATE 2026-10-07 (user): the 41 h gap ended with a manual power-down, and
# memory was near-full with swap climbing before it. A 16 GB host with 4 GB
# of swap can THRASH for hours without the kernel OOM killer ever firing: no
# disk writes complete, no SSH login completes, nothing is logged -- exactly
# the observed signature. Hence:
#   * optional --earlyoom -- kill the largest process (the agent) BEFORE the
#     box thrashes; sshd/journald/tailscaled/dockerd are protected. The
#     supervisor treats the kill as a crash and relaunches, so the host stays
#     reachable and the event is logged (journalctl -u earlyoom).
#
# DRY RUN BY DEFAULT: prints what it would change. Apply with --apply.
#   sudo bash scripts/host_hardening.sh            # show current state + plan
#   sudo bash scripts/host_hardening.sh --apply    # apply
#   sudo bash scripts/host_hardening.sh --apply --disable-wifi --panic-on-hung-io
#   sudo bash scripts/host_hardening.sh --apply --nvme-no-apst   (then reboot)
#   sudo bash scripts/host_hardening.sh --apply --earlyoom       (needs apt)
#   sudo bash scripts/host_hardening.sh --apply --netconsole node1   (see below)
#
# It does NOT auto-start training after a reboot: after a crash a human
# should look first (CLAUDE.md §5). Start it with host.yml -> launch.
# UPDATE 2026-10-07 (telemetry): the screen is a black box only if someone is
# standing at it. Hence:
#   * optional --netconsole <target>[:port] -- the kernel streams every console
#     message (the same loglevel-7 stream as the screen) as UDP to <target>
#     (default port 6666), from the kernel itself, with no disk and no
#     userspace involved: it keeps working when the NVMe or the memory is what
#     stalled. Costs 0 bytes until the kernel prints. <target> is a name or
#     address of the receiver (node1) and is only ever written to /etc on the
#     host, never into this tree. It must be ON-LINK (same LAN): netconsole
#     sends raw frames, so it cannot use tailscale0, and through a router it
#     needs the gateway's MAC -- both are refused with the reason. On the
#     receiver run (printed again at the end):
#         nc -u -l 6666 >> netconsole.log
#   sudo bash scripts/host_hardening.sh --apply --netconsole node1
#
set -u
APPLY=0; WIFI=0; HUNGIO=0; NOAPST=0; EOOM=0; NETCON=""
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --disable-wifi) WIFI=1 ;;
    --panic-on-hung-io) HUNGIO=1 ;;
    --nvme-no-apst) NOAPST=1 ;;
    --earlyoom) EOOM=1 ;;
    --netconsole)
      shift
      [ $# -gt 0 ] && [ -n "$1" ] && [ "${1#--}" = "$1" ] \
        || { echo "--netconsole needs a target: --netconsole <host>[:port]"; exit 2; }
      NETCON=$1 ;;
    --netconsole=*) NETCON=${1#--netconsole=} ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
  shift
done
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
do_() { if [ "$APPLY" -eq 1 ]; then eval "$@"; else echo "  would: $*"; fi; }

echo "== 1. panic -> reboot"
SYSCTL=/etc/sysctl.d/90-skybot-recovery.conf
echo "   now: panic=$(sysctl -n kernel.panic) panic_on_oops=$(sysctl -n kernel.panic_on_oops) softlockup_panic=$(sysctl -n kernel.softlockup_panic)"
EXTRA=""
if [ "$HUNGIO" -eq 1 ]; then
  EXTRA="'kernel.hung_task_panic = 1' 'kernel.hung_task_timeout_secs = 180'"
fi
do_ "printf '%s\n' '# scripts/host_hardening.sh (2026-10-07): a hang must become a reboot, and be recorded' 'kernel.panic = 10' 'kernel.panic_on_oops = 1' 'kernel.softlockup_panic = 1' 'kernel.printk = 7 4 1 7' $EXTRA > $SYSCTL"
echo "   kdump: $(systemctl is-active kdump-tools 2>/dev/null || echo unknown) (a panic saves a vmcore under /var/crash)"
do_ "sysctl -q -p $SYSCTL"

echo "== 2. hardware watchdog"
if [ ! -e /dev/watchdog ]; then
  do_ "modprobe sp5100_tco 2>/dev/null || true"
fi
if [ -e /dev/watchdog ] || { [ "$APPLY" -eq 1 ] && [ -e /dev/watchdog ]; }; then
  echo "   device: $(ls /dev/watchdog* | tr '\n' ' ')"
  do_ "echo sp5100_tco > /etc/modules-load.d/skybot-watchdog.conf"
  do_ "mkdir -p /etc/systemd/system.conf.d && printf '%s\n' '[Manager]' 'RuntimeWatchdogSec=60s' 'RebootWatchdogSec=10min' > /etc/systemd/system.conf.d/90-skybot-watchdog.conf"
  do_ "systemctl daemon-reexec"
else
  echo "   no /dev/watchdog (sp5100_tco not supported here?) -- skipped; panic->reboot still applies"
fi

echo "== 3. keep panic logs across reboot (pstore)"
echo "   backend: $(cat /sys/module/pstore/parameters/backend 2>/dev/null || echo none); saved: $(ls /var/lib/systemd/pstore 2>/dev/null | wc -l) file(s)"
do_ "systemctl enable --now systemd-pstore.service 2>/dev/null || true"

echo "== 4. NVMe power states (APST)"
echo "   now: default_ps_max_latency_us=$(cat /sys/module/nvme_core/parameters/default_ps_max_latency_us 2>/dev/null || echo ?)"
if [ "$NOAPST" -eq 1 ]; then
  do_ "printf '%s\n' '# scripts/host_hardening.sh: NVMe APST off (controller-hang suspect, 2026-10-06)' 'GRUB_CMDLINE_LINUX_DEFAULT=\"\$GRUB_CMDLINE_LINUX_DEFAULT nvme_core.default_ps_max_latency_us=0\"' > /etc/default/grub.d/90-skybot-nvme.cfg"
  do_ "update-grub"
  echo "   takes effect after a reboot; undo: rm /etc/default/grub.d/90-skybot-nvme.cfg && update-grub"
else
  echo "   unchanged (pass --nvme-no-apst to disable; needs a reboot)"
fi

echo "== 5. memory-pressure killer (earlyoom)"
echo "   now: $(systemctl is-active earlyoom 2>/dev/null || echo not installed); swap: $(swapon --show=SIZE,USED --noheadings 2>/dev/null | tr '\n' ' ')"
if [ "$EOOM" -eq 1 ]; then
  do_ "command -v earlyoom >/dev/null || apt-get install -y earlyoom"
  # -m 4: act at <4% RAM available; -s 15: and <15% swap free. Never kill the
  # processes that keep the host reachable and observable.
  do_ "printf '%s\n' '# scripts/host_hardening.sh (2026-10-07): kill the hog before the host thrashes' 'EARLYOOM_ARGS=\"-m 4 -s 15 -r 3600 --avoid (^|/)(sshd|systemd|systemd-journal|journald|tailscaled|dockerd|containerd|Xvfb)\$\"' > /etc/default/earlyoom"
  do_ "systemctl enable --now earlyoom && systemctl restart earlyoom"
else
  echo "   unchanged (pass --earlyoom to install; recommended on this 16 GB host)"
fi

echo "== 6. Wi-Fi / Bluetooth"
DEV=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="dev"){print $(i+1); exit}}')
echo "   default route via: ${DEV:-unknown}"
if [ "$WIFI" -eq 1 ]; then
  if [ -z "$DEV" ] || [ -d "/sys/class/net/$DEV/wireless" ]; then
    echo "   REFUSED: the default route is wireless or unknown; disabling Wi-Fi would cut this host off"
  else
    do_ "printf '%s\n' '# scripts/host_hardening.sh: host is wired; this card crashed its firmware 2026-10-05' 'blacklist iwlwifi' 'blacklist iwlmvm' 'blacklist btusb' > /etc/modprobe.d/skybot-no-wifi.conf"
    do_ "modprobe -r iwlmvm iwlwifi btusb 2>/dev/null || true"
    echo "   (takes full effect after the next reboot; undo: rm /etc/modprobe.d/skybot-no-wifi.conf)"
  fi
else
  echo "   unchanged (pass --disable-wifi to blacklist iwlwifi/btusb)"
fi

echo "== 7. netconsole (kernel log over UDP to another machine)"
echo "   now: $(lsmod 2>/dev/null | grep -q '^netconsole' && echo loaded || echo not loaded)"
NC_RECV=""
if [ -n "$NETCON" ]; then
  NC_HOST=${NETCON%:*}; NC_PORT=6666
  [ "$NC_HOST" != "$NETCON" ] && NC_PORT=${NETCON##*:}
  case "$NC_PORT" in ''|*[!0-9]*) echo "   REFUSED: bad port in '$NETCON'"; NC_HOST="" ;; esac
  NC_IP=""
  [ -n "$NC_HOST" ] && NC_IP=$(getent ahostsv4 "$NC_HOST" 2>/dev/null | awk 'NR==1 {print $1}')
  NC_ROUTE=""
  [ -n "$NC_IP" ] && NC_ROUTE=$(ip -4 route get "$NC_IP" 2>/dev/null | head -1)
  NC_DEV=$(printf '%s' "$NC_ROUTE" | awk '{for(i=1;i<NF;i++) if($i=="dev"){print $(i+1); exit}}')
  NC_SRC=$(printf '%s' "$NC_ROUTE" | awk '{for(i=1;i<NF;i++) if($i=="src"){print $(i+1); exit}}')
  if [ -z "$NC_HOST" ]; then
    :
  elif [ -z "$NC_IP" ]; then
    echo "   REFUSED: cannot resolve '$NC_HOST' to an IPv4 address (netconsole has no DNS)"
  elif [ -z "$NC_DEV" ] || [ -z "$NC_SRC" ]; then
    echo "   REFUSED: no route to $NC_HOST"
  elif [ "${NC_DEV#tailscale}" != "$NC_DEV" ]; then
    echo "   REFUSED: $NC_HOST routes via $NC_DEV; netconsole needs a real (LAN) interface"
  elif printf '%s' "$NC_ROUTE" | grep -q ' via '; then
    echo "   REFUSED: $NC_HOST is not on-link ($NC_ROUTE); use a LAN-local receiver"
  else
    # The receiver's MAC if the neighbour table has it, else broadcast (the
    # kernel default) -- broadcast still reaches an on-link receiver.
    NC_MAC=$(ip neigh show "$NC_IP" 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="lladdr"){print $(i+1); exit}}')
    NC_PARAM="+6665@${NC_SRC}/${NC_DEV},${NC_PORT}@${NC_IP}/${NC_MAC}"
    echo "   target: $NC_HOST:$NC_PORT via $NC_DEV (src $NC_SRC, mac ${NC_MAC:-broadcast})"
    echo "   param:  netconsole=$NC_PARAM   ('+' = extended format: sequence numbers)"
    do_ "modprobe -r netconsole 2>/dev/null || true"
    do_ "modprobe netconsole netconsole=$NC_PARAM"
    # Persist. NOT modules-load.d: at that point in boot $NC_DEV may not be up
    # yet and netconsole fails to bind. A oneshot after network-online instead.
    do_ "printf '%s\n' '# scripts/host_hardening.sh (2026-10-07): kernel log over UDP' 'options netconsole netconsole=$NC_PARAM' > /etc/modprobe.d/skybot-netconsole.conf"
    do_ "printf '%s\n' '[Unit]' 'Description=SkyBot netconsole (kernel log over UDP)' 'Wants=network-online.target' 'After=network-online.target' '[Service]' 'Type=oneshot' 'RemainAfterExit=yes' 'ExecStart=/sbin/modprobe netconsole' '[Install]' 'WantedBy=multi-user.target' > /etc/systemd/system/skybot-netconsole.service"
    do_ "systemctl daemon-reload && systemctl enable skybot-netconsole.service"
    echo "   undo: systemctl disable skybot-netconsole; rm /etc/systemd/system/skybot-netconsole.service /etc/modprobe.d/skybot-netconsole.conf; modprobe -r netconsole"
    echo "   test: echo 'netconsole test' > /dev/kmsg   (must appear on the receiver)"
    NC_RECV="nc -u -l $NC_PORT >> netconsole.log"
  fi
else
  echo "   unchanged (pass --netconsole <receiver>[:port] to stream kernel messages off-box)"
fi

[ "$APPLY" -eq 1 ] && echo "applied." || echo "dry run only -- re-run with --apply"
if [ -n "$NC_RECV" ]; then
  echo "On the receiver (node1), keep this running (OpenBSD nc: nc -klu $NC_PORT):"
  echo "    $NC_RECV"
fi
