#!/usr/bin/env bash
# install_node2.sh -- install the SkyBot brain mirror on node2. RUN ON node2.
#
# Idempotent: re-running rewrites the units with the same content and
# re-enables the timer. Installs from the directory this script lives in
# (the scp'd scripts/brain_mirror/ folder) -- keep it there.
#
# Usage:
#   ./install_node2.sh            # systemd USER service + timer (default)
#   ./install_node2.sh --system   # system service (needs sudo; runs as you)
#   ./install_node2.sh --no-timer # checks + dirs only; print the crontab line
#
# Checks, in order: rsync/python3(>=3.8)/ssh present -> brain_mirror.env
# exists and has no <placeholders> -> the ssh key reaches the training host
# non-interactively (BatchMode) -> BRAIN_DEST is created.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || true)"
SCRIPT="$HERE/brain_mirror.py"
ENVF="$HERE/brain_mirror.env"
MODE=user
TIMER=1
for a in "$@"; do
  case "$a" in
    --system) MODE=system ;;
    --no-timer) TIMER=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

die() { echo "INSTALL FAILED: $*" >&2; exit 1; }

echo "== 1/5 tools"
for t in rsync ssh python3; do
  command -v "$t" >/dev/null 2>&1 || die "$t not installed (apt install $t)"
done
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' \
  || die "python3 >= 3.8 required, found $("$PY" --version 2>&1)"
[ -f "$SCRIPT" ] || die "brain_mirror.py not found beside this script ($HERE)"
echo "   rsync: $(rsync --version | head -1)"
echo "   python3: $("$PY" --version 2>&1)"

echo "== 2/5 config ($ENVF)"
if [ ! -f "$ENVF" ]; then
  cp "$HERE/brain_mirror.env.example" "$ENVF"
  chmod 600 "$ENVF"
  die "created $ENVF from the example -- fill in MAIN_HOST, MAIN_USER, MAIN_SSH_KEYFILE, BRAIN_DEST, then re-run"
fi
chmod 600 "$ENVF"
"$PY" "$SCRIPT" --env "$ENVF" --check-config >/dev/null \
  || die "fix $ENVF (see the error above)"
DEST="$("$PY" "$SCRIPT" --env "$ENVF" --check-config \
        | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["dest"])')"

echo "== 3/5 ssh reachability (BatchMode: no password prompts)"
"$PY" "$SCRIPT" --env "$ENVF" --probe \
  || die "node2 cannot reach the training host with the key in $ENVF.
   Generate one and authorize it on the training host:
     ssh-keygen -t ed25519 -N '' -f ~/.ssh/skybot_brain_mirror_ed25519
     ssh-copy-id -i ~/.ssh/skybot_brain_mirror_ed25519.pub <MAIN_USER>@<MAIN_HOST>"

echo "== 4/5 destination $DEST"
mkdir -p "$DEST"
df -h "$DEST" | tail -1 | awk '{print "   filesystem " $1 " mounted at " $6 ", " $4 " free"}'
case "$(df -P "$DEST" | tail -1 | awk '{print $6}')" in
  /) echo "   NOTE: $DEST is on the ROOT filesystem. Fine if the root disk IS the"
     echo "         hard drive (single-disk node2); otherwise mount the HDD first."
     echo "         MIN_FREE_GB keeps the mirror from filling the OS disk." ;;
esac

CRON="*/5 * * * * nice -n 10 $PY $SCRIPT --env $ENVF >/dev/null 2>&1"
EXEC="$PY $SCRIPT --env $ENVF"

if [ "$TIMER" = 0 ]; then
  echo "== 5/5 skipped (--no-timer). Crontab fallback (crontab -e):"
  echo "   $CRON"
  exit 0
fi

echo "== 5/5 systemd ($MODE) service + timer"
SERVICE="[Unit]
Description=SkyBot brain mirror (pull training-host brain to node2)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=$EXEC
Nice=10
IOSchedulingClass=idle
# bounds a hung run, so its lock can never outlive it
TimeoutStartSec=45min
"
TIMERU="[Unit]
Description=Run the SkyBot brain mirror every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
AccuracySec=30s

[Install]
WantedBy=timers.target
"
if [ "$MODE" = system ]; then
  command -v sudo >/dev/null || die "--system needs sudo"
  UD=/etc/systemd/system
  SERVICE="${SERVICE}User=$(id -un)
"
  printf '%s' "$SERVICE" | sudo tee "$UD/skybot-brain-mirror.service" >/dev/null
  printf '%s' "$TIMERU"  | sudo tee "$UD/skybot-brain-mirror.timer" >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now skybot-brain-mirror.timer
  STATUS="systemctl status skybot-brain-mirror.timer"
  LOGS="journalctl -u skybot-brain-mirror.service"
else
  # A user timer needs the per-user systemd instance. Over ssh/su on a server
  # it is often absent ("Failed to connect to user scope bus"): enable linger
  # (which starts it), point at its runtime dir, and if there is still no bus
  # stop with the fix instead of a D-Bus error.
  if [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || echo no)" != yes ]; then
    loginctl enable-linger "$(id -un)" 2>/dev/null \
      || sudo loginctl enable-linger "$(id -un)" 2>/dev/null || true
  fi
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  systemctl --user show-environment >/dev/null 2>&1 \
    || die "no per-user systemd session here -- re-run as: $0 --system"
  UD="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "$UD"
  printf '%s' "$SERVICE" > "$UD/skybot-brain-mirror.service"
  printf '%s' "$TIMERU"  > "$UD/skybot-brain-mirror.timer"
  systemctl --user daemon-reload
  systemctl --user enable --now skybot-brain-mirror.timer
  # Without linger a user timer only runs while you are logged in.
  if [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || echo no)" != yes ]; then
    loginctl enable-linger "$(id -un)" 2>/dev/null \
      || sudo loginctl enable-linger "$(id -un)" \
      || echo "   WARNING: could not enable linger -- run: sudo loginctl enable-linger $(id -un)"
  fi
  STATUS="systemctl --user status skybot-brain-mirror.timer"
  LOGS="journalctl --user -u skybot-brain-mirror.service"
fi

echo
echo "Installed. First run ~2 min after boot, then every 5 min."
echo "  run now:   $EXEC      (no flag = take a snapshot)"
echo "  status:    $EXEC --status"
echo "  timer:     $STATUS"
echo "  logs:      $DEST/mirror.log   (and $LOGS)"
echo "Crontab fallback if systemd is unavailable (crontab -e):"
echo "  $CRON"
