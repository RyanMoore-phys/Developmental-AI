#!/usr/bin/env bash
# Manage which games SkyBot's ONE universal brain plays. Runs ON THE HOST,
# from the repo root (/workspace/devai). Called by the per-game workflows
# (.github/workflows/{doom,minecraft}.yml) through scripts/host_exec.sh; safe by hand.
#
#   bash scripts/game_ctl.sh <game> enable    add <game> to runlogs/games.enabled
#   bash scripts/game_ctl.sh <game> disable   remove it (refuses to remove the LAST)
#   bash scripts/game_ctl.sh <game> status    what is enabled, what is running
#   bash scripts/game_ctl.sh <game> install   engine + game files for <game>
#   bash scripts/game_ctl.sh any    legacy    delete games.enabled -> the old
#                                             Minecraft-only brain on next launch
#
# Takes effect at the NEXT launch: launch_skybot.sh composes the run config
# from games.enabled every time it starts the agent. There is always exactly
# one trainer and one brain; enabling a game adds that game's streams to it.
#
# docs/UNIVERSAL_GAMES.md §6. The no-latch rules (CLAUDE.md 4.1) for the two
# refusals here: "disable the last game" is reversed by enabling another or by
# `legacy`; an unknown game name is reversed by adding configs/games/<g>.yaml.
set -euo pipefail
cd "${SKYBOT_ROOT:-/workspace/devai}"
GAME="${1:?usage: game_ctl.sh <game> <enable|disable|status|install|legacy>}"
ACTION="${2:?usage: game_ctl.sh <game> <enable|disable|status|install|legacy>}"
GAME=$(echo "$GAME" | tr 'A-Z' 'a-z')
F=runlogs/games.enabled
mkdir -p runlogs
PY=./venv_mc/bin/python
[ -x "$PY" ] || PY=python3

enabled() { [ -f "$F" ] && sed 's/#.*//' "$F" | tr 'A-Z' 'a-z' | tr -s ' \t\n' '\n' | grep -v '^$' || true; }

known_game() {
    [ -f "configs/games/$1.yaml" ] || {
        echo "GAME-CTL REFUSED: unknown game '$1' (no configs/games/$1.yaml)."
        echo "  known: $(ls configs/games 2>/dev/null | sed 's/\.yaml$//' | tr '\n' ' ')"
        exit 2; }
}

write_list() {   # stdin: one game per line -> atomically replace $F
    { echo "# SkyBot universal brain — games played (one per line)."
      echo "# Managed by scripts/game_ctl.sh / the Game workflows. Read at every launch."
      cat; } > "$F.tmp"
    mv -f "$F.tmp" "$F"
}

case "$ACTION" in
  enable)
    known_game "$GAME"
    if [ ! -f "$F" ]; then
        echo "  switching this host from the LEGACY Minecraft-only brain to the"
        echo "  UNIVERSAL brain (fresh; the legacy brain is left untouched)."
        [ "$GAME" != minecraft ] && echo "  NOTE: Minecraft is not enabled. Run the Minecraft game action (enable) to add it."
    fi
    if enabled | grep -qx "$GAME"; then
        echo "  $GAME already enabled"
    else
        { enabled; echo "$GAME"; } | grep -v '^$' | write_list
        echo "  enabled $GAME"
    fi
    ;;
  disable)
    known_game "$GAME"
    if ! enabled | grep -qx "$GAME"; then
        echo "  $GAME is not enabled"
    else
        REST=$(enabled | grep -vx "$GAME" || true)
        if [ -z "$REST" ]; then
            echo "GAME-CTL REFUSED: $GAME is the only enabled game; disabling it leaves"
            echo "  the universal brain nothing to play. Enable another game first, stop"
            echo "  the run (host action: stop), or go back to the legacy Minecraft brain"
            echo "  (Minecraft game action: legacy)."
            exit 3
        fi
        echo "$REST" | write_list
        echo "  disabled $GAME"
    fi
    ;;
  legacy)
    if [ -f "$F" ]; then
        mv -f "$F" "$F.bak"
        echo "  removed $F (kept as $F.bak): the next launch runs the LEGACY"
        echo "  Minecraft-only brain from configs/minecraft_skybot.yaml."
    else
        echo "  already legacy (no $F)"
    fi
    ;;
  install)
    known_game "$GAME"
    # Everything a game needs is declared in configs/games/<game>.yaml
    # `install:` and installed by tools/game_install.py: the LAN game depot
    # (<repo>/depot, synced by the runner — scripts/depot_sync.sh) first,
    # the internet second, nothing already present re-downloaded.
    # docs/UNIVERSAL_GAMES.md §12.
    "$PY" tools/game_install.py "$GAME" --depot "${SKYBOT_DEPOT:-$PWD/depot}"
    ;;
  status)
    if [ -f "$F" ]; then
        echo "mode: UNIVERSAL brain — games: $(enabled | tr '\n' ' ')"
    else
        echo "mode: LEGACY Minecraft-only brain (no $F)"
    fi
    PID=$(cat runlogs/skybot_run.pid 2>/dev/null || true)
    if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
        echo "run: LIVE pid=$PID config=$(ps -p "$PID" -o args= | sed -n 's/.*--config \([^ ]*\).*/\1/p')"
    else
        echo "run: not running"
    fi
    if [ -f runlogs/run_config.yaml ]; then
        echo "last composed: $(sed -n '2,3p' runlogs/run_config.yaml | tr '\n' ' ')"
    fi
    if [ "$GAME" != any ] && [ "$GAME" != minecraft ]; then
        echo "--- $GAME save-state rings (brain_universal/snapshots):"
        ls -la brain_universal/snapshots/ 2>/dev/null | sed 's/^/    /' || echo "    (none yet)"
        echo "--- last $GAME lines in the run log:"
        grep -E "$GAME stream" runlogs/minecraft_skybot_run.log 2>/dev/null | tail -6 | sed 's/^/    /' || true
    fi
    ;;
  *)
    echo "usage: game_ctl.sh <game> <enable|disable|status|install|legacy>"; exit 2 ;;
esac
