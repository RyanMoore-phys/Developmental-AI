#!/usr/bin/env bash
# Populate roms/<game>/ with the game files SkyBot plays — FREE GAMES ONLY.
#
# Born 2026-10-11 with the universal-games work (docs/UNIVERSAL_GAMES.md).
# roms/ is gitignored (the repo is public), so every machine that runs a game
# stream — the Mac for local tests, `main` for training — runs this once.
#
# WHAT THIS FETCHES, AND WHAT IT NEVER WILL
#   doom   Freedoom phase 1 + 2 (BSD-licensed). ViZDoom's wheel already ships
#          them, so the primary source is the installed package — no network.
#          Fallback: the official Freedoom release on GitHub.
#   Commercial games (Pokemon, ...) are NEVER downloaded by this script. Put
#   your own dump at the path the game's configs/games/<game>.yaml names; the
#   adapter refuses to start with a message naming that path if it is absent.
#
# Idempotent. Verifies every file by sha256 and exits non-zero on a mismatch,
# so a provision log shows a corrupt or swapped WAD instead of training on it.
#
#   bash scripts/fetch_roms.sh [python]     # python defaults to ./venv_mc/bin/python, then ./venv/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${1:-}"
if [ -z "$PY" ]; then
    for c in ./venv_mc/bin/python ./venv/bin/python python3; do
        if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY="$c"; break; fi
    done
fi

# Pinned to vizdoom==1.3.1's bundled copies (the version provision installs).
# A different vizdoom may bundle a different Freedoom: the check then fails
# LOUDLY and names both hashes — update the pin deliberately, never silently.
# A `case`, not `declare -A`: macOS ships bash 3.2, where an associative
# array is an ARITHMETIC error that `set -e` does not catch -- the first
# draft printed FETCH-ROMS OK after failing every hash comparison.
pinned_sha() {
    case "$1" in
        freedoom1.wad) echo 7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d ;;
        freedoom2.wad) echo a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b ;;
        *) echo "FETCH-ROMS FAILED: no pinned sha256 for $1" >&2; exit 1 ;;
    esac
}
FREEDOOM_URL="https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

mkdir -p roms/doom
VZD_DIR=$("$PY" -c 'import os, vizdoom; print(os.path.dirname(vizdoom.__file__))' 2>/dev/null || true)

for f in freedoom1.wad freedoom2.wad; do
    dst="roms/doom/$f"
    want=$(pinned_sha "$f")
    if [ -f "$dst" ] && [ "$(sha "$dst")" = "$want" ]; then
        echo "  doom: $f present and verified"
        continue
    fi
    if [ -n "$VZD_DIR" ] && [ -f "$VZD_DIR/$f" ]; then
        cp "$VZD_DIR/$f" "$dst"
        echo "  doom: $f copied from the installed vizdoom package"
    else
        echo "  doom: vizdoom not importable with $PY — downloading the Freedoom release"
        tmp=$(mktemp -d)
        curl -fsSL "$FREEDOOM_URL" -o "$tmp/fd.zip"
        unzip -q -o "$tmp/fd.zip" -d "$tmp"
        src=$(find "$tmp" -name "$f" | head -1)
        [ -n "$src" ] || { echo "FETCH-ROMS FAILED: $f not in $FREEDOOM_URL"; exit 1; }
        cp "$src" "$dst"; rm -rf "$tmp"
    fi
    got=$(sha "$dst")
    if [ "$got" != "$want" ]; then
        echo "FETCH-ROMS FAILED: $dst sha256 $got != pinned $want"
        echo "  (a different vizdoom or Freedoom release? update the pin in this script deliberately)"
        exit 1
    fi
    echo "  doom: $f verified"
done
echo "FETCH-ROMS OK"
