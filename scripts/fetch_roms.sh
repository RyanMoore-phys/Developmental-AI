#!/usr/bin/env bash
# Populate roms/<game>/ with the game files SkyBot plays — FREE GAMES ONLY.
#
# Born 2026-10-11 with the universal-games work (docs/UNIVERSAL_GAMES.md).
# roms/ is gitignored (the repo is public), so every machine that runs a game
# stream — the Mac for local tests, `main` for training — runs this once.
#
# WHAT THIS FETCHES, AND WHAT IT NEVER WILL
#   doom   Freedoom phase 1 + 2 (BSD-licensed). Local copies first (the game
#          depot, ViZDoom's wheel, apt's freedoom package) — no network.
#          Fallback: the official Freedoom release on GitHub.
#   Commercial games (Pokemon, ...) are NEVER downloaded by this script. Put
#   your own dump at the path the game's configs/games/<game>.yaml names; the
#   adapter refuses to start with a message naming that path if it is absent.
#
# Idempotent. Verifies every file by sha256 and exits non-zero on a mismatch,
# so a provision log shows a corrupt or swapped WAD instead of training on it.
#
#   bash scripts/fetch_roms.sh [python]     # python defaults to ./venv_mc/bin/python, then ./venv/bin/python
#   SKYBOT_DEPOT=<dir>                      # optional: the game depot (tools/game_install.py)
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
DEPOT="${SKYBOT_DEPOT:-}"

# SOURCES, FIRST VERIFIED WINS (2026-10-14). This used to be "vizdoom package,
# else download the GitHub zip and `unzip` it" -- and the doom_player install
# never installs vizdoom, so it always took the download path on a slow link
# and needed an `unzip` nothing installs. Now every local copy is tried first,
# and a source whose sha256 does not match is SKIPPED (named), not fatal:
#   1. the game depot (files/doom/), filled by an earlier install
#   2. the installed vizdoom package (the pinned shas ARE its copies)
#   3. /usr/share/games/doom -- apt's `freedoom` package, which doom_player's
#      `apt install chocolate-doom` pulls in anyway
#   4. the official Freedoom release, unzipped with Python (no `unzip`)
# A verified file is copied back into the depot so the next host gets it
# over the LAN.
try_src() {   # $1 = candidate path, $2 = dst, $3 = wanted sha; 0 = installed
    [ -f "$1" ] || return 1
    local got; got=$(sha "$1")
    if [ "$got" != "$3" ]; then
        echo "  doom: $1 sha256 $got != pinned $3 -- skipped"
        return 1
    fi
    cp "$1" "$2"
    echo "  doom: $(basename "$2") from $1"
}

for f in freedoom1.wad freedoom2.wad; do
    dst="roms/doom/$f"
    want=$(pinned_sha "$f")
    if [ -f "$dst" ] && [ "$(sha "$dst")" = "$want" ]; then
        echo "  doom: $f present and verified"
    elif { [ -n "$DEPOT" ] && try_src "$DEPOT/files/doom/$f" "$dst" "$want"; } \
      || { [ -n "$VZD_DIR" ] && try_src "$VZD_DIR/$f" "$dst" "$want"; } \
      || try_src "/usr/share/games/doom/$f" "$dst" "$want"; then
        :
    else
        echo "  doom: no local copy of $f -- downloading the Freedoom release"
        tmp=$(mktemp -d)
        curl -fsSL --retry 3 --connect-timeout 30 "$FREEDOOM_URL" -o "$tmp/fd.zip" \
          || { echo "FETCH-ROMS FAILED: could not download $FREEDOOM_URL"; rm -rf "$tmp"; exit 1; }
        "$PY" - "$tmp/fd.zip" "$f" "$tmp/$f" <<'PYEOF' || { echo "FETCH-ROMS FAILED: $f not in $FREEDOOM_URL"; rm -rf "$tmp"; exit 1; }
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == sys.argv[2]]
if not names:
    sys.exit(1)
with z.open(names[0]) as src, open(sys.argv[3], "wb") as out:
    out.write(src.read())
PYEOF
        try_src "$tmp/$f" "$dst" "$want" || {
            echo "FETCH-ROMS FAILED: no source for $f matched the pinned sha256 $want"
            echo "  (a different vizdoom or Freedoom release? update the pin in this script deliberately)"
            rm -rf "$tmp"; exit 1; }
        rm -rf "$tmp"
    fi
    # missing OR corrupt in the depot -> replace it with the verified copy
    if [ -n "$DEPOT" ] && [ -d "$DEPOT" ] && { [ ! -f "$DEPOT/files/doom/$f" ] \
         || [ "$(sha "$DEPOT/files/doom/$f")" != "$want" ]; }; then
        mkdir -p "$DEPOT/files/doom" && cp "$dst" "$DEPOT/files/doom/$f" \
          && echo "  doom: $f saved to the depot"
    fi
done
echo "FETCH-ROMS OK"
