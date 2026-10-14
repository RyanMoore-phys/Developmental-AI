# roms/ — game files SkyBot plays

Everything in this directory except this README is **gitignored**. The repo is
public, so no game file ever enters it, free or not.

| Game | Path | Source |
|---|---|---|
| Doom | `roms/doom/freedoom2.wad` (and `freedoom1.wad`) | `bash scripts/fetch_roms.sh` (Freedoom, BSD-licensed; sha256-verified) |
| Commercial games | `roms/<game>/<file>` as named in `configs/games/<game>.yaml` | **Your own dump.** Nothing in this repo downloads commercial ROMs. |

A game's adapter refuses to start if its file is missing, and the error names
the exact path to fill. See `docs/UNIVERSAL_GAMES.md`.
