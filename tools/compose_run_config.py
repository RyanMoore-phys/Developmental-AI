#!/usr/bin/env python3
"""Compose the run config for SkyBot's universal brain from the enabled games.

docs/UNIVERSAL_GAMES.md §3. Run by scripts/launch_skybot.sh at every launch
when runlogs/games.enabled exists:

    base       configs/minecraft_skybot.yaml   the tuned learner
  + brain      configs/universal.yaml          one brain's shapes and paths
  + primary    configs/games/<stream-0 game>.yaml `primary:` block
  + streams    one parallel_envs.streams entry per stream of every game
  = out        runlogs/run_config.yaml

    python tools/compose_run_config.py --games minecraft,doom
    python tools/compose_run_config.py --games-file runlogs/games.enabled

Refuses (exit 2, message names the problem) rather than writing a config the
brain cannot run: unknown game, controller slot 0 not the no-op, more
controller slots than the brain has, no games at all.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any, Dict, List

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ComposeError(ValueError):
    pass


def deep_merge(base: Dict, over: Dict) -> Dict:
    """Maps merge key by key; everything else (lists included) replaces."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_games_file(path: str) -> List[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip().lower()
            if line and line not in out:
                out.append(line)
    return out


def order_games(games: List[str], primary_order: List[str]) -> List[str]:
    """primary_order first (in that order), then the rest alphabetically."""
    first = [g for g in primary_order if g in games]
    return first + sorted(g for g in games if g not in first)


def load_game(name: str, games_dir: str) -> Dict[str, Any]:
    path = os.path.join(games_dir, f"{name}.yaml")
    if not os.path.isfile(path):
        known = sorted(f[:-5] for f in os.listdir(games_dir) if f.endswith(".yaml"))
        raise ComposeError(f"unknown game {name!r}: no {path} (known: {known})")
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    g = doc.get("game") or {}
    if g.get("name") != name:
        raise ComposeError(f"{path}: game.name {g.get('name')!r} != {name!r}")
    if int(g.get("streams", 1)) < 1:
        raise ComposeError(f"{path}: game.streams must be >= 1")
    return doc


def compose(games: List[str], base_path: str, universal_path: str,
            games_dir: str) -> Dict[str, Any]:
    if not games:
        raise ComposeError("no games enabled")
    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(universal_path) as f:
        uni = yaml.safe_load(f)
    cfg = deep_merge(base, uni)
    ucfg = cfg.get("universal") or {}
    slots = int(ucfg.get("action_slots", 0))
    if slots < 1:
        raise ComposeError("universal.action_slots must be set")
    ordered = order_games(games, list(ucfg.get("primary_order") or []))
    docs = {g: load_game(g, games_dir) for g in ordered}

    streams: List[Dict[str, Any]] = []
    for name in ordered:
        g = docs[name]["game"]
        if g.get("backend") == "player":
            # keys/buttons/mouse controller: whitelist, OS-key refusal,
            # slot 0 = release-all, no one-way doors (CLAUDE.md §4.5)
            sys.path.insert(0, ROOT)
            from developmental_ai.environments.player.controller import (
                ControllerError, validate)
            try:
                validate(g, slots)
            except ControllerError as e:
                raise ComposeError(f"{name}: {e}")
        ctrl = g.get("controller")
        if ctrl is not None and g.get("backend") != "player":
            if not ctrl or ctrl[0]:
                raise ComposeError(f"{name}: controller slot 0 must be {{}} "
                                   f"(every game's slot 0 is its no-op)")
            if len(ctrl) > slots:
                raise ComposeError(
                    f"{name}: {len(ctrl)} controller slots > the brain's "
                    f"universal.action_slots={slots}")
        for _ in range(int(g.get("streams", 1))):
            streams.append({"game": name, "env_name": g["env_name"],
                            "backend": g.get("backend"),
                            "game_cfg": copy.deepcopy(g)})

    primary = ordered[0]
    cfg = deep_merge(cfg, docs[primary].get("primary") or {})
    if docs[primary]["game"].get("backend") != "minerl":
        cfg["environment"]["name"] = docs[primary]["game"]["env_name"]
    par = cfg.setdefault("parallel_envs", {})
    par["enabled"] = True
    par["num_envs"] = len(streams)
    par["streams"] = streams
    cfg["universal"]["games"] = ordered
    if len(streams) < 2:
        # num_envs 1 drops into the unwired legacy single-env body (CLAUDE.md
        # §4.2) — the pre-Wave-1 agent. One game with one stream is not
        # "SkyBot on one client"; refuse instead of silently running it.
        raise ComposeError("a universal run needs >= 2 streams in total "
                           "(num_envs 1 selects the legacy single-env body)")
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--games", help="comma-separated, e.g. minecraft,doom")
    src.add_argument("--games-file", help="one game per line (runlogs/games.enabled)")
    ap.add_argument("--base", default=os.path.join(ROOT, "configs/minecraft_skybot.yaml"))
    ap.add_argument("--universal", default=os.path.join(ROOT, "configs/universal.yaml"))
    ap.add_argument("--games-dir", default=os.path.join(ROOT, "configs/games"))
    ap.add_argument("--out", default="runlogs/run_config.yaml")
    a = ap.parse_args(argv)
    games = ([g.strip().lower() for g in a.games.split(",") if g.strip()]
             if a.games else read_games_file(a.games_file))
    try:
        cfg = compose(games, a.base, a.universal, a.games_dir)
    except (ComposeError, OSError, yaml.YAMLError) as e:
        print(f"COMPOSE FAILED: {e}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write("# GENERATED by tools/compose_run_config.py — do not edit.\n"
                f"# games (stream order): {', '.join(cfg['universal']['games'])}\n"
                f"# streams: {[s['game'] for s in cfg['parallel_envs']['streams']]}\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp, a.out)
    print(f"COMPOSED {a.out}: games={','.join(cfg['universal']['games'])} "
          f"streams={len(cfg['parallel_envs']['streams'])} "
          f"primary={cfg['universal']['games'][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
