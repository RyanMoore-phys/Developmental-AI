"""Reproducible run manifest (plan Stage 1, item 3).

WHAT IT CLAIMS
    Two runs whose manifests compare equal (ignoring VOLATILE keys) were
    started from the same source snapshot, the same RESOLVED config, the same
    dependency versions, seeds, checkpoints, environment settings and hardware
    class. A run whose manifest differs says WHERE, as dotted key paths.

WHY A CONTENT HASH AND NOT ONLY A GIT SHA
    The training host is deployed by `rsync` of the working tree
    (scripts/deploy_skybot.sh), not by `git clone` — it has no `.git` to ask,
    and the Mac tree that produced it is routinely dirty. A git revision alone
    therefore identifies neither. `source.tree_sha256` is a hash over the
    CONTENT of every source file under SOURCE_ROOTS (path + bytes, sorted), so
    a dirty tree and an rsynced copy are both identifiable, and identical
    trees hash identically on either machine. The git revision is recorded
    alongside when available, never instead.

"RESOLVED CONFIG" IS WHAT run_minecraft.py BUILDS
    run_minecraft.py does `yaml.safe_load(open(path))`, then
    `cfg["seed"] = --seed`, then (with --forever) sets lifelong.enabled and
    lifelong.forever. There is NO defaults merging anywhere: absent keys take
    in-code `.get(key, default)` values inside the loop, which are not
    statically enumerable. The manifest records exactly that, and says so in
    `config.defaults_note`, rather than pretending to a fuller resolution.

MISSING IS EXPLICIT
    Nothing is ever omitted. A value that could not be determined is the
    string "unknown"; a thing that does not exist (package not installed,
    checkpoint not on disk, env var unset) is "absent".
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA_VERSION = 1
UNKNOWN = "unknown"
ABSENT = "absent"

# What counts as "source". Data, weights, logs and venvs are not source.
SOURCE_ROOTS = ("developmental_ai", "configs", "scripts", "tools", "tests",
                "run_minecraft.py", "requirements.txt")
SOURCE_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".toml", ".cfg", ".ini",
                   ".txt")
_SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv", "venv_mc", "runlogs",
              "logs", "node_modules", ".pytest_cache", ".mypy_cache"}

PACKAGES = ("torch", "numpy", "scipy", "gymnasium", "minerl", "crafter",
            "minigrid", "PyYAML", "Pillow", "ollama")

ENV_VARS = ("OMP_NUM_THREADS", "MINERL_HEADLESS", "PYTORCH_CUDA_ALLOC_CONF",
            "CUDA_VISIBLE_DEVICES", "DISPLAY", "PYTHONHASHSEED")

# Keys whose value legitimately differs between two builds of the same run.
VOLATILE_KEYS = ("created_at", "created_unix", "build_seconds")

# Brain-state files the loop writes under <loop.log_dir>/checkpoints
# (developmental_loop.py _save_checkpoint_locked) plus side files.
CHECKPOINT_FILES = ("world_model.pt", "curiosity.pt", "policy.pt",
                    "dream_actor.pt", "symbolic_decoder.pt", "glue_layer.pt",
                    "symbolizer.pt", "familiarity.pt", "magnet.pt",
                    "knowledge_graph.json", "options_state.json")


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iter_source_files(root: str) -> List[str]:
    out: List[str] = []
    for top in SOURCE_ROOTS:
        p = os.path.join(root, top)
        if os.path.isfile(p):
            out.append(top)
            continue
        if not os.path.isdir(p):
            continue
        for d, dirs, files in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in _SKIP_DIRS
                             and not x.startswith("."))
            for fn in files:
                if fn.endswith(SOURCE_SUFFIXES):
                    rel = os.path.relpath(os.path.join(d, fn), root)
                    out.append(rel.replace(os.sep, "/"))
    return sorted(set(out))


def source_tree_hash(root: str) -> Dict[str, Any]:
    files = _iter_source_files(root)
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode("utf-8") + b"\0")
        h.update(sha256_file(os.path.join(root, rel)).encode("ascii") + b"\n")
    return {"tree_sha256": h.hexdigest(), "n_files": len(files),
            "roots": list(SOURCE_ROOTS), "suffixes": list(SOURCE_SUFFIXES)}


def tree_hash(path: str) -> Dict[str, Any]:
    """Content hash of a directory (relpath + file sha256), e.g. a skill bank."""
    h = hashlib.sha256()
    n, size = 0, 0
    for d, dirs, files in os.walk(path):
        dirs.sort()
        for fn in sorted(files):
            fp = os.path.join(d, fn)
            rel = os.path.relpath(fp, path).replace(os.sep, "/")
            h.update(rel.encode("utf-8") + b"\0")
            h.update(sha256_file(fp).encode("ascii") + b"\n")
            n += 1
            size += os.path.getsize(fp)
    return {"sha256": h.hexdigest(), "n_files": n, "bytes": size}


def _git(root: str, *args: str) -> Optional[str]:
    try:
        p = subprocess.run(["git", "-C", root] + list(args),
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def source_identity(root: str) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(source_tree_hash(root))
    rev = _git(root, "rev-parse", "HEAD")
    if rev is None:
        out.update({"git_revision": UNKNOWN, "git_dirty": UNKNOWN,
                    "git_dirty_paths": UNKNOWN, "git_branch": UNKNOWN})
        return out
    status = _git(root, "status", "--porcelain", "--untracked-files=no") or ""
    dirty = sorted(line[3:] for line in status.splitlines() if line.strip())
    br = (_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip()
    out.update({"git_revision": rev.strip(), "git_dirty": bool(dirty),
                "git_dirty_paths": dirty, "git_branch": br or UNKNOWN})
    return out


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def resolve_config(config_path: str, seed: Optional[int] = 0,
                   forever: bool = False) -> Dict[str, Any]:
    """Build the config dict exactly as run_minecraft.py main() does."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: top level is not a mapping")
    cfg["seed"] = seed
    if forever:
        cfg.setdefault("lifelong", {})
        cfg["lifelong"]["enabled"] = True
        cfg["lifelong"]["forever"] = True
    return cfg


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_section(config_path: str, cfg: Dict[str, Any],
                   cli: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": config_path,
        "file_sha256": sha256_file(config_path),
        "resolved_sha256": hashlib.sha256(
            canonical_json(cfg).encode()).hexdigest(),
        "cli": dict(cli),
        "defaults_note": ("no defaults merging exists: run_minecraft.py "
                          "loads the YAML and overrides seed (and lifelong "
                          "with --forever); absent keys take in-code "
                          ".get() defaults that are not statically "
                          "enumerable — infra/config_echo.py reports keys "
                          "read at runtime"),
        "resolved": cfg,
    }


# ---------------------------------------------------------------------------
# dependencies / hardware / env
# ---------------------------------------------------------------------------

def package_versions(names: Sequence[str] = PACKAGES) -> Dict[str, str]:
    try:
        from importlib import metadata as md
    except ImportError:                      # pragma: no cover (py<3.8)
        return {n: UNKNOWN for n in names}
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = ABSENT
        except Exception:
            out[n] = UNKNOWN
    return out


def _ram_bytes() -> Any:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        if sys.platform == "darwin":
            p = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                return int(p.stdout.strip())
    except Exception:
        pass
    return UNKNOWN


def hardware() -> Dict[str, Any]:
    gpu: Any
    try:
        import torch
        if torch.cuda.is_available():
            gpu = [{"name": torch.cuda.get_device_name(i),
                    "total_bytes": int(
                        torch.cuda.get_device_properties(i).total_memory)}
                   for i in range(torch.cuda.device_count())]
            cuda = torch.version.cuda or UNKNOWN
        else:
            gpu, cuda = ABSENT, ABSENT
    except ImportError:
        gpu, cuda = UNKNOWN, UNKNOWN
    except Exception:
        gpu, cuda = UNKNOWN, UNKNOWN
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or UNKNOWN,
            "hostname": platform.node() or UNKNOWN,
            "cpu_count": os.cpu_count() or UNKNOWN,
            "ram_total_bytes": _ram_bytes(), "gpu": gpu, "cuda": cuda}


def environment_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from .collection_path import select_collection_path
    env = cfg.get("environment") or {}
    par = cfg.get("parallel_envs") or {}
    life = cfg.get("lifelong") or {}
    try:
        d = select_collection_path(cfg)
        path = {"path": d.path, "reason": d.reason}
    except ValueError as e:
        path = {"path": "refused", "reason": str(e)}

    def g(d: Dict, k: str) -> Any:
        # a YAML null is a CONFIGURED value ("no limit"), not a missing one
        if k not in d:
            return ABSENT
        return "null" if d[k] is None else d[k]

    return {
        "env_name": g(env, "name"), "image_size": g(env, "image_size"),
        "render_size": g(env, "render_size"),
        "action_repeat": g(env, "action_repeat"),
        "max_episode_steps": g(env, "max_episode_steps"),
        "remote_server": g(env, "remote_server"),
        "remote_server_scope": g(env, "remote_server_scope"),
        "parallel_envs_enabled": g(par, "enabled"),
        "num_envs": g(par, "num_envs"),
        "allow_legacy_single_env": g(par, "allow_legacy_single_env"),
        "lifelong_enabled": g(life, "enabled"),
        "collection_path": path,
        "sensors_enabled": (cfg.get("sensors") or {}).get("enabled", ABSENT),
        "env_vars": {k: os.environ.get(k, ABSENT) for k in ENV_VARS},
    }


def seeds(cfg: Dict[str, Any], split_salt: Optional[str]) -> Dict[str, Any]:
    s = cfg.get("seed", ABSENT)
    return {"config_seed": ABSENT if s is None else s,
            "python_numpy_torch": ("seeded from config_seed in "
                                   "DevelopmentalAI.__init__"
                                   if s is not None else ABSENT),
            "env_reset_seed": ("first reset seeded from config_seed "
                               "(_seeded_reset); MineRL worlds are fresh per "
                               "mission and the remote server is shared "
                               "state — not reproducible"
                               if s is not None else ABSENT),
            "split_salt": split_salt if split_salt else ABSENT}


def checkpoint_paths(cfg: Dict[str, Any], run_root: str) -> Dict[str, str]:
    loop = cfg.get("loop") or {}
    ck = os.path.join(loop.get("log_dir", "./logs"), "checkpoints")
    paths = {f"checkpoints/{fn}": os.path.join(ck, fn)
             for fn in CHECKPOINT_FILES}
    sb = (cfg.get("skill_bank") or {}).get("storage_dir")
    if sb:
        paths["skill_bank"] = sb
    for sect in ("environment", "curiosity", "infra"):
        bm = (cfg.get(sect) or {}).get("break_memory_path")
        if bm:
            paths["break_memory"] = bm
    return {k: (v if os.path.isabs(v) else os.path.normpath(
        os.path.join(run_root, v))) for k, v in sorted(paths.items())}


def checkpoints(cfg: Dict[str, Any], run_root: str,
                hash_contents: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, p in checkpoint_paths(cfg, run_root).items():
        if os.path.isfile(p):
            rec: Dict[str, Any] = {"path": p, "exists": True,
                                   "bytes": os.path.getsize(p)}
            rec["sha256"] = sha256_file(p) if hash_contents else UNKNOWN
        elif os.path.isdir(p):
            rec = {"path": p, "exists": True}
            if hash_contents:
                rec.update(tree_hash(p))
            else:
                rec.update({"sha256": UNKNOWN, "n_files": UNKNOWN,
                            "bytes": UNKNOWN})
        else:
            rec = {"path": p, "exists": False, "sha256": ABSENT,
                   "bytes": ABSENT}
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# build / compare
# ---------------------------------------------------------------------------

def build_manifest(config_path: str, repo_root: str = ".",
                   seed: Optional[int] = 0, forever: bool = False,
                   timesteps: Any = ABSENT, run_root: Optional[str] = None,
                   hash_checkpoints: bool = True,
                   split_salt: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the manifest. `run_root` is where relative brain paths live
    (the repo root on the host, /workspace/devai)."""
    t0 = time.time()
    repo_root = os.path.abspath(repo_root)
    run_root = os.path.abspath(run_root or repo_root)
    from .splits import SPLIT_SALT, split_spec
    if split_salt is None:
        split_salt = SPLIT_SALT
    cfg = resolve_config(config_path, seed=seed, forever=forever)
    man: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "skybot_run_manifest",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "created_unix": t0,
        "source": source_identity(repo_root),
        "config": config_section(config_path, cfg, {
            "seed": seed, "forever": bool(forever), "timesteps": timesteps}),
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "packages": package_versions(),
        "seeds": seeds(cfg, split_salt),
        "evaluation_split": split_spec(salt=split_salt),
        "checkpoints": checkpoints(cfg, run_root, hash_checkpoints),
        "environment": environment_settings(cfg),
        "hardware": hardware(),
    }
    man["build_seconds"] = round(time.time() - t0, 3)
    validate_manifest(man)
    return man


_REQUIRED = ("schema_version", "kind", "created_at", "source", "config",
             "python", "packages", "seeds", "evaluation_split", "checkpoints",
             "environment", "hardware")


def validate_manifest(man: Dict[str, Any]) -> None:
    if not isinstance(man, dict):
        raise TypeError("manifest must be a dict")
    missing = [k for k in _REQUIRED if k not in man]
    if missing:
        raise ValueError(f"manifest missing required keys: {missing}")
    if man["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema_version "
                         f"{man['schema_version']!r} (this reader: "
                         f"{SCHEMA_VERSION})")

    def walk(x: Any, path: str) -> None:
        if x is None:
            raise ValueError(f"manifest value at {path or '<root>'} is None; "
                             f"use {UNKNOWN!r} or {ABSENT!r}")
        if isinstance(x, dict):
            for k, v in x.items():
                # resolved config may legitimately hold YAML nulls
                if path == "config" and k == "resolved":
                    continue
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
    walk(man, "")


def _flatten(x: Any, prefix: str, out: Dict[str, Any]) -> None:
    if isinstance(x, dict):
        if not x:
            out[prefix] = {}
        for k, v in x.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    else:
        out[prefix] = x


def compare_manifests(a: Dict[str, Any], b: Dict[str, Any],
                      ignore: Iterable[str] = VOLATILE_KEYS) -> List[str]:
    """Sorted dotted key paths whose values differ (or exist on one side).

    `ignore` holds top-level-or-dotted prefixes to skip (default: timestamps
    and build duration).
    """
    fa: Dict[str, Any] = {}
    fb: Dict[str, Any] = {}
    _flatten(a, "", fa)
    _flatten(b, "", fb)
    ign = tuple(ignore)

    def skipped(k: str) -> bool:
        return any(k == p or k.startswith(p + ".") for p in ign)

    _missing = object()
    diff = []
    for k in set(fa) | set(fb):
        if skipped(k):
            continue
        if fa.get(k, _missing) != fb.get(k, _missing):
            diff.append(k)
    return sorted(diff)


def write_manifest(man: Dict[str, Any], path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        man = json.load(f)
    validate_manifest(man)
    return man
