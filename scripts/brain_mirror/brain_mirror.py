#!/usr/bin/env python3
"""brain_mirror.py -- pull SkyBot's brain from the training host to node2.

RUNS ON node2 (the backup machine), not on the training host and not on the
Mac. Python 3.8+ STANDARD LIBRARY ONLY: node2 has no repo venv; this one file
is scp'd there with its env file and README.

WHY THIS EXISTS (CLAUDE.md section 6, "Backups")
    Brain state lives only on the training host's single NVMe. A rented host
    once died with ~11 days of unbacked state; owned hardware swaps that risk
    for disk failure. Nothing automated the copy. This does, every ~5 minutes,
    PULL-based: node2 initiates, node2 hashes, verifies, rotates and prunes.
    The training host only serves file reads, at idle CPU/IO priority.

WHAT IS COPIED (relative to MAIN_ROOT, default /workspace/devai, which is the
launch cwd -- scripts/launch_skybot.sh does `cd /workspace/devai`)
    logs/checkpoints/*.pt + knowledge_graph.json + options_state.json
        developmental_loop._save_checkpoint_locked writes to
        <loop.log_dir>/checkpoints and the live config has loop.log_dir:
        ./logs. NOT ATOMIC: torch.save / open("w") straight onto the final
        name, one file after another -- so a copy can catch a half-written
        file (truncated zip) or two generations side by side.
    runlogs/breaks_by_type.json      env.break_memory_path   (tmp+os.replace)
    runlogs/consequence_state.json   infra.log_dir           (tmp+os.replace)
    runlogs/anticipation_state.json  infra.log_dir           (tmp+os.replace)
    skill_bank_mc_rssm/              skill_bank.storage_dir (live). Registry,
        broadcaster_state.json, ltm_goals index are tmp+replace; skill
        policy .pt files are torch.save'd in place (NOT atomic).
    skill_bank_mc_curiosity/         the previous bank -- "never wipe" (sec 3)
NOT copied: logs/checkpoints/replay_buffer/ (GBs, regenerable experience),
logs, metrics, anything else. In-flight `*.tmp` files are always excluded.

CONSISTENCY DESIGN (because the checkpoint writes are not atomic)
    1. pull into snapshots/<UTC>.partial with --link-dest=<previous good>
       (unchanged files become hardlinks: an unchanged brain costs ~0 bytes);
    2. wait SETTLE_S, then a DRY-RUN rsync of the same file set against the
       partial. Anything it would transfer changed during the copy ->
       re-pull into the same partial, up to MAX_REPULLS times; still moving
       -> keep it, but MANIFEST says "mixed": true;
    3. verify on node2: every .pt that is a zip (torch >= 1.6 format) must
       pass zipfile.testzip; every .json must parse; sha256+size of every
       file goes into MANIFEST.json;
    4. only then: rename .partial -> final and atomically repoint `latest`.
    A failed verification renames to .failed, leaves `latest` alone and
    prunes nothing good.

GUARD ESCAPES (CLAUDE.md 4.1 -- every guard states what re-opens it)
    * lock: fcntl.flock, released by the KERNEL when the holder dies, so a
      lock file left by a dead process never latches; a hung live run is
      bounded by RSYNC_TIMEOUT_S (and systemd TimeoutStartSec).
    * verify failure: a file that fails the SAME way (same sha256) for
      DEGRADED_AFTER consecutive runs is not a mid-save race, it is what is
      on main. The snapshot is then promoted with "degraded": true and the
      bad files listed, exit code 15 -- otherwise one permanently corrupt
      skill file would freeze the mirror forever while the rest of the brain
      kept changing unbacked.
    * legacy (non-zip, pre-1.6) .pt files are reported, not failed: they
      cannot be truncated-zip checked and would otherwise latch the same way.
    * disk guard: re-checked every run; frees itself when space returns.

EXIT CODES
    0 ok | 3 config error | 10 lock held | 11 host unreachable / root missing
    12 verify failed | 13 disk full (pull skipped) | 14 rsync failed
    15 promoted DEGRADED | 16 --status: stale or no snapshot

TEST-ONLY HOOKS (environment; never set them in production)
    BRAIN_MIRROR_TEST_SOURCE   local dir standing in for user@host:MAIN_ROOT
    BRAIN_MIRROR_TEST_HOOK     shell command run after each pull, before the
                               consistency check (simulates a mid-copy save)
    BRAIN_MIRROR_TEST_NOW      epoch seconds used as "now"
    BRAIN_MIRROR_TEST_FREE_GB  pretend free space on BRAIN_DEST
"""
from __future__ import print_function

import argparse
import errno
import fcntl
import hashlib
import json
import logging
import logging.handlers
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_CONFIG = 3
EXIT_LOCK_HELD = 10
EXIT_UNREACHABLE = 11
EXIT_VERIFY_FAILED = 12
EXIT_DISK_FULL = 13
EXIT_RSYNC_FAILED = 14
EXIT_DEGRADED = 15
EXIT_STALE = 16

# Paths relative to MAIN_ROOT. A trailing "/" marks a directory, copied
# recursively. Derived from configs/minecraft_skybot.yaml + the loop's writers
# (see module docstring); override with BRAIN_PATHS if the config moves them.
CHECKPOINT_FILES = (
    "world_model.pt", "curiosity.pt", "policy.pt", "dream_actor.pt",
    "symbolic_decoder.pt", "glue_layer.pt", "symbolizer.pt",
    "familiarity.pt", "magnet.pt", "knowledge_graph.json",
    "options_state.json")
DEFAULT_BRAIN_PATHS = tuple(
    ["logs/checkpoints/" + f for f in CHECKPOINT_FILES]
    + ["runlogs/breaks_by_type.json",
       "runlogs/consequence_state.json",
       "runlogs/anticipation_state.json",
       "skill_bank_mc_rssm/",
       "skill_bank_mc_curiosity/"])
SHADOW_REL = "runlogs/foundation_shadow/"

DEFAULTS = {
    "MAIN_USER": "root",
    "MAIN_SSH_PORT": "22",
    "MAIN_SSH_KEYFILE": "",
    "MAIN_ROOT": "/workspace/devai",
    "KEEP_LAST": "12",
    "KEEP_HOURLY_HOURS": "48",
    "KEEP_DAILY_DAYS": "14",
    "MIN_FREE_GB": "20",
    "INCLUDE_SHADOW": "0",
    "BRAIN_PATHS": "",
    "RSYNC_PATH": "nice -n 19 ionice -c3 rsync",
    "MAX_REPULLS": "3",
    "SETTLE_S": "3",
    "DEGRADED_AFTER": "3",
    "RSYNC_TIMEOUT_S": "1800",
    "IO_TIMEOUT_S": "300",
    "STALE_MINUTES": "20",
}

NAME_RE = re.compile(r"^(\d{8}T\d{6}Z)(?:-(\d+))?(\.partial|\.failed)?$")
STALE_SCRATCH_S = 86400        # .partial / .failed older than this are pruned
MAX_FAILED_KEPT = 10           # ...and never more than this many .failed

log = logging.getLogger("brain_mirror")


class MirrorError(Exception):
    def __init__(self, code, msg):
        Exception.__init__(self, msg)
        self.code = code


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def default_env_path():
    return os.environ.get("BRAIN_MIRROR_ENV") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "brain_mirror.env")


def parse_env_file(path):
    out = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            out[k.strip()] = v
    return out


def is_placeholder(v):
    v = (v or "").strip()
    return (not v or "<" in v or ">" in v
            or v.upper() in ("CHANGE_ME", "CHANGEME", "TODO", "XXX"))


def load_config(env_path):
    """The env file is authoritative; only BRAIN_MIRROR_TEST_* come from the
    process environment (so a stray MAIN_HOST in a shell can't redirect it)."""
    if not os.path.isfile(env_path):
        raise MirrorError(EXIT_CONFIG,
                          "env file not found: %s (copy brain_mirror.env."
                          "example to it and fill in the values)" % env_path)
    raw = dict(DEFAULTS)
    raw.update(parse_env_file(env_path))
    for key in ("MAIN_HOST", "BRAIN_DEST"):
        if is_placeholder(raw.get(key)):
            raise MirrorError(
                EXIT_CONFIG,
                "%s is unset or still a placeholder (%r) in %s -- put the "
                "real value in brain_mirror.env on node2 (gitignored; "
                "never commit addresses)" % (key, raw.get(key, ""),
                                             env_path))
    if is_placeholder(raw.get("MAIN_USER")) or is_placeholder(
            raw.get("MAIN_ROOT")):
        raise MirrorError(EXIT_CONFIG, "MAIN_USER/MAIN_ROOT is a "
                          "placeholder in %s" % env_path)

    def _int(k):
        try:
            return int(float(raw[k]))
        except (TypeError, ValueError):
            raise MirrorError(EXIT_CONFIG, "%s=%r in %s is not a number"
                              % (k, raw[k], env_path))

    def _float(k):
        try:
            return float(raw[k])
        except (TypeError, ValueError):
            raise MirrorError(EXIT_CONFIG, "%s=%r in %s is not a number"
                              % (k, raw[k], env_path))

    paths = raw.get("BRAIN_PATHS", "").split()
    paths = paths or list(DEFAULT_BRAIN_PATHS)
    for p in paths:
        if p.startswith("/") or ".." in p.split("/"):
            raise MirrorError(EXIT_CONFIG, "BRAIN_PATHS entry %r must be "
                              "relative to MAIN_ROOT" % p)
    keyfile = raw.get("MAIN_SSH_KEYFILE", "")
    keyfile = "" if is_placeholder(keyfile) else os.path.expanduser(keyfile)
    cfg = {
        "env_path": os.path.abspath(env_path),
        "main_host": raw["MAIN_HOST"].strip(),
        "main_user": raw["MAIN_USER"].strip(),
        "main_port": _int("MAIN_SSH_PORT"),
        "keyfile": keyfile,
        "main_root": raw["MAIN_ROOT"].rstrip("/") or "/",
        "dest": os.path.abspath(os.path.expanduser(raw["BRAIN_DEST"])),
        "keep_last": _int("KEEP_LAST"),
        "keep_hourly_hours": _float("KEEP_HOURLY_HOURS"),
        "keep_daily_days": _float("KEEP_DAILY_DAYS"),
        "min_free_gb": _float("MIN_FREE_GB"),
        "include_shadow": raw.get("INCLUDE_SHADOW", "0").strip().lower()
        in ("1", "true", "yes", "on"),
        "brain_paths": paths,
        "rsync_path": raw.get("RSYNC_PATH", "").strip() or "rsync",
        "max_repulls": max(0, _int("MAX_REPULLS")),
        "settle_s": max(0.0, _float("SETTLE_S")),
        "degraded_after": max(1, _int("DEGRADED_AFTER")),
        "rsync_timeout_s": max(30, _int("RSYNC_TIMEOUT_S")),
        "io_timeout_s": max(10, _int("IO_TIMEOUT_S")),
        "stale_minutes": _float("STALE_MINUTES"),
        # test hooks
        "test_source": os.environ.get("BRAIN_MIRROR_TEST_SOURCE") or None,
        "test_hook": os.environ.get("BRAIN_MIRROR_TEST_HOOK") or None,
        "test_now": os.environ.get("BRAIN_MIRROR_TEST_NOW") or None,
        "test_free_gb": os.environ.get("BRAIN_MIRROR_TEST_FREE_GB") or None,
    }
    if cfg["keep_last"] < 1:
        raise MirrorError(EXIT_CONFIG, "KEEP_LAST must be >= 1")
    return cfg


def now_ts(cfg):
    return float(cfg["test_now"]) if cfg.get("test_now") else time.time()


def ts_name(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def name_ts(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# command construction (ssh is the default; a local source is test-only)
# ---------------------------------------------------------------------------

def ssh_argv(cfg):
    argv = ["ssh", "-p", str(cfg["main_port"])]
    if cfg["keyfile"]:
        argv += ["-i", cfg["keyfile"]]
    argv += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
             "-o", "StrictHostKeyChecking=accept-new"]
    return argv


def remote_login(cfg):
    return "%s@%s" % (cfg["main_user"], cfg["main_host"])


def source_spec(cfg, rel=""):
    """rsync source for MAIN_ROOT/<rel> (always with a trailing slash)."""
    if cfg.get("test_source"):
        base = os.path.abspath(cfg["test_source"])
    else:
        base = "%s:%s" % (remote_login(cfg), cfg["main_root"])
    return base.rstrip("/") + "/" + rel.strip("/") + ("/" if rel else "")


def rsync_base(cfg):
    # -rlpt, not -a: owner/group are node2's own; -o/-g would only add
    # attribute mismatches that defeat --link-dest. --whole-file and no -z:
    # LAN, so skip delta/compression CPU on the training host. NO --delete,
    # ever: nothing in this tool writes to or removes from the source.
    argv = ["rsync", "-rlpt", "--whole-file",
            "--timeout=%d" % cfg["io_timeout_s"]]
    if not cfg.get("test_source"):
        argv += ["-e", " ".join(shlex.quote(a) for a in ssh_argv(cfg)),
                 "--rsync-path=" + cfg["rsync_path"]]
    return argv


def build_pull_cmd(cfg, dest_dir, files_from, link_dest=None,
                   dry_run=False):
    argv = rsync_base(cfg) + ["--exclude=*.tmp",
                              "--files-from=" + files_from]
    if dry_run:
        argv += ["-n", "--out-format=%i %n"]
    else:
        argv += ["--stats"]
    if link_dest:
        argv += ["--link-dest=" + os.path.abspath(link_dest)]
    argv += [source_spec(cfg), dest_dir.rstrip("/") + "/"]
    return argv


def build_probe_cmd(cfg, paths):
    script = ("cd %s || exit 3; for p in %s; do if [ -e \"$p\" ]; then "
              "printf '%%s\\n' \"$p\"; fi; done"
              % (shlex.quote(cfg["main_root"]),
                 " ".join(shlex.quote(p) for p in paths)))
    if cfg.get("test_source"):
        script = script.replace(shlex.quote(cfg["main_root"]),
                                shlex.quote(os.path.abspath(
                                    cfg["test_source"])), 1)
        return ["sh", "-c", script]
    return ssh_argv(cfg) + [remote_login(cfg), script]


def _run(argv, timeout):
    log.debug("exec: %s", " ".join(shlex.quote(a) for a in argv))
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timed out after %ss" % timeout
    except OSError as e:
        return 127, "", str(e)
    return (p.returncode, p.stdout.decode("utf-8", "replace"),
            p.stderr.decode("utf-8", "replace"))


def probe(cfg):
    """Which brain paths exist on main. Also the reachability check."""
    rc, out, err = _run(build_probe_cmd(cfg, cfg["brain_paths"]), 60)
    if rc == 255 or rc == 124:
        raise MirrorError(EXIT_UNREACHABLE, "cannot reach %s over ssh "
                          "(rc=%d): %s" % (remote_login(cfg), rc,
                                           err.strip()[-400:]))
    if rc == 3:
        raise MirrorError(EXIT_UNREACHABLE, "MAIN_ROOT %s does not exist on "
                          "%s" % (cfg["main_root"], cfg["main_host"]))
    if rc != 0:
        raise MirrorError(EXIT_UNREACHABLE, "probe failed rc=%d: %s"
                          % (rc, err.strip()[-400:]))
    present = set(l.strip() for l in out.splitlines() if l.strip())
    return [p for p in cfg["brain_paths"] if p in present]


def _rsync(cfg, argv, what):
    rc, out, err = _run(argv, cfg["rsync_timeout_s"])
    # 24 = "some source files vanished" -- normal under a live writer; the
    # consistency check that follows decides whether that matters.
    if rc in (0, 24):
        return out
    code = EXIT_UNREACHABLE if rc in (255, 124) else EXIT_RSYNC_FAILED
    raise MirrorError(code, "%s: rsync rc=%d: %s" % (what, rc,
                                                     err.strip()[-600:]))


def changed_files(dry_out):
    """Regular files a dry run would transfer ('>f...', 'cf...', 'hf...').
    Works for GNU (11-char) and openrsync (9-char) itemize strings."""
    out = []
    for line in dry_out.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and len(parts[0]) >= 2 and parts[0][1] == "f" \
                and parts[0][0] in "<>ch":
            out.append(parts[1].strip())
    return out


# ---------------------------------------------------------------------------
# snapshot directory bookkeeping
# ---------------------------------------------------------------------------

def snap_root(cfg):
    return os.path.join(cfg["dest"], "snapshots")


def list_snaps(cfg):
    """(good, partial, failed): lists of (name, ts), oldest first."""
    good, partial, failed = [], [], []
    d = snap_root(cfg)
    if not os.path.isdir(d):
        return good, partial, failed
    for n in os.listdir(d):
        m = NAME_RE.match(n)
        if not m or not os.path.isdir(os.path.join(d, n)):
            continue
        t = name_ts(n)
        {None: good, ".partial": partial, ".failed": failed}[
            m.group(3)].append((n, t))
    key = lambda e: (e[1], e[0])
    return sorted(good, key=key), sorted(partial, key=key), sorted(
        failed, key=key)


def latest_name(cfg):
    link = os.path.join(cfg["dest"], "latest")
    if not os.path.islink(link):
        return None
    name = os.path.basename(os.readlink(link).rstrip("/"))
    if os.path.isdir(os.path.join(snap_root(cfg), name)):
        return name
    return None


def newest_good(cfg):
    good = list_snaps(cfg)[0]
    return good[-1][0] if good else None


def repoint_latest(cfg, name):
    link = os.path.join(cfg["dest"], "latest")
    tmp = link + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(os.path.join("snapshots", name), tmp)
    os.replace(tmp, link)                       # atomic on POSIX
    _fsync_dir(cfg["dest"])


def _fsync_dir(d):
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _rmtree(path):
    def _onerr(func, p, _exc):
        try:
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
        except OSError:
            pass
        func(p)
    shutil.rmtree(path, onerror=_onerr)


def unique_name(cfg, ts):
    base = ts_name(ts)
    existing = set(os.listdir(snap_root(cfg)))
    name, i = base, 0
    while any(x in existing for x in (name, name + ".partial",
                                      name + ".failed")):
        i += 1
        name = "%s-%d" % (base, i)
    return name


# ---------------------------------------------------------------------------
# retention (pure; tested against a synthetic timeline)
# ---------------------------------------------------------------------------

def select_keep(entries, now, keep_last=12, hourly_hours=48.0,
                daily_days=14.0, protect=()):
    """entries: [(name, ts)]. Returns the set of names to KEEP.

    keep_last newest; the newest snapshot of each UTC hour younger than
    hourly_hours; the newest of each UTC day younger than daily_days; plus
    every name in `protect` (latest, newest verified)."""
    newest_first = sorted(entries, key=lambda e: (e[1], e[0]), reverse=True)
    keep = set(n for n, _ in newest_first[:max(1, keep_last)])
    seen_h, seen_d = set(), set()
    for n, t in newest_first:
        age = now - t
        if age <= hourly_hours * 3600.0:
            b = int(t // 3600)
            if b not in seen_h:
                seen_h.add(b)
                keep.add(n)
        if age <= daily_days * 86400.0:
            b = int(t // 86400)
            if b not in seen_d:
                seen_d.add(b)
                keep.add(n)
    keep.update(p for p in protect if p)
    return keep


def newest_clean(cfg):
    """Newest promoted snapshot that is NOT degraded. Degraded promotion is
    the latch escape for a file that is corrupt ON MAIN; while that lasts,
    every new snapshot is degraded, and without this the last fully clean
    copy would age out of retention (14 days) and be pruned."""
    for name, _t in reversed(list_snaps(cfg)[0]):
        m = _load_manifest(os.path.join(snap_root(cfg), name))
        if m is not None and not m.get("degraded"):
            return name
    return None


def protected(cfg):
    return set(x for x in (latest_name(cfg), newest_good(cfg),
                           newest_clean(cfg)) if x)


def prune_retention(cfg, now):
    good = list_snaps(cfg)[0]
    keep = select_keep(good, now, cfg["keep_last"], cfg["keep_hourly_hours"],
                       cfg["keep_daily_days"], protected(cfg))
    removed = []
    for n, _ in good:
        if n not in keep:
            _rmtree(os.path.join(snap_root(cfg), n))
            removed.append(n)
    if removed:
        log.info("retention: pruned %d snapshot(s): %s", len(removed),
                 ", ".join(removed))
    return removed


def prune_scratch(cfg, now):
    _, partial, failed = list_snaps(cfg)
    removed = []
    for n, t in partial + failed:
        if now - t > STALE_SCRATCH_S:
            removed.append(n)
    extra = [n for n, _ in failed if n not in removed]
    removed += extra[:max(0, len(extra) - MAX_FAILED_KEPT)]
    for n in removed:
        _rmtree(os.path.join(snap_root(cfg), n))
    if removed:
        log.info("pruned stale scratch: %s", ", ".join(removed))
    return removed


def free_gb(cfg):
    if cfg.get("test_free_gb") is not None:
        return float(cfg["test_free_gb"])
    return shutil.disk_usage(cfg["dest"]).free / 1e9


def ensure_free(cfg):
    """Disk guard. Prunes scratch, then the oldest unprotected snapshots,
    one at a time, until MIN_FREE_GB is met. Still short -> skip the pull.
    Re-checked every run, so it reopens by itself once space returns."""
    if free_gb(cfg) >= cfg["min_free_gb"]:
        return
    good, partial, failed = list_snaps(cfg)
    prot = protected(cfg)
    order = [n for n, _ in partial + failed] + [n for n, _ in good
                                                 if n not in prot]
    for n in order:
        log.warning("DISK GUARD: %.1f GB free < MIN_FREE_GB=%.1f -- "
                    "pruning %s", free_gb(cfg), cfg["min_free_gb"], n)
        _rmtree(os.path.join(snap_root(cfg), n))
        if free_gb(cfg) >= cfg["min_free_gb"]:
            return
    raise MirrorError(EXIT_DISK_FULL, "DISK FULL: %.1f GB free on %s < "
                      "MIN_FREE_GB=%.1f even after pruning every eligible "
                      "snapshot -- pull SKIPPED (latest kept). Free space "
                      "on node2 or lower MIN_FREE_GB."
                      % (free_gb(cfg), cfg["dest"], cfg["min_free_gb"]))


# ---------------------------------------------------------------------------
# verification + manifest
# ---------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_file(path, rel):
    """(status, detail). status in ok | legacy | bad."""
    if rel.endswith(".pt"):
        if os.path.getsize(path) == 0:
            return "bad", "empty .pt"
        with open(path, "rb") as f:
            head = f.read(4)
        if head != b"PK\x03\x04":
            return "legacy", "not a zip (pre-1.6 torch format?)"
        try:
            with zipfile.ZipFile(path) as z:
                badname = z.testzip()
        except (zipfile.BadZipFile, OSError, EOFError, ValueError) as e:
            return "bad", "invalid zip: %s" % e
        if badname is not None:
            return "bad", "CRC error in member %s" % badname
        return "ok", ""
    if rel.endswith(".json"):
        try:
            with open(path, "rb") as f:
                json.loads(f.read().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            return "bad", "json does not parse: %s" % e
    return "ok", ""


def _load_manifest(snap_dir):
    try:
        with open(os.path.join(snap_dir, "MANIFEST.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def verify_tree(snap_dir, prev_dir=None, prev_manifest=None):
    """Hash + check every file. A file that is a HARDLINK of the same path in
    the previous verified snapshot (same inode) is byte-identical by
    construction -- nothing ever modifies a snapshot in place -- so its entry
    is reused instead of re-reading it from the HDD every 5 minutes."""
    prev_files = (prev_manifest or {}).get("files", {}) if prev_dir else {}
    files, bad, legacy = {}, {}, []
    reused, new_bytes = 0, 0
    for d, dirs, fns in os.walk(snap_dir):
        dirs.sort()
        for fn in sorted(fns):
            p = os.path.join(d, fn)
            rel = os.path.relpath(p, snap_dir).replace(os.sep, "/")
            if rel == "MANIFEST.json":
                continue
            st = os.lstat(p)
            if os.path.islink(p):
                files[rel] = {"symlink": os.readlink(p)}
                continue
            pe = prev_files.get(rel)
            if pe and "sha256" in pe and not pe.get("corrupt"):
                try:
                    pst = os.stat(os.path.join(prev_dir, rel))
                    if (pst.st_dev, pst.st_ino) == (st.st_dev, st.st_ino):
                        files[rel] = dict(pe)
                        reused += 1
                        if pe.get("legacy"):
                            legacy.append(rel)
                        continue
                except OSError:
                    pass
            new_bytes += st.st_size
            ent = {"size": st.st_size, "sha256": sha256_file(p)}
            status, detail = check_file(p, rel)
            if status == "bad":
                ent["corrupt"] = detail
                bad[rel] = ent["sha256"]
            elif status == "legacy":
                ent["legacy"] = True
                legacy.append(rel)
            files[rel] = ent
    return files, bad, legacy, reused, new_bytes


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _verify_state_path(cfg):
    return os.path.join(cfg["dest"], ".verify_state.json")


def note_verify_failure(cfg, bad):
    """Consecutive-identical-failure counter (the latch escape)."""
    path = _verify_state_path(cfg)
    try:
        with open(path) as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    if st.get("bad") == bad:
        st["count"] = int(st.get("count", 0)) + 1
    else:
        st = {"bad": bad, "count": 1}
    write_json_atomic(path, st)
    return st["count"]


def clear_verify_state(cfg):
    try:
        os.remove(_verify_state_path(cfg))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

class Lock(object):
    """fcntl.flock: held only while this process lives. A lock FILE left by a
    dead run is harmless (the kernel dropped the lock with the process), so
    it can never latch. The fd is non-inheritable (PEP 446) and subprocess
    closes fds, so an rsync child cannot keep it alive either."""

    def __init__(self, path):
        self.path = path
        self.f = None

    def acquire(self):
        f = open(self.path, "a+")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError) as e:
            if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise
            f.seek(0)
            holder = f.read().strip()
            f.close()
            raise MirrorError(EXIT_LOCK_HELD, "another brain_mirror run holds "
                              "%s (%s) -- skipped" % (self.path,
                                                      holder or "?"))
        f.seek(0)
        prior = f.read().strip()
        if prior:
            # release() empties the file, so a leftover record means its
            # writer died mid-run. The kernel already dropped its flock.
            log.warning("recovered stale lock %s (record %r, holder not "
                        "running)", self.path, prior)
        f.seek(0)
        f.truncate()
        f.write("%d %s\n" % (os.getpid(), ts_name(time.time())))
        f.flush()
        self.f = f

    def release(self):
        if self.f is not None:
            try:
                self.f.seek(0)
                self.f.truncate()
                fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            finally:
                self.f.close()
                self.f = None


def _stats_lines(out):
    keep = ("Number of", "Total", "Literal data", "Matched data",
            "Unmatched data", "sent ", "total size")
    return [l.strip() for l in out.splitlines()
            if l.strip().startswith(keep)]


def _run_hook(cfg):
    if cfg.get("test_hook"):
        subprocess.run(cfg["test_hook"], shell=True, check=False)


def mirror_once(cfg):
    t0 = time.time()
    now = now_ts(cfg)
    os.makedirs(snap_root(cfg), exist_ok=True)
    prune_scratch(cfg, now)
    ensure_free(cfg)

    present = probe(cfg)
    if not present:
        raise MirrorError(EXIT_VERIFY_FAILED, "none of the %d brain paths "
                          "exist under %s on %s -- nothing to mirror (wrong "
                          "MAIN_ROOT? fresh host?)"
                          % (len(cfg["brain_paths"]), cfg["main_root"],
                             cfg["main_host"]))
    missing = [p for p in cfg["brain_paths"] if p not in present]
    if missing:
        log.info("not present on main (skipped): %s", " ".join(missing))
    files_from = os.path.join(cfg["dest"], ".files-from")
    with open(files_from, "w") as f:
        f.write("".join(p + "\n" for p in present))

    prev = latest_name(cfg) or newest_good(cfg)
    prev_dir = os.path.join(snap_root(cfg), prev) if prev else None
    name = unique_name(cfg, now)
    partial = os.path.join(snap_root(cfg), name + ".partial")
    os.makedirs(partial)

    stats, changed_log, pulls, mixed = [], [], 0, False
    try:
        while True:
            out = _rsync(cfg, build_pull_cmd(cfg, partial, files_from,
                                             link_dest=prev_dir), "pull")
            pulls += 1
            stats = _stats_lines(out)
            _run_hook(cfg)
            if cfg["settle_s"] > 0:
                time.sleep(cfg["settle_s"])
            dry = _rsync(cfg, build_pull_cmd(cfg, partial, files_from,
                                             dry_run=True), "consistency")
            changed = changed_files(dry)
            if not changed:
                break
            changed_log.append(changed)
            log.info("changed during copy (pull %d): %s", pulls,
                     " ".join(changed[:20]))
            if pulls > cfg["max_repulls"]:
                mixed = True
                log.warning("brain still changing after %d re-pulls -- "
                            "keeping snapshot marked mixed: true", pulls - 1)
                break
    except MirrorError:
        _rmtree(partial)        # our own scratch; the source is untouched
        raise

    prev_manifest = _load_manifest(prev_dir) if prev_dir else None
    files, bad, legacy, reused, new_bytes = verify_tree(
        partial, prev_dir, prev_manifest)
    total = sum(e.get("size", 0) for e in files.values())
    manifest = {
        "schema": SCHEMA_VERSION,
        "snapshot": name,
        "created_utc": ts_name(now),
        "created_unix": now,
        "source": {"host": cfg["main_host"], "user": cfg["main_user"],
                   "root": cfg["main_root"],
                   "local_test_source": bool(cfg.get("test_source"))},
        "brain_paths_present": present,
        "brain_paths_missing": missing,
        "previous": prev,
        "mixed": mixed,
        "pulls": pulls,
        "changed_during_copy": changed_log,
        "rsync_stats": stats,
        "n_files": len(files),
        "total_bytes": total,
        "new_bytes": new_bytes,
        "hash_reused": reused,
        "legacy_pt": legacy,
        "corrupt": sorted(bad),
        "degraded": False,
        "verified": not bad,
        "files": files,
    }
    if prev_manifest and prev_manifest.get("n_files", 0) > 0 \
            and len(files) < 0.5 * prev_manifest["n_files"]:
        manifest["shrunk_from"] = prev_manifest["n_files"]
        log.warning("SNAPSHOT SHRANK: %d files vs %d in %s -- check main "
                    "(the older snapshots are kept by retention)",
                    len(files), prev_manifest["n_files"], prev)
    if legacy:
        log.warning("non-zip .pt files (not truncation-checkable): %s",
                    " ".join(legacy[:10]))

    code = EXIT_OK
    if not files:
        bad = {"<empty>": ""}
    if bad:
        count = note_verify_failure(cfg, bad)
        stable = bool(files) and count >= cfg["degraded_after"]
        detail = "; ".join("%s (%s)" % (r, files.get(r, {}).get("corrupt",
                                                                "no files"))
                           for r in sorted(bad))
        if not stable:
            manifest["duration_s"] = round(time.time() - t0, 3)
            write_json_atomic(os.path.join(partial, "MANIFEST.json"),
                              manifest)
            os.rename(partial, os.path.join(snap_root(cfg),
                                            name + ".failed"))
            raise MirrorError(
                EXIT_VERIFY_FAILED,
                "VERIFY FAILED (%d/%d identical failures%s): %s -- kept as "
                "%s.failed, latest unchanged (%s), nothing pruned"
                % (count, cfg["degraded_after"],
                   "; source was changing" if mixed or changed_log else "",
                   detail, name, latest_name(cfg)))
        manifest["degraded"] = True
        log.error("PROMOTING DEGRADED snapshot: the same file(s) failed "
                  "verification %d runs in a row with identical bytes, so "
                  "this is what is on main, not a mid-save race: %s",
                  count, detail)
        code = EXIT_DEGRADED
    else:
        clear_verify_state(cfg)

    manifest["duration_s"] = round(time.time() - t0, 3)
    write_json_atomic(os.path.join(partial, "MANIFEST.json"), manifest)
    final = os.path.join(snap_root(cfg), name)
    os.rename(partial, final)
    _fsync_dir(snap_root(cfg))
    repoint_latest(cfg, name)
    prune_retention(cfg, now)
    if cfg["include_shadow"]:
        mirror_shadow(cfg)
    return code, manifest


def mirror_shadow(cfg):
    """Single rolling copy of runlogs/foundation_shadow/ (append-only logs).
    Best-effort: a failure here never fails the brain snapshot."""
    dst = os.path.join(cfg["dest"], "shadow_mirror")
    os.makedirs(dst, exist_ok=True)
    argv = rsync_base(cfg) + ["--exclude=*.tmp", source_spec(cfg, SHADOW_REL),
                              dst + "/"]
    rc, _out, err = _run(argv, cfg["rsync_timeout_s"])
    if rc not in (0, 24):
        log.warning("shadow mirror failed rc=%d: %s", rc, err.strip()[-300:])


# ---------------------------------------------------------------------------
# --status / --restore-plan
# ---------------------------------------------------------------------------

def disk_use_bytes(path):
    seen, total = set(), 0
    for d, _dirs, fns in os.walk(path):
        for fn in fns:
            try:
                st = os.lstat(os.path.join(d, fn))
            except OSError:
                continue
            k = (st.st_dev, st.st_ino)
            if k not in seen:
                seen.add(k)
                total += st.st_size
    return total


def status(cfg):
    good, partial, failed = list_snaps(cfg)
    lat = latest_name(cfg)
    now = now_ts(cfg)
    lines = []
    if lat:
        m = _load_manifest(os.path.join(snap_root(cfg), lat)) or {}
        age = now - (name_ts(lat) or 0)
        lines.append("latest: %s  age %.1f min  files %s  %.1f MB  mixed=%s "
                     "degraded=%s" % (lat, age / 60.0, m.get("n_files", "?"),
                                      m.get("total_bytes", 0) / 1e6,
                                      m.get("mixed"), m.get("degraded")))
    else:
        age = None
        lines.append("latest: NONE")
    lines.append("snapshots: %d good, %d partial, %d failed"
                 % (len(good), len(partial), len(failed)))
    if os.path.isdir(cfg["dest"]):
        lines.append("disk: %.2f GB used by BRAIN_DEST (hardlinks counted "
                     "once), %.1f GB free (MIN_FREE_GB=%.1f)"
                     % (disk_use_bytes(cfg["dest"]) / 1e9, free_gb(cfg),
                        cfg["min_free_gb"]))
    vs = _verify_state_path(cfg)
    if os.path.exists(vs):
        try:
            with open(vs) as f:
                lines.append("PENDING VERIFY FAILURES: %s" % f.read().strip())
        except OSError:
            pass
    stale = age is None or age > cfg["stale_minutes"] * 60.0
    lines.append("state: %s" % ("STALE (older than %g min or none)"
                                % cfg["stale_minutes"] if stale else "OK"))
    print("\n".join(lines))
    return EXIT_STALE if stale else EXIT_OK


def resolve_snapshot(cfg, which):
    if which in ("latest", ""):
        n = latest_name(cfg)
        if not n:
            raise MirrorError(EXIT_CONFIG, "no latest snapshot")
        return os.path.join(snap_root(cfg), n)
    cand = which if os.path.isdir(which) else os.path.join(snap_root(cfg),
                                                           which)
    if not os.path.isdir(cand) or not NAME_RE.match(
            os.path.basename(cand.rstrip("/"))):
        raise MirrorError(EXIT_CONFIG, "no such snapshot: %s" % which)
    return os.path.abspath(cand)


def restore_plan(cfg, which):
    """PRINTS a restore procedure. Executes nothing -- by design."""
    snap = resolve_snapshot(cfg, which)
    m = _load_manifest(snap) or {}
    sh = " ".join(shlex.quote(a) for a in ssh_argv(cfg))
    login = remote_login(cfg)
    root = cfg["main_root"]
    q = shlex.quote
    rs_e = q(sh)
    stamp = "$(date -u +%Y%m%dT%H%M%SZ)"
    paths = " ".join(q(p.rstrip("/")) for p in m.get(
        "brain_paths_present", cfg["brain_paths"]))
    out = [
        "# RESTORE PLAN -- snapshot %s" % os.path.basename(snap),
        "#   created %s  files %s  mixed=%s  degraded=%s  corrupt=%s"
        % (m.get("created_utc", "?"), m.get("n_files", "?"), m.get("mixed"),
           m.get("degraded"), m.get("corrupt", [])),
        "# NOTHING HAS BEEN EXECUTED. Read every step and run them by hand.",
    ]
    if m.get("mixed") or m.get("degraded"):
        out.append("# WARNING: this snapshot is %s. Prefer an earlier "
                   "snapshot with mixed=false degraded=false if one exists."
                   % ("MIXED (may combine two checkpoint generations)"
                      if m.get("mixed") else "DEGRADED (corrupt files)"))
    out += [
        "#",
        "# LAYOUT MUST MATCH (CLAUDE.md sec 6): the snapshot mirrors %s"
        % root,
        "#   logs/checkpoints/  = <loop.log_dir>/checkpoints   (loop.log_dir: "
        "./logs)",
        "#   runlogs/breaks_by_type.json = environment.break_memory_path",
        "#   skill_bank_*/      = skill_bank.storage_dir",
        "# If configs/minecraft_skybot.yaml has moved any of these since "
        "the snapshot,",
        "# move the files to match the CONFIG before relaunch. A break "
        "memory restored",
        "# to the wrong path starts EMPTY silently and re-opens every "
        "mastered block",
        "# tier at full worth. A pre-rename backup with podlogs/ needs "
        "`mv podlogs runlogs` FIRST.",
        "",
        "# 1. Stop training GRACEFULLY. Never `kill` it: the supervisor "
        "treats that as a",
        "#    crash and relaunches, and the relaunched run's next "
        "checkpoint overwrites the",
        "#    restore. A graceful stop also WRITES a checkpoint (and the "
        "replay buffer,",
        "#    ~10 min) -- so wait for the process to exit before step 3.",
        "ssh %s %s %s" % (sh, login, q("touch %s/runlogs/STOP" % root)),
        "ssh %s %s %s" % (sh, login, q(
            "while pgrep -f 'run_minecraft[.]py' >/dev/null; do sleep 10; "
            "done; echo STOPPED")),
        "",
        "# 2. Keep what is there now, in case the restore is the mistake:",
        "ssh %s %s %s" % (sh, login, q(
            "cd %s && tar czf /workspace/brain_before_restore_%s.tgz "
            "--ignore-failed-read %s" % (root, stamp, paths))),
        "",
        "# 3. DRY RUN the push and read what it would change:",
        "rsync -rlpt --whole-file -n -i --exclude=/MANIFEST.json -e %s %s/ "
        "%s:%s/" % (rs_e, q(snap), login, root),
        "",
        "# 4. Push. NO --delete: the destination is the whole repo tree "
        "(code, configs,",
        "#    replay buffer); --delete would remove all of it. Skills minted "
        "after this",
        "#    snapshot stay on disk but its registry.json will not list "
        "them.",
        "rsync -rlpt --whole-file --exclude=/MANIFEST.json -e %s %s/ %s:%s/"
        % (rs_e, q(snap), login, root),
        "",
        "# 5. Re-allow training and relaunch the usual way "
        "(scripts/launch_skybot.sh / deploy):",
        "ssh %s %s %s" % (sh, login, q("rm -f %s/runlogs/STOP" % root)),
    ]
    print("\n".join(out))
    return EXIT_OK


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def setup_logging(cfg, verbose=False):
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.handlers[:] = []
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            "%Y-%m-%dT%H:%M:%S")
    try:
        os.makedirs(cfg["dest"], exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(cfg["dest"], "mirror.log"), maxBytes=1 << 20,
            backupCount=3)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError as e:
        print("brain_mirror: cannot open mirror.log: %s" % e, file=sys.stderr)
    eh = logging.StreamHandler(sys.stderr)
    eh.setLevel(logging.WARNING)
    eh.setFormatter(fmt)
    log.addHandler(eh)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Pull SkyBot's brain from the training host (run on "
                    "node2).")
    ap.add_argument("--env", default=default_env_path(),
                    help="env file (default: brain_mirror.env beside this "
                         "script, or $BRAIN_MIRROR_ENV)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true",
                   help="print last success age, snapshot count, disk use")
    g.add_argument("--restore-plan", metavar="SNAPSHOT",
                   help="PRINT (never run) the commands to push SNAPSHOT "
                        "('latest', a name, or a path) back to main")
    g.add_argument("--check-config", action="store_true",
                   help="validate the env file and print the resolved config")
    g.add_argument("--probe", action="store_true",
                   help="ssh to main and list which brain paths exist")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    try:
        cfg = load_config(a.env)
    except MirrorError as e:
        print("brain_mirror: CONFIG ERROR: %s" % e, file=sys.stderr)
        return e.code
    if a.check_config:
        print(json.dumps({k: v for k, v in cfg.items()
                          if not k.startswith("test_")}, indent=1))
        return EXIT_OK
    if a.restore_plan is not None:
        try:
            return restore_plan(cfg, a.restore_plan)
        except MirrorError as e:
            print("brain_mirror: %s" % e, file=sys.stderr)
            return e.code
    if a.status:
        return status(cfg)

    setup_logging(cfg, a.verbose)
    if a.probe:
        try:
            present = probe(cfg)
        except MirrorError as e:
            print("brain_mirror: %s" % e, file=sys.stderr)
            return e.code
        print("reachable: %s:%s -- present: %s" % (
            remote_login(cfg), cfg["main_root"], " ".join(present) or
            "(none)"))
        return EXIT_OK

    lock = Lock(os.path.join(cfg["dest"], ".brain_mirror.lock"))
    t0 = time.time()
    try:
        lock.acquire()
        code, m = mirror_once(cfg)
    except MirrorError as e:
        lvl = logging.INFO if e.code == EXIT_LOCK_HELD else logging.ERROR
        log.log(lvl, "%s", e)
        print("brain_mirror FAIL code=%d: %s" % (e.code, e))
        return e.code
    finally:
        lock.release()
    line = ("brain_mirror %s snapshot=%s files=%d total=%.1fMB new=%.1fMB "
            "reused=%d pulls=%d mixed=%s dur=%.1fs"
            % ("DEGRADED" if code == EXIT_DEGRADED else "OK", m["snapshot"],
               m["n_files"], m["total_bytes"] / 1e6, m["new_bytes"] / 1e6,
               m["hash_reused"], m["pulls"], m["mixed"], time.time() - t0))
    log.info("%s", line)
    print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
