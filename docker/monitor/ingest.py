"""JSONL -> SQLite for the SkyBot dashboard.

THE RAW JSONL IS THE SYSTEM OF RECORD; THIS DB IS A VIEW.
    That is the whole safety argument for choosing SQLite over a log-native
    store: an ingest bug can make the DB disagree with the log, so the DB must
    be reconstructible. Every insert is `INSERT OR IGNORE` on a UNIQUE `seq`,
    which makes a full re-read of the file idempotent — delete skybot.db,
    re-run, and you get a byte-equivalent table back. The verification step
    for this feature is exactly that.

WHY seq AND NOT rowid/timestamp
    The collector runs a live `tail -F` AND a periodic rsync backfill, so the
    same record legitimately arrives twice. `seq` is assigned by the host-side
    sink, is strictly increasing, and survives a restart (it is persisted in a
    sidecar precisely so a rotation cannot rewind it). Two records with the
    same seq are the same record.

NESTED MAPS STAY JSON
    `reward_shares`, `breaks_by_type` and `crafts_by_item` are open-ended —
    a new reward source or a newly-broken block type appears without warning.
    Storing them as JSON columns and reading with json_extract() in Grafana
    means adding a reward source never requires a schema migration.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

DB = os.environ.get("DB_PATH", "/data/skybot.db")
SRC = os.environ.get("METRICS_PATH", "/data/metrics.jsonl")
SERVER_SRC = os.environ.get("SERVER_PATH", "/data/server.jsonl")
HB_SRC = os.environ.get("HEARTBEAT_PATH", "/data/heartbeat.jsonl")
POLL = float(os.environ.get("POLL_SECONDS", "10"))
# The evidence mirror (collect_metrics.sh `mirror_pull`) drops these FLAT
# beside metrics.jsonl, so they default to METRICS_PATH's directory: one env
# var moves the whole set, and docker-compose.yml needs no new keys.
DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(SRC) or ".")
LEARNING_SRC = os.environ.get("LEARNING_PATH",
                              os.path.join(DATA_DIR, "learning.jsonl"))
POSITIONS_SRC = os.environ.get("POSITIONS_PATH",
                               os.path.join(DATA_DIR, "position_trace.jsonl"))
CENSUS_SRC = os.environ.get("MEMORY_CENSUS_PATH",
                            os.path.join(DATA_DIR, "memory_census.jsonl"))
INCIDENTS_DIR = os.environ.get("INCIDENTS_DIR",
                               os.path.join(DATA_DIR, "incidents"))

# Scalar columns worth having as real columns (Grafana graphs them directly).
# Anything not listed still survives — see the `extra` column.
SCALARS = [
    ("wall_time", "REAL"), ("total_timesteps", "INTEGER"),
    ("episode", "INTEGER"), ("uptime_s", "REAL"),
    ("reward_total", "REAL"), ("reward_hhi", "REAL"),
    ("episode_reward", "REAL"), ("intrinsic_reward", "REAL"),
    ("curiosity", "REAL"), ("policy_entropy", "REAL"),
    ("world_model_loss", "REAL"), ("episode_length", "REAL"),
    ("breaks_total", "INTEGER"), ("logs", "INTEGER"),
    ("breaks_per_log", "REAL"), ("crafts_total", "INTEGER"),
    ("attack_run_max", "REAL"), ("attack_run_mean", "REAL"),
    ("attack_runs", "INTEGER"),
    ("skills_bound", "INTEGER"), ("skills_minted", "INTEGER"),
    ("skills_refused", "INTEGER"), ("rssm_no_latent", "INTEGER"),
    ("cond_live", "INTEGER"), ("cond_load_failures", "INTEGER"),
    # Learning signals. These come from training_metrics deques, not from the
    # segment dict — see the note in _emit_metrics. Listed as real columns so
    # Grafana can graph them; without this they fall into `extra` as JSON and
    # the "Learning signals" panel has nothing to plot.
    ("kl_divergence", "REAL"), ("symbolic_decoder_loss", "REAL"),
    ("symbolic_decoder_accuracy", "REAL"), ("inverse_dynamics_loss", "REAL"),
    ("dream_distill_eff_weight", "REAL"), ("glue_mastery_score", "REAL"),
    ("glue_kg_density", "REAL"),
    ("skills_mastered", "INTEGER"), ("skills_composite", "INTEGER"),
    ("avg_success_rate", "REAL"),
]
JSON_COLS = ["reward_shares", "reward_alarms", "breaks_by_type",
             "crafts_by_type"]


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    cx = sqlite3.connect(DB)
    # WAL so Grafana can read while this process writes. Without it a reader
    # blocks the writer and the dashboard makes ingest stutter.
    cx.execute("PRAGMA journal_mode=WAL")
    cols = ", ".join(f"{n} {t}" for n, t in SCALARS)
    jcols = ", ".join(f"{n} TEXT" for n in JSON_COLS)
    cx.execute(
        f"CREATE TABLE IF NOT EXISTS segments ("
        f"seq INTEGER PRIMARY KEY, schema_version INTEGER, "
        f"{cols}, {jcols}, extra TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS server ("
               "wall_time REAL PRIMARY KEY, players_online INTEGER, "
               "max_players INTEGER, version TEXT)")
    # SEPARATE TABLE from `segments`, not a `kind` column on it. The two feeds
    # have different cadences (~15s vs ~7min) and different meanings: a
    # heartbeat is a running total sampled mid-segment, a segment row is a
    # closed observation. Mixing them would make every segment query need a
    # filter, and one forgotten filter would silently average the two.
    cx.execute("CREATE TABLE IF NOT EXISTS heartbeat ("
               "seq INTEGER PRIMARY KEY, wall_time REAL, "
               "total_timesteps INTEGER, uptime_s REAL, steps_per_s REAL, "
               "breaks_total INTEGER, logs INTEGER, attack_run REAL, "
               "episodes INTEGER, gpu_mem_mb REAL, income_now TEXT)")

    # ---- ADD-ONLY MIGRATION -------------------------------------------
    # `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a DB
    # created before a column was added keeps the OLD shape and every insert
    # then fails with "table segments has no column named ...". The tracker
    # would go quiet with the containers still healthy — the exact
    # looks-alive-stores-nothing failure this whole design guards against.
    #
    # Metrics are append-only and every column is nullable, so adding is
    # always safe. Nothing is ever dropped or renamed here: a removed field
    # simply stops being written, and old rows keep their history.
    have = {r[1] for r in cx.execute("PRAGMA table_info(segments)")}
    # `extra` belongs in this list too — it is a real column the INSERT names.
    # Omitting it made the migration add 34 columns and then fail every insert
    # with "table segments has no column named extra": the DB looked migrated
    # and stored nothing new.
    for name, typ in (SCALARS + [(n, "TEXT") for n in JSON_COLS]
                      + [("extra", "TEXT")]):
        if name not in have:
            cx.execute(f"ALTER TABLE segments ADD COLUMN {name} {typ}")
            print(f"migrate: added column {name} {typ}", flush=True)
    # SAME ADD-ONLY MIGRATION FOR `heartbeat` (2026-09-28). The block above
    # covers `segments` only. Adding heartbeat.income_now to a DB created
    # before it exists makes EVERY heartbeat insert fail with "table
    # heartbeat has no column named income_now" -- containers healthy, panels
    # frozen, nothing in any log. Precisely the failure the note above
    # describes, one table over.
    have_hb = {r[1] for r in cx.execute("PRAGMA table_info(heartbeat)")}
    for name, typ in [("income_now", "TEXT")]:
        if name not in have_hb:
            cx.execute(f"ALTER TABLE heartbeat ADD COLUMN {name} {typ}")
            print(f"migrate: added heartbeat.{name} {typ}", flush=True)
    connect_telemetry(cx)
    cx.commit()
    return cx


def ingest_segments(cx: sqlite3.Connection, path: str) -> int:
    if not os.path.exists(path):
        return 0
    names = ["seq", "schema_version"] + [n for n, _ in SCALARS] + JSON_COLS \
        + ["extra"]
    sql = (f"INSERT OR IGNORE INTO segments ({','.join(names)}) "
           f"VALUES ({','.join('?' * len(names))})")
    known = {"seq", "schema_version"} | {n for n, _ in SCALARS} \
        | set(JSON_COLS)
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                # A partially-written last line is normal while tail is
                # appending. Skip it; the next pass picks it up whole.
                continue
            if not isinstance(r, dict) or "seq" not in r:
                continue
            vals = [r.get("seq"), r.get("schema_version")]
            vals += [r.get(nm) for nm, _ in SCALARS]
            vals += [json.dumps(r.get(nm)) if r.get(nm) is not None else None
                     for nm in JSON_COLS]
            # Nothing is discarded: unrecognised keys are preserved verbatim,
            # so a field added to the sink later is still queryable today.
            vals.append(json.dumps({k: v for k, v in r.items()
                                    if k not in known}) or None)
            rows.append(vals)
        if rows:
            cur = cx.executemany(sql, rows)
            n = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            cx.commit()
    return n


def ingest_server(cx: sqlite3.Connection, path: str) -> int:
    if not os.path.exists(path):
        return 0
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows.append((r.get("wall_time"), r.get("players_online"),
                         r.get("max_players"), r.get("version")))
    if rows:
        cx.executemany("INSERT OR IGNORE INTO server VALUES (?,?,?,?)", rows)
        cx.commit()
    return len(rows)


HB_COLS = ["seq", "wall_time", "total_timesteps", "uptime_s", "steps_per_s",
           "breaks_total", "logs", "attack_run", "episodes", "gpu_mem_mb",
           # JSON, like reward_shares on `segments`: an open-ended map, so a
           # NEW reward source appears without a schema migration. The panel
           # must not hardcode its keys -- that is what hid `imagination`.
           "income_now"]


def ingest_heartbeat(cx: sqlite3.Connection, path: str) -> int:
    """Same idempotent-on-seq contract as segments, different table."""
    if not os.path.exists(path):
        return 0
    sql = (f"INSERT OR IGNORE INTO heartbeat ({','.join(HB_COLS)}) "
           f"VALUES ({','.join('?' * len(HB_COLS))})")
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue                      # torn tail line; next pass gets it
            if isinstance(r, dict) and "seq" in r:
                rows.append([json.dumps(r.get(c)) if c == "income_now"
                             and isinstance(r.get(c), dict) else r.get(c)
                             for c in HB_COLS])
    if rows:
        cur = cx.executemany(sql, rows)
        cx.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    return 0


# ===========================================================================
# TELEMETRY FEEDS (2026-10-07): learning.jsonl, position_trace.jsonl,
# memory_census.jsonl, incidents/<ts>-<class>/summary.json
# ===========================================================================
# SAME PRINCIPLE AS `segments`: the JSONL is the record, these tables are a
# view. Every insert is INSERT OR IGNORE on a natural key, so re-reading a
# file (the rsync mirror renames a fresh copy into place every 60 s, and the
# rotated `.N` siblings are re-read whenever their content shifts) is
# idempotent, and `rm skybot.db` + one pass rebuilds the same rows.
#
# KV, NOT COLUMNS, FOR learning.jsonl. Its reward/outcome/behaviour maps are
# open-ended (a new reward source, a new block type, a new option slot) and
# the producer adds keys without warning -- `segments` needed an add-only
# migration ritual and still lost a week to a missing `extra` column. Here a
# new key is just a new `key` string: no migration exists to forget.
#
# WHAT IS SKIPPED, AND COUNTED (never silently):
#   * a torn LAST line (no trailing newline) is not consumed at all -- the
#     read offset stops before it and the next pass gets it whole;
#   * a corrupt middle line, or a record whose (schema, v) this ingester does
#     not know, is skipped and counted in TELEMETRY_SKIPS (printed each pass
#     that changes it). An unknown version is NOT best-effort parsed: a v2
#     that renamed a key would otherwise land as plausible-looking wrong rows.
import glob as _glob
import math as _math
import re as _re

LEARNING_SCHEMA = ("skybot.learning", 1)
POSITION_SCHEMA = ("skybot.position", 1)
INCIDENT_SCHEMA = ("skybot.incident", 1)
CENSUS_SCHEMAS = {"memory_census/1"}
# Never flattened into kv: they are the row's own identity columns.
_LEARNING_META = {"schema", "v", "seq", "wall_time", "run_id"}
# memory_census sections whose leaves are unbounded (one per module/tensor):
# the census writes them for a human, not for a time series.
_CENSUS_SKIP = {"objects", "deep_tensors", "errors", "schema", "wall_time"}
_MAX_DEPTH = 6
_BATCH = 5000

TELEMETRY_SKIPS = {"learning": 0, "positions": 0, "memory_census": 0,
                   "incidents": 0}
# path -> (byte offset consumed, the bytes just before it). Kept IN MEMORY,
# not in the DB, so the DB holds nothing a rebuild could not reproduce; a
# restarted ingester simply re-reads every file once (idempotent).
_OFFSETS = {}


def connect_telemetry(cx: sqlite3.Connection) -> None:
    cx.execute("CREATE TABLE IF NOT EXISTS learning_kv ("
               "run_id TEXT NOT NULL, seq INTEGER NOT NULL, wall_time REAL, "
               "key TEXT NOT NULL, value REAL, "
               "PRIMARY KEY (run_id, seq, key))")
    cx.execute("CREATE INDEX IF NOT EXISTS learning_kv_key_time "
               "ON learning_kv (key, wall_time)")
    cx.execute("CREATE TABLE IF NOT EXISTS positions ("
               "run_id TEXT NOT NULL, stream INTEGER NOT NULL, "
               "step INTEGER NOT NULL, t_wall REAL NOT NULL, "
               "x REAL, y REAL, z REAL, yaw REAL, pitch REAL, "
               "PRIMARY KEY (run_id, stream, step, t_wall))")
    cx.execute("CREATE INDEX IF NOT EXISTS positions_time "
               "ON positions (t_wall)")
    # t_start / t_end are EPOCH SECONDS here (the bundle's *_epoch fields, or
    # its ISO strings parsed): Grafana filters on numbers, not on ISO text.
    cx.execute("CREATE TABLE IF NOT EXISTS incidents ("
               "dir TEXT PRIMARY KEY, class TEXT, exit_code INTEGER, "
               "signal TEXT, t_start REAL, t_end REAL, run_id TEXT, "
               "git_rev TEXT, config_hash TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS memory_census ("
               "wall_time REAL NOT NULL, key TEXT NOT NULL, value REAL, "
               "PRIMARY KEY (wall_time, key))")


def _num(v):
    """A finite float, or None. bool -> 0/1 (grad_norm_clipped et al.)."""
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        f = float(v)
        return f if _math.isfinite(f) else None
    return None


def flatten_numeric(obj, prefix="", out=None, depth=0, skip=()):
    """Dotted-key numeric leaves. Lists, strings and nulls are not numbers
    and are dropped (an absent producer logs `null` -> simply no row)."""
    if out is None:
        out = {}
    if not isinstance(obj, dict) or depth > _MAX_DEPTH:
        return out
    for k, v in obj.items():
        if depth == 0 and k in skip:
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flatten_numeric(v, key + ".", out, depth + 1, skip)
        else:
            f = _num(v)
            if f is not None and len(key) <= 200:
                out[key] = f
    return out


def _siblings(path: str):
    """`name` plus its rotated `name.<N>` siblings, oldest first."""
    rx = _re.compile(_re.escape(os.path.basename(path)) + r"(\.\d+)?$")
    found = [p for p in _glob.glob(_glob.escape(path) + "*")
             if rx.match(os.path.basename(p)) and os.path.isfile(p)]
    def _age(p):
        suf = p[len(path):].lstrip(".")
        return int(suf) if suf.isdigit() else 0
    return sorted(found, key=_age, reverse=True)


def _new_lines(path: str):
    """Yield complete new lines of `path` since the last pass. A shrunk or
    rewritten file (rotation, a fresh rsync copy whose prefix changed) is
    detected by the bytes just before the stored offset and re-read from 0."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    off, tail = _OFFSETS.get(path, (0, b""))
    with open(path, "rb") as f:
        if off:
            ok = size >= off
            if ok and tail:
                f.seek(off - len(tail))
                ok = f.read(len(tail)) == tail
            if not ok:
                off = 0
        f.seek(off)
        consumed = off
        last = b""
        for raw in f:
            if not raw.endswith(b"\n"):
                break                     # torn tail: leave it for next pass
            consumed += len(raw)
            last = raw
            yield raw
        if consumed:
            if last:
                tail = last[-64:]
            _OFFSETS[path] = (consumed, tail)


def _parse(raw: bytes):
    s = raw.decode("utf-8", errors="replace").strip()
    if not s:
        return None, False
    try:
        return json.loads(s), True
    except Exception:
        # A crash mid-write leaves a fragment that the next run's first
        # record is appended to, on the same line. Rescue the whole record.
        i = s.rfind('{"schema"')
        if i > 0:
            try:
                return json.loads(s[i:]), True
            except Exception:
                pass
        return None, True


def _flush(cx, sql, rows) -> int:
    if not rows:
        return 0
    before = cx.total_changes
    cx.executemany(sql, rows)
    cx.commit()
    n = cx.total_changes - before
    rows.clear()
    return n


def ingest_learning(cx: sqlite3.Connection, path: str = None) -> int:
    path = path or LEARNING_SRC
    sql = ("INSERT OR IGNORE INTO learning_kv (run_id, seq, wall_time, key, "
           "value) VALUES (?,?,?,?,?)")
    n, rows = 0, []
    for p in _siblings(path):
        for raw in _new_lines(p):
            r, nonblank = _parse(raw)
            if r is None:
                TELEMETRY_SKIPS["learning"] += int(nonblank)
                continue
            if (not isinstance(r, dict)
                    or (r.get("schema"), r.get("v")) != LEARNING_SCHEMA
                    or not isinstance(r.get("seq"), int)):
                TELEMETRY_SKIPS["learning"] += 1
                continue
            run_id = str(r.get("run_id") or "")
            wt = _num(r.get("wall_time"))
            for k, v in flatten_numeric(r, skip=_LEARNING_META).items():
                rows.append((run_id, r["seq"], wt, k, v))
            if len(rows) >= _BATCH:
                n += _flush(cx, sql, rows)
    return n + _flush(cx, sql, rows)


def ingest_positions(cx: sqlite3.Connection, path: str = None) -> int:
    path = path or POSITIONS_SRC
    sql = ("INSERT OR IGNORE INTO positions (run_id, stream, step, t_wall, "
           "x, y, z, yaw, pitch) VALUES (?,?,?,?,?,?,?,?,?)")
    n, rows = 0, []
    for p in _siblings(path):
        for raw in _new_lines(p):
            r, nonblank = _parse(raw)
            if r is None:
                TELEMETRY_SKIPS["positions"] += int(nonblank)
                continue
            try:
                assert isinstance(r, dict)
                assert (r.get("schema"), r.get("v")) == POSITION_SCHEMA
                x, y, z, t = (_num(r.get(c)) for c in ("x", "y", "z",
                                                         "t_wall"))
                assert None not in (x, y, z, t)
                row = (str(r.get("run_id") or ""), int(r["stream"]),
                       int(r["step"]), t, x, y, z, _num(r.get("yaw")),
                       _num(r.get("pitch")))
            except Exception:
                TELEMETRY_SKIPS["positions"] += 1
                continue
            rows.append(row)
            if len(rows) >= _BATCH:
                n += _flush(cx, sql, rows)
    return n + _flush(cx, sql, rows)


def ingest_memory_census(cx: sqlite3.Connection, path: str = None) -> int:
    path = path or CENSUS_SRC
    sql = ("INSERT OR IGNORE INTO memory_census (wall_time, key, value) "
           "VALUES (?,?,?)")
    n, rows = 0, []
    for p in _siblings(path):
        for raw in _new_lines(p):
            r, nonblank = _parse(raw)
            if r is None:
                TELEMETRY_SKIPS["memory_census"] += int(nonblank)
                continue
            wt = _num(r.get("wall_time")) if isinstance(r, dict) else None
            if wt is None or r.get("schema") not in CENSUS_SCHEMAS:
                TELEMETRY_SKIPS["memory_census"] += 1
                continue
            for k, v in flatten_numeric(r, skip=_CENSUS_SKIP).items():
                rows.append((wt, k, v))
            if len(rows) >= _BATCH:
                n += _flush(cx, sql, rows)
    return n + _flush(cx, sql, rows)


def _epoch(r: dict, name: str):
    v = _num(r.get(name + "_epoch"))
    if v is not None:
        return v
    v = r.get(name)
    if _num(v) is not None:
        return _num(v)
    if isinstance(v, str) and v:
        from datetime import datetime, timezone
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d.timestamp()
        except ValueError:
            return None
    return None


def ingest_incidents(cx: sqlite3.Connection, root: str = None) -> int:
    """One row per bundle dir. Reads each `<dir>/summary.json`, plus
    `index.jsonl` so a bundle main pruned for space still has its row."""
    root = root or INCIDENTS_DIR
    if not os.path.isdir(root):
        return 0
    recs = []
    for d in sorted(os.listdir(root)):
        sp = os.path.join(root, d, "summary.json")
        if d.startswith(".") or not os.path.isfile(sp):
            continue
        try:
            with open(sp, "r", encoding="utf-8", errors="replace") as f:
                recs.append((d, json.load(f)))
        except Exception:
            TELEMETRY_SKIPS["incidents"] += 1
    idx = os.path.join(root, "index.jsonl")
    if os.path.isfile(idx):
        for raw in _new_lines(idx):
            r, nonblank = _parse(raw)
            if isinstance(r, dict) and r.get("bundle"):
                recs.append((os.path.basename(str(r["bundle"]).rstrip("/")),
                             r))
            elif nonblank:
                TELEMETRY_SKIPS["incidents"] += 1
    rows = []
    for d, r in recs:
        if (not isinstance(r, dict)
                or (r.get("schema"), r.get("v")) != INCIDENT_SCHEMA):
            TELEMETRY_SKIPS["incidents"] += 1
            continue
        ec = _num(r.get("exit_code"))
        rows.append((d, r.get("class"), None if ec is None else int(ec),
                     None if r.get("signal") is None else str(r["signal"]),
                     _epoch(r, "t_start"), _epoch(r, "t_end"),
                     r.get("run_id"), r.get("git_rev"), r.get("config_hash")))
    return _flush(cx, "INSERT OR IGNORE INTO incidents (dir, class, "
                  "exit_code, signal, t_start, t_end, run_id, git_rev, "
                  "config_hash) VALUES (?,?,?,?,?,?,?,?,?)", rows)


def ingest_telemetry(cx: sqlite3.Connection) -> dict:
    """One pass over every telemetry feed. Each feed is isolated: a broken
    one is reported and the others still ingest."""
    out = {}
    for name, fn in (("learning", ingest_learning),
                     ("positions", ingest_positions),
                     ("memory_census", ingest_memory_census),
                     ("incidents", ingest_incidents)):
        try:
            out[name] = fn(cx)
        except Exception as exc:
            print(f"ingest {name} error: {exc!r}", flush=True)
            out[name] = 0
    return out


def main() -> None:
    cx = connect()
    once = "--once" in sys.argv
    last_skips = dict(TELEMETRY_SKIPS)
    while True:
        try:
            a = ingest_segments(cx, SRC)
            ingest_heartbeat(cx, HB_SRC)
            ingest_server(cx, SERVER_SRC)
            total = cx.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
            if a:
                print(f"ingest: +{a} new (total {total})", flush=True)
        except Exception as exc:                      # never die on one bad pass
            print(f"ingest error: {exc!r}", flush=True)
        # Separate try: a telemetry-feed bug must not stall `segments`.
        try:
            t = ingest_telemetry(cx)
            if any(t.values()):
                print(f"ingest telemetry: +{t}", flush=True)
            if TELEMETRY_SKIPS != last_skips:
                print(f"ingest telemetry skipped (torn/corrupt/unknown "
                      f"schema): {TELEMETRY_SKIPS}", flush=True)
                last_skips.update(TELEMETRY_SKIPS)
        except Exception as exc:
            print(f"ingest telemetry error: {exc!r}", flush=True)
        if once:
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
