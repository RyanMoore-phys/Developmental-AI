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
    same record legitimately arrives twice. `seq` is assigned by the pod-side
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
POLL = float(os.environ.get("POLL_SECONDS", "10"))

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


def main() -> None:
    cx = connect()
    once = "--once" in sys.argv
    while True:
        try:
            a = ingest_segments(cx, SRC)
            ingest_server(cx, SERVER_SRC)
            total = cx.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
            if a:
                print(f"ingest: +{a} new (total {total})", flush=True)
        except Exception as exc:                      # never die on one bad pass
            print(f"ingest error: {exc!r}", flush=True)
        if once:
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
