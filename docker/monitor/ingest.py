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
HB_SRC = os.environ.get("HEARTBEAT_PATH", "/data/heartbeat.jsonl")
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


def main() -> None:
    cx = connect()
    once = "--once" in sys.argv
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
        if once:
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
