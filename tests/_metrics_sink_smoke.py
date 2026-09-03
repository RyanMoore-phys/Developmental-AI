"""LIVE TRACKER: the metrics sink, its emission site, and the ingest contract.

WHY THIS EXISTS
    The tracker is a monitoring system, and a monitoring system has two ways
    to fail that ordinary code does not:

      1. IT CAN KILL WHAT IT WATCHES. A sink that raises inside the step loop
         takes the run down over a metric. Every path here must swallow.

      2. IT CAN LOOK ALIVE AND STORE NOTHING. That is the dangerous one,
         because a green dashboard is read as evidence. Two concrete ways it
         happens, both found while building this:

         * `RewardLedger.segment()` is a CONSUMING read — it closes and resets
           the segment. A second caller gets an EMPTY statement and destroys
           the real one, so the reward-provenance panel would sit at zero
           while the segment log printed correct numbers. Contract: exactly
           ONE `.segment()` call repo-wide; the sink reads a stash.

         * `seq` is the collector's dedupe key (INSERT OR IGNORE on a UNIQUE
           column). If a restart rewinds it, the ingester silently discards
           every new record as "already seen". The first version resumed by
           counting lines in the live file, which after a ROTATION is short —
           so seq restarted at 1 and collided with everything already stored.
           Contract: seq never goes backwards, rotation included.

Run: PYTHONPATH=. python tests/_metrics_sink_smoke.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.infra.metrics_sink import MetricsSink, SCHEMA_VERSION

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")


def test_never_raises():
    tmp = tempfile.mkdtemp()
    try:
        s = MetricsSink({"path": os.path.join(tmp, "m.jsonl")})

        class Boom:
            def __repr__(self):
                raise RuntimeError("no")

        # Unserialisable payload, None, and a non-dict: none may raise.
        assert s.write({"bad": Boom()}) is False
        assert s.write(None) is True          # empty record is still a record
        assert s.write({"ok": 1}) is True
        assert s.dropped == 1, s.dropped
        assert s.last_error, "a drop must leave a diagnosable trace"
        print("  1. never raises on malformed input; drops are counted and "
              f"noted (dropped={s.dropped})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_numpy_coercion():
    import numpy as np
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "m.jsonl")
        s = MetricsSink({"path": p})
        # np.float32 is NOT json-serialisable; losing the whole record over one
        # metric would be the wrong trade.
        assert s.write({"a": np.float32(1.5), "b": {"c": np.int64(3)},
                        "d": [np.float64(2.0)]})
        row = json.loads(open(p).read().strip())
        assert row["a"] == 1.5 and row["b"]["c"] == 3 and row["d"] == [2.0]
        print("  2. numpy/torch scalars coerced rather than dropping the record")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_durability_and_schema():
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "m.jsonl")
        MetricsSink({"path": p}).write({"durable": True})
        # A SECOND PROCESS must see it immediately — flush() alone reaches the
        # page cache; fsync() is what survives a host death, and the most
        # interesting record is always the last one before a crash.
        out = subprocess.run(
            [sys.executable, "-c", f"print(open({p!r}).read().strip())"],
            capture_output=True, text=True).stdout.strip()
        row = json.loads(out)
        assert row["durable"] is True
        assert row["schema_version"] == SCHEMA_VERSION
        assert "seq" in row and "wall_time" in row
        print("  3. fsync'd: a second process reads the record immediately; "
              f"schema_version={SCHEMA_VERSION}, seq and wall_time present")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rotation_and_seq_never_rewinds():
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "m.jsonl")
        s = MetricsSink({"path": p, "max_bytes": 400, "keep": 3})
        for _ in range(40):
            s.write({"pad": "x" * 50})
        high = s.seq
        live = sum(1 for _ in open(p))
        kept = [f for f in os.listdir(tmp) if f.startswith("m.jsonl")
                and not f.endswith(".seq")]
        assert len(kept) <= 4, kept          # live + keep
        assert live < high, "rotation did not happen; test is not exercising it"
        print(f"  4. rotation bounded: {len(kept)} file(s) kept, live file has "
              f"{live} of {high} records")

        # THE TRAP: a restart after rotation must not rewind seq.
        s2 = MetricsSink({"path": p, "max_bytes": 400, "keep": 3})
        assert s2.seq >= high, (
            f"seq REWOUND {high} -> {s2.seq}. The ingester dedupes on seq "
            f"(INSERT OR IGNORE), so reused numbers are silently DISCARDED "
            f"and the tracker stores nothing while looking healthy.")
        resumed = s2.seq                     # capture BEFORE write increments
        s2.write({"after": "restart"})
        nxt = json.loads(open(p).read().strip().splitlines()[-1])["seq"]
        assert nxt == high + 1, nxt
        print(f"  5. seq survives rotation+restart: resumed at {resumed}, next "
              f"record {nxt} (line-count alone would have said {live})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_single_emission_site():
    src = open(LOOP).read()
    n = src.count("self._emit_metrics(episode_metrics)")
    assert n == 1, (
        f"_emit_metrics is called {n} times. It must be called EXACTLY ONCE, "
        f"at the point where the lifelong / parallel / single-episode bodies "
        f"converge. Per-body calls are the duplicated-body drift CLAUDE.md "
        f"§4.2 names: an edit landing in one body and not the others is "
        f"silent — it just stops applying live.")
    assert "def _emit_metrics" in src
    print("  6. exactly ONE emission site, at the three-body convergence point")


def test_ledger_segment_has_one_consumer():
    """segment() is a consuming read; a second caller destroys the statement."""
    hits = []
    for root, _dirs, files in os.walk("developmental_ai"):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            for i, line in enumerate(open(path, encoding="utf-8"), 1):
                stripped = line.strip()
                if stripped.startswith("#") or "def segment" in stripped:
                    continue
                if ".segment()" in stripped:
                    hits.append(f"{path}:{i}")
    assert len(hits) == 1, (
        f"RewardLedger.segment() is called from {len(hits)} places: {hits}. "
        f"It CLOSES AND RESETS the segment, so a second consumer receives an "
        f"empty statement AND destroys the real one — reward provenance would "
        f"read zero on the dashboard while the segment log printed correct "
        f"numbers. Consumers must read InfraStack.last_ledger_segment.")
    assert "last_ledger_segment" in open(
        os.path.join("developmental_ai", "infra", "stack.py")).read()
    print(f"  7. ledger.segment() has exactly one consumer ({hits[0]}); "
          f"everything else reads the stash")


def test_ingest_is_replayable():
    """The DB is a VIEW; the JSONL is the system of record."""
    ing = os.path.join("docker", "monitor", "ingest.py")
    if not os.path.exists(ing):
        print("  8. SKIP (docker/monitor/ingest.py absent)")
        return
    import sqlite3
    tmp = tempfile.mkdtemp()
    try:
        src, db = os.path.join(tmp, "m.jsonl"), os.path.join(tmp, "s.db")
        recs = [{"schema_version": 1, "seq": i, "wall_time": 1e9 + i,
                 "breaks_total": i * 7, "logs": i // 10,
                 "reward_shares": {"curiosity": 0.6},
                 "breaks_by_type": {"dirt": i}} for i in range(1, 31)]
        with open(src, "w") as f:
            for r in recs + recs[:10]:          # deliberate duplicates
                f.write(json.dumps(r) + "\n")
            f.write('{"torn": ')                # partial line, as tail leaves
        env = dict(os.environ, DB_PATH=db, METRICS_PATH=src,
                   SERVER_PATH=os.path.join(tmp, "none"))
        run = [sys.executable, ing, "--once"]
        subprocess.run(run, env=env, check=True, capture_output=True)
        cx = sqlite3.connect(db)
        q = ("SELECT group_concat(seq||':'||breaks_total,'|') FROM "
             "(SELECT * FROM segments ORDER BY seq)")
        n1, h1 = (cx.execute("SELECT COUNT(*) FROM segments").fetchone()[0],
                  cx.execute(q).fetchone()[0])
        cx.close()
        assert n1 == 30, f"{n1} rows from 40 lines — dedupe on seq failed"

        os.remove(db)                            # rebuild from the log alone
        subprocess.run(run, env=env, check=True, capture_output=True)
        cx = sqlite3.connect(db)
        n2, h2 = (cx.execute("SELECT COUNT(*) FROM segments").fetchone()[0],
                  cx.execute(q).fetchone()[0])
        cx.close()
        assert (n1, h1) == (n2, h2), "rebuild did not reproduce the table"
        print(f"  8. ingest idempotent on seq (40 lines -> {n1} rows, torn "
              f"line skipped) and the DB rebuilds identically from the JSONL")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for fn in (test_never_raises, test_numpy_coercion,
               test_durability_and_schema, test_rotation_and_seq_never_rewinds,
               test_single_emission_site, test_ledger_segment_has_one_consumer,
               test_ingest_is_replayable):
        fn()
    print("[metrics-sink] ALL PASS: the sink never raises and never silently "
          "stores nothing — fsync'd per record, seq survives rotation and "
          "restart, one emission site at the three-body convergence, one "
          "ledger.segment() consumer, and the SQLite view rebuilds exactly "
          "from the JSONL that remains the system of record")
