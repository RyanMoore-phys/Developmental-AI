"""The telemetry VIEW: learning/position/census/incident JSONL -> SQLite ->
Grafana dashboards + alert rules, and the offline learning report (2026-10-07).

WHY THIS EXISTS. The view layer has failed QUIETLY before, every time the
same way: the containers stay healthy and the panels show nothing. An ingest
whose table lacked one column stored nothing for days ("table segments has no
column named extra"); a panel that hardcoded its reward keys hid a whole
reward source (`imagination`). A dashboard query naming a key the producer
never writes is not an error in Grafana -- it is an empty panel, which reads
as "the agent did nothing". So the claim tested here is not "the JSON is
valid"; it is that EVERY query shipped in grafana/provisioning returns rows
against an ingest of records shaped exactly like the spec's producers write,
and that the alert rules fire on the conditions they name.

PYTHONPATH=. ./venv/bin/python tests/_telemetry_ingest_smoke.py

Contracts:
    1. FLATTEN: learning.jsonl -> learning_kv one row per numeric leaf
       (dotted key), signed values kept, bool -> 0/1, lists/strings/null
       dropped, a null producer -> no rows for it (a gap, never a zero).
    2. SKIPS COUNTED: a corrupt middle line and an unknown (schema, v) are
       skipped AND counted; a torn LAST line is not consumed and not counted,
       and is ingested once its newline arrives.
    3. POSITIONS / INCIDENTS / CENSUS land with their natural keys; incident
       times are epoch seconds (ISO fallback parsed); an index.jsonl entry
       keeps a pruned bundle's row; census drops the unbounded `objects`.
    4. IDEMPOTENT: duplicates in the log, a second pass, a fresh process, and
       a ROTATION (live -> .1, new live file) never duplicate or lose a row.
    5. REPLAYABLE: delete skybot.db, one `--once` pass -> identical tables.
    6. DASHBOARDS: every SQLite target (macros substituted) executes and
       returns >= 1 row on the synthetic ingest; every quoted key literal
       matches a real key (one drifted name in an IN list would otherwise
       hide behind its siblings); PromQL names only known exporter metrics;
       datasource uids exist; dashboard uids are unique.
    7. ALERTS: the six named rules exist, every SQL rule query returns a
       numeric `value`, and they FIRE on the synthetic failure (a stuck
       stream at 6000 steps, 3 recent crashes, an empty heartbeat table) and
       stay quiet on the healthy stream.
    8. REPORT: tools/learning_report.py runs on the raw JSONL (with --png),
       leads with the scoreboard and signed reward, and reports skips.
"""
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

import yaml

NAME = "telemetry-ingest"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MON = os.path.join(ROOT, "docker", "monitor")
ING = os.path.join(MON, "ingest.py")
PROV = os.path.join(MON, "grafana", "provisioning")
NOW = time.time()
TABLES = {"learning_kv": "run_id, seq, key",
          "positions": "run_id, stream, step, t_wall",
          "incidents": "dir", "memory_census": "wall_time, key"}


def _learning(seq, stuck=False, ppo_null=False):
    wt = NOW - (10 - seq) * 420.0
    rec = {
        "schema": "skybot.learning", "v": 1, "seq": seq, "wall_time": wt,
        "run_id": "run-A", "total_timesteps": 1000 * seq,
        "segment_steps": 1000, "streams": 2,
        "reward": {f"stream-{i}": {"icm_base": 0.5 + i, "empowerment": 0.1,
                                   "approach": 0.02, "gui_dwell": -0.3,
                                   "env": 0.0} for i in (0, 1)},
        "reward_mix": {f"stream-{i}": {
            "intrinsic_raw_sum": 2.0, "intrinsic_after_damp": 1.5,
            "intrinsic_after_gui_zero": 1.2, "w_intrinsic": 1.0,
            "w_extrinsic": 1.0, "mixed_sum": 1.4, "farm_damp_mean": 0.9,
            "habituation_mean": 0.8} for i in (0, 1)},
        "outcomes": {f"stream-{i}": {
            "breaks_by_type": {"dirt": 3, "oak_log": i},
            "breaks_segment": 3 + i, "logs_segment": i,
            "breaks_run": 10 * seq, "logs_run": seq * i,
            "achievements_new": ["mine_dirt"]} for i in (0, 1)},
        "behaviour": {f"stream-{i}": {
            "action_hist": {"forward": 600, "attack": 400},
            "action_entropy_bits": 1.7, "source_hist": {
                "policy": 900, "option_slot:3": 100},
            "gui_frac": 0.05, "gui_longest_run": 12,
            "still_frac": 0.2} for i in (0, 1)},
        "exploration": {f"stream-{i}": {
            "path_blocks": 120.0, "net_displacement": 30.0,
            "unique_cells_segment": 50, "unique_cells_total": 50 * seq,
            "new_cells_segment": 40, "cells_per_hour": 300.0,
            "radius_of_gyration": 9.5,
            "steps_since_new_cell": 6000 if (stuck and i == 1) else 3,
            "seconds_since_new_cell": 1.0, "stationary_frac": 0.1,
            "y_min": 62.0, "y_max": 70.0, "n_obs": 1000,
            "teleports": 0} for i in (0, 1)},
        "ppo": None if ppo_null else {
            "policy_loss": -0.01, "value_loss": 0.4, "entropy": 1.9,
            "approx_kl": 0.01, "clip_frac": 0.1,
            "explained_variance": 0.3, "grad_norm": 0.8,
            "grad_norm_clipped": True, "lr": 3e-4, "adv_mean": 0.0,
            "adv_std": 1.0, "ret_mean": 0.5, "n_samples": 2048,
            "epochs": 4, "grad_norm_actor": 0.5, "grad_norm_critic": 0.6,
            "minibatches": 32},
        "wm": {"loss_total": 1.0, "loss_recon": 0.6, "loss_kl": 0.2,
               "loss_reward": 0.1, "loss_continue": 0.05, "loss_flow": 0.05,
               "kl_raw": 1.1, "grad_norm": 5.0, "prior_entropy": 3.0,
               "posterior_entropy": 2.5, "steps": 40, "replay_ratio": 8.0},
        "curiosity": {"evaluated": 100, "passed_sig": 30, "passed_abs": 20,
                      "paid": 15, "collapsed_visits": 2, "buckets": 64,
                      "mean_paid": 0.01},
        "replay": {"sequences_sampled": 640, "rejected_boundary": 12,
                   "mean_priority": 1.0, "rejected_boundary_frac": 0.02,
                   "sample_calls": 40, "size": 5000},
        "timing": {"env_ms": 40.0, "train_ms": 900.0, "steps_per_s": 9.5},
        "health": {"nan_steps": 0, "grad_spikes": 0, "sink_dropped": 0},
    }
    return rec


def _expected_leaves(obj, meta=("schema", "v", "seq", "wall_time",
                                "run_id")):
    """Independent count of numeric leaves (not ingest's own walker)."""
    n = 0
    stack = [(obj, 0)]
    while stack:
        o, d = stack.pop()
        for k, v in o.items():
            if d == 0 and k in meta:
                continue
            if isinstance(v, dict):
                stack.append((v, d + 1))
            elif isinstance(v, (bool, int, float)):
                n += 1
    return n


def _write(path, lines, torn=None):
    with open(path, "w") as f:
        for ln in lines:
            f.write((ln if isinstance(ln, str) else json.dumps(ln)) + "\n")
        if torn:
            f.write(torn)


def _load_ingest(env):
    old = dict(os.environ)
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location("_ing_under_test", ING)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.environ.clear()
        os.environ.update(old)


def _dump(db):
    cx = sqlite3.connect(db)
    out = {t: cx.execute(f"SELECT * FROM {t} ORDER BY {o}").fetchall()
           for t, o in TABLES.items()}
    cx.close()
    return out


def _fixture(tmp):
    recs = [_learning(i, ppo_null=(i == 5))
            for i in range(1, 10)]
    lines = [r for r in recs] + recs[:3]                  # duplicates
    lines.insert(4, '{"schema": "skybot.learning", "v": 1, "seq": 99, BROKEN')
    lines.append({"schema": "skybot.learning", "v": 2, "seq": 77,
                  "reward": {"stream-0": {"x": 1}}})        # unknown version
    torn_rec = dict(_learning(10), wall_time=NOW - 30)
    torn_txt = json.dumps(torn_rec)
    _write(os.path.join(tmp, "learning.jsonl"), lines,
           torn=torn_txt[: len(torn_txt) // 2])
    pos = []
    for s, npts in ((0, 20), (1, 20), (2, 5)):
        for k in range(npts):
            pos.append({"schema": "skybot.position", "v": 1,
                        "run_id": "run-A", "stream": s, "episode": "ep-1",
                        "step": 16 * k, "t_wall": NOW - 3000 + 10 * k,
                        "x": 1.0 * k, "y": 64.0 + (k % 3), "z": -2.0 * k + s,
                        "yaw": 90.0, "pitch": None})
    pos += pos[:5]
    pos.append({"schema": "skybot.position", "v": 9, "stream": 0, "step": 1,
                "t_wall": NOW, "x": 0, "y": 0, "z": 0})
    _write(os.path.join(tmp, "position_trace.jsonl"), pos)
    census = [{"schema": "memory_census/1", "wall_time": NOW - 600 + i,
               "total_timesteps": 5, "deep": False,
               "process": {"rss": 4e9, "swap": 1e8, "children": [{"x": 1}]},
               "objects": {"by_path": {f"m{j}": j for j in range(50)}},
               "summary": {"rss": 4e9, "rss_anon": 3e9, "swap": 1e8,
                           "cpu_accounted_total": 2e9,
                           "gpu_accounted_total": 1e9,
                           "residual_note": "text"}} for i in range(3)]
    _write(os.path.join(tmp, "memory_census.jsonl"), census)
    inc = os.path.join(tmp, "incidents")
    summaries = []
    for j, cls in enumerate(["crash", "crash", "crash", "clean"]):
        d = f"20261007T0{j}0000Z-{cls}"
        os.makedirs(os.path.join(inc, d))
        s = {"schema": "skybot.incident", "v": 1, "class": cls,
             "exit_code": 134 if cls == "crash" else 0,
             "signal": "SIGABRT" if cls == "crash" else None,
             "t_start": "2026-10-07T00:00:00Z", "t_end": "x",
             "t_start_epoch": NOW - 7200, "t_end_epoch": NOW - 500 * (j + 1),
             "run_id": "run-A", "git_rev": "abc123", "config_hash": "cfg1",
             "bundle": f"/workspace/devai/runlogs/incidents/{d}"}
        if j == 1:                       # ISO-only producer: parsed fallback
            s.pop("t_end_epoch")
            s["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(NOW - 1200))
        with open(os.path.join(inc, d, "summary.json"), "w") as f:
            json.dump(s, f)
        summaries.append(s)
    pruned = dict(summaries[0], **{
        "class": "abnormal_under_stop", "t_end_epoch": NOW - 90000,
        "bundle": "/workspace/devai/runlogs/incidents/20261001T000000Z-"
                  "abnormal_under_stop"})
    _write(os.path.join(inc, "index.jsonl"), summaries + [pruned])
    os.makedirs(os.path.join(inc, ".partial-20261007T090000Z-crash"))
    return recs, torn_rec, torn_txt


def run():
    tmp = tempfile.mkdtemp()
    try:
        recs, torn_rec, torn_txt = _fixture(tmp)
        db = os.path.join(tmp, "skybot.db")
        env = {"DB_PATH": db, "METRICS_PATH": os.path.join(tmp, "m.jsonl"),
               "SERVER_PATH": os.path.join(tmp, "none"),
               "HEARTBEAT_PATH": os.path.join(tmp, "hb.jsonl")}
        ing = _load_ingest(env)
        cx = ing.connect()
        got = ing.ingest_telemetry(cx)

        # 1. flatten
        exp = sum(_expected_leaves(r) for r in recs)
        n_kv = cx.execute("SELECT COUNT(*) FROM learning_kv").fetchone()[0]
        assert n_kv == exp == got["learning"], (n_kv, exp, got)
        kv = dict(cx.execute("SELECT key, value FROM learning_kv WHERE "
                             "seq = 2").fetchall())
        assert kv["reward.stream-0.gui_dwell"] == -0.3, "sign lost"
        assert kv["ppo.grad_norm_clipped"] == 1.0, "bool not 0/1"
        assert kv["outcomes.stream-1.breaks_by_type.oak_log"] == 1.0
        assert kv["exploration.stream-0.teleports"] == 0.0
        assert not any("achievements_new" in k for k in kv), "list leaked"
        n_ppo5 = cx.execute("SELECT COUNT(*) FROM learning_kv WHERE seq = 5 "
                            "AND key LIKE 'ppo.%'").fetchone()[0]
        assert n_ppo5 == 0, "null producer must be a gap, not rows"
        assert {r[0] for r in cx.execute(
            "SELECT DISTINCT run_id FROM learning_kv")} == {"run-A"}
        print(f"  1. {len(recs)} records -> {n_kv} learning_kv rows (one per "
              f"numeric leaf); signed, bool->0/1, lists + null producer "
              f"dropped")

        # 2. skips counted; torn tail not consumed, then ingested whole
        assert ing.TELEMETRY_SKIPS["learning"] == 2, ing.TELEMETRY_SKIPS
        assert cx.execute("SELECT COUNT(*) FROM learning_kv WHERE seq IN "
                          "(10, 77, 99)").fetchone()[0] == 0
        with open(os.path.join(tmp, "learning.jsonl"), "a") as f:
            f.write(torn_txt[len(torn_txt) // 2:] + "\n")
        assert ing.ingest_learning(cx) == _expected_leaves(torn_rec)
        assert ing.TELEMETRY_SKIPS["learning"] == 2, "torn line was counted"
        recs.append(torn_rec)
        print("  2. corrupt middle line + unknown v=2 skipped AND counted (2); "
              "torn tail left unconsumed, ingested once its newline arrived")

        # 3. positions / incidents / census
        assert cx.execute("SELECT COUNT(*) FROM positions").fetchone()[0] \
            == 45 and ing.TELEMETRY_SKIPS["positions"] == 1
        inc = {r[0]: r[1:] for r in cx.execute(
            "SELECT dir, class, exit_code, signal, t_end FROM incidents")}
        assert len(inc) == 5, sorted(inc)        # 4 dirs + 1 pruned, no .partial
        assert "20261001T000000Z-abnormal_under_stop" in inc
        iso = inc["20261007T010000Z-crash"][3]
        assert abs(iso - (NOW - 1200)) < 2, f"ISO t_end parsed as {iso}"
        assert inc["20261007T000000Z-crash"][1:3] == (134, "SIGABRT")
        ck = {r[0] for r in cx.execute("SELECT DISTINCT key FROM "
                                       "memory_census")}
        assert "summary.rss" in ck and "process.rss" in ck
        assert not any(k.startswith("objects.") for k in ck), "unbounded"
        print(f"  3. positions 45 (5 dupes merged, bad schema counted), "
              f"incidents {len(inc)} incl. a pruned bundle via index.jsonl, "
              f"ISO t_end -> epoch; census {len(ck)} keys, `objects` dropped")

        # 4. idempotent: second pass, fresh process, rotation
        before = _dump(db)
        again = ing.ingest_telemetry(cx)
        assert not any(again.values()), again
        cx.close()
        sub_env = dict(os.environ, **env)
        run_once = [sys.executable, ING, "--once"]
        subprocess.run(run_once, env=sub_env, check=True, capture_output=True,
                       timeout=60)
        assert _dump(db) == before, "a fresh full re-read changed the DB"
        lp = os.path.join(tmp, "learning.jsonl")
        os.replace(lp, lp + ".1")
        new = [_learning(11), _learning(12, stuck=True)]
        for r in new:
            r["wall_time"] = NOW - 20
        _write(lp, new)
        recs += new
        subprocess.run(run_once, env=sub_env, check=True, capture_output=True,
                       timeout=60)
        cx = sqlite3.connect(db)
        n_rot = cx.execute("SELECT COUNT(*) FROM learning_kv").fetchone()[0]
        assert n_rot == sum(_expected_leaves(r) for r in recs), n_rot
        print(f"  4. idempotent: 2nd pass +0, fresh process identical, "
              f"rotation live->.1 + new file -> {n_rot} rows, none duplicated")

        # 5. replayable
        built = _dump(db)
        cx.close()
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(db + suf):
                os.remove(db + suf)
        subprocess.run(run_once, env=sub_env, check=True, capture_output=True,
                       timeout=60)
        assert _dump(db) == built, "rebuild from JSONL differs"
        print("  5. rm skybot.db + one pass -> all four tables identical")

        cx = sqlite3.connect(db)
        check_dashboards(cx)
        check_alerts(cx)
        cx.close()
        check_report(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _sub(q):
    return (q.replace("$__unixEpochFrom()", "0")
             .replace("$__unixEpochTo()", "4000000000"))


PROM_METRICS = {
    "up", "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes",
    "node_memory_SwapTotal_bytes", "node_memory_SwapFree_bytes",
    "node_vmstat_pswpin", "node_vmstat_pswpout",
    "node_pressure_memory_waiting_seconds_total",
    "node_pressure_memory_stalled_seconds_total",
    "node_pressure_io_stalled_seconds_total",
    "nvidia_smi_utilization_gpu_ratio", "nvidia_smi_memory_used_bytes",
    "nvidia_smi_memory_total_bytes"}
PROM_FUNCS = {"rate", "max", "sum", "avg", "min", "by", "on", "and", "or",
              "bool"}


def _check_prom(expr):
    assert expr.count("(") == expr.count(")") and \
        expr.count("{") == expr.count("}") and \
        expr.count("[") == expr.count("]"), f"unbalanced: {expr}"
    bare = re.sub(r"\{[^}]*\}|\[[^\]]*\]|\"[^\"]*\"", "", expr)
    for name in re.findall(r"[A-Za-z_:][A-Za-z0-9_:]*", bare):
        assert name in PROM_METRICS or name in PROM_FUNCS, \
            f"unknown metric {name!r} in {expr}"


def _key_literals(cx, q):
    """Every quoted literal compared against `key` must match a real key."""
    lits = []
    for m in re.finditer(r"key\s+IN\s*\(([^)]*)\)", q):
        lits += re.findall(r"'([^']*)'", m.group(1))
    lits += re.findall(r"key\s*(?:=|<>|LIKE)\s*'([^']*)'", q)
    table = "memory_census" if "FROM memory_census" in q else "learning_kv"
    for lit in lits:
        op = "LIKE" if "%" in lit else "="
        hit = cx.execute(f"SELECT 1 FROM {table} WHERE key {op} ? LIMIT 1",
                         (lit,)).fetchone()
        assert hit, f"key literal {lit!r} matches nothing in {table}: {q}"
    return len(lits)


def _uids():
    out = set()
    for fn in os.listdir(os.path.join(PROV, "datasources")):
        with open(os.path.join(PROV, "datasources", fn)) as f:
            for ds in yaml.safe_load(f).get("datasources", []):
                out.add(ds["uid"])
    return out


def check_dashboards(cx):
    ddir = os.path.join(PROV, "dashboards")
    uids, titles = _uids(), {}
    n_sql = n_prom = n_lit = 0
    for fn in sorted(os.listdir(ddir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(ddir, fn)) as f:
            d = json.load(f)
        assert d["uid"] not in titles.values(), f"dup uid {d['uid']}"
        titles[d["title"]] = d["uid"]
        if fn == "skybot.json":
            continue                     # the pre-existing live board
        for p in d["panels"]:
            if p["type"] == "row":
                continue
            assert p["datasource"]["uid"] in uids, (fn, p["title"])
            assert p.get("targets"), (fn, p["title"])
            for t in p["targets"]:
                ds = (t.get("datasource") or p["datasource"])["uid"]
                if ds == "rack-prometheus":
                    _check_prom(t["expr"])
                    n_prom += 1
                    continue
                q = _sub(t["rawQueryText"])
                rows = cx.execute(q).fetchall()
                assert rows, f"{fn} / {p['title']}: no rows -> empty panel\n{q}"
                n_lit += _key_literals(cx, q)
                n_sql += 1
    for need in ("SkyBot Learning", "SkyBot Exploration", "SkyBot ML Health",
                 "SkyBot Host"):
        assert need in titles, f"missing dashboard {need}"
    ex = json.load(open(os.path.join(ddir, "skybot_exploration.json")))
    kinds = {p["type"] for p in ex["panels"]}
    assert "xychart" in kinds, "no XZ scatter"
    print(f"  6. {len(titles)} dashboards: {n_sql} SQL targets all return "
          f"rows, {n_lit} key literals all real, {n_prom} PromQL targets "
          f"name only known metrics, uids unique + datasources exist")


def check_alerts(cx):
    with open(os.path.join(PROV, "alerting", "skybot_alerts.yml")) as f:
        doc = yaml.safe_load(f)
    uids = _uids() | {"__expr__"}
    rules = {r["title"]: r for g in doc["groups"] for r in g["rules"]}
    need = {"main down", "main memory thrash", "heartbeat stale",
            "crash loop", "collector blind", "stuck (no new cell)"}
    assert need <= set(rules), need - set(rules)
    vals = {}
    for title, r in rules.items():
        refs = {d["refId"] for d in r["data"]}
        assert r["condition"] in refs, title
        for d in r["data"]:
            assert d["datasourceUid"] in uids, (title, d["datasourceUid"])
            m = d["model"]
            if d["datasourceUid"] == "rack-prometheus":
                _check_prom(m["expr"])
            elif d["datasourceUid"] == "skybot-sqlite":
                assert m["rawQueryText"] == m["queryText"], title
                cur = cx.execute(m["rawQueryText"])
                cols = [c[0] for c in cur.description]
                assert "value" in cols, (title, cols)
                vals[title] = {(row[0] if len(row) > 1 else ""): row[-1]
                               for row in cur.fetchall()}
            elif m.get("type") in ("reduce", "threshold", "math"):
                for ref in re.findall(r"\$?([A-Z]{1,2})\b",
                                      str(m["expression"])):
                    assert ref in refs, (title, ref)
    assert vals["stuck (no new cell)"] == {"stream-0": 3.0,
                                           "stream-1": 6000.0}, vals
    assert vals["crash loop"][""] == 3, vals["crash loop"]
    assert vals["heartbeat stale"][""] > 900, "empty heartbeat must fire"
    assert vals["collector blind"][""] < 900, "fresh learning rows: quiet"
    print(f"  7. {len(rules)} alert rules wired; on the synthetic ingest: "
          f"stuck fires only for stream-1 (6000 > 5000), crash loop counts "
          f"3 in 30 min, empty heartbeat fires, fresh learning_kv is quiet")


def check_report(tmp):
    out = os.path.join(tmp, "report")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "learning_report.py"),
         "--learning", os.path.join(tmp, "learning.jsonl.1"),
         "--positions", os.path.join(tmp, "position_trace.jsonl"),
         "--out", out, "--png"],
        capture_output=True, text=True, timeout=120,
        env=dict(os.environ, MPLBACKEND="Agg"))
    assert r.returncode == 0, r.stderr[-2000:]
    md = open(os.path.join(out, "report.md")).read()
    for must in ("## Scoreboard", "## Reward by source", "| gui_dwell | -3 |",
                 "## Exploration", "## Position trace",
                 "learning 2,", "ppo.grad_norm_actor"):
        assert must in md, f"report missing {must!r}"
    assert md.index("## Scoreboard") < md.index("## Reward by source")
    pngs = [f for f in os.listdir(out) if f.endswith(".png")]
    print(f"  8. learning_report.py: report.md (scoreboard first, signed "
          f"reward, skips reported) + {len(pngs)} PNGs")


if __name__ == "__main__":
    run()
    print(f"[{NAME}] ALL PASS: every telemetry feed lands in SQLite "
          f"idempotently and rebuilds identically from its JSONL, skips are "
          f"counted, and every shipped dashboard and alert query returns "
          f"rows against producer-shaped records")
