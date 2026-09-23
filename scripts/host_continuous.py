#!/usr/bin/env python3
"""Continuous regression — run the suites on a cadence and keep a LEDGER.

WHY A LEDGER AND NOT JUST A PASS/FAIL. A suite that passes today and passes
tomorrow tells you nothing about the run in between. What this project keeps
losing to is DRIFT: a number that slowly stops meaning what it meant, a
sensor that goes constant, a guard that quietly becomes a latch. None of
those flip a test red on the step they start.

So each cycle appends one JSON row: which suites passed, how long they took,
and the live metrics that have a direction. `--report` then diffs the rows
and says what CHANGED, which is the only question worth asking of a system
that was already green yesterday.

  # one cycle
  PYTHONPATH=. python scripts/host_continuous.py --once

  # every 30 min, forever (nohup it)
  PYTHONPATH=. python scripts/host_continuous.py --interval 1800

  # what has moved since the first row
  PYTHONPATH=. python scripts/host_continuous.py --report
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, ".")

LEDGER = "runlogs/continuous_ledger.jsonl"

# Metrics with a DIRECTION we care about, and what direction that is.
# "down" = should fall (a loss). "stable" = should not wander.
# "alive" = must not go constant, which is how a dead sensor looks.
WATCH = {
    "flow_loss": "down",
    "horizon_loss": "down",
    "reconstruction_error": "down",
    "flow_mask": "stable",
    "flow_residual": "alive",
    "oracle_dr_drift": "stable",
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_suites(tiers):
    out = {}
    for tier in tiers:
        t0 = time.time()
        p = subprocess.run(
            [sys.executable, "tests/run_all.py", tier],
            capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH="."))
        txt = p.stdout + p.stderr
        per = {}
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] in ("PASS", "FAIL", "TIMEOUT",
                                                "MISSING"):
                per[parts[-1]] = parts[0]
        out[tier] = {"rc": p.returncode, "secs": round(time.time() - t0, 1),
                     "suites": per}
    return out


def read_metrics(path):
    """Pull the watched series out of a metrics dump, if one exists."""
    if not os.path.isfile(path):
        return {}
    try:
        m = json.load(open(path))
    except Exception:
        return {}
    out = {}
    for k in WATCH:
        v = m.get(k)
        if not v:
            continue
        tail = [float(x) for x in v[-50:]]
        if not tail:
            continue
        n = len(tail)
        out[k] = {
            "n": n,
            "mean": round(sum(tail) / n, 6),
            "last": round(tail[-1], 6),
            # Std is the ALIVE check: a sensor or metric that goes constant
            # has died, and nothing else in this file would notice.
            "std": round((sum((x - sum(tail) / n) ** 2 for x in tail) / n)
                         ** 0.5, 6),
        }
    return out


def cycle(tiers, metrics_path):
    row = {"t": _now(), "tiers": run_suites(tiers),
           "metrics": read_metrics(metrics_path)}
    row["ok"] = all(v["rc"] == 0 for v in row["tiers"].values())
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    flag = "OK " if row["ok"] else "FAIL"
    detail = " ".join(f"{k}:{v['rc']}({v['secs']}s)"
                      for k, v in row["tiers"].items())
    print(f"[{row['t']}] {flag}  {detail}")
    if not row["ok"]:
        for tier, v in row["tiers"].items():
            bad = [s for s, st in v["suites"].items() if st != "PASS"]
            if bad:
                print(f"    {tier}: {bad}")
    return row["ok"]


def report():
    if not os.path.isfile(LEDGER):
        print("no ledger yet")
        return 0
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    if not rows:
        print("empty ledger")
        return 0
    first, last = rows[0], rows[-1]
    print(f"{len(rows)} cycles, {first['t']} -> {last['t']}")
    fails = [r for r in rows if not r["ok"]]
    print(f"failing cycles: {len(fails)}")
    for r in fails[-5:]:
        print(f"  {r['t']}: " + ", ".join(
            f"{k}={v['rc']}" for k, v in r["tiers"].items() if v["rc"]))

    print("\nwatched metrics (first -> last):")
    for k, direction in WATCH.items():
        a = first.get("metrics", {}).get(k)
        b = last.get("metrics", {}).get(k)
        if not b:
            print(f"  {k:24s} absent")
            continue
        if not a:
            print(f"  {k:24s} {b['mean']:.6f} (no baseline)")
            continue
        d = b["mean"] - a["mean"]
        verdict = ""
        if direction == "down":
            verdict = "FALLING" if d < 0 else "NOT FALLING"
        elif direction == "alive":
            verdict = "ALIVE" if b["std"] > 1e-9 else "CONSTANT — dead?"
        elif direction == "stable":
            rel = abs(d) / max(abs(a["mean"]), 1e-9)
            verdict = "stable" if rel < 0.5 else "DRIFTED"
        print(f"  {k:24s} {a['mean']:.6f} -> {b['mean']:.6f} "
              f"({d:+.6f})  {verdict}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--tiers", default="unit,contract")
    ap.add_argument("--metrics", default="runlogs/metrics.json")
    a = ap.parse_args()
    if a.report:
        return report()
    tiers = [t for t in a.tiers.split(",") if t]
    if a.once:
        return 0 if cycle(tiers, a.metrics) else 1
    print(f"continuous: {tiers} every {a.interval}s -> {LEDGER}")
    while True:
        try:
            cycle(tiers, a.metrics)
        except Exception as e:
            print(f"[{_now()}] cycle error: {type(e).__name__}: {e}")
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
