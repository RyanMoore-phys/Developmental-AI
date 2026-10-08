"""Offline report over runlogs/learning.jsonl + runlogs/position_trace.jsonl.

    PYTHONPATH=. python tools/learning_report.py \
        [--learning runlogs/learning.jsonl] \
        [--positions runlogs/position_trace.jsonl] \
        [--out runlogs/learning_report] [--png] [--run-id ID] [--last N]

Writes <out>/report.md (and, with --png, a few matplotlib PNGs beside it)
from the RAW JSONL -- the system of record -- so it needs neither the
collector, SQLite nor Grafana: copy the two files off main and run it.

WHAT IT LEADS WITH, AND WHY: the scoreboard (logs, breaks) and the
reward-by-source table come first, signed. Every historical "progress" was
the agent finding a way to get paid for doing nothing (CLAUDE.md section 9),
so the report puts income next to `still_frac` / `gui_frac` and exploration
novelty, never the reward total on its own.

Read-only. Torn lines and records of an unknown (schema, v) are skipped and
COUNTED in the report header, never silently dropped. No torch import.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

LEARNING_SCHEMA = ("skybot.learning", 1)
POSITION_SCHEMA = ("skybot.position", 1)


def load_jsonl(path, schema):
    """-> (records, skipped). Missing file -> ([], 0)."""
    recs, skipped = [], 0
    if not path or not os.path.exists(path):
        return recs, skipped
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(r, dict) or (r.get("schema"),
                                           r.get("v")) != schema:
                skipped += 1
                continue
            recs.append(r)
    return recs, skipped


def _num(v):
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _streams(recs, section):
    out = set()
    for r in recs:
        d = r.get(section)
        if isinstance(d, dict):
            out.update(k for k, v in d.items() if isinstance(v, dict))
    return sorted(out)


def _get(r, *path):
    for p in path:
        if not isinstance(r, dict):
            return None
        r = r.get(p)
    return r


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if abs(v) >= 1000 or v == int(v):
        return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}"
    return f"{v:.{nd}g}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def section_scoreboard(recs):
    lines = ["## Scoreboard", ""]
    rows = []
    for s in _streams(recs, "outcomes"):
        last = recs[-1].get("outcomes", {}).get(s, {}) if recs else {}
        logs = _num(last.get("logs_run"))
        brk = _num(last.get("breaks_run"))
        seg_logs = sum(_num(_get(r, "outcomes", s, "logs_segment")) or 0
                       for r in recs)
        rows.append([s, _fmt(brk), _fmt(logs), _fmt(seg_logs),
                     _fmt(brk / logs) if logs else "no log yet"])
    if rows:
        lines += _table(["stream", "breaks_run", "logs_run",
                         "logs (sum of segments)", "breaks per log"], rows)
    else:
        lines.append("_no `outcomes` in these records_")
    return lines + [""]


def section_reward(recs):
    lines = ["## Reward by source (signed sums over the window)", ""]
    for s in _streams(recs, "reward"):
        tot = defaultdict(float)
        for r in recs:
            for k, v in (_get(r, "reward", s) or {}).items():
                f = _num(v)
                if f is not None:
                    tot[k] += f
        gross = sum(abs(v) for v in tot.values()) or 1.0
        rows = [[k, f"{v:+.4g}", f"{100 * abs(v) / gross:.1f}%"]
                for k, v in sorted(tot.items(), key=lambda kv: -abs(kv[1]))]
        lines += [f"**{s}**", ""] + _table(["source", "signed sum",
                                            "share of |income|"], rows) + [""]
    mix = []
    for s in _streams(recs, "reward_mix"):
        for k in ("intrinsic_raw_sum", "intrinsic_after_damp",
                  "intrinsic_after_gui_zero", "mixed_sum"):
            vals = [_num(_get(r, "reward_mix", s, k)) for r in recs]
            vals = [v for v in vals if v is not None]
            if vals:
                mix.append([s, k, f"{sum(vals):+.4g}"])
    if mix:
        lines += ["Intrinsic pipeline (where reward is removed):", ""]
        lines += _table(["stream", "stage", "sum"], mix) + [""]
    return lines


def section_behaviour(recs):
    lines = ["## Behaviour (mean per segment)", ""]
    rows = []
    for s in _streams(recs, "behaviour"):
        def m(k):
            return _mean([_num(_get(r, "behaviour", s, k)) for r in recs])
        gl = [_num(_get(r, "behaviour", s, "gui_longest_run")) for r in recs]
        gl = [g for g in gl if g is not None]
        rows.append([s, _fmt(m("action_entropy_bits")), _fmt(m("gui_frac")),
                     _fmt(max(gl) if gl else None), _fmt(m("still_frac"))])
    if rows:
        lines += _table(["stream", "action entropy (bits)", "gui_frac",
                         "longest GUI run", "still_frac"], rows)
    else:
        lines.append("_no `behaviour` in these records_")
    return lines + [""]


def section_exploration(recs):
    lines = ["## Exploration (ExplorationTracker)", ""]
    rows = []
    for s in _streams(recs, "exploration"):
        def col(k):
            return [_num(_get(r, "exploration", s, k)) for r in recs]
        last = recs[-1].get("exploration", {}).get(s, {})
        ssn = [v for v in col("steps_since_new_cell") if v is not None]
        path = [v for v in col("path_blocks") if v is not None]
        rows.append([s, _fmt(_num(last.get("unique_cells_total"))),
                     _fmt(_mean(col("cells_per_hour"))),
                     _fmt(sum(path) if path else None),
                     _fmt(_mean(col("net_displacement"))),
                     _fmt(_mean(col("radius_of_gyration"))),
                     _fmt(max(ssn) if ssn else None),
                     _fmt(_mean(col("stationary_frac")))])
    if rows:
        lines += _table(["stream", "unique cells (total)", "cells/hour",
                         "path (blocks)", "mean displacement",
                         "mean radius", "max steps since new cell",
                         "stationary_frac"], rows)
    else:
        lines.append("_no `exploration` in these records_")
    return lines + [""]


def section_ml(recs):
    lines = ["## ML health (last value / mean over window)", ""]
    rows = []
    for sec in ("ppo", "wm", "curiosity", "replay", "timing", "health"):
        keys = sorted({k for r in recs if isinstance(r.get(sec), dict)
                       for k, v in r[sec].items() if _num(v) is not None})
        for k in keys:
            vals = [_num(_get(r, sec, k)) for r in recs]
            vals = [v for v in vals if v is not None]
            agg = (sum(vals) if sec == "health" else _mean(vals))
            rows.append([f"{sec}.{k}", _fmt(vals[-1]), _fmt(agg),
                         "sum" if sec == "health" else "mean"])
    if rows:
        lines += _table(["key", "last", "window", "agg"], rows)
    else:
        lines.append("_no ML producers reported (all null)_")
    return lines + [""]


def section_positions(pos):
    lines = ["## Position trace (EVALUATOR-ONLY)", ""]
    by = defaultdict(list)
    for p in pos:
        if None not in (_num(p.get("x")), _num(p.get("z"))):
            by[int(p.get("stream", 0))].append(p)
    rows = []
    for s, ps in sorted(by.items()):
        ps.sort(key=lambda p: (p.get("t_wall") or 0))
        path = 0.0
        for a, b in zip(ps, ps[1:]):
            if a.get("episode") != b.get("episode"):
                continue                      # a reset is not movement
            d = math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
            path += d
        ys = [p["y"] for p in ps]
        rows.append([f"stream-{s}", len(ps),
                     len({p.get("episode") for p in ps}),
                     f"{min(p['x'] for p in ps):.0f}..{max(p['x'] for p in ps):.0f}",
                     f"{min(p['z'] for p in ps):.0f}..{max(p['z'] for p in ps):.0f}",
                     f"{min(ys):.0f}..{max(ys):.0f}", _fmt(path)])
    if rows:
        lines += _table(["stream", "points", "episodes", "x range",
                         "z range", "y range", "trace path (sampled)"], rows)
        lines += ["", "Trace path undercounts true path: it joins points "
                      "sampled every `trace_every_steps`."]
    else:
        lines.append("_no position trace_")
    return lines + [""]


def write_pngs(recs, pos, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                  # optional by design
        print(f"learning_report: --png skipped ({exc!r})", file=sys.stderr)
        return []
    made = []
    t = [r.get("wall_time") for r in recs]
    for s in _streams(recs, "reward"):
        srcs = sorted({k for r in recs for k in (_get(r, "reward", s) or {})})
        fig, ax = plt.subplots(figsize=(10, 4))
        for k in srcs:
            ax.plot(t, [_num(_get(r, "reward", s, k)) for r in recs], label=k)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"reward by source (signed), {s}")
        ax.legend(fontsize=7, ncol=3)
        p = os.path.join(out, f"reward_{s}.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    ex = _streams(recs, "exploration")
    if ex:
        fig, ax = plt.subplots(figsize=(10, 4))
        for s in ex:
            ax.plot(t, [_num(_get(r, "exploration", s,
                                  "steps_since_new_cell")) for r in recs],
                    label=s)
        ax.axhline(5000, color="tab:red", ls="--", lw=0.8,
                   label="stuck alert (5000)")
        ax.set_title("steps since a new cell")
        ax.legend(fontsize=8)
        p = os.path.join(out, "steps_since_new_cell.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    if pos:
        by = defaultdict(list)
        for q in pos:
            by[int(q.get("stream", 0))].append(q)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        for s, ps in sorted(by.items()):
            ps.sort(key=lambda q: q.get("t_wall") or 0)
            a1.scatter([q["x"] for q in ps], [q["z"] for q in ps], s=3,
                       label=f"stream-{s}")
            a2.plot([q["t_wall"] for q in ps], [q["y"] for q in ps],
                    label=f"stream-{s}")
        a1.set_xlabel("x")
        a1.set_ylabel("z")
        a1.set_title("top-down position (episodes are different worlds)")
        a1.legend(fontsize=8)
        a2.set_title("y over time")
        a2.legend(fontsize=8)
        p = os.path.join(out, "positions.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made


def build(learning, positions, run_id=None, last=None):
    recs, skip_l = load_jsonl(learning, LEARNING_SCHEMA)
    pos, skip_p = load_jsonl(positions, POSITION_SCHEMA)
    if run_id:
        recs = [r for r in recs if r.get("run_id") == run_id]
        pos = [p for p in pos if p.get("run_id") == run_id]
    uniq = {}
    for r in recs:                            # same seq = same record
        if isinstance(r.get("seq"), int):
            uniq[r["seq"]] = r
    recs = [uniq[k] for k in sorted(uniq)]
    if last:
        recs = recs[-int(last):]
        if recs and pos:
            t0 = recs[0].get("wall_time") or 0
            pos = [p for p in pos if (p.get("t_wall") or 0) >= t0 - 1]
    lines = ["# SkyBot learning report", ""]
    if recs:
        t0, t1 = recs[0].get("wall_time"), recs[-1].get("wall_time")
        span = (t1 - t0) / 3600 if (t0 and t1) else None
        lines += [f"- records: {len(recs)} (seq {recs[0]['seq']}.."
                  f"{recs[-1]['seq']}), span {_fmt(span)} h",
                  f"- run_ids: {sorted({str(r.get('run_id')) for r in recs})}",
                  f"- total_timesteps (last): "
                  f"{_fmt(_num(recs[-1].get('total_timesteps')))}"]
    else:
        lines.append("- **no learning records** in the window")
    lines += [f"- skipped lines (torn / unknown schema): learning {skip_l}, "
              f"positions {skip_p}", ""]
    if recs:
        for fn in (section_scoreboard, section_reward, section_behaviour,
                   section_exploration, section_ml):
            lines += fn(recs)
    lines += section_positions(pos)
    return recs, pos, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--learning", default="runlogs/learning.jsonl")
    ap.add_argument("--positions", default="runlogs/position_trace.jsonl")
    ap.add_argument("--out", default="runlogs/learning_report")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--last", type=int, default=None)
    a = ap.parse_args(argv)
    recs, pos, lines = build(a.learning, a.positions, a.run_id, a.last)
    os.makedirs(a.out, exist_ok=True)
    if a.png:
        made = write_pngs(recs, pos, a.out)
        if made:
            lines += ["## Figures", ""] + [f"![{os.path.basename(p)}]"
                                           f"({os.path.basename(p)})"
                                           for p in made] + [""]
    path = os.path.join(a.out, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
