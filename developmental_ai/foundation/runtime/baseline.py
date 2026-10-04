"""Baseline measurement from an existing runlogs/ directory (Stage 1 item 5).

Reads only what the live run already writes (training-host layout):

    metrics.jsonl[.N]   one record per segment   (DevelopmentalAI._emit_metrics)
    heartbeat.jsonl[.N] wall-clock running totals (DevelopmentalAI._emit_heartbeat)
    *.log               free text; scanned only for crash signatures

and produces behaviour counters, throughput, memory and world-model error.
Anything the logs do not contain is reported as "unknown" WITH the reason,
never as 0 — a metric that reads zero is indistinguishable from an agent
doing nothing (the `_last_world_info` incident in _emit_metrics).

KNOWN GAP, stated rather than papered over: per-horizon prediction error is
NOT logged anywhere. rssm.compute_loss returns one aggregate "horizon" term,
recorded as training_metrics["horizon_loss"], which _emit_metrics does not
emit. Only aggregate WM losses are available from metrics.jsonl.

Counter semantics are NOT re-validated here (CLAUDE.md §5: "don't build an
argument on a counter you haven't validated"); values are reported as logged.
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

UNKNOWN = "unknown"
REPORT_SCHEMA = 1

BEHAVIOUR_KEYS = ("breaks_total", "logs", "breaks_per_log", "crafts_total",
                  "attack_run_max", "attack_run_mean", "skills_minted",
                  "skills_bound", "skills_refused", "skills_mastered",
                  "episode", "total_timesteps")
WM_KEYS = ("world_model_loss", "reconstruction_error", "kl_divergence",
           "inverse_dynamics_loss")
_HORIZON_RE = re.compile(r"^(wm_)?(err|mse|loss)_h(\d+)$")


def _rotated(path: str) -> List[str]:
    """Oldest first: path.N ... path.1, path."""
    rot = []
    for p in glob.glob(path + ".*"):
        suf = p[len(path) + 1:]
        if suf.isdigit():
            rot.append((int(suf), p))
    out = [p for _, p in sorted(rot, reverse=True)]
    if os.path.isfile(path):
        out.append(path)
    return out


def read_jsonl(paths: Iterable[str]) -> Tuple[List[Dict[str, Any]], int]:
    rows, bad = [], 0
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(r, dict):
                    rows.append(r)
                else:
                    bad += 1
    return rows, bad


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if x == x else None
    return None


def _summ(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "value": UNKNOWN}
    s = sorted(vals)

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return {"n": len(s), "first": vals[0], "last": vals[-1],
            "median": statistics.median(s), "p10": q(0.1), "p90": q(0.9),
            "min": s[0], "max": s[-1]}


def _unknown(reason: str) -> Dict[str, Any]:
    return {"value": UNKNOWN, "reason": reason}


def behaviour(seg: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not seg:
        return {k: _unknown("no metrics.jsonl records") for k in
                BEHAVIOUR_KEYS}
    out: Dict[str, Any] = {}
    for k in BEHAVIOUR_KEYS:
        vals = [v for v in (_num(r.get(k)) for r in seg) if v is not None]
        out[k] = ({"last": vals[-1], "max": max(vals), "n": len(vals)}
                  if vals else _unknown(f"key {k!r} absent from every record"))
    last = seg[-1]
    sh = last.get("reward_shares")
    out["reward_shares_last"] = sh if isinstance(sh, dict) and sh else \
        _unknown("no reward_shares in last record")
    alarms = [a for r in seg for a in (r.get("reward_alarms") or [])]
    out["reward_alarm_count"] = len(alarms)
    hhi = [v for v in (_num(r.get("reward_hhi")) for r in seg) if v is not None]
    out["reward_hhi"] = _summ(hhi) if hhi else _unknown("reward_hhi absent")
    return out


def throughput(seg: List[Dict[str, Any]],
               hb: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    sps = [v for v in (_num(r.get("steps_per_s")) for r in hb)
           if v is not None and v > 0]
    out["heartbeat_steps_per_s"] = _summ(sps) if sps else _unknown(
        "no heartbeat.jsonl steps_per_s")
    rates = []
    for a, b in zip(seg, seg[1:]):
        dt = (_num(b.get("wall_time")) or 0) - (_num(a.get("wall_time")) or 0)
        ds = (_num(b.get("total_timesteps")) or 0) - (
            _num(a.get("total_timesteps")) or 0)
        if dt > 0 and ds > 0:              # a restart makes ds <= 0: skip
            rates.append(ds / dt)
    out["segment_steps_per_s"] = _summ(rates) if rates else _unknown(
        "fewer than two consecutive segment records")
    up = [v for v in (_num(r.get("uptime_s")) for r in seg + hb)
          if v is not None]
    out["max_uptime_s"] = max(up) if up else UNKNOWN
    return out


def memory(seg: List[Dict[str, Any]],
           hb: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    g = [v for v in (_num(r.get("gpu_mem_mb")) for r in hb) if v is not None]
    out["gpu_allocated_mb"] = _summ(g) if g else _unknown(
        "no gpu_mem_mb in heartbeat (CPU run or no heartbeat)")
    rss_keys = sorted({k for r in seg + hb for k in r
                       if "rss" in k.lower() or k.lower().startswith("ram")})
    if rss_keys:
        out["host_ram"] = {k: _summ([v for v in (_num(r.get(k))
                                                 for r in seg + hb)
                                     if v is not None]) for k in rss_keys}
    else:
        out["host_ram"] = _unknown(
            "the run does not log process RSS; measure on the host "
            "(node exporter / ps) — see BASELINE_AUDIT.md")
    return out


def world_model(seg: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in WM_KEYS:
        v = [x for x in (_num(r.get(k)) for r in seg) if x is not None]
        out[k] = _summ(v) if v else _unknown(f"{k} not in metrics.jsonl")
    by_h: Dict[int, List[float]] = {}
    for r in seg:
        for k, v in r.items():
            m = _HORIZON_RE.match(k)
            if m and _num(v) is not None:
                by_h.setdefault(int(m.group(3)), []).append(float(v))
    out["error_by_horizon"] = ({str(h): _summ(v) for h, v in sorted(
        by_h.items())} if by_h else _unknown(
        "per-horizon prediction error is not logged (only the aggregate "
        "rssm 'horizon' loss, which _emit_metrics does not emit)"))
    return out


def failures(log_paths: List[str], seg: List[Dict[str, Any]]) -> Dict[str, Any]:
    tb, oom, timeouts = 0, 0, 0
    for p in log_paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("Traceback (most recent call last)"):
                        tb += 1
                    if "out of memory" in line:
                        oom += 1
                    if "TimeoutError" in line:
                        timeouts += 1
        except OSError:
            continue
    restarts = 0
    for a, b in zip(seg, seg[1:]):
        if (_num(b.get("total_timesteps")) or 0) < (
                _num(a.get("total_timesteps")) or 0):
            restarts += 1
    out: Dict[str, Any] = {"restarts_detected": restarts if seg else UNKNOWN}
    if log_paths:
        out.update({"tracebacks": tb, "oom_lines": oom,
                    "timeout_lines": timeouts,
                    "logs_scanned": len(log_paths)})
    else:
        out.update({k: UNKNOWN for k in ("tracebacks", "oom_lines",
                                         "timeout_lines")})
        out["logs_scanned"] = 0
    return out


def baseline_report(runlogs: str) -> Dict[str, Any]:
    if not os.path.isdir(runlogs):
        raise FileNotFoundError(f"runlogs dir not found: {runlogs}")
    mpaths = _rotated(os.path.join(runlogs, "metrics.jsonl"))
    hpaths = _rotated(os.path.join(runlogs, "heartbeat.jsonl"))
    seg, bad_m = read_jsonl(mpaths)
    hb, bad_h = read_jsonl(hpaths)
    seg.sort(key=lambda r: (_num(r.get("wall_time")) or 0.0))
    hb.sort(key=lambda r: (_num(r.get("wall_time")) or 0.0))
    logs = sorted(glob.glob(os.path.join(runlogs, "*.log")))
    schemas = sorted({str(r.get("schema_version")) for r in seg + hb})
    return {
        "report_schema": REPORT_SCHEMA,
        "runlogs": os.path.abspath(runlogs),
        "inputs": {"metrics_files": mpaths, "heartbeat_files": hpaths,
                   "log_files": logs, "segment_records": len(seg),
                   "heartbeat_records": len(hb),
                   "unparseable_lines": bad_m + bad_h,
                   "schema_versions": schemas or UNKNOWN},
        "behaviour": behaviour(seg),
        "throughput": throughput(seg, hb),
        "memory": memory(seg, hb),
        "world_model": world_model(seg),
        "failures": failures(logs, seg),
        "note": ("counters are reported as logged, not re-validated "
                 "(CLAUDE.md §5)"),
    }


def format_report(rep: Dict[str, Any]) -> str:
    lines = [f"baseline report: {rep['runlogs']}",
             f"  records: {rep['inputs']['segment_records']} segment, "
             f"{rep['inputs']['heartbeat_records']} heartbeat, "
             f"{len(rep['inputs']['log_files'])} log file(s)"]

    def fmt(v: Any) -> str:
        if isinstance(v, dict):
            if v.get("value") == UNKNOWN:
                return f"unknown ({v.get('reason', '')})"
            if "median" in v:
                return (f"median {v['median']:.4g} [p10 {v['p10']:.4g}, "
                        f"p90 {v['p90']:.4g}] n={v['n']}")
            if "last" in v:
                return f"last {v['last']:.4g} max {v['max']:.4g} n={v['n']}"
            return json.dumps(v, default=str)[:160]
        return str(v)

    for sect in ("behaviour", "throughput", "memory", "world_model",
                 "failures"):
        lines.append(f"  [{sect}]")
        for k, v in rep[sect].items():
            if isinstance(v, dict) and v and all(
                    isinstance(x, dict) for x in v.values()) and "value" \
                    not in v:
                lines.append(f"    {k}:")
                for kk, vv in v.items():
                    lines.append(f"      {kk}: {fmt(vv)}")
            else:
                lines.append(f"    {k}: {fmt(v)}")
    lines.append(f"  note: {rep['note']}")
    return "\n".join(lines)
