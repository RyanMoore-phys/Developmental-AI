"""Read-only report over a ShadowRecorder EvidenceStore (runtime/shadow.py).

    PYTHONPATH=. python tools/shadow_report.py <store_dir> [--out DIR]
            [--run <run_id|latest|all>]
        (default --out runlogs/shadow_report/) -> DIR/REPORT.md, DIR/report.json,
        DIR/ab/ (the A/B harness's own JSON + markdown, when section D runs)

RUNS (--run, 2026-10-05)
    The live store accumulates every launch; each ShadowRecorder has a run id
    and names episodes "<run_id>-e<N>". REPORT.md / report.json open with the
    runs found (id, first/last time, sampled steps). --run all (default) reads
    everything exactly as before; --run latest / <id> wraps the store ONCE in
    RunFilteredStore, a read-only proxy whose dev/held-out/evaluator views hide
    every record of another run (interpretations, which carry no episode, go
    with the record they target), so A-E all see the same run. BEFORE/AFTER a
    change: run the tool twice with different --out dirs, one per run id.

WHAT IT READS, AND HOW
    The store is opened `readonly=True`: no recovery write, no quarantine
    move, no manifest rewrite. A torn trailing journal line or a damaged
    sealed chunk (copying a LIVE store) is tolerated by the store's own
    recovery, which in readonly mode only NOTES what it would quarantine;
    this tool counts those notes and reports the records lost.
    tests/_foundation_shadow_report_smoke.py proves the directory is
    byte-identical before and after.

THE PARTITION RULE (plan §7.4: RED never reaches a learning path)
    A, B, D, E read ONLY the dev and held-out partitions, through
    `learning_view()` / `heldout_view()`. C reads ONLY records whose ref is in
    the evaluator partition. D fits on the learning view (dev) and scores on
    the held-out view, by the store's own episode split. A counts evaluator
    records by kind (from store.stats(), no decode) and nothing else.

SECTIONS
    A  inventory        runs, span, streams, episodes, records by kind and
                        partition, end reasons, sampled steps vs predictions,
                        model versions, action timing
    B  behaviour        action histogram per stream, and split by ACTION
                        SOURCE (Action payload "source": policy /
                        option_slot:<n> with the acting skill / dream_actor /
                        option_fallback; stores written before 2026-10-04
                        carry no source and land in "unrecorded"); reward channels
                        (reward_env, reward_replay = prim_extrinsic on stream-0
                        ONLY - the recorder receives the primary stream's
                        scalar - intrinsic); GUI-open fraction and run lengths
                        (proprio[5] = gui_open when proprio has the 13-field
                        PROPRIO_FIELDS layout); stillness of the GREEN vector
                        transport between t and t+1; what each reward channel
                        pays while still / while a GUI is open (CLAUDE.md §9)
    C  EVALUATOR ONLY   true_position: unique block cells (and per hour),
                        displacement per episode, stationary fraction, vertical
                        range
    D  mechanisms       plan Stage 6/7 on real data through experiments/ab.py,
                        with a Preregistration frozen BEFORE any data is read
                        for D; "insufficient data" instead of a verdict below
                        MIN_PAIRS valid pairs / MIN_EPISODES episodes / an
                        empty held-out side. An exploratory chronological
                        within-episode split (NO verdict, labelled) is added
                        when the preregistered comparison cannot run.
    E  RSSM prior       entropy of the logged prior probabilities over time and
                        by model version
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.foundation.contracts import ABSENT, UNKNOWN  # noqa: E402
from developmental_ai.foundation.contracts.streams import transition_problems  # noqa: E402
from developmental_ai.foundation.experience import EvidenceStore  # noqa: E402
from developmental_ai.foundation.runtime.assumptions import PROPRIO_FIELDS  # noqa: E402

SMALL_WIDTH = 16           # widest GREEN vector channel D may use
STILL_ATOL = 1e-4          # |x_{t+1} - x_t| at or below this = unchanged
VISUAL_STILL = 0.01        # mean fovea_delta below this = view unchanged
STATIONARY_BLOCKS = 0.05   # |true_position step| below this = stationary
MIN_PAIRS = 300
MIN_EPISODES = 6
MIN_HELDOUT_PAIRS = 30
SEEDS = (0, 1, 2)
GROUP_BOOTSTRAP_MIN = 8
PATIENCE = 5               # D early stopping: epochs without validation gain (preregistered)
BATCH_SIZE = 128           # D neural arms' minibatch size
EVENT_CLASSES = ("none", "event")
GUI_INDEX = PROPRIO_FIELDS.index("gui_open")
UNRECORDED = "unrecorded"  # action source of a store written before 2026-10-04


# ------------------------------------------------------------------ helpers
def _f(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _f(o)
    if o is ABSENT or o is UNKNOWN:
        return str(o)
    return o


def _spearman(a, b) -> Optional[float]:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    from scipy.stats import spearmanr
    return _f(spearmanr(a, b).correlation)


def _stats(x) -> Dict[str, Any]:
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "p50": float(np.median(x)),
            "max": float(x.max())}


def _run_of(episode: str) -> str:
    return episode.rsplit("-e", 1)[0] if "-e" in episode else episode


def _fmt(x, nd=4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def _ts(t) -> str:
    return "-" if t is None else time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                               time.gmtime(float(t)))


# ------------------------------------------------------------- store access
def open_store(path: str) -> EvidenceStore:
    if not os.path.exists(os.path.join(path, "STORE.json")):
        raise FileNotFoundError(f"no evidence store at {path} (STORE.json missing)")
    return EvidenceStore(path, readonly=True)


def integrity(store: EvidenceStore) -> Dict[str, Any]:
    """What readonly recovery found: notes plus quarantined record counts."""
    lines = chunks_lost = missing = 0
    for n in store.recovery:
        m = re.search(r"(\d+) damaged/truncated line\(s\) quarantined", n)
        if m:
            lines += int(m.group(1))
        m = re.search(r"(\d+)/(\d+) lines salvaged", n)
        if m:
            chunks_lost += int(m.group(2)) - int(m.group(1))
        m = re.search(r"missing on disk \((\d+) records lost\)", n)
        if m:
            missing += int(m.group(1))
    st = store.stats()
    return {"readonly": True, "recovery_notes": list(store.recovery),
            "quarantined_journal_lines": lines,
            "quarantined_chunk_records": chunks_lost,
            "manifest_records_missing_on_disk": missing,
            "quarantined_total": lines + chunks_lost + missing,
            "disk_bytes": st["disk_bytes"], "max_bytes": st["max_bytes"],
            "evicted": st["evicted"], "store_id": store.ident.get("store_id"),
            "split": store.ident.get("split")}


# ------------------------------------------------------------- run filter
def _ref_run(ev_view, ref: str, meta: Dict[str, Any]) -> Optional[str]:
    """The run a record belongs to. observation / action / prediction /
    evidence / episode_end carry `episode` in their index meta; an
    interpretation carries only `target`, so it belongs to its target's run
    (None when the target is gone - evicted - so it cannot be attributed)."""
    if "episode" in meta:
        return _run_of(str(meta["episode"]))
    tgt = meta.get("target")
    if isinstance(tgt, str):
        try:
            return _ref_run(ev_view, tgt, ev_view.meta(tgt))
        except Exception:
            return None
    return None


def list_runs(store) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """Every run in the store (index meta only, no decode) and each ref's run.
    t_first / t_last span every timestamp in the run's metas; sampled_steps
    is its action-record count."""
    ev = store.evaluator_view()
    attr: Dict[str, Optional[str]] = {}
    runs = collections.defaultdict(lambda: {"t_first": math.inf, "t_last": -math.inf,
                                            "sampled_steps": 0, "records": 0,
                                            "episodes": set(), "streams": set()})
    for ref in ev.refs():
        m = ev.meta(ref)
        rid = attr[ref] = _ref_run(ev, ref, m)
        if rid is None:
            continue
        R = runs[rid]
        R["records"] += 1
        if "stream" in m:
            R["streams"].add(m["stream"])
            R["episodes"].add((m["stream"], m.get("episode")))
        for key in ("t_wall", "t_dispatch", "t_complete"):
            if isinstance(m.get(key), (int, float)) and math.isfinite(m[key]):
                R["t_first"] = min(R["t_first"], float(m[key]))
                R["t_last"] = max(R["t_last"], float(m[key]))
        if ev.kind(ref) == "action":
            R["sampled_steps"] += 1
    out = []
    for rid, R in runs.items():
        ok = math.isfinite(R["t_first"])
        out.append({"run": rid, "t_first": R["t_first"] if ok else None,
                    "t_last": R["t_last"] if ok else None,
                    "first": _ts(R["t_first"]) if ok else None,
                    "last": _ts(R["t_last"]) if ok else None,
                    "sampled_steps": R["sampled_steps"], "records": R["records"],
                    "episodes": len(R["episodes"]), "streams": sorted(R["streams"])})
    out.sort(key=lambda r: (r["t_first"] is None, r["t_first"] or 0.0, r["run"]))
    return out, attr


def resolve_run(runs: List[Dict[str, Any]], want: str) -> Optional[str]:
    """`all` -> None (no filter); `latest` -> the run that STARTED last;
    otherwise an exact run id present in the store."""
    if want in (None, "", "all"):
        return None
    if not runs:
        raise ValueError("the store has no attributable runs")
    if want == "latest":
        timed = [r for r in runs if r["t_first"] is not None]
        return max(timed or runs, key=lambda r: (r["t_first"] or 0.0, r["run"]))["run"]
    ids = [r["run"] for r in runs]
    if want not in ids:
        raise ValueError(f"run {want!r} not in the store; runs found: {ids}")
    return want


class _RunView:
    """Read-only proxy over one store view that HIDES every record not
    attributed to the selected run. Exposes only what the report reads
    (refs / kind / meta / get); a hidden ref raises KeyError exactly like a
    ref outside the view's partitions."""

    def __init__(self, view, visible: frozenset, run_id: str):
        self._v, self._vis, self._run = view, visible, run_id

    @property
    def partitions(self):
        return self._v.partitions

    def _check(self, ref: str):
        if ref not in self._vis:
            raise KeyError(f"{ref} is not in run {self._run!r} (run-filtered view)")

    def refs(self, kind: Optional[str] = None) -> List[str]:
        return [r for r in self._v.refs(kind) if r in self._vis]

    def kind(self, ref: str) -> str:
        self._check(ref)
        return self._v.kind(ref)

    def meta(self, ref: str) -> Dict[str, Any]:
        self._check(ref)
        return self._v.meta(ref)

    def get(self, ref: str):
        self._check(ref)
        return self._v.get(ref)

    def __len__(self) -> int:
        return len(self.refs())

    def __getattr__(self, name):
        raise AttributeError(f"{name!r} is not supported by the run-filtered view "
                             f"(add it to _RunView so it filters too)")


class RunFilteredStore:
    """The ONE place the --run filter lives: every section reads the store
    through this, so A-E see the same records. Views are _RunView proxies;
    stats() reports the selected run's per-partition counts (index only, no
    decode); split parameters pass through. Never writes."""

    def __init__(self, store, run_id: str, attr: Dict[str, Optional[str]]):
        self._s, self.run_id = store, run_id
        self._vis = frozenset(r for r, rid in attr.items() if rid == run_id)
        self.heldout_fraction = store.heldout_fraction
        self.split_salt = store.split_salt
        self.ident = store.ident
        self.recovery = store.recovery

    def learning_view(self):
        return _RunView(self._s.learning_view(), self._vis, self.run_id)

    def heldout_view(self):
        return _RunView(self._s.heldout_view(), self._vis, self.run_id)

    def evaluator_view(self):
        return _RunView(self._s.evaluator_view(), self._vis, self.run_id)

    def stats(self) -> Dict[str, Any]:
        st = dict(self._s.stats())
        ev = self._s.evaluator_view()
        per: Dict[str, Dict[str, Any]] = {}
        for ref in sorted(self._vis):
            part = ref.split("/", 1)[0]
            d = per.setdefault(part, {"records": 0, "by_kind": collections.Counter()})
            d["records"] += 1
            d["by_kind"][ev.kind(ref)] += 1
        st["partitions"] = {p: {"records": d["records"], "by_kind": dict(d["by_kind"])}
                            for p, d in per.items()}
        st["run_filter"] = self.run_id
        return st


def _green_views(store):
    return {"dev": store.learning_view(), "heldout": store.heldout_view()}


def load_green(store) -> Dict[str, Any]:
    """Every non-evaluator record the report needs, from dev + held-out
    views only. Evidence rows carry obs t / obs t+1 vector channels."""
    views = _green_views(store)
    out = {"meta": [], "rows": [], "preds": [], "image_obs": {}, "fovea_delta": [],
           "actions": []}
    for part, v in views.items():
        for ref in v.refs():
            k = v.kind(ref)
            m = v.meta(ref)
            out["meta"].append((part, ref, k, m))
            if k == "action":
                # WHO chose it (payload "source", optional since 2026-10-04:
                # an older store has none and reads as "unrecorded")
                pay = dict(getattr(v.get(ref), "payload", None) or {})
                out["actions"].append({
                    "stream": m["stream"], "command": m["command"],
                    "source": str(pay.get("source", UNRECORDED)),
                    "skill": pay.get("skill")})
    for part, v in views.items():
        for ref in v.refs("evidence"):
            ev = v.get(ref)
            if ev.provenance != "sensor" or not ev.actions:
                continue
            act = ev.actions[0]
            t0, t1, probs = {}, {}, []
            pre, nxt = {}, {}
            for o in ev.observations:
                if o.seq == act.seq:
                    pre[o.channel] = o
                elif o.seq == act.seq + 1:
                    nxt[o.channel] = o
            for ch, o in pre.items():
                n = nxt.get(ch)
                if n is None:
                    probs.append(f"{ch}: no consequence observation")
                    continue
                probs += [f"{ch}: {p}" for p in transition_problems(o, n)]
                if o.value is ABSENT or n.value is ABSENT:
                    continue
                t0[ch] = np.asarray(o.value, np.float64)
                t1[ch] = np.asarray(n.value, np.float64)
            out["rows"].append({
                "part": part, "stream": ev.scope.stream, "episode": ev.scope.episode,
                "env": ev.scope.environment, "seq": int(act.seq),
                "t_wall": min(o.t_wall for o in pre.values()) if pre else act.t_dispatch,
                "spec": act.spec_id, "command": act.command,
                "pay": dict(ev.payload), "x0": t0, "x1": t1,
                "pids": tuple(ev.prediction_refs), "problems": probs})
        for ref in v.refs("prediction"):
            p = v.get(ref)
            m = v.meta(ref)
            probs_ = p.outcome.get("probs") if isinstance(p.outcome, dict) else None
            out["preds"].append({
                "part": part, "pid": p.prediction_id, "stream": m["stream"],
                "episode": m["episode"], "t_wall": float(p.t_wall),
                "version": str(p.model_versions.get("world_model.rssm",
                                                    next(iter(p.model_versions.values()), "unknown"))),
                "kind": p.outcome.get("kind") if isinstance(p.outcome, dict) else None,
                "probs": None if probs_ is None else np.asarray(probs_, np.float64)})
    # image channels: names + shape from one decode each; fovea_delta values
    # at t+1 (it IS |fovea_t+1 - fovea_t|) for the visual stillness proxy.
    vec_ch = {ch for r in out["rows"] for ch in r["x0"]}
    for part, v in views.items():
        for ref in v.refs("observation"):
            m = v.meta(ref)
            ch = m["channel"]
            if ch in vec_ch:
                continue
            if ch not in out["image_obs"]:
                val = v.get(ref).value
                out["image_obs"][ch] = (list(np.shape(val)) if val is not ABSENT
                                        else "ABSENT")
            if ch == "fovea_delta":
                val = v.get(ref).value
                if val is not ABSENT:
                    out["fovea_delta"].append((m["stream"], m["episode"], m["oseq"],
                                               float(np.mean(np.abs(val)))))
    for r in out["rows"]:
        r["valid"] = not r["problems"]
    out["rows"].sort(key=lambda r: (r["stream"], r["episode"], r["seq"]))
    return out


# ----------------------------------------------------------------- A
def section_a(store, g) -> Dict[str, Any]:
    kinds = collections.defaultdict(collections.Counter)
    streams, episodes = set(), collections.defaultdict(set)
    runs = collections.defaultdict(lambda: {"t_lo": math.inf, "t_hi": -math.inf,
                                            "episodes": set(), "streams": set(),
                                            "max_seq": collections.Counter()})
    env_names, ends, elapsed = set(), [], []
    acts_by_stream = collections.Counter()
    channels = collections.Counter()
    for part, ref, k, m in g["meta"]:
        kinds[part][k] += 1
        if "stream" not in m:
            continue
        streams.add(m["stream"])
        env_names.add(m.get("env"))
        episodes[m["stream"]].add(m["episode"])
        R = runs[_run_of(m["episode"])]
        R["episodes"].add((m["stream"], m["episode"]))
        R["streams"].add(m["stream"])
        for key in ("t_wall", "t_dispatch", "t_complete"):
            if isinstance(m.get(key), (int, float)):
                R["t_lo"] = min(R["t_lo"], m[key])
                R["t_hi"] = max(R["t_hi"], m[key])
        if isinstance(m.get("oseq"), int):
            ek = (m["stream"], m["episode"])
            R["max_seq"][ek] = max(R["max_seq"][ek], m["oseq"])
        if k == "observation":
            channels[m["channel"]] += 1
        elif k == "action":
            acts_by_stream[m["stream"]] += 1
            elapsed.append(m["elapsed"])
        elif k == "episode_end":
            ends.append({"stream": m["stream"], "episode": m["episode"],
                         "reason": m["reason"], "last_seq": m["oseq"]})
    st = store.stats()
    ev_counts = st["partitions"].get("evaluator", {}).get("by_kind", {})
    pred_ids = {p["pid"] for p in g["preds"]}
    resolved = {pid for r in g["rows"] for pid in r["pids"]} & pred_ids
    n_act = sum(acts_by_stream.values())
    versions = collections.Counter(p["version"] for p in g["preds"])
    run_out = {}
    for rid, R in sorted(runs.items()):
        span = (R["t_hi"] - R["t_lo"]) if math.isfinite(R["t_lo"]) else 0.0
        steps = sum(v + 1 for v in R["max_seq"].values())
        run_out[rid] = {"start": _ts(R["t_lo"]) if math.isfinite(R["t_lo"]) else None,
                        "end": _ts(R["t_hi"]) if math.isfinite(R["t_hi"]) else None,
                        "hours": span / 3600.0, "streams": sorted(R["streams"]),
                        "episodes": len(R["episodes"]),
                        "agent_steps_covered": steps,
                        "steps_per_s_per_stream": (steps / len(R["streams"]) / span
                                                   if span > 0 and R["streams"] else None)}
    ends_reasons = collections.Counter(e["reason"] for e in ends)
    n_eps = sum(len(v) for v in episodes.values())
    seq_gaps = []
    by_ep = collections.defaultdict(list)
    for r in g["rows"]:
        by_ep[(r["stream"], r["episode"])].append(r["seq"])
    for s in by_ep.values():
        seq_gaps += list(np.diff(sorted(s)))
    return {
        "environments": sorted(e for e in env_names if e),
        "runs": run_out, "streams": sorted(streams),
        "episodes_per_stream": {s: len(v) for s, v in sorted(episodes.items())},
        "episodes_total": n_eps,
        "records_by_partition_kind": {p: dict(c) for p, c in kinds.items()},
        "evaluator_partition_counts": dict(ev_counts),
        "observation_channels": dict(channels),
        "image_channels": g["image_obs"],
        "episode_ends": dict(ends_reasons),
        "episodes_without_end": n_eps - len(ends),
        "sampled_steps": n_act,
        "sampled_steps_per_stream": dict(acts_by_stream),
        "sampled_seq_gap_median": (float(np.median(seq_gaps)) if seq_gaps else None),
        "outcomes": len(g["rows"]),
        "valid_transitions": sum(r["valid"] for r in g["rows"]),
        "predictions": len(pred_ids),
        "prediction_coverage": (len(pred_ids) / n_act) if n_act else None,
        "predictions_resolved": len(resolved),
        "predictions_unresolved": len(pred_ids - resolved),
        "model_versions": dict(versions.most_common(20)),
        "model_versions_unique": len(versions),
        "action_elapsed_s": _stats(elapsed),
    }


# ----------------------------------------------------------------- B
def _cmd_key(c) -> str:
    if isinstance(c, (int, np.integer)):
        return str(int(c))
    return str(np.round(np.asarray(c, np.float64), 2).tolist())


def _runs(flags: Sequence[bool]) -> List[int]:
    out, n = [], 0
    for f in flags:
        if f:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def _hist(cmds) -> Dict[str, Any]:
    c = collections.Counter(cmds)
    n = sum(c.values())
    p = np.array([v / n for v in c.values()])
    ent = float(-(p * np.log2(p)).sum()) if n else None
    return {"n": n, "distinct": len(c), "entropy_bits": ent,
            "top": [(k, v, v / n) for k, v in c.most_common(10)],
            "histogram": dict(sorted(c.items(), key=lambda kv: (
                len(kv[0]), kv[0])))}


def _source_key(a) -> str:
    """policy / option_slot:<n> [skill] / dream_actor / ... / unrecorded."""
    if a["source"].startswith("option_slot:") and a.get("skill"):
        return f"{a['source']} [{a['skill']}]"
    return a["source"]


def section_b(g) -> Dict[str, Any]:
    rows = g["rows"]
    out: Dict[str, Any] = {"actions": {}, "actions_by_source": {},
                           "rewards": {}, "gui": None,
                           "stillness": {}, "pay_while": {}}
    by_stream = collections.defaultdict(list)
    for r in rows:
        by_stream[r["stream"]].append(r)
    # Histogram over every EXECUTED action record (an action whose outcome
    # was lost to a client recovery has no Evidence but was still taken).
    act_cmds = collections.defaultdict(list)
    for _part, _ref, k, m in g["meta"]:
        if k == "action":
            act_cmds[m["stream"]].append(m["command"])
    for s, cmds in sorted(act_cmds.items()):
        out["actions"][s] = _hist(cmds)
    # ... and split by WHO chose it: the shared policy, or an option slot
    # running a frozen skill copy (CLAUDE.md §4.6 — a skill is a byte copy
    # of the policy, so an action histogram that pools both cannot say which
    # one is stuck). Keyed on slot + skill_id, never on a skill's name.
    by_src = collections.defaultdict(lambda: collections.defaultdict(list))
    for a in g.get("actions", []):
        by_src[a["stream"]][_source_key(a)].append(a["command"])
    for s, d in sorted(by_src.items()):
        out["actions_by_source"][s] = {k: _hist(v) for k, v in sorted(d.items())}
    # rewards
    for s, rs in sorted(by_stream.items()):
        ch = {"reward_env": [r["pay"].get("reward_env") for r in rs]}
        if s == "stream-0":
            ch["prim_extrinsic"] = [r["pay"].get("reward_replay") for r in rs]
            ch["shaping (prim_extrinsic - env)"] = [
                (r["pay"]["reward_replay"] - r["pay"]["reward_env"])
                if r["pay"].get("reward_replay") is not None else None for r in rs]
        if any("intrinsic" in r["pay"] for r in rs):
            ch["intrinsic"] = [r["pay"].get("intrinsic") for r in rs]
        d = {}
        parts = {}
        for name, vals in ch.items():
            v = np.array([x for x in vals if x is not None], np.float64)
            d[name] = {"n": int(v.size), "mean": _f(v.mean()) if v.size else None,
                       "sum": _f(v.sum()), "frac_nonzero": _f((v != 0).mean()) if v.size else None,
                       "min": _f(v.min()) if v.size else None,
                       "max": _f(v.max()) if v.size else None}
            if name != "prim_extrinsic":          # disjoint income parts
                parts[name] = float(np.abs(v).sum())
        tot = sum(parts.values())
        for name in parts:
            d[name]["share_of_abs_income"] = parts[name] / tot if tot > 0 else None
        out["rewards"][s] = d
    # GUI
    gui_rows = [r for r in rows if "proprio" in r["x0"]
                and r["x0"]["proprio"].shape == (len(PROPRIO_FIELDS),)]
    if gui_rows:
        flags = collections.defaultdict(list)
        for r in gui_rows:
            flags[(r["stream"], r["episode"])].append(
                (r["seq"], bool(r["x0"]["proprio"][GUI_INDEX] > 0.5)))
        lens, per = [], {}
        for (s, e), lst in sorted(flags.items()):
            lst.sort()
            fl = [f for _, f in lst]
            rl = _runs(fl)
            lens += rl
            per.setdefault(s, []).extend(fl)
        gaps = [b - a for lst in flags.values() for (a, _), (b, _) in
                zip(sorted(lst), sorted(lst)[1:])]
        gap = float(np.median(gaps)) if gaps else None
        hist = collections.Counter(lens)
        out["gui"] = {
            "source": f"proprio[{GUI_INDEX}] (gui_open) at obs t of each sampled step",
            "fraction_open": {s: float(np.mean(v)) for s, v in per.items()},
            "fraction_open_all": float(np.mean([f for v in per.values() for f in v])),
            "runs": len(lens),
            "run_length_sampled_points": _stats(lens),
            "run_length_hist": dict(sorted(hist.items())),
            "sampled_point_gap_steps": gap,
            "longest_run_approx_steps": (max(lens) * gap) if (lens and gap) else None,
            "note": "runs are counted over SAMPLED points (every_n_steps apart); "
                    "a run of k points spans ~k x gap agent steps"}
    else:
        out["gui"] = {"available": False,
                      "reason": "no proprio channel with the 13-field PROPRIO_FIELDS "
                                "layout in the recorded transport"}
    # stillness
    vec_ch = sorted({ch for r in rows for ch in r["x0"] if r["x0"][ch].ndim == 1})
    still_any, per_ch = [], {c: [] for c in vec_ch}
    for r in rows:
        if not r["valid"]:
            continue
        mx = 0.0
        for c in vec_ch:
            if c in r["x0"]:
                d = float(np.max(np.abs(r["x1"][c] - r["x0"][c]))) if r["x0"][c].size else 0.0
                per_ch[c].append(d <= STILL_ATOL)
                mx = max(mx, d)
        r["_still"] = mx <= STILL_ATOL
        still_any.append(r["_still"])
    out["stillness"] = {
        "definition": f"every GREEN vector channel unchanged (|dx| <= {STILL_ATOL}) "
                      f"from obs t to obs t+1",
        "channels": vec_ch,
        "fraction_transport_unchanged": float(np.mean(still_any)) if still_any else None,
        "per_channel_unchanged": {c: float(np.mean(v)) for c, v in per_ch.items() if v}}
    fd = g["fovea_delta"]
    if fd:
        vals = np.array([x[3] for x in fd])
        out["stillness"]["visual"] = {
            "source": "fovea_delta observations (mean |fovea_t - fovea_t-1|)",
            "n": int(vals.size), "mean": float(vals.mean()),
            f"fraction_below_{VISUAL_STILL}": float((vals < VISUAL_STILL).mean())}
    # what the reward channels pay while still / while a GUI is open
    s0 = [r for r in by_stream.get("stream-0", []) if r["valid"]]
    conds = {"still": lambda r: r.get("_still", False),
             "moving": lambda r: not r.get("_still", False)}
    if gui_rows:
        conds["gui_open"] = lambda r: ("proprio" in r["x0"] and
                                       r["x0"]["proprio"].shape == (len(PROPRIO_FIELDS),)
                                       and r["x0"]["proprio"][GUI_INDEX] > 0.5)
    for name, fn in conds.items():
        sub = [r for r in s0 if fn(r)]
        d = {"n": len(sub)}
        for key, lab in (("reward_env", "env"), ("reward_replay", "prim_extrinsic"),
                         ("intrinsic", "intrinsic")):
            v = [r["pay"][key] for r in sub if key in r["pay"]]
            d[lab + "_mean"] = _f(np.mean(v)) if v else None
        out["pay_while"][name] = d
    out["pay_while"]["note"] = ("stream-0 only (prim_extrinsic is the primary "
                                "stream's scalar). intrinsic is the value passed "
                                "to the recorder by the loop")
    return out


# ----------------------------------------------------------------- C
def section_c(store) -> Dict[str, Any]:
    """EVALUATOR-ONLY. Reads records of the evaluator partition and nothing else."""
    ev = store.evaluator_view()
    pts = collections.defaultdict(dict)
    absent = 0
    channels = collections.Counter()
    for ref in ev.refs("observation"):
        if not ref.startswith("evaluator/"):
            continue
        m = ev.meta(ref)
        channels[m["channel"]] += 1
        if m["channel"] != "true_position":
            continue
        val = ev.get(ref).value
        if val is ABSENT or val is UNKNOWN:
            absent += 1
            continue
        v = np.asarray(val, np.float64).reshape(-1)
        if v.size != 3:
            absent += 1
            continue
        pts[(m["stream"], m["episode"])][m["oseq"]] = (m["t_wall"], v)
    if not pts:
        return {"available": False, "evaluator_channels": dict(channels),
                "reason": "no true_position readings in the evaluator partition"}
    per_ep, per_stream = {}, collections.defaultdict(lambda: {
        "cells": set(), "cols": set(), "t_lo": math.inf, "t_hi": -math.inf,
        "steps": [], "y": []})
    all_cells, still_all, ys = set(), [], []
    bbox_lo, bbox_hi = np.full(3, np.inf), np.full(3, -np.inf)
    for (s, e), d in sorted(pts.items()):
        seqs = sorted(d)
        P = np.stack([d[q][1] for q in seqs])
        T = np.array([d[q][0] for q in seqs])
        cells = {tuple(c) for c in np.floor(P).astype(np.int64).tolist()}
        cols = {(c[0], c[2]) for c in cells}
        steps = [float(np.linalg.norm(d[q + 1][1] - d[q][1]))
                 for q in seqs if q + 1 in d]
        still = [x < STATIONARY_BLOCKS for x in steps]
        hz = P[:, [0, 2]]
        path = float(np.linalg.norm(np.diff(hz, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
        hours = (T.max() - T.min()) / 3600.0
        per_ep[f"{s}|{e}"] = {
            "points": len(seqs), "hours": hours,
            "unique_cells": len(cells), "unique_columns_xz": len(cols),
            "unique_cells_per_hour": (len(cells) / hours) if hours > 0 else None,
            "net_horizontal_displacement": float(np.linalg.norm(hz[-1] - hz[0])),
            "max_horizontal_from_start": float(np.linalg.norm(hz - hz[0], axis=1).max()),
            "sampled_path_length_horizontal": path,
            "vertical_range": float(P[:, 1].max() - P[:, 1].min()),
            "y_min": float(P[:, 1].min()), "y_max": float(P[:, 1].max()),
            "stationary_fraction": float(np.mean(still)) if still else None,
            "one_step_pairs": len(steps)}
        S = per_stream[s]
        S["cells"] |= cells
        S["cols"] |= cols
        S["t_lo"], S["t_hi"] = min(S["t_lo"], T.min()), max(S["t_hi"], T.max())
        S["steps"] += steps
        S["y"] += P[:, 1].tolist()
        all_cells |= cells
        still_all += still
        ys += P[:, 1].tolist()
        bbox_lo, bbox_hi = np.minimum(bbox_lo, P.min(0)), np.maximum(bbox_hi, P.max(0))
    streams = {}
    for s, S in sorted(per_stream.items()):
        h = (S["t_hi"] - S["t_lo"]) / 3600.0
        streams[s] = {"unique_cells": len(S["cells"]),
                      "unique_columns_xz": len(S["cols"]), "hours": h,
                      "unique_cells_per_hour": len(S["cells"]) / h if h > 0 else None,
                      "stationary_fraction": (float(np.mean([x < STATIONARY_BLOCKS
                                                             for x in S["steps"]]))
                                              if S["steps"] else None),
                      "step_displacement": _stats(S["steps"]),
                      "vertical_range": float(max(S["y"]) - min(S["y"]))}
    disp = [v["net_horizontal_displacement"] for v in per_ep.values()]
    return {"available": True, "evaluator_channels": dict(channels),
            "absent_readings": absent,
            "definition": {"cell": "floor(x), floor(y), floor(z) of true_position",
                           "stationary": f"|true_position(t+1) - true_position(t)| < "
                                         f"{STATIONARY_BLOCKS} blocks over one agent step",
                           "coverage": "sampled steps only (every_n_steps): a LOWER "
                                       "bound on cells actually visited"},
            "unique_cells_total": len(all_cells),
            "stationary_fraction_all": float(np.mean(still_all)) if still_all else None,
            "vertical_range_all": float(max(ys) - min(ys)),
            "bbox_min": bbox_lo.tolist(), "bbox_max": bbox_hi.tolist(),
            "net_displacement_per_episode": _stats(disp),
            "streams": streams, "episodes": per_ep}


# ----------------------------------------------------------------- D
def preregistration(members=5, epochs=40):
    """Created and FROZEN before any data is read for section D."""
    from developmental_ai.foundation.experiments.ab import Preregistration
    return Preregistration(
        name="shadow-green-next-step-v2-validation",
        hypothesis=("a learned one-step predictor of the small-width GREEN "
                    "channels, conditioned on (x_t, a_t) (bootstrap ensemble "
                    "of MLPs), has lower held-out RMSE than persistence "
                    "(x_t+1 = x_t) on shadow-recorded SkyBot transitions"),
        primary_metric="rmse_ratio_vs_persistence", direction="lower",
        delta=0.05, baseline_arm="persistence", candidate_arm="ensemble_mlp",
        ablation_arm="ensemble_mlp_no_action", alternative_arm="ridge",
        seeds=SEEDS, basis="final",
        resource_budget={"max_wall_seconds": 1800.0},
        failure_conditions=("any arm raises",
                            "a held-out metric is non-finite",
                            "fewer than MIN_HELDOUT_PAIRS held-out pairs"),
        max_candidate_failure_fraction=0.0,
        notes=(f"channels: every GREEN vector channel of width <= {SMALL_WIDTH} "
               f"present in Evidence (image channels excluded); pairs: Evidence "
               f"obs t -> obs t+1, same stream and episode, transition_valid on "
               f"every channel; split: one per stream, the store's own episode "
               f"split (runtime.splits with the store's salt and fraction); fit "
               f"on the learning view (dev), reserving its chronologically last fifth "
               f"of episodes (ordered by each episode's earliest t_wall, then seq) "
               f"for validation (single episode: chronological split with a gap); "
               f"all arms use the same fit rows; neural arms select validation NLL "
               f"with patience {PATIENCE}, maximum {epochs} epochs per member and "
               f"{members} ensemble members; restore weights and optimizer together; "
               f"score once on the held-out view; metric: "
               f"mean over channels with nonzero held-out persistence error of "
               f"RMSE(arm)/RMSE(persistence); seeds {list(SEEDS)}; "
               f"insufficient below {MIN_PAIRS} pairs or {MIN_EPISODES} episodes")
    ).freeze()


def _channels_for_d(rows) -> List[Tuple[str, int]]:
    w = {}
    for r in rows:
        for ch, v in r["x0"].items():
            if v.ndim == 1 and 0 < v.size <= SMALL_WIDTH:
                w.setdefault(ch, v.size)
    ok = []
    for ch, n in sorted(w.items()):
        if all(ch in r["x0"] and r["x0"][ch].shape == (n,) for r in rows):
            ok.append((ch, n))
    return ok


def _action_dim(rows) -> Tuple[int, bool]:
    spec = rows[0]["spec"]
    m = re.search(r":discrete(\d+)$", spec)
    if m:
        return int(m.group(1)), True
    m = re.search(r":box(\d+)$", spec)
    if m:
        return int(m.group(1)), False
    return 1 + max(int(r["command"]) for r in rows), True


def _mats(rows, chans, ad, discrete):
    X = np.stack([np.concatenate([r["x0"][c] for c, _ in chans]) for r in rows])
    Y = np.stack([np.concatenate([r["x1"][c] for c, _ in chans]) for r in rows])
    A = np.zeros((len(rows), ad))
    for i, r in enumerate(rows):
        if discrete:
            A[i, int(r["command"])] = 1.0
        else:
            A[i, :] = np.asarray(r["command"], np.float64).reshape(-1)[:ad]
    G = np.array([f"{r['stream']}|{r['episode']}" for r in rows])
    return X, A, Y, G


def _slices(chans):
    out, o = {}, 0
    for c, n in chans:
        out[c] = slice(o, o + n)
        o += n
    return out


def validation_split(rows):
    """Development-only split, stable episode IDs. Never examines test rows.

    Multi-episode: the CHRONOLOGICALLY last fifth of episodes is held out,
    each episode ordered by its earliest (t_wall, seq). Not a sort of the
    ids: lexicographically "-e10" precedes "-e9", and in a multi-run store
    the run id would dominate, so a name sort holds out an arbitrary fifth.
    One-episode screening uses a chronological split with a transition gap;
    it remains exploratory and does not establish episode generalization.
    """
    def _when(r, i):
        # t_wall orders real shadow rows; synthetic/legacy rows without it
        # fall back to seq, then to input order -- never a KeyError.
        t = r.get("t_wall")
        return (float(t) if t is not None else float("inf"),
                int(r.get("seq", i)))
    start: Dict[Tuple[Any, Any, Any], Tuple[float, int]] = {}
    for i, r in enumerate(rows):
        k = (r["env"], r["stream"], r["episode"])
        when = _when(r, i)
        if k not in start or when < start[k]:
            start[k] = when
    # ties on (t_wall, seq) fall back to the key (repr: env may be None)
    keys = sorted(start, key=lambda k: (start[k], repr(k)))
    if len(keys) >= 2:
        held = set(keys[-max(1, len(keys)//5):])
        fit = [r for r in rows if (r["env"], r["stream"], r["episode"]) not in held]
        val = [r for r in rows if (r["env"], r["stream"], r["episode"]) in held]
        basis = "development-episode"
    else:
        ordered = [r for _, r in sorted(
            enumerate(rows), key=lambda ir: (_when(ir[1], ir[0])[1],
                                             _when(ir[1], ir[0])[0]))]
        cut = int(len(ordered)*0.8)
        fit, val = ordered[:max(0, cut-1)], ordered[cut:]
        basis = "development-chronological-gap-exploratory"
    if len(fit) < 3 or len(val) < 2:
        raise ValueError("insufficient development data for validation")
    return fit, val, basis


def fit_eval(arm: str, seed: int, train, test, chans, ad, discrete,
             members: int = 5, epochs: int = 40) -> Dict[str, Any]:
    """Fit `arm` on train rows, score on test rows. All metrics on the
    observable next GREEN vector, never a latent (plan §7.3)."""
    from developmental_ai.foundation.mechanisms import (EntityBatch, MixedOutcome,
                                                        OutcomeLayout, TransitionBatch,
                                                        score)
    supplied_train = len(train)
    train, validation, validation_basis = validation_split(train)
    X, A, Y, G = _mats(train, chans, ad, discrete)
    Xv, Av, Yv, _ = _mats(validation, chans, ad, discrete)
    Xt, At, Yt, _ = _mats(test, chans, ad, discrete)
    D = X.shape[1]
    sl = _slices(chans)
    dlt = Y - X
    dsd = dlt.std(0)
    dyn = np.flatnonzero(dsd > 1e-6)
    if dyn.size == 0:
        raise ValueError("no GREEN dim changes in the training pairs")
    t0 = time.perf_counter()
    lay = OutcomeLayout(tuple(f"d{i}" for i in range(D)))
    ens = None
    fit_reports = []
    if arm == "persistence":
        var = np.maximum((dlt ** 2).mean(0), 1e-12)
        dist = MixedOutcome.gaussian_categorical(lay, Xt, np.tile(var, (len(Xt), 1)))
    elif arm == "ridge":
        mu, sd = X.mean(0), np.where(X.std(0) < 1e-6, 1.0, X.std(0))
        tsd = np.where(dsd < 1e-6, 1.0, dsd)
        tmu = dlt.mean(0)

        def feats(x, a):
            return np.concatenate([(x - mu) / sd, a, np.ones((len(x), 1))], 1)
        F = feats(X, A)
        lam = 1.0
        W = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ ((dlt - tmu) / tsd))
        fit = (F @ W) * tsd + tmu
        var = np.maximum(((dlt - fit) ** 2).mean(0), 1e-12)
        pred = Xt + (feats(Xt, At) @ W) * tsd + tmu
        dist = MixedOutcome.gaussian_categorical(lay, pred, np.tile(var, (len(Xt), 1)))
    else:
        from developmental_ai.foundation.inference import BootstrapEnsemble
        from developmental_ai.foundation.mechanisms import MLPMechanism
        use_a = not arm.endswith("no_action")
        Atr = A if use_a else np.zeros_like(A)
        Ate = At if use_a else np.zeros_like(At)

        def eb(x):
            n = len(x)
            return EntityBatch(x[:, None, :], np.zeros((n, 1, D)), np.zeros((n, 1, 0)))
        batch = TransitionBatch(eb(X), Atr[:, None, :], np.ones(len(X)), eb(Y))

        def make(i):
            return MLPMechanism(1, D, 0, ad, EVENT_CLASSES, hidden=64, layers=2,
                                seed=1000 * seed + i, mechanism_id=f"{arm}.m{i}")
        val_batch = TransitionBatch(eb(Xv), (Av if use_a else np.zeros_like(Av))[:, None, :],
                                    np.ones(len(Xv)), eb(Yv))
        if arm == "mlp":
            model = make(0)
            fit_reports = [model.fit(batch, epochs=epochs, batch_size=BATCH_SIZE,
                                     lr=2e-3, validation=val_batch, patience=PATIENCE)]
        else:
            n_groups = len(set(G.tolist()))
            model = ens = BootstrapEnsemble(make, k=members, seed=seed)
            # Episode bootstrap only with enough episodes: with 4 unequal
            # episodes (live 2026-10-04: two of 40 pairs, two of 3534) a
            # member drawn on the two short ones extrapolates wildly and the
            # mixture mean is ruined (measured RMSE ratio 6.5 vs 0.96 for
            # one MLP). Below GROUP_BOOTSTRAP_MIN episodes: i.i.d. rows.
            fit_reports = model.fit(
                batch, groups=G if n_groups >= GROUP_BOOTSTRAP_MIN else None,
                epochs=epochs, batch_size=BATCH_SIZE, lr=2e-3,
                validation=val_batch, patience=PATIENCE)["members"]
        dist = model.predict(eb(Xt), Ate[:, None, :], np.ones(len(Xt)))
    secs = time.perf_counter() - t0
    tgt = {"continuous": (Yt if dist.layout.Dc == D else
                          np.concatenate([Yt, np.zeros_like(Yt)], 1))}
    sc = score(dist, tgt, cont_index=dyn.tolist(), disc_index=[])
    mean = dist.mean()[:, :D]
    err = mean - Yt
    perr = Xt - Yt
    per_ch, ratios = {}, []
    for c, s in sl.items():
        r_m = float(np.sqrt((err[:, s] ** 2).mean()))
        r_p = float(np.sqrt((perr[:, s] ** 2).mean()))
        ratio = r_m / r_p if r_p > 1e-9 else None
        per_ch[c] = {"rmse": r_m, "rmse_persistence": r_p, "ratio": ratio}
        if ratio is not None and np.any((dsd[s] > 1e-6)):
            ratios.append(ratio)
    if not ratios:
        raise ValueError("no channel has nonzero held-out persistence error")
    out = {"rmse_ratio_vs_persistence": float(np.mean(ratios)),
           "per_channel": per_ch, "nll_per_dim": sc.get("nll_cont"),
           "cov90": sc.get("cov90"), "cov50": sc.get("cov50"),
           "rmse_dyn_dims": sc.get("rmse"), "n_train": len(train), "n_test": len(test),
           "dyn_dims": int(dyn.size), "fit_seconds": secs,
           "interactions": supplied_train,
           # neural arms: optimizer steps actually taken, summed over members
           # (epochs early stopping discarded included); a fixed baseline is
           # one closed-form fit
           "model_updates": (int(sum(r["gradient_steps"] for r in fit_reports))
                             if fit_reports else 1),
           "validation_pairs": len(validation), "validation_basis": validation_basis,
           "epoch_budget_per_member": epochs,
           "epochs_run_per_member": [r["epochs"] for r in fit_reports],
           "selected_epoch_per_member": [r["selected_epoch"] for r in fit_reports],
           "selected_train_loss_per_member": [r["loss_selected"] for r in fit_reports],
           "selection": (f"validation NLL; patience {PATIENCE}; best weights and "
                         f"optimizer restored"
                         if fit_reports else "fixed baseline; no validation selection")}
    if ens is not None:
        dec = ens.decompose(eb(Xt), Ate[:, None, :], np.ones(len(Xt)))
        epi = dec["epistemic"][:, :D][:, dyn] / np.maximum(dsd[dyn] ** 2, 1e-12)
        ae = np.abs(err[:, dyn]) / np.maximum(dsd[dyn], 1e-12)
        out["spearman_epistemic_vs_abs_error"] = _spearman(epi.mean(1), ae.mean(1))
        per = [_spearman(epi[:, j], ae[:, j]) for j in range(len(dyn))]
        per = [p for p in per if p is not None]
        out["spearman_per_dim_median"] = float(np.median(per)) if per else None
    return out


ARMS = ("persistence", "ridge", "mlp", "ensemble_mlp", "ensemble_mlp_no_action")


def section_d(store, g, out_dir: str, members: int = 5, epochs: int = 40) -> Dict[str, Any]:
    from developmental_ai.foundation.experiments.ab import make_split, run_ab
    prereg = preregistration(members, epochs)  # FROZEN before any D data is touched
    res: Dict[str, Any] = {"prereg": prereg.to_dict(), "prereg_frozen_at_ns":
                           prereg.frozen_at_ns}
    rows = [r for r in g["rows"] if r["valid"]]
    res["invalid_pairs_dropped"] = len(g["rows"]) - len(rows)
    if not rows:
        res.update(status="insufficient data", reasons=["no valid transition pairs"])
        return res
    chans = _channels_for_d(rows)
    res["channels_used"] = [f"{c}[{n}]" for c, n in chans]
    excluded = sorted({ch for r in rows for ch in r["x0"]} - {c for c, _ in chans})
    res["channels_excluded"] = excluded + [f"{c} (image {s})" for c, s in
                                           sorted(g["image_obs"].items())]
    if not chans:
        res.update(status="insufficient data", reasons=["no small-width GREEN channel"])
        return res
    ad, discrete = _action_dim(rows)
    res["action"] = {"dim": ad, "discrete": discrete}
    hf = float(store.heldout_fraction)
    salt = store.split_salt
    eps = collections.defaultdict(set)
    for r in rows:
        eps[(r["env"], r["stream"])].add(r["episode"])
    splits, skipped = [], []
    for (env, s), es in sorted(eps.items()):
        try:
            sp = make_split(f"{env}:{s}", sorted(es), hf, salt,
                            split_id=f"{env}:{s}#store-split")
        except ValueError as e:
            skipped.append(f"{s}: {e}")
            continue
        # the split must BE the store's partition, or a learner read held-out
        part = {(r["stream"], r["episode"]): r["part"] for r in rows if r["stream"] == s}
        for e in sp.dev:
            assert part[(s, e)] == "dev", (s, e, part[(s, e)])
        for e in sp.heldout:
            assert part[(s, e)] == "heldout", (s, e, part[(s, e)])
        splits.append(sp)
    n_ep = sum(len(v) for v in eps.values())
    held = [r for r in rows if r["part"] == "heldout"]
    reasons = []
    if len(rows) < MIN_PAIRS:
        reasons.append(f"{len(rows)} valid pairs < {MIN_PAIRS}")
    if n_ep < MIN_EPISODES:
        reasons.append(f"{n_ep} episodes < {MIN_EPISODES}")
    if not splits:
        reasons.append("no stream has episodes on both sides of the store split "
                       f"({'; '.join(skipped)})")
    if len(held) < MIN_HELDOUT_PAIRS:
        reasons.append(f"{len(held)} held-out pairs < {MIN_HELDOUT_PAIRS}")
    res.update(valid_pairs=len(rows), episodes=n_ep, heldout_pairs=len(held),
               splits=[{"id": s.split_id, "dev": len(s.dev), "heldout": len(s.heldout),
                        "fingerprint": s.fingerprint} for s in splits],
               splits_skipped=skipped)
    if reasons:
        res.update(status="insufficient data", reasons=reasons)
        res["exploratory"] = exploratory_d(rows, chans, ad, discrete, members, epochs)
        return res
    by_split = {}
    for sp in splits:
        s = sp.scope.split(":", 1)[1]
        dev = {e: [] for e in sp.dev}
        hel = {e: [] for e in sp.heldout}
        for r in rows:
            if r["stream"] != s:
                continue
            (dev if r["episode"] in dev else hel)[r["episode"]].append(r)
        by_split[sp.split_id] = ([x for e in sp.dev for x in dev[e]],
                                 [x for e in sp.heldout for x in hel[e]])

    def arm_fn(name):
        def arm(seed, split):
            tr, te = by_split[split.split_id]
            if len(te) < 1:
                raise ValueError("empty held-out side")
            return fit_eval(name, seed, tr, te, chans, ad, discrete, members, epochs)
        return arm
    rep = run_ab(prereg, {a: arm_fn(a) for a in ARMS}, splits=splits,
                 out_dir=os.path.join(out_dir, "ab"), measure_memory=False)
    v = rep.verdict
    summary = {}
    for a in ARMS:
        runs = [r for r in rep.results.runs if r.arm == a and r.status == "ok"]
        summary[a] = _summarise([r.metrics for r in runs], [r.primary for r in runs])
        summary[a]["failed"] = sum(1 for r in rep.results.runs
                                   if r.arm == a and r.status != "ok")
        errs = [r.error for r in rep.results.runs if r.arm == a and r.error]
        if errs:
            summary[a]["errors"] = errs[:3]
    res.update(status="ran", verdict=v["label"], verdict_reasons=v["reasons"],
               comparison=v["comparisons"].get(prereg.basis),
               secondary=v.get("secondary"), arms=summary,
               ab_json=rep.json_path, ab_md=rep.md_path,
               results_started_ns=rep.results.started_at_ns)
    return res


def _summarise(metrics: List[Dict], primaries: List[float]) -> Dict[str, Any]:
    if not metrics:
        return {"runs_ok": 0}
    out = {"runs_ok": len(metrics),
           "rmse_ratio_mean": float(np.mean(primaries)),
           "rmse_ratio_std": float(np.std(primaries))}
    for k in ("nll_per_dim", "cov90", "cov50", "spearman_epistemic_vs_abs_error",
              "spearman_per_dim_median"):
        vals = [m.get(k) for m in metrics if m.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    chans = metrics[0]["per_channel"].keys()
    out["per_channel_ratio"] = {}
    for c in chans:
        vals = [m["per_channel"][c]["ratio"] for m in metrics
                if m["per_channel"][c]["ratio"] is not None]
        out["per_channel_ratio"][c] = float(np.mean(vals)) if vals else None
    out["n_train"] = metrics[0]["n_train"]
    out["n_test"] = metrics[0]["n_test"]
    return out


def exploratory_d(rows, chans, ad, discrete, members, epochs) -> Dict[str, Any]:
    """NOT A VERDICT. Chronological within-episode split: first 75% of each
    episode's sampled pairs train, the rest test. Episodes are not held out,
    so this measures fit to the SAME worlds later in time (adjacent sampled
    steps are every_n_steps apart; only the boundary pairs are neighbours)."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["part"] == "dev":                 # fit only what a learner may read
            by[(r["stream"], r["episode"])].append(r)
    tr, te = [], []
    for k, lst in sorted(by.items()):
        lst.sort(key=lambda r: r["seq"])
        cut = int(round(0.75 * len(lst)))
        tr += lst[:cut]
        te += lst[cut:]
    if len(tr) < 100 or len(te) < MIN_HELDOUT_PAIRS:
        return {"ran": False, "reason": f"{len(tr)} train / {len(te)} test pairs "
                                        f"too few even for a screening fit"}
    arms = {}
    for a in ARMS:
        ms, prim, errs = [], [], []
        for s in SEEDS:
            try:
                m = fit_eval(a, s, tr, te, chans, ad, discrete, members, epochs)
                ms.append(m)
                prim.append(m["rmse_ratio_vs_persistence"])
            except Exception as e:     # a failure is an outcome, not skipped
                errs.append(f"{type(e).__name__}: {e}")
            if a in ("persistence", "ridge"):
                break                  # deterministic: one seed is all seeds
        arms[a] = _summarise(ms, prim)
        if errs:
            arms[a]["errors"] = errs
    return {"ran": True, "label": "EXPLORATORY - chronological within-episode "
                                  "split, NOT episode-held-out, NO verdict",
            "train_pairs": len(tr), "test_pairs": len(te), "arms": arms}


# ----------------------------------------------------------------- E
def observable_scores(g):
    """Outcome scoring is separate from categorical entropy and record counts."""
    statuses = collections.Counter()
    groups = collections.defaultdict(list)
    for row in g["rows"]:
        score = row["pay"].get("observable_score", {})
        statuses[score.get("status", "unrecorded")] += 1
        if score.get("status") == "scored":
            groups[(row["part"], row["stream"], score["target"])].append(score)
    return {"statuses": dict(statuses), "groups": [
        {"partition": k[0], "stream": k[1], "target": k[2], "horizon": 1,
         "n": len(v), "mse": float(np.mean([x["mse"] for x in v])),
         "persistence_mse": float(np.mean([x["persistence_mse"] for x in v]))}
        for k, v in sorted(groups.items())]}


def section_e(g) -> Dict[str, Any]:
    observed = observable_scores(g)
    ps = [p for p in g["preds"] if p["probs"] is not None]
    if not ps:
        return {"available": False, "reason": "no prediction carries prior probs", "observable": observed}
    for p in ps:
        pr = np.clip(p["probs"].reshape(-1, p["probs"].shape[-1]), 1e-12, 1.0)
        h = -(pr * np.log(pr)).sum(-1)
        p["H"] = float(h.mean())
        p["Hn"] = float(h.mean() / math.log(pr.shape[-1]))
        p["pmax"] = float(pr.max(-1).mean())
        p["S"], p["C"] = pr.shape
    H = np.array([p["H"] for p in ps])
    out = {"observable": observed, "n": len(ps), "latent": f"{ps[0]['S']}x{ps[0]['C']}",
           "definition": "per-categorical entropy (nats) averaged over the S "
                         "latent groups; normalised = / ln C (1.0 = uniform)",
           "entropy": _stats(H),
           "entropy_normalised": _stats([p["Hn"] for p in ps]),
           "max_prob_mean": float(np.mean([p["pmax"] for p in ps])),
           "by_run": {}, "by_version": {}}
    runs = collections.defaultdict(list)
    for p in ps:
        runs[_run_of(p["episode"])].append(p)
    for rid, lst in sorted(runs.items()):
        lst.sort(key=lambda p: p["t_wall"])
        t = np.array([p["t_wall"] for p in lst])
        h = np.array([p["H"] for p in lst])
        bins = np.array_split(np.arange(len(lst)), min(10, len(lst)))
        out["by_run"][rid] = {
            "n": len(lst), "spearman_time_vs_entropy": _spearman(t, h),
            "deciles": [{"t_mid_h": float((t[b].mean() - t[0]) / 3600.0),
                         "entropy": float(h[b].mean()), "n": int(b.size)}
                        for b in bins if b.size]}
    vers = collections.defaultdict(list)
    for p in ps:
        vers[p["version"]].append(p["H"])
    steps = []
    for v, hs in vers.items():
        m = re.match(r"adam_step=(\d+)$", v)
        if m:
            steps += [(int(m.group(1)), x) for x in hs]
    if len(vers) <= 12:
        out["by_version"] = {v: {"n": len(hs), "entropy": float(np.mean(hs))}
                             for v, hs in sorted(vers.items())}
    elif steps:
        st = np.array(steps)
        order = np.argsort(st[:, 0])
        for b in np.array_split(order, 10):
            lo, hi = int(st[b, 0].min()), int(st[b, 0].max())
            out["by_version"][f"adam_step {lo}-{hi}"] = {
                "n": int(b.size), "entropy": float(st[b, 1].mean())}
    out["versions_unique"] = len(vers)
    if steps:
        st = np.array(steps)
        out["spearman_adam_step_vs_entropy"] = _spearman(st[:, 0], st[:, 1])
    if len(vers) == 1:
        out["note"] = (f"every prediction carries the same model version "
                       f"({next(iter(vers))}); version tracking cannot separate "
                       f"world-model states")
    return out


# ----------------------------------------------------------------- render
def _table(head, rows) -> List[str]:
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    L += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return L


def render(rep: Dict[str, Any]) -> str:
    I, A, B, C, D, E = (rep["integrity"], rep["A"], rep["B"], rep["C"],
                        rep["D"], rep["E"])
    L = [f"# Shadow store report", "",
         f"- store: `{rep['store']}` (opened readonly), generated {rep['generated']}",
         f"- quarantined on open: **{I['quarantined_total']}** record(s) "
         f"({I['quarantined_journal_lines']} journal line(s), "
         f"{I['quarantined_chunk_records']} from damaged chunks, "
         f"{I['manifest_records_missing_on_disk']} listed but missing); "
         f"{len(I['recovery_notes'])} recovery note(s)"]
    L += [f"  - {n}" for n in I["recovery_notes"][:10]]
    RF = rep.get("run_filter", {"requested": "all", "selected": "all"})
    L += ["", f"### Runs in the store (--run {RF['requested']} -> sections A-E "
          f"cover: **{RF['selected']}**)", ""]
    L += _table(["run", "first", "last", "sampled steps", "episodes", "streams",
                 "selected"],
                [[r["run"], r["first"], r["last"], r["sampled_steps"], r["episodes"],
                  ",".join(r["streams"]), "yes" if r["selected"] else ""]
                 for r in rep.get("runs_found", [])])
    if RF.get("unattributed_records"):
        L.append(f"- {RF['unattributed_records']} record(s) attributable to no run "
                 f"(shown only under --run all)")
    L += ["", "## A. Inventory", ""]
    L += _table(["run", "start", "end", "hours", "streams", "episodes",
                 "agent steps", "steps/s/stream"],
                [[k, v["start"], v["end"], v["hours"], ",".join(v["streams"]),
                  v["episodes"], v["agent_steps_covered"], v["steps_per_s_per_stream"]]
                 for k, v in A["runs"].items()])
    L += ["", f"- environments: {A['environments']}; streams: {A['streams']}; "
          f"episodes per stream: {A['episodes_per_stream']}",
          f"- records by partition/kind: {A['records_by_partition_kind']}; "
          f"evaluator partition (counts only): {A['evaluator_partition_counts']}",
          f"- episode ends: {A['episode_ends'] or 'none'}; episodes without an end "
          f"record: {A['episodes_without_end']}",
          f"- sampled steps (actions): {A['sampled_steps']} "
          f"{A['sampled_steps_per_stream']}; median seq gap between sampled steps "
          f"{_fmt(A['sampled_seq_gap_median'])}",
          f"- outcomes: {A['outcomes']} (valid transitions {A['valid_transitions']})",
          f"- predictions: {A['predictions']}; coverage {_fmt(A['prediction_coverage'])} "
          f"of sampled steps; resolved {A['predictions_resolved']}, unresolved "
          f"{A['predictions_unresolved']}",
          f"- model versions ({A['model_versions_unique']} unique): {A['model_versions']}",
          f"- action elapsed s: {_fmtd(A['action_elapsed_s'])}",
          f"- GREEN observation channels (records): {A['observation_channels']}; "
          f"image-shaped: {A['image_channels']}", ""]
    L += ["## B. Behaviour (policy-visible + reward channels only)", "",
          "### Actions per stream", ""]
    for s, a in B["actions"].items():
        L.append(f"- {s}: n={a['n']}, distinct={a['distinct']}, entropy "
                 f"{_fmt(a['entropy_bits'])} bits; top: " + ", ".join(
                     f"{k}:{v} ({p:.1%})" for k, v, p in a["top"][:8]))
    L += ["", "### Actions per stream, by source", "",
          "source = who chose the executed command (Action payload); "
          "`unrecorded` = a store written before sources were recorded.", ""]
    for s, d in B.get("actions_by_source", {}).items():
        for src, a in d.items():
            L.append(f"- {s} / {src}: n={a['n']}, distinct={a['distinct']}, "
                     f"entropy {_fmt(a['entropy_bits'])} bits; top: " + ", ".join(
                         f"{k}:{v} ({p:.1%})" for k, v, p in a["top"][:8]))
    L += ["", "### Reward channels per sampled step", ""]
    rr = []
    for s, d in B["rewards"].items():
        for name, v in d.items():
            rr.append([s, name, v["n"], v["mean"], v["sum"], v["frac_nonzero"],
                       v.get("share_of_abs_income"), v["min"], v["max"]])
    L += _table(["stream", "channel", "n", "mean", "sum", "frac nonzero",
                 "share |income|", "min", "max"], rr)
    L += ["", "prim_extrinsic exists for stream-0 only (the loop passes the "
          "primary stream's scalar; other streams' reward_replay = env reward). "
          "share = |sum| share among the disjoint parts env / shaping / intrinsic.", ""]
    G = B["gui"]
    L += ["### GUI", ""]
    if G.get("available", True):
        L += [f"- {G['source']}",
              f"- fraction open: {_fmtd(G['fraction_open'])}; all {_fmt(G['fraction_open_all'])}",
              f"- runs: {G['runs']}; run length (sampled points) {_fmtd(G['run_length_sampled_points'])}; "
              f"gap {_fmt(G['sampled_point_gap_steps'])} steps; longest ~{_fmt(G['longest_run_approx_steps'])} steps",
              f"- histogram (points: runs): {G['run_length_hist']}"]
    else:
        L.append(f"- not available: {G['reason']}")
    S = B["stillness"]
    L += ["", "### Stillness proxy", "",
          f"- {S['definition']}: **{_fmt(S['fraction_transport_unchanged'])}**",
          f"- per channel unchanged: {_fmtd(S['per_channel_unchanged'])}"]
    if "visual" in S:
        L.append(f"- visual: {_fmtd(S['visual'])}")
    L += ["", "### What stream-0 is paid while ...", ""]
    L += _table(["condition", "n", "env mean", "prim_extrinsic mean", "intrinsic mean"],
                [[k, v["n"], v.get("env_mean"), v.get("prim_extrinsic_mean"),
                  v.get("intrinsic_mean")] for k, v in B["pay_while"].items()
                 if isinstance(v, dict)])
    L += ["", "## C. EVALUATOR-ONLY (privileged true_position; computed from the "
          "evaluator partition; never used by A, B, D or E)", ""]
    if C.get("available"):
        L += [f"- definitions: {C['definition']}",
              f"- unique_cells total: {C['unique_cells_total']}; stationary fraction "
              f"{_fmt(C['stationary_fraction_all'])}; vertical range "
              f"{_fmt(C['vertical_range_all'])}; absent readings {C['absent_readings']}",
              f"- bbox min {np.round(C['bbox_min'], 2).tolist()} max "
              f"{np.round(C['bbox_max'], 2).tolist()}",
              f"- net displacement per episode: {_fmtd(C['net_displacement_per_episode'])}", ""]
        L += _table(["stream", "unique_cells", "columns xz", "hours", "cells/hour",
                     "stationary", "vertical range"],
                    [[s, v["unique_cells"], v["unique_columns_xz"], v["hours"],
                      v["unique_cells_per_hour"], v["stationary_fraction"],
                      v["vertical_range"]] for s, v in C["streams"].items()])
        L += [""]
        L += _table(["episode", "points", "hours", "unique_cells", "cells/hour",
                     "net disp", "max from start", "path (sampled)", "vertical range",
                     "stationary"],
                    [[k, v["points"], v["hours"], v["unique_cells"],
                      v["unique_cells_per_hour"], v["net_horizontal_displacement"],
                      v["max_horizontal_from_start"], v["sampled_path_length_horizontal"],
                      v["vertical_range"], v["stationary_fraction"]]
                     for k, v in C["episodes"].items()])
    else:
        L.append(f"- not available: {C.get('reason')}")
    L += ["", "## D. Mechanism comparison (preregistered A/B, plan Stage 6/7)", "",
          f"- preregistration `{D['prereg']['name']}` frozen at "
          f"{D['prereg_frozen_at_ns']} ns, hash `{D['prereg']['frozen_hash'][:16]}`",
          f"- hypothesis: {D['prereg']['hypothesis']}",
          f"- rule: {D['prereg']['acceptance_rule']}",
          f"- channels used: {D.get('channels_used')}; excluded: {D.get('channels_excluded')}",
          f"- valid pairs {D.get('valid_pairs')}, episodes {D.get('episodes')}, held-out "
          f"pairs {D.get('heldout_pairs')}, splits {D.get('splits')}"]
    if D["status"] == "ran":
        L += ["", f"**Verdict: {D['verdict']}** ({'; '.join(D['verdict_reasons'])})",
              f"- A/B harness report: `{D['ab_md']}`", ""]
        L += _arm_table(D["arms"])
    else:
        L += ["", f"**{D['status'].upper()}** — no verdict: {'; '.join(D['reasons'])}"]
        X = D.get("exploratory")
        if X and X.get("ran"):
            L += ["", f"### {X['label']}", "",
                  f"train {X['train_pairs']} / test {X['test_pairs']} pairs", ""]
            L += _arm_table(X["arms"])
        elif X:
            L.append(f"- exploratory not run: {X['reason']}")
    L += ["", "## E. RSSM prior entropy", ""]
    if E.get("available", True):
        L += [f"- {E['n']} predictions, latent {E['latent']}; {E['definition']}",
              f"- entropy {_fmtd(E['entropy'])}; normalised {_fmtd(E['entropy_normalised'])}; "
              f"mean max prob {_fmt(E['max_prob_mean'])}",
              f"- versions unique: {E['versions_unique']}; Spearman(adam_step, H) "
              f"{_fmt(E.get('spearman_adam_step_vs_entropy'))}"]
        if E.get("note"):
            L.append(f"- note: {E['note']}")
        for rid, v in E["by_run"].items():
            L.append(f"- run {rid}: n={v['n']}, Spearman(time, H) "
                     f"{_fmt(v['spearman_time_vs_entropy'])}; deciles (h, H): " +
                     ", ".join(f"({d['t_mid_h']:.2f}, {d['entropy']:.3f})"
                               for d in v["deciles"]))
        L.append(f"- by version: {_fmtd(E['by_version'])}")
    else:
        L.append(f"- not available: {E['reason']}")
    L.append("")
    L += ["", "## Observable forecast accuracy", "",
          "Latent entropy is not prediction accuracy. Boundary outcomes are excluded.", "",
          f"- scoring coverage: {E.get('observable', {}).get('statuses', {})}"]
    for group in E.get("observable", {}).get("groups", []):
        L.append(f"- {group['partition']} / {group['stream']} / {group['target']}: "
                 f"n={group['n']}, MSE={group['mse']:.6g}, "
                 f"persistence MSE={group['persistence_mse']:.6g}, horizon=1")
    return "\n".join(L)


def _arm_table(arms) -> List[str]:
    chans = sorted({c for a in arms.values() for c in (a.get("per_channel_ratio") or {})})
    head = ["arm", "ok", "RMSE ratio", "NLL/dim", "cov90", "Spearman(epi,|err|)"] + chans
    rows = []
    for a, s in arms.items():
        rows.append([a, s.get("runs_ok"), s.get("rmse_ratio_mean"), s.get("nll_per_dim"),
                     s.get("cov90"), s.get("spearman_epistemic_vs_abs_error")] +
                    [(s.get("per_channel_ratio") or {}).get(c) for c in chans])
    return _table(head, rows)


def _fmtd(d) -> str:
    if not isinstance(d, dict):
        return _fmt(d)
    return "{" + ", ".join(f"{k}: {_fmt(v) if not isinstance(v, dict) else _fmtd(v)}"
                           for k, v in d.items()) + "}"


# ----------------------------------------------------------------- main
def build_report(store_dir: str, out_dir: str, members: int = 5,
                 epochs: int = 40, run: str = "all") -> Dict[str, Any]:
    """run: "all" (every run, the pre-2026-10-05 behaviour), "latest" (the
    run that started last) or an exact run id. The filter is applied ONCE,
    by wrapping the store in RunFilteredStore; integrity stays whole-store
    (it describes the files, not a run)."""
    t0 = time.time()
    raw = open_store(store_dir)
    runs, attr = list_runs(raw)
    sel = resolve_run(runs, run)
    for r in runs:
        r["selected"] = sel is None or r["run"] == sel
    rep = {"store": os.path.abspath(store_dir),
           "generated": _ts(time.time()), "integrity": integrity(raw),
           "runs_found": runs,
           "run_filter": {"requested": run, "selected": sel if sel else "all",
                          "unattributed_records": sum(1 for v in attr.values()
                                                      if v is None)}}
    store = raw if sel is None else RunFilteredStore(raw, sel, attr)
    g = load_green(store)
    rep["A"] = section_a(store, g)
    rep["B"] = section_b(g)
    rep["C"] = section_c(store)
    rep["D"] = section_d(store, g, out_dir, members, epochs)
    rep["E"] = section_e(g)
    rep["seconds"] = time.time() - t0
    rep = _clean(rep)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True, allow_nan=False)
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write(render(rep))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("store")
    ap.add_argument("--out", default="runlogs/shadow_report")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--run", default="all",
                    help="run id, 'latest' (started last) or 'all' (default). "
                         "Before/after: run twice with different --out, e.g. "
                         "--run <old_id> --out runlogs/rep_before and --run "
                         "latest --out runlogs/rep_after")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    st = os.path.abspath(a.store)
    if out == st or out.startswith(st + os.sep):
        ap.error("--out must not be inside the store (the store is read-only)")
    rep = build_report(a.store, a.out, a.members, a.epochs, run=a.run)
    print(f"runs: " + ", ".join(f"{r['run']} ({r['first']} .. {r['last']}, "
                                f"{r['sampled_steps']} sampled)"
                                for r in rep["runs_found"])
          + f"; selected {rep['run_filter']['selected']}")
    print(f"wrote {os.path.join(a.out, 'REPORT.md')} and report.json "
          f"({rep['seconds']:.1f}s; quarantined {rep['integrity']['quarantined_total']}; "
          f"D: {rep['D']['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
