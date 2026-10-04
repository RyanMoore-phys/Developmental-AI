"""EvidenceStore — the immutable record of record for evaluation.

WHAT THIS IS, AND WHAT IT IS NOT
    The store is the experiment notebook: what was observed, what was done,
    what a model predicted BEFORE it knew, and what then happened. It exists
    so that scientific comparisons (plan §7) have a fixed, verifiable
    population to run on, and so a selected interaction can be reconstructed
    together with its original prediction and the snapshot that made it
    (plan Stage 5 completion gate).

    It is NOT a second training replay. `world_model/replay_buffer.py` keeps
    sole ownership of training data for the live learner; nothing in the
    live loop writes here or reads from here (plan §8: Offline/Shadow; "do
    not allow background experiments to silently alter the active learner's
    replay"). Offline consumers that FIT parameters (a Stage 6 mechanism
    fitted on stored experience) read through `learning_view()`, which is
    the dev split only. See docs/foundation/EXPERIENCE_RECONCILIATION.md.

LAYOUT (root/)
    STORE.json       identity, store format, split spec, budget (written once)
    manifest.json    sealed chunks + sha256, pins, eviction/quarantine log
    dev/ heldout/ evaluator/
                     one partition each: an fsynced append journal
                     (open-<cid>.jsonl) and sealed gzipped chunks
                     (chunk-<cid>.jsonl.gz), see chunks.py
    quarantine/      bytes that failed their hash on open, kept for
                     inspection (bounded to `quarantine_frac` of the budget)

    Partition is decided at write time and is structural: evaluator-
    provenance records go to evaluator/, everything else to the side of the
    deterministic episode split (foundation.runtime.splits) its scope falls
    on. The learning view is constructed holding ONLY the dev partition
    object, so it cannot return held-out or evaluator data — the same shape
    as SensorBus.read_policy never visiting a RED sensor.

RULES ENFORCED AT WRITE
    Raw records (Observation, Action, Prediction, Evidence) are immutable:
    a key is written once (DuplicateRecordError). Revisable readings of them
    are INTERPRETATIONS, separate records with (target ref, interpreter,
    interpreter version, supersedes); the raw line is never touched.

    Every write gets a global write sequence number (the `seq` in its ref
    "<partition>/<seq>"); the store's clock is that sequence, not t_wall.

    log_observation  imagined provenance refused (ProvenanceError): a model's
                     output is a Prediction, never an observation. Per
                     (scope, channel) seq strictly increases, t_env/t_wall
                     never run backwards; nothing after an episode end.
    log_action       must carry its ACTUAL completion time (t_complete);
                     elapsed = t_complete - t_dispatch is indexed beside the
                     commanded duration. Must answer an observation already
                     logged at the same (scope, seq), dispatched no earlier.
    log_prediction   needs the logged, non-evaluator observation it was
                     conditioned on (`context`); a prediction id is written
                     once.
    log_outcome      Evidence: imagined provenance refused; every
                     observation/action in it must be one this store already
                     recorded, byte-identical (content hash), so the only
                     door for observations into evidence is log_observation;
                     each linked prediction must be in the same scope (no
                     reset splice), and every observation of the outcome
                     after the prediction's context must have been WRITTEN
                     AFTER the prediction (OrderingError otherwise).
    log_episode_end  reason in {terminated, truncated, client_recovery}.

RETENTION (bounded growth)
    After every append the store's on-disk bytes (chunks + journals +
    manifest + quarantine) are <= max_bytes. Over budget, the OLDEST sealed
    chunk that is not pinned is evicted; if only journals remain they are
    sealed (gzip) first so they become evictable. Pinned = explicitly pinned
    (a promoted mechanism's evidence, `pin_mechanism`) or held-out, together
    capped at `max_pinned_frac` of the budget: explicit pins are refused past
    the cap (StoreBudgetError naming unpin as the escape), and held-out
    chunks beyond the cap become evictable oldest-first, so the store can
    never latch into "full and nothing may be deleted" (CLAUDE.md §4.1).

RECOVERY (on open)
    stray .tmp files deleted; a sealed chunk whose sha256 differs from the
    manifest is quarantined and its individually-verifying lines salvaged
    into a new chunk; a journal's bad or truncated lines are quarantined and
    the journal rewritten atomically with the good ones; a chunk sealed but
    not yet in the manifest (crash between rename and manifest write) is
    reconciled with its journal. Everything else loads. `recovery` lists
    what happened.

CONCURRENCY
    Writes are serialised by one lock. Views read the live index WITHOUT it:
    read from the writer's thread, or open a second store readonly. (A
    Shadow-stage writer thread is future work; nothing concurrent exists.)

MIGRATION
    Reads decode through the contracts migration registry and never write
    back. `migrate_store(src, dst)` converts into a NEW root, refusing to
    overwrite or nest in the source (plan §8).
"""

from __future__ import annotations

import bisect
import collections
import hashlib
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import (ABSENT, UNKNOWN, Action, Evidence, Mechanism,
                         Observation, ObsRef, Prediction, Scope, from_dict,
                         to_dict)
from ..contracts.codec import MIGRATIONS, decode_value, encode_value
from ..contracts.errors import SchemaVersionError
from ..runtime.splits import (DEV, HELDOUT, SPLIT_SALT, SplitLeakError,
                              EpisodeKey, assert_no_leak, assign_split,
                              split_spec)
from . import chunks as ck
from .errors import (DuplicateRecordError, ExperienceError, OrderingError,
                     PrivacyError, ProvenanceError, RecordEvicted,
                     StoreBudgetError, StoreCorruption)

EVALUATOR = "evaluator"
PARTS = (DEV, HELDOUT, EVALUATOR)
END_REASONS = ("terminated", "truncated", "client_recovery")
RECORD_KINDS = ("observation", "action", "prediction", "evidence")
STORE_KINDS = ("episode_end", "interpretation")
KINDS = RECORD_KINDS + STORE_KINDS
_RECENT_LOG = 64


# ------------------------------------------------------------------ helpers

def content_hash(record) -> str:
    """sha256 of a record's canonical wire form — its identity for content."""
    return hashlib.sha256(ck.canonical(to_dict(record)).encode()).hexdigest()


def split_scope(environment: str, stream: str) -> str:
    """The `scope` string foundation.runtime.splits assigns episodes under."""
    return f"{environment}:{stream}"


def parse_ref(ref: str) -> Tuple[str, int]:
    if not isinstance(ref, str) or "/" not in ref:
        raise KeyError(f"not a store ref: {ref!r} (expected '<part>/<seq>')")
    part, _, s = ref.partition("/")
    if part not in PARTS or not s.isdigit():
        raise KeyError(f"not a store ref: {ref!r}")
    return part, int(s)


def make_ref(part: str, seq: int) -> str:
    return f"{part}/{seq}"


def _cmd_key(c) -> str:
    if isinstance(c, np.ndarray):
        s = str(np.round(c.astype(float), 3).tolist())
    elif isinstance(c, tuple):
        s = str(tuple(_cmd_key(e) for e in c))
    else:
        s = repr(c)
    return s[:96]


def _num(v):
    return None if v is UNKNOWN else float(v)


@dataclass
class _Entry:
    seq: int
    kind: str
    cid: int
    line_no: int
    meta: Dict[str, Any]
    hash: str


# ---------------------------------------------------------------- partition

class _Partition:
    """One partition's files and its read index. Holds no reference to the
    store or to any other partition — that is what makes a view holding
    only some partitions structurally unable to reach the others."""

    def __init__(self, name: str, dirpath: str, registry):
        self.name = name
        self.dir = dirpath
        self.registry = registry
        self.entries: Dict[int, _Entry] = {}
        self.by_kind: Dict[str, Dict[int, None]] = {k: {} for k in KINDS}
        self.obs_at: Dict[Tuple, List[int]] = {}
        self.pred_by_id: Dict[str, int] = {}
        self.outcomes_by_pid: Dict[str, List[int]] = {}
        self.interps_by_target: Dict[int, List[int]] = {}
        self.superseded: Dict[int, int] = {}
        self.ends: Dict[Tuple, int] = {}
        self.chunk_seqs: Dict[int, List[int]] = {}
        self.evicted_ranges: List[Tuple[int, int]] = []
        self.journal_cid: Optional[int] = None
        self.journal_lines: List[str] = []
        self.journal_bytes = 0
        self._fh = None
        self._cache: "collections.OrderedDict[int, List[str]]" = \
            collections.OrderedDict()

    # paths
    def journal_path(self, cid: int) -> str:
        return os.path.join(self.dir, f"open-{cid:08d}.jsonl")

    def sealed_path(self, cid: int) -> str:
        return os.path.join(self.dir, f"chunk-{cid:08d}.jsonl.gz")

    # index maintenance
    def add(self, e: _Entry) -> None:
        self.entries[e.seq] = e
        self.by_kind[e.kind][e.seq] = None
        self.chunk_seqs.setdefault(e.cid, []).append(e.seq)
        m = e.meta
        if e.kind == "observation":
            self.obs_at.setdefault(
                (m["env"], m["stream"], m["episode"], m["oseq"]), []).append(e.seq)
        elif e.kind == "prediction":
            self.pred_by_id[m["pid"]] = e.seq
        elif e.kind == "evidence":
            for pid in m["pids"]:
                self.outcomes_by_pid.setdefault(pid, []).append(e.seq)
        elif e.kind == "interpretation":
            _, tseq = parse_ref(m["target"])
            self.interps_by_target.setdefault(tseq, []).append(e.seq)
            if m["supersedes"] is not None:
                self.superseded[parse_ref(m["supersedes"])[1]] = e.seq
        elif e.kind == "episode_end":
            self.ends[(m["env"], m["stream"], m["episode"])] = e.seq

    def _drop(self, d: Dict, key, seq: int) -> None:
        lst = d.get(key)
        if lst is None:
            return
        if isinstance(lst, list):
            if seq in lst:
                lst.remove(seq)
            if not lst:
                del d[key]
        elif lst == seq:
            del d[key]

    def remove_chunk(self, cid: int) -> List[_Entry]:
        gone = []
        for seq in self.chunk_seqs.pop(cid, []):
            e = self.entries.pop(seq)
            gone.append(e)
            self.by_kind[e.kind].pop(seq, None)
            m = e.meta
            if e.kind == "observation":
                self._drop(self.obs_at, (m["env"], m["stream"], m["episode"],
                                         m["oseq"]), seq)
            elif e.kind == "prediction":
                self._drop(self.pred_by_id, m["pid"], seq)
            elif e.kind == "evidence":
                for pid in m["pids"]:
                    self._drop(self.outcomes_by_pid, pid, seq)
            elif e.kind == "interpretation":
                self._drop(self.interps_by_target, parse_ref(m["target"])[1], seq)
                if m["supersedes"] is not None:
                    self._drop(self.superseded, parse_ref(m["supersedes"])[1], seq)
            elif e.kind == "episode_end":
                self._drop(self.ends, (m["env"], m["stream"], m["episode"]), seq)
        if gone:
            self.evicted_ranges.append((min(x.seq for x in gone),
                                        max(x.seq for x in gone)))
        self._cache.pop(cid, None)
        return gone

    def was_evicted(self, seq: int) -> bool:
        return any(lo <= seq <= hi for lo, hi in self.evicted_ranges)

    # reading
    def _lines(self, cid: int) -> List[str]:
        if cid == self.journal_cid:
            return self.journal_lines
        if cid in self._cache:
            self._cache.move_to_end(cid)
            return self._cache[cid]
        lines = ck.read_sealed(self.sealed_path(cid))
        self._cache[cid] = lines
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return lines

    def read_line(self, seq: int) -> Dict:
        e = self.entries.get(seq)
        if e is None:
            if self.was_evicted(seq):
                raise RecordEvicted(f"{self.name}/{seq} was evicted by the "
                                    f"retention rule")
            raise KeyError(f"no record {self.name}/{seq}")
        try:
            d, why = ck.decode_line(self._lines(e.cid)[e.line_no])
        except ck.READ_ERRORS + (IndexError,) as err:
            d, why = None, str(err)
        if d is None or d["hash"] != e.hash or d["seq"] != seq:
            raise StoreCorruption(
                f"{self.name}/{seq} failed verification on read "
                f"({why or 'hash changed since open'}); reopen the store to "
                f"quarantine it")
        return d

    def decode(self, seq: int):
        d = self.read_line(seq)
        if d["kind"] in RECORD_KINDS:
            return from_dict(d["body"], registry=self.registry)
        body = dict(d["body"])
        if d["kind"] == "interpretation":
            body["content"] = decode_value(body["content"], self.registry)
        return body


# ------------------------------------------------------------- reconstruct

@dataclass
class PredictionTrace:
    ref: str
    write_seq: int
    prediction: Prediction
    snapshot_id: str
    model_versions: Dict[str, Any]
    context: ObsRef
    context_observations: Tuple[Observation, ...]


@dataclass
class Interaction:
    """A selected outcome with everything that explains it, as written."""
    ref: str
    write_seq: int
    evidence: Evidence
    predictions: List[PredictionTrace]
    scores: Dict[str, Any]
    episode_end: Any                      # reason str, or ABSENT
    problems: List[str] = field(default_factory=list)

    @property
    def aligned(self) -> bool:
        return not self.problems


# -------------------------------------------------------------------- views

class _View:
    """Read access to a fixed set of partitions. Has no store reference."""

    _label = "view"

    def __init__(self, parts: Dict[str, _Partition]):
        self._parts = dict(parts)

    def _part_for(self, ref: str) -> Tuple[_Partition, int]:
        part, seq = parse_ref(ref)
        p = self._parts.get(part)
        if p is None:
            self._refuse(part, ref)
        return p, seq

    def _refuse(self, part: str, ref: str):
        raise KeyError(f"{ref} is not visible from the {self._label}")

    @property
    def partitions(self) -> Tuple[str, ...]:
        return tuple(self._parts)

    def get(self, ref: str):
        p, seq = self._part_for(ref)
        return p.decode(seq)

    def meta(self, ref: str) -> Dict[str, Any]:
        p, seq = self._part_for(ref)
        e = p.entries.get(seq)
        if e is None:
            p.read_line(seq)                      # raises evicted/missing
        return dict(e.meta)

    def kind(self, ref: str) -> str:
        p, seq = self._part_for(ref)
        if seq not in p.entries:
            p.read_line(seq)
        return p.entries[seq].kind

    def write_seq(self, ref: str) -> int:
        self.kind(ref)
        return parse_ref(ref)[1]

    def refs(self, kind: Optional[str] = None) -> List[str]:
        out = []
        for name, p in self._parts.items():
            seqs = p.by_kind[kind] if kind else p.entries
            out.extend((s, make_ref(name, s)) for s in seqs)
        out.sort()
        return [r for _, r in out]

    def __len__(self) -> int:
        return sum(len(p.entries) for p in self._parts.values())

    def query(self, kind: Optional[str] = None, environment: str = None,
              stream: str = None, episode: str = None, channel: str = None,
              t_wall: Tuple[float, float] = None) -> List[str]:
        """Retrieval by CONTEXT. Filters on index metadata only (no decode).
        `channel` matches an observation's channel or any channel inside an
        evidence record; `t_wall` is an inclusive [lo, hi] window."""
        if kind is not None and kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; known {KINDS}")
        out = []
        for ref in self.refs(kind):
            m = self.meta(ref)
            if environment is not None and m.get("env") != environment:
                continue
            if stream is not None and m.get("stream") != stream:
                continue
            if episode is not None and m.get("episode") != episode:
                continue
            if channel is not None and not (
                    m.get("channel") == channel
                    or channel in m.get("channels", ())):
                continue
            if t_wall is not None:
                lo, hi = t_wall
                a = m.get("t_wall", m.get("t_wall_lo", m.get("t_dispatch")))
                b = m.get("t_wall", m.get("t_wall_hi", m.get("t_complete")))
                if a is None or b < lo or a > hi:
                    continue
            out.append(ref)
        return out

    # interpretations -----------------------------------------------------
    def interpretations(self, ref: str, current_only: bool = True,
                        interpreter: Optional[str] = None) -> List[Tuple[str, Dict]]:
        p, seq = self._part_for(ref)
        out = []
        for iseq in p.interps_by_target.get(seq, ()):
            if current_only and iseq in p.superseded:
                continue
            body = p.decode(iseq)
            if interpreter is not None and body["interpreter"] != interpreter:
                continue
            out.append((make_ref(p.name, iseq), body))
        return out

    def interp_key(self, ref: str) -> Tuple[int, ...]:
        """Cheap change-detector: the current interpretation seqs of a ref."""
        p, seq = self._part_for(ref)
        return tuple(s for s in p.interps_by_target.get(seq, ())
                     if s not in p.superseded)

    def current_score(self, outcome_ref: str, prediction_id: str = None):
        """The newest un-superseded prediction_score on an outcome, or
        UNKNOWN when nothing has scored it (never 0.0: unscored is not
        'no error')."""
        best = None
        for iref, body in self.interpretations(outcome_ref):
            c = body["content"]
            if not (isinstance(c, dict) and c.get("kind") == "prediction_score"):
                continue
            if prediction_id is not None and c.get("prediction_id") != prediction_id:
                continue
            if best is None or parse_ref(iref)[1] > parse_ref(best[0])[1]:
                best = (iref, c)
        return UNKNOWN if best is None else dict(best[1], ref=best[0])

    def episode_end(self, scope: Scope):
        key = (scope.environment, scope.stream, scope.episode)
        for p in self._parts.values():
            if key in p.ends:
                return p.decode(p.ends[key])["reason"]
        return ABSENT

    def action_timing(self, **filters) -> List[Dict[str, Any]]:
        """Commanded vs measured duration of every executed action."""
        out = []
        for ref in self.query(kind="action", **filters):
            m = self.meta(ref)
            out.append({"ref": ref, "spec": m["spec"],
                        "commanded": UNKNOWN if m["duration"] is None
                        else m["duration"],
                        "elapsed": m["elapsed"],
                        "t_dispatch": m["t_dispatch"],
                        "t_complete": m["t_complete"]})
        return out

    # reconstruction ------------------------------------------------------
    def _context_obs(self, ctx: Dict) -> Tuple[Observation, ...]:
        key = (ctx["env"], ctx["stream"], ctx["episode"], ctx["oseq"])
        out = []
        for p in self._parts.values():
            for s in p.obs_at.get(key, ()):
                out.append(p.decode(s))
        return tuple(out)

    def reconstruct(self, outcome_ref: str) -> Interaction:
        """An outcome with its original predictions (as written, with write
        seq, snapshot id and model versions), their context observations,
        current scores and the episode's end — plus every alignment problem
        found (obs t -> action t -> obs t+1, prediction before outcome)."""
        p, seq = self._part_for(outcome_ref)
        if p.entries.get(seq) is None or p.entries[seq].kind != "evidence":
            p.read_line(seq)
            raise ExperienceError(f"{outcome_ref} is not an evidence record")
        ev = p.decode(seq)
        problems: List[str] = []
        traces = []
        for pid in ev.prediction_refs:
            found = None
            for name, q in self._parts.items():
                if pid in q.pred_by_id:
                    found = (q, q.pred_by_id[pid])
            if found is None:
                problems.append(f"prediction {pid!r} not visible from the "
                                f"{self._label} (other partition or evicted)")
                continue
            q, pseq = found
            pred = q.decode(pseq)
            c = q.entries[pseq].meta["ctx"]
            ctx = ObsRef(Scope(c["env"], c["stream"], c["episode"]), c["oseq"],
                         c["channel"])
            traces.append(PredictionTrace(
                make_ref(q.name, pseq), pseq, pred, pred.snapshot_id,
                dict(pred.model_versions), ctx, self._context_obs(c)))
            if pseq >= seq:
                problems.append(f"prediction {pid!r} written at {pseq} after "
                                f"its outcome at {seq}")
        oseqs = {o.seq for o in ev.observations}
        for a in ev.actions:
            if a.seq not in oseqs:
                problems.append(f"action seq {a.seq} has no observation at "
                                f"seq {a.seq} in the evidence")
            if a.seq + 1 not in oseqs:
                problems.append(f"action seq {a.seq} has no consequence "
                                f"observation at seq {a.seq + 1}")
        for t in traces:
            if t.context.scope != ev.scope:
                problems.append(f"prediction {t.prediction.prediction_id!r} "
                                f"context scope {t.context.scope} != evidence "
                                f"scope {ev.scope}")
            if not t.context_observations:
                problems.append(f"context observation of "
                                f"{t.prediction.prediction_id!r} not visible")
        scores = {}
        for t in traces:
            scores[t.prediction.prediction_id] = self.current_score(
                outcome_ref, t.prediction.prediction_id)
        return Interaction(outcome_ref, seq, ev, traces, scores,
                           self.episode_end(ev.scope), problems)


class LearningView(_View):
    """What a learner may read: the DEV partition and nothing else. Built
    holding only that partition object; held-out refs raise SplitLeakError
    and evaluator refs raise PrivacyError, without either being loaded."""

    _label = "learning view"

    def _refuse(self, part: str, ref: str):
        if part == HELDOUT:
            raise SplitLeakError(f"{ref} is held-out evaluation data; a "
                                 f"learner may not read it (plan §7.2.2)")
        if part == EVALUATOR:
            raise PrivacyError(f"{ref} is evaluator data; it never reaches a "
                               f"learning path (plan §7.4)")
        super()._refuse(part, ref)

    def assert_trainable(self, refs: Iterable[str]) -> None:
        for r in refs:
            self._part_for(r)


class HeldoutView(_View):
    _label = "held-out view"

    def _refuse(self, part: str, ref: str):
        if part == EVALUATOR:
            raise PrivacyError(f"{ref} is evaluator data")
        super()._refuse(part, ref)


class EvaluatorView(_View):
    """Every partition. For scoring and audit only — never hand this to a
    learner."""
    _label = "evaluator view"


# -------------------------------------------------------------------- store

class EvidenceStore:
    def __init__(self, root: str, max_bytes: Optional[int] = None,
                 chunk_bytes: Optional[int] = None,
                 heldout_fraction: Optional[float] = None,
                 split_salt: Optional[str] = None,
                 max_pinned_frac: Optional[float] = None,
                 quarantine_frac: Optional[float] = None,
                 fsync: bool = True, readonly: bool = False, registry=None):
        self.root = os.path.abspath(root)
        self.readonly = bool(readonly)
        self.fsync = bool(fsync)
        self.registry = MIGRATIONS if registry is None else registry
        self._lock = threading.RLock()
        self.recovery: List[str] = []
        ident_path = os.path.join(self.root, "STORE.json")
        if os.path.exists(ident_path):
            ident = _load_json(ident_path)
            if not isinstance(ident.get("format"), int) or ident["format"] > ck.STORE_FORMAT:
                raise SchemaVersionError(
                    f"store at {self.root} has format {ident.get('format')!r}; "
                    f"this code reads up to {ck.STORE_FORMAT}")
            sp = ident["split"]
            for name, want in (("heldout_fraction", heldout_fraction),
                               ("salt", split_salt)):
                if want is not None and want != sp[name]:
                    raise ExperienceError(
                        f"store was created with split {name}={sp[name]!r}; "
                        f"reopening with {want!r} would re-deal episodes "
                        f"between dev and held-out (a leak). Use a new root.")
            self.max_bytes = int(max_bytes if max_bytes is not None else ident["max_bytes"])
            self.chunk_bytes = int(chunk_bytes if chunk_bytes is not None else ident["chunk_bytes"])
            # Retention policy is part of the store: a reopen that silently
            # fell back to a default would change which evidence is
            # protected (caught by the bounded-growth smoke).
            if max_pinned_frac is None:
                max_pinned_frac = ident["max_pinned_frac"]
            if quarantine_frac is None:
                quarantine_frac = ident["quarantine_frac"]
        else:
            if self.readonly:
                raise FileNotFoundError(f"no evidence store at {self.root}")
            if max_bytes is None:
                raise ValueError("max_bytes is required when creating a store")
            hf = 0.2 if heldout_fraction is None else float(heldout_fraction)
            ident = {"format": ck.STORE_FORMAT, "store_id": uuid.uuid4().hex,
                     "created": time.time(),
                     "split": split_spec(hf, split_salt or SPLIT_SALT),
                     "max_bytes": int(max_bytes),
                     "chunk_bytes": int(chunk_bytes or 256 * 1024),
                     "max_pinned_frac": float(0.5 if max_pinned_frac is None
                                              else max_pinned_frac),
                     "quarantine_frac": float(0.1 if quarantine_frac is None
                                              else quarantine_frac),
                     "migrated_from": None}
            max_pinned_frac = ident["max_pinned_frac"]
            quarantine_frac = ident["quarantine_frac"]
            os.makedirs(self.root, exist_ok=True)
            ck.atomic_write_json(ident_path, ident, fsync=self.fsync)
            self.max_bytes = ident["max_bytes"]
            self.chunk_bytes = ident["chunk_bytes"]
        if self.chunk_bytes < 1024:
            raise ValueError(f"chunk_bytes {self.chunk_bytes} < 1024")
        if self.max_bytes < 8 * self.chunk_bytes:
            raise ValueError(f"max_bytes {self.max_bytes} must be >= 8 x "
                             f"chunk_bytes ({8 * self.chunk_bytes}); a budget "
                             f"of a few chunks cannot hold a journal per "
                             f"partition plus anything evictable")
        if not (0.0 <= max_pinned_frac <= 0.8):
            raise ValueError("max_pinned_frac must be in [0, 0.8] so some "
                             "budget is always evictable")
        self.max_pinned_frac = float(max_pinned_frac)
        self.quarantine_frac = float(quarantine_frac)
        self.ident = ident
        self.heldout_fraction = float(ident["split"]["heldout_fraction"])
        self.split_salt = ident["split"]["salt"]
        self._ident_bytes = os.path.getsize(ident_path)
        self._qdir = os.path.join(self.root, "quarantine")
        self._mpath = os.path.join(self.root, "manifest.json")
        self._parts = {n: _Partition(n, os.path.join(self.root, n), self.registry)
                       for n in PARTS}
        if not self.readonly:
            for p in self._parts.values():
                os.makedirs(p.dir, exist_ok=True)
            os.makedirs(self._qdir, exist_ok=True)
        # store-level write-validation indexes
        self._obs_keys: Dict[Tuple, Tuple[str, int, str]] = {}
        self._last_obs: Dict[Tuple, Tuple[int, float, float]] = {}
        self._act_keys: Dict[Tuple, Tuple[str, int, str]] = {}
        self._pred_ids: Dict[str, Tuple[str, int]] = {}
        self._evidence_ids: Dict[str, Tuple[str, int]] = {}
        self._ended: Dict[Tuple, int] = {}
        self.bytes_written = 0              # raw line bytes appended, this open
        self._recover()

    # ---------------------------------------------------------- manifest
    def _write_manifest(self) -> None:
        if self.readonly:
            return
        self._manifest_bytes = ck.atomic_write_json(self._mpath, self.manifest,
                                                    fsync=self.fsync)

    def _note(self, msg: str) -> None:
        self.recovery.append(msg)

    def _quarantine(self, name: str, data: bytes = None, src: str = None) -> None:
        if self.readonly:
            self._note(f"(readonly) would quarantine {name}")
            return
        dst = os.path.join(self._qdir, name)
        if src is not None:
            os.replace(src, dst)
        else:
            ck.atomic_write_bytes(dst, data, fsync=self.fsync)
        q = self.manifest["quarantine"]
        q["files"].append(name)
        cap = int(self.quarantine_frac * self.max_bytes)
        while q["files"] and self._quarantine_bytes() > cap:
            old = q["files"].pop(0)
            try:
                os.remove(os.path.join(self._qdir, old))
            except OSError:
                pass
            q["deleted"] += 1
            self._note(f"quarantine over its {cap}-byte share: deleted {old}")

    def _quarantine_bytes(self) -> int:
        n = 0
        for f in self.manifest["quarantine"]["files"]:
            try:
                n += os.path.getsize(os.path.join(self._qdir, f))
            except OSError:
                pass
        return n

    # ---------------------------------------------------------- recovery
    def _recover(self) -> None:
        empty = {"format": ck.STORE_FORMAT, "chunks": {}, "next_cid": 1,
                 "seq_hwm": 0, "pins": {},
                 "evicted": {"chunks": 0, "records": 0, "bytes": 0,
                             "ranges": {n: [] for n in PARTS}, "recent": []},
                 "quarantine": {"files": [], "deleted": 0}}
        if os.path.exists(self._mpath):
            try:
                self.manifest = _load_json(self._mpath)
            except (OSError, ValueError) as e:
                self._note(f"manifest unreadable ({e}); rebuilding from chunks")
                if not self.readonly:
                    with open(self._mpath, "rb") as f:
                        self._quarantine(f"manifest-{int(time.time()*1e6)}.bad",
                                         f.read())
                self.manifest = empty
        else:
            self.manifest = empty
        self._manifest_bytes = (os.path.getsize(self._mpath)
                                if os.path.exists(self._mpath) else 0)
        chunks_m = self.manifest["chunks"]
        lines_by_part: Dict[str, List[Tuple[int, int, Dict]]] = {n: [] for n in PARTS}
        hwm = self.manifest["seq_hwm"]
        # Every cid on disk, BEFORE anything allocates one (salvage does):
        # a journal opened after the last manifest write can be ahead of it.
        max_cid = self.manifest["next_cid"] - 1
        for p in self._parts.values():
            if os.path.isdir(p.dir):
                for f in os.listdir(p.dir):
                    if f.startswith("open-") and f.endswith(".jsonl"):
                        max_cid = max(max_cid, int(f[5:13]))
                    elif f.startswith("chunk-") and f.endswith(".jsonl.gz"):
                        max_cid = max(max_cid, int(f[6:14]))
        self.manifest["next_cid"] = max_cid + 1

        for name, p in self._parts.items():
            p.evicted_ranges = [tuple(r) for r in
                                self.manifest["evicted"]["ranges"].get(name, [])]
            if not os.path.isdir(p.dir):
                continue
            files = sorted(os.listdir(p.dir))
            for f in files:
                if f.endswith(".tmp"):
                    self._note(f"{name}/{f}: interrupted atomic write; removed")
                    if not self.readonly:
                        os.remove(os.path.join(p.dir, f))
            journals = {int(f[5:13]): f for f in files
                        if f.startswith("open-") and f.endswith(".jsonl")}
            sealed = {int(f[6:14]): f for f in files
                      if f.startswith("chunk-") and f.endswith(".jsonl.gz")}
            # sealed chunks
            for cid, f in sorted(sealed.items()):
                path = os.path.join(p.dir, f)
                ent = chunks_m.get(str(cid))
                if ent is None:
                    if cid in journals:
                        self._note(f"{name}/{f}: sealed but not committed to "
                                   f"the manifest; its journal is "
                                   f"authoritative, orphan removed")
                        if not self.readonly:
                            os.remove(path)
                        continue
                    self._note(f"{name}/{f}: not in manifest; adopted after "
                               f"per-line verification")
                    ent = self._adopt(name, cid, path)
                    if ent is None:
                        continue
                elif cid in journals:
                    self._note(f"{name}/{journals[cid]}: already sealed; "
                               f"stale journal removed")
                    if not self.readonly:
                        os.remove(os.path.join(p.dir, journals[cid]))
                    del journals[cid]
                good = self._load_sealed(name, cid, path, ent)
                lines_by_part[name].extend(good)
            for cid_s, ent in list(chunks_m.items()):
                if ent["part"] == name and int(cid_s) not in sealed:
                    self._note(f"{name}: sealed chunk {cid_s} listed in the "
                               f"manifest is missing on disk ({ent['n']} "
                               f"records lost)")
                    del chunks_m[cid_s]
            # journals: all but the newest are sealed after loading
            for cid in sorted(journals):
                good = self._load_journal(name, cid, os.path.join(p.dir, journals[cid]))
                lines_by_part[name].extend(good)
                p.journal_cid = cid
            # older journals (should not happen) are handled below by
            # sealing on the next append; keep only the newest open.
            older = sorted(journals)[:-1]
            for cid in older:
                self._note(f"{name}: extra journal {cid}; will be sealed")

        # ingest in GLOBAL write order
        allrec = []
        for name, lst in lines_by_part.items():
            for cid, ln, d in lst:
                allrec.append((d["seq"], name, cid, ln, d))
        allrec.sort(key=lambda x: x[0])
        seen = set()
        for seq, name, cid, ln, d in allrec:
            if seq in seen:
                self._note(f"duplicate write seq {seq} in {name}; second copy "
                           f"ignored")
                continue
            seen.add(seq)
            hwm = max(hwm, seq)
            self._ingest(name, _Entry(seq, d["kind"], cid, ln, d["meta"], d["hash"]))
        self._next_seq = hwm + 1
        self.manifest["seq_hwm"] = hwm
        # seal older extra journals now
        if not self.readonly:
            for name, p in self._parts.items():
                for cid in sorted(c for c in p.chunk_seqs
                                  if c != p.journal_cid
                                  and str(c) not in self.manifest["chunks"]):
                    self._seal_lines(p, cid, self._lines_for(p, cid))
            self._write_manifest()

    def _lines_for(self, p: _Partition, cid: int) -> List[str]:
        return ck.read_journal(p.journal_path(cid))[0]

    def _adopt(self, name: str, cid: int, path: str):
        try:
            lines = ck.read_sealed(path)
        except ck.READ_ERRORS as e:
            self._note(f"{name}/chunk {cid} unreadable ({e}); quarantined")
            self._quarantine(f"{name}-chunk-{cid:08d}.gz.bad", src=path)
            return None
        seqs = [d["seq"] for d in (ck.decode_line(ln)[0] for ln in lines)
                if d is not None] or [0]
        ent = {"part": name, "file": os.path.basename(path),
               "sha256": ck.file_sha256(path), "bytes": os.path.getsize(path),
               "n": len(lines), "seq_lo": min(seqs), "seq_hi": max(seqs),
               "sealed_t": time.time()}
        self.manifest["chunks"][str(cid)] = ent
        return ent

    def _load_sealed(self, name, cid, path, ent):
        bad_file = ck.file_sha256(path) != ent["sha256"]
        try:
            lines = ck.read_sealed(path)
        except ck.READ_ERRORS as e:
            lines = None
            why = str(e)
        if not bad_file and lines is not None:
            good = []
            for i, ln in enumerate(lines):
                d, why = ck.decode_line(ln)
                if d is None or d["part"] != name:
                    bad_file = True
                    break
                good.append((cid, i, d))
            if not bad_file:
                return good
        # corrupt sealed chunk: quarantine, salvage self-verifying lines
        salv = []
        for ln in (lines or []):
            d, _ = ck.decode_line(ln)
            if d is not None and d["part"] == name:
                salv.append(ln)
        self._note(f"{name}/chunk {cid}: sha256 mismatch or damaged gzip; "
                   f"quarantined, {len(salv)}/{ent['n']} lines salvaged")
        del self.manifest["chunks"][str(cid)]
        self._quarantine(f"{name}-chunk-{cid:08d}.gz.bad", src=path)
        if not salv:
            return []
        if self.readonly:
            return [(cid, i, ck.decode_line(ln)[0]) for i, ln in enumerate(salv)]
        ncid = self.manifest["next_cid"]
        self.manifest["next_cid"] = ncid + 1
        p = self._parts[name]
        self._write_sealed(p, ncid, salv)
        return [(ncid, i, ck.decode_line(ln)[0]) for i, ln in enumerate(salv)]

    def _load_journal(self, name, cid, path):
        lines, tail = ck.read_journal(path)
        good, bad = [], []
        for ln in lines:
            d, why = ck.decode_line(ln)
            if d is None or d["part"] != name:
                bad.append(ln.encode("utf-8", errors="replace"))
            else:
                good.append(ln)
        if tail:
            bad.append(tail)
        p = self._parts[name]
        if bad:
            self._note(f"{name}/journal {cid}: {len(bad)} damaged/truncated "
                       f"line(s) quarantined ({len(good)} intact)")
            self._quarantine(f"{name}-journal-{cid:08d}-{int(time.time()*1e6)}.bad",
                             b"".join(bad))
            if not self.readonly:
                ck.atomic_write_bytes(path, "".join(good).encode("utf-8"),
                                      fsync=self.fsync)
        p.journal_lines = list(good)
        p.journal_bytes = sum(len(x.encode("utf-8")) for x in good)
        return [(cid, i, ck.decode_line(ln)[0]) for i, ln in enumerate(good)]

    # ---------------------------------------------------------- ingest
    def _ingest(self, part: str, e: _Entry) -> None:
        self._parts[part].add(e)
        m = e.meta
        if e.kind == "observation":
            self._obs_keys[(m["env"], m["stream"], m["episode"], m["oseq"],
                            m["channel"])] = (part, e.seq, m["content_hash"])
            k = (m["env"], m["stream"], m["episode"], m["channel"])
            self._last_obs[k] = (m["oseq"], m["t_wall"], m["t_env"])
        elif e.kind == "action":
            self._act_keys[(m["env"], m["stream"], m["episode"], m["oseq"])] = (
                part, e.seq, m["content_hash"])
        elif e.kind == "prediction":
            self._pred_ids[m["pid"]] = (part, e.seq)
        elif e.kind == "evidence":
            self._evidence_ids[m["eid"]] = (part, e.seq)
        elif e.kind == "episode_end":
            self._ended[(m["env"], m["stream"], m["episode"])] = e.seq

    def _forget(self, part: str, e: _Entry) -> None:
        m = e.meta
        if e.kind == "observation":
            self._obs_keys.pop((m["env"], m["stream"], m["episode"], m["oseq"],
                                m["channel"]), None)
        elif e.kind == "action":
            self._act_keys.pop((m["env"], m["stream"], m["episode"], m["oseq"]), None)
        elif e.kind == "prediction":
            self._pred_ids.pop(m["pid"], None)
        elif e.kind == "evidence":
            self._evidence_ids.pop(m["eid"], None)
        # _last_obs and _ended are kept: ordering rules outlive eviction.

    # ---------------------------------------------------------- writing
    def _split_of(self, env: str, stream: str, episode: str) -> str:
        return assign_split(split_scope(env, stream), episode,
                            self.heldout_fraction, self.split_salt)

    def _append(self, part: str, kind: str, meta: Dict, body: Dict) -> str:
        if self.readonly:
            raise ExperienceError("store opened readonly")
        p = self._parts[part]
        if p.journal_cid is None:
            p.journal_cid = self.manifest["next_cid"]
            self.manifest["next_cid"] += 1
            p.journal_lines, p.journal_bytes = [], 0
        seq = self._next_seq
        line, h = ck.encode_line(seq, kind, part, meta, body)
        data = line.encode("utf-8")
        if p._fh is None:
            p._fh = open(p.journal_path(p.journal_cid), "ab")
        p._fh.write(data)
        p._fh.flush()
        if self.fsync:
            os.fsync(p._fh.fileno())
        self._next_seq = seq + 1
        self.bytes_written += len(data)
        p.journal_lines.append(line)
        p.journal_bytes += len(data)
        self._ingest(part, _Entry(seq, kind, p.journal_cid,
                                  len(p.journal_lines) - 1, meta, h))
        if p.journal_bytes >= self.chunk_bytes:
            self._seal(p)
        self._enforce_budget()
        return make_ref(part, seq)

    def _write_sealed(self, p: _Partition, cid: int, lines: List[str]) -> None:
        path = p.sealed_path(cid)
        data = ck.gzip_lines(lines)
        ck.atomic_write_bytes(path, data, fsync=self.fsync)
        seqs = [ck.decode_line(ln)[0]["seq"] for ln in lines]
        self.manifest["chunks"][str(cid)] = {
            "part": p.name, "file": os.path.basename(path),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "raw_bytes": sum(len(x.encode()) for x in lines), "n": len(lines),
            "seq_lo": min(seqs), "seq_hi": max(seqs), "sealed_t": time.time()}

    def _seal_lines(self, p: _Partition, cid: int, lines: List[str]) -> None:
        self._write_sealed(p, cid, lines)
        self.manifest["seq_hwm"] = self._next_seq - 1
        self._write_manifest()                      # COMMIT
        jp = p.journal_path(cid)
        if os.path.exists(jp):
            os.remove(jp)

    def _seal(self, p: _Partition) -> None:
        if p.journal_cid is None or not p.journal_lines:
            return
        if p._fh is not None:
            p._fh.close()
            p._fh = None
        cid = p.journal_cid
        lines = p.journal_lines
        self._seal_lines(p, cid, lines)
        p._cache[cid] = lines
        p.journal_cid, p.journal_lines, p.journal_bytes = None, [], 0

    def flush(self) -> None:
        """Seal every journal (e.g. before archiving the store)."""
        with self._lock:
            for p in self._parts.values():
                self._seal(p)

    def close(self) -> None:
        with self._lock:
            for p in self._parts.values():
                if p._fh is not None:
                    p._fh.close()
                    p._fh = None

    # ---------------------------------------------------------- budget
    def disk_bytes(self) -> int:
        sealed = sum(c["bytes"] for c in self.manifest["chunks"].values())
        journals = sum(p.journal_bytes for p in self._parts.values())
        return (sealed + journals + self._manifest_bytes + self._ident_bytes
                + self._quarantine_bytes())

    def _explicit_pin_cids(self) -> Dict[int, int]:
        out = {}
        for ref in self.manifest["pins"]:
            part, seq = parse_ref(ref)
            e = self._parts[part].entries.get(seq)
            if e is None:
                continue
            c = self.manifest["chunks"].get(str(e.cid))
            out[e.cid] = c["bytes"] if c else self.chunk_bytes
        return out

    def pinned_chunks(self) -> Dict[int, str]:
        """cid -> why it is pinned ("explicit" or "heldout")."""
        cap = self.max_pinned_frac * self.max_bytes
        explicit = self._explicit_pin_cids()
        used = sum(explicit.values())
        out = {c: "explicit" for c in explicit}
        # OLDEST held-out DATA first (seq_lo, not cid): a stable eval set.
        held = [c for _, c in sorted(
            (v["seq_lo"], int(c)) for c, v in self.manifest["chunks"].items()
            if v["part"] == HELDOUT and int(c) not in out)]
        for c in held:
            b = self.manifest["chunks"][str(c)]["bytes"]
            if used + b > cap:
                break
            used += b
            out[c] = "heldout"
        return out

    def _enforce_budget(self) -> None:
        guard = 0
        while self.disk_bytes() > self.max_bytes:
            guard += 1
            if guard > 100000:                       # pragma: no cover
                raise StoreBudgetError("budget enforcement did not converge")
            pinned = self.pinned_chunks()
            # Oldest DATA first: order by the chunk's newest write seq, not
            # its cid. A cid is allocated when a journal OPENS, so a slowly
            # filling partition's chunk has an old cid and fresh records —
            # evicting by cid destroyed observations an in-flight outcome
            # still had to cite (caught by the bounded-growth smoke).
            cands = sorted((v["seq_hi"], int(c))
                           for c, v in self.manifest["chunks"].items()
                           if int(c) not in pinned)
            if cands:
                self._evict(cands[0][1])
                continue
            open_parts = [p for p in self._parts.values() if p.journal_lines
                          and p.journal_cid not in pinned]
            if open_parts:
                self._seal(max(open_parts, key=lambda p: p.journal_bytes))
                continue
            raise StoreBudgetError(
                f"store at {self.disk_bytes()} bytes > max_bytes "
                f"{self.max_bytes} and everything left is pinned; unpin "
                f"evidence (unpin/unpin_mechanism) or raise max_bytes")

    def _evict(self, cid: int) -> None:
        ent = self.manifest["chunks"].pop(str(cid))
        p = self._parts[ent["part"]]
        gone = p.remove_chunk(cid)
        for e in gone:
            self._forget(p.name, e)
        try:
            os.remove(p.sealed_path(cid))
        except OSError:
            pass
        ev = self.manifest["evicted"]
        ev["chunks"] += 1
        ev.setdefault("by_part", {})
        ev["by_part"][p.name] = ev["by_part"].get(p.name, 0) + 1
        ev["records"] += len(gone)
        ev["bytes"] += ent["bytes"]
        if gone:
            ev["ranges"].setdefault(p.name, []).append(
                [min(e.seq for e in gone), max(e.seq for e in gone)])
            # Merge ranges with no LIVE record of this partition between
            # them, so the manifest stays bounded under endless eviction.
            rs = sorted(ev["ranges"][p.name])
            live = sorted(p.entries)
            merged = [rs[0]]
            for lo, hi in rs[1:]:
                i = bisect.bisect_right(live, merged[-1][1])
                if i >= len(live) or live[i] >= lo:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            ev["ranges"][p.name] = merged
            p.evicted_ranges = [tuple(r) for r in merged]
        ev["recent"].append({"cid": cid, "part": p.name, "n": len(gone),
                             "t": time.time()})
        del ev["recent"][:-_RECENT_LOG]
        self._write_manifest()

    # ---------------------------------------------------------- pins
    def pin(self, ref: str, reason: str) -> None:
        """Exempt the chunk holding `ref` from eviction. Refused past the
        pinned share of the budget — the escape is `unpin`."""
        with self._lock:
            part, seq = parse_ref(ref)
            e = self._parts[part].entries.get(seq)
            if e is None:
                self._parts[part].read_line(seq)
            if not isinstance(reason, str) or not reason:
                raise ValueError("a pin needs a reason")
            cur = self._explicit_pin_cids()
            if e.cid not in cur:
                c = self.manifest["chunks"].get(str(e.cid))
                add = c["bytes"] if c else self.chunk_bytes
                cap = self.max_pinned_frac * self.max_bytes
                if sum(cur.values()) + add > cap:
                    raise StoreBudgetError(
                        f"pinning {ref} would put explicit pins at "
                        f"{sum(cur.values()) + add} bytes > the pinned share "
                        f"{cap:.0f}; unpin older evidence first or raise "
                        f"max_bytes")
            self.manifest["pins"][ref] = reason
            self._write_manifest()

    def unpin(self, ref: str) -> None:
        with self._lock:
            self.manifest["pins"].pop(ref, None)
            self._write_manifest()

    def pin_mechanism(self, mech: Mechanism) -> List[str]:
        """Pin every store ref a promoted mechanism cites as evidence."""
        if not isinstance(mech, Mechanism):
            raise TypeError("pin_mechanism needs a Mechanism record")
        refs = [r for r in mech.evidence_refs]
        for r in refs:
            self.pin(r, f"mechanism {mech.mechanism_id} v{mech.version}")
        return refs

    def unpin_mechanism(self, mech: Mechanism) -> None:
        for r in mech.evidence_refs:
            self.unpin(r)

    # ---------------------------------------------------------- log API
    def log_observation(self, obs: Observation) -> str:
        with self._lock:
            if isinstance(obs, Prediction):
                raise ProvenanceError("a Prediction is not an observation; "
                                      "use log_prediction")
            if not isinstance(obs, Observation):
                raise TypeError(f"expected Observation, got {type(obs).__name__}")
            if obs.provenance == "imagined":
                raise ProvenanceError(
                    f"imagined observation {obs.channel!r} refused: a model's "
                    f"output is stored as a Prediction and cannot be filed as "
                    f"something the agent observed (plan §7.4)")
            sk = (obs.environment, obs.stream, obs.episode)
            if sk in self._ended:
                raise OrderingError(f"episode {sk} already ended; a reset or "
                                    f"client recovery starts a NEW episode id")
            key = sk + (obs.seq, obs.channel)
            if key in self._obs_keys:
                raise DuplicateRecordError(f"observation {key} already "
                                           f"recorded; raw records are "
                                           f"immutable")
            last = self._last_obs.get(sk + (obs.channel,))
            if last is not None:
                if obs.seq <= last[0]:
                    raise OrderingError(f"{obs.channel} seq {obs.seq} after "
                                        f"{last[0]} in {sk}")
                if obs.t_wall < last[1] or obs.t_env < last[2]:
                    raise OrderingError(f"{obs.channel} time runs backwards in "
                                        f"{sk}")
            part = (EVALUATOR if obs.provenance == "evaluator"
                    else self._split_of(*sk))
            body = to_dict(obs)
            meta = {"env": obs.environment, "stream": obs.stream,
                    "episode": obs.episode, "oseq": obs.seq,
                    "channel": obs.channel, "prov": obs.provenance,
                    "t_wall": obs.t_wall, "t_env": obs.t_env,
                    "content_hash": _hash_body(body)}
            return self._append(part, "observation", meta, body)

    def log_action(self, act: Action) -> str:
        with self._lock:
            if not isinstance(act, Action):
                raise TypeError(f"expected Action, got {type(act).__name__}")
            sk = (act.environment, act.stream, act.episode)
            if sk in self._ended:
                raise OrderingError(f"episode {sk} already ended")
            if act.t_complete is UNKNOWN:
                raise ExperienceError(
                    "action has no t_complete: the store records ACTUAL "
                    "timing; log the executed action once it completed")
            key = sk + (act.seq,)
            if key in self._act_keys:
                raise DuplicateRecordError(f"action {key} already recorded")
            part = self._split_of(*sk)
            prior = self._parts[part].obs_at.get(key, [])
            if not prior:
                raise OrderingError(
                    f"action at seq {act.seq} in {sk} answers no logged "
                    f"observation at seq {act.seq} (obs t is logged before "
                    f"action t)")
            t_obs = max(self._parts[part].entries[s].meta["t_wall"]
                        for s in prior)
            if act.t_dispatch < t_obs:
                raise OrderingError(f"action dispatched at {act.t_dispatch} "
                                    f"before the observation it answers "
                                    f"({t_obs})")
            body = to_dict(act)
            meta = {"env": act.environment, "stream": act.stream,
                    "episode": act.episode, "oseq": act.seq,
                    "spec": act.spec_id, "command": _cmd_key(act.command),
                    "duration": _num(act.duration),
                    "t_dispatch": act.t_dispatch, "t_complete": act.t_complete,
                    "elapsed": act.t_complete - act.t_dispatch,
                    "content_hash": _hash_body(body)}
            return self._append(part, "action", meta, body)

    def log_prediction(self, pred: Prediction, context: ObsRef) -> str:
        """Record a prediction BEFORE its outcome. Returns its ref; the ref's
        seq is the write sequence number every later outcome is checked
        against."""
        with self._lock:
            if not isinstance(pred, Prediction):
                raise TypeError(f"expected Prediction, got {type(pred).__name__}")
            if not isinstance(context, ObsRef):
                raise TypeError("context must be the ObsRef the prediction "
                                "was conditioned on")
            if pred.prediction_id in self._pred_ids:
                raise DuplicateRecordError(f"prediction {pred.prediction_id!r} "
                                           f"already recorded")
            s = context.scope
            sk = (s.environment, s.stream, s.episode)
            hit = self._obs_keys.get(sk + (context.seq, context.channel))
            if hit is None:
                raise OrderingError(f"context {context} is not a logged "
                                    f"observation")
            if hit[0] == EVALUATOR:
                raise ProvenanceError("a prediction may not be conditioned on "
                                      "evaluator data")
            if sk in self._ended:
                raise OrderingError(f"episode {sk} already ended; nothing "
                                    f"can follow to be this prediction's "
                                    f"outcome")
            body = to_dict(pred)
            meta = {"pid": pred.prediction_id, "snapshot": pred.snapshot_id,
                    "models": {k: str(v) for k, v in pred.model_versions.items()},
                    "ctx": {"env": s.environment, "stream": s.stream,
                            "episode": s.episode, "oseq": context.seq,
                            "channel": context.channel},
                    "t_wall": pred.t_wall, "horizon": pred.horizon,
                    "env": s.environment, "stream": s.stream,
                    "episode": s.episode,
                    "content_hash": _hash_body(body)}
            return self._append(self._split_of(*sk), "prediction", meta, body)

    def log_outcome(self, ev: Evidence) -> str:
        with self._lock:
            if not isinstance(ev, Evidence):
                raise TypeError(f"expected Evidence, got {type(ev).__name__}")
            if ev.provenance == "imagined":
                raise ProvenanceError(
                    "imagined evidence refused: a model cannot certify its "
                    "own imagined outcomes as real evidence (plan §7.4)")
            if ev.evidence_id in self._evidence_ids:
                raise DuplicateRecordError(f"evidence {ev.evidence_id!r} "
                                           f"already recorded")
            sc = ev.scope
            sk = (sc.environment, sc.stream, sc.episode)
            wseq = {}
            for o in ev.observations:
                hit = self._obs_keys.get(sk + (o.seq, o.channel))
                last = self._last_obs.get(sk + (o.channel,))
                if hit is None and last is not None and last[0] >= o.seq:
                    # Same refusal (provenance cannot be verified), but say
                    # which way it can happen: a forged seq, or a budget so
                    # small the context was evicted before its outcome.
                    raise ProvenanceError(
                        f"evidence observation {o.channel}@{o.seq} is not in "
                        f"the store: never recorded at that seq, or evicted "
                        f"before its outcome arrived (raise max_bytes)")
                if hit is None:
                    raise ProvenanceError(
                        f"evidence observation {o.channel}@{o.seq} was never "
                        f"recorded through log_observation; evidence may only "
                        f"cite what the store observed")
                if hit[2] != content_hash(o):
                    raise ProvenanceError(
                        f"evidence observation {o.channel}@{o.seq} differs "
                        f"from the recorded raw observation")
                wseq[(o.seq, o.channel)] = hit[1]
            for a in ev.actions:
                hit = self._act_keys.get(sk + (a.seq,))
                if hit is None or hit[2] != content_hash(a):
                    raise ProvenanceError(f"evidence action @{a.seq} is not "
                                          f"the recorded executed action")
            for pid in ev.prediction_refs:
                where = self._pred_ids.get(pid)
                if where is None:
                    raise OrderingError(
                        f"prediction {pid!r} is not recorded (never logged, or "
                        f"evicted): an outcome can only be linked to a "
                        f"prediction provably written before it")
                ppart, pseq = where
                c = self._parts[ppart].entries[pseq].meta["ctx"]
                if (c["env"], c["stream"], c["episode"]) != sk:
                    raise OrderingError(
                        f"prediction {pid!r} was made in another scope "
                        f"({c['env']}/{c['stream']}/{c['episode']}); an "
                        f"outcome never spans a reset or stream")
                after = [(k, w) for k, w in wseq.items() if k[0] > c["oseq"]]
                if not after:
                    raise OrderingError(
                        f"evidence has no observation after prediction "
                        f"{pid!r}'s context seq {c['oseq']}; it cannot be "
                        f"its outcome")
                late = [k for k, w in after if w < pseq]
                if late:
                    raise OrderingError(
                        f"prediction {pid!r} (write seq {pseq}) was written "
                        f"AFTER outcome observations {late} arrived; a "
                        f"prediction made with the answer in hand is not a "
                        f"prediction")
            part = (EVALUATOR if ev.provenance == "evaluator"
                    else self._split_of(*sk))
            body = to_dict(ev)
            meta = {"eid": ev.evidence_id, "env": sc.environment,
                    "stream": sc.stream, "episode": sc.episode,
                    "oseq_lo": min(o.seq for o in ev.observations),
                    "oseq_hi": max(o.seq for o in ev.observations),
                    "channels": sorted({o.channel for o in ev.observations}),
                    "prov": ev.provenance, "validity": ev.validity,
                    "pids": list(ev.prediction_refs),
                    "specs": [a.spec_id for a in ev.actions],
                    "commands": [_cmd_key(a.command) for a in ev.actions],
                    "t_wall_lo": min(o.t_wall for o in ev.observations),
                    "t_wall_hi": max(o.t_wall for o in ev.observations),
                    "content_hash": _hash_body(body)}
            return self._append(part, "evidence", meta, body)

    def log_episode_end(self, scope: Scope, reason: str, last_seq: int,
                        t_wall: float, detail: str = "") -> str:
        """terminated (env's own end, no bootstrap), truncated (external cut,
        bootstrap legitimate), client_recovery (not an env end; the next
        observations belong to a fresh episode id)."""
        with self._lock:
            if not isinstance(scope, Scope):
                raise TypeError("scope must be a Scope")
            if reason not in END_REASONS:
                raise ValueError(f"end reason must be one of {END_REASONS}, "
                                 f"got {reason!r}")
            sk = (scope.environment, scope.stream, scope.episode)
            if sk in self._ended:
                raise DuplicateRecordError(f"episode {sk} already ended")
            if isinstance(last_seq, bool) or not isinstance(last_seq, int) or last_seq < 0:
                raise ValueError("last_seq must be an int >= 0")
            t_wall = float(t_wall)
            if not math.isfinite(t_wall):
                raise ValueError("t_wall must be finite")
            meta = {"env": sk[0], "stream": sk[1], "episode": sk[2],
                    "oseq": last_seq, "reason": reason, "t_wall": t_wall}
            body = dict(meta, detail=str(detail), schema="episode_end",
                        version=1)
            return self._append(self._split_of(*sk), "episode_end", meta, body)

    def interpret(self, target: str, interpreter: str, interpreter_version: int,
                  content: Dict[str, Any], supersedes: Optional[str] = None,
                  t_wall: Optional[float] = None) -> str:
        """Attach a revisable reading to a raw record. Never mutates it."""
        with self._lock:
            part, tseq = parse_ref(target)
            p = self._parts[part]
            te = p.entries.get(tseq)
            if te is None:
                p.read_line(tseq)
            if te.kind == "interpretation":
                raise ExperienceError("interpret a raw record, not an "
                                      "interpretation (use supersedes)")
            if not isinstance(interpreter, str) or not interpreter:
                raise ValueError("interpreter must be a non-empty str")
            if (isinstance(interpreter_version, bool)
                    or not isinstance(interpreter_version, int)
                    or interpreter_version < 1):
                raise ValueError("interpreter_version must be an int >= 1")
            if not isinstance(content, dict):
                raise TypeError("content must be a dict")
            if supersedes is not None:
                spart, sseq = parse_ref(supersedes)
                se = p.entries.get(sseq) if spart == part else None
                if se is None or se.kind != "interpretation" \
                        or se.meta["target"] != target:
                    raise ExperienceError(f"{supersedes} is not an "
                                          f"interpretation of {target}")
                if sseq in p.superseded:
                    raise ExperienceError(
                        f"{supersedes} was already superseded by "
                        f"{make_ref(part, p.superseded[sseq])}; supersede the "
                        f"current one (no forks)")
            enc = encode_value(content)
            body = {"target": target, "interpreter": interpreter,
                    "interpreter_version": interpreter_version,
                    "supersedes": supersedes, "content": enc,
                    "t_wall": float(time.time() if t_wall is None else t_wall),
                    "schema": "interpretation", "version": 1}
            meta = {"target": target, "interpreter": interpreter,
                    "iversion": interpreter_version, "supersedes": supersedes,
                    "ckind": content.get("kind") if isinstance(
                        content.get("kind"), str) else None}
            return self._append(part, "interpretation", meta, body)

    def score(self, outcome_ref: str, prediction_id: str, error: float,
              contradicts: bool, interpreter: str, interpreter_version: int,
              supersedes: Optional[str] = None) -> str:
        """A prediction_score interpretation (what failure and contradiction
        retrieval read). Scores are revisable; outcomes are not."""
        with self._lock:
            return self._score(outcome_ref, prediction_id, error, contradicts,
                               interpreter, interpreter_version, supersedes)

    def _score(self, outcome_ref, prediction_id, error, contradicts,
               interpreter, interpreter_version, supersedes):
        part, seq = parse_ref(outcome_ref)
        e = self._parts[part].entries.get(seq)
        if e is None or e.kind != "evidence":
            raise ExperienceError(f"{outcome_ref} is not a live evidence record")
        if prediction_id not in e.meta["pids"]:
            raise ExperienceError(f"{outcome_ref} is not linked to "
                                  f"{prediction_id!r}")
        error = float(error)
        if not math.isfinite(error) or error < 0:
            raise ValueError(f"error must be finite and >= 0, got {error}")
        return self.interpret(outcome_ref, interpreter, interpreter_version,
                              {"kind": "prediction_score",
                               "prediction_id": prediction_id,
                               "error": error, "contradicts": bool(contradicts)},
                              supersedes=supersedes)

    # ---------------------------------------------------------- views
    def learning_view(self) -> LearningView:
        return LearningView({DEV: self._parts[DEV]})

    def heldout_view(self) -> HeldoutView:
        return HeldoutView({HELDOUT: self._parts[HELDOUT]})

    def evaluator_view(self) -> EvaluatorView:
        return EvaluatorView(self._parts)

    def get(self, ref: str):
        """Non-evaluator read (dev or held-out). Evaluator data only through
        evaluator_view()."""
        part, _ = parse_ref(ref)
        if part == EVALUATOR:
            raise PrivacyError(f"{ref}: evaluator data is read only through "
                               f"evaluator_view()")
        return EvaluatorView(self._parts).get(ref)

    def episode_key(self, ref: str) -> EpisodeKey:
        """The split unit a ref belongs to (for leakage checks)."""
        part, seq = parse_ref(ref)
        p = self._parts[part]
        e = p.entries.get(seq)
        if e is None:
            p.read_line(seq)
        m = e.meta
        if e.kind == "interpretation":
            return self.episode_key(m["target"])
        return EpisodeKey(split_scope(m["env"], m["stream"]), m["episode"])

    def assert_no_leak(self, train_refs: Iterable[str],
                       eval_refs: Iterable[str]) -> None:
        """SplitLeakError if any episode feeds both sides (by episode key,
        so adjacent frames of one episode are caught)."""
        assert_no_leak([self.episode_key(r) for r in train_refs],
                       [self.episode_key(r) for r in eval_refs])

    # ---------------------------------------------------------- privileged
    def iter_lines(self) -> List[Tuple[str, Dict]]:
        """Every live line, all partitions, in write order. For migration and
        audit tools; not a retrieval API."""
        out = []
        for name, p in self._parts.items():
            for seq in p.entries:
                out.append((seq, name))
        out.sort()
        return [(n, self._parts[n].read_line(s)) for s, n in out]

    def _import_line(self, d: Dict) -> None:
        """Append a line preserving its write seq (migration only)."""
        if d["seq"] < self._next_seq:
            raise OrderingError("import must preserve increasing write seq")
        self._next_seq = d["seq"]
        self._append(d["part"], d["kind"], d["meta"], d["body"])

    def stats(self) -> Dict[str, Any]:
        per = {}
        for n, p in self._parts.items():
            per[n] = {"records": len(p.entries),
                      "by_kind": {k: len(v) for k, v in p.by_kind.items() if v},
                      "sealed_chunks": sum(1 for c in self.manifest["chunks"].values()
                                           if c["part"] == n),
                      "journal_bytes": p.journal_bytes}
        raw = sum(c.get("raw_bytes", c["bytes"]) for c in self.manifest["chunks"].values())
        comp = sum(c["bytes"] for c in self.manifest["chunks"].values())
        return {"disk_bytes": self.disk_bytes(), "max_bytes": self.max_bytes,
                "partitions": per, "evicted": {k: v for k, v in
                                               self.manifest["evicted"].items()
                                               if k != "ranges"},
                "quarantine": dict(self.manifest["quarantine"]),
                "pinned_chunks": len(self.pinned_chunks()),
                "compression_ratio": (raw / comp) if comp else UNKNOWN,
                "recovery": list(self.recovery),
                "next_write_seq": self._next_seq,
                "bytes_written": self.bytes_written}


def _hash_body(body: Dict) -> str:
    return hashlib.sha256(ck.canonical(body).encode()).hexdigest()


def _load_json(path: str):
    import json
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- migration

def migrate_store(src_root: str, dst_root: str, registry=None,
                  max_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Convert every record to the current schema versions into a NEW store.

    The source is opened readonly (no recovery writes, no quarantine moves)
    and is never modified. Write seqs, partitions, pins and interpretation
    links are preserved, so prediction-before-outcome still holds in the
    copy. Each migrated line's meta records the source line hash."""
    src_abs = os.path.realpath(src_root)
    dst_abs = os.path.realpath(dst_root)
    if src_abs == dst_abs or dst_abs.startswith(src_abs + os.sep) \
            or src_abs.startswith(dst_abs + os.sep):
        raise ExperienceError("migration target must be a new location, "
                              "neither the source nor nested with it (plan §8)")
    if os.path.exists(dst_abs) and os.listdir(dst_abs):
        raise ExperienceError(f"{dst_abs} is not empty; migration never "
                              f"overwrites")
    reg = MIGRATIONS if registry is None else registry
    src = EvidenceStore(src_root, readonly=True, registry=reg)
    dst = EvidenceStore(dst_root, max_bytes=max_bytes or src.max_bytes,
                        chunk_bytes=src.chunk_bytes,
                        heldout_fraction=src.heldout_fraction,
                        split_salt=src.split_salt, fsync=src.fsync)
    report = {"records": 0, "upgraded": 0, "by_kind": {}}
    for part, d in src.iter_lines():
        kind = d["kind"]
        body = d["body"]
        if kind in RECORD_KINDS:
            old_v = body.get("version")
            rec = from_dict(body, registry=reg)
            body = to_dict(rec)
            if old_v != body["version"]:
                report["upgraded"] += 1
        elif kind == "interpretation":
            body = dict(body, content=encode_value(
                decode_value(body["content"], reg)))
        meta = dict(d["meta"])
        if "content_hash" in meta:
            meta["content_hash"] = _hash_body(body)
        meta["migrated_from_hash"] = d["hash"]
        dst._import_line({"seq": d["seq"], "kind": kind, "part": part,
                          "meta": meta, "body": body})
        report["records"] += 1
        report["by_kind"][kind] = report["by_kind"].get(kind, 0) + 1
    for ref, reason in src.manifest["pins"].items():
        try:
            dst.pin(ref, reason)
        except (KeyError, RecordEvicted):
            pass
    dst.flush()
    dst.ident["migrated_from"] = {
        "root": src.root, "store_id": src.ident["store_id"],
        "manifest_sha256": (ck.file_sha256(src._mpath)
                            if os.path.exists(src._mpath) else None)}
    ck.atomic_write_json(os.path.join(dst.root, "STORE.json"), dst.ident,
                         fsync=dst.fsync)
    dst._ident_bytes = os.path.getsize(os.path.join(dst.root, "STORE.json"))
    dst.close()
    return report
