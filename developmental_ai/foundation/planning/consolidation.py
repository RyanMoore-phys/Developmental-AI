"""Episode memory, candidate knowledge and consolidated knowledge — kept apart
(plan Stage 11 items 5-7, plan §7.4 last bullet).

WHAT IS CLAIMED
    1. THREE STORES, THREE STATES. EpisodeMemory holds REFERENCES to what
       happened (evidence ids / episode keys; bounded FIFO, no knowledge).
       KnowledgeStore holds mechanisms and skills as CANDIDATES until real
       validations promote them to CONSOLIDATED. Nothing moves between the
       two without `promote` / `revalidate`, and both are driven by real
       outcomes only (imagined -> ProvenanceError).
    2. RE-EVALUATION ON CHANGE. Every consolidated item is a dependent of
       the perception RepresentationRegistry (foundation.perception) bound
       to the semver it was validated under. A MAJOR bump without a
       translator — or a detected dynamics change (ReliabilityMonitor's
       CUSUM alarm) — sets status "needs_revalidation"; `usable()` is then
       False until `revalidate(passed=True)` rebinds it to the current
       version. A failed revalidation demotes it to candidate (it is not
       deleted: its history stays).
    3. RETENTION. RetentionMonitor keeps a bounded reservoir of probe states
       per skill and a time series of competence measured ON THOSE PROBES
       (real executions). forgetting() = best past rate - current rate,
       flagged "significant" only when the Beta intervals of the best and
       current measurements do not overlap — a single unlucky probe run is
       not forgetting.
    4. PERSISTENCE. save_planning_state / load_planning_state write ONE JSON
       file (contracts codec values; records as codec __record__ values) atomically
       (tmp + fsync + os.replace) with skills, mechanism records, the
       knowledge store (statuses and bound versions), retention, and the
       planner/reliability state — identity (ids) and versions intact. A
       store loaded AFTER a representation bump comes back
       needs_revalidation, never silently valid.

ESCAPE PATH (CLAUDE.md §4.1)
    needs_revalidation is not a latch: revalidate() is reached by the
    caller collecting real outcomes under the new representation/dynamics
    (refit, validate) — the same loop that produced the item. A failed
    revalidation demotes to candidate, from which promote() is reachable.

Never touches skill_bank_mc_curiosity/ or the live loop (plan §8).
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import ContractError, Mechanism, Record, Skill
from ..contracts.codec import decode_value, encode_value
from ..perception.errors import StaleDependentError
from ..perception.versioning import RepresentationRegistry, SemVer
from .planner import _check_real
from .skills import BetaCompetence, LearnedSkill

STATUSES = ("candidate", "consolidated", "needs_revalidation")
KINDS = ("mechanism", "skill")
FILE_SCHEMA = "foundation.planning.state"
FILE_VERSION = 1


class EpisodeMemory:
    """Bounded FIFO of episode references. Holds pointers, never knowledge."""

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = int(max_episodes)
        self._eps: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.evicted = 0

    def add(self, episode_key: str, evidence_refs: Sequence[str], meta=None) -> None:
        if not isinstance(episode_key, str) or not episode_key:
            raise ContractError("episode_key must be a non-empty str")
        self._eps[episode_key] = {"evidence_refs": [str(r) for r in evidence_refs],
                                  "meta": dict(meta or {})}
        self._eps.move_to_end(episode_key)
        while len(self._eps) > self.max_episodes:
            self._eps.popitem(last=False)
            self.evicted += 1

    def refs(self, episode_key: str) -> List[str]:
        return list(self._eps[episode_key]["evidence_refs"])

    def __len__(self):
        return len(self._eps)

    def state_dict(self):
        return {"max_episodes": self.max_episodes, "evicted": self.evicted,
                "episodes": [[k, v] for k, v in self._eps.items()]}

    def load_state_dict(self, d):
        self.max_episodes, self.evicted = int(d["max_episodes"]), int(d["evicted"])
        self._eps = OrderedDict((k, v) for k, v in d["episodes"])


class KnowledgeStore:
    """Candidate and consolidated mechanisms/skills, bound to representation
    versions in a RepresentationRegistry."""

    def __init__(self, registry: RepresentationRegistry, promote_min_n: int = 10,
                 promote_min_rate: float = 0.8):
        if not isinstance(registry, RepresentationRegistry):
            raise ContractError("registry must be a perception RepresentationRegistry")
        self.registry = registry
        self.promote_min_n, self.promote_min_rate = int(promote_min_n), float(promote_min_rate)
        self.items: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _dep_id(kind, item_id):
        return f"planning/{kind}/{item_id}"

    def add_candidate(self, kind: str, item_id: str, record: Record, representation: str
                      ) -> None:
        if kind not in KINDS:
            raise ContractError(f"kind must be one of {KINDS}")
        want = Mechanism if kind == "mechanism" else Skill
        if not isinstance(record, want):
            raise ContractError(f"a {kind} candidate needs a contracts.{want.__name__}")
        rid = record.mechanism_id if kind == "mechanism" else record.skill_id
        if rid != item_id:
            raise ContractError(f"item_id {item_id!r} != record id {rid!r}")
        if item_id in self.items:
            raise ContractError(f"{item_id!r} already stored; use revalidate/replace")
        self.registry.current(representation)          # must be declared
        self.items[item_id] = {"kind": kind, "record": record,
                               "representation": representation,
                               "bound_version": None, "status": "candidate",
                               "validation": BetaCompetence(), "history": ["candidate"]}

    def record_validation(self, item_id: str, passed: bool, *, provenance: str,
                          evidence_id: Optional[str] = None) -> None:
        _check_real(provenance, f"validation of {item_id}")
        self._get(item_id)["validation"].record(bool(passed), provenance=provenance,
                                                evidence_id=evidence_id)

    def _get(self, item_id):
        try:
            return self.items[item_id]
        except KeyError:
            raise ContractError(f"unknown item {item_id!r}") from None

    def promote(self, item_id: str) -> Tuple[bool, str]:
        it = self._get(item_id)
        if it["status"] == "consolidated":
            return True, "already consolidated"
        v = it["validation"]
        rate = v.successes / v.n if v.n else 0.0
        if v.n < self.promote_min_n or rate < self.promote_min_rate:
            return False, (f"needs >= {self.promote_min_n} real validations at rate >= "
                           f"{self.promote_min_rate}; has {v.n} at {rate:.2f}")
        self._bind(item_id, it)
        it["status"] = "consolidated"
        it["history"].append(f"consolidated@{it['bound_version']}")
        return True, "consolidated"

    def _bind(self, item_id, it, data=None):
        dep = self._dep_id(it["kind"], item_id)
        try:
            self.registry.dependent(dep)
            self.registry.rebuild(dep, data if data is not None else it["record"])
        except Exception:
            self.registry.register_dependent(dep, it["kind"], it["representation"],
                                             it["record"])
        it["bound_version"] = str(self.registry.current(it["representation"]))

    def status(self, item_id: str) -> str:
        self.sync()
        return self._get(item_id)["status"]

    def sync(self) -> List[str]:
        """Mark consolidated items whose representation dependency went stale.
        Returns the ids newly marked."""
        marked = []
        for item_id, it in self.items.items():
            if it["status"] != "consolidated":
                continue
            try:
                self.registry.use(self._dep_id(it["kind"], item_id))
            except StaleDependentError as e:
                it["status"] = "needs_revalidation"
                it["history"].append(f"needs_revalidation: {e}")
                marked.append(item_id)
                continue
            dep = self.registry.dependent(self._dep_id(it["kind"], item_id))
            it["bound_version"] = str(dep.bound_version)   # translated/compatible
        return marked

    def usable(self, item_id: str) -> Tuple[bool, str]:
        st = self.status(item_id)
        return st == "consolidated", st

    def gate(self, item_id: str):
        """A PlanningController.knowledge_gate for one mechanism."""
        return lambda: self.usable(item_id)

    def on_dynamics_change(self, item_ids: Sequence[str], reason: str) -> List[str]:
        if not reason:
            raise ContractError("a dynamics change must state its reason")
        marked = []
        for i in item_ids:
            it = self._get(i)
            if it["status"] == "consolidated":
                it["status"] = "needs_revalidation"
                it["history"].append(f"needs_revalidation: dynamics: {reason}")
                marked.append(i)
        return marked

    def revalidate(self, item_id: str, passed: bool, *, provenance: str,
                   new_record: Optional[Record] = None) -> str:
        """Escape path. passed -> rebind to the CURRENT representation version
        and consolidate (with `new_record`, e.g. the refitted mechanism at
        version + 1); failed -> demote to candidate with a fresh validation
        record."""
        _check_real(provenance, f"revalidation of {item_id}")
        it = self._get(item_id)
        if it["status"] != "needs_revalidation":
            raise ContractError(f"{item_id!r} is {it['status']}, not needs_revalidation")
        if new_record is not None:
            kind = it["kind"]
            rid = new_record.mechanism_id if kind == "mechanism" else new_record.skill_id
            if rid != item_id:
                raise ContractError("a revalidated record keeps its identity")
            if new_record.version < it["record"].version:
                raise ContractError("a revalidated record cannot go back a version")
            it["record"] = new_record
        if passed:
            self._bind(item_id, it, it["record"])
            it["status"] = "consolidated"
            it["history"].append(f"revalidated@{it['bound_version']}")
        else:
            it["status"] = "candidate"
            it["validation"] = BetaCompetence()
            it["history"].append("revalidation failed: demoted to candidate")
        return it["status"]

    def record(self, item_id: str) -> Record:
        return self._get(item_id)["record"]

    def state_dict(self) -> Dict[str, Any]:
        out = {}
        for k, it in self.items.items():
            out[k] = {"kind": it["kind"], "record": it["record"],
                      "representation": it["representation"],
                      "bound_version": it["bound_version"], "status": it["status"],
                      "validation": it["validation"].state_dict(),
                      "history": list(it["history"])}
        return {"promote_min_n": self.promote_min_n,
                "promote_min_rate": self.promote_min_rate, "items": out}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Re-registers dependents bound to their SAVED versions; anything
        consolidated under an older major than the registry's current one
        comes back needs_revalidation."""
        self.promote_min_n, self.promote_min_rate = int(d["promote_min_n"]), \
            float(d["promote_min_rate"])
        self.items = {}
        for k, s in d["items"].items():
            it = {"kind": s["kind"], "record": s["record"],
                  "representation": s["representation"],
                  "bound_version": s["bound_version"], "status": s["status"],
                  "validation": BetaCompetence.from_state_dict(s["validation"]),
                  "history": list(s["history"])}
            self.items[k] = it
            if it["bound_version"] is None:
                continue
            dep = self._dep_id(it["kind"], k)
            try:
                d0 = self.registry.dependent(dep)
            except Exception:
                d0 = self.registry.register_dependent(dep, it["kind"], it["representation"],
                                                      it["record"])
            d0.bound_version = SemVer.parse(it["bound_version"])
            if it["status"] == "needs_revalidation":
                d0.status = "invalidated"
        self.sync()


class RetentionMonitor:
    """Retention probes per skill + competence-on-probes over time."""

    def __init__(self, max_probes: int = 32, seed: int = 0, level: float = 0.9):
        self.max_probes, self.level = int(max_probes), float(level)
        self.rng = np.random.default_rng(seed)
        self.probes: Dict[str, List[np.ndarray]] = {}
        self.seen: Dict[str, int] = {}
        self.history: Dict[str, List[Dict[str, Any]]] = {}

    def add_probe(self, skill_id: str, state) -> None:
        """Reservoir sampling over every state offered (uniform retention)."""
        s = np.asarray(state, np.float64).copy()
        lst = self.probes.setdefault(skill_id, [])
        n = self.seen.get(skill_id, 0)
        if len(lst) < self.max_probes:
            lst.append(s)
        else:
            j = int(self.rng.integers(n + 1))
            if j < self.max_probes:
                lst[j] = s
        self.seen[skill_id] = n + 1

    def probe_states(self, skill_id: str) -> np.ndarray:
        if not self.probes.get(skill_id):
            raise ContractError(f"no retention probes for {skill_id!r}")
        return np.stack(self.probes[skill_id])

    def measure(self, skill_id: str, t: float, successes: Sequence[bool], *,
                provenance: str) -> Dict[str, Any]:
        """Competence on the retention probes at time t (REAL executions)."""
        _check_real(provenance, f"retention measurement of {skill_id}")
        s = np.asarray(successes, bool)
        if s.ndim != 1 or len(s) < 1:
            raise ContractError("successes must be a non-empty 1-D sequence")
        c = BetaCompetence()
        for x in s:
            c.record(bool(x), provenance=provenance)
        lo, hi = c.interval(self.level)
        m = {"t": float(t), "n": int(len(s)), "k": int(s.sum()),
             "rate": float(s.mean()), "lo": lo, "hi": hi}
        self.history.setdefault(skill_id, []).append(m)
        return m

    def forgetting(self, skill_id: str) -> Dict[str, Any]:
        h = self.history.get(skill_id, [])
        if len(h) < 2:
            raise ContractError(f"forgetting needs >= 2 measurements of {skill_id!r}")
        best = max(h[:-1], key=lambda m: m["rate"])
        cur = h[-1]
        drop = best["rate"] - cur["rate"]
        return {"best": best, "current": cur, "drop": float(drop),
                "significant": bool(cur["hi"] < best["lo"])}

    def state_dict(self):
        return {"max_probes": self.max_probes, "level": self.level,
                "probes": {k: np.stack(v) for k, v in self.probes.items() if v},
                "seen": dict(self.seen),
                "history": {k: [dict(m) for m in v] for k, v in self.history.items()},
                "rng": _rng_dump(self.rng)}

    def load_state_dict(self, d):
        self.max_probes, self.level = int(d["max_probes"]), float(d["level"])
        self.probes = {k: [np.array(r) for r in v] for k, v in d["probes"].items()}
        self.seen = {k: int(v) for k, v in d["seen"].items()}
        self.history = {k: [dict(m) for m in v] for k, v in d["history"].items()}
        self.rng.bit_generator.state = _rng_load(d["rng"])


def _rng_dump(rng):
    from .planner import _rng_state
    return _rng_state(rng)


def _rng_load(d):
    from .planner import _rng_restore
    return _rng_restore(d)


# ======================================================================
# Persistence
# ======================================================================

def save_planning_state(path: str, *, skills: Sequence[LearnedSkill] = (),
                        mechanisms: Sequence[Mechanism] = (),
                        knowledge: Optional[KnowledgeStore] = None,
                        retention: Optional[RetentionMonitor] = None,
                        controller=None, memory: Optional[EpisodeMemory] = None,
                        extra: Optional[Dict[str, Any]] = None) -> None:
    """Atomic single-file save. Refuses duplicate ids — two entries with one
    identity would make the load ambiguous."""
    ids = [s.skill_id for s in skills]
    mids = [m.mechanism_id for m in mechanisms]
    if len(set(ids)) != len(ids) or len(set(mids)) != len(mids):
        raise ContractError("duplicate skill or mechanism ids in a save")
    for m in mechanisms:
        if not isinstance(m, Mechanism):
            raise ContractError("mechanisms must be contracts.Mechanism records")
    body = {"schema": FILE_SCHEMA, "version": FILE_VERSION,
            "skills": [s.state_dict() for s in skills],
            "skill_records": [s.to_record() for s in skills],
            "mechanisms": list(mechanisms),
            "knowledge": None if knowledge is None else knowledge.state_dict(),
            "retention": None if retention is None else retention.state_dict(),
            "controller": None if controller is None else controller.state_dict(),
            "memory": None if memory is None else memory.state_dict(),
            "extra": dict(extra or {})}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(encode_value(body), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_planning_state(path: str, *, knowledge: Optional[KnowledgeStore] = None,
                        retention: Optional[RetentionMonitor] = None, controller=None,
                        memory: Optional[EpisodeMemory] = None) -> Dict[str, Any]:
    """Returns {"skills": [LearnedSkill], "mechanisms": [Mechanism], "extra"}
    and restores the given stateful objects in place. Each skill's record is
    rebuilt and checked against the saved record: identity, version and
    competence must match exactly or the load raises."""
    with open(path) as f:
        body = decode_value(json.load(f))
    if body.get("schema") != FILE_SCHEMA:
        raise ContractError(f"{path}: not a planning state file")
    if body.get("version") != FILE_VERSION:
        raise ContractError(f"{path}: planning state version {body.get('version')!r}, "
                            f"this code reads {FILE_VERSION} (no silent guessing)")
    skills = [LearnedSkill.from_state_dict(s) for s in body["skills"]]
    for s, saved in zip(skills, body["skill_records"]):
        if not isinstance(saved, Skill) or s.to_record() != saved:
            raise ContractError(f"skill {s.skill_id!r} did not survive the round trip "
                                f"with identity/version/competence intact")
    mechs = list(body["mechanisms"])
    if not all(isinstance(m, Mechanism) for m in mechs):
        raise ContractError(f"{path}: mechanism entries are not Mechanism records")
    for name, obj in (("knowledge", knowledge), ("retention", retention),
                      ("controller", controller), ("memory", memory)):
        if obj is not None:
            if body[name] is None:
                raise ContractError(f"{path} holds no {name} state")
            obj.load_state_dict(body[name])
    return {"skills": skills, "mechanisms": mechs, "extra": body["extra"]}
