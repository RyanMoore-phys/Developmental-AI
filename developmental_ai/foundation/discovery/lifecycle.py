"""Mechanism lifecycle: candidate -> probationary -> promoted -> retired, and
REVIVAL (plan Stage 9 items 4-5).

One PROMOTED mechanism per target is the INCUMBENT — the explanation in use.
Everything else is a challenger or an archive entry.

    admit(m)            -> candidate. Refused when the live cap is full and m
                           is no better (on its admission validation) than
                           the worst challenger, which is otherwise evicted
                           to the retired archive. No proliferation: live
                           entries per target <= max_live, retired <=
                           max_retired.
    evaluate_window(t)  on a window of FRESH validation episodes, each
                        challenger vs the incumbent (per-episode paired
                        log-lik test, forms.verdict):
                          advantage    streak+1; candidate -> probationary
                          disadvantage streak reset, contradiction recorded;
                                       retire_windows in a row -> retired
                          inconclusive no change; max_inconclusive in a row
                                       -> retired("stale")
                        A probationary challenger with promote_windows
                        consecutive advantages REPLACES the incumbent, which
                        is retired("replaced") with its parameters intact.
    consider_revival(t) a RETIRED mechanism that beats the incumbent on a
                        window is revived to probationary — same mechanism
                        id, same version, bit-identical parameters (no refit
                        from scratch) — and needs only revival_windows
                        consecutive advantages (the revival window counts)
                        to be promoted again.

CATASTROPHIC REPLACEMENT IS STRUCTURALLY IMPOSSIBLE: the only path to
promoted is a run of `advantage` verdicts against the incumbent on fresh
episodes; a disadvantage resets the run. A candidate worse on validation
therefore never replaces a promoted mechanism, however good its dev fit.

ESCAPE PATHS (CLAUDE.md §4.1 — every guard must be reachable from the
inside): a candidate stuck inconclusive expires to retired after
max_inconclusive windows (it does not hold a cap slot forever); a full cap
admits a better newcomer by evicting the worst challenger; every retired
entry (rejected, stale, evicted, replaced) can come back through
consider_revival when the evidence turns; the incumbent can be displaced by
any challenger with sustained advantage. What is permanent: an archive entry
dropped by the max_retired cap (oldest never-promoted first) — re-opened only
by the search re-deriving that structure.

All state round-trips through snapshot()/restore() (JSON-able), so a
rejected candidate is recoverable by construction: the incumbent object is
never touched by admission, evaluation or rejection of a challenger.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import from_dict, to_dict
from .forms import _MechBase, mechanism_from_record, per_episode_paired, verdict
from .table import DiscoveryError, EvidenceTable

CANDIDATE, PROBATIONARY, PROMOTED, RETIRED = ("candidate", "probationary",
                                              "promoted", "retired")
STATES = (CANDIDATE, PROBATIONARY, PROMOTED, RETIRED)
MAX_HISTORY = 32


@dataclass
class Entry:
    mechanism: _MechBase
    state: str
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    n_contradictions: int = 0
    windows: List[Dict[str, Any]] = field(default_factory=list)
    adv_streak: int = 0
    dis_streak: int = 0
    inconclusive: int = 0
    retired_reason: Optional[str] = None
    was_promoted: bool = False
    revived: bool = False
    score: float = float("-inf")         # latest known validation advantage
    order: int = 0

    @property
    def key(self) -> str:
        return self.mechanism.ref()

    @property
    def target(self) -> str:
        return self.mechanism.target

    def to_dict(self) -> Dict[str, Any]:
        d = {k: copy.deepcopy(getattr(self, k)) for k in
             ("state", "provenance", "contradictions", "n_contradictions", "windows",
              "adv_streak", "dis_streak", "inconclusive", "retired_reason",
              "was_promoted", "revived", "order")}
        d["score"] = self.score if math.isfinite(self.score) else None
        d["mechanism"] = to_dict(self.mechanism.to_record())
        return d

    @classmethod
    def from_dict(cls, d) -> "Entry":
        e = cls(mechanism_from_record(from_dict(d["mechanism"])), d["state"])
        for k in ("provenance", "contradictions", "n_contradictions", "windows",
                  "adv_streak", "dis_streak", "inconclusive", "retired_reason",
                  "was_promoted", "revived", "order"):
            setattr(e, k, copy.deepcopy(d[k]))
        e.score = float("-inf") if d["score"] is None else float(d["score"])
        return e


class MechanismRegistry:
    def __init__(self, promote_windows: int = 3, retire_windows: int = 2,
                 revival_windows: int = 2, max_live: int = 4, max_retired: int = 16,
                 max_inconclusive: int = 6, max_contradictions: int = 64,
                 t_crit: float = 2.0, margin: float = 0.0, min_episodes: int = 3):
        for k, v in (("promote_windows", promote_windows), ("retire_windows", retire_windows),
                     ("revival_windows", revival_windows), ("max_live", max_live),
                     ("max_retired", max_retired), ("max_inconclusive", max_inconclusive),
                     ("min_episodes", min_episodes), ("max_contradictions",
                                                      max_contradictions)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise DiscoveryError(f"{k} must be an int >= 1")
        if max_live < 2:
            raise DiscoveryError("max_live must be >= 2 (incumbent + one challenger)")
        self.cfg = dict(promote_windows=promote_windows, retire_windows=retire_windows,
                        revival_windows=revival_windows, max_live=max_live,
                        max_retired=max_retired, max_inconclusive=max_inconclusive,
                        max_contradictions=max_contradictions, t_crit=float(t_crit),
                        margin=float(margin), min_episodes=min_episodes)
        self._entries: Dict[str, Entry] = {}
        self.lineage: Dict[str, List[str]] = {}
        self.clock = 0
        self._order = 0

    # ---- queries -------------------------------------------------------------
    def entry(self, key: str) -> Entry:
        if key not in self._entries:
            raise DiscoveryError(f"no mechanism {key!r} in the registry")
        return self._entries[key]

    def entries(self, target: str, states=STATES) -> List[Entry]:
        return sorted((e for e in self._entries.values()
                       if e.target == target and e.state in states), key=lambda e: e.order)

    def live(self, target: str) -> List[Entry]:
        return self.entries(target, (CANDIDATE, PROBATIONARY, PROMOTED))

    def challengers(self, target: str) -> List[Entry]:
        return self.entries(target, (CANDIDATE, PROBATIONARY))

    def retired(self, target: str) -> List[Entry]:
        return self.entries(target, (RETIRED,))

    def incumbent_entry(self, target: str) -> Entry:
        p = self.entries(target, (PROMOTED,))
        if len(p) != 1:
            raise DiscoveryError(f"target {target!r} has {len(p)} promoted mechanisms; "
                                 f"install() one first")
        return p[0]

    def incumbent(self, target: str) -> _MechBase:
        return self.incumbent_entry(target).mechanism

    def chain(self, key: str) -> List[str]:
        """Provenance chain: key, its parents, theirs ... (breadth-first)."""
        out, todo = [], [key]
        while todo:
            k = todo.pop(0)
            if k in out:
                continue
            out.append(k)
            todo.extend(self.lineage.get(k, []))
        return out

    # ---- mutation --------------------------------------------------------------
    def _event(self, e: Entry, event: str, **kw):
        e.provenance.append({"event": event, "clock": self.clock, **kw})
        if len(e.provenance) > MAX_HISTORY:
            del e.provenance[1:2]          # keep the admission record (index 0)

    def _new(self, m: _MechBase, state: str) -> Entry:
        if not getattr(m, "fitted", False):
            raise DiscoveryError(f"{m.ref()} is not fitted")
        if m.ref() in self._entries:
            raise DiscoveryError(f"{m.ref()} already registered")
        self._order += 1
        e = Entry(m, state, order=self._order)
        self.lineage[m.ref()] = list(m.parents)
        self._entries[m.ref()] = e
        return e

    def install(self, m: _MechBase, note: str = "installed") -> Entry:
        if self.entries(m.target, (PROMOTED,)):
            raise DiscoveryError(f"{m.target!r} already has an incumbent; a "
                                 f"replacement must earn it through evaluate_window")
        e = self._new(m, PROMOTED)
        e.was_promoted = True
        self._event(e, note)
        return e

    def record_contradiction(self, key: str, info: Dict[str, Any]) -> None:
        e = self.entry(key)
        e.contradictions.append({"clock": self.clock, **info})
        e.n_contradictions += 1
        if len(e.contradictions) > self.cfg["max_contradictions"]:
            del e.contradictions[0]

    def admit(self, m: _MechBase, provenance: Optional[Dict[str, Any]] = None
              ) -> Tuple[bool, str]:
        """Add a validated search candidate. Returns (admitted, reason)."""
        self.incumbent_entry(m.target)                     # must exist
        if m.ref() in self._entries:
            return False, "duplicate"
        prov = dict(provenance or {})
        score = float(prov.get("val", {}).get("mean", float("-inf")))
        live = self.live(m.target)
        if len(live) >= self.cfg["max_live"]:
            ch = self.challengers(m.target)
            if not ch:
                return False, "cap_full"
            worst = min(ch, key=lambda e: (e.score, -e.order))
            if not score > worst.score:
                return False, "cap_full"
            self._retire(worst, "evicted_cap")
        e = self._new(m, CANDIDATE)
        e.score = score
        self._event(e, "admitted", parents=list(m.parents), **prov)
        for c in prov.get("contradictions", []):
            self.record_contradiction(e.key, c)
        return True, "admitted"

    def _retire(self, e: Entry, reason: str):
        e.state = RETIRED
        e.retired_reason = reason
        e.adv_streak = e.dis_streak = e.inconclusive = 0
        self._event(e, "retired", reason=reason)
        ret = self.retired(e.target)
        while len(ret) > self.cfg["max_retired"]:
            never = [r for r in ret if not r.was_promoted]
            drop = (never or ret)[0]
            del self._entries[drop.key]
            ret = self.retired(e.target)

    def _compare(self, m: _MechBase, inc_ll, table: EvidenceTable):
        st = per_episode_paired(m.loglik(table), inc_ll, table.episodes)
        return st, verdict(st, self.cfg["t_crit"], self.cfg["margin"],
                           self.cfg["min_episodes"])

    def evaluate_window(self, target: str, table: EvidenceTable) -> List[Dict[str, Any]]:
        """Score every challenger against the incumbent on `table` — which
        must be episodes NOT used to fit any of them."""
        self.clock += 1
        inc = self.incumbent_entry(target)
        inc_ll = inc.mechanism.loglik(table)
        events: List[Dict[str, Any]] = []
        for e in self.challengers(target):
            st, v = self._compare(e.mechanism, inc_ll, table)
            e.windows.append({"clock": self.clock, "verdict": v, **st})
            del e.windows[:-MAX_HISTORY]
            if math.isfinite(st["mean"]):
                e.score = st["mean"]
            if v == "advantage":
                e.adv_streak += 1
                e.dis_streak = e.inconclusive = 0
                self.record_contradiction(inc.key, {"kind": "outperformed",
                                                    "by": e.key, "mean": st["mean"],
                                                    "t": st["t"]})
                if e.state == CANDIDATE:
                    e.state = PROBATIONARY
                    self._event(e, "probationary", t=st["t"])
                    events.append({"event": "probationary", "key": e.key})
            elif v == "disadvantage":
                e.dis_streak += 1
                e.adv_streak = e.inconclusive = 0
                self.record_contradiction(e.key, {"kind": "validation_disadvantage",
                                                  "vs": inc.key, "mean": st["mean"],
                                                  "t": st["t"]})
                if e.dis_streak >= self.cfg["retire_windows"]:
                    self._retire(e, "disadvantage")
                    events.append({"event": "retired", "key": e.key,
                                   "reason": "disadvantage"})
            else:
                e.inconclusive += 1
                if e.inconclusive >= self.cfg["max_inconclusive"]:
                    self._retire(e, "stale")
                    events.append({"event": "retired", "key": e.key, "reason": "stale"})
        ready = [e for e in self.challengers(target) if e.state == PROBATIONARY and
                 e.adv_streak >= (self.cfg["revival_windows"] if e.revived
                                  else self.cfg["promote_windows"])]
        if ready:
            best = max(ready, key=lambda e: e.score)
            self._retire(inc, "replaced")
            best.state = PROMOTED
            best.was_promoted = True
            self._event(best, "promoted", replaced=inc.key)
            events.append({"event": "promoted", "key": best.key, "replaced": inc.key})
            for e in self.challengers(target):            # evidence was vs the old one
                e.adv_streak = 0
        return events

    def consider_revival(self, target: str, table: EvidenceTable) -> List[Dict[str, Any]]:
        """Revive at most one retired mechanism that beats the incumbent on
        `table` (parameters untouched — no refit)."""
        self.clock += 1
        inc = self.incumbent_entry(target)
        inc_ll = inc.mechanism.loglik(table)
        best = None
        for e in self.retired(target):
            st, v = self._compare(e.mechanism, inc_ll, table)
            if v == "advantage" and (best is None or st["mean"] > best[1]["mean"]):
                best = (e, st)
        if best is None:
            return []
        e, st = best
        if len(self.live(target)) >= self.cfg["max_live"]:
            ch = self.challengers(target)
            if not ch:
                return []
            self._retire(min(ch, key=lambda x: (x.score, -x.order)), "evicted_cap")
        e.state = PROBATIONARY
        e.revived = True
        e.retired_reason = None
        e.adv_streak, e.dis_streak, e.inconclusive = 1, 0, 0
        e.score = st["mean"]
        e.windows.append({"clock": self.clock, "verdict": "advantage", "revival": True, **st})
        self._event(e, "revived", vs=inc.key, mean=st["mean"], t=st["t"])
        self.record_contradiction(inc.key, {"kind": "outperformed_by_retired",
                                            "by": e.key, "mean": st["mean"]})
        return [{"event": "revived", "key": e.key, "mean": st["mean"], "t": st["t"]}]

    # ---- persistence -------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {"schema": "discovery.registry", "version": 1, "cfg": dict(self.cfg),
                "clock": self.clock, "order": self._order,
                "lineage": copy.deepcopy(self.lineage),
                "entries": [e.to_dict() for e in sorted(self._entries.values(),
                                                        key=lambda e: e.order)]}

    @classmethod
    def restore(cls, snap: Dict[str, Any]) -> "MechanismRegistry":
        if snap.get("schema") != "discovery.registry" or snap.get("version") != 1:
            raise DiscoveryError("not a discovery.registry v1 snapshot")
        r = cls(**snap["cfg"])
        r.clock, r._order = int(snap["clock"]), int(snap["order"])
        r.lineage = copy.deepcopy(snap["lineage"])
        for d in snap["entries"]:
            e = Entry.from_dict(d)
            r._entries[e.key] = e
        return r


__all__ = ["MechanismRegistry", "Entry", "STATES", "CANDIDATE", "PROBATIONARY",
           "PROMOTED", "RETIRED"]
