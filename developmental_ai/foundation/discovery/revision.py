"""Change-triggered structural revision (plan Stage 9 + inference.change).

ChangeTriggeredReviser watches the incumbent's predictive residuals with
inference.PredictiveCUSUM. A RULE CHANGE (an alarm — not one unlucky draw)
does not merely refit parameters: once `min_post_episodes` dev episodes
have arrived since the alarm it
    1. asks the registry whether a RETIRED mechanism now beats the
       incumbent on those episodes (revival — the old regime came back:
       reuse it, parameters intact, no search, no refit), and otherwise
    2. runs a BOUNDED structural search (search.search under `budget`,
       parent = the incumbent, evidence = post-alarm episodes only) and
       admits its best accepted candidates (the parameter-only REFIT
       included, if it validated — it competes on the same terms).
Admitted challengers are then judged by the registry on WINDOWS of later
episodes that no search ever saw (evaluate_window), and promoted only on
sustained advantage.

Gate/latch audit (CLAUDE.md §4.1): while challengers are being judged,
alarms are recorded as contradictions but do not start another search —
those challengers resolve (promoted / retired / expired "stale") within a
bounded number of windows, after which the next alarm triggers again. An
alarm with no successful search does not freeze anything — pending is cleared and the next alarm re-triggers; a
search lacking validation episodes keeps waiting for more episodes (buffer
bounded by max_buffer_episodes, oldest dropped), which arrive by themselves.
The detector restarts whenever the incumbent changes. Final held-out
episodes are refused at the door (EvidenceTable.require_dev).

MISSPECIFICATION TRIGGER (2026-10-03, verifier F6). The CUSUM answers "did
the rule CHANGE"; it is blind to an incumbent that is persistently, mildly
wrong (its dispersion drift is zero below a 1.65x variance ratio). A
MisspecificationMonitor (inference.hypotheses) on the same rows — Page's
CUSUM of "noisier than the incumbent's Gaussian predictive says" (half the
mass on a 4x-variance component) vs the predictive widened by tolerance 0.1,
h 14. MEASURED (D = 1, 200 runs x 5000 rows): variance exactly as
predicted 0 flags; 1.32x — what a root fitted on 120 rows can be off by
from sampling alone — 2% within 5000 rows; 1.5x median 2400 rows; 1.65x
540; 2.33x 78. Tolerance 0.2 / h 10 was slower everywhere (1.65x: 690)
with more 1.32x flags (6%); tolerance 0 flagged 1.32x in 53% — raises
`misspecified`; that opens a revision exactly like an alarm (event "misspecified", contradiction
recorded). Not a latch: the monitor is reset when that search runs and
whenever the incumbent changes, so a still-wrong incumbent must re-earn the
flag on fresh rows before the next (budgeted) search — repeated, bounded
attempts, never a frozen state. `misspecification()` exposes its report.

Offline/shadow only (plan §8): consumes evidence tables, emits events.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..inference.change import PredictiveCUSUM
from ..inference.hypotheses import MisspecificationMonitor
from .forms import fit_mechanism
from .lifecycle import MechanismRegistry
from .proposals import ALL_OPS
from .search import SearchBudget, search
from .table import EvidenceTable


class ChangeTriggeredReviser:
    def __init__(self, registry: MechanismRegistry, target: str, budget: SearchBudget,
                 *, ops: Sequence[str] = ALL_OPS,
                 detector_factory: Callable[[], PredictiveCUSUM] = PredictiveCUSUM,
                 min_post_episodes: int = 8, window_episodes: int = 4,
                 max_buffer_episodes: int = 24, admit_top: int = 2,
                 fitter: Callable = fit_mechanism, search_kw: Optional[Dict] = None,
                 misspec_factory: Optional[Callable[[], MisspecificationMonitor]] = (
                     lambda: MisspecificationMonitor(h=14.0, tolerance=0.1))):
        registry.incumbent_entry(target)                     # must be installed
        self.registry, self.target, self.budget = registry, target, budget
        self.ops = tuple(ops)
        self.detector_factory = detector_factory
        self.detector = detector_factory()
        self.misspec_factory = misspec_factory
        self.misspec = misspec_factory() if misspec_factory is not None else None
        self.misspec_triggers = 0
        self.min_post = int(min_post_episodes)
        self.window_n = int(window_episodes)
        self.max_buffer = int(max_buffer_episodes)
        self.admit_top = int(admit_top)
        self.fitter = fitter
        self.search_kw = dict(search_kw or {})
        self.pending = False
        self.post: List[EvidenceTable] = []
        self.window: List[EvidenceTable] = []
        self.searches: List[Any] = []
        self.revivals = 0
        self.alarms = 0
        self.log: List[Dict[str, Any]] = []
        self._inc_key = registry.incumbent_entry(target).key

    def _incumbent_changed(self):
        k = self.registry.incumbent_entry(self.target).key
        if k != self._inc_key:
            self._inc_key = k
            self.detector = self.detector_factory()
            if self.misspec_factory is not None:
                self.misspec = self.misspec_factory()
            return True
        return False

    def observe_episode(self, ep: EvidenceTable) -> List[Dict[str, Any]]:
        ep.require_dev()
        if len(ep.unique_episodes()) != 1:
            raise ValueError("observe_episode takes exactly one episode")
        ev: List[Dict[str, Any]] = []
        # 1. judge pending challengers on fresh episodes
        if self.registry.challengers(self.target):
            self.window.append(ep)
            if len(self.window) >= self.window_n:
                ev += self.registry.evaluate_window(self.target,
                                                    EvidenceTable.concat(self.window))
                self.window = []
        else:
            self.window = []
        if self._incumbent_changed():
            # alarms were against the old incumbent: they no longer apply
            self.pending, self.post = False, []
            ev.append({"event": "detector_reset", "incumbent": self._inc_key})
        # 2. change detection on the incumbent's predictive
        inc = self.registry.incumbent(self.target)
        mu, sd = inc.predict(ep)
        alarm = None
        for i in range(len(ep)):
            if not (np.isfinite(mu[i]) and np.isfinite(ep.y[i])):
                self.detector.update(None, None, None)      # missing: skipped
                continue
            out = self.detector.update(mu[i], sd[i] ** 2, ep.y[i])
            if out["alarm"] and alarm is None:
                alarm = out
            if self.misspec is not None:
                self.misspec.observe_gaussian(mu[i], sd[i] ** 2, ep.y[i])
        if alarm is None and self.misspec is not None and self.misspec.misspecified:
            rep = self.misspec.report()
            alarm = {"kind": "misspecified", "evidence": rep["evidence"],
                     "statistic": rep["statistic"]}
        if alarm is not None and alarm["kind"] == "misspecified":
            # Level-triggered, but it only ACTS (and counts) when it opens a
            # revision; while one is pending or challengers are being judged
            # the flag simply stays up, and the search that follows resets it.
            if not self.pending and not self.registry.challengers(self.target):
                self.misspec_triggers += 1
                self.registry.record_contradiction(self._inc_key, {
                    "kind": "misspecified", "evidence": alarm["evidence"],
                    "episode": ep.unique_episodes()[0]})
                ev.append({"event": "misspecified", "evidence": alarm["evidence"],
                           "statistic": alarm["statistic"],
                           "episode": ep.unique_episodes()[0]})
                self.pending, self.post = True, []
        elif alarm is not None:
            self.alarms += 1
            self.registry.record_contradiction(self._inc_key, {
                "kind": "change_alarm", "alarm": alarm["kind"],
                "episode": ep.unique_episodes()[0]})
            ev.append({"event": "alarm", "kind": alarm["kind"],
                       "episode": ep.unique_episodes()[0]})
            # Challengers already in evaluation ARE the response to earlier
            # alarms; a second search would only pile up more of them. They
            # resolve within max_inconclusive windows, after which alarms
            # trigger again (bounded wait, not a latch).
            if not self.pending and not self.registry.challengers(self.target):
                self.pending, self.post = True, []
        if self.pending:
            self.post.append(ep)
            del self.post[:-self.max_buffer]
            if len(self.post) >= self.min_post:
                ev += self._revise()
        self.log += ev
        return ev

    def _revise(self) -> List[Dict[str, Any]]:
        post = EvidenceTable.concat(self.post)
        rev = self.registry.consider_revival(self.target, post)
        if rev:
            self.revivals += 1
            self.pending, self.post, self.window = False, [], []
            return rev
        inc = self.registry.incumbent(self.target)
        rep = search(inc, post, self.budget, ops=self.ops, fitter=self.fitter,
                     **self.search_kw)
        self.searches.append(rep)
        if self.misspec is not None and rep.stopped != "insufficient_validation":
            # the flag has been acted on; a still-wrong incumbent must re-earn it
            self.misspec.reset("revision search ran")
        ev = [{"event": "search", "stopped": rep.stopped, "evaluated": rep.evaluated,
               "accepted": len(rep.accepted), "elapsed": rep.elapsed}]
        if rep.stopped == "insufficient_validation":
            return ev                                       # keep collecting
        for c in sorted(rep.accepted, key=lambda c: -c.val["mean"])[:self.admit_top]:
            from .causal import contradictions
            prov = c.provenance()
            prov["contradictions"] = contradictions(c.mechanism.causal)
            prov["search_episodes"] = rep.fit_episodes + rep.val_episodes
            ok, why = self.registry.admit(c.mechanism, prov)
            ev.append({"event": "admit", "key": c.mechanism.ref(), "ok": ok, "why": why,
                       "op": c.proposal.op})
        self.pending, self.post, self.window = False, [], []
        return ev


    def misspecification(self) -> Optional[Dict[str, Any]]:
        """The incumbent's misspecification report (None if disabled)."""
        if self.misspec is None:
            return None
        r = self.misspec.report()
        r["triggers"] = self.misspec_triggers
        return r


__all__ = ["ChangeTriggeredReviser"]
