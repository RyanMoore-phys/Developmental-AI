"""Budgeted structural search (plan Stage 9 items 2-3).

    search(parent, table, budget) -> SearchReport

WHAT HAPPENS
    1. `table` must be DEV evidence (final held-out episodes are refused) and
       is cut to at most budget.max_rows by WHOLE episodes.
    2. A deterministic episode-level split (table.VALIDATION_SALT) separates
       FIT episodes from VALIDATION episodes. Every fit sees fit episodes
       only; acceptance is decided on validation episodes only. Too few
       validation episodes -> nothing is accepted ("insufficient_validation").
    3. The PARAMETER-ONLY baseline is fitted first: the parent's structure
       refitted on the fit episodes (op REFIT). Structural candidates are
       judged against THAT, not against stale parameters — otherwise any
       refit at all would look like a structural win.
    4. Proposals are enumerated from the restricted language (proposals.py)
       — deterministic order, or sampled with `rng` — to `budget.depth`
       levels with a beam of `budget.beam`. A child is ACCEPTED only if
         (a) its penalised dev score is better: BIC(child) < BIC(reference)
             on the fit episodes, and
         (b) on the separate validation episodes it has a significant
             per-episode paired log-likelihood advantage over the reference
             (forms.verdict == "advantage"), with >= 95% coverage.
       Accepted children carry causal labels (causal.annotate_causal),
       computed on the FIT episodes' intervened rows too — no fit of any
       kind ever touches a validation episode.

BUDGETS (all explicit)
    wall_s          every candidate evaluation (fit + score + causal
                    annotation) runs under a timeout equal to the time left;
                    a fitter that overruns is ABANDONED (its result is
                    discarded, the search returns on time) — so a slow or
                    hung fitter cannot stretch the budget. The default fitter
                    is also cooperative (deadline checks between branches).
                    Proposal ENUMERATION is lazy and deadline-checked too
                    (iter_proposals): until 2026-10-03 it ran eagerly outside
                    the guard and a 150-var x 60k-row table took 4.1-4.7 s
                    of a 0.25 s budget (verifier B1). Overrun is now bounded
                    by one enumeration step (one sort of one column) plus
                    the up-front split of the table into fit/validation
                    episodes (O(rows), not budgeted).
    abandoned fits  a thread cannot be killed in Python, so an abandoned
                    evaluation is CANCELLED cooperatively: each evaluation
                    gets its own forms.CancelToken (a float deadline that
                    can also be cancelled) as the fitter's `deadline`; on
                    abandonment the token is cancelled, and every library
                    form (forms.check_deadline per branch / piecewise
                    threshold / lookup chunk), the causal annotation's
                    refits (which until 2026-10-03 ran with NO deadline and
                    kept going for whole seconds after the search returned)
                    and the scoring steps between them stop at their next
                    check. A fitter that never checks cannot be stopped:
                    at most `max_abandoned` (default MAX_ABANDONED_FITS)
                    abandoned evaluation threads may be ALIVE process-wide;
                    past that cap a search starts no new fit and returns
                    stopped="budget_exhausted" (counts
                    "refused_abandoned_cap"). ESCAPE (CLAUDE.md §4.1): the
                    cap counts LIVE threads only — each slot frees itself
                    the moment its fitter returns or raises, so the next
                    search proceeds with nothing reset by hand. Only a
                    fitter that never returns holds its slot forever, and
                    then refusing to stack more hung threads on it is the
                    point.
    max_candidates  number of candidate evaluations started (baseline incl.)
    max_bytes       serialized size of retained accepted candidates; the
                    worst (lowest validation advantage) is evicted beyond it
    max_rows        evidence rows used (whole episodes)

A fitter exception rejects that candidate and the search continues; nothing
the search does mutates `parent` or the library (fits are pure), so a
rejected or failed candidate leaves every existing mechanism as it was.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .causal import annotate_causal, with_causal
from .forms import (FORMS, MIN_BRANCH_ROWS, BudgetExceeded, CancelToken, Condition,
                    FitError, check_deadline,
                    StructuredMechanism, _MechBase,
                    fit_mechanism, per_episode_paired, verdict)
from .proposals import (ADD_DEPENDENCY, ALL_OPS, COMPOSE, REFIT, REPLACE_TRANSITION,
                        SPLIT_APPLICABILITY, Proposal, ProposalError)
from .table import DEFAULT_VAL_FRACTION, VALIDATION_SALT, DiscoveryError, EvidenceTable

SPLIT_QUANTILES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


@dataclass(frozen=True)
class SearchBudget:
    wall_s: float = 5.0
    max_candidates: int = 200
    max_bytes: int = 2_000_000
    max_rows: int = 20_000
    depth: int = 2
    beam: int = 2

    def __post_init__(self):
        if not (math.isfinite(self.wall_s) and self.wall_s > 0):
            raise DiscoveryError("wall_s must be finite and > 0")
        for k in ("max_candidates", "max_bytes", "max_rows", "depth", "beam"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise DiscoveryError(f"{k} must be an int >= 1, got {v!r}")

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("wall_s", "max_candidates", "max_bytes", "max_rows", "depth", "beam")}


@dataclass
class Candidate:
    mechanism: _MechBase
    proposal: Proposal
    reference: str                   # ref of the mechanism it had to beat
    dev_bic: float
    ref_dev_bic: float
    val: Dict[str, Any]
    depth: int
    nbytes: int

    def provenance(self) -> Dict[str, Any]:
        return {"proposal": self.proposal.to_dict(), "reference": self.reference,
                "dev_bic": self.dev_bic, "ref_dev_bic": self.ref_dev_bic,
                "val": dict(self.val), "depth": self.depth}


@dataclass
class SearchReport:
    parent: str
    baseline: Optional[_MechBase] = None
    baseline_val: Optional[Dict[str, Any]] = None
    accepted: List[Candidate] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    stopped: str = "exhausted"
    elapsed: float = 0.0
    evaluated: int = 0
    fit_episodes: List[str] = field(default_factory=list)
    val_episodes: List[str] = field(default_factory=list)
    peak_bytes: int = 0
    budget: Dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Optional[Candidate]:
        if not self.accepted:
            return None
        return max(self.accepted, key=lambda c: c.val["mean"])

    def structural(self) -> List[Candidate]:
        return [c for c in self.accepted if c.proposal.op != REFIT]

    def bump(self, k: str, n: int = 1):
        self.counts[k] = self.counts.get(k, 0) + n


class _Timeout:
    pass


_TIMEOUT = _Timeout()


MAX_ABANDONED_FITS = 4
_ABANDONED: List[threading.Thread] = []
_ABANDONED_LOCK = threading.Lock()


def live_abandoned_fits() -> int:
    """Abandoned evaluation threads still running, process-wide (dead ones
    are pruned here: a slot frees itself when its fitter returns)."""
    with _ABANDONED_LOCK:
        _ABANDONED[:] = [t for t in _ABANDONED if t.is_alive()]
        return len(_ABANDONED)


def _run_with_timeout(fn: Callable[[], Any], timeout: float,
                      token: Optional[CancelToken] = None):
    """Run fn in a daemon thread; return _TIMEOUT if it does not finish in
    `timeout` seconds. The thread is then ABANDONED (its result discarded),
    `token` is cancelled so a cooperative fn stops at its next check, and
    the thread is registered until it exits (live_abandoned_fits)."""
    box: Dict[str, Any] = {}

    def run():
        try:
            box["r"] = fn()
        except BaseException as e:            # noqa: BLE001 - re-raised below
            box["e"] = e

    th = threading.Thread(target=run, daemon=True, name="discovery-fit")
    th.start()
    th.join(max(0.0, timeout))
    if th.is_alive():
        if token is not None:
            token.cancel()
        with _ABANDONED_LOCK:
            _ABANDONED.append(th)
        return _TIMEOUT
    if "e" in box:
        raise box["e"]
    return box["r"]


def iter_proposals(parent: _MechBase, table: EvidenceTable,
                   ops: Sequence[str] = ALL_OPS, library: Sequence[_MechBase] = (),
                   deadline: Optional[float] = None) -> Iterator[Proposal]:
    """LAZY, deadline-checked enumeration of every proposal of the restricted
    language applicable to `parent` (REFIT excluded: it is the baseline).

    Why lazy (verifier B1, stage9_11 E9f): the eager list did a quantile, an
    argsort (_best_split) and three np.unique sorts PER VARIABLE before the
    first candidate could be fitted, all outside the search's deadline
    guard — 2.6 s of a 0.25 s budget on 150 vars x 60k rows. Now each
    data-touching step (a split threshold, a variable's quantile grid) is
    computed only when the consumer reaches it, after a deadline check that
    raises forms.BudgetExceeded; the per-step cost is bounded by one sort of
    one column. The ORDER is exactly that of the old eager list (ADD_DEPENDENCY
    per branch, then SPLIT_APPLICABILITY — best data-driven cuts of the
    continuous variables, then the quantile/level grid — then
    REPLACE_TRANSITION, then COMPOSE), so a max_candidates-limited search
    picks the same candidates it always did."""
    ref = (parent.ref(),)
    vars_ = table.variables

    def tick():
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExceeded("proposal enumeration reached the search deadline")

    grid_memo: Dict[str, List[Condition]] = {}

    def grid_for(v: str) -> List[Condition]:
        if v not in grid_memo:
            tick()
            g: List[Condition] = []
            if table.n_distinct_at_least(v, 2):
                if table.is_discrete(v):
                    lv = table.levels(v)
                    for x in (lv[:1] if len(lv) == 2 else lv):
                        g.append(Condition(v, "==", float(x)))
                else:
                    x = table.col(v)
                    for q in np.unique(np.quantile(x[np.isfinite(x)], SPLIT_QUANTILES)):
                        g.append(Condition(v, "<", float(q)))
            grid_memo[v] = g
        return grid_memo[v]

    def split_conditions(b) -> Iterator[Optional[Condition]]:
        for v in vars_:
            tick()
            if not table.is_discrete(v):
                yield _best_split(table, b, v)
        for v in vars_:
            for c in grid_for(v):
                yield c

    if isinstance(parent, StructuredMechanism):
        for i, b in enumerate(parent.branches):
            if ADD_DEPENDENCY in ops:
                for v in vars_:
                    if v not in b.inputs:
                        yield Proposal(ADD_DEPENDENCY, ref, {"var": v, "branch": i})
            if SPLIT_APPLICABILITY in ops:
                for c in split_conditions(b):
                    if c is not None and c not in b.predicate and c.negate() not in b.predicate:
                        yield Proposal(SPLIT_APPLICABILITY, ref, {"condition": c, "branch": i})
            if REPLACE_TRANSITION in ops:
                for f in FORMS:
                    if f == "piecewise":
                        for v in vars_:
                            if v != b.pivot:
                                yield Proposal(REPLACE_TRANSITION, ref,
                                               {"form": f, "pivot": v, "branch": i})
                    elif f != b.form and (f != "lookup" or (
                            b.inputs and all(table.is_discrete(v) for v in b.inputs))):
                        if f in ("linear", "lookup") and not b.inputs:
                            continue
                        yield Proposal(REPLACE_TRANSITION, ref, {"form": f, "branch": i})
    if COMPOSE in ops:
        for m1 in library:
            if isinstance(m1, StructuredMechanism) and m1.fitted and \
                    m1.target in parent.dependencies() and m1.target != parent.target:
                yield Proposal(COMPOSE, (m1.ref(), parent.ref()))


def enumerate_proposals(parent: _MechBase, table: EvidenceTable,
                        ops: Sequence[str] = ALL_OPS,
                        library: Sequence[_MechBase] = (),
                        deadline: Optional[float] = None) -> List[Proposal]:
    """Every proposal of the restricted language applicable to `parent` over
    the table's variables (REFIT excluded: it is the baseline), as a list.
    With a `deadline`, raises forms.BudgetExceeded once it passes."""
    return list(iter_proposals(parent, table, ops, library, deadline))


def _best_split(table: EvidenceTable, b, v: str) -> Optional[Condition]:
    """Data-driven threshold for a continuous variable: the cut of branch
    b's rows that minimises the two-constant SSE (cumulative sums, O(n log
    n)). Computed on whatever table is passed — search passes FIT episodes
    only, so validation never chooses a threshold."""
    x, y = table.col(v), table.y
    m = b.mask(table) & np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    k0 = MIN_BRANCH_ROWS
    if n < 2 * k0 + 1:
        return None
    o = np.argsort(x[m], kind="stable")
    xs, ys = x[m][o], y[m][o]
    c1, c2 = np.cumsum(ys), np.cumsum(ys * ys)
    k = np.arange(1, n)
    sse_l = c2[:-1] - c1[:-1] ** 2 / k
    sse_r = (c2[-1] - c2[:-1]) - (c1[-1] - c1[:-1]) ** 2 / (n - k)
    tot = sse_l + sse_r
    ok = (k >= k0) & (n - k >= k0) & (xs[1:] > xs[:-1])
    if not ok.any():
        return None
    j = int(np.argmin(np.where(ok, tot, np.inf)))
    return Condition(v, "<", float(0.5 * (xs[j] + xs[j + 1])))


def _cap_rows(table: EvidenceTable, max_rows: int, report: SearchReport) -> EvidenceTable:
    if len(table) <= max_rows:
        return table
    keep, n = [], 0
    uniq, counts = np.unique(table.episodes.astype(str), return_counts=True)
    count = dict(zip(uniq.tolist(), counts.tolist()))   # one sort, not one scan per episode
    for e in table.unique_episodes():
        k = int(count[e])
        if n + k > max_rows:
            break
        keep.append(e)
        n += k
    report.bump("rows_dropped", len(table) - n)
    return table.by_episodes(keep)


def search(parent: _MechBase, table: EvidenceTable, budget: SearchBudget = SearchBudget(),
           *, ops: Sequence[str] = ALL_OPS, library: Sequence[_MechBase] = (),
           fitter: Callable = fit_mechanism, val_fraction: float = DEFAULT_VAL_FRACTION,
           val_salt: str = VALIDATION_SALT, min_val_episodes: int = 3,
           t_crit: float = 2.0, margin: float = 0.0, rng: Optional[np.random.Generator] = None,
           annotate: bool = True, max_abandoned: Optional[int] = None) -> SearchReport:
    t0 = time.monotonic()
    cap = MAX_ABANDONED_FITS if max_abandoned is None else max_abandoned
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise DiscoveryError(f"max_abandoned must be an int >= 1, got {max_abandoned!r}")
    deadline = t0 + budget.wall_s
    bad = set(ops) - set(ALL_OPS)
    if bad:
        raise DiscoveryError(f"unknown ops {sorted(bad)}")
    if table.target != parent.target:
        raise DiscoveryError(f"table target {table.target!r} != parent target {parent.target!r}")
    rep = SearchReport(parent=parent.ref(), budget=budget.to_dict())
    if live_abandoned_fits() >= cap:          # before any O(rows) work
        rep.bump("refused_abandoned_cap")
        rep.stopped = "budget_exhausted"
        rep.elapsed = time.monotonic() - t0
        return rep
    table = _cap_rows(table.require_dev(), budget.max_rows, rep)
    fit_t, val_t = table.fit_validation_split(val_fraction, val_salt)
    rep.fit_episodes, rep.val_episodes = fit_t.unique_episodes(), val_t.unique_episodes()
    parents_map: Dict[str, _MechBase] = {m.mechanism_id: m for m in library}
    parents_map[parent.mechanism_id] = parent

    def finish(reason):
        rep.stopped = reason
        rep.elapsed = time.monotonic() - t0
        return rep

    if len(rep.val_episodes) < min_val_episodes:
        return finish("insufficient_validation")

    def evaluate(child: _MechBase, ref: Optional[_MechBase], tok: CancelToken):
        """fit on FIT episodes; dev BIC; validation verdict vs ref; causal.
        `tok` is checked between every step and handed to every fit."""
        f = fitter(child, fit_t, tok)
        check_deadline(tok)
        out = {"fitted": f, "dev_bic": f.bic(fit_t)}
        if ref is not None:
            out["ref_dev_bic"] = ref.bic(fit_t)
            check_deadline(tok)
            st = per_episode_paired(f.loglik(val_t), ref.loglik(val_t), val_t.episodes)
            out["val"] = st
            out["verdict"] = verdict(st, t_crit, margin, min_val_episodes)
            if annotate and out["verdict"] == "advantage" and out["dev_bic"] < out["ref_dev_bic"]:
                check_deadline(tok)
                out["fitted"] = with_causal(f, annotate_causal(f, fit_t, fitter=fitter,
                                                               deadline=tok))
        return out

    def guarded(child, ref):
        """None on a stop condition (rep.stopped set); else evaluate()'s dict
        or an {"error": ...} dict."""
        if rep.evaluated >= budget.max_candidates:
            rep.stopped = "candidates"
            return None
        left = deadline - time.monotonic()
        if left <= 0:
            rep.stopped = "wall"
            return None
        if live_abandoned_fits() >= cap:
            rep.bump("refused_abandoned_cap")
            rep.stopped = "budget_exhausted"
            return None
        rep.evaluated += 1
        tok = CancelToken(deadline)
        try:
            r = _run_with_timeout(lambda: evaluate(child, ref, tok), left, tok)
        except FitError as e:
            return {"error": "fit_error", "msg": str(e)}
        except Exception as e:                      # noqa: BLE001
            if type(e).__name__ == "BudgetExceeded":
                rep.stopped = "wall"
                return None
            return {"error": "fit_exception", "msg": f"{type(e).__name__}: {e}"}
        if r is _TIMEOUT:
            rep.bump("abandoned_timeout")
            rep.stopped = "wall"
            return None
        return r

    # ---- 1. parameter-only baseline (REFIT) ---------------------------------
    refit_p = Proposal(REFIT, (parent.ref(),))
    stored_ref = parent if getattr(parent, "fitted", False) else None
    r = guarded(refit_p.apply(parents_map), stored_ref)
    if r is None:
        return finish(rep.stopped)
    if "error" in r:
        rep.bump(r["error"])
        return finish("baseline_unfittable")
    baseline = r["fitted"]
    rep.baseline = baseline
    rep.baseline_val = r.get("val")
    retained_bytes = 0

    def retain(c: Candidate):
        nonlocal retained_bytes
        if c.nbytes > budget.max_bytes:
            rep.bump("rejected_too_large")
            return False
        rep.accepted.append(c)
        retained_bytes += c.nbytes
        while retained_bytes > budget.max_bytes and rep.accepted:
            worst = min(rep.accepted, key=lambda x: x.val["mean"])
            rep.accepted.remove(worst)
            retained_bytes -= worst.nbytes
            rep.bump("evicted_bytes")
        rep.peak_bytes = max(rep.peak_bytes, retained_bytes)
        return c in rep.accepted

    if REFIT in ops and stored_ref is not None and r.get("verdict") == "advantage":
        retain(Candidate(baseline, refit_p, parent.ref(), r["dev_bic"], r["ref_dev_bic"],
                         r["val"], 0, baseline.nbytes()))

    # ---- 2. structural proposals, depth x beam ------------------------------
    seen = {baseline.structure_key()}
    frontier = [(parent, baseline)]             # (structure source, reference)
    for depth in range(1, budget.depth + 1):
        nxt: List[Candidate] = []
        for source, ref in frontier:
            props = iter_proposals(source, fit_t, ops, library, deadline)
            if rng is not None:                 # a permutation needs the whole list
                try:
                    lst = list(props)
                except BudgetExceeded:
                    return finish("wall")
                props = iter([lst[i] for i in rng.permutation(len(lst))])
            while True:
                # Every step between guarded fits is deadline-checked too:
                # enumeration, apply and structure_key are work (B1).
                if time.monotonic() >= deadline:
                    return finish("wall")
                try:
                    p = next(props)
                except StopIteration:
                    break
                except BudgetExceeded:
                    return finish("wall")
                try:
                    child = p.apply({**parents_map, source.mechanism_id: source})
                except ProposalError:
                    rep.bump("inapplicable")
                    continue
                key = child.structure_key()
                if key in seen:
                    rep.bump("duplicate")
                    continue
                seen.add(key)
                r = guarded(child, ref)
                if r is None:
                    return finish(rep.stopped)
                if "error" in r:
                    rep.bump(r["error"])
                    continue
                if not r["dev_bic"] < r["ref_dev_bic"]:
                    rep.bump("rejected_dev_score")
                    continue
                if r["verdict"] != "advantage":
                    rep.bump("rejected_validation_" + r["verdict"])
                    continue
                f = r["fitted"]
                c = Candidate(f, p, ref.ref(), r["dev_bic"], r["ref_dev_bic"], r["val"],
                              depth, f.nbytes())
                if retain(c):
                    rep.bump("accepted")
                    nxt.append(c)
        nxt.sort(key=lambda c: -c.val["mean"])
        frontier = [(c.mechanism, c.mechanism) for c in nxt[:budget.beam]]
        if not frontier:
            break
    return finish("exhausted")


__all__ = ["SearchBudget", "SearchReport", "Candidate", "search", "enumerate_proposals",
           "iter_proposals", "live_abandoned_fits", "MAX_ABANDONED_FITS",
           "SPLIT_QUANTILES"]
