"""Experiment selection as an explicit, logged sum of declared terms
(plan Stage 10 item 3; CLAUDE.md §4.1 and §4.4).

    score(c) = + w_info  * info_gain(c)      expected nats about the open question
               + w_use   * usefulness(c)     expected competence gain (dev probes)
               + w_task  * task(c)           what the current task requires
               - w_eff   * effort(c)         ticks / resources it costs
               - w_risk  * risk(c)           expected harm

Every selection returns the full per-term breakdown (value, weight,
contribution, occluded) for EVERY candidate, so "why did it do that" is
answered from the log, not reconstructed.

THE FOUR DECLARED PROPERTIES (CLAUDE.md §4.4, plan §10). Before any term
may score anything it must say:
    channel    where its number goes (here: the selection score — not
               reward; nothing here writes prim_extrinsic or intrinsic)
    occlusion  what it does when its input is unavailable (UNKNOWN/ABSENT):
               it contributes `occluded_value` and the breakdown says so
    idle       what it pays a candidate that changes nothing
    recovery   what re-opens it if it pins a candidate down (§4.1: a guard
               no reachable event can satisfy is a latch)
`check_terms` raises TermMetadataError if any term lacks any of them, or if
the set is not exactly {info_gain, usefulness, task, effort, risk}. The
selector calls it at construction; it cannot be skipped.

NO HARD GATES. Terms only add finite numbers; no term removes a candidate.
With probability `epsilon` the selector picks uniformly (reason "explore"),
so every candidate offered keeps probability >= epsilon / N on every step —
the reachable recovery path for effort and risk, whose estimates can only
fall if the candidate is sometimes tried.

INFRA HOOKS. `InfraSink` books each selection's term contributions into an
infra.ledger.RewardLedger (sources "selection.<term>") and each term's raw
magnitude into an infra.signal_health.SignalMonitor, WITHOUT modifying
either: both take plain (name, float) calls. The ledger then reports when
one term dominates the decisions for several segments (the same alarm that
caught reward farms) and the monitor reports a term whose signal has gone
flat (e.g. info_gain pinned at 0 once everything is resolved). The ledger
instance is the sink's OWN, never the live loop's reward ledger — selection
scores are not income and must not appear on the reward statement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..contracts import ContractError, is_missing
from .candidates import MAX_CANDIDATES, SKILL, Candidate, CandidateSet

TERM_NAMES = ("info_gain", "usefulness", "task", "effort", "risk")
REQUIRED_PROPERTIES = ("channel", "occlusion", "idle", "recovery")
SELECTION_CHANNEL = ("selection score only (offline/shadow): never reward, never "
                     "prim_extrinsic or intrinsic, never replay priority")


class TermMetadataError(ContractError):
    """A selection term does not declare channel/occlusion/idle/recovery."""


@dataclass(frozen=True)
class TermSpec:
    name: str
    weight: float
    sign: int                     # +1 benefit, -1 cost
    channel: str
    occlusion: str
    idle: str
    recovery: str
    occluded_value: float = 0.0
    description: str = ""


def check_terms(terms: Iterable[Any]) -> Tuple[TermSpec, ...]:
    terms = tuple(terms)
    problems: List[str] = []
    names = []
    for t in terms:
        nm = getattr(t, "name", None)
        names.append(nm)
        for p in REQUIRED_PROPERTIES:
            v = getattr(t, p, None)
            if not isinstance(v, str) or not v.strip():
                problems.append(f"term {nm!r} does not declare {p!r}")
        w = getattr(t, "weight", None)
        if isinstance(w, bool) or not isinstance(w, (int, float)) or \
                not math.isfinite(w) or w < 0:
            problems.append(f"term {nm!r} weight must be a finite float >= 0, got {w!r}")
        if getattr(t, "sign", None) not in (1, -1):
            problems.append(f"term {nm!r} sign must be +1 or -1")
        ov = getattr(t, "occluded_value", None)
        if isinstance(ov, bool) or not isinstance(ov, (int, float)) or not math.isfinite(ov):
            problems.append(f"term {nm!r} occluded_value must be finite")
    if sorted(n for n in names if isinstance(n, str)) != sorted(TERM_NAMES) \
            or len(names) != len(TERM_NAMES):
        problems.append(f"terms must be exactly {TERM_NAMES}, got {names}")
    if problems:
        raise TermMetadataError("; ".join(problems))
    return terms


def default_terms(info_gain: float = 1.0, usefulness: float = 0.0, task: float = 0.0,
                  effort: float = 0.01, risk: float = 0.0) -> Tuple[TermSpec, ...]:
    ch = SELECTION_CHANNEL
    return check_terms((
        TermSpec("info_gain", info_gain, +1, ch,
                 occlusion="an outcome channel that is unavailable carries no "
                           "evidence: an occluded candidate scores 0 info and is "
                           "flagged occluded",
                 idle="an idle candidate whose outcome distribution is the same "
                      "under every hypothesis scores exactly 0 (I(H;O)=0); idle "
                      "observation that DOES discriminate scores what it is worth",
                 recovery="0 once the posterior is resolved; re-opens whenever "
                          "posterior entropy rises (HypothesisSet floor keeps every "
                          "hypothesis >= floor so contrary evidence moves it within "
                          "~log(1/floor)/L steps; change detection adds hypotheses) "
                          "and whenever the question's model check flags the set "
                          "misspecified (ExperimentQuestion.planning_probs re-opens "
                          "it; the flag clears when a hypothesis fits again or the "
                          "set is revised)",
                 description="expected nats about the open QUESTION (targeted "
                             "I(Q;O), nuisance factors marginalised; or robust "
                             "capped BALD on relevant outputs); a forecast, never "
                             "booked as learning"),
        TermSpec("usefulness", usefulness, +1, ch,
                 occlusion="no dev-probe measurement available -> 0, flagged",
                 idle="idle never improves dev-probe loss persistently: net "
                      "(signed) progress on dev probes, so noise fluctuations "
                      "cancel instead of accumulating",
                 recovery="re-opens when any dev probe's loss can still fall; "
                          "relearning credit is discounted, never zeroed, and the "
                          "forgetting flag clears after stable retention",
                 description="expected competence gain on the DEVELOPMENT probe "
                             "set only (held-out ids refused)"),
        TermSpec("task", task, +1, ch,
                 occlusion="task requirement unknown -> 0, flagged",
                 idle="pays an idle candidate only if the task requires idling",
                 recovery="follows the task definition; changes when the task does",
                 description="external task requirement (0 for pure discovery)"),
        TermSpec("effort", effort, -1, ch,
                 occlusion="cost unknown -> charged as 1 tick (never free)",
                 idle="an idle candidate is charged its ticks like any other: "
                      "doing nothing is not free",
                 recovery="never excludes a candidate: a finite subtraction, and "
                          "epsilon-exploration keeps every candidate reachable",
                 occluded_value=1.0,
                 description="ticks (x tick cost) the candidate consumes"),
        TermSpec("risk", risk, -1, ch,
                 occlusion="risk unknown -> 0 penalty, flagged (unknown is not "
                           "dangerous by fiat)",
                 idle="idle candidates carry their own estimated risk",
                 recovery="a finite subtraction, never a ban; estimates fall when "
                          "the candidate is executed safely, and epsilon-"
                          "exploration guarantees it is executed sometimes",
                 description="expected harm estimate"),
    ))


def effort_of(candidate: Candidate, tick_cost: float = 1.0,
              skill_overhead: float = 0.0) -> float:
    """Ticks consumed x tick cost (+ a fixed overhead for a skill call)."""
    e = float(candidate.duration_ticks) * float(tick_cost)
    if candidate.kind == SKILL:
        e += float(skill_overhead)
    return e


@dataclass
class Selection:
    step: int
    chosen: Candidate
    reason: str                                  # "argmax" | "explore"
    total: float
    breakdown: Dict[str, Dict[str, Any]]         # chosen candidate's terms
    table: Dict[str, Dict[str, Any]]             # every candidate: total + terms
    question: Optional[Dict[str, Any]] = None    # ExperimentQuestion.status() if given

    @property
    def misspecified(self) -> bool:
        return bool(self.question and self.question.get("misspecified"))

    def to_log(self) -> Dict[str, Any]:
        d = {"step": self.step, "chosen": self.chosen.candidate_id,
             "reason": self.reason, "total": self.total,
             "terms": {k: dict(v) for k, v in self.breakdown.items()}}
        if self.question is not None:
            d["question"] = {k: self.question.get(k) for k in
                             ("state", "leading", "p_leading", "reliable",
                              "misspecified", "request_revision")}
        return d


class ExperimentSelector:
    def __init__(self, terms: Optional[Sequence[TermSpec]] = None, epsilon: float = 0.05,
                 seed: int = 0, sink: Optional["InfraSink"] = None,
                 tie_tol: float = 1e-12):
        self.terms = check_terms(default_terms() if terms is None else terms)
        if not 0.0 <= float(epsilon) < 1.0:
            raise ContractError("epsilon must be in [0, 1)")
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.sink = sink
        self.tie_tol = float(tie_tol)
        self.steps = 0
        self.log: List[Dict[str, Any]] = []

    def score(self, values: Mapping[str, Any]) -> Tuple[float, Dict[str, Dict[str, Any]]]:
        missing = [t.name for t in self.terms if t.name not in values]
        if missing:
            raise ContractError(f"partial scoring: no value for {missing} (declare "
                                f"UNKNOWN for an unavailable input, never omit it)")
        extra = set(values) - set(TERM_NAMES)
        if extra:
            raise ContractError(f"undeclared terms {sorted(extra)}")
        total, out = 0.0, {}
        for t in self.terms:
            v = values[t.name]
            occluded = is_missing(v)
            if occluded:
                v = t.occluded_value
            if isinstance(v, (bool, np.bool_)):
                raise ContractError(f"term {t.name} value may not be bool")
            v = float(v)
            if not math.isfinite(v):
                raise ContractError(f"term {t.name} value {v} is not finite")
            if t.sign < 0 and v < 0:
                raise ContractError(f"cost term {t.name} must be >= 0 (its sign is "
                                    f"applied by the spec), got {v}")
            c = t.sign * t.weight * v
            total += c
            out[t.name] = {"value": v, "weight": t.weight, "sign": t.sign,
                           "contribution": c, "occluded": bool(occluded)}
        return total, out

    def select(self, candidates, term_values: Mapping[str, Mapping[str, Any]],
               status: Optional[Mapping[str, Any]] = None) -> Selection:
        """`status` (optional): the open question's ExperimentQuestion.status().
        It changes no score — it is carried into the Selection and the log so
        a misspecified question (unreliable posterior, revision requested)
        is visible on every decision taken under it."""
        cands = list(candidates.candidates if isinstance(candidates, CandidateSet)
                     else candidates)
        if not cands:
            raise ContractError("no candidates to select from")
        if len(cands) > MAX_CANDIDATES:
            raise ContractError(f"{len(cands)} candidates > MAX_CANDIDATES={MAX_CANDIDATES}")
        if not all(isinstance(c, Candidate) for c in cands):
            raise ContractError("candidates must be Candidate records "
                                "(from enumerate_candidates)")
        ids = [c.candidate_id for c in cands]
        if len(set(ids)) != len(ids):
            raise ContractError("duplicate candidate ids")
        unknown = set(term_values) - set(ids)
        if unknown:
            raise ContractError(f"term values for candidates not offered: {sorted(unknown)}")
        table = {}
        for c in cands:
            if c.candidate_id not in term_values:
                raise ContractError(f"candidate {c.candidate_id} has no term values")
            tot, br = self.score(term_values[c.candidate_id])
            table[c.candidate_id] = {"total": tot, "terms": br}
        totals = np.array([table[i]["total"] for i in ids])
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            k, reason = int(self.rng.integers(len(cands))), "explore"
        else:
            best = np.flatnonzero(totals >= totals.max() - self.tie_tol)
            k, reason = int(self.rng.choice(best)), "argmax"
        ch = cands[k]
        if status is not None and not isinstance(status, Mapping):
            raise ContractError("status must be a mapping (ExperimentQuestion.status())")
        sel = Selection(self.steps, ch, reason, float(totals[k]),
                        table[ch.candidate_id]["terms"], table,
                        None if status is None else dict(status))
        self.steps += 1
        self.log.append(sel.to_log())
        if self.sink is not None:
            self.sink.record(sel)
        return sel


class CostAccount:
    """What experiments actually cost and actually taught (plan Stage 10
    completion gate: knowledge per interaction, not intrinsic reward size).
    Forecast information is kept apart from realized information."""

    def __init__(self):
        self.interactions = 0
        self.ticks = 0
        self.effort = 0.0
        self.forecast_nats = 0.0
        self.realized_nats = 0.0
        self.by_candidate: Dict[str, Dict[str, float]] = {}

    def charge(self, selection: Selection, executed_ticks: int,
               realized_nats: float) -> None:
        if isinstance(executed_ticks, bool) or int(executed_ticks) < 0:
            raise ContractError("executed_ticks must be an int >= 0")
        r = float(realized_nats)
        if not math.isfinite(r) or r < 0:
            raise ContractError("realized information must be finite and >= 0")
        self.interactions += 1
        self.ticks += int(executed_ticks)
        e = float(selection.breakdown["effort"]["value"])
        f = float(selection.breakdown["info_gain"]["value"])
        self.effort += e
        self.forecast_nats += f
        self.realized_nats += r
        b = self.by_candidate.setdefault(selection.chosen.candidate_id,
                                         {"n": 0, "ticks": 0, "realized_nats": 0.0})
        b["n"] += 1
        b["ticks"] += int(executed_ticks)
        b["realized_nats"] += r

    def summary(self) -> Dict[str, Any]:
        n = max(self.interactions, 1)
        return {"interactions": self.interactions, "ticks": self.ticks,
                "effort": self.effort, "forecast_nats": self.forecast_nats,
                "realized_nats": self.realized_nats,
                "realized_nats_per_interaction": self.realized_nats / n,
                "realized_nats_per_tick": self.realized_nats / max(self.ticks, 1)}


class InfraSink:
    """Books selections into infra.ledger.RewardLedger and
    infra.signal_health.SignalMonitor (both unmodified; both contain their
    own failures and never raise).

    The ledger is PRIVATE and cannot be injected. `RewardLedger.segment()`
    is a consuming read (tests/_metrics_sink_smoke.py contract 7): handed
    the live loop's ledger, `segment()` below would close and reset the
    real statement. Owning a fresh one is what makes this the only safe
    second caller, and the metrics-sink smoke asserts it stays that way."""

    def __init__(self, monitor=None, monitor_window: int = 200,
                 monitor_min_obs: int = 50):
        from ...infra.ledger import RewardLedger
        from ...infra.signal_health import SignalMonitor
        self.ledger = RewardLedger()
        self.monitor = monitor if monitor is not None else SignalMonitor(
            window=monitor_window, min_obs=monitor_min_obs)

    def record(self, sel: Selection) -> None:
        for name, t in sel.breakdown.items():
            self.ledger.record(f"selection.{name}", t["contribution"])
            # SignalMonitor takes probabilities: squash |value| to [0, 1).
            self.monitor.observe(f"selection.{name}", 1.0 - math.exp(-abs(t["value"])))

    def segment(self) -> Dict[str, Any]:
        return {"statement": self.ledger.segment(),
                "degenerate_terms": sorted(self.monitor.degenerate()),
                "bits": self.monitor.bits()}
