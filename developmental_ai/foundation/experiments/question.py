"""An experiment's QUESTION: targeted information, a model check, and an
honest conclusion (verifier findings F6 / E10c / E10d, stage9_11/RESULTS.md).

Two failures this module exists for, both measured by the independent
verifier on the Stage 10 selector:

  E10c  a learnable but USELESS distractor (a deterministic 4-bit lamp next
        to the cue->response box) took 44% of all interactions, because
        I(H;O) over the JOINT hypothesis set (shift x lamp) pays log 2 nats
        for every lamp query against ~0.25 for a noisy response.
  E10d  with the truth NOT in the hypothesis set, selection reached > 0.9
        posterior on a (necessarily wrong) hypothesis in 100% of scenarios,
        and nothing said so. The posterior floor does not prevent it.

WHAT IT DOES

  ExperimentQuestion(hset, answer_of, ...) binds a HypothesisSet to the
  question being asked: `answer_of(name)` maps each hypothesis to its
  answer (for an `contracts.Experiment` record the answers must be its
  `alternatives`). Then:

    information(table)  targeted information I(Q;O) (information.
                        targeted_information): nuisance factors are
                        marginalised; I(H;O|Q) is returned as `nuisance`,
                        logged, never added to info_gain.
    observe(...)        the ONE real-evidence path (information.
                        check_real_evidence + update_from_real_outcome),
                        which also feeds the model check below.
    status()            open | provisional | resolved | misspecified, with
                        `reliable` and `request_revision`.
    conclusion(exp)     the Experiment record with its conclusion in the
                        payload, marked unreliable whenever it is.

  MisspecificationMonitor — a per-hypothesis anytime-valid model check
  (an e-detector). For hypothesis h and outcome o of an experiment whose
  outcome table is T, the bet

        r_{h,lam}(o) = ((1 - lam) T_h(o) + lam U(o)) / T_h(o)       U uniform

  has E_{o ~ T_h}[r] = 1 exactly: if h were the truth, no sequence of
  experiments could make its product grow except by chance (Ville:
  P(sup >= 1/alpha) <= alpha). The CUSUM form M <- max(1, M) * r restarts
  the bet whenever h fits again. Hypothesis h is REJECTED when
  max_lam M_{h,lam} >= |lambdas| / alpha (union bound over the grid). The
  SET is flagged misspecified when EVERY hypothesis is rejected: no
  alternative it contains is consistent with the real outcomes. A noisy TV
  (T_h = U for every h) bets nothing (r = 1): noise is not misfit.

  The inference layer's own flag is CONSUMED when HypothesisSet carries
  one: inference/hypotheses.py (2026-10-03) added `misspecified` (a
  posterior-predictive score-deficit CUSUM fed by `update_categorical`).
  observe() updates the set through update_categorical whenever it exists,
  so that monitor sees every real outcome, and its flag is OR-ed with the
  local one: either detector can flag. `_inference_flag` reads it
  defensively (bool attribute/property, zero-arg method, or a mapping with
  "flag"); if it is absent only the local check applies. The two are
  complementary: the inference CUSUM scores the posterior MIXTURE's
  predictive (sensitive while the posterior is spread), the e-detector
  scores each hypothesis separately (sensitive once it is confident).

RESOLUTION REQUIRES A SURVIVED CHECK. A posterior above `threshold` on one
answer makes the conclusion PROVISIONAL, not resolved: the model check can
only fail on outcomes the confident hypothesis predicted, so the
conclusion is reported as resolved only after it has PASSED tests worth
`confirm` expected failures: each later real outcome that bears on the
answer (`bears_on_question`: its distribution differs between answers — a
TV watch or a lamp query does not count) and that the leading answer did
not bet against is credited with its RISK, the probability it would have
failed the answer had another alternative been true (equal weights over
the other alternatives). A sharp test of the answer's own prediction is
worth ~1; a test whose outcome every alternative predicts alike is worth
~0. All while the set is not flagged. A FAILED test — an outcome the
leading answer gave less than uniform probability — resets the count and
demotes even a resolved conclusion back to provisional. That is
the price of reporting identification honestly: about `confirm` extra
interactions on a well-specified set (measured in POSTFIX_selection.md).

WHILE PROVISIONAL selection plans on the posterior mixed with uniform
(`planning_probs`): with the leading answer holding most of the mixture,
the most informative experiment is the one whose outcome that answer
predicts most sharply — i.e. selection seeks the outcome that could
falsify the conclusion, instead of idling on a question it believes is
settled. On a well-specified set those tests pass and cost ~`confirm`
interactions; on a misspecified one they fail, the posterior leaves the
leading answer, and the monitor accumulates misfit against every
alternative in turn.

WHEN FLAGGED (misspecified): status is "misspecified", `reliable` False,
`request_revision` True (the structural-revision request for the
discovery stage), the conclusion is never "resolved", and `planning_probs()`
returns the posterior mixed half-and-half with uniform so that targeted
information RE-OPENS and selection keeps probing instead of treating a
confident wrong posterior as a settled question.

RECOVERY (CLAUDE.md §4.1 — the flag is a guard, so what re-opens it):
  * automatic: the e-detector is CUSUM-form and capped at twice its
    threshold, so as soon as some hypothesis fits the incoming outcomes
    again its statistic decays below threshold within a bounded number of
    fitting outcomes (log_threshold / the per-outcome decrease; ~45 sharp
    outcomes at the default grid) and the flag clears by itself;
  * structural: when the hypothesis set's membership changes (discovery
    adds or retires a hypothesis) the monitor notices and restarts on the
    new set — revision is exactly what the flag requests;
  * explicit: `reset(reason)` (a change detector's alarm), reason recorded.
None of these needs a human.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..contracts import Action, ContractError, Experiment, Observation
from ..inference import HypothesisSet
from .information import (_TINY, check_real_evidence, targeted_information,
                          update_from_real_outcome)

OPEN, PROVISIONAL, RESOLVED, MISSPECIFIED = "open", "provisional", "resolved", "misspecified"
STATES = (OPEN, PROVISIONAL, RESOLVED, MISSPECIFIED)
_FLAT = 1e-9          # answer-conditional rows closer than this: the outcome tests nothing


def _inference_flag(hset: HypothesisSet) -> Optional[bool]:
    """HOOK for the inference layer's own misspecification flag.

    inference/hypotheses.py is gaining a `misspecified` flag (concurrent
    work). If HypothesisSet exposes it — as a bool attribute/property or a
    zero-argument method — its value is OR-ed with the local monitor. If it
    is absent (or not a bool), None is returned and only the local check
    applies. Nothing else in this module depends on its shape."""
    f = getattr(hset, "misspecified", None)
    if f is None:
        return None
    if callable(f):
        try:
            f = f()
        except TypeError:
            return None
    if isinstance(f, (bool, np.bool_)):
        return bool(f)
    if isinstance(f, Mapping) and isinstance(f.get("flag"), (bool, np.bool_)):
        return bool(f["flag"])
    return None


def bears_on_question(table, labels) -> bool:
    """True iff the outcome distribution depends on the ANSWER: the
    per-answer mean rows of `table` differ. A TV (identical rows) or a lamp
    (rows differ only within an answer, by the nuisance factor) is no test
    of a conclusion about the question."""
    T = np.asarray(table, np.float64)
    groups: Dict[Any, List[int]] = {}
    for i, a in enumerate(labels):
        groups.setdefault(a, []).append(i)
    means = np.stack([T[ix].mean(0) for ix in groups.values()])
    return bool(np.abs(means - means.mean(0)).max() > _FLAT)


class MisspecificationMonitor:
    def __init__(self, alpha: float = 0.01,
                 lambdas: Sequence[float] = (0.25, 0.5, 1.0)):
        if not 0.0 < float(alpha) < 1.0:
            raise ContractError("alpha must be in (0, 1)")
        lam = np.asarray([float(x) for x in lambdas], np.float64)
        if lam.size < 1 or np.any(lam <= 0) or np.any(lam > 1):
            raise ContractError("lambdas must be in (0, 1]")
        self.alpha, self.lambdas = float(alpha), lam
        self.log_threshold = math.log(lam.size / self.alpha)
        # capped at twice the threshold: a long misfit cannot bank unbounded
        # evidence, so recovery after the world fits again takes a bounded
        # number of fitting outcomes (CLAUDE.md §4.1)
        self.log_cap = 2.0 * self.log_threshold
        self.names: List[str] = []
        self._logM = np.zeros((0, lam.size))      # CUSUM log statistics (H, G)
        self.resets: List[Dict[str, Any]] = []
        self.history: List[bool] = []

    def _sync(self, names: Sequence[str]) -> None:
        names = list(names)
        if names != self.names:
            if self.names:
                self.resets.append({"reason": "hypothesis set membership changed "
                                              "(structural revision)",
                                    "before": list(self.names), "after": names})
            self.names = names
            self._logM = np.zeros((len(names), self.lambdas.size))

    def reset(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ContractError("a model-check reset needs a stated reason")
        self.resets.append({"reason": reason, "before": list(self.names),
                            "after": list(self.names)})
        self._logM = np.zeros_like(self._logM)

    def observe(self, names: Sequence[str], table, outcome: int) -> Dict[str, Any]:
        """Score one REAL outcome (callers go through ExperimentQuestion.
        observe, which enforces the real-evidence rule)."""
        self._sync(names)
        T = np.asarray(table, np.float64)
        if T.ndim != 2 or T.shape[0] != len(self.names) or T.shape[1] < 1:
            raise ContractError(f"outcome table shape {T.shape} != (H={len(self.names)}, O)")
        if np.any(T < 0) or not np.isfinite(T).all() or \
                not np.allclose(T.sum(1), 1.0, atol=1e-6):
            raise ContractError("outcome table rows must be distributions")
        o = int(outcome)
        if not 0 <= o < T.shape[1]:
            raise ContractError(f"outcome {o} outside the table's {T.shape[1]} outcomes")
        u = 1.0 / T.shape[1]
        th = np.maximum(T[:, o], 1e-12)                                   # (H,)
        q = (1.0 - self.lambdas)[None, :] * th[:, None] + self.lambdas[None, :] * u
        lr = np.log(q) - np.log(th)[:, None]                              # (H, G)
        self._logM = np.minimum(np.maximum(self._logM, 0.0) + lr, self.log_cap)
        flag = self.flagged()
        self.history.append(flag)
        return {"flagged": flag, "rejected": int(self.rejected().sum())}

    def statistic(self) -> np.ndarray:
        """(H,) log e-detector per hypothesis (max over the lambda grid)."""
        return self._logM.max(1) if self._logM.size else np.zeros(0)

    def rejected(self) -> np.ndarray:
        return self.statistic() >= self.log_threshold

    def flagged(self) -> bool:
        r = self.rejected()
        return bool(r.size > 0 and r.all())


AnswerOf = Union[Mapping[str, Any], Callable[[str], Any]]


class ExperimentQuestion:
    def __init__(self, hset: HypothesisSet, answer_of: AnswerOf,
                 threshold: float = 0.9, confirm: int = 5,
                 monitor: Optional[MisspecificationMonitor] = None,
                 experiment: Optional[Experiment] = None, flag_mix: float = 0.5):
        if not isinstance(hset, HypothesisSet):
            raise ContractError("an ExperimentQuestion is asked of a HypothesisSet")
        if not 0.5 <= float(threshold) < 1.0:
            raise ContractError("threshold must be in [0.5, 1)")
        if isinstance(confirm, bool) or not isinstance(confirm, (int, float)) \
                or not math.isfinite(confirm) or confirm < 0:
            raise ContractError("confirm must be a finite number >= 0")
        if not 0.0 < float(flag_mix) <= 1.0:
            raise ContractError("flag_mix must be in (0, 1]")
        if experiment is not None and not isinstance(experiment, Experiment):
            raise ContractError("experiment must be a contracts.Experiment record")
        self.hset, self.threshold, self.confirm = hset, float(threshold), float(confirm)
        self._answer_of = answer_of
        self.monitor = monitor if monitor is not None else MisspecificationMonitor()
        self.experiment, self.flag_mix = experiment, float(flag_mix)
        self._prov_answer: Any = None
        self._confirmed = 0.0
        self.resolved_at: Optional[int] = None
        self.steps = 0
        self.flag_steps = 0
        self._labels_key: Optional[tuple] = None
        self._labels: List[Any] = []
        self._plan_key: Optional[tuple] = None
        self._plan: Any = None
        self.labels()                                   # validate now, not later

    # ---- the question ---------------------------------------------------------
    def labels(self) -> List[Any]:
        key = tuple(self.hset.names)
        if self._labels_key == key:              # membership unchanged: reuse
            return list(self._labels)
        out = []
        for n in self.hset.names:
            a = self._answer_of(n) if callable(self._answer_of) else self._answer_of.get(n)
            if a is None:
                raise ContractError(f"hypothesis {n!r} gives no answer to the question")
            if self.experiment is not None and a not in self.experiment.alternatives:
                raise ContractError(f"answer {a!r} of {n!r} is not one of the "
                                    f"experiment's alternatives "
                                    f"{self.experiment.alternatives}")
            out.append(a)
        self._labels_key, self._labels = key, list(out)
        return out

    def answer_probs(self) -> Dict[Any, float]:
        out: Dict[Any, float] = {}
        for a, p in zip(self.labels(), self.hset.probs()):
            out[a] = out.get(a, 0.0) + float(p)
        return out

    def misspecified(self) -> bool:
        self.monitor._sync(self.hset.names)          # a revised set restarts the check
        inf = _inference_flag(self.hset)
        return bool(self.monitor.flagged() or inf is True)

    def planning_probs(self) -> np.ndarray:
        """The posterior selection should plan on. OPEN: the posterior itself.
        PROVISIONAL or MISSPECIFIED: the posterior mixed with uniform
        (`flag_mix`), i.e. SKEPTICALLY — information about the question
        re-opens, and the most informative experiment under the mixture is
        the one whose outcome the leading answer predicts most sharply: the
        test that could falsify it. RESOLVED: the posterior (the question is
        closed; its info is ~0 and selection moves on)."""
        p = self.hset.probs()
        if self.misspecified() or (self._prov_answer is not None
                                   and self.resolved_at is None):
            p = (1.0 - self.flag_mix) * p + self.flag_mix / p.size
        return p

    def information(self, table) -> Dict[str, Any]:
        # one planning posterior per belief state (candidates of one decision
        # share it); the key changes on any update, membership change or
        # question-state change
        key = (self.hset.steps, tuple(self.hset.names), self.steps, self._prov_answer,
               self.resolved_at, len(self.monitor.resets))
        if self._plan_key != key:
            self._plan = (self.planning_probs(), self.labels(), self.misspecified())
            self._plan_key = key
        p, labels, flagged = self._plan
        out = targeted_information(p, table, labels)
        out["misspecified"] = flagged
        return out

    # ---- real evidence --------------------------------------------------------
    def observe(self, table, outcome: int, *, executed_action: Action,
                evidence: Sequence[Observation]) -> float:
        check_real_evidence(executed_action, evidence)
        T = np.asarray(table, np.float64)
        o = int(outcome)
        if T.ndim != 2 or T.shape[0] != len(self.hset) or not 0 <= o < T.shape[1]:
            raise ContractError("table/outcome do not match the hypothesis set")
        labels = self.labels()
        tests_answer = bears_on_question(T, labels)
        # the prediction the PROVISIONAL conclusion made, before seeing o:
        # P(o | answer) with nuisance factors marginalised
        failed, risk = False, 0.0
        if self._prov_answer is not None and tests_answer:
            p = self.hset.probs()
            m = np.array([a == self._prov_answer for a in labels])
            pred = (p[m] @ T[m]) / max(float(p[m].sum()), _TINY)
            against = pred < 1.0 / T.shape[1]            # outcomes the answer bets against
            failed = bool(against[o])
            if (~m).any():
                # RISK: had the answer been wrong (any other alternative, equally
                # weighted), how likely was this test to fail it?
                risk = float(T[~m].mean(0)[against].sum())
        self.monitor.observe(self.hset.names, T, o)
        realized = update_from_real_outcome(
            self.hset, None, executed_action=executed_action, evidence=evidence,
            outcome_table=T, outcome=o)
        self.steps += 1
        flagged = self.misspecified()
        self.flag_steps += flagged
        ap = self.answer_probs()
        best = max(ap, key=ap.get)
        if flagged or ap[best] <= self.threshold:
            self._prov_answer, self._confirmed = None, 0.0
            self.resolved_at = None
        elif best != self._prov_answer:
            self._prov_answer, self._confirmed = best, 0.0
            self.resolved_at = None
        elif failed:
            # a failed test demotes the conclusion, even a resolved one
            self._confirmed, self.resolved_at = 0.0, None
        elif tests_answer and self.resolved_at is None:
            # a test the confident answer could have failed, and passed:
            # credited by how likely it was to fail a wrong answer
            self._confirmed += risk
            if self._confirmed >= self.confirm - 1e-9:
                self.resolved_at = self.steps
        if self._prov_answer is not None and self.confirm == 0 and self.resolved_at is None:
            self.resolved_at = self.steps
        return realized

    # ---- reading the answer -----------------------------------------------------
    def status(self) -> Dict[str, Any]:
        ap = self.answer_probs()
        best = max(ap, key=ap.get)
        flagged = self.misspecified()
        if flagged:
            state = MISSPECIFIED
        elif self._prov_answer is not None and self.resolved_at is not None:
            state = RESOLVED
        elif self._prov_answer is not None:
            state = PROVISIONAL
        else:
            state = OPEN
        return {"state": state, "answer": best if state == RESOLVED else None,
                "leading": best, "p_leading": ap[best],
                "reliable": state == RESOLVED, "misspecified": flagged,
                "request_revision": flagged, "confirmed": self._confirmed,
                "confirm_needed": self.confirm, "local_flag": self.monitor.flagged(),
                "inference_flag": _inference_flag(self.hset),
                "rejected_hypotheses": int(self.monitor.rejected().sum()),
                "n_hypotheses": len(self.hset)}

    def conclusion(self, experiment: Optional[Experiment] = None) -> Experiment:
        exp = experiment if experiment is not None else self.experiment
        if not isinstance(exp, Experiment):
            raise ContractError("conclusion() needs the Experiment record")
        st = self.status()
        payload = dict(exp.payload)
        payload["conclusion"] = {
            "state": st["state"], "answer": st["answer"], "leading": st["leading"],
            "p_leading": st["p_leading"], "reliable": st["reliable"],
            "request_revision": st["request_revision"],
            "why_unreliable": (None if st["reliable"] else
                               "hypothesis set misspecified: every alternative is "
                               "rejected by the model check" if st["misspecified"] else
                               f"not resolved: {st['state']} ({st['confirmed']:.2f}/"
                               f"{st['confirm_needed']:g} expected failures survived)")}
        return exp.replace(payload=payload)
