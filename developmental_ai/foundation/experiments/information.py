"""Expected information of an experiment, and the rule that only REAL
outcomes count as evidence (plan Stage 10 item 2; §7.4 "a model cannot
certify its own imagined outcomes as real evidence").

All quantities are in NATS (natural log), matching inference.HypothesisSet.

(a) FINITE HYPOTHESIS SET — `hypothesis_information(probs, table)`.
    `table[h, o]` = P(outcome o | hypothesis h, this candidate). The value of
    running the candidate is the mutual information between the hypothesis
    and the outcome, which equals the expected KL from prior to posterior:

        I(H; O) = H[ sum_h p_h T_h ]  -  sum_h p_h H[T_h]
                = E_o KL( p(h | o) || p(h) )

    An outcome whose distribution is the SAME under every hypothesis (a
    noisy TV, an idle observation of something no hypothesis is about) has
    I = 0 exactly (values under NUMERICAL_FLOOR = 1e-12 nats are float dust
    and are reported as 0), however random it is: its noise lives entirely in the
    second term. The first term alone ("predictive entropy") is what a
    novelty / prediction-error drive rewards, and it is maximal on a fair
    coin — that is the noisy-TV trap, kept below as the ablation witness.

(a') TARGETED — `targeted_information(probs, table, question)`. A hypothesis
    set often carries factors the experiment's question is not about (a
    learner that models everything it sees carries shift x lamp). I(H;O)
    then pays for resolving ANY of them: the verifier's E10c distractor (a
    deterministic 4-bit lamp, log 2 nats per query against ~0.25 for a
    noisy response) took 44% of all interactions. Targeted information is
    I(Q;O) with Q = the question variable (one answer label per
    hypothesis), nuisance factors marginalised; I(H;O|Q) is returned as
    `nuisance` and is never part of `info` (see its docstring for where it
    belongs instead). With Q = H it equals (a).

(b) ENSEMBLE — `ensemble_information(member_outcomes, ...)`, BALD-style:
    total predictive entropy minus the members' expected entropy.
      categorical groups: H[mean_m p_m] - mean_m H[p_m]   (exact BALD)
      Gaussian outputs:   0.5 * log(1 + epistemic / aleatoric) per output,
                          epistemic = variance of member means, aleatoric =
                          mean member variance. This is the exact I(y; m)
                          for y = mu_m + eps with mu_m Gaussian across
                          members — a moment-matched approximation otherwise.
    Pure aleatoric noise the members agree on scores ~0.

    FALSE DISAGREEMENT. Members can disagree about outputs that have nothing
    to do with the question being asked. `cont_index` / `disc_index` declare
    the RELEVANT outputs; disagreement elsewhere is not information about
    this question and is not scored. There is no default "all outputs" —
    the caller must say what the experiment is about.

    MODEL-GENERATED JACKPOTS. One member predicting something wild produces
    a huge raw disagreement that more data would "resolve" only by
    discarding that member. The robust score is the MINIMUM over
    leave-one-member-out scores: disagreement that a single member carries
    vanishes when that member is left out; disagreement spread across the
    ensemble survives. (Two colluding outliers survive this — the absolute
    cap below still bounds them.) Then an absolute cap, DEFAULT_INFO_CAP =
    log 16 nats: no question posed here distinguishes more than
    HypothesisSet's default 16 alternatives, so no single interaction is
    worth more. `jackpot` / `capped` flags are returned, never hidden.

REAL EVIDENCE ONLY — `update_from_real_outcome`. Expected information is a
FORECAST; it is never booked as learning. The posterior moves only through
this function, which requires (i) the EXECUTED Action record (completed:
t_complete known), and (ii) observations of provenance "sensor" from that
action's own scope, after it. "imagined" and "inferred" are model outputs —
a model cannot certify them — and "evaluator" is privileged; all three raise
ImaginedEvidenceError. The realized information (KL posterior || prior, in
nats) is returned so the caller books what was actually learned, not what
was predicted.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ..contracts import Action, ContractError, Observation, UNKNOWN
from ..inference import HypothesisSet
from ..mechanisms.api import MixedOutcome

DEFAULT_INFO_CAP = math.log(16.0)
REAL_PROVENANCE = "sensor"
_TINY = 1e-300
NUMERICAL_FLOOR = 1e-12   # nats; differences below this are float dust, reported as 0


class ImaginedEvidenceError(ContractError):
    """A model output (or evaluator data) was offered as real evidence."""


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 0.0, 1.0)
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, _TINY)), 0.0), axis=axis)


def _probs(x) -> np.ndarray:
    if isinstance(x, HypothesisSet):
        return x.probs()
    p = np.asarray(x, np.float64)
    if p.ndim != 1 or p.size < 1 or np.any(p < 0) or not np.isfinite(p).all() \
            or abs(p.sum() - 1.0) > 1e-6:
        raise ContractError("hypothesis probabilities must be a finite (H,) "
                            "vector >= 0 summing to 1")
    return p


def hypothesis_information(probs, table, cap: float = DEFAULT_INFO_CAP) -> Dict[str, float]:
    """I(H; O) for one candidate. `probs`: (H,) or a HypothesisSet; `table`:
    (H, O) rows = P(o | h). Returns info (capped), the two entropy terms
    and `capped`."""
    p = _probs(probs)
    T = np.asarray(table, np.float64)
    if T.ndim != 2 or T.shape[0] != p.size or T.shape[1] < 1:
        raise ContractError(f"outcome table shape {T.shape} must be (H={p.size}, O)")
    if np.any(T < 0) or not np.isfinite(T).all() or \
            not np.allclose(T.sum(1), 1.0, atol=1e-6):
        raise ContractError("outcome table rows must be distributions")
    pred = p @ T
    hp = float(_entropy(pred))
    hc = float(np.dot(p, _entropy(T, 1)))
    raw = hp - hc
    raw = 0.0 if raw < NUMERICAL_FLOOR else raw
    info = min(raw, float(cap))
    return {"info": info, "raw": raw, "predictive_entropy": hp,
            "expected_conditional_entropy": hc, "capped": raw > info + 1e-12}


def _question_codes(question, H: int) -> np.ndarray:
    labels = list(question)
    if len(labels) != H:
        raise ContractError(f"question labels {len(labels)} != hypotheses {H}: every "
                            f"hypothesis must answer the question (one label each)")
    for q in labels:
        if q is None or isinstance(q, float) and not math.isfinite(q):
            raise ContractError(f"question label {q!r} is not a valid answer")
    index: Dict[Any, int] = {}
    return np.array([index.setdefault(q, len(index)) for q in labels], np.int64)


def targeted_information(probs, table, question,
                         cap: float = DEFAULT_INFO_CAP) -> Dict[str, Any]:
    """I(Q; O): what the outcome says about the QUESTION, not about every
    factor the hypothesis set happens to carry (verifier finding F6/E10c).

    `question[h]` is hypothesis h's answer to the experiment's question (a
    label; hypotheses sharing a label are one alternative — e.g. the joint
    set shift x lamp labelled by shift). Nuisance factors are MARGINALISED:

        P(o | q) = sum_{h: Q(h)=q} p_h T_h(o) / p(q)
        I(Q; O)  = H[ sum_q p(q) P(o|q) ]  -  sum_q p(q) H[ P(o|q) ]

    By the chain rule I(H;O) = I(Q;O) + I(H;O|Q); the second term is what
    the outcome says about the nuisance factors alone. It is returned as
    `nuisance` (and the joint as `joint`) and is NEVER part of `info`: a
    learnable but irrelevant distractor (a deterministic lamp pays log 2 per
    query in the joint) scores exactly 0 here unless its outcome is
    correlated with the answer. If knowing a nuisance factor would improve
    competence, that claim belongs to the `usefulness` term (measured on dev
    probes, separately weighted and logged), not to info_gain.

    With one hypothesis per label (Q = H) this equals hypothesis_information
    up to float rounding, so a single-factor question is unchanged."""
    p = _probs(probs)
    T = np.asarray(table, np.float64)
    if T.ndim != 2 or T.shape[0] != p.size or T.shape[1] < 1:
        raise ContractError(f"outcome table shape {T.shape} must be (H={p.size}, O)")
    if np.any(T < 0) or not np.isfinite(T).all() or \
            float(np.abs(T.sum(1) - 1.0).max()) > 1e-6:
        raise ContractError("outcome table rows must be distributions")
    codes = _question_codes(question, p.size)
    nq = int(codes.max()) + 1
    pq = np.bincount(codes, weights=p, minlength=nq)                 # (Q,)
    mix = np.zeros((nq, T.shape[1]))
    np.add.at(mix, codes, p[:, None] * T)                            # sum p_h T_h
    live = pq > _TINY
    cond = np.where(live[:, None], mix / np.maximum(pq, _TINY)[:, None], 0.0)
    pred = mix.sum(0)
    hp = float(_entropy(pred))
    hc = float(np.dot(pq[live], _entropy(cond[live], 1)))
    joint = hp - float(np.dot(p, _entropy(T, 1)))                    # I(H;O)
    joint = 0.0 if joint < NUMERICAL_FLOOR else joint
    raw = hp - hc
    raw = 0.0 if raw < NUMERICAL_FLOOR else min(raw, joint)          # I(Q;O) <= I(H;O)
    info = min(raw, float(cap))
    nuis = joint - raw
    nuis = 0.0 if nuis < NUMERICAL_FLOOR else nuis
    return {"info": info, "raw": raw, "joint": joint, "nuisance": nuis,
            "predictive_entropy": hp, "expected_conditional_entropy": hc,
            "n_answers": nq, "capped": raw > info + 1e-12}


def bald_categorical(member_probs) -> np.ndarray:
    """(M, ..., C) member class probabilities -> (...) BALD in nats."""
    P = np.asarray(member_probs, np.float64)
    if P.ndim < 2 or P.shape[0] < 2:
        raise ContractError("need (M >= 2, ..., C) member probabilities")
    if np.any(P < 0) or not np.allclose(P.sum(-1), 1.0, atol=1e-5):
        raise ContractError("member probabilities must be distributions")
    return np.maximum(0.0, _entropy(P.mean(0)) - _entropy(P).mean(0))


def bald_gaussian(means, variances) -> np.ndarray:
    """(M, ..., D) member means / variances -> (..., D) per-output
    0.5*log(1 + epistemic/aleatoric) in nats."""
    mu = np.asarray(means, np.float64)
    var = np.asarray(variances, np.float64)
    if mu.shape != var.shape or mu.ndim < 2 or mu.shape[0] < 2:
        raise ContractError("need matching (M >= 2, ..., D) means and variances")
    if np.any(var <= 0) or not np.isfinite(var).all() or not np.isfinite(mu).all():
        raise ContractError("member variances must be finite and > 0")
    return 0.5 * np.log1p(mu.var(0) / var.mean(0))


def _raw_score(mu, var, P, cont_index, disc_index) -> np.ndarray:
    s = 0.0
    if cont_index:
        s = s + bald_gaussian(mu[..., cont_index], var[..., cont_index]).sum(-1)
    if disc_index:
        s = s + bald_categorical(P[:, :, disc_index, :]).sum(-1)
    return np.asarray(s, np.float64)


def ensemble_information(member_outcomes: Sequence[MixedOutcome],
                         cont_index: Optional[Sequence[int]] = None,
                         disc_index: Optional[Sequence[int]] = None,
                         cap: float = DEFAULT_INFO_CAP,
                         jackpot_ratio: float = 4.0) -> Dict[str, np.ndarray]:
    """BALD over the RELEVANT outputs of M >= 3 member predictives (each a
    MixedOutcome over the same batch). Returns per batch element: info
    (robust, capped), raw, robust (min leave-one-out), jackpot, capped."""
    outs = list(member_outcomes)
    if len(outs) < 3:
        raise ContractError("jackpot control needs >= 3 members (leave-one-out)")
    if not all(isinstance(o, MixedOutcome) for o in outs):
        raise ContractError("members must predict MixedOutcome")
    lay = outs[0].layout
    if any(o.layout != lay for o in outs) or len({o.batch_size for o in outs}) != 1:
        raise ContractError("members disagree on layout or batch size")
    ci = [] if cont_index is None else [int(i) for i in cont_index]
    di = [] if disc_index is None else [int(i) for i in disc_index]
    if not ci and not di:
        raise ContractError("declare the RELEVANT outputs (cont_index / disc_index): "
                            "disagreement on outputs the question is not about is "
                            "not information about it")
    if any(not 0 <= i < lay.Dc for i in ci) or any(not 0 <= i < lay.G for i in di):
        raise ContractError("relevant output index out of range")
    mu = np.stack([o.mean() for o in outs])                     # (M, B, Dc)
    var = np.stack([o.variance() for o in outs])
    P = np.stack([o.probs_marginal() for o in outs]) if lay.G else \
        np.zeros((len(outs), outs[0].batch_size, 0, 0))
    raw = _raw_score(mu, var, P, ci, di)
    M = len(outs)
    loo = np.stack([_raw_score(np.delete(mu, j, 0), np.delete(var, j, 0),
                               np.delete(P, j, 0), ci, di) for j in range(M)])
    robust = loo.min(0)
    info = np.minimum(robust, float(cap))
    jackpot = raw > float(jackpot_ratio) * robust + 1e-9
    return {"info": info, "raw": raw, "robust": robust, "jackpot": jackpot,
            "capped": raw > info + 1e-12}


def bald_from_ensemble(ensemble, state, action, dt, **kw) -> Dict[str, np.ndarray]:
    """Thin adapter for inference.BootstrapEnsemble (or anything with
    `.members` that each `.predict(state, action, dt) -> MixedOutcome`)."""
    members = getattr(ensemble, "members", None)
    if not members:
        raise ContractError("ensemble exposes no members")
    return ensemble_information([m.predict(state, action, dt) for m in members], **kw)


def update_from_real_outcome(hset: HypothesisSet, logliks, *,
                             executed_action: Action,
                             evidence: Sequence[Observation],
                             outcome_table=None, outcome: Optional[int] = None) -> float:
    """Bayes-update `hset` with REAL evidence; return the realized
    information KL(posterior || prior) in nats (0 when nothing moved).

    With `outcome_table` (H, O) and `outcome` (and `logliks` None), a set
    that offers `update_categorical` (inference/hypotheses.py) is updated
    through it, so the inference layer's own posterior-predictive
    misspecification monitor sees the outcome too; the posterior is the
    same as update(log table[:, outcome])."""
    if not isinstance(hset, HypothesisSet):
        raise ContractError("update_from_real_outcome needs a HypothesisSet")
    check_real_evidence(executed_action, evidence)
    before = hset.log_probs()
    names = list(hset.names)
    if outcome_table is not None:
        T = np.asarray(outcome_table, np.float64)
        if logliks is not None or outcome is None or T.ndim != 2 \
                or T.shape[0] != len(names) or not 0 <= int(outcome) < T.shape[1]:
            raise ContractError("give EITHER logliks OR (outcome_table (H, O), outcome)")
        L = np.log(np.maximum(T, _TINY))
        cat = getattr(hset, "update_categorical", None)
        moved = cat(L, int(outcome)) if callable(cat) else hset.update(L[:, int(outcome)])
    else:
        moved = hset.update(logliks)
    if not moved:
        return 0.0
    if list(hset.names) != names:                       # pragma: no cover
        raise ContractError("hypothesis set changed membership during an update")
    after = hset.log_probs()
    kl = float(np.sum(np.exp(after) * (after - before)))
    return 0.0 if kl < NUMERICAL_FLOOR else kl


def check_real_evidence(executed_action: Action, evidence: Sequence[Observation]) -> None:
    """The real-evidence rule, shared by every consumer of outcomes (the
    posterior update above and question.ExperimentQuestion's model check):
    a completed EXECUTED action, and sensor observations from its own scope,
    after it. Raises otherwise."""
    if not isinstance(executed_action, Action):
        raise ContractError("evidence must be tied to the EXECUTED Action record "
                            "(info['executed_action']), not to the intended one")
    if executed_action.t_complete is UNKNOWN:
        raise ContractError("executed action has no t_complete: it is still pending")
    ev = list(evidence)
    if not ev:
        raise ContractError("no evidence observations given")
    for o in ev:
        if not isinstance(o, Observation):
            raise ContractError("evidence must be Observation records")
        if o.provenance != REAL_PROVENANCE:
            raise ImaginedEvidenceError(
                f"observation {o.channel!r} has provenance {o.provenance!r}: only "
                f"{REAL_PROVENANCE!r} is evidence (imagined/inferred are model outputs "
                f"a model cannot certify; evaluator data is privileged)")
        if (o.environment, o.stream, o.episode) != (
                executed_action.environment, executed_action.stream,
                executed_action.episode):
            raise ContractError(f"evidence from {o.episode!r} does not belong to the "
                                f"executed action's scope {executed_action.episode!r}")
        if o.seq <= executed_action.seq:
            raise ContractError(f"evidence at seq {o.seq} precedes the action "
                                f"(seq {executed_action.seq}): it cannot be its outcome")
