"""Unit cases for foundation.experiments Stage 10 (curiosity as experiment
selection): candidates, information, selection terms, progress guard,
retention tracker.

Parts only: candidates come from the ActionSpec and nothing else (discrete
count, box bounds, skill admission with reasons, caps, truncation needs an
rng), information arithmetic (MI identities and bounds, BALD shapes, LOO
jackpot flag, relevant-output declaration required), real-evidence refusals
(provenance, scope, pending action), term metadata validation, selector
bookkeeping (breakdown, partial scoring refused, UNKNOWN -> occluded value,
cost terms non-negative), DevProbeSet refusals, RetentionTracker hysteresis
(sustained, noise-aware state changes) and its recovery path, targeted
information identities (Q = H, chain rule, nuisance-only tables),
MisspecificationMonitor bounds (noise bets nothing, cap, reset), the
ExperimentQuestion status and the inference-flag hook, Selection carrying
the question status, and the InfraSink hook into infra/ledger.py and
infra/signal_health.py. The design claims (noisy TV, idle, passive
observation, jackpots, forgetting loops, the A/B) live in
tests/_foundation_selection_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_foundation_selection_unit.py
"""
import dataclasses
import math
import sys

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import between, case, close, raises, run_all
from developmental_ai.foundation.contracts import (ABSENT, UNKNOWN, Action, ActionSpec,
                                                   ContractError, Observation, Skill)
from developmental_ai.foundation.experiments import (
    DEFAULT_INFO_CAP, MAX_CANDIDATES, MAX_HORIZON, TERM_NAMES, CostAccount,
    CueShiftHypotheses, DevProbeSet, ExperimentQuestion, ExperimentSelector,
    HeldOutAccessError, ImaginedEvidenceError, InfraSink, MisspecificationMonitor,
    RetentionTracker, TermMetadataError, bald_categorical, bald_gaussian,
    bears_on_question, check_terms, default_terms, effort_of, ensemble_information,
    enumerate_candidates, hypothesis_information, make_split, targeted_information,
    update_from_real_outcome)
from developmental_ai.foundation.inference import HypothesisSet
from developmental_ai.foundation.mechanisms.api import MixedOutcome, OutcomeLayout


def _skill(sid, spec_id, max_ticks):
    pl = {"spec_id": spec_id}
    if max_ticks is not None:
        pl["max_ticks"] = max_ticks
    return Skill(sid, 1, f"ctrl/{sid}", {"when": "always"}, {"after": "max_ticks"},
                 payload=pl)


# ---- candidates --------------------------------------------------------------
@case
def candidates_exactly_the_spec_commands():
    spec = ActionSpec.discrete("d", 5)
    cs = enumerate_candidates(spec, durations=(1, 3), max_candidates=16, max_horizon=4)
    cmds = sorted({c.command for c in cs})
    assert cmds == [0, 1, 2, 3, 4], cmds
    assert len(cs) == 10 and not cs.truncated and cs.n_admissible == 10
    assert {c.duration_ticks for c in cs} == {1, 3}


@case
def candidates_duration_and_caps():
    spec = ActionSpec.discrete("d", 3)
    cs = enumerate_candidates(spec, durations=(1, 9), max_horizon=4)
    assert ("duration=9", "exceeds max_horizon=4") in cs.rejected
    raises(lambda: enumerate_candidates(spec, max_candidates=MAX_CANDIDATES + 1),
           ContractError, "cap")
    raises(lambda: enumerate_candidates(spec, max_horizon=MAX_HORIZON + 1),
           ContractError, "horizon cap")
    raises(lambda: enumerate_candidates(spec, durations=(0,)), ContractError, "0 ticks")
    uc = ActionSpec.discrete("u", 2, duration_mode="until_complete")
    cs = enumerate_candidates(uc, durations=(1, 2))
    assert {c.duration_ticks for c in cs} == {1} and len(cs.rejected) == 1


@case
def candidates_truncation_needs_rng_and_redraws():
    spec = ActionSpec.discrete("d", 40)
    raises(lambda: enumerate_candidates(spec, max_candidates=8), ContractError,
           "fixed subset")
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        cs = enumerate_candidates(spec, max_candidates=8, rng=rng)
        assert cs.truncated and len(cs) == 8 and cs.n_admissible == 40
        seen |= {c.command for c in cs}
    assert len(seen) == 40, "every admissible candidate must remain reachable"


@case
def candidates_box_inside_bounds():
    spec = ActionSpec.box("b", [-1.0, 0.0], [1.0, 2.0], ("u", "v"))
    raises(lambda: enumerate_candidates(spec), ContractError, "box needs rng")
    cs = enumerate_candidates(spec, rng=np.random.default_rng(1), box_samples=6)
    assert len(cs) == 7
    for c in cs:
        assert np.all(c.command >= spec.low) and np.all(c.command <= spec.high)


@case
def candidates_skills_admitted_only_when_executable_and_bounded():
    spec = ActionSpec.discrete("d", 2)
    ok = _skill("chop", "d", 3)
    wrong = _skill("fly", "other", 3)
    unbounded = _skill("forever", "d", None)
    long_ = _skill("long", "d", 50)
    cs = enumerate_candidates(spec, skills=[ok, wrong, unbounded, long_], max_horizon=8)
    kinds = [c.kind for c in cs]
    assert kinds.count("skill") == 1 and cs.by_id("skill:chop@v1").duration_ticks == 3
    why = dict(cs.rejected)
    assert "other" in why["skill fly@v1"]
    assert "latch" in why["skill forever@v1"]
    assert "exceeds" in why["skill long@v1"]
    raises(lambda: enumerate_candidates(spec, skills=["chop"]), ContractError, "not Skill")


# ---- information ----------------------------------------------------------
@case
def mi_identities():
    p = np.array([0.5, 0.5])
    close(hypothesis_information(p, [[1, 0], [0, 1]])["info"], math.log(2), 1e-12,
          "perfect test of 2 hypotheses = log 2")
    close(hypothesis_information(p, [[0.5, 0.5], [0.5, 0.5]])["info"], 0.0, 1e-12,
          "identical rows")
    r = hypothesis_information(p, [[0.5, 0.5], [0.5, 0.5]])
    close(r["predictive_entropy"], math.log(2), 1e-12, "coin is maximally novel")
    hs = HypothesisSet(floor=0.0)
    for n in "abc":
        hs.add_hypothesis(n)
    assert hypothesis_information(hs, np.eye(3))["info"] <= math.log(3) + 1e-12
    raises(lambda: hypothesis_information(p, [[0.5, 0.6], [0.5, 0.5]]), ContractError, "rows")
    raises(lambda: hypothesis_information(p, np.eye(3)), ContractError, "shape")
    raises(lambda: hypothesis_information([0.7, 0.7], np.eye(2)), ContractError, "probs")
    capped = hypothesis_information(np.full(4, 0.25), np.eye(4), cap=0.5)
    assert capped["capped"] and capped["info"] == 0.5


@case
def bald_shapes_and_bounds():
    P = np.array([[[0.9, 0.1]], [[0.1, 0.9]], [[0.5, 0.5]]])     # (M=3, B=1, C=2)
    b = bald_categorical(P)
    assert b.shape == (1,) and 0 < b[0] <= math.log(2)
    close(bald_categorical(np.tile([[0.5, 0.5]], (4, 1))).item(), 0.0, 1e-12, "agree")
    g = bald_gaussian(np.array([[0.0], [0.0]]), np.array([[4.0], [4.0]]))
    close(g.item(), 0.0, 1e-12, "agreeing gaussians")
    raises(lambda: bald_gaussian(np.zeros((2, 1)), np.zeros((2, 1))), ContractError, "var 0")
    raises(lambda: bald_categorical(np.ones((1, 2)) / 2), ContractError, "M=1")


def _members(means, var=0.1):
    lay = OutcomeLayout(("a", "b"))
    return [MixedOutcome.gaussian_categorical(lay, np.array([m], float),
                                              np.full((1, 2), var)) for m in means]


@case
def ensemble_information_requires_relevance_and_three_members():
    ms = _members([[0, 0], [1, 0], [2, 0]])
    raises(lambda: ensemble_information(ms), ContractError, "no relevant outputs")
    raises(lambda: ensemble_information(ms[:2], cont_index=[0]), ContractError, "M<3")
    raises(lambda: ensemble_information(ms, cont_index=[5]), ContractError, "range")
    r = ensemble_information(ms, cont_index=[0])
    assert r["info"].shape == (1,) and r["info"][0] > 0 and not r["jackpot"][0]
    assert r["info"][0] <= DEFAULT_INFO_CAP


@case
def real_evidence_refusals():
    hs = HypothesisSet(floor=0.0)
    hs.add_hypothesis("a")
    hs.add_hypothesis("b")
    act = Action("env", "s", "ep1", 3, "d", 1, 1.0, 10.0, t_complete=11.0)
    good = Observation("env", "s", "ep1", 4, 4.0, 11.0, "x", "sensor", {"value": 1.0})
    for prov in ("imagined", "inferred", "evaluator"):
        bad = good.replace(provenance=prov)
        raises(lambda: update_from_real_outcome(hs, [0.0, -1.0], executed_action=act,
                                                evidence=[bad]),
               ImaginedEvidenceError, prov)
    raises(lambda: update_from_real_outcome(hs, [0.0, -1.0], executed_action=None,
                                            evidence=[good]), ContractError, "no action")
    raises(lambda: update_from_real_outcome(
        hs, [0.0, -1.0], executed_action=act.replace(t_complete=UNKNOWN),
        evidence=[good]), ContractError, "pending")
    raises(lambda: update_from_real_outcome(hs, [0.0, -1.0], executed_action=act,
                                            evidence=[good.replace(episode="ep2")]),
           ContractError, "other scope")
    raises(lambda: update_from_real_outcome(hs, [0.0, -1.0], executed_action=act,
                                            evidence=[good.replace(seq=3)]),
           ContractError, "precedes")
    assert np.allclose(hs.probs(), 0.5), "refused evidence must not move the posterior"
    kl = update_from_real_outcome(hs, [0.0, -1.0], executed_action=act, evidence=[good])
    assert kl > 0
    close(update_from_real_outcome(hs, None, executed_action=act, evidence=[good]),
          0.0, 0, "missing likelihood is a no-op")


# ---- selection terms ------------------------------------------------------
@case
def term_metadata_checked():
    t = default_terms()
    assert tuple(x.name for x in t) == TERM_NAMES
    for prop in ("channel", "occlusion", "idle", "recovery"):
        bad = tuple(dataclasses.replace(x, **{prop: ""}) if x.name == "risk" else x
                    for x in t)
        raises(lambda: check_terms(bad), TermMetadataError, f"missing {prop}")
        raises(lambda: ExperimentSelector(bad), TermMetadataError, "selector refuses")
    raises(lambda: check_terms(t[:4]), TermMetadataError, "term missing")
    raises(lambda: check_terms(tuple(dataclasses.replace(x, weight=-1.0) for x in t)),
           TermMetadataError, "negative weight")


@case
def selector_breakdown_and_refusals():
    spec = ActionSpec.discrete("d", 3)
    cs = enumerate_candidates(spec)
    sel = ExperimentSelector(epsilon=0.0, seed=0)
    vals = {c.candidate_id: {"info_gain": 0.1 * c.command, "usefulness": 0.0,
                             "task": 0.0, "effort": 1.0, "risk": UNKNOWN}
            for c in cs}
    s = sel.select(cs, vals)
    assert s.chosen.command == 2 and s.reason == "argmax"
    assert set(s.breakdown) == set(TERM_NAMES) and s.breakdown["risk"]["occluded"]
    close(s.total, 0.2 - 0.01, 1e-12, "total = sum of contributions")
    assert set(s.table) == {c.candidate_id for c in cs}
    v2 = {k: dict(v) for k, v in vals.items()}
    del v2[cs.candidates[0].candidate_id]["task"]
    raises(lambda: sel.select(cs, v2), ContractError, "partial scoring")
    v3 = {k: dict(v, effort=-1.0) for k, v in vals.items()}
    raises(lambda: sel.select(cs, v3), ContractError, "negative cost")
    v4 = {k: dict(v, info_gain=float("nan")) for k, v in vals.items()}
    raises(lambda: sel.select(cs, v4), ContractError, "nan")
    raises(lambda: sel.select([], {}), ContractError, "empty")
    occ = sel.score({"info_gain": 0.0, "usefulness": 0.0, "task": 0.0,
                     "effort": ABSENT, "risk": 0.0})[1]["effort"]
    assert occ["occluded"] and occ["value"] == 1.0, "unknown cost is never free"


@case
def selector_epsilon_keeps_every_candidate_reachable():
    spec = ActionSpec.discrete("d", 4)
    cs = enumerate_candidates(spec)
    sel = ExperimentSelector(epsilon=0.2, seed=3)
    vals = {c.candidate_id: {"info_gain": 10.0 if c.command == 0 else 0.0,
                             "usefulness": 0.0, "task": 0.0, "effort": 0.0,
                             "risk": 0.0} for c in cs}
    picks = [sel.select(cs, vals).chosen.command for _ in range(400)]
    assert set(picks) == {0, 1, 2, 3}
    between(picks.count(0) / 400.0, 0.75, 0.92, "argmax share")


@case
def cost_account_and_effort():
    spec = ActionSpec.discrete("d", 2)
    cs = enumerate_candidates(spec, durations=(1, 4), skills=[_skill("k", "d", 3)])
    c4 = [c for c in cs if c.duration_ticks == 4][0]
    sk = cs.by_id("skill:k@v1")
    close(effort_of(c4, tick_cost=0.5), 2.0, 0, "ticks x cost")
    close(effort_of(sk, skill_overhead=1.5), 4.5, 0, "skill overhead")
    sel = ExperimentSelector(epsilon=0.0)
    s = sel.select([c4], {c4.candidate_id: {"info_gain": 0.3, "usefulness": 0, "task": 0,
                                            "effort": effort_of(c4), "risk": 0}})
    acct = CostAccount()
    acct.charge(s, 4, 0.1)
    acct.charge(s, 4, 0.0)
    sm = acct.summary()
    assert sm["interactions"] == 2 and sm["ticks"] == 8 and sm["effort"] == 8.0
    close(sm["forecast_nats"], 0.6, 1e-12, "forecast kept apart")
    close(sm["realized_nats"], 0.1, 1e-12, "realized")
    raises(lambda: acct.charge(s, 1, -0.1), ContractError, "negative realized")


@case
def infra_sink_uses_ledger_and_signal_monitor():
    spec = ActionSpec.discrete("d", 2)
    cs = enumerate_candidates(spec)
    sink = InfraSink(monitor_window=50, monitor_min_obs=10)
    sel = ExperimentSelector(epsilon=0.0, sink=sink)
    vals = {c.candidate_id: {"info_gain": 0.0, "usefulness": 0.0, "task": 0.0,
                             "effort": 1.0, "risk": 0.0} for c in cs}
    for _ in range(20):
        sel.select(cs, vals)
    seg = sink.segment()
    sh = seg["statement"]["shares"]
    assert sh["selection.effort"] == 1.0 and set(sh) == {f"selection.{t}" for t in TERM_NAMES}, sh
    assert "selection.info_gain" in seg["degenerate_terms"], \
        "an info term pinned at 0 must show up as a flat signal"
    assert len(sel.log) == 20 and sel.log[0]["terms"]["effort"]["contribution"] == -0.01


# ---- dev probes / retention -----------------------------------------------
@case
def dev_probe_set_refuses_heldout():
    split = make_split("unit:sel", [f"e{i}" for i in range(20)], 0.25)
    ps = DevProbeSet(split)
    ps.add("p0", "unit:sel", split.dev[0])
    raises(lambda: ps.add("p1", "unit:sel", split.heldout[0]), HeldOutAccessError, "held")
    raises(lambda: ps.add("p2", "unit:sel", "nope"), ContractError, "neither side")
    raises(lambda: ps.add("p3", "other", split.dev[0]), ContractError, "scope")
    raises(lambda: ps.add("p0", "unit:sel", split.dev[1]), ContractError, "duplicate")
    assert len(ps) == 1


@case
def retention_tracker_hysteresis_and_recovery():
    raises(lambda: RetentionTracker(1.0, 0.5), ContractError, "hysteresis")
    raises(lambda: RetentionTracker(0.2, 0.8, window=1), ContractError, "window")
    raises(lambda: RetentionTracker(0.2, 0.8, noise_history=3), ContractError, "history")
    rt = RetentionTracker(known_below=0.2, forgotten_above=0.8, relearn_discount=0.5,
                          loop_flag=1, stable_evals=3, window=2, min_history=4)
    seq = [1.0] * 4 + [0.1] * 4 + [1.0] * 4 + [0.1] * 4     # learn, forget, relearn
    outs = [rt.observe(t, {"p": v}) for t, v in enumerate(seq)]
    close(outs[4]["progress"], 0.9, 1e-12, "first learning: full credit")
    assert rt.history[5]["flags"] == [] and outs[7]["flags"] == []
    close(outs[8]["progress"], -0.9, 1e-12, "forgetting a KNOWN probe is charged")
    close(outs[12]["progress"], 0.45, 1e-12, "relearning discounted")
    close(outs[12]["relearning_progress"], 0.45, 1e-12, "relearn share reported")
    assert outs[13]["flags"] == ["p"]
    need = rt.retention_needed("p")
    assert need == 2 * 3, need                     # known for 3 evals before it was forgotten
    t = 16
    for _ in range(need - 2):                      # retained -> flag clears
        o = rt.observe(t, {"p": 0.1})
        t += 1
    assert o["flags"] == [], "stable retention must clear the flag (no latch)"
    raises(lambda: rt.observe(t - 1, {"p": 0.1}), ContractError, "time must advance")
    raises(lambda: rt.reset_probe("p", ""), ContractError, "reason")
    rt.reset_probe("p", "PredictiveCUSUM alarm")
    assert rt.learned["p"] == 0 and rt.retention_needed("p") == 3


@case
def retention_tracker_single_excursions_are_noise():
    rt = RetentionTracker(known_below=0.2, forgotten_above=0.8, window=4)
    for t in range(10):
        rt.observe(t, {"p": 0.1})
    assert rt.state["p"] == "known"
    rt.observe(10, {"p": 1.0})                     # one bad evaluation
    rt.observe(11, {"p": 0.1})
    assert rt.state["p"] == "known" and rt.totals["forget_events"] == 0
    close(rt.totals["progress"], 0.9 - 0.9 + 0.9 * 0, 1e-12, "within a state: telescopes")
    assert rt.noise_floor("p") is not None and rt.noise_floor("q") is None


# ---- targeted information / the question ------------------------------------
@case
def targeted_information_identities():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(6))
    T = rng.dirichlet(np.ones(3), size=6)
    one = targeted_information(p, T, list(range(6)))           # Q = H
    close(one["info"], hypothesis_information(p, T)["info"], 1e-12, "Q = H -> I(H;O)")
    close(one["nuisance"], 0.0, 1e-12, "no nuisance when Q = H")
    two = targeted_information(p, T, ["a", "a", "b", "b", "c", "c"])
    assert two["info"] <= one["info"] + 1e-12
    close(two["info"] + two["nuisance"], two["joint"], 1e-9, "chain rule")
    single = targeted_information(p, T, ["x"] * 6)               # nothing to ask
    assert single["info"] == 0.0
    lamp = np.array([[0.999, 0.001], [0.001, 0.999]] * 3)       # rows differ only within
    assert targeted_information(np.full(6, 1 / 6), lamp, [0, 0, 1, 1, 2, 2])["info"] == 0.0
    raises(lambda: targeted_information(p, T, [0] * 5), ContractError, "one label each")
    raises(lambda: targeted_information(p, T, [None] * 6), ContractError, "label")


@case
def misspecification_monitor_bounds():
    raises(lambda: MisspecificationMonitor(alpha=0.0), ContractError, "alpha")
    raises(lambda: MisspecificationMonitor(lambdas=(0.0,)), ContractError, "lambdas")
    m = MisspecificationMonitor()
    sharp = np.array([[0.9, 0.1], [0.1, 0.9]])
    tv = np.full((2, 2), 0.5)
    for k in range(500):
        m.observe(["a", "b"], tv, k % 2)
    assert not m.flagged() and np.all(m.statistic() == 0), "noise is not misfit"
    for _ in range(200):
        m.observe(["a", "b"], sharp, 0)                        # a fits, b does not
    assert not m.flagged() and m.rejected().tolist() == [False, True]
    assert m.statistic().max() <= m.log_cap + 1e-12
    raises(lambda: m.observe(["a", "b"], sharp, 2), ContractError, "outcome")
    raises(lambda: m.observe(["a", "b"], np.ones((2, 2)), 0), ContractError, "distributions")
    raises(lambda: m.reset(""), ContractError, "reason")
    m.reset("change detector alarm")
    assert m.statistic().max() == 0 and m.resets[-1]["reason"] == "change detector alarm"


@case
def question_status_and_inference_hook():
    hs = HypothesisSet(floor=1e-3)
    for n in ("a", "b"):
        hs.add_hypothesis(n)
    q = ExperimentQuestion(hs, {"a": "A", "b": "B"})
    st = q.status()
    assert st["state"] == "open" and st["answer"] is None and not st["reliable"]
    assert st["inference_flag"] in (None, False), "inference flag read (if the set has one)"
    raises(lambda: ExperimentQuestion(hs, {"a": "A"}), ContractError, "no answer")
    raises(lambda: ExperimentQuestion(hs, {"a": "A", "b": "B"}, threshold=0.3),
           ContractError, "threshold")

    class Flagged(HypothesisSet):
        misspecified = True
    hf = Flagged(floor=1e-3)
    for n in ("a", "b"):
        hf.add_hypothesis(n)
    qf = ExperimentQuestion(hf, lambda n: n.upper())
    assert qf.misspecified() and qf.status()["state"] == "misspecified"
    assert qf.status()["request_revision"], "inference's flag is OR-ed in"
    assert bears_on_question(np.array([[0.9, 0.1], [0.1, 0.9]]), ["A", "B"])
    assert not bears_on_question(np.full((2, 2), 0.5), ["A", "B"])


@case
def selection_carries_question_status():
    spec = ActionSpec.discrete("d", 2)
    cs = enumerate_candidates(spec)
    sel = ExperimentSelector(epsilon=0.0)
    vals = {c.candidate_id: {"info_gain": 0.1, "usefulness": 0.0, "task": 0.0,
                             "effort": 1.0, "risk": 0.0} for c in cs}
    s = sel.select(cs, vals, status={"state": "misspecified", "misspecified": True,
                                     "reliable": False, "request_revision": True})
    assert s.misspecified and sel.log[-1]["question"]["misspecified"] is True
    s2 = sel.select(cs, vals)
    assert not s2.misspecified and "question" not in sel.log[-1]
    raises(lambda: sel.select(cs, vals, status=["bad"]), ContractError, "mapping")


@case
def cue_shift_tables_are_distributions():
    m = CueShiftHypotheses(4, 0.1)
    for cue in range(4):
        for cmd in range(5):
            T = m.outcome_table(cue, cmd)
            assert T.shape == (4, 2) and np.allclose(T.sum(1), 1)
    ml = CueShiftHypotheses(4, 0.1, lamp_bits=2)
    assert len(ml.names) == 16 and ml.answer(ml.names[5]) == "shift=1"
    for cmd in range(4 + 1 + 2):
        assert np.allclose(ml.outcome_table(0, cmd).sum(1), 1)
    raises(lambda: ml.outcome_table(0, 7), ContractError, "unknown action")
    assert np.allclose(m.outcome_table(0, 4), 0.5), "TV is the same coin for all"
    raises(lambda: CueShiftHypotheses(4, 0.6), ContractError, "slip")


if __name__ == "__main__":
    sys.exit(1 if run_all("foundation_selection_unit") else 0)
