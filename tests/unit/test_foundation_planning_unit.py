"""Unit cases for developmental_ai.foundation.planning (plan Stage 11).

Parts only: parameter validation, provenance refusals (imagined/evaluator
outcomes can never validate a model, a skill, a store item or a retention
measurement), the reliability window and its version reset, the
uncertainty cut and the truncation floor on a stub ensemble, the deadline
(deterministic fake clock: late chunks discarded, pre-emption, the reprobe
escape), CEM on a known model, set classifiers, Beta competence, JS
signatures, SMDP arithmetic, the knowledge store's status machine, the
retention reservoir, the read-only skill-bank bridge, and adapter
conformance of the point-mass fixture. Design claims (closed loop,
exploitation, remapping, calibration, resume) live in
tests/_foundation_planning_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_foundation_planning_unit.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, close, raises, run_all
from developmental_ai.foundation.contracts import (UNKNOWN, ActionSpec, ContractError,
                                                   Evidence, Mechanism, Observation)
from developmental_ai.foundation.experience.errors import ProvenanceError
from developmental_ai.foundation.mechanisms.api import EntityBatch, MixedOutcome, entity_layout
from developmental_ai.foundation.perception.versioning import RepresentationRegistry
from developmental_ai.foundation.planning import (
    BetaCompetence, BoundedPlanner, KnowledgeStore, LearnedSkill, PlanningController,
    ReliabilityMonitor, RetentionMonitor, SetClassifier, behaviour_signature,
    behavioural_divergence, find_duplicates, read_skill_bank, smdp_return, smdp_target)
from developmental_ai.foundation.planning.fixtures import (DT, PointMassAdapter,
                                                           PointMassMechanism, encode_discrete)


def _timed(measure, ok, what, attempts=3):
    """Wall-clock bound: re-run the measurement up to `attempts` times, pass
    if ANY attempt holds it (a scheduler stall on a loaded runner is not the
    code's fault; a code bug misses every time). Fake-clock asserts never use
    this. Returns (result, attempt)."""
    seen = []
    for k in range(1, attempts + 1):
        r = measure()
        if ok(r):
            if k > 1:
                print(f"  (timing: '{what}' held on attempt {k} of {attempts})")
            return r, k
        seen.append(r)
    raise AssertionError(f"{what}: bound missed on all {attempts} attempts: {seen}")


def one_state(pos=(0.0, 0.0), vel=(0.0, 0.0)):
    return EntityBatch(np.array(pos, float).reshape(1, 1, 2),
                       np.array(vel, float).reshape(1, 1, 2), np.ones((1, 1, 1)))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class StubEnsemble:
    """1 entity, 1 dim. Action 0: both members agree pos=-1. Action 1:
    members say -5 / +5 (mean 0 = better imagined reward, epistemic 25 on
    pos). Each rollout advances `clock` by `cost`."""

    def __init__(self, clock=None, cost=0.0):
        self.version, self.mechanism_id = 1, "stub"
        self.layout = entity_layout(1, 1, ("a", "b"))
        self.clock, self.cost = clock, cost

    def rollout(self, state, actions, dts, mode="mean"):
        if self.clock is not None:
            self.clock.t += self.cost
        H, C = actions.shape[:2]
        out = []
        for h in range(H):
            a = actions[h, :, 0, 0]
            p0 = np.where(a > 0.5, -5.0, -1.0)
            p1 = np.where(a > 0.5, 5.0, -1.0)
            mk = lambda p: MixedOutcome.gaussian_categorical(
                self.layout, np.stack([p, np.zeros(C)], 1), np.full((C, 2), 1e-3),
                np.tile([0.5, 0.5], (C, 1, 1)))
            out.append(MixedOutcome.mixture([mk(p0), mk(p1)]))
        return out


def stub_planner(**kw):
    clock = kw.pop("clock", None)
    m = StubEnsemble(clock, kw.pop("cost", 0.0))
    args = dict(n_entities=1, action_dim=1, encode=lambda c: np.asarray(c, float)[..., None, None],
                horizon=4, n_candidates=16, n_iters=2, deadline_s=10.0, seed=0)
    args.update(kw)
    if clock is not None:
        args["clock"] = clock
    return BoundedPlanner(m, lambda nc, a: nc[:, 0], ActionSpec.discrete("s", 2), **args)


def stub_state():
    return EntityBatch(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.ones((1, 1, 1)))


# ---- reliability -------------------------------------------------------------
@case
def reliability_refuses_imagined_and_evaluator():
    r = ReliabilityMonitor()
    for p in ("imagined", "evaluator"):
        raises(lambda: r.validate([0], [1], [0], provenance=p, model_version=1),
               ProvenanceError, p)
    assert r.total == 0


@case
def reliability_cold_then_valid_then_wrong():
    r = ReliabilityMonitor(window=10, min_count=3, max_z2=9.0)
    assert r.reliable()[0] is False and "cold" in r.reliable()[1]
    for _ in range(3):
        r.validate([0.0, 0.0], [1.0, 1.0], [0.5, -0.5], provenance="sensor", model_version=1)
    assert r.reliable()[0]
    r.validate([0.0, 0.0], [1e-6, 1e-6], [1.0, 1.0], provenance="sensor", model_version=1)
    ok, why = r.reliable()
    assert not ok and "overconfident" in why, why


@case
def reliability_window_resets_on_model_version():
    r = ReliabilityMonitor(window=5, min_count=2)
    for _ in range(5):
        r.validate([0], [1e-6], [1], provenance="sensor", model_version=1)
    assert not r.reliable()[0]
    r.validate([0], [1], [0.1], provenance="inferred", model_version=2)
    assert len(r.z2) == 1 and "cold" in r.reliable()[1]
    r.validate([0], [1], [0.1], provenance="sensor", model_version=2)
    assert r.reliable()[0], "a refitted model is judged on its own record"


@case
def reliability_nmse_and_params():
    raises(lambda: ReliabilityMonitor(max_nmse=1.0), ContractError, "nmse without scale")
    raises(lambda: ReliabilityMonitor(window=2, min_count=3), ContractError, "min>window")
    r = ReliabilityMonitor(window=4, min_count=2, max_nmse=0.1, scale=1.0)
    for _ in range(2):                                   # vague but wrong
        r.validate([0], [100.0], [3.0], provenance="sensor", model_version=1)
    ok, why = r.reliable()
    assert not ok and "nmse" in why
    s = r.state_dict()
    r2 = ReliabilityMonitor(window=4, min_count=2, max_nmse=0.1, scale=1.0)
    r2.load_state_dict(s)
    assert r2.reliable() == r.reliable() and list(r2.z2) == list(r.z2)


# ---- planner -------------------------------------------------------------------
@case
def planner_validates_construction():
    m = PointMassMechanism()
    raises(lambda: BoundedPlanner(m, None, ActionSpec.discrete("d", 5), n_entities=1,
                                  action_dim=2), ContractError, "discrete w/o encode")
    raises(lambda: BoundedPlanner(m, None, ActionSpec.multi_discrete("m", (2, 2)),
                                  n_entities=1, action_dim=2), ContractError, "multi")
    raises(lambda: BoundedPlanner(m, None, ActionSpec.box("b", [-1], [1], ("N",)),
                                  n_entities=1, action_dim=2), ContractError, "reshape")
    p = BoundedPlanner(m, lambda nc, a: np.zeros(3), ActionSpec.box("b", [-1, -1], [1, 1],
                       ("N", "N")), n_entities=1, action_dim=2)
    raises(lambda: p.plan(one_state(), DT), ContractError, "reward shape")
    two = EntityBatch(np.zeros((2, 1, 2)), np.zeros((2, 1, 2)), np.ones((2, 1, 1)))
    raises(lambda: p.plan(two, DT), ContractError, "B != 1")


@case
def uncertainty_cut_and_floor():
    # no cut: the planner exploits the members' disagreement (mean 0 > -1)
    r = stub_planner().plan(stub_state(), 0.1)
    assert r.ok and r.command == 1, "witness: without a cut the uncertain action wins"
    # cut at u > 1 with a declared floor below the confident reward
    r = stub_planner(epistemic_threshold=1.0, reward_floor=-2.0).plan(stub_state(), 0.1)
    assert r.ok and r.command == 0 and r.effective_horizon == 4, r
    # default floor = worst counted reward: uncertainty is never BETTER
    p = stub_planner(epistemic_threshold=1.0)
    seqs = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1]])
    v, e, _ = p._evaluate(stub_state(), 0.1, seqs)
    assert list(e) == [4, 0, 1] and v[1] <= v[0] + 1e-12 and v[2] <= v[0] + 1e-12, (v, e)
    # penalty alone (no cut) also flips the choice
    r = stub_planner(uncertainty_penalty=1.0).plan(stub_state(), 0.1)
    assert r.command == 0


@case
def all_uncertain_is_refused():
    class Always(StubEnsemble):
        def rollout(self, state, actions, dts, mode="mean"):
            return super().rollout(state, np.ones_like(actions), dts, mode)
    p = stub_planner(epistemic_threshold=1.0)
    p.model = Always()
    r = p.plan(stub_state(), 0.1)
    assert not r.ok and not r.timed_out and "epistemic" in r.reason
    assert r.imagined_value is UNKNOWN


@case
def deadline_discard_preempt_and_reprobe():
    clk = Clock()
    p = stub_planner(clock=clk, cost=0.03, deadline_s=0.02, reprobe_after=3, n_iters=1)
    r = p.plan(stub_state(), 0.1)                     # unknown cost: runs, late
    assert r.timed_out and abs(r.elapsed - 0.03) < 1e-9, "overrun bounded by one chunk"
    assert p.stats["late_chunks"] == 1
    for k in range(3):                                # estimate says it cannot fit
        r = p.plan(stub_state(), 0.1)
        assert r.timed_out and r.elapsed == 0.0
    assert p.consecutive_preempt == 3
    p.model.cost = 0.001                              # model became fast
    r = p.plan(stub_state(), 0.1)                     # the reprobe (escape path)
    assert p.stats["probes"] == 1 and r.ok, r
    r = p.plan(stub_state(), 0.1)
    assert r.ok and not r.timed_out, "planner resumes once the probe remeasures"


@case
def cem_box_steers_toward_goal():
    m = PointMassMechanism()
    m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
    goal = np.array([1.0, -1.0])
    rf = lambda nc, a: -np.linalg.norm(nc[:, :2] - goal, axis=1)
    p = BoundedPlanner(m, rf, ActionSpec.box("b", [-1, -1], [1, 1], ("N", "N")),
                       n_entities=1, action_dim=2, horizon=6, n_candidates=48, n_iters=3,
                       deadline_s=5.0, seed=0)
    r = p.plan(one_state(), DT)
    assert r.ok and r.command[0] > 0.3 and r.command[1] < -0.3, r.command
    assert r.model_versions == {"pm.linear": 1}


@case
def controller_logs_every_source():
    m = PointMassMechanism()
    m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
    p = BoundedPlanner(m, lambda nc, a: -np.abs(nc[:, 0]), ActionSpec.discrete("pm.push", 5),
                       n_entities=1, action_dim=2, encode=encode_discrete, horizon=3,
                       n_candidates=8, n_iters=1, deadline_s=5.0)
    rel = ReliabilityMonitor(window=3, min_count=1)
    c = PlanningController(p, lambda obs: 0, rel)
    d = c.decide(one_state(), DT, None)
    assert d.source == "fallback_unreliable" and d.command == 0
    c.observe(one_state(), 1, DT, one_state((0.04, 0), (0.4, 0)), provenance="sensor")
    assert c.decide(one_state((1, 0)), DT, None).source == "planner"
    raises(lambda: c.observe(one_state(), 1, DT, one_state(), provenance="imagined"),
           ProvenanceError, "imagined observe")
    raises(lambda: PlanningController(p, None, rel), ContractError, "non-callable fallback")
    c2 = PlanningController(p, lambda o: 7, ReliabilityMonitor())
    raises(lambda: c2.decide(one_state(), DT, None), ContractError, "out-of-spec fallback")
    assert sum(c.counts.values()) == len(c.log) == 2


@case
def planner_runs_on_bootstrap_ensemble():
    from developmental_ai.foundation.inference import BootstrapEnsemble
    from developmental_ai.foundation.planning.fixtures import transitions
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        s = one_state(rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2))
        f = rng.uniform(-1, 1, 2)
        v2 = 0.8 * s.vel[0, 0] + 4 * f * DT + rng.normal(0, 0.01, 2)
        rows.append((s, f.reshape(1, 1, 2), DT, one_state(s.pos[0, 0] + v2 * DT, v2)))
    ens = BootstrapEnsemble(lambda i: PointMassMechanism(f"pm{i}"), k=3, seed=0)
    ens.fit(transitions(rows))
    p = BoundedPlanner(ens, lambda nc, a: -np.linalg.norm(nc[:, :2] - 1.0, axis=1),
                       ActionSpec.box("b", [-1, -1], [1, 1], ("N", "N")), n_entities=1,
                       action_dim=2, horizon=4, n_candidates=16, n_iters=2,
                       epistemic_threshold=1.0, deadline_s=5.0)
    r = p.plan(one_state(), DT)
    assert r.ok and r.effective_horizon == 4 and set(r.model_versions) >= {"pm0", "pm1", "pm2"}
    p.threshold = 1e-12                                 # members never agree that well
    r = p.plan(one_state(), DT)
    assert not r.ok and "epistemic" in r.reason


# ---- skills ----------------------------------------------------------------------
@case
def set_classifier_unknown_linear_quadratic():
    rng = np.random.default_rng(0)
    c = SetClassifier(2, min_samples=10)
    c.add(rng.normal(size=(5, 2)), [0, 1, 0, 1, 0])
    c.fit()
    assert c.prob(np.zeros(2)) is UNKNOWN
    X = rng.uniform(-1, 1, (300, 2))
    c = SetClassifier(2)
    c.add(X, (X[:, 0] < 0.2).astype(int))
    c.fit()
    Xt = rng.uniform(-1, 1, (300, 2))
    assert np.mean((c.prob(Xt) > 0.5) == (Xt[:, 0] < 0.2)) > 0.95
    q = SetClassifier(2, "quadratic")
    q.add(X, (np.linalg.norm(X, axis=1) < 0.6).astype(int))
    q.fit()
    assert np.mean((q.prob(Xt) > 0.5) == (np.linalg.norm(Xt, axis=1) < 0.6)) > 0.93
    one = SetClassifier(2, min_samples=3)
    one.add(X[:4], [1, 1, 1, 1])
    one.fit()
    close(one.prob(X[:1])[0], 5 / 6, 1e-12, "one-class smoothed rate")
    r = SetClassifier.from_state_dict(q.state_dict())
    assert np.allclose(r.prob(Xt), q.prob(Xt))
    raises(lambda: c.add(np.zeros((2, 3)), [0, 1]), ContractError, "dim")
    raises(lambda: c.add(np.zeros((2, 2)), [0, 2]), ContractError, "label")


@case
def beta_competence_real_only():
    c = BetaCompetence()
    for p in ("imagined", "evaluator"):
        raises(lambda: c.record(True, provenance=p), ProvenanceError, p)
    assert c.n == 0 and c.rejected == 2
    for s in (True, True, False):
        c.record(s, provenance="sensor", evidence_id=f"e{c.n}")
    close(c.mean(), 3 / 5, 1e-12)
    lo, hi = c.interval(0.9)
    assert lo < c.mean() < hi
    raises(lambda: c.record(1, provenance="sensor"), ContractError, "non-bool")
    ob = Observation("env", "s", "ep", 0, 0.0, 0.0, "pos", "sensor", {"value": 1.0})
    c.record_evidence(Evidence("ev1", (ob,), (), (), "valid", "sensor"), True)
    assert c.evidence_ids[-1] == "ev1" and c.successes == 3
    bad = Evidence("ev2", (ob,), (), (), "invalid", "sensor", "reset splice")
    raises(lambda: c.record_evidence(bad, True), ContractError, "invalid evidence")


@case
def learned_skill_escapes_and_record():
    raises(lambda: LearnedSkill("s", "c", state_dim=2, representation="r",
                                representation_version="1.0.0", explore_floor=0.0),
           ContractError, "zero explore floor is a latch")
    s = LearnedSkill("s", "c", state_dim=1, representation="r",
                     representation_version="1.0.0", explore_floor=0.1, max_duration=5,
                     min_samples=4, features="linear")
    rng = np.random.default_rng(0)
    assert s.can_initiate(np.zeros(1), rng) == (True, "initiation unknown: exploring")
    for x in np.linspace(-1, 1, 40):
        s.record_attempt([x], [[x], [0.0]], bool(x < 0), provenance="sensor")
    raises(lambda: s.record_attempt([0], [[0]], True, provenance="imagined"),
           ProvenanceError, "imagined attempt")
    s.refit()
    outside = [s.can_initiate(np.array([0.9]), rng)[0] for _ in range(2000)]
    assert 0.07 < np.mean(outside) < 0.13, np.mean(outside)
    assert s.can_initiate(np.array([-0.9]), rng)[0]
    assert s.should_terminate(np.array([5.0]), 5)[0], "timeout always terminates"
    rec = s.to_record()
    assert rec.skill_id == "s" and rec.version == 1 and rec.payload["competence"]["n"] == 40
    close(s.expected_discount(0.9), 0.81, 1e-12, "two-step durations")
    s.new_version("c2", "controller replaced")
    assert s.version == 2 and s.competence.n == 0 and s.signature is None
    r2 = LearnedSkill.from_state_dict(s.state_dict())
    assert r2.to_record() == s.to_record()


@case
def signatures_are_behaviour_not_names():
    P = np.eye(3)
    a = behaviour_signature(P, probs_fn=lambda s: np.array([0.7, 0.3]))
    b = behaviour_signature(P, probs_fn=lambda s: np.array([0.7, 0.3]))
    c = behaviour_signature(P, probs_fn=lambda s: np.array([0.0, 1.0]))
    d = behaviour_signature(P, probs_fn=lambda s: np.array([1.0, 0.0]))
    close(behavioural_divergence(a, b), 0.0, 1e-12)
    close(behavioural_divergence(c, d), 1.0, 1e-12, "disjoint = 1 bit")
    other = behaviour_signature(np.eye(3) * 2, probs_fn=lambda s: np.array([0.7, 0.3]))
    raises(lambda: behavioural_divergence(a, other), ContractError, "probe sets differ")
    dup = find_duplicates({"chop": a, "reach": b, "noise": c})
    assert [(f.a, f.b) for f in dup] == [("chop", "reach")]
    assert dup[0].verdict == "indistinguishable_on_probes", "bare probes: no duplicate verdict"
    k = behaviour_signature(P, command_fn=lambda s: s[:2], scale=2.0)
    assert behavioural_divergence(k, k) == 0.0
    raises(lambda: behaviour_signature(P, probs_fn=lambda s: np.array([0.5, 0.6])),
           ContractError, "not a distribution")


@case
def smdp_arithmetic():
    g, disc = smdp_return([1, 1, 1], 0.5)
    close(g, 1.75, 1e-12)
    close(disc, 0.125, 1e-12)
    close(smdp_target([1, 1, 1], 0.5, 8.0, False), 2.75, 1e-12)
    close(smdp_target([1, 1, 1], 0.5, 8.0, True), 1.75, 1e-12)
    raises(lambda: smdp_return([], 0.9), ContractError, "empty option")


# ---- consolidation ------------------------------------------------------------------
def _mech(version=1):
    m = PointMassMechanism("pm.k")
    m.set_params([0.8, 0.8], np.eye(2), [1e-3] * 2, [1e-5] * 2)
    m.version = version
    return m.to_record()


@case
def knowledge_store_status_machine():
    reg = RepresentationRegistry()
    reg.declare("pm.state", "1.0.0")
    ks = KnowledgeStore(reg, promote_min_n=3, promote_min_rate=0.6)
    ks.add_candidate("mechanism", "pm.k", _mech(), "pm.state")
    raises(lambda: ks.add_candidate("mechanism", "pm.k", _mech(), "pm.state"),
           ContractError, "duplicate id")
    raises(lambda: ks.record_validation("pm.k", True, provenance="imagined"),
           ProvenanceError, "imagined validation")
    assert not ks.promote("pm.k")[0]
    for ok in (True, True, False, True):
        ks.record_validation("pm.k", ok, provenance="sensor")
    assert ks.promote("pm.k")[0] and ks.usable("pm.k")[0]
    reg.bump("pm.state", "1.1.0", "added a channel")
    assert ks.usable("pm.k")[0], "a minor bump keeps knowledge valid"
    reg.bump("pm.state", "2.0.0", "pos now in cm")
    assert ks.usable("pm.k") == (False, "needs_revalidation")
    raises(lambda: ks.revalidate("pm.k", True, provenance="sensor",
                                 new_record=_mech().replace(mechanism_id="other")),
           ContractError, "identity change")
    assert ks.revalidate("pm.k", True, provenance="sensor", new_record=_mech(2)) == \
        "consolidated"
    assert ks.usable("pm.k")[0] and ks.record("pm.k").version == 2
    ks.on_dynamics_change(["pm.k"], "controls remapped")
    assert ks.revalidate("pm.k", False, provenance="sensor") == "candidate"
    assert ks.items["pm.k"]["validation"].n == 0


@case
def retention_reservoir_and_forgetting():
    r = RetentionMonitor(max_probes=4, seed=0)
    for i in range(50):
        r.add_probe("s", [i, 0])
    assert r.probe_states("s").shape == (4, 2) and r.seen["s"] == 50
    raises(lambda: r.measure("s", 0, [True], provenance="imagined"), ProvenanceError, "imag")
    r.measure("s", 0, [True] * 20, provenance="sensor")
    r.measure("s", 1, [True] * 18 + [False] * 2, provenance="sensor")
    assert not r.forgetting("s")["significant"]
    r.measure("s", 2, [True] * 3 + [False] * 17, provenance="sensor")
    f = r.forgetting("s")
    assert f["significant"] and f["drop"] > 0.8


# ---- bridge -------------------------------------------------------------------------
@case
def bridge_is_read_only_and_refuses_live_bank():
    raises(lambda: read_skill_bank("/tmp/x/skill_bank_mc_curiosity"), ContractError, "live")
    with tempfile.TemporaryDirectory() as d:
        assert read_skill_bank(os.path.join(d, "missing")) == []
        assert not os.path.exists(os.path.join(d, "missing")), "bridge created a dir"
        rows = [{"skill_id": "a1", "name": "chop", "asked_log": [1, 0, 1],
                 "success_rate": 0.99, "preconditions": {"tree_visible": True}},
                {"skill_id": "b2", "name": "chop"}]
        with open(os.path.join(d, "registry.json"), "w") as f:
            json.dump({"skills": rows}, f)
        before = sorted(os.listdir(d)), os.path.getmtime(os.path.join(d, "registry.json"))
        sk = read_skill_bank(d)
        assert (sorted(os.listdir(d)), os.path.getmtime(os.path.join(d, "registry.json"))) \
            == before
        assert [s.skill_id for s in sk] == ["a1", "b2"], "same name, two skills"
        lc = sk[0].payload["legacy_competence"]
        assert (lc["a"], lc["b"]) == (3.0, 2.0), "asked_log, not success_rate"
        assert sk[0].initiation["kind"] == "legacy_preconditions"
        assert sk[1].initiation == {"kind": "unlearned"}


# ---- fixture ------------------------------------------------------------------------
@case
def point_mass_adapter_conforms():
    from developmental_ai.foundation.adapters.protocol import check_adapter_conformance
    from developmental_ai.foundation.transfer import ControlRemap
    for ctl in ("discrete", "box"):
        check_adapter_conformance(PointMassAdapter(ctl, episode_len=10), n_steps=25)
    check_adapter_conformance(ControlRemap(PointMassAdapter("discrete", episode_len=10),
                                           [0, 3, 4, 2, 1]), n_steps=25)
    m = PointMassMechanism()
    raises(lambda: m.to_record(), ContractError, "unfitted record")
    m.set_params([0.8, 0.8], np.eye(2), [1e-3] * 2, [1e-5] * 2)
    rec = m.to_record()
    assert isinstance(rec, Mechanism)
    m2 = PointMassMechanism.from_record(rec)
    assert m2.version == m.version and m2.to_record() == rec



# ---- verifier findings B3 / F7 / F8 (experiments/foundation_ab/stage9_11) ------------
class SleepyStub(StubEnsemble):
    """StubEnsemble whose k-th rollout sleeps `sleeps[k]` REAL seconds."""

    def __init__(self, sleeps):
        super().__init__()
        self.sleeps, self.calls = list(sleeps), 0

    def rollout(self, state, actions, dts, mode="mean"):
        import time as _t
        k = self.calls
        self.calls += 1
        if k < len(self.sleeps) and self.sleeps[k] > 0:
            _t.sleep(self.sleeps[k])
        return super().rollout(state, actions, dts, mode)


def sleepy_planner(sleeps, **kw):
    m = SleepyStub(sleeps)
    args = dict(n_entities=1, action_dim=1, encode=lambda c: np.asarray(c, float)[..., None, None],
                horizon=4, n_candidates=16, n_iters=2, deadline_s=0.02, seed=0)
    args.update(kw)
    return BoundedPlanner(m, lambda nc, a: nc[:, 0], ActionSpec.discrete("s", 2), **args)


@case
def deadline_holds_through_one_slow_model_call():
    """B3: a single 60 ms model call under a 20 ms deadline used to return
    after 60 ms (the chunk was synchronous). The wait is now bounded."""
    import time as _t

    def measure():
        p = sleepy_planner([0.06])
        t0 = _t.perf_counter()
        r = p.plan(stub_state(), 0.1)
        el = _t.perf_counter() - t0
        t0 = _t.perf_counter()                  # the abandoned call is still running:
        r2 = p.plan(stub_state(), 0.1)          # the model is busy -> no second call
        el2 = _t.perf_counter() - t0
        assert p.wait_idle(1.0), "abandoned call never finished"
        return p, r, el, r2, el2
    # bound with CI slack; the synchronous chunk took the full 60 ms here
    (p, r, el, r2, el2), _ = _timed(
        measure, lambda x: x[2] <= 0.035 and x[4] <= 0.035 and x[1].timed_out and
        x[3].timed_out and x[0].stats["late_chunks"] == 1,
        "one slow call: both plans <= 20 + 15 ms (planner, r, el, r2, el2)")
    assert not r.ok
    for _ in range(p.reprobe_after + 1):        # reachable escape: the planner returns
        r3 = p.plan(stub_state(), 0.1)
        if r3.ok:
            break
    assert r3.ok, "planner never recovered once the model was fast again"


@case
def deadline_returns_best_so_far():
    """First CEM iteration is fast, the second call stalls: the iteration-1
    plan is returned ON TIME instead of after the stall."""
    import time as _t

    def measure():
        p = sleepy_planner([0.0, 0.08], deadline_s=0.03)
        t0 = _t.perf_counter()
        r = p.plan(stub_state(), 0.1)
        el = _t.perf_counter() - t0
        p.wait_idle(1.0)
        return r, el
    _timed(measure, lambda x: x[1] <= 0.045 and x[0].ok and x[0].iterations == 1 and
           "best so far" in x[0].reason, "iteration-1 plan on time (synchronous: ~80 ms)")


class HangStub(StubEnsemble):
    """rollout() blocks on `release` for the call indices in `hang` (all
    calls when hang == "all") — a model call that does not return."""

    def __init__(self, hang):
        super().__init__()
        import threading
        self.hang, self.calls, self.release = hang, 0, threading.Event()

    def rollout(self, state, actions, dts, mode="mean"):
        k = self.calls
        self.calls += 1
        if self.hang == "all" or k in self.hang:
            self.release.wait(10.0)
        return super().rollout(state, actions, dts, mode)


def _model_threads():
    import threading
    return sum(1 for t in threading.enumerate()
               if t.name == "foundation-planner-model-call" and t.is_alive())


@case
def hung_model_call_is_replaced_and_planner_recovers():
    """Known limit closed 2026-10-03 (CLAUDE.md §4.1 guard-becomes-latch):
    one model call that never returned left the planner unavailable FOREVER
    (fallback permanently in control). The watchdog poisons the hung
    worker after hang_after_s, spawns a fresh one and re-probes."""
    import time as _t

    def measure():
        m = HangStub({0})
        p = BoundedPlanner(m, lambda nc, a: nc[:, 0], ActionSpec.discrete("s", 2),
                           n_entities=1, action_dim=1,
                           encode=lambda c: np.asarray(c, float)[..., None, None],
                           horizon=4, n_candidates=16, n_iters=2, deadline_s=0.02, seed=0,
                           hang_after_s=0.1, respawn_backoff_s=0.05)
        try:
            t0 = _t.perf_counter()
            r = first = p.plan(stub_state(), 0.1)
            while not r.ok and _t.perf_counter() - t0 < 1.0:
                _t.sleep(0.005)
                r = p.plan(stub_state(), 0.1)
            return (p, first.timed_out, r.ok, _t.perf_counter() - t0, dict(p.stats),
                    p.live_poisoned())
        finally:
            m.release.set()
    # recovery within hang_after (0.1 s) + 0.2 s; one hang -> exactly one respawn
    (p, *_), _ = _timed(measure, lambda x: x[1] and x[2] and x[3] < 0.3 and
                        x[4]["hung_calls"] == 1 and x[4]["respawns"] == 1 and
                        x[4]["probes"] >= 1 and x[5] == 1,
                        "planner regains control after one hung call (latch before the fix)")
    t1 = _t.perf_counter()
    while p.live_poisoned() and _t.perf_counter() - t1 < 1.0:
        _t.sleep(0.002)
    assert p.live_poisoned() == 0, "poisoned worker's slot not freed after its call returned"
    raises(lambda: stub_planner(hang_after_s=0), ContractError, "hang_after_s > 0")
    raises(lambda: stub_planner(max_respawns=-1), ContractError, "max_respawns >= 0")


@case
def always_hung_model_bounded_respawns():
    """A model on which EVERY call hangs: respawns stop at max_respawns
    live poisoned workers (no thread explosion), every plan times out (the
    fallback keeps control), and the moment calls return again the planner
    is back — the cap is not a latch."""
    import time as _t

    def measure():
        base = _model_threads()
        m = HangStub("all")
        p = BoundedPlanner(m, lambda nc, a: nc[:, 0], ActionSpec.discrete("s", 2),
                           n_entities=1, action_dim=1,
                           encode=lambda c: np.asarray(c, float)[..., None, None],
                           horizon=4, n_candidates=16, n_iters=2, deadline_s=0.01, seed=0,
                           hang_after_s=0.04, max_respawns=2, respawn_backoff_s=0.02)
        t0, res, peak = _t.perf_counter(), [], 0
        while _t.perf_counter() - t0 < 0.6:
            res.append(p.plan(stub_state(), 0.1))
            peak = max(peak, _model_threads() - base)
            _t.sleep(0.005)
        out = (p, m, all(r.timed_out and not r.ok for r in res), dict(p.stats),
               p.live_poisoned(), peak)
        if not (out[3]["respawns"] == 2 and out[4] == 2):
            m.release.set()                 # failed attempt: let its threads go
        return out
    (p, m, *_), _ = _timed(measure, lambda x: x[2] and x[3]["respawns"] == 2 and x[4] == 2
                           and x[3]["respawn_deferred"] > 0 and x[5] <= 1 + 2,
                           "always hangs: no plan, 2 respawns, <= 1 + max_respawns threads")
    m.release.set()
    t1, r = _t.perf_counter(), None
    while _t.perf_counter() - t1 < 1.0:
        r = p.plan(stub_state(), 0.1)
        if r.ok:
            break
        _t.sleep(0.005)
    assert r is not None and r.ok, "planner did not return once the model answered again"


@case
def region_memory_marks_failures_and_reopens():
    """F7: the global window forgot where the model failed. Region memory
    keeps it — and every way back is reachable (CLAUDE.md §4.1)."""
    rng = np.random.default_rng(0)
    A, B = np.array([3.0, 3.0]), np.array([0.0, 0.0])

    def feed(r, centre, fail, n, v=1):
        for _ in range(n):
            x = centre + 0.05 * rng.normal(size=2)
            r.validate([0.0], [1.0], [10.0 if fail else 0.1], provenance="sensor",
                       model_version=v, where=x)
    r = ReliabilityMonitor(window=5, min_count=2, max_z2=9.0, region_half_life=200,
                           region_stale_weight=0.25)
    feed(r, B, False, 30)
    feed(r, A, True, 5)
    feed(r, B, False, 10)                       # the global window refills ...
    assert r.reliable()[0], "global gate reopened (window)"
    assert list(r.region_unreliable(np.stack([A, B]))) == [True, False], \
        "... but the region where the model failed is remembered"
    ok, why = r.reliable_at(A)
    assert not ok and "region" in why, why
    # (a) the CURRENT model validated there again reopens it
    feed(r, A, False, 20)
    assert not r.region_unreliable(A[None])[0], "passes at A did not reopen it"
    # (b) a refit (version bump) discounts the old model's failures (x0.25
    #     here); the new model's own passes there reopen it
    feed(r, A, True, 5)
    assert r.region_unreliable(A[None])[0]
    feed(r, B, False, 3, v=2)
    assert r.region_unreliable(A[None])[0], "stale weight 0.25: still closed after refit"
    feed(r, A, False, 4, v=2)
    assert not r.region_unreliable(A[None])[0], "refit + validation did not reopen"
    rd = ReliabilityMonitor(window=5, min_count=2, max_z2=9.0)     # default stale 0
    feed(rd, B, False, 10)
    feed(rd, A, True, 5)
    assert rd.region_unreliable(A[None])[0]
    feed(rd, B, False, 1, v=2)
    assert not rd.region_unreliable(A[None])[0], "default: refit judged on its own record"
    feed(rd, A, True, 2, v=2)
    assert rd.region_unreliable(A[None])[0], "... and re-marked if it still fails there"
    # (c) with no visit at all, failures decay to UNKNOWN (deferred to the global gate)
    feed(r, A, True, 5, v=2)
    assert r.region_unreliable(A[None])[0]
    feed(r, B, False, 2000, v=2)
    assert not r.region_unreliable(A[None])[0], "decay never reopened the region"
    assert len(r.regions.fails) + len(r.regions.passes) <= \
        r.regions.max_fails + r.regions.max_passes, "memory unbounded"
    # disabled regions: the plain global gate
    r0 = ReliabilityMonitor(regions=False)
    feed(r0, A, True, 3)
    assert not r0.region_unreliable(A[None])[0]
    raises(lambda: r.validate([0], [1], [0], provenance="imagined", model_version=2,
                              where=A), ProvenanceError, "imagined region record")
    raises(lambda: r.validate([0], [1], [0], provenance="sensor", model_version=2,
                              where=[np.nan, 0]), ContractError, "non-finite where")
    r2 = ReliabilityMonitor(window=5, min_count=2, max_z2=9.0, region_half_life=200,
                            region_stale_weight=0.25)
    r2.load_state_dict(json.loads(json.dumps(r.state_dict(), default=lambda o: o.tolist())))
    X = rng.normal(size=(20, 2)) * 3
    assert np.array_equal(r2.region_unreliable(X), r.region_unreliable(X)), "round trip"


@case
def planner_cuts_rollouts_through_failed_regions():
    m = PointMassMechanism()
    m.set_params([0.8, 0.8], 4 * np.eye(2), [1e-4] * 2, [1e-6] * 2)
    goal = np.array([1.0, 0.0])
    rf = lambda nc, a: -np.linalg.norm(nc[:, :2] - goal, axis=1)
    p = BoundedPlanner(m, rf, ActionSpec.discrete("pm.push", 5), n_entities=1,
                       action_dim=2, encode=encode_discrete, horizon=6, n_candidates=32,
                       n_iters=2, deadline_s=5.0, seed=0)
    free = p.plan(one_state(), DT)
    assert free.ok and free.effective_horizon == 6 and free.command == 1   # +x
    bad = lambda X: np.asarray(X)[:, 0] > 0.02      # failed region: x > 0.02
    r = p.plan(one_state(), DT, region_check=bad)
    assert r.ok and r.command != 1 and p.stats["region_cut"] > 0, r   # does not enter
    refused = p.plan(one_state((0.5, 0)), DT, region_check=bad)
    assert not refused.ok and "region" in refused.reason
    rel = ReliabilityMonitor(window=3, min_count=1, max_z2=9.0)
    c = PlanningController(p, lambda o: 0, rel)
    for _ in range(3):                              # the model is WRONG at (2, 0)
        c.observe(one_state((2, 0)), 1, DT, one_state((0, 0)), provenance="sensor")
    for _ in range(3):                              # and right at the origin
        c.observe(one_state(), 0, DT, one_state(), provenance="sensor")
    assert rel.reliable()[0]
    d = c.decide(one_state((2, 0)), DT, None)
    assert d.source == "fallback_unreliable" and "region" in d.reason, d
    assert c.decide(one_state(), DT, None).source == "planner"


@case
def duplicates_require_evidence_coverage():
    """F8: identical on the probes is not identical. A duplicate verdict
    needs a probe set that covers BOTH skills' evidenced states."""
    from developmental_ai.foundation.planning import build_probe_set
    perm = np.array([1, 2, 0])

    def pol(offprobe=False):
        def f(s):
            z = np.array([s[0], -s[1], 0.3 * s[0] * s[1]])
            if offprobe and np.max(np.abs(s)) > 3.0:
                z = z[perm]
            e = np.exp(z - z.max())
            return e / e.sum()
        return f
    rng = np.random.default_rng(1)
    probes = rng.normal(size=(32, 2)) * 0.8          # never reaches |s| > 3
    a, b, c = pol(), pol(True), pol()
    bare = find_duplicates({"x": behaviour_signature(probes, probs_fn=a),
                            "y": behaviour_signature(probes, probs_fn=b)})
    assert len(bare) == 1 and bare[0].verdict == "indistinguishable_on_probes", bare
    assert "indistinguishable on 32 probes" in bare[0].statement, bare[0].statement
    ev = {k: rng.normal(size=(40, 2)) * 2.5 for k in ("x", "y", "z")}
    ps = build_probe_set(ev, shared=probes)
    sig = lambda f: behaviour_signature(ps, probs_fn=f)
    found = find_duplicates({"x": sig(a), "y": sig(b), "z": sig(c)}, probe_set=ps)
    assert [(f.a, f.b, f.verdict) for f in found] == [("x", "z", "duplicate")], found
    assert min(found[0].coverage.values()) >= 0.9
    thin = build_probe_set({"x": ev["x"][:3], "z": ev["z"]}, shared=probes)
    f2 = find_duplicates({"x": behaviour_signature(thin, probs_fn=a),
                          "z": behaviour_signature(thin, probs_fn=c)}, probe_set=thin)
    assert f2[0].verdict == "indistinguishable_on_probes", "3 evidenced states are not coverage"
    sparse = build_probe_set({"x": ev["x"], "z": ev["z"]}, max_per_skill=2)
    f3 = find_duplicates({"x": behaviour_signature(sparse, probs_fn=a),
                          "z": behaviour_signature(sparse, probs_fn=c)}, probe_set=sparse)
    assert f3[0].verdict == "indistinguishable_on_probes" and min(f3[0].coverage.values()) < 0.9
    raises(lambda: find_duplicates({"x": behaviour_signature(probes, probs_fn=a),
                                    "z": behaviour_signature(probes, probs_fn=c)},
                                   probe_set=ps), ContractError, "probe set mismatch")

@case
def graded_verdict_near_duplicate():
    """E11e: a tempered copy (same argmax everywhere) was 'distinct'. The
    verdict is now graded; near_duplicate needs modal agreement on probes
    resolvable by n executions, and coverage like duplicate."""
    from developmental_ai.foundation.planning import VERDICTS, build_probe_set, modal_agreement

    def pol(temp=1.0, flip=False):
        def f(s):
            z = np.array([2 * s[0], -2 * s[1], 0.5]) * temp
            if flip:
                z = z[[1, 0, 2]]
            e = np.exp(z - z.max())
            return e / e.sum()
        return f
    rng = np.random.default_rng(3)
    ev = {k: rng.normal(size=(40, 2)) * 2.0 for k in ("a", "b", "c", "d")}
    ps = build_probe_set(ev)
    sig = {"a": behaviour_signature(ps, probs_fn=pol()),
           "b": behaviour_signature(ps, probs_fn=pol(3.0)),        # tempered copy
           "c": behaviour_signature(ps, probs_fn=pol()),            # byte copy
           "d": behaviour_signature(ps, probs_fn=pol(flip=True))}   # other behaviour
    found = {(f.a, f.b): f for f in find_duplicates(sig, probe_set=ps, include_distinct=True)}
    assert len(found) == 6 and all(f.verdict in VERDICTS for f in found.values())
    assert found[("a", "c")].verdict == "duplicate"
    assert found[("a", "b")].verdict == "near_duplicate", found[("a", "b")].statement
    assert found[("a", "b")].divergence > 1e-3, "tempered copy differs in distribution"
    assert found[("a", "d")].verdict == "distinct" and found[("b", "d")].verdict == "distinct"
    m = modal_agreement(sig["a"], sig["b"])
    assert m["disagree"] == 0 and m["resolved"] >= 8 and m["exec_divergence"] == 0.0
    # positives without coverage are only indistinguishable_on_probes; default drops distinct
    bare = rng.normal(size=(32, 2))
    bs = {k: behaviour_signature(bare, probs_fn=f) for k, f in
          (("a", pol()), ("b", pol(3.0)), ("d", pol(flip=True)))}
    fb = find_duplicates(bs)
    assert [(f.a, f.b, f.verdict) for f in fb] == [("a", "b", "indistinguishable_on_probes")]
    # one probe where both modes are resolved and differ -> distinct
    flat = behaviour_signature(np.array([[0.0, 0.0]]), probs_fn=lambda s: np.ones(3) / 3)
    assert modal_agreement(flat, flat)["exec_divergence"] is UNKNOWN, "nothing resolved"
    raises(lambda: modal_agreement(sig["a"], bs["a"]), ContractError, "probe mismatch")


if __name__ == "__main__":
    sys.exit(1 if run_all("foundation-planning-unit") else 0)
