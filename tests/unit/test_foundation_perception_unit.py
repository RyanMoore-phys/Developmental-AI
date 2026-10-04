"""Unit cases for foundation.perception (plan Stage 8) — the PARTS.

tests/_foundation_perception_smoke.py argues the design claims (occlusion,
swaps, ego-motion, scale, downstream gate, families, revision). This file
checks each piece in isolation: validation raises the RIGHT error class,
track-state transitions happen on the documented frame, identities are
scoped and never reused, privileged inputs are refused at every door, the
oracle module is imported by nothing else in the package, and the version
registry's bookkeeping is exact.

Run: PYTHONPATH=. python tests/unit/test_foundation_perception_unit.py
"""
import os
import re
import sys

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, run_all, close, raises
from developmental_ai.foundation.contracts import (EntityId, ObsRef, Scope,
                                                   UNKNOWN, from_json, to_json)
from developmental_ai.foundation.geometry import M, Quantity
from developmental_ai.foundation import perception as P
from developmental_ai.foundation.perception import (
    FieldFamily, DiscreteProcessFamily, IdentityTracker, PerceptionError,
    PrivilegedInputError, RepresentationRegistry, ResidualStore,
    RevisionEngine, SemVer, StaleDependentError, TrackerConfig,
    UndeclaredFamilyError, declare_family, detect_peaks)
from developmental_ai.foundation.perception.oracle import (
    EvaluatorTruth, OracleReport, oracle_diagnostics)
from developmental_ai.foundation.perception.revision import fit_cv

SC = Scope("unit", "s0", "ep0")
F = 4


def _det(*pts):
    p = np.array(pts, dtype=float).reshape(-1, 2)
    return p, np.tile(np.eye(F)[0], (len(p), 1))


# ------------------------------------------------------------ versioning
@case
def semver_parse_and_order():
    assert SemVer.parse("1.2.3") < SemVer.parse("1.10.0") < SemVer.parse("2.0.0")
    assert str(SemVer.parse("3.0.1")) == "3.0.1"
    raises(lambda: SemVer.parse("1.2"), PerceptionError, "two-part version")
    raises(lambda: SemVer.parse("0.1.0"), PerceptionError, "major 0")
    raises(lambda: SemVer.parse(1), PerceptionError, "non-str")


@case
def registry_declare_bump_rules():
    r = RepresentationRegistry()
    r.declare("x", "1.0.0")
    r.declare("x", "1.0.0")                          # idempotent
    raises(lambda: r.declare("x", "2.0.0"), PerceptionError, "redeclare")
    raises(lambda: r.bump("x", "1.0.0", "same"), PerceptionError, "not newer")
    raises(lambda: r.bump("x", "1.1.0", "  "), PerceptionError, "no reason")
    raises(lambda: r.current("nope"), PerceptionError, "undeclared")
    raises(lambda: r.register_dependent("d", "widget", "x"), PerceptionError,
           "bad kind")
    r.register_dependent("d", "cache", "x", 1)
    raises(lambda: r.register_dependent("d", "cache", "x"), PerceptionError,
           "duplicate dep")
    assert r.stamp("x") == (1, "1.0.0")


@case
def registry_translator_chain_and_failure():
    r = RepresentationRegistry()
    r.declare("x", "1.0.0")
    r.register_dependent("c", "cache", "x", 1)
    r.register_dependent("bad", "cache", "x", 1)
    r.register_translator("x", 1, 2, lambda d, dep: d * 10 if dep.dep_id == "c"
                          else (_ for _ in ()).throw(ValueError("no")))
    r.register_translator("x", 2, 3, lambda d, dep: d + 1)
    rep = r.bump("x", "3.0.0", "two hops")
    assert rep["translated"] == ["c"] and rep["invalidated"] == ["bad"], rep
    assert r.use("c") == 11
    raises(lambda: r.use("bad"), StaleDependentError, "failed translator")
    assert r.dependent("bad").status == "invalidated"
    raises(lambda: r.register_translator("x", 3, 3, lambda d, e: d),
           PerceptionError, "non-advancing translator")


# --------------------------------------------------------------- tracker
@case
def tracker_config_validation():
    raises(lambda: TrackerConfig(meas_std=0).validate(), PerceptionError)
    raises(lambda: TrackerConfig(p_detect=1.0).validate(), PerceptionError)
    raises(lambda: TrackerConfig(ambiguity_threshold=0.6).validate(), PerceptionError)
    raises(lambda: TrackerConfig(reid_window=0).validate(), PerceptionError)


@case
def tracker_input_validation_and_privilege():
    tr = IdentityTracker(SC)
    p, f = _det((0, 0))
    raises(lambda: tr.step(p, f, provenance="evaluator"), PrivilegedInputError,
           "evaluator detections")
    raises(lambda: tr.step(p, f, provenance="imagined"), PrivilegedInputError,
           "imagined detections")
    truth = EvaluatorTruth(((0,),), "evaluator")
    raises(lambda: tr.step(truth, f), PrivilegedInputError, "truth object")
    raises(lambda: tr.step(np.zeros((1, 3)), f), PerceptionError, "(M,3)")
    raises(lambda: tr.step(p, np.zeros((2, F))), PerceptionError, "feat rows")
    raises(lambda: tr.step(np.array([[np.nan, 0]]), f), PerceptionError, "NaN")
    raises(lambda: tr.step(p, f, ego_shift=[1, 2, 3]), PerceptionError, "ego")
    other = ObsRef(Scope("unit", "s0", "other"), 0, "pov")
    raises(lambda: tr.step(p, f, support=other), PerceptionError, "foreign support")
    assert tr.frame == -1, "refused inputs must not advance the frame"
    tr.step(p, f)
    raises(lambda: tr.step(p, np.zeros((1, F + 1))), PerceptionError, "feat dim change")
    r = tr.step(np.zeros((0, 2)), np.zeros((0, F)))      # empty frame is legal
    assert r.frame == 1


@case
def track_state_transitions():
    cfg = TrackerConfig(confirm_hits=3, reid_window=2, tentative_max_misses=0)
    tr = IdentityTracker(SC, cfg)
    for _ in range(3):
        tr.step(*_det((0, 0)))
    (t,) = tr.tracks(("confirmed",))
    assert t.local == "t0" and t.state == "confirmed"
    empty = (np.zeros((0, 2)), np.zeros((0, F)))
    tr.step(*empty)
    assert t.state == "occluded"
    tr.step(*empty)
    assert t.state == "occluded"
    tr.step(*empty)                                  # misses 3 > window 2
    assert t.state == "lost" and t.end_reason == "reid_window_expired"
    tr.step(*_det((0.5, 0.5)))
    tr.step(*empty)                                  # tentative dies on 1st miss
    lost = {x.local: x.end_reason for x in tr.tracks(("lost",))}
    assert lost == {"t0": "reid_window_expired", "t1": "tentative_died"}, lost
    tr.step(*_det((0, 0)))
    assert tr.tracks(("tentative",))[0].local == "t2", "local ids never reused"


@case
def occluded_track_reidentified():
    tr = IdentityTracker(SC, TrackerConfig(reid_window=5))
    for k in range(5):
        tr.step(*_det((0.01 * k, 0)))
    for _ in range(3):
        tr.step(np.zeros((0, 2)), np.zeros((0, F)))
    r = tr.step(*_det((0.08, 0)))
    assert [e.local for e in r.reidentified] == ["t0"], r.reidentified


@case
def soft_probabilities_are_distributions():
    tr = IdentityTracker(SC)
    for _ in range(4):
        r = tr.step(*_det((0, 0), (0.03, 0)))
    for k, p in r.assoc_probs.items():
        assert p.shape == (3,) and abs(p.sum() - 1) < 1e-9 and np.all(p >= 0), (k, p)
        j = [d for d, e in r.assignments.items() if e.local == k][0]
        assert int(np.argmax(p)) == j, "soft argmax must agree with Hungarian here"


@case
def scope_and_reset():
    tr = IdentityTracker(SC)
    tr.step(*_det((0, 0)))
    old = tr.tracks(("tentative",))[0].entity_id
    raises(lambda: tr.reset(SC), PerceptionError, "same-scope reset")
    sc2 = Scope("unit", "s0", "ep1")
    tr.reset(sc2)
    assert tr.tracks(("tentative", "confirmed", "occluded", "lost")) == []
    tr.step(*_det((0, 0)))
    new = tr.tracks(("tentative",))[0].entity_id
    assert new.local == old.local and new != old, "same local, different scope"
    raises(lambda: IdentityTracker("ep0"), PerceptionError, "str scope")


@case
def beliefs_records_and_no_mutation():
    reg = P.default_registry()
    tr = IdentityTracker(SC)
    for k in range(4):
        tr.step(*_det((0.01 * k, 0)), support=ObsRef(SC, k, "pov"))
    before = tr.track("t0").x.copy()
    pred = tr.predict_positions()
    assert np.array_equal(tr.track("t0").x, before), "predict must not mutate"
    close(pred["t0"][0], before[0] + before[2], 1e-12)
    bs = tr.beliefs(reg, support=[ObsRef(SC, 3, "pov")])
    assert len(bs) == 1 and from_json(to_json(bs[0])) == bs[0]
    b = bs[0]
    assert isinstance(b.payload["entity_id"], EntityId)
    assert b.uncertainty["position"].shape == (2,)
    assert b.uncertainty["appearance"] is UNKNOWN
    raises(lambda: tr.beliefs(reg, support=[ObsRef(Scope("u", "s", "x"), 0, "p")]),
           PerceptionError, "foreign support")


@case
def merge_keeps_old_identity_and_aliases():
    tr = IdentityTracker(SC, TrackerConfig(reid_window=1))
    empty = (np.zeros((0, 2)), np.zeros((0, F)))
    for k in range(4):
        tr.step(*_det((0.01 * k, 0)))
    for _ in range(3):
        tr.step(*empty)                              # t0 lost
    for k in range(4):
        tr.step(*_det((0.07 + 0.01 * k, 0)))         # t1 born
    tr.apply_merge("t0", "t1", "unit")
    live = tr.tracks(("confirmed",))
    assert [t.local for t in live] == ["t0"] and tr.resolve("t1") == "t0"
    ghost = tr.track("t1")
    assert ghost.state == "lost" and ghost.entity_id.local == "t1"
    assert ghost.end_reason.startswith("merged_into:t0")
    assert len(live[0].history) == 8


# ---------------------------------------------------------------- oracle
@case
def oracle_refuses_non_evaluator_and_is_read_only():
    raises(lambda: EvaluatorTruth(((0,),), "sensor"), PrivilegedInputError)
    tr = IdentityTracker(SC)
    tr.step(*_det((0, 0)))
    log = tr.export_log()
    raises(lambda: oracle_diagnostics(log, {"det_entity": ((0,),)}),
           PrivilegedInputError, "dict truth")
    raises(lambda: oracle_diagnostics(log, EvaluatorTruth(((0,), (0,)), "evaluator")),
           PerceptionError, "length mismatch")
    rep = oracle_diagnostics(log, EvaluatorTruth(((0,),), "evaluator"))
    assert isinstance(rep, OracleReport) and rep.provenance == "evaluator"

    def _mutate():
        rep.metrics["x"] = 1
    raises(_mutate, TypeError, "mutable report")
    log[0]["assign"] = ("hacked",)
    assert tr.export_log()[0]["assign"] == ("t0",), "export_log must deep-copy"


@case
def oracle_imported_by_nothing_else():
    d = os.path.dirname(P.__file__)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".py") or fn == "oracle.py":
            continue
        src = open(os.path.join(d, fn)).read()
        assert not re.search(r"^\s*(from\s+\S*oracle\s+import|import\s+\S*oracle)",
                             src, re.M), \
            f"{fn} imports the oracle"
    assert not hasattr(P, "oracle_diagnostics") and "EvaluatorTruth" not in P.__all__


# ---------------------------------------------------------------- bridge
@case
def bridge_support_and_scale_rules():
    reg = P.default_registry()
    kw = dict(scale_ref="r", scope=SC, seq=3, registry=reg)
    raises(lambda: P.depth_belief(np.ones(4), support=(), **kw), PerceptionError,
           "no support")
    raises(lambda: P.depth_belief(np.ones(4), support=[ObsRef(SC, 4, "p")], **kw),
           PerceptionError, "support from the future")
    raises(lambda: P.inverse_depth_quantity(np.array([-1.0]), "r"), PerceptionError)
    raises(lambda: P.inverse_depth_quantity(np.array([1.0]), ""), PerceptionError)
    b = P.depth_belief(np.ones(4), support=[ObsRef(SC, 3, "p")], **kw)
    q = P.quantity_from_value(b.variables["inv_depth"])
    assert q.scale_status == "up_to_scale" and q.scale_exponent == -1
    raises(lambda: P.inverse_depth_from_flow(np.zeros((2, 8, 8)), [0.1, 0, 0],
                                             1.2, "r"),
           PerceptionError, "pure rotation reveals no depth")
    m = Quantity(np.array([1.0, 2.0]), M, frame="w", std=np.array([.1, .2]))
    assert P.quantity_from_value(P.quantity_to_value(m)).allclose(m)


@case
def bridge_slot_shape_checks():
    import torch
    from developmental_ai.slots import SAViSlots
    torch.manual_seed(0)
    m = SAViSlots(feat_dim=8, num_slots=3, slot_dim=16, grid=4)
    fm = torch.randn(1, 8, 4, 4)
    out = m.step(fm, m.initial(fm))
    g = SAViSlots.geometry(out["attn"], 4)
    reg = P.default_registry()
    kw = dict(scope=SC, seq=0, support=[ObsRef(SC, 0, "pov")], registry=reg)
    raises(lambda: P.slot_beliefs(out, g, grid=5, **kw), PerceptionError, "grid")
    g_bad = dict(g)
    g_bad["bearing"] = g["bearing"] + 0.5
    raises(lambda: P.slot_beliefs(out, g_bad, grid=4, **kw), PerceptionError,
           "geom from another attention")
    raises(lambda: P.slot_beliefs(out, g, grid=4, inv_depth=torch.rand(1, 1, 8, 8),
                                  **kw), PerceptionError, "depth without scale_ref")
    g_d = SAViSlots.geometry(out["attn"], 4, torch.rand(1, 1, 8, 8))
    raises(lambda: P.slot_beliefs(out, g_d, grid=4, inv_depth=torch.rand(1, 1, 8, 8),
                                  scale_ref="r", **kw),
           PerceptionError, "geom depth from a different map")


# -------------------------------------------------------------- families
@case
def field_family_recovers_known_operator():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(10, 10))
    frames = [f]
    for _ in range(12):
        p = np.pad(frames[-1], 1, mode="edge")
        frames.append(0.5 * frames[-1] + 0.125 * (p[:-2, 1:-1] + p[2:, 1:-1]
                                                  + p[1:-1, :-2] + p[1:-1, 2:]))
    fam = FieldFamily().fit(np.array(frames))
    pred = fam.one_step_predictions(np.array(frames))
    assert np.max(np.abs(pred - np.array(frames[1:]))) < 1e-5
    raises(lambda: FieldFamily().fit(np.zeros((1, 3, 3))), PerceptionError, "T<2")
    raises(lambda: FieldFamily().predict(np.zeros((3, 3))), PerceptionError, "unfit")


@case
def declarations_and_discrete_beliefs():
    fam = DiscreteProcessFamily(2)
    raises(lambda: declare_family("discrete", fam, ""), UndeclaredFamilyError)
    raises(lambda: declare_family("x", object(), "r"), UndeclaredFamilyError)
    raises(lambda: DiscreteProcessFamily(1), PerceptionError)
    frames = np.array([np.full((3, 3), float(t % 2)) for t in range(20)])
    fam.fit(frames)
    post = fam.filter(frames)
    assert post.shape == (20, 2) and np.allclose(post.sum(1), 1)
    d = declare_family("discrete", fam, "two-level blinker")
    reg = P.default_registry()
    b = d.belief(post[-1], reg, SC, 19, [ObsRef(SC, 19, "pov")])
    assert b.payload["family_declaration"]["evidence_scores"] is UNKNOWN
    raises(lambda: d.belief(np.array([0.7, 0.7]), reg, SC, 19,
                            [ObsRef(SC, 19, "pov")]), PerceptionError, "not a dist")


@case
def peak_detection_subpixel():
    yy, xx = np.mgrid[0:12, 0:12]
    f = np.exp(-((xx - 5.3) ** 2 + (yy - 7.0) ** 2) / 2.0)
    pos, amp = detect_peaks(f, 0.3)
    assert pos.shape == (1, 2) and abs(pos[0, 0] - 5.3) < 0.15 and abs(pos[0, 1] - 7) < 1e-6


# -------------------------------------------------------------- revision
@case
def residual_store_bounds():
    rs = ResidualStore(capacity=5)
    raises(lambda: rs.add(0, "nope", "t0", [1.0]), PerceptionError)
    raises(lambda: rs.add(0, "position_innovation", "t0", [np.inf]), PerceptionError)
    for k in range(9):
        rs.add(k, "position_innovation", "t0", [1.0, 0.0])
    assert len(rs) == 5 and rs.vectors("position_innovation").shape == (5, 2)
    assert rs.summary("feature_residual")["structured"] is False


@case
def cv_fit_and_engine_validation():
    t = np.arange(6, dtype=float)
    pos = np.stack([1 + 0.5 * t, -2 * t], 1)
    beta, inv = fit_cv(t, pos, 5.0)
    assert np.allclose(beta[0], pos[-1]) and np.allclose(beta[1], [0.5, -2])
    raises(lambda: RevisionEngine(max_proposals=0), PerceptionError)
    raises(lambda: RevisionEngine(split_window=10, split_holdout=8), PerceptionError)


if __name__ == "__main__":
    sys.exit(1 if run_all("foundation_perception_unit") else 0)
