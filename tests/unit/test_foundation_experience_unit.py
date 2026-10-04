"""Unit cases for developmental_ai.foundation.experience (plan Stage 5).

Parts only: the line format and its hash, ref parsing, constructor bounds,
and every write-time refusal of the store (each one called with the input
that should trip it AND a neighbouring input that should not, so a refusal
that fires on everything fails here too). The design claims — crash
recovery, bounded growth under 10x writes, sampling under extreme skew,
evaluator privacy, reconstruction — live in
tests/_foundation_experience_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_foundation_experience_unit.py
"""
import os
import sys
import tempfile

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, run_all, raises
from developmental_ai.foundation.contracts import (
    UNKNOWN, Action, Evidence, Observation, Prediction, Scope)
from developmental_ai.foundation.experience import (
    DuplicateRecordError, EvidenceStore, ExperienceError, MixedSampler,
    OrderingError, PrivacyError, ProvenanceError, Reservoir, StoreBudgetError,
    parse_ref, score_point_prediction)
from developmental_ai.foundation.experience import chunks as ck
from developmental_ai.foundation.runtime.splits import SplitLeakError

ENV = "unit-env"


def _store(**kw):
    kw.setdefault("max_bytes", 400_000)
    kw.setdefault("chunk_bytes", 8_000)
    kw.setdefault("heldout_fraction", 0.0)
    kw.setdefault("fsync", False)
    return EvidenceStore(os.path.join(tempfile.mkdtemp(), "ev"), **kw)


def _obs(seq, ep="e1", ch="x", prov="sensor", t=None, v=1.0, stream="s0"):
    t = float(seq) if t is None else float(t)
    return Observation(ENV, stream, ep, seq, float(seq), t, ch, prov,
                       {"value": np.float32(v)})


def _act(seq, ep="e1", t0=None, t1=UNKNOWN, stream="s0"):
    t0 = float(seq) + 0.1 if t0 is None else t0
    t1 = t0 + 0.05 if t1 is None else t1
    return Action(ENV, stream, ep, seq, "a", 1, 0.05, t0, t1)


def _pred(pid, t=0.2):
    return Prediction(pid, "b", {"wm": 1}, "wm#1", "a", (1,), 1,
                      {"kind": "gaussian", "mean": 2.0, "std": 1.0}, t)


def _step(st, seq=0, ep="e1", pid="p0"):
    """obs t, prediction, action t, obs t+1 -> (pref, evidence)."""
    o0 = _obs(seq, ep)
    st.log_observation(o0)
    pref = st.log_prediction(_pred(pid), o0.ref())
    a = _act(seq, ep, t1=None)
    st.log_action(a)
    o1 = _obs(seq + 1, ep, v=2.0)
    st.log_observation(o1)
    return pref, Evidence("ev-" + pid, (o0, o1), (a,), (pid,), "valid", "sensor")


# ---- line format ----------------------------------------------------------
@case
def line_round_trip_and_hash_detects_edit():
    ln, h = ck.encode_line(3, "observation", "dev", {"a": 1}, {"b": [1, 2]})
    d, why = ck.decode_line(ln)
    assert d is not None and d["hash"] == h and why == ""
    bad = ln.replace('"b":[1,2]', '"b":[1,3]')
    assert ck.decode_line(bad)[0] is None, "edited body passed its hash"
    assert ck.decode_line(ln[:-5])[0] is None, "truncated line accepted"
    assert ck.decode_line("{}\n")[0] is None


@case
def refs_parse_strictly():
    assert parse_ref("dev/12") == ("dev", 12)
    for r in ("dev12", "train/1", "dev/x", "", None, "dev/-1"):
        raises(lambda r=r: parse_ref(r), KeyError, repr(r))


# ---- constructor ----------------------------------------------------------
@case
def constructor_bounds():
    d = tempfile.mkdtemp()
    raises(lambda: EvidenceStore(os.path.join(d, "a"), max_bytes=10_000,
                                 chunk_bytes=8_000), ValueError, "tiny budget")
    raises(lambda: EvidenceStore(os.path.join(d, "b")), ValueError,
           "create without max_bytes")
    raises(lambda: EvidenceStore(os.path.join(d, "c"), readonly=True),
           FileNotFoundError, "readonly missing")
    EvidenceStore(os.path.join(d, "e"), max_bytes=400_000, chunk_bytes=8_000,
                  heldout_fraction=0.2, fsync=False).close()
    raises(lambda: EvidenceStore(os.path.join(d, "e"), heldout_fraction=0.3),
           ExperienceError, "re-dealing split on reopen")
    EvidenceStore(os.path.join(d, "e")).close()              # same split: ok


# ---- observations ---------------------------------------------------------
@case
def observation_refusals():
    st = _store()
    raises(lambda: st.log_observation(_obs(0, prov="imagined")),
           ProvenanceError, "imagined obs")
    raises(lambda: st.log_observation(_pred("p")), ProvenanceError,
           "prediction as obs")
    st.log_observation(_obs(0))
    raises(lambda: st.log_observation(_obs(0)), DuplicateRecordError, "dup")
    st.log_observation(_obs(2, t=5.0))
    raises(lambda: st.log_observation(_obs(1)), OrderingError, "seq back")
    raises(lambda: st.log_observation(_obs(3, t=4.0)), OrderingError, "t back")
    st.log_observation(_obs(3, ch="y", t=1.0))          # other channel: ok
    st.log_episode_end(Scope(ENV, "s0", "e1"), "terminated", 3, 6.0)
    raises(lambda: st.log_observation(_obs(4, t=7.0)), OrderingError,
           "after end")
    ref = st.log_observation(_obs(0, ep="e2", ch="truth", prov="evaluator"))
    assert ref.startswith("evaluator/"), ref


# ---- actions --------------------------------------------------------------
@case
def action_timing_is_actual_and_ordered():
    st = _store()
    raises(lambda: st.log_action(_act(0, t1=None)), OrderingError, "no obs")
    st.log_observation(_obs(0, t=1.0))
    raises(lambda: st.log_action(_act(0)), ExperienceError, "t_complete UNKNOWN")
    raises(lambda: st.log_action(_act(0, t0=0.5, t1=None)), OrderingError,
           "dispatched before obs")
    st.log_action(_act(0, t0=1.0, t1=1.25))
    raises(lambda: st.log_action(_act(0, t0=1.0, t1=1.3)),
           DuplicateRecordError, "dup action")
    tm = st.learning_view().action_timing()
    assert len(tm) == 1 and abs(tm[0]["elapsed"] - 0.25) < 1e-12
    assert tm[0]["commanded"] == 0.05


# ---- predictions / outcomes ----------------------------------------------
@case
def prediction_refusals():
    st = _store()
    o = _obs(0)
    raises(lambda: st.log_prediction(_pred("p"), o.ref()), OrderingError,
           "unlogged context")
    st.log_observation(o)
    r = st.log_prediction(_pred("p"), o.ref())
    assert parse_ref(r)[1] > 0
    raises(lambda: st.log_prediction(_pred("p"), o.ref()),
           DuplicateRecordError, "dup pid")
    e = _obs(0, ch="truth", prov="evaluator")
    st.log_observation(e)
    raises(lambda: st.log_prediction(_pred("q"), e.ref()), ProvenanceError,
           "evaluator context")


@case
def outcome_accepts_the_honest_case():
    st = _store()
    pref, ev = _step(st)
    oref = st.log_outcome(ev)
    assert st.learning_view().write_seq(oref) > parse_ref(pref)[1]


@case
def outcome_refusals():
    st = _store()
    _, ev = _step(st)
    raises(lambda: st.log_outcome(ev.replace(provenance="imagined",
           observations=tuple(o.replace(provenance="imagined")
                              for o in ev.observations))),
           ProvenanceError, "imagined evidence")
    forged = ev.replace(observations=(ev.observations[0],
                                      _obs(1, v=99.0)))
    raises(lambda: st.log_outcome(forged), ProvenanceError, "altered obs")
    unlogged = ev.replace(observations=ev.observations + (_obs(2, v=1.0),))
    raises(lambda: st.log_outcome(unlogged), ProvenanceError, "unlogged obs")
    raises(lambda: st.log_outcome(ev.replace(prediction_refs=("nope",))),
           OrderingError, "unknown prediction")
    raises(lambda: st.log_outcome(ev.replace(observations=ev.observations[:1],
                                             actions=())),
           OrderingError, "no obs after context")
    st.log_outcome(ev)
    raises(lambda: st.log_outcome(ev), DuplicateRecordError, "dup evidence")


@case
def outcome_refuses_prediction_written_after_it():
    st = _store()
    o0, o1 = _obs(0), _obs(1, v=2.0)
    st.log_observation(o0)
    st.log_observation(o1)                       # outcome arrives first ...
    st.log_prediction(_pred("late"), o0.ref())   # ... prediction after
    ev = Evidence("ev", (o0, o1), (), ("late",), "valid", "sensor")
    raises(lambda: st.log_outcome(ev), OrderingError, "post-hoc prediction")


@case
def outcome_refuses_cross_episode_prediction():
    st = _store()
    a = _obs(0, ep="e1")
    st.log_observation(a)
    st.log_prediction(_pred("p"), a.ref())
    b0, b1 = _obs(0, ep="e2"), _obs(1, ep="e2")
    st.log_observation(b0)
    st.log_observation(b1)
    ev = Evidence("ev", (b0, b1), (), ("p",), "valid", "sensor")
    raises(lambda: st.log_outcome(ev), OrderingError, "reset splice")


# ---- episode ends ---------------------------------------------------------
@case
def episode_end_reasons_are_distinct():
    st = _store()
    for i, r in enumerate(("terminated", "truncated", "client_recovery")):
        st.log_observation(_obs(0, ep=f"e{i}"))
        st.log_episode_end(Scope(ENV, "s0", f"e{i}"), r, 0, 1.0)
    raises(lambda: st.log_episode_end(Scope(ENV, "s0", "e9"), "done", 0, 1.0),
           ValueError, "bad reason")
    raises(lambda: st.log_episode_end(Scope(ENV, "s0", "e0"), "truncated", 0,
                                      1.0), DuplicateRecordError, "dup end")
    lv = st.learning_view()
    got = [lv.episode_end(Scope(ENV, "s0", f"e{i}")) for i in range(3)]
    assert got == ["terminated", "truncated", "client_recovery"], got


# ---- interpretations ------------------------------------------------------
@case
def interpretations_revise_without_touching_raw():
    st = _store()
    r = st.log_observation(_obs(0))
    raw_before = st.learning_view().get(r)
    i1 = st.interpret(r, "labeler", 1, {"label": "stone"})
    i2 = st.interpret(r, "labeler", 2, {"label": "log"}, supersedes=i1)
    raises(lambda: st.interpret(r, "labeler", 3, {"label": "x"},
                                supersedes=i1), ExperienceError, "fork")
    r2 = st.log_observation(_obs(1))
    raises(lambda: st.interpret(r2, "labeler", 1, {}, supersedes=i2),
           ExperienceError, "supersede across targets")
    raises(lambda: st.interpret(i2, "x", 1, {}), ExperienceError,
           "interpret an interpretation")
    lv = st.learning_view()
    cur = lv.interpretations(r)
    assert [b["content"]["label"] for _, b in cur] == ["log"]
    assert len(lv.interpretations(r, current_only=False)) == 2
    assert lv.get(r) == raw_before, "raw record changed"


# ---- views ----------------------------------------------------------------
@case
def learning_view_refuses_other_partitions():
    st = _store()
    e = st.log_observation(_obs(0, ch="truth", prov="evaluator"))
    raises(lambda: st.learning_view().get(e), PrivacyError, "evaluator")
    raises(lambda: st.get(e), PrivacyError, "store.get evaluator")
    assert st.evaluator_view().get(e).channel == "truth"
    raises(lambda: st.learning_view().get("heldout/1"), SplitLeakError,
           "heldout")


# ---- pins -----------------------------------------------------------------
@case
def pin_cap_refuses_and_unpin_reopens():
    st = _store(max_pinned_frac=0.0)
    r = st.log_observation(_obs(0))
    raises(lambda: st.pin(r, "mech"), StoreBudgetError, "pin past cap")
    st2 = _store(max_pinned_frac=0.5)
    r = st2.log_observation(_obs(0))
    st2.pin(r, "mech")
    st2.unpin(r)
    assert not st2.manifest["pins"]


# ---- sampling -------------------------------------------------------------
@case
def sampler_bounds():
    st = _store()
    lv = st.learning_view()
    raises(lambda: MixedSampler(lv, {"failure": 1}, representative_fraction=0.2),
           ValueError, "below floor")
    raises(lambda: MixedSampler(lv, {"bogus": 1}), ValueError, "component")
    raises(lambda: MixedSampler(lv, {"context": 1}), ValueError, "no context")
    s = MixedSampler(lv, {"failure": 1, "coverage": 1})
    for n in (1, 2, 3, 7, 64):
        q = s._quotas(n)
        assert sum(q.values()) == n and q["representative"] >= 0.25 * n, q
    raises(lambda: s.sample(4), ExperienceError, "empty population")


@case
def reservoir_is_uniform():
    counts = np.zeros(100)
    for seed in range(400):
        r = Reservoir(10, np.random.default_rng(seed))
        for i in range(100):
            r.offer(str(i))
        for x in r.items:
            counts[int(x)] += 1
    # expected 40 per item; first and last halves must agree
    assert abs(counts[:50].mean() - counts[50:].mean()) < 4, counts


@case
def scorer_bounds():
    o0, o1 = _obs(0), _obs(1, v=3.0)
    ev = Evidence("e", (o0, o1), (), (), "valid", "sensor")
    err, c = score_point_prediction(_pred("p"), ev, "x", 0)
    assert abs(err - 1.0) < 1e-9 and c is False
    far = _pred("p").replace(outcome={"kind": "gaussian", "mean": 2.0, "std": 0.1})
    assert score_point_prediction(far, ev, "x", 0)[1] is True
    pt = _pred("p").replace(outcome={"kind": "point", "mean": 2.0})
    raises(lambda: score_point_prediction(pt, ev, "x", 0), ExperienceError,
           "point without tol")
    raises(lambda: score_point_prediction(_pred("p"), ev, "x", 5),
           ExperienceError, "no target obs")


if __name__ == "__main__":
    sys.exit(run_all("foundation-experience-unit"))
