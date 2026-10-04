"""Unit cases for foundation.contracts — the parts, not the design.

tests/_foundation_contracts_smoke.py argues the DESIGN claims (SkyBot data
travels through records bit-identically with RED isolated; the nonspatial
adapter satisfies the same contract with no fabricated geometry; the
conformance check catches broken adapters). This file checks the PARTS:
that each sentinel is its own thing, that IDs carry scope, that every
registered record round-trips exactly, and that every malformed input the
plan names (§6 Stage 2 tests) is refused at construction or decode with the
right error class — RecordValidationError for bad fields, SchemaVersionError
for versions this code cannot read.

COVERAGE GUARD. `every_registered_record_has_a_fixture` fails if a record
type is registered without a round-trip fixture here, so a new record
cannot ship untested by omission.

Run: PYTHONPATH=. python tests/unit/test_foundation_contracts_unit.py
"""
import copy
import json
import math
import pickle
import sys

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, run_all, raises
from developmental_ai.foundation.contracts import (
    ABSENT, INAPPLICABLE, UNKNOWN, RECORD_TYPES, Action, ActionSpec,
    ChannelSpec, EntityId, Evidence, Experiment, FrameId, Mechanism,
    MigrationRegistry, ObsRef, Observation, ObservationSpec, Prediction,
    RecordValidationError, SchemaVersionError, Scope, Skill, StateBelief,
    from_dict, from_json, is_missing, same_scope, to_dict, to_json,
    transition_problems, transition_valid)

SC = Scope("env", "s0", "e0")


def _obs(seq=0, channel="c", stream="s0", episode="e0", prov="sensor", **kw):
    payload = kw.pop("payload", {"value": np.arange(3, dtype=np.float32)})
    return Observation("env", stream, episode, seq, float(seq), 100.0 + seq,
                       channel, prov, payload, **kw)


def _act(seq=0, **kw):
    d = dict(command=np.array([0.5, -0.25]), duration=0.05, t_dispatch=1.0,
             t_complete=1.1)
    d.update(kw)
    return Action("env", "s0", "e0", seq, "mv", d["command"], d["duration"],
                  d["t_dispatch"], d["t_complete"])


def fixtures():
    o = _obs(payload={"value": np.ones((2, 2), np.float16), "flag": ABSENT,
                      "nested": {"t": (1, 2.5, "x"), "l": [np.int8(3), None]}})
    return {
        "observation": o,
        "action": _act(),
        "action_spec": ActionSpec.box("mv", [-1, -1], [1, 1], ("m/s", "rad/s"),
                                      duration_mode="held", tick_seconds=0.05),
        "channel_spec": ChannelSpec("count", "scalar", (), "int64", "sensor",
                                    True, units="count", frame=INAPPLICABLE,
                                    neutral=np.int64(0)),
        "observation_spec": ObservationSpec("env", (
            ChannelSpec("a", "vector", (3,), "float32", "sensor", True),
            ChannelSpec("truth", "vector", (3,), "float32", "evaluator", False))),
        "state_belief": StateBelief("b1", "rssm", 1, {"h": np.zeros(4, np.float32),
                                                      "k": 3},
                                    {"h": INAPPLICABLE}, (o.ref(),)),
        "mechanism": Mechanism("m1", 2, ("a",), ("b",), {"regime": "day"},
                               {"gain": 1.5}, {"gain": {"std": 0.1}}, ("ev1",)),
        "prediction": Prediction("p1", "b1", {"rssm": "v3"}, "snap-7", "mv",
                                 (np.zeros(2), np.ones(2)), 2,
                                 {"kind": "gaussian", "mean": np.zeros((2, 3))},
                                 5.0),
        "experiment": Experiment("x1", "does lever A increment?", ("yes", "no"),
                                 {"command": 1}, {"yes": ("p1",)},
                                 {"steps": 20, "seconds": 1.5}),
        "evidence": Evidence("ev1", (o, _obs(1)), (_act(),), ("p1",), "valid",
                             "sensor"),
        "skill": Skill("sk1", 1, "ckpt://skills/sk1@3", {"min_competence": 0.0},
                       {"max_steps": 50}, ("ev1",)),
    }


# ---------------------------------------------------------------- sentinels

@case
def sentinels_are_distinct_from_each_other_and_from_values():
    ss = (UNKNOWN, ABSENT, INAPPLICABLE)
    for i, a in enumerate(ss):
        for j, b in enumerate(ss):
            assert (a == b) == (i == j), f"{a} vs {b}"
        for v in (0, 0.0, None, False, "", float("nan"), np.zeros(0)):
            assert not (a == v) and a is not v, f"{a} equals {v!r}"
        assert is_missing(a)
    for v in (0, None, float("nan"), False):
        assert not is_missing(v), f"{v!r} must not count as missing"


@case
def sentinel_truthiness_raises():
    raises(lambda: bool(UNKNOWN), TypeError, "bool(UNKNOWN)")
    raises(lambda: 1 if ABSENT else 0, TypeError, "if ABSENT")


@case
def sentinels_keep_identity_through_copy_pickle_json():
    for s in (UNKNOWN, ABSENT, INAPPLICABLE):
        assert copy.copy(s) is s and copy.deepcopy(s) is s
        assert pickle.loads(pickle.dumps(s)) is s
    o = from_json(to_json(_obs(payload={"value": UNKNOWN, "b": INAPPLICABLE})))
    assert o.payload["value"] is UNKNOWN and o.payload["b"] is INAPPLICABLE
    assert o.missingness == {"value": UNKNOWN, "b": INAPPLICABLE}


# ---------------------------------------------------------------- ids

@case
def ids_carry_scope():
    s2 = Scope("env", "s0", "e1")
    assert EntityId(SC, "7") == EntityId(SC, "7")
    assert EntityId(SC, "7") != EntityId(s2, "7"), "same local, new episode"
    assert EntityId(SC, "7") != FrameId(SC, "7"), "entity is not a frame"
    assert same_scope(EntityId(SC, "a"), FrameId(SC, "b"))
    assert not same_scope(EntityId(SC, "a"), EntityId(s2, "a"))
    assert not same_scope(EntityId(SC, "a"), "a")
    raises(lambda: Scope("env", "", "e"), RecordValidationError, "empty stream")
    raises(lambda: EntityId(("env", "s", "e"), "a"), RecordValidationError,
           "tuple is not a Scope")
    b = StateBelief("b", "r", 1, {"e": EntityId(SC, "7"), "f": FrameId(s2, "cam")})
    r = from_json(to_json(b))
    assert r.variables["e"] == EntityId(SC, "7")
    assert r.variables["f"].scope == s2


# ---------------------------------------------------------------- round trips

@case
def every_registered_record_has_a_fixture():
    missing = set(RECORD_TYPES) - set(fixtures())
    assert not missing, f"registered records with no round-trip fixture: {missing}"


@case
def every_record_round_trips_dict_json_pickle():
    for name, rec in fixtures().items():
        assert type(rec).SCHEMA == name
        for how, back in (("dict", from_dict(to_dict(rec))),
                          ("json", from_json(to_json(rec))),
                          ("pickle", pickle.loads(pickle.dumps(rec)))):
            assert type(back) is type(rec), f"{name}/{how} type"
            assert back == rec, f"{name} changed through {how}"
        json.loads(to_json(rec))                      # strict JSON


@case
def arrays_keep_dtype_shape_and_bits():
    vals = {"f32": np.array([1.5, np.nan, -np.inf], np.float32),
            "f16": np.ones((2, 3), np.float16),
            "i8": np.array([[-3]], np.int8), "u16": np.arange(4, dtype=np.uint16),
            "b": np.array([True, False]), "zero_d": np.array(2.0),
            "empty": np.zeros((0, 3), np.float64), "f64": np.float64(0.1),
            "i64": np.int64(-9), "be": np.arange(3, dtype=">f4")}
    back = from_json(to_json(_obs(payload=vals))).payload
    for k, v in vals.items():
        b = back[k]
        assert type(b) is type(v), f"{k}: {type(b)} vs {type(v)}"
        assert np.asarray(b).dtype == np.asarray(v).dtype, k
        assert np.asarray(b).shape == np.asarray(v).shape, k
        assert np.asarray(b).tobytes() == np.asarray(v).tobytes(), f"{k} bits"


@case
def tuples_lists_and_nonfinite_floats_survive():
    p = {"t": (1, (2, 3)), "l": [1, [2]], "nan": float("nan"), "inf": float("-inf")}
    b = from_json(to_json(_obs(payload=p))).payload
    assert b["t"] == (1, (2, 3)) and type(b["t"][1]) is tuple
    assert b["l"] == [1, [2]] and type(b["l"]) is list
    assert math.isnan(b["nan"]) and b["inf"] == float("-inf")


# ---------------------------------------------------------------- malformed

@case
def observation_field_validation():
    bad = [dict(seq=-1), dict(seq=True), dict(stream=""), dict(episode=""),
           dict(prov="dreamed"), dict(channel="")]
    for kw in bad:
        raises(lambda kw=kw: _obs(**kw), RecordValidationError, f"obs {kw}")
    for t in (float("nan"), float("inf"), "3", None):
        raises(lambda t=t: Observation("env", "s", "e", 0, t, 0.0, "c", "sensor",
                                       {}), RecordValidationError, f"t_env={t!r}")
    raises(lambda: _obs(payload={"value": 1.0}, missingness={"value": ABSENT}),
           RecordValidationError, "value present but declared ABSENT")
    raises(lambda: _obs(missingness={"x": 0}), RecordValidationError,
           "missingness must hold sentinels")
    raises(lambda: _obs(payload={"__nd__": 1}), RecordValidationError, "reserved key")
    raises(lambda: _obs(payload={1: 2}), RecordValidationError, "non-str key")
    raises(lambda: _obs(payload={"v": np.array(["a"], object)}),
           RecordValidationError, "object array")
    raises(lambda: _obs(payload={"v": object()}), RecordValidationError, "object")


@case
def action_timing_and_command_validation():
    raises(lambda: _act(t_complete=0.5), RecordValidationError, "complete<dispatch")
    raises(lambda: _act(t_dispatch=float("nan")), RecordValidationError, "nan dispatch")
    raises(lambda: _act(duration=-1.0), RecordValidationError, "negative duration")
    raises(lambda: _act(command=True), RecordValidationError, "bool command")
    raises(lambda: _act(command="left"), RecordValidationError, "str command")
    a = _act(t_complete=UNKNOWN, duration=UNKNOWN)
    assert a.t_complete is UNKNOWN and a.duration is UNKNOWN
    assert _act(t_complete=1.0).t_complete == 1.0          # equal is allowed


@case
def other_record_rules():
    raises(lambda: StateBelief("b", "r", 1, {"h": 1}, {"g": 1}),
           RecordValidationError, "uncertainty for undeclared variable")
    assert StateBelief("b", "r", 1, {"h": 1}).uncertainty["h"] is UNKNOWN
    raises(lambda: StateBelief("b", "r", 0, {}), RecordValidationError, "rep v0")
    raises(lambda: Mechanism("m", 1, (), (), {}), RecordValidationError, "no outputs")
    raises(lambda: Prediction("p", "b", {"m": 1}, "s", "mv", (1,), 2,
                              {"kind": "k"}, 0.0), RecordValidationError,
           "candidate len != horizon")
    raises(lambda: Prediction("p", "b", {}, "s", "mv", (1,), 1, {"kind": "k"}, 0.0),
           RecordValidationError, "no model versions")
    raises(lambda: Prediction("p", "b", {"m": 1}, "s", "mv", (1,), 1, {}, 0.0),
           RecordValidationError, "outcome without kind")
    raises(lambda: Experiment("x", "q", ("only",), {}, {}, {"steps": 1}),
           RecordValidationError, "one alternative")
    raises(lambda: Experiment("x", "q", ("a", "b"), {}, {"c": ("p",)}, {"s": 1}),
           RecordValidationError, "evidence for unknown alternative")
    raises(lambda: Experiment("x", "q", ("a", "b"), {}, {}, {"s": -1}),
           RecordValidationError, "negative budget")
    raises(lambda: Skill("k", 1, "ref", {"a": 1}, {}), RecordValidationError,
           "skill with no termination is a latch")
    raises(lambda: Evidence("e", (), (), (), "valid", "sensor"),
           RecordValidationError, "empty evidence")
    raises(lambda: Evidence("e", (_obs(),), (), (), "partial", "sensor"),
           RecordValidationError, "non-valid without reason")
    raises(lambda: Evidence("e", (_obs(), _obs(1, stream="s1")), (), (), "valid",
                            "sensor"), RecordValidationError, "mixed streams")
    raises(lambda: Evidence("e", (_obs(prov="imagined"),), (), (), "valid",
                            "sensor"), RecordValidationError,
           "imagined obs filed as sensor evidence")


@case
def decode_rejects_malformed_and_missing():
    d = to_dict(_obs())
    for mutate, what in (
            (lambda x: x.pop("fields"), "no fields"),
            (lambda x: x.update(extra=1), "extra envelope key"),
            (lambda x: x["fields"].pop("channel"), "missing field"),
            (lambda x: x["fields"].update(colour="red"), "unknown field"),
            (lambda x: x["fields"].update(seq="0"), "seq as str"),
            (lambda x: x["fields"]["payload"]["value"]["__nd__"].update(shape=[4]),
             "array bytes/shape mismatch"),
            (lambda x: x["fields"]["payload"].update(v={"__sentinel__": "MAYBE"}),
             "unknown sentinel")):
        x = json.loads(json.dumps(d))
        mutate(x)
        raises(lambda x=x: from_dict(x), RecordValidationError, what)
    raises(lambda: from_json("{not json"), RecordValidationError, "bad json")
    raises(lambda: from_dict([1]), RecordValidationError, "not a dict")


# ---------------------------------------------------------------- versions

@case
def unsupported_versions_raise_schema_version_error():
    d = to_dict(_obs())
    for v, what in ((2, "future"), (0, "old, no migration"), ("1", "str version")):
        x = dict(d, version=v)
        raises(lambda x=x: from_dict(x), SchemaVersionError, what)
    raises(lambda: from_dict(dict(d, schema="hologram")), SchemaVersionError,
           "unknown schema")


@case
def migration_registry_rules_and_path():
    reg = MigrationRegistry()
    raises(lambda: reg.register("observation", 0, 2, lambda f: f), ValueError,
           "multi-step migration")
    reg.register("observation", 0, 1,
                 lambda f: dict({k: v for k, v in f.items() if k != "sense"},
                                channel=f["sense"], missingness={}))
    raises(lambda: reg.register("observation", 0, 1, lambda f: f), ValueError,
           "duplicate step")
    d = to_dict(_obs())
    f = d["fields"]
    f["sense"] = f.pop("channel")
    f.pop("missingness")
    v0 = {"schema": "observation", "version": 0, "fields": f}
    o = from_dict(v0, registry=reg)
    assert o.channel == "c" and o.missingness == {}
    raises(lambda: from_dict(v0), SchemaVersionError, "global registry has no step")
    bad = MigrationRegistry()
    bad.register("observation", 0, 1, lambda f: None)
    raises(lambda: from_dict(v0, registry=bad), SchemaVersionError,
           "migration returning non-dict")


# ---------------------------------------------------------------- action spec

@case
def action_spec_kinds():
    box = ActionSpec.box("b", [-1.0, 0.0], [1.0, np.inf], ("m", "s"))
    assert box.validate_command([0.5, 9.0]).dtype == np.float64
    for bad in ([2.0, 0.0], [0.0, -1.0], [np.nan, 0.0], [0.0], True):
        raises(lambda c=bad: box.validate_command(c), RecordValidationError, f"box {bad}")
    d = ActionSpec.discrete("d", 4)
    assert d.validate_command(np.int64(3)) == 3
    for bad in (4, -1, 1.0, True):
        raises(lambda c=bad: d.validate_command(c), RecordValidationError, f"disc {bad}")
    m = ActionSpec.multi_discrete("m", (2, 3))
    assert m.validate_command(np.array([1, 2])) == (1, 2)
    for bad in ((2, 0), (0,), (0, 3)):
        raises(lambda c=bad: m.validate_command(c), RecordValidationError, f"md {bad}")
    rng = np.random.default_rng(0)
    for s in (box, d, m):
        for _ in range(50):
            s.validate_command(s.sample(rng))
    raises(lambda: ActionSpec.box("b", [1.0], [0.0], ("m",)), RecordValidationError,
           "low > high")
    raises(lambda: ActionSpec.box("b", [0.0, 0.0], [1.0, 1.0], ("m",)),
           RecordValidationError, "units length")
    raises(lambda: ActionSpec.discrete("d", 0), RecordValidationError, "n=0")
    raises(lambda: ActionSpec.discrete("d", 2, duration_mode="forever"),
           RecordValidationError, "duration mode")
    raises(lambda: ActionSpec.discrete("d", 2, tick_seconds=0.0),
           RecordValidationError, "zero tick")
    raises(lambda: ActionSpec("d", "discrete", (), np.zeros(0), INAPPLICABLE, 2,
                              INAPPLICABLE, ("i",), "tick"), RecordValidationError,
           "discrete spec with a low bound")


@case
def channel_spec_rules():
    raises(lambda: ChannelSpec("t", "vector", (3,), "float32", "evaluator", True),
           RecordValidationError, "evaluator channel policy_visible")
    raises(lambda: ChannelSpec("t", "image", (3, 4), "float32", "sensor", True),
           RecordValidationError, "image without CHW")
    raises(lambda: ChannelSpec("t", "vector", (3,), "U8", "sensor", True),
           RecordValidationError, "string dtype")
    raises(lambda: ChannelSpec("t", "vector", (3,), "float32", "imagined", True),
           RecordValidationError, "adapter cannot declare imagined")
    c = ChannelSpec("t", "vector", (3,), "float32", "sensor", True)
    assert c.value_problems(np.zeros(3, np.float32)) == []
    assert c.value_problems(ABSENT) == []
    assert c.value_problems(np.zeros(3, np.float64)), "dtype mismatch passes"
    assert c.value_problems(np.zeros(4, np.float32)), "shape mismatch passes"
    assert c.value_problems(INAPPLICABLE), "INAPPLICABLE is not a reading"
    raises(lambda: ObservationSpec("e", (c, c)), RecordValidationError, "dup channel")


# ---------------------------------------------------------------- streams

@case
def transition_rules():
    a = _obs(0)
    assert transition_valid(a, _obs(1))
    cases = {"cross-stream": _obs(1, stream="s1"),
             "cross-episode": _obs(1, episode="e1"),
             "cross-channel": _obs(1, channel="d"),
             "non-adjacent": _obs(2),
             "same seq": _obs(0),
             "imagined": _obs(1, prov="imagined"),
             "evaluator": _obs(1, prov="evaluator")}
    for what, b in cases.items():
        assert not transition_valid(a, b), f"{what} accepted as a transition"
        assert transition_problems(a, b), what
    back = Observation("env", "s0", "e0", 1, -5.0, 101.0, "c", "sensor", {"value": 1})
    assert not transition_valid(a, back), "t_env ran backwards"
    assert transition_valid(a, _obs(1, prov="imagined"),
                            allow=("sensor", "imagined")), "explicit opt-in"


if __name__ == "__main__":
    sys.exit(1 if run_all("foundation-contracts-unit") else 0)
