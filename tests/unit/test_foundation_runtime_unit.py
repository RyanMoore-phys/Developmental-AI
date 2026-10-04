"""Unit cases for developmental_ai.foundation.runtime (plan Stages 1 and 3).

Parts only: shapes, bounds, degenerate inputs and explicit errors for the
manifest, splits, snapshots, assumption register and collection-path rule.
The design claims (determinism of a whole manifest, concurrent publication,
the loop wiring) live in tests/_foundation_baseline_smoke.py and
tests/_collection_path_smoke.py.

Run: PYTHONPATH=. python tests/unit/test_foundation_runtime_unit.py
"""
import os
import sys
import tempfile

sys.path.insert(0, ".")

import numpy as np

from tests.unit._runner import case, run_all, between, raises
from developmental_ai.foundation.runtime import splits, manifest, assumptions
from developmental_ai.foundation.runtime import snapshots as snaps
from developmental_ai.foundation.runtime import collection_path as cp
from developmental_ai.foundation.runtime.baseline import baseline_report


# ---- splits ---------------------------------------------------------------
@case
def split_fraction_in_unit_interval():
    for i in range(200):
        between(splits.unit_fraction("s", i), 0.0, 1.0 - 1e-12, "fraction")


@case
def split_rejects_unscoped_and_bad_ids():
    raises(lambda: splits.assign_split("", 1), ValueError, "empty scope")
    raises(lambda: splits.assign_split(None, 1), ValueError, "None scope")
    raises(lambda: splits.assign_split("s", 1.5), TypeError, "float id")
    raises(lambda: splits.assign_split("s", True), TypeError, "bool id")
    raises(lambda: splits.assign_split("s", 1, heldout_fraction=1.5),
           ValueError, "fraction > 1")
    raises(lambda: splits.assign_frame("s", 1, -1), ValueError, "neg t")


@case
def split_extremes():
    assert all(splits.assign_split("s", i, 0.0) == splits.DEV
               for i in range(50))
    assert all(splits.assign_split("s", i, 1.0) == splits.HELDOUT
               for i in range(50))


@case
def split_salt_changes_assignment():
    a = [splits.assign_split("s", i) for i in range(400)]
    b = [splits.assign_split("s", i, salt="other") for i in range(400)]
    assert a != b, "salt is not part of the hash"


@case
def leak_detector_reduces_frames_to_episodes():
    splits.assert_no_leak([("s", 1, 0)], [("s", 2, 0)])
    raises(lambda: splits.assert_no_leak([("s", 1, 4)], [("s", 1, 5)]),
           splits.SplitLeakError, "frame-level leak")
    raises(lambda: splits.assert_no_leak([("s", 1)], ["bad"]), TypeError,
           "malformed id")


# ---- manifest -------------------------------------------------------------
@case
def compare_flags_added_removed_and_changed():
    a = {"x": {"y": 1, "z": 2}, "created_at": "t0"}
    b = {"x": {"y": 1, "w": 3}, "created_at": "t1"}
    assert manifest.compare_manifests(a, b) == ["x.w", "x.z"]
    assert manifest.compare_manifests(a, a) == []


@case
def validate_rejects_none_and_wrong_schema():
    raises(lambda: manifest.validate_manifest({}), ValueError, "empty")
    good = {k: {} for k in manifest._REQUIRED}
    good["schema_version"] = manifest.SCHEMA_VERSION
    manifest.validate_manifest(good)
    bad = dict(good, hardware={"gpu": None})
    raises(lambda: manifest.validate_manifest(bad), ValueError, "None value")
    raises(lambda: manifest.validate_manifest(
        dict(good, schema_version=99)), ValueError, "future schema")


@case
def resolve_config_mirrors_run_minecraft():
    c = manifest.resolve_config("configs/minecraft_skybot.yaml", seed=7,
                                forever=True)
    assert c["seed"] == 7
    assert c["lifelong"]["enabled"] is True and c["lifelong"]["forever"]


@case
def package_versions_marks_missing_absent():
    v = manifest.package_versions(("numpy", "no_such_pkg_xyz"))
    assert v["no_such_pkg_xyz"] == manifest.ABSENT
    assert v["numpy"] not in (manifest.ABSENT, manifest.UNKNOWN)


@case
def tree_hash_sees_content_not_mtime():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "a.bin")
    open(p, "wb").write(b"abc")
    h1 = manifest.tree_hash(d)["sha256"]
    os.utime(p, (1, 1))
    assert manifest.tree_hash(d)["sha256"] == h1
    open(p, "wb").write(b"abd")
    assert manifest.tree_hash(d)["sha256"] != h1


# ---- snapshots ------------------------------------------------------------
@case
def snapshot_arrays_are_read_only():
    r = snaps.SnapshotRegistry()
    s = r.publish("m", {"w": np.ones(3, np.float32), "step": 4})
    raises(lambda: s.arrays["w"].__setitem__(0, 5.0), ValueError,
           "write to snapshot array")
    raises(lambda: s.arrays["w"].setflags(write=True), ValueError,
           "re-enable write flag")
    raises(lambda: setattr(s, "snapshot_id", 9), AttributeError, "setattr")

    def _assign():
        s.arrays["w"] = 1
    raises(_assign, TypeError, "mapping write")


@case
def snapshot_ids_monotonic_and_hash_content_addressed():
    r = snaps.SnapshotRegistry(keep=2)
    a = r.publish("m", {"w": np.zeros(2)})
    b = r.publish("m", {"w": np.zeros(2)})
    c = r.publish("m", {"w": np.ones(2)})
    assert a.snapshot_id < b.snapshot_id < c.snapshot_id
    assert a.content_hash == b.content_hash != c.content_hash
    raises(lambda: r.get("m", a.snapshot_id), LookupError, "evicted")
    raises(lambda: r.current("nope"), LookupError, "unknown name")
    raises(lambda: r.publish("m", [1, 2]), TypeError, "non-mapping")
    raises(lambda: r.publish("m", {"x": object()}), TypeError, "object value")


@case
def snapshot_torch_roundtrip_including_bfloat16():
    import torch
    r = snaps.SnapshotRegistry()
    s = r.publish("m", {"a": torch.arange(4, dtype=torch.float32),
                        "b": torch.ones(2, dtype=torch.bfloat16)})
    sd = s.to_state_dict()
    assert sd["b"].dtype == torch.bfloat16
    sd["a"][0] = 99.0                      # a fresh copy: snapshot untouched
    s.verify()
    assert float(s.arrays["a"][0]) == 0.0


@case
def stamp_prediction_records_ids():
    r = snaps.SnapshotRegistry()
    r.publish("wm", {"w": np.zeros(1)})
    out = snaps.stamp_prediction({"mean": 1.0}, r.pin("wm"))
    assert out["snapshots"]["wm"]["snapshot_id"] == r.current("wm").snapshot_id
    raises(lambda: snaps.stamp_prediction(out, r.pin("wm")), ValueError,
           "double stamp")


# ---- assumptions ----------------------------------------------------------
@case
def register_is_well_formed():
    assumptions.validate_register()
    cats = assumptions.by_category()
    assert all(cats[c] for c in assumptions.CATEGORIES), "empty category"
    bad = assumptions.Assumption("X", assumptions.EVALUATOR_ONLY, "s", "w",
                                 True)
    raises(lambda: assumptions.validate_register([bad]), ValueError,
           "evaluator-only + policy-visible")


@case
def red_is_never_policy_visible():
    allnames = list(assumptions.SENSOR_CLASSIFICATION)
    vis = assumptions.policy_visible_sensors(allnames)
    red = [n for n, c in assumptions.SENSOR_CLASSIFICATION.items()
           if c == "red"]
    assert red and not set(red) & set(vis)
    assert set(assumptions.evaluator_only_sensors(allnames)) == set(red)


# ---- collection path ------------------------------------------------------
@case
def collection_path_rule_table():
    def c(env="CrafterReward-v1", en=None, n=None, ll=False, allow=None):
        par = {}
        if en is not None:
            par["enabled"] = en
        if n is not None:
            par["num_envs"] = n
        if allow is not None:
            par["allow_legacy_single_env"] = allow
        return {"environment": {"name": env}, "parallel_envs": par,
                "lifelong": {"enabled": ll}}
    sel = cp.select_collection_path
    assert sel(c()).path == cp.SINGLE_ENV_LEGACY
    assert sel(c(en=False, n=4)).path == cp.SINGLE_ENV_LEGACY
    assert sel(c(en=True, n=2)).path == cp.PARALLEL_EPISODE
    assert sel(c(en=True, n=2, ll=True)).path == cp.LIFELONG_SEGMENT
    raises(lambda: sel(c(en=True, n=1)), cp.UnsupportedCollectionPath,
           "requested parallel, num_envs 1")
    raises(lambda: sel(c(env="MineRLTreechop-v0")),
           cp.UnsupportedCollectionPath, "minecraft single env")
    raises(lambda: sel(c(ll=True, allow=True)), ValueError,
           "lifelong single env has no escape")
    assert sel(c(env="MineRLTreechop-v0", allow=True)).path == \
        cp.SINGLE_ENV_LEGACY
    assert sel({}).path == cp.SINGLE_ENV_LEGACY        # CartPole default


# ---- baseline -------------------------------------------------------------
@case
def baseline_missing_dir_raises_and_empty_dir_is_unknown():
    raises(lambda: baseline_report("/nonexistent/xyz"), FileNotFoundError)
    rep = baseline_report(tempfile.mkdtemp())
    assert rep["behaviour"]["logs"]["value"] == "unknown"
    assert rep["world_model"]["error_by_horizon"]["value"] == "unknown"


if __name__ == "__main__":
    sys.exit(run_all("foundation-runtime-unit"))
