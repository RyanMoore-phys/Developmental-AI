"""Foundation baseline smoke: plan Stage 1, plus Stage 3 item 4 (snapshots).

WHAT IS CLAIMED, AND WHY EACH CLAIM NEEDS A TEST

    Stage 1's completion gate is "another run can reproduce the baseline
    within documented variation". That only means something if:
      * the run's identity is recorded so a mismatch can be SEEN,
      * held-out data stays held out,
      * the list of supplied structure is current, and
      * the measurement tool works on what the host actually writes.
    Each of these has failed silently in this repo before:
      * STATE.md sent a session chasing a terminated host.
      * A tracker reported breaks_total=0 for a whole run because it read the
        wrong dict.
      * A suite listed nowhere never ran once.

    Contracts:
      A. Two manifests built from the same tree are identical except for the
         VOLATILE keys (timestamps, build duration).
      B. A manifest notices what changed. A config edit shows up as
         config.resolved.<key>, config.file_sha256 and environment.<key>.
         A seed change shows up under seeds. A source edit, including an
         UNTRACKED file, changes source.tree_sha256 (this is how an rsynced
         or dirty tree stays identifiable).
         Missing items appear as "absent"/"unknown". They are never None and
         never left out.
      C. Splits are a pure function of (salt, scope, episode):
         - the same in a fresh process (no per-process hash());
         - near the declared held-out fraction;
         - the frames of one episode never straddle the split.
         Falsification: a naive per-frame random split IS caught by
         assert_no_leak.
      D. The assumption register matches the live sensor registry and the
         adapter's proprio fields, and docs/foundation/ASSUMPTIONS.md
         matches the register. Falsification: a reclassified sensor IS
         reported as a mismatch.
      E. tools/baseline_report.py runs on the local (training-free)
         runlogs/, exits 0 and prints "unknown" for what it cannot know.
         On synthetic host-layout logs it computes throughput and detects a
         restart. A logged 0 is reported as 0, not as unknown.
      F. Snapshots stay immutable while being published to:
         - a background thread mutates the live module in place and publishes
           repeatedly while a reader holds a snapshot;
         - the held snapshot stays bit-identical and its hash verifies;
         - ids increase strictly;
         - with the trainer's lock, no published snapshot is torn.
         Falsification: tampering through `.base` IS caught by verify().

Run: PYTHONPATH=. python tests/_foundation_baseline_smoke.py
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.foundation.runtime import (
    assumptions, manifest, splits)
from developmental_ai.foundation.runtime.snapshots import (
    SnapshotRegistry, SnapshotCorruption, stamp_prediction)

CFG = "configs/minecraft_skybot.yaml"
TMP = tempfile.mkdtemp(prefix="foundation_baseline_")


_FROZEN = []


def _frozen_root():
    """A private copy of the source roots, built once.

    Other work may be editing the live tree while this runs; building both
    manifests from one frozen copy makes A a statement about the manifest,
    not about whether anyone saved a file in the meantime. A copy without
    .git is also exactly the rsynced-host case.
    """
    if not _FROZEN:
        root = os.path.join(TMP, "repo")
        os.makedirs(root)
        ign = shutil.ignore_patterns("__pycache__", "*.pyc")
        for top in manifest.SOURCE_ROOTS:
            if os.path.isdir(top):
                shutil.copytree(top, os.path.join(root, top), ignore=ign)
            elif os.path.isfile(top):
                shutil.copy2(top, os.path.join(root, top))
        _FROZEN.append(root)
    return _FROZEN[0]


def _build(cfg=None, seed=0):
    root = _frozen_root()
    return manifest.build_manifest(cfg or os.path.join(root, CFG),
                                   repo_root=root, seed=seed,
                                   hash_checkpoints=True)


def test_manifest_determinism():
    a, b = _build(), _build()
    assert manifest.compare_manifests(a, b) == [], (
        f"same tree, different manifests: {manifest.compare_manifests(a, b)}")
    raw = manifest.compare_manifests(a, b, ignore=())
    assert set(raw) <= set(manifest.VOLATILE_KEYS), (
        f"non-volatile keys differ between identical builds: {raw}")
    # No .git in the copy (the rsynced-host case): identity still recorded,
    # git fields explicitly unknown. The real checkout does have a revision.
    assert a["source"]["git_revision"] == "unknown"
    assert len(a["source"]["tree_sha256"]) == 64
    real = manifest.source_identity(".")
    assert re.match(r"^[0-9a-f]{40}$", str(real["git_revision"])) or \
        real["git_revision"] == "unknown"
    # Round-trip through disk changes nothing either.
    p = os.path.join(TMP, "m.json")
    manifest.write_manifest(a, p)
    assert manifest.compare_manifests(a, manifest.load_manifest(p)) == []
    print(f"  A. identical tree -> identical manifest (volatile only: "
          f"{sorted(raw)})")


def test_manifest_detects_change():
    import yaml
    base = _build()
    cfg = yaml.safe_load(open(os.path.join(_frozen_root(), CFG)))
    cfg["parallel_envs"]["num_envs"] = 3
    p = os.path.join(TMP, "cfg.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    m2 = _build(cfg=p)
    d = manifest.compare_manifests(base, m2)
    for k in ("config.resolved.parallel_envs.num_envs", "config.file_sha256",
              "config.resolved_sha256", "environment.num_envs"):
        assert k in d, f"config change not detected at {k}: {d}"
    d2 = manifest.compare_manifests(base, _build(seed=1))
    assert "seeds.config_seed" in d2 and "config.resolved.seed" in d2, d2
    # Source identity on a synthetic tree: an edit and an untracked file
    # both move the hash.
    root = os.path.join(TMP, "tree")
    os.makedirs(os.path.join(root, "developmental_ai", "pkg"))
    f = os.path.join(root, "developmental_ai", "pkg", "a.py")
    open(f, "w").write("x = 1\n")
    h0 = manifest.source_tree_hash(root)["tree_sha256"]
    open(f, "w").write("x = 2\n")
    h1 = manifest.source_tree_hash(root)["tree_sha256"]
    open(os.path.join(root, "developmental_ai", "new.py"), "w").write("")
    h2 = manifest.source_tree_hash(root)["tree_sha256"]
    open(os.path.join(root, "developmental_ai", "pkg", "x.pyc"), "w").write("z")
    h3 = manifest.source_tree_hash(root)["tree_sha256"]
    assert len({h0, h1, h2}) == 3, "source edit / new file not detected"
    assert h3 == h2, "non-source file changed the source hash"
    # Missing is explicit.
    flat = {}
    manifest._flatten({k: v for k, v in base.items() if k != "config"},
                      "", flat)
    assert None not in flat.values()
    absent = [k for k, v in base["checkpoints"].items() if not v["exists"]]
    assert all(base["checkpoints"][k]["sha256"] == "absent" for k in absent)
    assert base["packages"]["minerl"] in ("absent",) or \
        re.match(r"\d", base["packages"]["minerl"])
    assert base["environment"]["collection_path"]["path"] == \
        "lifelong_segment"
    print(f"  B. config/seed/source changes detected; {len(absent)} absent "
          f"checkpoints recorded as 'absent', not omitted")


def test_split_properties():
    ids = list(range(5000))
    here = [splits.assign_split("minerl:paper:s0", i) for i in ids[:300]]
    code = ("import sys; sys.path.insert(0, '.');"
            "from developmental_ai.foundation.runtime import splits;"
            "print(''.join('h' if splits.assign_split('minerl:paper:s0', i)"
            "=='heldout' else 'd' for i in range(300)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=dict(os.environ, PYTHONPATH=".",
                                             PYTHONHASHSEED="12345"))
    assert out.returncode == 0, out.stderr
    there = ["heldout" if c == "h" else "dev" for c in out.stdout.strip()]
    assert here == there, "assignment differs across processes"
    frac = sum(splits.assign_split("s", i) == splits.HELDOUT
               for i in ids) / len(ids)
    # Binomial sd at n=5000, p=0.2 is 0.0057; 5 sd is a safe bound.
    assert abs(frac - 0.2) < 0.03, f"held-out fraction {frac}"
    # Frames: 300 episodes x 50 adjacent frames.
    frames = [("s", e, t) for e in range(300) for t in range(50)]
    dev, held = splits.partition(frames)
    side = {}
    for x in dev:
        side.setdefault(x[1], set()).add("dev")
    for x in held:
        side.setdefault(x[1], set()).add("held")
    assert all(len(v) == 1 for v in side.values()), "episode straddles split"
    # Falsification: a per-frame random split leaks, and is caught.
    rng = random.Random(0)
    naive_tr = [f for f in frames if rng.random() >= 0.2]
    naive_ev = [f for f in frames if f not in set(naive_tr)]
    try:
        splits.assert_no_leak(naive_tr, naive_ev)
        raise AssertionError("per-frame split was NOT detected as a leak")
    except splits.SplitLeakError:
        pass
    print(f"  C. split deterministic across processes, held-out "
          f"{frac:.3f}, 0/300 episodes straddle; naive frame split caught")


def _doc_sensor_rows():
    rows = {}
    for line in open("docs/foundation/ASSUMPTIONS.md"):
        m = re.match(r"\|\s*`(\w+)`\s*\|\s*(green|amber|red)\s*\|\s*"
                     r"([\w-]+)\s*\|\s*([^|]+)\|", line)
        if m:
            rows[m.group(1)] = (m.group(2), m.group(3), m.group(4).strip())
    return rows


def test_register_matches_live():
    mm = assumptions.register_mismatches(".")
    assert mm == [], f"assumption register is stale: {mm}"
    docs = _doc_sensor_rows()
    assert {k: v[0] for k, v in docs.items()} == \
        assumptions.SENSOR_CLASSIFICATION, (
        "docs/foundation/ASSUMPTIONS.md sensor table disagrees with "
        "assumptions.SENSOR_CLASSIFICATION")
    import yaml
    enabled = yaml.safe_load(open(CFG))["sensors"]["enabled"]
    vis = assumptions.policy_visible_sensors(enabled)
    doc_vis = [k for k, v in docs.items() if v[2] == "policy"]
    assert sorted(vis) == sorted(doc_vis), (vis, doc_vis)
    red_en = assumptions.evaluator_only_sensors(enabled)
    assert red_en == ["true_position"] and "true_position" not in vis
    # Falsification: a reclassification is reported.
    saved = dict(assumptions.SENSOR_CLASSIFICATION)
    try:
        assumptions.SENSOR_CLASSIFICATION["true_position"] = "green"
        assert assumptions.register_mismatches("."), \
            "a reclassified sensor was NOT detected"
    finally:
        assumptions.SENSOR_CLASSIFICATION.clear()
        assumptions.SENSOR_CLASSIFICATION.update(saved)
    print(f"  D. register == live registry == docs; skybot policy sees "
          f"{len(vis)} sensors, evaluator-only {red_en}")


def test_baseline_report():
    # A TRAINING-FREE runlogs dir of our own (2026-10-04). This used the
    # repo's `runlogs/`, which is GITIGNORED: it existed on the dev Mac and
    # not in CI's fresh checkout, so CI died on "runlogs dir not found" while
    # every local run was green. A test must not depend on untracked state.
    empty = os.path.join(TMP, "runlogs_empty")
    os.makedirs(empty)
    p = subprocess.run([sys.executable, "tools/baseline_report.py",
                        empty], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH="."))
    assert p.returncode == 0, p.stderr
    assert "unknown" in p.stdout and "error_by_horizon: unknown" in p.stdout
    # A MISSING dir is an error, not a report of unknowns: a typo'd path must
    # not read as "nothing was logged".
    p = subprocess.run([sys.executable, "tools/baseline_report.py",
                        os.path.join(TMP, "no_such_runlogs")],
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH="."))
    assert p.returncode != 0 and "not found" in (p.stderr + p.stdout), \
        (p.returncode, p.stderr[-300:])
    # Synthetic host-layout logs.
    from developmental_ai.foundation.runtime.baseline import baseline_report
    d = os.path.join(TMP, "runlogs")
    os.makedirs(d)
    recs = [{"schema_version": 1, "seq": 1, "wall_time": 1000.0,
             "total_timesteps": 1000, "breaks_total": 0, "logs": 0},
            {"schema_version": 1, "seq": 2, "wall_time": 1100.0,
             "total_timesteps": 1400, "breaks_total": 5, "logs": 0},
            {"schema_version": 1, "seq": 3, "wall_time": 1200.0,
             "total_timesteps": 200, "breaks_total": 1, "logs": 0}]
    with open(os.path.join(d, "metrics.jsonl.1"), "w") as f:
        f.write(json.dumps(recs[0]) + "\n")
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        for r in recs[1:]:
            f.write(json.dumps(r) + "\n")
        f.write("{not json\n")
    with open(os.path.join(d, "run.log"), "w") as f:
        f.write("Traceback (most recent call last)\nCUDA out of memory\n")
    rep = baseline_report(d)
    seg = rep["throughput"]["segment_steps_per_s"]
    assert seg["n"] == 1 and abs(seg["median"] - 4.0) < 1e-9, seg
    assert rep["failures"]["restarts_detected"] == 1
    assert rep["failures"]["tracebacks"] == 1
    assert rep["failures"]["oom_lines"] == 1
    assert rep["inputs"]["unparseable_lines"] == 1
    assert rep["behaviour"]["logs"]["last"] == 0.0, \
        "a logged zero must be reported as 0, not unknown"
    assert rep["behaviour"]["skills_minted"]["value"] == "unknown"
    print("  E. baseline_report: local runlogs -> unknowns, exit 0; "
          "synthetic logs -> 4.0 steps/s, 1 restart, logged 0 kept as 0")


def test_snapshot_concurrency():
    import torch
    torch.manual_seed(0)
    mod = torch.nn.Linear(64, 64)
    reg = SnapshotRegistry(keep=3)
    held = reg.publish("wm", mod)
    ref = {k: np.array(v, copy=True) for k, v in held.arrays.items()}
    lock = threading.Lock()
    ids, torn, stop = [], [], threading.Event()

    def trainer():
        k = 0
        while not stop.is_set() and k < 300:
            k += 1
            with lock:                          # an optimizer step
                with torch.no_grad():
                    for p in mod.parameters():
                        p.fill_(float(k))
            s = reg.publish("wm", mod, lock=lock, meta={"step": k})
            ids.append(s.snapshot_id)
            for v in s.arrays.values():
                if not np.all(v == v.flat[0]):
                    torn.append(s.snapshot_id)

    t = threading.Thread(target=trainer)
    t.start()
    checks = 0
    try:
        while t.is_alive() or checks < 50:
            held.verify()
            for k, v in held.arrays.items():
                assert np.array_equal(v, ref[k]), "held snapshot mutated"
            cur = reg.current("wm")
            stamp = stamp_prediction({"x": 1}, {"wm": cur})
            assert stamp["snapshots"]["wm"]["snapshot_id"] == \
                cur.snapshot_id
            checks += 1
            if checks > 5000:
                break
    finally:
        stop.set()
        t.join(timeout=10)
    assert ids == sorted(ids) and len(set(ids)) == len(ids), \
        "snapshot ids not strictly increasing"
    assert held.snapshot_id < ids[0]
    assert not torn, f"torn snapshots published: {torn[:5]}"
    assert reg.current("wm").snapshot_id == ids[-1]
    # The live module really did change underneath the held snapshot.
    assert float(mod.weight.detach()[0, 0]) != float(ref["weight"][0, 0])
    # Falsification: deliberate tampering through .base is caught.
    base = held.arrays["bias"].base
    base.setflags(write=True)
    base[0] += 1.0
    try:
        held.verify()
        raise AssertionError("tampering was NOT detected by verify()")
    except SnapshotCorruption:
        pass
    print(f"  F. {len(ids)} publishes under a reader holding #"
          f"{held.snapshot_id}: {checks} checks, held unchanged, ids "
          f"monotonic, 0 torn; tamper detected")


if __name__ == "__main__":
    try:
        test_manifest_determinism()
        test_manifest_detects_change()
        test_split_properties()
        test_register_matches_live()
        test_baseline_report()
        test_snapshot_concurrency()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("[foundation-baseline] ALL PASS")
