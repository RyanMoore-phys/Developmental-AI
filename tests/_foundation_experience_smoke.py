"""Foundation experience smoke (2026-10-03): plan Stage 5 completion gate.

WHAT IS CLAIMED

    "improvement plan.md" §6 Stage 5 gate: "a selected interaction can be
    reconstructed with its original prediction, and sampling changes have
    measurable, documented effects". Also the Stage 5 test list:
    interrupted writes, recovery, sequence alignment, corrupted records,
    sampling frequencies, bounded growth, privacy of evaluator channels, and
    train/evaluation split leakage. Plus the §7.4 regression "a model cannot
    certify its own imagined outcomes as real evidence".

WHY IT IS WORTH A CONTRACT

    This project has repeatedly argued from records it could not check. The
    `places` counter listed `iron_axe: 2281` (CLAUDE.md §5). A training host died
    with ~11 days of unbacked state (§6). world_model.pt was once written
    non-atomically and a truncated copy was silently partially restored
    (replay_buffer.py header). And prioritised sampling is how "a quarter of
    the world model's capacity" went to memorising one behaviour. If the
    evidence store can be torn by a crash, edited without detection, fed
    model outputs as observations, or sampled until ordinary data starves,
    every Stage 6+ comparison built on it inherits the same problem.

Contracts (NonspatialAdapter runs on two interleaved streams, with client
recoveries, ABSENT readings and an EVALUATOR channel, filed by the
reference AdapterRecorder):
    A. Sequence alignment + the completion gate: every outcome reconstructs
       as obs t -> action t -> obs t+1 with its original Prediction
       (deep-equal to the object the model emitted), the snapshot id and
       content hash of the weights that made it (still verifying), and
       write order ctx < prediction < outcome observation. Predictions =
       outcomes + one unresolved per client recovery, exactly. The same
       interaction reconstructs identically after reopening.
    B. Prediction before outcome; imagined is never evidence. A post-hoc
       prediction on an already-observed step is refused; a model rollout
       filed as imagined observations or imagined evidence is refused; a
       model output RELABELLED "sensor" in place of the real reading is
       refused on content hash.
    C. Interrupted writes: a torn journal line plus a stray .tmp, a crash
       between sealing a chunk and committing the manifest, and a crash
       between the commit and removing the journal all recover. Every intact
       record loads, nothing is duplicated, the torn bytes are quarantined
       verbatim, and new writes continue at a fresh write seq.
    D. Corrupted records: a value edited inside a journal line (still valid
       JSON) and a flipped byte in a sealed chunk are both detected by hash.
       Only the damaged data is lost; bytes changed under a running store
       raise StoreCorruption on read.
    E. Sampling: under a 1e18:1 priority skew the representative share is
       >= the declared 0.25 in EVERY batch, and ordinary records are drawn
       at their uniform rate (+-15%), none starved. Removing the floor (the
       falsification) starves them. Changing the mix has measured effects
       in stats(): failure raises the sampled error, coverage moves the bin
       distribution.
    F. Bounded growth: writing > 10x max_bytes keeps disk <= max_bytes
       after every append (and on disk at the end). Pinned mechanism
       evidence and the oldest held-out chunks survive, held-out data past
       the pin cap is evictable (no latch), and evicted refs raise
       RecordEvicted rather than KeyError.
    G. Evaluator privacy: no record, sample, reconstruction context or
       object reachable from the learning view is evaluator data or another
       partition. The evaluator view does see it (positive control).
    H. Split leakage: learning refs are all dev, held-out refs all held-out,
       and assert_no_leak passes on them. A frame of a dev episode on the
       eval side is caught, and the learning view refuses held-out refs.
    I. Migration: a store holding schema-v0 observations is refused by
       ordinary reads, migrates into a NEW root through a registered step,
       keeps write seqs, prediction links and reconstruction, and leaves
       every source file byte-identical. Writing over or inside the source
       is refused.

Run: PYTHONPATH=. python tests/_foundation_experience_smoke.py   (< 60 s)
"""
import gc
import hashlib
import itertools
import os
import shutil
import sys
import tempfile

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.foundation.adapters import NonspatialAdapter
from developmental_ai.foundation.contracts import (
    ABSENT, UNKNOWN as UNKNOWN_, Evidence, MigrationRegistry, Observation, Mechanism,
    SchemaVersionError, deep_equal, to_dict)
from developmental_ai.foundation.experience import (
    AdapterRecorder, EvidenceStore, ExperienceError, MixedSampler,
    OrderingError, PrivacyError, ProvenanceError, RecordEvicted,
    StoreCorruption, migrate_store, parse_ref, split_scope)
from developmental_ai.foundation.experience import chunks as ck
from developmental_ai.foundation.experience.store import _Partition
from developmental_ai.foundation.runtime.snapshots import SnapshotRegistry
from developmental_ai.foundation.runtime.splits import (
    DEV, HELDOUT, SplitLeakError, assign_split)

TMP = tempfile.mkdtemp(prefix="exp_smoke_")


def _clock():
    c = itertools.count()
    return lambda: 1_000.0 + 0.01 * next(c)


def _run(root, steps=300, hf=0.3, max_bytes=4_000_000, chunk=16_000,
         fsync=True, recovery_every=37, policy=None):
    st = EvidenceStore(root, max_bytes=max_bytes, chunk_bytes=chunk,
                       heldout_fraction=hf, fsync=fsync)
    reg = SnapshotRegistry()
    snap = reg.publish("wm", {"w": np.array([0.0, 0.6, -0.4], np.float32)})
    w = snap.arrays["w"]

    def predictor(o, cmd):
        base = 0.0 if o.value is ABSENT else float(o.value)
        return base + float(w[cmd]), 0.5

    captured = {}
    real = st.log_prediction

    def capture(p, ctx):
        captured[p.prediction_id] = p
        return real(p, ctx)

    st.log_prediction = capture
    clock = _clock()
    recs = [AdapterRecorder(
        st, NonspatialAdapter(stream=f"lever-{i}", dropout=0.1, max_steps=25,
                              recovery_every=recovery_every, seed=i,
                              clock=clock),
        "count", predictor, snap, clock, policy=policy, seed=10 + i)
        for i in range(2)]
    for _ in range(steps):
        for r in recs:
            r.run(1)
    return st, recs, snap, captured


def _files(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _du(root):
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(root) for f in fs)


# --------------------------------------------------------------------- A
def test_alignment_and_reconstruction():
    root = os.path.join(TMP, "A")
    st, recs, snap, captured = _run(root, steps=150)
    ev = st.evaluator_view()
    outs = [r for rec in recs for r in rec.outcomes]
    unresolved = sum(len(rec.unresolved) for rec in recs)
    recoveries = len([r for r in ev.refs("episode_end")
                      if ev.get(r)["reason"] == "client_recovery"])
    assert len(captured) == len(outs) + unresolved, "prediction accounting"
    assert unresolved == recoveries > 0, (unresolved, recoveries)
    bad = 0
    for oref in outs:
        it = ev.reconstruct(oref)
        bad += not it.aligned
        (tr,) = it.predictions
        o = it.evidence.observations
        a = it.evidence.actions[0]
        assert o[0].seq == a.seq == tr.context.seq and o[1].seq == a.seq + 1
        ctx_seq = st._obs_keys[(o[0].environment, o[0].stream, o[0].episode,
                                o[0].seq, "count")][1]
        out_seq = st._obs_keys[(o[1].environment, o[1].stream, o[1].episode,
                                o[1].seq, "count")][1]
        assert ctx_seq < tr.write_seq < out_seq < it.write_seq, (
            ctx_seq, tr.write_seq, out_seq, it.write_seq)
    assert bad == 0, f"{bad} misaligned interactions"
    # the gate: pick one by FAILURE retrieval, reconstruct it from disk
    lv = st.learning_view()
    s = MixedSampler(lv, {"failure": 1.0}, representative_fraction=0.25, seed=3)
    pick = next(r for r, c in s.sample(32) if c == "failure")
    st.close()
    del st
    gc.collect()
    st2 = EvidenceStore(root)
    it = st2.learning_view().reconstruct(pick)
    tr = it.predictions[0]
    orig = captured[tr.prediction.prediction_id]
    assert tr.prediction == orig, "reconstructed prediction != original"
    assert tr.snapshot_id == f"wm#{snap.snapshot_id}"
    assert tr.prediction.payload["snapshots"]["wm"]["content_hash"] == \
        snap.content_hash
    snap.verify()
    assert it.scores[tr.prediction.prediction_id] is not None
    assert not st2.recovery, st2.recovery
    print(f"  A. {len(outs)} outcomes over 2 streams all aligned "
          f"(obs t -> act t -> obs t+1, ctx<pred<outcome by write seq); "
          f"{unresolved} predictions unresolved = {recoveries} client "
          f"recoveries; failure-picked {pick} reconstructs after reopen with "
          f"its original prediction + snapshot {tr.snapshot_id} "
          f"(hash {snap.content_hash[:10]} verifies)")
    st2.close()


# --------------------------------------------------------------------- B
def test_prediction_before_outcome_and_imagined():
    st, recs, _, captured = _run(os.path.join(TMP, "B"), steps=20, hf=0.0,
                                 recovery_every=None)
    lv = st.learning_view()
    oref = recs[0].outcomes[-1]
    ev = lv.get(oref)
    o0, o1 = ev.observations
    late = captured[ev.prediction_refs[0]].replace(prediction_id="posthoc")
    st.log_prediction(late, o0.ref())             # logging it is fine ...
    try:                                           # ... claiming it predicted
        st.log_outcome(ev.replace(evidence_id="ev-posthoc",
                                  prediction_refs=("posthoc",)))
        raise AssertionError("post-hoc prediction certified")
    except OrderingError:
        pass
    imagined = o1.replace(provenance="imagined",
                          payload={"value": np.int64(42)})
    for fn, what in (
            (lambda: st.log_observation(imagined.replace(seq=o1.seq + 50,
                                                         t_env=o1.t_env + 50,
                                                         t_wall=o1.t_wall + 50)),
             "imagined observation"),
            (lambda: st.log_outcome(Evidence("ev-img", (imagined,), (), (),
                                             "valid", "imagined")),
             "imagined evidence"),
            (lambda: st.log_outcome(ev.replace(
                evidence_id="ev-relabel",
                observations=(o0, imagined.replace(provenance="sensor")))),
             "relabelled model output")):
        try:
            fn()
            raise AssertionError(f"{what} accepted")
        except ProvenanceError:
            pass
    print("  B. post-hoc prediction refused (OrderingError); imagined "
          "observation, imagined evidence and a model output relabelled "
          "'sensor' all refused (ProvenanceError)")
    st.close()


# --------------------------------------------------------------------- C
def _count(st):
    return sum(len(p.entries) for p in st._parts.values())


def test_interrupted_writes():
    root = os.path.join(TMP, "C")
    st, _, _, _ = _run(root, steps=40, hf=0.0, chunk=8_000)
    n0, nxt0 = _count(st), st._next_seq
    p = st._parts[DEV]
    st.close()
    # 1. torn journal line + stray tmp
    torn, _ = ck.encode_line(nxt0, "observation", DEV, {"x": 1}, {"y": 2})
    torn = torn[: len(torn) // 2].encode()
    with open(p.journal_path(p.journal_cid), "ab") as f:
        f.write(torn)
    with open(os.path.join(p.dir, "chunk-99999999.jsonl.gz.tmp"), "wb") as f:
        f.write(b"partial")
    st = EvidenceStore(root)
    assert _count(st) == n0, (_count(st), n0)
    assert any("damaged/truncated" in m for m in st.recovery), st.recovery
    assert any(".tmp" in m for m in st.recovery)
    qf = [f for f in os.listdir(os.path.join(root, "quarantine")) if "journal" in f]
    with open(os.path.join(root, "quarantine", qf[0]), "rb") as f:
        assert f.read() == torn, "quarantine is not the torn bytes verbatim"
    for r in st.learning_view().refs():
        st.learning_view().get(r)                    # every survivor verifies
    # 2. crash between sealing and committing the manifest
    p = st._parts[DEV]
    cid, lines = p.journal_cid, list(p.journal_lines)
    st._write_sealed(p, cid, lines)                  # gz renamed in, no commit
    st.close()
    st = EvidenceStore(root)
    assert _count(st) == n0 and any("orphan removed" in m for m in st.recovery), \
        st.recovery
    # 3. crash after the commit, before the journal is removed
    p = st._parts[DEV]
    jp = p.journal_path(p.journal_cid)
    with open(jp, "rb") as f:
        saved = f.read()
    st.flush()
    with open(jp, "wb") as f:
        f.write(saved)
    st.close()
    st = EvidenceStore(root)
    assert _count(st) == n0, "duplicated records after stale journal"
    assert any("stale journal" in m for m in st.recovery), st.recovery
    r = st.log_observation(Observation("nonspatial-lever", "lever-9", "x", 0,
                                       0.0, 0.0, "count", "sensor",
                                       {"value": np.int64(1)}))
    assert parse_ref(r)[1] >= nxt0
    print(f"  C. torn line + stray .tmp, seal-before-commit and "
          f"commit-before-unlink crashes all recover: {n0} records intact, "
          f"torn bytes quarantined verbatim, writes resume at seq "
          f"{parse_ref(r)[1]} >= {nxt0}")
    st.close()


# --------------------------------------------------------------------- D
def test_corruption_detection():
    root = os.path.join(TMP, "D")
    st, _, _, _ = _run(root, steps=60, hf=0.0, chunk=8_000)
    n0 = _count(st)
    p = st._parts[DEV]
    st.close()
    # edit a value inside a journal line, keep it valid JSON
    jp = p.journal_path(p.journal_cid)
    with open(jp) as f:
        lines = f.readlines()
    i = next(k for k, ln in enumerate(lines) if '"t_wall":' in ln)
    a = lines[i].index('"t_wall":') + len('"t_wall":')
    lines[i] = lines[i][:a] + "7" + lines[i][a:]
    with open(jp, "w") as f:
        f.writelines(lines)
    # flip a byte in the oldest sealed chunk
    cid = min(int(c) for c in st.manifest["chunks"])
    n_chunk = st.manifest["chunks"][str(cid)]["n"]
    sp = p.sealed_path(cid)
    with open(sp, "r+b") as f:
        f.seek(40)
        b = f.read(1)
        f.seek(40)
        f.write(bytes([b[0] ^ 0xFF]))
    st = EvidenceStore(root)
    lost = n0 - _count(st)
    assert any("sha256 mismatch" in m for m in st.recovery), st.recovery
    assert any("1 damaged" in m for m in st.recovery), st.recovery
    assert 1 <= lost <= n_chunk + 1, (lost, n_chunk)
    # bytes changed under a RUNNING store
    p = st._parts[DEV]
    cid = min(int(c) for c in st.manifest["chunks"] if
              st.manifest["chunks"][c]["part"] == DEV)
    good = ck.read_sealed(p.sealed_path(cid))
    # whitespace-only edits are NOT corruption (hash is over parsed content);
    # change a value.
    k = next(i for i, ln in enumerate(good) if '"t_wall":' in ln)
    seq = ck.decode_line(good[k])[0]["seq"]
    good[k] = good[k].replace('"t_wall":', '"t_wall":9', 1)
    with open(p.sealed_path(cid), "wb") as f:
        f.write(ck.gzip_lines(good))
    p._cache.clear()
    try:
        st.learning_view().get(f"dev/{seq}")
        raise AssertionError("changed bytes read back without complaint")
    except StoreCorruption:
        pass
    print(f"  D. edited journal value and flipped sealed byte both detected "
          f"by hash; {lost} record(s) lost of {n0} (the damaged chunk of "
          f"{n_chunk} + 1 line); live tamper raises StoreCorruption")
    st.close()


# --------------------------------------------------------------------- E
def test_sampling_floor_and_effects():
    # a habit-bound policy (85% rest) so action bins are genuinely unequal
    st, recs, _, _ = _run(os.path.join(TMP, "E"), steps=200, hf=0.0,
                          fsync=False, policy=lambda rng: int(
                              rng.choice(3, p=[0.85, 0.10, 0.05])))
    lv = st.learning_view()
    pop = lv.refs("evidence")
    P = len(pop)
    hot = pop[len(pop) // 2]
    for r in pop:                       # extreme skew: one record 1e18 x rest
        cur = lv.current_score(r)
        pid = lv.meta(r)["pids"][0]
        st.score(r, pid, 1e9 if r == hot else 1e-9, False, "skew", 2,
                 supersedes=None if cur is UNKNOWN_ else cur["ref"])
    s = MixedSampler(lv, {"failure": 1.0}, representative_fraction=0.25, seed=0)
    min_rep = 1.0
    B, n = 200, 64
    for _ in range(B):
        batch = s.sample(n)
        min_rep = min(min_rep, sum(c == "representative" for _, c in batch) / n)
    f = s.frequencies()
    ordinary = np.array([f.get(r, 0) for r in pop if r != hot], float)
    expect = 0.25 * B * n / P
    st_ = s.stats()
    assert min_rep >= 0.25, min_rep
    assert abs(ordinary.mean() / expect - 1) < 0.15, (ordinary.mean(), expect)
    assert ordinary.min() > 0, "an ordinary record was starved"
    # falsification: the same sampler without the floor starves them
    nf = MixedSampler(lv, {"failure": 1.0}, seed=0)
    nf.declared = {"representative": 0.0, "failure": 1.0}
    for _ in range(B):
        nf.sample(n)
    starved = sum(v for r, v in nf.frequencies().items() if r != hot)
    assert starved == 0, starved
    # measurable effects of the mix
    base = MixedSampler(lv, {}, representative_fraction=1.0, seed=1)
    cov = MixedSampler(lv, {"coverage": 1.0}, representative_fraction=0.25, seed=1)
    for sm in (base, cov):
        for _ in range(100):
            sm.sample(64)
    sb, sc = base.stats(), cov.stats()
    assert st_["mean_error_sampled"] > 100 * sb["mean_error_sampled"]
    assert sc["bin_tv_distance"] > 2 * sb["bin_tv_distance"], (sc, sb)
    print(f"  E. 1e18:1 skew over {P} outcomes: representative share >= "
          f"{min_rep:.3f} in every batch, ordinary freq {ordinary.mean():.1f} "
          f"vs uniform {expect:.1f} (min {ordinary.min():.0f}), hot share "
          f"{st_['max_ref_share']:.3f}; without the floor ordinary draws = "
          f"{starved}. Effects: failure mean err {st_['mean_error_sampled']:.3g}"
          f" vs uniform {sb['mean_error_sampled']:.3g}; coverage bin TV "
          f"{sc['bin_tv_distance']:.3f} vs uniform {sb['bin_tv_distance']:.3f}")
    st.close()


# --------------------------------------------------------------------- F
def test_bounded_growth():
    root = os.path.join(TMP, "F")
    MAX, CH = 160_000, 8_000
    st = EvidenceStore(root, max_bytes=MAX, chunk_bytes=CH,
                       heldout_fraction=0.4, max_pinned_frac=0.4)
    clock = _clock()
    ad = NonspatialAdapter(stream="grow", dropout=0.1, max_steps=25, seed=4,
                           clock=clock)
    rec = AdapterRecorder(st, ad, "count",
                          lambda o, c: (0.0 if o.value is ABSENT else float(o.value), 1.0),
                          {"name": "wm", "snapshot_id": 1, "content_hash": "c" * 64},
                          clock)
    rec.run(10)
    early = rec.outcomes[0]
    mech = Mechanism("lever", 1, ("count",), ("count",), {"env": "lever"},
                     evidence_refs=(early,))
    st.pin_mechanism(mech)
    worst, steps = 0, 0
    while st.bytes_written < 10 * MAX:
        before = st._next_seq
        rec.run(1)                       # never refused: no latch
        steps += 1
        worst = max(worst, st.disk_bytes())
        assert st._next_seq > before
    written = st.bytes_written
    st.close()
    on_disk = _du(root)
    st = EvidenceStore(root)
    ev = st.manifest["evicted"]
    held = [c for c, v in st.manifest["chunks"].items() if v["part"] == HELDOUT]
    pins = st.pinned_chunks()
    assert worst <= MAX and on_disk <= MAX, (worst, on_disk, MAX)
    assert ev["chunks"] > 0
    st.evaluator_view().get(early)                   # pinned: survived
    evicted = [r for r in rec.outcomes[:40] if parse_ref(r)[0] == DEV
               and parse_ref(r)[1] not in st._parts[DEV].entries]
    assert evicted, "nothing early was evicted"
    try:
        st.learning_view().get(evicted[0])
        raise AssertionError("evicted ref read")
    except RecordEvicted:
        pass
    assert held and "heldout" in pins.values()
    held_pinned = sum(v == "heldout" for v in pins.values())
    held_evicted = ev.get("by_part", {}).get(HELDOUT, 0)
    assert held_evicted > 0, ("no held-out chunk ever became evictable: "
                              "pins latch", ev)
    pin_bytes = sum(st.manifest["chunks"][str(c)]["bytes"] for c in pins)
    assert pin_bytes <= 0.4 * MAX, pin_bytes
    print(f"  F. {steps} steps, ~{written / MAX:.1f}x max_bytes written "
          f"(raw): peak {worst} and on-disk {on_disk} <= {MAX}; "
          f"{ev['chunks']} chunks/{ev['records']} records evicted oldest-first;"
          f" pinned mechanism evidence {early} survives; "
          f"{held_pinned} oldest held-out chunks pinned ({pin_bytes} B <= cap), "
          f"{held_evicted} newer held-out chunks evicted past the cap (no "
          f"latch); evicted ref -> RecordEvicted")
    st.close()


# --------------------------------------------------------------------- G
def _reachable_partitions(obj, depth=4, seen=None):
    seen = set() if seen is None else seen
    out = []
    if id(obj) in seen or depth < 0:
        return out
    seen.add(id(obj))
    if isinstance(obj, EvidenceStore):
        out.append("STORE")
    if isinstance(obj, _Partition):
        out.append(obj.name)
    kids = []
    if isinstance(obj, dict):
        kids = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        kids = list(obj)[:50]
    elif hasattr(obj, "__dict__"):
        kids = list(vars(obj).values())
    for k in kids:
        if isinstance(k, (str, int, float, bytes, np.ndarray)) or k is None:
            continue
        out += _reachable_partitions(k, depth - 1, seen)
    return out


def test_evaluator_privacy():
    st, recs, _, _ = _run(os.path.join(TMP, "G"), steps=120, hf=0.3,
                          fsync=False)
    lv, ev = st.learning_view(), st.evaluator_view()
    for r in lv.refs():
        rec = lv.get(r)
        obs = (rec,) if isinstance(rec, Observation) else getattr(
            rec, "observations", ())
        for o in obs:
            assert o.provenance != "evaluator", r
    s = MixedSampler(lv, {"failure": 1, "coverage": 1, "contradiction": 1},
                     seed=5)
    drawn = {r for _ in range(100) for r, _ in s.sample(64)}
    assert all(parse_ref(r)[0] == DEV for r in drawn)
    evals = [r for r in ev.refs() if parse_ref(r)[0] == "evaluator"]
    assert len(evals) > 100
    for r in evals[:50]:
        try:
            lv.get(r)
            raise AssertionError("evaluator ref readable from learning view")
        except PrivacyError:
            pass
    reach = set(_reachable_partitions(lv))
    assert reach == {DEV}, reach
    oref = lv.refs("evidence")[0]
    ctx_l = lv.reconstruct(oref).predictions[0].context_observations
    ctx_e = ev.reconstruct(oref).predictions[0].context_observations
    assert {o.channel for o in ctx_l} <= {"count", "chime"}
    assert "regime" in {o.channel for o in ctx_e}, "positive control failed"
    print(f"  G. {len(lv.refs())} learning-view records, {len(drawn)} sampled "
          f"refs: zero evaluator data; {len(evals)} evaluator records refused "
          f"(PrivacyError); objects reachable from the view: {sorted(reach)}; "
          f"evaluator view sees 'regime' (control)")
    st.close()


# --------------------------------------------------------------------- H
def test_split_leakage():
    st, _, _, _ = _run(os.path.join(TMP, "H"), steps=120, hf=0.3, fsync=False)
    lv, hv = st.learning_view(), st.heldout_view()
    tr, te = lv.refs(), hv.refs()
    assert tr and te
    for refs, side in ((tr, DEV), (te, HELDOUT)):
        for r in refs:
            k = st.episode_key(r)
            assert assign_split(k.scope, k.episode_id, 0.3) == side, (r, side)
    st.assert_no_leak(tr, te)
    dev_obs = lv.refs("observation")
    other_frame = next(r for r in dev_obs[5:] if st.episode_key(r) ==
                       st.episode_key(dev_obs[0]) and r != dev_obs[0])
    try:
        st.assert_no_leak([dev_obs[0]], te + [other_frame])
        raise AssertionError("frame-level leak not caught")
    except SplitLeakError:
        pass
    try:
        lv.get(te[0])
        raise AssertionError("held-out readable from learning view")
    except SplitLeakError:
        pass
    print(f"  H. {len(tr)} dev / {len(te)} held-out refs on their deterministic "
          f"sides, no shared episode; a second frame of a dev episode on the "
          f"eval side -> SplitLeakError; learning view refuses held-out")
    st.close()


# --------------------------------------------------------------------- I
def test_migration_to_new_location():
    src = os.path.join(TMP, "I-src")
    st, recs, _, captured = _run(src, steps=30, hf=0.0, fsync=False,
                                 recovery_every=None)
    # legacy data: v0 observations (field 'sense' instead of 'channel')
    legacy = []
    for i in range(5):
        o = Observation("nonspatial-lever", "legacy", "old", i, float(i),
                        float(i), "count", "sensor", {"value": np.int64(i)})
        body = to_dict(o)
        body["version"] = 0
        body["fields"]["sense"] = body["fields"].pop("channel")
        meta = {"env": o.environment, "stream": o.stream, "episode": o.episode,
                "oseq": o.seq, "channel": o.channel, "prov": "sensor",
                "t_wall": o.t_wall, "t_env": o.t_env,
                "content_hash": "legacy"}
        legacy.append((st._append(DEV, "observation", meta, body), o))
    st.flush()
    st.close()
    try:
        EvidenceStore(src).learning_view().get(legacy[0][0])
        raise AssertionError("v0 record decoded without a migration")
    except SchemaVersionError:
        pass
    before = _files(src)
    for bad in (src, os.path.join(src, "inner")):
        try:
            migrate_store(src, bad)
            raise AssertionError(f"migration into {bad} allowed")
        except ExperienceError:
            pass
    reg = MigrationRegistry()
    reg.register("observation", 0, 1, lambda f: dict(
        {k: v for k, v in f.items() if k != "sense"}, channel=f["sense"]))
    dst = os.path.join(TMP, "I-dst")
    rep = migrate_store(src, dst, registry=reg)
    assert _files(src) == before, "migration modified the source"
    try:
        migrate_store(src, dst, registry=reg)
        raise AssertionError("migration over a non-empty target allowed")
    except ExperienceError:
        pass
    new = EvidenceStore(dst)
    lv = new.learning_view()
    for ref, o in legacy:
        assert lv.get(ref) == o, "migrated record differs"
    oref = recs[0].outcomes[-1]
    it = lv.reconstruct(oref)
    assert it.aligned and it.predictions[0].prediction == \
        captured[it.predictions[0].prediction.prediction_id]
    assert new.ident["migrated_from"]["root"] == os.path.abspath(src)
    print(f"  I. v0 records refused by ordinary reads; migrated {rep['records']}"
          f" records ({rep['upgraded']} upgraded) into a new root with write "
          f"seqs and prediction links intact; source byte-identical; target "
          f"= source, nested, or non-empty refused")
    new.close()


if __name__ == "__main__":
    try:
        test_alignment_and_reconstruction()
        test_prediction_before_outcome_and_imagined()
        test_interrupted_writes()
        test_corruption_detection()
        test_sampling_floor_and_effects()
        test_bounded_growth()
        test_evaluator_privacy()
        test_split_leakage()
        test_migration_to_new_location()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("[foundation-experience] ALL PASS")
