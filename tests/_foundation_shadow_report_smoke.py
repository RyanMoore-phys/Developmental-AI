"""Shadow REPORT smoke (2026-10-04): tools/shadow_report.py on recorded stores.

WHAT IS CLAIMED
    tools/shadow_report.py turns a ShadowRecorder EvidenceStore (the live
    runlogs/foundation_shadow, or a copy of it) into REPORT.md + report.json
    with five sections — A inventory, B behaviour, C EVALUATOR-ONLY,
    D preregistered mechanism comparison, E RSSM prior entropy — WITHOUT
    touching the store, and without letting privileged (RED) data leak into
    any section other than C.

WHY
    The first real pull (77 MB, 2026-10-04) is the first time the foundation
    stack meets live SkyBot data. Two failure modes would make its numbers
    worthless: (1) a "report" that quietly repairs or rewrites the store it
    reads (readonly recovery must only NOTE a torn line, never fix it), and
    (2) true_position — F3's information — reaching a behaviour metric or a
    fitted model, which plan §7.4 forbids. And a mechanism verdict computed
    with a hypothesis chosen after seeing the result is exactly the
    "favourable number after the fact" CLAUDE.md §9 warns about, so D must
    go through experiments/ab.py with a prereg frozen before fitting, and
    must say "insufficient data" rather than guess.

    The stores here are written by the REAL ShadowRecorder (direct path, as
    tests/_foundation_shadow_smoke.py F does) over the LIVE bus layout
    (fovea shrunk to 4x4 for speed, still image-shaped) with a RED
    true_position sensor, 2 streams, terminated / truncated / client_recovery
    endings, and a world model whose optimizer steps so model versions move.
    The synthetic world makes `motion`/`proprio.moved` depend on the action,
    so an action-conditioned predictor CAN beat persistence and the
    no-action ablation CANNOT — which is what lets D's wiring be falsified.

CONTRACTS
    A. Every section is present in REPORT.md and report.json; inventory
       counts equal the recorder's own counters (actions, predictions,
       outcomes, episode ends by reason).
    B. Action histogram totals = sampled steps; GUI fraction matches the
       generator's ground truth; prim_extrinsic is reported for stream-0 only.
       ACTION SOURCE (2026-10-04): with a stub option executor driving
       stream 0 (slot 2 skill sk_a, sometimes nested under sk_b, sometimes
       the policy), the by-source histograms equal the generator's ground
       truth per stream and per source, and sum to the stream's total; a
       store written the pre-2026-10-04 way (no payload "source") still
       loads and reports every action as "unrecorded".
    C. RED stays in C: the generator's distinctive x offset (77777) appears
       in C and NOWHERE else in REPORT.md / report.json; and with the
       evaluator partition's decode patched to raise, A, B, D and E still
       compute (they never read RED) while C does raise (the patch bites).
    D. On a sufficient store D RUNS through run_ab with a prereg frozen
       before any result (frozen_at_ns < every run's t_recorded_ns; the
       saved prereg verifies its hash); channels are small-width GREEN
       vectors only (no fovea); ridge beats persistence on `motion`
       (ratio < 0.8) and the no-action ablation is worse than the action-
       conditioned ensemble there (falsifies "D ignores actions"). On a tiny
       store D reports "insufficient data" with reasons and no verdict.
    E. Prior entropy is reported for every prediction, with > 1 model version.
    F. A torn trailing journal line and a truncated sealed chunk (a copy of a
       LIVE store) are tolerated; the quarantined count is reported.
    G. READ-ONLY: every store directory is byte-identical (same files, same
       sha256) before and after the report; the CLI runs.

Run: PYTHONPATH=. python tests/_foundation_shadow_report_smoke.py
"""
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, ".")

import numpy as np
import yaml

TMP = tempfile.mkdtemp(prefix="shadow_report_smoke_")
SKY = yaml.safe_load(open("configs/minecraft_skybot.yaml"))
X_OFFSET = 77777.0
N_ACT = 6        # 0 fwd, 1 left, 2 right, 3 open gui, 4 close gui, 5 noop


def _tree(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def _pick_run_id(n_eps):
    from developmental_ai.foundation.runtime.splits import assign_split
    for i in range(200):
        rid = f"synth{i}"
        ok = True
        for s in ("stream-0", "stream-1"):
            sides = [assign_split(f"synthetic:{s}", f"{rid}-e{k}", 0.2)
                     for k in range(n_eps + 1)]
            if sides[:n_eps].count("heldout") < 1 or sides[:n_eps].count("dev") < 3:
                ok = False
        if ok:
            return rid
    raise AssertionError("no run id with both split sides")


class _Frame:
    def __init__(self, slot, sid):
        self.slot, self.skill_id = slot, sid


class _StubExecutor:
    """The attributes runtime.shadow.action_sources reads from a real
    OptionExecutor after act(): root runtimes, nested substacks, and the
    primitive decision records."""
    def __init__(self):
        self.runtimes = [_Frame(None, None), _Frame(None, None)]
        self.substacks = [[], []]
        self._primary_primitive = {"action": 0}
        self._scout_primitive = [None, {"action": 0}]

        class _Bank:
            def is_scripted(self, slot):
                return False
        self.bank = _Bank()

    def set(self, k):
        """Stream 0, by step: policy / option slot 2 (sk_a) / sk_a nested
        under root sk_b. Returns the source key the report must show."""
        m = k % 5
        self.substacks[0] = []
        if m < 2:
            self.runtimes[0] = _Frame(None, None)
            self._primary_primitive = {"action": 0}
            return "policy"
        self._primary_primitive = None
        if m < 4:
            self.runtimes[0] = _Frame(2, "sk_a")
        else:
            self.runtimes[0] = _Frame(2, "sk_b")
            self.substacks[0] = [_Frame(3, "sk_a")]
        return "option_slot:2 [sk_a]"


def make_store(root, n_eps, steps, seed=0, executor=False, legacy=False):
    """Real ShadowRecorder over the live bus layout; returns ground truth.
    executor: drive action sources through _StubExecutor. legacy: write the
    Action payload exactly as the recorder did before sources existed."""
    import torch
    from developmental_ai.foundation.runtime import ShadowRecorder
    from developmental_ai.sensors import build_default_bus
    from developmental_ai.world_model.rssm import WorldModel
    enabled = list(SKY["sensors"]["enabled"])
    assert "true_position" in enabled and "fovea_native" in enabled
    bus = build_default_bus(13, lambda ctx: None, enabled=enabled, fovea_size=4)
    lay = {n: (o, w) for n, o, w in bus.layout()}
    wm = WorldModel(obs_dim=8, action_dim=N_ACT, stochastic_size=4,
                    stochastic_classes=4, deterministic_size=16, hidden_dim=16)
    wm.optimizer = torch.optim.Adam(wm.parameters(), lr=1e-2)
    rec = ShadowRecorder.from_config(
        {"foundation": {"shadow": {
            "enabled": True, "root": root, "max_bytes": 1 << 27,
            "chunk_bytes": 1 << 15, "every_n_steps": 1,
            "max_ms_per_step": 1e6, "max_ms_single_step": 1e7}}},
        bus=bus, action_dim=N_ACT, is_discrete=True, num_streams=2,
        environment="synthetic", layout_hash=bus.layout_hash())
    rec.run_id = _pick_run_id(n_eps) if n_eps >= 4 else "tiny"
    for st in rec.streams:
        st.ep_n = 0
        rec._new_episode(st)
    rng = np.random.default_rng(seed)
    S = [{"th": 0.0, "gui": 0, "pos": np.zeros(3), "light": 0.5, "a": 5, "t": 0}
         for _ in range(2)]
    truth = {"gui_obs_t": [], "s0_logged": 0,
             "src": {"stream-0": collections.Counter(),
                     "stream-1": collections.Counter()}}
    ex = _StubExecutor() if executor else None
    state = wm.rssm.initial_state(2, torch.device("cpu"))

    def transport(s, a, moved):
        v = np.zeros(bus.width, np.float32)
        o, w = lay["proprio"]
        v[o:o + w] = [1, 1, 1, 0, 0, s["gui"], 0, 0, float(a == 5), 0.5,
                      float(moved), np.sin(s["th"]), np.cos(s["th"])]
        o, w = lay["motion"]
        v[o] = float(moved)
        o, w = lay["light"]
        v[o:o + w] = s["light"]
        o, w = lay["sky"]
        v[o:o + w] = 0.5 + 0.1 * np.sin(s["t"] / 7.0)
        o, w = lay["screen_fx"]
        v[o:o + w] = rng.normal(0, 0.01, w)
        o, w = lay["dead_reckon"]
        v[o:o + w] = np.tile([np.sin(s["pos"][0] / 4), np.cos(s["pos"][0] / 4),
                              np.sin(s["pos"][2] / 4), np.cos(s["pos"][2] / 4)], 4)
        v[bus.vector_width:] = rng.random(bus.width - bus.vector_width)
        return v

    total = n_eps * steps
    recovered = False
    for k in range(total):
        acts = []
        for s in S:
            a = s["a"] if rng.random() < 0.5 else int(rng.integers(0, N_ACT))
            s["a"] = a
            acts.append(a)
        src0 = ex.set(k) if ex is not None else "policy"
        rec.before_step(state, acts, None, wm, executor=ex)
        if legacy:
            for st in rec.streams:      # the pre-2026-10-04 payload: no source
                st.source = None
        elif rec.streams[0].ctx is not None:
            truth["src"]["stream-0"][src0] += 1
        if not legacy and rec.streams[1].ctx is not None:
            truth["src"]["stream-1"]["policy"] += 1
        t = time.time()
        infos, rewards, dones, restarted, ends = [], [], [], [], []
        last = (k % steps) == steps - 1
        for e, s in enumerate(S):
            a = acts[e]
            if a == 3:
                s["gui"] = 1
            elif a == 4:
                s["gui"] = 0
            moved = (a == 0)      # action-determined: persistence cannot know it
            if a == 1:
                s["th"] += 0.3
            if a == 2:
                s["th"] -= 0.3
            if moved:
                s["pos"] += [np.cos(s["th"]), 0.0, np.sin(s["th"])]
            s["light"] = float(np.clip(0.9 * s["light"] + 0.05 + rng.normal(0, 0.02), 0, 1))
            s["t"] += 1
            rst = (e == 1 and not recovered and k == steps + steps // 2)
            recovered = recovered or rst
            infos.append(None if rst else {
                "sensors": transport(s, a, moved),
                "oracle": {"true_position": np.array(
                    [X_OFFSET + s["pos"][0], 64.0, s["pos"][2]], np.float32)}})
            rewards.append(1.0 if moved else 0.0)
            dones.append(bool(last or rst))
            restarted.append(bool(rst))
            ends.append((bool(last and (k // steps) % 2 == 0),
                         bool(last and (k // steps) % 2 == 1) or rst, time.time()))
        st0 = rec.streams[0]
        if st0.ctx is not None:                    # obs t logged this step
            truth["gui_obs_t"].append(float(st0.green[0].value[5]))
        rec.after_step(infos, acts, rewards, dones, restarted, ends, t,
                       prim_extrinsic=rewards[0] + 0.25 * float(acts[0] == 5),
                       intrinsic=np.array([0.1, 0.2]))
        if k % 25 == 24:                           # model version moves
            wm.optimizer.zero_grad()
            sum((p ** 2).sum() for p in wm.parameters()).mul(1e-4).backward()
            wm.optimizer.step()
    assert rec.enabled, rec.disabled_reason
    stats = rec.stats()
    rec.close()
    return stats, truth


def _sections(md):
    heads = ["## A.", "## B.", "## C.", "## D.", "## E."]
    idx = [md.index(h) for h in heads]
    assert idx == sorted(idx), idx
    return {h[3]: md[i:(idx[j + 1] if j + 1 < len(idx) else len(md))]
            for j, (h, i) in enumerate(zip(heads, idx))}


def main():
    import tools.shadow_report as SR
    from developmental_ai.foundation.experiments.ab import Preregistration, VERDICTS
    t_all = time.time()
    full, tiny = os.path.join(TMP, "full"), os.path.join(TMP, "tiny")
    st_full, truth = make_store(full, n_eps=8, steps=60, executor=True)
    st_tiny, _ = make_store(tiny, n_eps=1, steps=12, seed=1, legacy=True)
    torn = os.path.join(TMP, "torn")
    shutil.copytree(full, torn)
    # F. a LIVE copy: torn trailing journal line + a truncated sealed chunk
    dj = [f for f in os.listdir(os.path.join(torn, "dev")) if f.startswith("open-")]
    assert dj, "expected an open dev journal"
    p = os.path.join(torn, "dev", dj[0])
    raw = open(p, "rb").read()
    assert raw.endswith(b"\n")
    cut = raw.rfind(b"\n", 0, len(raw) - 1) + 1 + 40
    open(p, "wb").write(raw[:cut])
    sealed = sorted(f for f in os.listdir(os.path.join(torn, "dev")) if f.startswith("chunk-"))
    assert sealed, "expected sealed dev chunks"
    q = os.path.join(torn, "dev", sealed[-1])
    b = open(q, "rb").read()
    open(q, "wb").write(b[:len(b) // 2])
    before = {r: _tree(r) for r in (full, tiny, torn)}

    out_full = os.path.join(TMP, "out_full")
    rep = SR.build_report(full, out_full, members=3, epochs=25)
    md = open(os.path.join(out_full, "REPORT.md")).read()
    js = json.load(open(os.path.join(out_full, "report.json")))

    # A
    sec = _sections(md)
    for k in "ABCDE":
        assert k in js, k
    A, c = js["A"], st_full["counts"]
    assert A["sampled_steps"] == c["actions"], (A["sampled_steps"], c)
    assert A["predictions"] == c["predictions"], (A["predictions"], c)
    assert A["outcomes"] == c["outcomes"], (A["outcomes"], c)
    assert A["episode_ends"] == {r: c[f"end_{r}"] for r in
                                 ("terminated", "truncated", "client_recovery")
                                 if c.get(f"end_{r}")}, (A["episode_ends"], c)
    assert A["predictions_unresolved"] == c.get("unresolved_predictions", 0)
    assert A["evaluator_partition_counts"]["observation"] == c["evaluator_observations"]
    print(f"  A. all five sections present; inventory == recorder counters "
          f"({A['sampled_steps']} sampled, {A['predictions']} predictions, ends "
          f"{A['episode_ends']})")

    # B
    B = js["B"]
    assert sum(a["n"] for a in B["actions"].values()) == A["sampled_steps"]
    g = B["gui"]["fraction_open"]["stream-0"]
    gt = float(np.mean(truth["gui_obs_t"]))
    assert abs(g - gt) < 1e-9 and 0 < g < 1, (g, gt)
    assert "prim_extrinsic" in B["rewards"]["stream-0"]
    assert "prim_extrinsic" not in B["rewards"]["stream-1"]
    sh = B["rewards"]["stream-0"]["shaping (prim_extrinsic - env)"]
    assert sh["frac_nonzero"] > 0 and B["rewards"]["stream-0"]["intrinsic"]["mean"] > 0
    assert 0 <= B["stillness"]["fraction_transport_unchanged"] <= 1
    bs = B["actions_by_source"]
    for s_, a_ in B["actions"].items():
        assert sum(v["n"] for v in bs[s_].values()) == a_["n"], (s_, bs[s_])
    got = {s_: {k: v["n"] for k, v in d.items()} for s_, d in bs.items()}
    want = {s_: dict(c) for s_, c in truth["src"].items()}
    assert got == want, (got, want)
    assert set(got["stream-0"]) == {"policy", "option_slot:2 [sk_a]"}, got
    assert "by source" in sec["B"] and "option_slot:2 [sk_a]" in sec["B"]
    Bt = SR.section_b(SR.load_green(SR.open_store(tiny)))
    assert {k for d in Bt["actions_by_source"].values() for k in d} == {
        SR.UNRECORDED}, Bt["actions_by_source"]
    print(f"  B. by source == ground truth {got}; legacy store (no payload "
          f"source) loads, all actions '{SR.UNRECORDED}'")
    print(f"  B. action totals == sampled steps; GUI fraction {g:.3f} == ground "
          f"truth; prim_extrinsic on stream-0 only; runs "
          f"{B['gui']['runs']}; stillness {B['stillness']['fraction_transport_unchanged']:.3f}")

    # C: RED only in C, and A/B/D/E never decode the evaluator partition
    C = js["C"]
    assert C["available"] and C["unique_cells_total"] > 1
    assert C["bbox_min"][0] >= X_OFFSET - 100
    tok = "7777"
    assert tok in sec["C"], "C should show the true_position bbox"
    # A RED x is a number whose INTEGER part is ~77777 (X_OFFSET +- the walk).
    # Matching the bare digits also matched fractions — 53/288 prints as
    # 0.1840277777777778 in B's by-source histogram — so the token must not
    # sit after a digit or a decimal point.
    red = re.compile(r"(?<![\d.])777\d\d(?![\d])")
    assert red.search(sec["C"]), "C should show the true_position bbox"
    for k in "ABDE":
        assert not red.search(sec[k]), f"RED value leaked into section {k}"
        assert not red.search(json.dumps(js[k])), f"RED value leaked into json {k}"
    assert not red.search(md[:md.index("## A.")])
    from developmental_ai.foundation.experience import EvidenceStore
    s2 = EvidenceStore(full, readonly=True)

    def boom(*a, **k):
        raise RuntimeError("evaluator partition decoded")
    s2._parts["evaluator"].decode = boom
    s2._parts["evaluator"].read_line = boom
    g2 = SR.load_green(s2)
    SR.section_a(s2, g2), SR.section_b(g2), SR.section_e(g2)
    SR.section_d(s2, g2, os.path.join(TMP, "out_patched"), members=2, epochs=5)
    try:
        SR.section_c(s2)
        raise AssertionError("section C did not read the evaluator partition?")
    except RuntimeError:
        pass
    print(f"  C. {C['unique_cells_total']} cells, stationary "
          f"{C['stationary_fraction_all']:.3f}; RED token only in C; A/B/D/E run "
          f"with evaluator decode patched to raise, C does raise")

    # D
    D = js["D"]
    assert D["status"] == "ran", D.get("reasons")
    assert D["verdict"] in VERDICTS
    assert not any("fovea" in ch for ch in D["channels_used"]), D["channels_used"]
    assert all(int(ch.split("[")[1][:-1]) <= SR.SMALL_WIDTH for ch in D["channels_used"])
    ab = json.load(open(D["ab_json"]))
    pr = Preregistration.from_dict(ab["prereg"])
    pr.verify()
    assert pr.frozen_at_ns < ab["results"]["started_at_ns"]
    assert all(pr.frozen_at_ns < r["t_recorded_ns"] for r in ab["results"]["runs"])
    assert len(pr.seeds) >= 3 and pr.ablation_arm
    arms = D["arms"]
    rm = arms["ridge"]["per_channel_ratio"]["motion"]
    em = arms["ensemble_mlp"]["per_channel_ratio"]["motion"]
    nm = arms["ensemble_mlp_no_action"]["per_channel_ratio"]["motion"]
    assert rm < 0.8, rm
    assert nm > em, (nm, em)
    for a in SR.ARMS:
        assert arms[a]["runs_ok"] >= 3, (a, arms[a])
    Dt = SR.build_report(tiny, os.path.join(TMP, "out_tiny"), members=2, epochs=5)["D"]
    assert Dt["status"] == "insufficient data" and Dt["reasons"] and "verdict" not in Dt
    print(f"  D. ran via run_ab, prereg frozen before results, verdict "
          f"{D['verdict']}; motion ratio ridge {rm:.3f}, ensemble {em:.3f}, "
          f"no-action ablation {nm:.3f}; tiny store -> insufficient data "
          f"({'; '.join(Dt['reasons'])})")

    # E
    E = js["E"]
    assert E["n"] == A["predictions"] and E["versions_unique"] > 1, E
    print(f"  E. prior entropy over {E['n']} predictions, "
          f"{E['versions_unique']} model versions")

    # F
    rt = SR.build_report(torn, os.path.join(TMP, "out_torn"), members=2, epochs=5)
    I = rt["integrity"]
    assert I["quarantined_journal_lines"] == 1, I
    assert I["quarantined_chunk_records"] > 0, I
    assert rt["A"]["outcomes"] < A["outcomes"]
    print(f"  F. torn copy tolerated: {I['quarantined_total']} record(s) "
          f"quarantined ({I['quarantined_journal_lines']} journal line, "
          f"{I['quarantined_chunk_records']} chunk records), report complete")

    # G
    r = subprocess.run([sys.executable, "tools/shadow_report.py", tiny, "--out",
                        os.path.join(TMP, "out_cli"), "--members", "2", "--epochs", "3"],
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH="."))
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(os.path.join(TMP, "out_cli", "REPORT.md"))
    for root, h in before.items():
        assert _tree(root) == h, f"{root} changed: the report wrote to the store"
    print(f"  G. 3 stores byte-identical before/after ({sum(len(h) for h in before.values())} "
          f"files); CLI ok; total {time.time() - t_all:.1f}s")
    shutil.rmtree(TMP, ignore_errors=True)
    print("[foundation_shadow_report_smoke] ALL PASS")


if __name__ == "__main__":
    main()
