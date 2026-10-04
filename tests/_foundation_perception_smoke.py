"""Foundation perception smoke (2026-10-03) — plan Stage 8 completion gate.

WHAT IS CLAIMED

    "improvement plan.md" Stage 8: connect the existing motion, slot and
    spatial components through the state-belief interface; persistent
    identity with uncertain correspondence, occlusion, appearance change,
    births/deaths; alternative state families; bounded split/merge with
    residuals; versioned representations. Completion gate: "learned state
    improves a downstream outcome, not just visual attractiveness. Separate
    privileged-state diagnostics from pixel-based results."

WHY IT IS WORTH A CONTRACT

    Every representation this project shipped so far was PER-FRAME
    (spatial/memory.py's own docstring), and slots/attention.py says outright
    that a slot which re-mints on every occlusion "cannot support 'the tree I
    was walking toward is still there'". The honest scoreboard (CLAUDE.md §9)
    is 96% of option activity holding attack on an unreachable trunk — a
    target the agent could not keep identified across its own head motion is
    a target it cannot plan about. And the repo has paid for silent
    conflations before: a NEUTRAL reading written as 0.0 became a confident
    claim; geometry() here returns depth == 0 when it has no depth.

    These fixtures are synthetic 2-D on purpose: the slot model is UNTESTED
    on live frames (slots/__init__.py) and the identity layer has to be
    shown correct on cases with known answers before anyone trusts it on
    noise. Privileged truth is used ONLY through foundation.perception.oracle
    and is reported in clearly separated "[oracle]" lines.

Contracts:
    A. Slot bridge on REAL SAViSlots/geometry/slot_relations output (random
       tensor): valid StateBeliefs that JSON-round-trip, per-variable
       uncertainty, inv_depth up_to_scale; with no depth map the bridge says
       UNKNOWN while geometry() said 0, and drops the depth relation fields;
       slot detections drive the tracker; spatial memory -> declared field
       belief with an "unknown" cell scale.
    B. Occlusion: re-identified with the SAME EntityId inside the window; a
       NEW EntityId beyond it (the window is bounded — falsification).
    C. Identity swap: two indistinguishable entities meet, pause, part.
       Ambiguity events are raised, the oracle counts ZERO confident-wrong
       associations, and final identity_confidence is ~0.5 — while (shown,
       so the test bites) the track/entity mapping really did mix. With
       distinct appearance the same geometry raises no ambiguity and no
       switches.
    D. Camera movement: jerky ego-motion; with ego_shift compensation zero
       ID switches and zero extra births; without it, many (the
       compensation is load-bearing).
    E. Texture change: an abrupt appearance change keeps the identity when
       position is continuous; appearance change + teleport gets a new
       identity (so it is position, not stickiness, carrying it).
    F. Scale ambiguity: flow from rssm.flow_from_depth for (D, fwd) and
       (D/2, 2 fwd) is identical; the inverted inverse depth is up_to_scale,
       refuses metric_value, refuses mixing reconstructions, and a RATIO of
       two pixels is metric and correct.
    G. Distractors: Poisson flicker clutter spawns tentatives but ZERO
       confirmed clutter tracks; each true entity gets one track.
    H. Downstream gate: next-position prediction, tracker vs per-frame
       features (static) and frame-to-frame NN velocity, OSPA, 3 seeds; the
       tracker must win on the NON-PRIVILEGED metric (vs next detections) and
       on the [oracle] metric (vs truth). Numbers are printed.
    I. Representation migration: translator path translates a cache; no
       translator invalidates a mechanism (use raises); a minor bump keeps a
       skill valid; a stale belief is rejected; rebuild() reopens.
    J. Non-object fixtures: held-out one-step error recommends FIELD for a
       diffusion field, DISCRETE for a blinking indicator, OBJECTS for moving
       blobs (each family loses somewhere — falsification); beliefs only via
       a declaration, which records against_evidence.
    K. Revision: fragmented tracks are merged in >= 5/8 seeds (6 measured) and a
       different entity in 0/8; alternating pairs split in 8/8 with zero
       post-split switches, unimodal tracks never split; evaluations per
       step never exceed the budget though candidates do.
    L. Residuals: an unmodelled constant acceleration leaves a structured
       innovation residual; pure CV motion does not.

Run: PYTHONPATH=. python tests/_foundation_perception_smoke.py   (< 60 s CPU)
"""
import sys
import time

sys.path.insert(0, ".")

import numpy as np
from scipy.optimize import linear_sum_assignment

from developmental_ai.foundation.contracts import (ObsRef, Scope, UNKNOWN,
                                                   from_json, to_json)
from developmental_ai.foundation.geometry import ScaleStatusError
from developmental_ai.foundation.perception import (
    DiscreteProcessFamily, FieldFamily, IdentityTracker, ObjectFamily,
    ResidualStore, RevisionEngine, StaleDependentError, TrackerConfig,
    compare_families, declare_family, default_registry,
    inverse_depth_from_flow, occupancy_belief, slot_beliefs, slot_detections,
    walkability_belief)
from developmental_ai.foundation.perception.oracle import (EvaluatorTruth,
                                                           oracle_diagnostics)

F = 8
_n = [0]


def check(cond, msg):
    _n[0] += 1
    print(f"  {_n[0]:2d}. {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        raise SystemExit(f"[foundation_perception_smoke] FAIL: {msg}")


# ---------------------------------------------------------------- fixtures
def scene(T, P0, V, feats, rng, noise=0.01, fnoise=0.05, occl=(), clutter=0.0,
          cam=None, accel=None, featfn=None, posfn=None):
    """-> list of (positions (n,2), features (n,F), true entity ids (n,))."""
    out = []
    for t in range(T):
        P = P0 + V * t + (0 if accel is None else 0.5 * accel * t * t)
        ps, fs, es = [], [], []
        for i in range(len(P0)):
            if any(j == i and a <= t < b for j, a, b in occl):
                continue
            p = P[i] if posfn is None else posfn(i, t, P[i])
            p = p - (0 if cam is None else cam[t])
            ps.append(p + rng.normal(0, noise, 2))
            base = feats[i] if featfn is None else featfn(i, t)
            fs.append(base + rng.normal(0, fnoise, F))
            es.append(i)
        for _ in range(rng.poisson(clutter) if clutter else 0):
            ps.append(rng.uniform(-1, 1, 2))
            fs.append(rng.normal(0, 1, F))
            es.append(-1)
        idx = rng.permutation(len(ps))
        out.append((np.array(ps).reshape(-1, 2)[idx],
                    np.array(fs).reshape(-1, F)[idx],
                    np.array(es, int)[idx]))
    return out


def run(frames, cfg=None, ego=None, res=None, eng=None, ep="ep"):
    tr = IdentityTracker(Scope("fixture", "s0", ep), cfg or TrackerConfig(),
                         residuals=res)
    for t, (p, f, _) in enumerate(frames):
        tr.step(p, f, ego_shift=None if ego is None else ego[t])
        if eng is not None:
            eng.step(tr)
    truth = EvaluatorTruth(tuple(tuple(e) for _, _, e in frames), "evaluator")
    return tr, oracle_diagnostics(tr.export_log(), truth).metrics


def ospa(A, B, c=0.1):
    A, B = np.asarray(A).reshape(-1, 2), np.asarray(B).reshape(-1, 2)
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return c
    D = np.minimum(np.linalg.norm(A[:, None] - B[None], axis=-1), c)
    r, cc = linear_sum_assignment(D)
    return float((D[r, cc].sum() + c * abs(n - m)) / max(n, m))


# ---------------------------------------------------------------- contracts
def contract_a_bridge():
    print("A. slot / spatial bridge on real module output")
    import torch
    from developmental_ai.slots import SAViSlots, slot_relations
    from developmental_ai.spatial.memory import EgocentricOccupancy, WalkabilityMap
    torch.manual_seed(0)
    sc = Scope("fixture", "s0", "epA")
    reg = default_registry()
    m = SAViSlots(feat_dim=16, num_slots=4, slot_dim=32, grid=8)
    fm = torch.randn(2, 16, 8, 8)
    out = m.step(fm, m.initial(fm))
    inv = torch.rand(2, 1, 32, 32)
    g = SAViSlots.geometry(out["attn"], 8, inv)
    rel = slot_relations(g, out["attn"], out["alpha"])
    sup = [ObsRef(sc, 5, "pov")]
    bs = slot_beliefs(out, g, grid=8, scope=sc, seq=5, support=sup,
                      registry=reg, inv_depth=inv, scale_ref="recon:epA",
                      relations=rel)
    check(len(bs) == 2 and all(from_json(to_json(b)) == b for b in bs),
          "one belief per batch element; each JSON-round-trips exactly")
    b = bs[0]
    check(set(b.uncertainty) == set(b.variables)
          and b.uncertainty["bearing"].shape == (4,)
          and np.all(b.uncertainty["bearing"] > 0),
          f"per-variable uncertainty for every variable "
          f"(bearing std {np.round(b.uncertainty['bearing'], 3)})")
    check(b.variables["inv_depth"]["scale_status"] == "up_to_scale"
          and b.representation_version == 1
          and b.payload["representation_semver"] == "1.0.0",
          "inv_depth is up_to_scale; representation + version stamped")
    g0 = SAViSlots.geometry(out["attn"], 8)               # no depth map
    b0 = slot_beliefs(out, g0, grid=8, scope=sc, seq=5, support=sup,
                      registry=reg,
                      relations=slot_relations(g0, out["attn"], out["alpha"]))[0]
    check(float(g0["depth"].abs().max()) == 0.0
          and b0.variables["inv_depth"] is UNKNOWN
          and b0.payload["relation_fields"] == ("d_bearing", "size_ratio", "co_motion"),
          "no depth map: geometry() says depth 0, bridge says UNKNOWN and "
          "drops d_depth/occludes")
    tr = IdentityTracker(sc)
    for _ in range(3):
        out = m.step(fm, out)
        g = SAViSlots.geometry(out["attn"], 8)
        p, f, keep = slot_detections(out, g, 0, min_presence=0.0)
        tr.step(p, f)
    check(tr.frame == 2 and len(tr.tracks(("tentative", "confirmed"))) >= 1,
          f"slot detections drive the tracker ({len(keep)} slots/frame)")
    occ = EgocentricOccupancy(grid=16)
    occ.step(np.random.default_rng(0).uniform(0, 0.5, (8, 8)), 0.1, 0.5, 0.0)
    ob = occupancy_belief(occ, scope=sc, seq=5, support=sup, registry=reg)
    wb = walkability_belief(WalkabilityMap(), scope=sc, seq=5, support=sup,
                            registry=reg)
    check(ob.payload["family_declaration"]["name"] == "field"
          and ob.variables["cell_extent"]["scale_status"] == "unknown"
          and wb.uncertainty["passable"] is UNKNOWN
          and from_json(to_json(ob)) == ob,
          "occupancy -> declared FIELD belief, cell scale 'unknown'; "
          "walkability uncertainty UNKNOWN (an EMA is not a posterior)")


def contract_b_occlusion():
    print("B. occlusion and reappearance")
    rng = np.random.default_rng(1)
    P0 = np.array([[-.6, -.3], [-.6, .3], [.6, 0]])
    V = np.array([[.02, 0], [.02, 0], [-.02, .005]])
    fr = scene(50, P0, V, rng.normal(size=(3, F)), rng,
               occl=[(0, 15, 23), (1, 15, 32)])
    tr, m = run(fr, ep="epB")
    et = m["entity_tracks"]
    check(len(et[0]) == 1 and m["id_switches"][0] == 0,
          f"8-frame occlusion (window 10): same EntityId {et[0]}")
    check(len(et[1]) == 2,
          f"17-frame occlusion: beyond the bounded window -> new id {et[1]}")
    check(len(et[2]) == 1, "unoccluded entity untouched")


def _swapscene(same, rng, sp=0.03):
    out = []
    fa = np.eye(F)[0]
    fb = fa.copy() if same else np.eye(F)[1]
    for t in range(45):
        if t < 15:
            a = np.array([-15 * sp + sp * t, 0.0])
        elif t < 21:
            a = np.array([0, .003])
        else:
            a = np.array([0, sp * (t - 20)])
        b = -a
        ps = [a + rng.normal(0, .01, 2), b + rng.normal(0, .01, 2)]
        fs = [fa + rng.normal(0, .05, F), fb + rng.normal(0, .05, F)]
        idx = rng.permutation(2)
        out.append((np.array(ps)[idx], np.array(fs)[idx], np.array([0, 1])[idx]))
    return out


def contract_c_swap():
    print("C. identity swap ambiguity")
    cw, mixed, maxq, confs = 0, 0, 0.0, []
    for s in range(6):
        tr, m = run(_swapscene(True, np.random.default_rng(s)), ep=f"epC{s}")
        cw += m["confident_wrong"]
        mixed += any(len(v) > 1 for v in m["entity_tracks"].values())
        maxq = max([maxq] + [e["p_swap"] for e in tr.ambiguity_log])
        confs += [t.identity_confidence for t in tr.tracks()]
    check(maxq > 0.3, f"indistinguishable meeting raises ambiguity (max "
                      f"p_swap {maxq:.2f})")
    check(mixed >= 4, f"the test bites: track/entity mapping really mixed in "
                      f"{mixed}/6 seeds (a confident tracker would be wrong)")
    check(cw == 0 and max(confs) < 0.65,
          f"[oracle] confident-wrong = {cw}; final identity_confidence "
          f"{np.round(confs, 2).tolist()} (~0.5 = coin flip, reported)")
    cw2, sw2, amb2, confs2 = 0, 0, 0, []
    for s in range(6):
        tr, m = run(_swapscene(False, np.random.default_rng(s)), ep=f"epCd{s}")
        cw2 += m["confident_wrong"]
        sw2 += m["total_id_switches"]
        amb2 += len(tr.ambiguity_log)
        confs2 += [t.identity_confidence for t in tr.tracks()]
    check(amb2 == 0 and sw2 == 0 and min(confs2) > 0.95,
          f"distinct appearance, same geometry: {amb2} ambiguity events, "
          f"{sw2} switches, min confidence {min(confs2):.3f}")


def contract_d_camera():
    print("D. camera movement (ego-motion compensation)")
    rng = np.random.default_rng(2)
    g = np.array([[x, y] for x in (-.3, 0, .3) for y in (-.3, 0, .3)])
    cam = np.cumsum(rng.uniform(-.12, .12, (40, 2)), 0)
    cam -= cam[0]
    fr = scene(40, g, np.zeros_like(g), np.tile(np.eye(F)[0], (9, 1)), rng,
               cam=cam)
    ego = np.vstack([[0, 0], -(cam[1:] - cam[:-1])])
    tr, m = run(fr, ego=ego, ep="epD")
    tr2, m2 = run(fr, ep="epD2")
    check(m["total_id_switches"] == 0 and tr._next == 9,
          f"with ego_shift: 0 switches, 9 identities for 9 entities")
    check(m2["total_id_switches"] > 20 and tr2._next > 30,
          f"without it: {m2['total_id_switches']} switches, {tr2._next} "
          f"identities minted (compensation is load-bearing)")


def contract_e_texture():
    print("E. texture / appearance change")
    rng = np.random.default_rng(3)
    f2 = np.eye(F)[:2]
    fnew = np.eye(F)[5]
    P0, V = np.array([[-.5, 0], [.5, .2]]), np.array([[.01, 0], [-.01, 0]])
    featfn = lambda i, t: fnew if (i == 0 and t >= 20) else f2[i]
    tr, m = run(scene(40, P0, V, f2, rng, featfn=featfn), ep="epE")
    check(len(m["entity_tracks"][0]) == 1 and tr._next == 2,
          "orthogonal appearance change at t=20, continuous position: "
          "same identity, no birth")
    posfn = lambda i, t, p: p + (np.array([0, .5]) if (i == 0 and t >= 20) else 0)
    tr, m = run(scene(40, P0, V, f2, rng, featfn=featfn, posfn=posfn), ep="epE2")
    check(len(m["entity_tracks"][0]) == 2,
          "appearance change + 0.5 teleport: new identity (position, not "
          "stickiness, carried it)")


def contract_f_scale():
    print("F. scale ambiguity of monocular depth")
    import torch
    from developmental_ai.world_model.rssm import flow_from_depth
    torch.manual_seed(1)
    D = torch.rand(1, 1, 16, 16) * 0.5 + 0.1
    ego_cmd = [0.05, 0.01, 0.2]                    # what the agent commanded
    f1 = flow_from_depth(D, torch.tensor([[0.05, 0.01, 0.2]]), 1.22)
    f2 = flow_from_depth(D / 2, torch.tensor([[0.05, 0.01, 0.4]]), 1.22)
    check(float((f1 - f2).abs().max()) == 0.0,
          "world x2 + step x2 produces the identical flow field")
    q1, m1 = inverse_depth_from_flow(f1[0], ego_cmd, 1.22, "recon:1")
    q2, m2 = inverse_depth_from_flow(f2[0], ego_cmd, 1.22, "recon:2")
    check(np.allclose(q1.value, q2.value) and q1.scale_status == "up_to_scale",
          "same estimate for both worlds; scale_status up_to_scale")
    try:
        q1.metric_value()
        ok = False
    except ScaleStatusError:
        ok = True
    try:
        q1 + q2
        ok2 = False
    except ScaleStatusError:
        ok2 = True
    check(ok and ok2, "metric_value() refused; mixing two reconstructions refused")
    from developmental_ai.foundation.perception import inverse_depth_quantity
    a = inverse_depth_quantity(q1.value[3], "recon:1")
    b = inverse_depth_quantity(q1.value[40], "recon:1")
    true = D[0, 0].numpy()[m1]
    r = a / b
    check(r.scale_status == "metric"
          and abs(float(r) - true[3] / true[40]) < 1e-5,
          f"ratio within one reconstruction is metric and exact "
          f"({float(r):.4f})")


def contract_g_distractors():
    print("G. distractors (flicker clutter)")
    rng = np.random.default_rng(4)
    P0 = np.array([[-.6, -.3], [-.6, .3], [.6, 0]])
    V = np.array([[.01, 0], [.01, 0], [-.01, .0025]])
    fr = scene(100, P0, V, rng.normal(size=(3, F)), rng, clutter=2.0)
    tr, m = run(fr, cfg=TrackerConfig(clutter_density=0.5), ep="epG")
    check(m["false_confirmed_tracks"] == 0,
          f"[oracle] 0 confirmed clutter tracks out of {tr._next - 3} "
          f"tentatives spawned by ~200 flicker detections")
    check(all(len(v) == 1 for v in m["entity_tracks"].values())
          and len(m["entity_tracks"]) == 3,
          "each true entity holds exactly one identity")


def _bounce(T, N, rng, clutter=1.0, pmiss=0.1, noise=0.01):
    P = rng.uniform(-.8, .8, (N, 2))
    ang = rng.uniform(0, 2 * np.pi, N)
    sp = rng.uniform(.015, .035, N)
    V = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)
    feats = rng.normal(size=(N, F))
    fr, truth = [], []
    for _ in range(T):
        ps, fs, es = [], [], []
        for i in range(N):
            if rng.random() < pmiss:
                continue
            ps.append(P[i] + rng.normal(0, noise, 2))
            fs.append(feats[i] + rng.normal(0, .1, F))
            es.append(i)
        for _ in range(rng.poisson(clutter)):
            ps.append(rng.uniform(-1, 1, 2))
            fs.append(rng.normal(size=F))
            es.append(-1)
        idx = rng.permutation(len(ps))
        fr.append((np.array(ps).reshape(-1, 2)[idx],
                   np.array(fs).reshape(-1, F)[idx], np.array(es, int)[idx]))
        truth.append(P.copy())
        P = P + V
        for i in range(N):
            for d in range(2):
                if abs(P[i, d]) > 0.9:
                    V[i, d] *= -1
                    P[i, d] = np.clip(P[i, d], -.9, .9)
    return fr, truth


def contract_h_downstream():
    print("H. downstream gate: next-position prediction (OSPA, cutoff 0.1)")
    res = {"track": [[], []], "static": [[], []], "nnvel": [[], []]}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        fr, truth = _bounce(150, 6, rng)
        tr = IdentityTracker(Scope("fixture", "s0", f"epH{seed}"),
                             TrackerConfig(clutter_density=0.25))
        prev = None
        for t in range(len(fr) - 1):
            p, f, e = fr[t]
            tr.step(p, f)
            pred = np.array(list(tr.predict_positions().values())).reshape(-1, 2)
            nn = p.copy()
            if prev is not None and len(prev) and len(p):
                D = np.linalg.norm(p[:, None] - prev[None], axis=-1)
                r, c = linear_sum_assignment(D)
                for a, b in zip(r, c):
                    nn[a] = p[a] + (p[a] - prev[b])
            prev = p
            if t < 20:
                continue
            nxt = fr[t + 1][0]                               # observable
            vis = sorted(set(e[e >= 0].tolist()))
            tt = truth[t + 1][vis]                           # privileged
            for k, pr in (("track", pred), ("static", p), ("nnvel", nn)):
                res[k][0].append(ospa(pr, nxt))
                res[k][1].append(ospa(pr, tt))
    mo = {k: float(np.mean(v[0])) for k, v in res.items()}
    mt = {k: float(np.mean(v[1])) for k, v in res.items()}
    print(f"      observable (vs next detections): tracker {mo['track']:.4f}  "
          f"per-frame static {mo['static']:.4f}  NN-velocity {mo['nnvel']:.4f}")
    print(f"      [oracle] (vs true positions):    tracker {mt['track']:.4f}  "
          f"per-frame static {mt['static']:.4f}  NN-velocity {mt['nnvel']:.4f}")
    check(mo["track"] < 0.9 * min(mo["static"], mo["nnvel"]),
          "non-privileged gate: tracker state beats per-frame features by >10%")
    check(mt["track"] < 0.6 * min(mt["static"], mt["nnvel"]),
          "[oracle] gate: tracker error < 0.6x the best per-frame baseline")


def contract_i_migration():
    print("I. representation migration")
    reg = default_registry()
    sc = Scope("fixture", "s0", "epI")
    old_belief = None
    import torch
    from developmental_ai.slots import SAViSlots
    torch.manual_seed(0)
    m = SAViSlots(feat_dim=8, num_slots=3, slot_dim=16, grid=4)
    fm = torch.randn(1, 8, 4, 4)
    out = m.step(fm, m.initial(fm))
    old_belief = slot_beliefs(out, SAViSlots.geometry(out["attn"], 4), grid=4,
                              scope=sc, seq=0, support=[ObsRef(sc, 0, "pov")],
                              registry=reg)[0]
    cache = reg.register_dependent("cache:bearings", "cache", "savi_slots",
                                   {"bearing": old_belief.variables["bearing"]})
    reg.register_dependent("mech:approach", "mechanism", "savi_slots",
                           {"gain": 0.3})
    reg.register_dependent("skill:face", "skill", "savi_slots", {"k": 1})
    rep = reg.bump("savi_slots", "1.1.0", "added a field (compatible)")
    check(rep["unchanged"] == ["cache:bearings", "mech:approach", "skill:face"]
          and reg.use("skill:face") == {"k": 1},
          "minor bump: every dependent stays valid")
    half_fov = 0.61
    reg.register_translator("savi_slots", 1, 2,
                            lambda d, dep: {"bearing": d["bearing"] * half_fov})
    rep = reg.bump("savi_slots", "2.0.0", "bearing now in radians")
    check(rep["translated"] == ["cache:bearings"] and
          rep["invalidated"] == ["mech:approach", "skill:face"],
          f"major bump: translator path translated, others invalidated {rep}")
    check(np.allclose(reg.use("cache:bearings")["bearing"],
                      old_belief.variables["bearing"] * half_fov),
          "translated cache holds radians")
    try:
        reg.use("mech:approach")
        ok = False
    except StaleDependentError:
        ok = True
    try:
        reg.check_belief(old_belief)
        ok2 = False
    except StaleDependentError:
        ok2 = True
    check(ok and ok2, "stale mechanism use raises; v1 belief rejected under v2")
    reg.rebuild("mech:approach", {"gain": 0.25})
    check(reg.use("mech:approach") == {"gain": 0.25},
          "escape path: rebuild() against the current version reopens it")


def _blobs(T=80, H=24, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(4, 20, (3, 2))
    v = np.array([[0.9, 0.4], [-0.7, 0.8], [0.3, -1.0]])
    yy, xx = np.mgrid[0:H, 0:H]
    out = []
    for _ in range(T):
        f = np.zeros((H, H))
        for (x, y) in p:
            f += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
        out.append(f + rng.normal(0, .02, f.shape))
        p = p + v
        for i in range(3):
            for d in range(2):
                if p[i, d] < 2 or p[i, d] > H - 3:
                    v[i, d] *= -1
                    p[i, d] = np.clip(p[i, d], 2, H - 3)
    return np.array(out)


def _diffusion(T=60, H=16, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, (H, H))
    out = []
    for _ in range(T):
        p = np.pad(f, 1, mode="edge")
        lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * f
        f = f + 0.2 * lap + rng.normal(0, 0.3, (H, H))
        out.append(f.copy())
    return np.array(out)


def _blink(T=80, H=24, seed=0):
    rng = np.random.default_rng(seed)
    s, lv, out = 0, [0, .5, 1], []
    for _ in range(T):
        f = np.zeros((H, H))
        f[9:15, 9:15] = lv[s]
        out.append(f + rng.normal(0, .05, f.shape))
        if rng.random() < .9:
            s = (s + 1) % 3
    return np.array(out)


def contract_j_families():
    print("J. non-object fixtures and declared family selection")
    sc = Scope("fixture", "s0", "epJ")
    want = {"diffusion": "field", "blink": "discrete", "blobs": "objects"}
    comps = {}
    for name, gen in (("diffusion", _diffusion), ("blink", _blink),
                      ("blobs", _blobs)):
        fams = {"objects": ObjectFamily(sc), "field": FieldFamily(),
                "discrete": DiscreteProcessFamily(3)}
        c = compare_families(gen(), fams)
        comps[name] = (c, fams)
        sc_txt = "  ".join(f"{k} {v:.4f}" for k, v in c.scores.items())
        check(c.recommended == want[name],
              f"{name}: recommends {c.recommended} (held-out MSE {sc_txt})")
    c, fams = comps["diffusion"]
    check(not hasattr(c, "belief") and not hasattr(c, "family"),
          "a comparison hands back no family to encode with")
    reg = default_registry()
    d = declare_family("field", fams["field"], "diffusion: no persistent "
                       "compact things", evidence=c)
    fr = _diffusion()
    bel = d.belief(fr[-1], reg, sc, 59, [ObsRef(sc, 59, "field")],
                   std=fams["field"].resid_std)
    check(bel.payload["family_declaration"]["against_evidence"] is False
          and bel.representation == "field_grid"
          and float(bel.uncertainty["field"][0, 0]) > 0,
          "field belief only via a declaration, which it carries")
    d2 = declare_family("objects", fams["objects"], "forced for the test",
                        evidence=c)
    check(d2.against_evidence is True,
          "declaring against the evidence is allowed but recorded")


def contract_k_revision():
    print("K. bounded split/merge revision")
    f0 = np.eye(F)[0]
    merged, wrong = 0, 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        fr = scene(50, np.array([[-.6, 0.0]]), np.array([[.02, 0.003]]),
                   f0[None], rng, occl=[(0, 15, 30)])
        eng = RevisionEngine()
        tr, m = run(fr, eng=eng, ep=f"epKm{seed}")
        merged += len(m["entity_tracks"][0]) == 1
        fr1 = scene(15, np.array([[-.6, 0.0]]), np.array([[.02, 0.0]]),
                    f0[None], rng)
        fr2 = [(p, f, e + 1) for p, f, e in
               scene(20, np.array([[0.0, 0.0]]), np.array([[0.0, 0.025]]),
                     f0[None], rng)]
        gap = [(np.zeros((0, 2)), np.zeros((0, F)), np.zeros(0, int))] * 15
        eng2 = RevisionEngine()
        tr2, m2 = run(fr1 + gap + fr2, eng=eng2, ep=f"epKd{seed}")
        wrong += any(o.accepted for o in eng2.log)
    check(merged >= 5, f"fragment across a beyond-window occlusion merged "
                       f"back into one identity in {merged}/8 seeds")
    check(wrong == 0, f"different entity at the same spot merged in {wrong}/8")
    split_ok, post_sw, false_split = 0, 0, 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        frs = []
        for t in range(60):
            i = t % 2
            p = np.array([-.5 + .01 * t, 0.07 * i]) + rng.normal(0, .01, 2)
            frs.append((p[None], (f0 + rng.normal(0, .05, F))[None], np.array([i])))
        eng = RevisionEngine()
        tr, m = run(frs, eng=eng, ep=f"epKs{seed}")
        k, first = 0, None
        for st, n in enumerate(eng.evaluated_per_step):
            for _ in range(n):
                if eng.log[k].accepted and first is None:
                    first = st
                k += 1
        if first is not None:
            split_ok += 1
            log = tr.export_log()[first + 3:]
            tru = EvaluatorTruth(tuple(tuple(e) for _, _, e in frs[first + 3:]),
                                 "evaluator")
            post_sw += oracle_diagnostics(log, tru).metrics["total_id_switches"]
        rng = np.random.default_rng(100 + seed)
        frs = [(np.array([[-.5 + .01 * t, 0.0]]) + rng.normal(0, .01, (1, 2)),
                (f0 + rng.normal(0, .05, F))[None], np.array([0]))
               for t in range(60)]
        eng = RevisionEngine()
        run(frs, eng=eng, ep=f"epKu{seed}")
        false_split += sum(o.accepted for o in eng.log)
    check(split_ok == 8 and post_sw == 0,
          f"alternating pair (7 sigma apart) split in {split_ok}/8; "
          f"[oracle] post-split switches {post_sw}")
    check(false_split == 0, f"unimodal track split {false_split} times")
    rng = np.random.default_rng(9)
    P0 = rng.uniform(-.8, .8, (8, 2))
    fr = scene(40, P0, rng.uniform(-.02, .02, (8, 2)), np.eye(F), rng,
               occl=[(i, 10, 26) for i in range(8)])
    eng = RevisionEngine(max_proposals=2)
    tr = IdentityTracker(Scope("fixture", "s0", "epKb"))
    max_cands = 0
    for p, f, _ in fr:
        tr.step(p, f)
        max_cands = max(max_cands, len(eng.propose(tr)))
        eng.step(tr)
    check(max(eng.evaluated_per_step) <= 2 < max_cands,
          f"budget: up to {max_cands} candidates in a step, never more than "
          f"{max(eng.evaluated_per_step)} evaluated")


def contract_l_residuals():
    print("L. residual retention")
    zs = {}
    for acc in (0.0, 0.002):
        rng = np.random.default_rng(5)
        P0 = np.array([[-.6, -.3], [-.6, .3], [.6, 0], [0, .6]])
        V = np.array([[.02, 0], [.02, 0], [-.02, .005], [0, -.03]])
        fr = scene(60, P0, V - np.array([0, acc * 30]), rng.normal(size=(4, F)),
                   rng, accel=np.array([0, acc]))
        rs = ResidualStore()
        run(fr, res=rs, ep=f"epL{acc}")
        s = rs.summary("position_innovation")
        zs[acc] = s
    check(zs[0.002]["structured"] and zs[0.002]["z"][1] > 4,
          f"unmodelled acceleration: innovation mean z = "
          f"{np.round(zs[0.002]['z'], 2).tolist()} -> structured, retained")
    check(not zs[0.0]["structured"],
          f"pure CV motion: z = {np.round(zs[0.0]['z'], 2).tolist()} -> not "
          f"structured")


def main():
    t0 = time.process_time()
    for fn in (contract_a_bridge, contract_b_occlusion, contract_c_swap,
               contract_d_camera, contract_e_texture, contract_f_scale,
               contract_g_distractors, contract_h_downstream,
               contract_i_migration, contract_j_families, contract_k_revision,
               contract_l_residuals):
        fn()
    cpu = time.process_time() - t0
    print(f"  cpu {cpu:.1f}s")
    print("[foundation_perception_smoke] ALL PASS")


if __name__ == "__main__":
    main()
