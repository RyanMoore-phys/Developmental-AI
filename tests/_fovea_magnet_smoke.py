"""Fovea channel + gaze-righting smoke (Mac-ok; fake query_fn, small torch).

Contracts:
  1. _crop_center handles flat-CHW-float and HWC-uint8 frames and returns the
     central patch.
  2. Submission ALTERNATES full -> fovea -> full; a foveal label trains ONLY
     the fovea head, increments fovea counts, and collect() returns None for
     it (no fact minting from a crop).
  3. fovea_probs() learns the labels: after training on "trunk under gaze"
     latents vs "sky under gaze" latents, P(tree|trunk-latent) >
     P(tree|sky-latent).
  4. Magnet gating: _fovea_p is None below min_labels; with counts, phi's cen
     channel uses the foveal P so approaching-while-centred pays a positive
     potential delta pre-contact (reach still 0).
  5. Gaze righting: pitch clamped at -90 -> level pays positive seek delta
     once; away-and-back telescopes to ~0; disabled flag is inert.
"""
import json

import numpy as np
import torch

from developmental_ai.llm.vlm_symbolizer import (
    FOVEA_PREDICATES, VLMSymbolizer)
from developmental_ai.llm.vision_scaffold import VisionScaffold


def _drain(sym, timeout=5.0):
    """Wait for the single async worker to finish, then poll."""
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = sym.collect()
        if out is not None or not sym._channel.busy():
            # one more poll: done-but-unpolled still needs a collect()
            return out if out is not None else sym.collect()
        time.sleep(0.01)
    raise AssertionError("worker never finished")


def test_crop_center():
    hwc = np.zeros((128, 128, 3), np.uint8)
    hwc[30:98, 30:98] = 200                      # bright centre (covers crop)
    c = VLMSymbolizer._crop_center(hwc, 0.4)
    assert c.shape == (51, 51, 3) and c.mean() > 150, c.shape
    flat = (hwc.transpose(2, 0, 1).astype(np.float32) / 255.0).ravel()
    c2 = VLMSymbolizer._crop_center(flat, 0.4)
    assert c2.shape == (51, 51, 3) and c2.mean() > 150, c2.shape
    print("  1. crop_center ok on HWC-uint8 and flat-CHW-float")


class _FakeVLM:
    """Answers the FULL battery for big images and the fovea battery for
    crops, keying 'trunk in view' off the frame brightness the test set."""

    def __init__(self):
        self.calls = []

    def __call__(self, png_bytes):
        import io
        import imageio.v2 as imageio
        img = imageio.imread(io.BytesIO(png_bytes))
        # _encode_png upscales only frames SMALLER than 128px: the full
        # 128px frame passes through at 128, the 51px crop becomes 255.
        is_full = img.shape[0] == 128
        trunk = bool(img.mean() > 100)
        self.calls.append(("full" if is_full else "fovea", trunk))
        if is_full:
            return json.dumps({"tree_visible": trunk, "sky_visible": True,
                               "grass_visible": True})
        return json.dumps({"tree_visible": trunk, "leaves_visible": False,
                           "sky_visible": not trunk, "grass_visible": False})


def _mk_sym(fake):
    return VLMSymbolizer(model="fake", interval=1, latent_dim=16,
                         hidden_dim=32, enabled=True, query_fn=fake,
                         fovea=True, fovea_frac=0.4,
                         device=torch.device("cpu"))


def test_alternation_and_routing():
    fake = _FakeVLM()
    sym = _mk_sym(fake)
    bright = np.full((128, 128, 3), 220, np.uint8)     # trunk frame
    lat = torch.randn(1, 16)
    # cycle 1: FULL
    sym.maybe_label(bright, lat, 0)
    out1 = _drain(sym)
    assert out1 is not None and out1.get("tree_visible") is True, out1
    assert sym.total_labels == 1 and sym.total_fovea_labels == 0
    # cycle 2: FOVEA — collect() must return None (no fact path) but count it
    sym.maybe_label(bright, lat, 10)
    out2 = _drain(sym)
    assert out2 is None, f"foveal label leaked to the fact path: {out2}"
    assert sym.total_fovea_labels == 1, sym.total_fovea_labels
    assert sym.fovea_label_counts["tree_visible"] == 1
    # cycle 3: back to FULL
    sym.maybe_label(bright, lat, 20)
    out3 = _drain(sym)
    assert out3 is not None and sym.total_labels == 2
    kinds = [k for k, _ in fake.calls]
    assert kinds == ["full", "fovea", "full"], kinds
    print(f"  2. alternation + routing ok: {kinds}, "
          f"fovea_labels={sym.total_fovea_labels}")


def test_fovea_head_learns():
    fake = _FakeVLM()
    sym = _mk_sym(fake)
    torch.manual_seed(0)
    trunk_lat = torch.ones(1, 16)
    sky_lat = -torch.ones(1, 16)
    bright = np.full((128, 128, 3), 220, np.uint8)
    dark = np.full((128, 128, 3), 30, np.uint8)
    t = 0
    for _ in range(30):
        # alternate: full(bright) -> fovea(bright/trunk) -> full(dark)
        # -> fovea(dark/sky) ...
        sym.maybe_label(bright, trunk_lat, t); _drain(sym); t += 10
        sym.maybe_label(bright, trunk_lat, t); _drain(sym); t += 10
        sym.maybe_label(dark, sky_lat, t); _drain(sym); t += 10
        sym.maybe_label(dark, sky_lat, t); _drain(sym); t += 10
    pt = sym.fovea_probs(trunk_lat)["tree_visible"]
    ps = sym.fovea_probs(sky_lat)["tree_visible"]
    assert pt > 0.7 > 0.3 > ps, (pt, ps)
    assert sym.fovea_label_counts["tree_visible"] >= 5
    print(f"  3. fovea head learned: P(tree|trunk)={pt:.2f} "
          f"P(tree|sky)={ps:.2f} counts={sym.fovea_label_counts['tree_visible']}")


def _mk_vs(**kw):
    cats = ["tree_visible"]
    args = dict(target_categories=cats, min_labels=5, present_threshold=0.5,
                min_dwell=2, eps_abs=1e-3, instinct_bonus=0.0,
                approach_pull=0.0, align_bonus=0.0, aim_bonus=0.0,
                phi_from_evidence=True, seek_weight=0.5,
                seek_categories=cats, cold_start_weight=0.35,
                cold_start_budget=10**8, weight=0.6)
    args.update(kw)
    return VisionScaffold(**args)


REL = {"tree_visible": 1.0}
LAB = {"tree_visible": 99}
PROBS = {"tree_visible": 0.9, "leaves_visible": 0.2, "grass_visible": 0.2}


def test_fovea_gating_and_phi_gradient():
    vs = _mk_vs()
    few = {"tree_visible": 2}                     # below min_labels
    vs.step_shaping(0, 0, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                    fovea_probs={"tree_visible": 0.9}, fovea_counts=few)
    assert vs._fovea_p("tree_visible") is None, "ungated fovea leaked"
    # grounded fovea: rising foveal P with reach still 0 must pay a positive
    # phi delta (the pre-contact centring gradient that did not exist)
    cnt = {"tree_visible": 10}
    r = []
    for i, fp in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        r.append(vs.step_shaping(1, 1 + i, 0.0, PROBS, REL, LAB,
                                 reach_measured=0.0,
                                 fovea_probs={"tree_visible": fp},
                                 fovea_counts=cnt))
    assert sum(r[1:]) > 0.01, f"centring paid nothing pre-contact: {r}"
    # and telescoping: swing the fovea away and back -> net ~0
    away = vs.step_shaping(1, 6, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                           fovea_probs={"tree_visible": 0.1},
                           fovea_counts=cnt)
    back = vs.step_shaping(1, 7, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                           fovea_probs={"tree_visible": 0.9},
                           fovea_counts=cnt)
    assert abs(away + back) < 1e-6, (away, back)
    print(f"  4. fovea gating + phi gradient ok: climb={sum(r[1:]):+.4f}, "
          f"away+back={away + back:+.6f}")


def test_gaze_righting():
    vs = _mk_vs(seek_pitch_level=True)
    cnt = {"tree_visible": 10}
    fv = {"tree_visible": 0.0}
    # adopt baseline at the sky clamp
    vs.step_shaping(0, 0, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                    fovea_probs=fv, fovea_counts=cnt, pitch=-90.0)
    # coming level pays positive once...
    r_level = vs.step_shaping(0, 1, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                              fovea_probs=fv, fovea_counts=cnt, pitch=0.0)
    assert r_level > 0.01, f"righting paid nothing: {r_level}"
    # ...dwelling level pays 0...
    r_dwell = vs.step_shaping(0, 2, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                              fovea_probs=fv, fovea_counts=cnt, pitch=0.0)
    assert abs(r_dwell) < 1e-9, r_dwell
    # ...and clamp-and-back telescopes to ~0 (no farm in either direction)
    r_up = vs.step_shaping(0, 3, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                           fovea_probs=fv, fovea_counts=cnt, pitch=-90.0)
    r_dn = vs.step_shaping(0, 4, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                           fovea_probs=fv, fovea_counts=cnt, pitch=0.0)
    assert abs(r_up + r_dn) < 1e-9, (r_up, r_dn)
    # flag off -> pitch is inert
    vs2 = _mk_vs(seek_pitch_level=False)
    vs2.step_shaping(0, 0, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                     pitch=-90.0)
    r2 = vs2.step_shaping(0, 1, 0.0, PROBS, REL, LAB, reach_measured=0.0,
                          pitch=0.0)
    assert abs(r2) < 1e-9, f"disabled righting still paid: {r2}"
    print(f"  5. gaze righting ok: level={r_level:+.4f}, dwell={r_dwell:+.6f}, "
          f"cycle={r_up + r_dn:+.6f}, disabled={r2:+.6f}")


def test_gui_not_steerable():
    """2026-08-07: screen-state predicates must never be magnet targets
    (measured: the magnet courted the inventory for 55k floor-pulled steps,
    28% gui-open). A steerable set derived from `_visible` must exclude
    them the same way it excludes holding_tool."""
    from developmental_ai.llm.vision_scaffold import steerable_targets
    from developmental_ai.llm.vlm_symbolizer import PREDICATES
    s = set(steerable_targets(PREDICATES))
    for gui in ("inventory_visible", "crafting_grid_visible",
                "craft_output_visible"):
        assert gui not in s, f"{gui} is steerable — the menu-magnet returns"
    assert "tree_visible" in s and "zombie_visible" in s, s
    print(f"  6. GUI predicates non-steerable; {len(s)} world objects remain")


def test_stale_lp_discount():
    """A ghost category re-appearing after a long absence must re-earn its
    curiosity instead of cashing a frozen months-old EMA."""
    def _run(tau):
        vs = _mk_vs(target_categories=["tree_visible", "water_visible"],
                    seek_weight=0.0, ema_leak=0.0, lp_stale_tau=tau,
                    ema_beta=0.1)
        rel = {"tree_visible": 1.0, "water_visible": 1.0}
        lab = {"tree_visible": 99, "water_visible": 99}
        t = 0
        # water present + high LP -> its EMA climbs well above global
        for _ in range(50):
            vs.step_shaping(0, t, 1.0, {"water_visible": 0.9}, rel, lab)
            t += 1
        hi = vs._cat_lp["water_visible"]
        # water gone for 10k low-LP steps: EMA frozen (leak=0), global sinks
        for _ in range(10000):
            vs.step_shaping(0, t, 0.0, {"tree_visible": 0.9}, rel, lab)
            t += 1
        assert abs(vs._cat_lp["water_visible"] - hi) < 1e-9, "leak crept in"
        # water re-appears for ONE step
        vs.step_shaping(0, t, 0.0, {"water_visible": 0.9}, rel, lab)
        return vs._cat_lp["water_visible"], vs._global_lp
    ghost, g = _run(tau=0.0)          # control: old behaviour, fossil kept
    fresh, g2 = _run(tau=1000.0)      # 10k absence >> tau -> re-opens neutral
    assert ghost > g + 0.3, f"control lost the fossil: {ghost} vs {g}"
    assert fresh < g2 + 0.05, f"discount failed: {fresh} vs global {g2}"
    print(f"  7. stale-LP discount: fossil {ghost:.3f} -> neutral {fresh:.4f} "
          f"(global {g2:.4f})")


def test_nudge_regen():
    """A burned search budget must trickle back (measured stuck at 0 for
    10h+); sighting-refill semantics untouched."""
    vs = _mk_vs(seek_forward_nudge=0.02, seek_nudge_regen=5)
    vs._seek_nudge_left = 0
    for i in range(10):               # goal absent; noop -> nudge not burned
        vs.step_shaping(0, i, 0.0, {}, REL, LAB)
    assert vs._seek_nudge_left == 2, vs._seek_nudge_left
    # ...and a regenerated nudge is immediately SPENDABLE: a forward step
    # while the goal is off-screen earns the nudge again (search resumed —
    # this is the whole point of regen)
    r = vs.step_shaping(1, 10, 0.0, {}, REL, LAB)
    assert r > 0.0 and vs._seek_nudge_left == 1, (r, vs._seek_nudge_left)
    vs2 = _mk_vs(seek_forward_nudge=0.02, seek_nudge_regen=0)
    vs2._seek_nudge_left = 0
    for i in range(10):
        vs2.step_shaping(0, i, 0.0, {}, REL, LAB)
    assert vs2._seek_nudge_left == 0, "regen=0 must keep the old latch"
    print("  8. nudge regen: 0 -> 2 idle, forward step spends one (+pays); "
          "off stays latched")


def test_negative_evidence():
    """A fruitless full-length attack streak docks ONLY the predicates the
    fresh label actually claimed; a stale label is not evidence."""
    sym = _mk_sym(_FakeVLM())
    sym.last_labels = {"breakable_in_reach": True, "object_centered": False,
                       "object_adjacent": True}
    sym._last_full_submit = 100
    r0 = dict(sym.reliability)
    n = sym.observe_negative_event(now=102)  # fresh (window = 2*interval*2)
    assert n == 2, n
    assert sym.reliability["breakable_in_reach"] < r0["breakable_in_reach"]
    assert sym.reliability["object_adjacent"] < r0["object_adjacent"]
    assert sym.reliability["object_centered"] == r0["object_centered"], \
        "unclaimed predicate was docked — a miss does not contradict it"
    r1 = dict(sym.reliability)
    assert sym.observe_negative_event(now=99999) == 0, "stale label scored"
    assert sym.reliability == r1
    print(f"  9. negative evidence: 2 claimed docked "
          f"(breakable {r0['breakable_in_reach']:.2f}->"
          f"{sym.reliability['breakable_in_reach']:.2f}), "
          f"unclaimed + stale untouched")


def test_fovea_cadence_decoupled():
    """The fovea keeps its OWN fixed cadence while the full channel anneals
    wide (measured: inherited annealing starved the fovea to one label per
    ~908 steps)."""
    fake = _FakeVLM()
    sym = VLMSymbolizer(model="fake", interval=10, latent_dim=16,
                        hidden_dim=32, enabled=True, query_fn=fake,
                        fovea=True, fovea_interval=10,
                        device=torch.device("cpu"))
    sym.interval = 500                    # simulate a heavily annealed full channel
    bright = np.full((128, 128, 3), 220, np.uint8)
    lat = torch.randn(1, 16)
    for t in range(0, 600, 10):
        sym.maybe_label(bright, lat, t)
        _drain(sym)
    kinds = [k for k, _ in fake.calls]
    n_fov, n_full = kinds.count("fovea"), kinds.count("full")
    # 600 steps: fovea due every 10 -> dozens of labels; full due at ~0 and
    # ~500 -> a couple. The fovea must dominate but the full channel must
    # never starve.
    assert n_fov >= 30 and 1 <= n_full <= 5, (n_fov, n_full)
    print(f"  10. fovea cadence decoupled: {n_fov} fovea vs {n_full} full "
          f"labels under an annealed-wide full interval")


def _loop_cls():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    return DevelopmentalAI


class _Bare:
    pass


def test_habituation_factor():
    """Mastered-type breaks damp the step intrinsic; novel types keep full
    surprise; no-break steps and first-sighting baselines are untouched."""
    DL = _loop_cls()
    s = _Bare()
    s._habituation_scale = 50.0
    s._habit_prev = {}
    bbt = {"dirt": 400, "oak_log": 0}
    # first sighting = baseline, not a break
    f0 = DL._habituation_factor(
        s, {"achievements": {"mine_dirt": 5}, "breaks_by_type": bbt})
    assert f0 == 1.0, f0
    # dirt (400 lifetime) breaks again -> heavy habituation (1/(1+8)=0.111)
    f1 = DL._habituation_factor(
        s, {"achievements": {"mine_dirt": 6}, "breaks_by_type": bbt})
    assert 0.1 <= f1 < 0.13, f1
    # no break this step -> 1.0
    f2 = DL._habituation_factor(
        s, {"achievements": {"mine_dirt": 6}, "breaks_by_type": bbt})
    assert f2 == 1.0, f2
    # a log's FIRST counter appearance is baseline (counters only appear
    # once >0, and a crash-rebuilt client boots mid-count — a phantom break
    # at boot would damp a random step), so this step sees only the dirt...
    f3a = DL._habituation_factor(
        s, {"achievements": {"mine_dirt": 7, "mine_oak_log": 1},
            "breaks_by_type": bbt})
    assert 0.1 <= f3a < 0.13, f3a
    # ...and from the next log onward the least-familiar type governs: the
    # near-novel log keeps the step's surprise despite the mastered dirt.
    f3 = DL._habituation_factor(
        s, {"achievements": {"mine_dirt": 8, "mine_oak_log": 2},
            "breaks_by_type": {"dirt": 400, "oak_log": 1}})
    assert f3 > 0.9, f3
    # PLACEMENT events habituate identically (2026-08-08 — the pillar farm:
    # with breaks defunded, placing dirt was the last un-habituated way to
    # manufacture frame-change novelty)
    fp1 = DL._habituation_factor(
        s, {"achievements": {}, "placed_now": ["dirt"],
            "places_by_type": {"dirt": 500}})
    assert 0.1 <= fp1 < 0.12, fp1
    fp2 = DL._habituation_factor(
        s, {"achievements": {}, "placed_now": ["torch"],
            "places_by_type": {"torch": 0}})
    assert fp2 == 1.0, fp2
    # off switch
    s2 = _Bare(); s2._habituation_scale = 0.0; s2._habit_prev = {}
    assert DL._habituation_factor(s2, {"achievements": {}}) == 1.0
    print(f"  11. habituation: baseline {f0}, mastered-break {f1:.3f}, "
          f"no-event {f2}, novel-with-mastered {f3:.2f}, "
          f"mastered-place {fp1:.3f}, novel-place {fp2}")


def test_boring_view_factor():
    """Sky and mastered-material fovea views pay reduced novelty; trees and
    ungated categories never do."""
    DL = _loop_cls()
    s = _Bare()
    s._MASTERY_BLOCKS = DL._MASTERY_BLOCKS
    s._habituation_scale = 50.0
    s.symbolizer = None
    s._last_env_info = {"breaks_by_type": {"dirt": 400}}
    cnt = {k: 10 for k in ("sky_visible", "dirt_visible", "tree_visible",
                           "grass_visible", "stone_visible",
                           "leaves_visible")}
    # sky-filled gaze
    s._last_fovea_probs = {"sky_visible": 0.95}
    s._last_fovea_counts = cnt
    f_sky = DL._boring_view_factor(s)
    assert f_sky <= 0.16, f_sky
    # mastered-dirt-filled gaze (the fresh-hole farm)
    s._last_fovea_probs = {"dirt_visible": 0.9, "sky_visible": 0.0}
    f_dirt = DL._boring_view_factor(s)
    assert f_dirt < 0.3, f_dirt
    # trunk-filled gaze: never discounted
    s._last_fovea_probs = {"tree_visible": 0.9, "sky_visible": 0.0}
    f_tree = DL._boring_view_factor(s)
    assert f_tree == 1.0, f_tree
    # ungated fovea -> inert
    s._last_fovea_probs = {"dirt_visible": 0.9}
    s._last_fovea_counts = {}
    assert DL._boring_view_factor(s) == 1.0
    # MEASURED PITCH FALLBACK (2026-08-08): at either clamp the discount
    # applies BY GEOMETRY, independent of what the fovea head believes —
    # including when the fovea has no probs at all (cold boot). SYMMETRIC:
    # with the up-clamp defunded the bot moved to the DOWN clamp (mean
    # +76.7deg, 48.5% pinned staring at its feet), so both extremes are
    # boring; only the horizon band keeps full novelty.
    s._last_world_info = {"pitch": -90.0}
    s._last_fovea_probs = {}
    f_up = DL._boring_view_factor(s)
    assert f_up == 0.15, f_up
    s._last_world_info = {"pitch": 0.0}
    assert DL._boring_view_factor(s) == 1.0
    s._last_world_info = {"pitch": 90.0}
    f_dn = DL._boring_view_factor(s)
    assert f_dn == 0.15, f_dn
    s._last_world_info = {"pitch": 30.0}       # mild down-scan: untouched
    assert DL._boring_view_factor(s) == 1.0
    print(f"  12. boring-view: sky {f_sky:.2f}, mastered-dirt {f_dirt:.2f}, "
          f"trunk {f_tree:.2f}, ungated 1.0, clamps up/down "
          f"{f_up:.2f}/{f_dn:.2f}, horizon 1.0")


def test_break_memory_persistence():
    """breaks_by_type round-trips through the atomic save file."""
    import json
    import os
    import tempfile
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter
    d = tempfile.mkdtemp()
    path = os.path.join(d, "breaks.json")
    a = _Bare()
    a._break_memory_path = path
    a._break_mem_dirty = 0
    a._breaks_by_type = {"dirt": 123, "oak_log": 4}
    a._places_by_type = {"dirt": 55}
    for _ in range(20):                       # throttle: flushes on the 20th
        MineRLEnvAdapter._save_break_memory(a)
    assert os.path.exists(path), "memory never flushed"
    assert json.load(open(path)) == {
        "breaks": {"dirt": 123, "oak_log": 4}, "places": {"dirt": 55}}
    print("  13. break memory: atomic save (breaks+places) + round-trip ok")


if __name__ == "__main__":
    for fn in (test_crop_center, test_alternation_and_routing,
               test_fovea_head_learns, test_fovea_gating_and_phi_gradient,
               test_gaze_righting, test_gui_not_steerable,
               test_stale_lp_discount, test_nudge_regen,
               test_negative_evidence, test_fovea_cadence_decoupled,
               test_habituation_factor, test_boring_view_factor,
               test_break_memory_persistence):
        print(f"[fovea-magnet-smoke] {fn.__name__}")
        fn()
    print("[fovea-magnet-smoke] ALL PASS")
