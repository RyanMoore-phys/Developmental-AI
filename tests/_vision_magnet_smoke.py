"""Curiosity-ranked magnet smoke (Mac-ok; stubbed head probs, no Ollama/torch).

Contracts:
  1. Retargets when the top learning-progress category changes.
  2. Weight FADES to ~0 when the target is mastered (LP->0) and RE-ARMS when a
     novel object returns LP (this pins the wants_step != _w>0 decision).
  3. Head stubbed / no symbolizer: object_probs=None -> 0.0, no exception.
  4. Disabled is inert; the contrastive baseline blocks an always-present
     background predicate from ever becoming the target.
"""
from developmental_ai.llm.vision_scaffold import VisionScaffold


def _rel(cats):
    return {c: 1.0 for c in cats}


def _lab(cats):
    return {c: 5 for c in cats}


ADJ = {"object_adjacent": 0.9, "object_centered": 0.9}


def _feed(vs, cat, n, start, rel, lab, lp=1.0):
    """INTERMITTENT presence correlated with LP: on even steps `cat` is in
    view AND the agent earns lp; on odd steps nothing salient is in view and
    lp=0. This is what makes cat_lp[cat] > global_lp (the contrastive signal
    the magnet is designed for). The FINAL step is forced present so the
    steering target is engaged when we assert (the target is released the
    instant the object leaves the frame — correct, but not what we probe)."""
    for i in range(n):
        t = start + i
        present = (i % 2 == 0) or (i == n - 1)
        if present:
            vs.step_shaping(0, t, lp, {cat: 0.9, **ADJ}, rel, lab)
        else:
            vs.step_shaping(0, t, 0.0, {}, rel, lab)


def test_retarget_on_lp_shift():
    cats = ["tree_visible", "stone_visible"]
    vs = VisionScaffold(target_categories=cats, min_labels=1,
                        present_threshold=0.5, min_dwell=2,
                        retarget_margin=0.2, eps_abs=1e-6, ema_beta=0.1,
                        ema_leak=0.02)
    rel, lab = _rel(cats), _lab(cats)
    # tree correlates with LP -> becomes the target
    _feed(vs, "tree_visible", 600, 0, rel, lab, lp=1.0)
    assert vs._target == "tree_visible", vs.stats
    assert vs._score("tree_visible") > 0.05, vs.stats
    # curiosity shifts: stone now correlates with LP; tree no longer appears
    _feed(vs, "stone_visible", 800, 600, rel, lab, lp=1.0)
    assert vs._target == "stone_visible", (
        f"never retargeted to the newly-curious object: {vs.stats}")
    print(f"  1. retarget ok (tree -> stone as curiosity shifted): "
          f"{vs._target}")


def test_fade_and_rearm():
    cats = ["tree_visible"]
    vs = VisionScaffold(target_categories=cats, min_labels=1,
                        present_threshold=0.5, eps_abs=1e-4, ema_beta=0.1,
                        ema_leak=0.02)
    rel, lab = _rel(cats), _lab(cats)
    _feed(vs, "tree_visible", 600, 0, rel, lab, lp=1.0)  # novel -> engaged
    assert vs.current_weight() > 0.0, "magnet never engaged on a novel object"
    # mastered: tree still in view intermittently but yields NO more LP
    _feed(vs, "tree_visible", 4000, 600, rel, lab, lp=0.0)
    assert vs.current_weight() < 1e-3, (
        f"magnet did not fade when curiosity quenched: {vs.current_weight()}")
    assert vs.wants_step() is True, "stopped running while faded -> can't re-arm"
    _feed(vs, "tree_visible", 800, 4600, rel, lab, lp=1.0)  # novelty returns
    assert vs.current_weight() > 0.0, (
        "magnet did NOT re-arm when a novel object returned")
    print("  2. fade + re-arm ok (curiosity-driven weight, not a clock)")


def test_head_stubbed_and_none():
    vs = VisionScaffold(min_labels=1)
    # object_probs=None -> inert, no exception
    assert vs.step_shaping(0, 0, 1.0, object_probs=None) == 0.0
    # missing reliability/label_counts default safely
    r = vs.step_shaping(0, 1, 1.0, object_probs={"tree_visible": 0.9})
    assert r == 0.0  # no reliability/labels -> not trusted-present
    print("  3. head-stubbed / None-probs safe (0.0, no throw)")


def test_disabled_and_deconfound():
    # disabled -> fully inert
    vs = VisionScaffold(enabled=False, min_labels=1)
    assert vs.wants_step() is False
    assert vs.step_shaping(0, 0, 1.0, object_probs={"tree_visible": 0.9},
                           reliability={"tree_visible": 1.0},
                           label_counts={"tree_visible": 5}) == 0.0
    # anti-co-occurrence: a background predicate present EVERY step with LP
    # must never become the target (not in target_categories), and an
    # always-present interactable keeps s~0 via the contrastive baseline.
    cats = ["tree_visible", "stone_visible"]
    vs2 = VisionScaffold(target_categories=cats, min_labels=1,
                         present_threshold=0.5, eps_abs=1e-4, ema_beta=0.05)
    rel = {"tree_visible": 1.0, "stone_visible": 1.0, "sky_visible": 1.0}
    lab = {"tree_visible": 5, "stone_visible": 5, "sky_visible": 5}
    # sky always present (background, not a target); stone intermittently
    for t in range(400):
        p = {"sky_visible": 1.0, "tree_visible": 0.9}    # tree always present
        vs2.step_shaping(0, t, 1.0, p, rel, lab)
    # tree is present WHENEVER lp>0, so cat_lp[tree] ~ global_lp -> s~0
    assert vs2._score("tree_visible") < 0.05, (
        f"always-present interactable won by co-occurrence: {vs2.stats}")
    assert "sky_visible" != vs2._target  # never a background target
    print("  4. disabled inert + contrastive de-confounding ok")


def test_percategory_scale_no_cross_suppression():
    """MF-1: an early high-LP category must NOT cross-scale-suppress a later,
    lower-magnitude novel category. Per-category _lp_scale means the novel
    object arms from ITS OWN recent peak, not a global historical maximum."""
    cats = ["tree_visible", "stone_visible"]
    vs = VisionScaffold(target_categories=cats, min_labels=1, weight=0.6,
                        present_threshold=0.5, eps_abs=1e-4, ema_beta=0.1,
                        ema_leak=0.02, min_dwell=2)
    rel, lab = _rel(cats), _lab(cats)
    _feed(vs, "tree_visible", 600, 0, rel, lab, lp=1.0)      # tree dominates
    tree_scale = vs._lp_scale.get("tree_visible", 0.0)
    assert tree_scale > 0.1, vs.stats                        # its scale is high
    # tree gone; a NOVEL lower-LP object appears -> must arm near full strength
    _feed(vs, "stone_visible", 800, 600, rel, lab, lp=0.3)
    assert vs._target == "stone_visible", vs.stats
    w = vs.current_weight()
    assert w > 0.8 * 0.6, (
        f"novel lower-LP object crushed by a stale global scale: w={w} "
        f"(tree_scale={tree_scale}, {vs.stats})")
    print("  5. per-category scale: novel lower-LP object arms at full strength")


def test_flicker_tolerance_no_thrash():
    """MF-2: a single-step dip below present_threshold (raw head-sigmoid noise)
    must NOT release the target and thrash onto a lower-LP category; a SUSTAINED
    absence still releases."""
    cats = ["tree_visible", "stone_visible"]
    vs = VisionScaffold(target_categories=cats, min_labels=1,
                        present_threshold=0.6, min_dwell=25, absent_tolerance=3,
                        eps_abs=1e-4, ema_beta=0.1, ema_leak=0.02)
    rel, lab = _rel(cats), _lab(cats)
    _feed(vs, "tree_visible", 400, 0, rel, lab, lp=1.0)      # tree is the target
    assert vs._target == "tree_visible", vs.stats
    both = {"tree_visible": 0.9, "stone_visible": 0.9, **ADJ}
    stoneonly = {"stone_visible": 0.9, **ADJ}
    # give stone a positive contrastive score so it's a tempting alternative
    for t in range(400, 470):
        vs.step_shaping(0, t, 0.6, both, rel, lab)
    assert vs._target == "tree_visible", f"lost tree before flicker: {vs.stats}"
    # tree FLICKERS below present_threshold for 2 steps (stone present, scoring)
    vs.step_shaping(0, 470, 0.6, {"tree_visible": 0.59, "stone_visible": 0.9,
                                  **ADJ}, rel, lab)
    assert vs._target == "tree_visible", "single-step flicker thrashed the target"
    vs.step_shaping(0, 471, 0.6, {"tree_visible": 0.59, "stone_visible": 0.9,
                                  **ADJ}, rel, lab)
    assert vs._target == "tree_visible", "2-step flicker thrashed the target"
    vs.step_shaping(0, 472, 0.6, both, rel, lab)             # tree returns
    assert vs._target == "tree_visible", "target lost after flicker recovered"
    # a SUSTAINED absence (>= tolerance) does release
    for t in range(473, 480):
        vs.step_shaping(0, t, 0.6, stoneonly, rel, lab)
    assert vs._target != "tree_visible", "sustained absence failed to release"
    print("  6. flicker tolerance: brief dips hold target; sustained absence frees")


def test_antiphase_no_lockout():
    """MF-2 follow-up: an object that is genuinely MORE curious but visible
    only on the incumbent-absent frames must be able to PREEMPT the held
    incumbent (not be permanently locked out by the absence tolerance)."""
    cats = ["tree_visible", "stone_visible"]
    vs = VisionScaffold(target_categories=cats, min_labels=1,
                        present_threshold=0.6, min_dwell=5, absent_tolerance=3,
                        retarget_margin=0.2, eps_abs=1e-4, ema_beta=0.15,
                        ema_leak=0.02)
    rel, lab = _rel(cats), _lab(cats)
    _feed(vs, "tree_visible", 200, 0, rel, lab, lp=0.3)      # tree incumbent
    assert vs._target == "tree_visible", vs.stats
    tree_only = {"tree_visible": 0.9, **ADJ}
    stone_only = {"stone_visible": 0.9, **ADJ}
    for t in range(200, 900):                                # strict anti-phase
        if t % 2 == 0:
            vs.step_shaping(0, t, 0.05, tree_only, rel, lab)  # tree, low LP
        else:
            vs.step_shaping(0, t, 1.0, stone_only, rel, lab)  # stone, HIGH LP
    assert vs._target == "stone_visible", (
        f"anti-phase lockout: stayed on tree, not the more-curious stone: "
        f"{vs.stats}")
    print("  7. anti-phase: a more-curious alternate-frame object preempts")


def test_cold_start_bootstrap():
    """Cold-start instinct: with NO learning-progress differentiation yet
    (contrastive score 0 — the live-run deadlock), the magnet must STILL pull
    toward a trusted-present interactable to bootstrap the first interaction,
    prefer tree over a co-present stone, and hand over to the contrastive
    weight once tree becomes distinctively curious."""
    cats = ["tree_visible", "stone_visible"]
    vs = VisionScaffold(target_categories=cats, min_labels=1, weight=0.6,
                        cold_start_weight=0.12, present_threshold=0.5,
                        eps_abs=1e-3, ema_beta=0.1, ema_leak=0.02)
    rel, lab = _rel(cats), _lab(cats)
    both = {"tree_visible": 0.9, "stone_visible": 0.9, **ADJ}
    for t in range(30):                       # uniform LP, both present -> s*=0
        vs.step_shaping(0, t, 0.3, both, rel, lab)
    assert vs._target == "tree_visible", f"cold start didn't prefer tree: {vs.stats}"
    assert abs(vs.current_weight() - 0.12) < 1e-6, (
        f"cold-start floor did not engage (was the w=0 deadlock): "
        f"{vs.current_weight()}")
    # tree now becomes distinctively curious -> contrastive should take over
    for t in range(30, 400):
        if t % 2 == 0:
            vs.step_shaping(0, t, 1.0, {"tree_visible": 0.9, **ADJ}, rel, lab)
        else:
            vs.step_shaping(0, t, 0.0, {}, rel, lab)
    r = vs.step_shaping(0, 400, 1.0, {"tree_visible": 0.9, **ADJ}, rel, lab)
    assert vs._ever_curious.get("tree_visible"), "tree never marked engaged"
    assert vs.current_weight() > 0.12, (
        f"contrastive weight did not take over from the floor: "
        f"{vs.current_weight()}")
    print("  8. cold-start bootstrap: floor pulls toward tree, contrastive "
          "takes over once curious")


def test_trunk_focus_gates_instinct():
    """Lever 2: the chop instinct pays only when the TARGET (trunk) is present
    AND at least as prominent as its foliage distractors — chopping while leaves
    dominate the frame pays nothing (so the agent is rewarded for the log, not
    the leaves it was farming)."""
    cats = ["tree_visible"]
    vs = VisionScaffold(target_categories=cats, min_labels=1, weight=0.6,
                        cold_start_weight=0.2, instinct_bonus=0.5,
                        chop_actions=[5], present_threshold=0.5, eps_abs=1e-4,
                        focus_distractors=["leaves_visible"], ema_beta=0.1)
    rel = {"tree_visible": 1.0, "leaves_visible": 1.0}
    lab = {"tree_visible": 5, "leaves_visible": 5}
    FOC = {"object_centered": 0.9, "object_adjacent": 0.9}
    for t in range(20):                                  # engage (cold-start w>0)
        vs.step_shaping(0, t, 0.0, {"tree_visible": 0.9, **FOC}, rel, lab)
    assert vs.current_weight() > 0.0
    # chop while the TRUNK dominates (leaves low) -> instinct pays
    r_trunk = vs.step_shaping(5, 21, 0.0,
                              {"tree_visible": 0.9, "leaves_visible": 0.1, **FOC},
                              rel, lab)
    # chop while LEAVES dominate (trunk present but less prominent) -> no instinct
    r_leaf = vs.step_shaping(5, 22, 0.0,
                             {"tree_visible": 0.55, "leaves_visible": 0.95, **FOC},
                             rel, lab)
    assert r_trunk > 0.0, f"instinct should pay when trunk dominates: {r_trunk}"
    # UPDATED 2026-07-25: was `abs(r_leaf) < 1e-9`. phi is now TARGET-FOCUSED
    # (it drops when a distractor dominates), so shifting trunk -> leaf is a
    # real potential DESCENT and is charged. That satisfies this assertion's
    # intent ("a leaf-dominant chop must not pay") more strongly than 0 did:
    # the agent is now actively discouraged from chopping foliage, which is
    # the discrimination that was lost when the farmable instinct_bonus was
    # zeroed (live: dirt/fern/grass/leaves broken, almost no logs).
    assert r_leaf <= 0.0, f"leaf-dominant chop must not PAY: {r_leaf}"
    # ...and the charge TELESCOPES — re-focusing on the trunk returns it, so
    # there is no net drift and no way to farm either sign.
    r_back = vs.step_shaping(5, 23, 0.0,
                             {"tree_visible": 0.9, "leaves_visible": 0.1,
                              **FOC}, rel, lab)
    assert r_leaf + r_back > -1e-9, (
        f"trunk->leaf->trunk must net >= 0 (telescoping), got "
        f"{r_leaf:+.4f} then {r_back:+.4f}")
    print("  9. trunk-focus gate: instinct pays for chopping the trunk, not leaves")


if __name__ == "__main__":
    for fn in (test_retarget_on_lp_shift, test_fade_and_rearm,
               test_head_stubbed_and_none, test_disabled_and_deconfound,
               test_percategory_scale_no_cross_suppression,
               test_flicker_tolerance_no_thrash, test_antiphase_no_lockout,
               test_cold_start_bootstrap, test_trunk_focus_gates_instinct):
        print(f"[vision-magnet-smoke] {fn.__name__}")
        fn()
    print("[vision-magnet-smoke] ALL PASS")
