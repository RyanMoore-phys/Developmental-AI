"""Goal-SEEKING drive smoke (Mac-ok; stubbed head probs, no Ollama/torch).

The seek drive un-strands the fleet when the goal object (a tree) is off-screen:
approach/align/aim are inert with nothing in view, and lifelong mode never resets
the world to re-place the agent among trees.

Contracts:
  1. Potential on goal-visibility: reward tracks the RISE in tree-prob and is
     telescoping (a look-away/look-back round trip nets ~0 -> unfarmable).
  2. Inert once the goal is trusted-present (normal approach owns it) and the
     search budget refills.
  3. Forward nudge is BUDGETED: a bounded search burst while the goal is fully
     off-screen, then it stops (no runaway per-step exploration reward).
  4. Trust gate: an untrusted (low-reliability) head prob never drives seeking.
  5. OFF by default (seek_weight=0, nudge=0) -> identical to the old magnet.
"""
from developmental_ai.llm.vision_scaffold import VisionScaffold

REL = {"tree_visible": 1.0}
LAB = {"tree_visible": 5}
ADJ = {"object_adjacent": 0.9, "object_centered": 0.9}


def test_seek_potential_telescopes():
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=1,
                        present_threshold=0.6, seek_weight=0.5,
                        seek_forward_nudge=0.0, weight=0.6)
    # all probs < present_threshold -> tree never "present" -> main path w=0, so
    # the returned reward IS the seek term in isolation.
    r0 = vs.step_shaping(0, 0, 0.0, {"tree_visible": 0.1}, REL, LAB)  # adopt 0.1
    assert abs(r0) < 1e-9, f"first seek step must adopt without a delta: {r0}"
    # CONTRACT CHANGED 2026-08-02: the seek potential is now on ORIENTATION,
    # `gp * (0.6 + 0.4 * centred)`, not on bare visibility. With nothing
    # centred a rise in tree-prob earns 0.6 of what it used to; the withheld
    # 0.4 is what centring the goal now pays. The TOTAL budget is unchanged —
    # test_orientation_budget_is_conserved below pins that.
    r1 = vs.step_shaping(0, 1, 0.0, {"tree_visible": 0.4}, REL, LAB)  # +0.3
    assert abs(r1 - 0.5 * 0.3 * 0.6) < 1e-6, f"rise in tree-prob must pay: {r1}"
    r2 = vs.step_shaping(0, 2, 0.0, {"tree_visible": 0.1}, REL, LAB)  # -0.3
    assert abs(r2 + 0.5 * 0.3 * 0.6) < 1e-6, f"fall must debit symmetrically: {r2}"
    assert abs(r1 + r2) < 1e-9, "round trip must net ~0 (telescoping, unfarmable)"
    print("  1. seek potential rewards bringing the tree into view; telescopes")


def test_seek_inert_when_present_and_refills():
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=1,
                        present_threshold=0.6, seek_weight=0.5,
                        seek_forward_nudge=0.004, seek_nudge_budget=5, weight=0.6)
    for t in range(3):                       # forward while tree absent -> nudge
        vs.step_shaping(1, t, 0.0, {"tree_visible": 0.0}, REL, LAB)
    assert vs._seek_nudge_left == 2, f"3 nudges should be spent: {vs._seek_nudge_left}"
    # THE DEAD ZONE IS GONE (2026-08-02). This used to assert seek returns 0
    # the moment the goal is present, handing off to the approach potential —
    # which only ran when the magnet weight was non-zero AND the target was
    # present. "Goal visible but off-centre", the state that needs a turn,
    # fell between the two and nothing paid for closing it. Seek now keeps
    # paying, on centredness. The budget refill is unchanged.
    seek_only = vs._seek_shaping(1, {"tree_visible": 0.9, **ADJ}, REL, LAB,
                                 present={"tree_visible"})
    assert seek_only > 0.0, (
        f"a visible, well-centred goal must still pay ({seek_only}) — a 0 "
        "here is the seek->approach dead zone reopening")
    assert vs._seek_nudge_left == 5, "budget must refill when the goal is seen"
    print("  2. seek keeps paying on centredness once the goal is visible; "
          "search budget refills on sight")


def test_orientation_budget_is_conserved():
    """Splitting the potential between 'in frame' and 'centred' must not
    INFLATE it. Nothing-in-view -> visible -> centred has to total exactly
    seek_weight, the same ceiling bare visibility used to reach alone."""
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=1,
                        present_threshold=0.6, seek_weight=0.5,
                        seek_forward_nudge=0.0, weight=0.6)
    vs._seek_shaping(0, {"tree_visible": 0.0, "object_centered": 0.0},
                     REL, LAB, set())                      # adopt baseline
    total = vs._seek_shaping(0, {"tree_visible": 1.0, "object_centered": 0.0},
                             REL, LAB, {"tree_visible"})
    total += vs._seek_shaping(0, {"tree_visible": 1.0, "object_centered": 1.0},
                              REL, LAB, {"tree_visible"})
    assert abs(total - 0.5) < 1e-6, (
        f"orientation budget is {total}, must stay at seek_weight (0.5) — the "
        "split reallocates the pull, it must not create more of it")
    print("  2b. orientation budget conserved (0.6 in-frame + 0.4 centred)")


def test_forward_nudge_is_budgeted():
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=1,
                        present_threshold=0.6, seek_weight=0.0,
                        seek_forward_nudge=0.004, seek_nudge_budget=5)
    total = sum(vs.step_shaping(1, t, 0.0, {"tree_visible": 0.0}, REL, LAB)
                for t in range(20))          # 20 forward steps, tree absent
    assert abs(total - 5 * 0.004) < 1e-6, (
        f"nudge must cap at the budget (5x), got {total}")
    assert vs._seek_nudge_left == 0
    # a non-forward action pays no nudge even with budget
    vs._seek_nudge_left = 3
    r = vs.step_shaping(3, 99, 0.0, {"tree_visible": 0.0}, REL, LAB)  # turn-left
    assert abs(r) < 1e-9 and vs._seek_nudge_left == 3, "only forward spends budget"
    print("  3. forward nudge is bounded (budget cap; forward-only)")


def test_untrusted_head_does_not_seek():
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=5,
                        present_threshold=0.6, seek_weight=0.5,
                        seek_forward_nudge=0.0)
    rel_low = {"tree_visible": 0.1}          # below reliability_floor (0.35)
    vs.step_shaping(0, 0, 0.0, {"tree_visible": 0.2}, rel_low, LAB)   # gp->0
    r = vs.step_shaping(0, 1, 0.0, {"tree_visible": 0.5}, rel_low, LAB)  # still 0
    assert abs(r) < 1e-9, f"untrusted head prob must not drive seeking: {r}"
    print("  4. trust gate: a low-reliability head never drives seeking")


def test_off_by_default():
    vs = VisionScaffold(target_categories=["tree_visible"], min_labels=1,
                        present_threshold=0.6)   # seek params default to 0
    for t in range(5):
        r = vs.step_shaping(1, t, 0.0, {"tree_visible": 0.3}, REL, LAB)
        assert r == 0.0, f"seek must be OFF by default (backward-compat): {r}"
    print("  5. off by default: no seek params -> identical to the old magnet")


if __name__ == "__main__":
    for fn in (test_seek_potential_telescopes,
               test_seek_inert_when_present_and_refills,
               test_orientation_budget_is_conserved,
               test_forward_nudge_is_budgeted,
               test_untrusted_head_does_not_seek,
               test_off_by_default):
        print(f"[seek-drive-smoke] {fn.__name__}")
        fn()
    print("[seek-drive-smoke] ALL PASS")
