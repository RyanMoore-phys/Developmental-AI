"""Smoke: instinct that GENERALISES — derived targets, GUI perception, proposals.

THE GAP (measured 2026-07-26). The curiosity magnet had a vocabulary of FOUR
hand-written strings:

    DEFAULT_TARGETS = ["tree_visible","stone_visible","water_visible","animal_visible"]

Everything around it worked — contrastive learning-progress scoring, retarget
margins, dwell, cold-start pull, potential-based shaping. But in any scene
without a tree, stone, water or animal — an open inventory, a cave, a new
world, someone else's server — the present set was EMPTY, no target was
selected, and the instinct pulled toward NOTHING. The machinery was real; it
had almost nothing to point at.

THREE LAYERS, each testable on its own:
  1. the steerable set is DERIVED from the live predicate vocabulary, so a new
     predicate becomes wantable without anyone editing the magnet;
  2. the GUI is perceivable at all (crafting is GUI-only in this fork — the
     symbolic craft handlers close the Java socket, so the screen IS the
     interface);
  3. the VLM may PROPOSE a category for something it has no predicate for, on
     probation, judged by the same learning-progress bar as everything else.

THE INVARIANT THAT MATTERS MOST: a proposal is a hypothesis about WHERE TO
LOOK, never a claim about what is true. It must never reach the grounded head
or the knowledge graph. The LLM is instinct, not oracle.

Run: PYTHONPATH=. python tests/_instinct_layers_smoke.py
"""
import sys

sys.path.insert(0, ".")

from developmental_ai.llm import vlm_symbolizer as V
from developmental_ai.llm.vision_scaffold import (
    BACKGROUND_PREDICATES, DEFAULT_TARGETS, NON_STEERABLE_PREDICATES,
    VisionScaffold, steerable_targets)

LOOP = "developmental_ai/core/developmental_loop.py"


def main() -> None:
    # ---- LAYER 1: derived, not listed ----------------------------------
    assert len(DEFAULT_TARGETS) > 4, (
        f"instinct still has only {len(DEFAULT_TARGETS)} targets — the "
        f"vocabulary is not being derived")
    for t in DEFAULT_TARGETS:
        assert t in V.PREDICATES, f"{t} steerable but not a real predicate"
    # backgrounds must stay out: they are on screen constantly and would win
    # the magnet by CO-OCCURRENCE rather than by being interesting.
    for b in BACKGROUND_PREDICATES:
        assert b not in DEFAULT_TARGETS, f"background {b} became steerable"
    # ...and so must self-states: you cannot walk toward `holding_tool`.
    for n in NON_STEERABLE_PREDICATES:
        assert n not in DEFAULT_TARGETS, f"self-state {n} became steerable"
    # THE POINT OF LAYER 1: a brand-new predicate is wantable with no edit here
    grew = steerable_targets(list(V.PREDICATES) + ["lava_visible"])
    assert "lava_visible" in grew, (
        "adding a predicate did NOT make it steerable — the set is still "
        "effectively hand-maintained")

    # ---- LAYER 1b: EXPANDABLE MID-RUN ----------------------------------
    # Deriving the set at import/construction is still STATIC — right for the
    # world the agent booted into, stale the moment the vocabulary grows. An
    # instinct extendable only by restart goes blind exactly when the agent
    # walks somewhere new, which is when instinct matters most.
    vs0 = VisionScaffold(present_threshold=0.6, min_labels=0,
                         reliability_floor=0.0)
    n_before = len(vs0.target_categories)
    V.PREDICATES.append("lava_visible")
    try:
        added = vs0.refresh_targets()
        assert added == 1 and "lava_visible" in vs0.target_categories, (
            "the vocabulary grew mid-run and the instinct did NOT pick it up")
        # every per-category dict must be seeded or the new target reads as
        # "already known and uninteresting" and can never win the magnet
        for d, nm in ((vs0._cat_lp, "_cat_lp"), (vs0._lp_scale, "_lp_scale"),
                      (vs0._ever_curious, "_ever_curious"),
                      (vs0._cold_spent, "_cold_spent")):
            assert "lava_visible" in d, (
                f"{nm} not seeded for a new target — it would be present but "
                f"permanently unwantable")
        assert vs0.refresh_targets() == 0, "refresh is not idempotent"
        # an EXPLICIT config must never be silently overwritten by derivation
        pinned = VisionScaffold(target_categories=["tree_visible"])
        assert pinned.refresh_targets() == 0, "pinned config was overwritten"
        assert len(vs0.target_categories) == n_before + 1
    finally:
        V.PREDICATES.remove("lava_visible")

    # ---- LAYER 2: the GUI is perceivable, but NOT steerable ------------
    # REVERSED 2026-08-07 on live measurement: with GUI predicates in the
    # steerable set the magnet COURTED THE MENU — target=inventory_visible,
    # cold_spent[inventory]=55,889 floor-pulled steps, gui open 28% of the
    # run with a 25,923-step dwell. "Approaching" a screen overlay means
    # opening it and sitting there; a GUI is self-state like holding_tool,
    # not a place. Perception and KG facts stay; the magnet must not want it.
    gui = ["inventory_visible", "crafting_grid_visible", "craft_output_visible"]
    for g in gui:
        assert g in V.PREDICATES, f"{g} not perceivable"
        assert g in V._SCENE_PROMPT, f"VLM is never asked about {g}"
        assert g in V._FACT_TEMPLATES, f"{g} has no KG triple"
        assert g not in DEFAULT_TARGETS, (
            f"{g} is steerable — the magnet will court the menu again "
            f"(measured: 55.9k-step inventory floor-pull, 28% gui-open)")
    # presence only: the recipe must be LEARNED from watching the output slot,
    # never declared here.
    assert V._FACT_TEMPLATES["craft_output_visible"] == (
        "scene", "contains", "craft_output")
    # Check CODE only. Comments legitimately say "nothing here encodes a
    # RECIPE" while explaining exactly that, and a whole-file scan fails on
    # prose — failing for the wrong reason, which is its own bug.
    _sym_code = [ln for ln in open(V.__file__).read().splitlines()
                 if not ln.lstrip().startswith("#")]
    for banned in ("planks_make_sticks", "recipe_for", "CRAFT_RECIPES",
                   "RECIPES ="):
        hits = [ln for ln in _sym_code if banned in ln]
        assert not hits, (
            f"'{banned}' appears in symbolizer CODE — a recipe must be earned "
            f"from watching the output slot, not encoded: {hits[:1]}")

    # ---- LAYER 3: proposals, on probation ------------------------------
    # THE PRODUCTION CONDITION, asserted explicitly: a proposal can NEVER
    # appear in per-step probs, because the grounded head's output is fixed to
    # the base predicates at construction. The first cut of layer 3 read
    # probs.get(name) for presence and was therefore a silent no-op in the
    # live loop — while its unit test passed on hand-built probs dicts that
    # production never produces. Every presence check below uses EMPTY probs;
    # the ONLY sensor a proposal has is the VLM sighting itself
    # (propose_category re-invocation), and presence is a recency window.
    vs = VisionScaffold(max_proposed=3, probation_attends=5,
                        present_threshold=0.6, min_labels=0,
                        reliability_floor=0.0, ema_beta=0.5, eps_abs=0.01,
                        proposal_ttl=100)
    assert vs.propose_category("lava_visible", 0) is True
    assert vs.propose_category("tree_visible", 0) is False, "shadowed a predicate"
    assert vs.propose_category("!!!", 0) is False, "garbage admitted"
    assert "lava_visible" in vs.live_targets()
    # a repeat is a RE-SIGHTING, not an error: it is the sensor
    assert vs.propose_category("lava_visible", 10) is False
    assert vs._proposed["lava_visible"]["attends"] == 2.0, (
        "a re-sighting did not count — the proposal has no other sensor, so "
        "attends would never move and probation could never conclude")
    for i in range(5):
        vs.propose_category(f"thing{i}_visible", 0)
    assert len(vs._proposed) <= 3, "probation set is unbounded"

    # PRESENCE with EMPTY probs: recently sighted -> present; stale -> absent
    vs._now = 50
    pres = vs._trusted_present({}, {}, {})          # <-- empty, as in production
    assert "lava_visible" in pres, (
        "a just-sighted proposal is not PRESENT with empty probs — layer 3 "
        "is inert in production again (the original bug)")
    vs._now = 10 + 100 + 1                           # past the TTL
    pres = vs._trusted_present({}, {}, {})
    assert "lava_visible" not in pres, "presence did not expire with the TTL"

    # a proposal that TEACHES survives...
    good = VisionScaffold(max_proposed=4, probation_attends=5,
                          present_threshold=0.6, min_labels=0,
                          reliability_floor=0.0, ema_beta=0.5, eps_abs=0.01,
                          proposal_ttl=100)
    # INTERLEAVED as production actually runs: sighting, then steps of
    # presence + LP attribution, then the next sighting. A first draft of
    # this test compressed all sightings before any attribution and exposed
    # a real ordering bug (instant eviction at the first presence check).
    for t in range(6):
        good.propose_category("crafting_table_visible", t * 50)
        good._now = t * 50 + 10
        for _ in range(4):
            pres = good._trusted_present({}, {}, {})   # empty probs, as live
            b = good.ema_beta
            good._global_lp = (1.0 - b) * good._global_lp + b * 0.05
            for c in good.live_targets():
                cur = good._cat_lp.get(c, 0.0)
                good._cat_lp[c] = ((1.0 - b) * cur + b * 0.9) if c in pres else cur
    assert "crafting_table_visible" in good._proposed, (
        "a proposal that produced strong learning progress was EVICTED — "
        "layer 3 would reject everything and be a no-op that looks built")
    assert good._proposed["crafting_table_visible"].get("passed"), (
        "a paying proposal never PASSES its exam")
    # ...and once passed, a natural curiosity QUENCH must not evict it —
    # novel -> learned -> boring is the SUCCESS trajectory, not a failure.
    good._cat_lp["crafting_table_visible"] = 0.0
    good._now += 10
    good._trusted_present({}, {}, {})
    assert "crafting_table_visible" in good._proposed, (
        "a PASSED proposal was evicted when its curiosity quenched — the "
        "success trajectory (novel->learned->boring) is punished")

    # ...and one that teaches NOTHING is dropped, and cannot come back
    bad = VisionScaffold(max_proposed=4, probation_attends=5,
                         present_threshold=0.6, min_labels=0,
                         reliability_floor=0.0, ema_beta=0.5, eps_abs=0.01,
                         proposal_ttl=100)
    for t in range(6):
        bad.propose_category("noise_visible", t * 10)
    bad._now = 55
    for _ in range(3):
        bad._trusted_present({}, {}, {})
    assert "noise_visible" in bad._evicted, "a barren proposal was not evicted"
    assert bad.propose_category("noise_visible", 999) is False, (
        "an evicted hypothesis was re-admitted — it would keep costing "
        "attention forever")

    # ---- GRADUATION: a proven guess becomes permanent -------------------
    gr = VisionScaffold(max_proposed=4, probation_attends=3,
                        present_threshold=0.6, min_labels=0,
                        reliability_floor=0.0, ema_beta=0.5, eps_abs=0.01)
    gr.propose_category("crafting_table_visible", 0)
    assert gr.graduate("crafting_table_visible") is True
    assert "crafting_table_visible" in gr.target_categories, "not promoted"
    assert "crafting_table_visible" not in gr._proposed, "still on probation"
    assert gr.graduate("never_proposed_visible") is False, \
        "graduated something that was never proposed"

    # ---- THE INVARIANT: instinct, not oracle ---------------------------
    # A proposal must never train the grounded head or become a fact.
    src = open(V.__file__).read()
    assert '"__novel__"' in src, "no proposal channel"
    assert "__novel__" not in V.PREDICATES, "a proposal became a PREDICATE"
    assert "__novel__" not in V._FACT_TEMPLATES, "a proposal can become a FACT"
    # _train_head skips unknown keys BEFORE touching label_counts
    th = src[src.index("def _train_head"):]
    th = th[:th.index("lat = latent.reshape")]
    assert "if i is None:" in th and "continue" in th, (
        "head training does not skip unknown keys — a proposed category would "
        "be trained as if it were a grounded percept")
    loop = open(LOOP).read()
    assert loop.count("self.vision_scaffold.propose_category(") == 2, (
        "proposals are routed at only ONE of the two loop call sites "
        "(episodic + parallel) — patching one twin and not the other has "
        "recurred repeatedly in this project")

    print(f"[instinct-layers-smoke] ALL PASS: steerable set DERIVED "
          f"({len(DEFAULT_TARGETS)} targets, was 4) and grows with the "
          f"vocabulary; GUI perceivable but NOT steerable (menu-courting "
          f"measured 2026-08-07), no recipe encoded; "
          f"proposals bounded, admitted on probation, kept when they teach, "
          f"evicted and barred when they do not; a proposal can never reach "
          f"the grounded head or the knowledge graph")


if __name__ == "__main__":
    main()
