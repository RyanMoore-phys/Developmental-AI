"""Spatial-memory smoke (2026-09-19, roadmap G1/G2/P2/P3).

WHAT THESE MAPS ARE, AND WHY THEY ARE THE FIRST OF THEIR KIND HERE

    Every representation in this project has been PER-FRAME. The grounding
    head answers "is a tree visible NOW"; the magnet targets what is on
    screen NOW; the flow senses describe THIS step. Nothing carried the fact
    that a wall was on the left three steps ago — which is why "a tree at
    the edge of frame and a tree dead-centre pay exactly the same" fairly
    described the whole stack and not just one reward term.

    All four are built from the agent's OWN estimates: the world model's
    forward-probe magnitude, and the body's sensed ego-motion. Nothing reads
    a block name, a voxel grid or a true coordinate — the engine's position
    exists in this codebase only as the RED oracle, and only to measure
    these. They therefore inherit their inputs' errors, which is correct.

Contracts:
    A. THE RE-REGISTRATION SIGN. The map is anchored to the BODY, so when
       the body turns right the world turns left underneath it and when the
       body advances the world slides backwards. Both transforms are the
       INVERSE of the motion. Getting this backwards produces a map that
       smears in the direction of travel instead of holding still — which
       looks plausible on a static frame and is wrong the moment anything
       moves. Pinned by a known trajectory, not by argument.
    B. MEMORY, NOT SURVEY. An unobserved cell must FADE rather than assert
       an emptiness it cannot vouch for, and a wall must survive being
       turned away from.
    C. NEAR SWEEPS FAST. A larger probe magnitude must land CLOSER to the
       body — that inversion is the entire link between the flow head and
       the map, and if it is backwards every downstream map is mirrored.
    D. THE SR NEEDS A FRAME THAT HOLDS STILL. Over an egocentric grid the
       agent is always in the middle cell and every state is its own
       successor; this pins that the successor map is built over the
       dead-reckoned frame instead, and that it actually predicts.
    E. ANNOTATION BY INTERACTION. Walkability is learned from the GAP
       between command and outcome. Nobody labels a wall — the wall labels
       itself by stopping you.
    F. NO OPINION IS NOT A CLAIM. Walkability's prior is 0.5; zero would
       assert the world is solid, which is the belief that stops an agent
       from ever trying an untried direction.
    G. WIDTH AGREEMENT. _spatial_width() is the only declaration and
       _spatial_fields() the only writer; a mismatch would shift every map
       into the wrong slot, silently.

Run: PYTHONPATH=. python tests/_spatial_memory_smoke.py
"""
import math
import os
import sys

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.spatial import (
    EgocentricOccupancy, SuccessorMap, WalkabilityMap, rotate_translate)

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
CFG = os.path.join("configs", "minecraft_skybot.yaml")


def _peak(m):
    """(row, col) of the strongest cell."""
    return np.unravel_index(int(np.argmax(m)), m.shape)


def test_reregistration_sign():
    """A. A remembered landmark holds still while the body moves past it.

    EXERCISED THROUGH EgocentricOccupancy.step(), not through the raw
    helper: the signs that matter are the ones the live path actually
    passes, and testing the helper directly would let step() pass the wrong
    ones while this contract went green.

    CONVENTION: grid[row, col] = [z, x]; body at centre; AHEAD IS INCREASING
    ROW; the body's RIGHT is increasing column.

    THE BUG THIS CAUGHT. The intuitive phrasing "the map is anchored to the
    body, so the world slides backwards" gave the TRANSLATION the wrong sign
    — under a backward-mapping resample the inversion is already carried by
    the sampling, so negating again made a remembered wall RECEDE as the
    agent walked into it. The rotation sign, derived the same way, happened
    to be right. One of two, which is exactly why this is pinned by
    trajectory rather than by argument.
    """
    g, c = 15, 7
    occ = EgocentricOccupancy(grid=g, cell_blocks=1.0, decay=1.0,
                              max_range=10.0)
    scan = np.zeros((8, 8), np.float32)
    scan[:, 4] = 0.2                          # a wall ~5 blocks dead ahead
    occ.step(scan, 0.0, 0.0, 0.0)
    r0, c0 = _peak(occ.map)
    assert r0 > c, f"'ahead' wrote to row {r0}; ahead must be row > {c}"

    # Walk 2 blocks forward with nothing new observed: the remembered wall
    # must come CLOSER to the body, not recede.
    occ.step(None, 0.0, 2.0, 0.0)
    r1, _ = _peak(occ.map)
    assert r1 < r0, (
        f"after advancing 2 blocks the remembered wall moved from row {r0} "
        f"to {r1} — it RECEDED. The translation sign is inverted and the "
        f"map smears away from the direction of travel")
    assert abs((r0 - r1) - 2) <= 1, (
        f"advancing 2 blocks moved the wall {r0 - r1} cells at 1 block per "
        f"cell; the scale is wrong")

    # Turn 90 degrees RIGHT (+yaw): what was ahead must end up on the LEFT.
    occ.step(None, math.pi / 2, 0.0, 0.0)
    r2, c2 = _peak(occ.map)
    assert c2 < c and abs(r2 - c) <= 2, (
        f"after turning right the wall sits at ({r2}, {c2}); it must move to "
        f"the body's LEFT (col < {c}). This is the sign error the episodic "
        f"bearing shipped once, one module over")
    print(f"  A. ahead row {r0}; advance 2 -> row {r1}; turn right -> "
          f"col {c}->{c2}")


def test_memory_fades_but_survives_a_turn():
    """B. A wall you turned away from is still remembered, weakly."""
    occ = EgocentricOccupancy(grid=15, cell_blocks=1.0, decay=0.9,
                              max_range=10.0)
    scan = np.zeros((8, 8), np.float32)
    scan[:, 4] = 0.25                        # something ~4 blocks ahead
    occ.step(scan, 0.0, 0.0, 0.0)
    seen = float(occ.map.max())
    assert seen > 0.5, f"nothing was written into the map ({seen:.2f})"

    # Turn away and stop looking: it must decay, not vanish and not persist.
    for _ in range(5):
        occ.step(None, 0.0, 0.0, 0.0)
    faded = float(occ.map.max())
    assert 0.0 < faded < seen, (
        f"after 5 unobserved steps the map reads {faded:.3f} against "
        f"{seen:.3f}; it must FADE — neither assert stale geometry nor "
        f"claim an emptiness it cannot vouch for")
    assert abs(faded - seen * 0.9 ** 5) < 0.05

    occ.reset()
    assert float(occ.map.max()) == 0.0, "a new world must be a new map"
    print(f"  B. wrote {seen:.2f}, faded to {faded:.2f} over 5 steps "
          f"(decay 0.9^5 = {0.9 ** 5:.2f}), reset clears")


def test_near_sweeps_fast():
    """C. Larger sweep magnitude must land CLOSER to the body."""
    occ = EgocentricOccupancy(grid=21, cell_blocks=1.0, decay=1.0,
                              max_range=15.0)
    c = 10

    near = np.zeros((8, 8), np.float32); near[:, 4] = 0.5     # fast sweep
    occ.step(near, 0.0, 0.0, 0.0)
    r_near = _peak(occ.map)[0]

    occ.reset()
    far = np.zeros((8, 8), np.float32); far[:, 4] = 0.1       # slow sweep
    occ.step(far, 0.0, 0.0, 0.0)
    r_far = _peak(occ.map)[0]

    d_near, d_far = abs(r_near - c), abs(r_far - c)
    assert d_near < d_far, (
        f"magnitude 0.5 landed {d_near} cells away and 0.1 landed {d_far}; "
        f"NEAR MUST SWEEP FAST. If this inverts, every map downstream is "
        f"mirrored and the agent walks away from what it approaches")
    print(f"  C. magnitude 0.50 -> {d_near} cells, 0.10 -> {d_far} cells")


def test_successor_needs_a_frame_that_holds_still():
    """D. The SR predicts, and only because its frame does not move."""
    sr = SuccessorMap(grid=4, cell_blocks=1.0, gamma=0.9, lr=0.5)
    # Walk a repeating line so cell 0 reliably precedes cell 1.
    for _ in range(60):
        for x in (0.0, 1.0, 2.0, 3.0):
            sr.step(x, 0.0)
    m0 = sr.m[sr._index(0.0, 0.0)]
    nxt = sr._index(1.0, 0.0)
    far = sr._index(3.0, 0.0)
    assert m0[nxt] > m0[far], (
        f"from cell 0 the successor map rates the NEXT cell {m0[nxt]:.3f} "
        f"and the farthest {m0[far]:.3f}; it is not predicting")
    assert m0[nxt] > 0.1, "the immediate successor barely registers"

    # AND THE DEGENERACY THIS DESIGN AVOIDS: in an egocentric frame the
    # agent never leaves the centre cell, so every state is its own
    # successor and the map learns nothing about where it is going.
    ego = SuccessorMap(grid=4, cell_blocks=1.0, gamma=0.9, lr=0.5)
    for _ in range(60):
        ego.step(0.0, 0.0)                    # body always at its own origin
    row = ego.m[ego._index(0.0, 0.0)]
    assert int(np.argmax(row)) == ego._index(0.0, 0.0), (
        "the egocentric degeneracy did not reproduce, so this contract is "
        "not demonstrating why the dead-reckoned frame was chosen")
    print(f"  D. from cell 0: next {m0[nxt]:.2f} > far {m0[far]:.2f}; "
          f"egocentric frame collapses to self-succession as expected")


def test_walkability_is_learned_from_outcomes():
    """E/F. The wall labels itself, and 'untried' is not 'impassable'."""
    w = WalkabilityMap(bins=8, lr=0.3)
    assert np.allclose(w.w, 0.5), (
        "the prior must be 0.5 — no opinion. Zero is a CLAIM that the world "
        "is solid, and that is the belief that stops an agent trying")

    # Pushed forward and moved -> passable.
    for _ in range(10):
        w.step(True, 1.0, 0.0)
    assert w.w[0] > 0.9, f"open ground did not read passable ({w.w[0]:.2f})"

    # Pushed forward and did not move -> blocked.
    for _ in range(10):
        w.step(True, 0.0, 0.0)
    assert w.w[0] < 0.1, f"a wall did not read blocked ({w.w[0]:.2f})"

    # NOT COMMANDING MOVEMENT TEACHES NOTHING. Standing still is not
    # evidence that the way ahead is blocked, and treating it as evidence
    # would let an idle agent convince itself it is walled in.
    before = float(w.w[0])
    for _ in range(10):
        w.step(False, 0.0, 0.0)
    assert abs(float(w.w[0]) - before) < 1e-6, (
        "standing still changed the walkability belief; only a COMMANDED "
        "move that succeeded or failed is evidence")

    # A turn slides the bearings with the body.
    w2 = WalkabilityMap(bins=4, lr=1.0)
    w2.step(True, 1.0, 0.0)                   # ahead is open
    assert w2.w[0] > 0.9
    w2.step(False, 0.0, math.pi / 2)          # turn 90 deg right
    assert w2.w[0] < 0.9, "the open bearing did not move with the body"
    assert float(w2.w.max()) > 0.9, "the open bearing was lost entirely"
    print(f"  E/F. prior 0.5; open -> {1.0:.2f}, wall -> 0.0x; idle steps "
          f"teach nothing; bearings rotate with the body")


def test_width_declaration_matches_writer():
    """G. One declaration, one writer, and they must agree."""
    src = open(LOOP).read()
    assert src.count("def _spatial_width") == 1
    assert src.count("def _spatial_fields") == 1
    assert src.count("self._spatial_width()") == 1, (
        "_spatial_width must be consumed exactly once — by the policy's "
        "proprio_dim declaration")
    assert src.count("self._spatial_fields(e_i)") == 1, (
        "_spatial_fields must be written exactly once — in _augment_proprio, "
        "which is the shared assembly point both loop bodies use")
    # The maps must be advanced in BOTH loop bodies (CLAUDE.md 4.2):
    # _run_episode_parallel and _collect_segment. This project has already
    # lost the reach sense to exactly this — appended in one body only and
    # silently dead in every skybot run.
    assert src.count("def _spatial_step") == 1
    assert src.count("self._spatial_step(") == 2, (
        f"_spatial_step is called from "
        f"{src.count('self._spatial_step(')} loop bodies, expected 2")
    assert src.count("self._spatial_reset()") == 2, (
        "the maps must be cleared in both bodies too — a map carried across "
        "a world boundary is a memory of a place that no longer exists")

    import yaml
    sp = yaml.safe_load(open(CFG)).get("spatial") or {}
    if sp.get("enabled"):
        n = (sp["probe_grid"] ** 2 + sp["occupancy_grid"] ** 2
             + sp["successor_grid"] ** 2 + sp["walk_bins"])
        assert 0 < n < 2000, f"spatial fields = {n}; that is out of scale"
        assert 0.0 < sp["occupancy_decay"] < 1.0, (
            "decay must be a real half-life: 1.0 never forgets (stale "
            "geometry asserted forever) and 0 never remembers")
        print(f"  G. one declaration, one writer, two loop bodies; "
              f"{n} fields appended")
    else:
        print("  G. one declaration, one writer, two loop bodies; disabled")


if __name__ == "__main__":
    test_reregistration_sign()
    test_memory_fades_but_survives_a_turn()
    test_near_sweeps_fast()
    test_successor_needs_a_frame_that_holds_still()
    test_walkability_is_learned_from_outcomes()
    test_width_declaration_matches_writer()
    print("[spatial-memory] ALL PASS")
