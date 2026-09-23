"""Unit cases for spatial memory, object slots and the replay buffer column.

CONTRACTS vs UNITS. The contract suites argue the design (the
re-registration sign, the SR's frame, RED isolation). This file checks the
parts: bounds, shapes, resets, and the degenerate inputs that occur live —
a scan of all zeros, a single slot, a buffer whose layout changed.

Run: PYTHONPATH=. python tests/unit/test_spatial_slots_unit.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, ".")

import numpy as np
import torch

from tests.unit._runner import case, run_all, close, between, finite, raises
from developmental_ai.spatial import (
    EgocentricOccupancy, SuccessorMap, WalkabilityMap, rotate_translate)
from developmental_ai.slots import SAViSlots, slot_relations, RELATION_FIELDS
from developmental_ai.slots.relations import relation_width
from developmental_ai.world_model.replay_buffer import (
    ReplayBuffer, MultiStreamReplayBuffer)


# ------------------------------------------------------------ occupancy

@case
def rotate_translate_does_not_blow_up():
    """Bilinear BACKWARD mapping does not exactly conserve mass.

    The weight a source cell contributes across every output cell that reads
    it sums to ~1 on average but can EXCEED 1 locally, so a rotation can
    amplify slightly. Whole-cell translation is exact. The bound here is
    what the occupancy map's clamp exists to enforce — without it a turning
    agent's confidence would compound above 1/decay and every cell would
    eventually read "solid".
    """
    g = np.zeros((9, 9), np.float32)
    g[4, 4] = 1.0
    for dy, dx in ((1.0, 0.0), (0.0, 1.0)):
        out = rotate_translate(g, 0.0, dx, dy)
        close(float(out.sum()), 1.0, 1e-5, f"whole-cell shift {(dy, dx)}")
    for dy, dx, ang in ((0.0, 0.0, 0.7), (2.0, -1.5, -0.3), (0.5, 0.5, 1.9)):
        out = rotate_translate(g, ang, dx, dy)
        finite(out)
        assert out.sum() <= 2.0, f"mass {out.sum():.2f} at {(dy, dx, ang)}"


@case
def occupancy_confidence_cannot_compound_under_constant_turning():
    """THE BUG THE CASE ABOVE IMPLIES, checked where it matters.

    A turning agent re-registers every step. If amplification exceeded
    1/decay the map would grow without bound until everything read solid —
    the worst possible failure for a walkability-adjacent signal, because
    the agent would come to believe it was walled in everywhere.
    """
    occ = EgocentricOccupancy(grid=15, decay=0.93, max_range=10.0)
    scan = np.zeros((8, 8), np.float32); scan[:, 4] = 0.5
    for i in range(400):
        occ.step(scan if i % 3 == 0 else None, 0.31, 0.4, 0.1)
        between(float(occ.map.max()), 0.0, 1.0, f"confidence at step {i}")
    finite(occ.map)


@case
def identity_transform_is_the_identity():
    g = np.random.rand(11, 11).astype(np.float32)
    close(float(np.abs(rotate_translate(g, 0.0, 0.0, 0.0) - g).max()), 0.0,
          1e-5, "identity re-registration")


@case
def occupancy_stays_in_range_and_decays_geometrically():
    occ = EgocentricOccupancy(grid=11, decay=0.8, max_range=8.0)
    scan = np.zeros((8, 8), np.float32); scan[:, 4] = 0.3
    occ.step(scan, 0.0, 0.0, 0.0)
    peak = float(occ.map.max())
    between(peak, 0.0, 1.0, "occupancy confidence")
    for k in range(1, 4):
        occ.step(None, 0.0, 0.0, 0.0)
        close(float(occ.map.max()), peak * 0.8 ** k, 1e-4, f"decay step {k}")


@case
def occupancy_writes_nothing_for_an_empty_scan():
    occ = EgocentricOccupancy(grid=9)
    occ.step(np.zeros((8, 8), np.float32), 0.0, 0.0, 0.0)
    close(float(occ.map.max()), 0.0, 1e-9,
          "a scan of zeros means nothing was resolvable, not a wall at zero")


@case
def occupancy_tolerates_none_and_odd_shapes():
    occ = EgocentricOccupancy(grid=9)
    occ.step(None, 0.1, 0.2, 0.3)
    occ.step(np.zeros((1, 1), np.float32), 0.0, 0.0, 0.0)
    occ.step(np.zeros((0, 0), np.float32), 0.0, 0.0, 0.0)
    finite(occ.map)


@case
def occupancy_clamps_to_max_range():
    occ = EgocentricOccupancy(grid=41, cell_blocks=1.0, decay=1.0,
                              max_range=5.0)
    tiny = np.zeros((8, 8), np.float32); tiny[:, 4] = 1e-2   # very far
    occ.step(tiny, 0.0, 0.0, 0.0)
    r, _c = np.unravel_index(int(occ.map.argmax()), occ.map.shape)
    assert abs(int(r) - 20) <= 6, (
        f"a distant reading landed {abs(int(r) - 20)} cells out; max_range "
        f"must clamp it inside the map")


# ------------------------------------------------------------- successor

@case
def successor_index_wraps_rather_than_clipping():
    sr = SuccessorMap(grid=4, cell_blocks=1.0)
    assert sr._index(0.0, 0.0) == sr._index(4.0, 4.0), "did not wrap"
    assert sr._index(-1.0, 0.0) == sr._index(3.0, 0.0), "negative wrap"
    assert 0 <= sr._index(1e6, -1e6) < 16


@case
def successor_rows_are_finite_and_reset_clears():
    sr = SuccessorMap(grid=3, cell_blocks=1.0, gamma=0.9, lr=0.3)
    for t in range(50):
        finite(sr.step(float(t % 3), 0.0), "successor row")
    assert float(np.abs(sr.m).max()) > 0.0
    sr.reset()
    close(float(np.abs(sr.m).max()), 0.0, 1e-12)
    assert sr._prev is None


@case
def successor_first_step_teaches_nothing():
    sr = SuccessorMap(grid=3, cell_blocks=1.0)
    sr.step(0.0, 0.0)
    close(float(np.abs(sr.m).max()), 0.0, 1e-12,
          "the first observation has no predecessor to credit")


# ----------------------------------------------------------- walkability

@case
def walkability_stays_in_range_under_any_input():
    w = WalkabilityMap(bins=8, lr=0.5)
    for i in range(200):
        w.step(bool(i % 2), float(i % 3), (i % 7) - 3.0)
        between(float(w.w.min()), 0.0, 1.0)
        between(float(w.w.max()), 0.0, 1.0)
        finite(w.w)


@case
def walkability_rotation_preserves_total_belief():
    w = WalkabilityMap(bins=8, lr=0.0)
    w.w = np.linspace(0.1, 0.9, 8).astype(np.float32)
    before = float(w.w.sum())
    w.step(False, 0.0, 2.0 * math.pi)
    close(float(w.w.sum()), before, 1e-3, "a full turn changed total belief")


@case
def walkability_reset_restores_the_prior():
    w = WalkabilityMap(bins=4, lr=1.0, prior=0.5)
    w.step(True, 1.0, 0.0)
    assert float(w.w[0]) > 0.9
    w.reset()
    assert np.allclose(w.w, 0.5)


# ----------------------------------------------------------------- slots

@case
def slot_shapes_and_bounds():
    K, G, C = 4, 8, 32
    m = SAViSlots(feat_dim=C, num_slots=K, slot_dim=16, grid=G, flow_size=8)
    fm = torch.randn(3, C, G, G)
    st = m.initial(fm)
    out = m.step(fm, st)
    assert out["slots"].shape == (3, K, 16)
    assert out["attn"].shape == (3, K, G * G)
    assert out["alpha"].shape == (3, K)
    assert out["pred_flow"].shape == (3, 2, 8, 8)
    between(float(out["alpha"].min()), 0.0, 1.0)
    between(float(out["alpha"].max()), 0.0, 1.0)
    finite(out["slots"].detach().numpy())


@case
def slot_attention_competes_over_slots_not_positions():
    K, G, C = 4, 4, 8
    m = SAViSlots(feat_dim=C, num_slots=K, slot_dim=8, grid=G, flow_size=8)
    out = m.step(torch.randn(2, C, G, G), m.initial(torch.randn(2, C, G, G)))
    s = out["attn"].sum(dim=1)                  # sum over SLOTS per position
    close(float((s - 1.0).abs().max()), 0.0, 1e-4,
          "attention must normalise over slots — that single axis IS the "
          "decomposition mechanism")


@case
def slot_flow_starts_at_exactly_zero():
    m = SAViSlots(feat_dim=8, num_slots=3, slot_dim=8, grid=4, flow_size=8)
    out = m.step(torch.randn(2, 8, 4, 4), m.initial(torch.randn(2, 8, 4, 4)))
    close(float(out["pred_flow"].abs().max()), 0.0, 1e-9,
          "the correct prior before any movement is 'nothing moves'")


@case
def slot_conditioning_changes_the_initial_state():
    m = SAViSlots(feat_dim=8, num_slots=3, slot_dim=8, grid=4, flow_size=8)
    fm = torch.randn(1, 8, 4, 4)
    free = m.initial(fm)["slots"]
    hinted = m.initial(fm, hints=torch.tensor([[[0.5, -0.5]]]))["slots"]
    assert float((free[:, 0] - hinted[:, 0]).abs().max()) > 1e-6
    close(float((free[:, 1:] - hinted[:, 1:]).abs().max()), 0.0, 1e-9,
          "a hint must only claim the slots it was given")


@case
def slot_geometry_is_bounded():
    K, G = 5, 8
    attn = torch.rand(2, K, G * G)
    g = SAViSlots.geometry(attn, G, inv_depth=torch.rand(2, 1, 16, 16))
    for k in ("bearing", "elevation", "size", "depth"):
        assert g[k].shape == (2, K), k
        finite(g[k].numpy(), k)
    between(float(g["bearing"].abs().max()), 0.0, 1.0, "bearing")
    between(float(g["depth"].min()), 0.0, 1.0, "depth")


@case
def relations_width_and_gating():
    K, G = 4, 4
    attn = torch.rand(2, K, G * G)
    g = SAViSlots.geometry(attn, G, inv_depth=torch.rand(2, 1, 8, 8))
    rel = slot_relations(g, attn)
    assert rel.shape == (2, K * (K - 1), len(RELATION_FIELDS))
    assert rel.numel() // 2 == relation_width(K)
    finite(rel.numpy())
    # alpha=0 must neutralise, not zero: size ratio and co-motion are 0.5.
    off = slot_relations(g, attn, alpha=torch.zeros(2, K))
    close(float(off[..., 3].mean()), 0.5, 1e-6, "neutral size ratio")
    close(float(off[..., 4].mean()), 0.5, 1e-6, "neutral co-motion")
    close(float(off[..., 0].abs().max()), 0.0, 1e-6, "neutral bearing")


@case
def relations_are_directed():
    K, G = 3, 4
    attn = torch.rand(1, K, G * G)
    g = SAViSlots.geometry(attn, G, inv_depth=torch.rand(1, 1, 8, 8))
    g["bearing"] = torch.tensor([[-0.5, 0.0, 0.5]])
    rel = slot_relations(g, attn)
    # pair (0,1) and pair (1,0) must have opposite bearing deltas.
    close(float(rel[0, 0, 0]), -float(rel[0, 2, 0]), 1e-5,
          "i->j and j->i carry the same bearing; the relation is not directed")


# ---------------------------------------------------------------- buffer

@case
def buffer_without_sensors_is_unchanged():
    b = ReplayBuffer(capacity=64, obs_dim=4, action_dim=2)
    for _ in range(20):
        b.add(np.zeros(4, np.float32), 0, 0.0, False)
    assert b.proprio is None
    assert "proprio" not in b.sample_sequences(2, 4)


@case
def buffer_reward_threshold_selects():
    b = ReplayBuffer(capacity=512, obs_dim=4, action_dim=2)
    for i in range(200):
        b.add(np.zeros(4, np.float32), 0, 20.0 if i == 100 else 0.4, False)
    assert len(b._find_reward_starts(4, threshold=1e-3)) > 150
    assert 0 < len(b._find_reward_starts(4, threshold=5.0)) <= 8


@case
def buffer_reports_the_reward_pool():
    b = ReplayBuffer(capacity=256, obs_dim=4, action_dim=2)
    for i in range(80):
        b.add(np.zeros(4, np.float32), 0, 20.0 if i % 20 == 0 else 0.0, False)
    out = b.sample_sequences(4, 4, reward_fraction=0.5, reward_threshold=5.0)
    assert out["reward_pool"] > 0
    plain = ReplayBuffer(capacity=256, obs_dim=4, action_dim=2)
    for _ in range(40):
        plain.add(np.zeros(4, np.float32), 0, 0.0, False)
    assert plain.sample_sequences(2, 4)["reward_pool"] == -1, (
        "never asked must be distinguishable from asked and found nothing")


@case
def buffer_layout_mismatch_restores_neutral():
    W = 5
    a = ReplayBuffer(capacity=64, obs_dim=4, action_dim=2, proprio_dim=W,
                     sensor_layout="AAA")
    for _ in range(20):
        a.add(np.zeros(4, np.float32), 0, 0.0, False,
              proprio=np.full(W, 0.9, np.float32))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b")
        a.save(p)
        same = ReplayBuffer(capacity=64, obs_dim=4, action_dim=2,
                            proprio_dim=W, sensor_layout="AAA")
        n = same.load(p)
        assert np.allclose(same.proprio[:n], 0.9)
        other = ReplayBuffer(capacity=64, obs_dim=4, action_dim=2,
                             proprio_dim=W, sensor_layout="BBB")
        n2 = other.load(p)
        close(float(np.abs(other.proprio[:n2]).sum()), 0.0, 1e-9)


@case
def multistream_shares_one_schema():
    m = MultiStreamReplayBuffer(num_streams=3, capacity=600, obs_dim=4,
                                action_dim=2, proprio_dim=6,
                                sensor_layout="X")
    for st in range(3):
        for _ in range(40):
            m.add(np.zeros(4, np.float32), 0, 0.0, False, stream=st,
                  proprio=np.full(6, 0.3, np.float32))
    out = m.sample_sequences(6, 4)
    assert out["proprio"].shape == (6, 4, 6)


if __name__ == "__main__":
    sys.exit(run_all("spatial-slots-unit"))
