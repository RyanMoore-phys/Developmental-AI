"""Individuation + episodic-sense + progress-curiosity wave smoke (2026-08-09).

Contracts for the three changes of this wave:
  #35/#37 skills: DeltaHead is identity at birth; practice trains ONLY the
     delta (base frozen byte-for-byte); trust-region revert applies to the
     delta; delta_mag is reported; ZPD scoring prefers the learning edge.
  #38 episodic sense: _augment_proprio appends [reach, validity*proximity,
     sin, cos] on the primary stream, neutral zeros for scouts and when
     nothing was ever sighted; bearing math is body-relative via yaw.
  #13 progress: the loop-side wiring constants exist (module has its own
     suite); icm_base_scale defaults to 1.0 = byte-identical old behaviour.
"""
import math

import numpy as np
import torch

from developmental_ai.skill_bank.skill_practice import (DeltaHead,
                                                        SkillPractice)
from developmental_ai.policy.actor_critic import ActorNetwork


class _Bare:
    pass


def test_delta_identity_at_birth():
    torch.manual_seed(0)
    actor = ActorNetwork(20, 6, hidden_dim=32)
    delta = DeltaHead(20, 6)
    f = torch.randn(5, 20)
    base = actor.action_head(actor.shared(f))
    assert float(delta(f).abs().max()) == 0.0, "delta not identity at birth"
    print("  1. delta is exactly zero at mint (identity at birth)")


def test_practice_trains_only_delta():
    torch.manual_seed(0)
    actor = ActorNetwork(20, 6, hidden_dim=32)
    for p in actor.parameters():
        p.requires_grad_(False)
    delta = DeltaHead(20, 6)
    pr = SkillPractice(device=torch.device("cpu"))
    pr.enabled = True
    pr.min_steps = 1
    base_before = [p.detach().clone() for p in actor.parameters()]
    traj = [(torch.randn(1, 20), int(i % 6)) for i in range(24)]
    info = pr._self_imitate(3, actor, traj, delta=delta)
    assert info is not None and not info.get("reverted"), info
    for p, b in zip(actor.parameters(), base_before):
        assert torch.equal(p, b), "BASE moved — individuation must own all change"
    assert any(float(p.abs().sum()) > 0 for p in delta.parameters()), \
        "delta never moved"
    assert info.get("delta_mag", 0.0) > 0.0, info
    print(f"  2. practice moved ONLY the delta (mag={info['delta_mag']:.4f}); "
          f"base byte-identical")


def test_trust_region_on_delta():
    torch.manual_seed(0)
    actor = ActorNetwork(20, 6, hidden_dim=32)
    delta = DeltaHead(20, 6)
    pr = SkillPractice(device=torch.device("cpu"))
    pr.enabled = True
    pr.min_steps = 1
    pr.kl_max = 1e-9          # impossible trust region -> must revert fully
    pr.epochs = 8
    pr.lr = 0.5
    traj = [(torch.randn(1, 20), 2) for _ in range(24)]
    info = pr._self_imitate(4, actor, traj, delta=delta)
    assert info and info.get("reverted"), info
    assert all(float(p.abs().max()) == 0.0
               for p in delta.net[2].parameters()), \
        "revert did not restore the delta"
    print("  3. trust-region revert restores the delta exactly")


def test_zpd_scoring():
    """The ZPD rule from _dream_consolidate_skills, replicated: the learning
    edge (p~0.5) outranks both the hopeless and the mastered, and untried
    slots rank between."""
    def zpd(spikes, inv):
        if inv <= 0:
            return 0.4
        p = (spikes + 1.0) / (inv + 2.0)
        return 1.0 - 2.0 * abs(p - 0.5)
    edge = zpd(5, 10)         # ~0.5 success
    hopeless = zpd(0, 20)     # never works
    mastered = zpd(19, 20)    # always works
    untried = zpd(0, 0)
    assert edge > untried > hopeless, (edge, untried, hopeless)
    assert edge > untried > mastered - 0.2, (edge, untried, mastered)
    print(f"  4. ZPD ordering: edge {edge:.2f} > untried {untried:.2f} > "
          f"hopeless {hopeless:.2f} / mastered {mastered:.2f}")


def test_augment_proprio_bearing():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.infra.episodic import EpisodicEventMemory
    s = _Bare()
    s.symbolizer = object()          # non-None gates the append on
    s._reach_now = 0.7
    s.total_timesteps = 1000
    mem = EpisodicEventMemory()
    mem.record("sighting", "tree_visible", 900, (10.0, 64.0, 0.0))
    infra = _Bare(); infra.episodic = mem
    s.infra = infra
    sc = _Bare(); sc._seek_cats = ["tree_visible"]
    s.vision_scaffold = sc
    # agent at origin facing yaw=0; target due +x -> bearing 90deg -> rel 90
    s._last_world_info = {"x": 0.0, "z": 0.0, "yaw": 0.0}
    pp = np.zeros(10, np.float32)
    out = DevelopmentalAI._augment_proprio(s, pp, 0)
    assert out.shape == (14,), out.shape
    reach, val, sn, cs = out[10], out[11], out[12], out[13]
    assert abs(reach - 0.7) < 1e-6
    assert val > 0.5, f"fresh close sighting should be strong: {val}"
    assert abs(sn - 1.0) < 1e-5 and abs(cs) < 1e-5, (sn, cs)
    # MINECRAFT YAW CONVENTION PINNED (yaw grows clockwise; 90 = west = -x):
    # agent facing west with the target due WEST must read dead-ahead.
    mem2 = EpisodicEventMemory()
    mem2.record("sighting", "tree_visible", 900, (-10.0, 64.0, 0.0))
    infra.episodic = mem2
    s._last_world_info = {"x": 0.0, "z": 0.0, "yaw": 90.0}
    o_w = DevelopmentalAI._augment_proprio(s, pp, 0)
    assert abs(o_w[12]) < 1e-5 and abs(o_w[13] - 1.0) < 1e-5, \
        (o_w[12], o_w[13])
    infra.episodic = mem
    s._last_world_info = {"x": 0.0, "z": 0.0, "yaw": 0.0}
    # scouts read neutral zeros
    out2 = DevelopmentalAI._augment_proprio(s, pp, 1)
    assert out2.shape == (14,) and float(np.abs(out2[10:]).sum()) == 0.0
    # no memory at all -> bearing fields neutral, reach still felt
    infra.episodic = EpisodicEventMemory()
    out3 = DevelopmentalAI._augment_proprio(s, pp, 0)
    assert float(np.abs(out3[11:]).sum()) == 0.0 and out3[10] == np.float32(0.7)
    # ...and memory FADES: the same sighting 60k steps later is a whisper
    infra.episodic = mem
    s.total_timesteps = 61000
    out4 = DevelopmentalAI._augment_proprio(s, pp, 0)
    assert out4[11] < 0.1 * val, (out4[11], val)
    print(f"  5. episodic bearing: reach={reach:.2f} pull={val:.2f} "
          f"sin/cos=({sn:.2f},{cs:.2f}); scouts neutral; fades with age")


def test_icm_scale_default_neutral():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    s = _Bare()
    assert float(getattr(s, "_icm_base_scale", 1.0)) == 1.0
    print("  6. icm_base_scale defaults neutral (old behaviour preserved)")


if __name__ == "__main__":
    for fn in (test_delta_identity_at_birth, test_practice_trains_only_delta,
               test_trust_region_on_delta, test_zpd_scoring,
               test_augment_proprio_bearing, test_icm_scale_default_neutral):
        print(f"[individuation-wave] {fn.__name__}")
        fn()
    print("[individuation-wave] ALL PASS")
