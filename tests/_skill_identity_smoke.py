"""Skill identity + persistence smoke (Mac-ok, no MineRL/Ollama).

Contracts for the Layer-1 skill-brain groundwork:
  1. TIE DOES NOT CLOBBER: an upsert at equal success_rate leaves stored
     policy weights untouched (the live bug: fresh sigmoid(0)=0.5 competence
     tied stored 0.5 skills and replaced trained weights).
  2. Practice floor: higher success but < min_practice_episodes -> no
     overwrite either.
  3. A real winner (higher success + enough practice) DOES replace weights,
     and the superseded policy is archived under versions/ (bounded).
  4. Atomic registry: corrupt registry.json + intact .bak -> loads from .bak;
     orphaned skill dirs are adopted, not invisible.
  5. Honest mastery: is_mastered reflects the milestone, and sub-milestone
     skills are STILL selectable (competence-weighted, not gated).
  6. Broadcaster identity round-trip: signatures/slots/competence survive
     save_state/load_state; a re-observed event maps to its ORIGINAL slot in
     a fresh process (no relabelling); unlock_log survives for the DAG.
"""
import json
import os
import shutil

import numpy as np
import torch

SB = "/tmp/skill_identity_smoke_sb"


def _bank(**kw):
    from developmental_ai.skill_bank.skill_bank import SkillBank
    return SkillBank(storage_dir=SB, **kw)


def _weights(v: float):
    return {"w": torch.full((4, 4), float(v))}


def _stored_w(bank, sid):
    return float(torch.load(
        os.path.join(bank.storage_dir, sid, "policy.pt"),
        weights_only=True)["w"][0, 0])


def test_tie_does_not_clobber():
    shutil.rmtree(SB, ignore_errors=True)
    bank = _bank(min_practice_episodes=3)
    bank.save_skill("ach_00_discovered_0", "s0", _weights(1.0),
                    success_rate=0.5, total_episodes=8, dedup=True)
    # fresh-run challenger: same 0.5, minimal practice — the exact live bug
    bank.save_skill("ach_00_discovered_0", "s0", _weights(2.0),
                    success_rate=0.5, total_episodes=8, dedup=True)
    assert _stored_w(bank, "ach_00_discovered_0") == 1.0, (
        "a TIE overwrote trained weights — the clobbering bug is back")
    print("  1. tie keeps incumbent weights ok")


def test_practice_floor():
    bank = _bank(min_practice_episodes=3)
    bank.save_skill("ach_00_discovered_0", "s0", _weights(3.0),
                    success_rate=0.9, total_episodes=1, dedup=True)
    assert _stored_w(bank, "ach_00_discovered_0") == 1.0, (
        "an unpracticed challenger overwrote weights")
    print("  2. practice floor ok (0.9 succ but 1 ep -> kept incumbent)")


def test_winner_replaces_and_versions():
    bank = _bank(min_practice_episodes=3)
    bank.save_skill("ach_00_discovered_0", "s0", _weights(4.0),
                    success_rate=0.7, total_episodes=5, dedup=True)
    assert _stored_w(bank, "ach_00_discovered_0") == 4.0, (
        "a genuine winner failed to replace weights")
    versions = os.listdir(os.path.join(SB, "ach_00_discovered_0", "versions"))
    assert len(versions) == 1 and versions[0].endswith(".pt"), (
        f"superseded policy not archived: {versions}")
    old = torch.load(os.path.join(SB, "ach_00_discovered_0", "versions",
                                  versions[0]), weights_only=True)
    assert float(old["w"][0, 0]) == 1.0, "archived version has wrong weights"
    print(f"  3. winner replaced + archived {versions[0]} ok")


def test_registry_resilience():
    from developmental_ai.skill_bank.skill_bank import SkillBank
    reg = os.path.join(SB, "registry.json")
    assert os.path.exists(reg + ".bak"), "atomic write left no .bak"
    with open(reg, "w") as f:
        f.write('{"skills": [{"skill_id": "trunc')   # simulated torn write
    bank2 = SkillBank(storage_dir=SB)
    assert "ach_00_discovered_0" in bank2.skills, (
        "corrupt registry bricked the bank instead of falling back to .bak")
    # orphan adoption: dir with policy.pt, no registry row
    os.makedirs(os.path.join(SB, "orphan_skill"), exist_ok=True)
    torch.save(_weights(9.0), os.path.join(SB, "orphan_skill", "policy.pt"))
    bank3 = SkillBank(storage_dir=SB)
    assert "orphan_skill" in bank3.skills, "orphaned dir not adopted"
    bank3._save_registry()   # heal the registry for later tests
    print("  4. .bak fallback + orphan adoption ok")


def test_honest_mastery_and_selection():
    from developmental_ai.core.glue_layer import SkillSelector
    from developmental_ai.knowledge_graph.knowledge_graph import (
        InMemoryKnowledgeGraph)
    bank = _bank(mastery_milestone=0.8)
    weak = bank.save_skill("weak_skill", "weak", _weights(0.1),
                           success_rate=0.3, total_episodes=6, dedup=True)
    strong = bank.save_skill("strong_skill", "strong", _weights(0.2),
                             success_rate=0.9, total_episodes=6, dedup=True)
    assert weak.is_mastered is False, "0.3 skill flagged mastered"
    assert strong.is_mastered is True, "0.9 skill not flagged mastered"
    sel = SkillSelector()
    picks = sel.select_skills(None, bank, InMemoryKnowledgeGraph(),
                              max_skills=10)
    names = {s.skill_id for s, _ in picks}
    assert "weak_skill" in names, (
        "sub-milestone skill invisible to selection — mastery became a gate")
    print("  5. honest mastery + competence-weighted selection ok")


def test_broadcaster_identity_roundtrip():
    from developmental_ai.core.achievement_goals import (
        DiscoveredAchievementGoals)

    class _FakeEnv:
        def __init__(self):
            self.reward_history = [0.0]
            self.obs_history = [np.zeros(64, np.float32)] * 2

        def spike(self, seed):
            rng = np.random.RandomState(seed)
            prev = rng.rand(64).astype(np.float32)
            nxt = prev + rng.rand(64).astype(np.float32) * 2.0
            self.obs_history = [prev, nxt]
            self.reward_history = [5.0]

    g1 = DiscoveredAchievementGoals(max_slots=8, seed=0)
    env = _FakeEnv()
    env.spike(1); g1.update(env)          # -> discovered_0
    env.spike(2); g1.update(env)          # -> discovered_1 (different sig)
    assert g1.slot_names == ["discovered_0", "discovered_1"]
    # teach the self-model something non-default so restore is observable
    for _ in range(10):
        g1.competence.update(0, True)
    path = os.path.join(SB, "broadcaster_state.json")
    g1.save_state(path)

    g2 = DiscoveredAchievementGoals(max_slots=8, seed=99)   # "new process"
    assert g2.load_state(path) is True
    assert g2.slot_names == ["discovered_0", "discovered_1"], (
        f"slot names not restored: {g2.slot_names}")
    assert len(g2._signatures) == 2, "signatures not restored"
    c1 = g1.competence.predict_all()[:2]
    c2 = g2.competence.predict_all()[:2]
    assert np.allclose(c1, c2, atol=1e-5), (
        f"competence not restored: {c1} vs {c2} — fresh 0.5s would recreate "
        "the clobber tie")
    assert abs(c2[0] - 0.5) > 0.05, "restored competence is still the prior"
    # THE identity contract: the same physical event, re-observed in the new
    # process, lands on its ORIGINAL slot instead of minting a new one.
    env.spike(1)
    g2.update(env)
    assert len(g2.slot_names) == 2, (
        "re-observed event minted a NEW slot — identity not stable")
    assert len(g2.unlock_log) >= 3, "unlock_log lost across restore"
    dag = g2.estimated_dag()
    assert isinstance(dag, dict) and len(dag) >= 1
    print("  6. broadcaster identity round-trip ok "
          f"(competence {c2[0]:.2f} restored, re-observation -> same slot)")



def test_precondition_retrieval():
    """Layer 3: retrieval by grounded predicates is legible and ranked by
    match x competence; skills without preconditions are not retrievable
    here; min_match excludes half-matches."""
    from developmental_ai.skill_bank.skill_bank import SkillBank
    import torch
    bank = SkillBank(storage_dir=SB)
    bank.save_skill("chop", "chop_trunk", {"w": torch.zeros(1)},
                    success_rate=0.6, total_episodes=5, dedup=True,
                    preconditions={"tree_visible": True,
                                   "breakable_in_reach": True})
    bank.save_skill("swim", "cross_water", {"w": torch.zeros(1)},
                    success_rate=0.9, total_episodes=5, dedup=True,
                    preconditions={"water_visible": True,
                                   "tree_visible": False})
    bank.save_skill("legacy", "legacy_skill", {"w": torch.zeros(1)},
                    success_rate=0.9, total_episodes=5, dedup=True)

    at_tree = {"tree_visible": True, "breakable_in_reach": True,
               "water_visible": False}
    got = bank.retrieve_by_preconditions(at_tree, top_k=8, min_match=0.75)
    ids = [s.skill_id for s, _ in got]
    assert ids and ids[0] == "chop", f"wrong ranking at a tree: {ids}"
    assert "swim" not in ids, "half-matching skill leaked past min_match"
    assert "legacy" not in ids, "precondition-less skill retrievable"

    in_water = {"water_visible": True, "tree_visible": False}
    got2 = bank.retrieve_by_preconditions(in_water, top_k=8)
    assert got2 and got2[0][0].skill_id == "swim", "swim not top in water"
    # registry round-trip: preconditions survive save/load
    bank2 = SkillBank(storage_dir=SB)
    assert bank2.skills["chop"].preconditions == {
        "tree_visible": True, "breakable_in_reach": True}
    print("  7. precondition retrieval ok (legible, ranked, persisted)")


if __name__ == "__main__":
    for fn in (test_tie_does_not_clobber, test_practice_floor,
               test_winner_replaces_and_versions, test_registry_resilience,
               test_honest_mastery_and_selection,
               test_broadcaster_identity_roundtrip,
               test_precondition_retrieval):
        print(f"[skill-identity-smoke] {fn.__name__}")
        fn()
    print("[skill-identity-smoke] ALL PASS")
