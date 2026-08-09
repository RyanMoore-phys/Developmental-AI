"""Backfill + ghost-frontier + precondition-gating smoke (Mac-ok).

Contracts:
  1. LEGACY SKILLS STILL WORK: a bank built exactly like the pod's 16
     (action_dim == P, 34-d one-hot context, placeholder name, no
     preconditions, RELATIVE policy_path) binds to option slots, has its
     path repaired, and executes.
  2. Metadata backfill: an upsert whose weights LOSE still replaces a
     placeholder name and fills missing preconditions — and never touches
     a real name or the winning weights.
  3. Note backfill writes thin, honest notes for note-less skills,
     including observed prerequisites from the broadcaster's unlock log.
  4. GATING NEVER DISABLES THE LEGACY BANK: with gating on, skills with no
     preconditions stay offered; a skill whose preconditions are unmet is
     withheld; primitives are always offered; no predicates -> inert.
  5. Ghost frontier: an unresolved concept boosts the STEPPING-STONE slot
     (the one achieving the subject that reaches it), never invents a slot,
     and is capped; a fully-resolved graph produces no bonus.
  6. Frontier bonus actually moves IMGEP target selection, and survives a
     save_state/load_state round trip.
"""
import json
import os
import shutil

import numpy as np
import torch

SB = "/tmp/brain_upgrades_sb"
P = 10
OBS = 64
KDIM = 34          # 2*16+2, matching goals.max_slots=16


def _legacy_bank():
    """Reproduce the pod's 16-skill shape as closely as possible."""
    from developmental_ai.skill_bank.skill_bank import SkillBank
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    shutil.rmtree(SB, ignore_errors=True)
    bank = SkillBank(storage_dir=SB)
    for i in range(3):
        pol = StandaloneActorCritic(
            obs_dim=OBS, action_dim=P, hidden_dim=32, continuous=False,
            knowledge_dim=KDIM, device=torch.device("cpu"))
        ctx = np.zeros(KDIM, dtype=np.float32)
        ctx[i] = 1.0
        ctx[2 * 16] = 1.0                      # has-target flag
        bank.save_skill(f"ach_{i:02d}_discovered_{i}",
                        f"Achieve: discovered_{i}",     # placeholder
                        pol.get_state_dict(),
                        success_rate=0.5, total_episodes=8, dedup=True,
                        context_embedding=ctx, obs_dim=OBS, action_dim=P)
    # simulate a legacy registry: relative policy paths, no preconditions
    reg = os.path.join(SB, "registry.json")
    d = json.load(open(reg))
    for e in d["skills"]:
        e["policy_path"] = "./skill_bank_x/{}/policy.pt".format(e["skill_id"])
        e["preconditions"] = None
    json.dump(d, open(reg, "w"))
    return SkillBank(storage_dir=SB)           # reload -> path repair


def test_legacy_skills_still_work():
    from developmental_ai.policy.options import SkillOptionBank
    bank = _legacy_bank()
    sk = bank.skills["ach_00_discovered_0"]
    assert os.path.isabs(sk.policy_path), "relative legacy path not repaired"
    assert os.path.exists(sk.policy_path), sk.policy_path
    ob = SkillOptionBank(bank, OBS, P, 4, torch.device("cpu"))
    ob.refresh_slots()
    bound = [b for b in ob.slots if b is not None]
    assert len(bound) == 3, f"legacy skills failed to bind: {ob.refused}"
    b = ob.slots[0]
    assert b["kdim"] == KDIM and b["ctx"] is not None
    assert b["ctx"][0] == 1.0 and b["ctx"][32] == 1.0
    a = ob.skill_action(0, np.zeros(OBS, np.float32))
    assert 0 <= a < P
    print(f"  1. legacy bank binds + executes ok ({len(bound)}/3 slots)")


def test_metadata_backfill():
    from developmental_ai.skill_bank.skill_bank import SkillBank
    bank = SkillBank(storage_dir=SB)
    before = torch.load(bank.skills["ach_00_discovered_0"].policy_path,
                        weights_only=True)["actor"]["action_head.bias"].clone()
    # challenger LOSES on weights (equal success, no practice) but offers a
    # real name + preconditions
    bank.save_skill("ach_00_discovered_0", "chop_oak_trunk",
                    {"actor": {"action_head.bias": torch.ones(P)}},
                    description="Swing at a trunk until it breaks.",
                    success_rate=0.5, total_episodes=1, dedup=True,
                    preconditions={"tree_visible": True})
    sk = bank.skills["ach_00_discovered_0"]
    assert sk.name == "chop_oak_trunk", f"placeholder not replaced: {sk.name}"
    assert sk.preconditions == {"tree_visible": True}
    after = torch.load(sk.policy_path,
                       weights_only=True)["actor"]["action_head.bias"]
    assert torch.equal(before, after), "losing challenger overwrote weights"
    # a REAL name is never overwritten by a placeholder
    bank.save_skill("ach_00_discovered_0", "Achieve: discovered_0",
                    {"actor": {"action_head.bias": torch.zeros(P)}},
                    success_rate=0.5, total_episodes=1, dedup=True)
    assert bank.skills["ach_00_discovered_0"].name == "chop_oak_trunk"
    print("  2. metadata backfill ok (name+preconditions, weights intact)")


def test_note_backfill():
    from developmental_ai.skill_bank.skill_bank import SkillBank
    from developmental_ai.skill_bank.skill_notes import (
        backfill_notes, read_note_frontmatter)
    from developmental_ai.core.achievement_goals import (
        DiscoveredAchievementGoals)
    bank = SkillBank(storage_dir=SB)
    g = DiscoveredAchievementGoals(max_slots=16, seed=0)
    for i in range(3):
        g._slot_for(("disc", i), f"discovered_{i}")
    g._achieved_now[0, 0] = 1.0        # slot 0 achieved before slot 2
    g._record_unlock(2, 0)
    n = backfill_notes(bank, g)
    assert n == 3, f"expected 3 notes, wrote {n}"
    front = read_note_frontmatter(os.path.join(SB, "ach_02_discovered_2"))
    assert front is not None
    assert front["provenance"].get("backfilled") is True
    assert "ach_00_discovered_0" in front["prerequisites"], front
    assert backfill_notes(bank, g) == 0, "backfill not idempotent"
    print("  3. note backfill ok (observed prerequisites, idempotent)")


def test_gating_never_disables_legacy():
    from developmental_ai.policy.options import (
        OptionExecutor, SkillOptionBank)
    from developmental_ai.skill_bank.skill_bank import SkillBank
    bank = SkillBank(storage_dir=SB)
    # ach_00 now HAS preconditions (from the backfill test); 01/02 do not
    ob = SkillOptionBank(bank, OBS, P, 4, torch.device("cpu"))
    ob.refresh_slots()
    slot_of = {b["skill_id"]: i for i, b in enumerate(ob.slots)
               if b is not None}
    s0 = slot_of["ach_00_discovered_0"]

    # no predicates at all -> inert, everything offered
    m_none = ob.mask(None)
    assert m_none[:P].all() and m_none.sum() == P + 3

    # preconditions MET -> offered
    m_ok = ob.mask({"tree_visible": True}, min_match=0.6)
    assert m_ok[P + s0], "met preconditions withheld the skill"
    # preconditions UNMET -> withheld, but the others stay offered
    m_no = ob.mask({"tree_visible": False}, min_match=0.6)
    assert not m_no[P + s0], "unmet preconditions still offered"
    assert m_no[:P].all(), "primitives were gated"
    others = [i for sid, i in slot_of.items() if i != s0]
    assert all(m_no[P + i] for i in others), (
        "precondition-less legacy skills were disabled by gating")

    ex = OptionExecutor(ob, num_envs=2,
                        cfg={"gate_by_preconditions": True,
                             "max_option_steps": 4})
    assert ex.gate_by_preconditions
    snap = ex.snapshot()
    assert snap["gating"]["enabled"] is True
    json.dumps(snap)
    print("  4. gating ok (unmet withheld, legacy + primitives never gated)")


def test_ghost_frontier():
    from developmental_ai.skill_bank.skill_bank import SkillBank
    from developmental_ai.skill_bank import skill_notes
    from developmental_ai.skill_bank.brain_state import (
        compute_ghost_frontier)
    from developmental_ai.core.achievement_goals import (
        DiscoveredAchievementGoals)
    from developmental_ai.knowledge_graph.knowledge_graph import (
        InMemoryKnowledgeGraph, SymbolicFact)
    bank = SkillBank(storage_dir=SB)
    # slot 0's skill achieves oak_log; slot 1's achieves grass
    skill_notes.write_note(bank.skills["ach_00_discovered_0"],
                           os.path.join(SB, "ach_00_discovered_0"),
                           effects=["oak_log"])
    skill_notes.write_note(bank.skills["ach_01_discovered_1"],
                           os.path.join(SB, "ach_01_discovered_1"),
                           effects=["grass"])
    g = DiscoveredAchievementGoals(max_slots=16, seed=0)
    for i in range(3):
        g._slot_for(("disc", i), f"discovered_{i}")

    kg = InMemoryKnowledgeGraph()
    kg.add_fact(SymbolicFact("attack_held", "breaks", "oak_log", 1.0,
                             source="grounded_event"))
    # THE GAP: oak_log reaches planks, and nothing achieves planks
    kg.add_fact(SymbolicFact("oak_log", "crafts_into", "oak_planks", 1.0,
                             source="grounded_event"))
    bonus = compute_ghost_frontier(bank, kg, g)
    s0 = 0
    assert s0 in bonus, f"stepping-stone slot not boosted: {bonus}"
    assert 1 not in bonus, "unrelated slot boosted"
    assert all(v > 1.0 for v in bonus.values())
    # no gaps -> no bonus
    kg2 = InMemoryKnowledgeGraph()
    kg2.add_fact(SymbolicFact("attack_held", "breaks", "oak_log", 1.0,
                              source="grounded_event"))
    assert compute_ghost_frontier(bank, kg2, g) == {}, (
        "bonus produced with nothing unresolved")
    print(f"  5. ghost frontier ok (boosted stepping-stone slot {sorted(bonus)})")


def test_frontier_moves_selection_and_persists():
    from developmental_ai.core.achievement_goals import (
        DiscoveredAchievementGoals)
    g = DiscoveredAchievementGoals(max_slots=8, seed=0, epsilon=0.0)
    for i in range(3):
        g._slot_for(("disc", i), f"discovered_{i}")
    # make slot 2 clearly the default frontier pick
    for _ in range(30):
        g.competence.update(0, True)
        g.competence.update(1, True)
    g._attempts[:3] = 5
    base = g._select_target()
    g.set_frontier_bonus({0: 2.0})
    boosted = g._select_target()
    assert boosted == 0 and base != 0, (
        f"frontier bonus did not move selection: {base} -> {boosted}")
    # cap respected + round-trip
    g.set_frontier_bonus({0: 99.0}, cap=2.0)
    assert g._frontier_bonus[0] == 2.0
    path = os.path.join(SB, "bc_state.json")
    g.save_state(path)
    g2 = DiscoveredAchievementGoals(max_slots=8, seed=1)
    assert g2.load_state(path)
    assert g2._frontier_bonus.get(0) == 2.0, "frontier bonus lost on reload"
    print("  6. frontier moves selection, capped, and persists ok")
def test_reground_hallucinated_names():
    """The 16 legacy skills carry run-7 VLM hallucinations ('fish_blue_water').
    (a) startup reground replaces them with grounded names from note effects;
    (b) a grounded name from re-mint overrides a hallucination; (c) grounded
    names are never overwritten by another grounded name."""
    from developmental_ai.skill_bank.skill_bank import (
        SkillBank, is_grounded_name, is_replaceable_name)
    from developmental_ai.skill_bank import skill_notes
    import torch, os as _os
    shutil.rmtree("/tmp/reground_sb", ignore_errors=True)
    bank = SkillBank(storage_dir="/tmp/reground_sb")
    # a hallucinated skill WITH note effects, and one WITHOUT
    for sid, nm, eff in [("ach_00_discovered_0", "fish_blue_water",
                          ["oak_log", "grass"]),
                         ("ach_01_discovered_1", "collect_heart_items", [])]:
        sk = bank.save_skill(sid, nm, {"w": torch.zeros(1)},
                             success_rate=0.5, total_episodes=5, dedup=True)
        skill_notes.write_note(sk, _os.path.join("/tmp/reground_sb", sid),
                               effects=eff)
    assert is_replaceable_name("fish_blue_water")       # hallucination
    assert not is_grounded_name("collect_heart_items")

    n = skill_notes.reground_names(bank)
    assert n == 2, f"regrounded {n}, expected 2"
    # effects -> break_<highest-tier>; oak_log outranks grass
    assert bank.skills["ach_00_discovered_0"].name == "break_oak_log", \
        bank.skills["ach_00_discovered_0"].name
    # no effects -> neutral slot name
    assert bank.skills["ach_01_discovered_1"].name == "skill_slot_01"
    # persisted
    assert SkillBank(storage_dir="/tmp/reground_sb").skills[
        "ach_00_discovered_0"].name == "break_oak_log"

    # re-mint override: a fresh grounded name replaces even a hallucination
    bank.save_skill("ach_02_discovered_2", "craft_wooden_planks",
                    {"w": torch.zeros(1)}, success_rate=0.5,
                    total_episodes=5, dedup=True)
    bank.save_skill("ach_02_discovered_2", "break_spruce_log",
                    {"w": torch.ones(1)}, success_rate=0.9,
                    total_episodes=6, dedup=True)  # winner + grounded
    assert bank.skills["ach_02_discovered_2"].name == "break_spruce_log"
    # grounded is NOT overwritten by another grounded (first grounding wins)
    bank.save_skill("ach_02_discovered_2", "break_oak_log",
                    {"w": torch.zeros(1)}, success_rate=0.5,
                    total_episodes=1, dedup=True)  # loser
    assert bank.skills["ach_02_discovered_2"].name == "break_spruce_log"
    print("  7. reground + re-mint override ok (no more hallucinated names)")


if __name__ == "__main__":
    for fn in (test_legacy_skills_still_work, test_metadata_backfill,
               test_note_backfill, test_gating_never_disables_legacy,
               test_ghost_frontier,
               test_frontier_moves_selection_and_persists,
               test_reground_hallucinated_names):
        print(f"[brain-upgrades-smoke] {fn.__name__}")
        fn()
    print("[brain-upgrades-smoke] ALL PASS")
