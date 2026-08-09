"""Brain-state emitter smoke + viewer demo data (Mac-ok).

Contracts:
  1. BrainStateEmitter.emit() writes valid JSON with skill nodes, concept
     nodes, GHOST nodes (unresolved links), and earned edges only.
  2. Firing feed: option_started/option_ended round-trip; active map
     reflects a running option.
  3. Precedence edges come from broadcaster.estimated_dag() evidence.
  4. Notes' effects/prerequisites become achieves/prerequisite edges.
Also writes viewer/brain_state.json so brain_viewer.html renders a real
picture (served locally for eyeballing).
"""
import json
import os
import shutil

import numpy as np
import torch

SB = "/tmp/brain_demo_sb"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "viewer", "brain_state.json")


def build():
    from developmental_ai.skill_bank.skill_bank import SkillBank
    from developmental_ai.skill_bank import skill_notes
    from developmental_ai.skill_bank.brain_state import BrainStateEmitter
    from developmental_ai.core.achievement_goals import (
        DiscoveredAchievementGoals)
    from developmental_ai.knowledge_graph.knowledge_graph import (
        InMemoryKnowledgeGraph, SymbolicFact)

    shutil.rmtree(SB, ignore_errors=True)
    bank = SkillBank(storage_dir=SB)

    skills = [
        ("ach_00_discovered_0", "approach_tree_line", 0.61, 14,
         {"tree_visible": True, "open_space_ahead": True},
         ["oak_leaves"], []),
        ("ach_01_discovered_1", "clear_tall_grass", 0.55, 11,
         {"grass_visible": True, "breakable_in_reach": True},
         ["tall_grass", "fern"], []),
        ("ach_02_discovered_2", "chop_oak_trunk", 0.48, 9,
         {"tree_visible": True, "object_adjacent": True,
          "breakable_in_reach": True},
         ["oak_log"], ["ach_00_discovered_0"]),
        ("ach_03_discovered_3", "strip_low_leaves", 0.52, 8,
         {"leaves_visible": True, "breakable_in_reach": True},
         ["oak_leaves", "jungle_leaves"], ["ach_00_discovered_0"]),
        ("ach_04_discovered_4", "chop_birch_trunk", 0.33, 4,
         {"tree_visible": True, "object_adjacent": True},
         ["birch_log"], ["ach_02_discovered_2"]),
        ("ach_05_discovered_5", "dig_through_dirt", 0.29, 3,
         {"dirt_visible": True, "looking_down": True},
         ["dirt", "grass_block"], []),
    ]
    for sid, name, comp, eps, pre, effects, prereqs in skills:
        sk = bank.save_skill(
            sid, name, {"w": torch.zeros(2)},
            description=f"{name.replace('_', ' ').capitalize()}.",
            success_rate=comp, total_episodes=eps, dedup=True,
            preconditions=pre)
        skill_notes.write_note(
            sk, os.path.join(SB, sid), effects=effects,
            preconditions=pre, prerequisites=prereqs,
            provenance={"episode": eps, "timestep": eps * 4000,
                        "env": "TreechopFixed"})

    # broadcaster with real unlock evidence -> precedence edges
    g = DiscoveredAchievementGoals(max_slots=16, seed=0)
    for i, (sid, name, *_r) in enumerate(skills):
        g._slot_for(("disc", i), f"discovered_{i}")
    # discovered_2 (chop_oak) unlocked with 0 and 1 already achieved
    for i in range(6):
        g._achieved_now[0, :] = 0
        if i >= 2:
            g._achieved_now[0, 0] = 1.0
            g._achieved_now[0, 1] = 1.0
        g._record_unlock(2, 0)
    # discovered_4 unlocked with 2 achieved
    for i in range(3):
        g._achieved_now[0, :] = 0
        g._achieved_now[0, 2] = 1.0
        g._record_unlock(4, 0)

    # knowledge graph: grounded causal facts + one UNRESOLVED concept
    kg = InMemoryKnowledgeGraph()
    for obj in ("oak_log", "birch_log", "tall_grass"):
        kg.add_fact(SymbolicFact("attack_held", "breaks", obj, 1.0,
                                 source="grounded_event", timestamp=42))
    # the ghost: knowledge references planks; nothing achieves it
    kg.add_fact(SymbolicFact("oak_log", "crafts_into", "oak_planks", 1.0,
                             source="grounded_event", timestamp=99))

    em = BrainStateEmitter(bank, broadcaster=g, knowledge_graph=kg,
                           out_path=OUT)
    # firing feed: two finished invocations + one live
    em.option_started(1, "ach_01_discovered_1", 3200)
    em.option_ended(1, "ach_01_discovered_1", 3218, "achieved")
    em.option_started(3, "ach_02_discovered_2", 3300)
    em.option_ended(3, "ach_02_discovered_2", 3340, "horizon")
    em.option_started(0, "ach_02_discovered_2", 3391)
    path = em.emit(episode=31, timestep=3391 * 4)
    return path


def verify(path):
    d = json.load(open(path))
    kinds = {n["kind"] for n in d["nodes"]}
    assert "skill" in kinds and "concept" in kinds, kinds
    assert "ghost" in kinds, (
        "no unresolved-link ghost node — oak_planks should be one")
    ghost = [n for n in d["nodes"] if n["kind"] == "ghost"]
    assert any(n["label"] == "oak_planks" for n in ghost), ghost
    ekinds = {e["kind"] for e in d["edges"]}
    assert {"achieves", "prerequisite", "causal"} <= ekinds, ekinds
    assert "precedence" in ekinds, (
        "estimated_dag evidence produced no precedence edges")
    sim_edges = [e for e in d["edges"] if "similar" in e["kind"]]
    assert not sim_edges, "embedding-similarity edges are banned"
    assert d["active"]["0"] == "ach_02_discovered_2", d["active"]
    running = [f for f in d["firing"] if f["outcome"] == "running"]
    assert len(running) == 1 and running[0]["env"] == 0
    done = [f for f in d["firing"] if f["outcome"] == "achieved"]
    assert done and done[0]["t_end"] == 3218
    skill_nodes = [n for n in d["nodes"] if n["kind"] == "skill"]
    assert all(n.get("note_md") for n in skill_nodes), "notes missing"
    print(f"  nodes={len(d['nodes'])} ({len(skill_nodes)} skills, "
          f"{len(ghost)} ghost), edges={len(d['edges'])} "
          f"kinds={sorted(ekinds)}")
    print("[brain-state-demo] ALL PASS")


if __name__ == "__main__":
    verify(build())
