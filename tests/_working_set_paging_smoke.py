"""Working-set <-> long-term paging smoke (Mac-ok, no torch-heavy env).

The lifelong memory hierarchy: a fixed-cap goal registry (the network-shape
WORKING SET) pages against an unbounded LongTermStore (disk). Contracts:

  1. Paging OFF -> fixed-cap behaviour is unchanged (full registry refuses new
     kinds; no free slots ever appear).
  2. Paging ON -> a full registry EVICTS the stalest low-value slot to disk and
     reuses it, so total behaviours known exceeds max_slots while the working
     set stays bounded.
  3. Re-encountering an evicted behaviour's cue RECALLS it back into a slot
     (competence + attempts restored, not re-learned).
  4. Eviction is value-weighted: a mastered (high-competence) idle slot
     survives longer than a weak one.
  5. Target selection never picks a freed (empty) slot.
  6. Persistence round-trips: registry + dormant index + free slots survive a
     save/reload, and recall still works afterward.
  7. CompetencePredictor slot export/reset/import is exact.
"""
import os
import shutil

import numpy as np

from developmental_ai.core.achievement_goals import DiscoveredAchievementGoals
from developmental_ai.core.self_model import CompetencePredictor
from developmental_ai.memory.long_term_store import LongTermStore

SIG = 192  # obs size == pool -> pooled delta passes through 1:1 (clean cues)


class FakeEnv:
    def __init__(self):
        self.reward_history = []
        self.obs_history = []


def _feed(b, env, k, reward=1.0):
    """Present a DISTINCT behaviour k (orthogonal one-hot obs-delta)."""
    base = np.zeros(SIG, np.float32)
    ev = base.copy()
    ev[k] = 1.0
    env.obs_history = [base, ev]
    env.reward_history = [float(reward)]
    b.update(env, stream=0)


def _feed_sig(b, env, sig, reward=1.0):
    base = np.zeros(SIG, np.float32)
    env.obs_history = [base, np.asarray(sig, np.float32)]
    env.reward_history = [float(reward)]
    b.update(env, stream=0)


def _active_slots(b):
    return sorted(set(b.slot_of.values()))


def _tmp(name):
    d = os.path.join("/tmp", name)
    shutil.rmtree(d, ignore_errors=True)
    return d


def test_off_path_fixed_cap():
    b = DiscoveredAchievementGoals(max_slots=3, seed=0, pool=SIG,
                                   match_cosine=0.8)
    env = FakeEnv()
    for k in range(6):            # 6 distinct behaviours into a 3-slot registry
        _feed(b, env, k)
        b.reset()
    assert len(_active_slots(b)) == 3, _active_slots(b)
    assert len(b.slot_names) == 3
    assert not b._free_slots, "fixed-cap path must never free a slot"
    assert not b.paging_on
    print("  1. off-path fixed cap ok (3/3 slots, extras refused, no paging)")


def test_page_out_on_overflow():
    ltm = LongTermStore(_tmp("wsp_out"), cue_dim=SIG)
    b = DiscoveredAchievementGoals(max_slots=4, seed=1, pool=SIG,
                                   match_cosine=0.8)
    b.attach_long_term_store(ltm, min_idle_episodes=1, recall_threshold=0.9)
    env = FakeEnv()
    for k in range(4):            # fill the working set, advancing episodes
        _feed(b, env, k)
        b.reset()
    assert len(_active_slots(b)) == 4
    assert ltm.stats()["dormant"] == 0
    _feed(b, env, 50)            # a NEW distinct behaviour -> must evict + reuse
    po, _ = b.drain_page_events()
    assert len(po) >= 1, "overflow did not page a slot out"
    assert len(_active_slots(b)) == 4, "working set must stay bounded"
    assert ltm.stats()["dormant"] >= 1, "evicted behaviour not on disk"
    # total knowledge (active working set + dormant on disk) exceeds max_slots
    assert ltm.stats()["total"] >= 5, (
        f"total knowledge should exceed max_slots: {ltm.stats()}")
    assert ltm.stats()["active"] == 4, "active index should mirror the 4 slots"
    print(f"  2. page-out ok (evicted 1, working set still 4, "
          f"total known={ltm.stats()['total']})")


def test_recall_on_cue():
    ltm = LongTermStore(_tmp("wsp_recall"), cue_dim=SIG)
    b = DiscoveredAchievementGoals(max_slots=3, seed=2, pool=SIG,
                                   match_cosine=0.8)
    b.attach_long_term_store(ltm, min_idle_episodes=1, recall_threshold=0.9)
    env = FakeEnv()
    for k in range(3):
        _feed(b, env, k)
        for _ in range(2):
            b.reset()
    _feed(b, env, 77)           # overflow -> evict the stalest
    b.drain_page_events()
    dorm = ltm.dormant_ids("goal")
    assert dorm, "nothing paged out"
    victim = ltm.items[dorm[0]]
    vsig = np.asarray(victim.extra["sig"], np.float32)
    saved_attempts = victim.extra["attempts"]
    _feed_sig(b, env, vsig)     # re-encounter the evicted behaviour's cue
    _, pin = b.drain_page_events()
    assert len(pin) >= 1, "evicted behaviour was NOT recalled on re-cue"
    # the working set was full, so recall pages this memory back to ACTIVE
    # (evicting a different slot to make room — dormant count nets out, but
    # THIS item transitioned dormant -> active).
    assert ltm.items[dorm[0]].active is True, "recalled item still dormant"
    slot = pin[0]["slot"]
    assert b._slot_uid.get(slot) == dorm[0], "recalled slot not linked to memory"
    assert b._attempts[slot] == saved_attempts, "attempts not restored on recall"
    print("  3. recall ok (dormant cue -> paged back in, competence restored)")


def test_value_weighted_eviction():
    ltm = LongTermStore(_tmp("wsp_value"), cue_dim=SIG)
    b = DiscoveredAchievementGoals(max_slots=3, seed=3, pool=SIG,
                                   match_cosine=0.8)
    b.attach_long_term_store(ltm, min_idle_episodes=1, recall_threshold=0.9)
    env = FakeEnv()
    for k in range(3):
        _feed(b, env, k)
        b.reset()
    # make slot 0 MASTERED (high competence) and slot 1 WEAK, both idle
    for _ in range(30):
        b.competence.update(0, 1.0)   # slot 0 -> ~1.0
        b.competence.update(1, 0.0)   # slot 1 -> ~0.0
    b.target = 2                      # protect slot 2 from eviction
    victim = b._pick_evictable()
    assert victim == 1, (
        f"value-weighted eviction should drop the WEAK slot 1, got {victim} "
        f"(comp={b.competence.predict_all()[:3]})")
    print("  4. value-weighted eviction ok (weak slot evicted before mastered)")


def test_target_skips_freed_slot():
    ltm = LongTermStore(_tmp("wsp_tgt"), cue_dim=SIG)
    b = DiscoveredAchievementGoals(max_slots=4, seed=4, pool=SIG,
                                   match_cosine=0.8, epsilon=0.5)
    b.attach_long_term_store(ltm, min_idle_episodes=1, recall_threshold=0.9)
    env = FakeEnv()
    for k in range(4):
        _feed(b, env, k)
        b.reset()
    # SELF-PACKING invariant: overflow evicts-and-reuses atomically, so no hole
    # persists at rest — the registry stays fully packed at max_slots.
    _feed(b, env, 88)
    assert not b._free_slots, "overflow should reuse the freed slot, not hole it"
    assert sorted(_active_slots(b)) == [0, 1, 2, 3], "registry not fully packed"
    # SAFETY NET: force a hole WITHOUT refilling (a direct page-out), then the
    # empty slot must never be chosen as a target across many resets.
    b.target = None
    b._page_out_slot(2)
    b.drain_page_events()
    assert 2 in b._free_slots
    for _ in range(200):
        b.reset()
        assert b.target != 2, f"target selection picked the empty slot: {b.target}"
    print("  5. self-packing (no rest holes) + target skips a forced hole ok")


def test_persistence_roundtrip():
    d_ltm, d_bc = _tmp("wsp_persist_ltm"), _tmp("wsp_persist_bc")
    os.makedirs(d_bc, exist_ok=True)
    bc_path = os.path.join(d_bc, "broadcaster.json")

    ltm = LongTermStore(d_ltm, cue_dim=SIG)
    b = DiscoveredAchievementGoals(max_slots=3, seed=5, pool=SIG,
                                   match_cosine=0.8)
    b.attach_long_term_store(ltm, min_idle_episodes=1, recall_threshold=0.9)
    env = FakeEnv()
    for k in range(3):
        _feed(b, env, k)
        for _ in range(2):
            b.reset()
    _feed(b, env, 99)            # force one page-out
    b.drain_page_events()
    b.save_state(bc_path)
    ltm.save()
    active_before = len(_active_slots(b))
    dormant_before = ltm.stats()["dormant"]
    victim_sig = np.asarray(ltm.items[ltm.dormant_ids("goal")[0]].extra["sig"],
                            np.float32)

    # fresh process: reload both
    ltm2 = LongTermStore(d_ltm, cue_dim=SIG)
    b2 = DiscoveredAchievementGoals(max_slots=3, seed=5, pool=SIG,
                                    match_cosine=0.8)
    assert b2.load_state(bc_path), "broadcaster reload failed"
    b2.attach_long_term_store(ltm2, min_idle_episodes=1, recall_threshold=0.9)
    assert len(_active_slots(b2)) == active_before, "active slots not restored"
    assert ltm2.stats()["dormant"] == dormant_before, "dormant index lost"
    assert b2._uid_next > 0, "uid counter not restored (would collide on mint)"
    # recall still works after a full round-trip (age the slots first so the
    # full working set has an evictable occupant to make room for the recall)
    for _ in range(3):
        b2.reset()
    _feed_sig(b2, FakeEnv(), victim_sig)
    _, pin = b2.drain_page_events()
    assert len(pin) >= 1, "recall broken after persistence round-trip"
    print("  6. persistence round-trip ok (registry + dormant index + recall)")


def test_competence_slot_paging():
    c = CompetencePredictor(n_tasks=4)
    for _ in range(20):
        c.update(2, 1.0)         # train slot 2 up
    p2 = c.predict(2)
    assert p2 > 0.7, p2
    state = c.export_slot(2)
    assert state["n"] == 20 and len(state["hist"]) > 0
    c.reset_slot(2)
    # reset returns the slot to the fresh prior = what a NEVER-trained slot
    # predicts (the shared bias base-rate), i.e. it inherits no competence.
    assert abs(c.predict(2) - c.predict(3)) < 1e-6, (
        f"reset slot != untrained slot: {c.predict(2)} vs {c.predict(3)}")
    assert c.n_updates(2) == 0
    c.import_slot(2, state)
    assert abs(c.predict(2) - p2) < 1e-6, "import did not restore competence"
    assert c.n_updates(2) == 20
    print("  7. competence slot export/reset/import exact")


def _feed_effect(b, env, effect, sig_k, reward=1.0):
    """A grounded spike: block `effect` broke, with pixel-delta signature at
    index `sig_k` (so we can VARY the signature while keeping the effect)."""
    base = np.zeros(SIG, np.float32)
    ev = base.copy()
    ev[sig_k] = 1.0
    env.obs_history = [base, ev]
    env.reward_history = [float(reward)]
    b.update(env, stream=0, effect_key=effect)


def test_effect_keying_dedups():
    """THE duplication fix: the same grounded behaviour must key ONE slot even
    when its pixel-delta signature varies wildly across views (the real bug was
    break_birch_leaves occupying 8 slots). Distinct effects still get distinct
    slots."""
    b = DiscoveredAchievementGoals(max_slots=16, seed=10, pool=SIG,
                                   match_cosine=0.95)   # strict: sigs won't cluster
    env = FakeEnv()
    # break "birch_leaves" 8 times, each with a DIFFERENT signature index
    for k in range(8):
        _feed_effect(b, env, "birch_leaves", sig_k=k)
        b.reset()
    assert len(_active_slots(b)) == 1, (
        f"same grounded behaviour minted {len(_active_slots(b))} slots (dup!)")
    assert ("effect", "birch_leaves") in b.slot_of
    assert b.slot_names[_active_slots(b)[0]] == "break_birch_leaves"
    # a DIFFERENT effect gets its own slot
    _feed_effect(b, env, "grass_block", sig_k=20)
    b.reset()
    assert len(_active_slots(b)) == 2, "distinct effect should mint a 2nd slot"
    print("  10. effect-keying dedups (birch_leaves x8 -> 1 slot; grass -> 2nd)")


def test_effect_recall_dedups_across_paging():
    """Redundancy must be disallowed even ACROSS paging: after an effect slot
    is paged out, re-encountering that behaviour RECALLS the same memory (O(1)
    by deterministic effect id) instead of minting a duplicate."""
    ltm = LongTermStore(_tmp("wsp_effect_recall"), cue_dim=SIG)
    b = DiscoveredAchievementGoals(max_slots=2, seed=11, pool=SIG,
                                   match_cosine=0.95)
    b.attach_long_term_store(ltm, min_idle_episodes=1, recall_threshold=0.95)
    env = FakeEnv()
    _feed_effect(b, env, "oak_log", sig_k=0); b.reset()
    oak_uid = b._effect_uid("oak_log")
    # fill + overflow so oak_log is the stalest and gets paged out
    _feed_effect(b, env, "stone", sig_k=5); b.reset()
    _feed_effect(b, env, "iron_ore", sig_k=9); b.reset()   # overflow -> evict
    b.drain_page_events()
    assert not ltm.items[oak_uid].active, "oak_log should be dormant"
    n_before = len(ltm.dormant_ids("goal")) + len(_active_slots(b))
    # re-encounter oak_log with a TOTALLY different signature -> must RECALL,
    # not mint a new slot
    for _ in range(3):
        b.reset()
    _feed_effect(b, env, "oak_log", sig_k=40)
    _, pin = b.drain_page_events()
    assert len(pin) >= 1 and pin[0]["mem_id"] == oak_uid, "did not recall oak_log"
    assert ltm.items[oak_uid].active, "oak_log not reactivated"
    n_after = len(ltm.dormant_ids("goal")) + len(_active_slots(b))
    assert n_after == n_before, (
        f"recall created a duplicate behaviour: {n_before} -> {n_after}")
    print("  11. effect recall dedups across paging (no duplicate on re-cue)")


def test_grounded_adopts_ungrounded_slot():
    """A behaviour first seen UNGROUNDED (signature slot) then seen GROUNDED
    must ADOPT the existing slot (alias the effect key), never duplicate."""
    b = DiscoveredAchievementGoals(max_slots=8, seed=12, pool=SIG,
                                   match_cosine=0.8)
    env = FakeEnv()
    _feed(b, env, 3); b.reset()                 # ungrounded -> ("disc",0) slot
    assert len(_active_slots(b)) == 1
    # same behaviour (matching signature) now arrives grounded
    _feed_effect(b, env, "cobblestone", sig_k=3); b.reset()
    assert len(_active_slots(b)) == 1, "grounded view duplicated the behaviour"
    slot = _active_slots(b)[0]
    assert b.slot_of.get(("effect", "cobblestone")) == slot, "effect not aliased"
    print("  12. grounded spike adopts the matching ungrounded slot (no dup)")


def test_adopt_then_pageout_recall_no_dup():
    """GATE BLOCKER: a grounded spike that ADOPTS a pre-existing ungrounded
    signature slot must re-key it to the effect uid, so after the slot is paged
    out a re-encounter (even with a different signature) RECALLS it by effect
    instead of minting a duplicate goal + skill."""
    ltm = LongTermStore(_tmp("wsp_adopt_recall"), cue_dim=SIG)
    b = DiscoveredAchievementGoals(max_slots=3, seed=13, pool=SIG,
                                   match_cosine=0.8)
    b.attach_long_term_store(ltm, min_idle_episodes=1, recall_threshold=0.95)
    env = FakeEnv()
    _feed(b, env, 3); b.reset()                         # ungrounded ('disc',0)
    _feed_effect(b, env, "cobblestone", sig_k=3); b.reset()   # ADOPT slot 0
    slot = b.slot_of[("effect", "cobblestone")]
    cob_uid = b._effect_uid("cobblestone")
    assert b._slot_uid.get(slot) == cob_uid, "adopt did not re-key to effect uid"
    b._page_out_slot(slot); b.drain_page_events()       # deterministic page-out
    assert cob_uid in ltm.items and not ltm.items[cob_uid].active, (
        "dormant memory not stored under the effect uid")
    known_before = len(ltm.dormant_ids("goal")) + len(_active_slots(b))
    _feed_effect(b, env, "cobblestone", sig_k=45)       # re-encounter, diff sig
    _, pin = b.drain_page_events()
    assert len(pin) >= 1 and pin[0]["mem_id"] == cob_uid, (
        "adopted-then-paged behaviour was NOT recalled -> duplicate re-mint")
    known_after = len(ltm.dormant_ids("goal")) + len(_active_slots(b))
    assert known_after == known_before, (
        f"recall created a duplicate behaviour: {known_before} -> {known_after}")
    print("  13. adopt -> page-out -> recall-by-effect: no duplicate (gate fix)")


def test_competence_widening_safe():
    """Raising max_slots (the 24/48 working-set widening) must KEEP existing
    slots' learned competence, not drop it — load into the first N of the
    wider head. New slots start at the fresh prior."""
    d = _tmp("wsp_widen")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "bc.json")
    b = DiscoveredAchievementGoals(max_slots=3, seed=6, pool=SIG,
                                   match_cosine=0.8)
    env = FakeEnv()
    for k in range(3):
        _feed(b, env, k)
        b.reset()
    for _ in range(20):
        b.competence.update(1, 1.0)      # train slot 1 up
    p1 = b.competence.predict(1)
    assert p1 > 0.7
    b.save_state(path)
    # reload into a WIDER registry (3 -> 6 slots)
    b2 = DiscoveredAchievementGoals(max_slots=6, seed=6, pool=SIG,
                                    match_cosine=0.8)
    assert b2.load_state(path)
    assert abs(b2.competence.predict(1) - p1) < 1e-5, (
        f"widening dropped slot-1 competence: {b2.competence.predict(1)} vs {p1}")
    assert len(b2.slot_names) == 3 and b2.max_slots == 6
    # a brand-new slot in the widened region starts fresh
    assert abs(b2.competence.predict(5) - b2.competence.predict(4)) < 1e-6
    print("  9. competence widening-safe ok (3->6 slots keep learned values)")


def test_distinct_effects_not_merged():
    """Converse-dedup (gate): two DISTINCT grounded effects whose pixel-delta
    signatures are similar (here identical, cosine 1.0 >= match_cosine) must get
    SEPARATE slots — grounded identity beats pixel similarity, no silent merge
    / behaviour loss (was: birch_leaves stole oak_leaves' slot)."""
    b = DiscoveredAchievementGoals(max_slots=8, seed=15, pool=SIG,
                                   match_cosine=0.8)
    env = FakeEnv()
    _feed_effect(b, env, "oak_leaves", sig_k=0); b.reset()
    _feed_effect(b, env, "birch_leaves", sig_k=0); b.reset()   # identical sig
    assert len(_active_slots(b)) == 2, (
        f"distinct effects merged by signature similarity: "
        f"{len(_active_slots(b))} slot(s)")
    assert (("effect", "oak_leaves") in b.slot_of
            and ("effect", "birch_leaves") in b.slot_of)
    assert (b.slot_of[("effect", "oak_leaves")]
            != b.slot_of[("effect", "birch_leaves")])
    print("  16. distinct grounded effects (same signature) stay separate")


def test_serial_path_effect_dedup():
    """Serial-path fix: threading the grounded block as effect_key (the way the
    fixed _run_episode now calls broadcaster.update) dedups 8 viewpoints of one
    behaviour to ONE slot (was 8 slots + 8 skill-mint events on the serial
    path — the gate blocker)."""
    b = DiscoveredAchievementGoals(max_slots=16, seed=14, pool=SIG,
                                   match_cosine=0.95)   # strict: sigs won't cluster
    env = FakeEnv()
    for k in range(8):
        base = np.zeros(SIG, np.float32); ev = base.copy(); ev[k] = 1.0
        env.obs_history = [base, ev]; env.reward_history = [1.0]
        b.update(env, 0, effect_key="birch_leaves")   # serial loop now threads this
        b.reset()
    assert len(_active_slots(b)) == 1, (
        f"serial-path effect dedup failed: {len(_active_slots(b))} slots")
    print("  14. serial-path effect_key threading dedups 8 viewpoints -> 1 slot")


def test_grounded_break_extraction():
    """The _grounded_break_for helper (serial-path block identity): highest-tier
    NEW mine_* break, with per-env high-water marks (log > solid > plant)."""
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    fn = DevelopmentalAI._grounded_break_for

    class Stub:
        _mine_by_env = {}
    s = Stub()
    assert fn(s, {"achievements": {"mine_grass": 1, "mine_oak_log": 1}}, 0) \
        == "oak_log"                                   # log outranks plant
    assert fn(s, {"achievements": {"mine_oak_log": 1}}, 0) is None  # no increase
    assert fn(s, {"achievements": {"mine_oak_log": 2}}, 0) == "oak_log"  # +1
    assert fn(s, {}, 0) is None
    assert fn(s, None, 0) is None
    print("  15. grounded-break extraction (tiering + high-water) ok")


def test_loop_integration():
    """The loop attaches the store, and a real discovered-goals run drives the
    paging plumbing (attach + drain + persist) without error."""
    import copy
    import yaml
    from developmental_ai.core.developmental_loop import DevelopmentalAI

    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 2000})
    cfg["environment"]["max_episode_steps"] = 40
    cfg["skill_bank"]["storage_dir"] = _tmp("wsp_loop_sb")
    cfg["loop"]["verbose"] = 0
    cfg.setdefault("llm", {})["enabled"] = False
    cfg["dream_training"]["enabled"] = False
    cfg["symbolic"] = {"enabled": True, "knowledge_source": "goal"}
    cfg["goals"] = {"mode": "discovered", "max_slots": 3, "spike_threshold": 0.5,
                    "match_cosine": 0.8, "seed": 0}
    # tiny working set + eager paging so overflow is reachable in a short run
    cfg["lifelong"] = {"paging": {"enabled": True, "min_idle_episodes": 0,
                                  "recall_threshold": 0.8}}

    agent = DevelopmentalAI(config=copy.deepcopy(cfg))
    assert getattr(agent, "_goal_ltm", None) is not None, "store not attached"
    assert agent.broadcaster.paging_on, "broadcaster paging not enabled"
    agent.run(total_timesteps=400, verbose=0)
    # the plumbing ran; the store persisted its index
    assert os.path.exists(os.path.join(agent.skill_bank.storage_dir,
                                       "ltm_goals", "ltm_index.json")), \
        "long-term index not persisted"
    st = agent._goal_ltm.stats()
    assert st["active"] <= 3, f"working set exceeded max_slots: {st}"
    agent.close()
    print(f"  8. loop integration ok (store attached + persisted, "
          f"active<=3, total known={st['total']})")


if __name__ == "__main__":
    for fn in (test_off_path_fixed_cap, test_page_out_on_overflow,
               test_recall_on_cue, test_value_weighted_eviction,
               test_target_skips_freed_slot, test_persistence_roundtrip,
               test_competence_slot_paging, test_effect_keying_dedups,
               test_effect_recall_dedups_across_paging,
               test_grounded_adopts_ungrounded_slot,
               test_adopt_then_pageout_recall_no_dup,
               test_distinct_effects_not_merged,
               test_serial_path_effect_dedup, test_grounded_break_extraction,
               test_competence_widening_safe, test_loop_integration):
        print(f"[working-set-paging] {fn.__name__}")
        fn()
    print("[working-set-paging] ALL PASS")
