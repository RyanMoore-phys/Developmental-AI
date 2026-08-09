"""Smoke for the achievement-goal channel (rich-env program steps 1-3).

Unit level (fake env, no deps):
  1. Given-mode: slot registration, unlock events, once-per-slot minting,
     pre-existing-count baseline handling, feature layout.
  2. Discovered-mode: spike detection, signature clustering (distinct kinds
     -> distinct slots; repeats -> same slot), sub-threshold ignored.
  3. IMGEP targeting: mastered slots are mostly avoided; frontier preferred.
     Lesion vectors (constant/noise/scrambled) valid.
  4. Prerequisite DAG: B-always-after-A shows up in estimated_dag.

Integration level (real Crafter on this machine):
  5. given-mode: goal channel wired end-to-end (policy.knowledge_dim == DIM),
     slots register, achievement SKILLS mint into the bank.
  6. discovered-mode: slots discovered from reward spikes alone; purity vs
     the oracle names REPORTED (grading only — never used in decisions).
"""
import copy
import types

import numpy as np
import yaml

from developmental_ai.core.achievement_goals import (
    DiscoveredAchievementGoals, GivenAchievementGoals)

SB_DIR = "/tmp/goal_smoke_sb"


def _fake_env(info=None, obs_pair=None, reward=0.0):
    e = types.SimpleNamespace()
    e.last_info = info or {}
    e.obs_history = list(obs_pair) if obs_pair else []
    e.reward_history = [reward] if obs_pair else []
    return e


def test_given_mode():
    g = GivenAchievementGoals(max_slots=8, seed=0)
    g.reset()
    # first sight with pre-existing count: baseline, NOT an event
    g.update(_fake_env(info={"achievements": {"collect_wood": 1, "eat_cow": 0}}))
    assert g.stats["n_slots"] == 1 and g.stats["total_unlocks"] == 0
    # count increase -> unlock event
    g.update(_fake_env(info={"achievements": {"collect_wood": 2, "eat_cow": 0}}))
    assert g.stats["total_unlocks"] == 1
    # new achievement unlocks
    g.update(_fake_env(info={"achievements": {"collect_wood": 2, "eat_cow": 1}}))
    assert g.stats["n_slots"] == 2 and g.stats["total_unlocks"] == 2
    mints = g.pop_skill_mint_events()
    assert len(mints) == 2 and g.pop_skill_mint_events() == []
    # repeat unlock next episode: NOT re-minted
    g.reset()
    g.update(_fake_env(info={"achievements": {"collect_wood": 3, "eat_cow": 1}}))
    assert g.pop_skill_mint_events() == []
    f = g.feature()
    assert f.shape == (g.DIM,) and f[g.max_slots + g.slot_of["collect_wood"]] == 1.0
    print("  given ok: baseline, events, once-per-slot minting, feature")


def test_discovered_mode():
    d = DiscoveredAchievementGoals(max_slots=8, spike_threshold=0.9, seed=0)
    d.reset()
    base = np.zeros(400, np.float32)
    kindA = base.copy(); kindA[:40] = 1.0        # distinct delta pattern A
    kindB = base.copy(); kindB[200:240] = 1.0    # distinct delta pattern B
    d.update(_fake_env(obs_pair=(base, kindA), reward=1.0))
    d.update(_fake_env(obs_pair=(base, kindB), reward=1.0))
    assert d.stats["n_slots"] == 2, d.stats
    d.reset()
    d.update(_fake_env(obs_pair=(base, kindA * 1.05), reward=1.0))  # same kind
    assert d.stats["n_slots"] == 2 and d.stats["total_unlocks"] == 3
    d.update(_fake_env(obs_pair=(base, kindB), reward=0.1))  # sub-threshold
    assert d.stats["total_unlocks"] == 3
    print("  discovered ok: clustering, repeat-matching, threshold")


def test_imgep_targeting_and_lesions():
    g = GivenAchievementGoals(max_slots=8, epsilon=0.0, seed=1)
    g.reset()
    g.update(_fake_env(info={"achievements": {"a": 1, "b": 1}}))
    g._record_unlock(g.slot_of["a"])  # ensure both registered w/ events
    # teach the self-model: slot a mastered, slot b failing
    for _ in range(30):
        g.competence.update(g.slot_of["a"], 1.0)
        g.competence.update(g.slot_of["b"], 0.0)
    picks = []
    for _ in range(30):
        g.reset()
        picks.append(g.target)
    frac_b = np.mean([p == g.slot_of["b"] for p in picks])
    assert frac_b > 0.8, f"frontier slot not preferred ({frac_b:.2f})"
    rng = np.random.RandomState(0)
    for v in (g.constant_feature(), g.noise_feature(rng), g.scrambled_feature(rng)):
        assert v.shape == (g.DIM,) and np.isfinite(v).all()
    assert g.scrambled_feature(rng)[:g.max_slots].argmax() != g.target
    print(f"  imgep ok: frontier picked {frac_b:.0%}; lesions valid")


def test_dag_prior_selection():
    from developmental_ai.core.achievement_goals import GivenAchievementGoals
    g = GivenAchievementGoals(max_slots=8, epsilon=0.0, seed=2,
                              dag_prior={"deep": ["shallow"]})
    g.reset()
    g.update(_fake_env(info={"achievements": {"shallow": 1, "deep": 1}}))
    # both slots equally unknown -> the prior should damp 'deep' (prereq
    # 'shallow' unmastered) and target 'shallow' predominantly
    picks = []
    for _ in range(20):
        g.reset()
        picks.append(g.target)
    frac_shallow = np.mean([p == g.slot_of["shallow"] for p in picks])
    assert frac_shallow > 0.8, f"prior not damping deep goal ({frac_shallow})"
    # master the prerequisite -> deep becomes preferred (frontier + prereq met)
    for _ in range(40):
        g.competence.update(g.slot_of["shallow"], 1.0)
    picks2 = []
    for _ in range(20):
        g.reset()
        picks2.append(g.target)
    frac_deep = np.mean([p == g.slot_of["deep"] for p in picks2])
    assert frac_deep > 0.8, f"deep not unlocked by mastered prereq ({frac_deep})"
    print(f"  dag-prior ok: shallow first ({frac_shallow:.0%}), "
          f"deep after mastery ({frac_deep:.0%})")


def test_dag_estimation():
    g = GivenAchievementGoals(max_slots=8, seed=0)
    for _ in range(6):
        g.reset()
        g.update(_fake_env(info={"achievements": {"A": 1, "B": 0}}))
        g._counts = {0: {"A": 0, "B": 0}}  # per-stream: force stream-0 baselines
                                            # to 0 so the next update sees increases
        g.update(_fake_env(info={"achievements": {"A": 1, "B": 1}}))
    dag = g.estimated_dag()
    sB, sA = g.slot_of["B"], g.slot_of["A"]
    assert dag[sB][sA] > 0.9, f"A-precedes-B not captured: {dag[sB][sA]}"
    print(f"  dag ok: P(A already achieved | B unlocks) = {dag[sB][sA]:.2f}")


def _crafter_cfg(mode):
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 2000})
    cfg["environment"]["max_episode_steps"] = 250
    cfg["dream_training"]["enabled"] = False
    cfg["symbolic"] = {"enabled": True, "knowledge_source": "goal"}
    cfg["goals"] = {"mode": mode, "max_slots": 24, "mint_skills": True}
    cfg["skill_bank"]["storage_dir"] = f"{SB_DIR}_{mode}"
    cfg["loop"]["verbose"] = 0
    return cfg


def test_crafter_given_end_to_end():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.core.achievement_goals import GivenAchievementGoals
    agent = DevelopmentalAI(config=copy.deepcopy(_crafter_cfg("given")))
    assert isinstance(agent.broadcaster, GivenAchievementGoals)
    assert agent.policy.knowledge_dim == agent.broadcaster.DIM, (
        "goal channel must reach the policy input")
    agent.run(total_timesteps=2500, verbose=0)
    st = agent.broadcaster.stats
    n_skills = agent.skill_bank.get_stats()["total_skills"]
    assert st["n_slots"] >= 1, f"no achievements registered: {st}"
    assert n_skills >= 1, "no achievement skills minted"
    assert st["target"] is not None, "no goal targeted"
    print(f"  crafter given ok: {st['n_slots']} slots, "
          f"{st['total_unlocks']} unlocks, {n_skills} skills minted, "
          f"targeting slot {st['target']}")


def test_crafter_discovered_end_to_end():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.core.achievement_goals import (
        DiscoveredAchievementGoals)
    agent = DevelopmentalAI(config=copy.deepcopy(_crafter_cfg("discovered")))
    assert isinstance(agent.broadcaster, DiscoveredAchievementGoals)

    # grading tap (oracle used ONLY here, in the smoke, for purity stats)
    truth = []
    disc = agent.broadcaster
    orig = disc._record_unlock

    def spy(slot, stream=0):
        ach = agent.env.last_info.get("achievements") or {}
        truth.append((slot, tuple(sorted(k for k, v in ach.items() if v))))
        orig(slot, stream)

    disc._record_unlock = spy
    agent.run(total_timesteps=2500, verbose=0)
    st = disc.stats
    print(f"  crafter discovered: {st['n_slots']} slots, "
          f"{st['total_unlocks']} unlocks, graded events={len(truth)}")
    assert st["n_slots"] >= 1, "nothing discovered from reward spikes"


if __name__ == "__main__":
    for fn in (test_given_mode, test_discovered_mode,
               test_imgep_targeting_and_lesions, test_dag_prior_selection,
               test_dag_estimation,
               test_crafter_given_end_to_end,
               test_crafter_discovered_end_to_end):
        print(f"[goal-smoke] {fn.__name__}")
        fn()
    print("[goal-smoke] ALL PASS")
