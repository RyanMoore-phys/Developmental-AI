"""Skills-as-options smoke (Mac-ok; unit fixtures + the REAL loop on Crafter).

Contracts (subset of the design spec's S1-S13 that decides ship/no-ship):
  1. S4  SMDP GAE: hand-built taus fixture matches the closed form; taus
         all-ones is BITWISE-equal to the shipped recursion.
  2. S2  Masked head: 2000 samples never emit an unbound slot; renormalized
         log_prob matches a manual masked softmax.
  3. S7  Head adaptation: 10-head passes through, wider head row-truncates,
         narrow head refused; canonical ctx reconstructed from slot id when
         context_embedding is latent-shaped.
  4. S5  Termination: spike -> tau=2, done -> episode_end, horizon -> k;
         primary close emits exactly one SMDP decision with accumulated
         discounted reward.
  5. S12 Distill slice: log_softmax over the primitive slice backprops ZERO
         gradient into slot logit rows.
  6. S13 Two ledgers: record_invocation updates Skill fields, leaves goal
         competence untouched; mining adds a prerequisite edge at count 3.
  7. S1/S3/S11 integration on the REAL parallel loop (Crafter, 2 envs,
         2 stub skills): every env action < P; sum(taus) for the primary ==
         episode length; options fired (tau>1 stored) with firing telemetry;
         flag-off run keeps head == P and env-step trigger parity.
"""
import json
import os
import shutil

import numpy as np
import torch
import yaml

SB = "/tmp/options_smoke_sb"
P_CRAFTER = 17  # crafter action space


def _mk_policy(P, K, obs_dim=12, knowledge_dim=0):
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    return StandaloneActorCritic(
        obs_dim=obs_dim, action_dim=P + K, hidden_dim=32,
        continuous=False, knowledge_dim=knowledge_dim,
        device=torch.device("cpu"))


def test_smdp_gae():
    pol = _mk_policy(4, 0)
    pol.gamma, pol.gae_lambda = 0.9, 0.8
    r = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    v = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    d = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    taus = np.array([1.0, 3.0, 1.0], dtype=np.float32)
    adv = pol._compute_gae(r, v, d, taus=taus)
    # closed form, back to front
    a2 = (0.5 - 0.1)
    d1 = 2.0 + (0.9 ** 3) * 0.1 - 0.2
    a1 = d1 + ((0.9 * 0.8) ** 3) * a2
    d0 = 1.0 + 0.9 * 0.2 - 0.3
    a0 = d0 + (0.9 * 0.8) * a1
    assert np.allclose(adv, [a0, a1, a2], atol=1e-6), (adv, [a0, a1, a2])
    # all-ones taus == shipped recursion, bitwise
    legacy = np.zeros_like(r)
    last = 0.0
    for t in reversed(range(3)):
        nv = 0.0 if t == 2 else v[t + 1]
        delta = r[t] + 0.9 * nv * (1 - d[t]) - v[t]
        legacy[t] = delta + 0.9 * 0.8 * (1 - d[t]) * last
        last = legacy[t]
    assert np.array_equal(pol._compute_gae(r, v, d), legacy)
    assert np.array_equal(pol._compute_gae(r, v, d,
                                           taus=np.ones(3, np.float32)),
                          legacy)
    print("  1. SMDP GAE ok (fixture + all-ones bitwise parity)")


def test_masked_head():
    torch.manual_seed(0)
    pol = _mk_policy(4, 3)
    mask = np.array([1, 1, 1, 1, 0, 1, 0], dtype=bool)  # slot 1 bound only
    obs = np.random.randn(12).astype(np.float32)
    seen = set()
    for _ in range(2000):
        a, info = pol.select_action(obs, action_mask=mask)
        seen.add(a)
        assert mask[a], f"emitted masked action {a}"
    assert 5 in seen, "bound slot never sampled in 2000 draws"
    # log_prob matches manual masked softmax
    a, info = pol.select_action(obs, action_mask=mask)
    with torch.no_grad():
        logits = pol.actor.action_head(pol.actor.shared(
            torch.from_numpy(obs).reshape(1, -1)))
        logits = logits.masked_fill(
            ~torch.from_numpy(mask).reshape(1, -1), -1e9)
        lp = torch.log_softmax(logits, dim=-1)[0, a].item()
    assert abs(lp - info["log_prob"]) < 1e-4, (lp, info["log_prob"])
    print("  2. masked head ok (support + renormalized log_prob)")


def test_head_adaptation():
    from developmental_ai.policy.options import adapt_actor_sd, SlotRefused
    from developmental_ai.policy.actor_critic import ActorNetwork
    P = 10
    # 10-head: pass-through
    a10 = ActorNetwork(20, P, 16).state_dict()
    out, in_dim, hidden = adapt_actor_sd({"actor": a10}, P)
    assert out["action_head.weight"].shape[0] == P and in_dim == 20
    assert hidden == 16, "hidden dim not sniffed"
    # 18-head WITHOUT metadata: REFUSED (2026-07-26 contract change).
    # The old behaviour — silently slicing W[:P] — is exactly what would have
    # bound option-slot logits as the new crafting buttons after the action
    # space widened. A meta-width head is only sliceable when the skill's OWN
    # primitive count is recorded.
    a18 = ActorNetwork(20, P + 8, 16)
    sd18 = a18.state_dict()
    try:
        adapt_actor_sd({"actor": sd18}, P)
        raise AssertionError("meta-width head bound without primitive_dim")
    except SlotRefused:
        pass
    # ...and WITH metadata it truncates to the skill's OWN width
    out2, _, h2 = adapt_actor_sd({"actor": sd18}, P, primitive_dim=P)
    assert out2["action_head.weight"].shape[0] == P
    assert torch.equal(out2["action_head.weight"],
                       sd18["action_head.weight"][:P])
    # narrow head refused
    a6 = ActorNetwork(20, 6, 16).state_dict()
    try:
        adapt_actor_sd({"actor": a6}, P)
        raise AssertionError("narrow head not refused")
    except SlotRefused:
        pass
    print("  3. head adaptation ok (pass/truncate/refuse)")


def test_ctx_reconstruction():
    """A conditioner-bearing skill whose context_embedding is the 1536-d
    unlock latent still gets its CANONICAL 34-d goal vector, rebuilt from
    the slot id."""
    from developmental_ai.policy.options import SkillOptionBank
    from developmental_ai.policy.actor_critic import (
        StandaloneActorCritic)
    from developmental_ai.skill_bank.skill_bank import SkillBank
    shutil.rmtree(SB, ignore_errors=True)
    bank = SkillBank(storage_dir=SB)
    obs_dim, kdim, P = 12, 34, 4
    pol = StandaloneActorCritic(obs_dim=obs_dim, action_dim=P,
                                hidden_dim=16, continuous=False,
                                knowledge_dim=kdim,
                                device=torch.device("cpu"))
    bank.save_skill("ach_03_discovered_3", "chop", pol.get_state_dict(),
                    success_rate=0.6, total_episodes=5, dedup=True,
                    context_embedding=np.random.randn(1536),  # latent-shaped
                    obs_dim=obs_dim, action_dim=P)
    ob = SkillOptionBank(bank, obs_dim, P, 2, torch.device("cpu"))
    ob.refresh_slots()
    b = ob.slots[0]
    assert b is not None and b["kdim"] == kdim
    assert b["ctx"] is not None and b["ctx"].shape == (kdim,)
    assert b["ctx"][3] == 1.0 and b["ctx"][32] == 1.0, b["ctx"]
    assert b["ctx"].sum() == 2.0
    a = ob.skill_action(0, np.zeros(obs_dim, np.float32))
    assert 0 <= a < P
    print("  4. canonical ctx reconstructed from slot id ok")


def test_termination_and_accumulation():
    from developmental_ai.policy.options import (
        OptionExecutor, SkillOptionBank)
    from developmental_ai.skill_bank.skill_bank import SkillBank
    bank = SkillBank(storage_dir=SB)
    ob = SkillOptionBank(bank, 12, 4, 2, torch.device("cpu"))
    ex = OptionExecutor(ob, num_envs=2,
                        cfg={"max_option_steps": 5, "spike_threshold": 0.9},
                        gamma=0.5)
    # force env0 into a fake option
    ex.runtimes[0].open(0, "sk_a", 100, {
        "obs": np.zeros(12), "meta_action": 4, "log_prob": -1.0,
        "value": 0.2, "kv": None, "mask": np.ones(6, bool)})
    ex._start_event(0, ex.runtimes[0], 100)
    # two accumulation steps then a spike
    assert ex.observe_primary(0.0, False, 1.0, 101) is None
    closed = ex.observe_primary(5.0, False, 2.0, 102)   # raw 5.0 >= 0.9
    assert closed is not None and closed["outcome"] == "spike"
    assert closed["tau"] == 2
    assert abs(closed["reward"] - (1.0 + 0.5 * 2.0)) < 1e-6, closed["reward"]
    # done mid-option -> episode_end
    ex.runtimes[0].open(1, "sk_b", 110, {
        "obs": np.zeros(12), "meta_action": 5, "log_prob": -1.0,
        "value": 0.2, "kv": None, "mask": np.ones(6, bool)})
    ex._start_event(0, ex.runtimes[0], 110)
    closed2 = ex.observe_primary(0.0, True, 0.3, 111)
    assert closed2 is not None and closed2["outcome"] == "episode_end"
    # horizon
    ex.runtimes[0].open(0, "sk_a", 120, {
        "obs": np.zeros(12), "meta_action": 4, "log_prob": -1.0,
        "value": 0.2, "kv": None, "mask": np.ones(6, bool)})
    ex._start_event(0, ex.runtimes[0], 120)
    closed3 = None
    for i in range(5):
        closed3 = ex.observe_primary(0.0, False, 0.1, 121 + i)
    assert closed3 is not None and closed3["outcome"] == "horizon"
    assert closed3["tau"] == 5
    snap = ex.snapshot()
    json.dumps(snap)   # S9: JSON round-trip
    assert snap["active"]["0"] is None
    assert all(ev["t_end"] is not None for ev in snap["recent"])
    print("  5. termination + accumulation + snapshot ok")


def test_distill_slice_zero_slot_grad():
    P, K = 6, 3
    pol = _mk_policy(P, K)
    obs = torch.randn(8, 12)
    feats = pol.actor.shared(obs)
    logits = pol.actor.action_head(feats)
    student_logp = torch.log_softmax(logits[..., :P], dim=-1)
    teacher = torch.softmax(torch.randn(8, P), dim=-1)
    loss = -(teacher * student_logp).sum(-1).mean()
    loss.backward()
    g = pol.actor.action_head.weight.grad
    assert g is not None
    assert torch.allclose(g[P:], torch.zeros_like(g[P:])), (
        "slot logit rows received distill gradient")
    assert g[:P].abs().sum() > 0, "primitive rows got no gradient"
    print("  6. distill slice ok (zero slot-row gradient)")


def test_two_ledgers_and_mining():
    from developmental_ai.skill_bank.skill_bank import SkillBank
    from developmental_ai.core.achievement_goals import (
        DiscoveredAchievementGoals)
    bank = SkillBank(storage_dir=SB)
    sk = bank.save_skill("ach_00_discovered_0", "s0",
                         {"w": torch.zeros(1)}, success_rate=0.5,
                         total_episodes=5, dedup=True)
    g = DiscoveredAchievementGoals(max_slots=4, seed=0)
    comp_before = g.competence.predict_all().copy()
    bank.record_invocation("ach_00_discovered_0", "spike", 1.5)
    bank.record_invocation("ach_00_discovered_0", "horizon", 0.2)
    sk = bank.skills["ach_00_discovered_0"]
    assert sk.invocations == 2 and sk.option_spikes == 1
    assert sk.option_spike_ema > 0
    assert np.array_equal(comp_before, g.competence.predict_all()), (
        "option outcomes leaked into goal competence — two-ledger violation")
    # mining: 3 co-events -> prerequisite edge
    bank.save_skill("ach_01_discovered_1", "s1", {"w": torch.zeros(1)},
                    success_rate=0.5, total_episodes=5, dedup=True)
    co = [{"skill_id": "ach_00_discovered_0", "unlocked_slot": 1,
           "stream": 0, "t": i} for i in range(3)]
    added = bank.mine_cooccurrences(
        co, {1: "ach_01_discovered_1"}, min_count=3)
    assert added == 1
    assert ("ach_00_discovered_0"
            in bank.skills["ach_01_discovered_1"].prerequisites)
    bank.flush_if_dirty()
    from developmental_ai.skill_bank.skill_bank import SkillBank as SB2
    assert ("ach_00_discovered_0"
            in SB2(storage_dir=SB).skills[
                "ach_01_discovered_1"].prerequisites), "edge not persisted"
    print("  7. two ledgers + co-occurrence mining ok")


def _crafter_cfg(enable_options):
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 64, "encoder_hidden": 64, "batch_size": 4,
        "sequence_length": 8, "train_iters": 1, "buffer_capacity": 4000})
    cfg["environment"]["max_episode_steps"] = 40
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["skill_bank"]["storage_dir"] = SB + "_loop"
    cfg["loop"]["verbose"] = 0
    cfg["llm"]["enabled"] = False
    cfg["llm"]["vision"] = {"enabled": False}
    if enable_options:
        cfg["skills_as_options"] = {
            "enabled": True, "k_slots": 2, "max_option_steps": 6,
            "terminate_on_spike": True, "scouts_use_options": True}
    return cfg


def test_loop_integration():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    from developmental_ai.skill_bank.skill_bank import SkillBank
    from developmental_ai.policy.actor_critic import StandaloneActorCritic

    shutil.rmtree(SB + "_loop", ignore_errors=True)
    # seed the bank with two REAL stub skills at the env's obs/action dims
    cfg = _crafter_cfg(True)
    probe = DevelopmentalAI(_crafter_cfg(False))
    P, obs_dim = probe.action_dim, probe.obs_dim
    assert probe.meta_action_dim == P, "flag-off widened the head"
    assert probe.policy.rollout_env_steps() == len(
        probe.policy.rollout_obs), "flag-off env-step parity broken"
    del probe
    bank = SkillBank(storage_dir=SB + "_loop")
    for i in range(2):
        stub = StandaloneActorCritic(
            obs_dim=obs_dim, action_dim=P, hidden_dim=32,
            continuous=False, device=torch.device("cpu"))
        bank.save_skill(f"ach_0{i}_discovered_{i}", f"stub_{i}",
                        stub.get_state_dict(), success_rate=0.6,
                        total_episodes=5, dedup=True,
                        obs_dim=obs_dim, action_dim=P)

    ai = DevelopmentalAI(cfg)
    assert ai.option_executor is not None, "options not constructed"
    assert ai.meta_action_dim == P + 2
    ai._run_episode_parallel(use_dream_actor=False)

    ex = ai.option_executor
    bound = [b for b in ex.bank.slots if b is not None]
    assert len(bound) == 2, f"slots not bound: {ex.bank.slots}"
    taus = list(ai.policy.rollout_taus)
    assert taus, "nothing stored"
    assert sum(taus) == 40, f"sum(taus)={sum(taus)} != episode length 40"
    # ---- ASSERT THE CONTRACT, NOT A LUCKY DRAW (fixed 2026-09-01) ------
    # This used to require `any(t > 1)` — a PRIMARY-stream option inside 40
    # steps — and called its absence "astronomically unlikely". It is not:
    # ~33 primary decisions at 2/19 slot mass gives P(none) ~ 3%, so the
    # assertion was a coin flip the seed happened to win. It duly broke when
    # an unrelated change (dropping an unused module) shifted the torch RNG
    # stream, and reported "invocation path dead" for a path that was fine.
    # MEASURED across seeds with the same code:
    #     seed 0: 4 picks, 0 primary option rows   <- the old assertion fails
    #     seed 1: 8 picks, 4 primary option rows
    #     seed 2: 5 picks, 3 primary option rows
    #     seed 3: 5 picks, 2 primary option rows
    # The real contract is "the invocation path is alive", which
    # `option_picks` tests directly and seed-independently. The tau/meta-id
    # check still runs, but only over the rows that exist.
    assert ai.option_executor.option_picks > 0, (
        "no option invoked on ANY stream in 40 steps with 2/19 slot mass — "
        "the invocation path is dead (this is the seed-independent claim)")
    metas = [a for a, t in zip(ai.policy.rollout_actions, taus) if t > 1]
    assert all(a >= P for a in metas), "option decision stored primitive id"
    if not metas:
        print("     (no PRIMARY-stream option this seed — expected ~3% of "
              "the time; option_picks is what proves the path)")
    snap = ex.snapshot()
    assert snap["recent"], "no firing telemetry"
    json.dumps(snap)
    assert ai.policy.rollout_env_steps() == 40
    # a second episode must run clean too (slot freeze, clears, re-mint path)
    ai._run_episode_parallel(use_dream_actor=False)
    print(f"  8. loop integration ok ({len(metas)} invocations, "
          f"{len(taus)} decisions over 80 env steps, telemetry live)")
def test_full_construction_with_brain_viewer():
    """REGRESSION (pod deploy 2026-07-19): brain_viewer.enabled constructs
    the emitter, which references skill_bank + broadcaster + option_executor.
    No smoke exercised that combination -> an init-ordering AttributeError
    ('DevelopmentalAI has no attribute skill_bank') only surfaced on the pod.
    This constructs the exact failing combination on Crafter."""
    import shutil as _sh
    import yaml as _yaml
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    _sh.rmtree("/tmp/opt_brain_ctor", ignore_errors=True)
    cfg = _yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 64, "encoder_hidden": 64, "batch_size": 4,
        "sequence_length": 8, "train_iters": 1, "buffer_capacity": 4000})
    cfg["environment"]["max_episode_steps"] = 40
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["skill_bank"]["storage_dir"] = "/tmp/opt_brain_ctor"
    cfg["loop"]["verbose"] = 0
    cfg["llm"]["enabled"] = False
    cfg["llm"]["vision"] = {"enabled": False}
    cfg["brain_viewer"] = {"enabled": True, "out_dir": "/tmp/opt_brain_out",
                           "interval": 50, "ghost_frontier": True}
    cfg["skills_as_options"] = {
        "enabled": True, "k_slots": 2, "max_option_steps": 6,
        "gate_by_preconditions": True, "scouts_use_options": True}
    cfg["symbolic_grounding"] = {"enabled": True, "interval": 1,
                                 "min_labels": 2, "chop_actions": [0, 1]}
    ai = DevelopmentalAI(cfg)               # would AttributeError pre-fix
    assert ai.brain_emitter is not None
    assert ai.option_executor is not None
    assert ai.option_executor.on_start is not None, "firing hook not wired"
    import json as _json
    path = ai.brain_emitter.emit(0, 0)
    _json.load(open(path))                  # valid JSON, no throw
    print("  9. full construction with brain_viewer+options+grounding ok")
def test_competence_gate_warmup():
    """Condition-based warmup: options for a skill are offered only once its
    competence >= floor. Cold-start (all weak) offers NO options; a competent
    skill IS offered; the gate composes with preconditions; floor=0 disables."""
    from developmental_ai.policy.options import (
        OptionExecutor, SkillOptionBank)
    from developmental_ai.skill_bank.skill_bank import SkillBank
    import torch
    shutil.rmtree("/tmp/opt_comp_sb", ignore_errors=True)
    bank = SkillBank(storage_dir="/tmp/opt_comp_sb")
    for i in range(2):
        stub = _mk_policy(10, 0, obs_dim=12).get_state_dict()
        bank.save_skill(f"ach_0{i}_discovered_{i}", f"break_thing_{i}", stub,
                        success_rate=0.5, total_episodes=5, dedup=True,
                        obs_dim=12, action_dim=10)
    ob = SkillOptionBank(bank, 12, 10, 4, torch.device("cpu"))
    ob.refresh_slots()
    # all skills weak -> with floor 0.6, NO slot offered
    comp_weak = {"ach_00_discovered_0": 0.4, "ach_01_discovered_1": 0.45}
    m = ob.mask(competence=comp_weak, competence_floor=0.6)
    assert m[:10].all(), "primitives gated by warmup"
    assert m[10:].sum() == 0, "weak skills offered despite competence floor"
    # one skill matures -> only that one offered
    comp_mix = {"ach_00_discovered_0": 0.72, "ach_01_discovered_1": 0.45}
    m2 = ob.mask(competence=comp_mix, competence_floor=0.6)
    off = [i for i in range(4) if m2[10 + i]]
    assert len(off) == 1, f"expected 1 offered, got {off}"
    # floor 0 disables the gate -> both offered again
    m3 = ob.mask(competence=comp_weak, competence_floor=0.0)
    assert m3[10:].sum() == 2, "floor=0 should disable the competence gate"
    print("  10. competence-gate warmup ok (cold-start silent, self-enables)")


def test_grounded_naming():
    """Grounded skill naming: name from the block broken at unlock, not the
    VLM. Verifies the helper's ground-truth priority + fallbacks directly."""
    import shutil as _sh
    import yaml as _yaml
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    _sh.rmtree("/tmp/name_sb", ignore_errors=True)
    cfg = _yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 64, "encoder_hidden": 64, "batch_size": 4,
        "sequence_length": 8, "train_iters": 1, "buffer_capacity": 4000})
    cfg["environment"]["max_episode_steps"] = 40
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["skill_bank"]["storage_dir"] = "/tmp/name_sb"
    cfg["loop"]["verbose"] = 0
    cfg["llm"]["enabled"] = False
    cfg["llm"]["vision"] = {"enabled": False}
    ai = DevelopmentalAI(cfg)
    # ground-truth block present -> break_<block>
    ai._unlock_block[3] = "oak_log"
    n, d = ai._grounded_skill_name(3, {})
    assert n == "break_oak_log", n
    assert "oak_log" in d and "hallucin" not in d.lower()
    # no block, but a grounded predicate true -> act_where_<pred>
    n2, _ = ai._grounded_skill_name(4, {"tree_visible": True,
                                        "water_visible": False})
    # ...carrying an `_sNN` SLOT SUFFIX: two different slots can both ground
    # out on the same predicate, and without the suffix they collide on one
    # skill id — the second mint silently overwrites the first.
    assert n2 == "act_where_tree_visible_s04", n2
    # nothing -> neutral slot name (never a fabricated action)
    n3, _ = ai._grounded_skill_name(5, {})
    assert n3 == "skill_slot_05", n3
    # the hallucinated names from run 7 must be impossible now
    for bad in ("fish_blue_water", "collect_heart_items", "craft_wooden_planks"):
        assert bad not in (n, n2, n3)
    print("  11. grounded naming ok (block > predicate > neutral, no VLM)")


if __name__ == "__main__":
    for fn in (test_smdp_gae, test_masked_head, test_head_adaptation,
               test_ctx_reconstruction, test_termination_and_accumulation,
               test_distill_slice_zero_slot_grad,
               test_two_ledgers_and_mining, test_loop_integration,
               test_full_construction_with_brain_viewer,
               test_competence_gate_warmup, test_grounded_naming):
        print(f"[options-smoke] {fn.__name__}")
        fn()
    print("[options-smoke] ALL PASS")
