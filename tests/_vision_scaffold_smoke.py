"""All-stream discovery + magnet loop-wiring smoke (Mac-ok, no Ollama needed).

The curiosity-ranked magnet's OWN semantics (contrastive LP-EMA, retarget,
fade/re-arm, de-confounding) live in `_vision_magnet_smoke.py`. This file keeps
the two scaffold-INDEPENDENT discovery invariants plus the loop-integration
contract:

1. DiscoveredAchievementGoals ingests spikes from SECONDARY streams
   (the July 2026 all-stream discovery fix) and dedups identical events
   across streams by signature.
2. Per-stream competence: N streams -> N Bernoulli samples/episode.
3. Loop wiring: DevelopmentalAI on Crafter with vision.enabled -> the loop
   invokes the magnet with the NEW signature (lp_scalar float + grounding-head
   object_probs) and the shaping it returns reaches the replay buffer, while
   raw episode-reward metrics stay unshaped.
"""
import copy

import numpy as np
import yaml


def test_allstream_discovery():
    from developmental_ai.core.achievement_goals import DiscoveredAchievementGoals

    class FakeEnv:
        def __init__(self):
            self.reward_history = []
            self.obs_history = []

    b = DiscoveredAchievementGoals(max_slots=8, seed=0)
    b.reset()
    envs = [FakeEnv() for _ in range(4)]

    # spike on stream 2 only — must mint a slot (the old code only watched 0)
    base = np.zeros(400, np.float32)
    ev_a = base.copy(); ev_a[:50] = 1.0
    envs[2].obs_history = [base, ev_a]
    envs[2].reward_history = [1.0]
    for e in envs:
        b.update(e)
    assert b.stats["n_slots"] == 1, f"secondary-stream spike missed: {b.stats}"

    # the SAME event signature on stream 3 must dedup into the same slot
    envs[3].obs_history = [base, ev_a]
    envs[3].reward_history = [1.0]
    b.update(envs[3])
    assert b.stats["n_slots"] == 1, "identical event should dedup by signature"

    # a DIFFERENT event on stream 0 mints a second slot
    ev_b = base.copy(); ev_b[200:260] = 1.0
    envs[0].obs_history = [base, ev_b]
    envs[0].reward_history = [1.0]
    b.update(envs[0], 0)
    assert b.stats["n_slots"] == 2, f"distinct event should mint: {b.stats}"
    print("  all-stream discovery ok (secondary mint, cross-stream dedup)")


def test_per_stream_competence():
    """The high-severity fix: with N streams, the self-model must see N
    independent Bernoulli samples per episode (one per stream), NOT a single
    inflated P(any-of-N succeeds). And prerequisite evidence in the unlock
    log must be per-stream, never a union across concurrent episodes."""
    from developmental_ai.core.achievement_goals import DiscoveredAchievementGoals

    class FakeEnv:
        def __init__(self):
            self.reward_history = []
            self.obs_history = []

    b = DiscoveredAchievementGoals(max_slots=8, seed=1)
    b.set_num_streams(4)
    assert b._achieved_now.shape == (4, 8)

    base = np.zeros(400, np.float32)
    ev = base.copy(); ev[:50] = 1.0
    envs = [FakeEnv() for _ in range(4)]
    # register the slot + target it, via a spike on stream 0
    envs[0].obs_history = [base, ev]; envs[0].reward_history = [1.0]
    b.reset()  # picks a target once a slot exists; none yet -> None
    b.update(envs[0], 0)
    slot = 0
    b.target = slot  # force-target the discovered slot for the test

    # Only ONE of four streams achieves the target this episode.
    b._achieved_now[:] = 0.0
    b._achieved_now[2, slot] = 1.0
    attempts_before = b._attempts[slot]
    updates_before = b.competence.n_updates(slot)
    b.reset()
    # 4 attempts recorded (one per stream), not 1
    assert b._attempts[slot] - attempts_before == 4, (
        f"expected 4 per-stream attempts, got "
        f"{b._attempts[slot] - attempts_before}")
    # the self-model likewise saw 4 Bernoulli samples, not 1
    assert b.competence.n_updates(slot) - updates_before == 4, (
        "competence saw wrong number of samples")
    # competence reflects 1/4 success, not 1/1 — must be well under mastered
    comp = b.competence.predict_all()[slot]
    assert comp < 0.6, f"competence inflated (any-of-N leak?): {comp}"

    # per-stream prerequisite evidence: an unlock on stream 3 must log only
    # what stream 3 had, not a union with stream 2's concurrent achievement
    b._achieved_now[:] = 0.0
    b._achieved_now[2, 1] = 1.0  # stream 2 has slot 1 up
    b._record_unlock(0, stream=3)  # stream 3 unlocks slot 0, had nothing
    ev_log = b.unlock_log[-1]
    assert ev_log["stream"] == 3 and ev_log["achieved_before"][1] == 0.0, (
        "unlock log leaked another stream's achievement as prereq evidence")
    print("  per-stream competence ok (N samples/episode, isolated prereqs)")


def test_loop_wiring():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    import developmental_ai.llm.vision_scaffold as vsmod

    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 2000})
    cfg["environment"]["max_episode_steps"] = 40
    cfg["skill_bank"]["storage_dir"] = "/tmp/vision_smoke_sb"
    cfg["loop"]["verbose"] = 0
    cfg.setdefault("llm", {})["enabled"] = False
    cfg["llm"]["vision"] = {"enabled": True, "weight": 0.5, "min_labels": 1}

    # Spy on step_shaping to pin the INTEGRATION CONTRACT (not the magnet's own
    # curiosity dynamics — those are in _vision_magnet_smoke). The loop must:
    #   * call with lp_scalar as a float (this step's learning progress), and
    #   * thread object_probs/reliability/label_counts kwargs through.
    # A constant bonus is returned so we can prove the shaping reaches the
    # replay buffer while raw episode metrics stay unshaped.
    BONUS = 0.25
    _MISSING = object()   # sentinel: distinguishes "passed None" from "absent"
    calls = {"n": 0, "lp_float": False, "kw_threaded": False}
    orig = vsmod.VisionScaffold.step_shaping

    def spy(self, action, timestep, lp_scalar, object_probs=_MISSING,
            reliability=_MISSING, label_counts=_MISSING):
        calls["n"] += 1
        if isinstance(lp_scalar, float):
            calls["lp_float"] = True
        # object_probs may be None before grounding has evidence — that is a
        # valid inert call. The contract is that the loop PASSES the kwargs at
        # all (sentinel proves they were threaded, not defaulted).
        if (object_probs is not _MISSING and reliability is not _MISSING
                and label_counts is not _MISSING):
            calls["kw_threaded"] = True
        return BONUS

    vsmod.VisionScaffold.step_shaping = spy
    try:
        agent = DevelopmentalAI(config=copy.deepcopy(cfg))
        assert agent.vision_scaffold is not None, "magnet not constructed"
        agent.run(total_timesteps=120, verbose=0)
        assert calls["n"] >= 2, f"loop never invoked the magnet: {calls}"
        assert calls["lp_float"], "loop did not thread the LP scalar as a float"
        assert calls["kw_threaded"], "loop did not thread grounding kwargs"
        # shaping reached the replay buffer (stored reward = shaped extrinsic)
        rb = agent.replay_buffer
        rews = rb.rewards[:rb.size].flatten()
        assert (rews > BONUS - 1e-6).any(), (
            "no shaped reward reached the buffer")
        # raw episode-reward metric is NOT inflated by the injected bonus
        # (Crafter random policy over 120 steps earns only a few raw points;
        # a leak of +0.25/step would push it well past any sane bound)
        ep_r = (np.mean(agent.training_metrics["episode_reward"])
                if agent.training_metrics["episode_reward"] else 0.0)
        assert ep_r < 15.0, f"magnet bonus leaked into raw metrics: {ep_r}"
        agent.close()
    finally:
        vsmod.VisionScaffold.step_shaping = orig
    print(f"  loop wiring ok (magnet invoked {calls['n']}x with LP+probs, "
          f"shaped reward in buffer, raw metrics clean)")


if __name__ == "__main__":
    for fn in (test_allstream_discovery, test_per_stream_competence,
               test_loop_wiring):
        print(f"[vision-smoke] {fn.__name__}")
        fn()
    print("[vision-smoke] ALL PASS")
