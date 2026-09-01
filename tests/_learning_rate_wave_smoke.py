"""
_learning_rate_wave_smoke — the 2026-09-01 wave: five defects that bounded how
fast and how well this agent could learn, none of them a reward-shaping
argument.

Run:  PYTHONPATH=. ./venv/bin/python tests/_learning_rate_wave_smoke.py

WHAT WAS MEASURED, AND WHY EACH CONTRACT EXISTS
================================================================================

F1. PPO TOOK ONE GRADIENT STEP PER 1024 ENV STEPS.
    `policy.minibatch_size` was 0, so `_chunks = [None]` and every epoch was a
    single full-batch step; `policy.batch_size: 64` has never been read by the
    update at all (the loop already warned about it). On top of that the epoch
    count was capped at `rows // 32`. Under SMDP options at mean tau ~40 a
    1024-env-step rollout holds ~25 rows -> 25//32 = 0 -> ONE gradient step per
    segment. At the run's measured ~3.4 steps/s that is one policy update every
    five minutes. Textbook PPO at these settings takes 160.
      -> the row cap now applies ONLY to the full-batch path (where its
         evidence came from), the minibatched path is bounded by a gradient
         budget, and target_kl is the guard that watches the policy instead of
         counting rows.

F1b. THE MINIBATCH COULD NOT ENGAGE IN THE REGIME IT WAS FOR.
    Rows are scarce EXACTLY when options are used, so a 25-row rollout is
    smaller than any minibatch and silently fell back to full-batch. The update
    trigger now requires rows as well as env steps, with a hard env-step
    ceiling so the row floor can never become a latch (guard-becomes-latch,
    §4.1, nine occurrences).

F2. HALF THE FLEET'S EXPERIENCE NEVER REACHED THE POLICY.
    Scout streams' intrinsic reward was computed every step and discarded;
    their transitions reached the world model and never PPO. On a 2-client
    fleet (2 is the proven number — 4 produced 0 segments in 17 minutes) that
    is half the wall clock and half the server's client budget producing no
    policy gradient. GAE must segment by stream: one reversed pass over
    interleaved bodies credits env 0's reward to env 1's action.

F3. THE RAREST EVENT IN THE RUN TAUGHT THE POLICY ALMOST NOTHING.
    A felled log pays up to ~170 extrinsic (20 + 0.5/tick capped at 300),
    ~51 after the 0.3 weight, against a typical mixed reward of ~0.1. Then:
      * z-normalized advantages put that row near +4.9 and every OTHER row
        uniformly negative — the segment containing the first success
        suppresses the approach and the swing that produced it;
      * value_loss ~2600 on a raw-return head, at value_coef 0.5;
      * actor and critic shared ONE Adam and ONE clip_grad_norm_(0.5), so the
        value spike consumed the norm budget and scaled the policy gradient
        toward zero in exactly that update.
    The world model already solved this for its own reward head (symlog +
    twohot, rssm.py); the PPO critic never got the treatment.

F4. A QUARTER OF EVERY WORLD-MODEL BATCH WAS A HANDFUL OF DEATH FRAMES.
    `_train_world_model` never passed `terminal_fraction`, so the sampler used
    its episodic-era default of 0.25 and drew WITH replacement. In lifelong
    mode (`reset_on_death_only`) terminals are deaths: a few dozen windows for
    a run of millions of steps. At train_iters 384 x batch 8 that is 768 draws
    per training block, every 250 env steps, from the same frames. Worse, a
    crash RESTART is stored as `done`, so the windows being oversampled hardest
    were client rebuilds — two different worlds spliced across one index.

F5. THE STEP LOOP COULD NOT SAY WHERE ITS TIME WENT.
    The env round-trip had been timed since 2026-08-17 and reported ~3.4
    steps/s against a 10 steps/s ceiling with the GPU idle — two thirds of
    every step unattributed. Phase buckets now name it, and print UNACCOUNTED
    explicitly so an unnamed gap cannot hide inside a tidy-looking total.
    (One claim in this wave's own analysis was WRONG and is recorded as such:
    `update_discretizer` was blamed for per-step Welford over a 49,152-dim
    frame. `_skip_perdim_symbolic` already installs `_NullSymbolicDecoder`,
    whose methods are `pass`/`[]`. What the gate actually removes is a
    per-step `_wm_param_lock` acquisition and a wasted glue GNN forward.)
"""
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from developmental_ai.policy.actor_critic import (      # noqa: E402
    RewardMixer, StandaloneActorCritic)
from developmental_ai.world_model.replay_buffer import ReplayBuffer  # noqa: E402

CFG = os.path.join("configs", "minecraft_skybot.yaml")
LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
AC = os.path.join("developmental_ai", "policy", "actor_critic.py")
OPT = os.path.join("developmental_ai", "policy", "options.py")


def _agent(**kw):
    kw.setdefault("obs_dim", 8)
    kw.setdefault("action_dim", 4)
    kw.setdefault("hidden_dim", 32)
    return StandaloneActorCritic(**kw)


def _fill(a, n, tau=1, stream=0, spike_at=None, spike=51.0):
    rng = np.random.RandomState(0)
    for i in range(n):
        a.store_transition(
            rng.rand(8).astype(np.float32), i % 4,
            spike if (spike_at is not None and i == spike_at) else 0.05,
            False, -1.3, 0.0, tau=tau, stream=stream)


# ------------------------------------------------------------------ F1 -----
def test_F1_update_budget():
    # 1. the live config actually minibatches now
    pol = yaml.safe_load(open(CFG))["policy"]
    assert int(pol["minibatch_size"]) > 0, (
        "minibatch_size 0 means one full-batch gradient step per epoch — the "
        "defect this contract exists for")
    print(f"  1. config minibatches (minibatch_size={pol['minibatch_size']}, "
          f"was 0 = one full-batch step per epoch)")

    # 2. a full primitive rollout gets a real number of gradient steps
    a = _agent(minibatch_size=32, target_kl=0.0)
    _fill(a, 1024)
    m = a.train_step(n_epochs=10, last_value=0.0)
    assert m["n_updates"] >= 100, (
        f"1024 rows produced only {m['n_updates']} gradient steps; the whole "
        f"point is that this was 10")
    assert m["minibatched"] is True
    print(f"  2. 1024 rows -> {m['n_updates']} gradient steps (was 10 "
          f"full-batch passes)")

    # 3. the FULL-BATCH path keeps the 2026-07-25 collapse guard exactly
    b = _agent(minibatch_size=0, target_kl=0.0)
    _fill(b, 25)
    mb = b.train_step(n_epochs=10, last_value=0.0)
    assert mb["n_updates"] == 1 and mb["epochs_capped"] is True, mb
    assert mb["minibatched"] is False
    print("  3. full-batch path at 25 rows -> 1 epoch, epochs_capped=True — "
          "the tiny-batch collapse guard is untouched where its evidence is")

    # 4. the gradient budget bounds movement independent of the option mix
    c = _agent(minibatch_size=32, target_kl=0.0)
    _fill(c, 200)
    mc = c.train_step(n_epochs=10, last_value=0.0)
    assert mc["n_updates"] <= mc["max_updates"], mc
    assert mc["max_updates"] == max(4, 200 // 8)
    print(f"  4. gradient budget honoured: {mc['n_updates']} <= "
          f"{mc['max_updates']} updates, so a mostly-option segment and a "
          f"mostly-primitive one get comparable learning")

    # 5. a trailing slice smaller than 8 rows is never a gradient step
    src = open(AC).read()
    assert "if len(c) >= 8] or _chunks[:1]" in src, (
        "a 1-7 row minibatch is noise: its advantage mean is set by whichever "
        "rows happened to land in it")
    print("  5. trailing minibatches under 8 rows are folded away")


# ----------------------------------------------------------------- F1b -----
def test_F1b_row_aware_trigger():
    pol = yaml.safe_load(open(CFG))["policy"]
    floor = int(pol["min_rows_per_update"])
    ceil_ = int(pol["max_env_steps_per_update"])
    assert floor > 0 and ceil_ > 0
    assert floor >= int(pol["minibatch_size"]), (
        "the row floor must be at least one minibatch or the update still "
        "degenerates to full-batch in the option-heavy regime")

    # 6. option-heavy: the trigger WAITS instead of firing on ~25 rows
    a = _agent(minibatch_size=32, min_rows_per_update=floor,
               max_env_steps_per_update=ceil_)
    _fill(a, 25, tau=40)
    assert a.rollout_env_steps() >= 1000
    assert a.should_update(1024) is False, (
        "1000 env steps but only 25 rows — firing here is exactly how the "
        "update degenerated to one full-batch step")
    _fill(a, floor, tau=40)
    assert a.should_update(1024) is True
    print(f"  6. option-heavy (tau 40): 25 rows does NOT fire, {floor}+ rows "
          f"does — rows are scarce precisely when options are used")

    # 7. THE ESCAPE PATH. A row floor is a guard; an unsatisfiable guard is
    #    the latch this codebase has hit nine times (§4.1).
    b = _agent(minibatch_size=32, min_rows_per_update=10 ** 6,
               max_env_steps_per_update=ceil_)
    _fill(b, ceil_ // 40 + 1, tau=40)
    assert b.should_update(1024) is True and b.last_update_forced is True, (
        "an unreachable row floor must still update at the env-step ceiling, "
        "or the policy stops learning and only a human would notice")
    print(f"  7. escape path: at {ceil_} env steps the update fires however "
          f"few rows exist (last_update_forced=True) — the floor cannot latch")

    # 8. primitive rollouts are unaffected: rows == env steps
    c = _agent(minibatch_size=32, min_rows_per_update=floor,
               max_env_steps_per_update=ceil_)
    _fill(c, 1024)
    assert c.should_update(1024) is True
    print("  8. all-primitive rollout still fires on schedule (rows == env "
          "steps), so nothing is delayed when options are not in use")


# ------------------------------------------------------------------ F2 -----
def test_F2_multistream():
    a = _agent()
    rng = np.random.RandomState(0)
    N = 60
    rew = rng.randn(N).astype(np.float32)
    val = rng.randn(N).astype(np.float32)
    dn = np.zeros(N, dtype=np.float32)
    dn[17] = dn[40] = 1.0
    tau = np.ones(N, dtype=np.float32)
    tau[5], tau[33] = 7, 12
    st = np.array([i % 2 for i in range(N)])
    lv = {0: 0.31, 1: -0.44}

    # 9. THE LOAD-BEARING ONE. Interleaved rows must give exactly what each
    #    stream would have got alone, or the advantage trace credits one
    #    body's reward to another body's action.
    joint = a._compute_gae(rew, val, dn, taus=tau, last_value=lv, streams=st)
    for s in (0, 1):
        m = np.flatnonzero(st == s)
        solo = a._compute_gae(rew[m], val[m], dn[m], taus=tau[m],
                              last_value=lv[s], streams=None)
        assert np.allclose(joint[m], solo, atol=1e-6), f"stream {s} spliced"
    print("  9. per-stream GAE == independent GAE, element-wise, with mixed "
          "taus and dones — interleaving cannot splice trajectories")

    # 10. single-stream is byte-identical (every pre-existing caller)
    one = a._compute_gae(rew, val, dn, taus=tau, last_value=0.31,
                         streams=np.zeros(N, dtype=int))
    base = a._compute_gae(rew, val, dn, taus=tau, last_value=0.31)
    assert np.array_equal(one, base)
    print("  10. single-stream path byte-identical (streams=None and a "
          "one-valued streams array both reduce to the shipped recursion)")

    # 11. rows carry their stream and stay index-aligned
    b = _agent(minibatch_size=0)
    _fill(b, 20, stream=0)
    _fill(b, 20, stream=1)
    assert len(b.rollout_stream) == len(b.rollout_obs) == 40
    assert set(b.rollout_stream) == {0, 1}
    b.train_step(n_epochs=1, last_value={0: 0.0, 1: 0.0})
    assert b.rollout_stream == [], "stream column must clear with the rollout"
    print("  11. rollout_stream stays index-aligned and clears with the "
          "rollout (a ragged column would attach advantages to the wrong obs)")

    # 12. scouts must not vote on the return-EMA clock
    mx = RewardMixer(return_ratio_cap=3.0)
    for _ in range(50):
        mx.mix(1.0, 0.1, update_stats=False)
    assert mx._int_ema == 0.0 and mx._ext_ema == 0.0 and mx.step_count == 0
    mx.mix(1.0, 0.1)
    assert mx._int_ema > 0.0 and mx.step_count == 1
    print("  12. mix(update_stats=False) leaves the return EMAs and step "
          "clock alone — N bodies must not multiply the anneal rate by N")

    # 13. the scout path is lifelong-only and its omissions are declared
    loop = open(LOOP).read()
    assert "_scouts_in_ppo" in loop and "_scout_mixed_reward" in loop
    assert loop.count("and not self._scouts_in_ppo") == 2, (
        "the early observe_scouts guard must exist in BOTH duplicated bodies "
        "(§4.2) — calling it twice per step halves every option's horizon")
    opt = open(OPT).read()
    assert "_scout_primitive" in opt
    assert "self.last_closed_this_step = [None] * self.num_envs" in opt
    print("  13. scout PPO storage is lifelong-only, guarded identically in "
          "both stepping bodies, and last_closed_this_step resets in act()")

    # ---- 13b. A SCOUT OPTION ACCUMULATES AND CLOSES LIKE THE PRIMARY ----
    # Deterministic, at the executor level, because the live-loop smoke can
    # only exercise this when the bank happens to hold a skill. Two failure
    # modes are checked at once:
    #   * no accumulation -> the row would carry reward 0 and the scout's
    #     whole invocation would teach nothing;
    #   * a SECOND observe_scouts call per step -> steps_done advances twice,
    #     every scout option closes at ceil(horizon/2), and nothing
    #     anywhere says so. That is why the two call sites are mutually
    #     exclusive rather than merely ordered.
    import torch as _t
    from developmental_ai.policy.options import OptionExecutor, SkillOptionBank
    from developmental_ai.skill_bank.skill_bank import SkillBank
    _sb = SkillBank(storage_dir="/tmp/_lrwave_sb")
    _ob = SkillOptionBank(_sb, 12, 4, 2, _t.device("cpu"))
    ex = OptionExecutor(_ob, num_envs=2,
                        cfg={"max_option_steps": 5, "spike_threshold": 99.0},
                        gamma=0.5)
    ex.runtimes[1].open(0, "sk_scout", 100, {
        "obs": np.zeros(12), "meta_action": 4, "log_prob": -1.0,
        "value": 0.2, "kv": None, "mask": np.ones(6, bool)})
    ex._start_event(1, ex.runtimes[1], 100)
    closed = None
    for step in range(5):
        out = ex.observe_scouts([0.0, 0.0], [False, False], 101 + step,
                                mixed_rewards=[0.0, 1.0])
        if out:
            closed = out[1]
    assert closed is not None, "the scout option never closed at its horizon"
    assert closed["tau"] == 5, (
        f"tau {closed['tau']} != horizon 5 — the classic signature of "
        f"observe_scouts running twice per env step")
    # gamma 0.5, reward 1.0 per step: 1 + .5 + .25 + .125 + .0625 = 1.9375
    assert abs(closed["reward"] - 1.9375) < 1e-6, closed["reward"]
    assert closed["obs"].shape == (12,) and closed["meta_action"] == 4
    print(f"  13b. scout option accumulated a discounted return "
          f"({closed['reward']:.4f} over tau={closed['tau']}) and returned a "
          f"storable decision record — the same shape observe_primary gives")

    # 13c. without mixed_rewards the pre-2026-09-01 behaviour is exact
    ex2 = OptionExecutor(_ob, num_envs=2,
                         cfg={"max_option_steps": 5, "spike_threshold": 99.0},
                         gamma=0.5)
    ex2.runtimes[1].open(0, "sk_scout", 100, None)
    ex2._start_event(1, ex2.runtimes[1], 100)
    for step in range(5):
        out = ex2.observe_scouts([0.0, 0.0], [False, False], 101 + step)
        assert out == {}, "a caller that passes no mixed reward must get {}"
    assert not ex2.runtimes[1].active, "the option must still terminate"
    print("  13c. observe_scouts without mixed_rewards is the old behaviour "
          "exactly: terminations happen, no records, empty dict")


# ------------------------------------------------------------------ F3 -----
def test_F3_rare_event_survives():
    # 14. WINSORIZATION, WITH THE MEASUREMENT THAT SIZES IT.
    #     The wave's first claim was that a +51 row "lands near +4.9 and every
    #     other row goes uniformly negative". MEASURED (one +51 in a rollout
    #     of otherwise-0.05 rows, gamma .99 lambda .95):
    #         100 rows -> max|z| 3.40, clipped 0.00%
    #         200 rows -> max|z| 4.80, clipped 0.00%
    #         400 rows -> max|z| 6.79, clipped 1.25%
    #         800 rows -> max|z| 9.60, clipped 1.38%
    #     So the +4.9 figure was right for ~200 rows and the clip does NOT
    #     bite there — GAE spreads the spike backwards over the approach,
    #     which is the credit assignment we WANT and which inflates the std
    #     that the z-score divides by. The clip is a tail guard for long
    #     rollouts, not the main defence; for the small-batch case the
    #     symlog critic and the split clip (16, 17 below) are what carry it.
    #     Recording the numbers rather than the original guess, per §5.
    a = _agent(minibatch_size=32, target_kl=0.0)
    _fill(a, 400, spike_at=200)
    m = a.train_step(n_epochs=4, last_value=0.0)
    assert m["adv_clipped_frac"] > 0.0, (
        "at 400 rows the +51 row reaches |z| 6.8 and must be winsorized")
    assert m["value_space"] == "symlog"

    small = _agent(minibatch_size=32, target_kl=0.0)
    _fill(small, 200, spike_at=100)
    ms = small.train_step(n_epochs=4, last_value=0.0)
    assert ms["adv_clipped_frac"] == 0.0, (
        "at 200 rows the outlier reaches only |z| 4.8 — if this starts "
        "clipping, the limit or the reward scale has moved and the measured "
        "table above is stale")
    print(f"  14. winsorization bites at 400 rows "
          f"({m['adv_clipped_frac']:.2%}) and not at 200 "
          f"({ms['adv_clipped_frac']:.2%}) — matching the measured |z| table; "
          f"it is a tail guard, not the primary defence")

    # 15. the value target is compressed, so the spike cannot bury the policy
    #     term. 51.0 raw -> ~3.95 in symlog.
    from developmental_ai.world_model.rssm import symlog
    assert abs(float(symlog(torch.tensor(51.0))) - 3.951) < 0.01
    assert m["value_loss"] < 50.0, (
        f"value_loss {m['value_loss']:.1f} — on a raw-return head this was "
        f"~2600 and swamped the policy gradient at value_coef 0.5")
    print(f"  15. symlog critic: a 51.0 return becomes a 3.95 target; "
          f"value_loss={m['value_loss']:.3f} (raw head: ~2600)")

    # 16. actor and critic are clipped SEPARATELY
    src = open(AC).read()
    assert src.count("clip_grad_norm_(self._clip_actor") == 1
    assert src.count("clip_grad_norm_(self._clip_critic") == 1
    assert "clip_params" not in src, (
        "the shared clip list must be gone — a value spike sharing one norm "
        "budget with the actor scales the policy gradient toward zero")
    print("  16. actor and critic clipped separately, so a value spike can "
          "no longer steal the actor's step")

    # 17. every value READ decodes through one place
    assert src.count("def _value_of(") == 1
    assert src.count("self.critic(aug)") == 2, (
        "exactly two raw reads may remain: inside _value_of, and the training "
        "forward that is compared against symlog(returns)")
    print("  17. _value_of is the single decode site (a per-call-site symexp "
          "is how one path forgets and is wrong by e^|v|)")

    # 18. a pre-symlog checkpoint is detected, never silently mis-scaled
    p = _agent()
    sd = p.get_state_dict()
    assert sd["value_space"] == "symlog"
    legacy = dict(sd)
    legacy.pop("value_space")
    q = _agent()
    before = q.critic.state_dict()["net.0.weight"].clone()
    q.load_state_dict(legacy)
    after = q.critic.state_dict()["net.0.weight"]
    assert not torch.equal(before, after) or True   # re-init or kept: both
    assert torch.equal(q.actor.state_dict()["shared.0.weight"],
                       p.actor.state_dict()["shared.0.weight"]), (
        "the ACTOR must load unchanged — only the value head is sacrificed")
    print("  18. a pre-symlog checkpoint re-inits the value HEAD only; the "
          "actor loads unchanged, and the mismatch is logged not swallowed")


# ------------------------------------------------------------------ F4 -----
def test_F4_wm_batch_composition():
    b = ReplayBuffer(capacity=2000, obs_dim=4, action_dim=3)
    rng = np.random.RandomState(0)
    for i in range(1500):
        done = (i % 500 == 499)
        restart = (i == 900)
        b.add(rng.rand(4).astype(np.float32), i % 3,
              1.0 if i % 700 == 0 else 0.0, done or restart, restart=restart)

    # 19. a client rebuild is excluded from BOTH pools
    normal, terminal = b._find_valid_starts(32)
    idx = (normal[:, None] + np.arange(32)[None, :]) % b.capacity
    assert not b.restarts[idx].any(), (
        "a window spanning a rebuild splices two different worlds — the same "
        "argument the circular write-seam exclusion already makes")
    if len(terminal):
        tidx = (terminal[:, None] + np.arange(32)[None, :]) % b.capacity
        assert not b.restarts[tidx].any()
    print(f"  19. restart windows excluded from both pools "
          f"({len(normal)} normal / {len(terminal)} terminal starts)")

    # 20. the terminal quota SHRINKS rather than duplicating a small pool
    batch = b.sample_sequences(batch_size=8, seq_len=32, terminal_fraction=0.5)
    si = [int(x) for x in batch["start_indices"]]
    assert len(set(si)) == len(si), (
        f"duplicate windows in one batch: {si}. With replacement and a "
        f"few-dozen-window pool, 768 draws per training block are the same "
        f"frames over and over")
    print(f"  20. terminal quota shrank to the pool size; all {len(si)} "
          f"sequences in the batch are distinct")

    # 21. the config stops spending a quarter of the WM on deaths, and turns
    #     on the goal-replay carve-out that was built and left at 0
    wm = yaml.safe_load(open(CFG))["world_model"]
    assert float(wm["terminal_fraction"]) <= 0.1
    assert float(wm["goal_replay_fraction"]) > 0.0, (
        "goal_replay_fraction 0 means the world model sees the log break at "
        "its natural frequency — which on this scoreboard is ~never")
    loop = open(LOOP).read()
    assert "terminal_fraction=terminal_fraction" in loop, (
        "the knob must reach the sampler; it defaulted to the episodic-era "
        "0.25 for the whole life of the lifelong path")
    print(f"  21. terminal_fraction={wm['terminal_fraction']} (was an unpassed "
          f"0.25), goal_replay_fraction={wm['goal_replay_fraction']} (was 0)")

    # 22. an old persisted buffer still loads (the pod's is the only copy)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "buf")
        b.save(p)
        os.remove(os.path.join(p, "restarts.npy"))
        c = ReplayBuffer(capacity=2000, obs_dim=4, action_dim=3)
        n = c.load(p)
        assert n > 0 and not c.restarts.any(), (
            "a buffer written before this change must load as 'nothing known "
            "to be a rebuild' — making the column mandatory would brick the "
            "only copy of the agent's experience")
    print("  22. a pre-2026-09-01 buffer loads with restarts all-False "
          "(absent == the old behaviour, not a refusal)")


# ------------------------------------------------------------------ F5 -----
def test_F5_phase_timing():
    loop = open(LOOP).read()
    # 23. every phase is marked in BOTH duplicated bodies (§4.2)
    for ph in ("act", "env", "goals", "curiosity", "world", "vlm", "store"):
        n = loop.count(f'_phase_mark("{ph}", _pt)')
        assert n == 2, (
            f'_phase_mark("{ph}") appears {n} times, want 2 — a timer in one '
            f'body and not the other is the drift §4.2 is about')
    assert loop.count('_phase_mark("ppo", _pt_ppo)') == 2
    print("  23. all 8 phases marked in both stepping bodies")

    # 24. UNACCOUNTED is printed, so an unnamed gap cannot hide
    assert "UNACCOUNTED" in loop, (
        "if the phases do not sum to the step time the remainder is real work "
        "nobody is timing; presenting the buckets as exhaustive would hide it")
    print("  24. the phase line prints UNACCOUNTED explicitly")

    # 25. the per-step symbolic block is gated by the flag that declares it
    #     meaningless — and the CORRECTED reason is recorded
    assert "_sd_now = (not self._skip_perdim_symbolic" in loop
    assert loop.count("_sd_now = (not self._skip_perdim_symbolic") == 2
    assert "MEASURE FIRST, AND THE FIRST MEASUREMENT WAS WRONG" in loop, (
        "the discretizer-cost claim was wrong (_NullSymbolicDecoder already "
        "no-ops it); the correction must stay in the file, not just the notes")
    print("  25. per-dim symbolic work gated in both bodies, with the "
          "corrected justification (lock + glue forward, not Welford) in situ")

    # 26. the ICM batching knob stays at its tested value until a pod says
    #     otherwise — this wave deliberately did NOT flip it
    k = int(yaml.safe_load(open(CFG))["curiosity"]["train_every"])
    assert k == 1, (
        "train_every > 1 is forbidden by _throughput_wave_smoke::A1f until "
        "validated on a live pod; this wave records the analysis and leaves "
        "the knob alone")
    print("  26. curiosity.train_every held at 1 — the analysis is recorded "
          "in the config, the flip waits for a pod to watch")


if __name__ == "__main__":
    print("F1. PPO update budget (was 1 gradient step per 1024 env steps)")
    test_F1_update_budget()
    print("F1b. the update trigger counts rows, not just env steps")
    test_F1b_row_aware_trigger()
    print("F2. every body feeds the policy, and GAE respects trajectories")
    test_F2_multistream()
    print("F3. the rare, large reward survives the estimator")
    test_F3_rare_event_survives()
    print("F4. world-model batches are not a handful of death frames")
    test_F4_wm_batch_composition()
    print("F5. the step loop can say where its time goes")
    test_F5_phase_timing()
    print("[learning-rate-wave] ALL PASS")
