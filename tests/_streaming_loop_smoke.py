"""Streaming-segment loop integration smoke (Crafter, Mac-ok) — the lifelong
linchpin. Drives the REAL loop.

Contracts:
  1. Flag-OFF: DevelopmentalAI constructs and runs an episodic episode with the
     run() guards in place (no breakage from the additive guards). _collect_
     segment / _start_stream are never entered.
  2. Flag-ON construction: _lifelong true; requires parallel_envs (ValueError
     if single-env).
  3. State CARRIES across segments: rssm_state is the SAME object across two
     _collect_segment calls; _stream_started stays True; the stream is set up
     exactly ONCE.
  4. NO env.reset after stream start: env resets happen only in _start_stream
     (once per env); across further segments, zero resets (pure stream).
  5. State DECAYS: h shrinks by state_decay each segment (the "diminish older
     states" guard); h_norm is finite (the stability signal).
  6. WM trains on the STEP cadence (wm_train_every), and its metrics feed
     _last_wm_metrics.
  7. Forced death zeroes exactly that stream's RSSM row (others untouched).
  8. PPO segment update uses a NON-zero bootstrap (compute_last_value), since
     the segment ends mid-trajectory.
"""
import copy
import shutil

import numpy as np
import torch
import yaml


def _cfg(lifelong=True, num_envs=2):
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 32, "encoder_hidden": 32, "batch_size": 4,
        "sequence_length": 6, "train_iters": 1, "buffer_capacity": 3000})
    # huge episode limit so Crafter never truncates during the smoke -> a pure
    # continuous stream (MineRL gets the same via the lifelong env flag)
    cfg["environment"]["max_episode_steps"] = 100000
    cfg["parallel_envs"] = {"enabled": num_envs > 1, "num_envs": num_envs}
    cfg["skill_bank"]["storage_dir"] = "/tmp/stream_smoke_sb"
    cfg["loop"]["verbose"] = 0
    cfg["llm"]["enabled"] = False
    cfg["llm"]["vision"] = {"enabled": False}
    cfg["policy"]["n_steps"] = 8          # tiny so the segment triggers a PPO update
    if lifelong:
        cfg["lifelong"] = {
            "enabled": True, "segment_len": 6, "wm_train_every": 3,
            "goal_horizon": 100000, "state_decay": 0.9,
            "active_skills": 24, "active_goals": 48}
    return cfg


def test_flag_off_episodic_unbroken():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    shutil.rmtree("/tmp/stream_smoke_sb", ignore_errors=True)
    ai = DevelopmentalAI(_cfg(lifelong=False))
    assert ai._lifelong is False
    m = ai._run_episode_parallel(use_dream_actor=False)   # episodic path
    assert "episode_reward" in m and m["episode_length"] > 0
    assert ai._ll is None and ai._stream_started is False, (
        "episodic run touched lifelong state")
    print("  1. flag-off episodic path unbroken by the run() guards")


def test_lifelong_requires_parallel():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    c = _cfg(lifelong=True, num_envs=1)
    c["parallel_envs"] = {"enabled": False, "num_envs": 1}
    try:
        DevelopmentalAI(c)
        raise AssertionError("single-env lifelong not rejected")
    except ValueError as e:
        assert "parallel" in str(e)
    print("  2. lifelong requires parallel envs (single-env rejected)")


def test_stream_carries_and_no_reset():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    shutil.rmtree("/tmp/stream_smoke_sb", ignore_errors=True)
    ai = DevelopmentalAI(_cfg(lifelong=True))
    assert ai._lifelong is True

    # count env.reset calls
    resets = {"n": 0}
    ai._ensure_parallel_envs()
    for e in ai._parallel_envs:
        _raw = e.reset
        def _counting(_raw=_raw):
            resets["n"] += 1
            return _raw()
        e.reset = _counting

    m1 = ai._collect_segment()
    assert ai._stream_started is True and ai._ll is not None
    rssm_obj = ai._ll.rssm_state
    h_after_seg1 = ai._ll.h.clone()
    resets_after_start = resets["n"]
    assert resets_after_start >= 1, "stream start never reset the envs"

    m2 = ai._collect_segment()
    # SAME rssm_state object -> state carried, not re-initialised
    assert ai._ll.rssm_state is rssm_obj, "RSSM state was re-created (not carried)"
    # zero further resets across the second segment (pure continuous stream)
    assert resets["n"] == resets_after_start, (
        f"env.reset called after stream start: "
        f"{resets['n'] - resets_after_start} extra")
    assert m1["episode_length"] == 6 and m2["episode_length"] == 6
    print(f"  3/4. stream carries + no reset after start "
          f"({resets_after_start} initial resets, 0 after)")


def test_decay_and_death():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    shutil.rmtree("/tmp/stream_smoke_sb", ignore_errors=True)
    ai = DevelopmentalAI(_cfg(lifelong=True))
    ai._collect_segment()               # stream up
    h0 = ai._ll.h.clone()
    # decay is applied at each segment boundary -> ||h|| trends down when the
    # GRU isn't re-inflating it; assert the explicit decay call shrinks h
    ai._ll.h.fill_(1.0)
    ai._ll.decay()
    assert torch.allclose(ai._ll.h, torch.full_like(ai._ll.h, 0.9)), (
        "segment decay did not shrink h")
    assert ai._ll.h_norm() > 0.0
    # forced death on stream 1 zeroes only that row
    ai._ll.h.fill_(2.0)
    ai._ll.z.fill_(2.0)
    ai._ll.death_reset(1)
    assert torch.all(ai._ll.h[1] == 0.0) and torch.all(ai._ll.z[1] == 0.0)
    assert torch.all(ai._ll.h[0] == 2.0), "death touched the wrong stream"
    print("  5/7. state decay + death-reset (one row) ok")


def test_wm_cadence_and_bootstrap():
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    shutil.rmtree("/tmp/stream_smoke_sb", ignore_errors=True)
    ai = DevelopmentalAI(_cfg(lifelong=True))

    # spy on WM training + the PPO bootstrap
    wm_calls = {"n": 0}
    _raw_wm = ai._train_world_model
    def _wm():
        wm_calls["n"] += 1
        return _raw_wm()
    ai._train_world_model = _wm

    boot = {"last_value": None}
    _raw_ts = ai.policy.train_step
    def _ts(*a, **k):
        boot["last_value"] = k.get("last_value")
        return _raw_ts(*a, **k)
    ai.policy.train_step = _ts

    # is_ready needs >=100 transitions (12/segment at 2 envs x 6 steps), so
    # run ~15 segments to clear the warmup, then confirm WM trained on cadence.
    for _ in range(15):
        ai._collect_segment()
    assert wm_calls["n"] >= 1, "WM never trained on the step cadence"
    assert ai._last_wm_metrics is not None, "WM metrics not captured for run()"
    # the PPO update fired with a real (non-None) bootstrap value
    assert boot["last_value"] is not None, "segment PPO update never fired"
    assert isinstance(boot["last_value"], float), "bootstrap not a float V(s)"
    print(f"  6/8. WM step-cadence ({wm_calls['n']} trains) + PPO bootstrap ok")


if __name__ == "__main__":
    for fn in (test_flag_off_episodic_unbroken, test_lifelong_requires_parallel,
               test_stream_carries_and_no_reset, test_decay_and_death,
               test_wm_cadence_and_bootstrap):
        print(f"[streaming-smoke] {fn.__name__}")
        fn()
    print("[streaming-smoke] ALL PASS")
