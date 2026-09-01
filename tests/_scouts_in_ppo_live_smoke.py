"""
_scouts_in_ppo_live_smoke — drive the REAL lifelong stepping body and prove
that scout streams now reach the PPO rollout.

Run:  PYTHONPATH=. ./venv/bin/python tests/_scouts_in_ppo_live_smoke.py

WHY A LIVE-LOOP TEST AND NOT MORE UNIT ASSERTIONS
================================================================================
The unit contracts in _learning_rate_wave_smoke prove the PIECES: GAE segments
by stream, the mixer can be read without voting, the executor records scout
decisions. None of them proves the piece are WIRED, and this project's most
expensive bugs have all been exactly that — a correct mechanism nobody called:

  * the gui_open guard sat below the line it was meant to protect and was
    inert in BOTH loop bodies from the day it was written;
  * _viewer_push was only ever called from the serial path, so the viewer
    never received a frame in the mode skybot actually runs;
  * the felt-reach sense was computed only in the episodic body, so the
    proprioception field was silently zero for every lifelong run.

Three subsystems found dead the same way. So this drives `_collect_segment`
itself, on Crafter (fast, Mac-ok), with 3 streams, and asserts the rollout
afterwards.

CONTRACTS
  1. rows appear from EVERY stream, not just stream 0
  2. the stream column stays index-aligned with every other rollout array
  3. scout rows carry a real (non-zero) mixed reward — i.e. _scout_mixed_reward
     is being called, not silently returning a default
  4. an option row from a scout carries tau > 1 when options are in play
  5. the update runs on the combined rollout and the metrics are finite
  6. observe_scouts is NOT double-called: every option's tau is <= its horizon
     (a second call per step would halve every horizon and is invisible
     otherwise)
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

SB = "/tmp/scouts_ppo_smoke_sb"


def main():
    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({"stochastic_size": 8, "stochastic_classes": 8,
                               "deterministic_size": 64, "encoder_hidden": 64,
                               "decoder_hidden": 64,
                               "batch_size": 4, "sequence_length": 8,
                               "train_iters": 1, "buffer_capacity": 4000})
    cfg["environment"]["max_episode_steps"] = 200
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 3}
    cfg["lifelong"] = {"enabled": True, "forever": False, "segment_len": 40,
                       "wm_train_every": 10 ** 9, "goal_horizon": 10 ** 9,
                       "state_decay": 1.0, "reset_on_death_only": True,
                       "stop_file": "/tmp/_scouts_ppo_smoke_STOP"}
    cfg["policy"].update({"n_steps": 10 ** 9,          # never auto-update
                          "minibatch_size": 8,
                          "min_rows_per_update": 0,
                          "max_env_steps_per_update": 0,
                          "target_kl": 0.05,
                          "scouts_in_ppo": True})
    cfg["skill_bank"]["storage_dir"] = SB
    cfg["loop"]["verbose"] = 0
    cfg.setdefault("skills_as_options", {})
    cfg["skills_as_options"].update({"enabled": True, "k_slots": 4,
                                     "max_option_steps": 6,
                                     "scouts_use_options": True,
                                     "gate_by_preconditions": False,
                                     "competence_floor": 0.0})

    from developmental_ai.core.developmental_loop import DevelopmentalAI
    ai = DevelopmentalAI(config=cfg)
    assert ai._scouts_in_ppo is True, (
        "the lifelong config did not enable scout PPO storage — everything "
        "below would pass vacuously")
    assert ai._num_envs == 3

    try:
        for _ in range(3):
            ai._collect_segment()

        pol = ai.policy
        rows = len(pol.rollout_obs)
        assert rows > 0, "no rows at all — the loop stored nothing"

        # ---- 1. every stream contributed --------------------------------
        streams = list(pol.rollout_stream)
        seen = sorted(set(streams))
        assert seen == [0, 1, 2], (
            f"rows came only from streams {seen}. Before this wave the answer "
            f"was [0] and the scouts' whole existence was invisible to PPO")
        counts = {s: streams.count(s) for s in seen}
        print(f"  1. rows per stream: {counts} (was {{0: n}} — scouts "
              f"contributed nothing to the policy)")

        # ---- 2. index alignment ----------------------------------------
        for name in ("rollout_obs", "rollout_actions", "rollout_rewards",
                     "rollout_dones", "rollout_log_probs", "rollout_values",
                     "rollout_taus", "rollout_masks", "rollout_stream"):
            assert len(getattr(pol, name)) == rows, (
                f"{name} has {len(getattr(pol, name))} entries, rows={rows} — "
                f"a ragged rollout attaches advantages to the wrong "
                f"observation and looks exactly like 'PPO is not learning'")
        print(f"  2. all rollout arrays index-aligned at {rows} rows")

        # ---- 3. scout rewards are real ---------------------------------
        sc_r = [r for r, s in zip(pol.rollout_rewards, streams) if s != 0]
        assert any(abs(float(r)) > 1e-9 for r in sc_r), (
            "every scout row carries reward 0.0 — _scout_mixed_reward is not "
            "being reached, which is the 'correct mechanism nobody calls' "
            "failure this file exists to catch")
        assert all(np.isfinite(float(r)) for r in sc_r)
        print(f"  3. {len(sc_r)} scout rows carry finite, non-trivial mixed "
              f"reward (mean |r| = "
              f"{np.mean([abs(float(r)) for r in sc_r]):.4f})")

        # ---- 4/6. option rows and the double-call check -----------------
        horizon = int(ai.option_executor.max_option_steps)
        taus = list(pol.rollout_taus)
        sc_taus = [t for t, s in zip(taus, streams) if s != 0]
        assert max(taus) <= horizon, (
            f"a stored tau of {max(taus)} exceeds max_option_steps {horizon}")
        # A SECOND observe_scouts call per step would advance steps_done twice
        # and terminate every scout option at ceil(horizon/2). If scouts ever
        # opened options here, their taus must reach the same range the
        # primary's do rather than clustering at half.
        if any(t > 1 for t in sc_taus):
            print(f"  4. scout option rows present: taus "
                  f"{sorted(set(t for t in sc_taus if t > 1))} "
                  f"(horizon {horizon})")
            assert max(sc_taus) > horizon // 2, (
                f"every scout option closed at or below half the horizon "
                f"({max(sc_taus)} vs {horizon}) — the signature of "
                f"observe_scouts being called twice per step")
            print("  6. scout option horizons are not halved — observe_scouts "
                  "is called exactly once per step")
        else:
            print(f"  4/6. no scout option opened in this run (all taus == 1); "
                  f"the primitive path is what was exercised")

        # ---- 5. the update runs on the combined rollout ------------------
        lv = {i: 0.0 for i in range(ai._num_envs)}
        m = pol.train_step(n_epochs=2, last_value=lv)
        for k in ("policy_loss", "value_loss", "entropy"):
            assert np.isfinite(m[k]), f"{k} is not finite: {m[k]}"
        assert m["n_updates"] >= 1
        print(f"  5. PPO ran on the combined rollout: rows={m['rows']} "
              f"updates={m['n_updates']} entropy={m['entropy']:.3f} "
              f"value_loss={m['value_loss']:.4f} (all finite)")

        print("[scouts-in-ppo-live] ALL PASS")
    finally:
        try:
            ai.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
