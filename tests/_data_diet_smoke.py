"""Data-diet wave (2026-08-23) — what the brain is actually fed.

"Thin data diet" turned out to be five independent things. Each was measured
against the live config, not estimated.

W1. A TRAINING SEQUENCE COULD NOT CONTAIN A TREE-BREAK.
    `sequence_length: 16` at `action_repeat: 2` spans 32 GAME TICKS. Measured
    `Ticks-to-break` from the live logs: dirt/grass are 20-22t and fit, but a
    BAREHANDED LOG needs ~60t. The world model had therefore never seen a
    complete log-break as one trajectory — only its first or second half. We
    were asking it to learn an event that did not fit in its window. Batch is
    halved in exchange, so transitions per gradient step and the obs tensor
    are unchanged: temporal span bought at constant memory, which matters
    because VRAM is shared with the VLM.

W2. REPLAY RATIO 65.5 vs DreamerV3-on-Minecraft's ~512, on a GPU measured at
    2% utilisation with the async trainer reporting `skip 0.0%` — never
    behind. We were ~8x under-training on data already collected.

W3. MEMORY HORIZON 10.4 HOURS. `buffer_capacity` is SPLIT across streams, so
    240k with 2 bodies gave each 120k transitions. Everything older was
    overwritten, on a box with 125 GB of RAM and a buffer using 11.8 GB.

W4. THE MEMORY WAS VOLATILE. `ReplayBuffer` had no save/load path at all, so
    the supervisor's automatic crash-relaunch restarted world-model training
    from an EMPTY buffer. The effective diet was never the configured
    capacity; it was "however long since the last crash".

W5. THE BODY COULD NOT FEEL MOTION. Proprioception carried hunger, health,
    depth, menu-open, swing streak and pitch — but nothing for "am I actually
    going anywhere", for an agent whose every recorded failure is a failure to
    move (sky-staring, 10,149 steps in a trade menu, attacking an unreachable
    trunk). The adapter already read xpos/zpos every step for coverage.

Run: PYTHONPATH=. python tests/_data_diet_smoke.py
"""
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, ".")

import numpy as np
import yaml

CFG = os.path.join("configs", "minecraft_skybot.yaml")

# Measured from live logs, not assumed. `Ticks-to-break` shows dirt/grass at
# 20-22t; the audit records a barehanded log at ~60t.
LOG_BREAK_TICKS = 60


def _cfg():
    return yaml.safe_load(open(CFG))


# ---------------------------------------------------------------- W1 -----
def test_W1_sequence_spans_a_tree_break():
    c = _cfg()
    seq = int(c["world_model"]["sequence_length"])
    rep = int(c["environment"]["action_repeat"])
    span = seq * rep
    assert span >= LOG_BREAK_TICKS, (
        f"a training sequence spans {span} game ticks but a barehanded log "
        f"needs ~{LOG_BREAK_TICKS}. The world model cannot see a complete "
        f"log-break as one trajectory — it gets the first half or the second "
        f"half. This is the defect the wave exists to fix; raise "
        f"world_model.sequence_length or lower environment.action_repeat.")
    print(f"  W1. seq {seq} x action_repeat {rep} = {span} game ticks "
          f">= {LOG_BREAK_TICKS} — a sequence can now contain a whole "
          f"barehanded log-break")

    # ---- RE-BASED 256 -> 512 (2026-09-01 capacity wave) ---------------
    # The original contract was "this wave buys temporal span at CONSTANT
    # memory", written when batch 16 x seq 16 became 8 x 32 against a
    # measured 1.59 GB training step. The capacity wave deliberately spends
    # memory instead: batch 8 -> 16 at seq 32 is 512 transitions/grad-step,
    # ~3.2 GB, taken because a 54M-param world model updated on 8 sequences
    # carries high gradient noise and this model is the substrate for
    # imagination, prospection, the policy's input under arch: rssm, and
    # mastery's fidelity ratio.
    #
    # BE HONEST ABOUT WHAT IS AND IS NOT MEASURED. The 1.59 GB figure was a
    # real measurement on the pod; the ~3.2 GB is arithmetic, and there is no
    # GPU on the machine this was changed from. So this is a BUDGET CHANGE
    # TAKEN ON PURPOSE, not a verified one — `AsyncWM ... block ms/iter` and
    # the VRAM high-water on the first cluster run are what settle it, and
    # world_model.batch_size is the first thing to revert if they do not.
    #
    # ---- RE-BASED BACK 512 -> 256 (2026-09-24) — THE MEASUREMENT ARRIVED
    # The text above called the ~3.2 GB "arithmetic, not verified", said the
    # VRAM high-water on the first real run would settle it, and named
    # world_model.batch_size as THE FIRST THING TO REVERT if it did not fit.
    # It did not fit. Measured on `main` (RTX 5050, 7.56 GiB usable), the run
    # did not merely slow down -- it DIED, three times on the same fault, and
    # the supervisor stopped itself:
    #     [supervisor] STOPPING: same fault 3x in a row -- this is
    #                  deterministic and will not fix itself
    #     torch.OutOfMemoryError: Tried to allocate 96.00 MiB.
    #     this process has 6.97 GiB in use (6.66 GiB by PyTorch)
    # Both fatal tracebacks land in conv_transpose2d; the decoder is 71.7M of
    # 120.5M params. Ollama held 202 MiB of VRAM at the time -- the VLM was
    # the SYSTEM-RAM problem, never the VRAM one.
    # sequence_length stays 32 (W1 above depends on it spanning a whole
    # log-break); batch 16 -> 8 halves activations and changes only the
    # gradient estimate. REVERT TARGET on a >=16 GB card: 512.
    # Still re-based rather than deleted: the job is to stop this moving
    # UNREVIEWED in either direction.
    trans = int(c["world_model"]["batch_size"]) * seq
    assert trans == 256, (
        f"batch x seq is {trans}, expected 256. Re-based from 512 by a "
        f"MEASURED OOM death on an 8 GB card (see the note above). Raising "
        f"it again needs a card with the VRAM to hold it, not an argument")
    print(f"  W1b. batch {c['world_model']['batch_size']} x seq {seq} = "
          f"{trans} transitions/grad-step (re-based from 512 by the 8 GB "
          f"wave — a deliberate, UNVERIFIED memory spend; see comment)")


# ---------------------------------------------------------------- W2 -----
def test_W2_replay_ratio():
    c = _cfg()
    w, l, p = c["world_model"], c["lifelong"], c["parallel_envs"]
    ratio = (int(w["train_iters"]) * int(w["batch_size"])
             * int(w["sequence_length"])
             / (int(l["wm_train_every"]) * int(p["num_envs"])))
    assert ratio > 65.5, (
        f"replay ratio {ratio:.1f} is no better than the 65.5 this wave "
        f"measured — the GPU is at 2% and the async trainer skips 0.0%, so "
        f"this is free learning being declined")
    assert ratio <= 512, (
        f"replay ratio {ratio:.1f} exceeds DreamerV3-on-Minecraft's ~512 "
        f"reference; past that, verify against wm_fidelity rather than "
        f"assuming more is better")
    print(f"  W2. replay ratio {ratio:.1f} replayed per collected "
          f"(was 65.5, DreamerV3 reference ~512)")


# ---------------------------------------------------------------- W3 -----
def test_W3_memory_horizon():
    c = _cfg()
    cap = int(c["world_model"]["buffer_capacity"])
    n = int(c["parallel_envs"]["num_envs"])
    per_stream = cap // n                     # MultiStreamReplayBuffer splits
    hours = per_stream / 3.2 / 3600.0
    assert hours > 10.4, (
        f"{hours:.1f} h per stream is no better than the 10.4 h measured; "
        f"capacity is SPLIT across streams, so adding a body shortens memory")
    gb = cap * 49152 / 1e9
    assert gb < 100, f"{gb:.0f} GB of obs storage against 125 GB of RAM"
    print(f"  W3. {cap:,} total -> {per_stream:,}/stream = {hours:.1f} h "
          f"of memory each (was 10.4 h), ~{gb:.0f} GB uint8 of 125 GB RAM")


# ---------------------------------------------------------------- W4 -----
def _mk(cap=64, obs_dim=6, action_dim=3, seed=0):
    from developmental_ai.world_model.replay_buffer import ReplayBuffer
    b = ReplayBuffer(capacity=cap, obs_dim=obs_dim, action_dim=action_dim)
    rs = np.random.RandomState(seed)
    for i in range(cap + 20):                 # force a wrap
        b.add(rs.rand(obs_dim).astype(np.float32),
              rs.rand(action_dim).astype(np.float32),
              float(i), bool(i % 17 == 0))
    return b


def test_W4_buffer_persistence():
    from developmental_ai.world_model.replay_buffer import (
        ReplayBuffer, MultiStreamReplayBuffer)
    tmp = tempfile.mkdtemp()
    try:
        src = _mk()
        assert src.size == src.capacity, "test needs a wrapped buffer"
        path = os.path.join(tmp, "buf")
        n = src.save(path)
        assert n == src.size
        print(f"  W4. saved {n} transitions (buffer had wrapped, so this "
              f"exercises the circular -> chronological reordering)")

        # round-trip: identical content, in the same chronological order
        dst = ReplayBuffer(capacity=src.capacity, obs_dim=src.obs_dim,
                           action_dim=src.action_dim)
        assert dst.load(path) == n
        order = src._chronological()
        assert np.array_equal(dst.observations[:n], src.observations[order])
        assert np.array_equal(dst.rewards[:n], src.rewards[order])
        assert np.array_equal(dst.dones[:n], src.dones[order])
        assert dst.size == n
        print("  W4b. round-trip is exact, oldest-first — so a restore is "
              "valid even though the source had wrapped")

        # the newest experience survives a capacity SHRINK (and it is the
        # newest that is kept, not the oldest)
        small = ReplayBuffer(capacity=10, obs_dim=src.obs_dim,
                             action_dim=src.action_dim)
        assert small.load(path) == 10
        assert np.array_equal(small.rewards[:10], src.rewards[order][-10:])
        print("  W4c. capacity shrink keeps the NEWEST 10, exactly as the "
              "circular buffer would have")

        # a cap keeps the newest, not the oldest
        capped = os.path.join(tmp, "capped")
        assert src.save(capped, max_transitions=12) == 12
        c2 = ReplayBuffer(capacity=64, obs_dim=src.obs_dim,
                          action_dim=src.action_dim)
        c2.load(capped)
        assert np.array_equal(c2.rewards[:12], src.rewards[order][-12:])
        print("  W4d. max_transitions keeps the NEWEST slice (the cap bounds "
              "the write; it must not silently archive stale experience)")

        # ---- the failure modes that matter ----------------------------
        # 1. no manifest = not committed = absent, never a partial load
        broken = os.path.join(tmp, "broken")
        shutil.copytree(path, broken)
        os.remove(os.path.join(broken, "manifest.json"))
        try:
            ReplayBuffer(capacity=64, obs_dim=src.obs_dim,
                         action_dim=src.action_dim).load(broken)
            raise AssertionError("a manifest-less directory was loaded")
        except FileNotFoundError as e:
            assert "manifest" in str(e)
        print("  W4e. no manifest -> refused. The manifest is written LAST, "
              "so an interrupted save reads as absent rather than as a "
              "half-written buffer (world_model.pt's non-atomic write cost "
              "this project one silent partial restore)")

        # 2. shape drift is refused BEFORE the buffer is touched
        wrong = ReplayBuffer(capacity=64, obs_dim=src.obs_dim,
                             action_dim=src.action_dim + 1)
        before = wrong.observations.copy()
        try:
            wrong.load(path)
            raise AssertionError("action_dim mismatch was accepted")
        except ValueError as e:
            assert "action_dim" in str(e)
        assert np.array_equal(wrong.observations, before) and wrong.size == 0, (
            "the buffer was mutated before the shape check — that is the "
            "partial-hybrid failure the world-model loader was just fixed for")
        print("  W4f. action_dim drift (it moved 10 -> 12 once) -> refused "
              "with the buffer still pristine, not half-copied")

        # 3. it still samples after a restore
        dst2 = ReplayBuffer(capacity=src.capacity, obs_dim=src.obs_dim,
                            action_dim=src.action_dim)
        dst2.load(path)
        batch = dst2.sample_sequences(batch_size=2, seq_len=4)
        assert batch["observations"].shape == (2, 4, src.obs_dim)
        print("  W4g. a restored buffer samples sequences normally "
              f"{tuple(batch['observations'].shape)} — episode_starts are "
              f"deliberately not restored; `dones` carries the boundaries")

        # multi-stream delegation
        ms = MultiStreamReplayBuffer(num_streams=2, capacity=200, obs_dim=6,
                                     action_dim=3)
        rs = np.random.RandomState(1)
        for i in range(150):
            ms.add(rs.rand(6).astype(np.float32), rs.rand(3).astype(np.float32),
                   float(i), False, stream=i % 2)
        mpath = os.path.join(tmp, "ms")
        assert ms.save(mpath) == len(ms)
        ms2 = MultiStreamReplayBuffer(num_streams=2, capacity=200, obs_dim=6,
                                      action_dim=3)
        assert ms2.load(mpath) == len(ms)
        print(f"  W4h. multi-stream saves/restores per stream ({len(ms)} "
              f"transitions across 2 bodies)")

        # a run with FEWER bodies than were saved must still start
        ms1 = MultiStreamReplayBuffer(num_streams=1, capacity=200, obs_dim=6,
                                      action_dim=3)
        got = ms1.load(mpath)
        assert got > 0
        print(f"  W4i. restoring into FEWER streams works ({got} transitions) "
              f"— the fleet was scaled 4 -> 2 once when the Paper server "
              f"buckled, and a brain that refused to boot over that would be "
              f"worse than one that resumes with what it can account for")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_W4_wired_into_the_run():
    c = _cfg()
    w = c["world_model"]
    assert w.get("buffer_persist") is True, "persistence is off"
    assert float(w.get("buffer_persist_max_gb", 0)) > 0, (
        "an uncapped buffer write would put ~49 GB on a volume whose "
        "instance footprint measured ~42 GB")
    assert "replay_buffer" in c["loop"]["resume_components"], (
        "the buffer is saved but never restored — the crash-relaunch still "
        "starts world-model training from empty")
    src = open(os.path.join("developmental_ai", "core",
                            "developmental_loop.py")).read()
    i_lock = src.index("self._save_checkpoint_locked()")
    i_buf = src.index("self._save_replay_buffer()")
    assert i_lock < i_buf, (
        "the buffer write must happen AFTER _wm_param_lock is released — the "
        "acting thread takes that same lock on every step, so a multi-GB "
        "write inside it would stall the agent for the whole write")
    print("  W4j. persist on, capped, in resume_components, and written "
          "OUTSIDE the WM param lock (so a GB-scale write cannot stall the "
          "acting thread)")


# ---------------------------------------------------------------- W5 -----
def test_W5_the_body_can_feel_motion():
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter as A
    assert A.PROPRIO_DIM == 13, A.PROPRIO_DIM
    for k in ("moved", "head_sin", "head_cos"):
        assert k in A.PROPRIO_KEYS, k
    print(f"  W5. PROPRIO_KEYS is now {A.PROPRIO_DIM} wide, with "
          f"{A.PROPRIO_KEYS[-3:]} appended")

    stub = types.SimpleNamespace(PROPRIO_DIM=A.PROPRIO_DIM,
                                 ATTACK_RUN_SCALE=100.0,
                                 MOVE_SCALE=A.MOVE_SCALE)
    p = A._proprio

    still = p(stub, {"moved": 0.0, "yaw": 0.0})
    assert still[10] == 0.0
    walk = p(stub, {"moved": A.MOVE_SCALE * 0.8, "yaw": 0.0})
    assert 0.7 < walk[10] < 0.9, walk[10]
    sprint = p(stub, {"moved": 99.0, "yaw": 0.0})
    assert sprint[10] == 1.0, "moved must clip, not run away"
    print(f"  W5b. standing still reads {still[10]:.2f}, a walk "
          f"{walk[10]:.2f}, and a teleport clips to {sprint[10]:.2f} — the "
          f"agent can now feel that it is going nowhere")

    for yaw in (0.0, 90.0, 187.5, 359.0):
        v = p(stub, {"yaw": yaw})
        # rescaled to [0,1] like every other entry, so undo that to check the
        # heading really is a point on the unit circle
        _s, _c = (v[11] - 0.5) * 2.0, (v[12] - 0.5) * 2.0
        assert abs(_s ** 2 + _c ** 2 - 1.0) < 1e-5, yaw
    unknown = p(stub, {})
    assert unknown[11] == 0.5 and unknown[12] == 0.5, (
        "unknown yaw must read the NEUTRAL 0.5, the same convention `pitch` "
        "uses — not 0.0, which is a real heading")
    # THE NORMALIZATION INVARIANT the older proprio suite encodes: every
    # entry lives in [0,1]. A raw sine would have been the only member on a
    # different scale.
    for w in ({}, {"yaw": 187.5, "moved": 9.0}, {"yaw": 270.0}):
        vv = p(stub, w)
        assert (0.0 <= vv).all() and (vv <= 1.0).all(), (w, vv)
    print("  W5c. heading is sin/cos (continuous across the 360->0 wrap), "
          "rescaled to [0,1] like every other entry; unknown reads the "
          "neutral 0.5/0.5, matching `pitch`")

    # displacement is measured, and a world boundary must not read as travel
    ev = A._world_events
    s2 = types.SimpleNamespace(
        coverage_cell=16, _visits={}, _visit_writes=0,
        _read_scalar=A._read_scalar, _read_stat=A._read_stat,
        _save_break_memory=lambda **k: None,
        _prev_xz=None, _prev_life=None, _prev_food=None, _prev_deaths=None,
        _prev_damage_taken=None, _prev_mainhand=None, _prev_derived_hand=None,
        _prev_tool_damage=None, _last_placed_item=None, _start_tool=None)
    o1 = ev(s2, {"xpos": 100.0, "zpos": 50.0, "ypos": 64.0, "life": 20.0})
    assert o1["moved"] == 0.0, "first step after a reset must read 0"
    o2 = ev(s2, {"xpos": 100.3, "zpos": 50.4, "ypos": 64.0, "life": 20.0})
    assert abs(o2["moved"] - 0.5) < 1e-6, o2["moved"]
    print(f"  W5d. _world_events measures displacement ({o2['moved']:.2f} "
          f"blocks) and the first step after a reset reads 0 — a fresh world "
          f"teleports the body, and that must not read as a sprint")

    src = open(os.path.join("developmental_ai", "environments",
                            "minerl_env.py")).read()
    assert "self._prev_xz = None" in src
    i_reset = src.index("self._prev_agency_state = None")
    assert "self._prev_xz = None" in src[i_reset:i_reset + 400], (
        "the displacement anchor must be cleared on reset alongside the "
        "other cross-world state")
    print("  W5e. the anchor is cleared on reset with the other "
          "never-survives-a-world-boundary state")


if __name__ == "__main__":
    print("W1. a training sequence can contain a whole tree-break")
    test_W1_sequence_spans_a_tree_break()
    print("W2. replay ratio")
    test_W2_replay_ratio()
    print("W3. memory horizon")
    test_W3_memory_horizon()
    print("W4. the memory survives a restart")
    test_W4_buffer_persistence()
    test_W4_wired_into_the_run()
    print("W5. the body can feel motion")
    test_W5_the_body_can_feel_motion()
    print("[data-diet] ALL PASS")
