"""Foundation SHADOW smoke (2026-10-03): record live steps, change nothing.

WHAT IS CLAIMED (plan §8 step 2 "Shadow"; Stage 2 gate "existing SkyBot data
can travel through the interfaces"; Stage 5 "save predictions before outcomes
arrive, including source model versions")

    foundation/runtime/shadow.py's ShadowRecorder, wired into the two LIVE
    collection bodies, files every sampled step as foundation records —
    the sensor transport as Observations (RED oracle readings only in the
    evaluator partition), the executed Action with its real dispatch and
    completion times, the world model's one-step Prediction written BEFORE
    env.step, the Evidence that links them, and how each episode ended —
    WITHOUT moving the active learner by a single bit.

WHY EACH PART EXISTS
    The project's signature failure is a correct mechanism nobody calls
    (CLAUDE.md §4.2: three stepping bodies, edits that land in one). The
    second is a "harmless" observer that is not harmless: plan §8 says "do
    not allow background experiments to silently alter the active learner's
    replay, normalization, or memory". A shadow that draws one Gumbel sample
    shifts every later action. So this file does not trust the recorder's
    docstring; it runs the REAL bodies twice and diffs the learner.

CONTRACTS
    A. Off costs nothing: from_config returns None for an empty config and
       for enabled: false. The LIVE config ships it ON (2026-10-03), so for
       it the contract is BOUNDED: disk cap <= 2 GiB, every_n_steps >= 8,
       amortised budget <= 10 ms/step;
       the loop guards every call with `if self._shadow is not None:`.
       A config typo under foundation.shadow is refused (None + warning),
       not silently defaulted.
    B. Source wiring (§4.2): before_step and after_step are each called with
       IDENTICAL text exactly twice — once in _run_episode_parallel, once in
       _collect_segment — and zero times in _run_episode. In each body the
       order is before_step < env dispatch < replay add < PPO store <
       after_step < the advance/reset, i.e. the prediction precedes the
       outcome and the record sees FINAL reward values.
    C. NON-INTERFERENCE A/B on the real bodies (fake sensed CartPole, 2
       streams; lifelong _collect_segment with WM training + a PPO update,
       and episodic _run_episode_parallel): shadow OFF vs ON (every step
       sampled, predictions on) gives byte-identical actions, rewards,
       replay buffer arrays, PPO rollout, world-model / policy / ICM
       weights, returned metrics, and torch / numpy / python RNG states.
       (Excluded: the replay streams' lock_held_s / lock_wait_s wall-clock
       timers, which differ between two shadow-OFF runs as well.)
       Falsification: a recorder patched to draw ONE torch random number per
       step changes the fingerprints (so "identical" is a sensitive test),
       and a sampling imagine_step — RSSMPredictor's path — does advance the
       torch RNG while the shadow prior does not.
    D. The records: per-stream scopes, never mixed; RED only in the
       evaluator partition (learning view refuses it); every prediction was
       written before its outcome and reconstructs aligned; a reconstructed
       interaction's action and reward equal what the env actually received
       and returned at that step; episodes end as terminated (CartPole) and
       as client_recovery (driven directly) with the prediction left
       unresolved; the shadow prior equals RSSMPredictor's horizon-1 probs.
    E. Self-disable: a wall-time budget overrun and an exception inside the
       recorder each produce exactly ONE warning, disable the recorder, and
       the loop keeps collecting. An unstartable store returns None at
       construction (a shadow cannot stop the learner from booting).
    F. Overhead is measured and printed (ms/step off vs on).

Run: PYTHONPATH=. python tests/_foundation_shadow_smoke.py   (needs gymnasium)
"""
import collections
import hashlib
import importlib.util
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, ".")

import numpy as np
import yaml

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
SKY = yaml.safe_load(open("configs/minecraft_skybot.yaml"))
TMP = tempfile.mkdtemp(prefix="shadow_smoke_")
SHADOW_LOGGER = "developmental_ai.foundation.runtime.shadow"

BEFORE = ("self._shadow.before_step(rssm_state, env_actions,\n"
          "                                         self._wm_param_lock, "
          "self.world_model)")
AFTER = ("self._shadow.after_step(\n"
         "                    step_infos, env_actions, rewards, dones, "
         "restarted,\n"
         "                    _sh_ends, _t_env0, prim_extrinsic, intrinsic,\n"
         "                    _sh_fleet_reset, use_dream_actor)")


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _capture():
    h = _Warnings()
    logging.getLogger(SHADOW_LOGGER).addHandler(h)
    return h


def _release(h):
    logging.getLogger(SHADOW_LOGGER).removeHandler(h)


# ------------------------------------------------------------ A. disabled
def test_disabled():
    from developmental_ai.foundation.runtime import ShadowRecorder
    # The live config ships the shadow ON (user decision, 2026-10-03) — so
    # what must hold there is that it is BOUNDED: capped disk on a 256 GB
    # single NVMe, sampled not every-step, and an amortised time budget that
    # self-disables it before it can eat the step rate (CLAUDE.md §6: the
    # fovea at interval 12 once cost 28% of it).
    _sh = (SKY.get("foundation") or {}).get("shadow", {})
    assert _sh.get("enabled") is True, _sh
    assert 0 < int(_sh["max_bytes"]) <= 2 * (1 << 30), _sh
    assert int(_sh["every_n_steps"]) >= 8, _sh
    assert 0 < float(_sh["max_ms_per_step"]) <= 10.0, _sh
    assert ShadowRecorder.from_config({}) is None
    assert ShadowRecorder.from_config(
        {"foundation": {"shadow": {"enabled": False}}}) is None
    h = _capture()
    try:
        r = ShadowRecorder.from_config(
            {"foundation": {"shadow": {"enabled": True, "evry_n": 3}}})
    finally:
        _release(h)
    assert r is None and len(h.msgs) == 1 and "evry_n" in h.msgs[0], h.msgs
    src = open(LOOP).read()
    assert src.count("            if self._shadow is not None:\n") == 4
    assert "self._shadow = ShadowRecorder.from_config(" in src
    print("  A. live config ON and bounded; empty / false -> None; a typo "
          "key is refused with one warning; all 4 call sites guarded")


# --------------------------------------------------------------- B. source
def _body(src, name):
    i = src.index(f"    def {name}(")
    j = src.index("\n    def ", i + 10)
    return src[i:j]


def test_source():
    src = open(LOOP).read()
    assert src.count(BEFORE) == 2, src.count(BEFORE)
    assert src.count(AFTER) == 2, src.count(AFTER)
    tt = "_sh_ends.append((bool(term), bool(trunc), time.time()))"
    assert src.count(tt) == 2
    legacy = _body(src, "_run_episode")
    assert "_shadow" not in legacy and "_sh_" not in legacy, \
        "_run_episode must stay unwired (select_collection_path guards it)"
    for name in ("_run_episode_parallel", "_collect_segment"):
        b = _body(src, name)
        assert b.count(BEFORE) == 1 and b.count(AFTER) == 1, name
        order = [b.index(BEFORE),
                 b.index("futs = [self._env_pool.submit(envs[e_i].step"),
                 b.index(tt),
                 b.index("self.replay_buffer.add("),
                 b.rindex("self.policy.store_transition("),
                 b.index(AFTER),
                 b.index("obs_list[e_i] = next_obs_list[e_i]")]
        assert order == sorted(order), (name, order)
    print("  B. before_step x2 / after_step x2 (identical text), 0 in "
          "_run_episode; per body: predict < dispatch < replay < PPO store "
          "< record < advance")


# ------------------------------------------------------ the fake sensed env
ENV_LOG = {}   # env index -> list of (reset_no, step_no, action, reward)


def _sensed_cartpole(idx):
    import gymnasium as gym

    class SensedCartPole(gym.Wrapper):
        """CartPole with a body: info["proprio"], the bus transport in
        info["sensors"] (proprio[4] + screen_fx[6]) and a RED
        info["oracle"]["true_position"]. Resets are seeded by a counter, so two
        runs see identical worlds."""

        PROPRIO_DIM = 4
        PROPRIO_KEYS = ("x", "x_dot", "theta", "theta_dot")

        def __init__(self, idx):
            super().__init__(gym.make("CartPole-v1", max_episode_steps=200))
            self.idx = idx
            self.resets = -1
            self.t = 0
            ENV_LOG[idx] = []

        def _aug(self, o, info):
            info = dict(info)
            p = np.asarray(o, np.float32)
            fx = np.tanh(np.array([p[0], p[1], p[2], p[3], p[0] * p[2], 0.5],
                                  np.float32))
            info["proprio"] = p
            info["sensors"] = np.concatenate([p, fx]).astype(np.float32)
            info["oracle"] = {"true_position": np.array([p[0], 0.0, p[2]],
                                                        np.float32)}
            return info

        def reset(self, **kw):
            self.resets += 1
            self.t = 0
            kw["seed"] = 1000 * self.idx + self.resets
            o, i = self.env.reset(**kw)
            return o, self._aug(o, i)

        def step(self, a):
            o, r, te, tr, i = self.env.step(a)
            ENV_LOG[self.idx].append((self.resets, self.t, int(a), float(r)))
            self.t += 1
            return o, r, te, tr, self._aug(o, i)

    return SensedCartPole(idx)


def _patch_make_env():
    import developmental_ai.core.developmental_loop as dl
    from developmental_ai.environments.wrappers import DevelopmentalEnvWrapper
    state = {"n": 0}

    def fake(*a, **k):
        idx = state["n"]
        state["n"] += 1
        return DevelopmentalEnvWrapper(_sensed_cartpole(idx),
                                       normalize_obs=False), None
    dl.make_env = fake
    return dl


def _cfg(lifelong, shadow=None):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 32, "encoder_hidden": 32,
        "decoder_hidden": 32, "batch_size": 4, "sequence_length": 8,
        "train_iters": 1, "buffer_capacity": 2000, "proprio": True})
    cfg["sensors"] = {"enabled": ["proprio", "screen_fx", "true_position"]}
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["policy"].update({"n_steps": 40, "minibatch_size": 16,
                          "n_epochs": 2})
    if lifelong:
        cfg["lifelong"] = {"enabled": True, "forever": False,
                           "segment_len": 40, "wm_train_every": 25,
                           "goal_horizon": 10 ** 9, "state_decay": 1.0,
                           "reset_on_death_only": True,
                           "stop_file": os.path.join(TMP, "STOP")}
    else:
        cfg["environment"]["max_episode_steps"] = 60
    cfg.setdefault("loop", {})["verbose"] = 0
    cfg.setdefault("llm", {})["enabled"] = False
    cfg.setdefault("skill_bank", {})["storage_dir"] = os.path.join(TMP, "sb")
    if shadow is not None:
        cfg["foundation"] = {"shadow": shadow}
    return cfg


def _shadow_cfg(root, **kw):
    d = {"enabled": True, "root": root, "max_bytes": 1 << 26,
         "chunk_bytes": 1 << 16, "every_n_steps": 1,
         "max_ms_per_step": 1e6, "max_ms_single_step": 1e7}
    d.update(kw)
    return d


def _h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode() + str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()


# Wall-clock instrumentation, not learner state: these differ between two
# shadow-OFF runs too (measured: replay stream lock timers), so comparing
# them would make contract C fail for a reason that is not the shadow.
WALLCLOCK = {"lock_held_s", "lock_wait_s"}


def _arrays_of(obj, depth=0, seen=None):
    """Every ndarray / tensor reachable through an object's attributes and
    lists (the replay buffer's storage), in a stable order."""
    import torch
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return []
    seen.add(id(obj))
    out = []
    if isinstance(obj, (bool, int, float, np.generic)):
        return [np.asarray(obj)]
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, torch.Tensor):
        return [obj.detach().cpu().numpy()]
    if isinstance(obj, torch.nn.Module):
        out = [v.detach().cpu().numpy() for v in obj.state_dict().values()]
    if isinstance(obj, (list, tuple, collections.deque)):
        for x in obj:
            out += _arrays_of(x, depth + 1, seen)
        return out
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            out += _arrays_of(obj[k], depth + 1, seen)
        return out
    if hasattr(obj, "__dict__") and type(obj).__module__.startswith(
            "developmental_ai"):
        for k in sorted(vars(obj)):
            if k.startswith("_lock") or "thread" in k or k in WALLCLOCK:
                continue
            out += _arrays_of(vars(obj)[k], depth + 1, seen)
    return out


def _fingerprint(ai, metrics):
    import torch
    pol = ai.policy
    fp = {
        "env_log": repr({k: list(v) for k, v in ENV_LOG.items()}),
        "metrics": repr(metrics),
        "replay": _h(*_arrays_of(ai.replay_buffer)),
        "rollout": repr([np.asarray(x).tolist() for x in pol.rollout_rewards])
        + _h(*[np.asarray(x) for x in pol.rollout_obs]) + repr(
            [np.asarray(a).tolist() for a in pol.rollout_actions]),
        "wm": _h(*[v.cpu().numpy() for v in
                   ai.world_model.state_dict().values()]),
        "policy": _h(*_arrays_of(pol)),
        "curiosity": _h(*_arrays_of(ai.curiosity)),
        "reward_mixer": _h(*_arrays_of(ai.reward_mixer)),
        "torch_rng": _h(torch.get_rng_state().numpy()),
        "numpy_rng": repr(np.random.get_state()[1][:8].tolist())
        + str(np.random.get_state()[2]),
        "python_rng": repr(random.getstate()[1][:8]),
    }
    return fp


def _run(lifelong, shadow, segments=3):
    dl = _patch_make_env()
    ENV_LOG.clear()
    ai = dl.DevelopmentalAI(config=_cfg(lifelong, shadow))
    try:
        metrics = []
        t0 = time.perf_counter()
        steps0 = ai.total_timesteps
        for _ in range(segments):
            if lifelong:
                metrics.append(ai._collect_segment())
            else:
                metrics.append(ai._run_episode_parallel(
                    use_dream_actor=False))
        dt = time.perf_counter() - t0
        nsteps = (ai.total_timesteps - steps0) / max(1, ai._num_envs)
        fp = _fingerprint(ai, metrics)
        sh = ai._shadow
        return ai, fp, dt * 1000.0 / max(1, nsteps), sh
    finally:
        ai.close()


TIMING = {}


def test_non_interference():
    import torch
    from developmental_ai.foundation.runtime import (ShadowRecorder,
                                                     rssm_prior_probs)
    offs = {}
    for lifelong, label in ((True, "lifelong _collect_segment"),
                            (False, "episodic _run_episode_parallel")):
        root = os.path.join(TMP, f"store_{int(lifelong)}")
        _, off, ms_off, sh0 = _run(lifelong, None)
        assert sh0 is None
        _, on, ms_on, sh = _run(lifelong, _shadow_cfg(root))
        st = sh.stats()
        assert sh.enabled and st["counts"].get("predictions", 0) > 20, st
        assert st["counts"].get("outcomes", 0) > 20, st
        offs[lifelong] = off
        diff = [k for k in off if off[k] != on[k]]
        assert not diff, f"{label}: shadow ON changed {diff}"
        TIMING[label] = (ms_off, ms_on, st["ms_per_step_mean"])
        print(f"  C. {label}: {len(off)} learner fingerprints identical "
              f"OFF vs ON ({st['counts']['predictions']} predictions, "
              f"{st['counts']['outcomes']} outcomes recorded)")
    # Falsification: is the A/B sensitive at all? A recorder that draws ONE
    # random number per step (what a sampling imagine_step would do) must
    # show up in the fingerprints, or "identical" above proves nothing.
    orig = ShadowRecorder.before_step

    def leaky(self, *a, **k):
        orig(self, *a, **k)
        torch.rand(1)
    ShadowRecorder.before_step = leaky
    try:
        _, bad, _, _ = _run(True, _shadow_cfg(os.path.join(TMP, "leaky")))
    finally:
        ShadowRecorder.before_step = orig
    moved = [k for k in bad if bad[k] != offs[True][k]]
    assert "torch_rng" in moved and len(moved) >= 2, moved
    print(f"     falsifier: a recorder drawing 1 torch random/step changes "
          f"{moved} — the A/B detects interference")
    # Falsification witness: the sampling path WOULD have moved the RNG.
    from developmental_ai.world_model.rssm import WorldModel
    wm = WorldModel(obs_dim=4, action_dim=2, stochastic_size=8,
                    stochastic_classes=8, deterministic_size=32,
                    hidden_dim=32)
    s = wm.rssm.initial_state(2, torch.device("cpu"))
    a = np.eye(2, dtype=np.float32)
    g0 = torch.get_rng_state().clone()
    rssm_prior_probs(wm.rssm, s["h"], s["z"], a)
    assert torch.equal(g0, torch.get_rng_state())
    wm.rssm.train()
    with torch.no_grad():
        wm.rssm.imagine_step(s, torch.from_numpy(a))
    assert not torch.equal(g0, torch.get_rng_state()), \
        "witness failed: imagine_step drew nothing, so contract C proves less"
    print("     witness: shadow prior leaves the torch RNG untouched; a "
          "sampling imagine_step advances it (so C is not vacuous)")


# --------------------------------------------------------------- D. records
def test_records():
    from developmental_ai.foundation.adapters import RSSMPredictor
    from developmental_ai.foundation.contracts import ActionSpec, Scope
    from developmental_ai.foundation.experience import (
        EVALUATOR, EvidenceStore, PrivacyError)
    from developmental_ai.foundation.runtime import rssm_prior_probs
    import torch

    root = os.path.join(TMP, "store_1")
    # ENV_LOG holds the LAST run of contract C (episodic): rerun lifelong ON
    # so the env's own log and the store describe the same run.
    shutil.rmtree(root, ignore_errors=True)
    _, _, _, sh = _run(True, _shadow_cfg(root))
    log = {k: list(v) for k, v in ENV_LOG.items()}
    store = EvidenceStore(root, readonly=True)
    ev = store.evaluator_view()
    lv = store.learning_view()

    outs = ev.refs("evidence")
    sensor_outs = [r for r in outs if ev.get(r).provenance == "sensor"]
    red_outs = [r for r in outs if ev.get(r).provenance == "evaluator"]
    assert sensor_outs and red_outs
    streams = {ev.get(r).scope.stream for r in sensor_outs}
    assert streams == {"stream-0", "stream-1"}, streams

    # RED: only in the evaluator partition, and unreachable from learning
    for r in ev.refs("observation"):
        o = ev.get(r)
        red = o.channel == "true_position"
        assert red == (o.provenance == "evaluator"), (r, o.channel)
        assert red == r.startswith(EVALUATOR + "/"), r
        if red:
            try:
                lv.get(r)
                raise AssertionError("learning view returned RED data")
            except PrivacyError:
                pass
    assert not [r for r in lv.refs("observation")
                if lv.get(r).provenance == "evaluator"]

    # predictions before outcomes; reconstruction; ground truth
    n_checked = 0
    for r in sensor_outs:
        it = ev.reconstruct(r)
        assert it.aligned, (r, it.problems)
        assert len(it.predictions) == 1
        tr = it.predictions[0]
        assert tr.write_seq < it.write_seq
        assert tr.prediction.outcome["probs"].shape == (1, 8, 8)
        assert tr.snapshot_id.startswith(sh.run_id)
        assert "world_model.rssm" in tr.model_versions
        assert {o.channel for o in tr.context_observations} == {
            "proprio", "screen_fx", "true_position"}
        a = it.evidence.actions[0]
        sc = it.evidence.scope
        e_idx = int(sc.stream.split("-")[1])
        ep_no = int(sc.episode.rsplit("-e", 1)[1])
        hits = [x for x in log[e_idx] if x[0] == ep_no and x[1] == a.seq]
        assert len(hits) == 1, (sc, a.seq)
        assert hits[0][2] == a.command, (hits[0], a.command)
        assert hits[0][3] == it.evidence.payload["reward_env"]
        assert tr.prediction.candidate_action == (a.command,)
        assert a.t_complete >= a.t_dispatch
        n_checked += 1
    ends = [ev.meta(r)["reason"] for r in ev.refs("episode_end")]
    assert "terminated" in ends, ends
    print(f"  D. {n_checked} interactions reconstruct aligned (prediction "
          f"written first, action+reward == what the env saw); streams "
          f"{sorted(streams)} kept apart; RED only in evaluator/ "
          f"({len(red_outs)} evaluator outcomes); endings {sorted(set(ends))}")

    # client_recovery, driven directly; prediction left unresolved
    from developmental_ai.foundation.runtime import ShadowRecorder
    from developmental_ai.world_model.rssm import WorldModel
    wm = WorldModel(obs_dim=4, action_dim=2, stochastic_size=8,
                    stochastic_classes=8, deterministic_size=32,
                    hidden_dim=32)
    r2 = os.path.join(TMP, "store_direct")
    rec = ShadowRecorder.from_config(
        {"foundation": {"shadow": _shadow_cfg(r2)}}, bus=None, action_dim=2,
        is_discrete=True, num_streams=1, environment="fake")
    s = wm.rssm.initial_state(1, torch.device("cpu"))
    info = {"proprio": np.ones(3, np.float32)}
    for k, rst in enumerate([False, False, True, False, False]):
        rec.before_step(s, [k % 2], None, wm)
        t = time.time()                      # dispatch, then the result
        rec.after_step([None if rst else info], [k % 2], [1.0], [rst],
                       [rst], [(False, rst, time.time())], t)
    st = rec.stats()["counts"]
    assert st["end_client_recovery"] == 1 and st["unresolved_predictions"] == 1
    rec.close()
    s2 = EvidenceStore(r2, readonly=True).evaluator_view()
    end = [s2.meta(r) for r in s2.refs("episode_end")]
    assert [e["reason"] for e in end] == ["client_recovery"]
    preds = s2.refs("prediction")
    resolved = {p for r in s2.refs("evidence")
                for p in s2.get(r).prediction_refs}
    assert len(preds) - len(resolved) == 1
    # the shadow prior is RSSMPredictor's horizon-1 prior (no second model)
    wm.rssm.train()
    sp = ActionSpec.discrete("cp", 2)
    p = RSSMPredictor(wm.rssm, sp, "v", "s")
    ss = {"h": torch.randn(1, 32), "z": torch.zeros(1, 64)}
    ref = p.predict(ss, "b", [1], "pid", seed=0).outcome["probs"][0]
    mine = rssm_prior_probs(wm.rssm, ss["h"], ss["z"],
                            np.array([[0, 1]], np.float32))[0]
    assert np.allclose(ref, mine, atol=1e-6), np.abs(ref - mine).max()
    print("  D. client_recovery ends the episode and leaves its prediction "
          "unresolved; shadow prior == RSSMPredictor horizon-1 probs")


# ------------------------------------------------------------ E. disabling
def test_self_disable():
    from developmental_ai.foundation.runtime import ShadowRecorder
    h = _capture()
    try:
        ai, _, _, sh = _run(True, _shadow_cfg(
            os.path.join(TMP, "store_budget"), max_ms_per_step=1e-9,
            budget_window=3), segments=2)
    finally:
        _release(h)
    # DEGRADE, THEN DIE (2026-10-04): window 1 over budget -> predictions
    # off (warning 1); window 2 still over -> recorder off (warning 2).
    assert not sh.enabled and "max_ms_per_step" in sh.disabled_reason
    assert "predictions already off" in sh.disabled_reason
    assert len(h.msgs) == 2 and "predictions OFF" in h.msgs[0], h.msgs
    assert sh.n["steps"] == 6, sh.n          # two windows of 3
    assert ai.total_timesteps == 2 * 40 * 2  # the loop kept collecting
    print(f"  E. budget overrun -> predictions off after 3 steps, recorder "
          f"off after {sh.n['steps']} (2 warnings); loop finished all "
          f"{ai.total_timesteps} transitions")

    # an exception inside the recorder
    dl = _patch_make_env()
    h = _capture()
    ai = dl.DevelopmentalAI(config=_cfg(True, _shadow_cfg(
        os.path.join(TMP, "store_raise"))))
    try:
        def boom(*a, **k):
            raise OSError("disk full (injected)")
        ai._shadow.store.log_action = boom
        ai._collect_segment()
        ai._collect_segment()
        assert not ai._shadow.enabled
        assert "disk full" in ai._shadow.disabled_reason
        assert len(h.msgs) == 1, h.msgs
        assert ai.total_timesteps == 2 * 40 * 2
    finally:
        _release(h)
        ai.close()
    # unstartable store -> None at construction, one warning
    blocker = os.path.join(TMP, "a_file")
    open(blocker, "w").close()
    h = _capture()
    try:
        r = ShadowRecorder.from_config({"foundation": {"shadow": _shadow_cfg(
            os.path.join(blocker, "sub"))}})
    finally:
        _release(h)
    assert r is None and len(h.msgs) == 1, h.msgs
    print("  E. exception inside the recorder -> 1 warning, disabled, loop "
          "continues; unstartable store -> None (learner still boots)")


# ------------------------------------------------------------ F. overhead
def test_overhead():
    for label, (off, on, own) in TIMING.items():
        print(f"  F. {label}: {off:.2f} ms/step OFF, {on:.2f} ms/step ON "
              f"(every step sampled + predicted; recorder's own "
              f"{own:.3f} ms/step)")
    _, _, ms16, sh = _run(True, _shadow_cfg(os.path.join(TMP, "store_16"),
                                            every_n_steps=16))
    print(f"  F. lifelong at every_n_steps=16 (shipped default): "
          f"{ms16:.2f} ms/step loop, recorder "
          f"{sh.stats()['ms_per_step_mean']:.3f} ms/step mean "
          f"(p95 {sh.stats()['ms_per_step_p95']:.3f})")
    _live_shaped_cost()


def _live_shaped_cost():
    """CartPole's transport is 10 floats; SkyBot's is ~4K (two fovea
    images) and its RSSM prior is 48x48. Drive the recorder directly with
    the LIVE bus layout and the live latent sizes (CPU here; the host's GPU
    forward is faster, its disk similar) so the number printed is the one
    that matters against max_ms_per_step."""
    import torch
    from developmental_ai.foundation.runtime import ShadowRecorder
    from developmental_ai.sensors import build_default_bus
    from developmental_ai.world_model.rssm import WorldModel
    sen, wmc = SKY["sensors"], SKY["world_model"]
    bus = build_default_bus(13, lambda ctx: None,
                            enabled=list(sen["enabled"]),
                            fovea_size=int(sen.get("fovea_size", 32)))
    P, n = 24, 2
    S, C, D = (int(wmc["stochastic_size"]), int(wmc["stochastic_classes"]),
               int(wmc["deterministic_size"]))
    wm = WorldModel(obs_dim=16, action_dim=P, stochastic_size=S,
                    stochastic_classes=C, deterministic_size=D,
                    hidden_dim=64)
    rng = np.random.default_rng(0)                 # local: not the global RNG
    res = {}
    for every in (1, 16):
        rec = ShadowRecorder.from_config(
            {"foundation": {"shadow": _shadow_cfg(
                os.path.join(TMP, f"live_{every}"), every_n_steps=every)}},
            bus=bus, action_dim=P, is_discrete=True, num_streams=n,
            environment="MineRL", layout_hash=bus.layout_hash())
        st = {"h": torch.zeros(n, D), "z": torch.zeros(n, S * C)}
        for k in range(64):
            infos = [{"sensors": rng.random(bus.width, dtype=np.float32),
                      "oracle": {"true_position": rng.random(3)}}
                     for _ in range(n)]
            acts = [int(x) for x in rng.integers(0, P, n)]
            rec.before_step(st, acts, None, wm)
            t = time.time()
            rec.after_step(infos, acts, [0.0] * n, [False] * n, [False] * n,
                           [(False, False, time.time())] * n, t)
        assert rec.enabled, rec.disabled_reason
        res[every] = rec.stats()
        rec.close()
    per_sampled = res[1]["ms_per_step_mean"]
    print(f"  F. LIVE-SHAPED (transport {bus.width} floats, prior {S}x{C}, "
          f"h{D}, 2 streams, CPU): {per_sampled:.2f} ms per sampled step; "
          f"at every_n_steps=16 {res[16]['ms_per_step_mean']:.3f} ms/step "
          f"amortised (p95 {res[16]['ms_per_step_p95']:.2f}) vs budget "
          f"{SKY['foundation']['shadow']['max_ms_per_step']} ms; store "
          f"{res[16]['store_bytes'] / 64:.0f} B/step")


def main():
    logging.basicConfig(level=logging.ERROR)
    # the root is quiet; the shadow's own warnings must still reach _capture
    logging.getLogger(SHADOW_LOGGER).setLevel(logging.WARNING)
    logging.getLogger(SHADOW_LOGGER).propagate = False   # E counts them
    try:
        test_disabled()
        test_source()
        test_busy_lock()
        if importlib.util.find_spec("gymnasium") is None:
            print("  C-F. SKIPPED: gymnasium absent here (they drive the real "
                  "loop; CI and the host run them). A-B are the local "
                  "evidence.")
            print("[foundation_shadow] ALL PASS")
            return
        test_non_interference()
        test_records()
        test_self_disable()
        test_overhead()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("[foundation_shadow] ALL PASS")


# ------------------------------------------------- G. busy lock never blocks
def test_busy_lock():
    """LIVE INCIDENT 2026-10-04: the first live run disabled the shadow after
    one minute (5.127 ms/step > 5.0) — the moment the async WM trainer
    started. `_predict` did `with lock:` on the world model's param lock,
    which the trainer holds while it updates, so the ACTING thread waited on
    every sampled step. A held lock must cost at most the bounded timeout,
    skip only the prediction, and leave obs/action/outcome recorded."""
    import threading
    import torch
    from developmental_ai.foundation.runtime import ShadowRecorder
    from developmental_ai.world_model.rssm import WorldModel
    wm = WorldModel(obs_dim=4, action_dim=2, stochastic_size=8,
                    stochastic_classes=8, deterministic_size=32,
                    hidden_dim=32)
    rec = ShadowRecorder.from_config(
        {"foundation": {"shadow": _shadow_cfg(
            os.path.join(TMP, "store_lock"), predict_lock_timeout_ms=2.0)}},
        bus=None, action_dim=2, is_discrete=True, num_streams=1,
        environment="fake")
    lock = threading.RLock()
    held, release = threading.Event(), threading.Event()

    def trainer():                      # holds the lock like the WM trainer
        with lock:
            held.set()
            release.wait(10)
    th = threading.Thread(target=trainer, daemon=True)
    s = wm.rssm.initial_state(1, torch.device("cpu"))
    info = {"proprio": np.ones(3, np.float32)}
    rec.before_step(s, [0], lock, wm)            # no context yet (step 0)
    rec.after_step([info], [0], [1.0], [False], [False],
                   [(False, False, time.time())], time.time())
    th.start()
    held.wait(5)
    try:
        worst = 0.0
        for k in range(5):
            t0 = time.perf_counter()
            rec.before_step(s, [k % 2], lock, wm)
            worst = max(worst, time.perf_counter() - t0)
            rec.after_step([info], [k % 2], [1.0], [False], [False],
                           [(False, False, time.time())], time.time())
    finally:
        release.set()
        th.join(5)
    n = rec.stats()["counts"]
    assert rec.enabled, rec.disabled_reason
    assert n["skipped_lock_busy"] == 5 and n.get("predictions", 0) == 0, n
    assert n.get("outcomes", n.get("evidence", 1)) >= 1, n
    assert worst < 0.25, f"before_step blocked {worst*1e3:.0f} ms on a held lock"
    # lock free again -> predictions resume
    rec.before_step(s, [1], lock, wm)
    assert rec.stats()["counts"]["predictions"] == 1, rec.stats()["counts"]
    rec.close()
    print(f"  G. trainer holding the WM lock: 5/5 predictions skipped, worst "
          f"before_step {worst*1e3:.1f} ms (timeout 2 ms), recording kept; "
          f"lock free -> predictions resume")


if __name__ == "__main__":
    main()
