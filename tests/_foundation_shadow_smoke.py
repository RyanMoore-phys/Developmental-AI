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
       outcome and the record sees FINAL reward values. The observable
       path is handed the HOST observation lists (obs_list /
       next_obs_list), never the device tensors obs_t / next_obs_t: a
       device tensor forces a GPU->host copy on the acting thread
       (counted as observable_device_copies; 2026-10-05).
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
    E. Self-disable: a wall-time budget overrun DEGRADES IN STAGES — one
       warning per stage, a fresh window each: observable forecasts off,
       then categorical predictions off, then the recorder disabled
       (2026-10-05; three windows) — and an exception inside the core
       recorder produces exactly ONE warning and disables it; the loop
       keeps collecting either way. An unstartable store returns None at
       construction (a shadow cannot stop the learner from booting).
    F. Overhead is measured and printed (ms/step off vs on).
    G. A held WM lock never blocks the acting thread.
    H. ACTION SOURCE (2026-10-04). Each sampled Action's payload says WHO
       chose the command: "policy", "option_slot:<n>" (+ "skill" = the
       skill_id of the frame that emitted the primitive, "option_depth",
       "option_root_skill" when nested) or "option_fallback". Driven through
       the REAL OptionExecutor.act() (stub bank + scripted policy): a
       primitive pick reads policy, an option pick reads its slot + skill, a
       nested conv skill reads the CHILD as the acting skill and the root
       separately, and an option whose slot is unbound mid-flight reads
       option_fallback. action_sources is a pure read (executor state
       pickles byte-identical before/after; act() returns the same actions
       with and without it). Both bodies pass executor=self.option_executor
       (B); in the real-body run (no executor) every recorded action is
       "policy" (D). Optional field: stores without it still load
       (_foundation_shadow_report_smoke B, "unrecorded").
    I. PIXEL OBSERVABLE FORECAST (2026-10-05). Contract C only ever drove a
       vector (CartPole) model, so the pixel path — decoder forward, 8x8
       pooling, scoring — had never run. A tiny pixel world model (3x16x16,
       learner = posterior observe_step [draws torch RNG] + a real optimizer
       step under the WM lock, numpy obs exactly as the loop holds them),
       shadow OFF vs ON (observable on, every step): byte-identical weights,
       optimizer state, losses, latent state, obs and torch/numpy RNG. The
       ON run files a forecast for every prediction and scores every
       outcome; the forecast equals an INDEPENDENT second transition +
       decode (so reusing the categorical prediction's h1/logits changed
       nothing), the numpy pooling equals torch's adaptive_avg_pool2d, and
       the scored target / persistence MSE recompute from the raw frames.
       No device copies (numpy obs). Falsifier: a recorder drawing one torch
       random per step moves the fingerprints.
    J. OBSERVABLE ISOLATION + STAGES (2026-10-05). A decoder that raises
       never reaches the recorder-wide disable: categorical predictions keep
       being filed, `observable_max_failures` consecutive failures turn ONLY
       the observable forecast off (one warning), records then carry no
       "observable" key. A malformed outcome observation fails at the score
       site the same way. Driven directly over budget, the stages fire in
       order full -> categorical -> records -> disabled with exactly three
       warnings. observable_predict off: no "observable" key in prediction
       outcomes, no observable_score in evidence (old-store shape).

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
          "self.world_model,\n"
          "                                         "
          "executor=self.option_executor, observations=obs_list)")
AFTER = ("self._shadow.after_step(\n"
         "                    step_infos, env_actions, rewards, dones, "
         "restarted,\n"
         "                    _sh_ends, _t_env0, prim_extrinsic, intrinsic,\n"
         "                    _sh_fleet_reset, use_dream_actor, observations=next_obs_list)")


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
    # The config echo must see these keys as READ (live 2026-10-04: all ten
    # foundation.shadow.* keys were listed "never read" while the recorder
    # ran on them — dict.update bypassed TrackedConfig.__getitem__).
    from developmental_ai.infra.config_echo import TrackedConfig
    from developmental_ai.foundation.runtime.shadow import shadow_config
    _tc = TrackedConfig(SKY)
    shadow_config(_tc)
    _read = _tc.accessed_paths
    _miss = [k for k in _sh if f"foundation.shadow.{k}" not in _read]
    assert not _miss, f"config echo would report these as never read: {_miss}"
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
    # Falsifier: the device-tensor form must be gone from BOTH bodies.
    assert "observations=obs_t)" not in src, "device obs to shadow"
    assert "observations=next_obs_t)" not in src, "device next obs to shadow"
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
          "< record < advance; host obs lists, not device tensors")


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
         "chunk_bytes": 1 << 16, "every_n_steps": 1, "observable_predict": True,
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
        assert a.payload.get("source") == "policy", a.payload   # H: no executor
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
    # DEGRADE IN STAGES, THEN DIE (2026-10-05): window 1 over budget ->
    # observable forecasts off (warning 1); window 2 -> categorical
    # predictions off (warning 2); window 3 still over -> recorder off (3).
    assert not sh.enabled and "max_ms_per_step" in sh.disabled_reason
    assert "predictions already off" in sh.disabled_reason
    assert len(h.msgs) == 3, h.msgs
    assert "observable forecasts OFF" in h.msgs[0], h.msgs
    assert "predictions OFF" in h.msgs[1], h.msgs
    assert "DISABLED" in h.msgs[2], h.msgs
    assert [d["stage"] for d in sh.degraded] == ["categorical", "records"]
    assert sh.stage() == "disabled"
    assert sh.n["steps"] == 9, sh.n          # three windows of 3
    assert sh.n.get("observable_errors", 0) == 0, sh.n
    assert ai.total_timesteps == 2 * 40 * 2  # the loop kept collecting
    print(f"  E. budget overrun -> observable off after 3 steps, "
          f"predictions off after 6, recorder off after {sh.n['steps']} "
          f"(3 warnings, one per stage); loop finished all "
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
        test_action_source()
        test_pixel_observable()
        test_observable_isolation()
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


# --------------------------------------------- H. who chose the action
P_H = 4


class _BankH:
    """The surface OptionExecutor.act() touches: P primitives + 2 slots;
    sk_b is a conv skill whose head row 0 invokes sk_a (nesting)."""
    K = 2
    scripted = {}
    disabled_macros = None

    def __init__(self):
        self.P = P_H
        self.slots = [{"skill_id": "sk_a", "arch": "mlp"},
                      {"skill_id": "sk_b", "arch": "conv",
                       "head_dim": P_H + 2, "p_own": P_H,
                       "slot_map": {"0": "sk_a"}}]
        self.skill_bank = _SkillBankH()

    def mask(self, *a, **k):
        return np.ones(P_H + self.K, bool)

    def note_invoked(self, slot):
        pass

    def is_scripted(self, slot):
        return slot in self.scripted

    def slot_of_skill(self, sid):
        return {"sk_a": 0, "sk_b": 1}.get(sid)

    def skill_action(self, slot, obs, head_mask=None, **k):
        return 2 if slot == 0 else P_H + 0      # sk_b always invokes sk_a


class _SkillBankH:
    def record_invokes(self, *a):
        pass


class _PolicyH:
    arch = ""

    def __init__(self, seq):
        self.seq = list(seq)

    def select_action(self, obs, **k):
        return self.seq.pop(0), {"log_prob": 0.0, "value": 0.0}


def test_action_source():
    import pickle
    import torch
    from developmental_ai.policy.options import OptionExecutor
    from developmental_ai.foundation.experience import EvidenceStore
    from developmental_ai.foundation.runtime import ShadowRecorder
    from developmental_ai.foundation.runtime.shadow import action_sources
    from developmental_ai.world_model.rssm import WorldModel
    obs = [np.zeros(3, np.float32)] * 2
    cases = (
        ([1, 1], {"source": "policy"}),
        ([P_H + 0, 1], {"source": "option_slot:0", "skill": "sk_a",
                        "option_depth": 1}),
        ([P_H + 1, 1], {"source": "option_slot:1", "skill": "sk_a",
                        "option_depth": 2, "option_root_skill": "sk_b"}))
    for seq, want in cases:
        ex = OptionExecutor(_BankH(), 2, {})
        a = ex.act(obs, None, _PolicyH(seq), 0)
        st0 = pickle.dumps(vars(ex))
        got = action_sources(ex, 2)
        assert pickle.dumps(vars(ex)) == st0, "action_sources mutated it"
        assert got == [want, {"source": "policy"}], (seq, got)
        ex2 = OptionExecutor(_BankH(), 2, {})          # without the read
        assert ex2.act(obs, None, _PolicyH(seq), 0) == a
    # an option whose slot is unbound mid-flight closes inside act() and
    # emits a fallback primitive: neither the policy nor the skill chose it
    ex = OptionExecutor(_BankH(), 2, {})
    ex.act(obs, None, _PolicyH([P_H + 0, 1]), 0)
    ex.bank.slots[0] = None
    ex.act(obs, None, _PolicyH([1]), 1)
    assert action_sources(ex, 2)[0] == {"source": "option_fallback"}
    assert action_sources(None, 2) == [{"source": "policy"}] * 2
    assert action_sources(object(), 1) == [{"source": "unknown"}]
    # ... and the recorder files it in the Action payload, per stream
    root = os.path.join(TMP, "store_source")
    rec = ShadowRecorder.from_config(
        {"foundation": {"shadow": _shadow_cfg(root)}}, bus=None,
        action_dim=P_H, is_discrete=True, num_streams=2, environment="fake")
    wm = WorldModel(obs_dim=4, action_dim=P_H, stochastic_size=8,
                    stochastic_classes=8, deterministic_size=32,
                    hidden_dim=32)
    s = wm.rssm.initial_state(2, torch.device("cpu"))
    info = {"proprio": np.ones(3, np.float32)}
    ex = OptionExecutor(_BankH(), 2, {})
    for k, seq in enumerate([[1, 1], [1, 1], [P_H + 1, 1], [2]]):
        acts = ex.act(obs, None, _PolicyH(seq), k)
        rec.before_step(s, acts, None, wm, executor=ex)
        t = time.time()
        rec.after_step([info, info], acts, [0.0, 0.0], [False, False],
                       [False, False], [(False, False, time.time())] * 2, t)
    assert rec.enabled, rec.disabled_reason
    rec.close()
    ev = EvidenceStore(root, readonly=True).evaluator_view()
    pays = {}
    for r in ev.refs("action"):
        a = ev.get(r)
        pays[(a.stream, a.seq)] = a.payload
    assert pays[("stream-0", 1)]["source"] == "policy", pays
    assert pays[("stream-0", 2)]["source"] == "option_slot:1"
    assert pays[("stream-0", 2)]["skill"] == "sk_a"
    assert pays[("stream-0", 3)]["option_root_skill"] == "sk_b"   # continues
    assert {p["source"] for (st, _), p in pays.items()
            if st == "stream-1"} == {"policy"}
    assert all(p["actor"] == "policy" for p in pays.values())
    print(f"  H. real OptionExecutor.act(): policy / option_slot:<n>+skill / "
          f"nested child-as-actor / option_fallback read correctly, state "
          f"pickles identical after the read, act() unchanged; "
          f"{len(pays)} recorded Action payloads carry the source per stream")


# ------------------------------------------- I. pixel observable forecast
PIX = {"C": 3, "S": 16, "A": 3, "n": 2, "steps": 12}


def _pixel_wm():
    import torch
    from developmental_ai.world_model.rssm import WorldModel
    torch.manual_seed(0)
    return WorldModel(obs_dim=PIX["C"] * PIX["S"] ** 2, action_dim=PIX["A"],
                      stochastic_size=4, stochastic_classes=4,
                      deterministic_size=16, hidden_dim=16, pixel_obs=True,
                      image_channels=PIX["C"], image_size=PIX["S"])


def _tpool(x, size=None):
    """The torch reference for the 8x8 observable target."""
    import torch
    import torch.nn.functional as F
    size = size or PIX["S"]
    t = torch.as_tensor(np.asarray(x, np.float32)).reshape(
        -1, PIX["C"], size, size)
    return F.adaptive_avg_pool2d(t, (8, 8)).flatten(1).numpy()


def _pixel_run(shadow):
    """A tiny pixel learner shaped like the live one: numpy frames (what the
    loop holds in obs_list), a posterior observe_step that SAMPLES (train
    mode: Gumbel, torch RNG) and a real world-model optimizer step under the
    WM lock. Single-threaded so the A/B is deterministic. `shadow` is a
    foundation.shadow block, or None for the OFF arm. Both arms also compute
    the independent reference forecast (pure: no RNG, no grads, no writes),
    so that computation is not a difference between them."""
    import threading
    import torch
    import torch.nn.functional as F
    from developmental_ai.foundation.runtime import (ShadowRecorder,
                                                     rssm_prior_probs)
    from developmental_ai.foundation.runtime.observable import (
        predict_observation)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    wm = _pixel_wm()
    lock = threading.RLock()
    n, A, D = PIX["n"], PIX["A"], PIX["C"] * PIX["S"] ** 2
    rec = None if shadow is None else ShadowRecorder.from_config(
        {"foundation": {"shadow": shadow}}, bus=None, action_dim=A,
        is_discrete=True, num_streams=n, environment="pixel")
    rng = np.random.default_rng(7)            # the world: local generator
    frames = [rng.random((n, D), dtype=np.float32)]
    state = wm.rssm.initial_state(n, torch.device("cpu"))
    infos = [{"proprio": np.ones(3, np.float32)} for _ in range(n)]
    losses, expected, expected_probs = [], [], []
    for _k in range(PIX["steps"]):
        obs = frames[-1]
        acts = [int(x) for x in rng.integers(0, A, n)]
        a1h = torch.eye(A)[acts]
        expected.append(predict_observation(wm, state, a1h).numpy().copy())
        expected_probs.append(rssm_prior_probs(wm.rssm, state["h"],
                                               state["z"], a1h.numpy()))
        if rec is not None:
            rec.before_step(state, acts, lock, wm, observations=obs)
        nxt = np.clip(0.8 * obs + 0.2 * rng.random((n, D), dtype=np.float32),
                      0.0, 1.0).astype(np.float32)
        if rec is not None:
            t = time.time()
            rec.after_step(infos, acts, [0.0] * n, [False] * n, [False] * n,
                           [(False, False, time.time())] * n, t,
                           observations=nxt)
        with lock:                                # the trainer's update
            post, _ = wm.rssm.observe_step(state, a1h,
                                           wm.embed(torch.from_numpy(nxt)))
            loss = F.mse_loss(wm.decoder(wm.rssm.get_latent(post)),
                              torch.from_numpy(nxt))
            wm.optimizer.zero_grad()
            loss.backward()
            wm.optimizer.step()
        state = {kk: v.detach() for kk, v in post.items()}
        losses.append(float(loss))
        frames.append(nxt)
    fp = {"wm": _h(*[v.detach().numpy() for v in wm.state_dict().values()]),
          "optimizer": _h(*_arrays_of(wm.optimizer.state_dict())),
          "losses": repr(losses),
          "latent_state": _h(state["h"].numpy(), state["z"].numpy()),
          "frames": _h(*frames),
          "torch_rng": _h(torch.get_rng_state().numpy()),
          "numpy_rng": repr(np.random.get_state()[1][:8].tolist()),
          "python_rng": repr(random.getstate()[1][:8])}
    if rec is not None:
        rec.close()
    return fp, rec, frames, expected, expected_probs


def test_pixel_observable():
    import torch
    from developmental_ai.foundation.experience import EvidenceStore
    from developmental_ai.foundation.runtime import ShadowRecorder
    from developmental_ai.foundation.runtime.observable import (
        pool_observation)
    C, S, n = PIX["C"], PIX["S"], PIX["n"]
    # the numpy pooling IS torch's adaptive_avg_pool2d: block case (16 -> 8)
    # and the uneven-bin case (12 -> 8), so a future size cannot drift
    g = np.random.default_rng(3)
    x = g.random((2, C * S * S), dtype=np.float32)
    assert np.allclose(pool_observation(x, True, C, S), _tpool(x), atol=1e-6)
    y = g.random((2, C * 12 * 12), dtype=np.float32)
    assert np.allclose(pool_observation(y, True, C, 12), _tpool(y, 12),
                       atol=1e-6)
    x0 = x.copy()
    pool_observation(x, True, C, S)
    assert np.array_equal(x, x0), "pooling modified its input"

    root = os.path.join(TMP, "store_pixel")
    off, _, _, _, _ = _pixel_run(None)
    on, rec, frames, expected, eprobs = _pixel_run(_shadow_cfg(root))
    diff = [k for k in off if off[k] != on[k]]
    assert not diff, f"pixel: shadow ON changed the learner: {diff}"
    st = rec.stats()
    cn = st["counts"]
    want = n * (PIX["steps"] - 1)           # step 0 has no context yet
    assert rec.enabled and rec.stage() == "full", st
    assert cn.get("predictions") == want, cn
    assert cn.get("observable_scored") == want, cn
    assert cn.get("observable_errors", 0) == 0, cn
    assert cn.get("observable_device_copies", 0) == 0, cn   # numpy obs

    ev = EvidenceStore(root, readonly=True).evaluator_view()
    preds = {}
    for r in ev.refs("prediction"):
        p = ev.get(r)
        name, _ep, seq, _p = p.prediction_id.split(":")
        preds[(name, int(seq))] = p
    assert len(preds) == want, len(preds)
    for (name, seq), p in preds.items():
        e = int(name.split("-")[1])
        ob = p.outcome["observable"]
        assert ob["target"] == "pooled-pov-8x8" and bool(ob["pixel"]), ob
        pr = np.asarray(ob["prediction"], np.float32)
        assert pr.shape == (C * 64,), pr.shape
        # h1/logits reuse == an independent second transition + decode
        ref = _tpool(expected[seq][e:e + 1])[0]
        assert np.allclose(pr, ref, atol=1e-5), np.abs(pr - ref).max()
        assert np.allclose(np.asarray(ob["persistence"]),
                           _tpool(frames[seq][e:e + 1])[0], atol=1e-6)
        assert np.allclose(np.asarray(p.outcome["probs"])[0],
                           eprobs[seq][e], atol=1e-6)
    scored = 0
    for r in ev.refs("evidence"):
        evd = ev.get(r)
        if evd.provenance != "sensor":
            continue
        a = evd.actions[0]
        e = int(a.stream.split("-")[1])
        sc = evd.payload["observable_score"]
        assert sc["status"] == "scored", sc
        tgt = _tpool(frames[a.seq + 1][e:e + 1])[0]
        base = _tpool(frames[a.seq][e:e + 1])[0]
        pr = np.asarray(preds[(a.stream, a.seq)].outcome["observable"]
                        ["prediction"], np.float32)
        assert np.allclose(np.asarray(sc["target_value"]), tgt, atol=1e-6)
        assert abs(sc["mse"] - float(np.mean((pr - tgt) ** 2))) < 1e-6
        assert abs(sc["persistence_mse"]
                   - float(np.mean((base - tgt) ** 2))) < 1e-6
        scored += 1
    assert scored == want, scored
    print(f"  I. PIXEL 3x{S}x{S}: {len(off)} learner fingerprints identical "
          f"OFF vs ON (observable on, every step); {want} forecasts == an "
          f"independent transition+decode, {scored} outcomes scored against "
          f"the raw frames; numpy pool == adaptive_avg_pool2d (16->8, 12->8); "
          f"0 device copies")

    # falsifier: the pixel A/B is sensitive (one torch draw per step moves it)
    orig = ShadowRecorder.before_step

    def leaky(self, *a, **k):
        orig(self, *a, **k)
        torch.rand(1)
    ShadowRecorder.before_step = leaky
    try:
        bad = _pixel_run(_shadow_cfg(os.path.join(TMP, "store_pixel_leaky")))[0]
    finally:
        ShadowRecorder.before_step = orig
    moved = [k for k in bad if bad[k] != off[k]]
    assert "torch_rng" in moved and len(moved) >= 2, moved
    print(f"     falsifier: one torch draw/step in the recorder moves {moved}")


# ------------------------------- J. observable isolation + staged degrade
def test_observable_isolation():
    import torch
    from developmental_ai.foundation.experience import EvidenceStore
    from developmental_ai.foundation.runtime import ShadowRecorder
    C, S, A, n = PIX["C"], PIX["S"], PIX["A"], PIX["n"]
    D = C * S * S
    infos = [{"proprio": np.ones(3, np.float32)} for _ in range(n)]
    g = np.random.default_rng(11)

    def make(root, **kw):
        return ShadowRecorder.from_config(
            {"foundation": {"shadow": _shadow_cfg(root, **kw)}}, bus=None,
            action_dim=A, is_discrete=True, num_streams=n,
            environment="pixel")

    def drive(rec, wm, steps, next_width=D):
        s = wm.rssm.initial_state(n, torch.device("cpu"))
        for k in range(steps):
            acts = [k % A] * n
            rec.before_step(s, acts, None, wm, observations=g.random(
                (n, D), dtype=np.float32))
            rec.after_step(infos, acts, [0.0] * n, [False] * n, [False] * n,
                           [(False, False, time.time())] * n, time.time(),
                           observations=g.random((n, next_width),
                                                 dtype=np.float32))

    def records(root):
        ev = EvidenceStore(root, readonly=True).evaluator_view()
        return ([ev.get(r) for r in ev.refs("prediction")],
                [ev.get(r) for r in ev.refs("evidence")
                 if ev.get(r).provenance == "sensor"])

    # 1. a decoder that raises: categorical keeps going, observable turns off
    wm = _pixel_wm()

    def boom(*a, **k):
        raise RuntimeError("decoder exploded (injected)")
    wm.decoder.forward = boom
    root = os.path.join(TMP, "store_obs_fail")
    h = _capture()
    try:
        rec = make(root, observable_max_failures=2)
        drive(rec, wm, 6)
    finally:
        _release(h)
    cn = rec.stats()["counts"]
    assert rec.enabled, rec.disabled_reason
    assert rec.stage() == "categorical" and not rec.do_observable
    assert "forecast" in rec.observable_off_reason, rec.observable_off_reason
    assert cn["observable_errors"] == 2 == cn["observable_errors:forecast"], cn
    assert cn["predictions"] == n * 5, cn
    assert len(h.msgs) == 1 and "observable forecasts OFF" in h.msgs[0], h.msgs
    rec.close()
    preds, outs = records(root)
    assert len(preds) == n * 5 and len(outs) == n * 5
    assert not [p for p in preds if "observable" in p.outcome]
    assert not [o for o in outs if "observable_score" in o.payload]
    print("  J. raising decoder: 2 consecutive failures -> observable OFF "
          "(1 warning), recorder still enabled, all categorical predictions "
          "filed, no 'observable' key written")

    # 2. a malformed outcome observation fails at the SCORE site the same way
    wm = _pixel_wm()
    root = os.path.join(TMP, "store_score_fail")
    h = _capture()
    try:
        rec = make(root, observable_max_failures=2)
        drive(rec, wm, 4, next_width=D + 1)
    finally:
        _release(h)
    cn = rec.stats()["counts"]
    assert rec.enabled and rec.stage() == "categorical", rec.stats()
    assert cn["observable_errors"] == 2 == cn["observable_errors:score"], cn
    assert len(h.msgs) == 1 and "score" in h.msgs[0], h.msgs
    rec.close()
    preds, outs = records(root)
    st = collections.Counter(o.payload["observable_score"]["status"]
                             for o in outs if "observable_score" in o.payload)
    assert st == {"score-error": 2}, st
    assert len(preds) == n * 3, len(preds)
    print("  J. malformed outcome frame: 2 score errors -> observable OFF "
          "(1 warning), labelled score-error, never scored as success")

    # 3. budget stages, in order, one warning each
    wm = _pixel_wm()
    h = _capture()
    try:
        rec = make(os.path.join(TMP, "store_stages"), max_ms_per_step=1e-9,
                   budget_window=2)
        stages = []
        for _ in range(3):
            drive(rec, wm, 2)
            stages.append(rec.stage())
    finally:
        _release(h)
    assert stages == ["categorical", "records", "disabled"], stages
    assert len(h.msgs) == 3, h.msgs
    assert "observable forecasts OFF" in h.msgs[0], h.msgs
    assert "predictions OFF" in h.msgs[1] and "DISABLED" in h.msgs[2], h.msgs
    assert [d["stage"] for d in rec.degraded] == ["categorical", "records"]
    print(f"  J. over budget: stages {stages} after each window of 2, "
          f"exactly 3 warnings")

    # 4. feature off: old-store shape (no observable key, no score)
    root = os.path.join(TMP, "store_obs_off")
    rec = make(root, observable_predict=False)
    drive(rec, _pixel_wm(), 3)
    assert rec.enabled and rec.stage() == "categorical"
    rec.close()
    preds, outs = records(root)
    assert len(preds) == n * 2 and len(outs) == n * 2
    assert not [p for p in preds if "observable" in p.outcome]
    assert not [o for o in outs if "observable_score" in o.payload]
    print("  J. observable_predict off: predictions carry no 'observable' "
          "key and evidence no observable_score (reads as unrecorded)")


if __name__ == "__main__":
    main()
