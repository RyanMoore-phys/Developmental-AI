"""Learning telemetry (runlogs/learning.jsonl) — contract smoke (2026-10-07).

WHAT IS CLAIMED
    The agent's pay was reported only as unsigned ledger SHARES of stream 0,
    and the terms the ledger never books (icm_base, empowerment, approach,
    gui_dwell) plus every multiplier that shrank the drive (habituation, farm
    damp, GUI zeroing, the mixer's return-ratio scale) were invisible. The
    scoreboard (CLAUDE.md §9) has been misread every time the reward number
    was trusted, so a per-source breakdown is only worth having if it is
    PROVABLY the reward PPO stored. This test drives the REAL live bodies
    (`_collect_segment` and `_run_episode_parallel`) with a fake two-stream
    env that provides info["oracle"], GUI frames and break events, and checks:

      A. SOURCE: every telemetry stage line sits in BOTH live bodies with
         identical text (count 2, §4.2), none in the unwired `_run_episode`;
         `_lt_emit()` is called only inside `_emit_metrics`, whose call site
         is still exactly one; the ExplorationTracker is fed only inside
         `_oracle_observe`.
      B. SCHEMA: one record per segment with every spec section (reward,
         reward_mix, outcomes, behaviour, exploration, ppo, wm, curiosity,
         replay, timing, health); seq strictly increasing.
      C. KEY CLAIM — RECONCILIATION: for stream-0 AND the scout, the sum of
         the per-source parts equals the mixed reward the body actually
         passed on (captured by a pass-through wrapper on RewardMixer.mix),
         |diff| < 1e-6 per segment, with the GUI zeroing, the GUI dwell cost
         and the mixer's return-ratio scale all ACTIVE (asserted nonzero).
      C2. FALSIFIER: the check is not a tautology — the same run with the
         re-derived mixer scale forced to 1.0 FAILS reconciliation.
      D. EXPLORATION for BOTH streams, and the evaluator-only position trace
         is written for both.
      E. The episodic body reconciles too (both bodies are live).
      F. NON-INTERFERENCE: learner state (replay, rollout, WM, policy,
         curiosity, mixer, all RNGs, env action log) byte-identical with
         telemetry on vs off. The producers' consume-once counters (`_ss_*`,
         `_gs_*`) are excluded: popping them is their designed purpose and
         they feed nothing.
      G. DURABILITY: a second writer on the same path resumes seq (never
         rewinds).
      H. COST: telemetry ms/segment is reported (printed, bounded).

Run: PYTHONPATH=. ./venv/bin/python tests/_learning_telemetry_smoke.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import re
import sys
import tempfile

import numpy as np
import yaml

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
TMP = tempfile.mkdtemp(prefix="ltel_smoke_")
ENV_LOG = {}


# ------------------------------------------------------------ the fake env
def _sensed_cartpole(idx):
    import gymnasium as gym

    class SensedCartPole(gym.Wrapper):
        """CartPole with a body, a RED oracle position that MOVES, periodic
        GUI frames (stream 0 and 1 on different phases) and break events."""

        def __init__(self, idx):
            super().__init__(gym.make("CartPole-v1", max_episode_steps=200))
            self.idx = idx
            self.resets = -1
            self.t = 0
            self.T = 0
            ENV_LOG[idx] = []

        def _aug(self, o, info):
            info = dict(info)
            p = np.asarray(o, np.float32)
            fx = np.tanh(np.array([p[0], p[1], p[2], p[3], p[0] * p[2], 0.5],
                                  np.float32))
            info["proprio"] = p
            info["sensors"] = np.concatenate([p, fx]).astype(np.float32)
            # walks ~0.3 blocks/step along x with a wobble in z
            info["oracle"] = {"true_position": np.array(
                [0.3 * self.T + 5.0 * p[0], 64.0, 3.0 * np.sin(self.T / 9.0)],
                np.float32)}
            info["world"] = {"moved": 0.3 if self.T % 5 else 0.0,
                             "yaw": float(self.T % 360), "pitch": 0.0}
            info["gui_open"] = ((self.T + 3 * self.idx) // 6) % 4 == 0
            ev = []
            if self.T % 13 == 5:
                ev.append(("break", "oak_log" if self.T % 2 else "dirt"))
            info["events"] = ev
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
            self.T += 1
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


def _cfg(lifelong, telemetry, tag):
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
    d = os.path.join(TMP, tag)
    cfg["telemetry"] = {
        "enabled": bool(telemetry),
        "learning": {"path": os.path.join(d, "learning.jsonl")},
        "exploration": {"enabled": True, "trace_every_steps": 4,
                        "trace_path": os.path.join(d, "trace.jsonl")}}
    return cfg


def _activate(ai):
    """Turn on the stages a CartPole config leaves dormant, identically in
    every run: GUI dwell cost, scouts in PPO and the mixer's return-ratio
    damper (target tiny -> scale < 1 once extrinsic is seen)."""
    ai._gui_dwell_weight = 0.5
    ai._gui_dwell_step_cost = 0.01
    ai._gui_dwell_grace_steps = 2
    ai._scouts_in_ppo = True
    ai.reward_mixer.target_ratio = 1e-3
    ai.reward_mixer.min_intrinsic_scale = 0.25


def _capture_mix(ai):
    """Pass-through wrapper: records exactly what mix() returned."""
    box = {"primary": 0.0, "scout": 0.0}
    orig = ai.reward_mixer.mix

    def mix(i, e, update_stats=True):
        out = orig(i, e, update_stats=update_stats)
        box["primary" if update_stats else "scout"] += float(out)
        return out
    ai.reward_mixer.mix = mix
    return box


def _run(lifelong, telemetry, tag, segments=3, capture=False):
    dl = _patch_make_env()
    ENV_LOG.clear()
    ai = dl.DevelopmentalAI(config=_cfg(lifelong, telemetry, tag))
    _activate(ai)
    box = _capture_mix(ai) if capture else None
    caught = []
    try:
        for _ in range(segments):
            if box is not None:
                box["primary"] = box["scout"] = 0.0
            if lifelong:
                m = ai._collect_segment()
            else:
                m = ai._run_episode_parallel(use_dream_actor=False)
            ai._lt_snapshot_timing()   # as train(): before _log_progress
            ai._emit_metrics(m)
            if box is not None:
                caught.append(dict(box))
        return ai, caught
    finally:
        ai.close()


def _records(tag):
    p = os.path.join(TMP, tag, "learning.jsonl")
    return [json.loads(l) for l in open(p)]


# ------------------------------------------------------------- A. source
def _body(src, name):
    i = src.index(f"    def {name}(")
    j = src.index("\n    def ", i + 10)
    return src[i:j]


def test_source():
    src = open(LOOP).read()
    lines = [l.strip() for l in src.split("\n")
             if re.match(r"\s*self\._lt_(begin|mark|commit|pre_step|"
                         r"post_step)\(", l)]
    stage = collections.Counter(lines)
    par = _body(src, "_run_episode_parallel")
    col = _body(src, "_collect_segment")
    leg = _body(src, "_run_episode")
    assert "_lt_" not in leg, "_run_episode must stay unwired"
    only_collect = {l for l in stage if "damp_boring_view" in l}
    assert len(only_collect) == 1, only_collect     # collect-only stage
    n_shared = 0
    for l, c in stage.items():
        if l in only_collect:
            assert c == 1 and col.count(l) == 1, l
            continue
        assert c == 2, (l, c)
        assert par.count(l) == 1 and col.count(l) == 1, l
        n_shared += 1
    assert n_shared >= 12, n_shared
    # one emission site, and learning rides it
    assert len(re.findall(r"self\._emit_metrics\(", src)) == 1
    assert src.count("self._lt_emit()") == 1
    em = _body(src, "_emit_metrics")
    assert "self._lt_emit()" in em
    # exploration fed only in _oracle_observe
    assert src.count("_xt.observe(") == 1
    assert "_xt.observe(" in _body(src, "_oracle_observe")
    print(f"  A. {n_shared} stage lines x2 (identical, one per live body) + "
          "1 collect-only; 0 in _run_episode; one emission site; "
          "exploration fed only in _oracle_observe")


# -------------------------------------------------- B/C/D. lifelong run
SECTIONS = ("reward", "reward_mix", "outcomes", "behaviour", "exploration",
            "ppo", "wm", "curiosity", "replay", "timing", "health")


def _reconcile(recs, caught, tol=1e-6):
    worst = 0.0
    for r, c in zip(recs, caught):
        for key, ref in (("stream-0", c["primary"]), ("stream-1", c["scout"])):
            parts = sum(r["reward"][key].values())
            mixed = r["reward_mix"][key]["mixed_sum"]
            worst = max(worst, abs(parts - ref), abs(mixed - ref))
    return worst


def test_lifelong():
    ai, caught = _run(True, True, "on", segments=3, capture=True)
    recs = _records("on")
    assert len(recs) == 3, len(recs)
    for r in recs:
        assert r["schema"] == "skybot.learning" and r["v"] == 1
        for s in SECTIONS:
            assert s in r, s
        assert r["streams"] == 2 and r["segment_steps"] > 0
        for k in ("stream-0", "stream-1"):
            for s in ("reward", "reward_mix", "outcomes", "behaviour"):
                assert k in r[s], (s, k)
        b0 = r["behaviour"]["stream-0"]
        assert b0["action_hist"] and b0["source_hist"], b0
        assert 0.0 < b0["gui_frac"] < 1.0 and b0["gui_longest_run"] >= 1
        assert b0["still_frac"] is not None
        assert "steps" in r["wm"] and "replay_ratio" in r["wm"]
        assert "env_wait_ms" in r["timing"] and r["timing"]["steps_per_s"] > 0
        assert r["replay"]["size"] > 0
        for h in ("nan_steps", "grad_spikes", "sink_dropped"):
            assert h in r["health"]
    # the first segment is replay warm-up; later ones must carry WM means
    assert any(r["wm"]["steps"] >= 1 and "loss_total" in r["wm"]
               for r in recs[1:]), [r["wm"] for r in recs]
    seqs = [r["seq"] for r in recs]
    assert seqs == sorted(seqs) and len(set(seqs)) == 3
    print("  B. 3 records, every spec section, both streams, seq "
          f"{seqs} increasing")

    worst = _reconcile(recs, caught)
    assert worst < 1e-6, worst
    r0 = [r["reward"]["stream-0"] for r in recs]
    m0 = [r["reward_mix"]["stream-0"] for r in recs]
    assert any(abs(x.get("gui_zero", 0.0)) > 0 for x in r0), "gui zero idle"
    assert any(abs(x.get("gui_dwell", 0.0)) > 0 for x in r0), "dwell idle"
    assert any(abs(x.get("env", 0.0)) > 0 for x in r0)
    assert any(x["int_scale_mean"] < 0.999 for x in m0), "mixer scale idle"
    assert all(abs(x.get("unattributed", 0.0)) < 1e-9 for x in r0)
    s1 = [r["reward"]["stream-1"] for r in recs]
    assert any(abs(x.get("gui_zero", 0.0)) > 0 for x in s1)
    print(f"  C. reconciliation: sum(parts) == mixed PPO reward, stream-0 "
          f"and scout, worst |diff| {worst:.2e} (< 1e-6) with gui_zero, "
          "gui_dwell and the mixer scale active; unattributed == 0")

    ex = recs[-1]["exploration"]
    for k in ("stream-0", "stream-1"):
        assert ex[k]["n_obs"] > 0 and ex[k]["path_blocks"] > 0, ex[k]
        assert ex[k]["unique_cells_total"] > 1
    tr = [json.loads(l) for l in open(os.path.join(TMP, "on", "trace.jsonl"))]
    assert {t["stream"] for t in tr} == {0, 1}
    assert all(t["schema"] == "skybot.position" for t in tr)
    print(f"  D. exploration for both streams; position trace {len(tr)} "
          "lines, both streams")
    br = sum(r["outcomes"]["stream-0"]["breaks_segment"] for r in recs)
    assert br > 0 and recs[-1]["outcomes"]["stream-0"]["breaks_run"] == br
    cost = [r["timing"]["telemetry_ms"] for r in recs]
    return ai, cost


def test_falsifier():
    import developmental_ai.infra.learning_telemetry as lt
    orig = lt.intrinsic_scale
    lt.intrinsic_scale = lambda mixer, i: 1.0
    try:
        _ai, caught = _run(True, True, "fals", segments=2, capture=True)
    finally:
        lt.intrinsic_scale = orig
    worst = _reconcile(_records("fals"), caught)
    assert worst > 1e-6, worst
    print(f"  C2. falsifier: mixer scale ignored -> reconciliation breaks "
          f"(|diff| {worst:.3g})")


def test_episodic():
    _ai, caught = _run(False, True, "epi", segments=2, capture=True)
    recs = _records("epi")
    worst = 0.0
    for r, c in zip(recs, caught):
        worst = max(worst, abs(sum(r["reward"]["stream-0"].values())
                               - c["primary"]))
    assert worst < 1e-6, worst
    print(f"  E. episodic _run_episode_parallel reconciles stream-0 "
          f"(|diff| {worst:.2e})")


# --------------------------------------------------- F. non-interference
def _h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode() + str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()


SKIP = ("_ss_", "_gs_")              # producers' consume-once counters
WALLCLOCK = {"lock_held_s", "lock_wait_s"}


def _arrays_of(obj, depth=0, seen=None):
    import torch
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return []
    seen.add(id(obj))
    if isinstance(obj, (bool, int, float, np.generic)):
        return [np.asarray(obj)]
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, torch.Tensor):
        return [obj.detach().cpu().numpy()]
    out = []
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
            if (k.startswith("_lock") or "thread" in k or k in WALLCLOCK
                    or k.startswith(SKIP)):
                continue
            out += _arrays_of(vars(obj)[k], depth + 1, seen)
    return out


def _fingerprint(ai):
    import torch
    pol = ai.policy
    return {
        "env_log": repr({k: list(v) for k, v in ENV_LOG.items()}),
        "replay": _h(*_arrays_of(ai.replay_buffer)),
        "rollout": repr([np.asarray(x).tolist() for x in pol.rollout_rewards])
        + _h(*[np.asarray(x) for x in pol.rollout_obs]),
        "wm": _h(*[v.cpu().numpy() for v in
                   ai.world_model.state_dict().values()]),
        "policy": _h(*_arrays_of(pol)),
        "curiosity": _h(*_arrays_of(ai.curiosity)),
        "reward_mixer": _h(*_arrays_of(ai.reward_mixer)),
        "torch_rng": _h(torch.get_rng_state().numpy()),
        "numpy_rng": repr(np.random.get_state()[1][:8].tolist()),
        "python_rng": repr(random.getstate()[1][:8]),
    }


def test_non_interference():
    a, _ = _run(True, False, "off_a", segments=2)
    fa = _fingerprint(a)
    b, _ = _run(True, True, "on_b", segments=2)
    fb = _fingerprint(b)
    assert b._ltel is not None and a._ltel is None
    diff = [k for k in fa if fa[k] != fb[k]]
    assert not diff, diff
    print(f"  F. learner byte-identical telemetry ON vs OFF "
          f"({len(fa)} fingerprints: replay, rollout, wm, policy, "
          "curiosity, mixer, 3 RNGs, env actions)")


def test_durability():
    from developmental_ai.infra.learning_telemetry import LearningTelemetry
    p = os.path.join(TMP, "dur", "learning.jsonl")
    t1 = LearningTelemetry({"path": p})
    t1.record(1, {})
    t1.record(2, {})
    t2 = LearningTelemetry({"path": p})
    t2.record(3, {})
    seqs = [json.loads(l)["seq"] for l in open(p)]
    assert seqs == [1, 2, 3], seqs
    print("  G. restart resumes seq (1,2 -> 3), never rewinds")


def main():
    test_source()
    _ai, cost = test_lifelong()
    test_falsifier()
    test_episodic()
    test_non_interference()
    test_durability()
    mean = float(np.mean(cost[1:] or cost))
    assert mean < 1000.0, cost
    print(f"  H. telemetry cost {mean:.1f} ms/segment (40-step segments x 2 "
          f"streams; per-segment {['%.1f' % c for c in cost]})")
    print("[_learning_telemetry_smoke] ALL PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        import traceback
        traceback.print_exc()
        print(f"[_learning_telemetry_smoke] FAIL: {e!r}")
        sys.exit(1)
