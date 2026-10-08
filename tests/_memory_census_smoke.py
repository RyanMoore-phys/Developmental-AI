"""MEMORY CENSUS smoke (2026-10-07): an exact, read-only breakdown of RAM/VRAM.

WHY THIS EXISTS
    The training host has 16 GB. Measured live: the python agent at 7.57 GB
    RSS, the two MineRL java clients at 1.29 + 1.19 GB, 4.2 GB available of
    14.6, swap at 2-2.5 GB. The log claims the replay buffer is "25000
    transitions per stream (2.0 GB budget)". Deciding what to offload to the
    two CPU-only LAN nodes needs the 7.57 GB broken into movable parts — and
    CLAUDE.md §6 already records that the budget number lies ("a 2 GB ceiling
    measured 3.70 GB": the initial block per stream is allocated WITHOUT a
    budget request).

    A monitoring path has two failure modes (tests/_metrics_sink_smoke.py):
    it can kill what it watches, and it can look alive while measuring
    nothing. This test checks both, plus a third that a memory census is
    uniquely placed to commit: CHANGING what it measures (a walk that calls a
    property, consumes an RNG or touches a buffer would perturb learning).

Contracts:
    A. Every section is present and the census raises nothing on a real
       DevelopmentalAI that has run the live body (_collect_segment).
    B. Replay bytes are EXACT: every column of every stream equals the sum of
       its blocks' `.nbytes`, counted here independently of the census.
    C. Budget-vs-actual is reported and EXPLAINED from code: actual minus the
       live streams' reservations equals (initial block per stream) +
       (proprio column of every grown block), to the byte. Falsified by
       checking the undercount is nonzero — a census that reported actual ==
       reserved would be repeating the budget's claim, not measuring.
    D. The object walk finds the world model and the policy as nn.Modules,
       with param bytes equal to the modules' own parameter nbytes, and finds
       their optimizers' state.
    E. The hook writes one line at the first tick, respects the interval,
       and the MEMCENSUS trigger file forces one and is removed.
    F. Wired at the SINGLE emission site: `_emit_metrics` writes a census
       even with the metrics sink off, and there is still exactly one call
       to `_emit_metrics` in the loop.
    G. Learner state is byte-identical before and after a census (deep mode
       included): world model, policy, curiosity, replay, all three RNGs.
    H. Never raises into the loop: a census of a hostile object, and a hook
       whose path cannot be written, return without exception.

Run: PYTHONPATH=. ./venv/bin/python tests/_memory_census_smoke.py
"""
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, ".")

import numpy as np
import yaml

logging.disable(logging.WARNING)
TMP = tempfile.mkdtemp(prefix="memcensus_smoke_")
LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
BLOCK = 1000


# --------------------------------------------------------- fake sensed env
def _sensed_cartpole(idx):
    import gymnasium as gym

    class SensedCartPole(gym.Wrapper):
        PROPRIO_DIM = 4
        PROPRIO_KEYS = ("x", "x_dot", "theta", "theta_dot")

        def __init__(self, idx):
            super().__init__(gym.make("CartPole-v1", max_episode_steps=200))
            self.idx, self.resets = idx, -1

        def _aug(self, o, info):
            info = dict(info)
            p = np.asarray(o, np.float32)
            fx = np.tanh(np.array([p[0], p[1], p[2], p[3], p[0] * p[2], 0.5],
                                  np.float32))
            info["proprio"] = p
            info["sensors"] = np.concatenate([p, fx]).astype(np.float32)
            return info

        def reset(self, **kw):
            self.resets += 1
            kw["seed"] = 1000 * self.idx + self.resets
            o, i = self.env.reset(**kw)
            return o, self._aug(o, i)

        def step(self, a):
            o, r, te, tr, i = self.env.step(a)
            return o, r, te, tr, self._aug(o, i)

    return SensedCartPole(idx)


def _agent():
    import developmental_ai.core.developmental_loop as dl
    from developmental_ai.environments.wrappers import DevelopmentalEnvWrapper
    state = {"n": 0}

    def fake(*a, **k):
        idx = state["n"]
        state["n"] += 1
        return DevelopmentalEnvWrapper(_sensed_cartpole(idx),
                                       normalize_obs=False), None
    dl.make_env = fake
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["seed"] = 0
    cfg["world_model"].update({
        "stochastic_size": 8, "stochastic_classes": 8,
        "deterministic_size": 32, "encoder_hidden": 32,
        "decoder_hidden": 32, "batch_size": 4, "sequence_length": 8,
        "train_iters": 1, "buffer_capacity": 2000, "proprio": True,
        # growth ON: the live config's storage form (BlockArray + budget)
        "buffer_growth": {"enabled": True, "block_transitions": BLOCK,
                          "grow_at_frac": 0.85, "max_ram_frac": 0.5,
                          "hard_max_gb": 1.0}})
    cfg["sensors"] = {"enabled": ["proprio", "screen_fx"]}
    cfg["parallel_envs"] = {"enabled": True, "num_envs": 2}
    cfg["policy"].update({"n_steps": 40, "minibatch_size": 16, "n_epochs": 2})
    cfg["lifelong"] = {"enabled": True, "forever": False, "segment_len": 40,
                       "wm_train_every": 25, "goal_horizon": 10 ** 9,
                       "state_decay": 1.0, "reset_on_death_only": True,
                       "stop_file": os.path.join(TMP, "STOP")}
    cfg.setdefault("loop", {})["verbose"] = 0
    cfg.setdefault("llm", {})["enabled"] = False
    cfg.setdefault("skill_bank", {})["storage_dir"] = os.path.join(TMP, "sb")
    cfg["metrics"] = {"enabled": False}
    cfg.pop("diagnostics", None)
    return dl.DevelopmentalAI(config=cfg)


# ------------------------------------------------------------- fingerprint
def _h(arrs):
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return m.hexdigest()


def _replay_arrays(buf):
    out = []
    for s in buf.streams:
        for c in ("observations", "proprio", "actions", "rewards", "dones",
                  "restarts", "priorities"):
            col = getattr(s, c, None)
            if col is not None:
                out += list(col._blocks)
        out.append(np.array([s.size, s.position, s.capacity, s._grow_events,
                             len(s.episode_starts)]))
    return out


def _sd(module):
    return [v.detach().cpu().numpy() for v in module.state_dict().values()]


def _opt_state(ai):
    import torch
    out = []
    for o in (getattr(ai.world_model, "optimizer", None),
              getattr(ai.policy, "optimizer", None)):
        if o is None:
            continue
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    out.append(v.detach().cpu().numpy())
    return out


def _fingerprint(ai):
    import torch
    return {
        "wm": _h(_sd(ai.world_model)),
        "policy": _h(_sd(ai.policy) if hasattr(ai.policy, "state_dict")
                     else []),
        "curiosity": _h(_sd(ai.curiosity) if hasattr(ai.curiosity,
                                                     "state_dict") else []),
        "optim": _h(_opt_state(ai)),
        "replay": _h(_replay_arrays(ai.replay_buffer)),
        "torch_rng": _h([torch.get_rng_state().numpy()]),
        "numpy_rng": repr(np.random.get_state()[1][:8].tolist())
        + str(np.random.get_state()[2]),
        "python_rng": repr(random.getstate()[1][:8]),
        "timesteps": ai.total_timesteps,
    }


# ------------------------------------------------------------------- tests
def _fill_and_grow(ai):
    """Push each stream past grow_at_frac and grow once, so the census sees
    both an unbudgeted initial block and a budgeted grown one."""
    buf = ai.replay_buffer
    rng = np.random.RandomState(7)
    for i, s in enumerate(buf.streams):
        need = int(0.9 * s.capacity) - s.size + 1
        for _ in range(max(0, need)):
            s.add(rng.rand(s.obs_dim).astype(np.float32),
                  rng.rand(s.action_dim).astype(np.float32), 0.0, False,
                  proprio=(rng.rand(s.proprio_dim).astype(np.float32)
                           if s.proprio_dim else None))
    assert buf.maybe_grow(), "growth did not trigger; contract C untestable"
    for s in buf.streams:
        if s._grow_events == 0:          # any() short-circuits; grow the rest
            s.maybe_grow()


def test_sections_and_replay(ai):
    from developmental_ai.infra.memory_census import census
    rec = census(ai)
    for k in ("process", "replay", "sampler", "objects", "cuda", "summary",
              "errors"):
        assert k in rec, f"section {k} missing"
    assert not rec["errors"], rec["errors"]
    assert rec["process"].get("rss", 0) > 0, rec["process"]
    json.dumps(rec)                       # must be serialisable as-is
    print(f"  A. all sections present, no errors, JSON-serialisable "
          f"(census {rec['census_seconds']}s, rss "
          f"{rec['process']['rss'] / 1e6:.0f} MB)")

    rp = rec["replay"]
    assert rp["num_streams"] == len(ai.replay_buffer.streams) == 2
    n_cols = 0
    for s, si in zip(ai.replay_buffer.streams, rp["streams"]):
        for c in ("observations", "proprio", "actions", "rewards", "dones",
                  "restarts", "priorities"):
            col = getattr(s, c, None)
            if col is None:
                assert c not in si["columns"]
                continue
            truth = sum(b.nbytes for b in col._blocks)
            assert si["columns"][c]["bytes"] == truth, (c, si["columns"][c],
                                                        truth)
            n_cols += 1
        assert si["total_bytes"] == sum(v["bytes"]
                                        for v in si["columns"].values())
    assert "proprio" in rp["streams"][0]["columns"], \
        "proprio column absent; the proprio undercount is not exercised"
    print(f"  B. {n_cols} replay columns across 2 streams equal their "
          f"blocks' nbytes exactly (total {rp['total_bytes'] / 1e6:.2f} MB)")
    return rec


def test_budget_accounting(rec, ai):
    from developmental_ai.world_model.replay_buffer import MEMORY_BUDGET
    rp = rec["replay"]
    acc, bud = rp["accounting"], rp["budget"]
    assert bud["reserved_bytes"] == MEMORY_BUDGET.reserved
    assert bud["reserved_not_by_live_streams"] == 0, bud
    init = sum(s["columns"][c]["first_block_bytes"]
               for s in rp["streams"] for c in s["columns"])
    grows = sum(s["grow_events"] for s in rp["streams"])
    assert grows >= 2, "each stream should have grown once"
    prop = sum(s["grow_events"] * BLOCK * s_obj.proprio_dim * 4
               for s, s_obj in zip(rp["streams"], ai.replay_buffer.streams))
    assert acc["undercount_bytes"] == init + prop, (acc, init, prop)
    assert acc["unexplained_bytes"] == 0, acc
    assert acc["undercount_bytes"] > 0, \
        "census reports actual == reserved: it is echoing the budget"
    assert acc["actual_bytes"] == bud["reserved_bytes"] + \
        acc["undercount_bytes"]
    print(f"  C. budget reserved {bud['reserved_bytes']:,} B vs actual "
          f"{acc['actual_bytes']:,} B; undercount {acc['undercount_bytes']:,}"
          f" = initial blocks {init:,} + proprio-in-grows {prop:,}, "
          f"unexplained 0")


def test_modules(rec, ai):
    import torch
    mods = rec["objects"]["modules"]
    by_path = {m["path"]: m for m in mods}
    for attr in ("world_model", "policy"):
        mod = getattr(ai, attr)
        if not isinstance(mod, torch.nn.Module):
            continue
        assert attr in by_path, (attr, sorted(by_path)[:20])
        truth = sum(p.numel() * p.element_size() for p in mod.parameters())
        assert by_path[attr]["param_bytes"] == truth, (attr, truth,
                                                       by_path[attr])
        assert "cpu" in by_path[attr]["by_device"]
    opts = rec["objects"]["optimizers"]
    assert opts, "no optimizer found by the walk"
    assert any(o["state_bytes"] > 0 for o in opts), opts
    print(f"  D. {rec['objects']['modules_found']} modules (world_model "
          f"{by_path['world_model']['param_bytes']:,} B params, exact) and "
          f"{len(opts)} optimizers found; walk "
          f"{rec['objects']['walk']['nodes']} nodes in "
          f"{rec['objects']['walk']['seconds']}s")


def test_hook(ai):
    from developmental_ai.infra.memory_census import MemoryCensusHook
    p = os.path.join(TMP, "hook", "memory_census.jsonl")
    h = MemoryCensusHook({"memory_census_every_s": 100,
                          "memory_census_path": p})
    assert h.tick(ai, now=0.0) is True
    assert h.tick(ai, now=50.0) is False
    assert h.tick(ai, now=99.9) is False
    assert h.tick(ai, now=100.0) is True
    lines = open(p).read().splitlines()
    assert len(lines) == 2, len(lines)
    assert [json.loads(x)["trigger"] for x in lines] == ["startup",
                                                         "interval"]
    trig = os.path.join(os.path.dirname(p), "MEMCENSUS")
    open(trig, "w").close()
    assert h.tick(ai, now=101.0) is True
    assert not os.path.exists(trig), "trigger file not consumed"
    assert h.tick(ai, now=102.0) is False, "trigger fired twice"
    lines = open(p).read().splitlines()
    assert len(lines) == 3 and json.loads(lines[-1])["trigger"] == "file"
    off = MemoryCensusHook({})
    assert off.enabled is False and off.tick(ai) is False
    print("  E. hook: startup line at first tick, none inside the interval, "
          "one at the interval, MEMCENSUS forces one and is consumed; no "
          "path = no-op")


def test_emit_site(ai):
    src = open(LOOP).read()
    assert src.count("self._emit_metrics(episode_metrics)") == 1
    assert src.count("MemoryCensusHook(") == 1
    p = os.path.join(TMP, "emit", "memory_census.jsonl")
    ai.config["diagnostics"] = {"memory_census_every_s": 3600,
                                "memory_census_path": p}
    ai._memory_census_hook = None
    assert getattr(ai, "_metrics_sink", None) is None
    ai._emit_metrics({})
    ai._emit_metrics({})                  # inside the interval: no 2nd line
    lines = open(p).read().splitlines()
    assert len(lines) == 1, len(lines)
    assert json.loads(lines[0])["replay"]["present"] is True
    print("  F. _emit_metrics writes the census with the metrics sink OFF, "
          "once per interval; still exactly one emission site")


def test_non_interference(ai):
    from developmental_ai.infra.memory_census import census
    before = _fingerprint(ai)
    census(ai)
    census(ai, deep=True)
    after = _fingerprint(ai)
    moved = [k for k in before if before[k] != after[k]]
    assert not moved, f"census changed learner state: {moved}"
    # Falsification: the fingerprint is sensitive — one RNG draw moves it.
    import torch
    torch.rand(1)
    assert _fingerprint(ai)["torch_rng"] != after["torch_rng"]
    print(f"  G. {len(before)} learner fingerprints identical across a fast "
          f"and a deep census (and the fingerprint does detect one RNG draw)")


def test_never_raises():
    from developmental_ai.infra.memory_census import census, MemoryCensusHook

    class Hostile:
        def __getattr__(self, k):
            raise RuntimeError("proxy: " + k)

        @property
        def replay_buffer(self):
            raise RuntimeError("property must not be called")

    h = Hostile()
    h.__dict__["world"] = {"x": np.zeros(10), "self": h}
    rec = census(h, deep=True)
    assert rec["replay"] == {"present": False}, rec["replay"]
    assert rec["objects"]["numpy_total_bytes"] == 80
    bad = MemoryCensusHook({"memory_census_path":
                            "/dev/null/cannot/memory_census.jsonl"})
    assert bad.tick(h) is False and bad.failures == 1
    print("  H. hostile object (raising __getattr__ and property, cycle) and "
          "an unwritable path: no exception, no property called")


def main():
    ai = _agent()
    try:
        ai._collect_segment()             # the live body fills real rows
        _fill_and_grow(ai)
        rec = test_sections_and_replay(ai)
        test_budget_accounting(rec, ai)
        test_modules(rec, ai)
        test_hook(ai)
        test_emit_site(ai)
        test_non_interference(ai)
        test_never_raises()
    finally:
        try:
            ai.close()
        except Exception:
            pass
        shutil.rmtree(TMP, ignore_errors=True)
    print("[memory_census_smoke] ALL PASS")


if __name__ == "__main__":
    main()
