"""Standby actor: real-time games keep being played while SkyBot learns.

Born 2026-10-11 — stage R3 of docs/UNIVERSAL_GAMES.md §10. User decision:
the game does not wait for SkyBot and acting is independent of learning.
Between segments the loop runs the PPO update, dream training, reports and
checkpoints; on a real-time clock each of those seconds used to be SkyBot
standing still holding its last buttons. The standby actor plays the
real-time streams for exactly that window. Fakes only — runs anywhere.

Contracts:
    A. IT PLAYS: while running, every handed-over stream keeps deciding and
       stepping, on its own thread, with no call from the loop.
    B. HANDOFF IS THE TRUTH: stop() returns each stream's NEWEST frame and
       the belief updated through every step it took (the loop resumes from
       where the game really is, not from where it left it).
    C. A NEW LIFE IS A BOUNDARY: a terminated step zeroes that stream's
       belief, reset()s the env, and is reported as a boundary.
    D. NEVER TAKES THE RUN DOWN: an env that raises stops only that stream
       (error recorded); the others keep playing; stop() still returns.
    E. LEARNING LANDS: set_policy() swaps the policy mid-run (the PPO update
       finishing) and later decisions use it.
    F. IT ONLY ACTS: knowledge goes to stream 0 only, each stream's mask is
       passed, and the actor exposes no replay / rollout / curiosity hooks.
    G. LOOP WIRING (source): begin before the segment-end PPO update, refresh
       right after it, end (handoff) before the next segment reads its
       carry, close() stops it first, the replay gap flag is honoured in
       both bodies (count 2), and _reset_stream_potentials performs every
       per-stream reset the ADVANCE block performs on a death.

Run: PYTHONPATH=. python tests/_standby_actor_smoke.py
"""
import os
import sys
import threading
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.core.standby_actor import StandbyActor      # noqa: E402

NAME = "standby-actor"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


class FakeRSSM:
    def observe_step(self, state, action, emb):
        return {"h": state["h"] + 1.0, "z": state["z"]}, None

    def get_latent(self, state):
        return torch.cat([state["h"], state["z"].reshape(state["z"].shape[0], -1)], -1)


class FakeWM:
    rssm = FakeRSSM()

    def embed(self, x, proprio):
        return x.mean(dim=1, keepdim=True)


class FakePolicy:
    def __init__(self, tag):
        self.tag, self.calls = tag, []

    def select_action(self, obs, knowledge=None, **kw):
        self.calls.append((knowledge, kw.get("action_mask"), kw.get("feats") is not None))
        time.sleep(0.002)
        return 1, {"log_prob": 0.0, "value": 0.0}


class FakeEnv:
    def __init__(self, die_at=None, raise_at=None):
        self.t, self.die_at, self.raise_at, self.resets = 0, die_at, raise_at, 0
        self.threads = set()

    def step(self, a):
        self.threads.add(threading.current_thread().name)
        self.t += 1
        if self.raise_at is not None and self.t >= self.raise_at:
            raise RuntimeError("engine died")
        term = self.die_at is not None and self.t == self.die_at
        return np.full(4, self.t, np.float32), 0.0, term, False, {"proprio": np.ones(2) * self.t}

    def reset(self):
        self.resets += 1
        return np.full(4, -1.0, np.float32), {"proprio": np.zeros(2)}


def make(envs, policy):
    return StandbyActor(
        envs=envs, policy=policy, world_model=FakeWM(), wm_lock=threading.RLock(),
        wm_proprio_fn=lambda infos, n: None, device="cpu", action_dim=3,
        beliefs={e: {"h": torch.zeros(1, 2), "z": torch.zeros(1, 2)} for e in envs},
        obs={e: np.zeros(4, np.float32) for e in envs},
        infos={e: {} for e in envs},
        aug_proprio={e: np.zeros(5, np.float32) for e in envs},
        masks={e: np.array([True, True, e == 0]) for e in envs},
        knowledge=np.ones(3, np.float32), use_feats=True)


def main():
    print(f"[{NAME}]")
    # ---- A, B
    envs = {0: FakeEnv(), 1: FakeEnv()}
    pol = FakePolicy("old")
    sb = make(envs, pol)
    sb.start()
    time.sleep(0.3)
    out = sb.stop()
    d0, d1 = out[0]["decisions"], out[1]["decisions"]
    check(d0 > 10 and d1 > 10, "A. both streams played on their own (%d, %d decisions)" % (d0, d1))
    check(envs[0].threads == {"standby-actor"}, "A. stepped from the standby thread, not the loop")
    check(float(out[0]["obs"][0]) == envs[0].t and float(out[0]["h"][0, 0]) == envs[0].t,
          "B. handoff = newest frame (t=%d) and a belief updated through every step" % envs[0].t)
    check(np.allclose(out[1]["aug"][:2], envs[1].t),
          "B. body sense refreshed from the newest step")
    # ---- C
    envs = {0: FakeEnv(die_at=5)}
    sb = make(envs, FakePolicy("p"))
    sb.start(); time.sleep(0.15); out = sb.stop()
    check(out[0]["boundary"] and envs[0].resets == 1,
          "C. a terminated step is a boundary and reset() the env")
    # ---- D
    envs = {0: FakeEnv(raise_at=3), 1: FakeEnv()}
    sb = make(envs, FakePolicy("p"))
    sb.start(); time.sleep(0.2)
    t0 = time.time(); out = sb.stop(); dt = time.time() - t0
    check(0 in sb.errors and out[1]["decisions"] > out[0]["decisions"],
          "D. a raising env stops only its stream (%s); the other kept playing" % sb.errors[0])
    check(dt < 1.0, "D. stop() returns promptly (%.2fs)" % dt)
    # ---- E
    envs = {0: FakeEnv()}
    old, new = FakePolicy("old"), FakePolicy("new")
    sb = make(envs, old)
    sb.start(); time.sleep(0.1)
    sb.set_policy(new); n_old = len(old.calls); time.sleep(0.1)
    sb.stop()
    check(len(new.calls) > 5 and len(old.calls) <= n_old + 1,
          "E. set_policy swaps mid-run: %d decisions on the new policy" % len(new.calls))
    # ---- F
    envs = {0: FakeEnv(), 1: FakeEnv()}
    p = FakePolicy("p")
    sb = make(envs, p)
    sb.start(); time.sleep(0.1); sb.stop()
    k0 = [c for c in p.calls if c[0] is not None]
    check(len(k0) > 0 and all(c[2] for c in p.calls), "F. feats (belief latent) passed")
    masks = {tuple(c[1]) for c in p.calls}
    check(masks == {(True, True, True), (True, True, False)},
          "F. each stream's own mask passed: %s" % masks)
    check(not any(hasattr(sb, a) for a in ("replay_buffer", "store_transition", "curiosity")),
          "F. it only acts: no learner hooks")
    # ---- G
    loop = open(os.path.join(ROOT, "developmental_ai/core/developmental_loop.py")).read()
    i_begin = loop.index("        self._standby_begin(step_infos)")
    i_ppo = loop.index("pm = self.policy.train_step(", i_begin)
    i_ref = loop.index("self._standby_refresh_policy()", i_ppo)
    check(i_begin < i_ppo < i_ref, "G. begin before the PPO update; refresh right after")
    i_end = loop.index("        self._standby_end()\n        envs = self._parallel_envs\n"
                       "        rssm_state = self._ll.rssm_state")
    check(i_end > 0, "G. handoff happens before the segment reads its carry")
    c0 = loop.index("    def close(self) -> None:")
    check(loop.index('_sb = getattr(self, "_standby", None)', c0) < loop.index("self._shadow.close()", c0),
          "G. close() stops the standby actor before anything else")
    check(loop.count("or bool(self._rt_gap.pop(e_i, False))") == 2,
          "G. replay gap flag honoured in both bodies")
    adv = loop[loop.index("                if dones[e_i] or _crashed:"):
               loop.index("                    if restarted[e_i] and not _crashed:")]
    hlp = loop[loop.index("    def _reset_stream_potentials"):]
    hlp = hlp[:hlp.index("\n    def ", 10)]
    stmts = ["self._sc_phi.pop(e_i, None)", "self._persist_phi = 0.0",
             "self._reach_phi = 0.0", "self._pitch_level_phi = None",
             "self._gui_run = 0", "self._gui_dwell_phi = None",
             "self._habit_prev.clear()", "self._mine_by_env.pop(e_i, None)",
             "self._new_break_by_env.pop(e_i, None)",
             "self._loginv_by_env.pop(e_i, None)", "self._prev_mine.clear()",
             "self.option_executor.clear(e_i, self.total_timesteps)",
             "self._drop_curiosity_batch()"]
    missing = [x for x in stmts if x not in adv or x not in hlp]
    check(not missing, "G. _reset_stream_potentials mirrors the ADVANCE death resets%s"
          % ("" if not missing else ": MISSING %s" % missing))
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
