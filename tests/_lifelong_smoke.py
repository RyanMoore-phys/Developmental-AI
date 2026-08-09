"""Lifelong controller smoke (Mac-ok) — the continuous-run primitives.

Contracts:
  1. Disabled by default -> every method is inert (episodic path untouched):
     should_stop False, decay a no-op, budget passes the configured value.
  2. State decay: carried RSSM h/z, the broadcaster mask, and the scaffold
     belief each shrink by state_decay per segment; decay=1.0 is a no-op;
     within [0,1] it never grows state (no spin-out).
  3. Death reset: a true terminal zeroes exactly the dead env's RSSM row and
     clears its achievement-mask row, leaving other envs untouched.
  4. Forever-mode: budget()->None (run forever); should_stop fires on the
     stop-file and on a requested signal; NOT in non-forever mode.
  5. Working-set variables are surfaced (24/48 defaults, configurable).
"""
import os
import shutil

import numpy as np
import torch


def _ctrl(**kw):
    from developmental_ai.core.lifelong import LifelongController
    return LifelongController(kw)


class _FakeBroadcaster:
    def __init__(self, n, slots):
        self._achieved_now = np.ones((n, slots), dtype=np.float32)
        self.cleared = []

    def clear_stream(self, i):
        self._achieved_now[i, :] = 0.0
        self.cleared.append(i)


class _FakeScaffold:
    def __init__(self):
        self._phi_prev = 0.8


def test_disabled_is_inert():
    c = _ctrl(enabled=False)
    assert c.should_stop() is False
    assert c.budget(3_000_000) == 3_000_000, "disabled changed the budget"
    # a disabled controller still hands the configured budget straight back
    # and never requests a stop -> the episodic run() loop is unchanged.
    assert c.budget(None) is None
    c2 = _ctrl(enabled=True, forever=False)
    assert c2.budget(500) == 500, "non-forever changed the budget"
    print("  1. disabled is fully inert (episodic path untouched)")


class _MaskBroadcaster:
    """Real decay_masks semantics for the mask-fade contract."""
    def __init__(self, n, slots):
        self._achieved_now = np.ones((n, slots), dtype=np.float32)
        self.cleared = []
    def decay_masks(self, f):
        self._achieved_now *= float(f)
    def clear_stream(self, i):
        self._achieved_now[i, :] = 0.0
        self.cleared.append(i)


def test_state_decay():
    # LifelongState (built via the controller factory) owns carry + decay
    c = _ctrl(enabled=True, state_decay=0.9)
    rssm = {"h": torch.ones(2, 4) * 2.0, "z": torch.ones(2, 4)}
    bc = _MaskBroadcaster(2, 3)
    ll = c.make_state(rssm, broadcaster=bc)
    ll.decay()
    assert torch.allclose(rssm["h"], torch.full((2, 4), 1.8)), rssm["h"][0]
    assert np.allclose(bc._achieved_now, 0.9), bc._achieved_now
    # repeated decay shrinks monotonically toward 0 (never grows/spins out)
    prev = rssm["h"].clone()
    for _ in range(80):
        ll.decay()
        assert torch.all(rssm["h"] <= prev + 1e-6), "state grew under decay"
        prev = rssm["h"].clone()
    assert torch.all(rssm["h"] < 0.02), "state didn't diminish over time"
    # decay >= 1.0 is a no-op
    c2 = _ctrl(enabled=True, state_decay=1.0)
    r2 = {"h": torch.ones(1, 2)}
    c2.make_state(r2).decay()
    assert torch.all(r2["h"] == 1.0)
    # h_norm accessor (the stability signal to instrument)
    assert ll.h_norm() >= 0.0
    print("  2. state decay ok (LifelongState owns RSSM+mask; no-op at 1.0)")


def test_death_reset():
    c = _ctrl(enabled=True)
    rssm = {"h": torch.ones(3, 4), "z": torch.ones(3, 4)}
    bc = _MaskBroadcaster(3, 5)
    ll = c.make_state(rssm, broadcaster=bc)
    ll.death_reset(1)                   # env 1 dies
    assert torch.all(rssm["h"][1] == 0.0) and torch.all(rssm["z"][1] == 0.0)
    assert torch.all(rssm["h"][0] == 1.0) and torch.all(rssm["h"][2] == 1.0), \
        "death reset touched the wrong env rows"
    assert np.all(bc._achieved_now[1] == 0.0)
    assert np.all(bc._achieved_now[0] == 1.0)
    assert bc.cleared == [1]
    print("  3. death reset ok (only the dead env's row zeroed)")


def test_ppo_bootstrap_byte_identical():
    """EDIT 8: last_value defaults to 0.0 -> GAE bit-for-bit identical to the
    pre-change advantages; a non-zero last_value changes the final step."""
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    pol = StandaloneActorCritic(obs_dim=6, action_dim=4, hidden_dim=16,
                                continuous=False, device=torch.device("cpu"))
    pol.gamma, pol.gae_lambda = 0.99, 0.95
    r = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    v = np.array([0.3, 0.4, 0.2], dtype=np.float32)
    d = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # ends mid-trajectory
    base = pol._compute_gae(r, v, d)                  # default last_value=0.0
    # hand recursion with a 0.0 tail bootstrap == base (byte-identical)
    manual = np.zeros(3, dtype=np.float32); last = 0.0
    for t in reversed(range(3)):
        nv = 0.0 if t == 2 else v[t + 1]
        delta = r[t] + 0.99 * nv * (1 - d[t]) - v[t]
        manual[t] = delta + 0.99 * 0.95 * (1 - d[t]) * last
        last = manual[t]
    assert np.array_equal(base, manual), "default bootstrap not byte-identical"
    boot = pol._compute_gae(r, v, d, last_value=5.0)
    assert boot[-1] != base[-1] and boot[-1] > base[-1], (
        "non-zero last_value did not change the final advantage")
    # compute_last_value returns a real scalar without storing anything
    lv = pol.compute_last_value(np.zeros(6, dtype=np.float32))
    assert isinstance(lv, float)
    assert len(pol.rollout_obs) == 0, "compute_last_value stored a transition"
    print("  4b. PPO bootstrap ok (0.0 byte-identical, V(s) changes tail)")


def test_forever_mode():
    d = "/tmp/lifelong_stop_test"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    stop = os.path.join(d, "STOP")
    c = _ctrl(enabled=True, forever=True, stop_file=stop)
    assert c.budget(3_000_000) is None, "forever mode kept a finite budget"
    assert c.should_stop() is False, "stopped with no stop-file/signal"
    open(stop, "w").close()
    assert c.should_stop() is True, "stop-file didn't trigger stop"
    os.remove(stop)
    assert c.should_stop() is False
    c._stop_requested = True             # simulate SIGTERM
    assert c.should_stop() is True
    # non-forever never stops via this path
    c2 = _ctrl(enabled=True, forever=False, stop_file=stop)
    open(stop, "w").close()
    assert c2.should_stop() is False, "non-forever stopped on a stop-file"
    assert c2.budget(123) == 123
    print("  4. forever-mode ok (None budget, stop-file + signal)")


def test_working_set_vars():
    c = _ctrl(enabled=True, active_skills=24, active_goals=48)
    assert c.active_skills == 24 and c.active_goals == 48
    st = c.stats()
    assert st["active_skills"] == 24 and st["active_goals"] == 48
    # raise-later: bigger values flow through
    c2 = _ctrl(enabled=True, active_skills=96, active_goals=192)
    assert c2.active_skills == 96 and c2.active_goals == 192
    print("  5. working-set vars ok (24/48 default, raiseable)")


if __name__ == "__main__":
    for fn in (test_disabled_is_inert, test_state_decay, test_death_reset,
               test_ppo_bootstrap_byte_identical, test_forever_mode,
               test_working_set_vars):
        print(f"[lifelong-smoke] {fn.__name__}")
        fn()
    print("[lifelong-smoke] ALL PASS")
