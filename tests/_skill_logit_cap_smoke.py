"""Skill copies sample under the SHARED policy's logit-spread cap.

LIVE INCIDENT (13 h of shadow data, investigated 2026-10-04). Action 17
(turn-left micro) was 40% of the first 20 sampled actions of the run —
BEFORE any PPO update. A fresh policy never puts more than ~5% on one action
(odds of 8/20 by chance ~3e-6), and the policy itself is not resumed
(developmental_loop refuses), so the bias had to enter from the skill bank.

The shared policy bounds its action logits with `_bound_logits`
(`policy.logit_range`, 5.0 in minecraft_skybot.yaml): spread <= R, so the
best/worst probability ratio is <= e^R and max_prob over N actions is
<= e^R / (e^R + N - 1). It can be confident, never CERTAIN (saturation is a
one-way door: at max_prob 1.000 the entropy gradient is 9.5e-07).
`SkillOptionBank.skill_action` — which executes a frozen skill copy for every
step of an option — sampled straight from `actor.action_head(...)` and never
applied that bound. CLAUDE.md 4.6: a skill IS a byte copy of the shared
policy, so a copy taken from a saturated policy re-injected certainty the
live policy is forbidden to have.

FIX: the bank adopts the shared policy's OWN bound method on every
`OptionExecutor.act` (same function, same live `logit_range` — the two
cannot drift) and applies it to the skill's full head (delta included)
BEFORE masking, the same order `select_action` uses. Rows the rescale does
not engage on keep their original logits, so a well-behaved skill samples
byte-identically. It is a rescale, never a gate: nothing to latch (4.1).

WARM-START (developmental_loop ~3033): loads a stored skill's weights INTO
the shared policy. `logit_range` is a constructor attribute, not a weight,
so the cap survives the load and the masked acting path bounds it.

Contracts
  A. Pre-fix witness: an extreme-logit skill's raw-head max_prob exceeds the
     cap-implied maximum (this is what skill_action sampled from before).
  B. Production wiring: OptionExecutor.act adopts the shared policy's own
     `_bound_logits` (same function object, same instance).
  C. Capped: the extreme skill's sampling probs are <= the cap-implied max
     and equal the shared policy's own masked probs for the same weights.
  D. No drift: changing policy.logit_range moves the skill cap with it.
  E. Normal skill (spread inside the cap): probs BYTE-identical to raw head.
  F. Shared policy path unchanged: masked select_action log_prob ==
     log_softmax(bound(head)) by hand.
  G. Warm-start: extreme weights loaded into a capped shared policy act
     under the cap (logit_range is not part of the state dict).
  H. No policy adopted (or a double without the bound) -> identity.

Run:  PYTHONPATH=. ./venv/bin/python tests/_skill_logit_cap_smoke.py
"""
import contextlib
import math
import shutil
import sys
import tempfile

import numpy as np
import torch

OBS, P, K, R = 12, 6, 2, 5.0


def _cap_max(R_, n):
    return math.exp(R_) / (math.exp(R_) + n - 1)


def _mk_policy(action_dim, logit_range=R, extreme=None, seed=0):
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    torch.manual_seed(seed)
    pol = StandaloneActorCritic(
        obs_dim=OBS, action_dim=action_dim, hidden_dim=16, continuous=False,
        knowledge_dim=0, device=torch.device("cpu"), logit_range=logit_range)
    if extreme is not None:
        with torch.no_grad():
            pol.actor.action_head.bias[extreme] += 40.0
    return pol


@contextlib.contextmanager
def _capture_logits():
    """Record the logits every Categorical is built from."""
    seen = []
    orig = torch.distributions.Categorical

    class _Rec(orig):
        def __init__(self, probs=None, logits=None, validate_args=None):
            if logits is not None:
                seen.append(logits.detach().clone())
            super().__init__(probs=probs, logits=logits,
                             validate_args=validate_args)

    torch.distributions.Categorical = _Rec
    try:
        yield seen
    finally:
        torch.distributions.Categorical = orig


def _bind(tmp, pol, name):
    from developmental_ai.policy.options import SkillOptionBank
    from developmental_ai.skill_bank.skill_bank import SkillBank
    shutil.rmtree(tmp, ignore_errors=True)
    sb = SkillBank(storage_dir=tmp)
    sb.save_skill(name, "skill", pol.get_state_dict(), success_rate=0.6,
                  total_episodes=5, dedup=True, obs_dim=OBS, action_dim=P)
    ob = SkillOptionBank(sb, OBS, P, K, torch.device("cpu"))
    ob.refresh_slots()
    assert ob.slots[0] is not None, "skill did not bind"
    return ob


def _raw_probs(pol, obs):
    with torch.no_grad():
        x = torch.from_numpy(obs).reshape(1, -1)
        return torch.softmax(pol.actor.action_head(pol.actor.shared(x)), -1)


def _skill_probs(ob, obs):
    with _capture_logits() as seen:
        ob.skill_action(0, obs)
    assert seen, "skill_action built no Categorical"
    return torch.softmax(seen[-1], -1)


def main():
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    from developmental_ai.policy.options import OptionExecutor
    tmp = tempfile.mkdtemp(prefix="skill_cap_")
    obs = np.random.RandomState(0).randn(OBS).astype(np.float32)
    cap = _cap_max(R, P)
    try:
        # A — the pre-fix witness
        ext = _mk_policy(P, extreme=3)
        raw = _raw_probs(ext, obs)
        assert float(raw.max()) > cap + 0.02, (float(raw.max()), cap)
        print(f"  A. pre-fix witness: raw skill-head max_prob "
              f"{float(raw.max()):.6f} > cap-implied {cap:.4f}")

        # B — production wiring through OptionExecutor.act
        ob = _bind(tmp, ext, "ach_00_extreme")
        meta = _mk_policy(P + K, seed=1)
        ex = OptionExecutor(ob, num_envs=1, cfg={"max_option_steps": 50},
                            gamma=0.99)
        ex.runtimes[0].open(0, ob.slots[0]["skill_id"], 0, {
            "obs": obs, "meta_action": P, "log_prob": -1.0, "value": 0.0,
            "kv": None, "mask": np.ones(P + K, bool)})
        with _capture_logits() as seen:
            acts = ex.act([obs], None, meta, 1)
        assert 0 <= acts[0] < P, acts
        assert seen, "skill_action was not reached through act()"
        p_act = torch.softmax(seen[-1], -1)
        assert float(p_act.max()) <= cap + 1e-6, \
            f"skill sampled uncapped: max_prob {float(p_act.max())} > {cap}"
        _lb = getattr(ob, "logit_bound", None)
        assert _lb is not None, "act() did not adopt the bound"
        assert _lb.__func__ is StandaloneActorCritic._bound_logits
        assert _lb.__self__ is meta, "bound not the live policy's"
        print(f"  B. act() adopts policy._bound_logits (same fn, same "
              f"instance); skill max_prob via act {float(p_act.max()):.4f}")

        # C — capped, and identical to the shared policy's own probs
        ob.adopt_logit_bound(ext)          # same R, same weights as skill
        ps = _skill_probs(ob, obs)
        assert float(ps.max()) <= cap + 1e-6, (float(ps.max()), cap)
        _, info = ext.select_action(obs, action_mask=np.ones(P, bool))
        with _capture_logits() as seen:
            ext.select_action(obs, action_mask=np.ones(P, bool))
        p_shared = torch.softmax(seen[-1], -1)
        assert torch.allclose(ps, p_shared, atol=1e-6), (ps, p_shared)
        print(f"  C. capped skill max_prob {float(ps.max()):.4f} <= "
              f"{cap:.4f}, == shared policy's probs (atol 1e-6)")

        # D — no drift: the cap follows the policy's live value
        meta.logit_range = 3.0
        ob.adopt_logit_bound(meta)
        ps3 = _skill_probs(ob, obs)
        assert float(ps3.max()) <= _cap_max(3.0, P) + 1e-6, float(ps3.max())
        assert float(ps3.max()) < float(ps.max()) - 1e-3
        meta.logit_range = R
        print(f"  D. logit_range 5->3 moves the skill cap: max_prob "
              f"{float(ps3.max()):.4f} <= {_cap_max(3.0, P):.4f}")

        # E — a normal skill inside the cap: byte-identical
        nrm = _mk_policy(P, seed=2)
        rn = _raw_probs(nrm, obs)
        _l = torch.log(rn)
        assert float(_l.max() - _l.min()) < R, "fixture not inside the cap"
        obn = _bind(tmp, nrm, "ach_01_normal")
        obn.adopt_logit_bound(meta)
        pn = _skill_probs(obn, obs)
        assert torch.equal(pn, rn), (pn, rn)
        print("  E. in-range skill: probs byte-identical to the raw head")

        # F — shared policy path unchanged
        m = np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
        a, info = meta.select_action(obs, action_mask=m)
        with torch.no_grad():
            x = torch.from_numpy(obs).reshape(1, -1)
            lg = meta.actor.action_head(meta.actor.shared(x))
            c = lg - lg.mean(-1, keepdim=True)
            sp = float(c.max() - c.min())
            c = c * min(1.0, R / (sp + 1e-8))
            c = c.masked_fill(~torch.from_numpy(m).reshape(1, -1), -1e9)
            lp = float(torch.log_softmax(c, -1)[0, a])
        assert abs(lp - info["log_prob"]) < 1e-5, (lp, info["log_prob"])
        print("  F. shared masked path == log_softmax(bound(head)) by hand")

        # G — warm-start: extreme weights into a capped shared policy
        fresh = _mk_policy(P, seed=3)
        fresh.load_state_dict(ext.get_state_dict())
        assert fresh.logit_range == R, "load_state_dict changed logit_range"
        with _capture_logits() as seen:
            fresh.select_action(obs, action_mask=np.ones(P, bool))
        pw = torch.softmax(seen[-1], -1)
        assert float(pw.max()) <= cap + 1e-6, float(pw.max())
        print(f"  G. warm-started extreme weights act capped: max_prob "
              f"{float(pw.max()):.4f} (raw {float(raw.max()):.6f})")

        # H — no bound adopted / double without one -> identity
        obh = _bind(tmp, ext, "ach_02_unadopted")
        ph = _skill_probs(obh, obs)
        assert torch.equal(ph, raw), "unadopted bank must be identity"
        obh.adopt_logit_bound(object())
        assert obh.logit_bound is None
        print("  H. no policy adopted / boundless double -> identity")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[skill_logit_cap_smoke] ALL PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e!r}")
        sys.exit(1)
