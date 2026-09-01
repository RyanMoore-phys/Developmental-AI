"""Empowerment probe smoke (Mac-ok; fake world model, tiny torch).

Contracts:
  1. Free-vs-trapped: a fake model where distinct actions produce
     distinct latent updates scores strictly higher than one where EVERY
     action maps to the same update (score ~0); success clears
     empowerment_score.last_error; harvested policy actions are one-hot
     [n_seq, action_dim].
  2. Exception containment: a raising model returns 0.0 with the repr on
     empowerment_score.last_error, and a subsequent success clears it.
  3. EmpowermentPotential: first update returns 1.0; monotone-rising
     scores keep phi <= 1; a later smaller score gives phi < 1.
  4. Signature flexibility: a model WITHOUT a `knowledge` parameter is
     still probed correctly (no TypeError leaking into the 0.0 path).
"""
import torch

from developmental_ai.infra.empowerment import (
    EmpowermentPotential, empowerment_score)

H, Z, ACT = 5, 3, 4
LAT = H + Z


class FakeWorldModel:
    """Latent evolves as latent += action @ W per imagined step.

    trapped=True copies one row of W across all actions, so every action
    produces the identical update and rollouts can never diverge."""

    def __init__(self, trapped: bool = False, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(ACT, LAT, generator=g)
        if trapped:
            self.W = self.W[0:1].repeat(ACT, 1)
        self.seen_actions = []

    def imagine_trajectory(self, start, policy_fn, horizon=8,
                           knowledge=None):
        latent = torch.cat([start["h"], start["z"]], dim=-1)
        for _ in range(horizon):
            action = policy_fn(latent)
            self.seen_actions.append(action)
            latent = latent + action @ self.W
        return {"latents": latent}


class NoKnowledgeModel(FakeWorldModel):
    """Same dynamics, but imagine_trajectory refuses the knowledge kwarg
    (older-fork signature)."""

    def imagine_trajectory(self, start, policy_fn, horizon=8):
        return FakeWorldModel.imagine_trajectory(
            self, start, policy_fn, horizon=horizon)


class BrokenModel:
    def imagine_trajectory(self, *a, **k):
        raise RuntimeError("device-side assert (synthetic)")


def _mk_state(seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"h": torch.randn(1, H, generator=g),
            "z": torch.randn(1, Z, generator=g)}


def test_free_vs_trapped():
    torch.manual_seed(7)
    free = FakeWorldModel(trapped=False)
    s_free = empowerment_score(free, _mk_state(), ACT, n_seq=8, horizon=8)
    assert empowerment_score.last_error is None, empowerment_score.last_error
    torch.manual_seed(7)
    trapped = FakeWorldModel(trapped=True)
    s_trap = empowerment_score(trapped, _mk_state(), ACT, n_seq=8, horizon=8)
    assert s_free > s_trap, (s_free, s_trap)
    assert s_trap < 1e-6, s_trap          # identical updates: zero spread
    a = free.seen_actions[0]
    assert a.shape == (8, ACT), a.shape
    assert torch.all(a.sum(dim=-1) == 1.0) and torch.all(
        (a == 0) | (a == 1)), "actions must be one-hot"
    print(f"[infra-empowerment] 1. free {s_free:.3f} > trapped "
          f"{s_trap:.2e}; one-hot [n_seq, action_dim] actions")


def test_exception_containment():
    s = empowerment_score(BrokenModel(), _mk_state(), ACT)
    assert s == 0.0, s
    assert empowerment_score.last_error is not None
    assert "RuntimeError" in empowerment_score.last_error, \
        empowerment_score.last_error
    # success afterwards clears the error slot
    empowerment_score(FakeWorldModel(), _mk_state(), ACT)
    assert empowerment_score.last_error is None
    print("[infra-empowerment] 2. broken model -> 0.0 + last_error set; "
          "next success clears it")


def test_potential_normalisation():
    p = EmpowermentPotential(decay=0.999)
    assert p.update(0.5) == 1.0            # first call: its own max
    phis = [p.update(s) for s in (0.6, 0.7, 0.8)]
    assert all(phi <= 1.0 for phi in phis), phis
    low = p.update(0.4)
    assert low < 1.0, low                  # smaller than running max
    assert abs(low - 0.4 / (0.8 * 0.999)) < 1e-9, low
    print(f"[infra-empowerment] 3. potential: first=1.0, monotone "
          f"phis<=1 {['%.3f' % x for x in phis]}, later-smaller "
          f"phi={low:.3f}<1")


def test_signature_flexibility():
    torch.manual_seed(7)
    s = empowerment_score(NoKnowledgeModel(), _mk_state(), ACT)
    assert empowerment_score.last_error is None, empowerment_score.last_error
    assert s > 0.0, s
    print(f"[infra-empowerment] 4. knowledge-less signature probed ok "
          f"(score {s:.3f})")


if __name__ == "__main__":
    for fn in (test_free_vs_trapped, test_exception_containment,
               test_potential_normalisation, test_signature_flexibility):
        fn()
    print("[infra-empowerment] ALL PASS")
