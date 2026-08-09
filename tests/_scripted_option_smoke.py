"""Scripted (bootstrap) option smoke — the chop_trunk macro.

  1. reserve_scripted takes the LAST free slot, exposes the macro (brief aim,
     then HOLD attack), is offered by mask ONLY when preconditions hold and
     with the competence warmup gate BYPASSED, and survives refresh_slots.
  2. OptionExecutor opens the scripted option when the policy invokes its slot
     and drives the fixed macro (aim -> hold attack) using the option step
     count.
"""
import numpy as np
import torch

from developmental_ai.policy.options import SkillOptionBank, OptionExecutor


class SBStub:
    skills = {}


SPEC = {"skill_id": "chop_trunk", "name": "chop trunk (scripted)",
        "preconditions": {"tree_visible": True, "object_adjacent": True},
        "aim_steps": 1, "aim_action": 8, "act_action": 5}


def test_reserve_mask_macro():
    bank = SkillOptionBank(SBStub(), obs_dim=8, P=10, K=4,
                           device=torch.device("cpu"))
    slot = bank.reserve_scripted(dict(SPEC))
    assert slot == 3, f"scripted should take the LAST free slot: {slot}"
    assert bank.is_scripted(slot)
    # macro: 1 aim step (look down = 8), then hold attack (= 5)
    assert bank.scripted_action(slot, 0) == 8
    assert bank.scripted_action(slot, 1) == 5
    assert bank.scripted_action(slot, 25) == 5
    # offered when preconditions hold, EVEN with the competence gate high and
    # no competence recorded for chop_trunk (scripted bypasses the warmup gate)
    m = bank.mask(predicates={"tree_visible": True, "object_adjacent": True},
                  min_match=0.5, competence={}, competence_floor=0.9)
    assert m[bank.P + slot], "scripted option not offered (competence bypass?)"
    # NOT offered when the trunk isn't reachable
    m2 = bank.mask(predicates={"tree_visible": False, "object_adjacent": False},
                   min_match=0.5, competence={}, competence_floor=0.9)
    assert not m2[bank.P + slot], "offered despite failed preconditions"
    # refresh_slots must never displace the reserved scripted slot
    bank.refresh_slots()
    assert bank.slots[slot] is not None
    assert bank.slots[slot]["skill_id"] == "chop_trunk"
    print("  1. reserve + macro + mask(comp-bypass, precond-gated) + preserve ok")


def test_executor_drives_macro():
    bank = SkillOptionBank(SBStub(), obs_dim=8, P=10, K=4,
                           device=torch.device("cpu"))
    slot = bank.reserve_scripted(dict(SPEC))
    ex = OptionExecutor(bank, num_envs=1, cfg={
        "max_option_steps": 5, "gate_by_preconditions": False,
        "scouts_use_options": True})

    class Pol:
        def select_action(self, obs, knowledge=None, action_mask=None):
            return bank.P + slot, {"log_prob": 0.0, "value": 0.0}

    obs = [np.zeros(8, dtype=np.float32)]
    a1 = ex.act(obs, None, Pol(), timestep=0)      # opens + FIRST macro action
    assert a1[0] == 8, f"first macro action should be aim (look-down): {a1}"
    assert ex.runtimes[0].active, "option did not open"
    ex.runtimes[0].steps_done = 2                  # option has now run a bit
    a2 = ex.act(obs, None, Pol(), timestep=1)      # continuing -> hold attack
    assert a2[0] == 5, f"continuing macro action should be attack: {a2}"
    print("  2. executor opens the scripted option + drives aim -> hold attack")


if __name__ == "__main__":
    for fn in (test_reserve_mask_macro, test_executor_drives_macro):
        print(f"[scripted-option] {fn.__name__}")
        fn()
    print("[scripted-option] ALL PASS")
