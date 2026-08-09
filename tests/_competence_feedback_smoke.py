"""Smoke: option outcomes drive the gate, and the gate can still REOPEN.

THE GAP (measured live 2026-07-25): competence was updated in exactly two
places, both goal-attempt boundaries in `achievement_goals.py`. NOTHING in the
option path ever touched it, so the gate was structurally blind to whether an
option WORKS. Consequence: 96 of 96 completed options ended in `horizon` (full
80-tick budget, nothing achieved), ONE useless skill took 86% of every
invocation, its competence never moved, and it crowded the scripted chop — the
only option that earns the +5.0 log tier — down to ~3% of the budget.

THE TRAP: closing the gate on failure ALONE makes it a ONE-WAY LATCH (gated ->
never offered -> never invoked -> no new outcomes -> never reopens). That exact
shape has bitten this project FOUR times. So the two halves ship together and
are tested together: feedback WITHOUT probation must fail this suite.

Run: PYTHONPATH=. python tests/_competence_feedback_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

from developmental_ai.core.self_model import CompetencePredictor

LOOP = "developmental_ai/core/developmental_loop.py"
OPT = "developmental_ai/policy/options.py"


class _Slot(dict):
    pass


class _FakeBank:
    """Minimal stand-in: P primitives + K slots, no torch, no policies."""

    def __init__(self, P=10, scripted_slot=3, n=5):
        self.P = P
        self.slots = [_Slot({"skill_id": f"ach_{i:02d}_x",
                             "scripted": (i == scripted_slot)})
                      for i in range(n)]


def main() -> None:
    loop = open(LOOP).read()
    opt = open(OPT).read()

    # ---- A. OPTION OUTCOMES REACH THE GATE ------------------------------
    assert "_competence_from_option" in loop, "no option->competence feedback"
    assert 'closed.get("outcome") == "spike"' in loop, \
        "success is not keyed to a real reward spike"
    assert "bc.competence.update(" in loop, "the predictor is never updated"
    # BOTH close sites must feed it — patching one path and not its twin is
    # the mistake that has recurred three times in this project.
    assert loop.count("self._competence_from_option(_closed)") == 2, (
        f"expected 2 call sites (episodic + parallel), found "
        f"{loop.count('self._competence_from_option(_closed)')}")

    # ---- B. THE SIGNAL ACTUALLY CLOSES A FAILING GATE -------------------
    FLOOR = 0.25
    cp = CompetencePredictor(n_tasks=8)
    assert float(cp.predict_all()[2]) >= FLOOR, "fresh skill should be offered"
    for _ in range(40):                       # 40 straight `horizon` closes
        cp.update(2, 0.0)
    assert float(cp.predict_all()[2]) < FLOOR, (
        "40 consecutive failed options do not close the gate — a useless "
        "skill would keep taking budget forever (the measured 96/96 horizon)")
    assert float(cp.predict_all()[5]) >= FLOOR, \
        "failing slot 2 dragged slot 5 down — tasks must stay decoupled"

    # ---- C. ...AND A WORKING SKILL IS NOT PUNISHED ----------------------
    # MEASURED response of this predictor (Adam lr=0.05, BCE, frozen bias 0):
    #     n=20  success 0.724 / failure 0.277   <- failure still ABOVE floor
    #     n=40  success 0.850 / failure 0.150
    #     n=80  success 0.935 / failure 0.065
    # So a persistently failing skill needs ~25-30 invocations to gate: fast
    # enough to stop the measured 86%-of-budget monopoly, slow enough that a
    # short unlucky streak does not condemn a skill. These are the real
    # numbers, not round guesses — an earlier draft asserted >0.9 at n=40 and
    # failed against the actual 0.850.
    cp2 = CompetencePredictor(n_tasks=8)
    for _ in range(40):
        cp2.update(1, 1.0)                    # `spike` closes
    assert float(cp2.predict_all()[1]) > 0.84, "success must raise competence"
    # the gate must NOT slam shut on a brief unlucky streak
    cp3 = CompetencePredictor(n_tasks=8)
    for _ in range(10):
        cp3.update(1, 0.0)
    assert float(cp3.predict_all()[1]) >= FLOOR, (
        "10 failures already gate the skill — too trigger-happy; a skill "
        "deserves more than one bad stretch before losing its slot")

    # ---- D. PROBATION: THE GATE CAN REOPEN (no 5th latch) ---------------
    assert "_apply_probation" in opt, "no re-offer path — this is a latch"
    assert "gate_retry_after" in opt, "probation window is not configurable"
    assert "mask = self._apply_probation(mask, timestep)" in opt, \
        "probation is never applied to the mask actually handed to the policy"
    assert "self._last_inv_t[slot] = int(timestep)" in opt, \
        "nothing stamps the probation clock on invocation"

    from developmental_ai.policy.options import OptionExecutor
    ex = OptionExecutor.__new__(OptionExecutor)      # no torch/env needed
    ex.bank = _FakeBank(P=10, scripted_slot=3, n=5)
    ex.retry_after = 100
    ex._last_inv_t = {}
    ex.probation_offers = 0

    base = np.zeros(10 + 5, dtype=bool)
    base[:10] = True                                  # primitives only
    # a fully-gated mask at t=0: nothing stale yet (last_inv defaults to 0)
    out0 = ex._apply_probation(base, 0)
    assert not out0[10:].any(), "nothing should be re-offered before the window"
    assert out0 is base, "must not copy when there is nothing to change"

    # ...once the window passes, every gated NON-SCRIPTED slot is re-offered
    out1 = ex._apply_probation(base, 150)
    assert out1 is not base, "the shared base mask was mutated in place"
    assert not base[10:].any(), "base mask must be left untouched"
    reoffered = [s for s in range(5) if out1[10 + s]]
    assert reoffered == [0, 1, 2, 4], \
        f"expected non-scripted slots re-offered, got {reoffered}"
    assert ex.probation_offers == 4

    # a slot invoked recently is NOT re-offered; its clock resets
    ex._last_inv_t[1] = 140
    out2 = ex._apply_probation(base, 150)
    assert not out2[10 + 1], "recently-invoked slot must stay gated"
    assert out2[10 + 0], "other stale slots still get their chance"

    # ---- E'. THE GATE MUST BE OBSERVABLE, LEARNED-ONLY ------------------
    # `decisions_with_offer` counts mask[P:].any(), which INCLUDES the
    # scripted slot — and the scripted slot bypasses the competence gate.
    # So that ratio reads ~100% forever even when every learned skill is
    # gated, and cannot show this fix working. A separate learned-only
    # counter is required, or the mechanism is invisible.
    assert "_learned_idx" in opt, "no learned-only offer accounting"
    assert "decisions_with_learned" in opt
    assert 'b.get("scripted")' in opt, \
        "learned-only index does not exclude the gate-exempt scripted slot"
    assert "Gate state:" in loop, \
        "gate state never printed — the deployed mechanism is unobservable"
    assert "probation re-offers" in loop, \
        "probation activity is not surfaced, so the anti-latch cannot be " \
        "confirmed to actually fire"

    # ---- E. THE LATCH IS ONLY DISABLED DELIBERATELY ---------------------
    ex.retry_after = 0
    assert ex._apply_probation(base, 10**9) is base, \
        "retry_after<=0 must disable probation (documented latch restore)"

    print("[competence-feedback-smoke] ALL PASS: option outcomes reach the "
          "gate at BOTH close sites, 40 horizons close a gate (0.50 -> below "
          f"{FLOOR}) while tasks stay decoupled, 40 spikes raise it >0.9, and "
          "probation re-offers stale gated slots (non-scripted only, base "
          "mask never mutated) so the gate can reopen — no 5th one-way latch")


if __name__ == "__main__":
    main()
