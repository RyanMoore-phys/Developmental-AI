"""Smoke: BOUNDED skill-invokes-skill + hierarchical composition (arch v3).

Until now the hierarchy was exactly two levels deep BY CONSTRUCTION: every
stored head was truncated to primitives at bind, so a skill could never emit
an option-slot action. That truncation was load-bearing (it is what stopped
option-slot logits being re-labelled as the new crafting buttons in the
10->12 widening), so opening it up has to re-establish the safety it was
providing, explicitly.

The five properties, each one a way this could go wrong:
  1. DEPTH IS BOUNDED — at max_skill_depth every slot row is masked off, so
     a child can only emit primitives. No unbounded recursion, ever.
  2. NO CYCLES — a skill may not invoke itself or any ancestor on the stack.
  3. IDENTITY, NOT POSITION — a child is resolved through the mint-time
     slot_map by skill_id. Slots rebind constantly (paging); trusting the
     row index would make a frozen skill invoke a stranger. If the mapped
     skill is not currently bound, the row goes DARK rather than wrong.
  4. THE ENV ONLY EVER SEES PRIMITIVES — whatever the stack does, `act()`
     returns an action < P for every env.
  5. CREDIT IS UNCHANGED — the meta-policy still sees ONE SMDP decision per
     root option, whose tau spans every nested step. Nesting must not
     manufacture extra PPO rows or the option layer's credit assignment
     silently changes meaning.

Run: PYTHONPATH=. python tests/_nested_options_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from developmental_ai.policy.options import OptionExecutor, OptionRuntime

P = 12          # primitives (the live macro count)
K = 4           # option slots


class _FakeBank:
    """Minimal SkillOptionBank stand-in: two conv skills, A and B.

    A's slot_map maps its row 0 -> B, so A invoking its first slot row is a
    request to run B. B maps back to A, which is the CYCLE the executor must
    refuse.
    """

    def __init__(self, force_row=None):
        self.P, self.K = P, K
        self.slots = [
            {"skill_id": "ach_00_break_a", "name": "A", "arch": "conv",
             "p_own": P, "head_dim": P + K, "slot_map": {"0": "ach_01_break_b"},
             "preconditions": {}},
            {"skill_id": "ach_01_break_b", "name": "B", "arch": "conv",
             "p_own": P, "head_dim": P + K, "slot_map": {"0": "ach_00_break_a"},
             "preconditions": {}},
            None, None]
        self.force_row = force_row      # which head index skill_action returns
        self.calls = []
        self.skill_bank = _FakeSkillBank()

    def mask(self, predicates=None, min_match=0.6, competence=None,
             competence_floor=0.0):
        m = np.zeros(self.P + self.K, dtype=bool)
        m[:self.P] = True
        for i, b in enumerate(self.slots):
            if b is not None:
                m[self.P + i] = True
        return m

    def slot_of_skill(self, sid):
        for i, b in enumerate(self.slots):
            if b is not None and b["skill_id"] == sid:
                return i
        return None

    def is_scripted(self, slot):
        return False

    def note_invoked(self, slot):
        pass

    def skill_action(self, slot, obs, head_mask=None, env=0,
                     proprio=None):
        """Emit force_row if the mask allows it, else the first legal
        primitive — exactly what a masked Categorical would do.

        `env` mirrors the real bank: skill practice keys captured
        trajectories by (env, slot) so concurrent scout streams cannot have
        their evidence interleaved into one update. `proprio` mirrors it too:
        a bound skill receives its body state, and a stub that omits the
        parameter would let a real caller regress unnoticed."""
        self.calls.append((slot, None if head_mask is None
                           else tuple(np.flatnonzero(head_mask))))
        want = self.force_row
        if want is not None and (head_mask is None or head_mask[want]):
            return want
        return 0


class _FakeSkillBank:
    def __init__(self):
        self.asked = []
        self.edges = []

    def record_asked(self, sid, produced_effect):
        self.asked.append((sid, bool(produced_effect)))

    def record_invokes(self, parent, child):
        self.edges.append((parent, child))


class _FakePolicy:
    """Always picks option slot 0 (= skill A) when offered."""

    def select_action(self, obs, knowledge=None, action_mask=None):
        return P + 0, {"log_prob": -1.0, "value": 0.5}


def _mk(depth, force_row=None):
    bank = _FakeBank(force_row=force_row)
    ex = OptionExecutor(bank, num_envs=1,
                        cfg={"max_option_steps": 50, "max_skill_depth": depth,
                             "spike_threshold": 0.9}, gamma=0.9)
    return bank, ex


def main() -> None:
    obs = [np.zeros(8, np.float32)]

    # ---- 1. depth 1 (default): slot rows are NEVER open -------------------
    bank, ex = _mk(depth=1, force_row=P + 0)     # skill A wants to invoke B
    acts = ex.act(obs, None, _FakePolicy(), 0)
    assert acts[0] < P, f"env received non-primitive {acts[0]}"
    assert not ex.substacks[0], "nested frame pushed at max_skill_depth=1"
    assert ex.nested_depth_blocks > 0, "depth block not counted"
    _, m = bank.calls[-1]
    assert m is not None and all(i < P for i in m), \
        f"a slot row was offered to a skill at the depth cap: {m}"

    # ---- 2/3. depth 2: A invokes B by IDENTITY, then B is capped ----------
    bank, ex = _mk(depth=2, force_row=P + 0)
    a0 = ex.act(obs, None, _FakePolicy(), 0)
    assert a0[0] < P, "env saw a non-primitive during nesting"
    assert len(ex.substacks[0]) == 1, "A did not push a nested frame"
    child = ex.substacks[0][0]
    assert child.skill_id == "ach_01_break_b", \
        f"resolved to {child.skill_id} — identity mapping is wrong"
    assert child.depth == 2
    assert ex.nested_pushes == 1
    assert bank.skill_bank.edges == [("ach_00_break_a", "ach_01_break_b")], \
        "composition edge not recorded at the push"
    # B is now at the cap: its own slot row (back to A) must be closed, which
    # is BOTH the depth bound AND the cycle guard
    _, m_child = bank.calls[-1]
    assert all(i < P for i in m_child), \
        f"child at depth 2 was offered slot rows: {m_child}"

    # ---- cycle guard, isolated from the depth cap ------------------------
    bank, ex = _mk(depth=3, force_row=P + 0)
    ex.act(obs, None, _FakePolicy(), 0)          # A -> B pushed
    n_before = ex.nested_cycle_blocks
    ex.act(obs, None, _FakePolicy(), 1)          # B would invoke A: refused
    assert ex.nested_cycle_blocks > n_before, \
        "B invoking its ancestor A was not blocked as a cycle"
    assert len(ex.substacks[0]) == 1, "a cycle was actually pushed"

    # ---- 3b. an UNBOUND mapped skill goes dark, never wrong --------------
    bank, ex = _mk(depth=2, force_row=P + 0)
    bank.slots[1] = None                          # B paged out
    ex.act(obs, None, _FakePolicy(), 0)
    assert not ex.substacks[0], \
        "invoked a skill that is not bound — positions were trusted"

    # ---- 4/5. credit: ONE decision per root, tau spans nested steps -------
    bank, ex = _mk(depth=2, force_row=P + 0)
    ex.act(obs, None, _FakePolicy(), 0)
    assert ex.substacks[0], "precondition for the credit test not met"
    closed = None
    for t in range(1, 9):
        ex.act(obs, None, _FakePolicy(), t)
        c = ex.observe_primary(0.0, False, 0.1, t)
        assert c is None, "root closed early"
        closed = c
    # The keys must be MEASURABLE for the ledger to score them: the executor
    # abstains on a skill whose own effect key the env has never emitted
    # (otherwise discovered_N-style skills are scored 0 forever — a latch).
    ex.note_effect(0, "break_a")
    ex.note_effect(0, "break_b")
    closed = ex.observe_primary(5.0, False, 5.0, 9)     # spike ends it
    assert closed is not None, "spike did not close the root option"
    assert closed["skill_id"] == "ach_00_break_a", \
        "credit attributed to the CHILD — the meta-policy chose the root"
    assert closed["tau"] == 9, f"tau={closed['tau']} does not span nested steps"
    assert not ex.substacks[0], "children survived the root close"
    # mastery counted for BOTH frames (root + the nested child)
    ids = [s for s, _ in bank.skill_bank.asked]
    assert "ach_00_break_a" in ids and "ach_01_break_b" in ids, \
        f"nested practice invisible to the mastery ledger: {ids}"

    # ---- 5b. ROOT HORIZON must collapse children ------------------------
    # The spike path above collapses the stack on its OWN (spike ends every
    # frame), so it CANNOT prove the root close does. A first version of
    # this test asserted the property there and a mutation deleting the
    # root's collapse survived. Isolate it: end the root by HORIZON, which
    # the child (pushed later, so fewer steps) has not yet reached.
    # The child must be YOUNGER than the root, or the two hit the horizon on
    # the SAME step and _advance_substack pops the child first — which is
    # what let a mutation deleting the root's collapse survive. Open the
    # root emitting primitives, then let it invoke a child two steps in.
    bank, ex = _mk(depth=2, force_row=None)      # root emits primitives
    ex.max_option_steps = 5
    ex.act(obs, None, _FakePolicy(), 0)          # root opens, no child
    assert not ex.substacks[0]
    ex.observe_primary(0.0, False, 0.0, 1)
    bank.force_row = P + 0                       # NOW it invokes B
    ex.act(obs, None, _FakePolicy(), 2)
    assert len(ex.substacks[0]) == 1, "child never pushed"
    assert ex.substacks[0][0].steps_done < ex.runtimes[0].steps_done, \
        "child is not younger than the root — the test cannot isolate this"
    closed_h = None
    for t in range(3, 9):
        ex.act(obs, None, _FakePolicy(), t)
        closed_h = ex.observe_primary(0.0, False, 0.0, t)
        if closed_h is not None:
            break        # STOP: another act() would open a NEW option and
            # push a NEW child, and the assertion below would then be
            # inspecting the successor's stack instead of the orphan.
    assert closed_h is not None and closed_h["outcome"] == "horizon"
    assert not ex.substacks[0], (
        "the root closed on horizon and left nested frames RUNNING — the "
        "next option would execute inside a dead parent's stack")

    # ---- effect-vs-spike, THROUGH the executor --------------------------
    # The bank-level test proves record_asked's arithmetic; only this proves
    # the executor passes the SKILL'S OWN EFFECT and not "something good
    # happened". Same close, same reward, opposite effect key.
    for key, want in (("break_a", True), ("break_zzz", False)):
        bank, ex = _mk(depth=1)
        ex.act(obs, None, _FakePolicy(), 0)
        ex._effect_vocab.add("break_a")   # measurable in principle
        ex.note_effect(0, key)
        ex.observe_primary(5.0, False, 5.0, 1)      # spike closes it
        rec = [v for s, v in bank.skill_bank.asked if s == "ach_00_break_a"]
        assert rec and rec[-1] is want, (
            f"effect {key!r} on a SPIKING close recorded success={rec} — "
            f"mastery must count the skill's OWN effect, never the spike")

    # ---- abstention: an UNMEASURABLE key is never scored a failure -----
    bank, ex = _mk(depth=1)
    ex.act(obs, None, _FakePolicy(), 0)
    ex.observe_primary(5.0, False, 5.0, 1)          # spike, no note_effect
    assert not bank.skill_bank.asked, (
        "scored a skill whose effect key has never been observed — that is "
        "a latch: it can only ever record 0 and never reach mastery")
    assert ex.mastery_unmeasurable > 0, "abstention not counted"

    print("[nested-options-smoke] ALL PASS: depth bounded (rows masked at "
          "the cap, default 1 = legacy behaviour); cycles refused against "
          "the whole stack; children resolved by mint-time IDENTITY and dark "
          "when unbound; env only ever sees primitives; ONE SMDP decision "
          "per root with tau spanning nested steps and children collapsed "
          "on close; nested practice reaches the mastery ledger")


if __name__ == "__main__":
    main()
