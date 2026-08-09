"""Smoke: the mastery ledger is WIRED (task #39) — asked/success + fidelity.

Before this, `record_asked` had NO CALLER anywhere in the tree: the window
arithmetic existed, `is_mastered` was computed from it, and nothing ever fed
it — so "mastered" meant nothing and the WM-fidelity half was never measured.

Pins:
  1. record_asked IS CALLED from the executor's frame close, for the root
     option AND for nested frames AND on scout streams.
  2. SUCCESS = the skill's OWN grounded effect occurred while it was open —
     NOT a reward spike. (Measured counterexample: leaves pay 0.3, below the
     0.9 spike threshold, yet break_birch_leaves logged 7 "spikes" in 350
     invocations; those cannot have been leaves.)
  3. The window TRAILS: mastery is losable, not a permanent crown.
  4. wm_fidelity folds in as a ratio and gates `is_mastered` — and a
     never-measured (None) fidelity does NOT block mastery (that would be
     the 5th guard-becomes-latch).
  5. THE LEDGER SURVIVES AN UPSERT — a re-mint must not silently reset the
     asked window, or the metric zeroes itself on the very event that proves
     practice.

Run: PYTHONPATH=. python tests/_mastery_wiring_smoke.py
"""
import shutil
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from developmental_ai.skill_bank.skill_bank import SkillBank

SB = "/tmp/mastery_wiring_smoke"
OPTS = "developmental_ai/policy/options.py"
LOOP = "developmental_ai/core/developmental_loop.py"


def main() -> None:
    # ---- 1. the caller exists, at the frame close ------------------------
    src = open(OPTS).read()
    assert "def _record_mastery" in src, "no mastery recorder in the executor"
    assert src.count("self._record_mastery(") >= 2, (
        "mastery recorded at fewer than 2 sites — the root close AND the "
        "nested pop must both count, or nested practice is invisible")
    ic = src.index("def _close(")
    assert "_record_mastery" in src[ic:ic + 1400], \
        "root option close does not record mastery"
    ip = src.index("def _pop_frame(")
    assert "_record_mastery" in src[ip:ip + 2000], \
        "nested frame pop does not record mastery"
    # ...and a frame cut short by its PARENT must ABSTAIN, not be scored a
    # failure (it never got its own chance) — review MEDIUM.
    assert "if not truncated:" in src[ip:ip + 2000], \
        "nested pop scores parent-truncated frames as failures"
    assert "self.mastery_unmeasurable" in src, \
        "no abstention for skills whose own effect key is unmeasurable — "\
        "discovered_N/act_where_* would be scored 0 forever (a latch)"
    # scouts reach _close through observe_scouts -> _close: same path
    assert "def note_effect" in src, "no effect channel into the frames"
    # ...and the loop actually feeds it
    loop = open(LOOP).read()
    assert loop.count("option_executor.note_effect(") >= 2, (
        "the loop feeds effect keys at fewer than both twin sites")
    assert "record_wm_fidelity(" in loop, "WM fidelity never recorded"

    # ---- 2/3. success is EFFECT-based, and the window trails --------------
    shutil.rmtree(SB, ignore_errors=True)
    bank = SkillBank(storage_dir=SB)
    sd = {"actor": {"w": torch.zeros(2, 2)}, "critic": {}}
    bank.save_skill("ach_07_break_oak_log", "break oak log", sd,
                    obs_dim=12, action_dim=10)
    sk = bank.skills["ach_07_break_oak_log"]

    for _ in range(bank.MASTERY_MIN_N):
        bank.record_asked("ach_07_break_oak_log", produced_effect=True)
    assert sk.mastery_level == 1.0 and sk.is_mastered, \
        "20/20 successes did not confer mastery"
    assert sk.asked_total == 20
    # ...and it is LOSABLE
    for _ in range(bank.MASTERY_WINDOW):
        bank.record_asked("ach_07_break_oak_log", produced_effect=False)
    assert sk.mastery_level == 0.0 and not sk.is_mastered, \
        "mastery survived a full window of failures — it is a crown, not a "\
        "measurement"
    assert len(sk.asked_log) == bank.MASTERY_WINDOW, "window is not trailing"

    # ---- 4. fidelity gates, but None never latches ------------------------
    for _ in range(bank.MASTERY_WINDOW):
        bank.record_asked("ach_07_break_oak_log", produced_effect=True)
    assert sk.is_mastered and sk.wm_fidelity is None, \
        "unmeasured fidelity blocked mastery — that is a latch"
    bank.record_wm_fidelity("ach_07_break_oak_log", 5.0)   # wildly surprising
    assert sk.wm_fidelity is not None and sk.wm_fidelity > bank.WM_FIDELITY_MAX
    assert not sk.is_mastered, \
        "a skill the world model cannot predict still counts as mastered"
    for _ in range(30):                       # ...and it recovers by evidence
        bank.record_wm_fidelity("ach_07_break_oak_log", 0.5)
    assert sk.is_mastered, "fidelity never recovered — EMA is one-way"

    # ---- 5. the ledger SURVIVES a re-mint (upsert) -----------------------
    before_log = list(sk.asked_log)
    before_total, before_fid = sk.asked_total, sk.wm_fidelity
    bank.save_skill("ach_07_break_oak_log", "break oak log", sd,
                    success_rate=0.99, total_episodes=999,
                    obs_dim=12, action_dim=10, dedup=True)
    sk2 = bank.skills["ach_07_break_oak_log"]
    assert sk2.asked_log == before_log and sk2.asked_total == before_total, (
        "re-minting RESET the asked window — the metric zeroes itself on the "
        "event that proves practice")
    assert sk2.wm_fidelity == before_fid, "re-minting reset WM fidelity"
    # ...and the LEDGER, not the competence heuristic, decides the verdict.
    # The upsert above passed success_rate=0.99, which would set
    # is_mastered=True from the competence scalar alone; the preserved
    # asked_log is 20/20 so it agrees here. Force them to DISAGREE:
    for _ in range(bank.MASTERY_WINDOW):
        bank.record_asked("ach_07_break_oak_log", produced_effect=False)
    bank.save_skill("ach_07_break_oak_log", "break oak log", sd,
                    success_rate=0.99, total_episodes=1500,
                    obs_dim=12, action_dim=10, dedup=True)
    sk3 = bank.skills["ach_07_break_oak_log"]
    assert not sk3.is_mastered and sk3.mastery_level == 0.0, (
        f"upsert crowned a skill (mastered={sk3.is_mastered}, "
        f"level={sk3.mastery_level}) from the competence scalar while its "
        f"measured ledger says 0/{len(sk3.asked_log)} — the heuristic "
        f"overwrote the only real measurement")
    # option-practice history survives too
    assert sk3.invocations == sk2.invocations, "upsert reset invocations"

    # ---- composition edges are counted, not declared ---------------------
    bank.record_invokes("ach_07_break_oak_log", "ach_02_break_dirt")
    bank.record_invokes("ach_07_break_oak_log", "ach_02_break_dirt")
    bank.record_invokes("ach_07_break_oak_log", "ach_07_break_oak_log")  # self
    assert bank.skills["ach_07_break_oak_log"].invokes == {
        "ach_02_break_dirt": 2}, "invocation graph miscounted or self-looped"

    shutil.rmtree(SB, ignore_errors=True)
    print("[mastery-wiring-smoke] ALL PASS: record_asked called at root AND "
          "nested closes with the loop feeding effect keys at both twin "
          "sites; success is own-effect not reward-spike; window trails so "
          "mastery is losable and regainable; fidelity gates but None never "
          "latches; ledger survives upsert; invokes counted without self-loops")


if __name__ == "__main__":
    main()
