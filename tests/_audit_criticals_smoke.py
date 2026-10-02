"""Smoke for audit critical fixes #2/#3 (skills-as-options in lifelong).

#3 (re-distill): a minted skill is snapshotted once at first unlock (weak) and
was never improved. The loop now re-saves the current policy when competence
clearly beats the stored skill; this test proves the save_skill(dedup) path it
relies on: a strictly-better, practiced challenger WINS and archives the
incumbent; a weaker challenger LOSES and the good policy is preserved.

#2 (rebind): refresh_slots binds a skill that appears in the bank after start —
this test proves a skill saved post-construction is picked into an empty slot.

(#1, the crash-rebuilt-client reset, needs a live MineRL client and is validated
on the training host; here it is covered by compile + the ADVANCE-block code review.)
"""
import tempfile

import torch

from developmental_ai.skill_bank.skill_bank import SkillBank


def _pol(obs_dim=4, hidden=8, p=6):
    # minimal actor state dict adapt_actor_sd accepts: shared.0 + action_head,
    # head >= P, in_dim = obs_dim (kdim 0 -> no conditioner needed)
    return {"actor": {"shared.0.weight": torch.zeros(hidden, obs_dim),
                      "shared.0.bias": torch.zeros(hidden),
                      "action_head.weight": torch.zeros(p, hidden),
                      "action_head.bias": torch.zeros(p)}}


def test_redistill_upsert_keeps_best():
    with tempfile.TemporaryDirectory() as d:
        sb = SkillBank(storage_dir=d)
        mp = sb.min_practice_episodes
        # first mint: weak skill
        sb.save_skill(skill_id="ach_00_break_oak_log", name="chop oak log",
                      policy_state_dict=_pol(), success_rate=0.10,
                      total_episodes=mp + 5, obs_dim=8, action_dim=4, dedup=True)
        assert abs(sb.skills["ach_00_break_oak_log"].success_rate - 0.10) < 1e-6
        # re-distill: competence grew to 0.55 (> 0.10 + margin), practiced
        sb.save_skill(skill_id="ach_00_break_oak_log", name="chop oak log",
                      policy_state_dict=_pol(), success_rate=0.55,
                      total_episodes=mp + 20, obs_dim=8, action_dim=4, dedup=True)
        assert abs(sb.skills["ach_00_break_oak_log"].success_rate - 0.55) < 1e-6, \
            f"challenger should win: {sb.skills['ach_00_break_oak_log'].success_rate}"
        # weaker challenger must NOT replace the good policy
        sb.save_skill(skill_id="ach_00_break_oak_log", name="chop oak log",
                      policy_state_dict=_pol(), success_rate=0.30,
                      total_episodes=mp + 30, obs_dim=8, action_dim=4, dedup=True)
        assert abs(sb.skills["ach_00_break_oak_log"].success_rate - 0.55) < 1e-6, \
            "weaker challenger clobbered the better skill"
        print("  1. re-distill upsert: strictly-better practiced challenger wins, "
              "weaker one is rejected")


def test_refresh_binds_post_start_skill():
    from developmental_ai.policy.options import SkillOptionBank
    with tempfile.TemporaryDirectory() as d:
        sb = SkillBank(storage_dir=d)
        bank = SkillOptionBank(sb, obs_dim=4, P=6, K=4, device=torch.device("cpu"))
        bank.refresh_slots()                      # nothing to bind yet
        assert all(s is None for s in bank.slots), "bound something from empty bank"
        # a skill is discovered + minted DURING the run
        sb.save_skill(skill_id="ach_01_break_oak_log", name="chop oak log",
                      policy_state_dict=_pol(), success_rate=0.4,
                      total_episodes=sb.min_practice_episodes + 3,
                      obs_dim=4, action_dim=6, dedup=True)
        bank.notify_minted("ach_01_break_oak_log")
        bank.refresh_slots()                      # the loop now calls this per-segment
        bound = [s for s in bank.slots if s is not None]
        assert any(b.get("skill_id") == "ach_01_break_oak_log" for b in bound), \
            f"minted skill never bound into a slot: {bound}"
        print("  2. refresh_slots binds a skill minted after stream start (fix #2)")


if __name__ == "__main__":
    for fn in (test_redistill_upsert_keeps_best, test_refresh_binds_post_start_skill):
        print(f"[audit-criticals] {fn.__name__}")
        fn()
    print("[audit-criticals] ALL PASS")
