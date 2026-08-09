"""Smoke: the 10->12 ACTION widening — buttons, craft economy, safe binding.

(Distinct from tests/_widening_smoke.py, which covers the 24/48 goal
working-set widening of July 20.)

THE FIVE CONDITIONS this pins (from the crafting scout, AUDIT_FINDINGS #61):
  1. APPEND ONLY — indices 0-9 keep their exact meaning; the two new buttons
     sit at 10 (inventory) and 11 (use).
  2. P_old-AWARE TRUNCATION — a stored skill's head is sliced to ITS OWN
     primitive count, never today's P. Slicing W[:12] out of a 34-row head
     would bind option-slot-0/1 logits as the two new buttons: no refusal
     fires (34 >= 12), and a frozen skill silently emits "invoke option 0"
     re-labelled "open inventory".
  3. NULL action_dim is a HARD refusal (never infer rows - k_slots; k_slots
     is a config knob that has changed across eras) — with a backfill script
     recording the era fact action_dim=10 for the pre-widening bank.
  4. FRANKEN-STATE GUARD — torch >=2.6 copies matching params BEFORE raising
     on the mismatched one, so a failed warm-start must SNAPSHOT AND RESTORE
     or the run continues on a foreign trunk while logging "from scratch".
  5. The craft economy mirrors the break economy: ground-truth counters,
     first-craft-of-type top tier, same achievements -> goal-discovery path,
     re-armed by clear_break_marks, no phantom on reset.

Run: PYTHONPATH=. python tests/_action_widening_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from developmental_ai.environments.minerl_env import (
    MineRLEnvAdapter, TREECHOP_MACROS)
from developmental_ai.policy.options import SlotRefused, adapt_actor_sd

LOOP = "developmental_ai/core/developmental_loop.py"


def _meta_sd(rows: int, in_dim: int = 128, hidden: int = 64):
    """A synthetic stored policy: meta-width head, recognisable rows."""
    W = torch.zeros(rows, hidden)
    for r in range(rows):
        W[r] = float(r + 1)                 # row r is filled with r+1
    return {"actor": {
        "action_head.weight": W,
        "action_head.bias": torch.arange(rows, dtype=torch.float32),
        "shared.0.weight": torch.zeros(hidden, in_dim),
    }}


def main() -> None:
    # ---- 1. APPEND ONLY --------------------------------------------------
    # The contract is APPEND-ONLY, not a fixed width — the table has grown
    # 10 -> 12 (inventory, use) -> 13 (HOLD attack) and will grow again. What
    # must never change is the meaning of an existing index, because every
    # saved skill, policy and config addresses actions by position. Pinning
    # the COUNT just makes this test fail on every legitimate append; pinning
    # the PREFIX is what actually protects compatibility.
    assert len(TREECHOP_MACROS) >= 12, (
        f"macro table SHRANK to {len(TREECHOP_MACROS)} — indices were removed, "
        "which silently rebinds every stored skill")
    expected_first10 = [
        {}, {"forward": 1}, {"forward": 1, "jump": 1},
        {"camera": [0.0, -15.0]}, {"camera": [0.0, 15.0]}, {"attack": 1},
        {"attack": 1, "forward": 1}, {"camera": [-15.0, 0.0]},
        {"camera": [15.0, 0.0]}, {"back": 1}]
    assert TREECHOP_MACROS[:10] == expected_first10, (
        "indices 0-9 changed — every saved skill, policy and config indexes "
        "actions by position; append-only is the entire compat contract")
    assert TREECHOP_MACROS[10] == {"inventory": 1}
    assert TREECHOP_MACROS[11] == {"use": 1}
    if len(TREECHOP_MACROS) > 12:
        # 12 = HOLD attack: a sustained press, added 2026-08-05 because a
        # barehanded log needs ~60 ticks = 30 CONSECUTIVE attack decisions,
        # which a stochastic policy cannot produce (P ~ 1e-30 at a 10% attack
        # rate). It must be a real attack and must hold for several ticks.
        _h = TREECHOP_MACROS[12]
        assert _h.get("attack") and _h.get("_ticks", 0) >= 8, (
            f"index 12 is not a sustained attack: {_h}")
    # still no hotbar keys (deliberate: minimal exploration burden)
    assert not any("hotbar" in k for m in TREECHOP_MACROS for k in m)

    # ---- 2. P_old-AWARE TRUNCATION ---------------------------------------
    sd = _meta_sd(34)                       # 10 primitives + 24 slots, old era
    a, _, _ = adapt_actor_sd(sd, P=12, primitive_dim=10)
    W = a["action_head.weight"]
    assert W.shape[0] == 10, f"sliced to {W.shape[0]}, wanted the skill's OWN 10"
    assert float(W[9, 0]) == 10.0, "row 9 is not the skill's row 9"
    # THE BUG THIS PREVENTS: naive W[:P] would put old option-slot rows
    # (values 11.0, 12.0) at the new button indices 10, 11.
    naive = sd["actor"]["action_head.weight"][:12]
    assert float(naive[10, 0]) == 11.0, "sanity: naive slice grabs slot-0 row"

    # a same-era skill (head exactly P) still binds without metadata
    ok, _, _ = adapt_actor_sd(_meta_sd(12), P=12, primitive_dim=None)
    assert ok["action_head.weight"].shape[0] == 12

    # ---- 3. NULL action_dim REFUSES, never guesses ------------------------
    try:
        adapt_actor_sd(_meta_sd(34), P=12, primitive_dim=None)
        raise AssertionError(
            "34-row head with unrecorded primitive_dim bound in a 12-world — "
            "option-slot logits are now the crafting buttons, silently")
    except SlotRefused as e:
        assert "Backfill" in str(e), "refusal does not say how to fix it"
    # metadata that contradicts the tensor is also refused
    try:
        adapt_actor_sd(_meta_sd(8), P=12, primitive_dim=10)
        raise AssertionError("head narrower than its own primitive_dim bound")
    except SlotRefused:
        pass

    # ---- 4. FRANKEN-STATE GUARD is present at the warm-start -------------
    loop = open(LOOP).read()
    # Anchor on UNIQUE strings and assert ORDER (snapshot -> load attempt ->
    # restore). A first draft anchored on "Warm-start from", whose FIRST
    # occurrence is a comment 130 lines earlier, and searched backwards from
    # there — failing on correct code. An anchor must be unique or the
    # window is meaningless.
    i_snap = loop.find("_fresh_sd = _copy.deepcopy(self.policy.get_state_dict())")
    i_rest = loop.find("self.policy.load_state_dict(_fresh_sd)")
    i_log = loop.find("restored the fresh")
    assert i_snap != -1, (
        "no snapshot before warm-start load — torch >=2.6 partial-copies "
        "before raising, so a failed load leaves a franken-policy")
    assert i_rest != -1, "snapshot exists but is never restored on failure"
    assert i_snap < i_rest < i_log, \
        "guard pieces exist but in the wrong order — restore must follow load"

    # ---- 5. CRAFT ECONOMY -------------------------------------------------
    src = open("developmental_ai/environments/minerl_env.py").read()
    assert 'handlers.ObserveFromFullStats("craft_item")' in src, \
        "craft ground truth not observed — a GUI craft would be invisible"
    # counter reader: nested group, nonzero-only, garbage-safe
    C = MineRLEnvAdapter._craft_counts
    assert C({"craft_item": {"planks": np.array([3]),
                             "stick": np.array([0])}}) == {"planks": 3}
    assert C({"other": 1}) == {}
    assert C("garbage") == {}

    # first-craft pays once per type, re-craft does not, reset re-arms
    e = MineRLEnvAdapter.__new__(MineRLEnvAdapter)
    e._craft_prev = {}
    e._crafted_this_episode = set()

    def craft_step(counts):
        """Mirror of the step() craft block, run against the real state."""
        reward, ach = 0.0, {}
        now = C({"craft_item": counts})
        for it, cnt in now.items():
            d = int(cnt) - int(e._craft_prev.get(it, cnt))
            if d > 0:
                ach[f"craft_{it}"] = int(cnt)
                if it not in e._crafted_this_episode:
                    e._crafted_this_episode.add(it)
                    reward += 5.0
        e._craft_prev = dict(now)
        return reward, ach

    r0, a0 = craft_step({"planks": np.array([4])})
    assert r0 == 0.0 and not a0, "baseline absorption minted a phantom craft"
    r1, a1 = craft_step({"planks": np.array([8])})
    assert r1 == 5.0 and a1 == {"craft_planks": 8}, "first craft did not pay"
    r2, _ = craft_step({"planks": np.array([12])})
    assert r2 == 0.0, "re-craft of the same type paid the first-craft tier"
    e._crafted_this_episode = set()          # clear_break_marks re-arm
    r3, _ = craft_step({"planks": np.array([16])})
    assert r3 == 5.0, "re-armed craft did not pay again"

    # craft keys reach GOAL DISCOVERY at BOTH loop sites, top-ranked,
    # prefix kept (craft_planks can never collide with a planks block)
    assert loop.count('if _k.startswith("craft_"):') == 2, \
        "craft effects enter goal discovery at only one of the twin sites"
    assert loop.count("_b, _rank_base = _k, 3") == 2, \
        "crafts are not top-ranked as effects"
    # ...and can NEVER be claimed by the held tool as a causal enablement
    i2 = loop.index("_broke_now.append")
    blk2 = loop[max(0, i2 - 600):i2]
    assert 'startswith("mine_")' in blk2, (
        "tool-enablement no longer filters to mine_ — a craft while holding "
        "the axe would mint the FALSE fact 'iron_axe enables craft_planks'")

    print("[action-widening-smoke] ALL PASS: 12 macros append-only (0-9 "
          "bit-stable, 10=inventory 11=use, no hotbar); P_old-aware "
          "truncation slices to the skill's OWN width and naive W[:P] "
          "provably grabs slot rows; null metadata refuses with a fix hint; "
          "franken-state guard snapshots and restores; craft economy pays "
          "first-of-type once, re-arms, feeds goal discovery top-ranked at "
          "both sites, and can never be claimed by the held tool")


if __name__ == "__main__":
    main()
