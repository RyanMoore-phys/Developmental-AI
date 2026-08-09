"""Contract: NO HARDCODED SKILL RULES in the production Minecraft config.

The standing principle for this project is that meaning and skill must be
EARNED from experience, never declared. Crafting is not a recipe table; felling
a tree is not a fixed macro. The agent gets perception and buttons, and has to
find out what they do.

This test is the enforcement. It fails the build if a scripted option is
reintroduced, so that reversing the principle requires deleting a test that
says why — not quietly appending four lines of YAML.

HISTORY (why this is a test and not a comment): the scripted `chop_trunk`
bootstrap ran for 32 segments as 96% of all option activity and broke ZERO
logs, holding attack for up to 10726 consecutive ticks against a trunk it could
not reach — because `precondition_min_match: 0.4` let it fire with
`object_adjacent` FALSE. A fixed macro cannot notice that it is hitting
nothing. That is the failure mode this contract exists to prevent recurring.

Run: PYTHONPATH=. python tests/_no_scripted_skills_smoke.py
"""
import sys

sys.path.insert(0, ".")

import yaml

CFG = "configs/minecraft_skybot.yaml"


def main() -> None:
    cfg = yaml.safe_load(open(CFG))

    # 1. No scripted options, anywhere in the config tree.
    def walk(node, path="root"):
        hits = []
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "scripted_options" and v:
                    hits.append((f"{path}.{k}", v))
                hits += walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                hits += walk(v, f"{path}[{i}]")
        return hits

    found = walk(cfg)
    assert not found, (
        "SCRIPTED SKILL REINTRODUCED: "
        + "; ".join(f"{p} = {v}" for p, v in found)
        + "\nSkill learning must not be a hardcoded rule in the game. If this "
          "is a deliberate reversal, delete this test and say why in "
          "AUDIT_FINDINGS — do not weaken it.")

    # 2. The env exposes buttons, not recipes: the macro table must stay a list
    #    of raw key presses. A macro carrying a crafting/recipe payload would be
    #    declaring meaning through the back door.
    from developmental_ai.environments.minerl_env import TREECHOP_MACROS
    RAW_KEYS = {"forward", "back", "left", "right", "jump", "sneak", "sprint",
                "attack", "use", "inventory", "camera", "drop", "swapHands",
                "pickItem", "_ticks"}
    for i, m in enumerate(TREECHOP_MACROS):
        bad = set(m) - RAW_KEYS
        assert not bad, (
            f"macro {i} carries non-input keys {bad} — macros are BUTTONS, "
            f"not scripted behaviours or recipes")
        assert "craft" not in str(m).lower(), (
            f"macro {i} mentions crafting: {m} — a craft must be performed "
            f"through the GUI with the same buttons a player uses, never "
            f"declared as an action")

    # 3. No recipe table smuggled in as config (log->planks->sticks etc).
    flat = yaml.dump(cfg).lower()
    for word in ("planks", "stick", "crafting_table", "wooden_axe", "recipe"):
        assert word not in flat, (
            f"config names '{word}' — the tech tree must be DISCOVERED from "
            f"craft_item observations, never written down for the agent")

    print("[no-scripted-skills] ALL PASS: scripted_options empty; all 12 "
          "macros are raw button presses with no behavioural or recipe "
          "payload; no recipe vocabulary anywhere in the config. Felling and "
          "crafting both have to be earned.")


if __name__ == "__main__":
    main()
