"""Smoke: true mob perception with LEARNED (not declared) meaning.

The project's principle is no gifted knowledge. These contracts pin that down
for creatures:
  1. The agent can PERCEIVE specific creatures — they are real predicates in
     the grounded vocabulary, they reach the head's output layer, and the VLM
     is asked about them.
  2. NOTHING in the codebase declares any creature hostile/dangerous/passive.
     Meaning must be earned from experience, so no such prior may exist.
  3. Each creature maps to a presence-only KG triple (scene contains X) — the
     name, never the meaning.
  4. The head sizes itself to the vocabulary, so adding creatures cannot
     silently truncate outputs.
  5. The VLM prompt forbids guessing a specific creature (falls back to the
     generic animal_visible) — the reliability machinery can then suppress
     what llava genuinely cannot tell apart.

Run: PYTHONPATH=. python tests/_creature_perception_smoke.py
"""
import re
import sys

sys.path.insert(0, ".")

from developmental_ai.llm import vlm_symbolizer as V

HOSTILE = ["zombie_visible", "skeleton_visible", "creeper_visible",
           "spider_visible"]
PASSIVE = ["cow_visible", "pig_visible", "sheep_visible"]


def main() -> None:
    # ---- 1. perception exists ----
    for c in HOSTILE + PASSIVE:
        assert c in V.PREDICATES, f"{c} not perceivable"
        assert c in V.PREDICATE_INDEX
        assert c in V.CREATURE_PREDICATES, f"{c} missing from CREATURE_PREDICATES"
        assert c in V._SCENE_PROMPT, f"VLM is never asked about {c}"
    # the generic fallback survives for "a creature, but which?"
    assert "animal_visible" in V.PREDICATES
    assert "animal_visible" in V.CREATURE_PREDICATES

    # ---- 2. NO declared meaning anywhere (the load-bearing contract) ----
    src = open("developmental_ai/llm/vlm_symbolizer.py").read()
    loop = open("developmental_ai/core/developmental_loop.py").read()
    for word in ("hostile", "dangerous", "is_enemy", "enemy_list",
                 "DANGEROUS", "HOSTILE_MOBS", "passive_mobs"):
        # allow the word only inside comments/prose, never as a data structure
        for hay, name in ((src, "vlm_symbolizer"), (loop, "developmental_loop")):
            for m in re.finditer(re.escape(word), hay):
                line_start = hay.rfind("\n", 0, m.start()) + 1
                line = hay[line_start:hay.find("\n", m.start())]
                stripped = line.strip()
                assert stripped.startswith("#"), (
                    f"{name}: '{word}' appears as CODE, not a comment — "
                    f"creature meaning must be learned, not declared:\n"
                    f"  {stripped[:100]}")

    # ---- 3. presence-only triples (name, never meaning) ----
    for c in HOSTILE + PASSIVE:
        subj, rel, obj = V._FACT_TEMPLATES[c]
        assert (subj, rel) == ("scene", "contains"), (
            f"{c} maps to ({subj},{rel},{obj}) — creature triples must state "
            f"PRESENCE only; anything else smuggles in meaning")
        assert obj == c.replace("_visible", "")

    # ---- 4. head sizes to the vocabulary ----
    import torch
    head = V.GroundedSymbolHead(latent_dim=32)
    out = head(torch.zeros(2, 32))
    assert out.shape[-1] == len(V.PREDICATES), (
        f"head emits {out.shape[-1]} but vocabulary is {len(V.PREDICATES)} — "
        f"creature predicates would be truncated")

    # ---- 5. prompt forbids guessing ----
    assert "NEVER guess" in V._SCENE_PROMPT, "prompt lets llava guess mobs"
    assert "animal_visible" in V._SCENE_PROMPT

    # ---- 6. the loop learns meaning from DAMAGE, symmetrically ----
    assert "CREATURE MEANING, EARNED" in loop
    assert '"killed" if _wi.get("died") else "hurt"' in loop, \
        "damage/death must be what teaches creature meaning"
    # symmetry: the same code path applies to passive and hostile alike —
    # a cow simply never earns the link because it never causes damage
    assert "CREATURE_PREDICATES" in loop

    print(f"[creature-perception-smoke] ALL PASS: {len(HOSTILE)+len(PASSIVE)} "
          f"creatures perceivable ({len(V.PREDICATES)} predicates total), "
          f"presence-only triples, no declared hostility anywhere, meaning "
          f"earned from damage/death")


if __name__ == "__main__":
    main()


def test_earned_meaning_facts_are_wellformed() -> None:
    """The four EARNED-MEANING facts must be 4-tuples, or they crash on first use.

    These paths NEVER EXECUTED — both sensors that trigger them were dead
    (`equipped_items.mainhand` unpopulated; `life` pinned at MAX_LIFE). So the
    fact that they emitted 3-tuples into `_to_fact`, which unpacks
    (subject, relation, object, confidence), went unnoticed until the sensors
    were fixed — at which point the run crashed on the very first creature
    event with `ValueError: not enough values to unpack`.

    LESSON: "built, smoke-green, live" is NOT the same as "works". A code path
    that has never executed is untested no matter how green the suite is, and
    unblocking it is a CHANGE that needs its own verification.
    """
    import re
    src = open("developmental_ai/core/developmental_loop.py").read()
    bad = []
    for m in re.finditer(r"_to_fact\(\s*\n?\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", src):
        inner, depth, parts = m.group(1), 0, 1
        for ch in inner:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                parts += 1
        if parts != 4:
            bad.append((src[:m.start()].count("\n") + 1, parts, inner[:60]))
    assert not bad, (
        "_to_fact takes (subject, relation, object, confidence); these sites "
        "pass the wrong count and will crash the first time they fire: "
        + "; ".join(f"line {l}: {n} elems ({t})" for l, n, t in bad))
    for c in ("_CREATURE_FACT_CONF", "_TOOL_FACT_CONF"):
        assert c in src, f"{c} missing — confidence hard-coded or dropped again"
    print("  earned-meaning facts well-formed (all _to_fact sites arity 4)")


if __name__ == "__main__":
    test_earned_meaning_facts_are_wellformed()
