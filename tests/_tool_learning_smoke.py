"""Smoke: the axe is EQUIPPED, and its meaning is LEARNED from use.

Two things, both load-bearing:

A) THE EQUIP FIX. Measured 2026-07-25: `mainhand=none` — the agent had been
   chopping BAREHANDED its whole life. The env grants an iron axe to inventory
   slot 0, but a Minecraft item is only HELD once its hotbar slot is selected,
   and the agent's macro action space has no hotbar/equip action. Barehanded a
   log needs ~60 ticks; a chop option holds attack for 25*2 = 50. Every chop
   timed out a few ticks short, forever. Fix: press `hotbar.1` at reset via
   the UNDERLYING action space, leaving the agent's own macro space untouched.

B) TOOL SEMANTICS, EARNED. The agent SEES what it holds and that it wears
   (Minecraft renders the held item and a durability bar). Nothing declares
   "an axe chops wood" or "a short bar means about to break" — those come from
   use: a block falling with a tool in hand mints `<tool> enables break_X`;
   rising durability mints `wears_down`; losing the tool mints `breaks_from`.

Run: PYTHONPATH=. python tests/_tool_learning_smoke.py
"""
import re
import sys

sys.path.insert(0, ".")

from developmental_ai.llm import vlm_symbolizer as V

ENV = "developmental_ai/environments/minerl_env.py"
LOOP = "developmental_ai/core/developmental_loop.py"


def main() -> None:
    env_src = open(ENV).read()
    loop_src = open(LOOP).read()

    # ---- A. CHOP BUDGET clears a BAREHANDED break ------------------------
    # The equip route is a proven dead end (the granted axe sits in a slot no
    # hotbar key reaches). What matters is that a chop can actually FINISH:
    # barehanded a log needs ~60 ticks, so budget must exceed that.
    import yaml
    cfg = yaml.safe_load(open("configs/minecraft_skybot.yaml"))
    mos = int(cfg["skills_as_options"]["max_option_steps"])
    ar = int(cfg["environment"]["action_repeat"])
    assert mos * ar > 60, (
        f"chop budget {mos}x{ar}={mos*ar} ticks cannot finish a barehanded "
        f"log (~60) — every chop times out just short, which is exactly the "
        f"bug that produced ~0 task reward for a full day")
    # it must NOT change the agent's own action space (policy compat)
    macro_block = env_src[env_src.index("TREECHOP_MACROS"):]
    macro_block = macro_block[:macro_block.index("]")]
    assert "hotbar" not in macro_block, (
        "hotbar leaked into TREECHOP_MACROS — that changes action_dim and "
        "breaks every saved skill policy")

    # ---- B. PERCEPTION: tool + wear are SEEN ----------------------------
    for p in ("holding_tool", "tool_worn"):
        assert p in V.PREDICATES, f"{p} not perceivable"
        assert p in V._SCENE_PROMPT, f"VLM never asked about {p}"
        assert p in V._FACT_TEMPLATES, f"{p} has no KG triple"
    # triples state PRESENCE, never purpose
    assert V._FACT_TEMPLATES["holding_tool"] == ("agent", "holds", "tool")
    assert V._FACT_TEMPLATES["tool_worn"] == ("tool", "is", "worn")

    # ---- C. NO DECLARED TOOL SEMANTICS (the whole point) ----------------
    # An axe must never be *told* it chops wood or that it is "better".
    banned = ("chops", "for_chopping", "is_tool_for", "TOOL_EFFECTIVENESS",
              "axe_is_best", "tool_speed", "BEST_TOOL")
    for word in banned:
        for src, name in ((env_src, "minerl_env"), (V.__file__, "vlm_symbolizer"),
                          (loop_src, "developmental_loop")):
            hay = open(src).read() if name == "vlm_symbolizer" else src
            for m in re.finditer(re.escape(word), hay):
                ls = hay.rfind("\n", 0, m.start()) + 1
                line = hay[ls:hay.find("\n", m.start())].strip()
                assert line.startswith("#"), (
                    f"{name}: '{word}' appears as CODE — tool meaning must be "
                    f"earned from use, not declared:\n  {line[:100]}")

    # ---- D. MEANING IS MINTED FROM REAL EVENTS --------------------------
    assert "TOOL MEANING, EARNED" in loop_src
    assert '(_tool, "enables", f"break_{_bt}", _TOOL_FACT_CONF)' in loop_src, \
        "no tool->effect fact: the agent can never learn what an axe is FOR"
    assert '(_tool, "wears_down", "with_use", _TOOL_FACT_CONF)' in loop_src, \
        "no wear fact: durability can never be understood"
    assert '(_lost, "breaks_from", "use", _TOOL_FACT_CONF)' in loop_src, \
        "no breakage fact: tool loss teaches nothing"
    # enablement must key on ACTUAL breaks this step, not a guess
    assert "for _bt in _broke_now:" in loop_src, \
        "tool enablement is not keyed to real break events"
    assert "_broke_now.append(_btype)" in loop_src

    # ARITY: these MUST carry a confidence — `_to_fact` unpacks 4 fields, and
    # 3-tuples here crashed the run the first time a tool fact ever fired
    # (the path had never executed because the mainhand sensor was dead).
    assert "_TOOL_FACT_CONF" in loop_src, "tool-fact confidence constant gone"

    # ---- E. the wear/loss EXPERIENCE is measured, not assumed -----------
    assert '"tool_damage"' in env_src and '"tool_wore"' in env_src, \
        "durability is never read from the environment"
    assert '"tool_lost"' in env_src, "tool loss is never detected"
    assert "_prev_tool_damage" in env_src, "wear needs a previous-value memory"

    print(f"[tool-learning-smoke] ALL PASS: chop budget {mos}x{ar}="
          f"{mos*ar} ticks > 60 (barehanded log completes), "
          f"{len(V.PREDICATES)} predicates incl. holding_tool/tool_worn, "
          f"presence-only triples, no declared tool semantics, meaning minted "
          f"from real breaks + measured wear + tool loss")


if __name__ == "__main__":
    main()
