"""Consequence Frontier smoke (2026-08-13).

Pins the two tests that killed every rejected design, plus the defensive
contract.

  BOOTSTRAP AT STEP 0 — the mechanism must produce a pull toward wood
  BEFORE the agent has ever felled a log or crafted anything. Three of four
  candidate designs derived interest from a graph built by experience while
  the experience required the interest: a latch (nine in this project's
  history). Consequence deficit passes because the evidence is ALREADY
  BANKED — thousands of dirt breaks with a closed consequence set, against
  ~0 caused events for a category seen constantly.

  THE FARM TEST — every income stream a policy can cycle WILL be cycled
  (five defunded here). Possession income keys on the SET of kinds held, so
  dropping and re-taking, or stacking more of the same kind, pays nothing.

  RANKING NEVER PAYS — the magnet's score is a payer-scaler whose output is
  written to the replay buffer as the world model's reward label, so the
  deficit must never reach it. Enforced here by asserting the module has no
  path into reward magnitude.

Run: PYTHONPATH=. python tests/_consequence_smoke.py
"""
import os
import sys
import tempfile

sys.path.insert(0, ".")

from developmental_ai.infra.consequence import ConsequenceMap  # noqa: E402


def test_bootstrap_from_banked_evidence():
    """THE TEST THAT MATTERS. Replays the agent's real measured history:
    dirt seen constantly AND acted on thousands of times; tree seen
    constantly and never once acted on. No success required."""
    cm = ConsequenceMap({"possession_weight": 0.2})
    for _ in range(400):
        # both are in view; only dirt is ever CAUSED (a break lands)
        cm.observe({"dirt_visible": 0.9, "tree_visible": 0.6}, caused=False)
    for _ in range(200):
        # the dirt-grinding history: attention spikes on dirt as it breaks
        cm.observe({"dirt_visible": 0.99, "tree_visible": 0.6}, caused=True)
    d_dirt = cm.deficit("dirt_visible")
    d_tree = cm.deficit("tree_visible")
    assert d_tree > d_dirt, (d_tree, d_dirt)
    assert d_tree > 0.9, d_tree
    assert d_dirt < 0.2, d_dirt
    print(f"  1. BOOTSTRAP: tree deficit {d_tree:.2f} >> dirt {d_dirt:.2f} "
          f"with ZERO logs ever felled — banked evidence alone ranks the "
          f"unconsummated category first")


def test_always_present_category_cannot_inflate_itself():
    """A predicate reading 'present' half the time must not bank contact for
    events that had nothing to do with it — that would suppress exactly what
    the mechanism exists to promote (this killed two designs)."""
    cm = ConsequenceMap()
    for _ in range(600):
        # 'ambient' sits at a constant 0.5 forever; 'target' genuinely spikes
        cm.observe({"ambient": 0.5, "target": 0.05}, caused=False)
    for _ in range(100):
        cm.observe({"ambient": 0.5, "target": 0.95}, caused=True)
    assert cm.deficit("ambient") > 0.9, cm.deficit("ambient")
    assert cm.deficit("target") < 0.2, cm.deficit("target")
    print(f"  2. contrastive credit: a flat-0.5 predicate banks ~no contact "
          f"(deficit {cm.deficit('ambient'):.2f}) while the SPIKING one is "
          f"consummated ({cm.deficit('target'):.2f})")


def test_unseen_is_not_interesting():
    """'I have no idea' must not read as 'maximally interesting', or a dead
    sensor captures attention permanently."""
    cm = ConsequenceMap()
    assert cm.deficit("never_seen") == 0.0
    for _ in range(5):
        cm.observe({"barely": 0.5}, caused=False)
    assert cm.deficit("barely") == 0.0, "claimed a deficit on thin evidence"
    print("  3. an unseen / barely-seen category reports deficit 0.0 — "
          "ignorance is not interest")


def test_possession_farm_resistance():
    cm = ConsequenceMap({"possession_weight": 0.2})
    first, is_new = cm.possession_income({"log": 1}, step=100)
    assert is_new and first > 0, (first, is_new)
    # holding MORE of the same kind is not a new situation
    same, _ = cm.possession_income({"log": 9}, step=101)
    assert same == 0.0, same
    # THE FARM: drop and re-take, cycling between two known sets. Under the
    # first (1/sqrt) form this paid 2.085 vs a 0.2 first entry — a 10x farm
    # for two buttons, because sum(1/sqrt(k)) diverges.
    tot = 0.0
    for i in range(400):
        a, _ = cm.possession_income({}, step=200 + 2 * i)
        b, _ = cm.possession_income({"log": 1}, step=201 + 2 * i)
        tot += a + b
    assert tot == 0.0, f"drop/retake farm still pays {tot:.3f}"
    # a genuinely NEW rung — one that SPENDS the last (goal-oriented depth,
    # 2026-08-16) — pays more than the first. Note {"log":1,"planks":1}
    # would NOT: gaining without spending is a pickup, not a rung.
    nxt, is_new2 = cm.possession_income({"planks": 4}, step=999)
    assert is_new2 and nxt > first, (nxt, first)
    # ...and that rung is likewise one-shot: leave it and come back
    cm.possession_income({"log": 1}, step=1100)
    again, _ = cm.possession_income({"planks": 4}, step=1200)
    assert again == 0.0, again
    print(f"  4. possession: first entry {first:.3f}; 400 drop/retake "
          f"cycles pay {tot:.3f} (was 2.085 under 1/sqrt); a NEW rung "
          f"pays {nxt:.3f}, and re-entering it pays {again:.3f}")


def test_granted_items_are_not_earned():
    """An axe handed over by a human, or inventory restored on respawn, is
    not a frontier the agent crossed."""
    cm = ConsequenceMap({"possession_weight": 0.2})
    cm.suppress(until_step=500)
    paid, is_new = cm.possession_income({"iron_axe": 1}, step=100)
    assert paid == 0.0 and is_new, (paid, is_new)
    # ...but it IS counted, so it cannot be re-sold later
    again, is_new2 = cm.possession_income({"iron_axe": 1}, step=900)
    assert again == 0.0 and not is_new2
    print("  5. a granted item pays 0 while suppressed, and is still "
          "COUNTED so it cannot be re-sold once suppression lifts")


def test_persistence_round_trip():
    """These counters ARE the bootstrap; starting empty each boot would
    recreate the latch."""
    cm = ConsequenceMap({"possession_weight": 0.2})
    for _ in range(300):
        cm.observe({"tree_visible": 0.7, "dirt_visible": 0.9}, caused=False)
    for _ in range(120):
        cm.observe({"tree_visible": 0.7, "dirt_visible": 0.99}, caused=True)
    cm.possession_income({"log": 1}, step=10)
    before = cm.deficit("tree_visible"), cm.deficit("dirt_visible")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "consequence_state.json")
        assert cm.save(p)
        fresh = ConsequenceMap({"possession_weight": 0.2})
        assert fresh.deficit("tree_visible") == 0.0, "fresh should know nothing"
        assert fresh.load(p)
    after = fresh.deficit("tree_visible"), fresh.deficit("dirt_visible")
    assert abs(after[0] - before[0]) < 1e-9 and abs(after[1] - before[1]) < 1e-9
    assert fresh._set_counts.get("log") == 1
    print(f"  6. state round-trips: tree {after[0]:.2f} / dirt {after[1]:.2f} "
          f"survive a restart, so the bootstrap is not re-latched")


def test_never_raises_and_pays_nothing_by_default():
    cm = ConsequenceMap()
    # possession income is OFF unless a weight is configured
    assert cm.possession_income({"log": 1}, step=1) == (0.0, False)
    # malformed inputs are data, not exceptions
    cm.observe(None)
    cm.observe({"x": float("nan")}, caused=True)
    cm.observe({"y": "not-a-number"}, caused=True)
    assert cm.possession_income("garbage", step=1)[0] == 0.0
    assert cm.deficit("x") == 0.0
    assert isinstance(cm.report(), str)
    # and the module cannot reach reward magnitude: deficit() is the only
    # ranking export, and it is bounded [0,1]
    for v in cm.deficits().values():
        assert 0.0 <= v <= 1.0
    print(f"  7. off by default, never raises ({len(cm.errors)} errors "
          f"recorded as data), deficits bounded [0,1]")


if __name__ == "__main__":
    for fn in (test_bootstrap_from_banked_evidence,
               test_always_present_category_cannot_inflate_itself,
               test_unseen_is_not_interesting,
               test_possession_farm_resistance,
               test_granted_items_are_not_earned,
               test_persistence_round_trip,
               test_never_raises_and_pays_nothing_by_default):
        print(f"[consequence] {fn.__name__}")
        fn()
    print("[consequence] ALL PASS")


# ---------------------------------------------------------------------- #
# WIRING CONTRACTS — the separation that makes this safe
# ---------------------------------------------------------------------- #

def test_ranking_never_touches_the_payer():
    """THE LOAD-BEARING SEPARATION. `_score` sets `_w`, which scales the
    magnet shaping, which is added into `prim_extrinsic`, which is written
    to the replay buffer AS THE WORLD MODEL'S REWARD LABEL. Deficit must
    therefore reorder targets without changing any magnitude the agent is
    paid. Three of four candidate designs died here."""
    from developmental_ai.llm.vision_scaffold import VisionScaffold
    vs = VisionScaffold(weight=0.5, promise_weight=1.0)
    vs._cat_lp = {"a": 0.20, "b": 0.19}
    vs._global_lp = 0.0
    s_a, s_b = vs._score("a"), vs._score("b")
    vs.set_deficit_source(lambda c: 1.0 if c == "b" else 0.0)
    # ranking flips (b was behind on curiosity, but is unconsummated)...
    assert vs._rank("b") > vs._rank("a"), (vs._rank("a"), vs._rank("b"))
    # ...while the PAID quantity is untouched
    assert vs._score("a") == s_a and vs._score("b") == s_b
    print(f"  8. deficit flips RANK (b {vs._rank('b'):.3f} > a "
          f"{vs._rank('a'):.3f}) while _score is byte-identical "
          f"({s_a:.3f}/{s_b:.3f}) — never reaches the reward label")


def test_mastered_category_cannot_be_resurrected():
    """0 * anything is 0: a fully-understood category stays uninteresting no
    matter how unconsummated it is. Prevents deficit becoming an attractor
    for things that can never be affected (water)."""
    from developmental_ai.llm.vision_scaffold import VisionScaffold
    vs = VisionScaffold(weight=0.5, promise_weight=5.0)
    vs._cat_lp = {"mastered": 0.0}
    vs._global_lp = 0.0
    vs.set_deficit_source(lambda c: 1.0)
    assert vs._score("mastered") <= 0.0
    assert vs._rank("mastered") == vs._score("mastered")
    print("  9. a mastered category (score<=0) is NOT resurrected by a "
          "maximal deficit — LP retires what deficit alone would chase")


def test_default_off_everywhere_else():
    from developmental_ai.llm.vision_scaffold import VisionScaffold
    vs = VisionScaffold(weight=0.5)          # no promise_weight
    vs._cat_lp = {"a": 0.2}
    vs._global_lp = 0.0
    vs.set_deficit_source(lambda c: 1.0)
    assert vs._rank("a") == vs._score("a"), "default config changed meaning"
    print(" 10. promise_weight defaults to 0.0 — every other config ranks "
          "by pure learning progress, byte-identical")


for _fn in (test_ranking_never_touches_the_payer,
            test_mastered_category_cannot_be_resurrected,
            test_default_off_everywhere_else):
    print(f"[consequence] {_fn.__name__}")
    _fn()
print("[consequence-wiring] ALL PASS")


def test_loop_init_block_actually_runs():
    """SILENT-DEATH GUARD. The first deploy of this feature initialised the
    map inside a try/except that caught `name '_os' is not defined` — an
    alias that only exists in `_resume_components` — so the config was read,
    the flag was on, and the subsystem was INERT while the run looked
    healthy. Only the warning line revealed it.

    This executes the real init block's imports and path build the way
    __init__ does, so a name error there fails a test instead of a run."""
    import os as _real_os
    import developmental_ai.core.developmental_loop as dl
    src = open(dl.__file__).read()
    i = src.index("self.consequence = None")
    j = src.index("self.infra = None", i)
    block = src[i:j]
    # every attribute-style name used in the block must resolve in the
    # module's own globals (this is what `_os` failed)
    import re
    for name in sorted(set(re.findall(r"\b(_[a-z]+)\.", block))):
        if name in ("_cq_cfg", "_cqp", "_e"):
            continue                       # locals defined inside the block
        assert name in vars(dl), (
            f"init block uses `{name}.` which is not a module global — "
            f"this is the exact bug that made the subsystem inert")
    assert "os.path.join" in block, "path build not using the module's os"
    # and the real join works
    assert _real_os.path.join("runlogs", "consequence_state.json")
    print(" 11. loop init block references only resolvable names — the "
          "`_os` silent-death cannot recur")


print("[consequence] test_loop_init_block_actually_runs")
test_loop_init_block_actually_runs()
print("[consequence-init] ALL PASS")


def test_unaffectable_target_retires_itself():
    """THE LIVE TRAP (2026-08-14). The magnet locked onto stone, which is
    unbreakable barehanded. 400 fruitless swings, 182 of them >=8 ticks, max
    streak 202, ZERO breaks — so nothing ever closed stone's consequence
    set, its deficit stayed maximal, and it held attention indefinitely.

    Crediting attempts lets a dead end retire itself. Note this is what the
    user asked for from the other direction too: the agent should be ABLE to
    act on a thing and get nothing, because that is the only way to learn
    the thing is worthless."""
    cm = ConsequenceMap({"attempt_scale": 0.25})
    # both seen constantly; the agent swings at BOTH and only one yields
    for _ in range(300):
        cm.observe({"stone": 0.4, "tree": 0.4}, caused=False)
    for _ in range(200):
        # swinging at stone: attention spikes, nothing ever breaks
        cm.observe({"stone": 0.95, "tree": 0.3}, attempted=True)
    d_stone = cm.deficit("stone")
    assert d_stone < 0.2, f"an unaffectable target never retired: {d_stone}"
    # a category never acted on at all keeps its deficit
    assert cm.deficit("tree") > 0.9, cm.deficit("tree")
    print(f"  12. 200 fruitless swings retire stone (deficit {d_stone:.2f}) "
          f"while untried tree stays open ({cm.deficit('tree'):.2f})")


def test_attempt_is_weaker_evidence_than_outcome():
    """A single failed swing may be bad aim, not an inert target — so it
    must take several attempts to close what one success closes."""
    a = ConsequenceMap({"attempt_scale": 0.25})
    b = ConsequenceMap({"attempt_scale": 0.25})
    for _ in range(300):
        a.observe({"x": 0.4}, caused=False)
        b.observe({"x": 0.4}, caused=False)
    for _ in range(20):
        a.observe({"x": 0.95}, caused=True)        # 20 successes
        b.observe({"x": 0.95}, attempted=True)     # 20 failed swings
    assert a.deficit("x") < b.deficit("x"), (a.deficit("x"), b.deficit("x"))
    print(f"  13. 20 outcomes close harder ({a.deficit('x'):.3f}) than 20 "
          f"attempts ({b.deficit('x'):.3f}) — attempts are weaker evidence")


def test_attempts_off_restores_old_behaviour():
    cm = ConsequenceMap({"attempt_scale": 0.0})
    for _ in range(300):
        cm.observe({"stone": 0.4}, caused=False)
    for _ in range(300):
        cm.observe({"stone": 0.95}, attempted=True)
    assert cm.deficit("stone") > 0.9, "attempt_scale=0 should ignore attempts"
    print(" 14. attempt_scale=0.0 reproduces outcome-only behaviour exactly")


for _fn in (test_unaffectable_target_retires_itself,
            test_attempt_is_weaker_evidence_than_outcome,
            test_attempts_off_restores_old_behaviour):
    print(f"[consequence] {_fn.__name__}")
    _fn()
print("[consequence-attempts] ALL PASS")


def test_chain_depth_outpays_acquisition():
    """USER DIRECTIVE (2026-08-15): a multi-step chain — get wood -> craft
    planks -> make a crafting table — must be rewarded BY FAR MORE than
    collecting wood.

    Depth is read off the agent's own inventory; nothing here knows that
    planks come from logs. The same scoring applies to any chain in any
    domain."""
    cm = ConsequenceMap({"possession_weight": 0.3, "depth_exponent": 2.0})
    # each rung SPENDS the previous one — the transformation signature
    # (revised 2026-08-16: the old version added kinds without consuming
    # any, which is a pile, and is now correctly all depth 1)
    wood, _ = cm.possession_income({"log": 3}, step=10)
    planks, _ = cm.possession_income({"log": 1, "planks": 8}, step=20)
    table, _ = cm.possession_income(
        {"log": 1, "planks": 4, "crafting_table": 1}, step=30)
    assert planks > wood and table > planks, (wood, planks, table)
    assert table >= 8 * wood, f"a 3-rung chain only pays {table/wood:.1f}x"
    # and the whole chain is still one-shot: re-walking it pays nothing
    again = sum(cm.possession_income(s, step=100 + i)[0] for i, s in enumerate(
        [{"log": 3}, {"log": 1, "planks": 8},
         {"log": 1, "planks": 4, "crafting_table": 1}]))
    assert again == 0.0, again
    print(f" 15. chain pays {wood:.2f} -> {planks:.2f} -> {table:.2f} "
          f"({table/wood:.0f}x for the 3rd rung vs collecting wood); "
          f"re-walking the whole chain pays {again:.2f}")


def test_depth_cannot_be_farmed_by_hoarding():
    """Depth must reward a CHAIN, not a pile. Picking up many unrelated
    kinds would otherwise mint deep sets for free."""
    cm = ConsequenceMap({"possession_weight": 0.3, "depth_exponent": 2.0})
    # each new kind does deepen the set — that is intended — but every set
    # is one-shot, so the total is bounded by kinds ever held, not cyclable
    tot = 0.0
    for i in range(30):
        items = {f"junk{j}": 1 for j in range(i + 1)}
        tot += cm.possession_income(items, step=i)[0]
    before = tot
    # re-walking every one of those sets pays exactly nothing
    for i in range(30):
        items = {f"junk{j}": 1 for j in range(i + 1)}
        tot += cm.possession_income(items, step=500 + i)[0]
    assert tot == before, "hoarding loop re-paid"
    print(f" 16. 30 deepening sets pay {before:.1f} once and {tot-before:.1f} "
          f"on every repeat — depth scales a ONE-SHOT payment, not a cycle")


def test_depth_exponent_one_is_flat():
    cm = ConsequenceMap({"possession_weight": 0.3, "depth_exponent": 1.0})
    a, _ = cm.possession_income({"x": 3}, step=1)
    b, _ = cm.possession_income({"x": 1, "y": 1}, step=2)   # spends x
    assert b == 2 * a, (a, b)
    print(" 17. depth_exponent=1.0 gives flat per-kind scaling (opt-out)")


for _fn in (test_chain_depth_outpays_acquisition,
            test_depth_cannot_be_farmed_by_hoarding,
            test_depth_exponent_one_is_flat):
    print(f"[consequence] {_fn.__name__}")
    _fn()
print("[consequence-depth] ALL PASS")


def test_depth_is_capped():
    cm = ConsequenceMap({"possession_weight": 0.3, "depth_exponent": 2.0,
                         "max_depth": 8})
    deep, _ = cm.possession_income({f"k{i}": 1 for i in range(30)}, step=1)
    assert deep <= 0.3 * 64 + 1e-9, deep
    print(f" 18. a 30-kind pile is capped at depth 8 -> {deep:.1f} "
          f"(~one felled log), not 270 — a chain leads without swamping")


print("[consequence] test_depth_is_capped"); test_depth_is_capped()
print("[consequence-cap] ALL PASS")


def test_depth_requires_transformation_not_variety():
    """GOAL-ORIENTED DEPTH (2026-08-16, user directive).

    LIVE FAILURE this replaces: counting distinct kinds paid 4.8 — a
    quarter of a felled log — for holding dirt|poppy|stick|wheat_seeds,
    incidental drops from breaking grass. It rewarded VARIETY, so the agent
    went back to grinding ground cover and the median swing hold collapsed
    20t -> 2t.

    A rung now requires a TRANSFORMATION: a new kind appears while an
    existing kind is CONSUMED — what crafting looks like from outside, and
    what foraging does not."""
    cm = ConsequenceMap({"possession_weight": 0.3, "depth_exponent": 2.0})
    # THE PILE: four unrelated pickups, nothing ever consumed
    pile = 0.0
    for items in ({"dirt": 1}, {"dirt": 2, "stick": 1},
                  {"dirt": 2, "stick": 1, "wheat_seeds": 3},
                  {"dirt": 2, "stick": 1, "wheat_seeds": 3, "poppy": 1}):
        pile += cm.possession_income(items, step=1)[0]
    assert pile == 4 * 0.3, f"a pile still earns depth: {pile}"

    # THE CHAIN: each rung SPENDS the last one
    cm2 = ConsequenceMap({"possession_weight": 0.3, "depth_exponent": 2.0})
    wood, _ = cm2.possession_income({"log": 3}, step=1)
    planks, _ = cm2.possession_income({"log": 1, "planks": 8}, step=2)
    table, _ = cm2.possession_income(
        {"log": 1, "planks": 4, "crafting_table": 1}, step=3)
    assert wood == 0.3, wood
    assert planks == 0.3 * 4, planks
    assert table == 0.3 * 9, table
    assert table >= 8 * wood
    print(f"  19. PILE of 4 unrelated kinds pays {pile:.2f} (all depth 1); "
          f"a CHAIN that spends each rung pays {wood:.2f} -> {planks:.2f} "
          f"-> {table:.2f} ({table/wood:.0f}x)")


def test_gain_without_spend_is_not_a_rung():
    cm = ConsequenceMap({"possession_weight": 0.3, "depth_exponent": 2.0})
    cm.possession_income({"log": 1}, step=1)
    # gained planks but consumed NOTHING (e.g. picked them up) -> depth 1
    got, _ = cm.possession_income({"log": 1, "planks": 4}, step=2)
    assert got == 0.3, got
    print("  20. gaining a kind WITHOUT spending one restarts at depth 1 — "
          "finding planks is not making them")


for _fn in (test_depth_requires_transformation_not_variety,
            test_gain_without_spend_is_not_a_rung):
    print(f"[consequence] {_fn.__name__}")
    _fn()
print("[consequence-goal-depth] ALL PASS")
