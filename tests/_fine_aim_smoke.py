"""Action-space smoke (2026-09-19, roadmap A5/A7/A8/A9).

THE ARITHMETIC THAT FORCED THIS CHANGE

    Every camera action in the macro table moved the view by 15 degrees. At
    Minecraft's default 70-degree FOV that is 21% OF THE FRAME per press. A
    one-block target subtends:

        2 blocks   28.1 deg   fine
        5 blocks   11.4 deg   one step overshoots
       10 blocks    5.7 deg   the step is 2.6x the TARGET
       20 blocks    2.9 deg   the step is 5x the target

    So approaching a tree carried +/-7.5 degrees of UNAVOIDABLE aim error —
    about +/-1.3 blocks of lateral drift at 10 blocks — and every
    close-range correction then swung the view clean past the trunk. That is
    a limit cycle, and it is a better mechanical account of the scoreboard
    (13,305 blocks broken, 35 logs) than any perception story: the agent
    gets near trees, cannot converge on the trunk, and whatever sits under
    an unsteerable crosshair at ground level is dirt.

    PRECEDENT FOR THE SHAPE: VPT, the strongest Minecraft agent ever built,
    uses mu-law quantised camera bins — fine near zero, coarse far out.
    15/5/2 is that shape at three levels, so the agent can still turn fast.

    NOTHING HERE IS A SKILL. These are finer versions of buttons the agent
    already had. What to aim at, and when, remains entirely unlearned.

WHY THIS FILE PARSES SOURCE INSTEAD OF IMPORTING
    `minerl_env` imports gymnasium, which is not installed on a development
    Mac — so an import-based contract would be unrunnable exactly where it is
    most useful. The macro table is a module-level literal; reading it with
    `ast` is exact, and it keeps this contract offline-runnable everywhere.

Contracts:
    A. THREE camera magnitudes exist (15 / 5 / 2 degrees).
    B. APPEND ONLY — indices 0-12 keep their exact meaning. Every saved
       skill, policy and config indexes actions BY POSITION; rebinding index
       3 silently relabels every frozen skill that ever pressed it.
    C. The new indices add ONLY camera steps — a finer button, never a new
       capability smuggled in beside it.
    D. SYMMETRY — both axes, both signs, at every magnitude. A one-sided
       widening lets the agent correct left but not right, which is the
       crosshair equivalent of a one-way door.
    E. The finest step actually resolves a block at working distance, which
       is the whole point of the change.

Run: PYTHONPATH=. python tests/_fine_aim_smoke.py
"""
import ast
import math
import os
import sys

sys.path.insert(0, ".")

SRC = os.path.join("developmental_ai", "environments", "minerl_env.py")
FOV_DEG = 70.0          # Minecraft's default vertical FOV
FROZEN = 13             # indices 0..12 predate the aim widening
AIM_END = 21            # the aim block is [FROZEN, AIM_END); A7/A8/A9 follow


def _macros():
    """The TREECHOP_MACROS literal, read without importing the module."""
    tree = ast.parse(open(SRC).read())
    for node in tree.body:
        tgt = None
        if isinstance(node, ast.AnnAssign):
            tgt = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and node.targets:
            tgt = getattr(node.targets[0], "id", None)
        if tgt == "TREECHOP_MACROS":
            return ast.literal_eval(node.value)
    raise AssertionError(f"TREECHOP_MACROS not found in {SRC}")


def _subtends(blocks: float) -> float:
    """Angular size of a one-block face at `blocks` distance, in degrees."""
    return math.degrees(2.0 * math.atan(0.5 / blocks))


def test_three_magnitudes_exist():
    m = _macros()
    cam = [(i, d["camera"]) for i, d in enumerate(m) if "camera" in d]
    mags = sorted({abs(v) for _i, c in cam for v in c if v})
    assert mags == [2.0, 5.0, 15.0], (
        f"camera magnitudes are {mags}; expected 2/5/15. A single 15-degree "
        f"quantum cannot aim at a block beyond ~4 blocks (see docstring)")
    print(f"  A. {len(m)} macros, {len(cam)} camera actions, "
          f"magnitudes {mags} deg")


def test_append_only():
    m = _macros()
    assert len(m) > FROZEN, "no new macros were appended"
    assert m[3]["camera"] == [0.0, -15.0]
    assert m[4]["camera"] == [0.0, 15.0]
    assert m[7]["camera"] == [-15.0, 0.0]
    assert m[8]["camera"] == [15.0, 0.0]
    assert m[12] == {"attack": 1, "_ticks": 40}, (
        "HOLD attack moved off index 12 — the log economics reference it "
        "by position")
    for i in (0, 1, 2, 5, 6, 9, 10, 11):
        assert "camera" not in m[i], (
            f"macro {i} became a camera action; indices 0-{FROZEN - 1} are "
            f"frozen and every stored skill indexes them by position")
    print(f"  B. indices 0-{FROZEN - 1} unchanged; {len(m) - FROZEN} appended")


def test_new_indices_add_only_camera():
    """The AIM BLOCK adds only finer aim — no capability rides along with it.

    Bounded to [FROZEN, AIM_END) rather than "everything after FROZEN",
    because A7/A8/A9 later appended a second block. The point of the
    contract is unchanged: a widening that claims to be about aim must not
    quietly introduce a button, and the two blocks stay separately auditable.
    """
    m = _macros()
    assert len(m) >= AIM_END
    for i in range(FROZEN, AIM_END):
        assert set(m[i].keys()) == {"camera"}, (
            f"macro {i} is {m[i]}; the aim widening must add ONLY camera "
            f"steps, never a new capability beside one")
        assert "_ticks" not in m[i], f"macro {i} smuggled in a hold duration"
    for i in range(AIM_END, len(m)):
        assert "camera" not in m[i], (
            f"macro {i} is a camera action outside the aim block [{FROZEN}, "
            f"{AIM_END}); aim changes belong together")
    print(f"  C. aim block [{FROZEN}, {AIM_END}) is pure camera; "
          f"{len(m) - AIM_END} later macros contain none")


def test_symmetry():
    m = _macros()
    cam = [d["camera"] for d in m if "camera" in d]
    for mag in (15.0, 5.0, 2.0):
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                want = [0.0, 0.0]
                want[axis] = sign * mag
                assert want in cam, (
                    f"no camera macro for {want} — aim correction is "
                    f"asymmetric at {mag} deg, which is a one-way door for "
                    f"the crosshair")
    print("  D. both axes, both signs, at all three magnitudes")


def test_finest_step_resolves_a_block():
    """E. The change has to actually buy aim, not just more buttons."""
    m = _macros()
    finest = min(abs(v) for d in m if "camera" in d
                 for v in d["camera"] if v)
    before = 15.0

    rows = []
    for dist in (2, 5, 10, 20):
        ang = _subtends(dist)
        rows.append((dist, ang, before / ang, finest / ang))

    # At 10 blocks the OLD step was 2.6x the target; the new one must be
    # comfortably inside it, or nothing has changed where it matters.
    d10 = next(r for r in rows if r[0] == 10)
    assert d10[2] > 2.0, (
        f"the 15-degree step is only {d10[2]:.1f}x a block at 10 blocks — "
        f"recheck the FOV assumption before trusting this contract")
    assert d10[3] < 0.5, (
        f"the finest step is {d10[3]:.2f}x a block at 10 blocks; it must be "
        f"well under 1.0 or the agent still cannot centre a distant trunk")

    for dist, ang, old_r, new_r in rows:
        print(f"      {dist:2d} blocks: target {ang:5.1f} deg | "
              f"15 deg = {old_r:4.1f}x | {finest:.0f} deg = {new_r:4.2f}x")
    print(f"  E. finest step {finest:.0f} deg resolves a block at 10 blocks "
          f"({d10[3]:.2f}x) where 15 deg overshot by {d10[2]:.1f}x")


def test_player_hands_are_complete():
    """A7/A8/A9 — the buttons a player has, that the agent did not.

    STRAFE is the one that matters most. Turning and moving are coupled
    through the heading, so ORBITING a tree while keeping it centred was not
    expressible at all: the agent could only approach head-on, and every
    correction was a 15-degree swing (see the table above).

    HOLD DURATION had exactly one value. The config's own arithmetic: a
    barehanded log needs ~60 ticks and an iron axe ~8, so one fixed 40-tick
    hold is either far too long for a tool or two thirds too short for bare
    hands. It could not serve both and nothing could choose.

    HOTBAR KEYS are the only way the agent can ever deliberately equip
    anything. CLAUDE.md 7 records that there is no equip action and
    `equipped_items` is dead in MineRL 1.0, so mainhand is DERIVED FROM
    EVIDENCE — the agent has been holding whatever it happened to be holding.

    NONE OF THIS ENCODES A RECIPE OR A TOOL. They are buttons whose effects
    must be discovered by pressing them, exactly like `use` and `inventory`.
    """
    m = _macros()

    def has(*keys):
        want = set(keys)
        return any(set(d.keys()) == want for d in m)

    for k in ("left", "right", "sneak"):
        assert has(k), f"no macro presses {k!r}; a player has it"
    assert has("forward", "sprint"), "no sprint"
    assert has("jump"), "no standalone jump (jump-walk is not the same button)"

    holds = sorted(d["_ticks"] for d in m if "_ticks" in d)
    assert len(holds) >= 3, (
        f"attack hold durations are {holds}; one value cannot serve both a "
        f"~60-tick barehanded log and an ~8-tick axe")
    assert min(holds) <= 12, f"no short hold for a tool ({holds})"
    assert max(holds) >= 100, f"no long hold for bare hands ({holds})"
    assert max(holds) <= 120, (
        f"a hold of {max(holds)} ticks exceeds the 120 cap; `attack_run` has "
        f"been observed at 13538 and the cap is what bounds the payout")

    # HOTBAR KEYS ARE DELIBERATELY ABSENT, and that is a REVERSAL of what
    # this contract first asserted. The reasoning, kept because the next
    # person will have the same idea I did:
    #
    #   `tests/_action_widening_smoke.py` already carried "still no hotbar
    #   keys (deliberate: minimal exploration burden)" — a decision pinned
    #   in a test. Adding nine took the space from 28 to 37 macros, and then
    #   STAGE 0 MEASURED that MineRL does not render the HUD (bottom-band
    #   variance 0.029 vs 0.290 mid-frame). So the agent cannot see its
    #   selection change: nine actions with no perceptible effect, over an
    #   inventory that has never held two things at once.
    #
    # The measurement made the original decision MORE right, not less. This
    # asserts the reversal so nobody re-adds them without a selection sense.
    assert not any("hotbar" in k for d in m for k in d), (
        "hotbar keys are back; see the block in minerl_env.py — they are "
        "nine invisible actions until the agent can perceive its selection")

    # APPEND ONLY, still. The A5 block froze 0-12; this freezes 13-20 too.
    assert m[20]["camera"] == [2.0, 0.0], (
        "the fine-aim macros moved; indices 13-20 are frozen now as well")
    for i in range(21, len(m)):
        assert "camera" not in m[i], (
            f"macro {i} is a camera action outside the aim block")

    print(f"  F. {len(m)} macros: strafe/sprint/sneak/jump present, holds "
          f"{holds}, no hotbar keys (see docstring), indices 0-20 frozen")


def test_no_one_way_doors_in_the_new_buttons():
    """Every state a new button OPENS must have something that CLOSES it.

    CLAUDE.md 4.5, and the incident behind it: `use` opens a villager trade
    screen, `inventory` is the only thing that closes one, and disabling
    `inventory` left SkyBot sitting in a trade menu for 10,149 consecutive
    steps. A button that enters a state with no exit is the same trap.
    """
    m = _macros()
    keys = {k for d in m for k in d}

    # sneak is a HOLD in Minecraft, not a toggle — it ends when not pressed,
    # so every other macro releases it. Same for sprint, left, right.
    for held in ("sneak", "sprint", "left", "right"):
        if held in keys:
            assert any(held not in d for d in m), (
                f"{held!r} is pressed by every macro, so it can never be "
                f"released")

    # The GUI pair is unchanged: an entrance is never offered without its exit.
    if any("use" in d for d in m):
        assert any("inventory" in d for d in m), (
            "`use` opens villager and chest screens; `inventory` is the only "
            "close. This is the 10,149-step trade-menu incident exactly")

    print("  G. holds are releasable, GUI pair intact")


if __name__ == "__main__":
    test_three_magnitudes_exist()
    test_append_only()
    test_new_indices_add_only_camera()
    test_symmetry()
    test_finest_step_resolves_a_block()
    test_player_hands_are_complete()
    test_no_one_way_doors_in_the_new_buttons()
    print("[action-space] ALL PASS")
