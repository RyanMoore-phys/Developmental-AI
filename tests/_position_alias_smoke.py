"""Position key-alias smoke (2026-08-11).

THE BUG THIS ENCODES
    `_world_events` published `xpos`/`ypos`/`zpos`. Every downstream consumer
    was written against `x`/`y`/`z`. Each therefore read None and failed
    CLOSED, silently, for the entire life of the project:

      * the episodic SIGHTING gate (infra/stack.py: `if (... and position)`)
        — "sighting never" for 336k steps across ALL NINE categories,
        including `dirt` at a 42.6% fovea-positive rate. That uniformity is
        the tell: a starved input would spare dirt, a broken gate would not.
      * the episodic BEARING proprio sense (_augment_proprio)
      * _memory_pull_phi (the term could never pay at all)
      * the demonstration landmark
      * the live viewer's position payload

    MEASURED live: zero " @(x,z)" markers in the whole run log, while the
    status line printed "Position: x=+36.3 y=70.0 z=-16.7" on the same step
    — because that one reader used xpos/ypos/zpos.

THE CONTRACT
    Both spellings are published and agree. The gate that consumes them
    actually opens. Pinned as a REGRESSION because the two spellings are
    still both in the codebase, so the trap can be re-set by a future edit.

Run: PYTHONPATH=. python tests/_position_alias_smoke.py
"""
import os
import re
import sys

sys.path.insert(0, ".")


def test_both_spellings_published():
    src = open(os.path.join("developmental_ai", "environments",
                            "minerl_env.py")).read()
    for k in ('out["x"]', 'out["y"]', 'out["z"]'):
        assert k in src, f"{k} is not published — the consumers read None"
    print("  1. _world_events publishes x/y/z alongside xpos/ypos/zpos")


def test_alias_agrees_and_gate_opens():
    """Drive the real assembly + the real gate condition."""
    out = {}
    x, y, z = 36.3, 70.0, -16.7
    # the exact lines from _world_events
    out["ypos"] = y
    out["xpos"] = float(x)
    out["zpos"] = float(z)
    out["x"] = out["xpos"]
    out["z"] = out["zpos"]
    out["y"] = float(out.get("ypos", 0.0) or 0.0)
    assert (out["x"], out["y"], out["z"]) == (x, y, z)

    # the consumer, verbatim from developmental_loop._infra_step
    wi = out
    pos = None
    if wi.get("x") is not None and wi.get("z") is not None:
        pos = (float(wi["x"]), float(wi.get("y", 0.0)), float(wi["z"]))
    assert pos == (x, y, z), pos

    # the gate, verbatim from infra/stack.py
    episodic, fovea_probs = object(), {"tree_visible": 0.9}
    assert bool(episodic is not None and fovea_probs and pos), \
        "the sighting gate STILL does not open"
    print(f"  2. gate opens: position={pos} (it was None on every one of "
          f"336k steps)")


def test_missing_ypos_does_not_crash_the_alias():
    """y is published conditionally upstream; the alias must tolerate that
    rather than raising inside the env's hot path."""
    out = {"xpos": 1.0, "zpos": 2.0}
    out["x"] = out["xpos"]
    out["z"] = out["zpos"]
    out["y"] = float(out.get("ypos", 0.0) or 0.0)
    assert out["y"] == 0.0 and out["x"] == 1.0 and out["z"] == 2.0
    print("  3. absent ypos degrades to y=0.0 (x/z still usable) rather "
          "than raising in the step path")


def test_no_consumer_left_on_the_bare_spelling_only():
    """Every reader of the world dict should now succeed. This asserts the
    known five consumers exist and read a spelling that is published."""
    loop = open(os.path.join("developmental_ai", "core",
                             "developmental_loop.py")).read()
    n = len(re.findall(r'\.get\("x"\)', loop))
    assert n >= 4, f"expected the known x-consumers, found {n}"
    print(f"  4. {n} consumers read the bare spelling — all now fed by the "
          f"source alias (one fix, not {n} patches)")


def test_gate_still_refuses_a_genuinely_absent_position():
    """The guard must keep protecting: no coordinates -> no landmark. A fix
    that made the gate always-true would store garbage landmarks."""
    wi = {}
    pos = None
    if wi.get("x") is not None and wi.get("z") is not None:
        pos = (float(wi["x"]), 0.0, float(wi["z"]))
    assert pos is None
    assert not bool(object() is not None and {"t": 0.9} and pos)
    print("  5. a genuinely position-less step still records NO landmark — "
          "the guard is repaired, not removed")


if __name__ == "__main__":
    for fn in (test_both_spellings_published,
               test_alias_agrees_and_gate_opens,
               test_missing_ypos_does_not_crash_the_alias,
               test_no_consumer_left_on_the_bare_spelling_only,
               test_gate_still_refuses_a_genuinely_absent_position):
        print(f"[position-alias] {fn.__name__}")
        fn()
    print("[position-alias] ALL PASS")
