"""Does pressing the new `inventory` button make the agent INERT?

WHY: after the 10->12 action widening the live run went 21 segments with zero
extrinsic reward, and the attack-streak diagnostic hit 10726 consecutive game
ticks of pure attack with ZERO breaks — while holding an iron axe that fells a
log in ~8 ticks. `_attack_run` resets on any non-attack action AND on any
break, so that number can only mean: attack input is reaching the game and
nothing is breaking.

`inventory` (macro 10) is a TOGGLE. In Minecraft an open GUI swallows clicks —
attack no longer reaches the world. If the policy presses 10 and nothing
presses it again, the agent is a spectator holding down the mouse.

THE EXPERIMENT (causal, three phases, same aim throughout):
  A  attack barehanded-at-ground        -> breaks EXPECTED (control)
  B  press inventory once, then attack  -> breaks == 0 ? => the trap is real
  C  press inventory again, then attack -> breaks RESUME ? => it is the toggle

Frame-difference is recorded alongside, because "no breaks" alone cannot tell
"GUI swallowed the click" from "nothing breakable in front of me" — a big
pixel jump on the toggle press is the independent witness that the GUI opened.

Probes the PROJECT'S adapter (TreechopFixed, 12 macros), never the stock env —
a previous probe measured `MineRLTreechop-v0`'s 9-action space and returned a
false verdict.

    python scripts/probe_inventory_trap.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

ATTACK, PITCH_DOWN, INVENTORY = 5, 8, 10


def main() -> int:
    from developmental_ai.environments.minerl_env import (
        MineRLEnvAdapter, TREECHOP_MACROS)

    assert TREECHOP_MACROS[INVENTORY] == {"inventory": 1}, "macro map moved"
    assert TREECHOP_MACROS[ATTACK] == {"attack": 1}, "macro map moved"

    env = MineRLEnvAdapter()
    obs, _ = env.reset()                      # (obs, info) — gymnasium style
    print(f"[probe] env up | action_dim={len(TREECHOP_MACROS)}")

    def frame(o):
        a = o if isinstance(o, np.ndarray) else np.asarray(o)
        return a.astype(np.float32)

    # GROUND TRUTH break counter. `info` carries no "new_breaks" key — an
    # earlier draft of this probe invented one, which would have counted zero
    # breaks in EVERY phase and printed a confident "TRAP CONFIRMED".
    # `_mine_prev` is the env's own per-block stat dict; its sum only rises
    # when a block actually breaks.
    def mined():
        return sum(int(v) for v in getattr(env, "_mine_prev", {}).values())

    prev = frame(obs)

    def burst(n, action, tag):
        """Run `action` n times; report REAL breaks and mean frame change."""
        nonlocal prev
        before, diffs = mined(), []
        for _ in range(n):
            obs, _r, term, trunc, info = env.step(action)   # 5-tuple
            f = frame(obs)
            diffs.append(float(np.abs(f - prev).mean()))
            prev = f
            if term or trunc:
                obs, _ = env.reset()
                prev = frame(obs)
                before = mined()          # baseline moved; don't count a reset
        got = max(0, mined() - before)
        print(f"  {tag:<34} breaks={got:<3} "
              f"framediff={np.mean(diffs):6.2f} "
              f"attack_run={(info or {}).get('attack_run', '?')}")
        return got

    # aim at the ground so SOMETHING breakable is always under the crosshair —
    # otherwise "0 breaks" is trivially explained by facing the sky.
    for _ in range(4):
        env.step(PITCH_DOWN)
    prev = frame(env.step(PITCH_DOWN)[0])

    print("\n[A] control — attack with NO gui")
    a = burst(40, ATTACK, "attack x40")

    print("\n[B] press inventory ONCE, then attack")
    burst(1, INVENTORY, "inventory toggle (open?)")
    b = burst(80, ATTACK, "attack x80 (gui presumed open)")

    print("\n[C] press inventory AGAIN, then attack")
    burst(1, INVENTORY, "inventory toggle (close?)")
    c = burst(40, ATTACK, "attack x40")

    print("\n=== VERDICT ===")
    if a > 0 and b == 0 and c > 0:
        print("  TRAP CONFIRMED: one inventory press makes attack INERT until\n"
              "  pressed again. A policy that presses 10 and does not press it\n"
              "  again is a spectator — this explains the 10726-tick streak.")
        rc = 1
    elif a > 0 and b > 0:
        print("  NOT the trap: attack still breaks blocks with the gui open.\n"
              "  The zero-reward stall has another cause — do NOT 'fix' this.")
        rc = 0
    elif a == 0:
        print("  INCONCLUSIVE: the control broke nothing, so the probe never\n"
              "  established a baseline. Aim/terrain problem, not a gui result.")
        rc = 2
    else:
        print(f"  UNCLEAR: A={a} B={b} C={c} — read the frame diffs above.")
        rc = 2
    try:
        env.close()
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
