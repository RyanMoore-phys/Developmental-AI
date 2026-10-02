"""PROBE: can the agent SEE and MANIPULATE the crafting GUI at all?

This answers the single feasibility question behind learned crafting. Symbolic
craft handlers are unusable in this fork (the Java backend has no MissionHandlers
package and closes the socket on an unknown command), so the ONLY route to a
tool is the one a human uses: open the inventory, move a cursor, click. That is
only worth building if the agent can actually perceive and drive that screen.

Five questions, each MEASURED, none inferred:
  1. does the `inventory` key open the GUI?          (IsGuiOpen flips)
  2. does the GUI reach the agent's OWN frame?       (POV changes materially)
  3. do camera deltas move something while open?     (frame keeps changing)
  4. does `attack` register while the GUI is open?   (frame changes on click)
  5. is `craft_item` observable at all?              (counter key exists)

NOTHING here crafts anything and nothing is scripted toward a recipe. It probes
CAPABILITY. If these pass, the agent has real hands and real eyes in the menu
and can discover crafting itself. If any fail, we have a perception or control
problem to solve rather than a search problem — and that is worth knowing
BEFORE building on the assumption.

Run ON THE TRAINING HOST:
    DISPLAY=:77 MINERL_HEADLESS=1 ./venv_mc/bin/python scripts/probe_gui_capability.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.getcwd())


def _digest(pov) -> str:
    return hashlib.md5(np.asarray(pov, dtype=np.uint8).tobytes()).hexdigest()[:10]


def _frac_diff(a, b) -> float:
    """Fraction of pixels that changed — a material-change measure.

    A hash says "different"; a single dithered pixel would satisfy it. The GUI
    covers a large share of the screen, so the fraction is what distinguishes
    "the menu opened" from "a cloud moved".
    """
    a = np.asarray(a, dtype=np.int16)
    b = np.asarray(b, dtype=np.int16)
    if a.shape != b.shape:
        return 1.0
    return float((np.abs(a - b) > 8).mean())


def main() -> int:
    # PROBE THE ENV THE AGENT ACTUALLY LIVES IN, not the stock one. A first
    # cut called gym.make("MineRLTreechop-v0") and measured 9 action keys with
    # no `inventory` — a hard BLOCK that does not exist for us. Our spec
    # subclasses HumanSurvival -> HumanControlEnvSpec, whose create_actionables
    # returns the FULL 23-key KEYMAP. Measuring the wrong environment is the
    # same class of error as measuring the wrong log.
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter

    print("booting the PROJECT's env (TreechopFixed, one client, ~90s)...",
          flush=True)
    adapter = MineRLEnvAdapter(image_size=64, action_repeat=1, lifelong=True)
    adapter.reset()
    env = adapter._env                    # the underlying MineRL env
    obs = adapter._last_raw_obs if hasattr(adapter, "_last_raw_obs") else None
    if obs is None:
        _a = env.action_space.noop()
        obs, _, _, _ = env.step(_a)
    print("booted.", flush=True)

    noop = env.action_space.noop()
    keys = sorted(noop.keys())
    print(f"\nunderlying action keys ({len(keys)}): {keys}", flush=True)

    results = {}

    # --- Q5 first (free): is craft_item observable? ---------------------
    craft_keys = [k for k in obs.keys() if "craft" in str(k).lower()]
    results["craft_item_observable"] = bool(craft_keys)
    print(f"\n[Q5] craft_item in observation: {craft_keys or 'ABSENT'}")

    def step(**kw):
        a = env.action_space.noop()
        for k, v in kw.items():
            if k not in a:
                raise KeyError(f"action key {k!r} does not exist in this fork")
            a[k] = v
        o, r, d, i = env.step(a)
        return o, i, d

    # settle a few frames so we compare against a stable baseline
    for _ in range(5):
        obs, info, _ = step()
    base = obs["pov"]

    # --- Q1/Q2: does `inventory` open a GUI, and is it IN OUR FRAME? ----
    has_inv_key = "inventory" in noop
    results["inventory_key_exists"] = has_inv_key
    print(f"\n[Q1] 'inventory' action key exists: {has_inv_key}")
    if not has_inv_key:
        print("     -> GUI route is CLOSED at the action level.")
    else:
        obs, info, _ = step(inventory=1)
        for _ in range(4):                      # let the screen settle
            obs, info, _ = step()
        opened = obs["pov"]
        d_open = _frac_diff(base, opened)
        gui_flag = None
        for k in ("IsGuiOpen", "is_gui_open", "isGuiOpen"):
            if isinstance(info, dict) and k in info:
                gui_flag = bool(info[k])
                break
        results["gui_flag"] = gui_flag
        results["pov_change_on_open"] = d_open
        print(f"[Q1] IsGuiOpen after press: {gui_flag}")
        print(f"[Q2] POV changed on open  : {d_open:.1%} of pixels")
        print(f"     -> GUI {'IS' if d_open > 0.15 else 'is NOT'} rendering "
              f"into the agent's own frame")

        # --- Q3: does camera move anything while the GUI is open? -------
        pre = obs["pov"]
        for _ in range(3):
            obs, info, _ = step(camera=np.array([0.0, 12.0], dtype=np.float32))
        d_cam = _frac_diff(pre, obs["pov"])
        results["pov_change_on_camera_in_gui"] = d_cam
        print(f"[Q3] POV changed on camera while GUI open: {d_cam:.1%}")

        # --- Q4: does a click register? ---------------------------------
        pre = obs["pov"]
        for _ in range(2):
            obs, info, _ = step(attack=1)
        for _ in range(2):
            obs, info, _ = step()
        d_click = _frac_diff(pre, obs["pov"])
        results["pov_change_on_click_in_gui"] = d_click
        print(f"[Q4] POV changed on attack/click in GUI  : {d_click:.1%}")

        # close it again — leave the world as we found it
        obs, info, _ = step(inventory=1)

    print("\n" + "=" * 60)
    print("VERDICT")
    ok_open = results.get("pov_change_on_open", 0.0) > 0.15
    ok_ctrl = (results.get("pov_change_on_camera_in_gui", 0.0) > 0.005
               or results.get("pov_change_on_click_in_gui", 0.0) > 0.005)
    if not results.get("inventory_key_exists"):
        print("BLOCKED: no inventory action — GUI crafting is unreachable.")
    elif not ok_open:
        print("BLOCKED: the GUI does not reach the agent's frame. It cannot")
        print("         learn to use a screen it cannot see. Perception fix")
        print("         needed before any crafting work.")
    elif not ok_ctrl:
        print("PARTIAL: the GUI is VISIBLE but neither camera nor click")
        print("         changed it — the agent can see the menu and not")
        print("         operate it. Control problem, not a search problem.")
    else:
        print("CLEAR: the agent can open the GUI, it renders into its own")
        print("       frame, and its inputs change it. Discovery is possible;")
        print("       the remaining question is only how long it takes.")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")

    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("\nPROBE FAILED — the exception above is the finding.")
        raise SystemExit(2)
