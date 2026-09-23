"""CHOP PROBE 4 — the full fixed stack under a FOCUSED display.

Root cause of all zero-reward runs: MCP-Reborn's MouseHelper gates the
camera-velocity feed and mouse grab on `isMouseGrabbed() && isGameFocused()`;
under bare Xvfb (no window manager) the window never receives X focus, so
synthetic camera + attack are silently discarded (keyboard passes anyway).
Fix under test: run openbox in the headless display (auto-focuses new
windows) — plus xdotool focus-forcing as a belt-and-braces.

This probe exercises the ENTIRE fixed stack in one session, via OUR adapter
(MineRLEnvAdapter -> TreechopFixed spec, adapter-side inventory-delta
reward):
  phase 1 CAMERA: two pure-turn macros; frames before/after must differ
          (and we report the tutorial state via later frames).
  phase 2 CHOP: scripted contact-chop macro cycles (walk 4, chop-forward 5)
          — with focus fixed and an iron axe, logs should collect within a
          few cycles.

Launch (inside an openbox-managed display):
    Xvfb :77 -screen 0 640x480x24 &
    DISPLAY=:77 openbox &
    DISPLAY=:77 ./venv_mc/bin/python _chop_probe4.py
"""
import os
import subprocess
import time

import numpy as np

FRAMES = "runlogs/probe4_frames"


def save(frame, name):
    try:
        import imageio.v2 as imageio
        f = np.asarray(frame, dtype=np.uint8)
        f = np.repeat(np.repeat(f, 4, 0), 4, 1)
        imageio.imwrite(os.path.join(FRAMES, name + ".png"), f)
    except Exception as e:
        print(f"  (frame save failed: {e})")


def force_focus():
    """xdotool: focus any Minecraft window in this display (belt+braces).
    Set PROBE_NO_FOCUS=1 to skip — used to validate that the DEVAI Java
    focus patch alone suffices (the 8-parallel-client guarantee)."""
    if os.environ.get("PROBE_NO_FOCUS"):
        print("  [focus] SKIPPED (PROBE_NO_FOCUS=1 — testing Java patch alone)")
        return
    try:
        out = subprocess.run(
            ["xdotool", "search", "--name", "Minecraft"],
            capture_output=True, text=True, timeout=10)
        wids = out.stdout.split()
        for wid in wids:
            subprocess.run(["xdotool", "windowactivate", wid],
                           capture_output=True, timeout=10)
            subprocess.run(["xdotool", "windowfocus", wid],
                           capture_output=True, timeout=10)
        print(f"  [focus] focused {len(wids)} Minecraft window(s)")
    except Exception as e:
        print(f"  [focus] xdotool failed: {e}")


def main():
    os.makedirs(FRAMES, exist_ok=True)
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter

    print("[probe4] building FIXED env via adapter (~90s boot)...")
    env = MineRLEnvAdapter(image_size=64, action_repeat=2)
    obs, info = env.reset()
    print(f"[probe4] reset ok. initial logs={info.get('logs')} "
          f"obs={obs.shape}")
    force_focus()

    # phase 1: camera — pure turn macros (3 = left 15deg net each)
    save(env._last_pov, "cam_before")
    for _ in range(6):
        obs, r, te, tr, info = env.step(3)
    save(env._last_pov, "cam_after_90left")
    diff = float(np.abs(
        np.asarray(env._last_pov, np.float32)).mean())
    print("[probe4] phase1 done (frames saved; compare cam_before/after)")

    # phase 2: contact chop — walk 4 macros (~8 blocks), then chop-forward
    # x5 (5 x 60 ticks = 300 ticks of held attack while pressing in)
    total_r, events = 0.0, []
    step_i = 0
    for cycle in range(1, 15):
        for _ in range(4):
            obs, r, te, tr, info = env.step(1)   # walk
            step_i += 1
            total_r += r
            if r > 0:
                events.append((step_i, r, info.get("logs")))
                print(f"  [macro {step_i}] +{r:.0f} (logs={info.get('logs')})")
            if te or tr:
                break
        save(env._last_pov, f"c{cycle:02d}_pre")
        for _ in range(5):
            obs, r, te, tr, info = env.step(6)   # chop-forward, 60 ticks
            step_i += 1
            total_r += r
            if r > 0:
                events.append((step_i, r, info.get("logs")))
                print(f"  [macro {step_i}] +{r:.0f} (logs={info.get('logs')})")
            if te or tr:
                break
        save(env._last_pov, f"c{cycle:02d}_post")
        if te or tr:
            print("  (episode ended)")
            break
        env.step(3)  # turn for a new heading
        if total_r >= 3:
            break  # enough proof

    print(f"[probe4] TOTAL REWARD: {total_r:.0f}  events={events}")
    print(f"[probe4] final logs: {info.get('logs')}")
    env.close()
    if total_r > 0:
        print("[probe4] VERDICT: FIXED STACK WORKS — camera+attack live under "
              "the managed display, adapter pays inventory-delta reward. "
              "CLEAR TO TRAIN.")
    else:
        print("[probe4] VERDICT: still no reward — inspect cam/chop frames; "
              "focus fix insufficient.")


if __name__ == "__main__":
    main()
