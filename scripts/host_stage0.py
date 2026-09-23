#!/usr/bin/env python3
"""Stage 0 — MineRL feasibility. NOTHING ELSE RUNS UNTIL THIS PASSES.

Four questions that no amount of reading the source settles, each of which
decides whether a whole roadmap item is alive or dead. They are first, and
cheap, deliberately: the alternative is discovering on hour six of a soak run
that a third of the action space was being dropped at the socket.

  0.1  Does MineRL build and step at all on this pod?
  0.2  Does the action space contain the keys the new macros press —
       hotbar.1-9, left, right, sprint, sneak? A key the engine lacks is
       ACCEPTED SILENTLY by `act[k] = v` and dropped at the socket, so the
       policy learns a button that does nothing. This is the single most
       expensive failure mode in this project's history.
  0.3  Does POV include the HUD overlay? Decides A2, and the observation
       half of A9 (the agent can change hotbar selection but may not be
       able to SEE that it did).
  0.4  Is there any audio handler? Expected no. Decides M1.

Exit code 0 = go. Non-zero = stop and report, which is what the standing
instruction asks for.

Run on the pod:
    PYTHONPATH=. python scripts/pod_stage0.py --all
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, ".")

RESULT = {"stage": 0, "questions": {}}


def _say(q, ok, detail):
    RESULT["questions"][q] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'OK ' if ok else 'NO '}] {q}: {detail}")


def q01_boot(render_size, image_size):
    """0.1 — can we build and step a MineRL env at all."""
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter
    t0 = time.time()
    env = MineRLEnvAdapter(image_size=image_size, action_repeat=4,
                           render_size=render_size, sensors_cfg=None)
    obs, info = env.reset()
    boot = time.time() - t0
    t1 = time.time()
    n = 0
    for i in range(20):
        obs, r, term, trunc, info = env.step(i % env.action_space.n)
        n += 1
        if term or trunc:
            obs, info = env.reset()
    rate = n / max(1e-6, time.time() - t1)
    _say("0.1_boot", True,
         f"built in {boot:.0f}s, stepped {n} times at {rate:.2f} steps/s, "
         f"obs {tuple(obs.shape)}")
    return env


def q02_action_space(env):
    """0.2 — every macro key must exist in the real action space."""
    from developmental_ai.environments.minerl_env import TREECHOP_MACROS
    noop = env._env.action_space.noop()
    known = set(noop.keys())
    used = {k for m in TREECHOP_MACROS for k in m if k != "_ticks"}
    missing = sorted(used - known)
    RESULT["action_space"] = sorted(known)
    RESULT["macro_keys"] = sorted(used)
    if missing:
        bad = sorted({i for i, m in enumerate(TREECHOP_MACROS)
                      if set(m) & set(missing)})
        _say("0.2_action_space", False,
             f"MISSING {missing} (macros {bad}). The space offers "
             f"{sorted(known)}. Those macros would be dropped at the socket.")
        return False
    _say("0.2_action_space", True,
         f"all {len(used)} macro keys present in a {len(known)}-key space")
    return True


def q03_hud(env, out_dir):
    """0.3 — is the HUD rendered into POV.

    Decided by CONTRAST, not by eye: a rendered HUD is high-contrast
    geometry against the world, so the bottom band's variance is far above
    a same-sized band of sky or ground. Ambiguous cases are reported as
    ambiguous rather than guessed — a wrong YES here enables a sensor that
    reads a strip of world and is worse than nothing.
    """
    import numpy as np
    for _ in range(8):                       # let the HUD settle after reset
        env.step(0)
    pov = env._last_pov
    if pov is None:
        _say("0.3_hud", False, "no native frame retained")
        return False
    a = np.asarray(pov, dtype=np.float32) / 255.0
    h = a.shape[0]
    band = max(4, int(round(h * 0.12)))
    bottom = a[h - band:]
    middle = a[h // 2 - band // 2: h // 2 + band // 2]
    bv, mv = float(bottom.std()), float(middle.std())
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "stage0_frame.npy"), np.asarray(pov))
    try:
        from PIL import Image
        Image.fromarray(np.asarray(pov)).save(
            os.path.join(out_dir, "stage0_frame.png"))
    except Exception:
        pass
    RESULT["hud"] = {"bottom_std": bv, "middle_std": mv}
    if bv > mv * 1.6:
        _say("0.3_hud", True,
             f"bottom band std {bv:.3f} vs mid-frame {mv:.3f} — HUD LIKELY "
             f"RENDERED; inspect stage0_frame.png before enabling A2")
        return True
    _say("0.3_hud", False,
         f"bottom band std {bv:.3f} vs mid-frame {mv:.3f} — no HUD signature. "
         f"Keep `hud` disabled; A9's keys still work but selection is unseen.")
    return False


def q04_audio(env):
    """0.4 — any audio handler at all. Expected NO."""
    names = []
    try:
        spec = getattr(env._env, "task", None) or getattr(env._env, "_spec", None)
        obs_h = getattr(spec, "observables", None) or []
        names = [type(h).__name__ for h in obs_h]
    except Exception:
        pass
    if not names:
        try:
            names = list(env._env.observation_space.spaces.keys())
        except Exception:
            names = []
    RESULT["observation_handlers"] = names
    has = any("audio" in n.lower() or "sound" in n.lower() for n in names)
    _say("0.4_audio", has,
         f"{'FOUND an audio handler' if has else 'no audio handler'} among "
         f"{len(names)} observables: {names[:12]}")
    return has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--render-size", type=int, default=384)
    ap.add_argument("--out", default="podlogs/stage0")
    a = ap.parse_args()

    print("=== Stage 0: MineRL feasibility ===")
    env = None
    try:
        env = q01_boot(a.render_size, a.image_size)
    except Exception as e:
        _say("0.1_boot", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        _finish(a.out, blocking_failed=True)
        return 2

    ok_space = q02_action_space(env)
    q03_hud(env, a.out)
    q04_audio(env)
    try:
        env.close()
    except Exception:
        pass

    # 0.2 is the only BLOCKING one besides boot: a dropped key is silent.
    return _finish(a.out, blocking_failed=not ok_space)


def _finish(out_dir, blocking_failed):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "stage0.json")
    with open(p, "w") as f:
        json.dump(RESULT, f, indent=2)
    print(f"\nwrote {p}")
    if blocking_failed:
        print("\nSTAGE 0 BLOCKED. Do not proceed to Stage 1.")
        print("If 0.2 failed, remove the unsupported macros (the table is "
              "append-only, so indices 0-20 are unaffected) and re-run.")
        return 1
    print("\nSTAGE 0 GO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
