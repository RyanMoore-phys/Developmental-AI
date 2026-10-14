"""Doom on the universal brain: the real engine, end to end.

Born 2026-10-11 — stage U1 of docs/UNIVERSAL_GAMES.md. Runs ViZDoom headless
on Freedoom (needs `vizdoom` and roms/doom/freedoom2.wad: `bash
scripts/fetch_roms.sh`). Without vizdoom it aborts at import and
tests/run_all.py reports SKIP:vizdoom, not a pass.

Contracts:
    A. REFUSALS: a missing game file names fetch_roms.sh; a controller whose
       slot 0 is not the no-op is refused.
    B. SAME BRAIN, SAME WIRE. obs is the canonical 3x128x128 in [0,1];
       proprio is the 13 Minecraft body fields in [0,1]; the sensor
       transport has EXACTLY the width and layout hash a Minecraft stream's
       bus produces for the same config — that equality is what lets both
       games write the same replay rows and feed one world model.
    C. ONE FRAME OF REFERENCE. Doom heading converts to Minecraft yaw by
       cases (east->-90, north->180, west->90, south->0); walking forward
       accumulates positive move_fwd with little lateral drift, so dead
       reckoning and the loop's spatial map read Doom correctly.
    D. REWARD IS 0.0, ALWAYS. A player is never handed a number; SkyBot
       earns from curiosity. Kill/item counts go to info["evaluator"] (RED).
    E. SOUND. When OpenAL is loadable, firing produces non-silent stereo on
       info["audio"] (2,32,16). Without OpenAL that is reported, not faked.
    F. PROGRESSION: after a level exit the next reset starts the NEXT map;
       after a death it restarts the SAME map.
    G. REWIND, REAL ENGINE: a pending verdict ends the step with truncated,
       rewind AND env_restarted (the boundary the lifelong body honours);
       reset() restores the saved position; the ring is trimmed to 1.
    H. THROUGH make_env: universal Discrete(48), controller mask = the 16
       mapped slots, slot 40 executes the no-op; close() stops it for good.
    J. REAL TIME (realtime: true — the shipped Doom config). The game runs
       on its own 35 Hz clock: time passes while SkyBot "thinks" (a 0.5 s
       pause = ~17 tics in the next decision, and the `tempo` sense says
       so); a fast brain never decides twice on one tic; the viewer sink
       receives ~35 fps whatever SkyBot does; a rewind is serviced ON the
       clock thread (the engine is not thread-safe) and restores the saved
       position; close() stops the clock.
    I. VIEWER TIC FRAMES ARE THE SAME GAME. With keep_tic_frames on, the
       adapter steps one tic at a time so the live viewer can play every
       frame at 35 fps. Two identically seeded adapters fed the same actions
       — one stepping 4 tics at once, one tic by tic — end at the same
       position and heading, and the tic-by-tic one yields exactly
       action_repeat frames per step. Off by default (no viewer = unchanged).

Run: PYTHONPATH=. python tests/_vizdoom_smoke.py
"""
import ctypes
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import vizdoom  # noqa: E402,F401  (absent -> run_all reports SKIP:vizdoom)
import yaml     # noqa: E402

from developmental_ai.environments.vizdoom_env import (          # noqa: E402
    ViZDoomEnvAdapter, doom_angle_to_mc_yaw)

NAME = "vizdoom"
N = [0]
ROM = os.path.join(ROOT, "roms/doom/freedoom2.wad")


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def openal_present():
    for lib in ("libopenal.so.1", "libopenal.1.dylib"):
        try:
            ctypes.CDLL(lib)
            return True
        except OSError:
            pass
    return False


def main():
    print(f"[{NAME}]")
    if not os.path.isfile(ROM):
        raise SystemExit(f"[{NAME}] FAIL: {ROM} missing — run `bash scripts/fetch_roms.sh`")
    game = yaml.safe_load(open(os.path.join(ROOT, "configs/games/doom.yaml")))["game"]
    game["rom"] = ROM
    # A-I pin the SYNCHRONOUS semantics (still supported: realtime: false);
    # J covers the shipped real-time mode.
    game["realtime"] = False
    uni = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))
    sensors = {"enabled": uni["sensors"]["enabled"], "fovea_size": 32}

    # ---- A
    try:
        ViZDoomEnvAdapter(dict(game, rom="/nope/x.wad"), 128)
        check(False, "A. missing game file must refuse")
    except FileNotFoundError as e:
        check("fetch_roms.sh" in str(e), "A. missing game file refused, names fetch_roms.sh")
    try:
        ViZDoomEnvAdapter(dict(game, controller=[{"ATTACK": 1}, {}]), 128)
        check(False, "A. slot 0 must be the no-op")
    except ValueError:
        check(True, "A. controller slot 0 that is not {} refused")

    with tempfile.TemporaryDirectory() as snapdir:
        g = dict(game, maps=["MAP01", "MAP02"],
                 checkpoint={"interval_s": 0.0, "keep": 3})
        env = ViZDoomEnvAdapter(g, 128, sensors_cfg=sensors, snapshot_dir=snapdir)
        obs, info = env.reset()
        # ---- B
        check(obs.shape == (3 * 128 * 128,) and obs.dtype == np.float32
              and 0.0 <= obs.min() and obs.max() <= 1.0, "B. canonical obs 3x128x128 in [0,1]")
        p = info["proprio"]
        check(p.shape == (13,) and 0.0 <= p.min() and p.max() <= 1.0,
              "B. 13 body fields in [0,1]: %s" % np.round(p, 2))
        from developmental_ai.sensors import build_default_bus
        from developmental_ai.environments.minerl_env import MineRLEnvAdapter as MC
        mc_bus = build_default_bus(MC.PROPRIO_DIM, lambda ctx: None,
                                   enabled=list(sensors["enabled"]), fovea_size=32)
        check(info["sensors"].shape[0] == mc_bus.width,
              "B. sensor transport width == a Minecraft stream's (%d)" % mc_bus.width)
        check(info["sensor_layout"] == mc_bus.layout_hash(),
              "B. sensor layout hash == a Minecraft stream's (same replay rows)")
        # ---- C
        for a, want in [(0, -90), (90, 180), (180, 90), (270, 0)]:
            got = doom_angle_to_mc_yaw(a)
            assert abs(((got - want + 180) % 360) - 180) < 1e-6, (a, got)
        check(True, "C. heading cases east/north/west/south -> MC yaw -90/180/90/0")
        env.step(0)                                     # first action is inert
        fwd = lat = 0.0
        for _ in range(8):
            _, r, te, tr, inf = env.step(1)
            fwd += inf["world"]["move_fwd"]
            lat += abs(inf["world"]["move_lat"])
        check(fwd > 1.0 and lat < 0.25 * fwd,
              "C. walking forward: move_fwd sum %.2f blocks, |lateral| %.2f" % (fwd, lat))
        # ---- D
        rs = []
        for _ in range(10):
            _, r, te, tr, inf = env.step(5)
            rs.append(r)
        check(all(x == 0.0 for x in rs), "D. reward is 0.0 on every step")
        check({"kills", "items", "secrets", "map"} <= set(inf["evaluator"]),
              "D. game counters only on info['evaluator'] (RED)")
        # ---- E
        check(inf.get("audio") is not None and inf["audio"].shape == (2, 32, 16),
              "E. info['audio'] present, (2,32,16)")
        if openal_present():
            check(env._audio_nonsilent > 0,
                  "E. OpenAL present: firing is audible (%d non-silent steps)" % env._audio_nonsilent)
        else:
            print("  -- E. OpenAL not loadable here: audio is silent by design of the "
                  "engine, and the adapter warns after 5000 steps (not asserted)")
        # ---- F
        env._outcome = "exit"
        _, info = env.reset()
        check(info["evaluator"]["map"] == "MAP02", "F. after a level exit -> next map")
        env._outcome = "died"
        _, info = env.reset()
        check(info["evaluator"]["map"] == "MAP02", "F. after a death -> same map")
        # ---- G
        env.step(0)
        pos0 = env._prev_xz
        env._ring.interval_s = 1e9     # ONE snapshot: the one taken at pos0
        env._ring.clear()
        env._ring.maybe_save()
        for _ in range(6):
            env.step(1)
        moved = env._prev_xz
        env._pending = "rewind"
        _, r, te, tr, inf = env.step(1)
        check(tr and not te and inf["rewind"] and inf["env_restarted"],
              "G. rewind step: truncated + rewind + env_restarted (live-body boundary)")
        _, info = env.reset()
        back = (info["world"]["x"], info["world"]["z"])
        d_back = np.hypot(back[0] - pos0[0], back[1] - pos0[1])
        check(info["rewind"] and d_back < 0.5 < np.hypot(moved[0] - pos0[0],
                                                         moved[1] - pos0[1])
              and len(env._ring) == 1,
              "G. reset restored the snapshot (%.2f blocks from it, had moved %.2f); "
              "ring trimmed" % (d_back, np.hypot(moved[0] - pos0[0], moved[1] - pos0[1])))
        env.close()

    # ---- H
    from developmental_ai.environments.wrappers import make_env
    from developmental_ai.environments.universal import find_attr
    with tempfile.TemporaryDirectory() as snapdir:
        e, _c = make_env(env_name="ViZDoom-Freedoom", normalize=True, curriculum=True,
                         pixel_obs=True, image_size=128, sensors_cfg=sensors,
                         universal=dict(uni["universal"], snapshot_dir=snapdir),
                         game_cfg=game, stream_index=1)
        check(e.action_space.n == 48 and e.action_dim == 48,
              "H. make_env -> SkyBot's controller, Discrete(48)")
        m = find_attr(e, "action_mask")
        check(m is not None and int(m.sum()) == 16, "H. stream mask = the 16 mapped Doom slots")
        e.reset()
        e.step(40)                                        # unmapped slot
        check(find_attr(e, "unmapped_steps") == 1, "H. slot 40 executed the no-op")
        e.close()
        try:
            e.step(1)
            check(False, "H. a closed adapter must not step")
        except RuntimeError:
            check(True, "H. close() stops it for good")
    # ---- I
    acts = [1, 1, 4, 4, 1, 6, 3, 1, 9, 1, 11, 1]

    def trace(tics):
        e = ViZDoomEnvAdapter(dict(game, checkpoint={"enabled": False},
                                   stuck={"enabled": False}), 128, seed=7)
        check_off = e.keep_tic_frames
        e.set_keep_tic_frames(tics)
        e.reset()
        nf = []
        for a in acts:
            _, _, _, _, inf = e.step(a)
            nf.append(len(e.pop_tic_frames()))
        w = inf["world"]
        e.close()
        return (round(w["x"], 4), round(w["z"], 4), round(w["yaw"], 3)), nf, check_off
    whole, nf_whole, default_off = trace(False)
    tics, nf_tics, _ = trace(True)
    check(default_off is False and nf_whole == [0] * len(acts),
          "I. off by default: no frames kept, the step is unchanged")
    check(whole == tics, "I. tic-by-tic = 4-at-once: same position/heading %s vs %s"
          % (whole, tics))
    check(nf_tics[1:] == [game["action_repeat"]] * (len(acts) - 1),
          "I. viewer gets every tic: %s frames per step" % nf_tics[1:])
    # ---- J
    import time
    with tempfile.TemporaryDirectory() as snapdir:
        rg = dict(game, realtime=True, checkpoint={"interval_s": 1e9, "keep": 3})
        env = ViZDoomEnvAdapter(rg, 128, sensors_cfg=sensors, snapshot_dir=snapdir)
        seen = []
        env.set_frame_sink(lambda f, dt: seen.append(time.perf_counter()))
        _, info = env.reset()
        no_tempo = build_default_bus(MC.PROPRIO_DIM, lambda ctx: None,
                                     enabled=[x for x in sensors["enabled"] if x != "tempo"],
                                     fovea_size=32)
        check("tempo" in uni["sensors"]["enabled"]
              and info["sensors"].shape[0] == mc_bus.width == no_tempo.width + 2,
              "J. the tempo sense is on the wire (%d = %d + 2)"
              % (info["sensors"].shape[0], no_tempo.width))
        t0 = time.time()
        fast = []
        while time.time() - t0 < 1.0:
            fast.append(env.step(1)[4]["decision_tics"])
        check(min(fast) >= 1 and max(fast) <= 2 and len(fast) >= 20,
              "J. fast brain: %d decisions/s, %d-%d tics each (never twice on a tic)"
              % (len(fast), min(fast), max(fast)))
        time.sleep(0.5)
        _, _, _, _, inf = env.step(0)
        check(14 <= inf["decision_tics"] <= 22
              and abs(inf["world"]["decision_seconds"] - inf["decision_tics"] / 35) < 1e-6,
              "J. the game did not wait: 0.5 s of thinking = %d tics (tempo %.2f s)"
              % (inf["decision_tics"], inf["world"]["decision_seconds"]))
        fps = (len(seen) - 1) / (seen[-1] - seen[0])
        check(32 <= fps <= 37 and env.late_tics <= 2,
              "J. viewer fed at %.1f fps by the clock itself (late tics %d)" % (fps, env.late_tics))
        pos0 = (inf["world"]["x"], inf["world"]["z"])
        env._rt_request = "fresh"; env._rt_done.clear()       # clean ring via clock
        env._rt_done.wait(5)
        env._ring.interval_s = 0.0
        time.sleep(0.15)                                       # clock saves one
        env._ring.interval_s = 1e9
        _, _, _, _, a = env.step(0)
        here = (a["world"]["x"], a["world"]["z"])
        for _ in range(20):
            env.step(1)
            time.sleep(0.03)
        env._pending = "rewind"
        _, _, te, tr, inf = env.step(1)
        check(tr and inf["env_restarted"], "J. rewind ends the step as a boundary")
        _, info = env.reset()
        d = np.hypot(info["world"]["x"] - here[0], info["world"]["z"] - here[1])
        check(info["rewind"] and d < 1.0,
              "J. rewind serviced on the clock thread: back within %.2f blocks" % d)
        env.close()
        check(not env._clock.is_alive(), "J. close() stops the clock")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
