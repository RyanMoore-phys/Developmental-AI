"""The Player Adapter: any game, through a screen, speakers, keyboard, mouse.

Born 2026-10-13 — stage P1 of docs/UNIVERSAL_GAMES.md §11 (plan:
typed-petting-nautilus). User decision: one universal bridge for every game,
replacing per-game bridges (MineRL) once host-proven. This suite runs the
whole adapter against FAKE seat/screen/input/audio backends, so every
contract below holds on any machine; the real backends (Xvfb, XTest, mss,
parec) are measured on the host by tools/player_spike.py (stage P0).

Contracts:
    A. CONTROLLER SAFETY: slot 0 must release everything; OS/window-manager
       keys (Super, XF86*, Print...) and chords (Alt+F-key, Ctrl+Alt) are
       refused; unknown fields refused; a key that opens a state with no
       reachable closer is a one-way door and is refused (CLAUDE.md §4.5);
       the shipped doom_player.yaml passes.
    B. HELD INPUTS ARE A PERSON'S HANDS: changing slot releases exactly what
       is no longer held and presses exactly what newly is; nothing outside
       the game's whitelist ever reaches the backend; release_all lets go of
       everything.
    C. MOUSE IS A HELD VELOCITY: 400 px/s held for 1 s moves ~400 px whether
       the pump ticks at 60 Hz or 7 Hz (time-consistent, as `tempo` assumes).
    D. REAL TIME, SAME CONTRACT AS DOOM: step() returns the canonical obs,
       the time that really passed (0.3 s of thinking reads as ~0.3 s and
       ~0.3 x fps frames), the sound of that span, reward 0.0; the viewer
       sink is fed by the capture thread at the game's rate.
    E. HEALTH: a dead game or a screen frozen while inputs are held ends the
       life as a boundary (env_restarted) and reset() restarts the game; a
       frozen screen while NOTHING is held is not a fault.
    F. NO LATCH: the restart budget refuses an over-budget restart (counted)
       and re-opens by itself once the hour slides (CLAUDE.md §4.1).
    G. GETTING UNSTUCK: a stuck verdict quickloads when the game declares
       save keys, otherwise restarts the game.
    H. TEARDOWN: close() releases every held key, stops the threads, closes
       the seat; a closed adapter never steps.
    I. ONE BRAIN: wrapped in SkyBot's controller it is Discrete(48) with a
       16-slot mask for doom_player, and its sensor transport matches the
       other games' exactly; make_env refuses it outside the universal brain.

Run: PYTHONPATH=. python tests/_player_adapter_smoke.py
"""
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from developmental_ai.environments.player.adapter import PlayerAdapter     # noqa: E402
from developmental_ai.environments.player.audio import FakeAudio            # noqa: E402
from developmental_ai.environments.player.controller import (               # noqa: E402
    ControllerError, door_problems, parse_controller, validate)
from developmental_ai.environments.player.inputs import (                   # noqa: E402
    FakeInputBackend, HeldInputs, InputPump)
from developmental_ai.environments.player.screen import FakeScreen          # noqa: E402

NAME = "player-adapter"
N = [0]
GAME = yaml.safe_load(open(os.path.join(ROOT, "configs/games/doom_player.yaml")))["game"]
UNI = yaml.safe_load(open(os.path.join(ROOT, "configs/universal.yaml")))


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def refuses(entries, needle):
    try:
        parse_controller(entries)
        return False
    except ControllerError as e:
        return needle in str(e)


class FakeSeat:
    display_name, width, height, audio_sink = ":999", 64, 48, "fake"

    def __init__(self):
        self.alive, self.restarts, self.closed, self.started = True, 0, False, False

    def start(self):
        self.started = True

    def game_alive(self):
        return self.alive

    def restart_game(self):
        self.restarts += 1
        self.alive = True

    def close(self):
        self.closed = True


class FakeBackends:
    def __init__(self, frozen=False):
        self.seat = FakeSeat()
        self.inp = FakeInputBackend()
        self.frozen = frozen

    def make_seat(self):
        return self.seat

    def make_screen(self, seat):
        def make(n):
            v = 0 if self.frozen else (n * 7) % 255
            return np.full((48, 64, 3), v, np.uint8)
        return FakeScreen(make)

    def make_input(self, seat):
        return self.inp

    def make_audio(self, seat):
        t = np.arange(630) / 22050
        tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        return FakeAudio(lambda: np.stack([tone, tone], 1))


def adapter(backends, game=None, **kw):
    g = dict(game or GAME)
    g.update(kw)
    return PlayerAdapter(g, image_size=128,
                         sensors_cfg={"enabled": UNI["sensors"]["enabled"], "fovea_size": 32},
                         backends=backends)


def main():
    print(f"[{NAME}]")
    # ---- A
    check(refuses([{"keys": ["Up"]}], "slot 0"), "A. slot 0 must release everything")
    check(refuses([{}, {"keys": ["Super_L"]}], "OS/window-manager"), "A. Super key refused")
    check(refuses([{}, {"keys": ["XF86AudioMute"]}], "OS/window-manager"), "A. XF86 keys refused")
    check(refuses([{}, {"keys": ["Alt_L", "F4"]}], "Alt+F4"), "A. Alt+F4 refused")
    check(refuses([{}, {"keys": ["Control_L", "Alt_L", "Delete"]}], "Ctrl+Alt"), "A. Ctrl+Alt refused")
    check(refuses([{}, {"jump": 1}], "unknown field"), "A. unknown slot field refused")
    slots = parse_controller([{}, {"keys": ["Escape"]}])
    check(door_problems(slots, [{"opens": ["Escape"], "closes": ["Return"]}]),
          "A. Escape with no reachable closer = one-way door")
    check(not door_problems(slots, [{"opens": ["Escape"], "closes": ["Escape"]}]),
          "A. a key that closes what it opens is fine")
    check(len(validate(GAME, 48)) == 16, "A. the shipped doom_player.yaml validates (16 slots)")
    # ---- B
    fb = FakeInputBackend()
    held = HeldInputs(fb, parse_controller(GAME["controller"]))
    held.set_slot(1)                           # Up
    held.set_slot(6)                           # Up + Control_L
    held.set_slot(5)                           # Control_L
    held.set_slot(0)
    check(fb.events == [("key", "Up", True), ("key", "Control_L", True),
                        ("key", "Up", False), ("key", "Control_L", False)],
          "B. slot changes press/release exactly the difference: %s" % fb.events)
    held._key("Super_L", True)
    check(held.dropped == 1 and ("key", "Super_L", True) not in fb.events,
          "B. a key outside the whitelist never reaches the backend")
    held.set_slot(6)
    held.release_all()
    check(fb.events[-2:] == [("key", "Control_L", False), ("key", "Up", False)],
          "B. release_all lets go of everything held")
    # ---- C
    moved = []
    for hz in (60.0, 7.0):
        fb2 = FakeInputBackend()
        h2 = HeldInputs(fb2, parse_controller(GAME["controller"]))
        h2.set_slot(14)                        # mouse +400 px/s
        steps = int(hz)
        for _ in range(steps):
            h2.tick(1.0 / hz)
        moved.append(fb2.moved()[0])
    check(all(abs(m - 400) <= 1 for m in moved),
          "C. 400 px/s for 1 s = %s px at 60 Hz / 7 Hz (time-consistent)" % moved)
    fb3 = FakeInputBackend()
    h3 = HeldInputs(fb3, parse_controller(GAME["controller"]))
    h3.set_slot(14)
    pump = InputPump(h3, hz=60)
    pump.start(); time.sleep(0.5); pump.stop()
    check(150 <= fb3.moved()[0] <= 260, "C. the pump applies it in real time (%d px in 0.5 s)"
          % fb3.moved()[0])
    # ---- D
    be = FakeBackends()
    env = adapter(be)
    seen = []
    env.set_frame_sink(lambda f, dt: seen.append(time.perf_counter()))
    obs, info = env.reset()
    check(be.seat.started and obs.shape == (3 * 128 * 128,),
          "D. reset starts the seat and returns the canonical obs")
    time.sleep(0.3)
    obs, r, te, tr, info = env.step(1)
    s = info["world"]["decision_seconds"]
    check(0.25 <= s <= 0.45 and 7 <= info["decision_tics"] <= 16,
          "D. the game did not wait: %.2f s / %d frames since the last decision"
          % (s, info["decision_tics"]))
    check(r == 0.0 and not te and not tr and info.get("audio") is not None
          and info["audio"].shape == (2, 32, 16), "D. reward 0.0, sound of that span on info['audio']")
    check(("key", "Up", True) in be.inp.events, "D. the decision is held on the keyboard")
    time.sleep(0.6)
    fps = (len(seen) - 1) / (seen[-1] - seen[0])
    check(28 <= fps <= 40, "D. viewer fed by the capture thread at %.1f fps" % fps)
    # ---- E
    be.seat.alive = False
    _, _, te, tr, info = env.step(1)
    check(tr and info["env_restarted"] and "exited" in info["why"],
          "E. a dead game ends the life as a boundary (%s)" % info["why"])
    env.reset()
    check(be.seat.restarts == 1 and env.restarts == 1, "E. reset restarts the game")
    env.close()
    bf = FakeBackends(frozen=True)
    envf = adapter(bf, health={"frozen_s": 0.2, "max_restarts_per_hour": 6})
    envf.reset()
    envf.step(0); time.sleep(0.35)
    _, _, _, tr0, _ = envf.step(0)
    check(not tr0, "E. a static screen while NOTHING is held is not a fault")
    envf.step(1); time.sleep(0.35)
    _, _, _, tr1, info = envf.step(1)
    check(tr1 and "frozen" in info["why"], "E. frozen while inputs held -> boundary (%s)" % info["why"])
    envf.close()
    # ---- F
    bf2 = FakeBackends()
    env2 = adapter(bf2, health={"frozen_s": 999, "max_restarts_per_hour": 1})
    env2.reset()
    bf2.seat.alive = False
    env2.step(1); env2.reset()                       # restart 1 (budget used)
    bf2.seat.alive = False
    _, _, _, tr, _ = env2.step(1)
    check(not tr and env2.refused_restarts == 1, "F. over budget: restart refused and counted")
    env2._restart_times[0] -= 3601                   # the hour moves on
    _, _, _, tr, _ = env2.step(1)
    check(tr, "F. the budget slides: the restart is allowed again (no latch)")
    env2.close()
    # ---- G
    bs = FakeBackends()
    envs = adapter(bs, save={"quicksave": ["F6"], "quickload": ["F9"], "interval_s": 1e9})
    envs.reset()
    envs.step(1)                                     # first step quicksaves
    envs._pending = "rewind"
    _, _, _, tr, info = envs.step(1)
    envs.reset()
    check(tr and ("key", "F9", True) in bs.inp.events and bs.seat.restarts == 0
          and envs.rewinds == 1, "G. stuck + save keys -> quickload, not a restart")
    check(("key", "F6", True) in bs.inp.events, "G. quicksave taken on its interval")
    envs.close()
    bn = FakeBackends()
    envn = adapter(bn)
    envn.reset()
    envn._pending = "rewind"
    envn.step(1); envn.reset()
    check(bn.seat.restarts == 1, "G. stuck without save keys -> the game is restarted")
    envn.close()
    # ---- H
    bh = FakeBackends()
    envh = adapter(bh)
    envh.reset(); envh.step(6)                       # hold Up + Control_L
    envh.close()
    rel = [e for e in bh.inp.events if e[0] == "key" and not e[2]]
    check({e[1] for e in rel} >= {"Up", "Control_L"} and bh.seat.closed,
          "H. close() releases every held key and closes the seat")
    check(not envh._pump._t.is_alive() and not envh._inpump._t.is_alive(),
          "H. capture and input threads stopped")
    try:
        envh.step(1)
        check(False, "H. a closed adapter must not step")
    except RuntimeError:
        check(True, "H. a closed adapter never steps")
    # ---- I
    from developmental_ai.environments.universal import UniversalControllerWrapper
    from developmental_ai.sensors import build_default_bus
    bi = FakeBackends()
    w = UniversalControllerWrapper(adapter(bi), action_slots=48, image_size=128, game="doom_player")
    check(w.action_space.n == 48 and int(w.action_mask.sum()) == 16,
          "I. SkyBot's controller: Discrete(48), 16-slot mask")
    _, info = w.reset()
    ref = build_default_bus(13, lambda c: None, enabled=UNI["sensors"]["enabled"], fovea_size=32)
    check(info["sensors"].shape[0] == ref.width and info["sensor_layout"] == ref.layout_hash(),
          "I. sensor transport identical to every other game's (%d)" % ref.width)
    w.env.close()
    from developmental_ai.environments.wrappers import make_env
    try:
        make_env(env_name="Player-doom", pixel_obs=True, image_size=128, game_cfg=GAME)
        check(False, "I. must refuse outside the universal brain")
    except ValueError as e:
        check("universal brain" in str(e), "I. make_env refuses a player game outside the universal brain")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
