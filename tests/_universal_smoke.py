"""SkyBot's universal-brain contracts: one brain, every game.

Born 2026-10-11 with docs/UNIVERSAL_GAMES.md. User decision, non-negotiable:
ONE brain plays every game, as concurrent streams. The loop sizes that brain
from stream 0 (obs_dim, action_dim, PROPRIO_DIM, sensor layout), so a stream
that presents a different shape would either crash the shared networks or —
worse — be silently reinterpreted. These contracts hold the pieces that make
mixing games safe. No game engine needed; runs anywhere.

Contracts:
    A. CONTROLLER PADS AND NEVER EXECUTES A PHANTOM INPUT. A 28-action game
       on a 48-slot brain exposes Discrete(48); every slot >= 28 (and every
       unmapped native action) executes slot 0, the game's no-op. Its mask
       marks exactly the mapped slots, and slot 0 is always open.
    B. CONTROLLER REFUSES A MISMATCHED STREAM. Wrong observation size, or
       more native actions than slots, raises at construction — never a
       silent reinterpretation of every replay row.
    C. find_attr WALKS THE WRAPPER CHAIN. gymnasium >= 1.0 does not forward
       attributes, so a bare hasattr on the outer wrapper is False even when
       the adapter has the method. (Found while wiring this: the loop's
       `hasattr(_e, "clear_break_marks")` is that exact pattern.)
    D. AUDIO CONTRACT. stereo_mel returns float32 (2, 32, 16) in [0,1]; a
       440 Hz tone lands in a lower mel band than a 4 kHz tone; a left-only
       sound leaves the right channel at 0 (direction survives); None/empty
       input returns None (neutral), never a fabricated silence.
    E. NON-SQUARE IMAGE SENSORS RUN. SensorImageEncoder sized its Linear from
       H alone, so the (2, 32, 16) audio sensor crashed on its first forward
       pass. Fixed by passing W; the square layout is byte-identical, so
       existing checkpoints load. The old construction is kept as a witness.
    F. EXECUTOR ANDs THE STREAM MASK into primitive columns only, never masks
       the agent mute, and leaves streams without a mask untouched.
    G. COMPOSER: stream order puts Minecraft at slot 0 whenever enabled
       (slot 0 keeps the bare `SkyBot` player name on the Paper server);
       every brain path points under brain_universal/ (FRESH brain — none of
       the legacy Minecraft brain's paths survive the merge); audio is in the
       sensor list; a single-stream run is REFUSED (num_envs 1 selects the
       unwired legacy single-env body, CLAUDE.md 4.2); a controller whose
       slot 0 is not the no-op, or wider than the brain, is refused.

Run: PYTHONPATH=. python tests/_universal_smoke.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym                                         # noqa: E402

from developmental_ai.environments.universal import (           # noqa: E402
    UniversalControllerWrapper, find_attr, stereo_mel)

NAME = "universal"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


class FakeGame(gym.Env):
    def __init__(self, n_actions=28, obs=3 * 8 * 8):
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Box(0, 1, (obs,), np.float32)
        self.executed = []

    def observe_reward(self, r):
        self.last_r = r

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, a):
        self.executed.append(int(a))
        return np.zeros(self.observation_space.shape, np.float32), 0.0, False, False, {}


def main():
    print(f"[{NAME}]")
    # ---- A
    g = FakeGame(28)
    w = UniversalControllerWrapper(g, action_slots=48, image_size=8, game="fake")
    check(w.action_space.n == 48, "A. 28-action game exposes Discrete(48)")
    for a in (0, 5, 27, 28, 40, 47):
        w.step(a)
    check(g.executed == [0, 5, 27, 0, 0, 0],
          "A. slots >= native count execute slot 0 (the no-op): %s" % g.executed)
    check(int(w.action_mask.sum()) == 28 and w.action_mask[:28].all()
          and not w.action_mask[28:].any(), "A. mask = exactly the 28 mapped slots")
    g2 = FakeGame(4)
    w2 = UniversalControllerWrapper(g2, 48, 8, mapped=[False, True, False, True])
    w2.step(2)
    check(g2.executed == [0] and w2.action_mask[0] and not w2.action_mask[2],
          "A. an unmapped NATIVE action also falls back to the no-op; slot 0 always open")
    # ---- B
    try:
        UniversalControllerWrapper(FakeGame(28, obs=3 * 4 * 4), 48, 8)
        check(False, "B. wrong observation size must refuse")
    except ValueError as e:
        check("identical shape" in str(e), "B. wrong observation size refused")
    try:
        UniversalControllerWrapper(FakeGame(60), 48, 8)
        check(False, "B. more actions than slots must refuse")
    except ValueError as e:
        check("action_slots" in str(e), "B. 60 actions on a 48-slot brain refused")
    # ---- C
    outer = gym.Wrapper(gym.Wrapper(w))
    check(not hasattr(outer, "observe_reward"),
          "C. witness: gymnasium wrappers do NOT forward attributes")
    check(find_attr(outer, "observe_reward") is not None
          and find_attr(outer, "action_mask") is w.action_mask,
          "C. find_attr reaches the adapter and the controller through the chain")
    check(find_attr(outer, "no_such_thing") is None, "C. absent -> None")
    # ---- D
    sr, n = 22050, 2520
    t = np.arange(n) / sr
    silent = np.zeros(n, np.int16)
    lo = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    hi = (np.sin(2 * np.pi * 4000 * t) * 12000).astype(np.int16)
    m_lo = stereo_mel(np.stack([lo, silent], 1), sr)
    m_hi = stereo_mel(np.stack([hi, silent], 1), sr)
    check(m_lo.shape == (2, 32, 16) and m_lo.dtype == np.float32
          and 0.0 <= m_lo.min() and m_lo.max() <= 1.0, "D. (2,32,16) float32 in [0,1]")
    check(m_lo[0].mean(1).argmax() < m_hi[0].mean(1).argmax(),
          "D. 440 Hz lands below 4 kHz in mel bands")
    check(m_lo[1].max() == 0.0 and m_lo[0].max() > 0.2,
          "D. left-only sound leaves the right channel silent (direction kept)")
    check(stereo_mel(None, sr) is None and stereo_mel(np.zeros((0, 2)), sr) is None,
          "D. no input -> None (neutral), not a fabricated silence")
    # ---- E
    import torch
    from developmental_ai.world_model.rssm import SensorImageEncoder
    enc = SensorImageEncoder(2, 32, 64, width=16)
    check(tuple(enc(torch.zeros(3, 2, 32, 16)).shape) == (3, 64),
          "E. audio-shaped (2,32,16) sensor encodes")
    sq = SensorImageEncoder(3, 32, 64)
    sq_w = SensorImageEncoder(3, 32, 64, width=32)
    check([(k, tuple(v.shape)) for k, v in sq.state_dict().items()]
          == [(k, tuple(v.shape)) for k, v in sq_w.state_dict().items()],
          "E. square layout unchanged (existing checkpoints still load)")
    try:
        SensorImageEncoder(2, 32, 64)(torch.zeros(1, 2, 32, 16))
        check(False, "E. witness: H-only construction should fail on 32x16")
    except RuntimeError:
        check(True, "E. witness: the old H-only construction still fails on 32x16")
    # ---- F
    from developmental_ai.policy.options import OptionExecutor

    class Bank:
        P, K = 6, 2
    ex = OptionExecutor.__new__(OptionExecutor)
    ex.bank = Bank()
    ex.stream_masks = [None, np.array([True, True, False, False, True, False])]
    base = np.ones(8, dtype=bool)
    out0 = ex._and_stream(base, 0)
    out1 = ex._and_stream(base, 1)
    check(out0 is base, "F. a stream without a mask is untouched")
    check(list(out1) == [True, True, False, False, True, False, True, True],
          "F. primitive columns ANDed, option columns kept: %s" % list(out1))
    ex.stream_masks = [np.zeros(6, dtype=bool)]
    check(ex._and_stream(np.zeros(8, dtype=bool), 0)[0],
          "F. never masks the agent mute (slot 0 reopened)")
    # ---- G
    import yaml
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools"))
    from compose_run_config import ComposeError, compose
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_p = os.path.join(root, "configs/minecraft_skybot.yaml")
    uni_p = os.path.join(root, "configs/universal.yaml")
    gdir = os.path.join(root, "configs/games")
    c = compose(["doom", "minecraft"], base_p, uni_p, gdir)
    order = [s["game"] for s in c["parallel_envs"]["streams"]]
    check(order[0] == "minecraft" and order[-1] == "doom"
          and c["parallel_envs"]["num_envs"] == len(order),
          "G. Minecraft holds slot 0 whenever enabled: %s" % order)
    paths = {"loop.log_dir": c["loop"]["log_dir"],
             "skill_bank.storage_dir": c["skill_bank"]["storage_dir"],
             "infra.log_dir": c["infra"]["log_dir"],
             "break_memory_path": c["environment"]["break_memory_path"],
             "shadow.root": c["foundation"]["shadow"]["root"],
             "snapshot_dir": c["universal"]["snapshot_dir"]}
    check(all("brain_universal" in str(v) for v in paths.values()),
          "G. every brain path is under brain_universal/ (fresh brain): %s" % paths)
    check("audio" in c["sensors"]["enabled"], "G. the brain can hear (audio sensor on)")
    d = compose(["doom"], base_p, uni_p, gdir)
    check(d["environment"]["name"].startswith("ViZDoom")
          and d["llm"]["vision"]["enabled"] is False,
          "G. Doom primary: env name set, Minecraft-specific VLM off")
    with tempfile.TemporaryDirectory() as td:
        doc = yaml.safe_load(open(os.path.join(gdir, "doom.yaml")))
        doc["game"]["streams"] = 1
        yaml.safe_dump(doc, open(os.path.join(td, "doom.yaml"), "w"))
        try:
            compose(["doom"], base_p, uni_p, td)
            check(False, "G. one stream must refuse")
        except ComposeError as e:
            check("single-env" in str(e), "G. a 1-stream run is refused (legacy body)")
        doc["game"]["streams"] = 2
        doc["game"]["controller"] = [{"ATTACK": 1}] + doc["game"]["controller"]
        yaml.safe_dump(doc, open(os.path.join(td, "doom.yaml"), "w"))
        try:
            compose(["doom"], base_p, uni_p, td)
            check(False, "G. slot 0 not the no-op must refuse")
        except ComposeError as e:
            check("slot 0" in str(e), "G. controller slot 0 that is not {} refused")
        doc["game"]["controller"] = [{}] + [{"ATTACK": 1}] * 60
        yaml.safe_dump(doc, open(os.path.join(td, "doom.yaml"), "w"))
        try:
            compose(["doom"], base_p, uni_p, td)
            check(False, "G. too many slots must refuse")
        except ComposeError as e:
            check("action_slots" in str(e), "G. controller wider than the brain refused")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
