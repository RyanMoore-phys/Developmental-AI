"""The Player Adapter: SkyBot plays any game the way a person does.

docs/UNIVERSAL_GAMES.md §11; plan ~/.claude/plans/typed-petting-nautilus.md.
One unmodified game per seat (seat.py); SkyBot gets its screen (screen.py),
its sound (audio.py), and a keyboard + mouse (inputs.py) — nothing else.

REAL TIME, the same contract as the real-time Doom adapter (§10): the game
runs on its own clock and never waits. step(a) makes SkyBot HOLD controller
slot `a`, then returns the NEWEST frame, the sound since the last decision,
and how much time passed (`tempo`). The live viewer is fed by the capture
thread itself (`set_frame_sink`), so it shows the game at its own speed. The
standby actor and the loop find it through `realtime` like any real-time env.

What a universal adapter does NOT have: engine variables. There is no health,
position or "episode over" signal — SkyBot reads its body off the HUD like a
person, and play is continuous (a death screen is part of the game, and its
respawn key must be in the controller). Reward is always 0.0: curiosity pays.

Health and getting unstuck (each guard has a stated exit, CLAUDE.md §4.1):
  * the game process died, or the screen has not changed for `frozen_s`
    while SkyBot holds inputs  ->  restart the game (a boundary:
    env_restarted, like a death), at most `max_restarts_per_hour` — the
    budget slides, so a refused restart is retried once the hour moves on;
  * the stuck watch (rewind.py) fires  ->  quickload if the game declares
    save keys, else restart the game.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np

from developmental_ai.environments.minerl_env import MineRLEnvAdapter as _MC
from developmental_ai.environments.player.controller import validate
from developmental_ai.environments.player.inputs import (HeldInputs, InputPump)
from developmental_ai.environments.player.screen import FramePump
from developmental_ai.environments.rewind import FRESH, NONE, REWIND, StuckWatch
from developmental_ai.environments.universal import stereo_mel
from developmental_ai.sensors import build_default_bus, crop_centre
from developmental_ai.sensors.builtins import (AUDIO_CHANNELS, AUDIO_FRAMES,
                                               AUDIO_MELS)

logger = logging.getLogger(__name__)

RESTART = "restart"


class RealBackends:
    """Xvfb seat + mss screen + XTest input + parec audio (Linux host)."""

    def __init__(self, game: Dict[str, Any], stream_index: int):
        self.game, self.i = game, int(stream_index)

    def make_seat(self):
        from developmental_ai.environments.player.seat import Seat
        g = self.game
        w = g.get("window") or {}
        disp = int(g.get("display_base", 100)) + self.i
        launch = [str(x).format(stream=self.i, display=disp, name=self._name())
                  for x in (g.get("launch") or [])]
        return Seat(launch, display=disp, width=int(w.get("width", 640)),
                    height=int(w.get("height", 480)),
                    audio_sink=(f"skybot_seat_{disp}"
                                if (g.get("audio") or {}).get("enabled", True) else None),
                    gl=bool(g.get("gl", False)),
                    cpu=(None if g.get("cpu_base") is None
                         else int(g["cpu_base"]) + self.i),
                    cwd=g.get("cwd"), boot_s=float(g.get("boot_s", 4.0)))

    def _name(self) -> str:
        # same rule as the loop's _agent_name_for: slot 0 keeps the bare name
        p = str(self.game.get("player_name", "SkyBot"))
        return p if self.i <= 0 else f"{p}{self.i}"

    def make_screen(self, seat):
        from developmental_ai.environments.player.screen import MssScreen
        return MssScreen(seat.display_name, seat.width, seat.height)

    def make_input(self, seat):
        from developmental_ai.environments.player.inputs import XTestBackend
        return XTestBackend(seat.display_name)

    def make_audio(self, seat):
        if not seat.audio_sink:
            return None
        from developmental_ai.environments.player.audio import ParecAudio
        return ParecAudio(seat.audio_sink,
                          int((self.game.get("audio") or {}).get("sample_rate", 22050)))


class PlayerAdapter(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # ONE BODY layout for the one brain (all neutral here: no engine stats)
    PROPRIO_KEYS = _MC.PROPRIO_KEYS
    PROPRIO_DIM = _MC.PROPRIO_DIM
    ATTACK_RUN_SCALE = _MC.ATTACK_RUN_SCALE
    MOVE_SCALE = _MC.MOVE_SCALE
    _proprio = _MC._proprio

    def __init__(self, game: Dict[str, Any], image_size: int = 128,
                 sensors_cfg: Optional[Dict] = None, stream_index: int = 0,
                 seed: int = 0, action_slots: int = 48, backends=None):
        super().__init__()
        self.game_cfg = dict(game or {})
        self.stream_index = int(stream_index)
        self.image_size = int(image_size)
        self.slots = validate(self.game_cfg, action_slots)
        self.realtime = True
        self.fps = float(self.game_cfg.get("fps", 35))
        self.tic_seconds = 1.0 / self.fps
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (3 * self.image_size * self.image_size,), np.float32)
        self.action_space = gym.spaces.Discrete(len(self.slots))
        self.backends = backends or RealBackends(self.game_cfg, self.stream_index)
        self._move_scale = self.MOVE_SCALE
        self._fovea_size = int((sensors_cfg or {}).get("fovea_size", 32))
        self._sensor_bus = None
        if sensors_cfg and sensors_cfg.get("enabled") is not None:
            self._sensor_bus = build_default_bus(
                self.PROPRIO_DIM, lambda ctx: self._proprio(ctx["world"]),
                enabled=list(sensors_cfg.get("enabled") or []),
                fovea_size=self._fovea_size, move_scale=self._move_scale)
        a = self.game_cfg.get("audio") or {}
        self.audio_enabled = bool(a.get("enabled", True))
        h = self.game_cfg.get("health") or {}
        self.frozen_s = float(h.get("frozen_s", 60.0))
        self.max_restarts_per_hour = int(h.get("max_restarts_per_hour", 6))
        self.timeout_s = float(h.get("frame_timeout_s", 10.0))
        st = self.game_cfg.get("stuck") or {}
        self._watch = None if not bool(st.get("enabled", True)) else StuckWatch(
            window_s=float(st.get("window_s", 180)), ratio=float(st.get("ratio", 0.25)),
            warmup_s=float(st.get("warmup_s", 600)),
            baseline_halflife_s=float(st.get("baseline_halflife_s", 1800)),
            max_rewinds_without_recovery=int(st.get("max_rewinds_without_recovery", 3)))
        sv = self.game_cfg.get("save") or {}
        self._quicksave = list(sv.get("quicksave") or [])
        self._quickload = list(sv.get("quickload") or [])
        self._save_every = float(sv.get("interval_s", 300))
        self._last_save = None
        # runtime
        self._seat = self._pump = self._held = self._inpump = self._audio = None
        self._started = False
        self._closed = False
        self._pending = NONE
        self._seen = 0
        self._last_t: Optional[float] = None
        self._sink: Optional[Callable[[np.ndarray, float], None]] = None
        self._restart_times: deque = deque()
        self._last_frame: Optional[np.ndarray] = None
        self._last_obs = np.zeros(self.observation_space.shape, np.float32)
        self._cur_fovea = self._prev_fovea = self._last_pov_small = None
        # telemetry
        self.steps = 0
        self.restarts = 0
        self.rewinds = 0
        self.refused_restarts = 0
        self.decision_tics_last = 0

    # ------------------------------------------------------------ lifecycle
    def _start(self) -> None:
        b = self.backends
        self._seat = b.make_seat()
        self._seat.start()
        self._pump = FramePump(b.make_screen(self._seat), fps=self.fps)
        self._pump.sink = self._sink
        self._held = HeldInputs(b.make_input(self._seat), self.slots)
        self._inpump = InputPump(self._held)
        self._audio = b.make_audio(self._seat) if self.audio_enabled else None
        self._pump.start()
        self._inpump.start()
        self._started = True
        atexit.register(self.close)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for f in (lambda: self._held.release_all(), lambda: self._inpump.stop(),
                  lambda: self._pump.stop(), lambda: self._audio.close(),
                  lambda: self._held.backend.close(), lambda: self._seat.close()):
            try:
                f()
            except Exception:
                pass

    def set_frame_sink(self, sink) -> None:
        self._sink = sink
        if self._pump is not None:
            self._pump.sink = sink

    def render(self):
        return self._last_frame

    # ------------------------------------------------------------ senses
    def _compose(self, frame: np.ndarray, audio, seconds: float, frames: int):
        from PIL import Image
        small = np.asarray(Image.fromarray(frame).resize(
            (self.image_size, self.image_size), Image.BOX), dtype=np.uint8)
        if self._sensor_bus is not None:
            self._prev_fovea = self._cur_fovea
            self._cur_fovea = crop_centre(frame, self._fovea_size)
        self._last_frame, self._last_pov_small = frame, small
        obs = (small.astype(np.float32) / 255.0).transpose(2, 0, 1).reshape(-1)
        world: Dict[str, Any] = {"decision_seconds": float(seconds)}
        info: Dict[str, Any] = {"game": self.game_cfg.get("name", "player"),
                                "world": world, "decision_tics": int(frames),
                                "gui_open": False, "env_restarted": False,
                                "rewind": False}
        if self.audio_enabled and audio is not None:
            mel = stereo_mel(audio, getattr(self._audio, "sample_rate", 22050),
                             n_mels=AUDIO_MELS, n_frames=AUDIO_FRAMES)
            if mel is not None and mel.shape == (AUDIO_CHANNELS, AUDIO_MELS, AUDIO_FRAMES):
                info["audio"] = mel
        info["proprio"] = self._proprio(world)
        if self._sensor_bus is not None:
            ctx = {"world": world, "info": info, "pov_native": frame,
                   "pov_small": small, "prev_fovea": self._prev_fovea}
            try:
                info["sensors"] = self._sensor_bus.read_policy(ctx)
            except Exception:
                info["sensors"] = self._sensor_bus.neutral_policy()
            info["sensor_layout"] = self._sensor_bus.layout_hash()
        info["evaluator"] = {"restarts": self.restarts, "rewinds": self.rewinds,
                             "refused_restarts": self.refused_restarts,
                             "late_frames": self._pump.late if self._pump else 0}
        self._last_obs = obs
        return obs, info

    def _collect(self):
        frame, count = self._pump.wait_newer(self._seen, self.timeout_s)
        frames = count - self._seen
        self._seen = count
        now = time.monotonic()
        seconds = frames / self.fps if self._last_t is None else now - self._last_t
        self._last_t = now
        audio = self._audio.read() if self._audio is not None else None
        self.decision_tics_last = frames
        return frame, audio, seconds, frames

    # ------------------------------------------------------------ health
    def _unhealthy(self) -> Optional[str]:
        if not self._seat.game_alive():
            return "game process exited"
        if self._held.slot != 0 and self._pump.frozen_for() > self.frozen_s:
            return "screen frozen %.0fs while inputs held" % self._pump.frozen_for()
        return None

    def _restart_allowed(self) -> bool:
        now = time.monotonic()
        while self._restart_times and now - self._restart_times[0] > 3600.0:
            self._restart_times.popleft()            # the budget slides: no latch
        return len(self._restart_times) < self.max_restarts_per_hour

    # ------------------------------------------------------------ gym API
    def observe_reward(self, reward: float) -> None:
        if self._watch is None or self._pending != NONE:
            return
        v = self._watch.observe(reward)
        if v != NONE:
            self._pending = v
            logger.info("player %s stream %d: stuck (%s) -> %s",
                        self.game_cfg.get("name"), self.stream_index,
                        self._watch.stats(), v)

    def _boundary(self, why: str):
        """End this life here; reset() performs the restart / rewind."""
        if self._held is not None:
            self._held.release_all()
        info = {"game": self.game_cfg.get("name", "player"), "env_restarted": True,
                "rewind": self._pending == REWIND, "gui_open": False, "why": why}
        return self._last_obs, 0.0, False, True, info

    def step(self, action):
        if self._closed:
            raise RuntimeError("PlayerAdapter closed")
        if not self._started:
            self.reset()
        if self._pending != NONE:
            return self._boundary(str(self._pending))
        bad = self._unhealthy()
        if bad is not None:
            if self._restart_allowed():
                logger.warning("player %s stream %d: %s -> restarting the game",
                               self.game_cfg.get("name"), self.stream_index, bad)
                self._pending = RESTART
                return self._boundary(bad)
            self.refused_restarts += 1
        self._held.set_slot(int(action))
        if self._quicksave and (self._last_save is None
                                or time.monotonic() - self._last_save >= self._save_every):
            self._held.tap(self._quicksave)
            self._last_save = time.monotonic()
        frame, audio, seconds, frames = self._collect()
        self.steps += 1
        obs, info = self._compose(frame, audio, seconds, frames)
        return obs, 0.0, False, False, info

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("PlayerAdapter closed")
        if not self._started:
            self._start()
        pending, self._pending = self._pending, NONE
        if pending != NONE:
            self._held.release_all()
            if pending == REWIND and self._quickload:
                self._held.tap(self._quickload)
                self.rewinds += 1
            elif self._restart_allowed() or pending == RESTART:
                self._restart_times.append(time.monotonic())
                self._seat.restart_game()
                self.restarts += 1
            # nothing from before the boundary is handed back
            self._seen = self._pump.count
            self._last_t = None
            if self._audio is not None:
                self._audio.read()
        frame, audio, seconds, frames = self._collect()
        obs, info = self._compose(frame, audio, seconds, frames)
        info["rewind"] = pending == REWIND
        return obs, info
