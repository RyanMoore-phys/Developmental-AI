"""Doom (ViZDoom + Freedoom) for SkyBot's one universal brain.

Born 2026-10-11 — stage U1 of docs/UNIVERSAL_GAMES.md. Read that first.

WHAT SKYBOT GETS, AND ONLY THAT
    * the SCREEN: the rendered frame, resized to the brain's canonical square
      image (flat CHW float [0,1], same contract as the MineRL adapter);
    * the SPEAKERS: the stereo audio of the tics this step covered, as a
      (2, 32, 16) log-mel spectrogram on info["audio"] -> the GREEN `audio`
      sensor. Needs OpenAL at runtime (libopenal1 / openal-soft);
    * a CONTROLLER: the game YAML's `controller` list, slot 0 = no-op;
    * its own BODY: the same 13 proprio fields Minecraft fills
      (MineRLEnvAdapter.PROPRIO_KEYS, reused verbatim — one body layout, not
      two that can drift), from what a Doom player feels: health, being hurt,
      holding a weapon, sustained firing, which way it faces, whether it moved.

    Reward is ALWAYS 0.0. ViZDoom's own reward is scenario scripting, i.e. a
    number a player never sees; SkyBot earns from curiosity like it does
    everywhere. Kill/item/secret counts and map coordinates are RED: they go
    to info["evaluator"] / info["oracle"] and are logged, never learned.

ONE FRAME OF REFERENCE FOR THE WHOLE BRAIN
    Doom state is converted to MINECRAFT conventions at the source, so every
    downstream consumer (dead reckoning, the loop's spatial map, the heading
    proprio, coverage) runs unchanged:
        position  Doom units / 32 -> blocks (player 56 units vs 1.8 blocks)
                  x = doom_x, z = -doom_y (Doom's +y is north, MC's +z south)
        heading   Doom angle: 0 = east, counter-clockwise.
                  MC yaw:     0 = south (+z), clockwise.   yaw = -(angle + 90)
    tests/_vizdoom_smoke.py pins the heading conversion by cases.

EPISODES (docs §3.4)
    death       terminated=True; the next reset() reloads the CURRENT map.
    level exit  terminated=True; the next reset() starts the NEXT map.
    rewind      truncated=True, info["rewind"]=True AND info["env_restarted"]
                =True (the boundary the lifelong body honours); reset() loads
                the oldest save state (environments/rewind.py). The brain is
                not rewound.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from developmental_ai.environments.minerl_env import MineRLEnvAdapter as _MC
from developmental_ai.environments.rewind import (FRESH, NONE, REWIND,
                                                   SnapshotRing, StuckWatch)
from developmental_ai.environments.universal import stereo_mel
from developmental_ai.sensors import build_default_bus, crop_centre, DeadReckoner
from developmental_ai.sensors.builtins import (AUDIO_CHANNELS, AUDIO_FRAMES,
                                               AUDIO_MELS)

logger = logging.getLogger(__name__)

UNITS_PER_BLOCK = 32.0

_SAMPLING = {11025: "SR_11025", 22050: "SR_22050", 44100: "SR_44100"}


def _stop_clock_at_exit(stop: threading.Event, clock: threading.Thread) -> None:
    stop.set()
    clock.join(timeout=2.0)


def doom_angle_to_mc_yaw(angle_deg: float) -> float:
    """Doom (0=east, CCW) -> Minecraft yaw (0=south/+z, CW), in (-180, 180]."""
    y = -(float(angle_deg) + 90.0)
    return (y + 180.0) % 360.0 - 180.0


class ViZDoomEnvAdapter(gym.Env):
    """One Doom player for one SkyBot stream."""

    metadata = {"render_modes": ["rgb_array"]}

    # ONE BODY: the layout, its normalizers and the vector builder are
    # Minecraft's, verbatim (see module docstring).
    PROPRIO_KEYS = _MC.PROPRIO_KEYS
    PROPRIO_DIM = _MC.PROPRIO_DIM
    ATTACK_RUN_SCALE = _MC.ATTACK_RUN_SCALE
    MOVE_SCALE = _MC.MOVE_SCALE
    _proprio = _MC._proprio

    def __init__(self, game: Dict[str, Any], image_size: int = 128,
                 sensors_cfg: Optional[Dict] = None,
                 snapshot_dir: Optional[str] = None,
                 stream_index: int = 0, seed: int = 0,
                 _game_factory=None):
        super().__init__()
        import vizdoom as vzd                  # optional dep: imported here
        self._vzd = vzd
        self.game_cfg = dict(game or {})
        self.image_size = int(image_size)
        self.stream_index = int(stream_index)
        self.action_repeat = int(self.game_cfg.get("action_repeat", 4))
        self.controller: List[Dict[str, float]] = [
            dict(c or {}) for c in (self.game_cfg.get("controller") or [])]
        if not self.controller or self.controller[0]:
            raise ValueError("doom controller: slot 0 must be {} (the no-op) — "
                             "every game's slot 0 is its no-op (universal.py)")
        self.maps = list(self.game_cfg.get("maps") or ["MAP01"])
        self._map_i = 0
        rom = self.game_cfg.get("rom", "roms/doom/freedoom2.wad")
        if not os.path.isfile(rom):
            raise FileNotFoundError(
                f"Doom game file missing: {rom}. Run `bash scripts/fetch_roms.sh`"
                f" (Freedoom is free; nothing commercial is downloaded).")

        # buttons: union of every input the controller uses, in first-seen order
        self._buttons: List[str] = []
        for c in self.controller:
            for b in c:
                if b not in self._buttons:
                    self._buttons.append(b)
        for b in self._buttons:
            if not hasattr(vzd.Button, b):
                raise ValueError(f"doom controller: unknown ViZDoom button {b}")
        self._vectors = [[float(c.get(b, 0.0)) for b in self._buttons]
                         for c in self.controller]
        self._attack_idx = (self._buttons.index("ATTACK")
                            if "ATTACK" in self._buttons else None)

        audio = self.game_cfg.get("audio") or {}
        self.audio_enabled = bool(audio.get("enabled", True))
        self.sample_rate = int(audio.get("sample_rate", 22050))
        if self.sample_rate not in _SAMPLING:
            raise ValueError(f"doom audio.sample_rate must be one of "
                             f"{sorted(_SAMPLING)}")

        g = (_game_factory or vzd.DoomGame)()
        g.set_doom_game_path(rom)
        g.set_doom_map(self.maps[0])
        g.set_screen_resolution(getattr(vzd.ScreenResolution,
                                        self.game_cfg.get("render", "RES_256X192")))
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_window_visible(False)
        g.set_render_hud(True)                 # a player sees their HUD
        g.set_render_crosshair(False)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_episode_timeout(0)
        g.set_seed(int(seed) + self.stream_index)
        if self.audio_enabled:
            g.set_audio_buffer_enabled(True)
            g.set_audio_sampling_rate(getattr(vzd.SamplingRate,
                                              _SAMPLING[self.sample_rate]))
            # exactly this step (sync) / exactly one tic, accumulated by the
            # clock thread over however many tics a decision spans (realtime)
            g.set_audio_buffer_size(1 if self.game_cfg.get("realtime")
                                    else self.action_repeat)
        g.set_available_buttons([getattr(vzd.Button, b) for b in self._buttons])
        self._vars = ["HEALTH", "ARMOR", "POSITION_X", "POSITION_Y",
                      "POSITION_Z", "ANGLE", "PITCH", "KILLCOUNT",
                      "ITEMCOUNT", "SECRETCOUNT", "SELECTED_WEAPON_AMMO"]
        g.set_available_game_variables([getattr(vzd.GameVariable, v)
                                        for v in self._vars])
        g.init()
        self._g = g

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (3 * self.image_size * self.image_size,), np.float32)
        self.action_space = gym.spaces.Discrete(len(self.controller))

        # body-state carried between steps (Minecraft-convention units)
        self._move_scale = (float(self.game_cfg.get("move_units_per_tic", 8.0))
                            * self.action_repeat / UNITS_PER_BLOCK)
        self._dr = DeadReckoner()
        self._fovea_size = int((sensors_cfg or {}).get("fovea_size", 32))
        self._sensor_bus = None
        if sensors_cfg and sensors_cfg.get("enabled") is not None:
            self._sensor_bus = build_default_bus(
                self.PROPRIO_DIM, lambda ctx: self._proprio(ctx["world"]),
                enabled=list(sensors_cfg.get("enabled") or []),
                fovea_size=self._fovea_size, move_scale=self._move_scale)
        self._reset_body()

        # save states + stuck watch (docs §4)
        ck = self.game_cfg.get("checkpoint") or {}
        st = self.game_cfg.get("stuck") or {}
        self._ring = None
        if snapshot_dir and bool(ck.get("enabled", True)):
            self._ring = SnapshotRing(
                os.path.join(snapshot_dir, f"stream_{self.stream_index}"),
                save_fn=g.save, load_fn=g.load,
                interval_s=float(ck.get("interval_s", 300)),
                keep=int(ck.get("keep", 3)), ext="zds")
        self._watch = StuckWatch(
            window_s=float(st.get("window_s", 180)),
            ratio=float(st.get("ratio", 0.25)),
            warmup_s=float(st.get("warmup_s", 600)),
            baseline_halflife_s=float(st.get("baseline_halflife_s", 1800)),
            max_rewinds_without_recovery=int(
                st.get("max_rewinds_without_recovery", 3))) \
            if bool(st.get("enabled", True)) else None
        self._pending = NONE
        self._outcome = "start"          # start | died | exit
        self._closed = False

        # telemetry (RED / ops only)
        self.steps = 0
        self.deaths = 0
        self.exits = 0
        self.rewinds_done = 0
        self.fresh_starts = 0
        self._audio_nonsilent = 0
        self._audio_warned = False
        self._log_every = int(self.game_cfg.get("log_every", 5000))
        self._last_frame: Optional[np.ndarray] = None
        self._last_obs = np.zeros(self.observation_space.shape, np.float32)
        # LIVE VIEWER, EVERY TIC (2026-10-11). One agent step is
        # `action_repeat` tics of game time, but the frame the agent sees is
        # only the last. The viewer wants all of them so it can play the game
        # at its real speed (35 tics/s) on its own clock instead of one
        # still per decision. Off unless the loop's viewer asks for it
        # (set_keep_tic_frames), so a training run without a viewer steps
        # exactly as before. tests/_vizdoom_smoke.py pins that stepping one
        # tic at a time is the SAME game as stepping action_repeat at once.
        self.keep_tic_frames = False
        self.tic_seconds = 1.0 / 35.0             # Doom's fixed tic rate
        self._tic_frames: List[np.ndarray] = []

        # ---- REAL TIME (2026-10-11, docs/UNIVERSAL_GAMES.md §10) ----------
        # User decision: the game does NOT wait while SkyBot thinks. A clock
        # thread steps the engine one tic every 1/35 s with whatever buttons
        # are currently HELD; step() only sets the held buttons and returns
        # the newest frame plus how many tics passed since the last decision.
        # EVERY ViZDoom call happens on that one thread (the engine is not
        # thread-safe): ticks, deaths and map changes, save states, rewinds.
        self.realtime = bool(self.game_cfg.get("realtime", False))
        self.tic_hz = float(self.game_cfg.get("tic_hz", 35.0))
        self._held = self._vectors[0]
        self._cv = threading.Condition()
        self._tic = 0                       # tics the clock has run
        self._seen_tic = 0                  # tic of the last decision
        self._rt_latest: Optional[Tuple[np.ndarray, Dict[str, float]]] = None
        self._rt_audio: List[np.ndarray] = []
        self._rt_events: List[str] = []
        self._rt_attack_tics = 0
        self._rt_request: Optional[str] = None
        self._rt_result: Optional[bool] = None
        self._rt_done = threading.Event()
        self._rt_error: Optional[BaseException] = None
        self._rt_stop = threading.Event()
        self._frame_sink: Optional[Callable[[np.ndarray, float], None]] = None
        self.late_tics = 0                  # telemetry: clock fell behind
        self.decision_tics_last = 0
        # Started by the FIRST reset(), i.e. when SkyBot begins to play —
        # not at construction: the loop builds every stream and then its
        # world model before its first decision, and on a free-running clock
        # that build time would be spent standing still among monsters.
        self._clock: Optional[threading.Thread] = None

    # ------------------------------------------------------------ body
    def _reset_body(self) -> None:
        self._prev_xz: Optional[Tuple[float, float]] = None
        self._prev_yaw: Optional[float] = None
        self._prev_health: Optional[float] = None
        self._attack_run = 0
        self._dr.reset()
        self._cur_fovea = None
        self._prev_fovea = None
        self._last_pov_small = None
        self._visits: Dict[Tuple[int, int], int] = {}
        self._cells_ever = 0

    def _frame_to_obs(self, frame: np.ndarray) -> np.ndarray:
        from PIL import Image
        small = np.asarray(Image.fromarray(frame).resize(
            (self.image_size, self.image_size), Image.BOX), dtype=np.uint8)
        if self._sensor_bus is not None:
            self._prev_fovea = self._cur_fovea
            self._cur_fovea = crop_centre(frame, self._fovea_size)
        self._last_frame = frame
        self._last_pov_small = small
        return (small.astype(np.float32) / 255.0).transpose(2, 0, 1).reshape(-1)

    def _world(self, v: Dict[str, float]) -> Dict[str, Any]:
        """Doom variables -> the Minecraft-convention `world` dict."""
        x = v["POSITION_X"] / UNITS_PER_BLOCK
        z = -v["POSITION_Y"] / UNITS_PER_BLOCK
        yaw = doom_angle_to_mc_yaw(v["ANGLE"])
        health = v["HEALTH"]
        w: Dict[str, Any] = {
            # life on Minecraft's 0..20 scale, so `_proprio`'s /20 reads
            # Doom health 0..100 as 0..1 (over-heal clips at 1)
            "life": health * 0.2,
            "mainhand": "weapon",            # a Doom player always holds one
            "gui_open": False,               # no menus are reachable here
            "pitch": float(v["PITCH"]),
            "yaw": yaw,
            "x": x, "z": z, "xpos": x, "zpos": z,
            "y": v["POSITION_Z"] / UNITS_PER_BLOCK,
            "attack_run": float(self._attack_run),
        }
        if self._prev_health is not None and health < self._prev_health:
            w["damage"] = float(self._prev_health - health)
        self._prev_health = health
        cell = (int(x // 4), int(z // 4))
        n = self._visits.get(cell, 0)
        if n == 0:
            self._cells_ever += 1
        self._visits[cell] = n + 1
        if len(self._visits) > 50000:
            self._visits.pop(next(iter(self._visits)))
        w["coverage"] = float(1.0 / np.sqrt(1.0 + n))
        w["cell"] = cell
        w["cells_seen"] = int(self._cells_ever)
        pxz = self._prev_xz
        dx, dz = (0.0, 0.0) if pxz is None else (x - pxz[0], z - pxz[1])
        w["moved"] = float(np.hypot(dx, dz))
        r = np.radians(yaw)
        fx, fz = -np.sin(r), np.cos(r)               # MC forward unit vector
        w["move_fwd"] = float(dx * fx + dz * fz)
        w["move_lat"] = float(dx * fz - dz * fx)
        w["yaw_delta"] = (0.0 if self._prev_yaw is None else
                          float((yaw - self._prev_yaw + 180.0) % 360.0 - 180.0))
        self._dr.step(w["move_fwd"], w["move_lat"], yaw)
        w["dr_x"], w["dr_z"] = self._dr.x, self._dr.z
        w["move_dy"] = 0.0
        w["fall_dist"] = 0.0
        self._prev_xz = (x, z)
        self._prev_yaw = yaw
        return w

    def _read(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        st = self._g.get_state()
        if st is None:                                # episode just ended
            return self._last_obs, {}
        v = {k: float(x) for k, x in zip(self._vars, st.game_variables)}
        return self._compose(np.asarray(st.screen_buffer, dtype=np.uint8), v,
                             st.audio_buffer, self.action_repeat)

    def _compose(self, frame: np.ndarray, v: Dict[str, float], audio,
                 decision_tics: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Frame + variables + the audio of this decision -> (obs, info)."""
        obs = self._frame_to_obs(frame)
        info: Dict[str, Any] = {"game": "doom"}
        world = self._world(v)
        # how much game time this decision covered (the `tempo` sense)
        world["decision_seconds"] = float(decision_tics) / self.tic_hz
        info["decision_tics"] = int(decision_tics)
        info["world"] = world
        if self.audio_enabled:
            mel = stereo_mel(audio, self.sample_rate,
                             n_mels=AUDIO_MELS, n_frames=AUDIO_FRAMES)
            if mel is not None and mel.shape == (AUDIO_CHANNELS, AUDIO_MELS,
                                                 AUDIO_FRAMES):
                info["audio"] = mel
                if audio is not None and np.any(audio):
                    self._audio_nonsilent += 1
        info["proprio"] = self._proprio(world)
        if self._sensor_bus is not None:
            ctx = {"world": world, "info": info, "pov_native": self._last_frame,
                   "pov_small": self._last_pov_small,
                   "prev_fovea": self._prev_fovea}
            try:
                info["sensors"] = self._sensor_bus.read_policy(ctx)
                oracle = self._sensor_bus.read_oracle(ctx)
                if oracle:
                    info["oracle"] = oracle
            except Exception:
                info["sensors"] = self._sensor_bus.neutral_policy()
            info["sensor_layout"] = self._sensor_bus.layout_hash()
        # RED: evaluation only — no consumer in the learning path reads it
        info["evaluator"] = {"map": self.maps[self._map_i],
                             "kills": v["KILLCOUNT"], "items": v["ITEMCOUNT"],
                             "secrets": v["SECRETCOUNT"],
                             "health": v["HEALTH"], "armor": v["ARMOR"],
                             "deaths": self.deaths, "exits": self.exits,
                             "rewinds": self.rewinds_done,
                             "fresh_starts": self.fresh_starts}
        self._last_obs = obs
        return obs, info

    # ------------------------------------------------------------ real time
    def _clock_loop(self) -> None:
        """The game's own timer: one tic every 1/tic_hz s, whatever SkyBot is
        doing. Holds the last buttons SkyBot chose. Owns every engine call."""
        period = 1.0 / self.tic_hz
        nxt = time.perf_counter()
        g = self._g
        while not self._rt_stop.is_set():
            try:
                req = self._rt_request
                if req is not None:
                    self._rt_result = self._rt_service(req)
                    self._rt_request = None
                    self._rt_done.set()
                vec = self._held
                g.make_action(vec, 1)
                if self._attack_idx is not None and vec[self._attack_idx] > 0:
                    self._rt_attack_tics += 1
                else:
                    self._rt_attack_tics = 0
                ev = None
                if g.is_episode_finished():
                    # A player who dies reloads the level; one who exits
                    # goes on. The clock does it at once — the game never
                    # waits for SkyBot, not even for a respawn.
                    if g.is_player_dead():
                        ev = "died"
                    else:
                        ev = "exit"
                        self._map_i = (self._map_i + 1) % len(self.maps)
                        if self._ring is not None:
                            self._ring.clear()
                    self._new_episode(self.maps[self._map_i])
                st = g.get_state()
                if st is not None:
                    frame = np.array(st.screen_buffer, dtype=np.uint8)
                    v = {k: float(x) for k, x in zip(self._vars, st.game_variables)}
                    aud = (None if st.audio_buffer is None
                           else np.array(st.audio_buffer))
                    with self._cv:
                        self._rt_latest = (frame, v)
                        if ev is not None:
                            self._rt_events.append(ev)
                        if aud is not None:
                            self._rt_audio.append(aud)
                            if len(self._rt_audio) > 2 * int(self.tic_hz):
                                del self._rt_audio[:-2 * int(self.tic_hz)]
                        self._tic += 1
                        self._cv.notify_all()
                    sink = self._frame_sink
                    if sink is not None:
                        try:
                            sink(frame, period)
                        except Exception:
                            pass
                    if self._ring is not None and ev is None:
                        self._ring.maybe_save()
            except Exception as e:                     # engine died
                self._rt_error = e
                with self._cv:
                    self._cv.notify_all()
                self._rt_done.set()
                logger.error("doom stream %d: clock stopped: %s",
                             self.stream_index, e)
                return
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                self.late_tics += 1
                if d < -0.5:                           # far behind: resync
                    nxt = time.perf_counter()

    def _rt_service(self, req: str) -> bool:
        """Rewind / fresh start, executed ON the clock thread."""
        if req == REWIND and self._ring is not None:
            if self._g.is_episode_finished():
                self._new_episode(self.maps[self._map_i])
            if self._ring.load_oldest():
                self._ring.restart_interval()
                self.rewinds_done += 1
                return True
        if self._ring is not None:
            self._ring.clear()
        self.fresh_starts += 1
        self._new_episode(self.maps[self._map_i])
        return False

    def _rt_collect(self, timeout: float = 5.0):
        """Wait for at least one fresh tic; take everything since the last
        decision. Returns (frame, vars, audio, tics, events)."""
        with self._cv:
            ok = self._cv.wait_for(
                lambda: self._tic > self._seen_tic or self._rt_error is not None,
                timeout=timeout)
            if self._rt_error is not None:
                raise RuntimeError(f"doom clock died: {self._rt_error}")
            if not ok:
                raise RuntimeError("doom clock stalled (no tic in %.0fs)" % timeout)
            tics = self._tic - self._seen_tic
            self._seen_tic = self._tic
            frame, v = self._rt_latest
            audio = (np.concatenate(self._rt_audio, axis=0)
                     if self._rt_audio else None)
            self._rt_audio = []
            events, self._rt_events = self._rt_events, []
        return frame, v, audio, tics, events

    def set_frame_sink(self, sink: Optional[Callable[[np.ndarray, float], None]]) -> None:
        """Every frame the clock renders goes to `sink(frame, seconds)` —
        the live viewer, at the game's own 35 fps, independent of SkyBot."""
        self._frame_sink = sink

    # ------------------------------------------------------------ gym API
    def observe_reward(self, reward: float) -> None:
        """Called by the loop once per step with what this stream earned.
        A stuck verdict is acted on at the NEXT step boundary (truncate)."""
        if self._watch is None or self._pending != NONE:
            return
        verdict = self._watch.observe(reward)
        if verdict != NONE:
            self._pending = verdict
            logger.info("doom stream %d: stuck (%s) — %s", self.stream_index,
                        self._watch.stats(), verdict)

    def step(self, action):
        if self._closed:
            raise RuntimeError("ViZDoom adapter closed")
        if self._pending != NONE:
            # End this life here; reset() performs the rewind / fresh start.
            # SIGNALLED AS env_restarted, NOT ONLY truncated (2026-10-11).
            # The lifelong body stores `done = terminated or restart` and
            # deliberately ignores a plain truncation (a time limit is not a
            # boundary there), so `truncated` alone would never reset this
            # stream: every later step would land back here, pending forever
            # -- a guard become a latch (CLAUDE.md 4.1). env_restarted is the
            # discontinuity both bodies honour: death_reset, a bounded
            # reset(), and the replay restart flag, so nothing is spliced
            # across the jump in game state. tests/_vizdoom_smoke.py pins it.
            info = {"game": "doom", "rewind": True, "env_restarted": True,
                    "gui_open": False}
            if self.realtime:
                self._held = self._vectors[0]   # stand still until reset()
            return self._last_obs, 0.0, False, True, info
        a = int(action)
        if not 0 <= a < len(self._vectors):
            a = 0
        if self.realtime:
            return self._step_realtime(a)
        vec = self._vectors[a]
        if self._attack_idx is not None and vec[self._attack_idx] > 0:
            self._attack_run += 1
        else:
            self._attack_run = 0
        if self.keep_tic_frames:
            # Same inputs held for the same tics; only the rendering differs
            # (each tic's frame is kept for the viewer).
            for _t in range(self.action_repeat):
                self._g.make_action(vec, 1)
                if self._g.is_episode_finished():
                    break
                _st = self._g.get_state()
                if _st is not None:
                    self._tic_frames.append(
                        np.array(_st.screen_buffer, dtype=np.uint8))
            # bounded: if nobody pops (dream phase, viewer paused), keep only
            # the newest few steps' worth rather than growing forever
            _cap = 4 * self.action_repeat
            if len(self._tic_frames) > _cap:
                del self._tic_frames[:-_cap]
        else:
            self._g.make_action(vec, self.action_repeat)
        self.steps += 1
        terminated = False
        if self._g.is_episode_finished():
            terminated = True
            if self._g.is_player_dead():
                self.deaths += 1
                self._outcome = "died"
            else:
                self.exits += 1
                self._outcome = "exit"
            obs, info = self._last_obs, {"game": "doom",
                                         "world": {"died": self._outcome == "died"}}
        else:
            obs, info = self._read()
            if self._ring is not None:
                self._ring.maybe_save()
        info.setdefault("gui_open", False)
        info["env_restarted"] = False
        info["rewind"] = False
        self._periodic_checks(info)
        return obs, 0.0, terminated, False, info

    def _step_realtime(self, a: int):
        """Hold slot `a` from now on; return the newest frame and how much
        game time passed since the last decision. Never advances the game."""
        self._held = self._vectors[a]                  # atomic ref swap
        frame, v, audio, tics, events = self._rt_collect()
        self.steps += 1
        self.decision_tics_last = tics
        terminated = bool(events)
        if terminated:
            self.deaths += events.count("died")
            self.exits += events.count("exit")
            # the frame is already the NEW life's: no phantom movement or
            # damage may be computed across the boundary
            self._reset_body()
        self._attack_run = self._rt_attack_tics
        obs, info = self._compose(frame, v, audio, tics)
        info["gui_open"] = False
        info["env_restarted"] = False
        info["rewind"] = False
        if terminated:
            info["world"]["died"] = "died" in events
        self._periodic_checks(info)
        return obs, 0.0, terminated, False, info

    def _periodic_checks(self, info) -> None:
        if (self.audio_enabled and not self._audio_warned
                and self.steps == 5000 and self._audio_nonsilent == 0):
            self._audio_warned = True
            logger.warning("doom stream %d: 5000 steps of SILENT audio — is "
                           "OpenAL installed (libopenal1 / openal-soft)? "
                           "SkyBot is training deaf.", self.stream_index)
        if self._log_every and self.steps % self._log_every == 0:
            ev = dict(info.get("evaluator") or {})
            if self.realtime:
                ev.update(tic=self._tic, late_tics=self.late_tics,
                          decision_tics=self.decision_tics_last)
            logger.info("doom stream %d: %s", self.stream_index, ev)

    def _new_episode(self, map_name: str) -> None:
        self._g.set_doom_map(map_name)
        self._g.new_episode()

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("ViZDoom adapter closed")
        if self.realtime:
            return self._reset_realtime()
        rewound = False
        pending, self._pending = self._pending, NONE
        if pending == REWIND and self._ring is not None:
            if self._g.is_episode_finished():
                self._new_episode(self.maps[self._map_i])
            rewound = self._ring.load_oldest()
            if rewound:
                self._ring.restart_interval()
                self.rewinds_done += 1
            else:
                pending = FRESH                         # nothing to rewind to
        if pending == FRESH or (pending == REWIND and not rewound):
            if self._ring is not None:
                self._ring.clear()
            self.fresh_starts += 1
            self._new_episode(self.maps[self._map_i])
        elif not rewound:
            if self._outcome == "exit":
                self._map_i = (self._map_i + 1) % len(self.maps)
                if self._ring is not None:   # old map's states are not this map
                    self._ring.clear()
            if self._outcome != "start" or self._g.is_episode_finished():
                self._new_episode(self.maps[self._map_i])
        self._outcome = "start"
        self._reset_body()
        obs, info = self._read()
        info["rewind"] = rewound
        info["env_restarted"] = False
        return obs, info

    def _reset_realtime(self):
        """The game is already running: a reset never starts a new episode on
        its own (the clock already did, on a death or exit). It performs a
        pending rewind / fresh start ON the clock thread and returns the
        newest frame."""
        if self._clock is None:
            self._clock = threading.Thread(
                target=self._clock_loop, daemon=True,
                name=f"doom-clock-{self.stream_index}")
            self._clock.start()
            # STOP AT INTERPRETER EXIT (2026-10-11). An adapter whose clock
            # is still ticking when the process exits -- any crash path that
            # skips close() -- left a daemon thread inside an engine call
            # during finalization, and the process HUNG instead of exiting
            # (measured: a failing test never returned). A hung trainer is
            # worse than a crashed one: the supervisor sees a live pid.
            import atexit
            atexit.register(_stop_clock_at_exit, self._rt_stop, self._clock)
        pending, self._pending = self._pending, NONE
        rewound = False
        # Release the buttons FIRST: the clock keeps ticking while this
        # request waits, and it must not walk SkyBot off the restored spot
        # with the action it held before the rewind (measured: 4.45 blocks).
        self._held = self._vectors[0]
        if pending in (REWIND, FRESH):
            self._rt_done.clear()
            self._rt_request = REWIND if pending == REWIND else FRESH
            if not self._rt_done.wait(timeout=30.0):
                raise RuntimeError("doom clock did not service %s" % pending)
            if self._rt_error is not None:
                raise RuntimeError(f"doom clock died: {self._rt_error}")
            rewound = bool(self._rt_result)
            # Everything the clock produced before the load belongs to the
            # abandoned timeline: never hand back a pre-rewind frame, sound
            # or episode event. The next tic is after the load.
            with self._cv:
                self._seen_tic = self._tic
                self._rt_audio = []
                self._rt_events = []
        frame, v, audio, tics, _events = self._rt_collect()
        self._reset_body()
        self._attack_run = 0
        obs, info = self._compose(frame, v, audio, tics)
        info["rewind"] = rewound
        info["env_restarted"] = False
        return obs, info

    def render(self):
        return self._last_frame

    def set_keep_tic_frames(self, on: bool) -> None:
        self.keep_tic_frames = bool(on)
        if not on:
            self._tic_frames = []

    def pop_tic_frames(self) -> List[np.ndarray]:
        """Every tic frame since the last call, oldest first (for the viewer)."""
        out, self._tic_frames = self._tic_frames, []
        return out

    def close(self):
        # FIRST, like MineRLEnvAdapter.close (2026-10-09): anything still
        # running on an abandoned pool thread must see this and stop.
        self._closed = True
        if self._clock is not None:
            self._rt_stop.set()
            self._clock.join(timeout=5.0)
            if self._clock.is_alive():
                logger.warning("doom stream %d: clock thread did not stop",
                               self.stream_index)
        try:
            self._g.close()
        except Exception as e:
            logger.warning("ViZDoom close failed: %s", e)
