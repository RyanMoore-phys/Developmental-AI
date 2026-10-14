"""SkyBot's controller and the brain contract every game must satisfy.

Born 2026-10-11 (docs/UNIVERSAL_GAMES.md §3). ONE brain plays every game at
once, as concurrent streams. The loop sizes that brain from stream 0
(obs_dim, action_dim, PROPRIO_DIM, sensor layout), so every stream must
present the identical shape. This module is where that is enforced:

    UniversalControllerWrapper   pads any game's discrete actions to
                                 `action_slots`, executes slot 0 (the game's
                                 no-op) for every slot the game does not map,
                                 exposes `action_mask`, and REFUSES TO START
                                 if the observation is not the canonical
                                 3 x image_size x image_size.

    find_attr(env, name)         walk the wrapper chain (gymnasium >= 1.0 does
                                 not forward attributes; a bare hasattr on the
                                 outer wrapper silently returns False).

    stereo_mel(samples, ...)     int16 stereo samples -> (2, mels, frames) in
                                 [0,1], the contract of the GREEN `audio`
                                 sensor (sensors/builtins.py).

SLOT 0 IS THE NO-OP IN EVERY GAME. `curiosity.null_action: 0` (the learning-
progress baseline) and the safe-by-construction fallback both rely on it.
Minecraft's TREECHOP_MACROS[0] is already `{}`, so this costs it nothing.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np


def find_attr(env: Any, name: str, depth: int = 8) -> Any:
    """First `name` found walking env -> env.env -> ... ; None if absent."""
    e = env
    for _ in range(depth):
        if e is None:
            return None
        if name in getattr(e, "__dict__", {}) or hasattr(type(e), name):
            return getattr(e, name, None)
        e = getattr(e, "env", None)
    return None


class UniversalControllerWrapper(gym.Wrapper):
    """Present any discrete-action game as SkyBot's fixed-width controller.

    `mapped` (optional, length = the game's native action count) marks which
    native actions are real inputs; by default every native action is. Slots
    at or beyond the native count, and unmapped native actions, execute slot
    0. That makes a missed mask harmless (one idle step) rather than a crash
    or a phantom input; the mask then only saves wasted exploration.
    """

    def __init__(self, env: gym.Env, action_slots: int, image_size: int,
                 channels: int = 3, mapped: Optional[Sequence[bool]] = None,
                 game: str = "?"):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(f"universal controller: {game} has a non-discrete "
                             f"action space {env.action_space}")
        self.native_n = int(env.action_space.n)
        self.action_slots = int(action_slots)
        if self.native_n > self.action_slots:
            raise ValueError(
                f"universal controller: {game} needs {self.native_n} actions "
                f"but the brain has {self.action_slots} slots. Raising "
                f"universal.action_slots changes the brain's shapes — decide "
                f"that deliberately (docs/UNIVERSAL_GAMES.md §3.1).")
        want = (int(channels) * int(image_size) * int(image_size),)
        got = tuple(int(x) for x in env.observation_space.shape)
        if int(np.prod(got)) != want[0]:
            raise ValueError(
                f"universal controller: {game} observation {got} is not the "
                f"brain's {channels}x{image_size}x{image_size} = {want[0]} "
                f"(every stream must present the identical shape)")
        m = np.zeros(self.action_slots, dtype=bool)
        if mapped is None:
            m[:self.native_n] = True
        else:
            mapped = list(mapped)
            if len(mapped) != self.native_n:
                raise ValueError("universal controller: `mapped` must have one "
                                 "entry per native action")
            m[:self.native_n] = np.asarray(mapped, dtype=bool)
        m[0] = True                      # the no-op is always available
        self.action_mask = m
        self.game = game
        self.action_space = gym.spaces.Discrete(self.action_slots)
        self.unmapped_steps = 0

    def to_native(self, action: Any) -> int:
        a = int(action)
        if a < 0 or a >= self.native_n or not self.action_mask[a]:
            self.unmapped_steps += 1
            return 0
        return a

    def step(self, action):
        return self.env.step(self.to_native(action))


# ---------------------------------------------------------------- audio

_MEL_CACHE: dict = {}


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    key = (sr, n_fft, n_mels)
    fb = _MEL_CACHE.get(key)
    if fb is not None:
        return fb

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = bins[i], bins[i + 1], bins[i + 2]
        c = max(c, lo + 1)
        hi = max(hi, c + 1)
        for k in range(lo, min(c, n_bins)):
            fb[i, k] = (k - lo) / max(1, c - lo)
        for k in range(c, min(hi, n_bins)):
            fb[i, k] = (hi - k) / max(1, hi - c)
    _MEL_CACHE[key] = fb
    return fb


def stereo_mel(samples: Optional[np.ndarray], sample_rate: int,
               n_mels: int = 32, n_frames: int = 16,
               floor_db: float = -80.0) -> Optional[np.ndarray]:
    """(N, 2) or (2, N) int16/float stereo -> float32 (2, n_mels, n_frames).

    Log-mel magnitude, scaled so `floor_db`..0 dBFS maps to 0..1. Left and
    right stay separate channels: direction is half of what a sound tells a
    player. Returns None for no / empty / non-stereo input, which the `audio`
    sensor reads as neutral — never as a fabricated silence.
    """
    if samples is None:
        return None
    a = np.asarray(samples)
    if a.ndim != 2 or a.size == 0:
        return None
    if a.shape[0] != 2 and a.shape[1] == 2:
        a = a.T
    if a.shape[0] != 2:
        return None
    x = a.astype(np.float32)
    if np.issubdtype(np.asarray(samples).dtype, np.integer):
        x /= 32768.0
    n = x.shape[1]
    hop = max(1, n // n_frames)
    n_fft = 1
    while n_fft < 2 * hop:
        n_fft *= 2
    n_fft = max(n_fft, 64)
    win = np.hanning(n_fft).astype(np.float32)
    fb = _mel_filterbank(int(sample_rate), n_fft, n_mels)
    out = np.zeros((2, n_mels, n_frames), dtype=np.float32)
    pad = np.pad(x, ((0, 0), (n_fft // 2, n_fft // 2)))
    for t in range(n_frames):
        s = t * hop
        seg = pad[:, s:s + n_fft]
        if seg.shape[1] < n_fft:
            seg = np.pad(seg, ((0, 0), (0, n_fft - seg.shape[1])))
        spec = np.abs(np.fft.rfft(seg * win, axis=1)) / (n_fft / 2.0)
        mel = spec @ fb.T                                    # (2, n_mels)
        db = 20.0 * np.log10(np.maximum(mel, 1e-10))
        out[:, :, t] = np.clip((db - floor_db) / (-floor_db), 0.0, 1.0)
    return out
