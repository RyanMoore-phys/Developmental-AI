"""Episode video recorder — "let me SEE what the agent actually does."

A lightweight gymnasium wrapper that captures frames during episodes and
writes MP4s (imageio/ffmpeg). Designed for the rich-env program:

  * Crafter / pixel envs: the OBSERVATION IS the frame — recorded directly
    (upscaled for watchability).
  * MineRL-style envs: pass frame_key="pov" to record obs dicts.
  * Anything with render(): pass source="render".

Recording every episode of a long run would be huge, so it records episode
windows: the FIRST `head` episodes (what does an untrained agent do?), then
every `every`-th episode (development over time), plus always the episode
with a new best reward (the highlights reel).

Usage:
    env = VideoRecorder(env, out_dir="videos/run1", source="obs",
                        image_hw=(64, 64), channels=3, upscale=6)
    ... normal gym loop ...
    env.finalize()  # flush any open episode
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore


class VideoRecorder(gym.Wrapper):
    def __init__(
        self,
        env,
        out_dir: str,
        source: str = "obs",          # "obs" | "render" | "dict"
        frame_key: str = "pov",        # for source="dict"
        image_hw: Optional[Tuple[int, int]] = None,  # for flat CHW obs
        channels: int = 3,
        upscale: int = 6,
        head: int = 3,
        every: int = 50,
        fps: int = 12,
        max_frames: int = 0,
    ):
        super().__init__(env)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.source = source
        self.frame_key = frame_key
        self.image_hw = image_hw
        self.channels = channels
        self.upscale = int(upscale)
        self.head = int(head)
        self.every = int(every)
        self.fps = int(fps)
        # max_frames > 0 chops a NEVER-RESETTING stream (lifelong env 0) into
        # fixed-length windows: flush every max_frames steps and advance the
        # head/every sampling schedule per WINDOW instead of per episode.
        # Without this the lifelong stream never term/truncs -> reset() never
        # fires -> zero MP4s ever written AND the frame buffer grows unbounded
        # (~2.4 GB RAM/hour at 128px/13 fps — found live 2026-07-23).
        # 0 = off (episodic envs keep the original flush-on-reset behavior).
        self.max_frames = int(max_frames)

        self._episode = 0
        self._frames: list = []
        self._recording = False
        self._ep_reward = 0.0
        self._best_reward = -np.inf
        self._win_steps = 0

    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped env.

        gymnasium >=1.0 REMOVED implicit wrapper attribute forwarding, so
        wrapping DevelopmentalEnvWrapper broke every consumer that reads
        env.episode_length / obs_history / last_info / freeze_obs_stats /
        obs_dim ... (caught live on the Minecraft run). Underscore names are
        excluded so this can never mask our own missing internals or recurse
        during __init__.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    # ---- frame extraction -------------------------------------------------
    def _frame(self, obs) -> Optional[np.ndarray]:
        if self.source == "render":
            f = self.env.render()
        elif self.source == "dict":
            f = obs.get(self.frame_key) if isinstance(obs, dict) else None
        else:
            f = obs
        if f is None:
            return None
        f = np.asarray(f)
        if f.ndim == 1 and self.image_hw:  # flat CHW [0,1] -> HWC uint8
            h, w = self.image_hw
            try:
                f = f.reshape(self.channels, h, w).transpose(1, 2, 0)
            except ValueError:
                return None
            f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        elif f.dtype != np.uint8:
            f = np.clip(f * 255.0 if f.max() <= 1.0 else f, 0, 255).astype(np.uint8)
        if f.ndim == 2:
            f = np.stack([f] * 3, axis=-1)
        # NOTE: upscaling is deferred to _flush — buffering upscaled frames
        # held upscale^2 x the RAM (~1GB/episode at 128px, upscale 4).
        return f

    def _should_record(self) -> bool:
        # NOTE: _episode is 1-indexed at this point (incremented in reset
        # BEFORE this check), so head must be inclusive — `<` silently
        # recorded nothing when head=1 (caught by the MineRL smoke).
        return (self._episode <= self.head
                or (self.every > 0 and self._episode % self.every == 0))

    # ---- gym API ----------------------------------------------------------
    def reset(self, **kw):
        self._flush(tag=None)
        obs, info = self.env.reset(**kw)
        self._episode += 1
        self._recording = self._should_record()
        self._ep_reward = 0.0
        self._frames = []
        self._win_steps = 0
        if self._recording:
            f = self._frame(obs)
            if f is not None:
                self._frames.append(f)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._ep_reward += float(r)
        if self._recording:
            f = self._frame(obs)
            if f is not None:
                self._frames.append(f)
        if term or trunc:
            tag = None
            if self._ep_reward > self._best_reward:
                self._best_reward = self._ep_reward
                tag = "best"
            self._flush(tag=tag)
        elif self.max_frames > 0:
            # Stream-window mode: the lifelong env never term/truncs, so we
            # advance the schedule on a fixed step clock. Count steps even
            # while NOT recording — otherwise the window counter would stall
            # after the head windows and the every-th window never arrives.
            self._win_steps += 1
            if self._win_steps >= self.max_frames:
                tag = None
                if self._recording and self._ep_reward > self._best_reward:
                    self._best_reward = self._ep_reward
                    tag = "best"
                self._flush(tag=tag)   # no-op when not recording (no frames)
                self._win_steps = 0
                self._episode += 1     # a window IS an episode to the schedule
                self._recording = self._should_record()
                self._ep_reward = 0.0
        return obs, r, term, trunc, info

    # ---- writing ----------------------------------------------------------
    def _flush(self, tag: Optional[str]) -> None:
        if not self._frames:
            return
        keep = self._recording or tag == "best"
        if keep and len(self._frames) > 1:
            name = f"ep{self._episode:05d}_r{self._ep_reward:+.1f}"
            if tag:
                name += f"_{tag}"
            path = os.path.join(self.out_dir, name + ".mp4")
            try:
                import imageio.v2 as imageio
                u = self.upscale
                # Stream frames to the writer one at a time — upscaling the
                # whole episode up-front held upscale^2 x the RAM.
                with imageio.get_writer(path, fps=self.fps,
                                        macro_block_size=1) as w:
                    for f in self._frames:
                        if u > 1:
                            f = np.repeat(np.repeat(f, u, 0), u, 1)
                        w.append_data(f)
            except Exception:
                pass  # video is best-effort; never break training
        self._frames = []
        self._recording = False

    def finalize(self) -> None:
        self._flush(tag=None)
