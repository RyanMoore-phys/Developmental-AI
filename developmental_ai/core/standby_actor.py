"""Standby actor: SkyBot keeps playing real-time games while it learns.

Born 2026-10-11 — stage R3 of docs/UNIVERSAL_GAMES.md §10. User decision:
the game does not wait for SkyBot, and acting must be independent of
learning. Within a segment the loop acts (and since R2 never waits on
world-model training). BETWEEN segments the loop learns — the PPO update,
dream training, skill practice, reports, checkpoints — and on a real-time
clock every second of that used to be a second SkyBot stood still holding
its last buttons while the game went on without it.

This thread takes the real-time streams over for exactly that window:

    loop:    ...segment... | snapshot -> START standby | PPO, dream, saves... | STOP -> handoff | ...segment...
    standby:                 perceive -> believe -> decide -> act, per stream, as fast as it can

It only ACTS. It holds:
  * a frozen copy of the policy (refreshed as soon as the PPO update lands),
  * the published inference world model (R2) for its own belief per stream,
  * a snapshot of each stream's belief, body sense and decision mask.
It touches no learner state: no replay, no rollout buffer, no curiosity, no
broadcaster, no option executor (primitives only — options need learner-side
bookkeeping). The experience it gathers is not learned from; the handoff
tells the loop where each stream really is now (frame, belief) and marks the
gap so nothing is stitched across it.

Synchronous games (Minecraft) are never handed to it: they are paused while
the loop learns, so there is nothing to keep up with.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StandbyActor:
    def __init__(self, envs: Dict[int, Any], policy, world_model,
                 wm_lock, wm_proprio_fn: Callable, device, action_dim: int,
                 beliefs: Dict[int, Dict[str, torch.Tensor]],
                 obs: Dict[int, np.ndarray], infos: Dict[int, Dict],
                 aug_proprio: Dict[int, Optional[np.ndarray]],
                 masks: Dict[int, Optional[np.ndarray]],
                 knowledge: Optional[np.ndarray] = None,
                 use_feats: bool = True):
        self.envs = dict(envs)
        self._policy = policy
        self.world_model = world_model
        self.lock = wm_lock
        self.wm_proprio_fn = wm_proprio_fn
        self.device = device
        self.action_dim = int(action_dim)
        self.state = {e: {"h": b["h"].clone(), "z": b["z"].clone()}
                      for e, b in beliefs.items()}
        self.obs = {e: np.asarray(o, dtype=np.float32) for e, o in obs.items()}
        self.infos = dict(infos)
        self.aug = {e: (None if a is None else np.asarray(a, np.float32).copy())
                    for e, a in aug_proprio.items()}
        self.masks = dict(masks)
        self.knowledge = knowledge
        self.use_feats = bool(use_feats)
        self.boundary = {e: False for e in self.envs}
        self.decisions = {e: 0 for e in self.envs}
        self.errors: Dict[int, str] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.t_start = None
        self.t_stop = None

    # -- the policy can be swapped while running (after the PPO update) --
    def set_policy(self, policy) -> None:
        self._policy = policy                      # atomic reference swap

    def start(self) -> None:
        self.t_start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="standby-actor")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> Dict[int, Dict[str, Any]]:
        """Stop and hand every stream back: its newest frame, info, belief,
        body sense, and whether a boundary (death/exit/rewind) or a gap
        happened. Always marks the gap: time passed that the loop did not
        record, so the next stored transition must not splice across it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("standby actor did not stop within %.0fs", timeout)
        self.t_stop = time.time()
        out = {}
        for e in self.envs:
            out[e] = {"obs": self.obs[e], "info": self.infos.get(e) or {},
                      "h": self.state[e]["h"], "z": self.state[e]["z"],
                      "aug": self.aug.get(e), "boundary": self.boundary[e],
                      "decisions": self.decisions[e]}
        return out

    # ------------------------------------------------------------------
    def _decide(self, e: int) -> int:
        pol = self._policy
        kw: Dict[str, Any] = {}
        if self.masks.get(e) is not None:
            kw["action_mask"] = self.masks[e]
        if self.aug.get(e) is not None:
            kw["proprio"] = self.aug[e]
        if self.use_feats:
            with self.lock, torch.no_grad():
                kw["feats"] = self.world_model.rssm.get_latent(self.state[e]).detach()
        a, _info = pol.select_action(
            self.obs[e], knowledge=(self.knowledge if e == 0 else None), **kw)
        return int(a)

    def _believe(self, e: int, action: int, next_obs: np.ndarray, info) -> None:
        x = torch.as_tensor(np.asarray(next_obs, np.float32)).unsqueeze(0).to(self.device)
        a = torch.zeros(1, self.action_dim, device=self.device)
        if 0 <= action < self.action_dim:
            a[0, action] = 1.0
        with self.lock, torch.no_grad():
            emb = self.world_model.embed(x, self.wm_proprio_fn([info], 1))
            st, _ = self.world_model.rssm.observe_step(self.state[e], a, emb)
        self.state[e] = {"h": st["h"].detach(), "z": st["z"].detach()}

    def _zero_belief(self, e: int) -> None:
        self.state[e] = {k: torch.zeros_like(v) for k, v in self.state[e].items()}

    def _run(self) -> None:
        live = set(self.envs)
        while not self._stop.is_set() and live:
            for e in list(live):
                if self._stop.is_set():
                    break
                env = self.envs[e]
                try:
                    a = self._decide(e)
                    nobs, _r, term, trunc, info = env.step(a)
                    info = info or {}
                    if term or info.get("env_restarted"):
                        # a death, a level exit or a rewind: a new life. The
                        # real-time clock already restarted the game; reset()
                        # services a rewind (on the clock thread) and returns
                        # the newest frame.
                        self.boundary[e] = True
                        self._zero_belief(e)
                        nobs, info = env.reset()
                        info = info or {}
                    else:
                        self._believe(e, a, nobs, info)
                    self.obs[e] = np.asarray(nobs, dtype=np.float32)
                    self.infos[e] = info
                    raw = info.get("proprio")
                    if raw is not None and self.aug.get(e) is not None:
                        r = np.asarray(raw, np.float32).reshape(-1)
                        self.aug[e][:r.shape[0]] = r
                    self.decisions[e] += 1
                except Exception as ex:          # never take the run down
                    self.errors[e] = repr(ex)
                    logger.warning("standby actor: stream %d stopped acting: %s",
                                   e, ex)
                    live.discard(e)
