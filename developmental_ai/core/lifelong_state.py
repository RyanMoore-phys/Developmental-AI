"""Carried working-set state for the CONTINUOUS (lifelong) waking stream.

Owns what the EPISODIC loop wipes at every 8000-tick boundary but the
continuous loop must carry unbroken across software segments:
  * the batched RSSM recurrent state {h, z}  -> carry + per-segment soft decay
  * the broadcaster achievement mask         -> carry + soft decay (delegated)
  * the vision-scaffold belief               -> carried by NOT resetting it;
                                                hard-reset only on death

Hard-zeroing happens ONLY on a true terminal (death = 64-log success, or a
client-crash env restart) — NEVER on the 8000-tick time-truncation, which in
lifelong mode is not a boundary at all. In MineRL Treechop death effectively
never fires, so the interim validation is a pure continuous stream.

This is the ONE interface the LATER paging increment and the brain-viewer read
to inspect / page the working set — keep accessors here; do not reach into
rssm_state / broadcaster internals from the loop.

Design note (why decay is a SOFT guard, not a load-bearing fix): rssm.py
LayerNorms h every step (gru_norm), so ||h|| is likely already bounded. The
per-segment decay is a cheap guard against stale recurrent context dominating
over a multi-day life; whether it is even necessary is a MEASURE item — see
the streaming-loop stability instrumentation.
"""

from __future__ import annotations

from typing import Dict

import torch


class LifelongState:
    def __init__(self, rssm_state: Dict[str, torch.Tensor],
                 broadcaster=None, vision_scaffold=None,
                 state_decay: float = 0.995):
        self.rssm_state = rssm_state          # {"h": (N,H), "z": (N,Z)}
        self._broadcaster = broadcaster
        self._scaffold = vision_scaffold
        self.state_decay = float(state_decay)
        # cadence counters, in primary-stream step units
        self.seg_steps = 0
        self.wm_steps = 0
        self.goal_steps = 0

    def decay(self) -> None:
        """Per-SEGMENT soft decay so old state can neither spin up nor ossify.
        h is the meaningful lever (z is re-derived as the posterior every
        observe_step, so it only seeds the first GRU input)."""
        d = self.state_decay
        if d >= 1.0:
            return
        self.rssm_state["h"].mul_(d)
        if self._broadcaster is not None and hasattr(
                self._broadcaster, "decay_masks"):
            self._broadcaster.decay_masks(d)   # fade stale unlocks from the goal channel

    def death_reset(self, row: int) -> None:
        """Hard-zero ONE stream's carried rows on a true terminal / crash."""
        h = self.rssm_state.get("h")
        z = self.rssm_state.get("z")
        if h is not None and 0 <= row < h.shape[0]:
            h[row] = 0.0
        if z is not None and 0 <= row < z.shape[0]:
            z[row] = 0.0
        if self._broadcaster is not None and hasattr(
                self._broadcaster, "clear_stream"):
            self._broadcaster.clear_stream(row)
        # scaffold belief shapes ONLY the primary stream (0)
        if row == 0 and self._scaffold is not None and hasattr(
                self._scaffold, "reset"):
            self._scaffold.reset()

    # ---- read accessors for paging / viewer (LATER increment) ----
    @property
    def h(self) -> torch.Tensor:
        return self.rssm_state["h"]

    @property
    def z(self) -> torch.Tensor:
        return self.rssm_state["z"]

    def h_norm(self) -> float:
        """Mean ||h|| across streams. NOTE: rssm.py gru_norm (LayerNorm) pins
        this at ~sqrt(H) for ANY h, so it is a MAGNITUDE band check only and is
        BLIND to whether h actually evolves — use h_liveness() for that."""
        return float(self.rssm_state["h"].norm(dim=-1).mean().item())

    def h_liveness(self) -> float:
        """Cross-segment EVOLUTION signal: mean cosine DISTANCE between the
        carried h now and at the previous call (one call per segment). ~0 =>
        h is frozen across segments (the cross-segment carry is a no-op); a
        clearly positive value => the continuous state is genuinely evolving.
        This is the metric ||h|| could never provide (LayerNorm-pinned). NaN on
        the first call (no prior segment to compare)."""
        h = self.rssm_state["h"].detach()
        prev = getattr(self, "_h_prev_seg", None)
        self._h_prev_seg = h.clone()
        if prev is None or prev.shape != h.shape:
            return float("nan")
        cos = torch.nn.functional.cosine_similarity(h, prev, dim=-1)
        return float((1.0 - cos).mean().item())
