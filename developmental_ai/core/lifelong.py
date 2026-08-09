"""Lifelong controller — the machinery that makes the loop CONTINUOUS.

Everything the streaming (non-episodic) run needs, in one flag-gated place so
the episodic path is untouched when `lifelong.enabled` is false:

  * forever-mode termination: run until a stop-file appears or SIGTERM/SIGINT,
    not until a step budget (a lifelong agent has no natural end).
  * state carry + SOFT DECAY across segments: the RSSM recurrent state, the
    broadcaster achievement mask, and the vision-scaffold belief are carried
    between segments instead of reset — but decayed a little each segment so
    old state fades and cannot spin up or ossify over a multi-day life
    ("diminish older states over time", the user's requirement).
  * DEATH-ONLY hard reset: a true terminal (agent death) zeroes exactly that
    env's recurrent state + achievement mask row. On MineRL Treechop death may
    never fire — that is fine; the stream is then purely continuous.

Working-set SIZES (active_skills / active_goals) live here too as explicit,
raise-later variables: the bounded set the network holds "in mind" at once,
backed by the unbounded LongTermStore on disk.
"""

from __future__ import annotations

import logging
import os
import signal
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LifelongController:
    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.forever = bool(cfg.get("forever", False))
        self.stop_file = str(cfg.get("stop_file", "podlogs/STOP"))
        self.segment_steps = int(cfg.get("segment_steps", 1024))
        # per-segment soft leak of carried state (1.0 = no decay). Kept in
        # [0,1]; applied once per segment boundary, so within a segment the
        # RSSM updates normally and only the CARRY-OVER is gently forgotten.
        # 1.0 = NO decay (default). Measured 2026-07-19: the GRU's LayerNorm
        # (gru_norm) already bounds ||h|| perfectly flat over a continuous
        # stream, and the GRU forget gate already fades old context — so an
        # explicit decay is redundant, and decay<1 erodes ALL of h toward
        # zero (measured ||h|| = 22.5*decay^n). Leave at 1.0 unless a future
        # decay-toward-BASELINE design is added (never decay-to-zero).
        self.state_decay = float(cfg.get("state_decay", 1.0))
        self.reset_on_death_only = bool(cfg.get("reset_on_death_only", True))
        # WORKING-SET capacity — bounded active set, unbounded total on disk.
        # Explicit variables: raise later to upgrade the system.
        self.active_skills = int(cfg.get("active_skills", 24))
        self.active_goals = int(cfg.get("active_goals", 48))

        self._stop_requested = False
        # Handlers whenever lifelong is enabled, not only in forever-mode: a
        # budgeted run deserves the same graceful SIGTERM (finish the segment,
        # flush the brain) instead of dying mid-write. See should_stop().
        if self.enabled:
            self._install_signal_handlers()

    # ---- termination -------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        """Graceful stop on SIGTERM/SIGINT (main thread only). A lifelong run
        ends by request, never by step count."""
        def _handler(signum, _frame):
            logger.warning("lifelong: signal %d — requesting graceful stop",
                           signum)
            self._stop_requested = True
        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError) as e:
            # not the main thread (e.g. under a test harness) — non-fatal
            logger.info("lifelong: could not install signal handlers (%s)", e)

    def should_stop(self) -> bool:
        """True when a run should end early: stop-file present or a signal was
        received. Checked once per segment (cheap).

        Gated on `enabled`, NOT on `forever` (fixed 2026-07-25). Requiring
        forever-mode made the stop-file a no-op for every BUDGETED run — the
        mode actually in use — so `touch podlogs/STOP` did nothing and the only
        way to end a run was killing it mid-write. An operator asking a run to
        stop means the same thing whether or not a step budget exists.
        """
        if not self.enabled:
            return False
        if self._stop_requested:
            return True
        try:
            if self.stop_file and os.path.exists(self.stop_file):
                logger.warning("lifelong: stop-file %s present — stopping",
                               self.stop_file)
                return True
        except OSError:
            pass
        return False

    def budget(self, configured: Optional[int]) -> Optional[int]:
        """Resolve the step budget: None (run forever) in forever-mode, else
        the configured value. This is the None sentinel the loop honors."""
        if self.enabled and self.forever:
            return None
        return configured

    # ---- carried-state factory --------------------------------------------
    def make_state(self, rssm_state: Dict, broadcaster: Any = None,
                   scaffold: Any = None):
        """Build the per-stream LifelongState (owns the carried RSSM/mask/
        belief + decay + death-reset), wired with this controller's
        state_decay. The loop threads ONE object; paging + the brain-viewer
        read it through its accessors."""
        from developmental_ai.core.lifelong_state import LifelongState
        return LifelongState(rssm_state, broadcaster=broadcaster,
                             vision_scaffold=scaffold,
                             state_decay=self.state_decay)

    def stats(self) -> Dict:
        return {
            "enabled": self.enabled, "forever": self.forever,
            "segment_steps": self.segment_steps,
            "state_decay": self.state_decay,
            "active_skills": self.active_skills,
            "active_goals": self.active_goals,
            "stop_requested": self._stop_requested,
        }
