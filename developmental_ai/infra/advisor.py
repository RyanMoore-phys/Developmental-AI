"""Unstuck advisor (#46 — the answering half of help requests).

#51 gave the agent a voice: at stuck L3 it writes a structured help request
("no income, no new territory... consider relocating me").  Until now the
channel was write-only — the previous run stood at ONE coordinate for two
hours re-asking, and its most actionable self-diagnosis went into a file
nobody read until a human happened to.

This module closes the loop with the local VLM as the reader.  The request
(plus the agent's current view) goes to the same model that already teaches
its perception, and the answer comes back as GUIDANCE, not control: the
advisor picks one intervention from a bounded menu of existing, self-
restoring drive levers —

    look_around   re-open gaze novelty + refill the seek budget
    prime         socially prime ONE known category (curiosity injection,
                  the goal-emulation lever: "X is worth attending to";
                  the agent's own body still discovers the how)
    explore_wider double the exploration weights for a while (the L2
                  remedy, on request instead of on schedule)
    conserve      do nothing (a legitimate answer; not every zero-income
                  stretch is pathology)

Design constraints, in order:
  * SLIGHT guidance only.  Every remedy is a bias on drives the agent
    already has, expires on its own, and rides every existing safeguard
    (telescoping shaping, habituation, the farm detector, edge-triggered
    priming).  Nothing here moves the body or writes a reward.
  * The run NEVER depends on it.  No VLM, malformed JSON, hallucinated
    category, refused advice — all collapse to None and the run continues
    exactly as before this module existed.
  * Rate-limited by construction (`min_gap` steps between queries), because
    L3 can persist for many segments and one considered answer beats
    twenty identical ones (measured on the help-request emit side).
  * Every exchange is logged to `help_responses.jsonl` next to the
    requests, so the dialogue between the agent and its advisor is a
    readable artifact of the run.
"""

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

REMEDIES = ("look_around", "prime", "explore_wider", "conserve")

_PROMPT = """You are the caretaker of a young developmental AI playing \
Minecraft. It learns on its own from curiosity; you may only nudge WHAT it \
finds interesting, never control it. It just sent you this help request:

{situation}

Its known visual categories: {categories}
The attached image is its current first-person view (if present).

Choose exactly ONE gentle intervention:
- "look_around": reward scanning in new directions again (use when the view \
is monotonous — ground, sky, or a wall filling the frame — so it re-surveys \
its surroundings).
- "prime": make ONE category from the list temporarily fascinating (use when \
you can see or infer something worth its attention; set "category").
- "explore_wider": double its exploration drive for a while (use when the \
area looks exhausted and it should wander farther).
- "conserve": do nothing (use when the situation looks fine or transient).

Reply with ONLY this JSON:
{{"remedy": "<look_around|prime|explore_wider|conserve>", \
"category": "<one category or null>", \
"why": "<one short sentence>"}}"""


class UnstuckAdvisor:
    """Bridges stuck L3 help requests to a local VLM for slight guidance.

    `query_fn(prompt, images) -> Optional[str]` is INJECTED (the loop owns
    the ollama client and the model choice; infra stays dependency-free).
    `images` is a list of PNG byte strings or None.
    """

    def __init__(self, query_fn: Callable[[str, Optional[List[bytes]]],
                                          Optional[str]],
                 min_gap: int = 4096,
                 log_dir: str = "runlogs",
                 response_file: str = "help_responses.jsonl"):
        self._query = query_fn
        self.min_gap = int(min_gap)
        self.log_dir = str(log_dir)
        self.response_file = str(response_file)
        self._last_step = None      # step of the last accepted query
        self.asked = 0              # queries actually sent
        self.answered = 0           # answers that validated

    # ------------------------------------------------------------------ api
    def advise(self, situation: Dict[str, Any],
               categories: List[str],
               png: Optional[bytes],
               now_step: int) -> Optional[Dict[str, Any]]:
        """One considered answer to a help request, or None.

        None means "no guidance this time" for ANY reason — cooldown, no
        VLM, garbage output — and callers must treat it as a no-op, never
        an error state.
        """
        try:
            step = int(now_step)
            if (self._last_step is not None
                    and step - self._last_step < self.min_gap):
                return None
            self._last_step = step
            cats = [str(c) for c in (categories or [])]
            prompt = _PROMPT.format(
                situation=json.dumps(situation, default=str)[:2000],
                categories=", ".join(cats) if cats else "(none yet)")
            self.asked += 1
            raw = self._query(prompt, [png] if png else None)
            advice = self._parse(raw, cats)
            self._log(step, situation, raw, advice)
            if advice is not None:
                self.answered += 1
            return advice
        except Exception as e:
            logger.debug("UnstuckAdvisor.advise contained: %s", e)
            return None

    # ------------------------------------------------------------ internals
    @staticmethod
    def _parse(raw: Optional[str],
               categories: List[str]) -> Optional[Dict[str, Any]]:
        """Validate hard: the model gets a VOTE, not a syscall.

        Unknown remedy -> None.  "prime" with a category the agent does not
        actually know -> None (a hallucinated category must not mint a goal
        — the junk-goal factory taught that lesson five times).
        """
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        remedy = str(obj.get("remedy", "")).strip().lower()
        if remedy not in REMEDIES:
            return None
        cat = obj.get("category")
        cat = None if cat in (None, "", "null", "None") else str(cat)
        if remedy == "prime":
            if cat is None or cat not in categories:
                return None
        else:
            cat = None
        why = str(obj.get("why", ""))[:300]
        return {"remedy": remedy, "category": cat, "why": why}

    def _log(self, step: int, situation: Dict[str, Any],
             raw: Optional[str], advice: Optional[Dict[str, Any]]) -> None:
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, self.response_file)
            with open(path, "a") as f:
                f.write(json.dumps({
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "step": step,
                    "situation": situation.get("situation",
                                               situation.get("reason")),
                    "raw": (raw or "")[:1000],
                    "advice": advice,
                }, default=str) + "\n")
        except Exception:
            pass
