"""Curiosity-ranked vision magnet — INSTINCT for pixel environments.

July 2026 redesign (replaces the hardcoded tree-trunk magnet). The magnet no
longer has a VLM channel of its own and no fixed target: it consumes

  (a) the phase-2 GROUNDING HEAD's per-object probabilities (tree/stone/water/
      animal presence + generic object_left/right/centered/adjacent
      direction), which the symbolizer already computes every step, and
  (b) the primary stream's own LEARNING-PROGRESS reward (intrinsic[0]),

and steers the agent toward whatever it is currently most CURIOUS about. It
keeps a contrastive per-category LP-EMA — a category's score is
`cat_lp[c] - global_lp`, so a predicate that is present whenever ANYTHING is
learnable (sky/grass) cannot win by co-occurrence — retargets (with
hysteresis) to the highest-scoring in-view interactable, pulls toward it with
the same four shaping terms as before, and sets its OWN weight from that score
so it FADES as curiosity quenches and RE-ARMS when a novel object appears.

Design properties:
  * NO second VLM, NO render: objects come from the grounding head.
  * Reward-shaping only — never selects actions.
  * CONDITION-based fade (curiosity magnitude), not a timestep clock, so it is
    correct in a never-ending lifelong stream.
  * Curiosity memory (cat_lp/global_lp) SURVIVES reset — curiosity is not
    per-episode; only the per-episode approach-belief is cleared at a boundary.

Resolution ceiling (documented, not hidden): LP attribution is scene-level
per category, direction is the head's coarse 3-way, and phi proximity uses the
generic object_adjacent/centered predicates — no instance disambiguation of
two trees in one frame. See the deferred mechanism-C upgrade.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---- LAYER 1: the instinct's vocabulary is DERIVED, not listed -----------
# It used to be four hand-written strings. Everything around it worked —
# learning-progress scoring, retarget margins, dwell, cold-start pull toward an
# unengaged interactable, potential-based shaping — but it could only ever want
# FOUR things, while the symbolizer grounded 25. In any scene without a tree,
# stone, water or animal (an open inventory, a cave, a new world) the present
# set was EMPTY, no target was selected, and the instinct pulled toward nothing.
#
# Now the steerable set is computed from the live predicate vocabulary, so
# adding a predicate anywhere automatically gives the magnet something new it
# can want. Two exclusions, both principled rather than cosmetic:
#
#   BACKGROUND — always or near-always on screen. These would win the magnet by
#   CO-OCCURRENCE rather than by being interesting, which is the confound the
#   original four-item list existed to avoid. Keeping the exclusion explicit
#   (instead of implicit in a short list) is what lets the set grow safely.
#
#   NON-STEERABLE — the agent's own state or an affordance, not a thing out in
#   the world it can approach. You cannot walk toward `holding_tool`.
BACKGROUND_PREDICATES = frozenset({
    "sky_visible", "grass_visible", "dirt_visible", "leaves_visible",
})
NON_STEERABLE_PREDICATES = frozenset({
    "object_centered", "object_adjacent", "object_left", "object_right",
    "breakable_in_reach", "looking_up", "looking_down", "open_space_ahead",
    "holding_tool", "tool_worn",
    # GUI SCREENS ARE SELF-STATE, NOT PLACES (2026-08-07). These end in
    # `_visible` so layer-1 derivation made them steerable — and MEASURED
    # live, the magnet then courted the MENU: target=inventory_visible,
    # cold_spent[inventory]=55,889 floor-pulled steps, gui open 28% of the
    # run with one 25,923-step dwell. "Approaching" a screen overlay means
    # opening it and sitting there — the menu-dwell pathology, this time
    # caused by the instinct itself. You cannot walk toward your own
    # inventory any more than toward `holding_tool`.
    "inventory_visible", "crafting_grid_visible", "craft_output_visible",
})


def steerable_targets(predicates) -> List[str]:
    """Which predicates the instinct may steer toward, derived from vocabulary.

    A predicate is steerable when it names a THING that can be present in the
    world (the `_visible` convention) and is neither background nor a
    self-state. This is deliberately a RULE rather than a list: the whole point
    of layer 1 is that a new predicate becomes wantable without anyone
    remembering to edit the magnet.
    """
    out = []
    for p in predicates or ():
        if not str(p).endswith("_visible"):
            continue
        if p in BACKGROUND_PREDICATES or p in NON_STEERABLE_PREDICATES:
            continue
        out.append(str(p))
    return out


def _default_targets() -> List[str]:
    """Steerable set from the symbolizer's live vocabulary, with a fallback.

    Imported lazily so this module keeps working if the symbolizer is absent
    (tests, lesioned runs) — falling back to the historical four rather than
    to nothing, because an empty target set silently disables the instinct.
    """
    try:
        from developmental_ai.llm.vlm_symbolizer import PREDICATES
        derived = steerable_targets(PREDICATES)
        if derived:
            return derived
    except Exception:            # never let a bad import disable instinct
        pass
    return ["tree_visible", "stone_visible", "water_visible", "animal_visible"]


DEFAULT_TARGETS = _default_targets()


class VisionScaffold:
    def __init__(
        self,
        weight: float = 0.15,
        min_weight: float = 0.0,
        target_categories: Optional[List[str]] = None,
        # LAYER 3: bounded probation for VLM-proposed categories.
        # max_proposed caps memory AND caps how much of the instinct's
        # attention unproven guesses can occupy at once.
        max_proposed: int = 8,
        # probation counts VLM SIGHTINGS, not steps (labels land every
        # ~200-300 steps, so 6 sightings is already ~1500+ steps of evidence)
        probation_attends: int = 6,
        # how long after a VLM sighting a proposal counts as "present". This
        # IS the proposal's epistemic status: the VLM is its only sensor.
        proposal_ttl: int = 400,
        # COLD-START instinct: a small floor pull toward a trusted-present
        # interactable the agent has NOT yet engaged (never distinctively
        # curious), so the magnet bootstraps interaction BEFORE learning
        # progress can differentiate across objects. Without it the contrastive
        # score is 0 at cold start (nothing stands out), w stays 0, and the
        # agent is never steered toward the first log-break — a chicken-and-egg
        # deadlock. Fades to 0 for a category the instant it becomes curious
        # (contrastive takes over) or is later mastered.
        cold_start_weight: float = 0.10,
        # COLD-START BUDGET (audit fix): the floor above never faded for a
        # category that never becomes distinctively curious (e.g. static stone
        # yields below-average LP forever), so "park at stone" paid a standing
        # income that out-competed the seek drive 20-40:1. The floor now WANES:
        # each category gets this many floor-pulled steps to bootstrap an
        # interaction; after that, an un-engaged category's floor drops to 0
        # (contrastive engagement is untouched and can still take over).
        cold_start_budget: int = 4000,
        # TRUNK FOCUS: categories that, if more prominent than the target,
        # suppress the chop instinct — so the reward pays for chopping the LOG,
        # not the leaves/grass surrounding it (default foliage).
        focus_distractors: Optional[List[str]] = None,
        # attribution
        ema_beta: float = 0.01,        # per-category / global LP EMA rate
        ema_leak: float = 0.005,       # decay of a category's EMA while absent
        # trust gate (mirrors the symbolizer's perceptual_facts gate)
        present_threshold: float = 0.6,
        reliability_floor: float = 0.35,
        min_labels: int = 5,
        # retarget hysteresis
        retarget_margin: float = 0.20,
        min_dwell: int = 25,
        absent_tolerance: int = 3,     # steps the incumbent may flicker out of
                                       # view before it is released (MF-2: raw
                                       # per-step head sigmoids cross the
                                       # present_threshold on latent noise)
        # fade
        eps_abs: float = 1.0e-3,       # absolute LP floor for hard ON/OFF
        scale_decay: float = 0.999,    # running curiosity-scale decay
        # pull geometry (unchanged action sets + bonuses)
        instinct_bonus: float = 0.05,
        approach_pull: float = 0.0,
        align_bonus: float = 0.0,
        aim_bonus: float = 0.0,
        # GOAL-SEEKING / FRONTIER drive: when a designated goal object (e.g. a
        # tree) is NOT in view, approach/align/aim are inert (nothing to steer
        # toward) and in a lifelong stream the world never resets to re-place
        # the agent among trees. seek_weight is a POTENTIAL on the goal object's
        # visibility (rewards actions that bring it into frame; telescopes to 0
        # over any look-away/look-back cycle, so it cannot be farmed).
        # seek_forward_nudge is a small, BUDGETED forward reward that breaks the
        # agent out of parking on a non-goal object while the goal is entirely
        # off-screen. Both ride the reward mixer's anneal downstream — neither
        # hard-fixes the intrinsic/extrinsic ratio. Off when both are 0.
        seek_weight: float = 0.0,
        seek_categories: Optional[List[str]] = None,
        seek_forward_nudge: float = 0.0,
        seek_nudge_budget: int = 200,
        # NUDGE REGENERATION (2026-08-07). The budget refilled ONLY on a goal
        # sighting — but the nudge exists precisely for when the goal is
        # never sighted, so a bot that burned its budget without finding a
        # tree lost its search drive PERMANENTLY (measured: nudge_left=0 for
        # 10h+, parked in a self-dug pit). Guard-becomes-latch again. One
        # nudge trickles back every this-many steps (0 = off), so search can
        # starve but never die; sighting-refill still restores it instantly.
        seek_nudge_regen: int = 0,
        chop_actions: Optional[List[int]] = None,
        forward_actions: Optional[List[int]] = None,
        turn_left_actions: Optional[List[int]] = None,
        turn_right_actions: Optional[List[int]] = None,
        pitch_actions: Optional[List[int]] = None,
        enabled: bool = True,
        contrast_vs_peers: bool = False,
        phi_from_evidence: bool = False,
        # GAZE RIGHTING (2026-08-06): a levelness factor on the seek
        # potential. MEASURED: the agent spent 47.7% of a segment with its
        # pitch pinned at the sky clamp (mean -76deg) — a state from which no
        # terrestrial goal can ever enter the fovea — and nothing in the
        # reward paid for coming back to the horizon. Humans get this for
        # free (vestibular righting; infants orient to eye level before they
        # orient to objects). Folded INTO the seek potential (a bounded state
        # function of pitch), so it telescopes: dwelling level earns 0, only
        # RETURNING to level pays, and it scales with goal visibility so it
        # can never become a generic "stare at the horizon" income.
        seek_pitch_level: bool = False,
        # STALE-LP DISCOUNT (2026-08-07). ema_leak is deliberately 0, so an
        # absent category's LP FREEZES — correct as an anti-latch, but the
        # frozen value is EVIDENCE that ages. MEASURED live: water/zombie/
        # sheep held 0.38-0.43 for 10h+ from single early glimpses while the
        # global sat at 0.03, so any ghost re-entering the frame would open
        # with a spurious ~0.37 contrastive score and grab the target with
        # zero fresh evidence. On the absent->present transition the stored
        # EMA is decayed toward the CURRENT global with this time constant
        # (steps of absence): recently-absent keeps its standing, long-absent
        # re-opens neutral and must re-earn curiosity. 0 = off (exact old
        # behaviour); never decays toward zero, so the anti-latch survives.
        lp_stale_tau: float = 0.0,
    ):
        self.enabled = bool(enabled)
        # see _score: contrast against peer categories instead of a global
        # that includes this category. False = the historic behaviour.
        self.contrast_vs_peers = bool(contrast_vs_peers)
        # ---- SPATIAL PREDICATES ARE NOT LEARNABLE FROM CAPTIONS ----------
        # `object_centered` and `object_adjacent` are VLM-taught, and llava
        # cannot judge "is something under the crosshair" from a still frame.
        # The head faithfully learns "nothing is ever centred": measured
        # Centring Phi 0.003 and Reach 2.0% across a whole run, while the
        # head agreed with the VLM 96% of the time. It is not a training
        # failure — the label itself carries no such information.
        # With this on, phi is built from MEASURED CONTACT instead: a
        # completed block break is ground truth that something was both
        # centred and within range. Same reasoning as the reach sense.
        self.phi_from_evidence = bool(phi_from_evidence)
        self._reach_measured = 0.0
        self.seek_pitch_level = bool(seek_pitch_level)
        # ---- FOVEA (2026-08-06): "am I looking AT the target?" ----------
        # Per-step P(category under the crosshair) from the symbolizer's
        # fovea head — the pose-dependent signal every potential here was
        # missing. phi's centring channel and the seek orientation factor
        # both read it; presence/ranking stay on the full-frame head. A
        # category is only USED once its foveal label count clears
        # min_labels (no evidence, no assertion — same rule as everywhere).
        self._fovea_probs: Dict[str, float] = {}
        self._fovea_counts: Dict[str, int] = {}
        self._pitch: Optional[float] = None
        self.lp_stale_tau = float(lp_stale_tau)
        # consecutive steps each category has been ABSENT (stale-LP ageing)
        self._cat_absent: Dict[str, int] = {}
        self.weight = float(weight)
        self.min_weight = float(min_weight)
        self.cold_start_weight = float(cold_start_weight)
        self.cold_start_budget = int(cold_start_budget)
        self.focus_distractors = list(
            focus_distractors if focus_distractors is not None
            else ["leaves_visible", "grass_visible"])
        self.target_categories = list(target_categories or DEFAULT_TARGETS)
        self.ema_beta = float(ema_beta)
        self.ema_leak = float(ema_leak)
        self.present_threshold = float(present_threshold)
        self.reliability_floor = float(reliability_floor)
        self.min_labels = int(min_labels)
        self.retarget_margin = float(retarget_margin)
        self.min_dwell = int(min_dwell)
        self.absent_tolerance = int(absent_tolerance)
        self.eps_abs = float(eps_abs)
        self.scale_decay = float(scale_decay)
        self.instinct_bonus = float(instinct_bonus)
        self.approach_pull = float(approach_pull)
        self.align_bonus = float(align_bonus)
        self.aim_bonus = float(aim_bonus)
        # PURE-POTENTIAL MODE (2026-07-25): all four raw per-step income terms
        # are zero, so the ONLY shaping left telescopes -> a parked agent earns
        # exactly 0 and the floor cannot be farmed. That is what the
        # `not _ever_curious` guard on the cold-start floor was protecting
        # against ("park at stone" standing income), so in this mode the guard
        # is void — and keeping it would re-latch w to 0 forever the moment a
        # category's contrastive score fades AFTER it was once curious (the
        # second door onto the 19h zero-reward stall). Configs that still use
        # raw incomes keep the original strict behaviour.
        self._pure_potential = (self.instinct_bonus == 0.0
                                and self.approach_pull == 0.0
                                and self.align_bonus == 0.0
                                and self.aim_bonus == 0.0)
        self.seek_weight = float(seek_weight)
        # default seek target = the highest-priority interactable (tree first
        # for Treechop, per target_categories order)
        self._seek_cats = list(seek_categories) if seek_categories else (
            [self.target_categories[0]] if self.target_categories else [])
        self.seek_forward_nudge = float(seek_forward_nudge)
        self.seek_nudge_budget = int(seek_nudge_budget)
        self.seek_nudge_regen = int(seek_nudge_regen)
        self._nudge_regen_clock = 0
        self.chop_actions = set(chop_actions if chop_actions is not None
                                else [5, 6])
        self.forward_actions = set(forward_actions if forward_actions
                                   is not None else [1, 2, 6])
        self.turn_left_actions = set(turn_left_actions if turn_left_actions
                                     is not None else [3])
        self.turn_right_actions = set(turn_right_actions if turn_right_actions
                                      is not None else [4])
        self.pitch_actions = set(pitch_actions if pitch_actions is not None
                                 else [7, 8])
        # Nudge actions = PURE forward only (audit fix): action 6 is
        # attack+forward, so the anti-parking nudge was paying the parked
        # attack-at-stone action itself. The nudge must never carry a chop.
        self._nudge_actions = self.forward_actions - self.chop_actions

        # ---- LAYER 3: INSTINCT IN AN UNFAMILIAR PLACE -------------------
        # Layers 1-2 widened a hand-written vocabulary. That still hits a wall
        # the first time the agent stands somewhere nobody anticipated — a new
        # world, a cave, a menu, someone else's server. Instinct has to be able
        # to want something it was never told about.
        #
        # So the VLM may PROPOSE a category for something salient it has no
        # predicate for. A proposal is a HYPOTHESIS ABOUT WHERE TO LOOK, never
        # a claim about what is true:
        #   * it enters PROBATION and is steerable like any other category;
        #   * presence comes from the VLM's own label, because a proposal has
        #     no grounded head yet (base predicates do — that is the
        #     difference between a guess and a learned percept);
        #   * it is scored by the SAME contrastive learning-progress rule, so
        #     attending to it has to actually teach the agent something;
        #   * if it yields no learning progress after `probation_attends`
        #     sightings it is EVICTED and not re-proposed.
        # A proposed category NEVER becomes a knowledge-graph fact. The LLM
        # directs attention; experience decides what is real. That is what
        # keeps this instinct rather than an oracle, and what stops a
        # hallucinated category from turning into a belief.
        self.max_proposed = int(max_proposed)
        self.probation_attends = int(probation_attends)
        self.proposal_ttl = int(proposal_ttl)
        self._now = 0        # stamped each step_shaping; TTL presence
        self._proposed: Dict[str, Dict[str, float]] = {}
        self._evicted: Set[str] = set()
        # MID-RUN EXPANSION state. Layer 1 derived the steerable set from the
        # vocabulary — but at IMPORT and CONSTRUCTION, which is still STATIC.
        # It is right for the world the agent booted into and stale the moment
        # the vocabulary grows (a proposal graduating, a new environment
        # registering predicates, a server with things this world lacks). An
        # instinct extendable only by restart stops working exactly when the
        # agent walks somewhere new — which is when instinct matters most.
        # `_vocab_seen` makes the refresh O(1) when nothing changed; pinned
        # means the caller passed an explicit list and must not be overwritten.
        self._vocab_seen = 0
        self._targets_pinned = target_categories is not None

        # ---- LIFELONG curiosity memory (survives reset) ----
        self._cat_lp: Dict[str, float] = {c: 0.0
                                          for c in self.target_categories}
        self._global_lp = 0.0
        # PER-CATEGORY curiosity scale (MF-1): each category's weight is
        # normalized by ITS OWN recent peak, so an early high-LP object's peak
        # can no longer cross-scale-suppress a later, lower-magnitude novel
        # object (which a single shared global running-max did for ~2k steps).
        self._lp_scale: Dict[str, float] = {}
        # A category becomes "engaged" the first time it is distinctively
        # curious (contrastive score clears eps_abs). Until then the cold-start
        # instinct floor pulls toward it; after, the contrastive weight governs
        # and (once mastered) fades. Lifelong memory — survives reset.
        self._ever_curious: Dict[str, bool] = {}
        # steps each category has consumed of its cold-start floor budget
        # (lifelong memory — an instinct that waned stays waned)
        self._cold_spent: Dict[str, int] = {}
        self._w = 0.0                      # cached curiosity-driven weight
        # ---- per-episode belief (cleared on reset) ----
        self._target: Optional[str] = None
        self._target_since = 0
        self._target_absent = 0            # consecutive steps incumbent absent
        self._target_just_switched = False
        self._phi_prev: Optional[float] = None
        # goal-seeking belief (per-episode): last goal-visibility prob (for the
        # seek potential) + remaining forward-search nudge budget + consecutive
        # goal-sighting streak (weaker refill condition than trusted-present)
        self._seek_prob_prev: Optional[float] = None
        self._seek_nudge_left = self.seek_nudge_budget
        self._seek_sight_streak = 0
        self._closed = False
        # introspection
        self.assessment_log: deque = deque(maxlen=64)

    def reset(self) -> None:
        """Clear ONLY per-episode approach-belief. Curiosity memory
        (cat_lp/global_lp/lp_scale/w) is deliberately PRESERVED — the world
        continues across episode/dream/segment boundaries, so what the agent
        is curious about is not per-episode."""
        self._target = None
        self._target_since = 0
        self._target_absent = 0
        self._target_just_switched = False
        self._phi_prev = None              # next step adopts phi w/o a delta
        self._seek_prob_prev = None        # seek potential re-adopts w/o a delta
        self._seek_nudge_left = self.seek_nudge_budget   # fresh search budget
        self._seek_sight_streak = 0
        # NOTE: _cold_spent is deliberately NOT cleared — a waned instinct
        # stays waned across episode/death boundaries (lifelong memory).

    def current_weight(self, timestep: int = 0) -> float:
        """CONDITION-based now: the cached curiosity-driven weight; the
        timestep arg is ignored (kept for call-site compatibility)."""
        return self._w

    def wants_step(self, timestep: int = 0) -> bool:
        """Gate on enabled ONLY, never on _w>0: step_shaping must keep running
        while w==0 so the EMAs can detect a NEW novel object and re-arm. Cheap
        now (dict math; no VLM, no render)."""
        return self.enabled and not self._closed

    def _score(self, c: str) -> float:
        """How distinctively curious is this category vs the alternatives?

        THE ORIGINAL FORM CANNOT WORK FOR AN ALWAYS-PRESENT CATEGORY.
        `cat_lp[c]` is EMA'd with the same beta and the same lp as
        `global_lp` on every step c is present — so if c is present on EVERY
        step the two series are identical by construction and the score is
        exactly 0, forever. `tree_visible` is precisely that case: the
        grounding head reports it present essentially always (it read present
        while the agent stood in unrenderable void 65 blocks underground).
        MEASURED over a full run: 144 of 144 segments pinned at the cold-start
        floor w=0.3500 with cold_spent[tree]=144827 — the curiosity-ranked
        branch never fired once. The magnet was a constant pull, not guidance.

        With `contrast_vs_peers`, a category is compared against the MEAN OF
        THE OTHER live targets rather than a global that includes itself, so
        an always-present category can still be ranked. Off by default: the
        original semantics are what every other config expects.
        """
        _cur = self._cat_lp.get(c, 0.0)
        if not getattr(self, "contrast_vs_peers", False):
            return max(0.0, _cur - self._global_lp)
        # PEERS MUST BE CATEGORIES IT CAN ACTUALLY SEE (fix 2026-08-06).
        # ema_leak is 0.0 by deliberate design, so an ABSENT category's LP
        # FREEZES at whatever it last held. Categories glimpsed once early —
        # mobs, water — therefore keep stale HIGH values forever, while an
        # always-present category like tree_visible keeps tracking the
        # (declining) global. Comparing against all peers then ranks the
        # stale ghosts above the live target:
        #     tree=0.201 (== global) vs frozen peers averaging ~0.29
        #     -> score negative -> clamped to 0 -> back to the cold-start
        #        floor, with the magnet paying for ROTATION instead.
        # Restricting the peer set to what is present right now asks the
        # question that actually matters: of the things I can see, which am I
        # most curious about? Falls back to the global when nothing else is
        # in view, which is the single-target case.
        _seen = getattr(self, "_present_now", None) or set()
        _peers = [self._cat_lp.get(o, 0.0)
                  for o in self.live_targets() if o != c and o in _seen]
        _ref = (sum(_peers) / len(_peers)) if _peers else self._global_lp
        return max(0.0, _cur - _ref)

    def live_targets(self) -> List[str]:
        """Base steerable set PLUS categories still on probation."""
        return list(self.target_categories) + [
            n for n in self._proposed if n not in self._evicted]

    def refresh_targets(self) -> int:
        """Re-derive the steerable set from the LIVE vocabulary, MID-RUN.

        This is what makes layer 1 dynamic rather than merely derived. Cheap
        enough to call every step: it early-outs when the vocabulary size is
        unchanged, which is the common path. Returns how many categories
        became newly steerable.
        """
        if self._targets_pinned:      # an explicit config wins over derivation
            return 0
        try:
            from developmental_ai.llm.vlm_symbolizer import PREDICATES
        except Exception:
            return 0
        if len(PREDICATES) == self._vocab_seen:
            return 0
        self._vocab_seen = len(PREDICATES)
        added = [c for c in steerable_targets(PREDICATES)
                 if c not in self.target_categories]
        for c in added:
            self.target_categories.append(c)
            # EVERY per-category dict must be seeded. Miss one and the new
            # target reads as "already known and uninteresting", so it can
            # never win the magnet — present but permanently unwantable.
            self._cat_lp.setdefault(c, 0.0)
            self._lp_scale.setdefault(c, 0.0)
            self._ever_curious.setdefault(c, False)
            self._cold_spent.setdefault(c, 0)
        if added:
            logger.info("instinct: vocabulary grew MID-RUN, +%d steerable "
                        "%s (now %d)", len(added), added,
                        len(self.target_categories))
        return len(added)

    def graduate(self, name: str) -> bool:
        """Promote a proven proposal into the permanent steerable set.

        A proposal that keeps earning learning progress has stopped being a
        guess. Graduating moves it out of probation so it is no longer subject
        to eviction — the instinct's vocabulary GREW from experience rather
        than from someone editing a list.
        """
        if name not in self._proposed or name in self.target_categories:
            return False
        self._proposed.pop(name, None)
        self.target_categories.append(name)
        self._cat_lp.setdefault(name, 0.0)
        self._lp_scale.setdefault(name, 0.0)
        self._ever_curious.setdefault(name, False)
        self._cold_spent.setdefault(name, 0)
        logger.info("instinct: GRADUATED %r — a proposed category earned a "
                    "permanent place by repeatedly teaching the agent", name)
        return True

    def propose_category(self, name: str, timestep: int = 0) -> bool:
        """VLM proposes a salient thing it has no predicate for. Attention only.

        Returns True if newly admitted. Refuses silently when the name is
        already known, was previously evicted (a failed hypothesis does not get
        to keep costing attention), or the probation set is full.
        """
        n = str(name or "").strip().lower().replace(" ", "_")
        if not n or not n.replace("_", "").isalnum():
            return False
        if n in self.target_categories or n in self._evicted:
            return False
        if n in self._proposed:
            # RE-SIGHTING — this is the proposal's sensor. The grounded head
            # is fixed to the base predicates, so a proposal can NEVER appear
            # in per-step probs; the first cut read probs.get(name) there and
            # was therefore a silent no-op in production (attends never moved,
            # nothing was ever present, evicted, or steered toward — while
            # every unit test passed on hand-built probs dicts). The VLM
            # naming the thing again is the only observation that exists, so
            # it is the observation we count.
            self._proposed[n]["attends"] += 1.0
            self._proposed[n]["last_seen"] = float(timestep)
            return False
        if len(self._proposed) >= self.max_proposed:
            return False
        self._proposed[n] = {"attends": 1.0, "first_t": float(timestep),
                             "last_seen": float(timestep)}
        self._cat_lp.setdefault(n, 0.0)
        logger.info("instinct: PROPOSED new category %r (probation %d sightings)",
                    n, self.probation_attends)
        return True

    def _review_probation(self) -> None:
        """Evict a proposal that never paid: seen enough, taught nothing.

        The bar is the SAME contrastive score every category faces — a
        proposal must beat the global learning rate, not merely be present.
        """
        for n, d in list(self._proposed.items()):
            if d.get("passed"):
                continue        # already earned its place — see below
            if d["attends"] < self.probation_attends:
                continue
            if self._score(n) < self.eps_abs:
                self._proposed.pop(n, None)
                self._evicted.add(n)
                self._cat_lp.pop(n, None)
                logger.info(
                    "instinct: EVICTED proposed category %r after %d sightings "
                    "(no learning progress — hypothesis did not pay)",
                    n, int(d["attends"]))
            else:
                # ONE-SHOT exam, then done. Re-reviewing every step would
                # permanently bar any proposal the moment its curiosity
                # naturally QUENCHED — punishing exactly the trajectory a
                # successful hypothesis follows (novel -> learned -> boring).
                # Passing means: at the moment it had accumulated enough
                # sightings, attending to it was genuinely teaching. That is
                # the same bar a goal faces, judged once, like an exam.
                d["passed"] = True
                logger.info(
                    "instinct: %r PASSED probation after %d sightings "
                    "(contrastive LP %.4f) — permanently steerable",
                    n, int(d["attends"]), self._score(n))

    def _trusted_present(self, probs, reliability, label_counts) -> Set[str]:
        out: Set[str] = set()
        for c in self.target_categories:
            if (label_counts.get(c, 0) >= self.min_labels
                    and reliability.get(c, 0.0) >= self.reliability_floor
                    and float(probs.get(c, 0.0)) >= self.present_threshold):
                out.add(c)
        # PROPOSED categories have no grounded head — they can never appear
        # in `probs` (the head's output is fixed to the base predicates), so
        # presence is RECENCY OF SIGHTING: the VLM named it within the TTL.
        # That is the proposal's true epistemic status — "the only sensor I
        # have said this was here moments ago" — not a fabricated probability.
        for n, d in self._proposed.items():
            if n in self._evicted:
                continue
            if (self._now - float(d.get("last_seen", -1e18))
                    <= self.proposal_ttl):
                out.add(n)
        self._review_probation()
        return out

    def _select_target(self, present: Set[str], timestep: int) -> None:
        ranked = sorted(present, key=self._score, reverse=True)
        # eps_abs consistently (audit fix): "distinctively curious" is ONE bar.
        # A sub-epsilon score residue used to make best=stone, blocking the
        # cold-start handoff to a virgin tree and preempting via a numerically
        # meaningless margin (score > 0 here vs >= eps_abs at engagement).
        best = (ranked[0] if ranked and self._score(ranked[0]) >= self.eps_abs
                else None)
        # incumbent momentarily out of view -> TOLERATE a brief flicker before
        # releasing (MF-2). Raw per-step head sigmoids cross present_threshold
        # on latent noise; releasing on a single-step dip let a lower-LP
        # category grab the target and min_dwell then locked it there.
        if self._target is not None and self._target not in present:
            self._target_absent += 1
            if self._target_absent < self.absent_tolerance:
                # Tolerate the flicker, BUT still let a genuinely more-curious
                # PRESENT object preempt (past dwell + margin) — else an
                # incumbent visible only on alternate frames could permanently
                # lock out a challenger seen only on the incumbent-absent
                # frames. A weak/noise challenger (score not beating the
                # incumbent by the margin) cannot preempt, so single-step
                # flicker thrash stays fixed.
                if (best is not None and best != self._target
                        and (timestep - self._target_since) >= self.min_dwell
                        and self._score(best) > self._score(self._target)
                        * (1.0 + self.retarget_margin)):
                    self._target = best
                    self._target_since = timestep
                    self._target_absent = 0
                    self._target_just_switched = True
                return                     # otherwise keep the incumbent
            self._target = None            # genuinely gone -> release
            self._target_absent = 0
        elif self._target is not None:
            self._target_absent = 0        # back in view -> reset the counter
        if self._target is None:
            # contrastive best, else a cold-start bootstrap target
            chosen = best if best is not None else self._cold_start_candidate(
                present)
            if chosen is not None:
                self._target = chosen
                self._target_since = timestep
                self._target_absent = 0
                self._target_just_switched = True
            return
        # incumbent still present: hand off only past margin AND min dwell
        if best is not None and best != self._target:
            dwell_ok = (timestep - self._target_since) >= self.min_dwell
            if dwell_ok and self._score(best) > self._score(
                    self._target) * (1.0 + self.retarget_margin):
                self._target = best
                self._target_since = timestep
                self._target_absent = 0
                self._target_just_switched = True
        elif best is None and (self._pure_potential
                               or not self._ever_curious.get(
                                   self._target, False)):
            # incumbent is an UN-ENGAGED cold-start target and contrastive is
            # silent: prefer a higher-priority un-engaged interactable if one is
            # present (e.g. switch stone -> tree when tree comes into view). An
            # ENGAGED incumbent is NOT disturbed here — it fades or hands off
            # via the contrastive path, so this never thrashes a real target.
            cold = self._cold_start_candidate(present)
            if cold is not None and cold != self._target:
                self._target = cold
                self._target_since = timestep
                self._target_absent = 0
                self._target_just_switched = True

    def _cold_start_candidate(self, present: Set[str]) -> Optional[str]:
        """Highest-priority present interactable the agent has NOT yet engaged
        (never distinctively curious) — the cold-start bootstrap target.
        target_categories order is the priority (tree first for Treechop).
        A category whose floor BUDGET is spent no longer qualifies (its
        instinct waned), so the handoff falls through to the next candidate."""
        for c in self.target_categories:
            # THIRD LATCH DOOR (fixed 2026-07-25): in pure-potential mode the
            # `_ever_curious` exclusion must be lifted HERE too, not only on
            # the weight branch. Otherwise every present category eventually
            # becomes ever-curious, this returns None, _select_target leaves
            # target=None, and BOTH weight branches fail (the floor requires a
            # target) -> w=0.0 forever. That is the 19h zero-reward stall
            # arriving through target SELECTION instead of the weight.
            if (c in present
                    and (self._pure_potential
                         or not self._ever_curious.get(c, False))
                    and self._cold_spent.get(c, 0) < self.cold_start_budget):
                return c
        return None

    def _fovea_p(self, cat: Optional[str]) -> Optional[float]:
        """Gated foveal probability for a category, or None when the fovea
        has no earned evidence for it (untrained head ~= noise around 0.5,
        which must never steer)."""
        if not cat or not self._fovea_probs:
            return None
        if self._fovea_counts.get(cat, 0) < self.min_labels:
            return None
        p = self._fovea_probs.get(cat)
        return None if p is None else max(0.0, min(1.0, float(p)))

    def _pitch_level(self) -> float:
        """Levelness of the gaze in (0,1]: 1.0 at the horizon, 0.65 at a
        vertical clamp. Quadratic so small scanning tilts cost ~nothing."""
        if not self.seek_pitch_level or self._pitch is None:
            return 1.0
        t = min(90.0, abs(float(self._pitch))) / 90.0
        return 1.0 - 0.35 * t * t

    def _phi_head(self, probs) -> float:
        """Proximity ladder, WEIGHTED BY WHETHER THE THING IN FRONT IS THE
        TARGET (2026-07-25).

        The ladder used to read GENERIC affordance predicates only
        (object_centered / object_adjacent), so it said "you are close to
        something" without caring WHAT. Trunk-vs-leaf discrimination lived
        solely in `instinct_bonus`, which had to be zeroed because it was raw
        per-step income and farmable 16:1 against a real log. With it gone the
        agent correctly walked to a tree and then chopped the nearest LEAF —
        live: dirt/fern/grass/birch_leaves broken, almost no logs.

        Folding target-focus INTO the potential fixes that without
        reintroducing income: phi is a bounded state function, so the shaping
        still telescopes (any loop nets 0, parking earns 0) — it just now
        peaks on "adjacent to the TARGET" instead of "adjacent to anything".
        """
        if self.phi_from_evidence:
            # ground truth: recent break => something was centred AND in range
            adj = cen = float(self._reach_measured)
            # FOVEA restores the PRE-contact centring gradient that pure
            # break-evidence cannot supply (a break only proves centring
            # AFTER the fact — walking up to a trunk paid nothing). The
            # foveal P(target) rises continuously as the target fills the
            # centre of the view, which is exactly the missing slope.
            _fp = self._fovea_p(self._target)
            if _fp is not None:
                cen = max(cen, _fp)
        else:
            adj = float(probs.get("object_adjacent", 0.0))
            cen = float(probs.get("object_centered", 0.0))
        # ---- CONTINUOUS, NOT A THRESHOLD LADDER (2026-08-02) -------------
        # The ladder was `0.85 if adj>=0.5 else (0.6 if cen>=0.5 else 0.35)`.
        # As a POTENTIAL that is nearly useless for aiming: swinging the view
        # so the target goes from p(centered)=0.10 to 0.49 paid EXACTLY ZERO,
        # and 0.49 -> 0.51 paid the whole step. There was no gradient to
        # follow, so "turn toward the thing you are curious about" was not a
        # learnable direction — which is why the agent sat with its pitch
        # clamped at +90 while the symbol drive (position-blind and
        # count-decayed to ~0.01/step) could not pull it back.
        #
        # Linear in the head's own probabilities, so EVERY degree of
        # re-centring pays a little. Coefficients keep the old landmark
        # values so existing tuning still means what it meant:
        #   nothing (0,0) -> 0.35   centred (1,0) -> 0.60
        #   both    (1,1) -> 1.00
        # Still a bounded state function => Ng's theorem still applies: any
        # look-away/look-back cycle telescopes to ~0 and parking earns 0.
        phi = 0.35 + 0.25 * max(0.0, min(1.0, cen)) \
                   + 0.40 * max(0.0, min(1.0, adj))
        phi = max(0.0, min(1.0, phi))
        # target-focus factor: how much the thing in view IS the target rather
        # than one of its distractors (leaves/grass around a trunk).
        tgt = self._target
        if tgt is not None:
            # FOVEAL focus when grounded: "is the thing under my crosshair
            # the TRUNK or its leaves?" asked of the crop, which is the
            # question the full-frame probs could never answer (both are
            # trivially somewhere in a forest frame).
            _ftp = self._fovea_p(tgt)
            if _ftp is not None:
                tgt_p = _ftp
                dis_p = max((self._fovea_p(d) or float(probs.get(d, 0.0))
                             for d in self.focus_distractors), default=0.0)
            else:
                tgt_p = float(probs.get(tgt, 0.0))
                dis_p = max((float(probs.get(d, 0.0))
                             for d in self.focus_distractors), default=0.0)
            # SMOOTH for the same reason as above: a hard 1.0/0.5 step meant a
            # tiny change in the head's confidence halved or doubled the
            # potential, injecting a spurious +-0.3 shaping spike that had
            # nothing to do with the agent moving. Ramps between the same two
            # endpoints (0.5 when a distractor dominates, 1.0 when the target
            # does), so the floor and ceiling are unchanged.
            _margin = tgt_p - max(float(self.present_threshold), dis_p)
            focus = 0.5 + 0.5 * max(0.0, min(1.0, 0.5 + _margin / 0.4))
            phi *= focus
        return float(phi)

    def _seek_shaping(self, action: int, probs: Dict[str, float],
                      reliability: Dict[str, float],
                      label_counts: Dict[str, int], present: Set[str]) -> float:
        """Goal-SEEKING drive: a potential on how well the view is ORIENTED
        toward the designated goal object.

        NO LONGER SWITCHES OFF WHEN THE GOAL COMES INTO VIEW (2026-08-02).
        It used to `return 0.0` the instant the goal was trusted-present,
        handing over to the approach potential — which only ran when the
        magnet weight was non-zero AND the target was present. "Goal visible
        but off-centre", the exact state that needs a turn, fell into the gap
        between them and nothing paid for closing it. Combined with an
        approach potential that was a 0.5 THRESHOLD rather than a gradient,
        re-centring the view was unrewarded at every offset that mattered.

        The scalar is now `gp * (0.6 + 0.4 * centred)`:
          * goal not in frame        -> 0     (nothing to centre on)
          * goal in frame, off-centre-> 0.6 x visibility
          * goal in frame, centred   -> 1.0 x visibility
        Continuous through the trusted-present boundary, so there is no jump
        in the potential and no dead zone. Centring only ever pays in
        PROPORTION to the goal actually being in view, so it cannot become a
        generic "point at any object" bonus.

        (2) A budgeted forward nudge still breaks the agent out of parking
        when the goal is entirely off-screen; the budget refills the moment
        the goal is seen. Returns 0 unless configured on."""
        if not self._seek_cats or (self.seek_weight <= 0.0
                                   and self.seek_forward_nudge <= 0.0):
            return 0.0
        # Goal in view -> refill the search budget, but KEEP PAYING: the
        # potential now continues on centredness (see the docstring). The
        # early `return 0.0` here was the dead zone.
        # RATCHET FIX (2026-07-25): the baseline is CARRIED, not nulled —
        # nulling let sight -> lose -> re-sight re-adopt a 0 baseline and
        # re-pay the same climb every cycle. Carrying it charges the descent,
        # so a full look-away/look-back loop nets exactly 0.
        if any(c in present for c in self._seek_cats):
            self._seek_nudge_left = self.seek_nudge_budget
            self._seek_sight_streak = 0
        # trickle regen: search never permanently dies (see __init__)
        if (self.seek_nudge_regen > 0
                and self._seek_nudge_left < self.seek_nudge_budget):
            self._nudge_regen_clock += 1
            if self._nudge_regen_clock >= self.seek_nudge_regen:
                self._nudge_regen_clock = 0
                self._seek_nudge_left += 1
        # grounded + reliable goal-visibility prob (ignore a low-trust head so a
        # hallucinated tree can't drive seeking); absent categories read as 0
        gp = 0.0
        grounded = False
        for c in self._seek_cats:
            if (label_counts.get(c, 0) >= self.min_labels
                    and reliability.get(c, 0.0) >= self.reliability_floor):
                grounded = True
                gp = max(gp, float(probs.get(c, 0.0)))
        # weaker refill (audit fix): a few consecutive real sightings refill the
        # search budget even if the goal never crosses full trusted-present —
        # else the budget could never refill during a reliability dip, exactly
        # when the nudge is the only remaining search signal.
        if gp >= 0.3:
            self._seek_sight_streak += 1
            if self._seek_sight_streak >= 3:
                self._seek_nudge_left = self.seek_nudge_budget
        else:
            self._seek_sight_streak = 0
        r = 0.0
        if self.seek_weight > 0.0:
            # ORIENTATION, not mere visibility. Multiplicative in `gp` so a
            # goal that is not in frame contributes nothing at all and this
            # can never degenerate into "centre the crosshair on anything".
            _cen = (float(self._reach_measured) if self.phi_from_evidence
                    else float(probs.get("object_centered", 0.0)))
            # FOVEA: the goal entering the centre of the view IS the
            # orientation signal (break evidence only confirms it after
            # contact). Take the best gated foveal P over the seek goals.
            _fc = max((self._fovea_p(c) or 0.0 for c in self._seek_cats),
                      default=0.0)
            _cen = max(_cen, _fc)
            _cen = max(0.0, min(1.0, _cen))
            # GAZE RIGHTING: a goal cannot be foveated from a sky-clamped
            # pitch. Levelness is a bounded state function, so this stays a
            # potential (dwelling level pays 0; returning to level pays once,
            # scaled by how visible the goal is).
            gp_eff = max(0.0, min(
                1.0, gp * (0.6 + 0.4 * _cen) * self._pitch_level()))
            if self._seek_prob_prev is not None:
                r += self.seek_weight * (gp_eff - self._seek_prob_prev)
            self._seek_prob_prev = gp_eff
        # nudge burns ONLY once grounding is mature (audit fix: at cold boot the
        # trust gate forces gp=0, so the whole budget burned on the first ~200
        # undirected forward steps before seeking was even possible) and ONLY on
        # PURE forward actions (never attack+forward — see _nudge_actions).
        if (self.seek_forward_nudge > 0.0 and grounded and gp < 0.05
                and self._seek_nudge_left > 0
                and int(action) in self._nudge_actions):
            r += self.seek_forward_nudge
            self._seek_nudge_left -= 1
        return float(r)

    def step_shaping(self, action: int, timestep: int, lp_scalar: float,
                     object_probs: Optional[Dict[str, float]] = None,
                     reliability: Optional[Dict[str, float]] = None,
                     label_counts: Optional[Dict[str, int]] = None,
                     reach_measured: Optional[float] = None,
                     fovea_probs: Optional[Dict[str, float]] = None,
                     fovea_counts: Optional[Dict[str, int]] = None,
                     pitch: Optional[float] = None) -> float:
        if not self.enabled or self._closed:
            return 0.0
        self._now = int(timestep)     # TTL presence for proposals reads this
        if reach_measured is not None:
            self._reach_measured = float(reach_measured)
        self._fovea_probs = fovea_probs or {}
        self._fovea_counts = fovea_counts or {}
        self._pitch = None if pitch is None else float(pitch)
        probs = object_probs or {}
        reliability = reliability or {}
        label_counts = label_counts or {}

        # 0. MID-RUN VOCABULARY EXPANSION. Checked here, every step, because
        # the vocabulary can grow at any moment — a proposal graduating, a new
        # environment registering predicates. Deriving only at construction
        # would leave the instinct permanently blind to anything the agent had
        # not already met when it booted. O(1) when nothing changed.
        self.refresh_targets()

        # 1. trusted-present interactables (grounded, reliable, in view)
        present = self._trusted_present(probs, reliability, label_counts)
        # _score() reads this: only things currently in view are valid
        # comparison peers (see the note there about frozen stale LP).
        self._present_now = set(present)

        # 1b. goal-seeking drive (independent of the target-present gating below,
        # so it keeps pulling toward an off-screen goal object)
        seek_r = self._seek_shaping(action, probs, reliability, label_counts,
                                    present)

        # 2. attribute THIS step's actual LP (intrinsic[0]) to present cats
        lp = float(lp_scalar)
        lp = lp if math.isfinite(lp) else 0.0      # guard NaN/inf latching EMAs
        lp = max(0.0, lp)
        b = self.ema_beta
        self._global_lp = (1.0 - b) * self._global_lp + b * lp
        # PROPOSED categories are attributed EXACTLY like base ones — same
        # EMA, same contrastive score. A proposal earns its place by teaching
        # the agent something, or it does not keep it. Iterating only
        # `target_categories` here would leave every proposal at lp=0 forever,
        # which would evict all of them at probation and make layer 3 a no-op
        # that merely looked implemented.
        for c in self.live_targets():
            cur = self._cat_lp.get(c, 0.0)
            if c in present:
                # STALE-LP DISCOUNT: a fossil EMA re-entering view is decayed
                # toward the CURRENT global by how long it was absent, so a
                # ghost must re-earn its curiosity instead of cashing a
                # months-old score (see lp_stale_tau in __init__).
                _abs_n = self._cat_absent.pop(c, 0)
                if self.lp_stale_tau > 0.0 and _abs_n > 0:
                    _keep = math.exp(-float(_abs_n) / self.lp_stale_tau)
                    cur = self._global_lp + (cur - self._global_lp) * _keep
                self._cat_lp[c] = (1.0 - b) * cur + b * lp
            else:
                self._cat_absent[c] = self._cat_absent.get(c, 0) + 1
                self._cat_lp[c] = cur * (1.0 - self.ema_leak)

        # 3. retarget with hysteresis
        self._select_target(present, timestep)

        # 4. curiosity-driven weight (cache for current_weight/wants_step).
        # Decay EVERY category's scale each step (even while it is quiet) so a
        # category's own stale peak fades — otherwise its own later, lower-
        # magnitude re-arm would be crushed by its history (MF-1 residual).
        # Then normalize by the TARGET CATEGORY's OWN recent peak so a fresh
        # novel object arms at full strength regardless of any other category's
        # larger historical curiosity.
        for _c in self._lp_scale:
            self._lp_scale[_c] *= self.scale_decay
        s_star = self._score(self._target) if self._target else 0.0
        if self._target is not None and s_star >= self.eps_abs:
            # distinctively curious -> contrastive weight; mark the category
            # ENGAGED so the cold-start floor never applies to it again.
            c = self._target
            self._ever_curious[c] = True
            scale = max(self._lp_scale.get(c, 0.0), s_star)   # decay applied above
            self._lp_scale[c] = scale
            rel = s_star / (scale + 1e-8)
            self._w = max(self.min_weight,
                          self.weight * min(1.0, max(0.0, rel)))
        elif (self._target is not None
              and (self._pure_potential
                   or not self._ever_curious.get(self._target, False))
              and self._cold_spent.get(self._target, 0)
              < self.cold_start_budget):
            # COLD START: a present interactable the agent has never engaged.
            # A small floor pull bootstraps interaction; once it becomes curious
            # (branch above), is mastered, or its floor BUDGET is spent (the
            # instinct wanes — audit fix: an un-engageable category like static
            # stone used to collect this floor forever, making "park at stone"
            # a standing income that out-paid tree-seeking 20-40:1), this stops.
            self._w = self.cold_start_weight
            self._cold_spent[self._target] = (
                self._cold_spent.get(self._target, 0) + 1)
        else:
            self._w = 0.0                  # engaged-then-quiet -> fade
        w = self._w
        if w <= 0.0:
            # RATCHET FIX (2026-07-25): do NOT null _phi_prev here. Nulling
            # re-adopts a FRESH baseline when the pull re-arms, so
            # approach -> pull-drops -> re-approach paid the same climb twice
            # (the descent was never charged) — a farm loop that breaks the
            # potential-based/policy-invariance guarantee this term relies on.
            # CARRYING the baseline means returning to the phi you left at
            # pays exactly 0.
            return float(seek_r)           # ...but a seek pull may still apply
        # Area-2: while an incumbent is HELD but out of view (absence-tolerance
        # window), do not pay pull/instinct shaping — the generic scene
        # predicates now describe a DIFFERENT object, so bonuses would be
        # mis-credited to the absent target. Resume when it is back in view.
        if self._target not in present:
            # RATCHET FIX (2026-07-25): carry _phi_prev (see the w<=0 note).
            # Re-adopting a fresh baseline when the target came back into view
            # made lose-sight -> re-approach a repeatable income.
            return float(seek_r)

        # 5. potential + pull terms, keyed on the active target
        r = 0.0
        phi = self._phi_head(probs)
        if self._target_just_switched or self._phi_prev is None:
            self._phi_prev = phi           # adopt WITHOUT paying the delta
            self._target_just_switched = False
        else:
            r += w * (phi - self._phi_prev)
            self._phi_prev = phi

        act = int(action)
        left = float(probs.get("object_left", 0.0))
        right = float(probs.get("object_right", 0.0))
        cen = float(probs.get("object_centered", 0.0))
        adj = float(probs.get("object_adjacent", 0.0))
        pos = ("left" if left > max(right, cen)
               else ("right" if right > max(left, cen) else "center"))
        focused = cen >= 0.5 and adj >= 0.5
        # TRUNK-focused (Lever 2): the chop instinct pays only when the TARGET
        # (a wooden trunk when tree_visible) is present AND at least as prominent
        # as its foliage distractors — so the reward is for chopping the LOG,
        # not the leaves the agent was farming.
        tgt_p = float(probs.get(self._target, 0.0)) if self._target else 0.0
        distractor_p = max((float(probs.get(d, 0.0))
                            for d in self.focus_distractors), default=0.0)
        trunk_focused = (focused and tgt_p >= self.present_threshold
                         and tgt_p >= distractor_p)

        if act in self.chop_actions and trunk_focused:
            r += w * self.instinct_bonus
        if self.approach_pull > 0.0 and act in self.forward_actions:
            r += w * self.approach_pull * (0.3 + 0.7 * float(
                self._phi_prev or 0.0))
        if self.align_bonus > 0.0:
            if pos == "left" and act in self.turn_left_actions:
                r += w * self.align_bonus
            elif pos == "right" and act in self.turn_right_actions:
                r += w * self.align_bonus
        if (self.aim_bonus > 0.0 and not focused
                and float(self._phi_prev or 0.0) >= 0.6
                and act in self.pitch_actions):
            r += w * self.aim_bonus

        self.assessment_log.append({"t": timestep, "target": self._target,
                                    "s": round(s_star, 5), "w": round(w, 5)})
        return float(r + seek_r)

    def close(self) -> None:
        self._closed = True

    @property
    def stats(self) -> Dict:
        return {"enabled": self.enabled, "weight": round(self._w, 5),
                "target": self._target,
                "global_lp": round(self._global_lp, 5),
                "cat_lp": {k: round(v, 5) for k, v in self._cat_lp.items()},
                "seek": {"cats": self._seek_cats,
                         "nudge_left": self._seek_nudge_left},
                "fovea": {c: round(self._fovea_p(c) if self._fovea_p(c)
                                   is not None else -1.0, 3)
                          for c in (self._seek_cats or [])},
                "pitch_level": round(self._pitch_level(), 3),
                "cold_spent": dict(self._cold_spent),
                # budget included so a log line alone explains a w=0 latch
                "cold_start_budget": self.cold_start_budget}
