"""VLM-supervised symbolic grounding — "knows how" -> "knows why".

WHY THIS EXISTS
  On pixels the whole symbolic stack is dead: `_skip_perdim_symbolic` gates
  it off because the FactExtractor reads LABELLED observation dimensions and
  a raw 49152-d frame has none (dimension 8412 is some pixel's green
  channel). Minecraft runs therefore log "0 facts, 0 entities, 0 rules" —
  the agent knows HOW to break a block (implicit weights) but has no
  inspectable, transferable rule saying WHY.

  A vision LLM is exactly the missing converter: a learned pixels->language
  map. Point it at the frame and it emits the predicates the symbolic stack
  has been starving for.

THE DESIGN — the VLM TEACHES, then LEAVES
  The naive version (ask the VLM every step) is 100x too slow (~1.1s/call vs
  ~10 steps/s) and bolts a permanent oracle into perception. Instead:

    1. every `interval` steps the VLM labels a frame against a FIXED
       predicate vocabulary  ->  multi-label target
    2. `GroundedSymbolHead` learns to predict those predicates FROM THE
       AGENT'S OWN RSSM LATENT (BCE) — this is the step that converts
       BORROWED abstraction into OWNED abstraction
    3. as the head's agreement with the VLM rises, VLM queries get RARER
       (annealing, same philosophy as the magnet scaffold)
    4. eventually the head names the world alone, at full speed, no LLM

  Philosophically this is not cheating: no human derives the category "tree"
  from first principles either — it is handed over by language and then
  GROUNDED in one's own sensory experience. The LLM is the culture; the
  grounding still has to be earned by the head.

SELF-TESTING ABSTRACTIONS — and the limit of the test
  VLM claims are checked against the game's OWN ground truth (mine_block
  stats), giving a per-predicate reliability EMA. A predicate the world
  repeatedly contradicts loses confidence, stops emitting, and has its
  already-minted triple RETRACTED from the knowledge graph.

  Be honest about what this verifier can and cannot see. The only event that
  scores a predicate is a block break, and `hit` is "did the VLM claim this
  when the break happened". So it detects FALSE NEGATIVES — the VLM said
  "nothing breakable in reach" and the world proved otherwise — but it is
  structurally blind to FALSE POSITIVES: a predicate stuck at True is scored
  correct on every break and drifts to reliability 1.0, because no event ever
  supplies negative evidence for an over-claim. Only 4 of the 16 predicates
  are scored at all (breakable_in_reach, object_centered, object_adjacent,
  tree_visible); the other 12 sit at their prior forever and can never be
  retracted. A low `retracted=` count in the log is therefore expected, and
  is NOT evidence the channel is working. CLOSED 2026-08-07: the loop now
  feeds exactly that negative source — `observe_negative_event` docks the
  claimed affordance predicates when a full break-worth of held attack
  produced no break (see the method for scope and gentler alpha).

TWO KINDS OF FACT reach the knowledge graph:
  * PERCEPTUAL (from the VLM/head): ("scene","contains","tree")
  * CAUSAL    (from ground truth) : ("attack_held","breaks","oak_log")
  The second kind is the one rule induction actually needs, and it is never
  taken on the VLM's word — it comes from observed action->effect events.
"""

from __future__ import annotations

import io
import logging
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from developmental_ai.llm.llm_module import (
    _AsyncChannel,
    _check_ollama_server,
    _get_ollama_client,
    _parse_json_object,
)

logger = logging.getLogger(__name__)

# FIXED vocabulary — the head's output layer and every emitted fact key off
# this list, so it must stay stable across a run. Generic object/affordance
# concepts, deliberately not Minecraft-jargon, so rules can transfer.
PREDICATES: List[str] = [
    "tree_visible",        # a wooden trunk anywhere in view
    "leaves_visible",
    "grass_visible",
    "stone_visible",
    "dirt_visible",
    "water_visible",
    "sky_visible",
    "animal_visible",
    "object_centered",     # something is under the crosshair
    "object_adjacent",     # ...and within reach
    "object_left",
    "object_right",
    "breakable_in_reach",  # a solid, breakable face is touchable
    "looking_up",
    "looking_down",
    "open_space_ahead",    # can walk forward unobstructed
    # ---- CREATURES (2026-07-25): TRUE mob perception, learned meaning ----
    # The agent SEES these; it is never TOLD what they do. The VLM supplies
    # only the NAME from pixels (like a child hearing a word); the
    # GroundedSymbolHead then learns to recognise each from the agent's OWN
    # latent, and MEANING is earned from experience — a creature that is in
    # view when health drops becomes causally linked to being hurt (see
    # `_creature_causal_facts` in the loop). Nothing here encodes "hostile".
    # Unreliable predicates self-suppress: per-predicate reliability + the
    # retraction margin stop llava from poisoning the KG if it cannot in fact
    # tell a zombie from a skeleton at 128px.
    "zombie_visible",
    "skeleton_visible",
    "creeper_visible",
    "spider_visible",
    "cow_visible",
    "pig_visible",
    "sheep_visible",
    # ---- TOOLS (2026-07-25): the agent SEES what it is holding ----
    # Minecraft renders the held item in the first-person view, and a
    # durability bar under the hotbar icon once the tool is worn. So both are
    # genuinely VISIBLE and can be learned from pixels like anything else.
    # As with creatures, the VLM supplies only the NAME of what is on screen —
    # nothing here says an axe is "for chopping" or that a short bar means
    # "about to break". Those meanings are earned from use: an axe in hand
    # when a log falls becomes `axe enables break_oak_log`; a bar that shrinks
    # as the agent works becomes `axe wears_down`; a tool that vanishes
    # becomes `axe breaks_from_use` (see `_tool_causal_facts` in the loop).
    "holding_tool",        # something tool-shaped is in the agent's hand
    "tool_worn",           # a durability bar is visible under the held item
    # ---- GUI / CRAFTING (2026-07-26): the screen is part of the world ----
    # Crafting in this MineRL fork is GUI-only — the symbolic craft handlers
    # exist in Python but the live Java backend has no MissionHandlers package
    # and closes the socket on an unknown command, so a "craft planks" action
    # is not merely unavailable, it is destructive. What IS available is what
    # a human uses: open the inventory, move a cursor, click.
    #
    # These are named `_visible` for vocabulary consistency, but they are
    # explicitly NON-STEERABLE for the magnet (vision_scaffold): a GUI screen
    # is self-state, not a place in the world. Letting the derived steerable
    # set include them was measured to make the magnet court the MENU
    # (cold_spent[inventory]=55k, 25.9k-step gui dwell, 2026-08-07).
    #
    # As everywhere else: the VLM supplies only what is ON SCREEN. Nothing
    # here encodes a RECIPE. That two planks make four sticks must be learned
    # from watching the output slot fill, exactly as `axe enables break_log`
    # was learned from watching a log fall.
    "inventory_visible",     # the inventory/crafting screen is open
    "crafting_grid_visible", # empty crafting slots are on screen
    "craft_output_visible",  # THE causal signal: the result slot has an item
    # ---- SOCIAL PERCEPTION (2026-08-09): another PLAYER is a first-class
    # percept. The user plays on the same server, and the single most human
    # learning channel — watching a co-present adult demonstrate — was
    # invisible: the avatar rendered in the POV but nothing could NAME it.
    # As everywhere: the VLM supplies only presence; what a player MEANS
    # (joint attention, demonstrations) is learned by the loop from
    # observed-change events, never declared here.
    "player_visible",        # another human player's avatar is in view
]
# Creature predicates — used by the loop to associate what was in view with
# what happened to the agent (damage/death). Order-independent.
CREATURE_PREDICATES: List[str] = [
    "zombie_visible", "skeleton_visible", "creeper_visible",
    "spider_visible", "cow_visible", "pig_visible", "sheep_visible",
    "animal_visible",
]
PREDICATE_INDEX = {p: i for i, p in enumerate(PREDICATES)}

# How each predicate becomes a knowledge-graph triple.
_FACT_TEMPLATES: Dict[str, Tuple[str, str, str]] = {
    "tree_visible": ("scene", "contains", "tree"),
    "leaves_visible": ("scene", "contains", "leaves"),
    "grass_visible": ("scene", "contains", "grass"),
    "stone_visible": ("scene", "contains", "stone"),
    "dirt_visible": ("scene", "contains", "dirt"),
    "water_visible": ("scene", "contains", "water"),
    "sky_visible": ("scene", "contains", "sky"),
    "animal_visible": ("scene", "contains", "animal"),
    "object_centered": ("object", "is", "centered"),
    "object_adjacent": ("object", "is", "adjacent"),
    "object_left": ("object", "is", "left"),
    "object_right": ("object", "is", "right"),
    "breakable_in_reach": ("object", "is", "breakable"),
    "looking_up": ("agent", "is", "looking_up"),
    "looking_down": ("agent", "is", "looking_down"),
    "open_space_ahead": ("path", "is", "clear"),
    # creatures: presence only — what they MEAN is learned, not declared
    "zombie_visible": ("scene", "contains", "zombie"),
    "skeleton_visible": ("scene", "contains", "skeleton"),
    "creeper_visible": ("scene", "contains", "creeper"),
    "spider_visible": ("scene", "contains", "spider"),
    "cow_visible": ("scene", "contains", "cow"),
    "pig_visible": ("scene", "contains", "pig"),
    "sheep_visible": ("scene", "contains", "sheep"),
    # tools: what is HELD and whether it looks worn — presence only.
    # What a tool is FOR is learned from using it, never declared.
    "holding_tool": ("agent", "holds", "tool"),
    "tool_worn": ("tool", "is", "worn"),
    # GUI facts state PRESENCE only — never what a recipe produces.
    "inventory_visible": ("scene", "contains", "inventory"),
    "crafting_grid_visible": ("scene", "contains", "crafting_grid"),
    "craft_output_visible": ("scene", "contains", "craft_output"),
    "player_visible": ("scene", "contains", "player"),
}

# ---- FOVEA (2026-08-06): presence-in-a-patch instead of spatial judgment --
# MEASURED on real 128px SkyBot frames: llava:7b answered tree_visible=true on
# EVERY crop shown to it, including a canopy-only patch (and hallucinated
# freely — "a dog", "YOUR INVESTMENT" — on unambiguous single-material
# patches). The saturated tree_visible that froze the whole guidance stack was
# therefore the TEACHER's failure, not the head's. qwen2.5vl:7b on the same
# battery scored 8/8, same ~1.1s warm latency. But even a good VLM cannot
# judge "is something under the crosshair" from a full frame (it picked the
# far-left trunk over the near-right one when asked for the nearest) — what it
# CAN do reliably is presence-in-a-patch. So the spatial question is converted
# into a presence question by CROPPING: label the centre of the view
# separately. That is foveation — the answer to "am I looking AT it?" comes
# from what the fovea contains, exactly as it does in a human.
#
# The foveal vocabulary is the small material/object subset that can occupy a
# patch of world. GUI/self-state predicates are whole-screen properties and
# never foveal. Deliberately reuses the `_visible` convention so the magnet's
# steerable-set derivation applies unchanged.
FOVEA_PREDICATES: List[str] = [
    "tree_visible", "leaves_visible", "grass_visible", "stone_visible",
    "dirt_visible", "water_visible", "sky_visible", "animal_visible",
    # social perception (2026-08-09): "am I looking AT the other player" is
    # the gaze half of joint attention
    "player_visible",
]
FOVEA_INDEX = {p: i for i, p in enumerate(FOVEA_PREDICATES)}

_FOVEA_PROMPT = (
    "This is a small patch cropped from the CENTER of a low-resolution "
    "Minecraft first-person view (the crosshair area — what the player is "
    "looking directly at). Answer strictly about THIS PATCH ONLY; if unsure "
    "answer false.\n"
    'Return ONLY a JSON object with exactly these keys: '
    '{{"tree_visible": bool, "leaves_visible": bool, "grass_visible": bool, '
    '"stone_visible": bool, "dirt_visible": bool, "water_visible": bool, '
    '"sky_visible": bool, "animal_visible": bool, "player_visible": bool}}\n'
    'Guidance: "tree_visible" = a wooden TRUNK occupies part of the patch — '
    "a VERTICAL log column with bark texture (birch is white with black "
    "streaks, oak/spruce are brown). Leaves alone are NOT a trunk. "
    '"leaves_visible" = foliage blocks. "sky_visible" = open sky. '
    '"player_visible" = another human player\'s avatar (person-like, not a '
    "mob). Name only what is actually in the patch."
)


_SCENE_PROMPT = (
    "You label what a Minecraft agent can see, for a symbolic reasoner. "
    "Look at this first-person view and answer each question true/false. "
    "Be literal and strict — answer false if unsure.\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    '{{"tree_visible": bool, "leaves_visible": bool, "grass_visible": bool, '
    '"stone_visible": bool, "dirt_visible": bool, "water_visible": bool, '
    '"sky_visible": bool, "animal_visible": bool, "object_centered": bool, '
    '"object_adjacent": bool, "object_left": bool, "object_right": bool, '
    '"breakable_in_reach": bool, "looking_up": bool, "looking_down": bool, '
    '"open_space_ahead": bool, "zombie_visible": bool, '
    '"skeleton_visible": bool, "creeper_visible": bool, '
    '"spider_visible": bool, "cow_visible": bool, "pig_visible": bool, '
    '"sheep_visible": bool, "holding_tool": bool, "tool_worn": bool, '
    '"inventory_visible": bool, "crafting_grid_visible": bool, '
    '"craft_output_visible": bool, "player_visible": bool, '
    '"novel": string}}\n'
    '"novel" = LAYER 3 INSTINCT. If something SALIENT is on screen that '
    'none of the other keys describe, name it in ONE lower_snake_case '
    'word ending in _visible (e.g. "lava_visible"). Otherwise return "". '
    'Name only what you can SEE. This is a suggestion about where to '
    'look, NOT a claim about what anything does.\n'
    'Guidance: "tree_visible" = a wooden TRUNK — a vertical log column '
    "with bark texture (birch is white with black streaks, oak/spruce are "
    "brown). Leaves alone do NOT count. "
    '"object_centered" = the centre crosshair rests on some block. '
    '"object_adjacent" = that block is within arm\'s reach. '
    '"breakable_in_reach" = a solid block face is close enough to hit. '
    '"looking_up"/"looking_down" = the view is angled at sky / at the ground. '
    '"open_space_ahead" = the agent could walk forward without hitting a wall. '
    '"inventory_visible" = the inventory/crafting SCREEN is open (a grid '
    'of item slots covering the view), not the small hotbar. '
    '"crafting_grid_visible" = the small square crafting slots are on '
    'screen. "craft_output_visible" = the RESULT slot (right of the '
    'arrow) currently holds an item. '
    '"player_visible" = another human PLAYER\'s avatar (a Steve/Alex-like '
    'humanoid in normal skin colours, often holding an item) — NOT a green '
    'zombie, NOT a bony skeleton; a player looks like a person. '
    'Creatures: name ONLY what you can actually identify — a green humanoid '
    'is a zombie, a white bony humanoid is a skeleton, a green four-legged '
    'creature with a flat face is a creeper, a dark many-legged creature is a '
    'spider, and cow/pig/sheep are the farm animals. If a creature is present '
    'but you cannot tell which, set "animal_visible" true and leave the '
    'specific ones false. NEVER guess a specific creature. '
    'Held item: "holding_tool" = a tool or weapon is visible in the '
    'agent\'s hand at the lower-right of the view (an axe, pickaxe or '
    'sword shape) — false if the hand is empty. "tool_worn" = a small '
    'coloured durability bar is visible under the held item, meaning it '
    'has been used. Report only what you SEE; never infer what a tool '
    'is for.'
)


# Hysteresis for retraction. A predicate must climb this far back ABOVE
# reliability_floor before it may be asserted again, so a reliability EMA
# resting exactly on the floor cannot flap the knowledge graph for two days.
_REARM_MARGIN = 0.05

# ---- FALSIFIABILITY REGISTRY (infra #3, 2026-08-08) -----------------------
# Which predicates have a GROUND-TRUTH event class that can DISCONFIRM them:
# break events + the fruitless-swing negative channel score these four; every
# other predicate has confirmation-only or no evidence at all. A belief that
# no observation could refute must never accumulate confidence from one-sided
# evidence — that is exactly how a stuck-true tree_visible drifted to
# reliability 1.0 unopposed for weeks. Unfalsifiable predicates get a hard
# reliability CEILING at their 0.7 prior: usable, never authoritative.
FALSIFIABLE_PREDICATES = frozenset({
    "breakable_in_reach", "object_centered", "object_adjacent",
    "tree_visible",
    # ---- SCORED AGAINST TELEMETRY SINCE 2026-09-01 -------------------
    # The module docstring is candid that only 4 of 16 predicates were ever
    # scored, and that the other 12 "sit at their prior forever and can never
    # be retracted". These four are not approximations — the env emits their
    # ground truth every single step, and it was simply never compared:
    #   looking_up / looking_down  <- world["pitch"]
    #   inventory_visible          <- world["gui_open"]
    #   holding_tool               <- world["mainhand"] not in (none, air)
    # See `observe_truth`. They are also a CALIBRATION PROBE: a VLM that
    # cannot say which way it is looking is not one to trust on
    # breakable_in_reach, and that is now visible before it poisons anything.
    "looking_up", "looking_down", "inventory_visible", "holding_tool",
})
_UNFALSIFIABLE_RELIABILITY_CAP = 0.7

# predicate -> callable(world_info) -> Optional[bool] ground truth.
# None means "the world cannot answer right now"; only a definite answer
# scores, so a missing telemetry field never counts as a miss.
_TELEMETRY_TRUTH = {
    "looking_up": lambda w: (None if w.get("pitch") is None
                             else float(w["pitch"]) < -20.0),
    "looking_down": lambda w: (None if w.get("pitch") is None
                               else float(w["pitch"]) > 20.0),
    "inventory_visible": lambda w: (None if w.get("gui_open") is None
                                    else bool(w["gui_open"])),
    "holding_tool": lambda w: (
        None if w.get("mainhand") is None
        else str(w["mainhand"]) not in ("none", "air", "")),
}


class LabelReplay:
    """Ring buffer of (latent, target, mask) triples from VLM labels.

    ---- THE SCARCEST RESOURCE IN THE SUBSYSTEM (2026-09-01) --------------
    A VLM label costs ~1.5 s of shared-GPU inference and arrives roughly
    every 20 s of wall clock. Before this existed, each one produced exactly
    one batch-size-1 gradient step and was then thrown away. Keeping them
    turns "4-8k labels a day" into "4-8k labels a day, each seen as often as
    training can afford" — which is the difference between a head that has
    seen 8k examples and one that has taken 8k steps.

    ON HOST MEMORY, NOT VRAM, deliberately. A latent is 4352 float32 = 17.4
    KB, so 50k labels is ~870 MB; that is affordable in RAM and would be a
    third of the VRAM budget on a 12 GB card. Batches move to the device on
    sample, which is a ~1 MB transfer for batch 64.

    Stores the MASK alongside the target because VLM labels are partial: a
    predicate the model declined to report must not teach the head a false
    negative, and that distinction has to survive into the replay.
    """

    def __init__(self, capacity: int, dim: int, n_predicates: int):
        self.capacity = max(0, int(capacity))
        self.dim = int(dim)
        self.n = int(n_predicates)
        self._lat = None
        self._tgt = None
        self._msk = None
        self._size = 0
        self._pos = 0
        if self.capacity > 0:
            self._lat = torch.zeros(self.capacity, self.dim)
            self._tgt = torch.zeros(self.capacity, self.n)
            self._msk = torch.zeros(self.capacity, self.n)

    def __len__(self) -> int:
        return self._size

    def add(self, latent: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        i = self._pos
        self._lat[i] = latent.detach().reshape(-1).float().cpu()
        self._tgt[i] = target.detach().reshape(-1).float().cpu()
        self._msk[i] = mask.detach().reshape(-1).float().cpu()
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device=None):
        if self._size == 0:
            return None, None, None
        idx = torch.randint(0, self._size, (min(batch_size, self._size),))
        lat, tgt, msk = self._lat[idx], self._tgt[idx], self._msk[idx]
        if device is not None:
            lat, tgt, msk = lat.to(device), tgt.to(device), msk.to(device)
        return lat, tgt, msk


class GroundedSymbolHead(nn.Module):
    """latent -> P(predicate) for the fixed vocabulary (multi-label).

    This is the OWNED half of the design: it learns to name the world from
    the agent's own world-model latent, so once trained the symbols come for
    free at full speed with no LLM in the loop.
    """

    def __init__(self, latent_dim: int, n_predicates: int = len(PREDICATES),
                 hidden_dim: int = 256, lr: float = 1e-3,
                 n_layers: int = 3, n_members: int = 1):
        super().__init__()
        self.n_predicates = int(n_predicates)
        self.n_members = max(1, int(n_members))
        self.n_layers = max(2, int(n_layers))

        def _mlp():
            layers = [nn.Linear(latent_dim, hidden_dim), nn.ELU()]
            for _ in range(self.n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
            layers += [nn.Linear(hidden_dim, self.n_predicates)]
            return nn.Sequential(*layers)

        # ---- ENSEMBLE (2026-09-01) ---------------------------------------
        # K independently initialised MLPs. Members are trained on
        # INDEPENDENTLY SAMPLED replay minibatches (see train_replay), so
        # they genuinely diverge rather than being K copies of one trajectory.
        #
        # WHY: `fact_threshold` currently reads a single sigmoid as
        # confidence, which cannot tell "the head is sure" from "the head has
        # collapsed to a constant" — and this project has watched a predicate
        # collapse to a stuck-true constant and ride to reliability 1.0
        # unopposed (llava's tree_visible). Spread ACROSS members is
        # epistemic uncertainty: a collapsed predicate shows near-constant
        # output AND near-zero disagreement, which is distinguishable from
        # genuine confidence by construction rather than by tuning.
        #
        # n_members=1 keeps the module arithmetically identical to the
        # pre-2026-09-01 head (mean over one member is that member).
        self.members = nn.ModuleList([_mlp() for _ in range(self.n_members)])
        # One optimizer per member: a shared one would couple their updates
        # through Adam's moment estimates and erode the diversity that makes
        # the disagreement signal mean anything.
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=lr)
                           for m in self.members]
        # kept so existing code paths that reach for `.optimizer` still work
        self.optimizer = self.optimizers[0]

    @property
    def net(self):
        """Back-compat alias for the single-member case."""
        return self.members[0]

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """MEAN LOGITS across the ensemble (identity when n_members == 1)."""
        if self.n_members == 1:
            return self.members[0](latent)
        return torch.stack([m(latent) for m in self.members], 0).mean(0)

    def member_probs(self, latent: torch.Tensor) -> torch.Tensor:
        """(K, B, P) per-member probabilities — the raw material for spread."""
        with torch.no_grad():
            return torch.stack(
                [torch.sigmoid(m(latent)) for m in self.members], 0)

    def disagreement(self, latent: torch.Tensor) -> torch.Tensor:
        """(P,) std across members. 0 for a single member, by definition —
        which is the honest answer: one head cannot disagree with itself, so
        an ensemble of one supplies no uncertainty and the caller must not
        pretend otherwise."""
        if self.n_members == 1:
            return torch.zeros(self.n_predicates)
        p = self.member_probs(latent)              # (K, B, P)
        return p.std(dim=0).mean(dim=0).cpu()      # (P,)

    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(latent))

    def train_step(self, latent: torch.Tensor, target: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   pos_weight: Optional[torch.Tensor] = None) -> float:
        """One BCE step against VLM labels. `mask` (same shape) zeroes out
        predicates the VLM did not report, so a partial label never teaches
        the head a false negative. `pos_weight` (per-predicate) upweights the
        POSITIVE term for rare classes — without it a predicate present in
        ~9% of frames is best served by answering "no" forever.

        EVERY MEMBER trains on this sample (it is the one fresh label and all
        of them should see it); diversity comes from the independently
        sampled replay minibatches in train_replay, not from withholding new
        evidence."""
        total = 0.0
        for m, opt in zip(self.members, self.optimizers):
            loss = self._member_loss(m, latent, target, mask, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        return total / len(self.members)

    def _member_loss(self, member, latent, target, mask, pos_weight):
        logits = member(latent)
        loss_el = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none",
            pos_weight=(pos_weight.reshape(1, -1).to(logits.device)
                        if pos_weight is not None else None))
        if mask is not None:
            return (loss_el * mask).sum() / mask.sum().clamp(min=1.0)
        return loss_el.mean()

    def train_replay(self, buf, batch_size: int, n_batches: int,
                     pos_weight: Optional[torch.Tensor] = None,
                     device=None) -> float:
        """Train on minibatches drawn from a label REPLAY buffer.

        ---- WHY THIS IS THE HIGHEST-VALUE CHANGE HERE (2026-09-01) --------
        Before this, `_train_head` took ONE gradient step, at batch size 1,
        on each VLM label and then discarded it. At `interval: 60` and ~3
        agent steps/s that is a label every ~20 s — roughly 4-8k labels in a
        24-hour run, and therefore 4-8k single-sample gradient steps, to fit
        29 binary predicates from a 4352-d input. Labels are by far the
        scarcest resource in this subsystem and they were being used once.
        Adding head capacity under that regime would have overfit, not
        improved anything, which is why this lands before the width change.

        Each MEMBER draws its OWN minibatch. That is what makes the ensemble
        an ensemble: identical batches in the same order would leave K heads
        following nearly the same trajectory from different inits, and the
        disagreement signal would understate real uncertainty.
        """
        if not buf or batch_size <= 0 or n_batches <= 0:
            return 0.0
        total, steps = 0.0, 0
        for m, opt in zip(self.members, self.optimizers):
            for _ in range(n_batches):
                lat, tgt, msk = buf.sample(batch_size, device=device)
                if lat is None:
                    break
                loss = self._member_loss(m, lat, tgt, msk, pos_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()); steps += 1
        return total / max(1, steps)


class VLMSymbolizer:
    """Orchestrates VLM labelling -> head training -> verification -> facts.

    Call `maybe_label(frame, timestep)` each step (cheap; it self-throttles),
    then `train_and_emit(latent, ...)` when a label lands. Ground-truth
    events are fed via `observe_event()` which both scores predicate
    reliability and mints CAUSAL facts.
    """

    def __init__(
        self,
        model: str = "llava:7b",
        interval: int = 60,
        max_interval: int = 600,
        latent_dim: int = 1536,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        fact_threshold: float = 0.7,      # P(pred) to emit a perceptual fact
        min_labels: int = 5,              # VLM sightings before a predicate may assert
        reliability_floor: float = 0.35,  # below this a predicate is muted
        anneal_agreement: float = 0.9,    # head agreement that widens interval
        device: Optional[torch.device] = None,
        enabled: bool = True,
        query_fn: Optional[Callable[[bytes], Optional[str]]] = None,
        # FOVEA channel (see FOVEA_PREDICATES above). Off by default: every
        # existing config keeps exactly its old labelling cadence and head.
        fovea: bool = False,
        fovea_frac: float = 0.4,
        input_max_side: int = 0,
        # FOVEA CADENCE DECOUPLING (2026-08-08). The fovea used to alternate
        # with the full channel and therefore inherit its ANNEALED interval —
        # but the anneal is driven by FULL-frame agreement, which says nothing
        # about whether the fovea head has learned its (much harder, class-
        # imbalanced) trunk-under-gaze discrimination. MEASURED: full interval
        # annealed to 454, so the fovea got one label per ~908 steps and
        # ground its way to 284 labels in 11 hours. The fovea now keeps its
        # OWN fixed cadence (None -> the base interval); the full channel
        # anneals independently. One worker still serves both — whichever is
        # more overdue relative to its own cadence goes first.
        fovea_interval: Optional[int] = None,
        # Input width of the FOVEA head. Set this to the world model's
        # encoder width to put the head on crop features (see below).
        fovea_latent_dim: Optional[int] = None,
        # None -> hidden_dim, so an unset config is byte-identical.
        head_hidden: Optional[int] = None,
        head_layers: int = 3,
        head_members: int = 1,
        replay_capacity: int = 0,
        replay_batch: int = 64,
        replay_steps: int = 0,
        max_disagreement: float = 1.0,
        reprobe_after_failures: int = 5,
    ):
        self.model = model
        # See _ollama_scene_query: re-run the boot smoke probe every this
        # many CONSECUTIVE failures, so a mid-run Ollama crash is reported
        # instead of silently producing zero labels at DEBUG forever. 0 =
        # off (never re-probe).
        self.reprobe_after_failures = max(0, int(reprobe_after_failures))
        self._consec_failures = 0
        self.base_interval = max(1, int(interval))
        self.interval = self.base_interval
        self.max_interval = max(self.base_interval, int(max_interval))
        self.fact_threshold = float(fact_threshold)
        self.min_labels = int(min_labels)
        self.reliability_floor = float(reliability_floor)
        self.anneal_agreement = float(anneal_agreement)
        # Deadband: de-anneal only when agreement falls well BELOW the
        # widening gate, so a value hovering at the boundary cannot make the
        # interval oscillate for the length of a multi-day run.
        self._deanneal_agreement = max(0.0, self.anneal_agreement - 0.15)
        self.device = device or torch.device("cpu")
        self.enabled = bool(enabled)
        self._query_fn = query_fn or self._ollama_scene_query
        # an injected query_fn (tests, lesions) serves BOTH channels — the
        # fovea must never silently fall back to a live ollama call under it
        self._fovea_query_fn = query_fn or self._fovea_query

        self._available = bool(enabled) and (
            query_fn is not None or _check_ollama_server())
        # the falsifiable/unfalsifiable split, stated once at startup so the
        # epistemics of every predicate are on record (infra #3)
        _n_fals = len(FALSIFIABLE_PREDICATES & set(PREDICATES))
        logger.info(
            "symbolizer epistemics: %d/%d predicates falsifiable %s; the "
            "other %d are confirmation-only and reliability-capped at %.2f",
            _n_fals, len(PREDICATES), sorted(FALSIFIABLE_PREDICATES),
            len(PREDICATES) - _n_fals, _UNFALSIFIABLE_RELIABILITY_CAP)

        import concurrent.futures
        self._executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vlm-symbol")
            if self._available else None)
        self._channel = _AsyncChannel(self._executor, True)

        head_hidden = int(head_hidden or hidden_dim)
        self.head_hidden = head_hidden      # part of the arch signature
        self.head = GroundedSymbolHead(
            latent_dim, len(PREDICATES), head_hidden, lr,
            n_layers=head_layers, n_members=head_members).to(self.device)
        # See LabelReplay: labels are the scarce resource, not parameters.
        self.replay = LabelReplay(replay_capacity, latent_dim, len(PREDICATES))
        self.replay_batch = int(replay_batch)
        self.replay_steps = int(replay_steps)
        self.max_disagreement = float(max_disagreement)
        self.last_disagreement: Dict[str, float] = {}

        # ---- FOVEA: a second, SEPARATE head over the small foveal
        # vocabulary. Separate on purpose: it never emits knowledge-graph
        # facts, never changes PREDICATES (so no observation/head dimension
        # anywhere else moves), and its labels come from a different image
        # (the centre crop) so they must never train the full-frame head.
        # It exists for exactly one consumer: the magnet's "am I looking AT
        # the target?" gradient.
        self.fovea_enabled = bool(fovea)
        self.fovea_frac = float(fovea_frac)
        # Downscale the WHOLE-FRAME request to this longest side before the
        # PNG encode; 0 = off. 336 is qwen2.5vl's ViT tile size, so the model
        # sees the same tiling it would anyway and the extra pixels the 384
        # render carries were being re-tiled at cost for nothing. The fovea
        # crop is deliberately exempt — it is the high-detail channel and the
        # whole reason it exists is that a downscaled full frame could not
        # tell a zombie from a skeleton.
        self.input_max_side = int(input_max_side or 0)
        # CROP-CONSISTENT INPUT (2026-08-12). The head used to read the
        # WHOLE-FRAME RSSM latent while its labels described the CENTRE CROP
        # — it was asked to decode "what is in the middle 16% of the image"
        # from a global, temporally-mixed embedding. Measured result: fovea
        # agreement 0.876 against 0.8757 for a head that always answers "no",
        # i.e. essentially zero information, with P(tree) pinned at its own
        # 0.094 prior. When `fovea_latent_dim` is given, the head instead
        # reads ENCODER FEATURES OF THE CROP ITSELF (see set_crop_encoder):
        # a static function of the patch pixels, which is what a patch label
        # actually describes. None keeps the old whole-frame wiring.
        self.latent_dim = int(latent_dim)
        self.fovea_latent_dim = (int(fovea_latent_dim)
                                 if fovea_latent_dim else int(latent_dim))
        self._crop_encode = None
        self._fovea_lat_cache = None
        self._fovea_lat_step = -1
        self.fovea_head = (GroundedSymbolHead(
            self.fovea_latent_dim, len(FOVEA_PREDICATES), head_hidden, lr,
            n_layers=head_layers, n_members=head_members
        ).to(self.device) if self.fovea_enabled else None)
        self.fovea_replay = (
            LabelReplay(replay_capacity, self.fovea_latent_dim,
                        len(FOVEA_PREDICATES))
            if self.fovea_enabled else None)
        self.fovea_label_counts = {p: 0 for p in FOVEA_PREDICATES}
        # POSITIVE sightings per foveal predicate (review 2026-08-09): the
        # label count says "the head was TAUGHT about this key", which is
        # free after ~5 fovea labels for every key at once — it can never
        # gate "has this thing actually been SEEN". Presence-claims (the
        # social gate, episodic landmarks, the boring-view association)
        # must gate on positives.
        self.fovea_pos_counts = {p: 0 for p in FOVEA_PREDICATES}
        self.total_fovea_labels = 0
        self.last_fovea_labels: Dict[str, bool] = {}
        self._fovea_agreement: deque = deque(maxlen=50)
        # per-channel cadence (see fovea_interval in the signature): the
        # fovea keeps a FIXED cadence while the full channel anneals.
        self.fovea_interval = int(fovea_interval
                                  if fovea_interval is not None
                                  else self.base_interval)
        self._last_fovea_submit = -(10 ** 9)
        self._pending_is_fovea = False
        # freshness for reliability scoring must track FULL-frame labels only
        # (observe_event compares against last_labels, which the fovea channel
        # never writes).
        self._last_full_submit = -(10 ** 9)

        # per-predicate reliability (EMA of ground-truth agreement); starts
        # neutral-optimistic so predicates are usable before evidence lands
        self.reliability = {p: 0.7 for p in PREDICATES}
        # how many times the VLM has actually LABELLED each predicate. An
        # untrained head sits near P=0.5 and shared-trunk drift can push
        # never-seen predicates over the fact threshold — that would flood
        # the KG with fabricated facts. No evidence, no assertion.
        self.label_counts = {p: 0 for p in PREDICATES}
        # POSITIVE label counts for the FULL channel (2026-08-09): same
        # rationale as fovea_pos_counts — presence-certification ("has a
        # player actually been SEEN") must count positives, and the full
        # frame sees the whole scene, so it certifies far faster than the
        # centre crop.
        self.pos_counts = {p: 0 for p in PREDICATES}
        # Predicates whose triple has already been handed back for retraction.
        # Makes retraction EDGE-triggered (once per collapse, not once per
        # step) and lets a recovered predicate be asserted again.
        self._retracted: set = set()
        self.total_retracted = 0
        self._pending_latent: Optional[torch.Tensor] = None
        self._last_submit = -(10 ** 9)
        self._recent_agreement: deque = deque(maxlen=50)

        # introspection
        self.total_labels = 0
        self.total_parse_failures = 0
        self.total_query_failures = 0
        self.total_empty_labels = 0
        self.total_head_steps = 0
        self.total_facts = 0
        self.last_labels: Dict[str, bool] = {}
        self.last_head_loss = 0.0

    # ---- VLM plumbing ------------------------------------------------------

    def _ollama_scene_query(self, png_bytes: bytes) -> Optional[str]:
        try:
            client = _get_ollama_client()
            if client is None:
                return None
            resp = client.generate(
                model=self.model, prompt=_SCENE_PROMPT, images=[png_bytes],
                format="json", keep_alive=-1,
                options={"num_predict": 256, "temperature": 0.0})
            txt = (resp.get("response") or "").strip()
            self._consec_failures = 0
            return txt
        except Exception as e:
            logger.debug("VLMSymbolizer query failed: %s", e)
            # ---- MID-RUN FAILURE IS NOT JUST A BOOT-TIME QUESTION --------
            # (2026-09-02) probe_ollama_model catches a bad pull or a dead
            # server AT BOOT. It says nothing about Ollama crashing or being
            # restarted six hours into a run — that failure mode looked
            # identical to "the VLM is just being slow" until now: silent,
            # at DEBUG, forever. After a real streak of failures (not one —
            # a single dropped request is normal on a shared GPU) re-run the
            # same smoke check that boot uses and warn at the same volume.
            self._consec_failures = getattr(self, "_consec_failures", 0) + 1
            # modulo, not ==: re-arms on its own during a prolonged outage
            # instead of warning once and going quiet for the rest of the run
            if (self.reprobe_after_failures > 0
                    and self._consec_failures % self.reprobe_after_failures
                    == 0):
                try:
                    from developmental_ai.llm.llm_module import (
                        probe_ollama_model)
                    probe_ollama_model(self.model)
                except Exception:
                    pass
                logger.warning(
                    "VLM: %d consecutive failed generate calls for %s — "
                    "re-ran the smoke probe (see the line above). If it "
                    "also failed, the server likely crashed or restarted "
                    "mid-run; this channel is now producing zero labels.",
                    self._consec_failures, self.model)
            return None

    @staticmethod
    def _encode_png(frame, max_side: int = 0) -> Optional[bytes]:
        """Frame -> PNG bytes for the VLM, optionally downscaled first.

        ---- max_side (2026-09-01) ---------------------------------------
        The env renders natively at 384 so the VLM gets a sharp frame, but
        qwen2.5vl's vision tower tiles at 336 — anything above that is
        re-tiled at cost and buys nothing. Downscaling to `max_side` before
        the PNG encode cuts bytes per request ~(384/336)^2 and, more to the
        point, cuts the tiles the model has to attend over. 0 = off = the
        pre-2026-09-01 behaviour exactly.

        NOTHING ABOUT THE AGENT'S PERCEPTION CHANGES: the policy and the
        world model read the 128px observation, which never passes through
        here. This is only how many bytes carry the picture to the labeller.

        CROP BEFORE RESIZE, always: the fovea crop is taken from the full
        frame by _crop_center and encoded separately, so it keeps the render
        resolution. Resizing first and cropping second would silently shrink
        the fovea — the fovea exists because a downscaled whole frame could
        not tell a zombie from a skeleton.
        """
        try:
            f = np.asarray(frame)
            if f.ndim == 1:
                n = f.size // 3
                side = int(np.sqrt(n))
                f = f.reshape(3, side, side).transpose(1, 2, 0)
                f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
            if f.dtype != np.uint8:
                f = np.clip(f * 255.0 if f.max() <= 1.0 else f,
                            0, 255).astype(np.uint8)
            if f.shape[0] < 128:
                k = max(1, 256 // f.shape[0])
                f = np.repeat(np.repeat(f, k, 0), k, 1)
            _ms = int(max_side or 0)
            if _ms > 0 and max(f.shape[:2]) > _ms:
                # integer-factor box downsample: no interpolation library, no
                # new dependency, and an exact block mean rather than a
                # resampling filter whose kernel we would then have to reason
                # about when the labels move.
                k = int(np.ceil(max(f.shape[:2]) / float(_ms)))
                h, w = f.shape[0] // k * k, f.shape[1] // k * k
                if h >= k and w >= k:
                    f = (f[:h, :w].reshape(h // k, k, w // k, k, -1)
                         .mean(axis=(1, 3)).astype(np.uint8))
            import imageio.v2 as imageio
            buf = io.BytesIO()
            imageio.imwrite(buf, f, format="png")
            return buf.getvalue()
        except Exception:
            return None

    @staticmethod
    def _crop_center(frame, frac: float):
        """Centre crop as HWC uint8, accepting every frame format
        `_encode_png` accepts (flat CHW floats, CHW, HWC, float or uint8)."""
        try:
            f = np.asarray(frame)
            if f.ndim == 1:
                n = f.size // 3
                side = int(np.sqrt(n))
                f = f.reshape(3, side, side).transpose(1, 2, 0)
            elif f.ndim == 3 and f.shape[0] == 3 and f.shape[2] != 3:
                f = f.transpose(1, 2, 0)
            if f.dtype != np.uint8:
                f = np.clip(f * 255.0 if f.max() <= 1.0 else f,
                            0, 255).astype(np.uint8)
            h, w = f.shape[:2]
            c = max(8, int(min(h, w) * max(0.1, min(1.0, frac))))
            y0 = (h - c) // 2
            x0 = (w - c) // 2
            return f[y0:y0 + c, x0:x0 + c]
        except Exception:
            return None

    def set_crop_encoder(self, fn) -> None:
        """Supply `crop_hwc_uint8 -> [1, fovea_latent_dim]` features.

        The OWNER of the world model supplies this, because turning a patch
        into the encoder's input is the observation contract's business
        (resize to image_size, /255, CHW, flatten) and this class has no
        business knowing it. Passing None reverts to whole-frame latents.
        """
        self._crop_encode = fn

    def _refresh_fovea_latent(self, frame, timestep: int):
        """Encode THIS step's centre crop, at most once per step.

        Called from maybe_label, which runs every step and is the only place
        holding the frame. Cached because two consumers want it in the same
        step (the label pairing and the magnet's per-step read) and an
        encoder forward is not free.
        """
        if self._crop_encode is None or not self.fovea_enabled:
            return None
        if self._fovea_lat_step == int(timestep) and \
                self._fovea_lat_cache is not None:
            return self._fovea_lat_cache
        try:
            crop = self._crop_center(frame, self.fovea_frac)
            if crop is None:
                return None
            lat = self._crop_encode(crop)
            if lat is None:
                return None
            self._fovea_lat_cache = lat.detach()
            self._fovea_lat_step = int(timestep)
            return self._fovea_lat_cache
        except Exception:
            # a failed encode costs this step's foveal reading, never the run
            return None

    def _fovea_query(self, png_bytes: bytes) -> Optional[str]:
        try:
            client = _get_ollama_client()
            if client is None:
                return None
            resp = client.generate(
                model=self.model, prompt=_FOVEA_PROMPT, images=[png_bytes],
                format="json", keep_alive=-1,
                options={"num_predict": 128, "temperature": 0.0})
            return (resp.get("response") or "").strip()
        except Exception as e:
            logger.debug("VLMSymbolizer fovea query failed: %s", e)
            return None

    def _assess_fovea(self, png: bytes) -> Optional[Dict]:
        """Worker-thread path for a FOVEAL label. Same failure accounting as
        `_assess`; result is tagged so `collect()` can route it to the fovea
        head instead of the full-frame head / fact pipeline."""
        text = self._fovea_query_fn(png)
        if not text:
            self.total_query_failures += 1
            return None
        obj = _parse_json_object(text)
        if not isinstance(obj, dict):
            self.total_parse_failures += 1
            return None
        out = {}
        for p in FOVEA_PREDICATES:
            if p in obj:
                out[p] = bool(obj[p])
        if not out:
            self.total_empty_labels += 1
            return None
        out["__fovea__"] = True
        return out

    def _assess(self, png: bytes) -> Optional[Dict]:
        """Runs on the WORKER thread. Counts its own failure modes here —
        `collect()` cannot tell them apart, because everything used to
        collapse into a bare `None` and `parse_failures` sat at 0 forever
        whether the VLM was healthy, erroring, or dead.

        Thread-safety: these three counters are the ONLY worker-thread writes
        in the class. They are independent `int += 1` on attributes no other
        thread writes, and are read only for logging, so a lost update under
        CPython would cost an off-by-one in a diagnostic — not correctness.
        """
        text = self._query_fn(png)
        if not text:
            self.total_query_failures += 1      # server down / timed out
            return None
        obj = _parse_json_object(text)
        if not isinstance(obj, dict):
            self.total_parse_failures += 1      # answered, but not JSON
            return None
        # keep only known predicates with bool-ish values
        out = {}
        for p in PREDICATES:
            if p in obj:
                out[p] = bool(obj[p])
        if not out:
            self.total_empty_labels += 1        # valid JSON, no known keys
            return None
        # LAYER 3: carry the VLM's proposed category through, under a key that
        # can never collide with a predicate. It is a HYPOTHESIS about where to
        # look — the scaffold puts it on probation and learning progress
        # decides whether it survives. Deliberately NOT a predicate: the label
        # dict trains the grounded head and mints facts, and a guess must never
        # become either a fact or a training target.
        nov = obj.get("novel")
        if isinstance(nov, str):
            nov = nov.strip().lower().replace(" ", "_")
            if (nov and nov.endswith("_visible") and nov not in PREDICATES
                    and nov.replace("_", "").isalnum() and len(nov) <= 40):
                out["__novel__"] = nov
        return out

    # ---- main hooks --------------------------------------------------------

    def maybe_label(self, frame, latent: torch.Tensor, timestep: int) -> None:
        """Submit a frame for labelling if due. Pairs the CURRENT latent with
        it so the head trains on the state that produced the frame."""
        # Reliability REGRESSION-TO-PRIOR (audit fix): absent evidence, each
        # predicate's reliability drifts slowly back toward its 0.7 prior
        # (time constant ~2000 steps). Predicates with real evidence flowing
        # are unaffected (the alpha=0.1 event updates dominate); a predicate
        # that fell below reliability_floor on a bad streak is no longer
        # PERMANENTLY muted — the floor-crossing stops being absorbing.
        for _p, _r in self.reliability.items():
            self.reliability[_p] = _r + 0.0005 * (0.7 - _r)
        if not self._available or frame is None:
            return
        # BEFORE any early return: the magnet reads the foveal probs EVERY
        # step, and the busy-check below skips most steps while a ~1s query
        # is in flight. Refreshing here (not inside the submit branch) is
        # what keeps the per-step reading on crop features.
        self._refresh_fovea_latent(frame, timestep)
        if self._channel.busy():
            return
        # PER-CHANNEL SCHEDULER (2026-08-08, replaces strict alternation):
        # each channel is due on its OWN cadence — the full channel on the
        # ANNEALED interval, the fovea on its fixed one — and the single
        # worker serves whichever is more overdue relative to its cadence.
        # Ties go to the full channel (it grounds the main head and mints
        # facts; at cold boot that is the right first label).
        due_full = (timestep - self._last_full_submit) >= self.interval
        due_fovea = (self.fovea_enabled
                     and (timestep - self._last_fovea_submit)
                     >= self.fovea_interval)
        if not due_full and not due_fovea:
            return
        pick_fovea = due_fovea and (
            not due_full
            or (timestep - self._last_fovea_submit)
            / max(1.0, float(self.fovea_interval))
            > (timestep - self._last_full_submit)
            / max(1.0, float(self.interval)))
        if pick_fovea:
            crop = self._crop_center(frame, self.fovea_frac)
            # The CROP is already small (fovea_frac of the render) and is
            # the high-detail channel — never downscale it. See _encode_png.
            png = self._encode_png(crop) if crop is not None else None
            if png is None:
                return
            if self._channel.submit(self._assess_fovea, png):
                self._last_submit = timestep
                self._last_fovea_submit = timestep
                # Pair the label with the CROP's own features when we have
                # them, so the head is trained on the same region the VLM
                # was shown. Falls back to the whole-frame latent (old,
                # broken-but-compatible behaviour) when no crop encoder.
                _cl = self._refresh_fovea_latent(frame, timestep)
                self._pending_latent = (_cl.clone() if _cl is not None
                                        else latent.detach().clone())
                self._pending_is_fovea = True
            return
        png = self._encode_png(frame, max_side=self.input_max_side)
        if png is None:
            return
        if self._channel.submit(self._assess, png):
            self._last_submit = timestep
            self._last_full_submit = timestep
            self._pending_latent = latent.detach().clone()
            self._pending_is_fovea = False

    def collect(self) -> Optional[Dict[str, bool]]:
        """Absorb a finished label (non-blocking). Trains the head on the
        latent that was paired with the submitted frame."""
        labels = self._channel.poll()
        if labels is None:
            # Either nothing is ready yet (the common case — this runs every
            # step while a ~1s query is in flight) or the job failed and
            # _assess already recorded WHICH failure it was. Do NOT clear
            # _pending_latent here: on the in-flight path that would destroy
            # the latent paired with the frame currently being labelled.
            # A failed job cannot cross-pair either, because submit() refuses
            # while a future is running OR done-but-unpolled, so the next
            # submit always sets a fresh latent.
            return None
        if labels.pop("__fovea__", False):
            # FOVEAL label: trains ONLY the fovea head, and returns None so
            # the caller's fact-minting / novel-proposal path never runs on a
            # crop (a patch label describes a patch, not the scene).
            self.total_fovea_labels += 1
            self.last_fovea_labels = dict(labels)
            lat = self._pending_latent
            self._pending_latent = None
            if lat is not None and self.fovea_head is not None:
                self._train_fovea_head(lat, labels)
            return None
        self.total_labels += 1
        self.last_labels = labels
        lat = self._pending_latent
        self._pending_latent = None
        if lat is not None:
            self._train_head(lat, labels)
        return labels

    def _train_head(self, latent: torch.Tensor, labels: Dict[str, bool]):
        target = torch.zeros(1, len(PREDICATES), device=self.device)
        mask = torch.zeros(1, len(PREDICATES), device=self.device)
        for p, v in labels.items():
            i = PREDICATE_INDEX.get(p)
            if i is None:
                continue
            target[0, i] = 1.0 if v else 0.0
            mask[0, i] = 1.0
            self.label_counts[p] = self.label_counts.get(p, 0) + 1
            if v:
                self.pos_counts[p] = self.pos_counts.get(p, 0) + 1
        lat = latent.reshape(1, -1).to(self.device)
        # agreement BEFORE the update = how well the head already knew this
        with torch.no_grad():
            pred = (torch.sigmoid(self.head(lat)) > 0.5).float()
            agree = float(((pred == target) * mask).sum() /
                          mask.sum().clamp(min=1.0))
        self._recent_agreement.append(agree)
        # ---- CLASS IMBALANCE, MAIN HEAD (2026-09-01) --------------------
        # The FOVEA head has been getting `pos_weight` since 2026-08-12, with
        # a comment explaining exactly why: at a ~9-12% positive rate plain
        # BCE is minimised by answering "no" forever, and the head duly
        # learned to (agreement 0.876 vs 0.8757 for the trivial always-no
        # head). The main head — 29 predicates, several of them far RARER
        # than that (creeper_visible, tool_worn, water_visible) — was never
        # given the same treatment, though `label_counts`/`pos_counts` were
        # already being tracked for it. Same computation, same clamp.
        pw = torch.ones(1, len(PREDICATES), device=self.device)
        for _i, _p in enumerate(PREDICATES):
            _n = int(self.label_counts.get(_p, 0))
            _pos = int(self.pos_counts.get(_p, 0))
            if _n > 0 and _pos > 0:
                pw[0, _i] = min(20.0, max(1.0, (_n - _pos) / float(_pos)))
        self.last_head_loss = self.head.train_step(
            lat, target, mask, pos_weight=pw)
        # KEEP THE LABEL. One step at batch size 1 was all a ~1.5 s VLM call
        # ever bought; the replay lets every later update see it again.
        self.replay.add(lat, target, mask)
        if self.replay_steps > 0:
            self.head.train_replay(
                self.replay, self.replay_batch, self.replay_steps,
                pos_weight=pw, device=self.device)
        self.total_head_steps += 1
        self._maybe_anneal()

    def _train_fovea_head(self, latent: torch.Tensor,
                          labels: Dict[str, bool]) -> None:
        target = torch.zeros(1, len(FOVEA_PREDICATES), device=self.device)
        mask = torch.zeros(1, len(FOVEA_PREDICATES), device=self.device)
        for p, v in labels.items():
            i = FOVEA_INDEX.get(p)
            if i is None:
                continue
            target[0, i] = 1.0 if v else 0.0
            mask[0, i] = 1.0
            self.fovea_label_counts[p] = self.fovea_label_counts.get(p, 0) + 1
            if v:
                self.fovea_pos_counts[p] = (
                    self.fovea_pos_counts.get(p, 0) + 1)
        lat = latent.reshape(1, -1).to(self.device)
        with torch.no_grad():
            pred = (torch.sigmoid(self.fovea_head(lat)) > 0.5).float()
            agree = float(((pred == target) * mask).sum()
                          / mask.sum().clamp(min=1.0))
        self._fovea_agreement.append(agree)
        # CLASS IMBALANCE (2026-08-12). The foveal positive rate is ~9-12%,
        # so plain BCE is minimised by answering "no" forever — which is
        # exactly what the head learned (agreement 0.876 vs 0.8757 for the
        # trivial always-no head). Weight each predicate's positive term by
        # its own measured neg/pos ratio so a miss costs what it should.
        # Clamped: an unseen predicate would otherwise produce an infinite
        # weight the first time it appears.
        pw = torch.ones(1, len(FOVEA_PREDICATES), device=self.device)
        for _i, _p in enumerate(FOVEA_PREDICATES):
            _n = int(self.fovea_label_counts.get(_p, 0))
            _pos = int(self.fovea_pos_counts.get(_p, 0))
            if _n > 0 and _pos > 0:
                pw[0, _i] = min(20.0, max(1.0, (_n - _pos) / float(_pos)))
        self.fovea_head.train_step(lat, target, mask, pos_weight=pw)
        # Same replay treatment as the main head — the fovea channel labels
        # twice as often but each label is still a 1.5 s VLM call.
        if self.fovea_replay is not None:
            self.fovea_replay.add(lat, target, mask)
            if self.replay_steps > 0:
                self.fovea_head.train_replay(
                    self.fovea_replay, self.replay_batch, self.replay_steps,
                    pos_weight=pw, device=self.device)

    def fovea_probs(self, latent: torch.Tensor) -> Optional[Dict[str, float]]:
        """Per-step P(object under my gaze) from the agent's own latent —
        the magnet's centring signal. None when the channel is off. The
        caller gates per-category trust via `fovea_label_counts` (same
        no-evidence-no-assertion rule as everything else)."""
        if self.fovea_head is None:
            return None
        try:
            # CROP FEATURES WHEN AVAILABLE (2026-08-12): inference must read
            # the same thing training reads, or the head is evaluated on a
            # distribution it never saw. `latent` stays the fallback so a
            # config without a crop encoder behaves exactly as before.
            lat = self._fovea_lat_cache if self._crop_encode is not None \
                else None
            if lat is None:
                if latent is None:
                    return None
                lat = latent
            probs = self.fovea_head.predict(
                lat.reshape(1, -1).to(self.device)).squeeze(0)
            return {p: float(probs[i])
                    for i, p in enumerate(FOVEA_PREDICATES)}
        except Exception:
            return None

    def _maybe_anneal(self) -> None:
        """As the head learns to name things itself, ask the VLM less often —
        the teacher steps back. And if the head later STOPS agreeing, the
        teacher comes back: annealing used to be a one-way ratchet, so a head
        that degraded (distribution shift into a new biome, a latent that
        drifted as the RSSM kept training) could never recall its teacher.
        """
        if len(self._recent_agreement) < self._recent_agreement.maxlen:
            return
        agr = float(np.mean(self._recent_agreement))
        if agr >= self.anneal_agreement and self.interval < self.max_interval:
            # +1 floor: int(1 * 1.5) == 1, so a small base interval would be a
            # fixed point and the teacher could never step back.
            self.interval = min(self.max_interval,
                                max(self.interval + 1,
                                    int(self.interval * 1.5)))
            self._recent_agreement.clear()
            logger.info("VLMSymbolizer: head agreement %.2f — VLM interval "
                        "widened to %d steps", agr, self.interval)
        elif agr < self._deanneal_agreement and self.interval > self.base_interval:
            # DE-ANNEAL: agreement fell well below the widening gate, so the
            # head no longer deserves the autonomy it earned. Halve back
            # toward (never below) the base interval.
            self.interval = max(self.base_interval, int(self.interval / 1.5))
            self._recent_agreement.clear()
            logger.info("VLMSymbolizer: head agreement fell to %.2f — VLM "
                        "interval narrowed back to %d steps", agr,
                        self.interval)

    # ---- grounding / verification -----------------------------------------

    def observe_event(self, event_kind: str, detail: str,
                      was_chopping: bool,
                      now: Optional[int] = None
                      ) -> List[Tuple[str, str, str, float]]:
        """Feed a GROUND-TRUTH event (e.g. a block break from mine_block
        stats). Two jobs:
          1. score predicate reliability — a break proves something breakable
             really was in reach, so `breakable_in_reach` earns/loses credit
             against what the VLM last claimed;
          2. mint a CAUSAL fact, which is never taken on the VLM's word.
        Returns triples (subject, relation, object, confidence).

        `now` (optional, the caller's symbol clock): reliability is scored ONLY
        against a FRESH label (audit fix). last_labels can be up to `interval`
        steps stale (60 annealing to 600); scoring tree_visible against a
        600-step-old frame's label produced streaks of false misses that drove
        the goal category below reliability_floor — an absorbing trap, since
        recovery needed the very log breaks the muted magnet/seek could no
        longer steer toward. Causal facts (ground truth) are always minted.
        """
        facts: List[Tuple[str, str, str, float]] = []
        if event_kind != "block_break":
            return facts

        # 1. verification — only against a label young enough to describe the
        # CURRENT scene (the label describes the frame captured at
        # _last_submit; 2x base_interval is one full labeling cycle of slack).
        fresh = self._label_fresh(now)
        if fresh:
            # the world just proved a breakable was in reach
            for p in ("breakable_in_reach", "object_centered",
                      "object_adjacent"):
                claimed = bool(self.last_labels.get(p, False))
                self._update_reliability(p, hit=claimed)
            if "log" in detail:
                self._update_reliability(
                    "tree_visible",
                    hit=bool(self.last_labels.get("tree_visible")))

        # 2. causal fact from ground truth (confidence 1.0 — observed, not
        #    claimed). This is what rule induction actually needs.
        if was_chopping:
            facts.append(("attack_held", "breaks", str(detail), 1.0))
        facts.append((str(detail), "is", "breakable", 1.0))
        self.total_facts += len(facts)
        return facts

    def _label_fresh(self, now: Optional[int]) -> bool:
        """Is last_labels young enough to describe the CURRENT scene?
        Keys on the last FULL-frame submit — the fovea channel never writes
        last_labels, so with alternation _last_submit may describe a crop."""
        _ref = (self._last_full_submit
                if getattr(self, "fovea_enabled", False)
                else self._last_submit)
        # 2x base interval of slack. (The alternation-era x2 allowance is
        # gone: with per-channel scheduling the full channel keeps its own
        # cadence, so full labels are no more stale than pre-fovea.)
        return (now is None or (now - _ref) <= 2 * self.base_interval)

    def observe_negative_event(self, now: Optional[int] = None) -> int:
        """GROUND-TRUTH NEGATIVE evidence (2026-08-07): the agent held attack
        for a full break-worth of ticks and NOTHING broke.

        This is the missing half the module docstring warned about — the
        break-event verifier detects false NEGATIVES only, so a predicate
        stuck at True drifted to reliability 1.0 forever (llava's stuck-true
        tree_visible went undetected for weeks exactly this way). A long
        fruitless swing is the world contradicting the claimed affordance:
        whatever the VLM said was breakable/centred/adjacent was not.

        Only predicates the fresh label actually CLAIMED are docked — an
        unclaimed predicate is not contradicted by the miss. Gentler alpha
        than a positive hit: one stuck swing is weaker evidence than a
        completed break. Returns how many predicates were docked."""
        if not self._label_fresh(now):
            return 0
        n = 0
        for p in ("breakable_in_reach", "object_centered",
                  "object_adjacent"):
            if bool(self.last_labels.get(p, False)):
                self._update_reliability(p, hit=False, alpha=0.05)
                n += 1
        return n

    def observe_truth(self, world_info: Dict, latent: torch.Tensor) -> int:
        """Score the head against GROUND TRUTH the env already emits.

        ---- CLOSING PART OF THE VERIFIER'S BLIND SPOT (2026-09-01) --------
        The module docstring is honest that only 4 of 16 predicates are ever
        scored, and that the rest can never be retracted however wrong they
        are. But four of the unscored ones are not judgement calls at all —
        `pitch`, `gui_open` and `mainhand` arrive every step and settle
        looking_up / looking_down / inventory_visible / holding_tool exactly.

        This scores the HEAD (not the VLM) because the head is what emits
        facts. A predicate the head gets wrong about the agent's own body
        loses reliability and stops asserting, which is the same mechanism
        the block-break events already drive — just pointed at evidence that
        was sitting unused.

        Cheap and unconditional: one no-grad forward the caller already has a
        latent for, four dict lookups. Returns how many predicates scored.
        """
        if not isinstance(world_info, dict) or not world_info:
            return 0
        try:
            probs = self.head.predict(
                latent.reshape(1, -1).to(self.device)).squeeze(0)
        except Exception:
            return 0
        n = 0
        for p, fn in _TELEMETRY_TRUTH.items():
            i = PREDICATE_INDEX.get(p)
            if i is None:
                continue
            try:
                truth = fn(world_info)
            except Exception:
                truth = None
            if truth is None:
                continue          # the world cannot answer; never a miss
            said = bool(float(probs[i]) >= 0.5)
            self._update_reliability(p, said == bool(truth))
            # These count as observations, so the min_labels gate opens for
            # them on evidence rather than on VLM attention alone.
            self.label_counts[p] = self.label_counts.get(p, 0) + 1
            if truth:
                self.pos_counts[p] = self.pos_counts.get(p, 0) + 1
            self.telemetry_scored = getattr(self, "telemetry_scored", 0) + 1
            n += 1
        return n

    def _update_reliability(self, predicate: str, hit: bool,
                            alpha: float = 0.1) -> None:
        if predicate not in self.reliability:
            return
        prev = self.reliability[predicate]
        self.reliability[predicate] = (1 - alpha) * prev + alpha * (
            1.0 if hit else 0.0)

    def perceptual_facts(self, latent: torch.Tensor
                         ) -> List[Tuple[str, str, str, float]]:
        """Facts the agent asserts FROM ITS OWN LATENT (no VLM call). Each is
        gated by both head confidence and the predicate's earned reliability,
        so a hallucination-prone predicate quietly stops emitting."""
        out: List[Tuple[str, str, str, float]] = []
        _lat = latent.reshape(1, -1).to(self.device)
        probs = self.head.predict(_lat).squeeze(0)
        # ---- ENSEMBLE DISAGREEMENT AS A FACT GATE (2026-09-01) ----------
        # `fact_threshold` reads ONE sigmoid, which cannot distinguish "the
        # head is confident" from "the head has collapsed to a constant" —
        # and a collapsed predicate riding to reliability 1.0 unopposed is
        # the documented llava failure. Spread across independently trained
        # members is the missing axis. All-ones when n_members == 1, so a
        # single-head config is unchanged and honestly reports no
        # uncertainty rather than a fabricated one.
        _dis = (self.head.disagreement(_lat)
                if self.head.n_members > 1 else None)
        if _dis is not None:
            self.last_disagreement = {
                p: float(_dis[i]) for i, p in enumerate(PREDICATES)}
        for i, p in enumerate(PREDICATES):
            if (_dis is not None
                    and float(_dis[i]) > self.max_disagreement):
                continue          # the ensemble does not agree: assert nothing
            if self.label_counts.get(p, 0) < self.min_labels:
                continue          # never observed enough to assert anything
            rel = self.reliability.get(p, 0.0)
            if rel < self.reliability_floor:
                continue
            # falsifiability ceiling: a predicate with no disconfirming
            # evidence source may never be MORE trusted than its prior
            # (see FALSIFIABLE_PREDICATES above)
            if p not in FALSIFIABLE_PREDICATES:
                rel = min(rel, _UNFALSIFIABLE_RELIABILITY_CAP)
            conf = float(probs[i]) * rel
            if float(probs[i]) >= self.fact_threshold:
                subj, relation, obj = _FACT_TEMPLATES[p]
                out.append((subj, relation, obj, conf))
        self.total_facts += len(out)
        return out

    def stale_facts(self) -> List[Tuple[str, str, str]]:
        """Triples that must be RETRACTED from the knowledge graph.

        `perceptual_facts` gates EMISSION, which only ever stops NEW facts. A
        predicate that crossed `fact_threshold` once left its triple resident
        forever, still feeding the graph embedding long after the world had
        discredited it. This is the missing other half: when a predicate's
        earned reliability falls below the floor, its triple is handed back
        for deletion.

        EDGE-triggered — each collapse yields the triple exactly once. If the
        predicate later earns its reliability back (floor + _REARM_MARGIN) it
        is re-armed and `perceptual_facts` may assert it again.
        """
        out: List[Tuple[str, str, str]] = []
        for p in PREDICATES:
            rel = self.reliability.get(p, 0.0)
            if rel < self.reliability_floor:
                if p not in self._retracted:
                    self._retracted.add(p)
                    triple = _FACT_TEMPLATES.get(p)
                    if triple is not None:
                        out.append(triple)
            elif rel >= self.reliability_floor + _REARM_MARGIN:
                self._retracted.discard(p)
        self.total_retracted += len(out)
        return out

    # ---- PERCEPTION PERSISTENCE (2026-08-10) ------------------------------
    # The grounded heads, reliability EMAs and label evidence died with every
    # process, so each boot re-ran hours of re-grounding: 21/28 predicates
    # DEGENERATE at cold start, the magnet idling on an untrained head, the
    # social certification reset to zero, and a fresh symbols-novelty
    # windfall. Perception is EXPERIENCE — it persists like the skill bank.

    def state(self) -> Dict:
        """Everything earned that a restart should not destroy."""
        return {
            "predicates": list(PREDICATES),
            "fovea_predicates": list(FOVEA_PREDICATES),
            "head": self.head.state_dict(),
            "fovea_head": (self.fovea_head.state_dict()
                           if self.fovea_head is not None else None),
            "fovea_latent_dim": int(self.fovea_latent_dim),
            "latent_dim": int(self.latent_dim),
            # HEAD ARCHITECTURE SIGNATURE (2026-09-01). The head became an
            # ensemble of configurable depth, so its state-dict KEYS moved
            # (`net.0.weight` -> `members.0.0.weight`) and its shapes follow
            # head_hidden/head_layers. The file's own note says torch copies
            # matching tensors BEFORE raising on a mismatch — so an
            # unguarded load leaves a half-restored hybrid, which for a
            # perception system is the worst possible failure. Guarded the
            # same way vocabulary and input width already are.
            "head_arch": [int(self.head.n_members), int(self.head.n_layers),
                          int(self.head_hidden)],
            "reliability": dict(self.reliability),
            "label_counts": dict(self.label_counts),
            "pos_counts": dict(self.pos_counts),
            "fovea_label_counts": dict(self.fovea_label_counts),
            "fovea_pos_counts": dict(self.fovea_pos_counts),
            "interval": int(self.interval),
            "total_labels": int(self.total_labels),
            "total_fovea_labels": int(self.total_fovea_labels),
            "retracted": sorted(self._retracted),
        }

    def load_state(self, st: Dict) -> str:
        """Defensive restore. HEADS load only when the vocabulary matches
        exactly (a grown vocabulary means new output rows — partial surgery
        would silently misalign predicates, the worst possible failure for a
        perception system); the evidence DICTS merge on common keys always,
        so even across a vocabulary change the earned counts and reliability
        survive. Returns a one-line summary for the log."""
        out = []
        _head_ok = False
        _fovea_ok = False
        try:
            # WIDTH as well as VOCABULARY (2026-08-14): the head reads the
            # RSSM latent, so scaling the world model changes its input
            # width. torch copies matching tensors BEFORE raising on a
            # mismatch, so attempting the load would leave a half-restored
            # hybrid rather than a clean fresh head — the same trap already
            # guarded on the fovea head.
            _want_w = int(st.get("latent_dim", self.latent_dim))
            _want_arch = list(st.get("head_arch") or [1, 3, self.head_hidden])
            _live_arch = [int(self.head.n_members), int(self.head.n_layers),
                          int(self.head_hidden)]
            if (list(st.get("predicates") or []) == list(PREDICATES)
                    and _want_w == int(self.latent_dim)
                    and _want_arch == _live_arch):
                self.head.load_state_dict(st["head"])
                _head_ok = True
                out.append("head")
            elif _want_w != int(self.latent_dim):
                out.append(f"head=FRESH(latent {_want_w}->{self.latent_dim})")
            elif _want_arch != _live_arch:
                out.append(f"head=FRESH(arch {_want_arch}->{_live_arch})")
            else:
                out.append("head=FRESH(vocab changed)")
            if (self.fovea_head is not None and st.get("fovea_head")
                    and list(st.get("fovea_predicates") or [])
                    == list(FOVEA_PREDICATES)
                    # INPUT WIDTH must match too (2026-08-12): the fovea head
                    # moved from the whole-frame RSSM latent to crop encoder
                    # features, so an older checkpoint carries a different
                    # input width. torch copies matching tensors BEFORE
                    # raising on a mismatch, so attempting this would leave a
                    # half-restored hybrid rather than a clean fresh head.
                    and int(st.get("fovea_latent_dim",
                                   self.fovea_latent_dim))
                    == int(self.fovea_latent_dim)):
                self.fovea_head.load_state_dict(st["fovea_head"])
                _fovea_ok = True
                out.append("fovea_head")
            elif self.fovea_head is not None and st.get("fovea_head"):
                out.append("fovea_head=FRESH(shape/vocab changed)")
            # EVIDENCE BELONGS TO THE HEAD THAT EARNED IT (2026-08-11, review
            # finding). label_counts and reliability are not trivia — they ARE
            # the grounding gate: a predicate with counts >= min_labels and
            # reliability >= floor is allowed to assert facts into the
            # knowledge graph at confidence p*reliability, and to mint
            # grounded symbols for reward. Restoring them onto a FRESH
            # (random) head therefore certifies noise as trusted perception
            # from step 1, and it does not self-correct: only a few
            # predicates ever receive disconfirming evidence, so reliability
            # drifts back toward its prior instead of falling under the
            # floor. A head that was not restored must re-earn its grounding.
            _pairs = ((("reliability", "label_counts", "pos_counts"), _head_ok),
                      (("fovea_label_counts", "fovea_pos_counts"), _fovea_ok))
            _kept = []
            for names, ok in _pairs:
                if not ok:
                    continue
                for name in names:
                    mine = getattr(self, name)
                    for k, v in (st.get(name) or {}).items():
                        if k in mine:
                            mine[k] = v
                    _kept.append(name)
            out.append("evidence" if _kept
                       else "evidence=DROPPED(no head restored)")
            # The ANNEAL is also head-specific: `interval` widens as the head
            # agrees with the teacher, so inheriting a wide interval for a
            # fresh head would starve it of the very labels it needs to
            # relearn. A fresh head restarts at the base cadence.
            if _head_ok:
                # CLAMP BOTH ENDS (2026-08-17). This clamped only the FLOOR,
                # so a restored anneal outlived the config that bounded it:
                # `max_interval` was lowered 600 -> 120 and the run still
                # printed `vlm_every=600`, because the checkpoint's widened
                # value was inherited unchecked. A tuning knob that a resume
                # silently ignores is worse than no knob — every run after
                # the first would have quietly kept the old cadence.
                self.interval = min(
                    self.max_interval,
                    max(self.base_interval,
                        int(st.get("interval", self.interval))))
                self.total_labels = int(st.get("total_labels", 0))
            else:
                self.interval = self.base_interval
                self.total_labels = 0
            if _fovea_ok:
                self.total_fovea_labels = int(
                    st.get("total_fovea_labels", 0))
            self._retracted = set(st.get("retracted") or [])
        except Exception as e:      # a bad checkpoint must never block boot
            out.append(f"partial({type(e).__name__})")
        return "+".join(out)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    @property
    def stats(self) -> Dict:
        return {
            "available": self._available,
            "labels": self.total_labels,
            "head_steps": self.total_head_steps,
            "head_loss": round(self.last_head_loss, 4),
            "agreement": (round(float(np.mean(self._recent_agreement)), 3)
                          if self._recent_agreement else None),
            "vlm_interval": self.interval,
            # predicates with enough VLM evidence to be allowed to assert
            "grounded_predicates": sum(1 for c in self.label_counts.values()
                                       if c >= self.min_labels),
            "n_predicates": len(PREDICATES),
            "facts": self.total_facts,
            "retracted": self.total_retracted,
            "parse_failures": self.total_parse_failures,
            "query_failures": self.total_query_failures,
            "empty_labels": self.total_empty_labels,
            "worst_predicate": (min(self.reliability.items(),
                                    key=lambda kv: kv[1])
                                if self.reliability else None),
            "fovea": ({
                "labels": self.total_fovea_labels,
                "agreement": (round(float(np.mean(self._fovea_agreement)), 3)
                              if self._fovea_agreement else None),
                "grounded": sum(1 for c in self.fovea_label_counts.values()
                                if c >= self.min_labels),
                "last": {k: int(v)
                         for k, v in self.last_fovea_labels.items()},
            } if self.fovea_enabled else None),
        }
