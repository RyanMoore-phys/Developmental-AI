"""Consequence Frontier — interest as a deficit of CONSEQUENCE, and
possession as part of location.

THE PRINCIPLE (domain-independent; nothing here knows what a tree is)

    You are bored of what you have already CAUSED, and curious about what
    you have SEEN but never AFFECTED.

    ...and: WHERE YOU ARE includes WHAT YOU HAVE. Acquiring a kind of thing
    you have rarely held is moving somewhere new, in the same sense that
    walking onto new ground is.

WHY IT EXISTS (measured, 2026-08-12/13)
    The vision magnet ranks categories by LEARNING PROGRESS, so it steers
    toward whatever currently teaches it most: live LP was water 0.186,
    stone 0.143, tree 0.138 — so it went to stone, and the user watched the
    agent look straight at a tree, walk past it, and mine dirt in the
    distance. Prediction-error progress measures "how much am I learning
    ABOUT this", never "how much does this OPEN UP for me". Wood is visually
    dull and enormous in consequence, and nothing in the economy could
    represent that gap.

    Every other candidate mechanism died on the BOOTSTRAP test: interest
    derived from a graph built by experience, where getting the experience
    requires the interest, is a latch (this project has produced nine).
    Consequence deficit is the exception because THE EVIDENCE IS ALREADY
    BANKED: 4,340 dirt breaks with a closed consequence set, against ~0
    caused events for a category seen thousands of times. No first success
    is required.

TWO PARTS
    A. CONSEQUENCE DEFICIT (ranking only, ZERO income). Per category:
       seen a lot, caused little  ->  high deficit  ->  worth attending to.
    B. POSSESSION FRONTIER (income, count-decayed). Entering a rarely-held
       possession SET pays once, decaying as 1/sqrt(n). This is what makes
       a chain recede on its own — each rung is a set never occupied —
       without any written recipe or tech tree.

THREE STRUCTURAL RULES, each answering a defect that killed a rejected design

    1. RANKING, NEVER PAYING. The magnet's score is a PAYER-SCALER: it sets
       the shaping weight, that shaping is added into `prim_extrinsic`, and
       `prim_extrinsic` is written to the replay buffer AS THE WORLD MODEL'S
       REWARD LABEL. Injecting a derived quantity there would train the
       world model on the agent's own inference. Deficit therefore enters
       ONLY a target-selection tie-break and never a magnitude.
    2. CONTRASTIVE CREDIT. A predicate that reads "present" half the time
       would otherwise bank contact credit for every unrelated event and
       suppress exactly what it should promote. Credit is
       `max(0, p(c) - base(c))` against that category's OWN slow baseline,
       so only a genuine spike counts.
    3. PERSISTED. These counters ARE the bootstrap. Starting empty each boot
       would recreate the latch, so they round-trip through JSON.

DEFENSIVE CONTRACT (mirrors infra/affordance.py): never raises, errors are
data, memory bounded, and it only ever RETURNS numbers — the loop decides
what to do with them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class ConsequenceMap:
    """Per-category consequence deficit + possession-set frontier."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        # slow EMA of each category's presence: its OWN baseline
        self.base_beta = float(cfg.get("base_beta", 0.001))
        # minimum attention spike that counts as "attending to this"
        self.min_credit = float(cfg.get("min_credit", 0.05))
        # a category must have been SEEN this much before a deficit is
        # claimed — otherwise a never-observed predicate (or a broken
        # sensor) reads as maximally interesting
        self.min_seen = float(cfg.get("min_seen", 20.0))
        # weight of a FRUITLESS attempt vs an actual outcome (see observe).
        # <1 because a failed swing is weaker evidence than a break: it may
        # be bad aim rather than an inert target, so it should take several
        # to close what one success closes. 0.0 = outcomes only (the
        # original, which let unbreakable targets trap the magnet forever).
        self.attempt_scale = float(cfg.get("attempt_scale", 0.25))
        self.max_keys = int(cfg.get("max_keys", 512))
        # possession frontier
        self.possession_weight = float(cfg.get("possession_weight", 0.0))
        self.max_sets = int(cfg.get("max_sets", 4096))
        # how steeply a DEEPER rung out-pays a shallow one (see
        # possession_income). 2.0 => a 3-kind set pays 9x a 1-kind set.
        # 1.0 restores flat per-set payment.
        self.depth_exponent = float(cfg.get("depth_exponent", 2.0))
        self.max_depth = int(cfg.get("max_depth", 8))

        self._base: Dict[str, float] = {}      # slow EMA of p(c), every step
        self._seen: Dict[str, float] = {}      # Σ p(c): opportunity to act
        self._contact: Dict[str, float] = {}   # Σ contrastive credit at events
        self._set_counts: Dict[str, int] = {}  # possession-set visit counts
        self._set_depth: Dict[str, int] = {}   # goal-chain depth per set
        self._prev_items: Dict[str, float] = {}   # last inventory counts
        self._cur_set: Optional[str] = None
        self._suppress_until: int = -1         # respawn / granted-tool guard
        self.errors: List[str] = []
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    def _note(self, msg: str) -> None:
        self.last_error = str(msg)
        if len(self.errors) < 64:
            self.errors.append(str(msg))

    def _prune(self, d: Dict, cap: int) -> None:
        while len(d) > cap:
            d.pop(next(iter(d)))

    # ------------------------------------------------------------------ #
    # A. CONSEQUENCE DEFICIT
    # ------------------------------------------------------------------ #
    def observe(self, probs: Optional[Dict[str, float]],
                caused: bool = False, attempted: bool = False) -> None:
        """One step of evidence.

        `probs` is per-category presence (the fovea's read of what is under
        the gaze). `caused` is True on a step where the agent's OWN action
        produced a world event. Ambient change must never count: watching a
        tree sway is not affecting it.

        `attempted` is True when the agent ACTED ON something and nothing
        happened — a swing that ended with no break.

        WHY ATTEMPTS COUNT (2026-08-14, from a live trap). Crediting only
        successes means a category the agent CANNOT affect never closes.
        Measured: the magnet locked onto stone, which is unbreakable
        barehanded; the agent threw 400 fruitless swings (182 of them >=8
        ticks, max streak 202) and broke nothing, so stone kept a maximal
        deficit AND its learning progress, and held the magnet's attention
        indefinitely. "I tried this repeatedly and nothing happened" is
        genuine evidence about a thing — it explores the frontier just as
        surely as succeeding does, and a mechanism that ignores it can never
        let a dead end retire.

        Attempts are credited at `attempt_scale` (< 1) because they are
        WEAKER evidence than an outcome: one fruitless swing may be bad aim
        rather than an inert target, so it should take many to close what a
        single success closes.
        """
        if not self.enabled or not probs:
            return
        try:
            for c, p in probs.items():
                p = float(p)
                if p != p:                     # NaN
                    continue
                b = self._base.get(c, p)
                self._base[c] = b + self.base_beta * (p - b)
                self._seen[c] = self._seen.get(c, 0.0) + max(0.0, p)
                _w = (1.0 if caused
                      else (self.attempt_scale if attempted else 0.0))
                if _w > 0.0:
                    # CONTRASTIVE: only the part of attention that exceeds
                    # this category's own habitual level is credited, so an
                    # always-on predicate cannot bank contact for events
                    # that had nothing to do with it.
                    credit = max(0.0, p - self._base.get(c, 0.0))
                    if credit >= self.min_credit:
                        self._contact[c] = (self._contact.get(c, 0.0)
                                            + credit * _w)
            self._prune(self._base, self.max_keys)
            self._prune(self._seen, self.max_keys)
            self._prune(self._contact, self.max_keys)
        except Exception as exc:               # pragma: no cover
            self._note(f"observe failed: {exc!r}")

    def deficit(self, category: str) -> float:
        """How UNCONSUMMATED is this category? in [0, 1].

        1.0 = seen a great deal, never once affected.
        ~0  = every sighting has been followed by causing something.

        Returns 0.0 (not 1.0) for anything not yet seen enough: "I have no
        idea" must not read as "maximally interesting", or a dead sensor
        would capture the agent's attention permanently.
        """
        try:
            seen = float(self._seen.get(category, 0.0))
            if seen < self.min_seen:
                return 0.0
            contact = float(self._contact.get(category, 0.0))
            return float(max(0.0, min(1.0, 1.0 / (1.0 + contact))))
        except Exception as exc:
            self._note(f"deficit failed: {exc!r}")
            return 0.0

    def deficits(self) -> Dict[str, float]:
        return {c: self.deficit(c) for c in self._seen}

    # ------------------------------------------------------------------ #
    # B. POSSESSION FRONTIER
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(items) -> str:
        """A possession SET (kinds held), order- and count-independent.

        Counts are deliberately ignored: holding 3 logs rather than 1 is not
        a new kind of situation, and paying per unit would be a farm.
        """
        try:
            return "|".join(sorted({str(k) for k, v in dict(items).items()
                                    if v and float(v) > 0}))
        except Exception:
            return ""

    def suppress(self, until_step: int) -> None:
        """Ignore acquisitions until this step.

        Used after respawn/rebuild and for externally-granted tools: an item
        the agent did not obtain by acting is not a frontier it crossed, and
        paying for it would reward being handed things.
        """
        self._suppress_until = int(until_step)

    def possession_income(self, items, step: int) -> Tuple[float, bool]:
        """Income for ENTERING a rarely-held possession set.

        FIRST ENTRY ONLY. A set you have held before is not a frontier.

        The first cut of this used 1/sqrt(1+visits), by analogy with
        territory coverage — and the farm contract caught it immediately:
        sum(1/sqrt(k)) ~ 2*sqrt(n) DIVERGES, so dropping and re-taking one
        item paid 2.085 against a 0.2 first entry, i.e. a 10x farm reachable
        with two buttons. Coverage survives that form only because walking
        back to a cell costs real travel; an inventory toggle costs nothing.

        Making it strictly one-shot bounds LIFETIME possession income at
        possession_weight x (number of distinct sets ever discovered), which
        no cycle can inflate — you can only discover a set once. It also
        states the semantics honestly: this pays for CROSSING a frontier,
        not for standing on the far side.

        Deliberately NOT a potential either: a potential refunds itself when
        the set is left, so the first log would net zero — and that is the
        one payment that has to land.

        Returns (income, is_new_set).
        """
        if not self.enabled or self.possession_weight <= 0.0:
            return 0.0, False
        try:
            key = self._key(items)
            _now = {str(k): float(v) for k, v in dict(items).items()
                    if v and float(v) > 0}
            _prev, self._prev_items = dict(self._prev_items), _now
            _prev_key = self._cur_set
            if key == self._cur_set:
                return 0.0, False              # no transition, no payment
            self._cur_set = key
            if not key:
                return 0.0, False              # empty-handed is not a rung
            # ---- GOAL-ORIENTED DEPTH (2026-08-16) ----------------------
            # DEPTH MEANS "I USED WHAT I HAD TO MAKE SOMETHING NEW", not
            # "I am holding more kinds".
            #
            # The first cut counted distinct kinds, and live it paid 4.8 —
            # a quarter of a felled log — for holding
            # dirt|poppy|stick|wheat_seeds: incidental drops from breaking
            # grass, no chain at all. It rewarded VARIETY, so the agent
            # went back to grinding ground cover for pickups and the median
            # swing hold collapsed 20t -> 2t, undoing the persistence gain.
            #
            # The observable signature of a real rung is a TRANSFORMATION:
            # a new kind appears WHILE an existing kind is consumed. That
            # is what crafting looks like from the outside — planks cost
            # logs — and it is what picking a flower does not look like.
            # A chain only deepens through transformation; a plain pickup
            # restarts at depth 1 no matter how full the inventory is.
            #
            # Still DISCOVERED, not declared: nothing knows what consumes
            # what. Any domain where making something spends something else
            # scores identically.
            _gained = [k for k in _now if k not in _prev]
            _spent = [k for k in _prev
                      if _now.get(k, 0.0) < _prev.get(k, 0.0)]
            if _gained and _spent:
                _depth = int(self._set_depth.get(_prev_key, 1)) + 1
            else:
                _depth = 1                     # acquisition, not construction
            self._set_depth[key] = max(int(self._set_depth.get(key, 0)),
                                       _depth)
            self._prune(self._set_depth, self.max_sets)
            n = int(self._set_counts.get(key, 0))
            self._set_counts[key] = n + 1
            self._prune(self._set_counts, self.max_sets)
            if n > 0:
                return 0.0, False              # already discovered
            if int(step) <= int(self._suppress_until):
                return 0.0, True               # counted, never paid
            # ---- CHAIN DEPTH (2026-08-15, user directive) --------------
            # "What should reward SkyBot the most is delegating goals of
            #  getting wood -> crafting planks -> making a crafting table,
            #  and that should be rewarded by far more than collecting
            #  wood."
            #
            # A set reachable only AFTER assembling other kinds sits deeper
            # in the agent's own dependency chain, and depth is exactly what
            # distinguishes a multi-step achievement from an acquisition.
            # Payout is SUPERLINEAR in depth, so a third rung pays ~9x a
            # first — "by far more", as asked — while a first rung still
            # pays enough to be found.
            #
            # DISCOVERED, NOT DECLARED: depth is the number of distinct
            # kinds the set contains, read off the agent's own inventory.
            # Nothing here knows that planks come from logs, or that a
            # crafting table is worth having; a chain the agent finds in any
            # other order, or in another domain entirely, is scored the same
            # way. Still strictly one-shot per set, so depth multiplies a
            # payment that can only ever be collected once.
            depth = max(1, min(int(self.max_depth), int(_depth)))
            return (float(self.possession_weight)
                    * float(depth) ** float(self.depth_exponent)), True
        except Exception as exc:
            self._note(f"possession failed: {exc!r}")
            return 0.0, False

    # ------------------------------------------------------------------ #
    def report(self) -> str:
        try:
            d = self.deficits()
            if not d:
                return ""
            top = sorted(d.items(), key=lambda kv: -kv[1])[:4]
            return ("deficit " + ", ".join(f"{k}={v:.2f}" for k, v in top)
                    + f" | possession sets {len(self._set_counts)}")
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    def state(self) -> Dict[str, Any]:
        return {"base": dict(self._base), "seen": dict(self._seen),
                "contact": dict(self._contact),
                "set_counts": dict(self._set_counts)}

    def load_state(self, st: Dict[str, Any]) -> bool:
        """Merge, never replace. THESE COUNTERS ARE THE BOOTSTRAP — starting
        empty each boot would recreate the latch this design exists to
        avoid."""
        try:
            for attr, key in (("_base", "base"), ("_seen", "seen"),
                              ("_contact", "contact"),
                              ("_set_counts", "set_counts")):
                cur = getattr(self, attr)
                for k, v in (st.get(key) or {}).items():
                    cur[k] = (cur.get(k, 0) or 0) + v if key != "base" else v
            return True
        except Exception as exc:
            self._note(f"load_state failed: {exc!r}")
            return False

    def save(self, path: str) -> bool:
        try:
            tmp = str(path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state(), f)
            os.replace(tmp, path)
            return True
        except Exception as exc:
            self._note(f"save failed: {exc!r}")
            return False

    def load(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            with open(path) as f:
                return self.load_state(json.load(f))
        except Exception as exc:
            self._note(f"load failed: {exc!r}")
            return False
