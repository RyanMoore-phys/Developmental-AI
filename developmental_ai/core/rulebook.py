"""RuleBook — intervention-verified rule mining + schema transfer (Stages 2-4
of the law-induction program; the "scientific method" loop).

Hypotheses are precondition->action->effect statements with BETA POSTERIORS
(verified/refuted counts from the agent's OWN interventions). Deliberately
imperfect, like human rule internalization: a hard cap of `max_trials`
experiments per hypothesis (no certainty-chasing), probabilistic acceptance
thresholds, and no exhaustive search. Rules are tables — the whole layer is
~zero-FLOP.

Two levels:

  RuleBook   — per-world evidence: counts for concrete events
               ("toggle door d carrying k" -> opened?; "step on floor c" ->
               hazard?). This is what the agent's experiments fill in.
  SchemaPrior — the ABSTRACT, world-invariant structure induced across
               training worlds: "exactly one floor color is hazardous",
               "door<-key is a permutation over colors". Transferring the
               SchemaPrior to a new world tells the agent WHAT experiments
               bind the world's laws fastest — not the bindings themselves.

The `scrambled()` lesion returns a valid-but-wrong SchemaPrior (permuted
inference outputs) — the project's signature control: if scramble ~= intact,
the schema is decorative; if scramble craters, it is load-bearing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class Beta:
    """Beta(1,1)-prior evidence counter."""

    __slots__ = ("pos", "neg")

    def __init__(self):
        self.pos = 0
        self.neg = 0

    @property
    def trials(self) -> int:
        return self.pos + self.neg

    @property
    def mean(self) -> float:
        return (self.pos + 1) / (self.trials + 2)

    def update(self, outcome: bool) -> None:
        if outcome:
            self.pos += 1
        else:
            self.neg += 1


class RuleBook:
    """Per-world evidence store, filled by the agent's interventions."""

    def __init__(self, key_colors: List[str], floor_colors: List[str],
                 max_trials: int = 6):
        self.key_colors = list(key_colors)
        self.floor_colors = list(floor_colors)
        self.max_trials = int(max_trials)
        # OPENS evidence: (door_color, key_color) -> Beta over "door opened"
        self.opens: Dict[Tuple[str, str], Beta] = {}
        # HAZARD evidence: floor_color -> Beta over "stepping terminated badly"
        self.hazard: Dict[str, Beta] = {}

    # ---- evidence intake (call from the interaction loop) ----
    def observe_toggle(self, door_color: str, key_color: Optional[str],
                       opened: bool) -> None:
        if key_color is None:
            return
        self.opens.setdefault((door_color, key_color), Beta()).update(opened)

    def observe_floor(self, floor_color: str, hazardous: bool) -> None:
        self.hazard.setdefault(floor_color, Beta()).update(hazardous)

    # ---- queries ----
    def p_opens(self, door_color: str, key_color: str) -> float:
        b = self.opens.get((door_color, key_color))
        return b.mean if b else 0.5

    def p_hazard(self, floor_color: str) -> float:
        b = self.hazard.get(floor_color)
        return b.mean if b else 0.5

    def needs_experiment(self, door_color: str, key_color: str) -> bool:
        """True while this hypothesis is unsettled AND under the trial cap —
        the imperfection principle: stop experimenting once confident-enough
        or once the budget is spent, never chase certainty."""
        b = self.opens.get((door_color, key_color))
        if b is None:
            return True
        if b.trials >= self.max_trials:
            return False
        # Loose band on purpose (imperfection principle): 3 consistent trials
        # settle a rule (Beta mean 0.8 / 0.2), like human rule internalization
        # — never chase certainty.
        return 0.25 < b.mean < 0.75

    def believed_key_for(self, door_color: str,
                         min_p: float = 0.6) -> Optional[str]:
        """The key this world's evidence says opens `door_color` (or None)."""
        best, best_p = None, min_p
        for k in self.key_colors:
            p = self.p_opens(door_color, k)
            b = self.opens.get((door_color, k))
            if b and b.trials > 0 and p > best_p:
                best, best_p = k, p
        return best

    def believed_hazards(self, min_p: float = 0.6) -> List[str]:
        return [
            c for c, b in self.hazard.items()
            if b.trials > 0 and b.mean > min_p
        ]

    def refuted_keys(self, door_color: str, max_p: float = 0.35) -> Set[str]:
        """Keys this world's evidence says do NOT open `door_color` —
        never worth retrying (a refuted hypothesis stays refuted until
        contrary evidence, which a trap-on-wrong-key design never gives)."""
        return {
            k for (d, k), b in self.opens.items()
            if d == door_color and b.trials > 0 and b.mean < max_p
        }


class SchemaPrior:
    """The abstract, cross-world invariants induced from training worlds —
    the transferable 'laws', held with tolerated imperfection."""

    def __init__(self):
        # P(structure) estimates, each a Beta over "held in a training world"
        self.one_hazard = Beta()        # exactly one hazardous floor color
        self.binding_is_perm = Beta()   # door<-key map is a 1-1 permutation
        self.wrong_key_inert = Beta()   # wrong-key toggle never opens

    # ---- induction from a completed world's rulebook + ground truth-free
    #      summary of what the agent EXPERIENCED ----
    def absorb_world(self, rb: RuleBook) -> None:
        hazards = rb.believed_hazards()
        # Only worlds that produced hazard EVIDENCE inform the one-hazard
        # structure — absence of evidence (never stepped on the hazard) is
        # not evidence of absence.
        if hazards:
            self.one_hazard.update(len(hazards) == 1)
        keys = [
            rb.believed_key_for(d)
            for d in rb.key_colors
            if rb.believed_key_for(d) is not None
        ]
        # permutation-ness among observed doors: no two doors share a key
        self.binding_is_perm.update(len(keys) == len(set(keys)))
        inert = [
            b.mean < 0.3
            for (d, k), b in rb.opens.items()
            if b.trials > 0 and rb.believed_key_for(d) not in (None, k)
        ]
        if inert:
            self.wrong_key_inert.update(all(inert))

    # ---- what the schema buys in a NEW world ----
    def hazard_experiment_plan(self, observed_colors: List[str],
                               rb: RuleBook) -> List[str]:
        """Colors still worth treating as potentially hazardous. Under the
        one-hazard schema, a single confirmed hazard clears every other
        color WITHOUT testing them — the schema's whole value: fewer costly
        experiments."""
        if self.one_hazard.mean > 0.6 and rb.believed_hazards():
            return []  # bound: everything else is safe by schema
        return [
            c for c in observed_colors
            if rb.hazard.get(c) is None or rb.hazard[c].trials == 0
        ]

    def key_candidates(self, door_color: str, rb: RuleBook) -> List[str]:
        """Keys worth testing on this door, best-first. Evidence level
        (shared by every arm): the believed binding short-circuits, refuted
        keys are excluded. Schema level (the arm difference): keys bound to
        OTHER doors are pruned when binding-is-permutation is believed."""
        believed = rb.believed_key_for(door_color)
        if believed:
            return [believed]
        dead = rb.refuted_keys(door_color)
        taken = set()
        if self.binding_is_perm.mean > 0.6:
            for d in rb.key_colors:
                if d != door_color:
                    k = rb.believed_key_for(d)
                    if k:
                        taken.add(k)
        return [k for k in rb.key_colors if k not in taken and k not in dead]

    def scrambled(self, seed: int = 0) -> "SchemaPrior":
        """Valid-but-wrong lesion: same confidence numbers, inference DELIBERATELY
        permuted — hazard plan tests exactly the colors the intact schema would
        skip, and key candidates are reversed (worst-first, believed excluded)."""
        rng = np.random.RandomState(seed)

        class _Scrambled(SchemaPrior):
            def hazard_experiment_plan(inner, observed_colors, rb):
                plan = SchemaPrior.hazard_experiment_plan(
                    self, observed_colors, rb)
                inverted = [c for c in observed_colors if c not in plan]
                return inverted if inverted else plan

            def key_candidates(inner, door_color, rb):
                # Same EVIDENCE as every arm (refuted keys excluded, believed
                # key known) — only the SCHEMA-level ordering is wrong-first.
                cands = SchemaPrior.key_candidates(self, door_color, rb)
                dead = rb.refuted_keys(door_color)
                usable = [k for k in rb.key_colors if k not in dead]
                wrong = [k for k in usable if k not in cands[:1]]
                rng.shuffle(wrong)
                return wrong + [k for k in cands[:1] if k in usable]

        s = _Scrambled()
        s.one_hazard = self.one_hazard
        s.binding_is_perm = self.binding_is_perm
        s.wrong_key_inert = self.wrong_key_inert
        return s
