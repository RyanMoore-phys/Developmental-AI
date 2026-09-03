"""SKILLS AS MEMORIES — practice, consolidation and decay.

WHY THIS EXISTS
  A skill mints ONCE, at its first-ever unlock, when (the loop's own comment
  says it) "the policy could barely do it". It is then FROZEN: `refresh_slots`
  states that bound slots are frozen for the run, "identity AND behaviour",
  and re-distillation only ever refills EMPTY slots. So a skill invoked a
  thousand times successfully is byte-identical afterwards. That is a
  RECORDING, not a memory.

  A memory is different in three ways, and this module supplies all three:

    1. PRACTICE      — using it successfully makes it better.
    2. CONSOLIDATION — the more it has been proven, the less it moves
                       (early plasticity, later stability).
    3. DECAY         — going unused makes it weaker, and eventually stale
                       enough that it should be re-derived rather than kept.

WHY FREEZING WAS THERE, AND HOW THIS KEEPS IT SAFE
  The freeze is not superstition. The meta-policy learns a value for
  "invoke slot k"; if slot k's behaviour changes underneath it, that value is
  stale and SMDP credit assignment degrades. Freezing is simply the crudest
  way to guarantee stationarity. Three mechanisms here keep drift bounded
  instead of forbidding it:

    * SELF-IMITATION ONLY. Updates happen ONLY on invocations that produced
      the skill's own grounded effect, and they pull the policy toward the
      actions it ACTUALLY TOOK on that occasion. The skill is moved toward
      something it already did successfully — the least disruptive direction
      that still constitutes learning. Failures never push it anywhere:
      "don't do that" is a far larger, far less identified change than
      "do more of what worked", and this is a live agent, not a benchmark.

    * TRUST REGION. After the update, the KL between the old and new action
      distributions on that same batch is measured. If it exceeds `kl_max`
      the new weights are blended back toward the old until it fits. The
      meta-value can go stale by at most a bounded amount per invocation.

    * CONSOLIDATION. The effective step size is scaled by (1 - strength), so
      a skill that has succeeded many times becomes progressively harder to
      move. Fresh skills are plastic; proven ones are stable. This is also
      what stops a long run from slowly wandering a good skill into a bad one.

  Everything is measurable: `stats()` reports per-slot strength, cumulative
  drift, update counts and rejected (trust-region-clipped) updates, so
  "are skills actually changing, and by how much" is a readable number rather
  than an inference.

DELIBERATELY NOT DONE HERE
  No reward is invented. Success is the SAME ground-truth signal the mastery
  ledger uses — the skill's own grounded effect key occurring while it was
  open — never a reward spike (the bank's own comment records that leaves,
  which pay 0.3, were logged as "spikes" 7 times in 350 invocations). If the
  effect ledger cannot score an invocation, this module abstains too rather
  than guessing.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaHead(nn.Module):
    """THE OWNED PART OF A SKILL (2026-08-09, infra #35).

    Minted skills were byte copies of the shared policy — measured cosine
    similarity 0.869-1.000 across the bank, three pairs bit-identical. A copy
    cannot be independently improved, evaluated or composed, so everything
    above it (practice, mastery, composition) was decorative. A skill is now

        logits = frozen_base(feats) + delta(feats)

    where `delta` is this small head, ZERO-INITIALIZED at mint: at birth the
    skill IS the base behaviour (which is honest — at first unlock the agent
    could only barely do the thing), and INDIVIDUATION is what practice
    produces, by training ONLY the delta on the skill's own successes. The
    delta's output magnitude is therefore a direct, per-skill measurement of
    how much the skill has become its own thing — the number that was
    previously unmeasurable.
    """

    def __init__(self, feat_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(feat_dim), int(hidden)), nn.Tanh(),
            nn.Linear(int(hidden), int(out_dim)))
        # identity at birth: the LAST layer is zeroed so delta(x) == 0
        # everywhere until practice moves it
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

logger = logging.getLogger(__name__)


class SkillPractice:
    """Per-slot practice, consolidation and decay for bound option skills.

    Lives inside the OptionExecutor's bank. Holds no policy of its own: it
    updates the RESIDENT actor module for a slot in place, then writes the
    improved weights back into the slot binding so they survive eviction.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        lr: float = 1e-4,
        kl_max: float = 0.02,
        min_steps: int = 4,
        epochs: int = 1,
        strength_up: float = 0.10,
        strength_down: float = 0.02,
        decay_per_segment: float = 0.01,
        stale_below: float = 0.15,
        max_traj: int = 256,
        device: str = "cpu",
    ):
        self.enabled = bool(enabled)
        self.lr = float(lr)
        self.kl_max = float(kl_max)
        self.min_steps = int(min_steps)
        self.epochs = max(1, int(epochs))
        self.strength_up = float(strength_up)
        self.strength_down = float(strength_down)
        self.decay_per_segment = float(decay_per_segment)
        self.stale_below = float(stale_below)
        self.max_traj = int(max_traj)
        self.device = device

        # (env, slot) -> [(feats, action), ...] for the invocation in flight.
        # Keyed by ENV as well as slot: scouts run options concurrently on
        # streams 1..N-1, and mixing two envs' trajectories into one update
        # would train a skill on interleaved evidence from two different
        # situations.
        self._traj: Dict[Tuple[int, int], List[Tuple[torch.Tensor, int]]] = {}
        # slot -> optimizer over that slot's resident actor
        self._opt: "OrderedDict[int, torch.optim.Optimizer]" = OrderedDict()
        # slot -> consolidation strength in [0, 1]
        self.strength: Dict[int, float] = {}
        # slot -> cumulative L2 parameter drift since bind (telemetry)
        self.drift: Dict[int, float] = {}
        self.updates: Dict[int, int] = {}
        # dream consolidation is counted SEPARATELY from waking practice:
        # folding them together would make "did this skill improve because it
        # worked, or because the world model said it would?" unanswerable —
        # and on a weak WM those are very different claims.
        self.dream_updates: Dict[int, int] = {}
        self.clipped: Dict[int, int] = {}
        self.abstained = 0

    # ---- trajectory capture ------------------------------------------------
    def record(self, env: int, slot: int, feats: torch.Tensor,
               action: int) -> None:
        """One (features, action) pair executed by a bound skill.

        FEATURES, not raw observations: under the shared-perception arch the
        actor's input is already a 256-d encoding, so storing that instead of
        49152 raw pixels makes an invocation's trajectory a few hundred KB
        rather than tens of MB — and means the update never has to re-run the
        encoder (which belongs to the world model and must not be touched).
        """
        if not self.enabled:
            return
        buf = self._traj.setdefault((int(env), int(slot)), [])
        if len(buf) < self.max_traj:
            buf.append((feats.detach().reshape(1, -1).cpu(), int(action)))

    def discard(self, env: int, slot: int) -> None:
        self._traj.pop((int(env), int(slot)), None)

    # ---- the practice step -------------------------------------------------
    def on_close(self, env: int, slot: int, actor: Any,
                 produced_effect: Optional[bool],
                 delta: Any = None) -> Optional[Dict[str, float]]:
        """An invocation ended. Practise it if it demonstrably worked.

        `produced_effect` is the mastery ledger's verdict: True (its own
        grounded effect occurred), False (it did not), or None (UNMEASURABLE —
        the skill's key is not something the environment has ever been
        observed to emit). None abstains, exactly as the ledger does: scoring
        an unmeasurable skill 0 forever is a latch, not a measurement.
        """
        traj = self._traj.pop((int(env), int(slot)), None)
        if not self.enabled or not traj:
            return None
        if produced_effect is None:
            self.abstained += 1
            return None
        if not produced_effect:
            # failure: weaken confidence, change no weights (see module docs)
            self.strength[slot] = max(
                0.0, self.strength.get(slot, 0.0) - self.strength_down)
            return None
        if len(traj) < self.min_steps:
            return None                      # too little evidence to learn from
        return self._self_imitate(slot, actor, traj, delta=delta)

    # ---- consolidation during dreaming -------------------------------------
    def consolidate(self, slot: int, actor: Any,
                    traj: List[Tuple[torch.Tensor, int]],
                    lr_scale: float = 1.0,
                    delta: Any = None) -> Optional[Dict[str, float]]:
        """Practise a skill on IMAGINED elite experience (sleep consolidation).

        Same machinery as waking practice — identical trust region, identical
        consolidation curve — but the evidence comes from world-model rollouts
        instead of real invocations. That is the point: the agent runs at ~10
        environment steps per second, so real successful invocations are
        precious and rare, while imagined ones are cheap. Sleep is when
        biological memory consolidates, and this is the same trade.

        `lr_scale` < 1 is the honest discount for dreamed evidence: an
        imagined success is weaker evidence than a real one, because it is
        only as true as the world model. The caller additionally gates this on
        WM trust, so a model that has not earned trust consolidates nothing.
        """
        if not self.enabled or not traj or len(traj) < self.min_steps:
            return None
        info = self._self_imitate(slot, actor, traj, lr_scale=lr_scale,
                                  dreamed=True, delta=delta)
        if info and not info.get("reverted"):
            info["dreamed"] = 1.0
        return info

    @staticmethod
    def _logits_of(actor: Any, delta: Any):
        """The skill's ACTUAL policy function: frozen base + owned delta
        (infra #35). With no delta this is exactly the legacy behaviour, so
        every pre-delta caller and test is untouched. A width-mismatched
        delta contributes nothing rather than crashing a live run."""
        if delta is None:
            return lambda f: actor.action_head(actor.shared(f))

        def fn(f):
            out = actor.action_head(actor.shared(f))
            try:
                d = delta(f)
                if d.shape == out.shape:
                    return out + d
            except Exception:
                pass
            return out
        return fn

    def _self_imitate(self, slot: int, actor: Any,
                      traj: List[Tuple[torch.Tensor, int]],
                      lr_scale: float = 1.0, dreamed: bool = False,
                      delta: Any = None) -> Optional[Dict[str, float]]:
        feats = torch.cat([f for f, _ in traj], dim=0).to(self.device)
        acts = torch.tensor([a for _, a in traj], dtype=torch.long,
                            device=self.device)

        # INDIVIDUATION (infra #35): when the slot owns a delta head, practice
        # trains ONLY the delta — the base stays the frozen record of what the
        # policy was at mint, and everything the skill becomes lives in its
        # own parameters. The trust region, consolidation curve and revert
        # machinery below operate on `params` generically, so the guarantees
        # are identical in both modes.
        _logits = self._logits_of(actor, delta)
        if delta is not None:
            params = [p for p in delta.parameters()]
        else:
            params = [p for p in actor.parameters()]
        if not params:
            return None
        for p in params:
            p.requires_grad_(True)

        # CONSOLIDATION: a proven skill resists change. Fresh skills move at
        # the full rate; one at strength 1.0 moves at a quarter of it.
        s = float(self.strength.get(slot, 0.0))
        lr_eff = self.lr * max(0.25, 1.0 - 0.75 * s) * float(lr_scale)
        # OPTIMIZER IDENTITY (review finding, 2026-08-09): the cache is keyed
        # by slot but Adam holds references to the PARAMETER TENSORS of
        # whichever module was resident at creation. _materialize builds a
        # brand-new module after every LRU eviction (max_resident=8 vs 24
        # bound slots -> constant churn), so a stale optimizer steps ORPHANED
        # tensors: backward() puts grads on the new params, step() moves the
        # old ones -> weights never change, KL reads 0, strength still climbs
        # — practice becomes a silent no-op that REPORTS healthy learning.
        # Identity of the first param tensor is the honest cache key; a mode
        # flip (actor<->delta) changes it too, so one check covers both.
        if not hasattr(self, "_opt_pid"):
            self._opt_pid = {}
        _pid = id(params[0])
        if self._opt_pid.get(slot) != _pid:
            self._opt.pop(slot, None)
            self._opt_pid[slot] = _pid
        opt = self._opt.get(slot)
        if opt is None:
            opt = torch.optim.Adam(params, lr=lr_eff)
            self._opt[slot] = opt
        for g in opt.param_groups:
            g["lr"] = lr_eff

        with torch.no_grad():
            old_logits = _logits(feats).detach()
            old_logp = F.log_softmax(old_logits, dim=-1)
        before = [p.detach().clone() for p in params]

        _trainee = delta if delta is not None else actor
        _trainee.train()
        loss_val = 0.0
        for _ in range(self.epochs):
            logits = _logits(feats)
            # cross-entropy toward the actions this invocation ACTUALLY took
            loss = F.cross_entropy(logits, acts)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            loss_val = float(loss.item())
        _trainee.eval()

        # ---- TRUST REGION -------------------------------------------------
        # Blend back toward the pre-update weights until the induced KL fits
        # under kl_max. The UPDATED weights are snapshotted first: the blend
        # writes into `p` each round, so interpolating from the live tensor
        # would compound (each halving would measure from the previous blend,
        # not from the original step) and the alpha would mean nothing.
        updated = [p.detach().clone() for p in params]
        kl = self._kl(_logits, feats, old_logp)
        alpha, tries = 1.0, 0
        while kl > self.kl_max and tries < 5:
            alpha *= 0.5
            tries += 1
            with torch.no_grad():
                for p, b, u in zip(params, before, updated):
                    p.copy_(b + alpha * (u - b))
            kl = self._kl(_logits, feats, old_logp)
        if tries:
            self.clipped[slot] = self.clipped.get(slot, 0) + 1
        if kl > self.kl_max:
            # could not fit the trust region — revert entirely rather than
            # let one invocation move a skill an unbounded distance.
            # (grad flags cleared HERE too: the early return used to leave
            # params trainable on the hot invoke path — review 2026-08-09)
            with torch.no_grad():
                for p, b in zip(params, before):
                    p.copy_(b)
            for p in params:
                p.requires_grad_(False)
            return {"slot": slot, "reverted": 1.0, "kl": kl}

        with torch.no_grad():
            d = sum(float((p - b).pow(2).sum()) for p, b in zip(params, before))
        for p in params:
            p.requires_grad_(False)

        self.drift[slot] = self.drift.get(slot, 0.0) + d ** 0.5
        # exactly ONE counter per update — see the dream_updates note above
        if dreamed:
            self.dream_updates[slot] = self.dream_updates.get(slot, 0) + 1
        else:
            self.updates[slot] = self.updates.get(slot, 0) + 1
        # Dreamed rehearsal does NOT fully consolidate a skill. Strength is
        # what makes a skill resist further change and survive decay, and
        # letting imagination alone drive it to 1.0 would mean a skill could
        # become permanent without ever having worked in the world.
        gain = self.strength_up * (0.5 if dreamed else 1.0)
        self.strength[slot] = min(1.0, s + gain)
        out = {"slot": slot, "loss": loss_val, "kl": kl,
               "step": d ** 0.5, "strength": self.strength[slot],
               "n": float(len(traj))}
        if delta is not None:
            # INDIVIDUATION, measured: mean |delta logits| on this batch is
            # how far this skill has become its own thing (0 at mint by the
            # zero-init; grows only through practice)
            with torch.no_grad():
                out["delta_mag"] = float(delta(feats).abs().mean().item())
        return out

    @staticmethod
    def _kl(logits_fn, feats: torch.Tensor,
            old_logp: torch.Tensor) -> float:
        with torch.no_grad():
            new_logp = F.log_softmax(logits_fn(feats), dim=-1)
            return float(
                (old_logp.exp() * (old_logp - new_logp)).sum(-1).mean().item())

    # ---- forgetting --------------------------------------------------------
    def decay(self) -> List[int]:
        """Time passes; unrehearsed skills weaken. Returns slots now STALE.

        Called once per segment. A stale slot is not deleted — deletion would
        throw away a real discovery. It is reported so the caller can re-derive
        it from the agent's CURRENT competence (the re-distill path), which is
        the useful sense of "forgetting": the specific old motor trace fades,
        the fact that this achievement exists does not.
        """
        stale: List[int] = []
        for slot in list(self.strength):
            self.strength[slot] = max(
                0.0, self.strength[slot] - self.decay_per_segment)
            if self.strength[slot] < self.stale_below:
                stale.append(slot)
        return stale

    def on_bind(self, slot: int) -> None:
        """A slot was (re)bound: its resident module is new, so the optimizer
        state and drift accounting must not carry over from the old tenant."""
        self._opt.pop(slot, None)
        self.drift.pop(slot, None)
        self.updates.pop(slot, None)
        self.dream_updates.pop(slot, None)
        self.clipped.pop(slot, None)
        self.strength.setdefault(slot, 0.0)
        for k in [k for k in self._traj if k[1] == slot]:
            self._traj.pop(k, None)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "slots_practised": len(set(self.updates) | set(self.dream_updates)),
            "updates_total": int(sum(self.updates.values())),
            "dream_updates_total": int(sum(self.dream_updates.values())),
            "clipped_total": int(sum(self.clipped.values())),
            "abstained": int(self.abstained),
            "mean_strength": (
                float(sum(self.strength.values()) / len(self.strength))
                if self.strength else 0.0),
            "mean_drift": (
                float(sum(self.drift.values()) / len(self.drift))
                if self.drift else 0.0),
        }
