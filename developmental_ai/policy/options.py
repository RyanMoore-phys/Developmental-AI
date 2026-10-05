"""Skills as options — mid-episode skill invocation (July 2026, Layer 4).

The meta-policy's discrete head is widened from P primitives to P + K:
picking index >= P INVOKES a stored skill, whose frozen policy then drives
the env for up to `max_option_steps` primitive steps (early exit on reward
spike or episode end). PPO learns WHEN to invoke via proper SMDP credit
assignment (gamma^tau bootstrap, (gamma*lambda)^tau trace — see
StandaloneActorCritic._compute_gae).

Design invariants (judge-panel design, wf_085afa1a):
  * Every dynamics consumer (world model FiLM, ICM/LP curiosity, replay,
    goal discovery, symbolizer, scaffold) sees the executed PRIMITIVE.
    Only the PPO buffer and telemetry ever see a meta-action.
  * Skill policies are FROZEN during execution — no intra-option gradients.
    A bound slot is never rebound and never weight-refreshed within a run,
    so Q(s, slot_j) stays stationary for the meta-policy.
  * Head adaptation is ROW-TRUNCATION: a skill saved with a P+K head is
    sliced to its first P rows. Sampling softmax(logits[:P]) equals the
    renormalized no-slot conditional, and an option becomes structurally
    unable to invoke another option — no recursion guard needed.
  * Skill execution SAMPLES (skills were trained stochastic; argmax
    brittleness is a known audit failure).
  * Scouts invoke options too (behaviour-distribution match for the WM +
    4x the unlock-producing behaviour), but never touch the PPO buffer.
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid as _uuid
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SlotRefused(Exception):
    """A stored skill cannot be adapted into an option slot."""


_ACH_SLOT_RE = re.compile(r"^ach_(\d+)_")


def _slot_index_of(skill_id: str) -> int:
    """Broadcaster slot index frozen into an `ach_NN_<name>` skill id, else -1.

    The NAME half of a skill id is a snapshot of what the slot was called at
    MINT time and drifts as goals re-key; the INDEX half does not. Callers use
    it to look up per-slot competence without depending on the name matching.
    """
    m = _ACH_SLOT_RE.match(str(skill_id or ""))
    return int(m.group(1)) if m else -1


def adapt_actor_sd(sd: Dict, P: int,
                   primitive_dim: Optional[int] = None) -> Tuple[Dict, int, int]:
    """Adapt a stored actor state dict to an execution head.

    P_old-AWARE TRUNCATION (2026-07-26, the action-space widening). A stored
    skill's head is META-width: its OWN primitive count plus the option slots
    of its era (e.g. 10 + 24 = 34 rows). Slicing `W[:P]` with today's P was
    only correct while P equalled the skill's own primitive count. The moment
    P grew (10 -> 12 for the crafting buttons), `W[:12]` would bind the
    skill's OPTION-SLOT-0/1 logits as if they were the two new primitives —
    no refusal fires (34 >= 12), nothing crashes, and a frozen skill quietly
    emits "invoke option 0" re-labelled as "open inventory". Measured on real
    unlock frames when this was scouted: 0.02%-5.21% of action mass diverted.

    So the slice width is the skill's OWN `primitive_dim`, recorded at mint.
    The resulting head emits indices in [0, primitive_dim) — all valid in any
    APPEND-ONLY widened space, because append-only is precisely the promise
    that old indices keep their meaning. A skill minted with 10 actions simply
    never presses the two buttons that did not exist in its world.

    When `primitive_dim` is unknown (legacy nulls), the ONLY safe case is a
    head exactly P wide. Anything else is refused loudly: guessing
    `P_old = rows - k_slots` is forbidden because k_slots is a config knob
    that has changed across eras.
    """
    if "actor" not in sd:
        raise SlotRefused("state dict has no actor")
    a = dict(sd["actor"])
    W, b = a.get("action_head.weight"), a.get("action_head.bias")
    if W is None or b is None:
        raise SlotRefused("no discrete action head")
    p_own = int(primitive_dim) if primitive_dim else None
    if p_own is not None:
        if W.shape[0] < p_own:
            raise SlotRefused(
                f"head {W.shape[0]} narrower than its own recorded "
                f"primitive_dim {p_own} — metadata is wrong, refuse")
        cut = p_own
    else:
        if W.shape[0] != P:
            raise SlotRefused(
                f"primitive_dim unrecorded and head {W.shape[0]} != current "
                f"P {P}: cannot know which rows are primitives (inferring "
                f"rows - k_slots is forbidden; k_slots has changed across "
                f"eras). Backfill action_dim on this skill.")
        cut = P
    a["action_head.weight"], a["action_head.bias"] = W[:cut], b[:cut]
    if "shared.0.weight" not in a:
        raise SlotRefused("unexpected actor topology")
    in_dim = a["shared.0.weight"].shape[1]
    hidden = a["shared.0.weight"].shape[0]   # sniffed, never assumed
    return a, int(in_dim), int(hidden)


class SkillOptionBank:
    """Slot table + frozen-weight cache for invocable skills."""

    def __init__(self, skill_bank, obs_dim: int, P: int, K: int,
                 device, cache_cfg: Optional[Dict] = None,
                 hidden_dim: int = 256):
        self.skill_bank = skill_bank
        self.obs_dim = int(obs_dim)
        self.P = int(P)
        self.K = int(K)
        self.device = device
        self.hidden_dim = int(hidden_dim)
        cache_cfg = cache_cfg or {}
        self.max_resident = int(cache_cfg.get("max_resident", 8))
        self.dtype = (torch.float16 if str(cache_cfg.get("dtype", "fp32"))
                      == "fp16" else torch.float32)

        # slot -> binding {skill_id, actor_sd(cpu), in_dim, kdim,
        #                  cond_sd or None, ctx(np or None), mtime}
        self.slots: List[Optional[Dict]] = [None] * self.K
        # slot -> SCRIPTED macro spec (hardcoded action generator instead of a
        # learned policy). Reserved slots bootstrap a behaviour the emergent
        # policy can't stumble into; the meta-policy still learns WHEN to invoke.
        self.scripted: Dict[int, Dict] = {}
        self._bound_ids: set = set()
        self._pending_minted: List[str] = []
        # per-slot invocation counts (this run). A bound slot the meta-policy
        # has NEVER invoked carries no trained meta-value, so evicting it for
        # a fresh mint loses nothing real (see refresh_slots evict-for-mint).
        self.invoked: Dict[int, int] = {}
        # resident eval modules, LRU by slot: slot -> (actor_module, cond)
        self._resident: "OrderedDict[int, Tuple[Any, Any]]" = OrderedDict()
        self.disk_loads = 0
        self.refused: Dict[str, str] = {}
        # policy-file mtime at the moment of refusal, so a REWRITTEN policy
        # clears the refusal instead of latching it forever (see refresh_slots)
        self._refused_mtime: Dict[str, float] = {}
        # symbolic-conditioning health: how many bound skills actually carry a
        # live knowledge gate vs how many silently fell back to zeros
        # ---- PHASE-SCOPED ACTION SPACE (2026-08-02) --------------------
        # Primitive indices the policy may not select in this developmental
        # phase. Indices are NEVER renumbered (the macro table is
        # append-only, and every stored skill indexes actions by position) —
        # the action simply stops being offered.
        #
        # MEASURED NEED: with `inventory` (10) selectable, the agent spent
        # 80% of its steps inside the menu, where 5 of its 12 actions do
        # nothing at all and the other camera actions move a CURSOR rather
        # than its head — so its pitch stayed clamped at -90 (straight up)
        # and it could not look down to reach anything. Exiting needs a
        # two-step release-then-press sequence on an action it picked <4% of
        # the time. This is not a rule about what the agent should want; it
        # is the same scoping already applied when the macro set chose
        # inventory+use "over hotbar.1-9 to keep the exploration burden
        # small". Re-enable when crafting is the actual objective.
        self.disabled_macros: set = set()
        self.cond_live = 0
        self.cond_load_failures = 0
        self.cond_failed_ids: set = set()
        # ---- SKILLS AS MEMORIES (2026-08-01) ------------------------------
        # Practice/consolidation/decay for bound skills. None => the historical
        # behaviour (mint once, frozen for the run) is byte-for-byte intact,
        # which is what every config other than skybot still expects.
        self.practice = None
        # ---- SKILL LOGIT-SPREAD CAP (2026-10-04) --------------------------
        # The shared policy's `_bound_logits` (policy.logit_range) caps the
        # best/worst action probability ratio at e^logit_range so it can be
        # confident but never CERTAIN. skill_action sampled straight from
        # `actor.action_head` and skipped it, so a frozen skill copy could
        # be arbitrarily more deterministic than the policy it was copied
        # from. MEASURED (13 h live shadow, 2026-10-04): action 17 (turn
        # left micro) was 40% of the first 20 sampled actions BEFORE any
        # PPO update; the policy itself is not resumed, so the bias entered
        # through executed skill copies. Holds the shared policy's OWN bound
        # method (adopted in OptionExecutor.act), never a copy of the value,
        # so the two cannot drift. None = identity (no policy seen yet, or a
        # policy/test double without the bound) — never a gate, so no latch.
        self.logit_bound = None

    def adopt_logit_bound(self, policy) -> None:
        """Use the SHARED policy's own logit bound for skill sampling.

        Reuses `policy._bound_logits` itself (same function, same live
        `policy.logit_range`), so a skill copy is bounded exactly as the
        shared policy is and a config change reaches both at once."""
        fn = getattr(policy, "_bound_logits", None)
        self.logit_bound = (fn if callable(fn)
                            and hasattr(policy, "logit_range") else None)

    # ---- slot management ---------------------------------------------------
    def notify_minted(self, skill_id: str) -> None:
        self._pending_minted.append(skill_id)

    def note_invoked(self, slot: int) -> None:
        self.invoked[slot] = self.invoked.get(slot, 0) + 1

    # ---- scripted (bootstrap) options --------------------------------------
    def reserve_scripted(self, spec: Dict) -> Optional[int]:
        """Reserve a slot for a SCRIPTED option — a fixed low-level macro
        (e.g. 'look down onto the trunk, then hold attack') instead of a
        learned skill policy. Bootstraps a behaviour the emergent policy cannot
        stumble into (the first log break: logs need SUSTAINED attack that a
        step-wise stochastic policy never produces). The meta-policy still
        learns WHEN to invoke it via SMDP credit; only the low-level actions
        are fixed. Placed in the LAST free slot so it never displaces a learned
        skill; frozen for the run like any bound slot."""
        slot = next((i for i in range(self.K - 1, -1, -1)
                     if self.slots[i] is None), None)
        if slot is None:
            logger.warning("no free option slot for scripted %s",
                           spec.get("skill_id"))
            return None
        sid = str(spec.get("skill_id", "scripted"))
        self.slots[slot] = {
            "skill_id": sid, "name": spec.get("name", sid),
            "preconditions": dict(spec.get("preconditions") or {}),
            "scripted": True,
        }
        self.scripted[slot] = {
            "aim_steps": int(spec.get("aim_steps", 0)),
            "aim_action": int(spec.get("aim_action", 8)),   # 8 = look down
            "act_action": int(spec.get("act_action", 5)),   # 5 = attack (hold)
        }
        self._bound_ids.add(sid)
        logger.info("option slot %d <- SCRIPTED %s (aim %dx a%d, then hold a%d)",
                    slot, sid, self.scripted[slot]["aim_steps"],
                    self.scripted[slot]["aim_action"],
                    self.scripted[slot]["act_action"])
        return slot

    def is_scripted(self, slot: int) -> bool:
        return slot in self.scripted

    def scripted_action(self, slot: int, steps_done: int) -> int:
        """The macro's action for this option step: a brief aim, then a HOLD
        (repeated attack) — which is exactly the sustained press a log needs."""
        spec = self.scripted[slot]
        if steps_done < spec["aim_steps"]:
            return spec["aim_action"]
        return spec["act_action"]

    def refresh_slots(self) -> None:
        """Fill EMPTY slots from the bank. Bound slots are frozen for the
        run — identity AND behaviour — even if save_skill later upserts a
        better policy (logged, ignored: meta-values must stay stationary)."""
        # A REFUSAL IS ABOUT A FILE, NOT A NAME (2026-07-27 review).
        # `refused` was keyed by skill_id alone and never cleared, so a skill
        # refused once — e.g. for arch metadata a later save REPAIRS — stayed
        # refused for the rest of the run AND every future run. Re-key on the
        # policy file's mtime so a rewritten policy gets a fresh hearing, and
        # a genuinely-broken one is still refused exactly once per version.
        for _sid in list(self.refused):
            _sk = self.skill_bank.skills.get(_sid)
            if _sk is None or not _sk.policy_path:
                continue
            try:
                _mt = os.path.getmtime(_sk.policy_path)
            except OSError:
                continue
            if self._refused_mtime.get(_sid) != _mt:
                self.refused.pop(_sid, None)
                self._refused_mtime.pop(_sid, None)
                logger.info("re-examining previously-refused skill %s "
                            "(policy rewritten)", _sid)
        candidates = []
        for sid, sk in self.skill_bank.skills.items():
            if sid in self._bound_ids or sid in self.refused:
                continue
            if sk.obs_dim is not None and int(sk.obs_dim) != self.obs_dim:
                continue
            if not sk.policy_path or not os.path.exists(sk.policy_path):
                continue
            candidates.append(sk)
        # prefer achievement skills, then competence, then stable id order
        candidates.sort(key=lambda s: (
            not s.skill_id.startswith("ach_"),
            -float(s.success_rate), s.skill_id))
        for sk in candidates:
            free = next((i for i, b in enumerate(self.slots) if b is None),
                        None)
            if free is None:
                break
            try:
                self._bind(free, sk)
            except SlotRefused as e:
                self.refused[sk.skill_id] = str(e)
                self._note_refused_mtime(sk)
                logger.warning("option slot refused %s: %s", sk.skill_id, e)
        # EVICT-FOR-MINT (audit fix): with every slot bound (24 persisted
        # skills fill 24 slots at stream start), a FRESH mint had nowhere to
        # bind — the newly-discovered behaviour stayed un-invocable forever.
        # A fresh mint may displace a bound, NEVER-INVOKED, non-scripted slot:
        # an uninvoked slot's meta-value is untrained, so the frozen-slot
        # stationarity invariant loses nothing real.
        for sid in list(self._pending_minted):
            if (sid in self._bound_ids or sid in self.refused
                    or any(b is None for b in self.slots)):
                continue                     # bound / refused / free slot left
            sk = self.skill_bank.skills.get(sid)
            if sk is None:
                continue
            victims = [i for i, b in enumerate(self.slots)
                       if b is not None and not b.get("scripted")
                       and self.invoked.get(i, 0) == 0]
            if not victims:
                break                        # everything is practiced: keep all

            def _stored_sr(i: int) -> float:
                s = self.skill_bank.skills.get(self.slots[i]["skill_id"])
                return float(s.success_rate) if s is not None else 0.0
            v = min(victims, key=_stored_sr)
            old_sid = self.slots[v]["skill_id"]
            self._bound_ids.discard(old_sid)
            self._resident.pop(v, None)
            self.slots[v] = None
            try:
                self._bind(v, sk)
                logger.info("option slot %d: evicted never-invoked %s for "
                            "fresh mint %s", v, old_sid, sid)
            except SlotRefused as e:
                self.refused[sid] = str(e)
                logger.warning("evict-for-mint refused %s: %s (slot %d left "
                               "empty for next refresh)", sid, e, v)
        self._pending_minted.clear()

    def _bind(self, slot: int, sk) -> None:
        sd = self.skill_bank.load_skill_policy(sk.skill_id)
        if sd is None:
            raise SlotRefused("policy load failed")
        self.disk_loads += 1
        # ---- CONV FAMILY (arch v3): branch BEFORE the flat adapter --------
        # A conv sd fed to adapt_actor_sd would die on the kdim sanity check
        # (in_dim ~354 vs obs_dim 49152), but by accident, with a misleading
        # message. Detect the family by its own marker instead.
        # ENCODED FAMILY = conv | wm | rssm. All three read FEATURES rather
        # than raw pixels, so they share the encoded bind path. The family
        # test cannot be `"encoder" in sd` any more (2026-09-02): arch="rssm"
        # reads the LIVE world model's latents and therefore ships no encoder
        # snapshot, so it fell through to the flat adapter, which computed
        # kdim = in_dim - obs_dim = 363 - 49152 and refused every rssm skill
        # with "unexplainable input_dim". Mint succeeded, binding never did —
        # skills-as-options was silently off. Test the arch MARKER instead.
        if "encoder" in sd or sd.get("arch") == "rssm":
            return self._bind_conv(slot, sk, sd)
        # the skill's OWN primitive count decides the head slice — see
        # adapt_actor_sd. getattr: legacy Skill objects may lack the field.
        actor_sd, in_dim, hidden = adapt_actor_sd(
            sd, self.P, primitive_dim=getattr(sk, "action_dim", None))
        p_own = int(getattr(sk, 'action_dim', None) or self.P)
        kdim = in_dim - self.obs_dim
        if kdim < 0 or kdim > 512:
            raise SlotRefused(f"unexplainable input_dim {in_dim}")
        cond_sd = sd.get("conditioner") if kdim > 0 else None
        ctx = None
        if kdim > 0 and cond_sd is not None:
            # The skill trained under the broadcaster GOAL vector (one-hot
            # slot + has-target flag, DIM = 2*max_slots+2). Since Layer 1,
            # Skill.context_embedding stores the RSSM unlock latent instead
            # (a different object for a different purpose), so the canonical
            # conditioning vector is RECONSTRUCTED from the slot id when the
            # stored embedding is not kdim-shaped. Missing everything ->
            # conditioner dropped and the tail zero-padded, which is
            # BIT-EXACT to conditioning on null knowledge (gate * 0 == 0).
            raw = sk.context_embedding
            arr = (np.asarray(raw, dtype=np.float32).reshape(-1)
                   if raw is not None else None)
            if arr is not None and arr.size == kdim:
                ctx = arr.copy()
            else:
                m = re.match(r"ach_(\d+)_", sk.skill_id)
                max_slots = (kdim - 2) // 2
                if m and 0 <= int(m.group(1)) < max_slots:
                    v = np.zeros(kdim, dtype=np.float32)
                    v[int(m.group(1))] = 1.0      # target slot one-hot
                    v[2 * max_slots] = 1.0        # has-target flag
                    ctx = v
                else:
                    cond_sd = None  # bind-time decision, never at invoke
        elif kdim > 0:
            cond_sd = None
        self.slots[slot] = {
            "skill_id": sk.skill_id, "name": sk.name,
            # STABLE IDENTITY (infra #35): survives slot reuse and renames —
            # 3 of 16 skills were once orphaned from their competence records
            # by a slot rename. Never derived from name or slot.
            "uuid": _uuid.uuid4().hex[:12],
            # the skill's OWNED parameters (DeltaHead state) — restored from
            # disk when the skill was ever practised, else None and
            # materialize builds a zero-init head. See skill_practice.
            "delta_sd": ({k: v.to(self.dtype) for k, v in sd["delta"].items()}
                         if isinstance(sd.get("delta"), dict) else None),
            # None/empty => ALWAYS ELIGIBLE. Legacy skills carry no
            # grounded preconditions, and gating them out would silently
            # disable the entire existing bank.
            "preconditions": dict(sk.preconditions or {}),
            "actor_sd": {k: v.to(self.dtype) for k, v in actor_sd.items()},
            # the slot's OWN head width. The adapted head has p_own rows, so
            # the rebuilt module must be p_own-wide too — rebuilding at
            # today's P would raise a shape mismatch at first materialize for
            # every skill minted under a narrower action space.
            "p_own": p_own,
            "in_dim": in_dim, "kdim": kdim, "hidden": hidden,
            "cond_sd": cond_sd, "ctx": ctx,
            "mtime": os.path.getmtime(sk.policy_path),
        }
        # A slot's practice state belongs to its TENANT, not the slot number.
        # Carrying an Adam state (or accumulated drift) across a rebind would
        # apply one skill's momentum to a different skill's weights.
        if self.practice is not None:
            self.practice.on_bind(slot)
        self._bound_ids.add(sk.skill_id)
        logger.info("option slot %d <- %s (%s, kdim=%d)",
                    slot, sk.skill_id, sk.name, kdim)

    def _bind_conv(self, slot: int, sk, sd: Dict) -> None:
        """Bind a conv-family skill (arch v3).

        Differences from the flat path, each load-bearing:
          * NO head truncation. The stored head keeps its full meta width
            (primitives + option-slot rows) — that is what lets a frozen
            skill invoke other skills. `p_own` still marks where primitives
            end; rows above it are only reachable through `slot_map`.
          * Metadata is MANDATORY. Conv skills are always minted with
            action_dim/enc_dim/head_dim recorded; their absence means a
            hand-edited registry, and guessing widths is how option-slot
            logits become buttons (see the 10->12 widening postmortem).
          * `slot_map` records what each slot ROW meant at mint, by skill
            IDENTITY. Binding never trusts row positions across eras.
        """
        p_own = getattr(sk, "action_dim", None)
        enc_dim = getattr(sk, "enc_dim", 0) or 0
        if not enc_dim:
            # RECOVER FROM THE WEIGHTS BEFORE REFUSING (2026-07-27 review).
            # `get_state_dict` stamps arch/enc_dim into the state dict, so a
            # registry row that lost them (a caller that omitted the arch
            # kwargs) is repairable from the tensors themselves. Refusing
            # here instead made improving a skill delete it from the bank
            # permanently, since `refused` is sticky.
            enc_dim = int(sd.get("enc_dim", 0) or 0)
            if enc_dim:
                logger.warning(
                    "conv skill %s had no enc_dim in the registry — "
                    "recovered %d from its state dict", sk.skill_id, enc_dim)
        if not p_own or not enc_dim:
            raise SlotRefused(
                f"conv skill {sk.skill_id} missing action_dim/enc_dim "
                f"metadata and unrecoverable from the state dict "
                f"(action_dim={p_own}, enc_dim={enc_dim})")
        if "actor" not in sd or not isinstance(sd.get("actor"), dict):
            raise SlotRefused(f"conv skill {sk.skill_id}: no actor in sd")
        actor_sd = {k: v for k, v in sd["actor"].items()}
        W = actor_sd.get("action_head.weight")
        if W is None:
            raise SlotRefused("conv sd has no action_head")
        if "shared.0.weight" not in actor_sd:
            raise SlotRefused(
                f"conv skill {sk.skill_id}: unexpected actor topology")
        head_dim = int(W.shape[0])
        rec_head = getattr(sk, "head_dim", None)
        if rec_head is not None and int(rec_head) != head_dim:
            raise SlotRefused(
                f"conv skill {sk.skill_id}: recorded head_dim {rec_head} != "
                f"stored tensor {head_dim} — registry and weights disagree")
        if head_dim < int(p_own):
            raise SlotRefused(
                f"conv skill {sk.skill_id}: head {head_dim} narrower than "
                f"its own primitive count {p_own}")
        in_dim = int(actor_sd["shared.0.weight"].shape[1])
        hidden = int(actor_sd["shared.0.weight"].shape[0])
        # KDIM FROM THE CONDITIONER ITSELF, NOT BY SUBTRACTION (fix
        # 2026-08-01, caught by tests/_skybot_boot_smoke.py).
        # The actor input is [features, PROPRIO, gated_knowledge], so
        #     in_dim - enc_dim  ==  proprio_dim + kdim
        # and inferring kdim by subtraction over-counts it by exactly the
        # proprio width. The rebuilt KnowledgeConditioner then had the wrong
        # shape, `load_state_dict` raised, and the except-branch below
        # silently dropped the conditioner and zero-padded — which the code
        # itself notes is "BIT-EXACT to conditioning on null knowledge".
        # i.e. since proprioception was added, EVERY encoded skill has been
        # binding with its symbolic conditioning quietly switched off, and the
        # only evidence was one warning line.
        # The conditioner's own weight shape is (kdim, enc_dim) — ground
        # truth. Fall back to subtraction only when it is absent.
        cond_sd = sd.get("conditioner")
        _cw = (cond_sd or {}).get("gate_net.0.weight")
        if _cw is not None:
            kdim = int(_cw.shape[0])
        else:
            kdim = in_dim - int(enc_dim)
        if kdim < 0 or kdim > 512:
            raise SlotRefused(f"conv skill {sk.skill_id}: unexplainable "
                              f"in_dim {in_dim} for enc_dim {enc_dim}")
        if kdim <= 0:
            cond_sd = None
        # whatever the actor's input width has left over after perception and
        # knowledge IS the proprio block the skill was trained with
        proprio_dim = max(0, in_dim - int(enc_dim) - int(kdim))
        ctx = None
        if kdim > 0 and cond_sd is not None:
            raw = sk.context_embedding
            arr = (np.asarray(raw, dtype=np.float32).reshape(-1)
                   if raw is not None else None)
            if arr is not None and arr.size == kdim:
                ctx = arr.copy()
            else:
                m = re.match(r"ach_(\d+)_", sk.skill_id)
                max_slots = (kdim - 2) // 2
                if m and 0 <= int(m.group(1)) < max_slots:
                    v = np.zeros(kdim, dtype=np.float32)
                    v[int(m.group(1))] = 1.0
                    v[2 * max_slots] = 1.0
                    ctx = v
                else:
                    cond_sd = None
        elif kdim > 0:
            cond_sd = None
        slot_map = dict(getattr(sk, "slot_map", None) or {})
        self.slots[slot] = {
            "skill_id": sk.skill_id, "name": sk.name,
            "uuid": _uuid.uuid4().hex[:12],     # stable identity (infra #35)
            # owned params, restored from disk when ever practised
            "delta_sd": ({k: v.to(self.dtype) for k, v in sd["delta"].items()}
                         if isinstance(sd.get("delta"), dict) else None),
            "preconditions": dict(sk.preconditions or {}),
            # CARRY THE REAL FAMILY. "conv" and "wm" both store an encoder and
            # both arrive here, but they rebuild with DIFFERENT encoder classes
            # (CNNFeatureEncoder vs the world model's CNNEncoder) whose
            # state_dict keys do not match. Hardcoding "conv" would load wm
            # weights into the wrong module and refuse the skill.
            "arch": str(sd.get("arch") or getattr(sk, "arch", None) or "conv"),
            "actor_sd": {k: v.to(self.dtype) for k, v in actor_sd.items()},
            # None for arch="rssm": it reads the LIVE world model's latents,
            # so there is no encoder to snapshot. _materialize must therefore
            # branch on has-encoder, not on is-encoded.
            "encoder_sd": ({k: v.to(self.dtype)
                            for k, v in sd["encoder"].items()}
                           if isinstance(sd.get("encoder"), dict) else None),
            "p_own": int(p_own),
            "head_dim": head_dim,
            "slot_map": slot_map,
            "enc_dim": int(enc_dim),
            "proprio_dim": int(proprio_dim),
            "in_dim": in_dim, "kdim": kdim, "hidden": hidden,
            "cond_sd": cond_sd, "ctx": ctx,
            "mtime": os.path.getmtime(sk.policy_path),
        }
        # A slot's practice state belongs to its TENANT, not the slot number.
        # Carrying an Adam state (or accumulated drift) across a rebind would
        # apply one skill's momentum to a different skill's weights.
        if self.practice is not None:
            self.practice.on_bind(slot)
        self._bound_ids.add(sk.skill_id)
        logger.info("option slot %d <- %s (%s, CONV kdim=%d head=%d "
                    "slot_map=%d entries)", slot, sk.skill_id, sk.name,
                    kdim, head_dim, len(slot_map))

    def _note_refused_mtime(self, sk) -> None:
        """Stamp the policy file's mtime alongside a refusal so refresh_slots
        can tell 'still the same broken file' from 'rewritten, try again'."""
        try:
            self._refused_mtime[sk.skill_id] = os.path.getmtime(
                sk.policy_path)
        except (OSError, AttributeError, TypeError):
            self._refused_mtime.pop(sk.skill_id, None)

    def slot_of_skill(self, skill_id: str) -> Optional[int]:
        """Current slot binding a skill_id, or None. Identity->position
        resolution for nested invocation (positions are never trusted)."""
        for i, b in enumerate(self.slots):
            if b is not None and b["skill_id"] == skill_id:
                return i
        return None

    def primitive_mask(self, width: Optional[int] = None) -> np.ndarray:
        """Primitives-only mask honouring the phase-scoped action space.

        SINGLE SOURCE OF TRUTH for "which raw macros may be emitted right
        now". `disabled_macros` used to be applied ONLY in mask(), i.e. only
        to the meta-policy's own primitive choices — so the scoping leaked
        through both of the other action sources (a skill choosing inside an
        invoked option, and scouts running unmasked). Given the whole reason
        macro 10 is disabled is that the agent spent 80% of its steps inside
        the inventory screen, a hole that lets any of those paths reopen it
        defeats the measure entirely.

        `width` defaults to the primitive count; pass a larger head width to
        get the primitive columns of a wider mask (option rows stay False).
        """
        w = int(self.P if width is None else width)
        m = np.zeros(w, dtype=bool)
        lim = min(self.P, w)
        m[:lim] = True
        for _d in self.disabled_macros:
            _d = int(_d)
            if 0 <= _d < lim:
                m[_d] = False
        if not m[:lim].any():          # never mask the agent mute
            m[:lim] = True
        return m

    def mask(self, predicates: Optional[Dict[str, bool]] = None,
             min_match: float = 0.6,
             competence: Optional[Dict[str, float]] = None,
             competence_floor: float = 0.0) -> np.ndarray:
        """Valid actions: all primitives, plus ELIGIBLE bound slots.

        A bound slot is offered only when BOTH hold:
          * COMPETENCE gate (the condition-based "warmup"): the skill's live
            competence >= competence_floor. This solves the cold-start drag
            — at the start every skill is weak, so NO options are offered and
            the base policy learns primitively; as competence rises (the base
            policy gets reliable at the underlying goals), options self-enable
            per skill. It is a CONDITION, not an episode timer, so it works
            identically for a fixed run and for continuous lifelong running.
          * INITIATION SET: enough of the skill's recorded preconditions match
            the current grounded belief (`predicates`).

        Skills without preconditions stay initiation-eligible (legacy bank);
        `competence=None` skips the competence gate (backward compatible).
        """
        m = self.primitive_mask(self.P + self.K)
        for i, b in enumerate(self.slots):
            if b is None:
                continue
            if b.get("scripted"):
                # SCRIPTED bootstrap option: always "competent" (the macro is
                # fixed), so it bypasses the competence warmup gate — offered
                # whenever its preconditions hold (a reachable trunk in view).
                pre = b.get("preconditions") or {}
                if predicates and pre:
                    hits = sum(1 for k, v in pre.items()
                               if predicates.get(k) == bool(v))
                    if hits / len(pre) < min_match:
                        continue
                m[self.P + i] = True
                continue
            if competence is not None and competence_floor > 0.0:
                # NAME-DRIFT-PROOF LOOKUP (2026-07-25). The caller keys this
                # dict by the RECONSTRUCTED name `ach_{slot:02d}_{slot_name}`,
                # but a skill_id freezes the slot name it was MINTED under.
                # When a broadcaster slot is later renamed (grounded effect ->
                # `discovered_N` and back), the reconstructed key stops
                # matching and `.get(...)` silently returns 0.0 — permanently
                # gating a skill that may be perfectly competent. Measured
                # live: 3 of 16 skills orphaned this way
                # (ach_00_break_birch_leaves, ach_01_break_oak_leaves,
                # ach_02_break_dirt vs slots now named discovered_0/1/2).
                # The SLOT INDEX embedded in the id is stable, so fall back to
                # it. Missing from BOTH -> 0.0, the old conservative default.
                _c = competence.get(b["skill_id"])
                if _c is None:
                    _c = competence.get(_slot_index_of(b["skill_id"]), 0.0)
                if float(_c) < competence_floor:
                    continue          # not yet worth invoking -> not offered
            pre = b.get("preconditions") or {}
            if predicates and pre:
                hits = sum(1 for k, v in pre.items()
                           if predicates.get(k) == bool(v))
                if hits / len(pre) < min_match:
                    continue          # preconditions unmet -> not offered
            m[self.P + i] = True
        return m

    # ---- execution ----------------------------------------------------------
    def _materialize(self, slot: int):
        """Resident eval modules for a slot (LRU, actor+conditioner only —
        critics are never loaded). Cost paid at option START, not per step."""
        if slot in self._resident:
            self._resident.move_to_end(slot)
            return self._resident[slot]
        b = self.slots[slot]
        from developmental_ai.policy.actor_critic import (
            ActorNetwork, KnowledgeConditioner)
        # ENCODED FAMILY = conv | wm. Both read features rather than raw
        # pixels, so everything downstream (head width, conditioner input
        # width) is identical; only the encoder MODULE differs.
        _arch = b.get("arch")
        # IS-ENCODED vs HAS-ENCODER are different questions (2026-09-02).
        # All three encoded archs read features, so head width and the
        # conditioner's input width follow enc_dim for all of them. Only
        # conv/wm own an encoder MODULE to rebuild; rssm's features arrive
        # from the live world model via `feats`, so it must not try.
        is_conv = _arch in ("conv", "wm", "rssm")
        has_encoder = _arch in ("conv", "wm")
        if has_encoder:
            # conv skills rebuild at their FULL stored head width — the slot
            # rows are the composition interface, sliced off only by MASKING
            # at sample time, never by truncation at build time.
            side = int(round((self.obs_dim / 3) ** 0.5))
            if _arch == "wm":
                # the world model's own encoder class: a wm skill snapshotted
                # WM weights, and CNNFeatureEncoder has a different layout
                # (`head.*` vs `fc.*`), so loading one into the other raises.
                from developmental_ai.world_model.rssm import CNNEncoder
                enc = CNNEncoder(3, b["enc_dim"], side)
            else:
                from developmental_ai.curiosity.icm import CNNFeatureEncoder
                enc = CNNFeatureEncoder(3, side, b["enc_dim"])
            enc.load_state_dict({k: v.to(torch.float32)
                                 for k, v in b["encoder_sd"].items()})
            enc.to(self.device).eval()
            for p in enc.parameters():
                p.requires_grad_(False)
        else:
            enc = None
        # HEAD WIDTH FOLLOWS IS-ENCODED, NOT HAS-ENCODER. Every encoded skill
        # (conv|wm|rssm) rebuilds at its FULL stored head — the option-slot
        # rows above p_own are the composition interface, sliced off by
        # MASKING at sample time, never by truncation at build time. Only the
        # flat legacy family truncates. Building rssm at p_own would silently
        # drop those rows and disable skill-invokes-skill for it.
        if is_conv:
            actor = ActorNetwork(b["in_dim"], int(b["head_dim"]),
                                 b["hidden"], continuous=False)
        else:
            actor = ActorNetwork(b["in_dim"], int(b.get("p_own") or self.P),
                                 b["hidden"], continuous=False)
        actor.load_state_dict(
            {k: v.to(torch.float32) for k, v in b["actor_sd"].items()})
        actor.to(self.device).eval()
        for p in actor.parameters():
            p.requires_grad_(False)
        cond = None
        if b["cond_sd"] is not None:
            try:
                # the conditioner reads what the skill's policy read: raw obs
                # for flat, encoder features for conv
                cond = KnowledgeConditioner(
                    b["enc_dim"] if is_conv else self.obs_dim, b["kdim"])
                cond.load_state_dict(b["cond_sd"])
                cond.to(self.device).eval()
                for p in cond.parameters():
                    p.requires_grad_(False)
            except Exception as e:
                # LOUD, AND COUNTED. This branch silently disabled symbolic
                # conditioning on EVERY encoded skill for weeks: the only
                # evidence was one warning line, and "zero-pad fallback"
                # reads like a graceful degradation rather than what it is —
                # the skill's entire symbolic input replaced by zeros, which
                # the code elsewhere notes is bit-exact to conditioning on
                # null knowledge. A degradation nobody can see is a bug that
                # cannot be found, so it now leaves a countable trace that
                # the segment log can surface.
                self.cond_load_failures += 1
                self.cond_failed_ids.add(str(b["skill_id"]))
                logger.error(
                    "SYMBOLIC CONDITIONING DISABLED for %s (%s) — this skill "
                    "will act on ZERO knowledge, not on degraded knowledge",
                    b["skill_id"], e)
                cond = None
            else:
                self.cond_live += 1
        # THE OWNED HALF (infra #35): a zero-init DeltaHead per skill —
        # identity at birth, individuated only by practice. Built to the
        # actor's exact geometry so `base + delta` is always shape-safe.
        delta = None
        try:
            from developmental_ai.skill_bank.skill_practice import DeltaHead
            delta = DeltaHead(int(actor.shared[0].in_features),
                              int(actor.action_head.out_features))
            if b.get("delta_sd"):
                delta.load_state_dict({k: v.to(torch.float32)
                                       for k, v in b["delta_sd"].items()})
            delta.to(self.device).eval()
            for p in delta.parameters():
                p.requires_grad_(False)
        except Exception as e:
            logger.warning("delta head unavailable for slot %d (%s) — "
                           "skill runs as base-only", slot, e)
            delta = None
        self._resident[slot] = (actor, cond, enc, delta)
        while len(self._resident) > self.max_resident:
            old_slot, _mods = self._resident.popitem(last=False)
            logger.debug("option cache evicted slot %d", old_slot)
        return self._resident[slot]

    @torch.no_grad()
    def resident_actor(self, slot: int):
        """The slot's live actor module, or None if the slot is unbound or
        scripted. Materializes on demand (same LRU path as acting)."""
        if slot is None or slot < 0 or slot >= self.K:
            return None
        if self.slots[slot] is None or self.is_scripted(slot):
            return None
        try:
            return self._materialize(slot)[0]
        except Exception:
            return None

    @torch.no_grad()
    def resident_delta(self, slot: int):
        """The slot's OWNED DeltaHead (infra #35), or None."""
        if slot is None or slot < 0 or slot >= self.K:
            return None
        if self.slots[slot] is None or self.is_scripted(slot):
            return None
        try:
            return self._materialize(slot)[3]
        except Exception:
            return None

    def writeback_delta(self, slot: int, delta, delta_mag: float = None
                        ) -> None:
        """Persist a practised delta into the slot binding (mirrors
        writeback_actor: without this, individuation would vanish on the
        first LRU eviction)."""
        try:
            b = self.slots[slot]
            if b is None or delta is None:
                return
            b["delta_sd"] = {k: v.detach().to(self.dtype).cpu()
                             for k, v in delta.state_dict().items()}
            if delta_mag is not None:
                b["delta_mag"] = float(delta_mag)
            # the practised counter is what triggers the disk flush
            # (_persist_practised_skills) — without this the delta lived
            # only until the next restart (review finding, 2026-08-09)
            b["practised"] = int(b.get("practised", 0)) + 1
        except Exception:
            pass

    def writeback_actor(self, slot: int, actor) -> None:
        """Persist a practised actor into the slot binding.

        The binding's `actor_sd` is what `_materialize` rebuilds from after an
        LRU eviction, so an improvement that is not written back here would
        silently vanish the moment the cache filled — the skill would appear
        to learn and then forget for reasons invisible in any log.
        """
        b = self.slots[slot]
        if b is None:
            return
        b["actor_sd"] = {k: v.detach().to(self.dtype).cpu()
                         for k, v in actor.state_dict().items()}
        b["practised"] = int(b.get("practised", 0)) + 1

    def skill_action(self, slot: int, obs: np.ndarray,
                     head_mask: Optional[np.ndarray] = None,
                     env: int = 0,
                     proprio: Optional[np.ndarray] = None,
                     feats: Optional[np.ndarray] = None) -> int:
        """One SAMPLED action from the slot's frozen skill policy.

        Flat legacy skills return a primitive by construction (their heads
        were truncated at bind). Conv skills sample over their FULL stored
        head under `head_mask` — the executor's depth/cycle/boundness gate —
        so an action >= the skill's own primitive count is a NESTED
        INVOCATION REQUEST, resolved by the caller through slot_map. A
        returned index is in the SKILL'S OWN action space, valid in today's
        wider primitive space because macros are append-only."""
        b = self.slots[slot]
        actor, cond, enc, _delta = self._materialize(slot)
        obs_t = torch.from_numpy(
            np.asarray(obs, dtype=np.float32)).reshape(1, -1).to(self.device)
        # WHERE FEATURES COME FROM, per family (2026-09-02):
        #   conv|wm  the skill's OWN snapshotted encoder, applied to obs
        #   rssm     the LIVE world model's latent, passed in as `feats`
        #   flat     raw obs
        # rssm MUST NOT fall back to obs_t: the actor was trained on a
        # ~4352-wide latent and obs is 49152 raw pixels, so the matmul would
        # either raise or — if widths ever coincided — feed the skill noise
        # shaped like perception. Refuse loudly instead; a skill that cannot
        # see must not act.
        if b.get("arch") == "rssm":
            if feats is None:
                raise SlotRefused(
                    f"rssm skill in slot {slot} got no latent: skill_action "
                    f"needs feats= from the world model, and obs are pixels")
            # NEVER ROUTE A TENSOR THROUGH numpy HERE (fix 2026-09-03).
            # In production `feats` is the LIVE RSSM latent — a torch tensor
            # on cuda:0 — and np.asarray() on a CUDA tensor raises
            #   TypeError: can't convert cuda:0 device type tensor to numpy
            # This killed the first live run after 4 hours: skill_action only
            # runs once a skill has been minted AND invoked, so the crash
            # waited for the first invocation, then repeated deterministically
            # until the supervisor gave up (3 identical faults).
            # It survived the boot test because that test passed a NUMPY array
            # for feats — the one type production never sends.
            if isinstance(feats, torch.Tensor):
                feats_t = feats.detach().to(
                    device=self.device, dtype=torch.float32).reshape(1, -1)
            else:
                feats_t = torch.as_tensor(
                    np.asarray(feats, dtype=np.float32)
                ).reshape(1, -1).to(self.device)
        else:
            feats_t = enc(obs_t) if enc is not None else obs_t
        feats = feats_t
        # ---- PROPRIOCEPTION (fix 2026-08-01, caught by the boot test) ----
        # The policy input is [features, PROPRIO, gated_knowledge] — the
        # conditioner gate deliberately reads perception only, so proprio sits
        # BETWEEN them. This path rebuilt [features, knowledge] and omitted
        # proprio entirely. It went unnoticed because kdim was being derived
        # by subtraction and so absorbed proprio's width: the tensor shapes
        # matched by coincidence while the conditioner's output was being fed
        # into the slots the actor reads as body state. Both halves were
        # wrong and they cancelled into a runnable, meaningless input.
        # Absent proprio reads NEUTRAL (zeros), which is the same convention
        # the env uses for a sensor it cannot supply — never alarming.
        _pdim = int(b.get("proprio_dim", 0) or 0)
        if _pdim > 0:
            if proprio is not None:
                _p = torch.as_tensor(
                    np.asarray(proprio, dtype=np.float32),
                    device=self.device).reshape(1, -1)[:, :_pdim]
                if _p.shape[1] < _pdim:
                    _p = torch.cat([_p, torch.zeros(
                        1, _pdim - _p.shape[1], device=self.device)], dim=-1)
            else:
                _p = torch.zeros(1, _pdim, device=self.device)
            feats = torch.cat([feats, _p], dim=-1)
        if b["kdim"] > 0:
            if cond is not None and b["ctx"] is not None:
                ctx_t = torch.from_numpy(b["ctx"]).reshape(1, -1).to(
                    self.device)
                # the gate conditions on PERCEPTION ONLY — strip proprio back
                # off before handing features to it
                tail = cond(feats[:, :feats.shape[1] - _pdim] if _pdim
                            else feats, ctx_t)
            else:
                tail = torch.zeros(1, b["kdim"], device=self.device)
            feats = torch.cat([feats, tail], dim=-1)
        logits = actor.action_head(actor.shared(feats))
        # the skill's OWNED adjustment (infra #35): zero at mint, grown by
        # practice — this line is where individuation becomes behaviour
        if _delta is not None:
            try:
                _d = _delta(feats)
                if _d.shape == logits.shape:
                    logits = logits + _d
            except Exception:
                pass
        # SAME CAP AS THE SHARED POLICY, SAME ORDER: bound the full head
        # (delta included, so practice cannot route around it), THEN mask —
        # exactly as ActorCriticAgent.select_action does on its masked path.
        # The bound mean-centres every row (softmax-invariant, but not bit-
        # exact), so a row the rescale did NOT engage on keeps its original
        # logits: a skill already inside the cap samples byte-identically.
        if self.logit_bound is not None:
            _b = self.logit_bound(logits)
            if _b is not logits:
                def _spr(t):
                    return (t.max(dim=-1, keepdim=True).values
                            - t.min(dim=-1, keepdim=True).values)
                _hit = _spr(_b) < _spr(logits) * (1.0 - 1e-5)
                logits = torch.where(_hit, _b.to(logits.dtype), logits)
        if head_mask is not None:
            m = torch.as_tensor(np.asarray(head_mask, dtype=bool),
                                device=logits.device).reshape(1, -1)
            if int(m.shape[1]) == int(logits.shape[1]):
                logits = logits.masked_fill(~m, -1e9)
            else:  # never crash a run on a stale mask; fall back to prims
                logits = logits[:, :int(b.get("p_own") or self.P)]
        elif b.get("arch") in ("conv", "wm", "rssm"):
            # no mask supplied (e.g. a caller predating nesting): an encoded-
            # family skill must still never emit a raw slot row.
            # "rssm" added 2026-09-02 — it keeps its full head like conv/wm,
            # so omitting it here let an unmasked call sample an option-slot
            # row and return it as if it were a primitive. That is the
            # 10->12 widening postmortem's failure mode exactly: slot logits
            # becoming buttons.
            logits = logits[:, :int(b.get("p_own") or self.P)]
        a = int(torch.distributions.Categorical(logits=logits).sample())
        # PRACTICE CAPTURE: remember what this skill saw and did, so a
        # successful invocation can be learned from when it closes. `feats` is
        # the actor's actual input (post-encoder, post-conditioner), which is
        # both what the update needs and ~200x smaller than the raw frame.
        if self.practice is not None and self.practice.enabled:
            self.practice.record(env, slot, feats, a)
        return a


class OptionRuntime:
    """Per-env option execution state (one FRAME: root or nested)."""

    __slots__ = ("slot", "skill_id", "steps_done", "t_start",
                 "r_disc_acc", "gamma_pow", "decision", "effects", "depth",
                 "err_sum", "err_n")

    def __init__(self):
        self.slot: Optional[int] = None
        self.skill_id: Optional[str] = None
        self.steps_done = 0
        self.t_start = 0
        self.r_disc_acc = 0.0
        self.gamma_pow = 1.0
        self.decision: Optional[Dict] = None   # env 0 only
        # effect keys (break_X / craft_Y) observed WHILE this frame was open
        # — the ground truth for "did this skill produce ITS OWN effect",
        # which is what mastery counts (a reward spike is not evidence; see
        # SkillBank.record_asked's docstring for the measured counterexample)
        self.effects: set = set()
        self.depth = 1                          # 1 = meta-policy invocation
        # forward-model surprise accumulated over this invocation (primary
        # stream only) — the numerator of the WM-fidelity ratio
        self.err_sum = 0.0
        self.err_n = 0

    @property
    def active(self) -> bool:
        return self.slot is not None

    def open(self, slot: int, skill_id: str, t: int,
             decision: Optional[Dict], depth: int = 1) -> None:
        self.slot, self.skill_id = slot, skill_id
        self.steps_done, self.t_start = 0, t
        self.r_disc_acc, self.gamma_pow = 0.0, 1.0
        self.decision = decision
        self.effects = set()
        self.depth = int(depth)
        self.err_sum, self.err_n = 0.0, 0

    def close(self) -> None:
        self.slot = None
        self.skill_id = None
        self.decision = None
        self.effects = set()


class OptionExecutor:
    """Action resolver + SMDP bookkeeping for the waking parallel path."""

    def __init__(self, bank: SkillOptionBank, num_envs: int,
                 cfg: Optional[Dict] = None, gamma: float = 0.99,
                 on_start=None, on_end=None):
        cfg = cfg or {}
        self.bank = bank
        self.num_envs = int(num_envs)
        self.gamma = float(gamma)
        self.max_option_steps = int(cfg.get("max_option_steps", 25))
        self.terminate_on_spike = bool(cfg.get("terminate_on_spike", True))
        self.spike_threshold = float(cfg.get("spike_threshold", 0.9))
        self.scouts_use_options = bool(cfg.get("scouts_use_options", True))
        # initiation-set gating (Layer 3 wired into Layer 4)
        self.gate_by_preconditions = bool(
            cfg.get("gate_by_preconditions", False))
        self.precondition_min_match = float(
            cfg.get("precondition_min_match", 0.6))
        # condition-based warmup: options for a skill turn on only once its
        # competence crosses this floor (0 disables the gate)
        self.competence_floor = float(cfg.get("competence_floor", 0.0))
        self.warmup_gated = 0
        self.gated_offers = 0
        self.total_offers = 0
        # OFFERED-vs-CHOSEN (2026-07-25). Without these two counters there is
        # no way to tell "the gate never offered the option" from "the option
        # was offered and the meta-policy never picked it" — opposite bugs
        # with opposite fixes. Diagnosing an option stall cost hours precisely
        # because this distinction was unobservable from the logs.
        self.decisions_open = 0        # decision points with no option active
        self.decisions_with_offer = 0  # ...of those, >=1 learned slot offered
        self.option_picks = 0          # ...of those, an option was chosen
        # [competence probation] — see _apply_probation(). Option outcomes now
        # close the gate on failing skills, so there MUST be a way back open or
        # it is a one-way latch (the 4-times-repeated bug in this project).
        self.retry_after = int(cfg.get("gate_retry_after", 3000))
        self._last_inv_t: Dict[int, int] = {}
        # CONTACT GATE (2026-08-07): an optional per-slot veto the loop can
        # install (a callable binding -> bool, True = may be offered). Why:
        # the chop option's whole 80-tick budget used to start wherever the
        # meta-policy happened to invoke it — MEASURED 61 invocations of
        # break_spruce_log with 0 successes because the budget was spent
        # walking, not chopping. With a grounded "target under the gaze /
        # in reach" sense available, offering the option only at contact
        # makes the budget start where the work does. Vetoes are counted so
        # a wrongly-strict gate is visible in the segment log.
        self.contact_gate = None
        self.contact_vetoes = 0
        self.probation_offers = 0
        # LEARNED-only offer accounting (the scripted slot bypasses the gate,
        # so it must not be counted when asking "is the gate closing?").
        self._learned_idx = None          # lazy: slots are bound after init
        self.learned_offered_sum = 0
        self.decisions_with_learned = 0
        self.runtimes = [OptionRuntime() for _ in range(self.num_envs)]
        # Per-scout primitive decision records (2026-09-01), the mirror of
        # `_primary_primitive`. Index 0 is unused and stays None so the two
        # paths can never be confused for one another.
        self._scout_primitive: List[Optional[Dict]] = [
            None] * self.num_envs
        # Scout option-steps whose reward could not be attributed to any
        # storable decision (see observe_scouts). Reported, never silent.
        self.scout_rows_dropped = 0
        # ---- BOUNDED NESTING (arch v3) -----------------------------------
        # `substacks[e]` holds frames ABOVE the root runtime: a conv skill
        # sampling one of its slot rows pushes a child frame here. The root
        # runtime stays the ONLY thing the meta-policy's SMDP path sees, so
        # credit assignment is untouched — the root's tau simply spans every
        # env step consumed, nested or not. Depth 1 = a plain option; the
        # config cap bounds total depth (2 = one level of skill-in-skill).
        self.max_skill_depth = max(1, int(cfg.get("max_skill_depth", 1)))
        self.substacks: List[List[OptionRuntime]] = [
            [] for _ in range(self.num_envs)]
        self.nested_pushes = 0
        self.nested_cycle_blocks = 0
        self.nested_depth_blocks = 0
        # every effect key the env has EVER emitted. A skill whose own key
        # has never been observable cannot be scored — see _record_mastery.
        self._effect_vocab: set = set()
        self.mastery_unmeasurable = 0
        self.mastery_truncated = 0
        # telemetry
        self.events: deque = deque(maxlen=512)
        self.co_log: deque = deque(maxlen=1024)
        self.slot_stats: Dict[int, Dict] = {}
        self.last_closed_this_step: List[Optional[str]] = (
            [None] * self.num_envs)
        # brain-emitter hooks (option_started/option_ended)
        self.on_start = on_start
        self.on_end = on_end

    # ---- action resolution ---------------------------------------------------
    def _apply_probation(self, mask: np.ndarray, timestep: int) -> np.ndarray:
        """[competence probation] Re-offer a gated slot that has gone stale.

        Option outcomes now drive competence (see
        DevelopmentalAI._competence_from_option), so a skill that keeps failing
        drops below `competence_floor` and stops being offered — which is the
        whole point. But gating on failure ALONE is a ONE-WAY LATCH: gated ->
        never offered -> never invoked -> no new outcomes -> can never reopen.
        That exact shape has bitten this project four times already.

        So a gated slot is force-offered again once `gate_retry_after`
        timesteps have passed since its last invocation. It gets one honest
        chance to prove itself: succeed and its competence climbs back above
        the floor on its own; fail and it simply returns to probation. The
        cost is bounded — at most one stale re-offer per slot per window.

        `gate_retry_after <= 0` disables probation (and restores the latch).
        """
        if self.retry_after <= 0:
            return mask
        P = self.bank.P
        out, copied = mask, False
        for s, b in enumerate(self.bank.slots):
            if b is None or b.get("scripted"):
                continue
            idx = P + s
            if idx >= len(out) or out[idx]:
                continue                      # already offered: nothing to do
            if timestep - self._last_inv_t.get(s, 0) < self.retry_after:
                continue                      # not stale yet
            if not copied:                    # never mutate the shared base
                out, copied = mask.copy(), True
            out[idx] = True
            self.probation_offers += 1
        return out

    def act(self, obs_list, primary_kv, policy, timestep: int,
            predicates_per_env: Optional[List[Dict[str, bool]]] = None,
            competence: Optional[Dict[str, float]] = None,
            proprio_per_env: Optional[List] = None,
            feats_per_env: Optional[List] = None,
            verify_feats: bool = False
            ) -> List[int]:
        """Resolve one primitive action per env; opens options as chosen.

        `predicates_per_env` (optional) is each env's grounded-predicate
        readout; with gating on, it restricts which skills are OFFERED to
        those whose initiation conditions currently hold.
        """
        P = self.bank.P
        # skill copies sample under the shared policy's own logit cap
        _adopt = getattr(self.bank, "adopt_logit_bound", None)
        if _adopt is not None:
            _adopt(policy)

        def _pro(e):
            """This env's self-state, or None. The meta-policy decides WHEN
            to invoke a skill; being starving, hurt or stuck in a menu is
            exactly the kind of thing that decision should depend on."""
            if not proprio_per_env or e >= len(proprio_per_env):
                return None
            return proprio_per_env[e]

        def _feats(e):
            """This env's ALREADY-ENCODED features, or None to recompute.

            The loop encodes every env's next_obs for the RSSM; one step
            later that is exactly this env's obs, so the policy can skip a
            duplicate encoder forward. None is always safe and is what the
            caller passes whenever the frame did NOT carry — an env that
            reset gets a FRESH observation, not last step's next_obs, and
            handing over stale features there would silently act on the
            world that existed before the rebuild.
            """
            if not feats_per_env or e >= len(feats_per_env):
                return None
            return feats_per_env[e]

        def _feats_np(e):
            """This env's decision features as a plain numpy row, or None.

            Only arch='rssm' needs these stored (the update cannot recompute
            a recurrent state from an observation); every other arch
            recomputes from the stored obs and gets None here, so nothing
            changes shape or cost for them.
            """
            if getattr(policy, "arch", "") != "rssm":
                return None
            _f = _feats(e)
            if _f is None:
                return None
            try:
                return np.asarray(
                    _f.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            except Exception:
                return None
        # base mask already applies the competence gate (offered set shrinks
        # to competent skills); precondition gating narrows further per env
        base_mask = self.bank.mask(competence=competence,
                                   competence_floor=self.competence_floor)
        # Cleared every act(): a record present after this call means a
        # decision was MADE THIS STEP on that stream. A stream continuing
        # inside an option made no decision and must store no row — leaving
        # last step's record in place would store the same decision twice.
        self._scout_primitive = [None] * self.num_envs
        # Once per step, before any option can close (see observe_scouts).
        self.last_closed_this_step = [None] * self.num_envs
        actions: List[int] = []
        for e_i in range(self.num_envs):
            rt = self.runtimes[e_i]
            if not rt.active:
                if e_i > 0 and not self.scouts_use_options:
                    _pk = {"proprio": _pro(e_i)} if _pro(e_i) is not None else {}
                    # MASKED (fix 2026-08-02): this path passed no mask at all,
                    # so scouts ignored the phase-scoped action space and fed
                    # the shared world model experience the primary can never
                    # produce. Primitives only — scouts opt out of options by
                    # construction here, which is what the branch means.
                    _pmask = getattr(self.bank, "primitive_mask", None)
                    if _pmask is not None:
                        _pk["action_mask"] = _pmask(P + int(
                            getattr(self.bank, "K", 0) or 0))
                    a, _info = policy.select_action(
                        obs_list[e_i], feats=_feats(e_i),
                        verify_feats=verify_feats, **_pk)
                    # RECORD IT (2026-09-01): this decision was made by the
                    # shared policy under a known mask, so it is on-policy
                    # experience and belongs in the rollout. See
                    # `_scout_primitive`.
                    self._scout_primitive[e_i] = {
                        "action": int(a) if int(a) < P else 0,
                        "log_prob": _info["log_prob"],
                        "value": _info["value"],
                        "mask": (_pk["action_mask"].copy()
                                 if "action_mask" in _pk else None),
                        "proprio": (None if _pro(e_i) is None
                                    else np.asarray(_pro(e_i)).copy()),
                        "feats": _feats_np(e_i)}
                    actions.append(int(a) if int(a) < P else 0)
                    continue
                kv = primary_kv if e_i == 0 else None
                if (self.gate_by_preconditions and predicates_per_env
                        and e_i < len(predicates_per_env)):
                    mask = self.bank.mask(
                        predicates_per_env[e_i],
                        min_match=self.precondition_min_match,
                        competence=competence,
                        competence_floor=self.competence_floor)
                    self.total_offers += 1
                    if mask.sum() < base_mask.sum():
                        self.gated_offers += 1
                else:
                    mask = base_mask
                mask = self._apply_probation(mask, timestep)
                # contact gate (see __init__): veto slots whose target is not
                # engaged right now. Copy-on-veto — the mask may still BE the
                # shared base_mask.
                if self.contact_gate is not None:
                    _vetoed = False
                    for _s, _b in enumerate(self.bank.slots):
                        if (_b is not None and mask[P + _s]
                                and not self.contact_gate(_b)):
                            if not _vetoed:
                                mask = mask.copy()
                                _vetoed = True
                            mask[P + _s] = False
                            self.contact_vetoes += 1
                # OFFERED-vs-CHOSEN accounting, recorded at the decision point
                # itself so the two failure modes are distinguishable later.
                self.decisions_open += 1
                if bool(mask[P:].any()):
                    self.decisions_with_offer += 1
                # LEARNED-slot offers, counted SEPARATELY. `mask[P:].any()`
                # above includes the SCRIPTED slot, which bypasses the
                # competence gate — so that ratio reads 100% forever even if
                # every learned skill is gated, and cannot show the gate
                # working. This is the number that can.
                if self._learned_idx is None:
                    self._learned_idx = np.array(
                        [P + s for s, b in enumerate(self.bank.slots)
                         if b is not None and not b.get("scripted")],
                        dtype=int)
                if self._learned_idx.size:
                    _nl = int(mask[self._learned_idx].sum())
                    self.learned_offered_sum += _nl
                    if _nl:
                        self.decisions_with_learned += 1
                # only pass the kwarg when there IS a body sense, so a
                # policy (or test double) predating proprioception still works
                _pk = {"proprio": _pro(e_i)} if _pro(e_i) is not None else {}
                a, info = policy.select_action(
                    obs_list[e_i], knowledge=kv, action_mask=mask,
                    feats=_feats(e_i), verify_feats=verify_feats, **_pk)
                if int(a) >= P:
                    self.option_picks += 1
                if a < P:
                    _rec = {
                        "action": int(a), "log_prob": info["log_prob"],
                        "value": info["value"], "mask": mask.copy(),
                        "proprio": (None if _pro(e_i) is None
                                    else np.asarray(_pro(e_i)).copy()),
                        # arch='rssm': the latent this decision was made on.
                        # Captured HERE, at the decision, because the update
                        # cannot recompute it and by close time the belief
                        # has moved on. None for every other arch.
                        "feats": _feats_np(e_i)}
                    if e_i == 0:
                        # primitive decision: recorded by the caller's store
                        # block exactly as before (tau=1)
                        self._primary_primitive = _rec
                    else:
                        self._scout_primitive[e_i] = _rec
                    actions.append(int(a))
                    continue
                slot = int(a) - P
                # DECISION RECORDED FOR EVERY STREAM (2026-09-01), not just
                # the primary. A scout's option invocation is the same
                # shared policy making the same kind of choice; without the
                # record its SMDP row could not be stored and the whole
                # stream stayed invisible to PPO.
                decision = {
                    "obs": np.asarray(obs_list[e_i]),
                    # THE BELIEF AT THE MOMENT OF CHOOSING. An option runs for
                    # tau steps and the latent moves the whole time, so this
                    # has to be frozen at the decision or the update would
                    # score the choice against a state the option itself
                    # produced. None for every arch but 'rssm'.
                    "feats": _feats_np(e_i),
                    "meta_action": int(a),
                    "log_prob": info["log_prob"],
                    "value": info["value"],
                    "kv": (None if (e_i != 0 or primary_kv is None)
                           else np.asarray(primary_kv).copy()),
                    "proprio": (None if _pro(e_i) is None
                                else np.asarray(_pro(e_i)).copy()),
                    "mask": mask.copy()}
                if e_i == 0:
                    self._primary_primitive = None
                else:
                    self._scout_primitive[e_i] = None
                rt.open(slot, self.bank.slots[slot]["skill_id"],
                        timestep, decision)
                self.bank.note_invoked(slot)
                self._last_inv_t[slot] = int(timestep)   # probation clock
                self._start_event(e_i, rt, timestep)
            # inside an option (fresh or continuing): a scripted macro emits a
            # fixed action (using the option's step count), else the frozen
            # skill stack resolves one — possibly pushing nested frames.
            if self.bank.is_scripted(rt.slot):
                actions.append(
                    self.bank.scripted_action(rt.slot, rt.steps_done))
            else:
                # A BOUND SKILL FEELS ITS BODY (2026-08-01). `act` already
                # received proprio_per_env for the meta-policy; the skill
                # stack was the one consumer that never got it, so every
                # invoked skill ran with a neutral (all-zero) body — unable
                # to tell starving from fed, or in-a-menu from in-the-world,
                # in exactly the frames where it is in control.
                _pr = (proprio_per_env[e_i]
                       if (proprio_per_env is not None
                           and e_i < len(proprio_per_env)) else None)
                # LATENT FOR arch="rssm" skills. `act` already receives
                # feats_per_env for the meta-policy; the option stack was the
                # one consumer that never got it, which is the same omission
                # the proprio note above records. Without it an rssm skill
                # raises SlotRefused on every invocation.
                _ft = (feats_per_env[e_i]
                       if (feats_per_env is not None
                           and e_i < len(feats_per_env)) else None)
                actions.append(
                    self._stack_action(e_i, obs_list[e_i], timestep,
                                       proprio=_pr, feats=_ft))
        assert all(0 <= a < P for a in actions), actions
        return actions

    # ---- bounded skill-invokes-skill (arch v3) -----------------------------
    def _child_mask(self, b: Dict, stack_ids: set, depth: int) -> np.ndarray:
        """Head mask for a conv frame: which of ITS actions are allowed NOW.

        Primitives: always. A slot row is open only when ALL hold:
          * depth budget remains (depth < max_skill_depth),
          * the row's mint-time skill IDENTITY (slot_map) is CURRENTLY bound
            somewhere (paging may have evicted it — positions lie, so the row
            is resolved by identity or not at all),
          * the target is not scripted and not already on this env's stack
            (cycle detection: a skill may never invoke itself or an ancestor).
        """
        width = int(b["head_dim"])
        p_own = int(b["p_own"])
        m = np.zeros(width, dtype=bool)
        m[:p_own] = True
        # PHASE SCOPING APPLIES INSIDE OPTIONS TOO (fix 2026-08-02).
        # `m[:p_own] = True` opened every primitive unconditionally, so an
        # invoked skill could select a macro the current developmental phase
        # has disabled — reopening the inventory screen the scoping exists to
        # close. A skill's own head may be narrower than the live macro table
        # (append-only indices), hence the p_own bound.
        # getattr, not attribute access: the executor accepts any bank-like
        # object (test doubles, and older pickled banks predating the
        # phase-scoped action space). A missing scoping means "nothing is
        # disabled", which is the pre-2026-08-02 behaviour — never a crash on
        # the first NESTED invocation, i.e. only under load.
        for _d in (getattr(self.bank, "disabled_macros", None) or ()):
            _d = int(_d)
            if 0 <= _d < p_own:
                m[_d] = False
        if not m[:p_own].any():           # never mask a skill mute
            m[:p_own] = True
        smap = b.get("slot_map") or {}
        if depth >= self.max_skill_depth:
            if smap:
                self.nested_depth_blocks += 1
            return m
        for j_str, child_sid in smap.items():
            row = p_own + int(j_str)
            if row >= width or not child_sid:
                continue
            if child_sid in stack_ids:
                self.nested_cycle_blocks += 1
                continue
            child_slot = self.bank.slot_of_skill(child_sid)
            if child_slot is None or self.bank.is_scripted(child_slot):
                continue
            m[row] = True
        return m

    def _stack_action(self, e_i: int, obs: np.ndarray, timestep: int,
                      proprio: Optional[np.ndarray] = None,
                      feats: Optional[np.ndarray] = None) -> int:
        """Resolve one PRIMITIVE from this env's option stack, pushing nested
        frames when a conv skill invokes a child. Bounded by construction:
        each hop either returns a primitive or increases depth, and the mask
        closes every slot row at max_skill_depth."""
        root = self.runtimes[e_i]
        stack = self.substacks[e_i]
        for _hop in range(self.max_skill_depth + 1):
            frame = stack[-1] if stack else root
            b = self.bank.slots[frame.slot]
            if b is None:
                # slot unbound mid-flight (paging) — close the orphan frame
                if stack:
                    self._pop_frame(e_i, timestep, "unbound")
                    continue
                self._close(e_i, root, timestep, "unbound")
                return 0
            # AN rssm SKILL WITHOUT A LATENT CANNOT ACT (2026-09-02).
            # Its actor was trained on world-model latents; obs are raw
            # pixels. Close the option the same way an unbound slot is closed
            # rather than raising: skill_action's own guard would propagate
            # out of act() — nothing on this path catches SlotRefused — and
            # kill the run over a missing optional argument. Degrading to the
            # meta-policy is recoverable; a crash mid-segment is not.
            if b.get("arch") == "rssm" and feats is None:
                self.rssm_no_latent = getattr(self, "rssm_no_latent", 0) + 1
                logger.warning(
                    "option slot %d (%s) is arch=rssm but no latent was "
                    "supplied — closing the option. The caller must pass "
                    "feats_per_env; without it this skill can never run.",
                    frame.slot, b.get("skill_id"))
                if stack:
                    self._pop_frame(e_i, timestep, "no_latent")
                    continue
                self._close(e_i, root, timestep, "no_latent")
                return 0
            depth = 1 + len(stack)
            stack_ids = {root.skill_id} | {f.skill_id for f in stack}
            # ENCODED FAMILY (conv|wm) keeps its FULL head, so it needs the
            # child mask to nest. Omitting "wm" here would have silently
            # disabled skill-invokes-skill for every skill minted under the
            # shared-perception arch.
            # "rssm" ADDED 2026-09-02 — it keeps its full head exactly like
            # conv/wm, so omitting it here would silently disable
            # skill-invokes-skill for every rssm skill, which is precisely
            # the failure this comment already records for "wm".
            if b.get("arch") in ("conv", "wm", "rssm"):
                mask = self._child_mask(b, stack_ids, depth)
                a = self.bank.skill_action(frame.slot, obs, head_mask=mask,
                                           env=e_i, proprio=proprio,
                                           feats=feats)
            else:
                a = self.bank.skill_action(frame.slot, obs, env=e_i,
                                           proprio=proprio, feats=feats)
            p_own = int(b.get("p_own") or self.bank.P)
            if a < p_own:
                return int(a)
            # nested invocation request: resolve IDENTITY through slot_map
            child_sid = (b.get("slot_map") or {}).get(str(a - p_own))
            child_slot = (None if child_sid is None
                          else self.bank.slot_of_skill(child_sid))
            if child_slot is None:          # stale between mask and resolve
                return int(a % p_own)
            child = OptionRuntime()
            child.open(child_slot, child_sid, timestep, None,
                       depth=depth + 1)
            stack.append(child)
            self.nested_pushes += 1
            # deliberately NOT bank.note_invoked / st["invocations"]: those
            # mean "the meta-policy chose this", and a nested push is a
            # SKILL's choice. Counted under nested_* in _pop_frame.
            self.slot_stats.setdefault(child_slot, {
                "invocations": 0, "taus": 0, "spikes": 0})
            self.events.append({
                "env": e_i, "slot": child_slot, "skill_id": child_sid,
                "t_start": int(timestep), "t_end": None,
                "outcome": "running", "depth": child.depth,
                "parent": frame.skill_id})
            try:      # EARNED composition edge (never kill a run over it)
                self.bank.skill_bank.record_invokes(frame.skill_id,
                                                    child_sid)
            except Exception:
                pass
        # unreachable by construction (mask closes rows at the cap), but a
        # frozen net + a stale mask must degrade to a primitive, not crash
        return 0

    def _pop_frame(self, e_i: int, t: int, outcome: str,
                   truncated: bool = False) -> None:
        """Close the TOP nested frame only (its own horizon/eviction)."""
        stack = self.substacks[e_i]
        if not stack:
            return
        fr = stack.pop()
        # NESTED ACTIVITY IS COUNTED SEPARATELY (review MEDIUM): folding it
        # into `invocations`/`taus` made the same numbers mean two different
        # things and corrupted every per-slot rate derived from them (the
        # option-oscillation diagnosis reads exactly these).
        st = self.slot_stats.setdefault(fr.slot, {
            "invocations": 0, "taus": 0, "spikes": 0})
        st["nested_invocations"] = st.get("nested_invocations", 0) + 1
        st["nested_taus"] = st.get("nested_taus", 0) + fr.steps_done
        if outcome == "spike":
            st["nested_spikes"] = st.get("nested_spikes", 0) + 1
        for ev in reversed(self.events):
            if (ev["env"] == e_i and ev["t_end"] is None
                    and ev.get("depth", 1) == fr.depth):
                ev["t_end"] = int(t)
                ev["outcome"] = outcome
                break
        if not truncated:
            self._record_mastery(fr)
            self._practice(e_i, fr)
        else:
            # A truncated child was interrupted by its PARENT's clock, not by
            # its own performance. The ledger abstains from scoring it;
            # practice abstains from learning from it, for the same reason —
            # and its captured trajectory is dropped rather than left to leak
            # into the next invocation of the same slot.
            self.mastery_truncated += 1
            _pr = getattr(self.bank, "practice", None)
            if _pr is not None:
                _pr.discard(e_i, fr.slot)
        fr.close()

    def _collapse_stack(self, e_i: int, t: int, outcome: str) -> None:
        """Close EVERY nested frame (spike/episode_end/root horizon).

        `truncated` marks frames cut short by an ANCESTOR rather than by
        their own budget: on a spike every frame legitimately shares the
        credit, but on `horizon`/`episode_end`/`unbound` a child was simply
        interrupted and scoring it a mastery FAILURE would punish it for its
        parent's clock (review MEDIUM). Those are abstained from instead.
        """
        truncated = outcome != "spike"
        while self.substacks[e_i]:
            self._pop_frame(e_i, t, outcome, truncated=truncated)

    def _advance_substack(self, e_i: int, reward_raw: float,
                          done: bool, timestep: int) -> None:
        """Per-env-step bookkeeping for nested frames, deepest first. A
        spike or episode end collapses the whole stack (the root closes via
        its own _termination on the same signals); a frame that exhausts its
        own budget pops alone and control returns to its parent."""
        stack = self.substacks[e_i]
        if not stack:
            return
        for fr in stack:
            fr.steps_done += 1
        if done:
            self._collapse_stack(e_i, timestep, "episode_end")
            return
        if self.terminate_on_spike and reward_raw >= self.spike_threshold:
            self._collapse_stack(e_i, timestep, "spike")
            return
        while (self.substacks[e_i]
               and self.substacks[e_i][-1].steps_done
               >= self.max_option_steps):
            self._pop_frame(e_i, timestep, "horizon")

    def note_effect(self, e_i: int, key: str) -> None:
        """An effect key (break_X / craft_Y) occurred on env e_i this step.
        Every open frame on that env witnesses it — mastery later asks each
        closing frame whether its OWN key is among them."""
        self._effect_vocab.add(key)
        rt = self.runtimes[e_i]
        if rt.active:
            rt.effects.add(key)
        for fr in self.substacks[e_i]:
            fr.effects.add(key)

    _OWN_EFFECT_RE = re.compile(r"^ach_\d+_(.+)$")

    def _practice(self, e_i: int, rt: OptionRuntime) -> None:
        """PRACTICE ON CLOSE — the step that turns a recording into a memory.

        Reuses the mastery ledger's verdict exactly, including its abstention
        rule: an effect key the environment has never been observed to emit is
        UNMEASURABLE, not failed, and a skill must not be trained (or punished)
        on a verdict that cannot be earned. Scripted slots have no learnable
        actor and are skipped.
        """
        pr = getattr(self.bank, "practice", None)
        if pr is None or not pr.enabled:
            return
        slot, sid = rt.slot, rt.skill_id
        if slot is None or not sid or self.bank.is_scripted(slot):
            pr.discard(e_i, slot if slot is not None else -1)
            return
        m = self._OWN_EFFECT_RE.match(sid)
        if m is None:
            pr.discard(e_i, slot)
            return
        own = m.group(1)
        produced = (None if (own not in self._effect_vocab
                             and own not in rt.effects)
                    else (own in rt.effects))
        try:
            actor = self.bank.resident_actor(slot)
            if actor is None:
                pr.discard(e_i, slot)
                return
            # INDIVIDUATION (infra #35): waking practice trains the slot's
            # OWNED delta; the base actor stays the frozen mint-time record.
            _delta = self.bank.resident_delta(slot)
            info = pr.on_close(e_i, slot, actor, produced, delta=_delta)
            if info and not info.get("reverted"):
                # persist the improvement into the slot binding, so it
                # survives LRU eviction and is written to disk at the next
                # consolidation flush. Without this the skill would "learn"
                # only until its module was evicted from the cache.
                if _delta is not None:
                    self.bank.writeback_delta(slot, _delta,
                                              info.get("delta_mag"))
                else:
                    self.bank.writeback_actor(slot, actor)
        except Exception as e:      # practice must never kill a run
            logger.warning("skill practice failed on slot %s (%s): %s",
                           slot, sid, e)

    def _record_mastery(self, rt: OptionRuntime) -> None:
        """MASTERY LEDGER (task #39, wired 2026-07-26): one counted
        invocation — asked (it ran) and eligible (it passed the offer mask) —
        success iff the skill's OWN grounded effect occurred while it was
        open. Never a reward spike: leaves pay 0.3 yet 'spikes' were logged
        against break_birch_leaves 7 times in 350 invocations — those cannot
        have been leaves. Effect keys are ground truth; spikes are not."""
        sid = rt.skill_id
        if not sid:
            return
        m = self._OWN_EFFECT_RE.match(sid)
        if m is None:
            return                    # scripted/foreign id: not in the ledger
        own = m.group(1)
        # UNMEASURABLE != FAILED (2026-07-27 review, HIGH). Many skills carry
        # ids whose "effect" is not an effect the env can ever emit —
        # `discovered_7`, `act_where_*`, `skill_slot_03`. Scoring those 0 on
        # every invocation is not a measurement, it is a latch: the ledger
        # would drive them to mastery 0.0 permanently and no amount of
        # competent play could move it. Only score a skill whose own key the
        # environment has ACTUALLY been observed to emit; otherwise abstain
        # and say so. Self-correcting: the first real occurrence of the key
        # enters the vocabulary and the skill becomes scorable from then on.
        if own not in self._effect_vocab and own not in rt.effects:
            self.mastery_unmeasurable += 1
            return
        try:
            self.bank.skill_bank.record_asked(
                sid, produced_effect=(own in rt.effects))
        except Exception as e:        # telemetry must never kill a run
            logger.warning("record_asked failed for %s: %s", sid, e)

    # ---- post-step bookkeeping -----------------------------------------------
    def observe_scouts(self, rewards, dones, timestep: int,
                       mixed_rewards=None) -> Dict[int, Dict]:
        """Scout accumulation + termination checks.

        Returns {stream: closed decision record} for every scout option that
        ENDED this step, in the same shape `observe_primary` returns — the
        caller stores them as SMDP rows.

        `mixed_rewards` (optional, per-stream): the shaped reward the policy
        is actually optimizing. Supplied => scout options accumulate a
        discounted return exactly as the primary does and their rows can be
        stored. Omitted => the pre-2026-09-01 behaviour (terminations only,
        no credit assignment) and an empty dict, so any caller that has not
        been updated is unchanged.
        """
        # NOTE: `last_closed_this_step` is reset in act(), once per step,
        # BEFORE any close can happen. It used to be reset here, which was
        # only safe while this ran before the primary's observe_primary —
        # moving scout accumulation after the reward mix (it needs the mixed
        # reward, which does not exist until then) would otherwise have wiped
        # the primary's entry every step.
        closed_by_stream: Dict[int, Dict] = {}
        for e_i in range(1, self.num_envs):
            rt = self.runtimes[e_i]
            if not rt.active:
                continue
            self._advance_substack(e_i, float(rewards[e_i]),
                                   bool(dones[e_i]), timestep)
            if mixed_rewards is not None:
                # ONLY ACCUMULATE WHERE A ROW CAN ACTUALLY BE PRODUCED
                # (2026-09-01). A runtime opened by a path that builds no
                # decision record — a nested frame pushed by _stack_action,
                # or restored runtime state — would otherwise pile up a
                # discounted return that _close() silently throws away, with
                # nothing anywhere saying so. Count those instead: a counter
                # at zero is a proof, a counter climbing is the next finding.
                if rt.decision is not None:
                    rt.r_disc_acc += rt.gamma_pow * float(mixed_rewards[e_i])
                    rt.gamma_pow *= self.gamma
                else:
                    self.scout_rows_dropped += 1
            rt.steps_done += 1
            outcome = self._termination(rt, float(rewards[e_i]),
                                        bool(dones[e_i]))
            if outcome:
                # Snapshot BEFORE _close(), which calls rt.close() and wipes
                # the runtime — the primary path has the same ordering and
                # the same reason.
                if mixed_rewards is not None and rt.decision is not None:
                    _d = rt.decision
                    closed_by_stream[e_i] = {
                        "obs": _d["obs"], "feats": _d.get("feats"),
                        "meta_action": _d["meta_action"],
                        "reward": rt.r_disc_acc, "log_prob": _d["log_prob"],
                        "value": _d["value"], "kv": _d["kv"],
                        "mask": _d["mask"], "tau": rt.steps_done,
                        "outcome": outcome, "skill_id": rt.skill_id,
                        "proprio": _d.get("proprio"),
                        "wm_err_mean": None,
                    }
                self._close(e_i, rt, timestep, outcome)
        return closed_by_stream

    def observe_primary(self, reward_raw: float, done: bool,
                        mixed_reward: float, timestep: int,
                        wm_err: Optional[float] = None
                        ) -> Optional[Dict]:
        """Primary-stream accumulation; returns a CLOSED decision record
        (for the PPO store) when the option ends this step, else None.

        `wm_err` (optional): this step's forward-model surprise. Accumulated
        per invocation so mastery's WM-fidelity ratio (this skill's mean
        surprise / the agent's typical surprise) can be computed at close.
        """
        rt = self.runtimes[0]
        if not rt.active:
            return None
        self._advance_substack(0, float(reward_raw), bool(done), timestep)
        rt.r_disc_acc += rt.gamma_pow * float(mixed_reward)
        rt.gamma_pow *= self.gamma
        rt.steps_done += 1
        if wm_err is not None and np.isfinite(wm_err):
            rt.err_sum += float(wm_err)
            rt.err_n += 1
        outcome = self._termination(rt, float(reward_raw), bool(done))
        if not outcome:
            return None
        dec = rt.decision
        closed = {
            "obs": dec["obs"], "feats": dec.get("feats"),
            "meta_action": dec["meta_action"],
            "reward": rt.r_disc_acc, "log_prob": dec["log_prob"],
            "value": dec["value"], "kv": dec["kv"], "mask": dec["mask"],
            "tau": rt.steps_done, "outcome": outcome,
            "skill_id": rt.skill_id, "proprio": dec.get("proprio"),
            "wm_err_mean": (rt.err_sum / rt.err_n) if rt.err_n else None,
        }
        self._close(0, rt, timestep, outcome)
        return closed

    def _termination(self, rt: OptionRuntime, reward_raw: float,
                     done: bool) -> Optional[str]:
        if done:
            return "episode_end"
        if self.terminate_on_spike and reward_raw >= self.spike_threshold:
            return "spike"
        if rt.steps_done >= self.max_option_steps:
            return "horizon"
        return None

    def _start_event(self, e_i: int, rt: OptionRuntime, t: int) -> None:
        st = self.slot_stats.setdefault(rt.slot, {
            "invocations": 0, "taus": 0, "spikes": 0})
        st["invocations"] += 1
        self.events.append({
            "env": e_i, "slot": rt.slot, "skill_id": rt.skill_id,
            "t_start": int(t), "t_end": None, "outcome": "running"})
        if self.on_start is not None:
            try:
                self.on_start(e_i, rt.skill_id, t)
            except Exception:
                pass

    def _close(self, e_i: int, rt: OptionRuntime, t: int,
               outcome: str) -> None:
        # the root never closes over live children: whatever ends the root
        # (horizon/episode_end/spike) ends every nested frame with it
        self._collapse_stack(e_i, t, outcome)
        st = self.slot_stats.setdefault(rt.slot, {
            "invocations": 0, "taus": 0, "spikes": 0})
        st["taus"] += rt.steps_done
        if outcome == "spike":
            st["spikes"] += 1
        for ev in reversed(self.events):
            if (ev["env"] == e_i and ev["t_end"] is None):
                ev["t_end"] = int(t)
                ev["outcome"] = outcome
                break
        self.last_closed_this_step[e_i] = rt.skill_id
        skill_id = rt.skill_id
        self._record_mastery(rt)
        self._practice(e_i, rt)
        rt.close()
        if self.on_end is not None:
            try:
                self.on_end(e_i, skill_id, t, outcome)
            except Exception:
                pass

    # ---- lifecycle -------------------------------------------------------------
    def clear(self, e_i: int, timestep: int = 0) -> None:
        rt = self.runtimes[e_i]
        if rt.active:
            self._close(e_i, rt, timestep, "episode_end")

    def clear_all(self, timestep: int = 0) -> None:
        for e_i in range(self.num_envs):
            self.clear(e_i, timestep)
        self._primary_primitive = None

    def record_cooccurrence(self, skill_id: str, unlocked_slot: int,
                            stream: int, t: int) -> None:
        self.co_log.append({"skill_id": skill_id,
                            "unlocked_slot": int(unlocked_slot),
                            "stream": int(stream), "t": int(t)})

    def snapshot(self) -> Dict:
        return {
            "active": {str(i): rt.skill_id
                       for i, rt in enumerate(self.runtimes)},
            "recent": list(self.events)[-100:],
            "slots": {str(k): dict(v) for k, v in self.slot_stats.items()},
            "co": list(self.co_log)[-50:],
            "bound": [None if b is None else b["skill_id"]
                      for b in self.bank.slots],
            "offered_vs_chosen": {
                "decisions": self.decisions_open,
                "with_offer": self.decisions_with_offer,
                "picked": self.option_picks,
                "probation": self.probation_offers,
                "retry_after": self.retry_after,
                "with_learned": self.decisions_with_learned,
                "learned_offered_sum": self.learned_offered_sum,
                "scout_rows_dropped": self.scout_rows_dropped,
            },
            "gating": {"enabled": self.gate_by_preconditions,
                       "offers": self.total_offers,
                       "restricted": self.gated_offers,
                       "competence_floor": self.competence_floor},
            "nested": {"max_depth": self.max_skill_depth,
                       "mastery_unmeasurable": self.mastery_unmeasurable,
                       "mastery_truncated": self.mastery_truncated,
                       "pushes": self.nested_pushes,
                       "cycle_blocks": self.nested_cycle_blocks,
                       "depth_blocks": self.nested_depth_blocks},
        }
