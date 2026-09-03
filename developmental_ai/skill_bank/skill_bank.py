"""
Skill Bank — Learned Behavior Storage & Retrieval
====================================================
Stores and retrieves learned policies (skills) that the agent has mastered.
Skills compound over time — earlier skills scaffold later ones, just like
a child learning to grasp enables learning to stack blocks.

Key operations:
  1. DETECT mastery: Has the agent consistently succeeded at a behavior?
  2. SAVE skill: Store the policy weights + metadata for later reuse
  3. RETRIEVE skills: Find relevant prior skills when facing a new challenge
  4. COMPOSE skills: Combine simple skills into complex behaviors

Design inspired by:
  - Autotelic Agents (Colas 2022): store goal representations alongside policies
  - H-GRAIL (2025): hierarchical skill organization with sub-goal decomposition
  - WALL-E 2.0: skill libraries that constrain action selection

The skill bank stores skills as:
  - Policy weights (the actual learned behavior)
  - Goal description (what the skill achieves)
  - Context embedding (what situations the skill applies to)
  - Performance statistics (mastery level, success rate)
  - Prerequisites (what other skills must be learned first)
"""

import torch
import numpy as np
import os
import re
import json
import time
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import logging

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, hf_hub_download, upload_folder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


_GROUNDED_PREFIXES = ("break_", "act_where_", "skill_slot_")
# WEAKLY grounded (2026-07-25): produced by the ground-truth namer's FALLBACK
# paths, not by an actual observed effect. Treating these as fully grounded
# LATCHED a wrong name permanently — `is_grounded_name` gates every rename
# path, so a slot that got `act_where_breakable_in_reach` before its real
# effect was ever attributed could never be corrected to `break_oak_log`.
# Same latch shape as the two magnet latches (#40/#41): a guard that is right
# once becomes a permanent block. Only `break_*` is a real, observed effect.
_WEAK_PREFIXES = ("act_where_", "skill_slot_")


def is_grounded_name(name: str) -> bool:
    """True for a name produced by the GROUND-TRUTH namer (break_<block>,
    act_where_<predicate>, skill_slot_NN)."""
    n = (name or "").strip()
    return any(n.startswith(pfx) for pfx in _GROUNDED_PREFIXES)


def is_weakly_grounded_name(name: str) -> bool:
    """True for a FALLBACK name (act_where_*, skill_slot_NN) — one that a
    real observed-effect name (break_*) is allowed to replace."""
    n = (name or "").strip()
    return any(n.startswith(pfx) for pfx in _WEAK_PREFIXES)


def is_placeholder_name(name: str) -> bool:
    """True for auto-generated stand-in names ("Achieve: discovered_3")."""
    n = (name or "").strip()
    return (not n) or n.startswith("Achieve: ") or bool(
        re.fullmatch(r"discovered_\d+", n))


def is_replaceable_name(name: str) -> bool:
    """True when a stored name should yield to a fresh GROUNDED name on
    re-mint. That means placeholders AND stale VLM HALLUCINATIONS
    ("fish_blue_water", "collect_heart_items") — anything that is not
    already a grounded-format name. A grounded name is never overwritten by
    another grounded name (first grounding wins, stable identity)."""
    return (not is_grounded_name(name)) or is_placeholder_name(name)


def _json_default(o):
    """Fallback encoder for json.dump so numpy scalars/arrays are serializable.

    Env-derived values leak numpy types into skill records — e.g. gymnasium's
    Discrete(n).n is an np.int64, and episode counters / metrics can be numpy
    scalars. json.dump only calls this for objects it can't natively encode, so
    native ints/floats are untouched; numpy types are coerced to Python ones.
    Without this, minting a skill raises 'Object of type int64 is not JSON
    serializable' and the whole training run crashes at save time.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(
        f"Object of type {o.__class__.__name__} is not JSON serializable"
    )


# ---------------------------------------------------------------------------
# Skill data structure
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    """
    A single learned skill — a behavior the agent has mastered.

    A skill has:
      - A unique ID and name
      - The policy weights that implement the behavior
      - A goal description (what it achieves)
      - A context vector (what situations it applies to)
      - Performance metrics (how good the agent is at it)
      - Prerequisites (skills that must be learned first)

    Skills are the atoms of capability — they combine to form complex behaviors.
    """
    skill_id: str
    name: str
    description: str = ""

    # When this skill was learned
    created_at: float = 0.0
    updated_at: float = 0.0

    # Performance metrics
    success_rate: float = 0.0
    total_episodes: int = 0
    successful_episodes: int = 0
    avg_reward: float = 0.0
    avg_episode_length: float = 0.0

    # Mastery status
    is_mastered: bool = False
    mastery_level: float = 0.0  # 0.0 to 1.0

    # ---- MASTERY, MEASURED (2026-07-26) ---------------------------------
    # The user's definition: "when called upon to do that task, it knows
    # exactly what to do" — succeeds >=80% of the time it is ASKED.
    #
    # None of that was measurable before. `success_rate` above is NOT a
    # success ratio at all: at mint it is `competence.predict_all()[slot]`,
    # an online BCE logit with no denominator, and `successful_episodes` is
    # literally `int(success_rate * total_episodes)`. `is_mastered` then
    # thresholded that scalar, so a "mastered" skill meant nothing.
    #
    # `asked_log` is a TRAILING window of 0/1 over COUNTED invocations —
    # counted meaning the skill was actually asked for (it was the target)
    # AND its initiation set held, so failures in states where the effect was
    # impossible do not count against it. Trailing (not lifetime) so early
    # fumbling cannot poison a skill forever AND so mastery can be LOST
    # again — skills decay without practice, and a lucky streak should not
    # crown one permanently. Serialized as a plain list of ints.
    asked_log: List[int] = field(default_factory=list)
    asked_total: int = 0          # lifetime counted invocations (context)
    # WM FIDELITY: the second half of the user's definition — the world model
    # can imagine this action with near-fidelity to the real thing. Stored as
    # a RATIO of this skill's mean WM prediction error to the agent's current
    # typical error, so it stays meaningful as the WM improves globally (an
    # absolute threshold would eventually mark everything "mastered").
    # <1.0 means "better predicted than average". None until measured.
    wm_fidelity: Optional[float] = None

    # ---- ARCHITECTURE + COMPOSITION (2026-07-26, arch v3) -----------------
    # These describe the WEIGHTS in policy.pt and must always travel WITH
    # them (an upsert that swaps the policy swaps these too):
    #   arch      — "flat" (Linear on raw pixels, the legacy family) or
    #               "conv" (shared CNN trunk; ~16x smaller, spatial prior).
    #   enc_dim   — conv feature width (0 for flat).
    #   head_dim  — the FULL stored action-head width incl. option-slot rows.
    #               Conv skills keep their meta-width head so they can invoke
    #               other skills; flat legacy heads were truncated at bind.
    #   slot_map  — {str(row_j - P): skill_id} recorded AT MINT: what each
    #               option-slot row of this skill's head MEANT. Slots rebind
    #               over time (paging), so nested invocation resolves the
    #               skill IDENTITY through this map and never trusts the row
    #               POSITION — a frozen skill invokes the skill it learned to
    #               invoke, or nothing.
    #   parent_skills — skills observed running when this skill's effect was
    #               first unlocked (compositional provenance, additive).
    #   invokes   — {child_skill_id: count} nested invocations actually
    #               performed by this skill at runtime (earned, not declared).
    arch: str = "flat"
    enc_dim: int = 0
    head_dim: Optional[int] = None
    slot_map: Optional[Dict[str, str]] = None
    parent_skills: List[str] = field(default_factory=list)
    invokes: Dict[str, int] = field(default_factory=dict)

    # Context: what situations this skill applies to
    # Stored as a numpy array, serialized as a list for JSON
    context_embedding: Optional[List[float]] = None

    # Environment shape this skill's policy was trained for. Used to gate
    # warm-start: a policy can only be loaded into a network with matching
    # obs/action dimensions (loading e.g. a 4-obs CartPole net into a 2-obs
    # MountainCar net would raise a shape mismatch). None on legacy skills
    # minted before this field existed.
    obs_dim: Optional[int] = None
    action_dim: Optional[int] = None

    # WHEN THIS SKILL APPLIES — grounded predicate state at unlock (phase-2
    # vocabulary, e.g. {"tree_visible": True, "breakable_in_reach": True}).
    # This is the legible replacement for embedding-similarity retrieval:
    # "which skills' preconditions match what I currently perceive" is a
    # debuggable question at 16 skills and at 600. None on legacy skills.
    preconditions: Optional[Dict[str, bool]] = None

    # Prerequisites: skill IDs that should be learned first
    prerequisites: List[str] = field(default_factory=list)

    # Goal representation (symbolic description of what the skill achieves)
    goal_facts: List[Dict[str, str]] = field(default_factory=list)

    # Option-invocation lifecycle (skills-as-options, July 2026). These are
    # PRACTICE statistics — the skill's life as a callable behaviour.
    # TWO-LEDGER RULE (hard invariant): option outcomes update THESE fields
    # only; they must never feed the goal broadcaster's CompetencePredictor
    # (within-episode option-level Bernoulli samples would re-inflate
    # mastery exactly like the pre-refactor any-of-N bug).
    invocations: int = 0
    option_spikes: int = 0
    option_spike_ema: float = 0.0
    # PRIMARY stream only (above) vs EVERY stream (below). See
    # record_invocation: the scouts run without the magnet or the VLM, so
    # pooling them answers "works somewhere" while every reader is asking
    # "works here". Reported, not gated on.
    option_spike_ema_all: float = 0.0
    avg_option_return: float = 0.0
    last_invoked_at: float = 0.0

    # File path to saved policy weights
    policy_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Mastery Detector
# ---------------------------------------------------------------------------

class MasteryDetector:
    """
    Determines when the agent has "mastered" a behavior well enough to save it.

    A skill is considered mastered when:
      1. The agent has practiced it for at least min_episodes
      2. The success rate over the last mastery_window episodes exceeds the threshold
      3. Performance is stable (not still rapidly improving or fluctuating)

    This prevents saving half-learned skills that would be unreliable when reused.
    """

    def __init__(
        self,
        mastery_threshold: float = 0.8,
        mastery_window: int = 50,
        min_episodes: int = 20,
        stability_threshold: float = 0.1,
    ):
        self.mastery_threshold = mastery_threshold
        self.mastery_window = mastery_window
        self.min_episodes = min_episodes
        self.stability_threshold = stability_threshold

        # Rolling window of recent performance
        self.reward_history: deque = deque(maxlen=mastery_window)
        self.success_history: deque = deque(maxlen=mastery_window)

    def record_episode(self, reward: float, success: bool) -> None:
        """Record the outcome of one episode."""
        self.reward_history.append(reward)
        self.success_history.append(float(success))

    def check_mastery(self) -> Tuple[bool, float]:
        """
        Check if the current behavior qualifies as mastered.

        Returns:
            (is_mastered, mastery_level) where mastery_level is 0.0 to 1.0
        """
        if len(self.success_history) < self.min_episodes:
            return False, 0.0

        # Calculate success rate over recent window
        recent_successes = list(self.success_history)
        success_rate = np.mean(recent_successes)

        # Check stability: is performance consistent?
        if len(recent_successes) >= 10:
            first_half = np.mean(recent_successes[:len(recent_successes)//2])
            second_half = np.mean(recent_successes[len(recent_successes)//2:])
            stability = abs(second_half - first_half)
            is_stable = stability < self.stability_threshold
        else:
            is_stable = True

        is_mastered = success_rate >= self.mastery_threshold and is_stable
        mastery_level = min(1.0, success_rate / self.mastery_threshold)

        return is_mastered, mastery_level

    def reset(self) -> None:
        """Reset the mastery detector for tracking a new skill."""
        self.reward_history.clear()
        self.success_history.clear()


# ---------------------------------------------------------------------------
# Skill Bank
# ---------------------------------------------------------------------------

class SkillBank:
    """
    Stores and retrieves learned skills.

    The skill bank is the agent's long-term memory of learned behaviors.
    As the agent masters new skills, they are saved here for later reuse.

    Key operations:
      save_skill(): Store a mastered skill's policy and metadata
      retrieve_skills(): Find skills relevant to a new situation
      load_skill(): Load a specific skill's policy weights
      get_prerequisites(): Find skills that should be learned before attempting something

    Storage format:
      Each skill is saved as a directory containing:
        - metadata.json: skill info, performance stats, context
        - policy.pt: PyTorch state dict of the policy network
    """

    def __init__(self, storage_dir: str = "./skill_bank_data",
                 min_practice_episodes: int = 3,
                 policy_versions_kept: int = 5,
                 mastery_milestone: float = 0.8,
                 max_skills_loaded: int = 5,
                 similarity_threshold: float = 0.5):
        self.storage_dir = storage_dir
        # A challenger policy must have at least this much real practice
        # before it may replace stored weights (see save_skill upsert).
        self.min_practice_episodes = int(min_practice_episodes)
        # How many superseded policy versions to keep per skill (provenance:
        # a skill's note can show its own history; a bad overwrite is
        # recoverable instead of destructive).
        self.policy_versions_kept = int(policy_versions_kept)
        # is_mastered is a MILESTONE marker, not an availability gate: skills
        # below it are still fully usable (selection weights by competence).
        # It used to be hardcoded True at save time, which made the
        # "16 mastered" stat meaningless.
        self.mastery_milestone = float(mastery_milestone)
        # ---- RETRIEVAL POLICY (config: skill_bank.max_skills_loaded /
        # skill_bank.similarity_threshold) --------------------------------
        # How many prior skills a new goal may consider, and how relevant a
        # candidate must be before it is allowed to seed the live policy.
        # Both were config-only keys with NO python reader until 2026-08-01
        # (they sat in all 15 configs and did nothing); the retrieval sites
        # hardcoded 3 / 0.7 instead. The bank is the canonical home so every
        # retrieval path answers to one setting.
        #
        # SCALE: `similarity_threshold` is on the NORMALIZED [0,1] relevance
        # scale used by SkillSelector._cosine_score ((cos+1)/2) — NOT raw
        # cosine. 0.5 means "orthogonal", 1.0 means "identical context".
        # Do not reuse it against a raw-cosine comparison; the two scales
        # disagree by exactly the remap and a shared constant would silently
        # mean two different angles.
        self.max_skills_loaded = max(1, int(max_skills_loaded))
        self.similarity_threshold = float(similarity_threshold)
        os.makedirs(storage_dir, exist_ok=True)

        # In-memory skill registry (metadata only, not full policy weights)
        self.skills: Dict[str, Skill] = {}

        # Skill dependency graph: skill_id → [prerequisite_skill_ids]
        self.dependency_graph: Dict[str, List[str]] = {}

        # Load existing skills from disk
        self._load_registry()

    def _load_registry(self) -> None:
        """Load skill metadata from disk on startup.

        Resilient by design: the registry is the skill bank's spine, and a
        truncated/corrupt file used to raise at construction — bricking every
        future run until repaired by hand. Now: fall back to registry.json.bak,
        skip individually-broken entries, and ADOPT orphaned skill dirs (a dir
        with policy.pt but no registry row is recovered, not invisible).
        """
        registry_path = os.path.join(self.storage_dir, "registry.json")
        loaded_from = None
        for candidate in (registry_path, registry_path + ".bak"):
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, "r") as f:
                    data = json.load(f)
                loaded_from = candidate
                break
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Skill registry %s unreadable (%s) — trying "
                             "fallback", candidate, e)
                data = None
        if loaded_from is not None and data is not None:
            for skill_data in data.get("skills", []):
                try:
                    # FORWARD-COMPATIBLE LOAD (2026-07-26). `Skill(**d)` raises
                    # TypeError on ANY key the current dataclass does not
                    # declare — so a registry written by newer code and read by
                    # older code lost the whole entry to the `continue` below,
                    # and the orphan-adoption path then re-added it with
                    # MINIMAL metadata: no preconditions, no context_embedding,
                    # no obs_dim. That silently breaks option binding
                    # (options.py obs_dim check and precondition gating) while
                    # looking like a healthy bank. Unknown keys are now dropped
                    # with a warning instead, so a rollback degrades gracefully
                    # rather than quietly disabling skills.
                    _fields = getattr(Skill, "__dataclass_fields__", None)
                    if _fields is not None:
                        _extra = [k for k in skill_data if k not in _fields]
                        if _extra:
                            logger.warning(
                                "skill %s has unknown fields %s (newer schema?)"
                                " — ignoring them, keeping the skill",
                                skill_data.get("skill_id", "?"), _extra)
                            skill_data = {k: v for k, v in skill_data.items()
                                          if k in _fields}
                    skill = Skill(**skill_data)
                except TypeError as e:  # genuinely malformed entry
                    logger.error("Skipping malformed skill entry %s: %s",
                                 skill_data.get("skill_id", "?"), e)
                    continue
                if skill.policy_path and not os.path.isabs(
                        skill.policy_path):
                    # legacy relative path -> re-anchor on this bank's dir
                    skill.policy_path = os.path.abspath(os.path.join(
                        self.storage_dir, skill.skill_id, "policy.pt"))
                self.skills[skill.skill_id] = skill
            if loaded_from.endswith(".bak"):
                logger.warning("Registry recovered from backup %s",
                               loaded_from)
            logger.info(f"Loaded {len(self.skills)} skills from registry")

        # Adopt orphaned skill directories (present on disk, absent from the
        # registry — e.g. after a partial write or a hand-restored backup).
        try:
            for entry in sorted(os.listdir(self.storage_dir)):
                d = os.path.join(self.storage_dir, entry)
                if (entry not in self.skills and os.path.isdir(d)
                        and os.path.exists(os.path.join(d, "policy.pt"))):
                    logger.warning(
                        "Orphaned skill dir '%s' (policy.pt on disk, no "
                        "registry row) — adopting with minimal metadata",
                        entry)
                    self.skills[entry] = Skill(
                        skill_id=entry, name=entry,
                        description="(recovered from orphaned directory)",
                        created_at=os.path.getmtime(d),
                        updated_at=os.path.getmtime(d),
                        policy_path=os.path.join(d, "policy.pt"),
                    )
        except OSError:
            pass

    def _save_registry(self) -> None:
        """Save the skill registry to disk ATOMICALLY.

        write tmp -> fsync -> keep previous as .bak -> os.replace. A kill
        landing mid-write can no longer leave a truncated registry.json;
        the worst case is an intact previous version.
        """
        data = {
            "skills": [asdict(s) for s in self.skills.values()],
            "last_updated": time.time(),
        }
        registry_path = os.path.join(self.storage_dir, "registry.json")
        tmp_path = registry_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(registry_path):
            try:  # best-effort backup of the last good version
                os.replace(registry_path, registry_path + ".bak")
            except OSError:
                pass
        os.replace(tmp_path, registry_path)

    def record_invocation(self, skill_id: str, outcome: str,
                          r_disc: float, stream: int = 0) -> None:
        """In-memory option-outcome update (no disk I/O — the registry is
        flushed once per episode via flush_if_dirty).

        ---- WHY `stream` EXISTS (2026-09-01) ----------------------------
        Scout streams started storing PPO rows, and with them their option
        closes reach this method for the first time. More invocations is the
        right direction — `invocations = 0 forever` was the 4th
        guard-becomes-latch — but the streams are NOT interchangeable
        evidence: the primary runs on the user's server with the vision
        magnet and the VLM, the scouts run without either. Pooling them makes
        `option_spike_ema` mean "works somewhere", while everything that
        READS it (paging via `_stored_sr`, the competence gate) is asking
        "works here".

        So both are kept. `option_spike_ema` stays PRIMARY-ONLY and therefore
        byte-identical for every existing reader; `option_spike_ema_all`
        pools every stream and is reported so the divergence between the two
        is a number someone can look at. Nothing gates on the pooled figure
        until it has been.
        """
        sk = self.skills.get(skill_id)
        if sk is None:
            return
        sk.invocations += 1
        spiked = 1.0 if outcome == "spike" else 0.0
        if spiked:
            sk.option_spikes += 1
        if not hasattr(sk, "option_spike_ema_all"):
            # skills unpickled from before this field existed
            sk.option_spike_ema_all = sk.option_spike_ema
        sk.option_spike_ema_all = 0.9 * sk.option_spike_ema_all + 0.1 * spiked
        if int(stream) == 0:
            sk.option_spike_ema = 0.9 * sk.option_spike_ema + 0.1 * spiked
        n = sk.invocations
        sk.avg_option_return += (float(r_disc) - sk.avg_option_return) / n
        sk.last_invoked_at = time.time()
        self._invocation_dirty = True

    # ---- MASTERY: "succeeds >=80% of the time it is ASKED" ---------------
    MASTERY_WINDOW = 20        # trailing counted invocations
    MASTERY_MIN_N = 20         # ...and this many before any claim is made
    MASTERY_CONFIRM_N = 50     # tighter tier worth quoting
    WM_FIDELITY_MAX = 1.0      # this skill's WM error <= the agent's typical

    def record_asked(self, skill_id: str, produced_effect: bool) -> None:
        """Record ONE invocation where the skill was ASKED and was ELIGIBLE.

        `produced_effect` must mean "this skill's OWN effect occurred", not
        "some reward arrived". The old `outcome == "spike"` fires on ANY
        reward >= 0.9 during the option, so the ledger credited skills for
        effects they did not produce — proven live: leaves pay 0.3, BELOW the
        0.9 threshold, yet break_birch_leaves logged 7 spikes over 350
        invocations. Those spikes cannot have been leaves.

        The caller is responsible for the two filters (asked, eligible); this
        method only owns the window arithmetic.
        """
        sk = self.skills.get(skill_id)
        if sk is None:
            return
        sk.asked_log = (list(sk.asked_log) + [1 if produced_effect else 0]
                        )[-self.MASTERY_WINDOW:]
        sk.asked_total += 1
        self._recompute_mastery(sk)
        self._invocation_dirty = True

    def record_wm_fidelity(self, skill_id: str, ratio: float) -> None:
        """Fold one invocation's WM-error ratio into the skill's fidelity.

        `ratio` = (mean forward-model error during THIS invocation) /
        (the agent's running typical error). EMA (0.2) so one chaotic
        invocation cannot brand a skill unpredictable forever, and so the
        measure tracks the WM as it improves globally. Feeds directly into
        _recompute_mastery's `predictable` clause.
        """
        sk = self.skills.get(skill_id)
        if sk is None or not np.isfinite(ratio):
            return
        r = float(max(0.0, ratio))
        sk.wm_fidelity = (r if sk.wm_fidelity is None
                          else 0.8 * float(sk.wm_fidelity) + 0.2 * r)
        self._recompute_mastery(sk)
        self._invocation_dirty = True

    def record_invokes(self, parent_id: str, child_id: str) -> None:
        """One EARNED composition edge: `parent_id` (a running skill) invoked
        `child_id` as a nested option. Counts, never declares — the hierarchy
        this builds is whatever the agent actually does."""
        sk = self.skills.get(parent_id)
        if sk is None or not child_id or child_id == parent_id:
            return
        inv = dict(sk.invokes or {})
        inv[child_id] = int(inv.get(child_id, 0)) + 1
        sk.invokes = inv
        self._invocation_dirty = True

    def _recompute_mastery(self, sk: "Skill") -> None:
        """Mastery = enough practice AND >=80% success AND predictable.

        Demotion is automatic: the window trails, so this is re-evaluated on
        every counted invocation and a skill that degrades loses the badge.
        """
        log = list(sk.asked_log or [])
        sk.mastery_level = (sum(log) / len(log)) if log else 0.0
        enough = len(log) >= self.MASTERY_MIN_N
        accurate = sk.mastery_level >= float(self.mastery_milestone)
        # WM fidelity is the SECOND half of the definition: it is not enough
        # to succeed, the world model must also predict what will happen.
        # Unmeasured (None) does not block mastery — it is additive evidence,
        # and gating on a metric that may never be populated would be a latch.
        predictable = (sk.wm_fidelity is None
                       or float(sk.wm_fidelity) <= self.WM_FIDELITY_MAX)
        sk.is_mastered = bool(enough and accurate and predictable)

    def flush_if_dirty(self) -> None:
        """Persist invocation stats at episode boundaries only."""
        if getattr(self, "_invocation_dirty", False):
            self._invocation_dirty = False
            self._save_registry()

    def mine_cooccurrences(self, co_log, slot_to_skill_id,
                           min_count: int = 3) -> int:
        """Fold option/unlock co-occurrences into EARNED prerequisite
        edges: skill A was running when slot B unlocked, >= min_count times
        -> A joins B's skill's prerequisites (additive). The skill graph
        enriched by PRACTICE, not just minting. Never raises."""
        added = 0
        try:
            from collections import Counter
            pairs = Counter((e["skill_id"], int(e["unlocked_slot"]))
                            for e in co_log)
            for (sid_a, slot_b), cnt in pairs.items():
                if cnt < min_count:
                    continue
                sid_b = slot_to_skill_id.get(slot_b)
                if not sid_b or sid_b == sid_a:
                    continue
                sk_b = self.skills.get(sid_b)
                if sk_b is None or sid_a in sk_b.prerequisites:
                    continue
                sk_b.prerequisites.append(sid_a)
                self._invocation_dirty = True
                added += 1
                logger.info("co-occurrence edge: %s -> prerequisite of %s "
                            "(%d co-events)", sid_a, sid_b, cnt)
        except Exception as e:
            logger.warning("co-occurrence mining failed: %s", e)
        return added

    def retrieve_by_preconditions(
        self,
        current_predicates: Dict[str, bool],
        top_k: int = 8,
        min_match: float = 0.5,
    ) -> List[Tuple["Skill", float]]:
        """Which skills apply to what the agent currently perceives?

        The legible retrieval path (July 2026): score = fraction of a
        skill's recorded preconditions matched by the current grounded
        predicates, weighted by competence (floor 0.3 — weak skills pull
        less, never vanish). Deterministic, debuggable ("it fired because
        tree_visible+breakable_in_reach matched"), and O(skills) with a
        16-key dict compare — identical cost at 16 skills or 600.

        Skills without preconditions (legacy, or minted before phase-2
        grounding produced evidence) are not retrievable here — they remain
        reachable via select_skills. min_match keeps half-matching skills
        out of the eligible set.
        """
        if not current_predicates:
            return []
        scored: List[Tuple[Skill, float]] = []
        for skill in self.skills.values():
            pre = skill.preconditions
            if not pre:
                continue
            hits = sum(1 for k, v in pre.items()
                       if current_predicates.get(k) == bool(v))
            match = hits / len(pre)
            if match < min_match:
                continue
            weight = 0.3 + 0.7 * float(skill.success_rate)
            scored.append((skill, match * weight))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _version_policy(self, skill_dir: str, policy_path: str,
                        existing: "Skill") -> None:
        """Archive the current policy.pt before it is overwritten.

        versions/policy_v{n}_s{success:.2f}.pt — bounded to
        `policy_versions_kept`, oldest evicted. Overwriting used to be
        destructive; now a skill keeps its own history.
        """
        if not os.path.exists(policy_path):
            return
        try:
            versions_dir = os.path.join(skill_dir, "versions")
            os.makedirs(versions_dir, exist_ok=True)
            stamped = os.path.join(
                versions_dir,
                f"policy_v{int(existing.updated_at)}_"
                f"s{existing.success_rate:.2f}.pt")
            os.replace(policy_path, stamped)
            # bounded: evict oldest beyond the cap
            versions = sorted(
                f for f in os.listdir(versions_dir) if f.endswith(".pt"))
            for old in versions[:-self.policy_versions_kept]:
                os.remove(os.path.join(versions_dir, old))
        except OSError as e:
            logger.warning("Could not version policy for %s: %s",
                           existing.skill_id, e)

    def save_skill(
        self,
        skill_id: str,
        name: str,
        policy_state_dict: Dict,
        description: str = "",
        success_rate: float = 0.0,
        total_episodes: int = 0,
        avg_reward: float = 0.0,
        context_embedding: Optional[np.ndarray] = None,
        goal_facts: Optional[List[Dict[str, str]]] = None,
        prerequisites: Optional[List[str]] = None,
        preconditions: Optional[Dict[str, bool]] = None,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        dedup: bool = False,
        arch: str = "flat",
        enc_dim: int = 0,
        head_dim: Optional[int] = None,
        slot_map: Optional[Dict[str, str]] = None,
        parent_skills: Optional[List[str]] = None,
    ) -> Skill:
        """
        Save a mastered skill to the skill bank.

        Args:
            skill_id: Unique identifier
            name: Human-readable name
            policy_state_dict: PyTorch state dict of the policy network
            description: What this skill does
            success_rate: Mastery success rate
            total_episodes: How many episodes of practice
            avg_reward: Average reward achieved
            context_embedding: Vector describing when to use this skill
            goal_facts: Symbolic description of what the skill achieves
            prerequisites: Skill IDs that should be learned first

        Returns:
            The saved Skill object
        """
        # ---- ARCH METADATA IS DERIVED FROM THE WEIGHTS, NOT TRUSTED -------
        # (2026-07-27, adversarial review, CRITICAL). The re-distill and
        # _save_skill call sites did not pass the arch kwargs, so a CONV
        # state dict was written to policy.pt while the registry recorded
        # `arch='flat', enc_dim=0`. `_bind` sniffs the family from the sd's
        # own `encoder` key, routes to `_bind_conv`, reads enc_dim=0 from the
        # registry and raises SlotRefused — and `refused` is never cleared
        # and the bad row is persisted. Net effect: the BETTER a skill got,
        # the more permanently it deleted itself from the option bank. The
        # 5th guard-becomes-a-latch in this project.
        #
        # Fixing the two call sites is necessary but not sufficient: any
        # future caller could reintroduce it. The state dict already carries
        # `arch`/`enc_dim` markers (get_state_dict writes them), so derive
        # from the WEIGHTS whenever the caller is silent, and REFUSE a caller
        # whose claim contradicts them. Metadata can then never disagree
        # with the tensors it describes.
        # ENCODED FAMILY = conv | wm. Both ship an "encoder", so the family
        # test is the presence of that key; WHICH member it is comes from the
        # state dict's own `arch` marker. Hardcoding "conv" here (as this did
        # until 2026-08-01) rejected every arch="wm" skill outright — nothing
        # could ever be minted under shared perception.
        _ENC_ARCHS = ("conv", "wm")
        _sd_conv = isinstance(policy_state_dict, dict) and \
            "encoder" in policy_state_dict
        _sd_arch = (policy_state_dict.get("arch")
                    if isinstance(policy_state_dict, dict) else None)
        # LATENT FAMILY = rssm. It carries an `arch` marker but NO "encoder"
        # key, because it reads the LIVE world model's RSSM latents instead of
        # owning an encoder. That makes the `_sd_conv` family test below blind
        # to it: the whole block is gated on "encoder" in the state dict, so an
        # rssm skill fell through every branch and was registered arch="flat"
        # (fix 2026-09-02). `_bind` reads the registry, so those skills were
        # minted successfully and then refused at every invocation — with
        # arch: rssm live that silently disabled skills-as-options, which is
        # the whole SMDP mechanism.
        _sd_latent = (_sd_arch == "rssm")
        if _sd_latent:
            if arch not in ("flat", "rssm"):
                raise ValueError(
                    f"save_skill({skill_id}): caller claims arch={arch!r} but "
                    f"the weights are marked 'rssm' — refusing rather than "
                    f"recording metadata that contradicts the tensors")
            arch = "rssm"
            if not enc_dim:
                enc_dim = int(policy_state_dict.get("enc_dim", 0) or 0)
            if not head_dim:
                _w = policy_state_dict.get("actor", {}).get(
                    "action_head.weight")
                if _w is not None:
                    head_dim = int(_w.shape[0])
        elif _sd_conv:
            if arch not in _ENC_ARCHS:
                if arch != "flat":
                    raise ValueError(
                        f"save_skill({skill_id}): arch={arch!r} contradicts an "
                        f"encoded (conv/wm) state dict")
                # caller silent -> believe the weights' own marker
                arch = _sd_arch if _sd_arch in _ENC_ARCHS else "conv"
            elif _sd_arch in _ENC_ARCHS and arch != _sd_arch:
                raise ValueError(
                    f"save_skill({skill_id}): caller claims arch={arch!r} but "
                    f"the weights are {_sd_arch!r} — the two encoder classes "
                    f"have different state-dict layouts and would not load")
            if not enc_dim:
                enc_dim = int(policy_state_dict.get("enc_dim", 0) or 0)
            if not head_dim:
                _w = policy_state_dict.get("actor", {}).get(
                    "action_head.weight")
                if _w is not None:
                    head_dim = int(_w.shape[0])
        elif arch in _ENC_ARCHS:
            raise ValueError(
                f"save_skill({skill_id}): arch={arch!r} but the state dict has "
                f"no encoder — metadata would describe weights that do not "
                f"exist")

        # Create skill directory
        skill_dir = os.path.join(self.storage_dir, skill_id)
        os.makedirs(skill_dir, exist_ok=True)
        # ABSOLUTE: option binding checks os.path.exists(policy_path), and a
        # relative path silently resolves to nothing if cwd ever differs
        # from the launch directory (legacy registries stored "./...").
        policy_path = os.path.abspath(os.path.join(skill_dir, "policy.pt"))

        existing = self.skills.get(skill_id) if dedup else None

        if existing is not None:
            # --- UPSERT: this environment already has a skill ---
            # Keep the BEST policy on disk (highest success_rate). The
            # comparison is STRICT and requires real practice: it used to be
            # `>=` with no practice floor, and the competence self-model
            # zero-inits to sigmoid(0)=0.5 — so a minutes-old policy at the
            # start of a new run TIED any stored 0.5 skill and silently
            # replaced weights trained over many episodes (observed live:
            # ach_00..ach_03 clobbered on 2026-07-19). Ties now keep the
            # incumbent, and a challenger must have actually been practiced.
            new_context = (
                context_embedding.tolist()
                if context_embedding is not None
                else None
            )
            challenger_wins = (
                success_rate > existing.success_rate
                and total_episodes >= self.min_practice_episodes
            )
            # arch metadata describes the WEIGHTS in policy.pt, so it follows
            # whichever policy wins the upsert — a conv challenger replacing
            # flat weights must flip every arch field with them, and a losing
            # challenger must leave them untouched (a conv slot_map attached
            # to flat weights would be silently corrupt at bind).
            if challenger_wins:
                best_arch, best_enc = arch, int(enc_dim)
                best_head = head_dim
                # SLOT_MAP IS NOT RECOVERABLE FROM WEIGHTS — it is a
                # mint-time record of which skill each option-slot ROW meant,
                # stored nowhere else. A caller that re-saves a policy
                # without supplying one must INHERIT the existing map, never
                # null it: `None` would leave every nested-invocation row
                # dark forever with no way to reconstruct it.
                best_slot_map = (slot_map if slot_map is not None
                                 else existing.slot_map)
            else:
                best_arch = existing.arch
                best_enc = int(existing.enc_dim or 0)
                best_head, best_slot_map = existing.head_dim, existing.slot_map
            if challenger_wins:
                self._version_policy(skill_dir, policy_path, existing)
                torch.save(self._deltaify_encoder(policy_state_dict),
                           policy_path)
                best_success = success_rate
                best_avg_reward = avg_reward
                # New (better) policy → its context/dims are authoritative,
                # falling back to the existing values when not supplied.
                best_context = new_context if new_context is not None else existing.context_embedding
                best_obs_dim = obs_dim if obs_dim is not None else existing.obs_dim
                best_action_dim = action_dim if action_dim is not None else existing.action_dim
                best_name = name
                best_description = description
                best_goal_facts = goal_facts if goal_facts else existing.goal_facts
                best_preconditions = (preconditions
                                      if preconditions is not None
                                      else existing.preconditions)
            else:
                # Existing policy is better — leave policy.pt untouched, but
                # BACKFILL retrieval metadata the old skill lacked (legacy
                # skills were minted with context_embedding/dims = None).
                best_success = existing.success_rate
                best_avg_reward = existing.avg_reward
                best_context = existing.context_embedding if existing.context_embedding is not None else new_context
                best_obs_dim = existing.obs_dim if existing.obs_dim is not None else obs_dim
                best_action_dim = existing.action_dim if existing.action_dim is not None else action_dim
                best_goal_facts = existing.goal_facts
                best_preconditions = (existing.preconditions
                                      if existing.preconditions is not None
                                      else preconditions)
                # METADATA BACKFILL on the losing branch: a legacy skill
                # keeps its (better) weights but ADOPTS a real name and
                # description the moment one is offered. Without this, the
                # 16 pre-naming skills would stay "Achieve: discovered_N"
                # forever, since their weights rarely lose to a fresh run.
                # Name precedence: a GROUNDED name beats any non-grounded
                # one (so break_oak_log overrides a "fish_blue_water"
                # hallucination), and ANY real name beats a placeholder. A
                # grounded name is never overwritten by another grounded one
                # (first grounding wins -> stable identity).
                # ...and a REAL observed-effect name (break_*) may replace a
                # WEAK fallback name (act_where_*/skill_slot_NN), which was
                # previously latched permanently because the fallback counted
                # as "grounded" (2026-07-25 fix).
                _replace_name = (
                    (is_grounded_name(name)
                     and not is_grounded_name(existing.name))
                    or (is_grounded_name(name)
                        and not is_weakly_grounded_name(name)
                        and is_weakly_grounded_name(existing.name))
                    or (is_placeholder_name(existing.name)
                        and not is_placeholder_name(name)))
                if _replace_name:
                    best_name = name
                    best_description = description or existing.description
                else:
                    best_name = existing.name
                    best_description = existing.description

            cumulative_episodes = existing.total_episodes + total_episodes

            skill = Skill(
                skill_id=skill_id,
                name=best_name,
                description=best_description,
                created_at=existing.created_at,          # preserve original
                updated_at=time.time(),
                success_rate=best_success,
                total_episodes=cumulative_episodes,
                successful_episodes=int(best_success * cumulative_episodes),
                avg_reward=best_avg_reward,
                avg_episode_length=existing.avg_episode_length,
                # NOTE: these two are provisional — _recompute_mastery()
                # below overrules them from the PRESERVED asked_log whenever
                # there is one. `best_success` is the competence self-model's
                # logit, not a success ratio (see the asked_log docstring);
                # letting it set the mastery verdict threw away the only
                # real measurement the moment a skill was re-saved.
                is_mastered=best_success >= self.mastery_milestone,
                mastery_level=min(1.0, best_success),
                # OPTION-PRACTICE LEDGER SURVIVES THE UPSERT TOO (review
                # HIGH): rebuilding the Skill without these silently reset
                # invocation/spike history on every re-distill.
                invocations=int(existing.invocations or 0),
                option_spikes=int(existing.option_spikes or 0),
                context_embedding=best_context,
                obs_dim=best_obs_dim,
                action_dim=best_action_dim,
                prerequisites=prerequisites or existing.prerequisites,
                goal_facts=best_goal_facts,
                preconditions=best_preconditions,
                policy_path=policy_path,
                # MASTERY LEDGER SURVIVES UPSERT (2026-07-26): the upsert used
                # to rebuild the Skill without these, silently resetting the
                # asked window / fidelity / invocation graph every re-mint —
                # a metric that zeroes itself on the very event that proves
                # practice is no metric at all.
                asked_log=list(existing.asked_log or []),
                asked_total=int(existing.asked_total or 0),
                wm_fidelity=existing.wm_fidelity,
                arch=best_arch,
                enc_dim=best_enc,
                head_dim=best_head,
                slot_map=best_slot_map,
                parent_skills=sorted(set(list(existing.parent_skills or [])
                                         + list(parent_skills or []))),
                invokes=dict(existing.invokes or {}),
            )

            # THE PRESERVED LEDGER IS THE SOURCE OF TRUTH for mastery.
            # Without this, the competence-derived is_mastered/mastery_level
            # set above would overwrite a verdict that was computed from
            # actual asked/succeeded counts (review HIGH).
            if skill.asked_log:
                self._recompute_mastery(skill)
            self.skills[skill_id] = skill
            self._save_registry()
            logger.info(
                f"Updated skill '{best_name}' (id={skill_id}, "
                f"success_rate={best_success:.2f}, "
                f"practice={cumulative_episodes} eps)"
            )
            return skill

        # --- INSERT: brand-new skill identity ---
        # Shared perception is stored ONCE per bank, not once per skill.
        torch.save(self._deltaify_encoder(policy_state_dict), policy_path)

        skill = Skill(
            skill_id=skill_id,
            name=name,
            description=description,
            created_at=time.time(),
            updated_at=time.time(),
            success_rate=success_rate,
            total_episodes=total_episodes,
            successful_episodes=int(success_rate * total_episodes),
            avg_reward=avg_reward,
            is_mastered=success_rate >= self.mastery_milestone,
            mastery_level=min(1.0, success_rate),
            context_embedding=context_embedding.tolist() if context_embedding is not None else None,
            obs_dim=obs_dim,
            action_dim=action_dim,
            prerequisites=prerequisites or [],
            goal_facts=goal_facts or [],
            preconditions=preconditions,
            policy_path=policy_path,
            arch=arch,
            enc_dim=int(enc_dim),
            head_dim=head_dim,
            slot_map=slot_map,
            parent_skills=sorted(set(parent_skills or [])),
        )

        self.skills[skill_id] = skill
        self._save_registry()

        logger.info(f"Saved skill '{name}' (id={skill_id}, success_rate={success_rate:.2f})")
        return skill

    # ------------------------------------------------------------------
    # Skill de-duplication / consolidation
    # ------------------------------------------------------------------

    @staticmethod
    def _identity_key(skill: Skill) -> str:
        """
        A skill's *identity* is the environment it operates in.

        Historically skills were minted with count-based ids
        (skill_000_Acrobot-v1, skill_001_Acrobot-v1, ...) so a single
        competent policy spawned dozens of near-identical entries. Two
        skills that act in the same environment are the same skill in
        different states of practice, not distinct behaviors — difficulty
        labels were cosmetic (_apply_difficulty was a no-op).

        We recover the environment from, in order of preference:
          0. a GROUNDED behaviour name ("break_birch_leaves") — the same
             grounded effect from different achievement slots is ONE behaviour,
             so offline consolidation can collapse the historical duplication
             (the primary defence is grounded-effect slot keying at mint time;
             this lets a one-time pass clean legacy ach_* duplicates on disk).
          1. the skill name  ("Behavior in <env> (difficulty N)")
          2. the skill_id    ("skill_<n>_<env>" or "skill_<env>")
        Falling back to the raw skill_id keeps genuinely unknown skills
        separate rather than merging them by accident.
        """
        if is_grounded_name(skill.name or ""):
            return f"grounded:{skill.name}"

        m = re.search(r"Behavior in (\S+)", skill.name or "")
        if m:
            return m.group(1)

        sid = skill.skill_id or ""
        # skill_<digits>_<env>  -> <env>
        m = re.match(r"skill_\d+_(.+)$", sid)
        if m:
            return m.group(1)
        # skill_<env>  -> <env>
        m = re.match(r"skill_(.+)$", sid)
        if m:
            return m.group(1)

        return sid

    def _remove_skill_dir(self, skill_id: str) -> None:
        """Delete a skill's on-disk directory (policy.pt + metadata)."""
        skill_dir = os.path.join(self.storage_dir, skill_id)
        if os.path.isdir(skill_dir):
            shutil.rmtree(skill_dir, ignore_errors=True)

    def deduplicate(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Consolidate duplicate skills that share the same environment identity.

        For each identity group:
          - keep the single best member (highest success_rate, then most
            practice) as the canonical skill,
          - re-home it under a deterministic id ("skill_<env>"),
          - accumulate the group's total practice onto the survivor,
          - remove every other member from the registry AND from disk.

        This is CONSOLIDATION, never a wipe — the surviving policy and the
        accumulated experience are preserved. Callers should back up
        skill_bank_data before a non-dry run.

        Args:
            dry_run: if True, report what *would* happen without touching
                     the registry or disk.

        Returns:
            A report dict: before/after counts and per-group decisions.
        """
        # Group skills by environment identity.
        groups: Dict[str, List[Skill]] = {}
        for skill in self.skills.values():
            groups.setdefault(self._identity_key(skill), []).append(skill)

        report: Dict[str, Any] = {
            "before": len(self.skills),
            "groups": {},
            "removed": [],
            "kept": [],
            "dry_run": dry_run,
        }

        new_skills: Dict[str, Skill] = {}
        dirs_to_remove: List[str] = []

        for env, members in groups.items():
            # Best = highest success_rate, then most total practice.
            best = max(
                members,
                key=lambda s: (s.success_rate, s.total_episodes),
            )
            cumulative = sum(s.total_episodes for s in members)
            canonical_id = f"skill_{env}"

            report["groups"][env] = {
                "merged": len(members),
                "kept_from": best.skill_id,
                "canonical_id": canonical_id,
                "best_success_rate": round(best.success_rate, 4),
                "cumulative_episodes": cumulative,
            }
            report["kept"].append(canonical_id)

            # Survivor inherits best stats + accumulated practice.
            survivor = Skill(
                skill_id=canonical_id,
                name=best.name,
                description=best.description,
                created_at=min(s.created_at for s in members),
                updated_at=time.time(),
                success_rate=best.success_rate,
                total_episodes=cumulative,
                successful_episodes=int(best.success_rate * cumulative),
                avg_reward=best.avg_reward,
                avg_episode_length=best.avg_episode_length,
                is_mastered=best.success_rate >= self.mastery_milestone,
                mastery_level=best.mastery_level,
                context_embedding=best.context_embedding,
                obs_dim=best.obs_dim,
                action_dim=best.action_dim,
                prerequisites=best.prerequisites,
                goal_facts=best.goal_facts,
                policy_path=os.path.join(
                    self.storage_dir, canonical_id, "policy.pt"
                ),
            )

            if not dry_run:
                # Re-home the best policy under the canonical id (if needed).
                canonical_dir = os.path.join(self.storage_dir, canonical_id)
                os.makedirs(canonical_dir, exist_ok=True)
                src_policy = best.policy_path
                dst_policy = survivor.policy_path
                if (
                    src_policy
                    and os.path.exists(src_policy)
                    and os.path.abspath(src_policy) != os.path.abspath(dst_policy)
                ):
                    shutil.copyfile(src_policy, dst_policy)

            # Schedule every non-canonical directory for removal.
            for s in members:
                if s.skill_id != canonical_id:
                    report["removed"].append(s.skill_id)
                    dirs_to_remove.append(s.skill_id)

            new_skills[canonical_id] = survivor

        report["after"] = len(new_skills)

        if not dry_run:
            self.skills = new_skills
            self._save_registry()
            for sid in dirs_to_remove:
                self._remove_skill_dir(sid)
            logger.info(
                f"Deduplicated skill bank: {report['before']} -> "
                f"{report['after']} skills "
                f"({len(report['removed'])} removed)"
            )

        return report

    # Shared base actor for delta-form skills. Loaded lazily, once, and
    # reused by every skill in the bank — that is the entire point of the
    # representation: ONE base, many small modulations.
    BASE_ACTOR_FILE = "base_actor.pt"
    # ---- SHARED PERCEPTION, STORED ONCE (2026-08-01) ---------------------
    # Under arch="wm" a skill snapshots the world model's encoder so it stays
    # self-contained when bound into a frozen option slot. MEASURED: that
    # snapshot is 8.11 MB of a 9.56 MB skill — 84.8%, and near-identical
    # across the whole bank, which is the "skills are copies" problem
    # reappearing one level down from where it was just fixed.
    #
    # The fix keeps the self-containment guarantee and drops the cost: the
    # bank stores ONE base encoder, and each skill stores only its DIFFERENCE
    # from it. A skill minted while the encoder is unchanged stores literally
    # nothing (to_delta emits no entry for a tensor equal to the base); a
    # skill minted after the encoder has trained on stores an int8 delta.
    # Reconstruction is exact-to-tolerance and verified by the same
    # machinery as actor deltas, so no skill's perception is altered.
    BASE_ENCODER_FILE = "base_encoder.pt"

    def _shared_base_encoder(self) -> Optional[Dict]:
        if getattr(self, "_base_encoder_cache", None) is not None:
            return self._base_encoder_cache
        path = os.path.join(self.storage_dir, self.BASE_ENCODER_FILE)
        if not os.path.exists(path):
            return None
        try:
            self._base_encoder_cache = torch.load(path, map_location="cpu")
        except Exception as e:
            logger.error("base encoder at %s failed to load (%s)", path, e)
            return None
        return self._base_encoder_cache

    def _deltaify_encoder(self, sd: Dict) -> Dict:
        """Replace a full encoder snapshot with a delta against the bank's
        shared base encoder, writing that base on first use.

        Returns a NEW dict; the caller's state dict is never mutated (the
        same object is often still held by the live policy).
        """
        from developmental_ai.skill_bank import skill_delta as _sdelta
        enc = sd.get("encoder") if isinstance(sd, dict) else None
        if not isinstance(enc, dict) or _sdelta.is_delta(enc):
            return sd
        if not all(hasattr(v, "numel") for v in enc.values()):
            return sd                       # not a tensor dict; leave alone
        base = self._shared_base_encoder()
        if base is None:
            # FIRST skill defines the base. Written before the skill itself so
            # a crash between the two leaves a base with no dependents (
            # harmless) rather than dependents with no base (unloadable).
            path = os.path.join(self.storage_dir, self.BASE_ENCODER_FILE)
            tmp = path + ".tmp"
            torch.save({k: v.detach().cpu().clone() for k, v in enc.items()},
                       tmp)
            os.replace(tmp, path)
            self._base_encoder_cache = None
            base = self._shared_base_encoder()
            if base is None:
                return sd
        if set(base) != set(enc):
            return sd                       # different encoder family: as-is
        try:
            out = dict(sd)
            # ON CPU, like the base beside it (2026-08-17). The base is
            # written with .cpu() and read back with map_location="cpu",
            # while `enc` is the LIVE encoder on the run's device — so on a
            # GPU box this subtraction used to raise a device mismatch,
            # which was caught below and answered by storing the skill
            # whole. Measured on the live pod: 14 MB per skill and the
            # compression win absent on exactly the hardware that trains.
            # Matching the base's device here also keeps CUDA tensors out
            # of the checkpoint file. to_delta aligns devices defensively
            # too; this makes the storage layer's own convention explicit.
            _enc_cpu = {k: v.detach().cpu() for k, v in enc.items()}
            out["encoder"] = _sdelta.to_delta(base, _enc_cpu, tol=0.01)
            return out
        except Exception as e:
            logger.warning("encoder delta failed (%s) — storing in full", e)
            return sd

    def _shared_base_actor(self) -> Optional[Dict]:
        if getattr(self, "_base_actor_cache", None) is not None:
            return self._base_actor_cache
        path = os.path.join(self.storage_dir, self.BASE_ACTOR_FILE)
        if not os.path.exists(path):
            return None
        try:
            self._base_actor_cache = torch.load(path, map_location="cpu")
        except Exception as e:
            logger.error("shared base actor at %s failed to load (%s) — "
                         "delta-form skills cannot be reconstructed", path, e)
            return None
        return self._base_actor_cache

    def load_skill_policy(self, skill_id: str) -> Optional[Dict]:
        """Load a skill's policy weights from disk.

        Transparently reconstructs DELTA-FORM skills (actor stored as
        base + low-rank modulation) so no caller has to know which
        representation a given skill uses. A dense skill is returned exactly
        as before, byte for byte.
        """
        if skill_id not in self.skills:
            logger.warning(f"Skill '{skill_id}' not found in registry")
            return None

        skill = self.skills[skill_id]
        if skill.policy_path and os.path.exists(skill.policy_path):
            sd = torch.load(skill.policy_path, map_location="cpu")
            return self._materialize_delta(skill_id, sd)

        logger.warning(f"Policy file not found for skill '{skill_id}'")
        return None

    def _materialize_delta(self, skill_id: str, sd: Optional[Dict]
                           ) -> Optional[Dict]:
        """Expand a delta-form actor in place; pass anything else through."""
        from developmental_ai.skill_bank import skill_delta as _sdelta
        if not isinstance(sd, dict):
            return sd
        # ---- encoder delta (shared perception, stored once) --------------
        if _sdelta.is_delta(sd.get("encoder")):
            enc_base = self._shared_base_encoder()
            if enc_base is None:
                logger.error(
                    "skill %s stores a delta encoder but %s is missing from "
                    "%s — refusing rather than binding a skill that cannot "
                    "see", skill_id, self.BASE_ENCODER_FILE, self.storage_dir)
                return None
            try:
                sd = dict(sd)
                sd["encoder"] = _sdelta.from_delta(enc_base, sd["encoder"])
            except Exception as e:
                logger.error("encoder reconstruction failed for %s: %s",
                             skill_id, e)
                return None
        if not _sdelta.is_delta(sd.get("actor")):
            return sd
        base = self._shared_base_actor()
        if base is None:
            # LOUD, not silent: a delta skill without its base is unusable,
            # and returning it half-formed would surface later as a baffling
            # shape error inside an option slot.
            logger.error(
                "skill %s is delta-form but %s is missing from %s — the "
                "skill cannot be reconstructed and will be refused",
                skill_id, self.BASE_ACTOR_FILE, self.storage_dir)
            return None
        try:
            out = dict(sd)
            out["actor"] = _sdelta.from_delta(base, sd["actor"])
            return out
        except Exception as e:
            logger.error("delta reconstruction failed for %s: %s",
                         skill_id, e)
            return None

    def retrieve_skills(
        self,
        context_embedding: Optional[np.ndarray] = None,
        max_skills: Optional[int] = None,
        min_mastery: float = 0.0,
        min_similarity: Optional[float] = None,
    ) -> List[Skill]:
        """
        Find skills relevant to the current situation.

        Uses cosine similarity between the current context embedding
        and stored skill context embeddings to find the most relevant
        prior skills.

        This enables transfer learning: when facing a new challenge,
        the agent can start from a relevant prior skill rather than
        learning from scratch.

        Args:
            context_embedding: Current situation vector
            max_skills: Maximum number of skills to return. None ->
                `skill_bank.max_skills_loaded` from config.
            min_mastery: Minimum mastery level for returned skills. Defaults
                to 0.0 because mastery is a MILESTONE, not an availability
                gate (same decision as SkillSelector.select_skills) — a
                0.3-competence skill is a real candidate that ranks lower,
                not an invisible one.
            min_similarity: Minimum NORMALIZED relevance in [0,1] for a
                candidate to be returned at all. None ->
                `skill_bank.similarity_threshold` from config. Only applies
                when a context embedding is supplied.

        Returns:
            List of relevant Skill objects, sorted by relevance
        """
        max_skills = (self.max_skills_loaded if max_skills is None
                      else max(1, int(max_skills)))
        min_similarity = (self.similarity_threshold if min_similarity is None
                          else float(min_similarity))

        # Filter by mastery level
        candidates = [
            s for s in self.skills.values()
            if s.mastery_level >= min_mastery
        ]

        if not candidates:
            return []

        if context_embedding is None:
            # No context → return most mastered skills. The similarity cut
            # cannot apply here (nothing to compare against); returning the
            # top-N by competence is the documented no-context behaviour.
            candidates.sort(key=lambda s: s.mastery_level, reverse=True)
            return candidates[:max_skills]

        context_embedding = np.asarray(context_embedding, dtype=np.float64)

        # Compute similarity scores
        scored = []
        for skill in candidates:
            if skill.context_embedding is not None:
                skill_emb = np.asarray(skill.context_embedding,
                                       dtype=np.float64)
                # DIMENSION-INCOMPARABLE embeddings score 0 instead of
                # raising. A persisted bank routinely mixes goal spaces (a
                # 16-d legacy glue goal vs a 1024-d RSSM unlock latent), and
                # the bare np.dot below used to crash the run the first time
                # a cross-space skill was reached. Mirrors
                # SkillSelector._cosine_score.
                if skill_emb.shape != context_embedding.shape:
                    scored.append((skill, 0.0))
                    continue
                # Cosine similarity, remapped to [0,1] so it shares a scale
                # with `similarity_threshold` and with select_skills().
                cos = float(np.dot(context_embedding, skill_emb) / (
                    np.linalg.norm(context_embedding)
                    * np.linalg.norm(skill_emb) + 1e-8
                ))
                similarity = float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))
            else:
                similarity = 0.0

            scored.append((skill, similarity))

        # Sort by similarity (most relevant first), then drop everything the
        # caller would not want loaded. Without this cut `similarity_threshold`
        # is decorative: an irrelevant skill still came back as long as it
        # placed in the top-N.
        scored.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, sim in scored[:max_skills]
                if sim >= min_similarity]

    # ------------------------------------------------------------------
    # Hugging Face Hub integration
    # ------------------------------------------------------------------

    def push_to_hub(
        self,
        repo_id: str,
        skill_ids: Optional[List[str]] = None,
        token: Optional[str] = None,
        private: bool = True,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Push skills to Hugging Face Hub for sharing and persistence.

        Args:
            repo_id: HF Hub repo (e.g. "username/my-skills")
            skill_ids: Specific skills to push (None = all mastered)
            token: HF token (uses cached login if None)
            private: Whether the repo should be private
            commit_message: Custom commit message

        Returns:
            URL of the uploaded repo
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub not installed. pip install huggingface-hub"
            )

        api = HfApi(token=token)
        api.create_repo(repo_id, exist_ok=True, private=private)

        skills_to_push = skill_ids or [
            s.skill_id for s in self.skills.values() if s.is_mastered
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_skills = []
            for sid in skills_to_push:
                if sid not in self.skills:
                    logger.warning(f"Skill '{sid}' not found, skipping")
                    continue
                skill = self.skills[sid]
                registry_skills.append(asdict(skill))

                skill_src = os.path.join(self.storage_dir, sid)
                skill_dst = os.path.join(tmp_dir, sid)
                if os.path.isdir(skill_src):
                    shutil.copytree(skill_src, skill_dst)

            with open(os.path.join(tmp_dir, "registry.json"), "w") as f:
                json.dump({
                    "skills": registry_skills,
                    "last_updated": time.time(),
                }, f, indent=2, default=_json_default)

            msg = commit_message or (
                f"Push {len(skills_to_push)} skills from developmental-ai"
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=tmp_dir,
                commit_message=msg,
                token=token,
            )

        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Pushed {len(skills_to_push)} skills to {url}")
        return url

    def pull_from_hub(
        self,
        repo_id: str,
        skill_ids: Optional[List[str]] = None,
        token: Optional[str] = None,
        overwrite: bool = False,
    ) -> List[str]:
        """
        Pull skills from Hugging Face Hub into the local skill bank.

        Args:
            repo_id: HF Hub repo (e.g. "username/my-skills")
            skill_ids: Specific skills to pull (None = all)
            token: HF token (uses cached login if None)
            overwrite: Whether to overwrite existing local skills

        Returns:
            List of skill IDs that were imported
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub not installed. pip install huggingface-hub"
            )

        registry_path = hf_hub_download(
            repo_id=repo_id,
            filename="registry.json",
            token=token,
        )
        with open(registry_path, "r") as f:
            remote_data = json.load(f)

        imported = []
        for skill_data in remote_data.get("skills", []):
            sid = skill_data["skill_id"]
            if skill_ids and sid not in skill_ids:
                continue
            if sid in self.skills and not overwrite:
                logger.debug(f"Skill '{sid}' already exists, skipping")
                continue

            skill = Skill(**skill_data)

            skill_dir = os.path.join(self.storage_dir, sid)
            os.makedirs(skill_dir, exist_ok=True)

            try:
                policy_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{sid}/policy.pt",
                    token=token,
                )
                local_policy = os.path.join(skill_dir, "policy.pt")
                shutil.copy2(policy_path, local_policy)
                skill.policy_path = local_policy
            except Exception:
                logger.debug(f"No policy.pt for skill '{sid}'")

            try:
                comp_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{sid}/composition.json",
                    token=token,
                )
                shutil.copy2(
                    comp_path,
                    os.path.join(skill_dir, "composition.json"),
                )
            except Exception:
                pass

            self.skills[sid] = skill
            imported.append(sid)

        if imported:
            self._save_registry()
            logger.info(
                f"Pulled {len(imported)} skills from {repo_id}: "
                f"{', '.join(imported)}"
            )
        return imported

    def get_skill_count(self) -> int:
        """Number of skills in the bank."""
        return len(self.skills)

    def get_mastered_skills(self) -> List[Skill]:
        """Get all fully mastered skills."""
        return [s for s in self.skills.values() if s.is_mastered]

    def get_skill_tree(self) -> Dict[str, List[str]]:
        """
        Get the skill dependency tree (which skills build on which).

        This shows the developmental progression — the order in which
        skills were learned and their prerequisites.
        """
        tree = {}
        for skill in self.skills.values():
            tree[skill.skill_id] = {
                "name": skill.name,
                "prerequisites": skill.prerequisites,
                "mastery_level": skill.mastery_level,
            }
        return tree

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics of the skill bank."""
        skills = list(self.skills.values())
        composite_count = sum(1 for s in skills if s.prerequisites)
        return {
            "total_skills": len(skills),
            "mastered_skills": sum(1 for s in skills if s.is_mastered),
            "composite_skills": composite_count,
            "avg_mastery": np.mean([s.mastery_level for s in skills]) if skills else 0.0,
            "avg_success_rate": np.mean([s.success_rate for s in skills]) if skills else 0.0,
            "total_episodes_trained": sum(s.total_episodes for s in skills),
            "skill_names": [s.name for s in skills],
        }


# ---------------------------------------------------------------------------
# Skill Composer — Detects and creates composite skills
# ---------------------------------------------------------------------------

class SkillComposer:
    """
    Detects and creates composite skills from pairs of mastered skills.

    Composition types:
    1. Sequential: Execute skill A then skill B (A's effects enable B)
    2. Conditional: Use skill A or B depending on observed state

    Detection uses the knowledge graph to find causal chains:
    if skill A's effects satisfy skill B's preconditions, they can chain.
    """

    def __init__(
        self,
        min_mastery: float = 0.7,
        max_compositions_per_cycle: int = 3,
    ):
        self.min_mastery = min_mastery
        self.max_compositions_per_cycle = max_compositions_per_cycle
        self.attempted_pairs: set = set()
        self.compositions_created = 0

    def find_composable_pairs(
        self,
        skill_bank: SkillBank,
        knowledge_graph,
    ) -> List[Tuple[Skill, Skill, float, str]]:
        """
        Find pairs of mastered skills that could be composed.
        Returns list of (skill_a, skill_b, score, composition_type).
        """
        mastered = [
            s for s in skill_bank.skills.values()
            if s.is_mastered and s.mastery_level >= self.min_mastery
        ]

        candidates = []
        for a in mastered:
            for b in mastered:
                if a.skill_id == b.skill_id:
                    continue
                pair_key = (a.skill_id, b.skill_id)
                if pair_key in self.attempted_pairs:
                    continue

                score, comp_type = self._score_composition(
                    a, b, knowledge_graph
                )
                if score > 0.3:
                    candidates.append((a, b, score, comp_type))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[: self.max_compositions_per_cycle]

    def _score_composition(
        self,
        skill_a: Skill,
        skill_b: Skill,
        knowledge_graph,
    ) -> Tuple[float, str]:
        score = 0.0

        causal_score = self._causal_chain_score(
            skill_a, skill_b, knowledge_graph
        )
        score += 0.5 * causal_score

        complement_score = self._complementarity_score(skill_a, skill_b)
        score += 0.3 * complement_score

        temporal_score = self._temporal_score(skill_a, skill_b)
        score += 0.2 * temporal_score

        comp_type = (
            "sequential" if causal_score > complement_score else "conditional"
        )
        return score, comp_type

    def _causal_chain_score(
        self, skill_a: Skill, skill_b: Skill, knowledge_graph
    ) -> float:
        if not skill_a.goal_facts or not skill_b.goal_facts:
            return 0.0

        a_effects = set()
        for fact in skill_a.goal_facts:
            a_effects.add((fact.get("relation", ""), fact.get("object", "")))

        b_preconditions = set()
        for fact in skill_b.goal_facts:
            b_preconditions.add(
                (fact.get("relation", ""), fact.get("subject", ""))
            )

        if not a_effects or not b_preconditions:
            return 0.0

        overlap = len(a_effects & b_preconditions)
        return min(1.0, overlap / max(len(b_preconditions), 1))

    def _complementarity_score(
        self, skill_a: Skill, skill_b: Skill
    ) -> float:
        a_dims = {f.get("subject", "") for f in (skill_a.goal_facts or [])}
        b_dims = {f.get("subject", "") for f in (skill_b.goal_facts or [])}

        if not a_dims or not b_dims:
            return 0.5

        overlap = len(a_dims & b_dims)
        total = len(a_dims | b_dims)
        return 1.0 - (overlap / max(total, 1))

    def _temporal_score(self, skill_a: Skill, skill_b: Skill) -> float:
        if skill_a.created_at == 0 or skill_b.created_at == 0:
            return 0.5
        time_diff = abs(skill_a.created_at - skill_b.created_at)
        return max(0.0, 1.0 - time_diff / 3600.0)

    def compose_skills(
        self,
        skill_a: Skill,
        skill_b: Skill,
        composition_type: str,
        skill_bank: SkillBank,
    ) -> Optional[Skill]:
        """Create a composite skill from two constituent skills."""
        pair_key = (skill_a.skill_id, skill_b.skill_id)
        self.attempted_pairs.add(pair_key)

        composite_id = f"composite_{skill_a.skill_id}_{skill_b.skill_id}"
        if composite_id in skill_bank.skills:
            return None

        merged_goals = list(skill_a.goal_facts or []) + list(
            skill_b.goal_facts or []
        )

        skill = Skill(
            skill_id=composite_id,
            name=f"{skill_a.name} -> {skill_b.name}",
            description=(
                f"{composition_type} composition: "
                f"{skill_a.description} then {skill_b.description}"
            ),
            created_at=time.time(),
            updated_at=time.time(),
            success_rate=0.0,
            total_episodes=0,
            is_mastered=False,
            mastery_level=0.0,
            prerequisites=[skill_a.skill_id, skill_b.skill_id],
            goal_facts=merged_goals[:20],
        )

        skill_bank.skills[composite_id] = skill

        comp_dir = os.path.join(skill_bank.storage_dir, composite_id)
        os.makedirs(comp_dir, exist_ok=True)
        comp_meta = {
            "type": composition_type,
            "skill_a": skill_a.skill_id,
            "skill_b": skill_b.skill_id,
            "switch_condition": "episode_midpoint",
        }
        with open(os.path.join(comp_dir, "composition.json"), "w") as f:
            json.dump(comp_meta, f, indent=2, default=_json_default)

        skill_bank._save_registry()
        self.compositions_created += 1

        logger.info(
            f"Composed skill: '{skill.name}' from "
            f"'{skill_a.name}' + '{skill_b.name}' ({composition_type})"
        )
        return skill


class CompositeSkillExecutor:
    """
    Executes composite skills by switching between constituent policies.

    Sequential: run skill A's policy for first half, switch to B for second half.
    Conditional: select A or B based on observation state.
    """

    def __init__(self, skill_bank: SkillBank):
        self.skill_bank = skill_bank
        self.active_composition: Optional[Dict] = None

    def load_composition(self, composite_skill_id: str) -> bool:
        comp_path = os.path.join(
            self.skill_bank.storage_dir,
            composite_skill_id,
            "composition.json",
        )
        if not os.path.exists(comp_path):
            return False
        with open(comp_path, "r") as f:
            self.active_composition = json.load(f)
        return True

    def get_active_policy_id(
        self,
        current_step: int,
        episode_length: int,
        subgoal_done: bool = False,
    ) -> Optional[str]:
        if self.active_composition is None:
            return None

        comp_type = self.active_composition.get("type", "sequential")

        if comp_type == "sequential":
            # State-aware switch (preferred): hand off from skill_a to skill_b
            # the moment skill_a's subgoal is achieved, rather than guessing an
            # episode midpoint. Falls back to the midpoint split for legacy
            # compositions whose switch_condition isn't "subgoal".
            if self.active_composition.get("switch_condition") == "subgoal":
                return (self.active_composition["skill_b"] if subgoal_done
                        else self.active_composition["skill_a"])
            midpoint = max(episode_length // 2, 1)
            if current_step < midpoint:
                return self.active_composition["skill_a"]
            return self.active_composition["skill_b"]

        if comp_type == "conditional":
            if current_step % 2 == 0:
                return self.active_composition["skill_a"]
            return self.active_composition["skill_b"]

        return self.active_composition["skill_a"]

    def clear(self):
        self.active_composition = None
