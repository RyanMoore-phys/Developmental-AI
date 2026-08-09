"""Achievement goal broadcast — a REAL goal representation (Steps 1-3 of the
rich-env program; replaces hash-random goal embeddings for rich worlds).

WHAT THIS IS
------------
A pluggable broadcaster (same contract as EpisodicWorkingMemory /
RuleRegimeBroadcast: DIM / reset() / update(env) / feature() + lesion hooks)
that gives the policy a GOAL CHANNEL over a world's achievements:

    feature = [ target-slot one-hot | achieved-this-episode mask |
                has-target flag  | target-competence estimate ]

Target selection is IMGEP-style goal babbling driven by the agent's OWN
self-model: a CompetencePredictor (the rung-5 organ) tracks P(success) per
goal slot, and each episode targets a FRONTIER slot — known to exist, not yet
reliably achieved — via score = (1 - P(success)) + UCB novelty, ε-greedy.
Curiosity at the goal level, not the pixel level.

TWO SOURCES, ONE INTERFACE (the pre-registered ablation of rung 10):
  GivenAchievementGoals      — reads env.last_info["achievements"] (the
      oracle tech tree; the PRIVILEGED arm).
  DiscoveredAchievementGoals — reads ONLY reward spikes + observation deltas:
      a spike >= spike_threshold is an "unlock event"; events are clustered
      into slots by a pooled obs-delta signature (cosine matching). The agent
      EARNS the tech tree. Deliberately imperfect (the imperfection
      principle): clusters may split/merge; grading vs ground truth is the
      HARNESS's job (rung-7 convention — oracle names never enter decisions).

HOOKS FOR THE REST OF THE PROGRAM
  pop_skill_mint_events()  — first-time-global unlocks -> per-achievement
      skill minting (Step 2; both 1M rich-env runs minted ZERO skills under
      the old one-skill-per-env @0.8-mastery rule).
  unlock_log / estimated_dag() — prerequisite structure (which slots were
      already achieved when a slot unlocked) -> rulebook evidence, rung-10
      law induction, and the Crafter->Craftax DAG transfer (Step 6).

The env wrapper must expose `last_info` (dict of the latest step's info) and
the standard obs/reward histories (DevelopmentalEnvWrapper does).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import re
import numpy as np
import torch

from developmental_ai.core.self_model import CompetencePredictor

logger = logging.getLogger(__name__)



# ---- FOSSIL SLOT NAMES (2026-07-27) --------------------------------------
# ONE definition, shared by the mint guard here and scripts/retire_fossil_skills.py.
# A fossil is a slot whose NAME encodes no grounded, breakable effect:
#   `discovered_N` — an ungrounded signature cluster (the junk-goal factory)
#   `break_air`    — an effect that is not a block at all
# Minting one produces a "skill" for a behaviour nobody can name, which is how
# this bank filled with 48 slots and 0 usable capabilities. The retire tool
# cleaned them; nothing stopped them being re-minted, so they came back.
FOSSIL_EXACT = frozenset({"break_air", "air"})
FOSSIL_RE = re.compile(r"^discovered_\d+$")


def is_fossil_slot_name(name) -> bool:
    n = str(name or "").strip()
    return (not n) or n in FOSSIL_EXACT or bool(FOSSIL_RE.match(n))

class AchievementGoalBroadcast:
    """Base class: slot registry, IMGEP target selection, broadcast feature."""

    def __init__(self, max_slots: int = 32, epsilon: float = 0.1,
                 competence_mastered: float = 0.85, seed: int = 0,
                 dag_prior: Optional[Dict[str, List[str]]] = None,
                 mint_min_repeats: int = 3, mint_window_eps: int = 12):
        self.max_slots = int(max_slots)
        self.DIM = 2 * self.max_slots + 2
        self.epsilon = float(epsilon)
        self.competence_mastered = float(competence_mastered)
        self._rng = np.random.RandomState(seed)
        # Optional TRANSFERRED prerequisite structure (step 6: rule-level
        # Crafter->Craftax transfer): name -> list of prerequisite names.
        # Target selection damps goals whose known prerequisites are not yet
        # mastered/achieved — "learn to walk before dungeon-diving". A prior,
        # not a constraint: damped, never forbidden (imperfection principle).
        self.dag_prior: Dict[str, List[str]] = dict(dag_prior or {})

        # slot registry: source-specific key -> slot index
        self.slot_of: Dict = {}
        self.slot_names: List[str] = []
        # self-model over goal slots (rung-5 organ, reused)
        self.competence = CompetencePredictor(n_tasks=self.max_slots)
        self._attempts = np.zeros(self.max_slots, dtype=np.int64)
        self._global_unlocks = np.zeros(self.max_slots, dtype=np.int64)

        # per-episode state. PER-STREAM (July 2026): the parallel path runs N
        # concurrent envs, each an independent episode-length attempt by the
        # same policy. A single shared mask made the self-model learn
        # P(ANY of N streams succeeds) — mastery inflates ~N-fold and IMGEP
        # frontier selection moves off goals the agent has not learned. Rows
        # are per-stream; the broadcast feature uses their union (identical to
        # the old behaviour when num_streams == 1, the serial path).
        self.target: Optional[int] = None
        self.num_streams = 1
        self._achieved_now = np.zeros((1, self.max_slots), dtype=np.float32)
        # streams whose window evidence was ALREADY scored (banked at a
        # mid-window clear_stream) — reset() skips them so a death/crash
        # boundary neither erases a real success nor double-counts it.
        self._stream_scored: set = set()
        self._mint_queue: List[Tuple[int, str]] = []
        # ---- REPEATABILITY GATE state (2026-07-26) ----------------------
        # An effect must REPRODUCE across distinct episodes before it earns a
        # skill; one spike is evidence something happened, not evidence of a
        # capability. All fixed-size: no unbounded growth on a multi-day run.
        self.mint_min_repeats = max(1, int(mint_min_repeats))
        self.mint_window_eps = max(1, int(mint_window_eps))
        self._cand_eps: List[deque] = [
            deque(maxlen=self.mint_window_eps) for _ in range(self.max_slots)]
        self._cand_first_ep = np.zeros(self.max_slots, dtype=np.int64)
        self._minted = np.zeros(self.max_slots, dtype=bool)
        self.unlock_log: List[Dict] = []
        # Bound the precedence-evidence log for multi-day runs. Trimmed only
        # at EPISODE BOUNDARIES (in reset()), never mid-step, so the
        # per-step `unlock_log[_unlocks_before:]` slice invariant in the
        # parallel loop is preserved. estimated_dag keeps ample evidence at
        # this cap; memory stays a few MB instead of growing unbounded.
        self._unlock_log_cap = 8000
        self._episode = 0
        # GHOST FRONTIER (July 2026): slot -> multiplier >= 1.0, set from
        # UNRESOLVED knowledge links — concepts the agent's grounded facts
        # reference but no skill achieves. A slot whose achievements are a
        # stepping stone toward such a gap is worth more attention.
        #
        # Deliberately a BIAS ON EXISTING SLOTS, never a new slot: a slot
        # with no obs-delta signature could never be achieved, so its
        # competence would stay 0 and the (1-p) frontier score would target
        # it forever. Ghosts steer; they do not invent goals.
        self._frontier_bonus: Dict[int, float] = {}

        # ---- WORKING-SET <-> LONG-TERM PAGING (lifelong; default OFF) ----
        # The max_slots registry is the network-shape-bounded WORKING SET. With
        # a LongTermStore attached, a full registry no longer refuses new
        # discoveries: the stalest low-value slot is PAGED OUT to disk (its
        # identity + competence serialized, the slot freed) and reused, and a
        # re-encountered dormant behaviour is RECALLED back into a slot. Total
        # knowledge grows unbounded on disk while the network only ever sees
        # the bounded working set. When no store is attached EVERY path below
        # is inert and behaviour is byte-identical to the fixed-cap original.
        self._ltm = None                       # LongTermStore or None
        self._page_min_idle = 0                # episodes a slot must be idle
        self._page_recall_thresh = 0.8
        self._free_slots: set = set()          # freed working slots to reuse
        self._slot_last_used = np.zeros(self.max_slots, dtype=np.int64)
        self._slot_uid: Dict[int, str] = {}    # slot -> its LTM mem_id
        self._uid_next = 0
        self._page_out_queue: List[Dict] = []  # events the loop drains to
        self._page_in_queue: List[Dict] = []   # physically page skills/dicts

    def set_num_streams(self, n: int) -> None:
        """Resize per-stream state (called by the parallel collection path)."""
        n = max(1, int(n))
        if n == self.num_streams:
            return
        self.num_streams = n
        self._achieved_now = np.zeros((n, self.max_slots), dtype=np.float32)

    @property
    def achieved_union(self) -> np.ndarray:
        """Any-stream achieved mask (what the broadcast feature exposes)."""
        return self._achieved_now.max(axis=0)

    def clear_stream(self, stream: int) -> None:
        """Reset one stream's episode state (secondary-env autoreset).

        BANK EVIDENCE FIRST (audit fix): zeroing the row mid-window used to
        erase an in-window success, which reset() then scored as a FAILURE —
        every death/crash boundary converted real achievements into recorded
        regressions. If this stream achieved the current target, score the
        success NOW; either way mark the stream scored so reset() skips it
        (an interrupted window is not a fair failure sample)."""
        if 0 <= stream < self.num_streams:
            if (self.target is not None
                    and stream not in self._stream_scored):
                if self._achieved_now[stream, self.target] > 0:
                    self.competence.update(self.target, True)
                    self._attempts[self.target] += 1
                self._stream_scored.add(stream)
            self._achieved_now[stream, :] = 0.0

    # ---- slot management -------------------------------------------------
    def decay_masks(self, factor: float) -> None:
        """Soft-fade the per-stream achievement mask (continuous/lifelong
        path). The broadcast feature (achieved_union) fades stale unlocks so
        the goal channel reflects RECENT achievement. Competence/attempt
        bookkeeping is UNTOUCHED here — it advances only at a goal-horizon
        reset(). Within a goal-horizon window the mask starts at 0, unlocks
        set a row to 1.0, and each segment multiplies by factor (>0 for finite
        k), so the reset() Bernoulli sample `_achieved_now[s,target] > 0`
        still means "achieved at any point this window" — identical boolean
        semantics to the episodic "achieved this episode"; only the feature
        MAGNITUDE fades.
        """
        self._achieved_now *= float(factor)

    def _slot_for(self, key, name: str) -> Optional[int]:
        if key in self.slot_of:
            self._touch_slot(self.slot_of[key])
            return self.slot_of[key]
        slot = self._acquire_slot()
        if slot is None:
            return None  # registry full and nothing evictable — ignore (ok)
        self.slot_of[key] = slot
        if slot < len(self.slot_names):
            self.slot_names[slot] = name           # reused a freed slot
        else:
            self.slot_names.append(name)           # fresh slot (slot == len)
        self._free_slots.discard(slot)
        self._touch_slot(slot)
        return slot

    # ---- working-set <-> long-term paging --------------------------------
    def attach_long_term_store(self, ltm, min_idle_episodes: int = 8,
                               recall_threshold: float = 0.8) -> None:
        """Opt IN to unbounded paging. Without this the registry keeps its
        fixed-cap behaviour (a full registry refuses new kinds)."""
        self._ltm = ltm
        self._page_min_idle = int(min_idle_episodes)
        self._page_recall_thresh = float(recall_threshold)

    @property
    def paging_on(self) -> bool:
        return self._ltm is not None

    def _new_uid(self) -> str:
        uid = f"goal_{self._uid_next:08d}"
        self._uid_next += 1
        return uid

    def _effect_uid(self, effect: str) -> str:
        """Deterministic LTM id for a grounded behaviour, so recall-by-effect
        is an O(1) exact lookup and dedups across paging (base default; the
        discovered arm relies on it). A short stable hash of the RAW effect is
        appended so two distinct names that sanitize/truncate to the same
        readable stem ('foo-bar' vs 'foo_bar', or >48 shared chars) still get
        DISTINCT uids — no silent cross-effect merge on the paging path."""
        raw = str(effect)
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in raw)[:40]
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        return f"goal_effect_{safe}_{h}"

    def _effect_key_of_slot(self, slot: int):
        """A slot's ("effect", name) key if it has one (aliases mean a slot may
        carry several keys); grounded identity takes precedence."""
        for k, v in self.slot_of.items():
            if v == slot and isinstance(k, tuple) and len(k) == 2 \
                    and k[0] == "effect":
                return k
        return None

    def _uid_for_slot(self, slot: int) -> str:
        """The LTM id for a slot: a DETERMINISTIC effect uid if the slot is
        grounded (so a re-encounter recalls the same memory), else its existing
        id, else a fresh counter id for signature-keyed slots."""
        ek = self._effect_key_of_slot(slot)
        if ek is not None:
            return self._effect_uid(ek[1])
        existing = self._slot_uid.get(slot)
        if existing is not None:
            return existing
        return self._new_uid()

    def _touch_slot(self, slot: int) -> None:
        if 0 <= slot < self.max_slots:
            self._slot_last_used[slot] = self._episode
        if self._ltm is not None:
            uid = self._slot_uid.get(slot)
            if uid is not None:
                self._ltm.touch(uid, self._episode)

    def _acquire_slot(self) -> Optional[int]:
        """A working slot index to (re)use: a freed slot, a fresh index while
        under cap, or — with paging on — a slot freed by evicting the stalest
        low-value occupant. None only when full AND nothing is evictable."""
        if self._free_slots:
            return min(self._free_slots)
        if len(self.slot_names) < self.max_slots:
            return len(self.slot_names)
        if self._ltm is None:
            return None                            # fixed-cap original
        victim = self._pick_evictable()
        if victim is None:
            return None
        self._page_out_slot(victim)
        return victim

    def _pick_evictable(self) -> Optional[int]:
        """Stalest occupant, discounted by competence (a mastered goal must be
        idle much longer to be evicted). Never the active target."""
        comp = self.competence.predict_all()
        best, best_score = None, -1.0
        for slot in set(self.slot_of.values()):
            if slot == self.target:
                continue
            idle = self._episode - int(self._slot_last_used[slot])
            if idle < self._page_min_idle:
                continue
            score = idle * (1.0 - 0.6 * float(np.clip(comp[slot], 0.0, 1.0)))
            if score > best_score:
                best, best_score = slot, score
        return best

    def _key_of_slot(self, slot: int):
        for k, v in self.slot_of.items():
            if v == slot:
                return k
        return None

    @staticmethod
    def _encode_key(key):
        return {"t": list(key)} if isinstance(key, tuple) else key

    @staticmethod
    def _decode_key(k):
        return tuple(k["t"]) if isinstance(k, dict) and "t" in k else k

    def _page_out_slot(self, slot: int) -> None:
        """Consolidate a working slot to disk: serialize its identity +
        competence into the LTM (dormant), then free the slot for reuse. Emits
        a page-out event the loop drains to unbind the paired option/skill."""
        keys = [k for k, v in self.slot_of.items() if v == slot]  # incl aliases
        # canonical key is the grounded effect key if present, so the dormant
        # memory carries the effect identity and recall-by-effect finds it.
        key = self._effect_key_of_slot(slot) or (keys[0] if keys else None)
        name = self.slot_names[slot] if slot < len(self.slot_names) else ""
        uid = self._uid_for_slot(slot)
        extra = {
            "key": self._encode_key(key), "name": name,
            "comp": self.competence.export_slot(slot),
            "attempts": int(self._attempts[slot]),
            "global_unlocks": int(self._global_unlocks[slot]),
        }
        self._augment_pageout_extra(slot, extra)
        value = float(np.clip(self.competence.predict_all()[slot], 0.0, 1.0))
        self._ltm.register(uid, "goal", self._slot_cue(slot), self._episode,
                           value=value, active=False, extra=extra)
        self._ltm.set_active(uid, False)   # register() won't flip an existing
                                           # active item; do it explicitly
        # free the working slot back to the fresh prior
        self.competence.reset_slot(slot)
        self._attempts[slot] = 0
        self._global_unlocks[slot] = 0
        self._achieved_now[:, slot] = 0.0
        self._frontier_bonus.pop(slot, None)
        # REPEATABILITY-GATE state must be freed with the slot. A recycled
        # slot that inherited a dead behaviour's candidate episodes would mint
        # a NEW skill on the strength of a DIFFERENT behaviour's evidence —
        # and inheriting `_minted=True` would permanently block the incoming
        # behaviour from ever minting at all.
        self._minted[slot] = False
        self._cand_eps[slot].clear()
        self._cand_first_ep[slot] = 0
        for k in keys:                      # drop the canonical key AND aliases
            self.slot_of.pop(k, None)
        self._free_slots.add(slot)
        self._slot_uid.pop(slot, None)
        self._on_pageout_freed(slot)
        self._page_out_queue.append(
            {"mem_id": uid, "slot": slot, "name": name})

    def _page_in_item(self, mem_id: str, slot: int) -> None:
        """Recall a dormant memory into a working slot. Emits a page-in event
        the loop drains to rebind the paired option/skill."""
        it = self._ltm.items.get(mem_id)
        if it is None:
            return
        ex = it.extra or {}
        key = self._decode_key(ex.get("key"))
        name = ex.get("name", "")
        while len(self.slot_names) <= slot:
            self.slot_names.append("")
        self.slot_names[slot] = name
        if key is not None:
            self.slot_of[key] = slot
        self.competence.import_slot(slot, ex.get("comp"))
        self._attempts[slot] = int(ex.get("attempts", 0))
        self._global_unlocks[slot] = int(ex.get("global_unlocks", 0))
        self._achieved_now[:, slot] = 0.0
        self._free_slots.discard(slot)
        self._slot_uid[slot] = mem_id
        self._ltm.set_active(mem_id, True)
        self._on_pagein_restore(slot, ex)
        self._touch_slot(slot)
        self._page_in_queue.append(
            {"mem_id": mem_id, "slot": slot, "name": name})

    def _recall_into_slot(self, cue) -> Optional[int]:
        """If `cue` strongly matches a DORMANT memory, page it into a slot
        (evicting a stale occupant if the working set is full)."""
        if self._ltm is None or cue is None:
            return None
        hits = self._ltm.recall(cue, "goal", k=1,
                                threshold=self._page_recall_thresh)
        if not hits:
            return None
        slot = self._acquire_slot()
        if slot is None:
            return None
        self._page_in_item(hits[0][0], slot)
        return slot

    def drain_page_events(self) -> Tuple[List[Dict], List[Dict]]:
        """Hand the loop the page-out/page-in events since the last drain so it
        can physically unbind/rebind the paired option heads + skill dirs."""
        out, inn = self._page_out_queue, self._page_in_queue
        self._page_out_queue, self._page_in_queue = [], []
        return out, inn

    def _register_active_slot(self, slot: int) -> None:
        """Keep the LTM a COMPLETE index: every active working slot with a
        recall cue is mirrored (active=True) so the store's total reflects all
        knowledge and the viewer can list active vs dormant. Inert without a
        store or a cue (e.g. Given goals have no signature)."""
        if self._ltm is None:
            return
        cue = self._slot_cue(slot)
        if cue is None:
            return
        uid = self._uid_for_slot(slot)     # deterministic for effect-keyed
        self._slot_uid[slot] = uid
        value = float(np.clip(self.competence.predict_all()[slot], 0.0, 1.0))
        name = self.slot_names[slot] if slot < len(self.slot_names) else ""
        self._ltm.register(
            uid, "goal", cue, self._episode, value=value, active=True,
            extra={"key": self._encode_key(self._key_of_slot(slot)),
                   "name": name})
        self._ltm.set_active(uid, True)
        self._ltm.touch(uid, self._episode)

    # ---- paging hooks (base = identity; Discovered overrides for signatures)
    def _slot_cue(self, slot: int):
        return None

    def _augment_pageout_extra(self, slot: int, extra: Dict) -> None:
        pass

    def _on_pageout_freed(self, slot: int) -> None:
        pass

    def _on_pagein_restore(self, slot: int, extra: Dict) -> None:
        pass

    def _slot_name(self, slot: int) -> str:
        """Name of a slot, safe when `slot_names` has not caught up yet.

        `slot_names` is grown as slots are keyed, so a bare index can raise
        during early discovery or in a direct unit-test call. A crash in the
        unlock hot path would take down a multi-day run over a log string.
        """
        names = getattr(self, "slot_names", None) or []
        return names[slot] if 0 <= slot < len(names) else f"slot_{slot:02d}"

    def _record_unlock(self, slot: int, stream: int = 0) -> None:
        first_global = self._global_unlocks[slot] == 0
        self._global_unlocks[slot] += 1
        # Prerequisite evidence is what THIS stream had achieved before this
        # unlock — a union over concurrent, causally unrelated episodes would
        # be false evidence for estimated_dag().
        self.unlock_log.append({
            "slot": slot,
            "episode": self._episode,
            "stream": stream,
            "achieved_before": self._achieved_now[stream].copy(),
        })
        self._achieved_now[stream, slot] = 1.0
        self._touch_slot(slot)          # unlocking a slot is a USE (idle reset)
        # ---- REPEATABILITY GATE (2026-07-26) --------------------------------
        # A skill used to be minted from ONE spike (`if first_global`). Measured
        # consequence: 3 of 10 skills were minted at total_episodes=0, the bank
        # filled with behaviours the agent cannot actually perform, and 96 of 96
        # completed option invocations achieved nothing while a single useless
        # skill took 86% of the option budget.
        #
        # A one-off spike is EVIDENCE THAT SOMETHING HAPPENED, not evidence of a
        # capability. A capability is something that REPRODUCES. So an effect
        # must recur `mint_min_repeats` times across `mint_window_eps` DISTINCT
        # episodes before it is allowed to become a skill.
        #
        # DISTINCT EPISODES is the load-bearing part: N spikes inside one lucky
        # episode is the same accident counted N times, which is exactly the
        # noise this gate exists to reject. Requiring separate episodes also
        # makes the evidence robust to a single freak world layout.
        #
        # Cost is O(1) per slot and bounded: one counter and one small deque.
        self._cand_eps[slot].append(int(self._episode))
        _distinct = len(set(self._cand_eps[slot]))
        if first_global:
            self._cand_first_ep[slot] = int(self._episode)
        _mintable = (_distinct >= self.mint_min_repeats)
        if _mintable and not self._minted[slot]:
            self._minted[slot] = True
            self._mint_queue.append((slot, self._slot_name(slot)))
            logger.info(
                "MINT: slot %d (%s) reproduced in %d distinct episodes "
                "(>= %d required) — promoting candidate to skill",
                slot, self._slot_name(slot), _distinct, self.mint_min_repeats)
        elif first_global:
            logger.info(
                "CANDIDATE: slot %d (%s) seen once — needs %d distinct "
                "episodes before it becomes a skill",
                slot, self._slot_name(slot), self.mint_min_repeats)

    # ---- broadcaster contract --------------------------------------------
    def reset(self) -> None:
        # 1) close out the previous episode: teach the self-model whether the
        #    targeted slot was achieved. ONE Bernoulli sample PER STREAM: each
        #    stream is an independent attempt by the same policy, so N streams
        #    give N samples (not one inflated "any-of-N" success).
        if self.target is not None:
            # skip streams already scored at a mid-window clear_stream (their
            # success was banked there; re-reading the zeroed row here would
            # record a false failure — audit fix)
            _unscored = [s for s in range(self.num_streams)
                         if s not in self._stream_scored]
            for s in _unscored:
                success = bool(self._achieved_now[s, self.target] > 0)
                self.competence.update(self.target, success)
            self._attempts[self.target] += len(_unscored)
        self._stream_scored.clear()
        self._episode += 1
        # bound the precedence log (episode boundary only — see __init__)
        if len(self.unlock_log) > self._unlock_log_cap:
            del self.unlock_log[:-self._unlock_log_cap]
        self._achieved_now[:] = 0.0
        # 2) IMGEP frontier selection over REGISTERED slots
        self.target = self._select_target()

    def _select_target(self) -> Optional[int]:
        # every selection either produces a fresh prospection record or None —
        # a stale record must never be attributed to a later retarget
        self.last_selection = None
        n = len(self.slot_names)
        if n == 0:
            return None
        # Freed (paged-out) slots sit empty in the middle of the registry with
        # competence reset to 0; excluding them keeps the (1-p) frontier score
        # from targeting an empty slot. Gated on _free_slots so the fixed-cap
        # path (no paging / pre-first-eviction) is byte-identical.
        occ = sorted(set(self.slot_of.values())) if self._free_slots else None
        p = np.asarray(self.competence.predict_all()[:n], dtype=np.float64)
        if self._rng.rand() < self.epsilon:
            if occ is not None:
                return int(occ[self._rng.randint(len(occ))]) if occ else None
            return int(self._rng.randint(n))
        # frontier score: hard-but-known, with a try-things-you-haven't bonus
        score = (1.0 - p) + 1.0 / np.sqrt(1.0 + self._attempts[:n])
        score[p > self.competence_mastered] *= 0.15  # mastered: mostly move on
        # transferred prerequisite prior (step 6): damp goals whose known
        # prerequisites the agent cannot yet reliably deliver
        if self.dag_prior:
            for i in range(n):
                prereqs = self.dag_prior.get(self.slot_names[i])
                if not prereqs:
                    continue
                known = [self.slot_of[x] for x in prereqs if x in self.slot_of]
                if not known:
                    continue
                _ach = self.achieved_union
                met = float(np.mean([
                    (p[k] > 0.5) or (_ach[k] > 0)
                    for k in known]))
                score[i] *= (0.3 + 0.7 * met)
        # unresolved-link steering: boost slots that lead toward gaps the
        # agent knows about but cannot yet close
        if self._frontier_bonus:
            for i, mult in self._frontier_bonus.items():
                if i < n:
                    score[i] *= mult
        jitter = 1e-6 * self._rng.rand(n)
        if occ is not None:                        # never target an empty slot
            mask = np.full(n, -np.inf)
            mask[occ] = 0.0
            score = score + mask
        # ---- GOAL-LEVEL PROSPECTION (imagine-before-committing) ----
        # The loop may install a hook that IMAGINES pursuing each of the
        # frontier's top-M candidates from the CURRENT world state (world-
        # model rollouts under the candidate's counterfactual goal broadcast)
        # and returns {slot: imagined payoff}. The prospective signal is
        # blended with the retrospective frontier score OVER THE SHORTLIST
        # ONLY — prospection re-ranks the frontier's own candidates, it can
        # never promote a slot the frontier ruled out. hook=None / weight=0 /
        # hook failure / degenerate scores -> byte-identical frontier argmax.
        hook = getattr(self, "prospection_hook", None)
        weight = float(getattr(self, "prospection_weight", 0.0))
        if hook is not None and weight > 0.0:
            try:
                finite = np.where(np.isfinite(score))[0]
                top_m = int(getattr(self, "prospection_top_m", 4))
                if finite.size >= 2 and top_m >= 2:
                    cand = finite[np.argsort(score[finite])[::-1][:top_m]]
                    pros = hook([int(i) for i in cand]) or {}
                    p_vals = np.array(
                        [pros.get(int(i), np.nan) for i in cand], dtype=np.float64)
                    valid = np.isfinite(p_vals)
                    if valid.sum() >= 2:
                        f_sub = score[cand]
                        f_rng = float(f_sub.max() - f_sub.min())
                        p_rng = float(np.nanmax(p_vals) - np.nanmin(p_vals))
                        f_n = (f_sub - f_sub.min()) / (f_rng + 1e-9)
                        p_n = (p_vals - np.nanmin(p_vals)) / (p_rng + 1e-9)
                        w = float(np.clip(weight, 0.0, 1.0))
                        # candidates missing a prospective score keep their
                        # frontier rank inside the blend
                        blend = (1.0 - w) * f_n + w * np.where(valid, p_n, f_n)
                        # rescale into (at least) the shortlist's score range.
                        # span floor 1e-3 (review fix): a FLAT shortlist
                        # (f_rng ~ 0) is exactly the regime where imagination
                        # is the only discriminating signal — without the
                        # floor the blend collapsed under the 1e-6 jitter and
                        # prospection was silently nullified. Safe: every
                        # non-candidate scores <= f_sub.min(), so a wider
                        # span can never let one interleave.
                        span = max(f_rng, 1e-3)
                        score = score.copy()
                        score[cand] = f_sub.min() + blend * span
                        self.last_selection = {
                            "candidates": [int(i) for i in cand],
                            "frontier": [round(float(x), 4) for x in f_sub],
                            "prospective": [
                                None if not np.isfinite(v) else round(float(v), 4)
                                for v in p_vals],
                            "weight": w,
                            "chosen": int(np.argmax(score + jitter)),
                        }
            except Exception:  # prospection must never break selection
                logger.exception("prospection hook failed — frontier-only")
        return int(np.argmax(score + jitter))

    def update(self, env, stream: int = 0) -> None:  # pragma: no cover
        raise NotImplementedError

    def feature(self) -> np.ndarray:
        f = np.zeros(self.DIM, dtype=np.float32)
        if self.target is not None:
            f[self.target] = 1.0
            f[2 * self.max_slots] = 1.0
            f[2 * self.max_slots + 1] = float(
                self.competence.predict_all()[self.target])
        f[self.max_slots:2 * self.max_slots] = self.achieved_union
        return f

    # ---- rung-6-style lesion controls (valid-but-uninformative) ----------
    def constant_feature(self) -> np.ndarray:
        f = np.zeros(self.DIM, dtype=np.float32)
        f[0] = 1.0
        f[2 * self.max_slots] = 1.0
        return f

    def noise_feature(self, rng: np.random.RandomState) -> np.ndarray:
        f = np.zeros(self.DIM, dtype=np.float32)
        n = max(1, len(self.slot_names))
        f[int(rng.randint(n))] = 1.0
        f[2 * self.max_slots] = 1.0
        return f

    def scrambled_feature(self, rng: np.random.RandomState) -> np.ndarray:
        """Valid-but-wrong: a DIFFERENT slot than the true target."""
        f = self.feature().copy()
        n = len(self.slot_names)
        if self.target is not None and n > 1:
            wrong = [i for i in range(n) if i != self.target]
            f[:self.max_slots] = 0.0
            f[int(rng.choice(wrong))] = 1.0
        return f

    def set_frontier_bonus(self, bonus: Dict[int, float],
                           cap: float = 2.0) -> None:
        """Install per-slot attention multipliers from unresolved links."""
        self._frontier_bonus = {
            int(k): float(max(1.0, min(cap, v))) for k, v in
            (bonus or {}).items() if 0 <= int(k) < self.max_slots}

    # ---- persistence (stable skill identity, July 2026) -------------------
    # The slot registry used to be in-memory only: every process relabelled
    # clusters from 0 in whatever order novelty spikes happened, so
    # "discovered_0" could name a DIFFERENT behaviour each run — and skill
    # upserts landed on the wrong directories. Persisting the registry (and
    # the competence self-model, whose zero-init sigmoid(0)=0.5 was the tie
    # that let fresh policies clobber trained ones) makes slot->behaviour
    # identity stable across runs.

    def _extra_state(self) -> Dict:
        return {}

    def _load_extra_state(self, d: Dict) -> None:
        pass

    def save_state(self, path: str) -> None:
        """Persist slot identity + competence, atomically."""
        try:
            keys = [{"t": list(k)} if isinstance(k, tuple) else k
                    for k in self.slot_of.keys()]
            slots = [self.slot_of[k] for k in self.slot_of.keys()]
            state = {
                "max_slots": self.max_slots,
                "slot_names": self.slot_names,
                "slot_keys": keys,
                "slot_indices": slots,
                "global_unlocks": self._global_unlocks.tolist(),
                "attempts": self._attempts.tolist(),
                # REPEATABILITY-GATE state. Without this a restart resets
                # candidate evidence (a slot 2-of-3 of the way to minting
                # starts over) and — far worse — `_minted` returns to all-False,
                # so ALREADY-MINTED skills re-accumulate repeats and MINT
                # AGAIN. That is the skill-duplication bug this project has
                # already fixed once; it must not come back through the gate.
                "minted": self._minted.tolist(),
                # stamped once the one-shot bank-derived repair has run,
                # so it never runs again (see load_state)
                "minted_reconciled_v1": True,
                "cand_eps": [list(d) for d in self._cand_eps],
                "competence_weight":
                    self.competence.head.weight.detach().reshape(-1).tolist(),
                "competence_bias":
                    float(self.competence.head.bias.detach().item()),
                # precedence evidence for estimated_dag / the skill graph;
                # bounded so the file cannot grow without limit
                "unlock_log": [
                    {"slot": int(e["slot"]), "episode": int(e["episode"]),
                     "stream": int(e.get("stream", 0)),
                     "achieved_before":
                         np.asarray(e["achieved_before"]).tolist()}
                    for e in self.unlock_log[-500:]],
                "episode": self._episode,
                "frontier_bonus": {str(k): v for k, v
                                   in self._frontier_bonus.items()},
                # paging identity: slot -> its LTM mem_id, so a reloaded active
                # slot re-uses its store item instead of orphaning it under a
                # fresh uid on the next page-out.
                "slot_uid": {str(k): v for k, v in self._slot_uid.items()},
                "uid_next": int(self._uid_next),
            }
            state.update(self._extra_state())
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception as e:  # persistence must never kill training
            logger.warning("broadcaster save_state failed: %s", e)

    def load_state(self, path: str, minted_probe=None) -> bool:
        """Restore slot identity + competence. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("broadcaster state unreadable (%s) — starting "
                           "fresh registry", e)
            return False
        try:
            saved_max = int(d.get("max_slots", self.max_slots))
            n = min(saved_max, self.max_slots)
            self.slot_names = list(d.get("slot_names", []))[:n]
            self.slot_of = {}
            for k, idx in zip(d.get("slot_keys", []),
                              d.get("slot_indices", [])):
                if int(idx) >= n:
                    continue
                key = tuple(k["t"]) if isinstance(k, dict) and "t" in k else k
                self.slot_of[key] = int(idx)

            def _fit(arr, dtype):
                a = np.zeros(self.max_slots, dtype=dtype)
                src = np.asarray(arr, dtype=dtype)[:self.max_slots]
                a[:len(src)] = src
                return a
            self._global_unlocks = _fit(d.get("global_unlocks", []), np.int64)
            self._attempts = _fit(d.get("attempts", []), np.int64)
            # ---- REPEATABILITY-GATE state, with MIGRATION -----------------
            # A state file written before the gate existed has no "minted"
            # key. Defaulting it to all-False would let every EXISTING skill
            # re-accumulate repeats and mint a SECOND time — reintroducing the
            # skill duplication this project already fixed. A slot that has
            # ever unlocked globally has, by definition, already been minted
            # under the old one-spike rule, so derive it from global_unlocks.
            _m = d.get("minted")
            if _m is not None:
                self._minted = _fit(_m, np.int64).astype(bool)
                # ---- ONE-SHOT REPAIR of state written by the broken
                # migration (2026-07-27). Live pod state had minted=48/48
                # against a 10-skill bank, permanently disabling minting.
                # Re-derive ONCE from the bank, then stamp a flag so this
                # never runs again — an unconditional every-load rewrite of
                # a private latch would itself be a standing override that
                # silently undoes any future deliberate latch.
                if not d.get("minted_reconciled_v1") and callable(minted_probe):
                    _before = int(self._minted.sum())
                    self._minted = np.array(
                        [bool(minted_probe(i, self.slot_names[i]
                                           if i < len(self.slot_names) else ""))
                         for i in range(self.max_slots)], dtype=bool)
                    self._minted_reconciled = True
                    logger.warning(
                        "repeatability gate: ONE-SHOT REPAIR — minted %d -> "
                        "%d slots, re-derived from the skill bank (the old "
                        "global_unlocks migration over-counted). Fossil slots "
                        "stay un-mintable via the mint-site guard.",
                        _before, int(self._minted.sum()))
            else:
                # ---- MIGRATION, CORRECTED (2026-07-27) -------------------
                # The old line was `self._minted = self._global_unlocks > 0`
                # — "ever unlocked" equated with "ever minted". FALSE: minting
                # also required the loop's mint block to run AND succeed.
                # Measured consequence on the live pod: minted=48/48 while the
                # bank held 10 skills, so the repeatability gate was reached
                # and PASSED four times in the stalled run and blocked every
                # time by the latch. Zero skills could ever be minted again —
                # arch v3's conv path never ran in production because of it.
                #
                # Derive from what the BANK ACTUALLY CONTAINS when a probe is
                # supplied; fall back to the old rule only when it is not, so
                # callers that cannot answer keep the conservative behaviour
                # (never re-mint an existing skill).
                if callable(minted_probe):
                    self._minted = np.array(
                        [bool(minted_probe(i, self.slot_names[i]
                                           if i < len(self.slot_names) else ""))
                         for i in range(self.max_slots)], dtype=bool)
                    logger.info(
                        "repeatability gate: no 'minted' in state — derived "
                        "from the skill bank (%d/%d slots already minted)",
                        int(self._minted.sum()), self.max_slots)
                else:
                    self._minted = self._global_unlocks > 0
                    logger.warning(
                        "repeatability gate: no 'minted' in state and no bank "
                        "probe — falling back to global_unlocks (%d slots "
                        "treated as minted). This OVER-COUNTS: 'unlocked' is "
                        "not 'minted'.", int(self._minted.sum()))
            _ce = d.get("cand_eps")
            if isinstance(_ce, list):
                for i, eps in enumerate(_ce[:self.max_slots]):
                    self._cand_eps[i] = deque(
                        [int(e) for e in (eps or [])],
                        maxlen=self.mint_window_eps)
            w = d.get("competence_weight")
            if w is not None and len(w) > 0:
                # WIDENING-SAFE: load the saved per-slot logits into the first
                # min(saved, current) slots instead of demanding an exact width
                # match. Raising goals.max_slots (24/48 working-set widening)
                # then keeps existing slots' learned competence; new slots start
                # at the shared-bias prior. Same-width reloads are unchanged.
                with torch.no_grad():
                    k = min(len(w), self.max_slots)
                    self.competence.head.weight[0, :k] = torch.tensor(
                        w[:k], dtype=torch.float32)
                    # BIAS IS RESTORED AT 0, NEVER FROM THE FILE (2026-07-25
                    # LATCH FIX). self_model froze the shared bias to stop a
                    # failure-dominated stream dragging every task's competence
                    # down together — but a legacy state file carries a bias
                    # ALREADY poisoned by exactly that drift, and restoring it
                    # into a frozen parameter makes the poison PERMANENT and
                    # re-applies it on every restart. Measured live: stored
                    # bias = -5.68 => competence 0.0034 for EVERY task, against
                    # competence_floor 0.25, so a task needed w[task] >= +4.58
                    # to be offered at all. Result: all 16 skills sat at
                    # invocations=0 across 139k steps and 2227 option decisions
                    # (100% of them the scripted bootstrap, which bypasses this
                    # gate). Per-task logits live ENTIRELY in head.weight; the
                    # frozen bias must be 0 or the gate is a one-way latch.
                    self.competence.head.bias.fill_(0.0)
            self.unlock_log = [
                {"slot": int(e["slot"]), "episode": int(e["episode"]),
                 "stream": int(e.get("stream", 0)),
                 "achieved_before": np.asarray(
                     e["achieved_before"], dtype=np.float32)}
                for e in d.get("unlock_log", [])]
            self._episode = int(d.get("episode", 0))
            self.set_frontier_bonus(
                {int(k): float(v) for k, v
                 in (d.get("frontier_bonus", {}) or {}).items()})
            self._load_extra_state(d)
            # Reconstruct the free-slot set: any index below the registry high
            # water mark that no key maps to was paged out and is reusable.
            occupied = set(self.slot_of.values())
            self._free_slots = {s for s in range(len(self.slot_names))
                                if s not in occupied}
            self._slot_last_used[:] = self._episode
            self._slot_uid = {int(k): v for k, v
                              in (d.get("slot_uid", {}) or {}).items()
                              if int(k) in occupied}
            self._uid_next = int(d.get("uid_next", 0))
            logger.info("Broadcaster identity restored: %d slots (%s), "
                        "%d unlock events",
                        len(self.slot_names),
                        ", ".join(self.slot_names[:6])
                        + ("..." if len(self.slot_names) > 6 else ""),
                        len(self.unlock_log))
            return True
        except Exception as e:
            logger.warning("broadcaster load_state failed (%s) — starting "
                           "fresh registry", e)
            return False

    # ---- hooks for skills / rulebook / transfer --------------------------
    def pop_skill_mint_events(self) -> List[Tuple[int, str]]:
        ev, self._mint_queue = self._mint_queue, []
        return ev

    def estimated_dag(self) -> Dict[int, np.ndarray]:
        """slot -> mean achieved-before mask over its unlock events (soft
        prerequisite evidence; feeds rung-10 law induction + Step-6 transfer)."""
        acc: Dict[int, List[np.ndarray]] = defaultdict(list)
        for e in self.unlock_log:
            acc[e["slot"]].append(
                np.asarray(e["achieved_before"], dtype=np.float32).reshape(-1))
        # MIXED-WIDTH GUARD (audit fix): persisted unlock_log entries predate
        # the 48-slot widening (16-wide rows observed live next to 48-wide),
        # and np.mean over a ragged list crashes — silently wiping the DAG.
        # Zero-pad to the widest row (absent slots were unachieved: 0 is
        # exactly right).
        out: Dict[int, np.ndarray] = {}
        for s, v in acc.items():
            w = max(x.size for x in v)
            out[s] = np.mean(
                [np.pad(x, (0, w - x.size)) if x.size < w else x
                 for x in v], axis=0)
        return out

    @property
    def stats(self) -> Dict:
        n = len(self.slot_names)
        return {
            "n_slots": n,
            "total_unlocks": int(self._global_unlocks.sum()),
            "distinct_unlocked": int((self._global_unlocks > 0).sum()),
            "target": self.target,
            "competence": [round(float(x), 3)
                           for x in self.competence.predict_all()[:n]],
        }


class GivenAchievementGoals(AchievementGoalBroadcast):
    """The PRIVILEGED arm: reads env.last_info['achievements'] (name->count).
    Slots are the oracle achievement names."""

    def __init__(self, **kw):
        super().__init__(**kw)
        # PER-STREAM baselines: counts are diffed against the same env's last
        # reading, so one shared dict across concurrent envs would act as a
        # fleet max — dropping real unlocks and mis-baselining after resets.
        self._counts: Dict[int, Dict[str, int]] = {}

    def reset(self) -> None:
        super().reset()
        self._counts = {}

    def clear_stream(self, stream: int) -> None:
        super().clear_stream(stream)
        self._counts.pop(stream, None)

    def update(self, env, stream: int = 0) -> None:
        info = getattr(env, "last_info", None) or {}
        ach = info.get("achievements")
        if not isinstance(ach, dict):
            return
        counts = self._counts.setdefault(stream, {})
        for name, count in ach.items():
            c = int(count)
            prev = counts.get(name)
            if prev is None:
                # first sight this episode: establish the baseline WITHOUT
                # counting pre-existing unlocks as new events
                counts[name] = c
                if c > 0:
                    slot = self._slot_for(name, name)
                    if slot is not None:
                        self._achieved_now[stream, slot] = 1.0
                continue
            if c > prev:
                counts[name] = c
                slot = self._slot_for(name, name)
                if slot is not None:
                    self._record_unlock(slot, stream)


class DiscoveredAchievementGoals(AchievementGoalBroadcast):
    """The EARNED arm: reward spikes + pooled obs-delta signatures ONLY.
    Never reads info['achievements'] — grading is external."""

    def __init__(self, spike_threshold: float = 0.9, pool: int = 192,
                 match_cosine: float = 0.8, **kw):
        super().__init__(**kw)
        self.spike_threshold = float(spike_threshold)
        self.pool = int(pool)
        self.match_cosine = float(match_cosine)
        # SLOT-ALIGNED: _signatures[slot] is the ACTIVE working slot's cluster
        # signature (None for a freed/empty slot). Paged-out signatures live in
        # the LongTermStore as recall cues, not here. In the fixed-cap path
        # (paging off) there are never Nones and this is exactly the old list.
        self._signatures: List = []
        self._disc_next = 0             # monotonic id for new discovered slots

    def _extra_state(self) -> Dict:
        return {"signatures": [None if s is None else np.asarray(s).tolist()
                               for s in self._signatures],
                "disc_next": int(self._disc_next)}

    def _load_extra_state(self, d: Dict) -> None:
        # Restoring signatures is what makes slot->behaviour identity real:
        # a re-observed break event matches its ORIGINAL cluster instead of
        # minting discovered_0 again under a new process.
        self._signatures = [
            None if sig is None else np.asarray(sig, dtype=np.float32)
            for sig in d.get("signatures", [])][:self.max_slots]
        self._disc_next = int(d.get("disc_next", len(self._signatures)))

    def _set_slot_signature(self, slot: int, sig) -> None:
        while len(self._signatures) <= slot:
            self._signatures.append(None)
        self._signatures[slot] = sig

    # ---- paging hooks: the signature IS the goal's recall cue -------------
    def _slot_cue(self, slot: int):
        if 0 <= slot < len(self._signatures):
            return self._signatures[slot]
        return None

    def _augment_pageout_extra(self, slot: int, extra: Dict) -> None:
        s = self._signatures[slot] if slot < len(self._signatures) else None
        if s is not None:
            extra["sig"] = np.asarray(s, dtype=np.float32).tolist()

    def _on_pageout_freed(self, slot: int) -> None:
        if slot < len(self._signatures):
            self._signatures[slot] = None

    def _on_pagein_restore(self, slot: int, extra: Dict) -> None:
        sig = extra.get("sig")
        if sig is not None:
            self._set_slot_signature(slot, np.asarray(sig, dtype=np.float32))

    def _pooled_delta(self, prev_obs: np.ndarray, obs: np.ndarray) -> np.ndarray:
        d = np.abs(np.asarray(obs, np.float32) - np.asarray(prev_obs, np.float32))
        if d.size <= self.pool:
            return d
        # block-mean pool the flat delta to a fixed small size
        k = d.size // self.pool
        return d[: k * self.pool].reshape(self.pool, k).mean(axis=1)

    def update(self, env, stream: int = 0, effect_key=None) -> None:
        rh = getattr(env, "reward_history", None)
        oh = getattr(env, "obs_history", None)
        if not rh or oh is None or len(oh) < 2:
            return
        if rh[-1] < self.spike_threshold:
            return
        sig = self._pooled_delta(oh[-2], oh[-1])
        norm = np.linalg.norm(sig)
        if norm < 1e-8:
            return
        sig = sig / norm
        # GROUNDED-EFFECT dedup (July 2026): when the loop tells us WHAT broke
        # (e.g. "birch_leaves"), that block type is the STABLE behaviour
        # identity — key the slot by it so the same grounded behaviour can
        # never mint two slots regardless of how much the raw pixel-delta
        # signature varies across viewpoint/lighting. The noisy-signature
        # clustering below is only the FALLBACK for ungrounded spikes.
        if effect_key:
            slot = self._slot_for_effect(str(effect_key), sig)
        else:
            slot = self._slot_for_signature(sig)
        if slot is None:
            return
        # once per episode per slot PER STREAM (each stream is its own episode)
        if self._achieved_now[stream, slot] == 0:
            self._record_unlock(slot, stream)
        # mirror the active slot into the LTM (complete active+dormant index)
        self._register_active_slot(slot)

    def _match_active_signature(self, sig):
        """Nearest ACTIVE working-slot by signature cosine (>= match_cosine),
        or None. Slot-aligned list; None entries are freed slots."""
        best, best_cos = None, self.match_cosine
        for i, s in enumerate(self._signatures):
            if s is None:
                continue
            c = float(sig @ s)
            if c > best_cos:
                best, best_cos = i, c
        return best

    def _blend_sig(self, slot: int, sig) -> None:
        """EMA-refine a slot's cluster signature toward a fresh observation."""
        if slot < len(self._signatures) and self._signatures[slot] is not None:
            s = 0.9 * self._signatures[slot] + 0.1 * sig
            self._signatures[slot] = s / (np.linalg.norm(s) + 1e-8)
        else:
            self._set_slot_signature(slot, sig)

    def _slot_for_effect(self, effect: str, sig) -> Optional[int]:
        """Resolve a grounded behaviour to ONE canonical slot: existing
        effect-slot -> dormant effect-slot (recall) -> an existing slot whose
        signature already matches (adopt/alias, never duplicate) -> mint."""
        key = ("effect", effect)
        if key in self.slot_of:                     # already the canonical slot
            slot = self.slot_of[key]
            self._blend_sig(slot, sig)
            self._touch_slot(slot)
            return slot
        slot = self._recall_effect(effect)          # dormant effect-slot (O(1))
        if slot is not None:
            self._blend_sig(slot, sig)
            return slot
        # a slot for this behaviour may already exist under a signature key
        # (ungrounded discovery, or pre-fix data) — ADOPT it (alias the effect
        # key on) so grounded + ungrounded views of the same behaviour unify
        # instead of duplicating. But ONLY adopt a slot that is NOT already
        # grounded to a DIFFERENT effect: two distinct blocks whose pixel-delta
        # signatures happen to be similar (leaf/log/ore variants at
        # match_cosine) must NOT be merged into one behaviour (silent loss of
        # the second effect). Grounded identity beats pixel similarity.
        match = self._match_active_signature(sig)
        if match is not None:
            ek = self._effect_key_of_slot(match)
            if ek is None or ek[1] == effect:      # ungrounded or same effect
                self.slot_of[key] = match
                # CRITICAL: re-key the slot's LTM identity to the deterministic
                # effect uid, so page-out stores it under the effect uid and
                # recall-by-effect finds it (else it re-mints a duplicate).
                self._rekey_slot_to_effect(match, effect)
                self._blend_sig(match, sig)
                self._touch_slot(match)
                return match
        # NAMESPACE-AWARE NAMING (2026-07-27 review, HIGH). The `break_`
        # prefix used to be unconditional, so a CRAFT effect (`craft_planks`,
        # which already carries its own namespace) became the slot
        # `break_craft_planks` -> skill `ach_NN_break_craft_planks`. The
        # mastery ledger derives a skill's own effect key from its id and
        # compares it against the key the env emits (`craft_planks`), so a
        # craft skill could NEVER be scored a success — the metric would read
        # 0/N forever for exactly the behaviour the crafting phase exists to
        # measure. Effects that already name their own namespace keep it.
        _name = (effect if effect.startswith(("break_", "craft_", "mine_"))
                 or effect == "log_pickup" else f"break_{effect}")
        slot = self._slot_for(key, _name)
        if slot is None:
            return None
        self._set_slot_signature(slot, sig)
        return slot

    def _rekey_slot_to_effect(self, slot: int, effect: str) -> None:
        """Bind a slot's canonical LTM identity to the grounded effect uid and
        refresh the store entry under it, so page-out/recall dedup by effect."""
        uid = self._effect_uid(effect)
        old = self._slot_uid.get(slot)
        self._slot_uid[slot] = uid
        if self._ltm is not None and old is not None and old != uid:
            self._ltm.forget(old)          # drop the stale signature-uid entry
        self._register_active_slot(slot)   # (re)register under the effect uid

    def _slot_for_signature(self, sig) -> Optional[int]:
        """Ungrounded fallback: cluster by noisy pixel-delta signature.

        KNOWN LIMITATION (documented): if a behaviour is discovered UNGROUNDED,
        paged out, then later seen GROUNDED at a DIFFERENT viewpoint (a fresh
        effect slot is minted), an ungrounded re-cue at the ORIGINAL viewpoint
        can resurrect the dormant twin as a 2nd active slot for that one
        behaviour. It is unlinkable by pixel signature (the two views share no
        key) and is the inherent imperfection of signature-only discovery; it
        mints NO extra skill (global_unlocks restore on recall) and is deduped
        in the brain VIEW by grounded label. Rare on grounded envs like
        Treechop (a block-break spikes reward + the mine_* achievement in the
        SAME step, so the first sighting is already grounded)."""
        best = self._match_active_signature(sig)
        if best is not None:
            self._blend_sig(best, sig)
            self._touch_slot(best)
            return best
        slot = self._recall_into_slot(sig)          # dormant cue match
        if slot is not None:
            return slot
        slot = self._slot_for(("disc", self._disc_next),
                              f"discovered_{self._disc_next}")
        if slot is None:
            return None
        self._disc_next += 1
        self._set_slot_signature(slot, sig)
        return slot

    def _recall_effect(self, effect: str) -> Optional[int]:
        if self._ltm is None:
            return None
        uid = self._effect_uid(effect)
        it = self._ltm.items.get(uid)
        if it is None or it.active:
            return None
        slot = self._acquire_slot()
        if slot is None:
            return None
        self._page_in_item(uid, slot)
        return slot


def make_goal_broadcast(goal_cfg: Dict) -> AchievementGoalBroadcast:
    mode = str(goal_cfg.get("mode", "given")).lower()
    kw = dict(
        max_slots=int(goal_cfg.get("max_slots", 32)),
        epsilon=float(goal_cfg.get("epsilon", 0.1)),
        seed=int(goal_cfg.get("seed", 0)),
        dag_prior=goal_cfg.get("dag_prior"),
        # REPEATABILITY GATE: an effect must reproduce in this many DISTINCT
        # episodes (within a rolling window) before it can become a skill.
        # 1 restores the old one-spike-one-skill behaviour.
        mint_min_repeats=int(goal_cfg.get("mint_min_repeats", 3)),
        mint_window_eps=int(goal_cfg.get("mint_window_eps", 12)),
    )
    if mode == "discovered":
        return DiscoveredAchievementGoals(
            spike_threshold=float(goal_cfg.get("spike_threshold", 0.9)),
            match_cosine=float(goal_cfg.get("match_cosine", 0.8)),
            **kw)
    return GivenAchievementGoals(**kw)
