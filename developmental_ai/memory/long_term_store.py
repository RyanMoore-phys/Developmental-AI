"""Long-term memory store — the SSD tier of a human-like memory hierarchy.

WHY THIS EXISTS (July 2026, the lifelong pivot)
  Working memory (RAM + the network's fixed slots) is BOUNDED; long-term
  memory — this store, on disk — is UNBOUNDED. Items unused for a while are
  PAGED OUT here (consolidation: short-term -> long-term); when the current
  perception matches a dormant item's cue, it is RECALLED back into working
  memory. Like a person: you think about a few dozen things at once, but you
  KNOW an unbounded amount, and a cue pulls the right memory back into focus.

  This ONE mechanism resolves two problems the audit flagged as separate:
    * OOM over a multi-day run — the in-RAM working set stays bounded.
    * The max_slots / K network-shape cap — the network only ever sees the
      bounded working set, so its input/output width is fixed while total
      knowledge grows without limit on disk.

  Kind-agnostic: skills, goals, and facts share one index. The store tracks
  metadata + does the ranking/matching; the CALLER decides what a page-out or
  page-in physically means (unbind an option slot, drop a KG fact, archive a
  goal signature). The store never owns the heavy payloads (skill policies
  already live in their own dirs) — it owns the recall index.

CONTRACT
  * register/upsert an item's metadata (cue vector + bookkeeping)
  * touch(id) on every use -> recency + frequency
  * evict_candidates(...) -> stalest, lowest-value ACTIVE items to page out
  * recall(cue, ...) -> best-matching DORMANT items to page in
  * the index persists atomically and survives restarts (lifelong runs are
    killed and resumed; the store is the memory that outlives a process)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """One long-term memory (a skill, goal, or fact) and its recall metadata.

    `cue` is the embedding recall matches against — for a skill its
    context/precondition vector, for a fact an entity embedding, for a goal
    its obs-delta signature. `extra` holds kind-specific pointers (skill_id,
    fact triple, goal slot signature) the caller needs to physically page.
    """
    mem_id: str
    kind: str                      # "skill" | "goal" | "fact"
    cue: List[float]
    value: float = 0.5             # competence / confidence / importance
    usage_count: int = 0
    created_step: int = 0
    last_used_step: int = 0
    active: bool = True            # in working memory (RAM/network) vs dormant
    extra: Dict = field(default_factory=dict)


class LongTermStore:
    """Disk-backed, unbounded recall index for a memory hierarchy."""

    def __init__(self, storage_dir: str, cue_dim: int,
                 recall_scan_cap: int = 20000):
        self.storage_dir = storage_dir
        self.cue_dim = int(cue_dim)
        # brute-force cosine recall is fine into the tens of thousands; beyond
        # that, sample the scan (logged) until an ANN index is added.
        self.recall_scan_cap = int(recall_scan_cap)
        os.makedirs(storage_dir, exist_ok=True)
        self.index_path = os.path.join(storage_dir, "ltm_index.json")
        self.items: Dict[str, MemoryItem] = {}
        self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self) -> None:
        for cand in (self.index_path, self.index_path + ".bak"):
            if not os.path.exists(cand):
                continue
            try:
                with open(cand) as f:
                    data = json.load(f)
                for d in data.get("items", []):
                    it = MemoryItem(**d)
                    self.items[it.mem_id] = it
                logger.info("LongTermStore loaded %d items", len(self.items))
                return
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.error("LTM index %s unreadable (%s) — trying fallback",
                             cand, e)

    def save(self) -> None:
        """Atomic index write (tmp -> .bak -> replace); the store must
        survive a kill mid-write on a multi-day run."""
        data = {"items": [asdict(it) for it in self.items.values()],
                "saved_at": time.time()}
        tmp = self.index_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(self.index_path):
                try:
                    os.replace(self.index_path, self.index_path + ".bak")
                except OSError:
                    pass
            os.replace(tmp, self.index_path)
        except OSError as e:
            logger.warning("LTM index save failed: %s", e)

    # ---- registration / use ------------------------------------------------
    def _fit_cue(self, cue) -> List[float]:
        v = np.zeros(self.cue_dim, dtype=np.float32)
        if cue is not None:
            a = np.asarray(cue, dtype=np.float32).reshape(-1)[:self.cue_dim]
            v[:len(a)] = a
        n = float(np.linalg.norm(v))
        if n > 1e-8:
            v /= n                 # store unit cues -> cosine == dot
        return v.tolist()

    def register(self, mem_id: str, kind: str, cue, step: int,
                 value: float = 0.5, active: bool = True,
                 extra: Optional[Dict] = None) -> MemoryItem:
        """Add or refresh an item. Existing items keep their usage history
        and are only re-cued/re-valued (a skill re-minted with a better
        policy updates its cue but not its recall record)."""
        ex = self.items.get(mem_id)
        if ex is None:
            it = MemoryItem(
                mem_id=mem_id, kind=kind, cue=self._fit_cue(cue),
                value=float(value), created_step=int(step),
                last_used_step=int(step), active=bool(active),
                extra=dict(extra or {}))
            self.items[mem_id] = it
            return it
        ex.cue = self._fit_cue(cue)
        ex.value = float(value)
        if extra:
            ex.extra.update(extra)
        return ex

    def touch(self, mem_id: str, step: int) -> None:
        it = self.items.get(mem_id)
        if it is not None:
            it.last_used_step = int(step)
            it.usage_count += 1

    def set_active(self, mem_id: str, active: bool) -> None:
        it = self.items.get(mem_id)
        if it is not None:
            it.active = bool(active)

    def forget(self, mem_id: str) -> None:
        self.items.pop(mem_id, None)

    # ---- views -------------------------------------------------------------
    def active_ids(self, kind: Optional[str] = None) -> List[str]:
        return [i.mem_id for i in self.items.values()
                if i.active and (kind is None or i.kind == kind)]

    def dormant_ids(self, kind: Optional[str] = None) -> List[str]:
        return [i.mem_id for i in self.items.values()
                if not i.active and (kind is None or i.kind == kind)]

    # ---- consolidation: RAM -> disk ---------------------------------------
    def evict_candidates(self, now_step: int, kind: str,
                         min_idle_steps: int, k: int = 1,
                         protect: Optional[set] = None) -> List[str]:
        """Stalest, least-valuable ACTIVE items of `kind` that have been idle
        at least `min_idle_steps` — candidates to page OUT to disk.

        Score = idle_time discounted by value (a high-value skill must be
        idle much longer before it is a candidate), so competent-but-quiet
        skills persist in working memory longer than weak ones. `protect`
        ids (e.g. the currently-executing option) are never evicted.
        """
        protect = protect or set()
        scored: List[Tuple[float, str]] = []
        for it in self.items.values():
            if not it.active or it.kind != kind or it.mem_id in protect:
                continue
            idle = now_step - it.last_used_step
            if idle < min_idle_steps:
                continue
            # higher score = more evictable: idle time, damped by value
            score = idle * (1.0 - 0.6 * float(np.clip(it.value, 0.0, 1.0)))
            scored.append((score, it.mem_id))
        scored.sort(reverse=True)
        return [mid for _s, mid in scored[:k]]

    # ---- recall: disk -> RAM ----------------------------------------------
    def recall(self, cue, kind: str, k: int = 1,
               threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Best-matching DORMANT items of `kind` for the given cue — the
        candidates to page IN. Returns (mem_id, similarity), most-similar
        first, above `threshold`. Cosine over unit cues == dot product.
        """
        q = np.asarray(self._fit_cue(cue), dtype=np.float32)
        if float(np.linalg.norm(q)) < 1e-8:
            return []
        dormant = [it for it in self.items.values()
                   if not it.active and it.kind == kind]
        if not dormant:
            return []
        if len(dormant) > self.recall_scan_cap:
            # bounded scan (deterministic: most-recently-used slice) until an
            # ANN index lands; logged so the cap is never silent.
            dormant.sort(key=lambda it: it.last_used_step, reverse=True)
            logger.info("LTM recall scan capped at %d/%d dormant %s items",
                        self.recall_scan_cap, len(dormant), kind)
            dormant = dormant[:self.recall_scan_cap]
        mat = np.asarray([it.cue for it in dormant], dtype=np.float32)
        sims = mat @ q
        order = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for idx in order[:k]:
            s = float(sims[idx])
            if s < threshold:
                break
            out.append((dormant[idx].mem_id, s))
        return out

    # ---- introspection -----------------------------------------------------
    def stats(self) -> Dict:
        active = sum(1 for i in self.items.values() if i.active)
        by_kind: Dict[str, int] = {}
        for i in self.items.values():
            by_kind[i.kind] = by_kind.get(i.kind, 0) + 1
        return {"total": len(self.items), "active": active,
                "dormant": len(self.items) - active, "by_kind": by_kind}
