"""Brain-state emitter — the live picture of what the agent knows.

Writes brain_state.json: the skill graph (nodes + EARNED edges only) plus a
live firing feed (which skill is executing in which env right now). A
self-contained HTML viewer (viewer/brain_viewer.html) polls the file and
renders it — the user's "Obsidian brain": watch skills light up as the
agent thinks with them.

EDGE DISCIPLINE (user decision, July 2026): every edge must be earned.
  precedence   slot B's unlock events found slot A already achieved
               (broadcaster.estimated_dag(), observed evidence)
  achieves     skill -> concept, from ground-truth effects recorded at mint
  causal       concept-level facts minted from real events
               (("attack_held","breaks","oak_log") @ confidence 1.0)
  prerequisite skill -> skill, from the achieved-before mask of the actual
               unlock event
NO embedding-similarity links. A sparse true picture beats a dense one.

UNRESOLVED LINKS (ghost nodes): a concept the knowledge mentions but no
skill achieves — visible gaps, the Obsidian ghost-note. These are goals.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BrainStateEmitter:
    """Assembles brain_state.json from the live organs. Cheap by design:
    called every `interval` env steps and at episode boundaries; all inputs
    are in-memory reads except note frontmatter (refreshed lazily)."""

    def __init__(self, skill_bank, broadcaster=None, knowledge_graph=None,
                 symbolizer=None, out_path: str = "runlogs/brain/brain_state.json",
                 precedence_threshold: float = 0.34, long_term_store=None):
        self.skill_bank = skill_bank
        self.broadcaster = broadcaster
        self.knowledge_graph = knowledge_graph
        self.symbolizer = symbolizer
        # Long-term memory index (lifelong paging). When present, the emitter
        # shows the memory HIERARCHY: goals currently in the bounded working
        # set vs those consolidated to disk (dormant), so the viewer renders
        # the whole memory system, not just what is resident right now.
        self.long_term_store = long_term_store
        self.out_path = out_path
        self.precedence_threshold = float(precedence_threshold)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # firing feed: Layer-4 options push events here
        #   {"env": int, "skill_id": str, "t_start": int, "t_end": int|None,
        #    "outcome": "running"|"achieved"|"horizon"|"episode_end"}
        self.firing_events: deque = deque(maxlen=300)
        self.active_options: Dict[int, Optional[str]] = {}

        self._notes_cache: Dict[str, Dict] = {}
        self._notes_cache_t = 0.0

    # ---- firing feed (called by the options machinery) ---------------------
    def option_started(self, env: int, skill_id: str, timestep: int) -> None:
        self.active_options[int(env)] = skill_id
        self.firing_events.append({
            "env": int(env), "skill_id": skill_id,
            "t_start": int(timestep), "t_end": None, "outcome": "running"})

    def option_ended(self, env: int, skill_id: str, timestep: int,
                     outcome: str) -> None:
        self.active_options[int(env)] = None
        for ev in reversed(self.firing_events):
            if (ev["env"] == int(env) and ev["skill_id"] == skill_id
                    and ev["t_end"] is None):
                ev["t_end"] = int(timestep)
                ev["outcome"] = str(outcome)
                break

    # ---- assembly ----------------------------------------------------------
    def _note_fronts(self) -> Dict[str, Dict]:
        """Frontmatter per skill dir, cached for 10s (files change rarely)."""
        now = time.time()
        if now - self._notes_cache_t < 10.0:
            return self._notes_cache
        from developmental_ai.skill_bank.skill_notes import (
            read_note_frontmatter)
        fronts = {}
        for sid in self.skill_bank.skills:
            d = os.path.join(self.skill_bank.storage_dir, sid)
            f = read_note_frontmatter(d)
            if f:
                fronts[sid] = f
        self._notes_cache, self._notes_cache_t = fronts, now
        return fronts

    def _note_markdown(self, sid: str) -> Optional[str]:
        path = os.path.join(self.skill_bank.storage_dir, sid, "note.md")
        try:
            with open(path, "r") as f:
                return f.read()[:20000]
        except OSError:
            return None

    def emit(self, episode: int = 0, timestep: int = 0) -> Optional[str]:
        try:
            return self._emit(episode, timestep)
        except Exception as e:  # telemetry must never kill training
            logger.warning("brain_state emit failed: %s", e)
            return None

    def _emit(self, episode: int, timestep: int) -> str:
        fronts = self._note_fronts()
        nodes: List[Dict] = []
        edges: List[Dict] = []
        concepts: Dict[str, Dict] = {}

        # slot index -> skill_id (for precedence edges)
        slot_to_sid: Dict[int, str] = {}
        if self.broadcaster is not None and hasattr(
                self.broadcaster, "slot_names"):
            for i, name in enumerate(self.broadcaster.slot_names):
                slot_to_sid[i] = f"ach_{i:02d}_{name}"[:64]

        # ---- skill nodes (DEDUPED by grounded label) ----
        # The same grounded behaviour historically minted several slots (the
        # break_birch_leaves x8 duplication). Collapse skills that share a
        # grounded label into ONE representative node (most competent, then
        # most practised) so the brain shows the behaviour once, with an
        # `instances` count. `canon` remaps every merged id -> its
        # representative so edges dedup too. Nothing on disk is touched.
        by_label: Dict[str, List] = {}
        for sid, sk in self.skill_bank.skills.items():
            by_label.setdefault(sk.name, []).append((sid, sk))
        canon: Dict[str, str] = {}
        reps: List = []
        for label, members in by_label.items():
            members.sort(key=lambda t: (float(t[1].success_rate),
                                        int(t[1].total_episodes), t[0]),
                         reverse=True)
            rep_sid, rep_sk = members[0]
            reps.append((rep_sid, rep_sk, len(members)))
            for sid, _sk in members:
                canon[sid] = rep_sid

        for sid, sk, n_inst in reps:
            front = fronts.get(sid, {})
            node = {
                "id": sid, "kind": "skill",
                "label": sk.name,
                "competence": round(float(sk.success_rate), 3),
                "milestone": bool(sk.is_mastered),
                "episodes": int(sk.total_episodes),
                "preconditions": sk.preconditions or {},
                "memory_tier": "working",       # resident in the network
                "note_md": self._note_markdown(sid),
            }
            if n_inst > 1:
                node["instances"] = n_inst      # merged duplicate slots
            nodes.append(node)
            # achieves edges from note-recorded ground-truth effects
            for eff in front.get("effects", []):
                concepts.setdefault(eff, {"achieved_by": 0})
                concepts[eff]["achieved_by"] += 1
                edges.append({"source": sid, "target": f"concept:{eff}",
                              "kind": "achieves"})
            for pre in front.get("prerequisites", []):
                if pre in self.skill_bank.skills:
                    edges.append({"source": canon.get(pre, pre), "target": sid,
                                  "kind": "prerequisite"})

        # ---- precedence edges (estimated_dag: observed evidence) ----
        if (self.broadcaster is not None
                and hasattr(self.broadcaster, "estimated_dag")):
            try:
                dag = self.broadcaster.estimated_dag()
                for slot, mask in dag.items():
                    tgt = slot_to_sid.get(int(slot))
                    if tgt is None or tgt not in self.skill_bank.skills:
                        continue
                    mask = np.asarray(mask)
                    for pre_slot in np.nonzero(
                            mask > self.precedence_threshold)[0]:
                        src = slot_to_sid.get(int(pre_slot))
                        if (src and src != tgt
                                and src in self.skill_bank.skills):
                            edges.append({
                                "source": src, "target": tgt,
                                "kind": "precedence",
                                "weight": round(float(mask[pre_slot]), 2)})
            except Exception:
                pass

        # ---- causal facts from the knowledge graph (ground truth only) ----
        # Every grounded fact's subject AND object become concept mentions;
        # subjects that only ever CAUSE (attack_held) are actors, not gaps.
        kg_facts = []
        actors = set()
        if self.knowledge_graph is not None:
            try:
                for f in self.knowledge_graph.query():
                    if f.source == "grounded_event":
                        kg_facts.append([f.subject, f.relation, f.obj])
                        concepts.setdefault(f.subject, {"achieved_by": 0})
                        concepts.setdefault(f.obj, {"achieved_by": 0})
                        actors.add(f.subject)
                        edges.append({
                            "source": f"concept:{f.subject}",
                            "target": f"concept:{f.obj}",
                            "kind": "causal", "label": f.relation})
            except Exception:
                pass

        # ---- concept + ghost nodes ----
        # ghost = the knowledge REFERENCES it, but no skill achieves it and
        # nothing shows the agent can cause it: an unresolved [[link]] — a
        # visible gap, which is to say a GOAL.
        for cname, meta in concepts.items():
            unresolved = (meta.get("achieved_by", 0) == 0
                          and cname not in actors)
            nodes.append({
                "id": f"concept:{cname}",
                "kind": "ghost" if unresolved else "concept",
                "label": cname,
            })

        # ---- memory hierarchy: dormant (paged-out) goals ----
        # The working set is bounded; a goal idle too long is consolidated to
        # the long-term store on disk. Surface those as distinct DORMANT nodes
        # so the viewer shows the whole memory system — what the agent KNOWS,
        # not just what is resident in the network right now. A cue can recall
        # any of them back into the working set.
        working_set = {}
        if self.long_term_store is not None:
            try:
                st = self.long_term_store.stats()
                working_set = {
                    "active_goals": st.get("active", 0),
                    "dormant_goals": st.get("dormant", 0),
                    "total_known": st.get("total", 0),
                }
                for mid in self.long_term_store.dormant_ids("goal"):
                    it = self.long_term_store.items.get(mid)
                    if it is None:
                        continue
                    nodes.append({
                        "id": f"dormant:{mid}",
                        "kind": "dormant_goal",
                        "label": (it.extra or {}).get("name", mid),
                        "competence": round(float(it.value), 3),
                        "usage": int(it.usage_count),
                        "memory_tier": "long_term",
                    })
            except Exception:
                pass

        # ---- live perception overlay (what the head asserts right now) ----
        perception = {}
        if self.symbolizer is not None:
            try:
                st = self.symbolizer.stats
                perception = {
                    "grounded_predicates": st.get("grounded_predicates", 0),
                    "labels": st.get("labels", 0),
                }
            except Exception:
                pass

        # ---- canonicalize edges through the skill-dedup map + drop dupes ----
        # Remap any edge endpoint that pointed at a merged-away skill to its
        # representative, drop self-loops the merge creates, and dedup so the
        # collapsed behaviour has clean single edges.
        node_ids = {n["id"] for n in nodes}
        seen_e = set()
        deduped_edges = []
        for e in edges:
            s = canon.get(e["source"], e["source"])
            t = canon.get(e["target"], e["target"])
            if s == t or s not in node_ids or t not in node_ids:
                continue
            # include the relation label so distinct-relation edges between the
            # same concept pair (breaks_into vs crafts_into) are NOT collapsed
            k = (s, t, e.get("kind"), e.get("label"))
            if k in seen_e:
                continue
            seen_e.add(k)
            deduped_edges.append({**e, "source": s, "target": t})
        edges = deduped_edges

        state = {
            "meta": {
                "episode": int(episode), "timestep": int(timestep),
                "written_at": time.time(),
                "n_skills": len(self.skill_bank.skills),
                "n_behaviours": len(reps),   # distinct grounded behaviours
            },
            "nodes": nodes,
            "edges": edges,
            "firing": list(self.firing_events)[-100:],
            "active": {str(k): v for k, v in self.active_options.items()},
            "working_set": working_set,
            "kg_causal_facts": kg_facts[:200],
            "perception": perception,
        }
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, default=_np_safe)
        os.replace(tmp, self.out_path)
        return self.out_path


def compute_ghost_frontier(skill_bank, knowledge_graph, broadcaster,
                           bonus: float = 1.4) -> Dict[int, float]:
    """Slots worth extra attention because they lead toward UNRESOLVED links.

    A ghost is a concept the agent's GROUNDED facts reference as an object
    but that no skill achieves — "oak_log crafts_into oak_planks" with no
    planks skill. The agent knows the gap exists and cannot close it.

    The actionable signal is not the ghost itself (it has no goal slot and
    inventing one would create an unreachable target that the frontier
    score would chase forever). It is the STEPPING STONE: the slot whose
    skill achieves the SUBJECT of that unresolved fact. Keep valuing
    oak_log, because oak_log is how planks eventually become reachable.

    Returns {slot_index: multiplier}. Empty when nothing is unresolved.
    """
    out: Dict[int, float] = {}
    if knowledge_graph is None or broadcaster is None:
        return out
    try:
        from developmental_ai.skill_bank.skill_notes import (
            read_note_frontmatter)
        # what each skill achieves (ground-truth effects from its note)
        achieved, effects_of = set(), {}
        for sid in skill_bank.skills:
            front = read_note_frontmatter(
                os.path.join(skill_bank.storage_dir, sid)) or {}
            eff = list(front.get("effects", []))
            effects_of[sid] = eff
            achieved.update(eff)

        # unresolved objects, and the subjects that reach them
        reachers: Dict[str, set] = {}
        for f in knowledge_graph.query():
            if f.source != "grounded_event":
                continue
            if f.obj in achieved:
                continue                      # not a gap: something makes it
            reachers.setdefault(f.obj, set()).add(f.subject)
        if not reachers:
            return out

        stepping_stones = set()
        for _ghost, subjects in reachers.items():
            stepping_stones.update(subjects)

        names = list(getattr(broadcaster, "slot_names", []) or [])
        for i, nm in enumerate(names):
            sid = f"ach_{i:02d}_{nm}"[:64]
            if set(effects_of.get(sid, [])) & stepping_stones:
                out[i] = bonus
        if out:
            logger.info("ghost frontier: %d unresolved concept(s) -> "
                        "boosting slots %s", len(reachers), sorted(out))
    except Exception as e:
        logger.warning("ghost frontier computation failed: %s", e)
    return out


def _np_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
