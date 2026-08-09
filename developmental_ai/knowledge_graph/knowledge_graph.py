"""
Knowledge Graph — Symbolic Fact Storage & Reasoning
=====================================================
Stores explicit symbolic facts as a graph: entities (nodes) connected by
relations (edges). Examples: 'cart → has_velocity → 0.5', 'pole → angle → 12deg'.

This is the SYMBOLIC half of the hybrid architecture. While the RSSM world model
learns compressed neural representations, the knowledge graph stores human-readable
facts that can be inspected, queried, and reasoned about.

Two backends:
  1. Neo4j (production): Full graph database with Cypher queries
  2. In-memory (prototyping): NetworkX-based graph, no external dependencies

PyKEEN integration:
  Trains graph embedding models (TransE, RotatE, etc.) that convert the symbolic
  graph structure into neural vectors the world model can process. This bridges
  the neural ↔ symbolic gap: the world model gets structured knowledge as vectors.

Key insight from WALL-E 2.0 (Zhou et al., 2025):
  Extract three types of knowledge from trajectories:
    1. Action rules: "push → object moves"
    2. Knowledge graph facts: "cup → on → table"
    3. Scene graphs: spatial relationships between entities

Key insight from NeSyS (2026):
  The symbolic model should CONSTRAIN the neural model's predictions,
  not just run in parallel. Symbolic rules act as hard guardrails.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os
import logging

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures for symbolic facts
# ---------------------------------------------------------------------------

@dataclass
class SymbolicFact:
    """
    A single fact in the knowledge graph: (subject, relation, object).

    Examples:
      - SymbolicFact("cart", "has_velocity", "positive")
      - SymbolicFact("push_right", "causes", "cart_moves_right")
      - SymbolicFact("pole_angle", "above_threshold", "falling")

    Confidence tracks how certain we are about this fact (0.0 to 1.0).
    Source tracks where the fact came from (observation, inference, etc.).
    """
    subject: str
    relation: str
    obj: str
    confidence: float = 1.0
    source: str = "observation"
    timestamp: int = 0  # When this fact was learned (timestep)

    def to_triple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.obj)

    def __hash__(self):
        return hash((self.subject, self.relation, self.obj))

    def __eq__(self, other):
        if not isinstance(other, SymbolicFact):
            return False
        return self.to_triple() == other.to_triple()


@dataclass
class ActionRule:
    """
    A learned action rule: "if preconditions are met and action is taken, then effects happen."

    Inspired by WALL-E 2.0's action rule extraction from trajectories.
    These rules constrain the world model's predictions — if the agent learns
    "push always makes things move", the world model can't predict otherwise.

    Example:
      preconditions: {"cart_position": "center", "pole_angle": "small"}
      action: "push_right"
      effects: {"cart_velocity": "positive", "pole_angle": "slightly_right"}
    """
    preconditions: Dict[str, str]
    action: str
    effects: Dict[str, str]
    confidence: float = 0.5
    usage_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


# ---------------------------------------------------------------------------
# In-Memory Knowledge Graph (no external dependencies)
# ---------------------------------------------------------------------------

class InMemoryKnowledgeGraph:
    """
    Lightweight in-memory knowledge graph using adjacency lists.

    This is the prototyping backend — no Neo4j server required.
    Good enough for simple environments. For production scale,
    switch to the Neo4j backend.

    Stores:
      - Entity set: all unique entities
      - Relation set: all unique relation types
      - Triples: (subject, relation, object) facts
      - Action rules: learned cause-effect patterns
      - Scene snapshots: periodic state of the world
    """

    def __init__(self):
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.facts: Dict[Tuple[str, str, str], SymbolicFact] = {}
        self.action_rules: List[ActionRule] = []

        # Adjacency: subject → [(relation, object)]
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # Reverse adjacency: object → [(relation, subject)]
        self.reverse_adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # Scene history (periodic snapshots of entity states)
        self.scene_history: List[Dict[str, Any]] = []

    def add_fact(self, fact: SymbolicFact, refresh: bool = False) -> bool:
        """
        Add a fact to the knowledge graph.

        Returns True if this is a NEW fact (novel information),
        False if it already existed (just updates confidence/timestamp).

        `refresh=True` marks the caller as a source that can REVISE its own
        belief downward (the VLM symbolizer): confidence is OVERWRITTEN with
        the current value instead of being ratcheted to a high-water mark.
        Default False keeps the original max() semantics for every existing
        caller. Timestamp stays monotone either way — it is a last-seen mark.
        """
        triple = fact.to_triple()

        if triple in self.facts:
            # Update existing fact's confidence and timestamp
            existing = self.facts[triple]
            if refresh:
                existing.confidence = float(fact.confidence)
            else:
                existing.confidence = max(existing.confidence, fact.confidence)
            existing.timestamp = max(existing.timestamp, fact.timestamp)
            return False  # Not novel

        # Add new fact
        self.facts[triple] = fact
        self.entities.add(fact.subject)
        self.entities.add(fact.obj)
        self.relations.add(fact.relation)

        self.adjacency[fact.subject].append((fact.relation, fact.obj))
        self.reverse_adjacency[fact.obj].append((fact.relation, fact.subject))

        return True  # Novel fact

    def remove_fact(self, subject: str, relation: str, obj: str,
                    source: Optional[str] = None) -> bool:
        """
        RETRACT a fact. Returns True if a fact was actually removed.

        `source` scopes the retraction: when given, the stored fact is only
        removed if its source matches. Safety interlock — one subsystem must
        never delete a triple another subsystem asserted, even when the two
        happen to agree on the (s, r, o) strings.

        Entities/relations are deliberately NOT pruned: an entity may still
        participate in other facts and a correct prune needs a full scan, so
        get_stats()['num_entities'] becomes an upper bound after a retraction.
        Nothing load-bearing reads it — get_triples_for_embedding (the GNN's
        only input) reads self.facts.
        """
        triple = (subject, relation, obj)
        existing = self.facts.get(triple)
        if existing is None:
            return False
        if source is not None and existing.source != source:
            return False

        del self.facts[triple]
        # Keep adjacency in sync. add_fact appends an edge on every NEW fact,
        # so a retract/re-assert cycle would otherwise grow these lists
        # without bound and make get_neighbors() return duplicates.
        fwd = self.adjacency.get(subject)
        if fwd is not None and (relation, obj) in fwd:
            fwd.remove((relation, obj))
        rev = self.reverse_adjacency.get(obj)
        if rev is not None and (relation, subject) in rev:
            rev.remove((relation, subject))
        return True

    def add_action_rule(self, rule: ActionRule) -> None:
        """Add or update an action rule."""
        # Check if a similar rule already exists
        for existing in self.action_rules:
            if (existing.action == rule.action and
                existing.preconditions == rule.preconditions):
                # Update effects and confidence
                existing.effects.update(rule.effects)
                existing.confidence = max(existing.confidence, rule.confidence)
                return

        self.action_rules.append(rule)

    def query(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[SymbolicFact]:
        """
        Query the knowledge graph for matching facts.

        Any combination of subject/relation/obj can be specified.
        Unspecified fields match anything (wildcard).

        Examples:
            query(subject="cart")         → all facts about the cart
            query(relation="causes")      → all causal relationships
            query(subject="push_right", relation="causes")  → effects of pushing right
        """
        results = []
        for triple, fact in self.facts.items():
            if subject is not None and fact.subject != subject:
                continue
            if relation is not None and fact.relation != relation:
                continue
            if obj is not None and fact.obj != obj:
                continue
            results.append(fact)
        return results

    def get_neighbors(self, entity: str) -> List[SymbolicFact]:
        """Get all facts where the entity appears as subject or object."""
        results = []
        for rel, obj in self.adjacency.get(entity, []):
            triple = (entity, rel, obj)
            if triple in self.facts:
                results.append(self.facts[triple])
        for rel, subj in self.reverse_adjacency.get(entity, []):
            triple = (subj, rel, entity)
            if triple in self.facts:
                results.append(self.facts[triple])
        return results

    def get_applicable_rules(self, current_state: Dict[str, str]) -> List[ActionRule]:
        """
        Find action rules whose preconditions match the current state.

        Used for symbolic planning: "given what I know about the current state,
        what actions are predicted to have known effects?"

        Inspired by WALL-E 2.0's rule-based prediction constraining.
        """
        applicable = []
        for rule in self.action_rules:
            # Check if all preconditions are met
            all_met = True
            for key, value in rule.preconditions.items():
                if current_state.get(key) != value:
                    all_met = False
                    break
            if all_met and rule.confidence > 0.3:
                applicable.append(rule)
        return applicable

    def save_scene_snapshot(self, scene: Dict[str, Any]) -> None:
        """Save a scene graph snapshot (spatial/state relationships at a point in time)."""
        self.scene_history.append(scene)

    def get_stats(self) -> Dict[str, int]:
        """Summary statistics of the knowledge graph."""
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "num_facts": len(self.facts),
            "num_action_rules": len(self.action_rules),
            "num_scene_snapshots": len(self.scene_history),
        }

    def get_triples_for_embedding(self) -> List[Tuple[str, str, str]]:
        """Get all triples in a format suitable for PyKEEN embedding training."""
        return [fact.to_triple() for fact in self.facts.values()]

    def save(self, filepath: str) -> None:
        """Save the knowledge graph to a JSON file."""
        data = {
            "facts": [
                {
                    "subject": f.subject,
                    "relation": f.relation,
                    "object": f.obj,
                    "confidence": f.confidence,
                    "source": f.source,
                    "timestamp": f.timestamp,
                }
                for f in self.facts.values()
            ],
            "action_rules": [
                {
                    "preconditions": r.preconditions,
                    "action": r.action,
                    "effects": r.effects,
                    "confidence": r.confidence,
                    "usage_count": r.usage_count,
                    "success_count": r.success_count,
                }
                for r in self.action_rules
            ],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load the knowledge graph from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        for fact_data in data.get("facts", []):
            fact = SymbolicFact(
                subject=fact_data["subject"],
                relation=fact_data["relation"],
                obj=fact_data["object"],
                confidence=fact_data.get("confidence", 1.0),
                source=fact_data.get("source", "loaded"),
                timestamp=fact_data.get("timestamp", 0),
            )
            self.add_fact(fact)

        for rule_data in data.get("action_rules", []):
            rule = ActionRule(
                preconditions=rule_data["preconditions"],
                action=rule_data["action"],
                effects=rule_data["effects"],
                confidence=rule_data.get("confidence", 0.5),
                usage_count=rule_data.get("usage_count", 0),
                success_count=rule_data.get("success_count", 0),
            )
            self.add_action_rule(rule)


# ---------------------------------------------------------------------------
# Neo4j Knowledge Graph (production backend)
# ---------------------------------------------------------------------------

class Neo4jKnowledgeGraph:
    """
    Production knowledge graph backend using Neo4j.

    Same interface as InMemoryKnowledgeGraph but backed by a real graph database.
    Supports Cypher queries, scales to millions of facts, and persists across
    restarts without manual save/load.

    Requires a running Neo4j instance:
      docker run -d --name neo4j -p 7687:7687 -p 7474:7474 \
        -e NEO4J_AUTH=neo4j/password neo4j:5-community
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package not installed. pip install neo4j"
            )
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connection()
        self._ensure_indexes()

        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.action_rules: List[ActionRule] = []
        self.scene_history: List[Dict[str, Any]] = []

        self._sync_from_db()

    def _verify_connection(self) -> None:
        with self._driver.session() as session:
            session.run("RETURN 1")
        logger.info("Neo4j connection established")

    def _ensure_indexes(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)"
            )

    def _sync_from_db(self) -> None:
        """Populate entity/relation caches from existing DB state."""
        with self._driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN e.name AS name")
            for record in result:
                self.entities.add(record["name"])
            result = session.run(
                "MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel"
            )
            for record in result:
                self.relations.add(record["rel"])

    def close(self) -> None:
        self._driver.close()

    def add_fact(self, fact: SymbolicFact, refresh: bool = False) -> bool:
        """`refresh=True` lets a source REVISE its own confidence downward
        instead of ratcheting to a high-water mark. Mirrors the in-memory
        backend exactly so the two cannot diverge."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:Entity {name: $subj})-[r:`" + fact.relation + "`]->"
                "(o:Entity {name: $obj}) "
                "RETURN r.confidence AS conf",
                subj=fact.subject, obj=fact.obj,
            )
            record = result.single()
            if record is not None:
                session.run(
                    "MATCH (s:Entity {name: $subj})-[r:`" + fact.relation + "`]->"
                    "(o:Entity {name: $obj}) "
                    "SET r.confidence = CASE WHEN $refresh OR r.confidence < $conf "
                    "THEN $conf ELSE r.confidence END, "
                    "r.timestamp = CASE WHEN r.timestamp < $ts "
                    "THEN $ts ELSE r.timestamp END",
                    subj=fact.subject, obj=fact.obj,
                    conf=fact.confidence, ts=fact.timestamp,
                    refresh=bool(refresh),
                )
                return False

            session.run(
                "MERGE (s:Entity {name: $subj}) "
                "MERGE (o:Entity {name: $obj}) "
                "CREATE (s)-[r:`" + fact.relation + "` {"
                "confidence: $conf, source: $src, timestamp: $ts}]->(o)",
                subj=fact.subject, obj=fact.obj,
                conf=fact.confidence, src=fact.source, ts=fact.timestamp,
            )
        self.entities.add(fact.subject)
        self.entities.add(fact.obj)
        self.relations.add(fact.relation)
        return True

    def remove_fact(self, subject: str, relation: str, obj: str,
                    source: Optional[str] = None) -> bool:
        """Retract a fact. Same source-scoping contract as the in-memory
        backend. Returns True if a relationship was deleted.

        The relation name is interpolated into the Cypher exactly as add_fact
        and query already do; the symbolizer's relation vocabulary is fixed
        (_FACT_TEMPLATES), so this adds no new injection surface.
        """
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:Entity {name: $subj})-[r:`" + relation + "`]->"
                "(o:Entity {name: $obj}) "
                "WHERE $src IS NULL OR r.source = $src "
                "WITH r LIMIT 1 "
                "DELETE r "
                "RETURN 1 AS removed",
                subj=subject, obj=obj, src=source,
            )
            return result.single() is not None

    def add_action_rule(self, rule: ActionRule) -> None:
        for existing in self.action_rules:
            if (existing.action == rule.action and
                    existing.preconditions == rule.preconditions):
                existing.effects.update(rule.effects)
                existing.confidence = max(existing.confidence, rule.confidence)
                return
        self.action_rules.append(rule)

    def query(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[SymbolicFact]:
        clauses = []
        params: Dict[str, Any] = {}
        if subject is not None:
            clauses.append("s.name = $subj")
            params["subj"] = subject
        if obj is not None:
            clauses.append("o.name = $obj")
            params["obj"] = obj

        rel_pattern = f":`{relation}`" if relation else ""
        where = " WHERE " + " AND ".join(clauses) if clauses else ""

        cypher = (
            f"MATCH (s:Entity)-[r{rel_pattern}]->(o:Entity){where} "
            f"RETURN s.name AS subj, type(r) AS rel, o.name AS obj, "
            f"r.confidence AS conf, r.source AS src, r.timestamp AS ts"
        )

        results = []
        with self._driver.session() as session:
            for record in session.run(cypher, **params):
                results.append(SymbolicFact(
                    subject=record["subj"],
                    relation=record["rel"],
                    obj=record["obj"],
                    confidence=record["conf"] or 1.0,
                    source=record["src"] or "observation",
                    timestamp=record["ts"] or 0,
                ))
        return results

    def get_neighbors(self, entity: str) -> List[SymbolicFact]:
        results = []
        with self._driver.session() as session:
            for record in session.run(
                "MATCH (s:Entity {name: $name})-[r]->(o:Entity) "
                "RETURN s.name AS subj, type(r) AS rel, o.name AS obj, "
                "r.confidence AS conf, r.source AS src, r.timestamp AS ts "
                "UNION "
                "MATCH (s:Entity)-[r]->(o:Entity {name: $name}) "
                "RETURN s.name AS subj, type(r) AS rel, o.name AS obj, "
                "r.confidence AS conf, r.source AS src, r.timestamp AS ts",
                name=entity,
            ):
                results.append(SymbolicFact(
                    subject=record["subj"],
                    relation=record["rel"],
                    obj=record["obj"],
                    confidence=record["conf"] or 1.0,
                    source=record["src"] or "observation",
                    timestamp=record["ts"] or 0,
                ))
        return results

    def get_applicable_rules(self, current_state: Dict[str, str]) -> List[ActionRule]:
        applicable = []
        for rule in self.action_rules:
            all_met = True
            for key, value in rule.preconditions.items():
                if current_state.get(key) != value:
                    all_met = False
                    break
            if all_met and rule.confidence > 0.3:
                applicable.append(rule)
        return applicable

    def save_scene_snapshot(self, scene: Dict[str, Any]) -> None:
        self.scene_history.append(scene)

    def get_stats(self) -> Dict[str, int]:
        with self._driver.session() as session:
            fact_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]
            entity_count = session.run(
                "MATCH (e:Entity) RETURN count(e) AS c"
            ).single()["c"]
        return {
            "num_entities": entity_count,
            "num_relations": len(self.relations),
            "num_facts": fact_count,
            "num_action_rules": len(self.action_rules),
            "num_scene_snapshots": len(self.scene_history),
        }

    def get_triples_for_embedding(self) -> List[Tuple[str, str, str]]:
        triples = []
        with self._driver.session() as session:
            for record in session.run(
                "MATCH (s:Entity)-[r]->(o:Entity) "
                "RETURN s.name AS subj, type(r) AS rel, o.name AS obj"
            ):
                triples.append((record["subj"], record["rel"], record["obj"]))
        return triples

    def save(self, filepath: str) -> None:
        """Export Neo4j graph to JSON for backup/portability."""
        data = {
            "facts": [
                {
                    "subject": f.subject,
                    "relation": f.relation,
                    "object": f.obj,
                    "confidence": f.confidence,
                    "source": f.source,
                    "timestamp": f.timestamp,
                }
                for f in self.query()
            ],
            "action_rules": [
                {
                    "preconditions": r.preconditions,
                    "action": r.action,
                    "effects": r.effects,
                    "confidence": r.confidence,
                    "usage_count": r.usage_count,
                    "success_count": r.success_count,
                }
                for r in self.action_rules
            ],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Import facts from a JSON backup into Neo4j."""
        with open(filepath, "r") as f:
            data = json.load(f)
        for fact_data in data.get("facts", []):
            fact = SymbolicFact(
                subject=fact_data["subject"],
                relation=fact_data["relation"],
                obj=fact_data["object"],
                confidence=fact_data.get("confidence", 1.0),
                source=fact_data.get("source", "loaded"),
                timestamp=fact_data.get("timestamp", 0),
            )
            self.add_fact(fact)
        for rule_data in data.get("action_rules", []):
            rule = ActionRule(
                preconditions=rule_data["preconditions"],
                action=rule_data["action"],
                effects=rule_data["effects"],
                confidence=rule_data.get("confidence", 0.5),
                usage_count=rule_data.get("usage_count", 0),
                success_count=rule_data.get("success_count", 0),
            )
            self.add_action_rule(rule)

    def clear(self) -> None:
        """Delete all nodes and relationships."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        self.entities.clear()
        self.relations.clear()
        self.action_rules.clear()
        self.scene_history.clear()


def create_knowledge_graph(
    config: Optional[Dict[str, Any]] = None,
) -> "InMemoryKnowledgeGraph | Neo4jKnowledgeGraph":
    """
    Factory: create the best available knowledge graph backend.

    Tries Neo4j first (if configured and reachable), falls back to in-memory.
    """
    config = config or {}
    use_fallback = config.get("use_in_memory_fallback", True)

    if NEO4J_AVAILABLE and not config.get("force_in_memory", False):
        try:
            kg = Neo4jKnowledgeGraph(
                uri=config.get("neo4j_uri", "bolt://localhost:7687"),
                user=config.get("neo4j_user", "neo4j"),
                password=config.get("neo4j_password", "password"),
            )
            logger.info("Using Neo4j knowledge graph backend")
            return kg
        except Exception as e:
            if use_fallback:
                logger.warning(
                    f"Neo4j unavailable ({e}), falling back to in-memory graph"
                )
            else:
                raise

    logger.info("Using in-memory knowledge graph backend")
    return InMemoryKnowledgeGraph()


# ---------------------------------------------------------------------------
# Fact Extractor — bridges neural observations to symbolic facts
# ---------------------------------------------------------------------------

class FactExtractor:
    """
    Extracts symbolic facts from raw observations and transitions.

    This is one of the key bridge components between the neural and symbolic
    layers. It takes raw environment observations (numbers) and produces
    human-readable symbolic facts.

    The extraction is environment-specific — different environments have
    different observation meanings. Override extract_facts_from_obs() for
    custom environments.

    Inspired by:
      - WALL-E 2.0: extracts action rules and scene graphs from trajectories
      - NeSyS: alternates symbolic rule learning with neural model training
      - Curiosity-Driven Imagination (Lorang 2025): uses symbolic transitions
        to guide curiosity toward structurally novel experiences
    """

    def __init__(
        self,
        obs_labels: Optional[List[str]] = None,
        discretize_bins: int = 5,
        min_confidence: float = 0.7,
    ):
        """
        Args:
            obs_labels: Human-readable names for each observation dimension.
                        E.g., ["cart_position", "cart_velocity", "pole_angle", "pole_velocity"]
            discretize_bins: How many categories to discretize continuous values into.
                            E.g., 5 → ["very_low", "low", "medium", "high", "very_high"]
            min_confidence: Minimum confidence threshold for storing facts.
        """
        self.obs_labels = obs_labels
        self.discretize_bins = discretize_bins
        self.min_confidence = min_confidence

        # Bin labels for discretization
        self.bin_labels = self._make_bin_labels(discretize_bins)

        # Track observation statistics for adaptive discretization
        self.obs_min = None
        self.obs_max = None
        self.obs_count = 0

    def _make_bin_labels(self, n_bins: int) -> List[str]:
        """Create human-readable labels for discretized bins."""
        if n_bins <= 3:
            return ["low", "medium", "high"]
        elif n_bins <= 5:
            return ["very_low", "low", "medium", "high", "very_high"]
        else:
            return [f"level_{i}" for i in range(n_bins)]

    def _discretize(self, value: float, dim_idx: int) -> str:
        """
        Convert a continuous value to a symbolic label.

        Uses running min/max to adaptively determine bin boundaries.
        """
        if self.obs_min is None:
            return "unknown"

        vmin = self.obs_min[dim_idx]
        vmax = self.obs_max[dim_idx]

        if vmax - vmin < 1e-8:
            return "constant"

        # Normalize to [0, 1]
        normalized = np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)

        # Map to bin
        bin_idx = min(int(normalized * self.discretize_bins), self.discretize_bins - 1)
        return self.bin_labels[bin_idx]

    def update_statistics(self, obs: np.ndarray) -> None:
        """Update running min/max statistics for discretization."""
        if self.obs_min is None:
            self.obs_min = obs.copy()
            self.obs_max = obs.copy()
        else:
            self.obs_min = np.minimum(self.obs_min, obs)
            self.obs_max = np.maximum(self.obs_max, obs)
        self.obs_count += 1

    def extract_state_facts(
        self,
        obs: np.ndarray,
        timestep: int = 0,
    ) -> List[SymbolicFact]:
        """
        Extract symbolic facts from a single observation.

        Converts each dimension of the observation vector into a symbolic fact.
        E.g., observation[2] = 0.15 → SymbolicFact("pole_angle", "has_value", "medium")
        """
        self.update_statistics(obs)

        facts = []
        for i, value in enumerate(obs):
            label = self.obs_labels[i] if self.obs_labels and i < len(self.obs_labels) else f"dim_{i}"
            symbolic_value = self._discretize(float(value), i)

            fact = SymbolicFact(
                subject=label,
                relation="has_value",
                obj=symbolic_value,
                confidence=0.9,
                source="observation",
                timestamp=timestep,
            )
            facts.append(fact)

        return facts

    def extract_transition_facts(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        reward: float,
        timestep: int = 0,
        action_labels: Optional[List[str]] = None,
    ) -> Tuple[List[SymbolicFact], Optional[ActionRule]]:
        """
        Extract facts from a state transition (obs → action → next_obs).

        Produces:
          1. State facts for the next observation
          2. Change facts (what changed between obs and next_obs)
          3. An action rule capturing the causal relationship

        This is the core of trajectory-to-symbol extraction, inspired by WALL-E 2.0.
        """
        self.update_statistics(next_obs)
        facts = []

        # Get action label. Discrete actions are scalar indices into
        # action_labels; continuous actions are vectors with no discrete label.
        if np.ndim(action) == 0:
            a = int(action)
            action_name = (action_labels[a] if (action_labels and a < len(action_labels))
                           else f"action_{a}")
        else:
            action_name = "action_continuous"

        # Extract change facts: what dimensions changed significantly?
        preconditions = {}
        effects = {}

        for i in range(len(obs)):
            label = self.obs_labels[i] if self.obs_labels and i < len(self.obs_labels) else f"dim_{i}"

            old_value = self._discretize(float(obs[i]), i)
            new_value = self._discretize(float(next_obs[i]), i)

            preconditions[label] = old_value

            if old_value != new_value:
                # Something changed — record it
                effects[label] = new_value

                change_fact = SymbolicFact(
                    subject=action_name,
                    relation=f"changed_{label}",
                    obj=f"{old_value}_to_{new_value}",
                    confidence=0.8,
                    source="transition",
                    timestamp=timestep,
                )
                facts.append(change_fact)

        # Create action rule if there were observable effects
        action_rule = None
        if effects:
            action_rule = ActionRule(
                preconditions=preconditions,
                action=action_name,
                effects=effects,
                confidence=0.5,  # Low initial confidence, increases with repetition
            )

        # Add reward fact
        if abs(reward) > 0.01:
            reward_label = "positive" if reward > 0 else "negative"
            facts.append(SymbolicFact(
                subject=action_name,
                relation="received_reward",
                obj=reward_label,
                confidence=1.0,
                source="reward",
                timestamp=timestep,
            ))

        return facts, action_rule

    def extract_scene_graph(
        self,
        obs: np.ndarray,
        timestep: int = 0,
    ) -> Dict[str, Any]:
        """
        Extract a scene graph snapshot from the current observation.

        A scene graph captures the spatial and state relationships between
        all entities at a single moment in time. This bridges raw perception
        and the knowledge graph.
        """
        scene = {
            "timestep": timestep,
            "entities": {},
            "relationships": [],
        }

        for i, value in enumerate(obs):
            label = self.obs_labels[i] if self.obs_labels and i < len(self.obs_labels) else f"dim_{i}"
            symbolic_value = self._discretize(float(value), i)
            scene["entities"][label] = {
                "raw_value": float(value),
                "symbolic_value": symbolic_value,
            }

        return scene
