"""
Glue Layer — Neural-Symbolic Bridge Implementation
=====================================================
Concrete implementation connecting the RSSM latent space to the
symbolic knowledge graph, skill bank, and curriculum system.

Components:
  1. KnowledgeIntegrator: GNN + Gate → knowledge-augmented latent states
  2. GoalGenerator: Autotelic goal setting from curiosity + KG analysis
  3. AdvancedMasteryDetector: Multi-signal mastery (KL + reward + rules)
  4. AdvancedCurriculumManager: Zone of proximal development calculation
  5. SkillSelector: Multi-signal skill retrieval (cosine + graph + prereqs)

Reference:
  - NeSyS (2026): symbolic embeddings constrain neural predictions via gating
  - Autotelic Agents (Colas 2022): self-generated goals from curiosity
  - H-GRAIL (2025): hierarchical intrinsic motivation + bandit curriculum
  - WALL-E 2.0 (Zhou 2025): action rule extraction from trajectories
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging

from developmental_ai.knowledge_graph.graph_embeddings import (
    KnowledgeGraphGNN,
    SymbolicNeuralGate,
    KGEmbeddingTrainer,
)
from developmental_ai.knowledge_graph.knowledge_graph import (
    InMemoryKnowledgeGraph,
    SymbolicFact,
)
from developmental_ai.skill_bank.skill_bank import SkillBank, Skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge Integrator — GNN + Gate for knowledge-augmented latents
# ---------------------------------------------------------------------------

class KnowledgeIntegrator:
    """
    Bridges symbolic knowledge into the neural pipeline.

    Data flow:
      KG facts → PyKEEN embeddings → GNN → knowledge vector
      → Gate(RSSM latent, knowledge vector) → augmented latent

    The gate is initialized to favor neural representations (gate ≈ 0),
    so the augmented latent starts close to the raw RSSM latent and
    gradually incorporates symbolic knowledge as the KG grows.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        latent_dim: int = 512,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.gnn_output_dim = gnn_output_dim
        self.latent_dim = latent_dim

        self.gnn = KnowledgeGraphGNN(
            input_dim=embedding_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
        ).to(device)

        self.gate = SymbolicNeuralGate(
            neural_dim=latent_dim,
            symbolic_dim=gnn_output_dim,
            output_dim=latent_dim,
        ).to(device)

        # Initialize gate to trust neural by default (sigmoid(-2) ≈ 0.12)
        # This ensures the augmented latent starts close to the raw RSSM output
        with torch.no_grad():
            self.gate.gate_net[0].bias.fill_(-2.0)

        self.knowledge_vector: Optional[torch.Tensor] = None

    def update_knowledge_vector(
        self,
        kg_embedder: KGEmbeddingTrainer,
        knowledge_graph: InMemoryKnowledgeGraph,
    ) -> Optional[torch.Tensor]:
        """Recompute the knowledge vector from current KG state."""
        entity_to_id, embeddings = kg_embedder.get_all_entity_embeddings()

        if embeddings.shape[0] == 0:
            return None

        embeddings = embeddings.to(self.device)
        triples = knowledge_graph.get_triples_for_embedding()
        edge_index = self.gnn.build_edge_index(triples, entity_to_id)

        if edge_index.shape[1] > 0:
            edge_index = edge_index.to(self.device)
        else:
            edge_index = None

        with torch.no_grad():
            self.knowledge_vector = self.gnn(embeddings, edge_index)

        return self.knowledge_vector

    def augment_latent(self, neural_latent: torch.Tensor) -> torch.Tensor:
        """Blend neural latent with symbolic knowledge via learned gate."""
        if self.knowledge_vector is None:
            return neural_latent

        symbolic = self.knowledge_vector.unsqueeze(0).expand(
            neural_latent.shape[0], -1
        )
        with torch.no_grad():
            return self.gate(neural_latent, symbolic)

    @property
    def has_knowledge(self) -> bool:
        return self.knowledge_vector is not None


# ---------------------------------------------------------------------------
# Goal Generator — Autotelic goal setting
# ---------------------------------------------------------------------------

class GoalGenerator:
    """
    Generates exploration goals from curiosity signals and KG analysis.
    Inspired by Autotelic Agents (Colas 2022).

    Strategy:
    1. Find KG entities with fewest connections (least explored areas)
    2. Weight by symbolic decoder uncertainty (least confident predictions)
    3. Avoid recently targeted goals (diversity pressure)
    """

    def __init__(self, obs_dim: int, embedding_dim: int = 16):
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim
        self.current_goal: Optional[np.ndarray] = None
        self.goal_history: List[np.ndarray] = []

    def generate_goal(
        self,
        knowledge_graph: InMemoryKnowledgeGraph,
        symbolic_decoder=None,
        scene_history: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """
        Generate a goal embedding targeting underexplored areas.

        Returns:
            goal_embedding: shape (embedding_dim,) normalized vector
        """
        entities = list(knowledge_graph.entities)

        if not entities:
            goal = self._random_goal()
            self._record_goal(goal)
            return goal

        # Find least-connected entities (most to explore)
        entity_degrees = {}
        for entity in entities:
            entity_degrees[entity] = len(knowledge_graph.get_neighbors(entity))

        min_degree = min(entity_degrees.values())
        frontier_entities = [
            e for e, d in entity_degrees.items() if d <= min_degree + 1
        ]

        # Build goal from frontier + uncertainty
        seed = hash(tuple(sorted(frontier_entities))) % (2**31)
        rng = np.random.RandomState(seed)
        goal = rng.randn(self.embedding_dim).astype(np.float32)

        # Bias toward uncertain dimensions if decoder info available
        if symbolic_decoder is not None and symbolic_decoder.avg_confidence > 0:
            uncertainty = 1.0 - symbolic_decoder.avg_confidence
            goal[:min(self.obs_dim, self.embedding_dim)] += uncertainty

        # Diversity pressure: push away from recent goals
        if self.goal_history:
            recent = np.mean(self.goal_history[-5:], axis=0)
            goal -= 0.3 * recent

        # Scene dynamics: bias toward dimensions that change most
        if scene_history and len(scene_history) >= 2:
            dynamics = self._compute_scene_dynamics(scene_history)
            goal[:min(len(dynamics), self.embedding_dim)] += dynamics[:self.embedding_dim] * 0.5

        goal /= np.linalg.norm(goal) + 1e-8
        self._record_goal(goal)
        return goal

    def _random_goal(self) -> np.ndarray:
        goal = np.random.randn(self.embedding_dim).astype(np.float32)
        goal /= np.linalg.norm(goal) + 1e-8
        return goal

    def _record_goal(self, goal: np.ndarray) -> None:
        self.current_goal = goal
        self.goal_history.append(goal.copy())
        if len(self.goal_history) > 100:
            self.goal_history = self.goal_history[-50:]

    def _compute_scene_dynamics(
        self, scene_history: List[Dict]
    ) -> np.ndarray:
        """Identify which observation dimensions change most across scenes."""
        if len(scene_history) < 2:
            return np.zeros(self.obs_dim, dtype=np.float32)

        recent = scene_history[-10:]
        changes = np.zeros(self.obs_dim, dtype=np.float32)

        for i in range(1, len(recent)):
            prev_ents = recent[i - 1].get("entities", {})
            curr_ents = recent[i].get("entities", {})

            for dim_idx, label in enumerate(sorted(curr_ents.keys())):
                if dim_idx >= self.obs_dim:
                    break
                prev_val = prev_ents.get(label, {}).get("symbolic_value", "")
                curr_val = curr_ents.get(label, {}).get("symbolic_value", "")
                if prev_val != curr_val:
                    changes[dim_idx] += 1.0

        # Normalize
        total = changes.sum()
        if total > 0:
            changes /= total

        return changes


# ---------------------------------------------------------------------------
# Hierarchical Goal Manager — H-GRAIL-inspired multi-level goals
# ---------------------------------------------------------------------------

class HierarchicalGoalManager:
    """
    Manages a hierarchy of goals: high-level goals decompose into sub-goals
    that the agent pursues in sequence. Inspired by H-GRAIL (2025).

    Structure:
      - A goal stack (deepest sub-goal on top, root goal on bottom)
      - Each goal has: embedding, description, completion condition, children
      - When the top sub-goal is completed, it pops and the next one starts
      - When all sub-goals complete, the parent goal completes
      - Failed sub-goals trigger re-planning

    The LLM can suggest goal decompositions; the system also auto-decomposes
    based on skill bank analysis (if a skill exists that solves a sub-problem,
    it becomes a sub-goal).
    """

    def __init__(
        self,
        obs_dim: int,
        embedding_dim: int = 16,
        max_depth: int = 3,
        subgoal_patience: int = 100,
    ):
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim
        self.max_depth = max_depth
        self.subgoal_patience = subgoal_patience

        self.goal_stack: List[Dict[str, Any]] = []
        self.completed_goals: List[Dict[str, Any]] = []
        self.failed_goals: List[Dict[str, Any]] = []
        self._episode_counter = 0

    def set_root_goal(
        self,
        embedding: np.ndarray,
        description: str = "",
        completion_reward_threshold: float = 0.0,
    ) -> None:
        """Set a new root goal, clearing the stack."""
        self.goal_stack = [{
            "embedding": embedding,
            "description": description,
            "depth": 0,
            "episodes_active": 0,
            "best_reward": float("-inf"),
            "completion_threshold": completion_reward_threshold,
            "children_completed": 0,
            "total_children": 0,
        }]
        self._episode_counter = 0

    def decompose_goal(
        self,
        skill_bank: SkillBank,
        knowledge_graph,
        llm_suggestions: Optional[List[Dict[str, Any]]] = None,
        stage: Optional[str] = None,
    ) -> int:
        """
        Decompose the current top goal into sub-goals.

        Strategy (in priority order):
        1. LLM suggestions — if the LLM proposed sub-goals, use them
        2. Skill-based — if mastered skills cover parts of the goal,
           create sub-goals around them
        3. KG-based — split goal along knowledge graph clusters

        Returns number of sub-goals created.
        """
        if not self.goal_stack:
            return 0

        current = self.goal_stack[-1]
        if current["depth"] >= self.max_depth:
            return 0

        sub_goals = []

        # Strategy 1: LLM-suggested decomposition
        if llm_suggestions:
            for suggestion in llm_suggestions[:4]:
                emb = suggestion.get("embedding")
                if emb is None:
                    emb = self._derive_subgoal_embedding(
                        current["embedding"],
                        len(sub_goals),
                        len(llm_suggestions),
                    )
                else:
                    emb = np.array(emb, dtype=np.float32)
                sub_goals.append({
                    "embedding": emb,
                    "description": suggestion.get("description", ""),
                    "depth": current["depth"] + 1,
                    "episodes_active": 0,
                    "best_reward": float("-inf"),
                    "completion_threshold": suggestion.get("threshold", 0.0),
                    "children_completed": 0,
                    "total_children": 0,
                })

        # Strategy 2: skill-based decomposition
        if not sub_goals:
            mastered = skill_bank.get_mastered_skills()
            relevant = []
            for skill in mastered:
                if skill.context_embedding is not None:
                    skill_emb = np.array(skill.context_embedding)
                    # Pad or truncate to match embedding_dim
                    if len(skill_emb) > self.embedding_dim:
                        skill_emb = skill_emb[:self.embedding_dim]
                    elif len(skill_emb) < self.embedding_dim:
                        skill_emb = np.pad(
                            skill_emb, (0, self.embedding_dim - len(skill_emb))
                        )
                    sim = np.dot(
                        current["embedding"], skill_emb
                    ) / (
                        np.linalg.norm(current["embedding"])
                        * np.linalg.norm(skill_emb) + 1e-8
                    )
                    if sim > 0.3:
                        relevant.append((skill, sim, skill_emb))

            relevant.sort(key=lambda x: x[1], reverse=True)
            for skill, sim, emb in relevant[:3]:
                sub_goals.append({
                    "embedding": emb,
                    "description": f"Apply skill: {skill.name}",
                    "depth": current["depth"] + 1,
                    "episodes_active": 0,
                    "best_reward": float("-inf"),
                    "completion_threshold": 0.0,
                    "children_completed": 0,
                    "total_children": 0,
                })

        # Strategy 3: KG cluster-based split — EXPLORATION only.
        # Splitting a goal along KG clusters asks the agent to visit novel
        # regions of the graph. A converged exploiter generates no novelty, so
        # such a sub-goal is unsatisfiable and merely burns the patience budget.
        # We therefore only do cluster-exploration in EXPLORE (or when stage is
        # unknown, preserving the original behavior).
        exploring = stage in (None, DevelopmentalStageController.EXPLORE)
        if not sub_goals and exploring and hasattr(knowledge_graph, "entities"):
            entities = list(knowledge_graph.entities)
            if len(entities) >= 4:
                n_splits = min(3, len(entities) // 2)
                for i in range(n_splits):
                    sub_emb = self._derive_subgoal_embedding(
                        current["embedding"], i, n_splits
                    )
                    sub_goals.append({
                        "embedding": sub_emb,
                        "description": f"Explore cluster {i+1}/{n_splits}",
                        "depth": current["depth"] + 1,
                        "episodes_active": 0,
                        "best_reward": float("-inf"),
                        "completion_threshold": 0.0,
                        "children_completed": 0,
                        "total_children": 0,
                    })

        if not sub_goals:
            return 0

        current["total_children"] = len(sub_goals)
        # Push sub-goals in reverse order so the first one is on top
        for sg in reversed(sub_goals):
            self.goal_stack.append(sg)

        logger.info(
            f"Decomposed goal (depth={current['depth']}) into "
            f"{len(sub_goals)} sub-goals"
        )
        return len(sub_goals)

    def _derive_subgoal_embedding(
        self,
        parent_embedding: np.ndarray,
        index: int,
        total: int,
    ) -> np.ndarray:
        """Create a sub-goal embedding by perturbing the parent."""
        rng = np.random.RandomState(
            hash((parent_embedding.tobytes(), index)) % (2**31)
        )
        offset = rng.randn(self.embedding_dim).astype(np.float32) * 0.3
        angle = 2 * np.pi * index / max(total, 1)
        offset[0] += 0.5 * np.cos(angle)
        if self.embedding_dim > 1:
            offset[1] += 0.5 * np.sin(angle)
        sub = parent_embedding + offset
        sub /= np.linalg.norm(sub) + 1e-8
        return sub

    def step(self, episode_reward: float) -> Dict[str, Any]:
        """
        Called after each episode. Updates the active sub-goal and checks
        for completion or failure.

        Returns dict with status information for the developmental loop.
        """
        self._episode_counter += 1
        result: Dict[str, Any] = {
            "goal_completed": False,
            "subgoal_completed": False,
            "subgoal_failed": False,
            "replan_needed": False,
            "stack_depth": len(self.goal_stack),
        }

        if not self.goal_stack:
            return result

        current = self.goal_stack[-1]
        current["episodes_active"] += 1

        # Track genuine progress: did this episode beat the goal's best so far?
        improved = episode_reward > current["best_reward"] + 1e-6
        current["best_reward"] = max(current["best_reward"], episode_reward)
        if improved:
            current["episodes_since_improvement"] = 0
        else:
            current["episodes_since_improvement"] = (
                current.get("episodes_since_improvement", 0) + 1
            )

        # Check completion
        if episode_reward >= current["completion_threshold"] and current["episodes_active"] >= 5:
            self._complete_current_goal()
            result["subgoal_completed"] = True
            if not self.goal_stack:
                result["goal_completed"] = True

        # Check abandonment — fire ONLY on a genuine stall (no improvement for
        # `subgoal_patience` episodes), NOT on a flat episode clock. A goal whose
        # reward is still climbing toward its threshold is never abandoned. This
        # makes the safety valve regime-fair: an EXPLOIT consolidation goal that
        # is actively being satisfied won't be killed mid-progress, and the
        # budget now matches its original stated intent ("patience without
        # improvement").
        elif current.get("episodes_since_improvement", 0) > self.subgoal_patience:
            self._fail_current_goal()
            result["subgoal_failed"] = True
            result["replan_needed"] = True

        return result

    def _complete_current_goal(self) -> None:
        completed = self.goal_stack.pop()
        self.completed_goals.append(completed)
        logger.info(
            f"Sub-goal completed: '{completed['description']}' "
            f"(depth={completed['depth']}, "
            f"reward={completed['best_reward']:.2f})"
        )

        # Notify parent
        if self.goal_stack:
            parent = self.goal_stack[-1]
            parent["children_completed"] += 1
            if (parent["total_children"] > 0 and
                    parent["children_completed"] >= parent["total_children"]):
                self._complete_current_goal()

    def _fail_current_goal(self) -> None:
        failed = self.goal_stack.pop()
        self.failed_goals.append(failed)
        logger.info(
            f"Sub-goal abandoned (stalled): '{failed['description']}' "
            f"after {failed['episodes_active']} episodes "
            f"({failed.get('episodes_since_improvement', 0)} without progress)"
        )

    def reset_active_goal(self) -> None:
        """
        Drop the active goal stack so the next generate_goal() builds a fresh
        root. Completed/failed history (stats) is preserved.

        Called on a developmental stage change: a goal's *intent* is tied to
        the regime it was born in (a frontier goal in EXPLORE, a consolidation
        goal in EXPLOIT). A long-lived goal whose completion threshold is
        unreachable in the new regime would otherwise persist forever — the
        stage-gating in generate_goal only takes effect when a NEW root is
        built. Resetting here lets the regime switch actually re-target the goal.
        """
        self.goal_stack = []
        self._episode_counter = 0

    @property
    def current_goal(self) -> Optional[Dict[str, Any]]:
        return self.goal_stack[-1] if self.goal_stack else None

    @property
    def current_embedding(self) -> Optional[np.ndarray]:
        if self.goal_stack:
            return self.goal_stack[-1]["embedding"]
        return None

    @property
    def depth(self) -> int:
        return len(self.goal_stack)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "stack_depth": len(self.goal_stack),
            "completed_goals": len(self.completed_goals),
            "failed_goals": len(self.failed_goals),
            "current_description": (
                self.goal_stack[-1]["description"]
                if self.goal_stack else ""
            ),
            "current_depth": (
                self.goal_stack[-1]["depth"]
                if self.goal_stack else -1
            ),
        }


# ---------------------------------------------------------------------------
# Advanced Mastery Detector — Multi-signal mastery detection
# ---------------------------------------------------------------------------

class AdvancedMasteryDetector:
    """
    Detects skill mastery using three complementary signals:
    1. KL divergence trend — decreasing KL means the world model
       has learned to predict this behavior (low surprise)
    2. Reward stability — low variance means consistent performance
    3. Action rule confidence — high-confidence rules mean the agent
       has built reliable causal understanding

    Requires at least 2 of 3 signals to agree before declaring mastery.
    """

    def __init__(
        self,
        kl_window: int = 50,
        reward_window: int = 50,
        kl_slope_threshold: float = -0.001,
        reward_cv_threshold: float = 0.5,
        rule_confidence_threshold: float = 0.6,
        min_samples: int = 20,
        reward_target: float = 475.0,
        reward_floor: float = 0.0,
    ):
        self.kl_window = kl_window
        self.reward_window = reward_window
        self.kl_slope_threshold = kl_slope_threshold
        self.reward_cv_threshold = reward_cv_threshold
        self.rule_confidence_threshold = rule_confidence_threshold
        self.min_samples = min_samples
        # Competence is normalized against an env-relative band
        # [reward_floor, reward_target]: floor = random/worst-case return,
        # target = "excellent" return. Defaults (0.0, 475.0) reproduce the
        # original CartPole behavior EXACTLY (competence = reward / 475).
        # Negative-reward envs (e.g. Acrobot floor=-500 target=-100) need a
        # shifted band so a strong-but-negative return maps to high competence.
        self.reward_target = reward_target
        self.reward_floor = reward_floor

        self.kl_history: deque = deque(maxlen=kl_window)
        self.reward_history: deque = deque(maxlen=reward_window)

    def record_kl(self, kl_value: float) -> None:
        self.kl_history.append(kl_value)

    def record_reward(self, reward: float) -> None:
        self.reward_history.append(reward)

    def detect_mastery(
        self,
        knowledge_graph: InMemoryKnowledgeGraph,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Multi-signal mastery check. Returns (is_mastered, signal_details).
        """
        signals = {}

        kl_signal = self._compute_kl_signal()
        signals["kl_trend"] = kl_signal

        reward_signal = self._compute_reward_signal()
        signals["reward_stability"] = reward_signal

        rule_signal = self._compute_rule_signal(knowledge_graph)
        signals["rule_confidence"] = rule_signal

        votes = sum([
            kl_signal >= 0.5,
            reward_signal >= 0.5,
            rule_signal >= 0.5,
        ])

        is_mastered = votes >= 2
        # Behavior-dominant blend: mastering a control task is primarily about
        # task competence (reward), supported by a converged world model (kl).
        # The symbolic rule signal is inherently sparse for simple control envs,
        # so it contributes but must not cap a genuinely-mastered agent's score.
        signals["combined_score"] = (
            0.55 * reward_signal + 0.30 * kl_signal + 0.15 * rule_signal
        )
        signals["votes"] = float(votes)

        return is_mastered, signals

    def _compute_kl_signal(self) -> float:
        """Decreasing, low KL → mastered. Returns 0.0 to 1.0."""
        if len(self.kl_history) < self.min_samples:
            return 0.0

        kl_values = np.array(list(self.kl_history))
        x = np.arange(len(kl_values))
        slope = np.polyfit(x, kl_values, 1)[0]

        mean_kl = np.mean(kl_values[-self.min_samples:])

        slope_score = np.clip(-slope / abs(self.kl_slope_threshold), 0, 1)
        level_score = np.clip(1.0 - mean_kl / 5.0, 0, 1)

        # Level-dominated: a converged (low, flat) KL is the goal. A flat slope
        # near zero must NOT penalize an already-low KL — once the world model
        # has learned the dynamics there is nothing left to decrease, so the
        # slope term is only a small bonus, never a drag on a mastered model.
        return float(0.9 * level_score + 0.1 * slope_score)

    def _compute_reward_signal(self) -> float:
        """Stable, positive reward → mastered. Returns 0.0 to 1.0."""
        if len(self.reward_history) < self.min_samples:
            return 0.0

        rewards = np.array(list(self.reward_history))
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        if abs(mean_reward) < 1e-8:
            return 0.0

        cv = std_reward / (abs(mean_reward) + 1e-8)

        # Competence-dominated: how far mean reward has climbed across the
        # env-relative band [reward_floor, reward_target] (e.g. 0..475 for
        # CartPole-v1, -500..-100 for Acrobot-v1). This makes competence work
        # for NEGATIVE-reward envs: with floor=0/target=475 it reduces exactly
        # to the original mean_reward/475. Consistency (low CV) is a secondary
        # signal — a noisy-but-strong agent should still score well, so
        # competence carries the bulk of the weight.
        denom = self.reward_target - self.reward_floor
        if abs(denom) < 1e-8:
            competence = 0.0
        else:
            competence = np.clip(
                (mean_reward - self.reward_floor) / denom, 0, 1
            )
        consistency = np.clip(1.0 - cv / self.reward_cv_threshold, 0, 1)

        return float(0.7 * competence + 0.3 * consistency)

    def _compute_rule_signal(
        self, knowledge_graph: InMemoryKnowledgeGraph
    ) -> float:
        """High-confidence action rules → mastered. Returns 0.0 to 1.0."""
        rules = knowledge_graph.action_rules
        if not rules:
            return 0.0

        confidences = [r.confidence for r in rules]
        avg_conf = np.mean(confidences)
        high_conf_ratio = sum(
            1 for c in confidences if c >= self.rule_confidence_threshold
        ) / len(confidences)

        return float(0.5 * avg_conf + 0.5 * high_conf_ratio)

    def reset(self) -> None:
        self.kl_history.clear()
        self.reward_history.clear()


# ---------------------------------------------------------------------------
# Advanced Curriculum Manager — Zone of proximal development
# ---------------------------------------------------------------------------

class AdvancedCurriculumManager:
    """
    Determines the optimal difficulty level using:
    1. Skill tree depth — how many layers of prerequisite skills exist
    2. KG density — facts/entity ratio measures understanding connectedness
    3. Prediction error trend — plateau detection (stopped learning → go harder)

    Increases difficulty when: deep skills, dense KG, and flat error curve.
    Decreases difficulty when: error is rising fast (overwhelmed).
    """

    def __init__(
        self,
        max_difficulty: int = 10,
        pred_error_window: int = 50,
        density_threshold: float = 2.0,
        stable_error_threshold: float = 0.01,
    ):
        self.max_difficulty = max_difficulty
        self.pred_error_window = pred_error_window
        self.density_threshold = density_threshold
        self.stable_error_threshold = stable_error_threshold

        self.prediction_error_history: deque = deque(maxlen=pred_error_window)

    def record_prediction_error(self, error: float) -> None:
        self.prediction_error_history.append(error)

    def recommend_difficulty(
        self,
        skill_bank: SkillBank,
        knowledge_graph: InMemoryKnowledgeGraph,
        current_difficulty: int,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Recommend difficulty level. Returns (level, signal_details).
        """
        signals = {}

        tree_depth = self._get_skill_tree_depth(skill_bank)
        signals["skill_tree_depth"] = float(tree_depth)

        kg_density = self._get_kg_density(knowledge_graph)
        signals["kg_density"] = kg_density

        pred_slope, pred_level = self._get_pred_error_trend()
        signals["pred_error_slope"] = pred_slope
        signals["pred_error_level"] = pred_level

        should_increase = (
            tree_depth >= current_difficulty
            and kg_density >= self.density_threshold
            and pred_slope >= -self.stable_error_threshold
        )

        should_decrease = (
            pred_slope > self.stable_error_threshold * 5
            and len(self.prediction_error_history) >= 20
        )

        if should_increase and current_difficulty < self.max_difficulty:
            recommended = current_difficulty + 1
        elif should_decrease and current_difficulty > 0:
            recommended = current_difficulty - 1
        else:
            recommended = current_difficulty

        signals["recommended"] = float(recommended)
        signals["should_increase"] = float(should_increase)

        return recommended, signals

    def _get_skill_tree_depth(self, skill_bank: SkillBank) -> int:
        if not skill_bank.skills:
            return 0

        def depth(skill_id: str, visited: set) -> int:
            if skill_id in visited or skill_id not in skill_bank.skills:
                return 0
            visited.add(skill_id)
            skill = skill_bank.skills[skill_id]
            if not skill.prerequisites:
                return 1
            return 1 + max(depth(p, visited) for p in skill.prerequisites)

        return max(depth(sid, set()) for sid in skill_bank.skills)

    def _get_kg_density(self, knowledge_graph: InMemoryKnowledgeGraph) -> float:
        stats = knowledge_graph.get_stats()
        if stats["num_entities"] == 0:
            return 0.0
        return stats["num_facts"] / stats["num_entities"]

    def _get_pred_error_trend(self) -> Tuple[float, float]:
        if len(self.prediction_error_history) < 10:
            return 0.0, 0.0

        errors = np.array(list(self.prediction_error_history))
        x = np.arange(len(errors))
        slope = float(np.polyfit(x, errors, 1)[0])
        level = float(np.mean(errors[-10:]))

        return slope, level


# ---------------------------------------------------------------------------
# Developmental Stage Controller — closed-loop EXPLORE <-> EXPLOIT (-> IMAGINE)
# ---------------------------------------------------------------------------

class DevelopmentalStageController:
    """
    Closed-loop developmental stage controller.

    Replaces the old *open-loop* schedules — a config flag for dream training
    and a wall-clock anneal for the curiosity->task reward mix — with an
    *endogenous* transition driven by the agent's own learning signals. The
    stage advances because the agent's learning state changed, not because a
    timer elapsed.

    Signal roles (deliberately ASYMMETRIC by direction):

      - WM prediction-error (learning progress) is the SPINE. It drives
        transitions in BOTH directions because it is a *leading* indicator:
        novel states spike prediction error immediately, before any skill is
        minted or applied.
      - Skill-acquisition rate is a FORWARD-ONLY confirm. EXPLORE->EXPLOIT
        requires BOTH a flat WM-error slope AND a flat skill rate ("I've
        stopped learning the world AND stopped discovering skills"). We NEVER
        use skill-rate for the reverse transition: when the agent re-enters
        exploration it *applies* skills (application != acquisition), so the
        rate stays flat even when there is genuinely new material. Using it to
        detect re-entry would be lagging and confounded — hence the reverse
        trigger relies on the WM-error spike alone.

    Stages:
      EXPLORE  - curiosity-heavy; PPO drives; world model learns from diverse
                 real data (dream-control OFF). The reward mixer holds intrinsic
                 weight high (anneal paused).
      EXPLOIT  - task-heavy; PPO drives; the reward mixer anneals intrinsic down,
                 re-anchored to the moment of transition (not wall-clock t=0).
      IMAGINE  - (Stage 2, trust-gated, optional) dream-control ON. Only
                 reachable when dream training is enabled in config AND the world
                 model has earned trust (low, stable prediction error). Dormant
                 on cheap/dense envs where dream-control isn't worth it and where
                 the historical "dream-only starves the world model" trap lives.

    Robustness: every edge uses hysteresis (a sustained streak of confirming
    checks) plus a minimum-episode guard so early noise can't fire it. A hard
    timestep backstop forces EXPLORE->EXPLOIT if the plateau never arrives, so a
    never-plateauing run can't stay curious forever.
    """

    EXPLORE = "explore"
    EXPLOIT = "exploit"
    IMAGINE = "imagine"

    def __init__(
        self,
        window: int = 30,
        patience: int = 5,
        min_episodes: int = 50,
        flat_slope_threshold: float = 1e-3,
        skill_rate_threshold: float = 0.02,
        spike_factor: float = 1.5,
        trust_level: float = 0.0,
        force_exploit_timestep: Optional[int] = None,
        force_imagine_timestep: Optional[int] = None,
        dream_enabled: bool = False,
    ):
        self.window = window
        self.patience = patience
        self.min_episodes = min_episodes
        self.flat_slope_threshold = flat_slope_threshold
        self.skill_rate_threshold = skill_rate_threshold
        self.spike_factor = spike_factor
        self.trust_level = trust_level          # <=0 disables the IMAGINE gate
        self.force_exploit_timestep = force_exploit_timestep
        self.force_imagine_timestep = force_imagine_timestep
        self.dream_enabled = dream_enabled

        self.stage = self.EXPLORE
        self.error_history: deque = deque(maxlen=window)
        self.skill_history: deque = deque(maxlen=window)
        self.episodes_seen = 0

        # Hysteresis streak counters (consecutive confirming checks).
        self._plateau_streak = 0
        self._spike_streak = 0
        self._trust_streak = 0

        # WM-error level captured at EXPLORE->EXPLOIT; the reverse "spike" band
        # is measured relative to this learned-good level, so it adapts to the
        # natural error scale of whatever the agent just mastered.
        self._baseline_error: Optional[float] = None

    # -- signal estimators ---------------------------------------------------

    def _slope(self) -> float:
        """Linear-fit slope of WM error over the window. Negative = error still
        dropping = still learning. Returns -inf until enough data so a sparse
        window never reads as 'plateaued'. The requirement is capped at the
        deque's maxlen (=window): with window<30 the old max(10, window//3)
        exceeded what the history could ever hold, so slope stayed -inf forever
        and the IMAGINE trust gate was unreachable (bit the 320k Minecraft run,
        which configured window=6 for its long episodes)."""
        need = min(self.window, max(10, self.window // 3))
        if len(self.error_history) < need:
            return float("-inf")
        errs = np.array(self.error_history, dtype=float)
        x = np.arange(len(errs))
        return float(np.polyfit(x, errs, 1)[0])

    def _level(self) -> float:
        """Recent absolute WM-error level (mean of the last few samples)."""
        if not self.error_history:
            return float("inf")
        n = min(10, len(self.error_history))
        return float(np.mean(list(self.error_history)[-n:]))

    def _skill_rate(self) -> float:
        """New skills per episode across the window. inf until enough data so it
        can't trivially satisfy the 'flat' confirm before we've observed any."""
        if len(self.skill_history) < 2:
            return float("inf")
        span = len(self.skill_history) - 1
        delta = self.skill_history[-1] - self.skill_history[0]
        return delta / max(1, span)

    def _reset_streaks(self) -> None:
        self._plateau_streak = 0
        self._spike_streak = 0
        self._trust_streak = 0

    # -- main update ---------------------------------------------------------

    def update(
        self,
        prediction_error: Optional[float],
        skill_count: int,
        timestep: int,
    ) -> Dict[str, Any]:
        """Feed one episode's signals; advance the stage machine if warranted.

        Returns a dict with the current stage, a ``transitioned`` flag, the
        from/to stages, the raw signals, and a human-readable ``reason``.
        """
        self.episodes_seen += 1
        if prediction_error is not None and np.isfinite(prediction_error):
            self.error_history.append(float(prediction_error))
        self.skill_history.append(int(skill_count))

        prev = self.stage
        slope = self._slope()
        level = self._level()
        srate = self._skill_rate()
        transitioned = False
        reason = ""

        if self.stage == self.EXPLORE:
            # Forward: WM-error plateau (spine) AND skill-rate flat (confirm).
            plateau = (
                self.episodes_seen >= self.min_episodes
                and slope >= -self.flat_slope_threshold
                and srate <= self.skill_rate_threshold
            )
            forced = (
                self.force_exploit_timestep is not None
                and timestep >= self.force_exploit_timestep
            )
            self._plateau_streak = self._plateau_streak + 1 if plateau else 0
            if self._plateau_streak >= self.patience or forced:
                self.stage = self.EXPLOIT
                self._baseline_error = level if np.isfinite(level) else None
                reason = (
                    "wm-plateau+skill-flat"
                    if self._plateau_streak >= self.patience
                    else "forced-backstop"
                )
                self._reset_streaks()
                transitioned = True

        elif self.stage == self.EXPLOIT:
            # Reverse: WM-error spike relative to the learned-good baseline.
            # NEVER skill-rate (application != acquisition keeps it flat).
            spiked = (
                self._baseline_error is not None
                and self._baseline_error > 0
                and level > self._baseline_error * self.spike_factor
            )
            self._spike_streak = self._spike_streak + 1 if spiked else 0

            # Stage-2 trust gate (dream-control), optional + dream-enabled only.
            trusted = (
                self.dream_enabled
                and self.trust_level > 0
                and level <= self.trust_level
                and slope >= -self.flat_slope_threshold
            )
            self._trust_streak = self._trust_streak + 1 if trusted else 0

            # Backstop for long-episode envs (Minecraft: ~1 controller update
            # per 10 wall-minutes): if the gate hasn't fired by this timestep,
            # force dream-control anyway — mirrors force_exploit_timestep.
            forced_imagine = (
                self.dream_enabled
                and self.force_imagine_timestep is not None
                and timestep >= self.force_imagine_timestep
            )

            if self._spike_streak >= self.patience:
                self.stage = self.EXPLORE
                reason = "wm-error-spike (re-entry)"
                self._reset_streaks()
                transitioned = True
            elif self._trust_streak >= self.patience or forced_imagine:
                self.stage = self.IMAGINE
                reason = (
                    "wm-trust (dream-control)"
                    if self._trust_streak >= self.patience
                    else "forced-imagine-backstop"
                )
                self._reset_streaks()
                transitioned = True

        elif self.stage == self.IMAGINE:
            # Trust broken (world surprised the model again) -> back to EXPLORE.
            broken = (
                self.trust_level > 0
                and level > self.trust_level * self.spike_factor
            )
            self._spike_streak = self._spike_streak + 1 if broken else 0
            if self._spike_streak >= self.patience:
                self.stage = self.EXPLORE
                reason = "wm-trust-broken (re-entry)"
                self._reset_streaks()
                transitioned = True

        return {
            "stage": self.stage,
            "transitioned": transitioned,
            "from": prev,
            "to": self.stage,
            "slope": slope,
            "level": level,
            "skill_rate": srate,
            "reason": reason,
        }


# ---------------------------------------------------------------------------
# Skill Selector — Multi-signal skill retrieval
# ---------------------------------------------------------------------------

class SkillSelector:
    """
    Selects relevant prior skills for a goal using three signals:
    1. Cosine similarity of goal/skill context embeddings (neural)
    2. Knowledge graph fact overlap (symbolic)
    3. Prerequisite satisfaction (structural)

    Skills with unmet prerequisites are filtered out. The remaining are
    ranked by a weighted combination of neural and symbolic scores.
    """

    def select_skills(
        self,
        goal_embedding: Optional[np.ndarray],
        skill_bank: SkillBank,
        knowledge_graph: InMemoryKnowledgeGraph,
        max_skills: Optional[int] = None,
    ) -> List[Tuple[Skill, float]]:
        """
        Returns list of (Skill, relevance_score) sorted by relevance.

        `max_skills` defaults to the bank's configured
        `skill_bank.max_skills_loaded` rather than a local constant, so the
        config key governs every retrieval path (getattr-guarded: test stubs
        and hand-built banks predate the attribute).

        Mastery is a MILESTONE, not an availability gate (user decision,
        July 2026): every stored skill is a candidate, weighted by its
        competence. A 0.3-competence skill is a real node that pulls less —
        not an invisible one. (is_mastered used to be hardcoded True at save
        time, so the old get_mastered_skills() filter passed everything by
        accident; now that the flag is honest, filtering on it would have
        silently emptied selection.)
        """
        if max_skills is None:
            max_skills = int(getattr(skill_bank, "max_skills_loaded", 3))
        max_skills = max(1, int(max_skills))

        candidates = list(skill_bank.skills.values())
        if not candidates:
            return []

        scored = []
        for skill in candidates:
            neural_score = self._cosine_score(goal_embedding, skill)
            graph_score = self._graph_overlap_score(skill, knowledge_graph)
            prereq_ok = self._prerequisites_met(skill, skill_bank)

            if not prereq_ok:
                continue

            relevance = 0.6 * neural_score + 0.4 * graph_score
            # competence scales influence: floor 0.3 keeps weak-but-relevant
            # skills discoverable instead of invisible
            combined = relevance * (0.3 + 0.7 * float(skill.success_rate))
            scored.append((skill, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_skills]

    def _cosine_score(
        self, goal_embedding: Optional[np.ndarray], skill: Skill
    ) -> float:
        if goal_embedding is None or skill.context_embedding is None:
            return 0.0

        skill_emb = np.array(skill.context_embedding)
        # Dimension-incomparable embeddings (e.g. a 16-d legacy glue goal vs
        # a 50-d achievement-slot embedding from a different goal space) are
        # NOT retrievable by this goal — score 0, don't crash (this raised
        # whenever a persisted bank held skills from another goal space).
        if goal_embedding.shape != skill_emb.shape:
            return 0.0
        dot = np.dot(goal_embedding, skill_emb)
        norm = np.linalg.norm(goal_embedding) * np.linalg.norm(skill_emb) + 1e-8
        return float(np.clip((dot / norm + 1) / 2, 0, 1))

    def _graph_overlap_score(
        self,
        skill: Skill,
        knowledge_graph: InMemoryKnowledgeGraph,
    ) -> float:
        if not skill.goal_facts:
            return 0.5

        matches = 0
        for fact_dict in skill.goal_facts:
            results = knowledge_graph.query(
                subject=fact_dict.get("subject"),
                relation=fact_dict.get("relation"),
            )
            if results:
                matches += 1

        return matches / len(skill.goal_facts)

    def _prerequisites_met(
        self, skill: Skill, skill_bank: SkillBank
    ) -> bool:
        for prereq_id in skill.prerequisites:
            if prereq_id not in skill_bank.skills:
                return False
            if not skill_bank.skills[prereq_id].is_mastered:
                return False
        return True


# ---------------------------------------------------------------------------
# GlueLayer — Orchestrator for all glue components
# ---------------------------------------------------------------------------

class GlueLayer:
    """
    The complete glue layer connecting neural world model to symbolic reasoning.

    Orchestrates all five sub-components and provides the interface that
    the DevelopmentalAI loop calls during training and inference.

    Usage in the loop:
      - Episode start: generate_goal() → select_skills()
      - Each step: augment_latent() to blend knowledge into latent states
      - After training: record_training_signals() for mastery/curriculum
      - Periodically: update_knowledge() when KG embeddings retrain
      - After episode: detect_mastery(), recommend_difficulty()
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        embedding_dim: int = 64,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 64,
        goal_embedding_dim: int = 16,
        max_difficulty: int = 10,
        device: torch.device = torch.device("cpu"),
        reward_target: float = 475.0,
        reward_floor: float = 0.0,
    ):
        self.knowledge_integrator = KnowledgeIntegrator(
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            device=device,
        )

        self.goal_generator = GoalGenerator(
            obs_dim=obs_dim,
            embedding_dim=goal_embedding_dim,
        )

        self.goal_manager = HierarchicalGoalManager(
            obs_dim=obs_dim,
            embedding_dim=goal_embedding_dim,
        )

        self.mastery_detector = AdvancedMasteryDetector(
            reward_target=reward_target,
            reward_floor=reward_floor,
        )

        # Env-relative reward band — used to set an *achievable* completion
        # threshold for consolidation goals during EXPLOIT (see generate_goal).
        self.reward_target = reward_target
        self.reward_floor = reward_floor

        self.curriculum_manager = AdvancedCurriculumManager(
            max_difficulty=max_difficulty,
        )

        self.skill_selector = SkillSelector()

    def update_knowledge(
        self,
        kg_embedder: KGEmbeddingTrainer,
        knowledge_graph: InMemoryKnowledgeGraph,
    ) -> Optional[torch.Tensor]:
        return self.knowledge_integrator.update_knowledge_vector(
            kg_embedder, knowledge_graph
        )

    def augment_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.knowledge_integrator.augment_latent(latent)

    def generate_goal(
        self,
        knowledge_graph: InMemoryKnowledgeGraph,
        symbolic_decoder=None,
        skill_bank: Optional[SkillBank] = None,
        llm_subgoals: Optional[List[Dict[str, Any]]] = None,
        stage: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate a goal, using hierarchical decomposition when possible.

        If the goal manager has an active sub-goal, returns that. Otherwise
        generates a new root goal via GoalGenerator and attempts to decompose
        it into sub-goals using skills, KG, or LLM suggestions.

        Stage-gated (developmental):
          - EXPLORE  -> a *frontier* goal (chase novelty), threshold 0.0.
          - EXPLOIT  -> a *competence* goal whose completion threshold is an
                        achievable, env-relative reward. A converged policy
                        can't grow novelty, so a frontier goal there is
                        unsatisfiable and just burns the patience budget; a
                        competence goal completes when the agent performs well.
        """
        # If there's an active sub-goal, keep pursuing it
        if self.goal_manager.current_embedding is not None:
            return self.goal_manager.current_embedding

        # Generate a fresh root goal
        root = self.goal_generator.generate_goal(
            knowledge_graph,
            symbolic_decoder,
            scene_history=knowledge_graph.scene_history,
        )

        exploiting = stage in (
            DevelopmentalStageController.EXPLOIT,
            DevelopmentalStageController.IMAGINE,
        )
        goal_n = len(self.goal_generator.goal_history)
        if exploiting:
            band = self.reward_target - self.reward_floor
            completion_threshold = self.reward_floor + 0.7 * band
            description = f"Consolidate competence (goal #{goal_n})"
        else:
            completion_threshold = 0.0
            description = f"Explore frontier (goal #{goal_n})"

        self.goal_manager.set_root_goal(
            embedding=root,
            description=description,
            completion_reward_threshold=completion_threshold,
        )

        # Decompose ONLY while exploring. Decomposition breaks a frontier goal
        # into explorable chunks (cluster / skill / LLM sub-goals) — that is an
        # exploration tool. In EXPLOIT the objective is unitary (reach a
        # competent return), so we pursue the single achievable consolidation
        # goal and never push sub-goals a converged policy can't satisfy
        # (including LLM-suggested exploration sub-goals, which carry a 0.0
        # threshold that is unreachable on a negative-reward env).
        if skill_bank is not None and not exploiting:
            self.goal_manager.decompose_goal(
                skill_bank, knowledge_graph, llm_subgoals, stage=stage
            )

        current = self.goal_manager.current_embedding
        if current is not None:
            return current
        return root

    def step_goal(self, episode_reward: float) -> Dict[str, Any]:
        """Update the hierarchical goal manager after an episode."""
        return self.goal_manager.step(episode_reward)

    def select_skills(
        self,
        goal_embedding: Optional[np.ndarray],
        skill_bank: SkillBank,
        knowledge_graph: InMemoryKnowledgeGraph,
        max_skills: Optional[int] = None,
    ) -> List[Tuple[Skill, float]]:
        # None flows through to SkillSelector, which resolves it from the
        # bank's configured max_skills_loaded.
        return self.skill_selector.select_skills(
            goal_embedding, skill_bank, knowledge_graph, max_skills,
        )

    def detect_mastery(
        self,
        knowledge_graph: InMemoryKnowledgeGraph,
    ) -> Tuple[bool, Dict[str, float]]:
        return self.mastery_detector.detect_mastery(knowledge_graph)

    def recommend_difficulty(
        self,
        skill_bank: SkillBank,
        knowledge_graph: InMemoryKnowledgeGraph,
        current_difficulty: int,
    ) -> Tuple[int, Dict[str, float]]:
        return self.curriculum_manager.recommend_difficulty(
            skill_bank, knowledge_graph, current_difficulty,
        )

    def record_training_signals(
        self,
        kl_value: Optional[float] = None,
        prediction_error: Optional[float] = None,
        episode_reward: Optional[float] = None,
    ) -> None:
        if kl_value is not None:
            self.mastery_detector.record_kl(kl_value)
        if prediction_error is not None:
            self.curriculum_manager.record_prediction_error(prediction_error)
        if episode_reward is not None:
            self.mastery_detector.record_reward(episode_reward)

    @property
    def stats(self) -> Dict[str, Any]:
        gm = self.goal_manager.stats
        return {
            "has_knowledge_vector": self.knowledge_integrator.has_knowledge,
            "goals_generated": len(self.goal_generator.goal_history),
            "kl_samples": len(self.mastery_detector.kl_history),
            "reward_samples": len(self.mastery_detector.reward_history),
            "pred_error_samples": len(
                self.curriculum_manager.prediction_error_history
            ),
            "goal_stack_depth": gm["stack_depth"],
            "completed_goals": gm["completed_goals"],
            "failed_goals": gm["failed_goals"],
        }
