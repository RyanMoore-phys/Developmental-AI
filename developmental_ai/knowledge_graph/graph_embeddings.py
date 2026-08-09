"""
Graph Embeddings — PyKEEN + PyTorch Geometric Integration
==========================================================
Converts the symbolic knowledge graph into neural vectors that the
world model (RSSM) can process. This is the bridge between:
  - Symbolic layer: discrete facts like "push → causes → movement"
  - Neural layer: continuous latent vectors the RSSM operates on

Two complementary tools:
  1. PyKEEN: Trains knowledge graph embeddings (TransE, RotatE, etc.)
     These embed entities and relations into vector space where:
     subject_vec + relation_vec ≈ object_vec (for TransE)

  2. PyTorch Geometric: Runs Graph Neural Networks (GNNs) over the
     knowledge graph structure. GNNs learn patterns OVER graph topology,
     not just individual nodes — they capture multi-hop reasoning.

The embeddings are fed back into the world model as structured knowledge,
allowing the RSSM to benefit from symbolic reasoning without losing
the flexibility of neural computation.

Key insight from NeSyS (2026): the symbolic embeddings should CONSTRAIN
the neural model, not just provide additional input. We implement this
via a gating mechanism that lets the model learn when to trust symbolic
vs. neural predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

# PyKEEN is optional but recommended
try:
    from pykeen.pipeline import pipeline as pykeen_pipeline
    from pykeen.triples import TriplesFactory
    PYKEEN_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    # PyKEEN can fail on Python 3.9 due to typing compatibility issues
    PYKEEN_AVAILABLE = False
    logger.warning(f"PyKEEN not available ({type(e).__name__}). Using fallback embeddings.")

# PyTorch Geometric is optional
try:
    import torch_geometric
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import GCNConv, GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("PyTorch Geometric not installed. GNN features disabled.")


# ---------------------------------------------------------------------------
# PyKEEN Graph Embedding Trainer
# ---------------------------------------------------------------------------

class KGEmbeddingTrainer:
    """
    Trains knowledge graph embeddings using PyKEEN.

    Takes symbolic triples (subject, relation, object) and learns vector
    representations where geometric relationships capture semantic meaning.

    For example, with TransE:
      embedding("cart") + embedding("has_velocity") ≈ embedding("positive")

    These embeddings are then fed to the world model as additional input,
    providing structured symbolic knowledge in a format neural networks
    can process.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        model_name: str = "TransE",
        num_epochs: int = 100,
        device: str = "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.device = device

        # Trained model (set after training)
        self.model = None
        self.triples_factory = None

        # Entity/relation to index mappings
        self.entity_to_id: Dict[str, int] = {}
        self.relation_to_id: Dict[str, int] = {}

    def train(self, triples: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """
        Train KG embeddings on the current set of triples.

        Args:
            triples: List of (subject, relation, object) string tuples

        Returns:
            Training metrics (loss, hits@k, etc.)
        """
        if not PYKEEN_AVAILABLE:
            logger.info("PyKEEN not available, using random embeddings as fallback")
            return self._train_fallback(triples)

        if len(triples) < 3:
            logger.warning("Need at least 3 triples to train embeddings")
            return {"status": "insufficient_data"}

        # Convert to PyKEEN format
        triples_array = np.array(triples, dtype=str)
        self.triples_factory = TriplesFactory.from_labeled_triples(triples_array)

        # Store mappings for later lookup
        self.entity_to_id = dict(self.triples_factory.entity_to_id)
        self.relation_to_id = dict(self.triples_factory.relation_to_id)

        # Train embedding model.
        # NOTE: PyKEEN's pipeline() requires BOTH a training and a testing
        # triples factory — newer versions raise
        #   ValueError: Must specify either dataset or both training/testing
        #   triples factories
        # if only `training=` is given. We just want the learned embeddings,
        # not a held-out evaluation, so we reuse the same factory for testing.
        # Passing `testing=` is also valid on older PyKEEN, so this is
        # backward-compatible. A fixed random_seed also silences PyKEEN's
        # "No random seed is specified" warning and makes runs reproducible.
        result = pykeen_pipeline(
            training=self.triples_factory,
            testing=self.triples_factory,
            model=self.model_name,
            model_kwargs={"embedding_dim": self.embedding_dim},
            training_kwargs={
                "num_epochs": self.num_epochs,
                "use_tqdm": False,
            },
            random_seed=0,
            device=self.device,
        )

        self.model = result.model

        return {
            "status": "trained",
            "num_entities": len(self.entity_to_id),
            "num_relations": len(self.relation_to_id),
            "num_triples": len(triples),
        }

    def _train_fallback(self, triples: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """Fallback: create random embeddings when PyKEEN is not available."""
        entities = set()
        relations = set()
        for s, r, o in triples:
            entities.add(s)
            entities.add(o)
            relations.add(r)

        self.entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation_to_id = {r: i for i, r in enumerate(sorted(relations))}

        return {
            "status": "fallback_random",
            "num_entities": len(self.entity_to_id),
            "num_relations": len(self.relation_to_id),
        }

    def get_entity_embedding(self, entity: str) -> Optional[torch.Tensor]:
        """Get the learned embedding vector for a specific entity."""
        if entity not in self.entity_to_id:
            return None

        if self.model is not None and PYKEEN_AVAILABLE:
            idx = self.entity_to_id[entity]
            with torch.no_grad():
                # Access entity representations from the PyKEEN model
                entity_repr = self.model.entity_representations[0]
                return entity_repr(torch.tensor([idx])).squeeze(0)

        # Fallback: return a random but consistent embedding
        idx = self.entity_to_id[entity]
        rng = np.random.RandomState(idx)
        return torch.from_numpy(rng.randn(self.embedding_dim).astype(np.float32))

    def get_all_entity_embeddings(self) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Get embeddings for all entities.

        Returns:
            entity_to_id: Mapping from entity name to index
            embeddings: Tensor of shape (num_entities, embedding_dim)
        """
        num_entities = len(self.entity_to_id)
        if num_entities == 0:
            return {}, torch.zeros(0, self.embedding_dim)

        embeddings = torch.zeros(num_entities, self.embedding_dim)
        for entity, idx in self.entity_to_id.items():
            emb = self.get_entity_embedding(entity)
            if emb is not None:
                embeddings[idx] = emb

        return self.entity_to_id, embeddings

    def save(self, dirpath: str) -> None:
        """Save the trained embeddings to disk."""
        os.makedirs(dirpath, exist_ok=True)
        import json

        # Save mappings
        with open(os.path.join(dirpath, "entity_to_id.json"), "w") as f:
            json.dump(self.entity_to_id, f)
        with open(os.path.join(dirpath, "relation_to_id.json"), "w") as f:
            json.dump(self.relation_to_id, f)

        # Save embeddings as tensor
        _, embeddings = self.get_all_entity_embeddings()
        torch.save(embeddings, os.path.join(dirpath, "entity_embeddings.pt"))

    def load(self, dirpath: str) -> None:
        """Load previously trained embeddings from disk."""
        import json

        with open(os.path.join(dirpath, "entity_to_id.json"), "r") as f:
            self.entity_to_id = json.load(f)
        with open(os.path.join(dirpath, "relation_to_id.json"), "r") as f:
            self.relation_to_id = json.load(f)


# ---------------------------------------------------------------------------
# GNN Bridge — PyTorch Geometric integration
# ---------------------------------------------------------------------------

class KnowledgeGraphGNN(nn.Module):
    """
    Graph Neural Network that learns over the knowledge graph structure.

    While PyKEEN embeddings capture individual entity/relation semantics,
    the GNN captures STRUCTURAL patterns — multi-hop reasoning, neighborhood
    effects, and graph-level features.

    Architecture:
      1. GCN layers process the graph structure
      2. Global pooling aggregates node features into a single vector
      3. This vector is fed to the world model as structured knowledge

    The output is a fixed-size "knowledge vector" that summarizes
    what the knowledge graph knows, in a format the RSSM can use.
    """

    def __init__(
        self,
        input_dim: int = 64,    # Entity embedding dimension
        hidden_dim: int = 128,
        output_dim: int = 64,   # Knowledge vector dimension
        num_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Fallback MLP is always available (used when no edge_index is provided)
        self.fallback_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        if TORCH_GEOMETRIC_AVAILABLE:
            # GCN layers for processing graph structure
            self.conv_layers = nn.ModuleList()
            self.conv_layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            if num_layers > 1:
                self.conv_layers.append(GCNConv(hidden_dim, output_dim))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process the knowledge graph and produce a knowledge vector.

        Args:
            node_features: (num_nodes, input_dim) entity embeddings
            edge_index: (2, num_edges) edge connectivity

        Returns:
            knowledge_vector: (output_dim,) summarizing graph knowledge
        """
        if node_features.shape[0] == 0:
            return torch.zeros(self.output_dim, device=node_features.device)

        if TORCH_GEOMETRIC_AVAILABLE and edge_index is not None:
            x = node_features
            for conv in self.conv_layers:
                x = conv(x, edge_index)
                x = torch.nn.functional.elu(x)

            # Global mean pooling: average all node features
            knowledge_vector = x.mean(dim=0)
        else:
            # Fallback: just average and project
            x = node_features.mean(dim=0, keepdim=True)
            knowledge_vector = self.fallback_net(x).squeeze(0)

        return knowledge_vector

    def build_edge_index(
        self,
        triples: List[Tuple[str, str, str]],
        entity_to_id: Dict[str, int],
    ) -> torch.Tensor:
        """
        Convert triples to edge_index format for PyTorch Geometric.

        Each triple (s, r, o) becomes a directed edge from s to o.
        Relations are encoded as edge attributes (not used in basic GCN).
        """
        src_ids = []
        dst_ids = []

        for s, r, o in triples:
            if s in entity_to_id and o in entity_to_id:
                src_ids.append(entity_to_id[s])
                dst_ids.append(entity_to_id[o])

        if not src_ids:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor([src_ids, dst_ids], dtype=torch.long)


# ---------------------------------------------------------------------------
# Symbolic-Neural Gate
# ---------------------------------------------------------------------------

class SymbolicNeuralGate(nn.Module):
    """
    Gating mechanism that blends symbolic knowledge with neural predictions.

    Inspired by NeSyS (2026): rather than just concatenating symbolic and
    neural representations, this gate LEARNS when to trust each source.

    When the symbolic knowledge is confident (well-tested action rules),
    the gate weighs it more heavily. When the knowledge graph is sparse
    or uncertain, it falls back to the neural model's predictions.

    This prevents catastrophic failures where bad symbolic rules override
    correct neural predictions, while still leveraging symbolic knowledge
    when available.
    """

    def __init__(self, neural_dim: int, symbolic_dim: int, output_dim: int):
        super().__init__()

        # Projects both representations to the same space
        self.neural_proj = nn.Linear(neural_dim, output_dim)
        self.symbolic_proj = nn.Linear(symbolic_dim, output_dim)

        # Gate: learns how much to trust each source
        self.gate_net = nn.Sequential(
            nn.Linear(neural_dim + symbolic_dim, output_dim),
            nn.Sigmoid(),  # Output in [0, 1]: 0 = trust neural, 1 = trust symbolic
        )

    def forward(
        self,
        neural_repr: torch.Tensor,
        symbolic_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend neural and symbolic representations.

        Args:
            neural_repr: From the RSSM world model
            symbolic_repr: From the knowledge graph GNN

        Returns:
            Blended representation that leverages both sources
        """
        # Project to common space
        neural = self.neural_proj(neural_repr)
        symbolic = self.symbolic_proj(symbolic_repr)

        # Compute gate value (how much to trust symbolic)
        gate = self.gate_net(torch.cat([neural_repr, symbolic_repr], dim=-1))

        # Weighted blend
        return gate * symbolic + (1 - gate) * neural
