"""
Symbolic Decoder — Latent State → Discrete Symbolic Categories
================================================================
A dedicated neural network head that maps the RSSM's latent state
directly into probability distributions over symbolic categories,
bypassing the continuous observation decoder entirely.

Why not just decode to observations and then discretize?
  - The observation decoder was trained for reconstruction, not classification.
    Small numerical errors (0.499 vs 0.501) can flip bin boundaries.
  - The symbolic decoder learns SHARP decision boundaries in latent space
    because it's trained with cross-entropy (a classification loss).
  - Softmax outputs give us confidence for free — if P(medium) = 0.95,
    that's a high-confidence fact; if P(medium) = 0.4, skip it.

Architecture:
  Input:  RSSM latent vector [h; z] — shape (batch, latent_dim)
  Output: For each observation dimension, a probability distribution
          over num_categories symbolic bins.

  Example for CartPole (obs_dim=4, num_categories=5):
    Input:  (batch, 1536)      — the full latent state
    Output: (batch, 4, 5)      — 4 distributions, each over 5 categories
      cart_position:     [P(very_low), P(low), P(medium), P(high), P(very_high)]
      cart_velocity:     [P(very_low), P(low), P(medium), P(high), P(very_high)]
      pole_angle:        [P(very_low), P(low), P(medium), P(high), P(very_high)]
      pole_angular_vel:  [P(very_low), P(low), P(medium), P(high), P(very_high)]

Training:
  Supervised — during the normal training loop, when we have both the
  latent state AND the real observation, we discretize the real observation
  into category indices and train the decoder with cross-entropy loss.
  This piggybacks on the world model training loop at near-zero extra cost.

Reference:
  - NeSyS (2026): symbolic model constrains neural predictions
  - Curiosity-Driven Imagination (Lorang 2025): symbolic transitions
    guide ICM toward structurally novel experiences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discretizer — converts continuous observations to category indices
# ---------------------------------------------------------------------------

class ObservationDiscretizer:
    """
    Converts continuous observation values to discrete category indices.

    Maintains running min/max statistics per dimension and assigns each
    value to one of num_categories bins. These bin indices become the
    training targets for the symbolic decoder.

    The discretizer adapts as it sees more data — early bins may be
    inaccurate, but they stabilize after a few hundred observations.
    """

    def __init__(
        self,
        obs_dim: int,
        num_categories: int = 5,
        obs_labels: Optional[List[str]] = None,
    ):
        self.obs_dim = obs_dim
        self.num_categories = num_categories
        self.obs_labels = obs_labels or [f"dim_{i}" for i in range(obs_dim)]

        # Category names for human-readable output
        if num_categories <= 3:
            self.category_names = ["low", "medium", "high"]
        elif num_categories <= 5:
            self.category_names = ["very_low", "low", "medium", "high", "very_high"]
        else:
            self.category_names = [f"level_{i}" for i in range(num_categories)]

        # Running statistics per dimension (Welford's online algorithm)
        self.obs_min = None   # shape (obs_dim,)
        self.obs_max = None   # shape (obs_dim,)
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_m2 = np.zeros(obs_dim, dtype=np.float64)
        self.obs_count = 0

    def update(self, obs: np.ndarray) -> None:
        """Update running statistics with a new observation."""
        obs = np.asarray(obs, dtype=np.float64)
        self.obs_count += 1

        if self.obs_min is None:
            self.obs_min = obs.copy()
            self.obs_max = obs.copy()
        else:
            self.obs_min = np.minimum(self.obs_min, obs)
            self.obs_max = np.maximum(self.obs_max, obs)

        # Welford's algorithm for online mean/variance
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_m2 += delta * delta2

    def discretize(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert a continuous observation to category indices.

        Args:
            obs: shape (obs_dim,) continuous observation

        Returns:
            shape (obs_dim,) integer array of category indices in [0, num_categories)
        """
        if self.obs_min is None or self.obs_count < 2:
            # Not enough data — assign everything to the middle bin
            return np.full(self.obs_dim, self.num_categories // 2, dtype=np.int64)

        obs = np.asarray(obs, dtype=np.float64)
        indices = np.zeros(self.obs_dim, dtype=np.int64)

        for i in range(self.obs_dim):
            vmin = self.obs_min[i]
            vmax = self.obs_max[i]

            if vmax - vmin < 1e-8:
                indices[i] = self.num_categories // 2
                continue

            # Normalize to [0, 1] then map to bin index
            normalized = np.clip((obs[i] - vmin) / (vmax - vmin), 0.0, 1.0)
            indices[i] = min(int(normalized * self.num_categories), self.num_categories - 1)

        return indices

    def discretize_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Discretize a batch of observations.

        Args:
            obs_batch: shape (batch, obs_dim)

        Returns:
            shape (batch, obs_dim) integer category indices
        """
        batch_size = obs_batch.shape[0]
        result = np.zeros((batch_size, self.obs_dim), dtype=np.int64)
        for b in range(batch_size):
            result[b] = self.discretize(obs_batch[b])
        return result

    def index_to_label(self, dim_idx: int, cat_idx: int) -> Tuple[str, str]:
        """Convert a (dimension_index, category_index) pair to human-readable labels."""
        dim_name = self.obs_labels[dim_idx] if dim_idx < len(self.obs_labels) else f"dim_{dim_idx}"
        cat_name = self.category_names[cat_idx] if cat_idx < len(self.category_names) else f"level_{cat_idx}"
        return dim_name, cat_name


# ---------------------------------------------------------------------------
# Symbolic Decoder Network
# ---------------------------------------------------------------------------

class SymbolicDecoder(nn.Module):
    """
    Neural network that maps RSSM latent states to symbolic category
    probability distributions.

    Instead of:  latent → decoder → continuous obs → discretize → symbol
    This does:   latent → symbolic_decoder → P(category) → symbol

    The softmax output gives confidence for free:
      P(medium) = 0.95  →  high-confidence fact, store it
      P(medium) = 0.35  →  uncertain, skip or mark low-confidence

    Trained with cross-entropy against discretized real observations.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        num_categories: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.num_categories = num_categories

        # Shared feature extraction from latent state
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Per-dimension classification heads
        # Each head outputs logits for num_categories classes
        # Using separate heads lets each dimension learn its own
        # decision boundaries independently
        self.dim_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_categories)
            for _ in range(obs_dim)
        ])

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Map latent state to category logits for each observation dimension.

        Args:
            latent: shape (batch, latent_dim) — the [h; z] RSSM state

        Returns:
            logits: shape (batch, obs_dim, num_categories)
                    Raw logits (pre-softmax) for each dimension's categories
        """
        shared_features = self.shared_net(latent)

        # Stack outputs from each dimension head
        logits_list = [head(shared_features) for head in self.dim_heads]
        logits = torch.stack(logits_list, dim=1)  # (batch, obs_dim, num_categories)

        return logits

    def predict_categories(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the most likely category and its confidence for each dimension.

        Args:
            latent: shape (batch, latent_dim)

        Returns:
            categories: shape (batch, obs_dim) — predicted category indices
            confidences: shape (batch, obs_dim) — confidence (max probability)
        """
        logits = self.forward(latent)
        probs = F.softmax(logits, dim=-1)  # (batch, obs_dim, num_categories)

        confidences, categories = probs.max(dim=-1)  # both (batch, obs_dim)

        return categories, confidences

    def compute_loss(
        self,
        latent: torch.Tensor,
        target_categories: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss against discretized real observations.

        Args:
            latent: shape (batch, latent_dim)
            target_categories: shape (batch, obs_dim) — integer indices
                              from the ObservationDiscretizer

        Returns:
            Scalar loss value
        """
        logits = self.forward(latent)  # (batch, obs_dim, num_categories)

        # Reshape for cross_entropy: (batch * obs_dim, num_categories) vs (batch * obs_dim,)
        batch_size = logits.shape[0]
        logits_flat = logits.view(-1, self.num_categories)
        targets_flat = target_categories.view(-1)

        return F.cross_entropy(logits_flat, targets_flat)


# ---------------------------------------------------------------------------
# Symbolic Fact Producer — converts decoder output to SymbolicFact objects
# ---------------------------------------------------------------------------

class SymbolicFactProducer:
    """
    Converts SymbolicDecoder output into SymbolicFact objects for the
    knowledge graph.

    This is the final step in the latent → symbol pipeline:
      RSSM latent → SymbolicDecoder → (category, confidence) → SymbolicFact

    Only produces facts above the confidence threshold to avoid polluting
    the knowledge graph with uncertain information.
    """

    def __init__(
        self,
        discretizer: ObservationDiscretizer,
        confidence_threshold: float = 0.6,
    ):
        self.discretizer = discretizer
        self.confidence_threshold = confidence_threshold

    def produce_facts(
        self,
        categories: torch.Tensor,
        confidences: torch.Tensor,
        timestep: int = 0,
    ) -> List:
        """
        Convert predicted categories + confidences into SymbolicFact objects.

        Args:
            categories: shape (obs_dim,) — predicted category index per dimension
            confidences: shape (obs_dim,) — probability of predicted category

        Returns:
            List of SymbolicFact objects (only those above confidence threshold)
        """
        # Import here to avoid circular dependency
        from developmental_ai.knowledge_graph.knowledge_graph import SymbolicFact

        facts = []
        cats_np = categories.detach().cpu().numpy()
        confs_np = confidences.detach().cpu().numpy()

        for dim_idx in range(len(cats_np)):
            confidence = float(confs_np[dim_idx])

            # Skip low-confidence predictions
            if confidence < self.confidence_threshold:
                continue

            dim_name, cat_name = self.discretizer.index_to_label(
                dim_idx, int(cats_np[dim_idx])
            )

            fact = SymbolicFact(
                subject=dim_name,
                relation="has_value",
                obj=cat_name,
                confidence=confidence,
                source="symbolic_decoder",  # distinguishes from raw observation facts
                timestamp=timestep,
            )
            facts.append(fact)

        return facts


# ---------------------------------------------------------------------------
# Integrated Symbolic Decoder Manager
# ---------------------------------------------------------------------------

class SymbolicDecoderManager:
    """
    Manages the full symbolic decoding pipeline:
      1. Tracks observation statistics (ObservationDiscretizer)
      2. Trains the SymbolicDecoder network alongside the world model
      3. Produces SymbolicFact objects from RSSM latent states

    This is the concrete implementation of "Approach 2" from the glue layer.
    It plugs into the DevelopmentalAI loop at two points:

      Training: After the world model processes a sequence, the symbolic
        decoder trains on the same latent states using discretized
        observations as targets. Piggybacks on world model training.

      Inference: Each step, the RSSM latent state is passed through the
        symbolic decoder to produce high-confidence symbolic facts that
        go into the knowledge graph.

    Usage:
        manager = SymbolicDecoderManager(latent_dim=1536, obs_dim=4)

        # During training (latent states come from world model observe_sequence):
        loss = manager.train_step(latent_states, raw_observations)

        # During inference (produce facts from current latent state):
        facts = manager.extract_facts(latent_state, timestep=1000)
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        num_categories: int = 5,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        confidence_threshold: float = 0.6,
        obs_labels: Optional[List[str]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.num_categories = num_categories

        # Discretizer tracks observation statistics and produces training targets
        self.discretizer = ObservationDiscretizer(
            obs_dim=obs_dim,
            num_categories=num_categories,
            obs_labels=obs_labels,
        )

        # Neural network: latent → category probabilities
        self.decoder = SymbolicDecoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            num_categories=num_categories,
            hidden_dim=hidden_dim,
        ).to(device)

        # Optimizer (separate from world model optimizer)
        self.optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=learning_rate
        )

        # Fact producer: converts decoder output → SymbolicFact objects
        self.fact_producer = SymbolicFactProducer(
            discretizer=self.discretizer,
            confidence_threshold=confidence_threshold,
        )

        # Training statistics
        self.train_steps = 0
        self.total_loss = 0.0
        self.avg_confidence = 0.0
        self.avg_accuracy = 0.0

    def update_discretizer(self, obs: np.ndarray) -> None:
        """
        Feed a raw observation to the discretizer to update its statistics.

        Call this every step so the discretizer adapts its bin boundaries
        as it sees more data. Early bins may be inaccurate, but they
        stabilize after ~100-200 observations.
        """
        self.discretizer.update(obs)

    def train_step(
        self,
        latent_states: torch.Tensor,
        raw_observations: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Train the symbolic decoder on a batch of (latent, observation) pairs.

        Called during world model training — we reuse the same latent states
        the RSSM just computed, and the raw observations are already available
        in the replay buffer batch. Near-zero extra compute cost.

        Args:
            latent_states: shape (batch, seq_len, latent_dim) or (batch*seq_len, latent_dim)
                          Latent states from world model's observe_sequence
            raw_observations: shape matching latent_states but with obs_dim
                             The real observations these latent states encode

        Returns:
            Dict with loss, accuracy, and avg_confidence
        """
        self.decoder.train()

        # Flatten sequence dimension if present
        if latent_states.dim() == 3:
            batch_size, seq_len, latent_dim = latent_states.shape
            latent_flat = latent_states.reshape(-1, latent_dim)
            obs_flat = raw_observations.reshape(-1, self.obs_dim)
        else:
            latent_flat = latent_states
            obs_flat = raw_observations

        # Update discretizer with all observations in this batch
        obs_np = obs_flat.detach().cpu().numpy()
        for i in range(obs_np.shape[0]):
            self.discretizer.update(obs_np[i])

        # Create training targets: discretize real observations → category indices
        target_indices = self.discretizer.discretize_batch(obs_np)
        target_tensor = torch.from_numpy(target_indices).long().to(self.device)

        # Forward pass and loss
        loss = self.decoder.compute_loss(latent_flat.detach(), target_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Compute accuracy for logging
        with torch.no_grad():
            pred_cats, pred_confs = self.decoder.predict_categories(latent_flat.detach())
            accuracy = (pred_cats == target_tensor).float().mean().item()
            avg_conf = pred_confs.mean().item()

        self.train_steps += 1
        self.total_loss = loss.item()
        self.avg_confidence = avg_conf
        self.avg_accuracy = accuracy

        return {
            "symbolic_decoder_loss": loss.item(),
            "symbolic_decoder_accuracy": accuracy,
            "symbolic_decoder_confidence": avg_conf,
        }

    def extract_facts(
        self,
        latent_state: torch.Tensor,
        timestep: int = 0,
    ) -> List:
        """
        Extract symbolic facts from an RSSM latent state.

        This is the core inference method — called each step during the
        developmental loop to produce symbolic facts from the current
        latent state.

        Args:
            latent_state: shape (latent_dim,) or (1, latent_dim) — single latent state
            timestep: Current timestep for the fact's timestamp

        Returns:
            List of SymbolicFact objects (only high-confidence ones)
        """
        self.decoder.eval()

        with torch.no_grad():
            if latent_state.dim() == 1:
                latent_state = latent_state.unsqueeze(0)

            categories, confidences = self.decoder.predict_categories(latent_state)

            # Produce facts from the first (and typically only) batch element
            facts = self.fact_producer.produce_facts(
                categories[0], confidences[0], timestep=timestep,
            )

        return facts

    @property
    def stats(self) -> Dict[str, float]:
        """Current training statistics for logging."""
        return {
            "symbolic_decoder_loss": self.total_loss,
            "symbolic_decoder_accuracy": self.avg_accuracy,
            "symbolic_decoder_confidence": self.avg_confidence,
            "symbolic_decoder_train_steps": self.train_steps,
            "discretizer_obs_count": self.discretizer.obs_count,
        }
