"""
ICM — Intrinsic Curiosity Module
==================================
Generates curiosity-based intrinsic reward from prediction error.
This is what drives the agent to EXPLORE — no external reward labels needed.

How it works:
  1. Feature encoder: Maps raw observations → compact feature space
     (strips out irrelevant visual details, focuses on controllable aspects)
  2. Forward model: Given current features + action → predict next features
     (tests: "do I understand what my actions do?")
  3. Inverse model: Given current + next features → predict what action was taken
     (ensures the feature space captures action-relevant information)

The FORWARD model prediction error IS the curiosity reward:
  - High error = "I don't understand what happened" = HIGH curiosity = explore more
  - Low error  = "I predicted this correctly" = LOW curiosity  = move on

The INVERSE model is a regularizer — it forces the features to encode things
the agent can actually control (not random noise like TV static).

Reference: Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction" (2017)
  - arxiv: 1705.05363
  - Extended by Curiosity-Driven Imagination (Lorang et al., 2025) which adds symbolic planning

Design insight from H-GRAIL (2025): Use a bandit-based mechanism to balance between
curiosity-driven exploration (novelty seeking) and skill refinement (competence improvement).
We implement a simple version of this via adaptive reward scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class FeatureEncoder(nn.Module):
    """
    Encodes raw observations into a compact feature space.

    The feature space should capture aspects of the environment that are:
    1. Relevant to the agent's actions (not background noise)
    2. Stable enough to learn predictions over
    3. Compact enough that prediction errors are meaningful

    For simple environments (vector observations), this is an MLP.
    For pixel environments, replace with a CNN.
    """

    def __init__(self, obs_dim: int, feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.ELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CNNFeatureEncoder(nn.Module):
    """CNN feature encoder for PIXEL observations (rich-env program, step 8).

    The MLP FeatureEncoder on a flat 12288-d pixel vector burns ~3.1M params
    on its first layer and has no spatial prior — curiosity quality suffers
    exactly where exploration matters. This mirrors the world model's
    CNNEncoder layout (stride-2 convs to a 4x4 bottleneck; 4 stages at 64px,
    5 at 128px, channels capped at 128) and accepts the same FLAT CHW
    [0,1] vectors the pipeline passes everywhere (reshapes internally), so
    it is a drop-in for the MLP.
    """

    def __init__(self, in_channels: int = 3, image_size: int = 64,
                 feature_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        # Stride-2 ladder to a 4x4 bottleneck, channels capped at 128:
        # 64 -> [32,64,128,128] (the original layout, byte-identical);
        # 128 -> [32,64,128,128,128] (one extra stage).
        from developmental_ai.world_model.rssm import _cnn_channel_ladder
        chans = _cnn_channel_ladder(image_size, cap=128)
        layers, c_in = [], in_channels
        for c_out in chans:
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                       nn.ELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        feat = image_size // (2 ** len(chans))  # always 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chans[-1] * feat * feat, feature_dim),
            nn.ELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], self.in_channels,
                           self.image_size, self.image_size)
        return self.head(self.conv(obs))


class ForwardModel(nn.Module):
    """
    Predicts NEXT state features given current features + action taken.

    This is where curiosity comes from: if the forward model can't predict
    what happens after an action, the agent is in unfamiliar territory
    and should explore more.

    The prediction error of this model = intrinsic reward.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, action], dim=-1))


class InverseModel(nn.Module):
    """
    Predicts what ACTION was taken given current and next state features.

    This doesn't directly produce curiosity reward — its role is to REGULARIZE
    the feature encoder. By training the encoder so that an inverse model can
    recover the action, we ensure the features capture action-relevant information
    rather than random noise (which would cause the "noisy TV problem" — the agent
    gets stuck watching unpredictable but uncontrollable noise).
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor, next_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, next_features], dim=-1))


class IntrinsicCuriosityModule(nn.Module):
    """
    Complete ICM that generates curiosity-driven intrinsic rewards.

    Usage in the developmental loop:
        1. Agent observes state, takes action, sees next state
        2. ICM computes prediction error (forward model)
        3. Prediction error → intrinsic reward
        4. Intrinsic reward drives exploration toward novel experiences

    The key insight: high prediction error = the agent encountered something
    it doesn't understand yet = this is where learning happens most.

    Adaptive scaling (inspired by H-GRAIL):
        We track running statistics of curiosity rewards and normalize them.
        This prevents the curiosity signal from overwhelming or being dwarfed by
        extrinsic rewards as the agent's understanding grows.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        forward_loss_weight: float = 0.8,
        inverse_loss_weight: float = 0.2,
        reward_scale: float = 1.0,
        reward_clip: float = 5.0,
        learning_rate: float = 1e-4,
        novelty_window: int = 1000,
        discrete_actions: bool = False,
        pixel_obs: bool = False,
        image_channels: int = 3,
        image_size: int = 64,
    ):
        super().__init__()

        self.forward_loss_weight = forward_loss_weight
        self.inverse_loss_weight = inverse_loss_weight
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.discrete_actions = discrete_actions
        # Stored for subclasses (LearningProgressCuriosity pixel bucketing).
        self.pixel_obs = bool(pixel_obs)
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)

        # Sub-modules (CNN encoder on pixel envs — step 8)
        if pixel_obs:
            self.encoder = CNNFeatureEncoder(
                image_channels, image_size, feature_dim)
        else:
            self.encoder = FeatureEncoder(obs_dim, feature_dim)
        self.forward_model = ForwardModel(feature_dim, action_dim, hidden_dim)
        self.inverse_model = InverseModel(feature_dim, action_dim, hidden_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Running statistics for adaptive reward scaling
        # Tracks recent curiosity values to normalize them
        self.reward_history = deque(maxlen=novelty_window)
        self._running_mean = 0.0
        self._running_median = 0.0
        self.last_pred_error = 0.0
        self._running_std = 1.0

        # Tracks whether the agent is in "exploration" or "exploitation" mode
        # Inspired by H-GRAIL's bandit-based motivation selector
        self.exploration_ratio = 1.0  # Start fully curious, decay as skills develop

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        update_state: bool = True,
    ) -> torch.Tensor:
        """
        Compute curiosity reward for a batch of transitions.

        High prediction error → high reward → agent explores here more.
        Low prediction error → low reward → agent has learned this, move on.

        Args:
            obs: Current observations (batch, obs_dim)
            action: Actions taken (batch, action_dim)
            next_obs: Next observations (batch, obs_dim)
            update_state: when False, the call is READ-ONLY — no running-stat
                update (and, in the LP subclass, no prototype/bucket/history
                mutation). Set False for IMAGINED (dream) transitions so
                decoder reconstructions never pollute the state the waking
                curiosity signal is computed from.

        Returns:
            Intrinsic reward per transition (batch,)
        """
        with torch.no_grad():
            # Encode observations
            features = self.encoder(obs)
            next_features = self.encoder(next_obs)

            # Forward model predicts next features
            predicted_next = self.forward_model(features, action)

            # Prediction error = curiosity
            prediction_error = F.mse_loss(
                predicted_next, next_features, reduction="none"
            ).mean(dim=-1)

            reward = torch.clamp(prediction_error, max=self.reward_clip)
            # RAW forward-model surprise, before clipping/z-normalization and
            # before any LP / imagination / coverage bonus is mixed in.
            # `wm_fidelity` (skill mastery) needs "how well does the world
            # model predict THIS", which is exactly this quantity — the
            # SHAPED reward that leaves this method is a different thing
            # (mean-centred, clipped, and in the LP subclass a derivative),
            # and feeding it to fidelity measured the wrong variable
            # entirely. Detached scalar mean; costs nothing.
            self.last_pred_error = float(prediction_error.mean().item())

            # MEAN-CENTERED z-normalization — deliberately zero-mean, and the
            # July 2026 audit's original "punishes familiar states" complaint
            # is hereby WITHDRAWN for the novelty signal: two attempted
            # non-negative replacements (std-only, then median-clamp-at-zero)
            # both re-taught the agent to farm curiosity by surviving to
            # timeout (measured: ~81 units of farmable intrinsic per episode
            # at weight 0.7 vs task reward 1.0; 5x5 learning collapsed from
            # ~0.74 to ~0.1 frac_solved both times). Zero-mean is the
            # anti-farming mechanism: per-step intrinsic averages ~0, so
            # padding the episode buys nothing. The negative half is not
            # punishment of familiarity per se — it is what makes novelty a
            # RELATIVE signal ("more novel than the recent typical step").
            # The LP subclass overrides this with clamp-at-zero, which is
            # safe THERE because LP is mostly zeros (no farmable mass).
            if update_state:
                self._update_stats(reward)
            if self._running_std > 1e-8:
                reward = (reward - self._running_mean) / (self._running_std + 1e-8)
                reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            # Apply the exploration-ratio scale AFTER normalization so it
            # actually changes the delivered magnitude — a pre-normalization
            # scale cancels out of the division (the old code's H-GRAIL decay
            # was behaviorally inert).
            return reward * self.reward_scale

    def train_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One training step for the ICM.

        Both the forward and inverse models are trained together, but the
        ENCODER is trained ONLY by the inverse loss (Pathak et al. 2017):
        if the forward loss reached the encoder it would reward making
        features trivially predictable — collapsing the representation and
        decaying curiosity everywhere regardless of actual novelty
        (CODE_AUDIT_2026-07.md §H.3).

        Returns:
            Dict of loss values for logging
        """
        self.train()

        # Encode observations
        features = self.encoder(obs)
        next_features = self.encoder(next_obs)

        # Forward model: predict next features from current + action.
        # Detached on BOTH sides so the encoder gets no gradient from it.
        predicted_next = self.forward_model(features.detach(), action)
        forward_loss = F.mse_loss(predicted_next, next_features.detach())

        # Inverse model: predict action from current + next features — this
        # is the loss that shapes the encoder toward action-relevant features.
        predicted_action = self.inverse_model(features, next_features)
        if self.discrete_actions:
            # One-hot actions: cross-entropy over logits is the correct
            # objective (MSE-on-one-hot gives weak, poorly scaled gradients).
            inverse_loss = F.cross_entropy(predicted_action, action.argmax(dim=-1))
        else:
            inverse_loss = F.mse_loss(predicted_action, action)

        # Combined loss
        total_loss = (
            self.forward_loss_weight * forward_loss
            + self.inverse_loss_weight * inverse_loss
        )

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "icm_total": total_loss.item(),
            "icm_forward": forward_loss.item(),
            "icm_inverse": inverse_loss.item(),
        }

    def get_novelty_score(self, obs: torch.Tensor) -> float:
        """
        Estimate how novel a single observation is based on the encoder.

        This is used by the curiosity engine to decide whether to set new goals.
        High novelty → unfamiliar territory → good candidate for exploration.

        Uses the magnitude of the encoded features as a proxy — observations
        that activate the encoder strongly are typically more unusual.
        """
        with torch.no_grad():
            features = self.encoder(obs)
            # Use feature variance as novelty proxy
            return features.var(dim=-1).mean().item()

    def _update_stats(self, rewards: torch.Tensor) -> None:
        """Update running mean/median/std for reward normalization."""
        for r in rewards.detach().cpu().numpy():
            self.reward_history.append(float(r))

        if len(self.reward_history) > 10:
            values = list(self.reward_history)
            self._running_mean = float(np.mean(values))
            self._running_median = float(np.median(values))
            self._running_std = float(np.std(values)) + 1e-8

    def update_exploration_ratio(self, skill_count: int, max_ratio: float = 1.0) -> None:
        """
        Adjust exploration vs exploitation balance based on skill accumulation.

        As the agent masters more skills, it should spend less time on raw
        exploration and more time refining/combining existing skills.
        Inspired by H-GRAIL's bandit-based motivation selector.

        Args:
            skill_count: Number of skills the agent has mastered so far
            max_ratio: Maximum exploration ratio (1.0 = fully curious)
        """
        # Exponential decay: more skills → less raw exploration.
        # Rate is configurable (exploration_decay_rate attr, default 0.1 —
        # legacy behaviour): the July 2026 Minecraft curiosity program mints
        # a skill PER BLOCK TYPE, so at 0.1 twelve cheap achievements would
        # quench curiosity to 30% right as discovery starts working.
        decay_rate = float(getattr(self, "exploration_decay_rate", 0.1))
        self.exploration_ratio = max_ratio * np.exp(-decay_rate * skill_count)
        # COMPOSE, never overwrite (audit fix): this used to clobber the
        # configured curiosity_reward_scale with the ratio itself, silently
        # multiplying ALL delivered intrinsic (mixer + magnet lp_scalar) by
        # exp(-rate * count). The configured base is preserved; the ratio is
        # a factor on top of it.
        if not hasattr(self, "_base_reward_scale"):
            self._base_reward_scale = float(self.reward_scale)
        self.reward_scale = self._base_reward_scale * self.exploration_ratio

    @property
    def stats(self) -> Dict[str, float]:
        """Current curiosity statistics for logging."""
        return {
            "curiosity_mean": self._running_mean,
            "curiosity_std": self._running_std,
            "exploration_ratio": self.exploration_ratio,
            "reward_history_size": len(self.reward_history),
        }
