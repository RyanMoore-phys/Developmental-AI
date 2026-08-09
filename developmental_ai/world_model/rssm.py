"""
RSSM — Recurrent State Space Model (DreamerV3-style)
=====================================================
The core world model that encodes observations into a compressed latent space
and imagines future trajectories WITHOUT actually interacting with the environment.

Architecture overview:
  - The latent state has TWO parts:
    1. Deterministic (h): GRU hidden state — carries history forward
    2. Stochastic (z): Categorical distribution — captures uncertainty
  - The "prior" predicts z from h alone (what the model expects)
  - The "posterior" uses h + actual observation (what really happened)
  - The gap between prior and posterior = prediction error = learning signal

This is a simplified PyTorch implementation inspired by SimpleDreamer and DreamerV3.
The original DreamerV3 uses JAX; this version keeps the same core math in PyTorch.

Reference: Hafner et al., "Mastering Diverse Domains through World Models" (2023/2025)
  - arxiv: 2301.04104
  - Key insight: symlog predictions + fixed hyperparameters across all domains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Utility: Symlog transform (from DreamerV3)
# ---------------------------------------------------------------------------
# Symlog compresses large values while preserving sign and small values.
# This is what lets DreamerV3 use the SAME hyperparameters across all domains —
# rewards of 0.01 and rewards of 10000 both become manageable numbers.

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic compression: sign(x) * ln(|x| + 1)"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ---------------------------------------------------------------------------
# Twohot distributional encoding (DreamerV3 reward head)
# ---------------------------------------------------------------------------
# A scalar MSE-on-symlog reward head is mean-collapse-prone: when the goal
# reward fires on ~1% of steps, the regression target's least-squares optimum is
# the mean (~0), so the head goes flat near the goal. DreamerV3 instead predicts
# a categorical distribution over a fixed grid of symlog values and trains it
# with cross-entropy to a TWOHOT target (probability split between the two bins
# bracketing the true value). This keeps gradients sharp at the rare nonzero
# reward and is robust across reward magnitudes. Bins span symlog space; the
# range [-5, 5] in symlog covers real rewards up to symexp(5) ~= 147 while
# keeping fine resolution (~0.04) right where DoorKey rewards live (0..~1).
TWOHOT_BINS = 255
TWOHOT_LOW = -5.0
TWOHOT_HIGH = 5.0


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Soft two-hot encoding of x over `bins`. x:(...,) -> (..., B).

    Splits unit mass between the two adjacent bins bracketing x, weighted by
    linear distance, so the expected value of the encoding equals x exactly."""
    B = bins.shape[0]
    x = x.clamp(min=float(bins[0].item()), max=float(bins[-1].item()))
    # upper = index of the first bin >= x (>=1 so lower=upper-1 is valid)
    ge = bins.view(*([1] * x.dim()), B) >= x.unsqueeze(-1)        # (..., B)
    upper = ge.float().argmax(dim=-1).clamp(min=1)                # (...,)
    lower = upper - 1
    bl = bins[lower]
    bu = bins[upper]
    w_u = ((x - bl) / (bu - bl + 1e-8)).clamp(0.0, 1.0)
    w_l = 1.0 - w_u
    target = torch.zeros(*x.shape, B, device=x.device, dtype=w_u.dtype)
    target.scatter_(-1, lower.unsqueeze(-1), w_l.unsqueeze(-1))
    target.scatter_(-1, upper.unsqueeze(-1), w_u.unsqueeze(-1))
    return target


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value (in symlog space) of the categorical reward head.
    logits:(..., B) -> (...,)."""
    p = F.softmax(logits, dim=-1)
    return (p * bins).sum(dim=-1)


# ---------------------------------------------------------------------------
# Straight-Through Categorical
# ---------------------------------------------------------------------------
# DreamerV3's stochastic state uses categorical distributions with
# straight-through gradients — sample discretely in forward pass,
# but use continuous probabilities for backpropagation.

class StraightThroughCategorical(nn.Module):
    """
    Samples from a categorical distribution but passes gradients through
    as if we used the soft probabilities. This lets the model be discrete
    (good for representing distinct concepts) while still being trainable
    with gradient descent.
    """

    def __init__(self, num_categoricals: int, num_classes: int):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: shape (batch, num_categoricals * num_classes)
        Returns:
            One-hot samples with straight-through gradients,
            shape (batch, num_categoricals * num_classes)
        """
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.num_categoricals, self.num_classes)

        if self.training:
            # Gumbel-softmax with hard=True ALREADY applies the straight-through
            # estimator internally (returns y_hard - y_soft.detach() + y_soft).
            # Do not add a second ST on top — that sums two Jacobians into the
            # logits and silently doubles/noises gradients through z.
            result = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            # Deterministic: pick the most likely class, with a single
            # straight-through path in case gradients are ever needed here.
            indices = torch.argmax(logits, dim=-1)
            samples = F.one_hot(indices, self.num_classes).float()
            probs = F.softmax(logits, dim=-1)
            result = samples + probs - probs.detach()
        return result.view(batch_size, -1)

    def log_prob(self, logits: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        """Compute log probability of samples under the distribution."""
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.num_categoricals, self.num_classes)
        samples = samples.view(batch_size, self.num_categoricals, self.num_classes)

        log_probs = F.log_softmax(logits, dim=-1)
        # Sum log probs of selected classes across all categoricals
        return (log_probs * samples).sum(dim=(-1, -2))


# ---------------------------------------------------------------------------
# FiLM Action Conditioner
# ---------------------------------------------------------------------------
# Instead of concatenating a small action vector with a large stochastic state
# (diluting the action signal ~150:1), FiLM produces scale (gamma) and shift
# (beta) parameters from the action that MULTIPLICATIVELY modulate the state.
# This gives the action influence proportional to its learned effect, not its
# raw dimension. Initialized to identity (gamma=1, beta=0) so the model starts
# equivalent to no conditioning and learns to use actions as training demands.

class FiLMActionConditioner(nn.Module):
    """Feature-wise Linear Modulation: action → (gamma, beta) to modulate z."""

    def __init__(self, action_dim: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
            self.net[-1].bias[:feature_dim].fill_(1.0)  # gamma=1, beta=0

    def forward(self, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gb = self.net(action)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma * z + beta


# ---------------------------------------------------------------------------
# RSSM Core
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    """
    Recurrent State Space Model — the heart of the world model.

    The RSSM maintains a latent state that has two components:
      - Deterministic state (h): Updated by a GRU, carries temporal information
      - Stochastic state (z): Sampled from a categorical distribution, captures uncertainty

    Two modes of operation:
      1. OBSERVE (posterior): Given h and a real observation, compute what z should be
         → This is used during training to learn from actual experience
      2. IMAGINE (prior): Given h alone, predict what z will be
         → This is used during planning to imagine future trajectories

    The KL divergence between prior and posterior is the prediction error —
    when the model is surprised, this gap is large, driving learning.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        stochastic_size: int = 32,
        stochastic_classes: int = 32,
        deterministic_size: int = 512,
        hidden_size: int = 256,
        film_conditioning: bool = False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stochastic_size = stochastic_size
        self.stochastic_classes = stochastic_classes
        self.deterministic_size = deterministic_size
        self.stoch_dim = stochastic_size * stochastic_classes  # Total stochastic dimension
        self.film_conditioning = film_conditioning

        # --- Sequence model: GRU updates deterministic state ---
        # Input: previous stochastic state + action (+ optional knowledge) → next deterministic state
        # DreamerV3 uses LayerNorm on both the GRU input and output to prevent
        # the hidden state from saturating into a fixed point during imagination.
        #
        # With FiLM conditioning (film_conditioning=True), the action modulates z
        # multiplicatively instead of being concatenated, so the GRU input is just
        # the modulated z (stoch_dim). Without FiLM, the original concatenation
        # path is preserved exactly.
        if self.film_conditioning:
            self.film = FiLMActionConditioner(
                action_dim, self.stoch_dim, hidden_dim=max(128, action_dim * 16))
            self._base_input_dim = self.stoch_dim
        else:
            self.film = None
            self._base_input_dim = self.stoch_dim + action_dim
        self.gru_input_proj = nn.Sequential(
            nn.Linear(self._base_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(hidden_size, deterministic_size)
        self.gru_norm = nn.LayerNorm(deterministic_size)

        # Knowledge conditioning projection (initialized lazily on first use)
        self._knowledge_proj: Optional[nn.Module] = None
        self._knowledge_dim: int = 0

        # --- Prior: predict stochastic state from deterministic state alone ---
        # "What do I EXPECT to see next?" (before seeing the observation)
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.stoch_dim),
        )

        # --- Posterior: compute stochastic state from deterministic + observation ---
        # "What did I ACTUALLY see?" (after seeing the observation)
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_size + obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.stoch_dim),
        )

        # --- Categorical sampler for stochastic state ---
        self.categorical = StraightThroughCategorical(stochastic_size, stochastic_classes)

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Create a blank initial state (all zeros) for the start of an episode."""
        return {
            "h": torch.zeros(batch_size, self.deterministic_size, device=device),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def observe_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        observation: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        One step of OBSERVE mode — uses the real observation.

        This is called during training: the agent took an action, saw what happened,
        and now updates its internal state using both its prediction AND reality.

        Returns:
            posterior_state: The updated latent state (using real observation)
            model_info: Contains prior/posterior logits for computing KL loss
        """
        # Step 1: Update deterministic state using GRU
        # With FiLM: action multiplicatively modulates z, then z feeds the GRU.
        # Without FiLM: original concatenation [z, action] feeds the GRU.
        if self.film is not None:
            gru_input = self.gru_input_proj(self.film(action, prev_state["z"]))
        else:
            gru_input = self.gru_input_proj(torch.cat([prev_state["z"], action], dim=-1))
        h = self.gru_norm(self.gru(gru_input, prev_state["h"]))

        # Step 2: Compute PRIOR — what the model expected (before seeing observation)
        prior_logits = self.prior_net(h)
        prior_z = self.categorical(prior_logits)

        # Step 3: Compute POSTERIOR — what actually happened (using real observation)
        posterior_logits = self.posterior_net(torch.cat([h, observation], dim=-1))
        posterior_z = self.categorical(posterior_logits)

        posterior_state = {"h": h, "z": posterior_z}
        model_info = {
            "prior_logits": prior_logits,
            "posterior_logits": posterior_logits,
            "prior_z": prior_z,
            "posterior_z": posterior_z,
        }

        return posterior_state, model_info

    def _ensure_knowledge_proj(
        self, knowledge_dim: int, device: torch.device
    ) -> None:
        """Lazily create the projection that fuses knowledge into the GRU input."""
        if self._knowledge_proj is not None and self._knowledge_dim == knowledge_dim:
            return
        self._knowledge_dim = knowledge_dim
        # gru_input_proj is Sequential(Linear, LayerNorm, ELU)
        base_linear = self.gru_input_proj[0]
        hidden_size = base_linear.out_features
        proj_linear = nn.Linear(
            self._base_input_dim + knowledge_dim, hidden_size,
        ).to(device)
        with torch.no_grad():
            proj_linear.weight[:, :self._base_input_dim].copy_(
                base_linear.weight
            )
            proj_linear.bias.copy_(base_linear.bias)
            nn.init.zeros_(proj_linear.weight[:, self._base_input_dim:])
        self._knowledge_proj = nn.Sequential(
            proj_linear,
            nn.LayerNorm(hidden_size).to(device),
            nn.ELU(),
        )

    def imagine_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        knowledge: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        One step of IMAGINE mode — no real observation, pure prediction.

        This is called during planning: the agent imagines what WOULD happen
        if it took a given action, without actually doing it.
        Used for: dreaming trajectories, planning, evaluating potential actions.

        Args:
            prev_state: Previous latent state (h, z)
            action: Action taken
            knowledge: Optional knowledge vector from GNN. When provided,
                       the GRU transition is conditioned on symbolic knowledge,
                       letting the world model use learned facts during dreaming.
        """
        if self.film is not None:
            base_input = self.film(action, prev_state["z"])
        else:
            base_input = torch.cat([prev_state["z"], action], dim=-1)

        if knowledge is not None:
            self._ensure_knowledge_proj(
                knowledge.shape[-1], base_input.device
            )
            full_input = torch.cat([base_input, knowledge], dim=-1)
            gru_input = self._knowledge_proj(full_input)
        else:
            gru_input = self.gru_input_proj(base_input)

        h = self.gru_norm(self.gru(gru_input, prev_state["h"]))

        # Only the prior is available (no observation to form posterior)
        prior_logits = self.prior_net(h)
        prior_z = self.categorical(prior_logits)

        return {"h": h, "z": prior_z}

    def get_latent(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate h and z into a single latent vector for downstream use."""
        return torch.cat([state["h"], state["z"]], dim=-1)

    @property
    def latent_dim(self) -> int:
        """Total dimension of the latent state (h + z)."""
        return self.deterministic_size + self.stoch_dim

    def compute_kl_loss(
        self,
        prior_logits: torch.Tensor,
        posterior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        DreamerV3-style KL balancing between posterior and prior.

        The key insight: split KL into two terms so the prior ALWAYS gets
        a learning signal, even when KL is small.

        - Prior loss:     KL(sg(posterior) || prior)     — trains prior toward posterior
        - Posterior loss:  KL(posterior || sg(prior))     — regularizes posterior

        The prior loss is weighted higher (0.5 vs 0.1) because accurate
        prior predictions are what make dreams accurate. Without this,
        free nats can completely block gradient flow to the prior.
        """
        batch_size = prior_logits.shape[0]
        prior = prior_logits.view(batch_size, self.stochastic_size, self.stochastic_classes)
        posterior = posterior_logits.view(batch_size, self.stochastic_size, self.stochastic_classes)

        prior_probs = F.softmax(prior, dim=-1)
        posterior_probs = F.softmax(posterior, dim=-1)

        # Prior loss: train the prior to match the posterior (posterior is detached)
        # This is the critical term for dream accuracy
        prior_loss = (posterior_probs.detach() * (
            torch.log(posterior_probs.detach() + 1e-8) - torch.log(prior_probs + 1e-8)
        )).sum(dim=-1)

        # Posterior loss: regularize posterior to not diverge too far from prior
        posterior_loss = (posterior_probs * (
            torch.log(posterior_probs + 1e-8) - torch.log(prior_probs.detach() + 1e-8)
        )).sum(dim=-1)

        # Free nats applied per term (small, just to prevent collapse)
        free_nats = 0.1
        prior_loss = torch.clamp(prior_loss, min=free_nats)
        posterior_loss = torch.clamp(posterior_loss, min=free_nats)

        # Weight prior learning much higher than posterior regularization
        kl = 1.0 * prior_loss.mean() + 0.1 * posterior_loss.mean()
        return kl


# ---------------------------------------------------------------------------
# Observation Encoder & Decoder
# ---------------------------------------------------------------------------
# These translate between raw observations and the latent space.
# For simple environments (CartPole, etc.) these are MLPs.
# For pixel environments, swap these for CNNs.

class ObservationEncoder(nn.Module):
    """Encodes raw observations into a feature vector for the RSSM."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ObservationDecoder(nn.Module):
    """Decodes latent state back into predicted observations (for reconstruction loss)."""

    def __init__(self, latent_dim: int, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class RewardPredictor(nn.Module):
    """Predicts reward from latent state as a categorical distribution over a
    fixed symlog grid (DreamerV3 twohot head). Returns `num_bins` logits;
    decode via twohot_decode(...) then symexp(...) for the real-scale reward."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256,
                 num_bins: int = TWOHOT_BINS):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)  # (..., num_bins) logits


class ContinuePredictor(nn.Module):
    """Predicts whether the episode continues (1) or terminates (0) from latent state.

    Returns raw logits — caller applies sigmoid or uses BCE-with-logits.
    This allows pos_weight for class imbalance (terminations are rare).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


# ---------------------------------------------------------------------------
# CNN Encoder & Decoder (for pixel-based environments)
# ---------------------------------------------------------------------------
# These replace the MLP encoder/decoder when the environment provides
# image observations (e.g., Atari, rendered MuJoCo, any rgb_array env).
# Architecture: stride-2 convolutions downsample from image_size to a fixed
# 4x4 bottleneck (4 stages at 64px, 5 at 128px — see _cnn_channel_ladder),
# then a linear layer projects to hidden_dim.

def _cnn_channel_ladder(image_size: int, cap: int = 256) -> list:
    """Stride-2 stage output channels from image_size down to a 4x4 bottleneck.

    64 -> [32, 64, 128, 256] (the original hardcoded ladder, byte-identical
    layer structure); 128 -> [32, 64, 128, 256, 256] (one extra stage, same
    4x4x256 bottleneck so everything downstream of the CNN is unchanged).
    """
    if image_size < 16 or (image_size & (image_size - 1)) != 0:
        raise ValueError(
            f"image_size must be a power of two >= 16, got {image_size}")
    chans, c, size = [], 32, image_size
    while size > 4:
        chans.append(c)
        c = min(c * 2, cap)
        size //= 2
    return chans


class CNNEncoder(nn.Module):
    """Encodes pixel observations (C, H, W) into feature vectors for the RSSM."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels

        chans = _cnn_channel_ladder(image_size)
        layers, c_in = [], in_channels
        for c_out in chans:
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.ELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        feat_size = image_size // (2 ** len(chans))  # always 4
        self.flatten_dim = chans[-1] * feat_size * feat_size
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, self.in_channels, self.image_size, self.image_size)
        x = self.conv(obs)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class CNNDecoder(nn.Module):
    """Decodes latent state back to pixel observations for reconstruction loss."""

    def __init__(self, latent_dim: int, out_channels: int = 3, hidden_dim: int = 256, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.out_channels = out_channels

        chans = _cnn_channel_ladder(image_size)
        self.feat_size = image_size // (2 ** len(chans))  # always 4
        self._bottleneck = chans[-1]
        self.fc = nn.Linear(
            latent_dim, self._bottleneck * self.feat_size * self.feat_size)
        rev = chans[::-1]
        layers = []
        for i in range(len(rev) - 1):
            layers += [
                nn.ConvTranspose2d(rev[i], rev[i + 1], 3, stride=2,
                                   padding=1, output_padding=1),
                nn.ELU(),
            ]
        layers.append(
            nn.ConvTranspose2d(rev[-1], out_channels, 3, stride=2,
                               padding=1, output_padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        x = x.reshape(x.shape[0], self._bottleneck, self.feat_size, self.feat_size)
        x = self.deconv(x)
        return x.reshape(x.shape[0], -1)


# ---------------------------------------------------------------------------
# Inverse Dynamics Head (auxiliary action-prediction loss)
# ---------------------------------------------------------------------------
# Forces the RSSM transition to encode action-distinguishable next states:
# given (latent_t, latent_{t+1}), predict the action that caused the
# transition. Without this, reconstruction loss alone lets the model
# mostly ignore actions (static backgrounds dominate MiniGrid's obs).

class InverseDynamicsHead(nn.Module):
    """Predict the action from consecutive latent states."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256,
                 discrete: bool = True):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, latent_t: torch.Tensor,
                latent_t1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent_t, latent_t1], dim=-1))


# ---------------------------------------------------------------------------
# Complete World Model (wraps RSSM + encoder/decoder/predictors)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    Complete world model that combines the RSSM with observation encoder/decoder
    and reward/continue predictors.

    This is the main class you interact with. It handles:
    1. Encoding observations and feeding them to the RSSM
    2. Decoding latent states back to predicted observations
    3. Predicting rewards and episode continuation
    4. Computing all training losses (reconstruction, KL, reward, continue)
    5. Imagining future trajectories for planning

    The world model is trained on real experience from the replay buffer,
    then used to IMAGINE trajectories for training the policy without
    needing more real environment interaction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        stochastic_size: int = 32,
        stochastic_classes: int = 32,
        deterministic_size: int = 512,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        pixel_obs: bool = False,
        image_channels: int = 3,
        image_size: int = 64,
        film_conditioning: bool = False,
        inverse_dynamics: bool = False,
        inverse_dynamics_scale: float = 1.0,
        discrete_actions: bool = True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pixel_obs = pixel_obs

        # Choose encoder/decoder based on observation type
        if pixel_obs:
            self.encoder = CNNEncoder(image_channels, hidden_dim, image_size)
            self.decoder = CNNDecoder(
                0, image_channels, hidden_dim, image_size
            )  # latent_dim set below after RSSM init
        else:
            self.encoder = ObservationEncoder(obs_dim, hidden_dim)

        self.rssm = RSSM(
            obs_dim=hidden_dim,  # Encoder output dim (same for MLP and CNN)
            action_dim=action_dim,
            stochastic_size=stochastic_size,
            stochastic_classes=stochastic_classes,
            deterministic_size=deterministic_size,
            hidden_size=hidden_dim,
            film_conditioning=film_conditioning,
        )

        # Decoder needs latent_dim from RSSM — rebuild CNN decoder with correct dim
        if pixel_obs:
            self.decoder = CNNDecoder(
                self.rssm.latent_dim, image_channels, hidden_dim, image_size
            )
        else:
            self.decoder = ObservationDecoder(self.rssm.latent_dim, obs_dim, hidden_dim)

        self.reward_predictor = RewardPredictor(
            self.rssm.latent_dim, hidden_dim, num_bins=TWOHOT_BINS)
        self.continue_predictor = ContinuePredictor(self.rssm.latent_dim, hidden_dim)
        # Fixed symlog grid for the twohot reward head (not a learned parameter).
        self.register_buffer(
            "reward_bins", torch.linspace(TWOHOT_LOW, TWOHOT_HIGH, TWOHOT_BINS))

        # Causal action alignment (ON by default since the July 2026 audit —
        # see CODE_AUDIT_2026-07.md §A). When True, observe_sequence builds
        # state_t from the PREVIOUS action a_{t-1} (the action that CAUSED
        # o_t), so the prior/dynamics learn true action->next-state causality
        # and imagine_step(state_t, a_t) predicts o_{t+1}. The original
        # convention paired a_t with o_t, so the prior predicted the next obs
        # from the NEXT action (a policy-correlation leak) — set
        # world_model.causal_align=false in config to reproduce it.
        self.action_shift = True

        # --- Inverse dynamics head (auxiliary action-prediction loss) ---
        # Forces the RSSM latent transition to encode which action caused it.
        # From (latent_t, latent_{t+1}), predict the action that was taken.
        # This creates direct gradient pressure for action-distinguishable
        # transitions — without it, the model can satisfy reconstruction loss
        # while mostly ignoring actions (static backgrounds dominate obs).
        self.inverse_dynamics_enabled = inverse_dynamics
        self.inverse_dynamics_scale = inverse_dynamics_scale
        self.inverse_head = (
            InverseDynamicsHead(
                self.rssm.latent_dim, action_dim, hidden_dim,
                discrete=discrete_actions)
            if inverse_dynamics else None
        )

        # Optimizer for all world model parameters (includes inverse head if present)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _reward_symlog(self, latent: torch.Tensor) -> torch.Tensor:
        """Expected reward in SYMLOG space from the twohot head. (...,)."""
        return twohot_decode(self.reward_predictor(latent), self.reward_bins)

    def predict_reward(self, latent: torch.Tensor) -> torch.Tensor:
        """Real-scale reward prediction from the twohot head. (...,)."""
        return symexp(self._reward_symlog(latent))

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a sequence of real experience through the world model.

        Args:
            observations: (batch, seq_len, obs_dim) — real observations
            actions: (batch, seq_len, action_dim) — actions taken

        Returns:
            states: Dict with 'h' and 'z' tensors, each (batch, seq_len, dim)
            infos: Dict with prior/posterior logits for KL computation
        """
        batch_size, seq_len, _ = observations.shape
        device = observations.device

        # Causal alignment: build state_t from the action that CAUSED o_t, i.e.
        # a_{t-1} (shift actions right by one; a_{-1} = zeros). The GRU transition
        # into state_t then uses the causal action, so the prior learns true
        # dynamics and imagine_step(state_t, a_t) predicts o_{t+1}.
        if self.action_shift:
            shifted = torch.zeros_like(actions)
            shifted[:, 1:] = actions[:, :-1]
            actions = shifted

        # Encode all observations at once (more efficient)
        encoded = self.encoder(observations.reshape(-1, self.obs_dim))
        encoded = encoded.view(batch_size, seq_len, -1)

        # Step through the sequence
        state = self.rssm.initial_state(batch_size, device)
        all_h, all_z = [], []
        all_prior_logits, all_posterior_logits = [], []

        for t in range(seq_len):
            state, info = self.rssm.observe_step(state, actions[:, t], encoded[:, t])
            all_h.append(state["h"])
            all_z.append(state["z"])
            all_prior_logits.append(info["prior_logits"])
            all_posterior_logits.append(info["posterior_logits"])

        states = {
            "h": torch.stack(all_h, dim=1),
            "z": torch.stack(all_z, dim=1),
        }
        infos = {
            "prior_logits": torch.stack(all_prior_logits, dim=1),
            "posterior_logits": torch.stack(all_posterior_logits, dim=1),
        }

        return states, infos

    def imagine_trajectory(
        self,
        initial_state: Dict[str, torch.Tensor],
        policy_fn,
        horizon: int = 15,
        knowledge: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory by rolling out the world model with a policy.

        This is how the agent "dreams" — it uses its learned world model to
        simulate what would happen if it followed a given policy, WITHOUT
        interacting with the real environment. The imagined trajectory is
        used to train the policy (actor-critic in latent space).

        Args:
            initial_state: Starting latent state
            policy_fn: Function that takes latent state → action
            horizon: How many steps to imagine
            knowledge: Optional (batch, knowledge_dim) vector from the GNN.
                       When provided, the RSSM GRU transition is conditioned
                       on symbolic knowledge at every imagination step,
                       so the world model's dreams respect learned facts.

        Returns:
            Dict with imagined latents, actions, rewards, continues
        """
        state = initial_state
        latents, actions, rewards, continues = [], [], [], []

        for _ in range(horizon):
            latent = self.rssm.get_latent(state)
            action = policy_fn(latent)

            # Predict reward (SYMLOG scale; caller applies symexp) and
            # continuation from current state. Reward via the twohot head.
            reward = self._reward_symlog(latent).unsqueeze(-1)
            cont = torch.sigmoid(self.continue_predictor(latent))

            latents.append(latent)
            actions.append(action)
            rewards.append(reward)
            continues.append(cont)

            # Imagine next state, conditioned on symbolic knowledge
            state = self.rssm.imagine_step(state, action, knowledge=knowledge)

        return {
            "latents": torch.stack(latents, dim=1),
            "actions": torch.stack(actions, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continues": torch.stack(continues, dim=1),
        }

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
        return_per_sample: bool = False,
    ):
        """
        Compute all world model training losses.

        The world model learns from four signals:
        1. Reconstruction loss: Can it reproduce the observations from latent state?
        2. KL loss: How surprised was it? (prior vs posterior divergence)
        3. Reward loss: Can it predict what reward the environment gives?
        4. Continue loss: Can it predict when episodes end?

        All predictions use symlog for scale-invariant learning.

        Prioritized Experience Replay support (opt-in, fully backward
        compatible — with importance_weights=None the math is identical to
        the original uniform path):
          - importance_weights: (batch,) IS weights applied per-sequence to the
            reconstruction and reward terms to correct PER's sampling bias.
          - return_per_sample: if True, also return the per-sequence
            reconstruction error (the priority signal) as a detached tensor.
        """
        # Forward pass through the sequence
        states, infos = self.observe_sequence(observations, actions)

        # Get latent representations
        batch_size, seq_len, _ = observations.shape
        latents = torch.cat([states["h"], states["z"]], dim=-1)
        latents_flat = latents.reshape(-1, latents.shape[-1])

        w = None
        if importance_weights is not None:
            w = importance_weights.detach().reshape(-1)

        # Causal alignment shifts the REWARD/CONTINUE targets to match the shifted
        # states: state_t (built from a_{t-1}, o_t) predicts the reward/continue for
        # ARRIVING at o_t — i.e. r_{t-1}, continue_{t-1}. Index 0 has no predecessor
        # action and is masked out of the reward/continue losses. Recon (state_t ->
        # o_t) and KL are unchanged.
        if self.action_shift:
            rewards = torch.cat([rewards[:, :1] * 0.0, rewards[:, :-1]], dim=1)
            continues = torch.cat([continues[:, :1] * 0.0 + 1.0, continues[:, :-1]], dim=1)
            step_valid = torch.ones_like(rewards)
            step_valid[:, 0] = 0.0                                   # (B, L)
        else:
            step_valid = None

        # 1. Reconstruction loss: decode latent → predicted observation
        pred_obs = self.decoder(latents_flat).view(batch_size, seq_len, -1)
        recon_target = observations if self.pixel_obs else symlog(observations)
        recon_per_elem = F.mse_loss(pred_obs, recon_target, reduction="none")
        per_seq_recon = recon_per_elem.mean(dim=tuple(range(1, recon_per_elem.dim())))
        if w is not None:
            recon_loss = (w * per_seq_recon).sum() / (w.sum() + 1e-8)
        else:
            recon_loss = per_seq_recon.mean()

        # 2. KL divergence loss: how surprised was the model?
        kl_loss = self.rssm.compute_kl_loss(
            infos["prior_logits"].reshape(-1, self.rssm.stoch_dim),
            infos["posterior_logits"].reshape(-1, self.rssm.stoch_dim),
        )

        # 3. Reward prediction loss — TWOHOT distributional, CLASS-BALANCED.
        # The goal reward fires on ~1% of steps. A scalar MSE head collapses to
        # the majority mean (~0) and goes BLIND to the goal; the twohot head
        # predicts a categorical over a symlog grid and is trained by
        # cross-entropy to the two-hot target, keeping gradients sharp at the
        # rare nonzero reward. We still upweight nonzero-reward steps (weight =
        # #zero/#nonzero, capped) and pool the weighted mean GLOBALLY so a goal
        # step counts against the whole batch (per-sequence normalization would
        # dilute the ~1%-rare goal steps back toward zero).
        pred_reward_logits = self.reward_predictor(latents_flat).view(
            batch_size, seq_len, -1)
        reward_symlog_target = symlog(rewards)                       # (B, L)
        twohot_target = twohot_encode(reward_symlog_target, self.reward_bins)
        reward_logp = F.log_softmax(pred_reward_logits, dim=-1)      # (B, L, K)
        reward_per_elem = -(twohot_target * reward_logp).sum(dim=-1)  # (B, L) CE
        nonzero = rewards.abs() > 1e-3                               # (B, L)
        n_nz = nonzero.sum().float().clamp(min=1.0)
        n_z = (float(reward_symlog_target.numel()) - n_nz).clamp(min=1.0)
        nz_weight = (n_z / n_nz).clamp(max=100.0)
        rw_elem = torch.where(nonzero,
                              nz_weight * torch.ones_like(reward_per_elem),
                              torch.ones_like(reward_per_elem))      # (B, L)
        if w is not None:
            rw_elem = rw_elem * w.reshape(batch_size, 1)
        if step_valid is not None:
            rw_elem = rw_elem * step_valid
        reward_loss = (rw_elem * reward_per_elem).sum() / rw_elem.sum().clamp(min=1e-8)

        # 4. Continue prediction loss (binary cross-entropy with class balancing)
        # ContinuePredictor returns logits; we use BCE-with-logits for stability.
        # Termination events are rare (~1% of steps), so pos_weight amplifies them.
        pred_continue_logits = self.continue_predictor(latents_flat).view(batch_size, seq_len, 1)
        continue_targets = continues.unsqueeze(-1).float()
        n_total = continue_targets.numel()
        n_done = (continue_targets < 0.5).sum().float().clamp(min=1.0)
        n_cont = n_total - n_done
        done_weight = (n_cont / n_done).clamp(max=50.0)
        pos_weight = torch.ones_like(continue_targets)
        pos_weight[continue_targets < 0.5] = done_weight
        continue_ce = F.binary_cross_entropy_with_logits(
            pred_continue_logits, continue_targets,
            weight=pos_weight, reduction="none",
        )
        if step_valid is not None:
            m = step_valid.unsqueeze(-1)
            continue_loss = (continue_ce * m).sum() / m.sum().clamp(min=1e-8)
        else:
            continue_loss = continue_ce.mean()

        # 5. Inverse dynamics loss: from consecutive latents, predict the action.
        #    This forces the RSSM transition to encode action-distinguishable
        #    states — the architectural complement to FiLM conditioning.
        if self.inverse_head is not None and seq_len >= 2:
            latent_t = latents[:, :-1].reshape(-1, latents.shape[-1])
            latent_t1 = latents[:, 1:].reshape(-1, latents.shape[-1])
            pred_action = self.inverse_head(latent_t, latent_t1)
            # The action that caused transition (t -> t+1) is the one fed to
            # observe_step at step t+1. observe_sequence right-shifts actions
            # internally when action_shift=True, so the action consumed at
            # step t+1 is a_t = actions[:, t] under causal alignment, and
            # a_{t+1} = actions[:, t+1] under the original convention. Using
            # the unshifted slice with action_shift=True would train the head
            # to predict the agent's NEXT action — a pure policy leak.
            true_actions = (
                actions[:, :-1] if self.action_shift else actions[:, 1:]
            )                                                     # (B, L-1, A)
            true_actions_flat = true_actions.reshape(-1, self.action_dim)
            if self.inverse_head.discrete:
                inverse_loss = F.cross_entropy(
                    pred_action, true_actions_flat.argmax(dim=-1))
            else:
                inverse_loss = F.mse_loss(pred_action, true_actions_flat)
        else:
            inverse_loss = torch.tensor(0.0, device=latents.device)

        recon_scale = 1.0
        kl_scale = 0.3
        reward_scale = 1.0
        continue_scale = 5.0
        inverse_scale = self.inverse_dynamics_scale if self.inverse_head is not None else 0.0

        total_loss = (
            recon_scale * recon_loss
            + kl_scale * kl_loss
            + reward_scale * reward_loss
            + continue_scale * continue_loss
            + inverse_scale * inverse_loss
        )

        losses = {
            "total": total_loss,
            "reconstruction": recon_loss,
            "kl": kl_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "inverse": inverse_loss,
        }

        if return_per_sample:
            # Per-sequence reconstruction error is the PER priority signal.
            return losses, per_seq_recon.detach()
        return losses

    def train_step(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
        return_per_sample: bool = False,
    ):
        """One gradient update step. Returns loss values as plain floats for logging.

        If return_per_sample is True, returns (metrics, per_sequence_error) where
        per_sequence_error is a numpy array used to update PER priorities.
        """
        self.train()
        out = self.compute_loss(
            observations,
            actions,
            rewards,
            continues,
            importance_weights=importance_weights,
            return_per_sample=return_per_sample,
        )
        if return_per_sample:
            losses, per_sample = out
        else:
            losses = out

        self.optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping for stability (DreamerV3 uses 100.0)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.optimizer.step()

        metrics = {k: v.item() for k, v in losses.items()}
        if return_per_sample:
            return metrics, per_sample.detach().cpu().numpy()
        return metrics

    def get_prediction_error(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prediction error for a single transition.
        This feeds into the curiosity engine — high prediction error = high novelty.

        Returns scalar prediction error per batch element.
        """
        with torch.no_grad():
            # Imagine what would happen
            imagined_state = self.rssm.imagine_step(state, action)
            imagined_latent = self.rssm.get_latent(imagined_state)

            # Predict what we'd observe
            pred_obs = self.decoder(imagined_latent)

            # Compare prediction to reality
            if self.pixel_obs:
                error = F.mse_loss(pred_obs, next_observation, reduction="none")
            else:
                error = F.mse_loss(pred_obs, symlog(next_observation), reduction="none")
            return error.mean(dim=-1)  # (batch,)
