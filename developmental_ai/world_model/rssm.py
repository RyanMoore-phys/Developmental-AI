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

import math

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
# ---- WORLD-MODEL STEP TELEMETRY (2026-10-07, measurement only) ------------
# When True, WorldModel.train_step fills `last_step_stats` (every loss head as
# loss_<name>, the unclamped KL, the pre-clip grad norm and the prior /
# posterior entropies). All of it is computed under no_grad from tensors the
# step already produced and draws no random numbers, so parameters and RNG
# are byte-identical on or off (tests/_ml_stats_smoke.py). Cost: ONE extra
# device sync per gradient step (a single .tolist() of four scalars).
COLLECT_STATS = True

# Short, stable names for the telemetry keys (anything else -> loss_<key>).
_STAT_LOSS_NAMES = {"total": "loss_total", "reconstruction": "loss_recon",
                    "kl": "loss_kl", "reward": "loss_reward",
                    "continue": "loss_continue", "flow": "loss_flow"}

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

    def prior_divergence(self, prior_logits: torch.Tensor,
                         posterior_logits: torch.Tensor) -> torch.Tensor:
        """UNCLAMPED KL(posterior || prior), posterior detached.

        WHY NOT compute_kl_loss FOR THE HORIZON TERM (2026-09-19). That
        method applies free nats of 0.1 to BOTH halves, so its value cannot
        fall below 1.0*0.1 + 0.1*0.1 = 0.11 and its gradient vanishes
        entirely underneath. MEASURED while building the multi-horizon term:
        two models with completely different training reported 0.1100 on a
        16-step rollout across three seeds — the instrument was pinned at
        its own floor and could not see anything.

        Free nats exist to stop the POSTERIOR being crushed toward the prior
        early in training. That is not the concern sixteen steps out, where
        the whole point is to keep pressing on a rollout that is already
        roughly right. So the horizon term uses the raw divergence, and
        compute_kl_loss keeps its floor for the one-step term it was tuned
        for.
        """
        b = prior_logits.shape[0]
        pri = F.log_softmax(
            prior_logits.view(b, self.stochastic_size, self.stochastic_classes),
            dim=-1)
        post = F.softmax(
            posterior_logits.view(b, self.stochastic_size,
                                  self.stochastic_classes), dim=-1).detach()
        return (post * (torch.log(post + 1e-8) - pri)).sum(-1).mean()


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
# Architecture: stride-2 convolutions downsample from image_size to a small
# square bottleneck (`min_grid`, historically always 4), then a linear layer
# projects to hidden_dim.
#
# ---- WHY min_grid AND coord_channels EXIST (2026-09-18) --------------------
# An earlier reading of this file claimed the encoder "throws away spatial
# identity at the bottleneck". That is WRONG and is recorded here so nobody
# re-derives it: forward() does reshape -> Linear, and a flatten+Linear is
# POSITION-SPECIFIC — every bottleneck cell owns its own weights, so position
# is representable today.
#
# The real defect is RESOLUTION. `while size > 4` drives every image_size to a
# 4x4 grid, so at 128px one bottleneck cell covers a 32x32 PIXEL SLAB of
# scene. A mid-distance tree trunk lives entirely inside one cell, indistinct
# from the dirt behind it. That is the measured shape of "position-blind"
# (configs/minecraft_skybot.yaml, symbol_weight) and of the vision magnet's
# admitted "no instance disambiguation of two trees in one frame".
#
# min_grid=8 at 128px costs ONE conv stage and makes a cell 16x16 px — about
# one block face at working distance. coord_channels appends normalized (x, y)
# planes to the INPUT so the convolutional stack itself can express "this
# texture, over there" instead of leaving all position information to be
# reconstructed by the final Linear.
#
# Both default to the historical behaviour (min_grid=4, no coord channels), so
# every config that does not opt in builds the same layer structure as before.

def _cnn_channel_ladder(image_size: int, cap: int = 256,
                        min_grid: int = 4) -> list:
    """Stride-2 stage output channels from image_size down to `min_grid`.

    min_grid=4 (the default, unchanged): 64 -> [32, 64, 128, 256] (the
    original hardcoded ladder, byte-identical layer structure); 128 ->
    [32, 64, 128, 256, 256] (one extra stage, same 4x4x256 bottleneck).

    min_grid=8: 128 -> [32, 64, 128, 256], i.e. one FEWER stage and an
    8x8x256 bottleneck. Four times the spatial resolution for the heads that
    read the feature map.
    """
    if image_size < 16 or (image_size & (image_size - 1)) != 0:
        raise ValueError(
            f"image_size must be a power of two >= 16, got {image_size}")
    if min_grid < 1 or (min_grid & (min_grid - 1)) != 0:
        raise ValueError(f"min_grid must be a power of two, got {min_grid}")
    if min_grid >= image_size:
        raise ValueError(
            f"min_grid ({min_grid}) must be smaller than image_size "
            f"({image_size}) — there would be no downsampling stages")
    chans, c, size = [], 32, image_size
    while size > min_grid:
        chans.append(c)
        c = min(c * 2, cap)
        size //= 2
    return chans


class CNNEncoder(nn.Module):
    """Encodes pixel observations (C, H, W) into feature vectors for the RSSM.

    Two outputs, one forward pass:
      * the VECTOR (width `vec_dim`), which is what the RSSM has always
        consumed and what the fovea head reads via `set_crop_encoder`;
      * the FEATURE MAP (B, map_channels, grid, grid), which the flow head and
        any future object-slot head read. Exposed through `encode_with_map`
        rather than cached on the module: a cached activation would retain the
        autograd graph between calls and would be wrong under the loop's
        two-body stepping.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 256,
                 image_size: int = 64, min_grid: int = 4,
                 coord_channels: bool = False,
                 readout_channels: int = 0):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels          # channels of the OBSERVATION
        self.coord_channels = bool(coord_channels)
        self.min_grid = int(min_grid)

        chans = _cnn_channel_ladder(image_size, min_grid=min_grid)
        conv_in = in_channels + (2 if self.coord_channels else 0)
        layers, c_in = [], conv_in
        for c_out in chans:
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.ELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        feat_size = image_size // (2 ** len(chans))
        self.grid = feat_size
        self.map_channels = chans[-1]

        # READOUT BOTTLENECK. At min_grid=8 the flatten is 8*8*256 = 16384 and
        # a Linear to 768 would be 12.6M params (vs 3.1M at 4x4). A 1x1 conv
        # to `readout_channels` first keeps that at ~6.3M for 128 channels.
        # 0 = no 1x1 conv = the historical path.
        if readout_channels and readout_channels != chans[-1]:
            self.readout = nn.Sequential(
                nn.Conv2d(chans[-1], readout_channels, 1), nn.ELU())
            flat_c = readout_channels
        else:
            self.readout = None
            flat_c = chans[-1]
        self.flatten_dim = flat_c * feat_size * feat_size
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ELU(),
        )
        # THE ENCODER'S OWN OUTPUT WIDTH. Read this, never `rssm.obs_dim` —
        # once proprioception is concatenated onto the embed (see
        # WorldModel.proprio_dim) the two are no longer the same number, and
        # the fovea head's input width is derived from the ENCODER.
        self.vec_dim = int(hidden_dim)

        if self.coord_channels:
            ys = torch.linspace(-1.0, 1.0, image_size).view(1, 1, -1, 1)
            xs = torch.linspace(-1.0, 1.0, image_size).view(1, 1, 1, -1)
            coords = torch.cat([
                ys.expand(1, 1, image_size, image_size),
                xs.expand(1, 1, image_size, image_size),
            ], dim=1)
            self.register_buffer("_coords", coords, persistent=False)

    def _as_image(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, self.in_channels,
                           self.image_size, self.image_size)
        if self.coord_channels:
            obs = torch.cat(
                [obs, self._coords.to(obs.dtype).expand(obs.shape[0], -1, -1, -1)],
                dim=1)
        return obs

    def encode_with_map(self, obs: torch.Tensor):
        """-> (vector [B, vec_dim], feature map [B, map_channels, grid, grid])."""
        x = self.conv(self._as_image(obs))
        fmap = x
        if self.readout is not None:
            x = self.readout(x)
        return self.fc(x.reshape(x.shape[0], -1)), fmap

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encode_with_map(obs)[0]


class CNNDecoder(nn.Module):
    """Decodes latent state back to pixel observations for reconstruction loss."""

    def __init__(self, latent_dim: int, out_channels: int = 3, hidden_dim: int = 256,
                 image_size: int = 64, min_grid: int = 4):
        super().__init__()
        self.image_size = image_size
        self.out_channels = out_channels

        # MUST share the encoder's min_grid or the deconv stack lands on the
        # wrong output resolution (8x8 start with a 5-stage ladder would
        # produce 256px from a 128px frame).
        chans = _cnn_channel_ladder(image_size, min_grid=min_grid)
        self.feat_size = image_size // (2 ** len(chans))
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
# PERSPECTIVE FROM MOTION — the flow head and its warp objective
# ---------------------------------------------------------------------------
# THE CLAIM: depth is what your own movement reveals, and objectness is what
# moves together when you move. A monocular agent that never moves cannot know
# how far anything is; one that moves and PREDICTS how the scene will sweep
# past it has to represent depth to get that prediction right.
#
# THE MECHANISM. Given the latent at t and the action about to be taken, the
# FlowHead predicts a dense displacement field. That field is used to WARP
# frame t into a prediction of frame t+1, and the photometric error of that
# warp is the loss. Nothing supervises the flow itself — novel-view synthesis
# IS the supervision (the SfMLearner / monodepth2 construction, and the
# ego-motion-as-supervision result of Agrawal/Carreira/Malik 2015 and
# Jayaraman/Grauman 2015).
#
# WHY THIS AND NOT A RADIAL-EXPANSION MEASUREMENT. The intuitive version of
# this idea is behavioural: centre on a blob, step forward, watch its radius
# grow, call it converged when the radius stops growing. That needs a
# hand-written notion of what a blob is, a hand-written controller to chase
# it, and it pays the agent for walking into walls (a wall is the cheapest
# source of monotone radial growth there is). Predicting the field instead
# collapses the whole loop into one head forward: under a forward action,
# PREDICTED FLOW MAGNITUDE IS INVERSE DEPTH. Near things sweep fast. That is
# the same fact the radius test measures, available without moving, learned
# rather than declared.
#
# WHAT THE RESIDUAL IS. Whatever the action-conditioned warp CANNOT explain is
# either (a) something that moved on its own, or (b) geometry the model has
# not learned yet. Both are worth being curious about and neither can be
# obtained by standing still: no ego-motion -> no predicted flow -> no
# residual -> no income. That property is the reason this design is safe to
# attach to the curiosity channel at all, and `tests/_perspective_smoke.py`
# asserts it directly.
#
# COST. Everything here runs inside world-model TRAINING, on replayed
# sequences that already hold consecutive frames. Nothing is added to the
# env-step critical path (the lesson of fovea_interval 12 costing 28% of the
# step rate); the per-step senses in the loop are one small head forward on a
# latent the loop already computed, the same deal _reach_sense gets.

def flow_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample `img` at (identity + flow) — i.e. warp frame t toward frame t+1.

    CONVENTION, stated because getting it backwards is silent: `flow` is the
    BACKWARD field in NORMALIZED grid units (the [-1, 1] coordinate system
    grid_sample uses, so 1.0 == half the image). output[p] = img[p + flow[p]]:
    "the content that appears at p in the next frame came from p + flow[p] in
    this one". This is the view-synthesis convention, not the tracking one.

    img:  (B, C, H, W)      flow: (B, 2, H, W) with channel 0 = x, 1 = y
    """
    b, _, h, w = img.shape
    ys = torch.linspace(-1.0, 1.0, h, device=img.device, dtype=img.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=img.device, dtype=img.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    grid = base + flow.permute(0, 2, 3, 1)
    # ---- align_corners=True IS LOAD-BEARING (bug, fixed 2026-09-18) -------
    # It shipped as False, which does NOT pair with a linspace(-1, 1, n) base
    # grid: under that convention -1 and +1 are the OUTER EDGES of the corner
    # pixels, not their centres, so the identity grid resamples at a half-pixel
    # offset. MEASURED on a random 16x16 image: max |warp(img, 0) - img| =
    # 0.490 with False, 0.000001 with True.
    #
    # WHAT THAT COST. Zero flow was not the identity, so the photometric loss
    # charged a large floor of blur error that NO flow field could remove, and
    # the head's cheapest way to reduce it was to learn a constant
    # compensating offset — a bias on every depth reading. The residual was
    # inflated the same way, since its denominator uses no warp at all.
    #
    # WHY NO TEST CAUGHT IT: the zero-motion contract gates on `motion_floor`
    # BEFORE the warp runs, so identical frames returned 0 by the gate and
    # never exercised this path. tests/_perspective_smoke.py now asserts the
    # identity directly.
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def photometric(src: torch.Tensor, tgt: torch.Tensor, flow: torch.Tensor,
                eps: float = 1e-3):
    """-> (warp_error, identity_error), each (B, 1, H, W) per-pixel.

    ONE IMPLEMENTATION, TWO CONSUMERS. The training loss and the `mover` sense
    both need "how wrong was the warp" and "how much did the frame change at
    all", and before this they computed them separately. That is the
    tensor-level form of the duplicated-body hazard (CLAUDE.md 4.2): an edit to
    one would silently not apply to the other, and the two numbers are supposed
    to be comparable — auto-masking below depends on exactly that.

    `identity_error` is the error of NOT warping: the null hypothesis that
    nothing moved. It is the denominator of the residual and the threshold for
    auto-masking.
    """
    warped = flow_warp(src, flow)
    return (charbonnier(warped - tgt, eps).mean(dim=1, keepdim=True),
            charbonnier(src - tgt, eps).mean(dim=1, keepdim=True))


CHARBONNIER_EPS = 1e-3


def charbonnier(x: torch.Tensor, eps: float = CHARBONNIER_EPS) -> torch.Tensor:
    """Robust L1. Standard in photometric warp losses — a plain L2 lets a few
    occlusion pixels (where the warp is *correctly* undefined) dominate."""
    return torch.sqrt(x * x + eps * eps)


def edge_aware_smoothness(flow: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Penalize flow gradients, discounted where the IMAGE has an edge.

    THIS IS THE APERTURE-PROBLEM DEFENCE, and it is why a flat sky cannot
    become an income source. Over a texture-free region the photometric term
    is satisfied by ANY flow — the field is unidentifiable — so without a
    prior the head would emit arbitrary large values there and manufacture
    residual out of nothing. The smoothness prior has no image edge to excuse
    a gradient on flat sky, so it pulls the field to the locally constant
    (ego) solution and the residual to zero.
    """
    fx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean(1, keepdim=True)
    fy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean(1, keepdim=True)
    ix = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(1, keepdim=True)
    iy = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(1, keepdim=True)
    return ((fx * torch.exp(-ix)).mean() + (fy * torch.exp(-iy)).mean())


# ---------------------------------------------------------------------------
# GEOMETRY THAT IS GIVEN RATHER THAN GUESSED (2026-09-18)
# ---------------------------------------------------------------------------
# In a driving paper, ego-motion has to be regressed because nobody knows it.
# Here it is essentially KNOWN: camera rotation is the commanded action (the
# macro table's +/-15 degree steps, and looking is never blocked), and realised
# translation is the `moved` proprio field along the current heading.
#
# That matters because rotation-induced flow carries NO depth information and
# DOMINATES in first person, while translation-induced flow carries all of it.
# Asked to predict a raw 2-channel field, the head must learn that separation
# from scratch and will mostly learn the rotation, because that is where the
# error mass is. Factoring flow as pi(ego, depth) makes the separation
# structural: the head predicts ONE channel of inverse depth and the geometry
# does the rest.
#
# NO NEW BUFFER COLUMN IS NEEDED. Delta-yaw, delta-pitch and forward distance
# are a FINITE DIFFERENCE of the proprio the buffer already stores (see
# MineRLEnvAdapter.PROPRIO_KEYS: pitch, head_sin, head_cos, moved).

# Index of each field within the env's raw proprio vector. Supplied by the
# LOOP, derived from the env's own PROPRIO_KEYS, so an env with a different
# body supplies its own mapping and this module never has to know one.
#
# The fallback below exists only so a bare WorldModel can be constructed in a
# test without a running environment. It is NOT a default the live path uses
# — developmental_loop derives the real layout from the adapter — and it is
# deliberately not named after any engine, because a constant with an engine
# in its name is an invitation to reason from it.
FALLBACK_PROPRIO_LAYOUT = {"pitch": 9, "moved": 10, "head_sin": 11,
                           "head_cos": 12}


def ego_from_proprio(pp_t: torch.Tensor, pp_t1: torch.Tensor,
                     layout: Dict[str, int],
                     move_scale: float = 1.0) -> torch.Tensor:
    """Body motion between two steps, from stored proprioception alone.

    -> (B, 3) = [d_yaw_rad, d_pitch_rad, forward_distance]

    YAW COMES FROM THE SIN/COS PAIR, NOT A SUBTRACTION. Heading is stored as
    (sin, cos) precisely because it wraps, and `atan2(sin_t1, cos_t1) -
    atan2(sin_t, cos_t)` would jump by 2*pi once per revolution. The rotation
    between two unit vectors is obtained directly:

        sin(d) = sin_t1*cos_t - cos_t1*sin_t
        cos(d) = cos_t1*cos_t + sin_t1*sin_t

    which is continuous everywhere.

    ENGINE-INDEPENDENT BY CONSTRUCTION. This returns the rotation BETWEEN two
    stored headings, which is the same number whichever axis yaw 0 points
    along and whichever way the angle grows — the convention cancels. That is
    why no convention argument appears here, while DeadReckoner (which must
    turn a heading into a direction in the world) takes one. The contract in
    tests/_perspective_smoke.py pins the sign with a known-answer case rather
    than trusting this paragraph.

    THE STORED PAIR IS RESCALED, NOT UNIT (fixed 2026-10-04). The env writes
    head_sin = 0.5*sin(yaw) + 0.5 and head_cos = 0.5*cos(yaw) + 0.5
    (minerl_env._proprio: every proprio entry lives in [0, 1]), so the pair
    is mapped back with 2x - 1 before the rotation formula. Feeding the
    rescaled values straight in — what this did until now — computed the
    angle between two vectors offset by (0.5, 0.5): a 90-degree turn read as
    ~37 degrees, and the error depended on the absolute heading. An unknown
    heading (0.5, 0.5) decodes to (0, 0) and yields d_yaw = 0. INERT LIVE:
    only flow_mode "depth" consumes ego, and the live config runs
    flow_mode: raw — so this fix changes no live number today.

    SIGN, pinned against the world (tests/unit/test_world_model_unit.py, and
    DeadReckoner's convention in tests/_oracle_isolation_smoke.py F):
    Minecraft yaw grows CLOCKWISE seen from above (yaw 0 faces +z/south,
    90 faces -x/west), and d_yaw > 0 exactly when yaw grew — i.e. a turn to
    the RIGHT (south -> west) is positive.

    PITCH is a plain difference: it is clamped to [-90, 90] in the game and
    stored normalized to [0, 1], so it cannot wrap.

    `forward_distance` is `moved` (normalized by MOVE_SCALE in the env)
    re-expanded by `move_scale`. It is a MAGNITUDE — the body's direction of
    travel is its heading — which is correct for this action space: `forward`
    and `back` move along the look direction and there is no strafe macro.
    """
    ps, pm = layout.get("pitch"), layout.get("moved")
    hs, hc = layout.get("head_sin"), layout.get("head_cos")
    if None in (ps, pm, hs, hc):
        raise ValueError(f"proprio layout is missing a field: {layout}")
    # stored as 0.5*v + 0.5 (see above) -> back to [-1, 1]
    sin_t, cos_t = 2.0 * pp_t[:, hs] - 1.0, 2.0 * pp_t[:, hc] - 1.0
    sin_1, cos_1 = 2.0 * pp_t1[:, hs] - 1.0, 2.0 * pp_t1[:, hc] - 1.0
    d_yaw = torch.atan2(sin_1 * cos_t - cos_1 * sin_t,
                        cos_1 * cos_t + sin_1 * sin_t)
    # pitch is stored as (90 - pitch_deg)/180 style normalization in [0,1];
    # the DIFFERENCE is what matters and scales linearly either way.
    d_pitch = (pp_t1[:, ps] - pp_t[:, ps]) * math.pi
    fwd = pp_t1[:, pm] * float(move_scale)
    return torch.stack([d_yaw, d_pitch, fwd], dim=-1)


def flow_from_depth(inv_depth: torch.Tensor, ego: torch.Tensor,
                    fov_rad: float) -> torch.Tensor:
    """Analytic optical flow from inverse depth and known body motion.

    inv_depth: (B, 1, H, W) in (0, 1]   ego: (B, 3)   -> flow (B, 2, H, W)
    in the same normalized grid units flow_warp consumes.

    A deliberately SMALL-ANGLE pinhole model, and the approximation is stated
    rather than buried: per agent step the camera turns at most 15 degrees, so
    tan(theta) ~ theta holds to within ~2%, and the residual is exactly the
    kind of error the photometric loss is there to absorb.

        rotation    x += d_yaw / (fov/2)          — DEPTH-INDEPENDENT, which
                    y += d_pitch / (fov/2)          is the whole point: no
                                                    amount of turning tells you
                                                    how far anything is.
        translation x += fwd * inv_depth * x      — the radial expansion a
                    y += fwd * inv_depth * y        forward step produces, and
                                                    the ONLY depth-bearing term

    The translation term is the "things sweep outward as you advance, faster
    when near" fact the whole design rests on, written as geometry instead of
    left for a network to rediscover. Sign: this is the BACKWARD field
    (where did the content at p come from), matching flow_warp's convention.
    """
    b, _, h, w = inv_depth.shape
    ys = torch.linspace(-1.0, 1.0, h, device=inv_depth.device,
                        dtype=inv_depth.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=inv_depth.device,
                        dtype=inv_depth.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx.unsqueeze(0).unsqueeze(0)
    gy = gy.unsqueeze(0).unsqueeze(0)

    half_fov = max(float(fov_rad) * 0.5, 1e-6)
    d_yaw = ego[:, 0].view(b, 1, 1, 1)
    d_pitch = ego[:, 1].view(b, 1, 1, 1)
    fwd = ego[:, 2].view(b, 1, 1, 1)

    fx = d_yaw / half_fov + fwd * inv_depth * gx
    fy = d_pitch / half_fov + fwd * inv_depth * gy
    return torch.cat([fx, fy], dim=1)


class SensorImageEncoder(nn.Module):
    """A small conv encoder for one IMAGE sensor (the native foveal crop).

    WHY IMAGE SENSORS ARE NOT FLAT-CONCATENATED. The foveal crop is 3,072
    numbers; the body vector is 13. Concatenating them raw would drown the
    body outright — the config already flags a 17:1 compression through one
    Linear as a problem at far gentler ratios, and this would be 236:1.
    Each modality gets its own encoder so they arrive at comparable width and
    the RSSM posterior sees a balanced input rather than a pile of pixels
    with a rounding error attached.

    Deliberately SMALL (three stride-2 stages to 4x4). This is a 32x32 patch
    covering ~6 degrees, not a scene: its job is to report "cracks are
    appearing", "a wireframe is present", "particles are flying", and those
    are texture-scale facts a shallow stack resolves fine.
    """

    def __init__(self, in_channels: int, size: int, out_dim: int = 64):
        super().__init__()
        chans, c_in, layers, cur = [32, 64, 64], in_channels, [], size
        for c_out in chans:
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.ELU()]
            c_in, cur = c_out, (cur + 1) // 2
        self.conv = nn.Sequential(*layers)
        self.out_dim = int(out_dim)
        self.fc = nn.Sequential(nn.Linear(c_in * cur * cur, out_dim), nn.ELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.fc(h.reshape(h.shape[0], -1))


class FlowHead(nn.Module):
    """latent (+ action) -> the displacement field that action would produce.

    ZERO-INITIALIZED OUTPUT, on purpose: at step 0 the prediction is exactly
    zero flow, i.e. "the world is static". That is the correct prior for an
    agent that has not moved yet, and it means the warp starts as the identity
    instead of scrambling the first thousand frames with random resampling.

    `max_flow` bounds the field (via tanh) to a fraction of the half-image, so
    a single bad gradient cannot ask grid_sample for a wrap-around warp.
    """

    def __init__(self, latent_dim: int, action_dim: int, out_size: int = 32,
                 hidden_dim: int = 512, base_channels: int = 128,
                 max_flow: float = 0.5, film_conditioning: bool = True,
                 mode: str = "raw"):
        super().__init__()
        # MODE IS A PARAMETERIZATION, NOT A DIFFERENT HEAD. Both emit a field
        # that the identical warp/loss path consumes, so `raw` and `depth` are
        # A/B-able by one config line and either reverts without touching a
        # checkpoint's shape beyond this head's final conv.
        #   raw   — 2 channels: the displacement field directly.
        #   depth — 1 channel of inverse depth; flow is then computed
        #           analytically from the KNOWN body motion (flow_from_depth).
        if mode not in ("raw", "depth"):
            raise ValueError(f"flow mode must be 'raw' or 'depth', got {mode}")
        self.mode = mode
        out_ch = 2 if mode == "raw" else 1
        if out_size < 4 or (out_size & (out_size - 1)) != 0:
            raise ValueError(f"out_size must be a power of two >= 4, got {out_size}")
        self.out_size = int(out_size)
        self.max_flow = float(max_flow)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        # Action conditions MULTIPLICATIVELY, matching the RSSM's own FiLM
        # path (see FiLMActionConditioner): "what would happen if I did THIS"
        # is a modulation of the scene representation, not a concatenated
        # afterthought a wide Linear can learn to ignore.
        self.film = (FiLMActionConditioner(
            action_dim, hidden_dim, hidden_dim=max(128, action_dim * 16))
            if film_conditioning else None)
        self._action_dim = action_dim
        if not film_conditioning:
            self.act_proj = nn.Linear(hidden_dim + action_dim, hidden_dim)

        self.proj = nn.Linear(hidden_dim, base_channels * 4 * 4)
        self._base_channels = base_channels
        n_up = int(math.log2(out_size // 4))
        layers, c = [], base_channels
        for _ in range(n_up):
            c_out = max(32, c // 2)
            layers += [
                nn.ConvTranspose2d(c, c_out, 3, stride=2, padding=1,
                                   output_padding=1),
                nn.ELU(),
            ]
            c = c_out
        self.deconv = nn.Sequential(*layers)
        self.out = nn.Conv2d(c, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _trunk(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        if self.film is not None:
            x = self.film(action, x)
        else:
            x = self.act_proj(torch.cat([x, action], dim=-1))
        x = self.proj(x).view(-1, self._base_channels, 4, 4)
        return self.out(self.deconv(x))

    def inv_depth(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """-> (B, 1, out_size, out_size) inverse depth in (0, 1].

        SIGMOID-BOUNDED, so depth is always positive and finite — an
        unbounded head can and will emit a negative depth, which is a mirror
        world, and the photometric loss cannot tell it from the real one.

        The zero-init out layer means this starts at sigmoid(0) = 0.5
        everywhere: "the scene is at a uniform middle distance", which is the
        right prior for an agent that has not moved yet and, paired with
        ego = 0, still yields exactly zero flow.
        """
        if self.mode != "depth":
            raise RuntimeError("inv_depth requires mode='depth'")
        return torch.sigmoid(self._trunk(latent, action))

    def forward(self, latent: torch.Tensor, action: torch.Tensor,
                ego: Optional[torch.Tensor] = None,
                fov_rad: float = 1.221730476,
                range_scale: float = 1.0) -> torch.Tensor:
        """-> (B, 2, out_size, out_size), normalized grid units.

        In `depth` mode `ego` is REQUIRED — without the body motion there is
        no way to turn a depth map into a displacement field. Passing None
        there raises rather than silently falling back, because a silent
        fallback would make the mode switch look like it worked while the
        geometry it exists for was never applied.

        ---- range_scale, AND THE BUG IT EXISTS TO FIX (2026-09-18) --------
        `max_flow` bounds the field so one bad gradient cannot ask grid_sample
        for a wrap-around warp. But when the same head is asked to explain a
        STRIDE-k pair, the true displacement is k times larger — and at
        max_flow 0.5 on a 32px field, a 4-step gap in Minecraft is simply
        NOT REPRESENTABLE.

        MEASURED, on the two-plane synthetic: adding strides [1,2,4] with a
        fixed bound took the near/far ratio from 1.60x to 0.86x across three
        seeds — i.e. it did not merely fail to help, it INVERTED the depth
        ordering. An unlearnable stride term does not sit quietly; it drags
        the shared head to a compromise that destroys the stride the agent
        actually acts on.

        So the bound has to grow with the baseline it is bounding. Callers
        pass range_scale = k. This is exactly the kind of "the guard made the
        thing impossible" failure CLAUDE.md 4.1 is about, one level down in
        the tensor math.
        """
        cap = self.max_flow * float(max(range_scale, 1e-6))
        if self.mode == "raw":
            return torch.tanh(self._trunk(latent, action)) * cap
        if ego is None:
            raise ValueError(
                "flow mode 'depth' needs ego motion; pass ego=(B,3) from "
                "ego_from_proprio")
        d = torch.sigmoid(self._trunk(latent, action))
        # NOTE depth mode needs no range_scale in principle — a larger `fwd`
        # in ego already produces a proportionally larger field, which is the
        # factorisation doing its job. The cap still scales so that the clamp
        # does not become the binding constraint at long baselines.
        return flow_from_depth(d, ego, fov_rad).clamp(-cap, cap)


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

    # Diagnostics of the most recent train_step (see COLLECT_STATS); None
    # until the first step. Plain floats only.
    last_step_stats = None
    _stats_logits = None

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
        # ---- PERSPECTIVE FROM MOTION (2026-09-18) ------------------------
        # All five default to the historical behaviour, so every config that
        # does not opt in builds the same model it built before.
        min_grid: int = 4,
        coord_channels: bool = False,
        readout_channels: int = 0,
        flow_head: bool = False,
        flow_size: int = 32,
        flow_photo_size: int = 0,
        flow_mode: str = "raw",
        flow_automask: bool = False,
        flow_scales: Optional[list] = None,
        flow_strides: Optional[list] = None,
        camera_fov_deg: float = 70.0,
        proprio_layout: Optional[Dict[str, int]] = None,
        move_scale: float = 1.0,
        flow_weight: float = 1.0,
        flow_smooth_weight: float = 0.05,
        recon_residual_lambda: float = 0.0,
        latent_horizons: Optional[list] = None,
        latent_horizon_weight: float = 1.0,
        slots: bool = False,
        num_slots: int = 6,
        slot_dim: int = 64,
        slot_iters: int = 3,
        slot_weight: float = 1.0,
        proprio_dim: int = 0,
        sensor_image_specs: Optional[list] = None,
        sensor_feature_dim: int = 64,
        # VRAM, not behaviour. See the block near self.optimizer below.
        # Defaults keep the CPU path (every test on the Mac) byte-identical.
        amp_dtype: Optional[str] = None,
        grad_checkpoint: bool = False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pixel_obs = pixel_obs
        self.image_channels = image_channels
        self.image_size = image_size

        # PROPRIOCEPTION INTO THE WORLD MODEL. The model used to see pixels
        # and the INTENDED action only, which makes "I turned my head" and
        # "the world spun around me" the same observation. Optical flow in a
        # first-person game is dominated by camera rotation, so a flow head
        # without a body sense is being asked to re-derive its own heading
        # from pixels every step. proprio_t (pitch, head_sin/cos, moved, ...)
        # enters the EMBED, i.e. the posterior only — imagination runs on the
        # prior and is therefore untouched by this.
        self.proprio_dim = int(proprio_dim or 0)
        # ---- IMAGE SENSORS (2026-09-19) ----------------------------------
        # [(offset, C, H, W)] into the transport vector. Everything BEFORE
        # the first offset is plain vector sensors and is concatenated
        # directly; each image slice is reshaped and run through its own
        # encoder. Empty list => the transport is all vectors and `embed`
        # behaves exactly as it did in Wave 1.
        self.sensor_image_specs = [tuple(int(x) for x in spec)
                                   for spec in (sensor_image_specs or [])]
        self.sensor_vector_dim = (
            min(spec[0] for spec in self.sensor_image_specs)
            if self.sensor_image_specs else self.proprio_dim)
        self.sensor_encoders = nn.ModuleList([
            SensorImageEncoder(c, h, sensor_feature_dim)
            for (_o, c, h, _w) in self.sensor_image_specs])
        # What the RSSM actually receives from the sensor side: the vector
        # fields verbatim plus one feature block per image sensor.
        self.sensor_embed_dim = (
            self.sensor_vector_dim
            + sensor_feature_dim * len(self.sensor_image_specs))

        # Choose encoder/decoder based on observation type
        if pixel_obs:
            self.encoder = CNNEncoder(
                image_channels, hidden_dim, image_size,
                min_grid=min_grid, coord_channels=coord_channels,
                readout_channels=readout_channels)
            self.decoder = CNNDecoder(
                0, image_channels, hidden_dim, image_size, min_grid=min_grid
            )  # latent_dim set below after RSSM init
        else:
            self.encoder = ObservationEncoder(obs_dim, hidden_dim)

        self.rssm = RSSM(
            # Encoder output dim + proprio. NOTE this is why nothing may read
            # `rssm.obs_dim` as "the encoder's width" any more — read
            # `encoder.vec_dim` for that.
            obs_dim=hidden_dim + self.sensor_embed_dim,
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
                self.rssm.latent_dim, image_channels, hidden_dim, image_size,
                min_grid=min_grid
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

        # --- Flow head (perspective from motion) ---------------------------
        # Pixel observations only: there is no image to warp otherwise.
        self.flow_enabled = bool(flow_head) and bool(pixel_obs)
        self.flow_weight = float(flow_weight)
        self.flow_smooth_weight = float(flow_smooth_weight)
        # PHOTOMETRIC RESOLUTION, separate from the field's own resolution.
        # The warp comparison is the memory hot spot in this loss: at B=16,
        # L=32 there are 496 frame pairs per batch, and holding warped+target
        # at full 128px would cost ~100 MB per tensor on top of the ~100 MB
        # the reconstruction already materializes. Parallax and depth ordering
        # are low-frequency facts — halving the resolution quarters the memory
        # and costs nothing that matters here. 0 -> min(64, image_size).
        self.flow_photo_size = int(flow_photo_size or min(64, image_size))
        # ---- WAVE 2 KNOBS, every default reproducing Wave 1 exactly -------
        self.flow_mode = str(flow_mode)
        self.flow_automask = bool(flow_automask)
        # Scales DIVIDE flow_photo_size; strides are frame gaps. [1] / [1] is
        # the Wave 1 loss term for term, which is what makes contract P
        # ("every switch reverts") assertable rather than aspirational.
        self.flow_scales = [int(x) for x in (flow_scales or [1]) if int(x) >= 1]
        self.flow_strides = [int(x) for x in (flow_strides or [1]) if int(x) >= 1]
        if not self.flow_scales:
            self.flow_scales = [1]
        if not self.flow_strides:
            self.flow_strides = [1]
        # Minecraft's default vertical FOV is 70 degrees. A DECLARED SENSOR
        # CONSTANT, and the honest note about it: getting this wrong rescales
        # every depth uniformly and PRESERVES ORDERING, and every consumer
        # (flow_fovea, flow_edge, flow_ratio) reads only relative magnitude —
        # so a mis-set FOV is invisible to everything downstream today. It
        # would matter the moment a metric distance is wanted.
        self.camera_fov = math.radians(float(camera_fov_deg))
        self.proprio_layout = dict(proprio_layout or FALLBACK_PROPRIO_LAYOUT)
        self.move_scale = float(move_scale)
        # RESIDUAL-WEIGHTED RECONSTRUCTION. Flat per-pixel MSE over a 128x128
        # frame is dominated by sky, ground and texture; a three-block trunk is
        # rounding error, which is the documented reason autoencoder world
        # models fail to perceive small objects. Weighting recon by where the
        # warp FAILED spends capacity where motion says something is there.
        #
        # ESCAPE PATH (CLAUDE.md 4.1): the weight is bounded in [1, 1+lambda]
        # and never reaches zero, so no region of the frame can ever become
        # permanently unlearnable — the failure mode a naive "only train on
        # the interesting pixels" mask would have. lambda=0 restores the
        # historical loss exactly and is the revert.
        self.recon_residual_lambda = float(recon_residual_lambda)
        # ---- MULTI-HORIZON LATENT PREDICTION (2026-09-19, roadmap P1) -----
        # The model is trained to predict ONE step ahead. Everything it knows
        # about the future beyond that is whatever one-step accuracy happens
        # to imply — and the policy trains on IMAGINED ROLLOUTS fifteen steps
        # long, so the horizon the dreams live at has never been supervised.
        #
        # This adds a term at each extra horizon k: roll the PRIOR forward k
        # steps with the actions actually taken, and charge the divergence
        # from the posterior that actually resulted. More supervision per
        # frame from unchanged data, which is the one answer to data
        # starvation that needs no new data.
        #
        # WHY THIS IS NOT THE MULTI-STRIDE WARP THAT FAILED. Wave 2 measured
        # that warping t->t+k from a shared flow head INVERTED depth ordering,
        # because one field cannot represent k times the displacement while
        # `max_flow` bounds it. Latent rollout has no such representability
        # bound — the RSSM's job already IS iterated transition, so a longer
        # horizon asks it to do more of exactly what it does.
        #
        # [] reproduces the one-step objective exactly.
        self.latent_horizons = sorted({
            int(k) for k in (latent_horizons or []) if int(k) >= 2})
        self.latent_horizon_weight = float(latent_horizon_weight)
        # ---- OBJECT SLOTS (2026-09-19, roadmap R1-R4) --------------------
        # Slots DECODE THE FLOW FIELD, never pixels: DINOSAUR found pixel
        # targets make slots latch onto colour and texture, and Minecraft is
        # maximally textured, so a pixel-target slot model here would
        # confidently segment grass noise. SAVi showed flow is the target
        # that works, because what moves together IS an object — which is
        # this project's own thesis one level down.
        #
        # THEREFORE GATED ON THE FLOW HEAD. Without a flow field that has
        # been shown to learn something on live frames there is no
        # reconstruction target, and this is a slot model trained on noise.
        # SAVi's own stated limitation is that fully-unsupervised
        # decomposition "still fails to scale to diverse realistic data".
        self.slots_enabled = bool(slots) and bool(pixel_obs) and self.flow_enabled
        self.num_slots = int(num_slots)
        self.slot_weight = float(slot_weight)
        self.slot_module = None
        if self.slots_enabled:
            from developmental_ai.slots import SAViSlots
            self.slot_module = SAViSlots(
                feat_dim=self.encoder.map_channels,
                num_slots=num_slots, slot_dim=slot_dim, iters=slot_iters,
                grid=self.encoder.grid, flow_size=flow_size,
                max_flow=0.5)
        self.flow_head = (
            FlowHead(self.rssm.latent_dim, action_dim, out_size=flow_size,
                     hidden_dim=hidden_dim, film_conditioning=film_conditioning,
                     mode=self.flow_mode)
            if self.flow_enabled else None
        )

        # ---- MIXED PRECISION / CHECKPOINTING (2026-09-24) -----------------
        # There was NO mixed precision anywhere in this repo: training was
        # full fp32. On `main` (RTX 5050, 7.56 GiB usable) that does not fit --
        # torch held 6.66 GiB and the supervisor stopped the run after the same
        # CUDA OOM three times. bf16 halves activation memory (measured
        # 50.3 MB -> 25.2 MB for the same tensor).
        # bf16 AND NOT fp16: bf16 keeps fp32's EXPONENT RANGE, so the underflow
        # that GradScaler exists to fix does not arise and no scaler is needed.
        # CUDA-ONLY BY CONSTRUCTION. Every test on the Mac runs on CPU; if
        # autocast were live there, every contract asserting a loss value would
        # move for reasons unrelated to what it is testing.
        _amp = str(amp_dtype or "off").lower()
        self.amp_dtype = torch.bfloat16 if _amp in ("bf16", "bfloat16") else None
        # Time-for-memory on the decoder only, OFF by default: turn it on if
        # bf16 alone does not fit. 40-60% of activation memory for ~25-30%
        # more compute, and mathematically identical.
        self.grad_checkpoint = bool(grad_checkpoint)

        # Optimizer for all world model parameters (includes inverse head if present)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _reward_symlog(self, latent: torch.Tensor) -> torch.Tensor:
        """Expected reward in SYMLOG space from the twohot head. (...,)."""
        return twohot_decode(self.reward_predictor(latent), self.reward_bins)

    def predict_reward(self, latent: torch.Tensor) -> torch.Tensor:
        """Real-scale reward prediction from the twohot head. (...,)."""
        return symexp(self._reward_symlog(latent))

    def as_images(self, observations: torch.Tensor) -> torch.Tensor:
        """Flat pixel rows -> (N, C, H, W). Raises on non-pixel observations."""
        if not self.pixel_obs:
            raise RuntimeError("as_images is only meaningful for pixel_obs models")
        return observations.reshape(-1, self.image_channels,
                                    self.image_size, self.image_size)

    def embed(self, observations: torch.Tensor,
              proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encoder vector (+ proprio) — exactly what the RSSM posterior reads.

        A model built with proprio_dim > 0 and handed no proprio reads NEUTRAL
        ZEROS rather than raising, the same convention every other missing
        sense in this project uses (see _augment_proprio). That keeps a scout
        stream, a test harness and a restored checkpoint from having to
        fabricate a body they do not have.
        """
        vec = self.encoder(observations)
        if self.proprio_dim <= 0:
            return vec
        if proprio is None:
            proprio = vec.new_zeros(vec.shape[0], self.proprio_dim)
        return self._fuse_sensors(vec, proprio)

    def _fuse_sensors(self, vec: torch.Tensor,
                      proprio: torch.Tensor) -> torch.Tensor:
        """Encoder vector + sensor transport -> the RSSM's observation.

        ONE implementation, two callers (embed and observe_sequence's
        map-capturing path). Splitting it would be the tensor-level form of
        the duplicated-body hazard: the ORDER here must match
        SensorBus.read_policy exactly, and two copies of that ordering is two
        chances to get it wrong.
        """
        if not self.sensor_image_specs:
            return torch.cat([vec, proprio], dim=-1)
        # Vector fields pass through; each image slice is reshaped to its
        # declared (C, H, W) and encoded. The ORDER here must match
        # SensorBus.read_policy (vectors first, then images in registration
        # order) — that ordering is what `layout_hash` pins.
        parts = [vec, proprio[:, :self.sensor_vector_dim]]
        for enc, (off, c, h, w) in zip(self.sensor_encoders,
                                       self.sensor_image_specs):
            sl = proprio[:, off:off + c * h * w]
            parts.append(enc(sl.reshape(-1, c, h, w)))
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def flow_probe(self, latent: torch.Tensor, probe_action: torch.Tensor,
                   fovea_frac: float = 0.4,
                   ego: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """THREE GEOMETRIC SENSES from one counterfactual query. Batch of 1.

        `probe_action` is a question, not a record: the loop passes a
        walk-forward action and asks "if I stepped forward from here, how
        would the scene sweep past me?".

          flow_fovea  magnitude at the CENTRE. Under a forward translation,
                      flow magnitude is inverse depth — near things sweep
                      fast. This is the counterfactual form of "walk at it and
                      watch it grow", available without walking.
          flow_edge   the same for the PERIPHERY. The centre/edge gradient is
                      the geometry of heading itself: content toward the
                      direction of travel barely moves while the edges sweep
                      hardest.
          flow_ratio  fovea / (fovea + edge). Figure-ground: 0.5 means the
                      attended thing is at the same depth as its surround,
                      higher means it stands nearer — an object, not backdrop.

        SENSES, NOT REWARDS. These enter proprioception and nothing else, on
        the doctrine _reach_sense states.
        """
        out = {"flow_fovea": 0.0, "flow_edge": 0.0, "flow_ratio": 0.5}
        if not self.flow_enabled or self.flow_head is None:
            return out
        if self.flow_mode == "depth" and ego is None:
            # THE PROBE IS A COUNTERFACTUAL, so its ego is the one the probe
            # ACTION would produce, not the one that just happened: "if I
            # walked one step forward from here". A unit forward translation
            # with no rotation is exactly that question, and it is also the
            # ego under which flow magnitude IS inverse depth.
            ego = latent.new_tensor([[0.0, 0.0, 1.0]])
        flow = self.flow_head(latent, probe_action, ego=ego,
                              fov_rad=self.camera_fov)            # (1, 2, S, S)
        mag = flow.norm(dim=1, keepdim=True) / max(self.flow_head.max_flow, 1e-6)
        s_side = mag.shape[-1]
        half = max(1, int(round(s_side * float(fovea_frac) / 2.0)))
        c0, c1 = s_side // 2 - half, s_side // 2 + half
        centre = mag[:, :, c0:c1, c0:c1]
        n_all, n_c = mag.numel(), centre.numel()
        # Periphery without materializing a mask: frame total minus centre.
        edge_mean = ((mag.sum() - centre.sum()) / max(1, n_all - n_c)).clamp(0, 1)
        fov_mean = centre.mean().clamp(0, 1)
        out["flow_fovea"] = float(fov_mean)
        out["flow_edge"] = float(edge_mean)
        out["flow_ratio"] = float(fov_mean / (fov_mean + edge_mean + 1e-6))
        return out

    @torch.no_grad()
    def probe_map(self, latent: torch.Tensor, probe_action: torch.Tensor,
                  ego: Optional[torch.Tensor] = None,
                  map_size: int = 16) -> Optional[torch.Tensor]:
        """"If I stepped forward, how fast would each part of the scene sweep
        past me?" -> (B, 1, map_size, map_size) in [0, 1]. Roadmap G1.

        WHY THIS WORKS IN BOTH MODES, which is what lets it ship before the
        depth factorisation has been validated on live frames:

          flow_mode: depth  the field IS pi(ego, depth), so magnitude under a
                            pure forward translation is inverse depth exactly.
          flow_mode: raw    the head predicts the field directly, and under a
                            forward probe the magnitude is whatever it has
                            LEARNED about how fast things sweep — the same
                            quantity, discovered rather than constructed.

        Either way the ordering is the useful part: near sweeps fast, far
        sweeps slow. `flow_probe` already reduces this to three scalars for
        proprioception; those three are a 341:1 compression of the first
        continuous, spatially-resolved quantity this system has ever had,
        which is the plurality failure happening to our own new sensor.
        """
        if not self.flow_enabled or self.flow_head is None:
            return None
        if self.flow_mode == "depth" and ego is None:
            # The counterfactual is "one step forward, no rotation" — which
            # is also the only ego under which magnitude IS inverse depth.
            ego = latent.new_tensor([[0.0, 0.0, 1.0]]).expand(
                latent.shape[0], 3)
        flow = self.flow_head(latent, probe_action, ego=ego,
                              fov_rad=self.camera_fov)
        mag = flow.norm(dim=1, keepdim=True) / max(self.flow_head.max_flow, 1e-6)
        if mag.shape[-1] != map_size:
            mag = F.adaptive_avg_pool2d(mag, map_size)
        return mag.clamp(0.0, 1.0)

    @torch.no_grad()
    def flow_residual(self, prev_obs: torch.Tensor, obs: torch.Tensor,
                      prev_latent: torch.Tensor, taken_action: torch.Tensor,
                      motion_floor: float = 0.004,
                      valid: Optional[torch.Tensor] = None,
                      ego: Optional[torch.Tensor] = None,
                      return_map: bool = False,
                      map_size: int = 8):
        """Per-sample fraction of the frame change the warp could NOT explain.

        -> (B,) in [0, 1]. 0 = the action-conditioned warp accounted for
        everything that changed (or nothing changed at all); 1 = the change is
        entirely unexplained by the agent's own movement, which is what a mover
        — another player, a mob, falling leaves — looks like.

        SCALE-FREE BY CONSTRUCTION. The numerator is the photometric error
        after warping, the denominator the error of NOT warping at all. A ratio
        rather than a magnitude, so a dim cave and a bright field are on the
        same footing and no exposure change can be farmed.

        THE ZERO CONTRACT, and why `motion_floor` is not a latch (CLAUDE.md
        4.1). If the image did not change, the denominator is ~0, there is
        nothing to explain, and this returns exactly 0. That is the property
        that makes this safe to feed the curiosity channel: no ego-motion ->
        no change -> no residual -> no income, structurally. `motion_floor`
        gates on an INSTANTANEOUS measurement and re-opens on the very next
        frame in which anything moves; nothing has to notice, and no human is
        in the loop.

        `valid` masks out streams whose previous frame belongs to a DIFFERENT
        WORLD (a reset, a death, a client rebuild). Without it, every episode
        boundary would report a near-total residual and read as the most
        interesting thing that ever happened.
        """
        b = obs.shape[0]
        zeros = obs.new_zeros(b)
        zmap = obs.new_zeros(b, 1, map_size, map_size)
        if (not self.flow_enabled or self.flow_head is None
                or not self.pixel_obs):
            return (zeros, zmap) if return_map else zeros
        ps = self.flow_photo_size
        a = self.as_images(prev_obs)
        c = self.as_images(obs)
        if ps != self.image_size:
            a = F.interpolate(a, size=(ps, ps), mode="area")
            c = F.interpolate(c, size=(ps, ps), mode="area")
        if self.flow_mode == "depth" and ego is None:
            # No body reading for this pair — the residual would be measured
            # against a geometry we cannot construct. Zero is the honest
            # answer ("no evidence of unexplained motion"), not a guess.
            return (zeros, zmap) if return_map else zeros
        f = self.flow_head(prev_latent, taken_action, ego=ego,
                           fov_rad=self.camera_fov)
        if f.shape[-1] != ps:
            f = F.interpolate(f, size=(ps, ps), mode="bilinear",
                              align_corners=True)
        # SAME HELPER THE TRAINING LOSS USES, so "how wrong was the warp" and
        # "how much changed at all" mean the same thing in both places.
        warp_err, id_err = photometric(a, c, f)
        # ---- THE CHARBONNIER FLOOR IS NOT MOTION (fixed 2026-09-19) ------
        # charbonnier(0) = eps = 1e-3, so a pixel where NOTHING changed
        # reports 1e-3 of "error" from both the warp and the identity, and
        # their ratio is 1e-3/(1e-3+1e-6) ~ 0.999. The scalar hid this behind
        # `motion_floor`, which gates on the frame mean; the per-cell map has
        # no such shelter, and reported 1.00 surprise for a completely static
        # region — manufacturing curiosity out of stillness, which is the one
        # failure mode this whole project keeps paying for.
        #
        # Subtracting the floor makes "nothing moved" read as zero MOTION
        # rather than as a small amount of unexplained motion.
        _floor = float(charbonnier(torch.zeros(1), CHARBONNIER_EPS).item())
        warp_err = (warp_err - _floor).clamp(min=0.0)
        id_err = (id_err - _floor).clamp(min=0.0)
        dims = (1, 2, 3)
        resid_warp = warp_err.mean(dim=dims)                          # (B,)
        resid_identity = id_err.mean(dim=dims)                        # (B,)
        out = (resid_warp / (resid_identity + 1e-6)).clamp(0.0, 1.0)
        gate = resid_identity >= float(motion_floor)
        out = torch.where(gate, out, zeros)
        if valid is not None:
            gate = gate & valid.reshape(-1).to(torch.bool)
            out = torch.where(valid.reshape(-1).to(torch.bool), out, zeros)
        if not return_map:
            return out
        # ---- WHERE the warp failed, not just how much (roadmap P4) -------
        # The per-pixel error is already computed above and was being
        # averaged away. Curiosity has therefore always been SCENE-LEVEL:
        # the agent could be surprised, but never surprised BY A PLACE IN
        # THE FRAME — which is why the vision magnet can only want a
        # category and never an instance.
        #
        # Same scale-free ratio as the scalar, per cell: the local warp error
        # over the local frame change. A cell where nothing moved reads 0 for
        # the same reason standing still pays nothing.
        # POOL FIRST, THEN DIVIDE. A per-pixel ratio averaged over a cell is
        # not the cell's ratio — one bright pixel with a tiny denominator
        # would dominate the average. Pooling the two errors and dividing the
        # pooled values is the cell-level version of the same scale-free
        # quantity the scalar uses frame-wide.
        we_p = F.adaptive_avg_pool2d(warp_err, map_size)
        ie_p = F.adaptive_avg_pool2d(id_err, map_size)
        pm = (we_p / (ie_p + 1e-6)).clamp(0.0, 1.0)
        # PER-CELL motion gate, the local form of the frame-level one: a cell
        # that did not change has nothing to explain and reads exactly 0.
        pm = torch.where(ie_p >= float(motion_floor), pm,
                         torch.zeros_like(pm))
        pm = pm * gate.reshape(-1, 1, 1, 1).to(pm.dtype)
        return out, pm

    @torch.no_grad()
    def perspective_senses(self, latent: torch.Tensor, probe_action: torch.Tensor,
                           prev_obs: Optional[torch.Tensor] = None,
                           obs: Optional[torch.Tensor] = None,
                           prev_latent: Optional[torch.Tensor] = None,
                           taken_action: Optional[torch.Tensor] = None,
                           fovea_frac: float = 0.4,
                           motion_floor: float = 0.004) -> Dict[str, float]:
        """flow_probe + flow_residual as one dict, for a batch of 1.

        The loop computes the two halves at different points in the step (the
        residual with the curiosity channel, the probe after the RSSM observe)
        and does NOT use this; it exists so a caller that just wants all four
        numbers — a test, a viewer, a probe script — has one entry point that
        cannot get the pairing wrong.
        """
        out = dict(self.flow_probe(latent, probe_action, fovea_frac))
        out["mover"] = 0.0
        if (prev_obs is not None and obs is not None
                and prev_latent is not None and taken_action is not None):
            out["mover"] = float(self.flow_residual(
                prev_obs, obs, prev_latent, taken_action,
                motion_floor=motion_floor)[0])
        return out

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        return_maps: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a sequence of real experience through the world model.

        Args:
            observations: (batch, seq_len, obs_dim) — real observations
            actions: (batch, seq_len, action_dim) — actions taken
            proprio: (batch, seq_len, proprio_dim) — the agent's own body
                state at each step, or None for neutral zeros

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
        fmap = None
        if return_maps and self.pixel_obs:
            # The slot head needs the encoder's SPATIAL map, which embed()
            # discards. Captured here rather than by re-encoding, which would
            # double the encoder cost for the same tensor.
            _vec, fmap = self.encoder.encode_with_map(
                observations.reshape(-1, self.obs_dim))
            if self.proprio_dim > 0:
                _pp = (proprio.reshape(-1, self.proprio_dim)
                       if proprio is not None
                       else _vec.new_zeros(_vec.shape[0], self.proprio_dim))
                encoded = self._fuse_sensors(_vec, _pp)
            else:
                encoded = _vec
        else:
            encoded = self.embed(
                observations.reshape(-1, self.obs_dim),
                None if proprio is None
                else proprio.reshape(-1, self.proprio_dim))
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

        if return_maps:
            infos["feature_map"] = fmap
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
        proprio: Optional[torch.Tensor] = None,
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
        states, infos = self.observe_sequence(
            observations, actions, proprio,
            return_maps=self.slots_enabled)

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

        # 0. PERSPECTIVE FROM MOTION — the action-conditioned warp.
        #
        # For each consecutive pair (o_t, o_{t+1}) predict the displacement
        # field that action a_t produced, warp o_t through it, and charge the
        # photometric error. Nothing labels the flow; novel-view synthesis is
        # the supervision. See the FlowHead block above for why this replaces
        # the behavioural "walk at it and watch the radius grow" formulation.
        #
        # PAIRING. `actions` here is the CALLER's unshifted tensor —
        # observe_sequence shifts only its own local copy — so actions[:, t] is
        # the action taken AT t, which is the one that carries o_t to o_{t+1}.
        # This is the same alignment the causal comment above describes for
        # imagine_step(state_t, a_t) -> o_{t+1}. With causal_align=false the
        # RSSM's own pairing changes but this one does not, which is correct:
        # the warp is a fact about the frames, not about the RSSM's convention.
        flow_loss = observations.new_zeros(())
        flow_smooth_loss = observations.new_zeros(())
        resid_map = None
        mask_frac = 1.0
        if self.flow_enabled and seq_len >= 2:
            imgs = self.as_images(observations).view(
                batch_size, seq_len, self.image_channels,
                self.image_size, self.image_size)
            # Running action sums, so the stride-k window's action input is
            # SUM(a_t .. a_{t+k-1}) in one slice instead of a Python loop.
            # That sum encodes both WHICH actions and HOW MANY steps, and
            # reduces to the plain one-hot at k=1 — which is why enabling
            # longer baselines does not change what k=1 means.
            acts_cum = torch.cat(
                [actions.new_zeros(batch_size, 1, actions.shape[-1]),
                 torch.cumsum(actions, dim=1)], dim=1)
            pp = None
            if proprio is not None and self.flow_mode == "depth":
                pp = proprio

            n_terms = 0
            mask_keep, mask_total = 0.0, 0.0
            for k in self.flow_strides:
                if k < 1 or seq_len <= k:
                    continue
                lat_k = latents[:, :seq_len - k].reshape(-1, latents.shape[-1])
                act_k = (acts_cum[:, k:seq_len] - acts_cum[:, :seq_len - k]
                         ).reshape(-1, actions.shape[-1])
                ego_k = None
                if pp is not None:
                    ego_k = ego_from_proprio(
                        pp[:, :seq_len - k].reshape(-1, pp.shape[-1]),
                        pp[:, k:seq_len].reshape(-1, pp.shape[-1]),
                        self.proprio_layout, self.move_scale)
                # ONE head forward per stride; the scales below resample the
                # SAME field rather than predicting it again.
                # range_scale = k: the bound must grow with the baseline or
                # the stride term is unrepresentable and poisons the head.
                flow_k = self.flow_head(lat_k, act_k, ego=ego_k,
                                        fov_rad=self.camera_fov,
                                        range_scale=float(k))
                src_k = imgs[:, :seq_len - k].reshape(
                    -1, self.image_channels, self.image_size, self.image_size)
                tgt_k = imgs[:, k:seq_len].reshape(
                    -1, self.image_channels, self.image_size, self.image_size)

                for sc in self.flow_scales:
                    ps = max(8, self.flow_photo_size // max(1, int(sc)))
                    src = (src_k if ps == self.image_size else
                           F.interpolate(src_k, size=(ps, ps), mode="area"))
                    tgt = (tgt_k if ps == self.image_size else
                           F.interpolate(tgt_k, size=(ps, ps), mode="area"))
                    fl = (flow_k if flow_k.shape[-1] == ps else
                          F.interpolate(flow_k, size=(ps, ps), mode="bilinear",
                                        align_corners=True))
                    warp_err, id_err = photometric(src, tgt, fl)

                    if self.flow_automask:
                        # ---- AUTO-MASKING (monodepth2) -------------------
                        # A pixel whose error is WORSE after warping is better
                        # explained by "nothing moved": it is occluded, or it
                        # is texture that travels with the camera. Charging it
                        # as error supervises the field toward a wrong answer,
                        # and those pixels cluster at depth discontinuities —
                        # which in this world means trunk edges, the one place
                        # the geometry has to be right.
                        #
                        # `<=` AND A TIE EPSILON ARE NOT COSMETIC. At init the
                        # head emits zero flow, so warp_err == id_err exactly;
                        # a strict `<` would mask EVERY pixel, make the loss
                        # identically zero, and the head could never produce a
                        # gradient to escape with. That is a guard-becomes-
                        # latch (CLAUDE.md 4.1) of the purest kind. With `<=`
                        # the mask starts at 100% retained and only ever drops
                        # pixels the head has actively made worse.
                        #
                        # ESCAPE: the retained fraction is reported as
                        # `flow_mask` in the losses dict, so a collapse is
                        # visible in telemetry rather than silent. Setting
                        # flow_automask=false restores the unmasked loss.
                        keep = (warp_err <= id_err + 1e-6).to(warp_err.dtype)
                        mask_keep += float(keep.sum().detach())
                        mask_total += float(keep.numel())
                        term = ((warp_err * keep).sum()
                                / keep.sum().clamp(min=1.0))
                    else:
                        term = warp_err.mean()
                    flow_loss = flow_loss + term
                    n_terms += 1

                    if sc == self.flow_scales[0]:
                        tgt_s = (tgt if tgt.shape[-1] == flow_k.shape[-1] else
                                 F.interpolate(tgt, size=flow_k.shape[-2:],
                                               mode="area"))
                        flow_smooth_loss = flow_smooth_loss + \
                            edge_aware_smoothness(flow_k, tgt_s)
                    # THE RESIDUAL MAP IS STRIDE 1, FINEST SCALE, ALWAYS.
                    # It weights RECONSTRUCTION, which is a per-frame loss, so
                    # it has to describe adjacent frames at full resolution —
                    # a stride-4 residual would smear four steps of motion
                    # across one frame's weights.
                    if k == 1 and sc == self.flow_scales[0]:
                        resid_map = warp_err.detach().view(
                            batch_size, seq_len - 1, 1, ps, ps)

            if n_terms:
                flow_loss = flow_loss / n_terms
                flow_smooth_loss = flow_smooth_loss / max(
                    1, len(self.flow_strides))
            if mask_total > 0:
                mask_frac = mask_keep / mask_total

        # 1. Reconstruction loss: decode latent → predicted observation
        # OPTIONAL TIME-FOR-MEMORY ON THE DECODER (2026-09-24). Off unless
        # world_model.grad_checkpoint is set. Both fatal CUDA OOMs on `main`
        # landed in conv_transpose2d and the decoder is 71.7M of 120.5M params,
        # so this is where recomputation pays.
        # use_reentrant=False is REQUIRED: the reentrant path mishandles the
        # no-grad/eval calls this same decoder serves (imagination, below).
        # Safe to wrap ONLY the decoder -- CNNDecoder is deterministic, so
        # there is no RNG state to replay; the RSSM's stochastic sampling
        # happens outside this call.
        if (self.grad_checkpoint and self.training
                and latents_flat.requires_grad):
            from torch.utils.checkpoint import checkpoint as _ckpt
            _dec = _ckpt(self.decoder, latents_flat, use_reentrant=False)
        else:
            _dec = self.decoder(latents_flat)
        pred_obs = _dec.view(batch_size, seq_len, -1)
        recon_target = observations if self.pixel_obs else symlog(observations)
        recon_per_elem = F.mse_loss(pred_obs, recon_target, reduction="none")
        # RESIDUAL-WEIGHTED RECONSTRUCTION. A flat per-pixel MSE over a 128px
        # frame is dominated by sky, ground and repeated texture; a three-block
        # trunk contributes rounding error, which is the documented reason
        # autoencoder world models fail to perceive small objects. Weighting by
        # where the warp FAILED spends capacity where motion says something is
        # there. The weight is bounded in [1, 1+lambda] and never reaches zero
        # (CLAUDE.md 4.1 — a mask that can reach zero is a region the model can
        # never recover), and the batch mean is divided out so lambda changes
        # the DISTRIBUTION of reconstruction effort, not its total magnitude
        # (otherwise lambda would silently double as a learning-rate knob).
        if (self.recon_residual_lambda > 0.0 and resid_map is not None
                and self.pixel_obs):
            r = resid_map
            # Robust per-frame scale: a pixel at >= 2x its frame's mean
            # residual gets the full weight. Not amax — one specular pixel
            # would flatten every other weight to nothing.
            scale = 2.0 * r.mean(dim=(2, 3, 4), keepdim=True) + 1e-6
            rn = (r / scale).clamp(0.0, 1.0)
            if rn.shape[-1] != self.image_size:
                rn = F.interpolate(
                    rn.view(-1, 1, rn.shape[-2], rn.shape[-1]),
                    size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False
                ).view(batch_size, seq_len - 1, 1, self.image_size, self.image_size)
            wmap = rn.new_ones(batch_size, seq_len, 1,
                               self.image_size, self.image_size)
            # The residual of pair t is evidence about content in o_{t+1}, so
            # it weights FRAME t+1. Frame 0 has no pair and keeps weight 1.
            wmap[:, 1:] = 1.0 + self.recon_residual_lambda * rn
            # BROADCAST, DO NOT EXPAND-THEN-RESHAPE (2026-09-24).
            # `expand` is a free view, but `.reshape()` on an expanded tensor
            # FORCES A COPY -- and this weight is IDENTICAL across channels, so
            # the old line copied the same values 3x. With its two temporaries
            # that is ~300 MB transient at batch 16 (measured 50.3 MB x 3), on
            # the exact line the run died at:
            #   rssm.py:2095 recon_per_elem = recon_per_elem * (wflat/...)
            #   torch.OutOfMemoryError: Tried to allocate 96.00 MiB
            # MATHEMATICALLY identical, NOT bit-identical -- measured, after
            # this comment first claimed otherwise. `expand` replicates the
            # same values, so wflat.mean() and wmap.mean() are the same
            # quantity, but the old form reduces 3x MORE ELEMENTS and float32
            # rounds differently: max abs difference in the per-sequence loss
            # is 1.19e-07, i.e. fp32 epsilon. That is ~six orders of magnitude
            # below the bf16 change landing alongside it, and it is recorded
            # here rather than rounded up to "identical".
            # The reduction below takes the mean over every non-batch dim, so
            # the extra channel axis here needs no reshape back.
            _w = wmap.reshape(batch_size, seq_len, 1, -1)
            recon_per_elem = recon_per_elem.view(
                batch_size, seq_len, self.image_channels, -1
            ) * (_w / wmap.mean())
        per_seq_recon = recon_per_elem.mean(dim=tuple(range(1, recon_per_elem.dim())))
        if w is not None:
            recon_loss = (w * per_seq_recon).sum() / (w.sum() + 1e-8)
        else:
            recon_loss = per_seq_recon.mean()

        # 2. KL divergence loss: how surprised was the model?
        # ---- fp32 ISLAND (2026-09-24) -------------------------------------
        # The KL is where reduced mantissa bites: free_nats clamps at a FLOOR
        # of ~0.1, and a measurement pinned at exactly 0.1100 for every config
        # and seed is how the horizon defect hid last time. Differences that
        # small must not be an artefact of the dtype. `enabled=False` is a
        # no-op when autocast was never on (CPU), so this costs nothing there.
        with torch.autocast("cuda", enabled=False):
            kl_loss = self.rssm.compute_kl_loss(
                infos["prior_logits"].reshape(-1, self.rssm.stoch_dim).float(),
                infos["posterior_logits"].reshape(
                    -1, self.rssm.stoch_dim).float(),
            )
        if COLLECT_STATS and self.training:
            # Detached references only (no copy, no sync); train_step consumes
            # and clears them. Not part of `losses`, whose every value is
            # .item()'d by train_step.
            self._stats_logits = (
                infos["prior_logits"].detach(),
                infos["posterior_logits"].detach())

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
        # fp32 ISLAND: the twohot reward target interpolates between bin edges
        # (symlog buckets, top of this file), and the class-balancing weight
        # can reach 100x. A weighted sum over both in bf16 is exactly the kind
        # of reduction that loses the small-reward tail this agent lives on.
        with torch.autocast("cuda", enabled=False):
            reward_loss = ((rw_elem.float() * reward_per_elem.float()).sum()
                           / rw_elem.float().sum().clamp(min=1e-8))

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

        # ---- R1: what the SLOTS collectively explain ---------------------
        # The slots are trained by the SAME photometric warp the flow head
        # is: their combined field must carry frame t into frame t+1. No
        # segmentation label exists anywhere in this objective — a slot earns
        # its region by explaining the motion in it, which is the only kind
        # of objectness this project permits.
        slot_loss = observations.new_zeros(())
        if (self.slots_enabled and self.slot_module is not None
                and seq_len >= 2 and infos.get("feature_map") is not None):
            fm = infos["feature_map"]
            g = self.encoder.grid
            fm = fm.view(batch_size, seq_len, -1, g, g)
            ps = self.flow_photo_size
            imgs_s = self.as_images(observations).view(
                batch_size, seq_len, self.image_channels,
                self.image_size, self.image_size)
            st = self.slot_module.initial(fm[:, 0])
            n_s = 0
            for t in range(seq_len - 1):
                out_t = self.slot_module.step(fm[:, t], st)
                st = {"slots": out_t["slots"], "identity": out_t["identity"]}
                src = imgs_s[:, t]
                tgt = imgs_s[:, t + 1]
                if ps != self.image_size:
                    src = F.interpolate(src, size=(ps, ps), mode="area")
                    tgt = F.interpolate(tgt, size=(ps, ps), mode="area")
                fl = out_t["pred_flow"]
                if fl.shape[-1] != ps:
                    fl = F.interpolate(fl, size=(ps, ps), mode="bilinear",
                                       align_corners=True)
                we, ie = photometric(src, tgt, fl)
                if self.flow_automask:
                    keep = (we <= ie + 1e-6).to(we.dtype)
                    slot_loss = slot_loss + (we * keep).sum() / keep.sum().clamp(min=1.0)
                else:
                    slot_loss = slot_loss + we.mean()
                n_s += 1
            if n_s:
                slot_loss = slot_loss / n_s

        # ---- P1: horizons beyond one step --------------------------------
        # STARTS ARE STRIDED BY k, not taken at every t. Every-t would cost
        # k times a full observe pass per horizon; striding costs about one,
        # and the starts still tile the sequence so no region goes unseen.
        horizon_loss = observations.new_zeros(())
        n_h = 0
        for k in self.latent_horizons:
            if seq_len <= k:
                continue
            starts = list(range(0, seq_len - k, k))
            if not starts:
                continue
            idx = torch.tensor(starts, device=latents.device)
            state = {"h": states["h"][:, idx].reshape(-1, states["h"].shape[-1]),
                     "z": states["z"][:, idx].reshape(-1, states["z"].shape[-1])}
            # Roll the PRIOR forward with the actions actually taken. Under
            # causal alignment the action that carries state_t to state_{t+1}
            # is actions[:, t], which is the same pairing imagine_step and
            # the flow head use.
            for j in range(k):
                a_j = actions[:, idx + j].reshape(-1, actions.shape[-1])
                state = self.rssm.imagine_step(state, a_j)
            # The posterior at t+k is what actually happened. Detached: this
            # term trains the TRANSITION to reach the right place, not the
            # encoder to make the place easier to reach — without the detach
            # the cheapest solution is a posterior that collapses toward
            # whatever the prior already predicts.
            tgt = infos["posterior_logits"][:, idx + k].reshape(
                -1, self.rssm.stoch_dim).detach()
            pri = self.rssm.prior_net(state["h"])
            # RAW divergence, not compute_kl_loss — see prior_divergence for
            # the measurement that forced the distinction.
            horizon_loss = horizon_loss + self.rssm.prior_divergence(pri, tgt)
            n_h += 1
        if n_h:
            horizon_loss = horizon_loss / n_h

        recon_scale = 1.0
        kl_scale = 0.3
        reward_scale = 1.0
        continue_scale = 5.0
        inverse_scale = self.inverse_dynamics_scale if self.inverse_head is not None else 0.0

        flow_scale = self.flow_weight if self.flow_enabled else 0.0
        flow_smooth_scale = self.flow_smooth_weight if self.flow_enabled else 0.0

        total_loss = (
            recon_scale * recon_loss
            + kl_scale * kl_loss
            + reward_scale * reward_loss
            + continue_scale * continue_loss
            + inverse_scale * inverse_loss
            + flow_scale * flow_loss
            + flow_smooth_scale * flow_smooth_loss
            + (self.latent_horizon_weight if n_h else 0.0) * horizon_loss
            + (self.slot_weight if self.slots_enabled else 0.0) * slot_loss
        )

        losses = {
            "total": total_loss,
            "reconstruction": recon_loss,
            "kl": kl_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "inverse": inverse_loss,
            "flow": flow_loss,
            "flow_smooth": flow_smooth_loss,
            # Fraction of pixels the auto-mask RETAINED. 1.0 when masking is
            # off or nothing was dropped; a number trending toward 0 means the
            # loss is starving itself and is the thing to watch.
            "flow_mask": observations.new_tensor(mask_frac),
            # Divergence between an k-step rollout and what actually
            # happened. The number the dream substrate has always needed and
            # never had: if this does not fall, imagined trajectories are
            # fiction past step one.
            "horizon": horizon_loss,
            # How well the SLOTS' combined field explains the motion. If this
            # does not fall below what a single undifferentiated field
            # achieves, the decomposition is not buying anything.
            "slot": slot_loss,
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
        proprio: Optional[torch.Tensor] = None,
    ):
        """One gradient update step. Returns loss values as plain floats for logging.

        If return_per_sample is True, returns (metrics, per_sequence_error) where
        per_sequence_error is a numpy array used to update PER priorities.
        """
        self.train()
        # ---- bf16 AUTOCAST, FORWARD ONLY (2026-09-24) ---------------------
        # Wraps the LOSS COMPUTATION only. backward(), clip_grad_norm_ and
        # optimizer.step() stay outside, so master weights and gradients remain
        # fp32 -- bf16 is used for activations, which is where the memory is.
        # `enabled=` IS LOAD-BEARING: on CPU (every test on the Mac) this must
        # be a hard no-op, or contracts asserting loss values move for reasons
        # that have nothing to do with what they test.
        _use_amp = (self.amp_dtype is not None
                    and torch.is_tensor(observations) and observations.is_cuda)
        with torch.autocast("cuda", dtype=self.amp_dtype or torch.bfloat16,
                            enabled=_use_amp):
            out = self.compute_loss(
                observations,
                actions,
                rewards,
                continues,
                importance_weights=importance_weights,
                return_per_sample=return_per_sample,
                proprio=proprio,
            )
        if return_per_sample:
            losses, per_sample = out
        else:
            losses = out

        self.optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping for stability (DreamerV3 uses 100.0)
        _gn = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.optimizer.step()

        metrics = {k: v.item() for k, v in losses.items()}
        if COLLECT_STATS:
            self.last_step_stats = self._step_stats(metrics, _gn)
        self._stats_logits = None
        if return_per_sample:
            return metrics, per_sample.detach().cpu().numpy()
        return metrics

    def _step_stats(self, metrics, grad_norm):
        """Plain-float diagnostics of one gradient step. ONE device sync.
        Never raises: a failure yields the losses with the extras as None."""
        out = {}
        for k, v in metrics.items():
            out[_STAT_LOSS_NAMES.get(k, "loss_" + str(k))] = float(v)
        out.update({"kl_raw": None, "grad_norm": None,
                    "prior_entropy": None, "posterior_entropy": None})
        try:
            with torch.no_grad():
                vals = [grad_norm.detach().float().reshape(())]
                lg = self._stats_logits
                if lg is not None:
                    pri = lg[0].reshape(-1, self.rssm.stoch_dim).float()
                    post = lg[1].reshape(-1, self.rssm.stoch_dim).float()
                    S = self.rssm.stochastic_size
                    C = self.rssm.stochastic_classes
                    lp_pri = F.log_softmax(pri.view(-1, S, C), dim=-1)
                    lp_post = F.log_softmax(post.view(-1, S, C), dim=-1)
                    # Mean entropy PER categorical variable, nats; the
                    # ceiling is log(C). A posterior falling toward 0 is
                    # collapse; a prior stuck at log(C) has learned nothing.
                    ent_pri = -(lp_pri.exp() * lp_pri).sum(-1).mean()
                    ent_post = -(lp_post.exp() * lp_post).sum(-1).mean()
                    # Unclamped KL(post || prior): the loss_kl head is
                    # floored by free nats and cannot fall below 0.11.
                    kl_raw = self.rssm.prior_divergence(pri, post)
                    vals += [kl_raw.float(), ent_pri, ent_post]
                flat = torch.stack(vals).tolist()          # the one sync
            out["grad_norm"] = float(flat[0])
            if len(flat) == 4:
                out["kl_raw"] = float(flat[1])
                out["prior_entropy"] = float(flat[2])
                out["posterior_entropy"] = float(flat[3])
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "world-model step stats failed: %s", e)
        return out

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
