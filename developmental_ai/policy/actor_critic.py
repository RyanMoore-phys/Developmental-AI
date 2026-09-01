"""
Policy — Goal-Conditioned Actor-Critic
========================================
The action selection layer. Uses Stable-Baselines3 (SB3) as the backbone
for policy optimization (PPO by default), but wraps it with:

  1. Curiosity reward integration: Blends intrinsic (curiosity) and extrinsic
     (environment) rewards to drive exploration
  2. World model integration: Can use imagined trajectories from the RSSM
     to train the policy without real environment interaction
  3. Goal conditioning: The policy receives a goal description alongside
     observations, enabling it to pursue different objectives

The policy operates in the standard RL loop:
  observation → policy → action → environment → reward → update policy

But with the developmental AI additions:
  - Reward = (intrinsic_weight * curiosity_reward) + (extrinsic_weight * env_reward)
  - Policy also trains on imagined trajectories from the world model
  - Goals are set by the curiosity engine, not by an external task specification

Reference: Stable-Baselines3 uses PyTorch and supports PPO, SAC, A2C, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
import logging

# The world model already solved "one reward stream spanning four orders of
# magnitude" for its own reward head (symlog + twohot, rssm.py). The PPO
# critic never got the treatment; see `value_space` in PPOAgent.
from developmental_ai.world_model.rssm import symlog, symexp

logger = logging.getLogger(__name__)

# SB3 is optional — we provide a standalone actor-critic as fallback
try:
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable-Baselines3 not installed. Using standalone actor-critic.")


# ---------------------------------------------------------------------------
# Reward Mixer — blends curiosity and environment rewards
# ---------------------------------------------------------------------------

class RewardMixer:
    """
    Blends intrinsic curiosity reward with extrinsic environment reward.

    The mix ratio changes over time:
    - Early training: mostly curiosity reward (explore widely)
    - Later training: more environment reward (exploit learned knowledge)

    This implements a simple version of H-GRAIL's bandit-based mechanism
    for balancing exploration vs. exploitation.

    The mixed reward is what the policy actually optimizes. By starting
    with high intrinsic weight, the agent explores even in environments
    with sparse or no extrinsic rewards.
    """

    def __init__(
        self,
        intrinsic_weight: float = 0.7,
        extrinsic_weight: float = 0.3,
        decay_rate: float = 0.0001,
        min_intrinsic: float = 0.1,
        gated: bool = False,
        reentry_bump: float = 0.2,
        adaptive_gating: bool = False,
        adaptive_rate: float = 5e-5,
        solve_ema_beta: float = 0.05,
        solve_threshold: float = 0.1,
        return_ratio_cap: float = 0.0,
        ret_ema_alpha: float = 1e-4,
        min_intrinsic_scale: float = 0.1,
    ):
        self.intrinsic_weight = intrinsic_weight
        self.extrinsic_weight = extrinsic_weight
        self.initial_intrinsic_weight = intrinsic_weight
        self.decay_rate = decay_rate
        self.min_intrinsic = min_intrinsic
        self.reentry_bump = reentry_bump
        self.step_count = 0
        # ---- Signal-driven (adaptive) gating ----
        # When on, the curiosity->task anneal is driven by HOW RELIABLY the agent
        # is earning EXTRINSIC reward (a per-episode solve-rate EMA), not just the
        # wall-clock/milestone clock. On a findable-reward task the solve rate
        # climbs -> intrinsic weight (and its floor) fall toward ~0, so the system
        # behaves like plain PPO instead of paying a curiosity tax. On a sparse /
        # reward-free task the solve rate stays ~0 -> curiosity is retained. This
        # is a NATURAL, reward-driven anneal, not a hand-fixed ratio. Off by
        # default so every prior run is byte-for-byte unchanged.
        self.adaptive_gating = bool(adaptive_gating)
        self.adaptive_rate = float(adaptive_rate)
        self.solve_ema_beta = float(solve_ema_beta)
        self.solve_threshold = float(solve_threshold)
        self._solve_ema = 0.0
        # When ``gated`` is True the anneal is HELD until a developmental
        # milestone calls ``start_anneal()`` — this re-anchors the
        # curiosity->task drift to a *learned* EXPLORE->EXPLOIT transition
        # instead of wall-clock t=0. When False the mixer anneals from the
        # start (original open-loop behavior), so existing callers/tests are
        # unaffected.
        self.gated = gated
        self.annealing = not gated
        # ---- RETURN NORMALIZATION (2026-07-27) ---------------------------
        # Slow EMAs of each stream's magnitude, used by mix() to keep the
        # two RETURNS within `target_ratio` instead of letting a dense
        # per-step signal bury a sparse per-segment one (measured 1922:1).
        # alpha is deliberately slow: the extrinsic stream is sparse, so a
        # fast EMA would collapse to ~0 between spikes and the guard would
        # oscillate. target_ratio 3.0 keeps curiosity DOMINANT (it is the
        # exploration engine) while leaving the task signal visible.
        self._ret_ema_alpha = float(ret_ema_alpha)
        # target_ratio <= 0 DISABLES normalization entirely (old behaviour).
        self.target_ratio = float(return_ratio_cap)
        self.min_intrinsic_scale = float(min_intrinsic_scale)
        self._int_ema = 0.0
        self._ext_ema = 0.0

    def mix(self, intrinsic_reward: float, extrinsic_reward: float,
            update_stats: bool = True) -> float:
        """
        Combine intrinsic and extrinsic rewards into a single training signal.

        RETURN-NORMALIZED (2026-07-27 stall assessment). The weights used to
        blend a DENSE per-step intrinsic against a SPARSE per-segment
        extrinsic, so the *weights* said 0.45/0.55 while the *returns* were
        1922:1 curiosity — measured over the stalled run: weighted intrinsic
        return 9179.4 vs weighted extrinsic 4.78. The anneal was cosmetic:
        shifting a weight cannot matter when one stream fires every step and
        the other fires four times in 60 segments. PPO optimised curiosity
        and never saw the task.

        The fix rescales intrinsic so the two RETURNS sit at a bounded ratio
        before the weights apply. Both streams are tracked as slow EMAs of
        their magnitude; intrinsic is divided by (its scale / extrinsic
        scale) whenever that would otherwise exceed `target_ratio`.

        DELIBERATELY ONE-SIDED: it only ever shrinks intrinsic, never
        amplifies it. With no extrinsic yet seen (`_ext_ema` ~ 0) the guard
        is inert and behaviour is the old blend — otherwise a task-free
        warmup would divide by ~0 and detonate the intrinsic drive, which is
        the exploration engine this project depends on.
        """
        # `update_stats=False` (2026-09-01): mix WITHOUT advancing the return
        # EMAs. Scout streams pass False. The EMAs are a per-STEP clock —
        # `ret_ema_alpha` 1e-4 was chosen against one stream's step rate — so
        # letting N bodies each tick it would silently multiply the anneal
        # rate by N and retune the return-ratio damper by changing the fleet
        # size. The primary stream remains the single clock; scouts read the
        # damping it has learned without voting on it.
        i_abs, e_abs = abs(float(intrinsic_reward)), abs(float(extrinsic_reward))
        if update_stats:
            a = self._ret_ema_alpha
            self._int_ema = (1 - a) * self._int_ema + a * i_abs
            self._ext_ema = (1 - a) * self._ext_ema + a * e_abs
        scaled = float(intrinsic_reward)
        if (self.target_ratio > 0.0
                and self._ext_ema > 1e-8 and self._int_ema > 1e-8):
            ratio = self._int_ema / self._ext_ema
            if ratio > self.target_ratio:
                # FLOOR IS LOAD-BEARING. With a genuinely sparse task signal
                # (4 spikes in 60k steps) the raw correction is ~1/1000 and
                # would ZERO the intrinsic drive — killing the exploration
                # engine this project depends on, in the name of making the
                # task visible. Bound the correction so curiosity is damped,
                # never deleted.
                scaled *= max(self.min_intrinsic_scale,
                              self.target_ratio / ratio)
        mixed = (self.intrinsic_weight * scaled +
                 self.extrinsic_weight * extrinsic_reward)
        if update_stats:
            self.step_count += 1
        return mixed

    def decay(self, steps: int = 1) -> None:
        """
        Gradually shift from curiosity-driven to reward-driven behavior.

        Decays by ``decay_rate`` per timestep. Pass ``steps`` (e.g. the
        episode length) so the natural high-curiosity -> high-task drift
        advances by elapsed timesteps rather than episode count. This keeps
        the anneal schedule consistent regardless of how long episodes get —
        otherwise, once episodes lengthen, far fewer per-episode decay calls
        occur and the shift toward task stalls mid-way.

        When the mixer is ``gated`` (closed-loop developmental control), this
        is a no-op until ``start_anneal()`` fires at the EXPLORE->EXPLOIT
        milestone — so intrinsic weight is held high throughout exploration.
        """
        if self.adaptive_gating:
            # Signal-driven anneal (active from the start, independent of the
            # milestone gate): the more reliably extrinsic reward is being earned,
            # the faster intrinsic weight falls AND the lower its floor drops
            # (toward 0 -> plain PPO). At solve-rate 0 it reduces to the slow base
            # anneal with the usual min_intrinsic floor (curiosity retained).
            s = max(0.0, min(1.0, self._solve_ema))
            floor = self.min_intrinsic * (1.0 - s)
            rate = self.decay_rate + self.adaptive_rate * s
            self.intrinsic_weight = max(floor, self.intrinsic_weight - rate * steps)
            self.extrinsic_weight = 1.0 - self.intrinsic_weight
            return
        if not self.annealing:
            return
        self.intrinsic_weight = max(
            self.min_intrinsic,
            self.intrinsic_weight - self.decay_rate * steps
        )
        self.extrinsic_weight = 1.0 - self.intrinsic_weight

    def observe_episode(self, extrinsic_return: float) -> None:
        """Update the per-episode solve-rate EMA that drives adaptive gating.
        Call once per episode with the episode's EXTRINSIC return. No-op effect
        unless ``adaptive_gating`` is on (the EMA is only read there)."""
        solved = 1.0 if extrinsic_return > self.solve_threshold else 0.0
        self._solve_ema += self.solve_ema_beta * (solved - self._solve_ema)

    def start_anneal(self) -> None:
        """Begin the curiosity->task anneal. Called once the developmental
        stage controller detects EXPLORE->EXPLOIT, re-anchoring the drift to a
        learned milestone rather than wall-clock t=0."""
        self.annealing = True

    def restore_exploration(self) -> None:
        """Re-ignite curiosity on an EXPLOIT->EXPLORE re-entry with a PARTIAL
        bump: nudge intrinsic weight UP by ``reentry_bump`` (capped at its
        initial high value) and pause the anneal until the next plateau.

        A full reset to the initial high value made a single transient WM-error
        blip erase the entire curiosity->task schedule built up over the run.
        A bounded bump still re-engages exploration in proportion to how far
        the anneal had progressed, but the drift resumes from near where it
        left off rather than from scratch."""
        self.intrinsic_weight = min(
            self.initial_intrinsic_weight,
            self.intrinsic_weight + self.reentry_bump,
        )
        self.extrinsic_weight = 1.0 - self.intrinsic_weight
        self.annealing = False

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "intrinsic": self.intrinsic_weight,
            "extrinsic": self.extrinsic_weight,
        }


# ---------------------------------------------------------------------------
# Standalone Actor-Critic (no SB3 dependency)
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    """
    Actor network: maps observations to action probabilities (discrete)
    or action parameters (continuous).

    For discrete action spaces, outputs a probability distribution over actions.
    For continuous action spaces, outputs mean and log_std of a Gaussian.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, continuous: bool = False):
        super().__init__()
        self.continuous = continuous

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        if continuous:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.shared(obs)
        if self.continuous:
            return self.mean_head(features)
        else:
            return F.softmax(self.action_head(features), dim=-1)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action (or take the mode when deterministic=True) and
        return it with its log probability. deterministic=True gives the
        greedy/argmax action for honest evaluation (July 2026 audit H4 —
        eval previously SAMPLED while claiming to be deterministic)."""
        features = self.shared(obs)

        if self.continuous:
            mean = self.mean_head(features)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob
        else:
            logits = self.action_head(features)
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob


class CriticNetwork(nn.Module):
    """
    Critic network: estimates the value of being in a given state.

    The critic's estimate is used to:
    1. Compute advantage estimates (how much better/worse than expected)
    2. Guide the actor toward higher-value states
    3. Reduce variance in policy gradient estimates
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class KnowledgeConditioner(nn.Module):
    """
    Learned gate that conditions the policy on the (detached) knowledge graph
    vector — the load-bearing half of "Path A" (feature-conditioning).

    Before this, the symbolic knowledge vector was blended into an RSSM latent
    that only ever fed a fact-extractor (a closed self-feeding loop), under
    torch.no_grad(), so it could never change the agent's actions and the gate
    received no training signal. Here the knowledge vector enters the policy as
    an INPUT FEATURE, and this gate — whose parameters live in the policy
    optimizer — learns *how much* of it to admit from the policy/value loss.

    The gate is initialized near-closed (bias = -3 -> sigmoid ~= 0.047) so the
    policy starts behaviorally identical to the obs-only baseline and only
    *opens* the gate if the symbolic knowledge measurably reduces the loss.
    That keeps the symbolic-on vs symbolic-off ablation honest: if knowledge
    does not help, the gate stays shut and the two conditions converge.

    The knowledge vector itself is passed in detached (a feature, not a
    backprop path into the GNN), so the GNN keeps learning on graph structure
    while the policy learns to *exploit* the resulting features.
    """

    GATE_MODES = ("learned", "open")

    def __init__(self, obs_dim: int, knowledge_dim: int, gate_mode: str = "learned"):
        super().__init__()
        assert gate_mode in self.GATE_MODES, f"unknown gate_mode {gate_mode!r}"
        self.knowledge_dim = knowledge_dim
        self.gate_mode = gate_mode
        self.gate_net = nn.Sequential(
            nn.Linear(obs_dim + knowledge_dim, knowledge_dim),
            nn.Sigmoid(),  # per-dim gate in [0,1]
        )
        with torch.no_grad():
            self.gate_net[0].bias.fill_(-3.0)  # start ~closed

    def forward(
        self, obs: torch.Tensor, knowledge: torch.Tensor
    ) -> torch.Tensor:
        # "open" force-concat: the gate is fixed fully open so the knowledge
        # vector reaches the policy unattenuated. This DECOUPLES two questions
        # the learned gate conflates — "is the knowledge useful" vs "did the
        # policy choose to admit it". For the cross-task transfer test we WANT
        # the source policy to wire the knowledge in unconditionally, so that
        # lesioning its content at transfer time genuinely bites.
        if self.gate_mode == "open":
            return knowledge
        gate = self.gate_net(torch.cat([obs, knowledge], dim=-1))
        return gate * knowledge


class StandaloneActorCritic:
    """
    A simple PPO-style actor-critic that works without SB3.

    This is a minimal but functional implementation for prototyping.
    For production use, prefer the SB3-based CuriosityPolicy below.

    Implements:
      - Clipped surrogate objective (PPO)
      - Generalized Advantage Estimation (GAE)
      - Entropy bonus for exploration

    arch="conv" (2026-07-26): a shared conv trunk in front of the SAME MLP
    stack. Measured motivation: with flat pixel input, 30.1 of a stored
    skill's 30.2M params were THREE separate copies of a (·, 49250) matrix
    reading raw pixels (actor.shared.0, critic.net.0, conditioner.gate_net.0)
    — 116 MB per minted skill, ~99% of it near-duplicate across the whole
    bank. The conv trunk encodes pixels ONCE into `enc_dim` features shared
    by actor, critic and conditioner: ~1.7M params (~7 MB) per skill, a
    spatial prior instead of a flat one, and the conditioner shrinks from
    4.8M to ~44k params. The encoder is the ICM's CNNFeatureEncoder — the
    proven in-repo conv that accepts the SAME flat CHW [0,1] vectors the
    pipeline passes everywhere (it reshapes internally), so nothing upstream
    changes shape. Default "flat" keeps every existing env, test and stored
    skill byte-identical.

    arch="wm" (2026-08-01): ONE VENTRAL STREAM, MANY READERS. Perception is
    not the policy's own module at all — it is the WORLD MODEL's CNNEncoder,
    shared by reference and read DETACHED.

    Measured motivation: with arch="flat" the agent carried TWO independent
    visual systems. The world model's encoder is trained on every frame from
    all four streams by reconstruction + KL; the policy's own 50 MB pixel
    layer is trained only by PPO gradients, a far weaker signal. Inspecting a
    stored skill, 86% of that layer's weights were still inside their
    initialization bound and the 98 knowledge columns sat at 1.009x init —
    i.e. the policy was acting on a near-random projection of the screen
    while a well-trained representation of the same pixels already existed
    one module away.

    Under "wm" the policy consumes the world model's features and its own
    first layer becomes the READOUT (Linear(enc_dim + kdim + proprio, hidden))
    — the small, trainable, per-consumer part. The dream actor already reads
    the world model's latents; this makes the waking policy do the same.

    GRADIENT ISOLATION IS LOAD-BEARING: the shared encoder is excluded from
    the policy optimizer AND its output is detached, so PPO can never reach
    into the world model. Perception is shaped by prediction (what the world
    is actually like), never by reward (what the agent currently wants) —
    which is both the safe choice and the biologically honest one.
    """

    # Bound on the symlog-space value before symexp decodes it. See _value_of.
    _VALUE_SYMLOG_CLAMP = 20.0

    ARCHS = ("flat", "conv", "wm", "rssm")

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        knowledge_dim: int = 0,
        knowledge_gate_mode: str = "learned",
        device: Optional[torch.device] = None,
        arch: str = "flat",
        enc_dim: int = 256,
        proprio_dim: int = 0,
        minibatch_size: int = 0,
        target_kl: float = 0.0,
        logit_range: float = 0.0,
        min_rows_per_update: int = 0,
        max_env_steps_per_update: int = 0,
        max_rows_per_update: int = 0,
        rollout_uint8: bool = False,
    ):
        assert arch in self.ARCHS, f"unknown policy arch {arch!r}"
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        # ---- MINIBATCHING (2026-08-02) -----------------------------------
        # `policy.batch_size` shipped in every config from the start and had
        # NO reader: train_step did `n_epochs` FULL-batch passes and nothing
        # else, so a 1024-step rollout bought ~10 gradient steps where standard
        # PPO takes ~10 x 16 = 160. Learning was ~16x slower than the config
        # claimed, which is most of why the policy looked frozen between
        # segments. 0 keeps the old full-batch behaviour exactly.
        self.minibatch_size = int(minibatch_size or 0)
        # See should_update(). 0/0 = the historical env-step-only trigger.
        self.min_rows_per_update = int(min_rows_per_update or 0)
        self.max_env_steps_per_update = int(max_env_steps_per_update or 0)
        self.max_rows_per_update = int(max_rows_per_update or 0)
        self.last_update_forced = False
        # See store_transition. Pixel envs only — a vector env's normalized
        # obs can be negative and unbounded, where x255-and-round is nonsense.
        self._rollout_uint8 = bool(rollout_uint8)
        # ---- KL EARLY STOPPING (2026-08-05) ------------------------------
        # THE MISSING TRUST REGION. PPO's clip bounds the SURROGATE
        # OBJECTIVE, not how far the policy actually moves — so running
        # n_epochs passes over one batch lets the policy walk arbitrarily far
        # from the one that collected the data. Every reference PPO (SB3,
        # CleanRL, Spinning Up) stops epochs early once approximate KL
        # exceeds ~0.01-0.02. This codebase had no such check at all.
        #
        # MEASURED consequence, on every single restart: entropy 90% -> 0.00
        # within one or two segments and max_prob EXACTLY 1.000 (a one-hot
        # softmax), independent of entropy_coef (tried 0.01 and 0.03) and
        # independent of minibatching (tried on and off). Neither of those
        # knobs could ever have fixed it, because neither bounds policy
        # movement — which is why two rounds of tuning them failed.
        # 0.0 = off = byte-identical for every config that does not set it.
        self.target_kl = float(target_kl or 0.0)
        # ---- SATURATION IS A ONE-WAY DOOR (2026-08-05) -------------------
        # MEASURED directly on the softmax:
        #   max_prob 0.174 -> |d(entropy)/d(logits)| = 2.88e-01
        #   max_prob 0.920 -> 4.41e-01   (the bonus pushes back hardest here)
        #   max_prob 1.000 -> 9.54e-07   (five orders of magnitude gone)
        # An entropy bonus can PREVENT collapse but can never REVERSE it, and
        # every entropy_coef tuned this session (0.01, then 0.03) was applied
        # to a policy already at max_prob = 1.000 — where the knob has no
        # leverage at all. That is why each attempt failed.
        #
        # Prevention needs a HARD bound. `_bound_logits` rescales the action
        # logits so their spread never exceeds this, preserving their ORDER
        # (learning is unaffected) while capping the best/worst probability
        # ratio at e^logit_range. The policy may become confident; it cannot
        # become certain, so gradient and sampling always remain.
        # 0.0 = off = byte-identical for every config that omits it.
        self.logit_range = float(logit_range or 0.0)
        self.last_approx_kl = 0.0
        self.last_epochs_run = 0
        # PPO update-rate forensics (2026-08-23). Under options a stored ROW
        # spans tau env steps, so the row count — not the env-step count —
        # is what the collapse guard divides by. Kept on the object so the
        # loop's segment print can distinguish a starved update from a
        # healthy one; see train_step's guard block.
        # largest observed |carried - recomputed| feature delta while the A2
        # self-check is on; ~0 means no encoder update fell between the two
        self.last_feat_drift = 0.0
        self.last_epochs_requested = 0
        self.last_epochs_capped = False
        self.last_rows = 0
        self.last_rows_option = 0
        self.last_rows_primitive = 0
        self.last_tau_mean = 1.0
        self.last_env_steps = 0
        self.continuous = continuous
        # Device (July 2026 audit H-GPU fix): StandaloneActorCritic is a plain
        # class, not an nn.Module, so it was never `.to(device)`'d and built all
        # input tensors on CPU — crashing on CUDA when its submodules or the WM
        # latents live on the GPU. We now move the submodules to `device` and
        # create every input tensor on it. Defaults to CUDA if available.
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # ---- Perception trunk (arch="conv") ----
        # `feat_dim` is what the MLP stack sees: raw obs for flat, encoder
        # features for conv. The encoder is SHARED by actor/critic/conditioner
        # and trained end-to-end by the PPO loss (single optimizer below).
        self.arch = arch
        self.enc_dim = int(enc_dim)
        self.encoder = None
        # arch="wm": perception is the WORLD MODEL's encoder, attached after
        # construction via attach_shared_encoder(). It is deliberately NOT a
        # submodule (no self.encoder assignment) so it never enters
        # self.parameters(), the policy optimizer, or state_dict() by
        # accident — the sharing is by reference, and the gradient isolation
        # has to survive someone later writing `params = self.parameters()`.
        self._shared_encoder = None
        if arch == "rssm":
            # ---- THE POLICY SEES THE WORLD MODEL'S STATE (2026-09-01) ----
            # arch='wm' feeds the actor DETACHED ENCODER FEATURES OF THE
            # CURRENT FRAME. The 2048-d deterministic RSSM state h — the
            # entire reason for having an RSSM — never reached it, so the
            # policy was memoryless on a task whose defining difficulty is
            # sustaining one action through ~60 ticks of no visible feedback.
            # A frame mid-swing and a frame at rest look nearly identical
            # (the crack overlay is a few pixels at 128px); h does not.
            #
            # `enc_dim` here is the LATENT width the caller must pass:
            # deterministic_size + stochastic_size * stochastic_classes.
            # There is no encoder to attach — the loop already computes this
            # latent every step for the RSSM and hands it over via
            # select_action(feats=...).
            feat_dim = self.enc_dim
        elif arch == "wm":
            side = int(round((obs_dim / 3) ** 0.5))
            if 3 * side * side != int(obs_dim):
                raise ValueError(
                    f"arch='wm' needs flat CHW pixel obs (3*s*s); "
                    f"obs_dim={obs_dim} is not. Use arch='flat'.")
            # enc_dim MUST equal the world model's encoder output width
            # (world_model.encoder_hidden). The caller is responsible for
            # passing it; attach_shared_encoder re-checks and raises rather
            # than letting a silent shape mismatch reach the first forward.
            feat_dim = self.enc_dim
        elif arch == "conv":
            # pixel obs are flat CHW: obs_dim must be 3 * s * s
            side = int(round((obs_dim / 3) ** 0.5))
            if 3 * side * side != int(obs_dim):
                raise ValueError(
                    f"arch='conv' needs flat CHW pixel obs (3*s*s); "
                    f"obs_dim={obs_dim} is not. Use arch='flat'.")
            from developmental_ai.curiosity.icm import CNNFeatureEncoder
            self.encoder = CNNFeatureEncoder(
                in_channels=3, image_size=side,
                feature_dim=self.enc_dim).to(self.device)
            feat_dim = self.enc_dim
        else:
            feat_dim = obs_dim
        # ---- PROPRIOCEPTION (2026-07-27, rung 0) --------------------------
        # Self-state (hunger, health, depth, holding-something, in-a-menu,
        # just-hurt, carrying) concatenated UNGATED. Deliberately not routed
        # through the KnowledgeConditioner: that gate initialises near-closed
        # (bias -3, sigmoid ~0.047) so symbolic knowledge has to earn its way
        # in, which is right for a KG feature and wrong for a body. An agent
        # should not have to learn that it is allowed to feel hungry.
        self.proprio_dim = int(proprio_dim)
        feat_dim = feat_dim + self.proprio_dim
        self.feat_dim = feat_dim

        # ---- Symbolic feature-conditioning (Path A) ----
        # When knowledge_dim > 0 the policy consumes [feats, gated_knowledge];
        # the gate is trained by the policy loss so symbolic knowledge can
        # actually shape behavior. When 0, the policy is the original
        # feats-only baseline (so the symbolic-off ablation is a clean A/B).
        # Under conv the gate reads ENCODER FEATURES, not raw pixels — that is
        # where the 4.8M -> ~44k conditioner shrink comes from.
        self.knowledge_dim = knowledge_dim
        self.knowledge_gate_mode = knowledge_gate_mode
        input_dim = feat_dim + knowledge_dim
        # the gate conditions on PERCEPTION, not on the body — keeping its
        # input width independent of proprio means adding a sense never
        # reshapes the knowledge gate.
        self.conditioner = (
            KnowledgeConditioner(feat_dim - self.proprio_dim, knowledge_dim,
                                 gate_mode=knowledge_gate_mode).to(self.device)
            if knowledge_dim > 0
            else None
        )

        # Networks (input is augmented with the gated knowledge feature).
        # Moved to self.device so forward passes match the input-tensor device.
        self.actor = ActorNetwork(input_dim, action_dim, hidden_dim, continuous).to(self.device)
        self.critic = CriticNetwork(input_dim, hidden_dim).to(self.device)

        # Single optimizer for actor + critic (+ encoder, + the knowledge
        # gate) so the whole stack is trained end-to-end by the policy loss
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.encoder is not None:
            params += list(self.encoder.parameters())
        if self.conditioner is not None:
            params += list(self.conditioner.parameters())
        # NOTE (arch="wm"): _shared_encoder is intentionally absent here. The
        # world model's encoder is trained by PREDICTION (reconstruction+KL on
        # every frame of every stream), never by REWARD. Adding it to this
        # optimizer would let PPO reshape the agent's perception to suit its
        # current policy — the failure mode where an agent learns to see what
        # pays rather than what is there.
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        # Clip groups, built ONCE. The actor and the critic are clipped
        # SEPARATELY (see train_step): with one shared clip, a value-loss
        # spike consumed the whole norm budget and scaled the policy gradient
        # toward zero — in exactly the update that carried the run's only real
        # signal. On this task a felled log is ~51 in one row against a
        # typical ~0.1, so `value_loss` jumps ~4 orders of magnitude on the
        # single most informative rollout the agent will ever collect.
        self._clip_actor = list(self.actor.parameters())
        if self.conditioner is not None:
            self._clip_actor += list(self.conditioner.parameters())
        if self.encoder is not None:
            # the shared trunk serves both heads; clipped with the actor so
            # perception is never starved by a value spike either
            self._clip_actor += list(self.encoder.parameters())
        self._clip_critic = list(self.critic.parameters())

        # ---- VALUE SPACE (2026-09-01) ------------------------------------
        # "symlog": the critic predicts symlog(return) and every READ decodes
        # with symexp, so the head is scale-free and a 51.0 return is a target
        # of ~3.95 instead of 51. "linear": the pre-2026-09-01 raw-return
        # head, kept so a checkpoint written before this change can be
        # identified rather than silently mis-scaled by a factor of e^|v|.
        self.value_space = "symlog"
        self._value_clamp_hits = 0
        self._value_clamp_n = 0
        # Rollout storage
        self.rollout_obs = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.rollout_log_probs = []
        self.rollout_values = []
        self.rollout_knowledge = []  # per-step KG feature (empty if knowledge_dim==0)
        self.rollout_proprio = []    # per-step self-state (empty if proprio_dim==0)
        # SMDP extensions (skills-as-options, July 2026): tau = how many env
        # steps decision t spanned (1 for primitives, k for options); mask =
        # the action mask AT SELECTION TIME (slots flip empty->bound at
        # episode boundaries mid-rollout, and a masked softmax changes the
        # normalizer, so ratios must be computed on the selection-time
        # support). Empty/all-ones when options are off -> byte-identical.
        self.rollout_taus = []
        self.rollout_masks = []
        # ---- WHICH BODY PRODUCED THIS ROW (2026-09-01) -------------------
        # All rows used to come from stream 0; the scouts' experience reached
        # the world model and never the policy. Rows now carry their stream
        # so _compute_gae can run its recursion PER TRAJECTORY — interleaving
        # two streams in one reversed pass would splice one body's future
        # onto another body's present, which is a worse error than the
        # missing data it would be fixing.
        self.rollout_stream = []
        # ---- arch='rssm': the FEATURES the decision was made on -----------
        # For flat/conv/wm the update recomputes features from the stored
        # observation. For rssm it cannot — h is recurrent — so the latent
        # that was actually acted on is stored alongside the row. Empty on
        # every other arch, so nothing else changes shape or cost.
        self.rollout_feats = []

    def _prep_knowledge(self, knowledge: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        """Coerce a knowledge feature to a (1, knowledge_dim) tensor, or None
        when conditioning is disabled. Missing knowledge -> zeros (gate sees a
        null feature and the augmented input reduces to ~[obs, 0])."""
        if self.knowledge_dim == 0:
            return None
        if knowledge is None:
            vec = np.zeros(self.knowledge_dim, dtype=np.float32)
        else:
            vec = np.asarray(knowledge, dtype=np.float32).reshape(-1)
        return torch.FloatTensor(vec).unsqueeze(0).to(self.device)

    def attach_shared_encoder(self, encoder: nn.Module) -> None:
        """Point arch='wm' perception at the world model's encoder.

        Called once by the loop after BOTH modules exist (the world model is
        built long before the policy, but the policy must be constructed with
        the right feat_dim, so the wiring is a second step rather than a
        constructor argument).

        Validates the output width against enc_dim with a real forward pass:
        a mismatch here would otherwise surface as a confusing shape error
        inside the actor on the first real observation, thousands of steps
        into a run.
        """
        if self.arch != "wm":
            raise ValueError(
                f"attach_shared_encoder is only meaningful for arch='wm' "
                f"(this policy is arch='{self.arch}')")
        encoder = encoder.to(self.device)
        with torch.no_grad():
            probe = torch.zeros(1, self.obs_dim, device=self.device)
            out_dim = int(encoder(probe).reshape(1, -1).shape[1])
        if out_dim != self.enc_dim:
            raise ValueError(
                f"shared encoder emits {out_dim} features but the policy was "
                f"built for enc_dim={self.enc_dim}. Set policy.enc_dim to "
                f"world_model.encoder_hidden.")
        self._shared_encoder = encoder

    def _encode(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Perception hop: identity for flat, conv features for conv, and the
        WORLD MODEL's features for wm.

        For flat/conv this runs WITH grad inside train_step (the encoder
        learns from the PPO loss) and under the caller's no_grad during action
        selection.

        For wm the result is DETACHED unconditionally: the shared encoder is
        the world model's, and PPO must not backpropagate into it from any
        call site — including the ones inside train_step that deliberately run
        with grad enabled. Detaching here rather than at each caller means a
        future call site cannot reintroduce the leak by forgetting.
        """
        if self.arch == "rssm":
            # THERE IS NOTHING TO COMPUTE HERE, AND THAT IS THE POINT.
            # h is recurrent: it depends on the whole history, not on this
            # observation, so no function of `obs_tensor` can produce it.
            # Every caller must hand the latent over explicitly
            # (select_action(feats=...), or the stored rollout_feats at update
            # time). Raising is the same refusal arch='wm' already makes when
            # no encoder is attached — a silent obs-shaped fallback would
            # look like it worked and quietly train the policy on the wrong
            # input.
            raise RuntimeError(
                "arch='rssm' policy cannot encode an observation: the "
                "deterministic state h is recurrent and is not a function of "
                "one frame. Pass the latent explicitly — select_action(..., "
                "feats=latent) when acting, rollout_feats when training.")
        if self.arch == "wm":
            if self._shared_encoder is None:
                raise RuntimeError(
                    "arch='wm' policy has no shared encoder attached — call "
                    "attach_shared_encoder(world_model.encoder) before acting. "
                    "Refusing to silently fall back to raw pixels, which would "
                    "look like it worked and quietly train on a 49152-wide "
                    "input the network was never built for.")
            with torch.no_grad():
                feats = self._shared_encoder(obs_tensor)
            return feats.reshape(obs_tensor.shape[0], -1).detach()
        if self.encoder is None:
            return obs_tensor
        return self.encoder(obs_tensor)

    def _value_of(self, aug: torch.Tensor) -> torch.Tensor:
        """V(s) in REWARD SPACE, whatever space the head is trained in.

        THE ONE DECODE SITE. Every consumer of a value — action selection,
        the GAE bootstrap, the option decision record — reads through here,
        so the head's parameterization can never disagree with the arithmetic
        performed on its output. Putting the symexp at each call site instead
        is how you get one path that forgets, and a value function that is
        wrong by a factor of e^|v| on exactly that path.
        """
        v = self.critic(aug)
        if self.value_space != "symlog":
            return v
        # CLAMP BEFORE symexp. The critic head is an unbounded MLP, and
        # symexp is sign(v)*(exp(|v|)-1): a head output of 20 is 4.8e8, 90 is
        # inf. That value flows into _compute_gae as both `values` and the
        # bootstrap, so ONE diverging update turns every advantage into NaN,
        # NaNs the weights, and poisons the checkpoint — on a brain that
        # lives only on the pod. Before symlog a divergence grew linearly and
        # was survivable; this makes it exponential, so the bound comes with
        # it. Sizing: the largest single-row reward here is ~51 (a felled
        # log) and a discounted option-horizon return stays well under 1e3,
        # i.e. symlog ~6.9. 20 is ~3x beyond anything legitimate, so it never
        # binds in normal operation and only ever catches divergence —
        # `value_clamped_frac` says which.
        _c = v.clamp(-self._VALUE_SYMLOG_CLAMP, self._VALUE_SYMLOG_CLAMP)
        if torch.is_grad_enabled() is False:
            self._value_clamp_hits += int((v != _c).sum().item())
            self._value_clamp_n += int(v.numel())
        return symexp(_c)

    def _bound_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Cap the SPREAD of the action logits, preserving their order.

        Rescales (not clips) so relative preferences are kept intact and only
        their magnitude is bounded: the best/worst probability ratio can never
        exceed e^logit_range. The policy stays free to become confident; it
        cannot become certain, so some gradient and some sampling always
        remain. Identity when logit_range <= 0 or the spread is already under
        it — so this is a no-op on every config that does not ask for it, and
        on any well-behaved policy.

        MUST be applied identically when ACTING and when TRAINING, or the
        stored log_probs would not match the ones PPO recomputes and every
        importance ratio would be wrong.
        """
        if self.logit_range <= 0.0:
            return logits
        _c = logits - logits.mean(dim=-1, keepdim=True)
        _spread = (_c.max(dim=-1, keepdim=True).values
                   - _c.min(dim=-1, keepdim=True).values)
        _scale = torch.clamp(self.logit_range / (_spread + 1e-8), max=1.0)
        return _c * _scale

    def _fit_proprio_width(self, proprio) -> np.ndarray:
        """One row of self-state, exactly proprio_dim wide.

        A SHORT vector is a MISSING SENSE (a reset or client rebuild handing
        back the env's raw width before loop-computed senses are appended),
        and the convention throughout this project is that a missing sense
        reads NEUTRAL rather than being fatal. Pad with zeros; truncate an
        over-long one rather than guessing which field to drop.
        """
        if proprio is None:
            return np.zeros(self.proprio_dim, dtype=np.float32)
        v = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if v.shape[0] < self.proprio_dim:
            v = np.concatenate([v, np.zeros(
                self.proprio_dim - v.shape[0], dtype=np.float32)])
        elif v.shape[0] > self.proprio_dim:
            v = v[:self.proprio_dim]
        return v

    def _prep_proprio(self, proprio, n: int = 1) -> Optional[torch.Tensor]:
        """Coerce self-state to (n, proprio_dim); missing -> neutral zeros."""
        if self.proprio_dim == 0:
            return None
        if proprio is None:
            v = np.zeros((n, self.proprio_dim), dtype=np.float32)
        else:
            # WIDTH-TOLERANT (fix 2026-08-02). proprio is assembled by
            # several paths — the env's own vector, plus loop-computed senses
            # appended on the primary stream — and a reset or client rebuild
            # can hand back the env's raw width. A hard reshape turned that
            # into "cannot reshape array of size 9 into shape (10)" and KILLED
            # a live multi-day run. A SHORT vector is a MISSING SENSE, and the
            # convention everywhere else in this project is that a missing
            # sense reads NEUTRAL, never fatal. Pad with zeros; truncate an
            # over-long one rather than guessing which field to drop.
            v = np.asarray(proprio, dtype=np.float32).reshape(1, -1)
            w = int(v.shape[1])
            if w < self.proprio_dim:
                v = np.concatenate(
                    [v, np.zeros((1, self.proprio_dim - w), dtype=np.float32)],
                    axis=1)
            elif w > self.proprio_dim:
                v = v[:, :self.proprio_dim]
            if v.shape[0] != n:
                v = np.repeat(v[:1], n, axis=0)
        return torch.from_numpy(v).to(self.device)

    def _augment(
        self, feats_tensor: torch.Tensor,
        knowledge_tensor: Optional[torch.Tensor],
        proprio_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the policy input: [features, proprio, gated_knowledge].

        ORDER MATTERS AND IS FIXED: the knowledge gate is constructed to read
        PERCEPTION ONLY (feat_dim - proprio_dim wide), so the gate must see
        `feats_tensor` before proprio is appended. Proprio then rides
        UNGATED — a body is not a hypothesis to be admitted, it is the state
        the policy acts from.
        """
        gated = None
        if self.conditioner is not None and knowledge_tensor is not None:
            gated = self.conditioner(feats_tensor, knowledge_tensor)
        parts = [feats_tensor]
        # FAIL-SAFE, NOT FAIL-CRASH (2026-07-27). A caller that forgets the
        # body channel used to build a vector 8 columns too narrow and raise
        # `mat1 and mat2 shapes cannot be multiplied (8x354 and 362x256)`.
        # Every such caller (prospection, imagination, dream-distill) is
        # inside a try/except, so the crash did not stop the run — it
        # SILENTLY KILLED those subsystems, which is strictly worse than a
        # slightly-wrong input. Missing self-state now pads to NEUTRAL, the
        # same convention the env uses for an unavailable sense.
        if self.proprio_dim > 0:
            if proprio_tensor is None:
                proprio_tensor = torch.zeros(
                    feats_tensor.shape[0], self.proprio_dim,
                    device=feats_tensor.device, dtype=feats_tensor.dtype)
            elif proprio_tensor.shape[0] != feats_tensor.shape[0]:
                proprio_tensor = proprio_tensor[:1].expand(
                    feats_tensor.shape[0], -1)
        if proprio_tensor is not None:
            parts.append(proprio_tensor)
        if gated is not None:
            parts.append(gated)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def select_action(
        self, obs: np.ndarray, knowledge: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
        proprio: Optional[np.ndarray] = None,
        feats: Optional[torch.Tensor] = None,
        verify_feats: bool = False,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Select an action given an observation (and optional KG knowledge vector).

        deterministic=True takes the policy mode (argmax / Gaussian mean) —
        use for evaluation; training must keep the default sampling.

        Returns:
            action: The chosen action (int for discrete, array for continuous)
            info: Dict with log_prob and value for later training
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            knowledge_tensor = self._prep_knowledge(knowledge)
            # ---- REUSE THE ENCODER FORWARD THE LOOP ALREADY DID (A2) ------
            # For arch='wm' this is the world model's encoder, and the loop
            # ran it on THIS SAME FRAME one step ago (as `encoded_next`, for
            # the RSSM). Recomputing it here is a duplicate forward on every
            # env step. `feats` lets the caller hand the result over.
            #
            # EXPLICIT DATAFLOW, NOT A CACHE, ON PURPOSE: a cache keyed on
            # anything but exact provenance could serve one env's features
            # for another's frame, or a pre-reset frame after a world
            # rebuild — silent, and indistinguishable from a bad policy.
            # Passing them in means they are right by construction, and the
            # caller is the only place that knows whether the frame carried.
            _feat = None
            if feats is not None:
                _f = feats if feats.dim() == 2 else feats.reshape(1, -1)
                if _f.shape[0] != 1 or _f.shape[1] != int(self.enc_dim):
                    raise ValueError(
                        f"select_action got feats of shape {tuple(_f.shape)}, "
                        f"expected (1, {self.enc_dim}) — refusing to act on a "
                        f"feature vector that is not this observation's")
                if verify_feats and self.arch != "rssm":
                    # arch='rssm' EXEMPT, and not as a convenience: the check
                    # recomputes features from the observation, and h is
                    # recurrent, so there is nothing to recompute and
                    # _encode would raise. The failure this check exists to
                    # catch — a carried feature from the wrong frame or the
                    # wrong env — cannot occur there either: the latent is
                    # produced fresh from that env's own RSSM row every step
                    # and never carried across a step boundary.
                    # LIVE SELF-CHECK: recompute and compare. The tolerance is
                    # deliberate, not sloppy — the shared encoder TRAINS
                    # asynchronously (wm_train_every 250), so features carried
                    # one step can legitimately differ by one optimizer step
                    # of drift. A WRONG FRAME or a wrong env index differs by
                    # O(1) in conv features, so this separates the two rather
                    # than pretending drift is a bug.
                    _ref = self._encode(obs_tensor)
                    _d = float((_f - _ref).abs().max().item())
                    self.last_feat_drift = max(
                        float(getattr(self, "last_feat_drift", 0.0)), _d)
                    if _d > 1e-2:
                        raise RuntimeError(
                            f"carried features do not match this observation "
                            f"(max|delta| {_d:.4f}) — far beyond one optimizer "
                            f"step of encoder drift. The frame or the env "
                            f"index is wrong. Refusing to act.")
                _feat = _f
            if _feat is None:
                _feat = self._encode(obs_tensor)
            aug = self._augment(_feat, knowledge_tensor,
                                self._prep_proprio(proprio))
            if action_mask is not None and not self.continuous:
                # masked meta-head (options): sample from the renormalized
                # distribution over VALID actions only
                feats = self.actor.shared(aug)
                logits = self._bound_logits(self.actor.action_head(feats))
                mask_t = torch.as_tensor(
                    np.asarray(action_mask, dtype=bool),
                    device=logits.device).reshape(1, -1)
                logits = logits.masked_fill(~mask_t, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                action = (logits.argmax(dim=-1) if deterministic
                          else dist.sample())
                log_prob = dist.log_prob(action)
            else:
                action, log_prob = self.actor.get_action(
                    aug, deterministic=deterministic)
            value = self._value_of(aug)

        # ---- ONE DEVICE->HOST TRANSFER, NOT THREE (2026-08-23) -----------
        # Every `.item()` is a full pipeline drain: the CPU blocks until the
        # GPU has finished all queued work. This path ran three of them
        # (action, log_prob, value) on EVERY env step, and the drains are
        # worst exactly when they hurt most — the VLM shares this device, so
        # a sync can wait behind a 1.5s inference rather than a kernel.
        # Measured context: ~2350 aten ops and ~25 syncs per step at 2% GPU
        # utilisation; the device is idle and we are paying latency.
        #
        # Stacking is exact: the action INDEX is a small integer (action_dim
        # is tens), far inside float32's exactly-representable range, so the
        # int(...) round-trip cannot lose it.
        if self.continuous:
            action_np = action.squeeze(0).cpu().numpy()
            _lp, _v = log_prob.reshape(-1)[0], value.reshape(-1)[0]
            _scalars = torch.stack((_lp, _v)).tolist()   # 1 sync
            return action_np, {"log_prob": float(_scalars[0]),
                               "value": float(_scalars[1])}
        _scalars = torch.stack((
            action.reshape(-1)[0].to(value.dtype),
            log_prob.reshape(-1)[0].to(value.dtype),
            value.reshape(-1)[0],
        )).tolist()                                       # 1 sync, not 3
        return int(_scalars[0]), {
            "log_prob": float(_scalars[1]),
            "value": float(_scalars[2]),
        }

    def compute_last_value(self, obs: np.ndarray,
                           knowledge: Optional[np.ndarray] = None,
                           proprio: Optional[np.ndarray] = None,
                           feats: Optional[np.ndarray] = None) -> float:
        """V(s) for the state AFTER the last stored transition — the GAE
        bootstrap for a rollout that ends MID-trajectory (a lifelong segment
        boundary). Mirrors select_action's value path; stores nothing, no
        grad. Not used on the episodic path (which ends on real terminals)."""
        with torch.no_grad():
            knowledge_tensor = self._prep_knowledge(knowledge)
            if feats is not None:
                _f = torch.as_tensor(
                    np.asarray(feats, dtype=np.float32)
                ).reshape(1, -1).to(self.device)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                _f = self._encode(obs_tensor)   # raises for arch='rssm'
            aug = self._augment(_f, knowledge_tensor,
                                self._prep_proprio(proprio))
            return float(self._value_of(aug).item())

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        knowledge: Optional[np.ndarray] = None,
        tau: int = 1,
        action_mask: Optional[np.ndarray] = None,
        proprio: Optional[np.ndarray] = None,
        stream: int = 0,
        feats: Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition (tau = env steps this decision spanned).

        `stream` names the body this row came from. Rows from different
        streams may be interleaved in the buffer in any order; GAE segments
        them before running its recursion. Default 0 keeps every existing
        caller single-stream and byte-identical.
        """
        if self.arch == "rssm":
            if feats is None:
                raise ValueError(
                    "arch='rssm' store_transition needs `feats` (the latent "
                    "the decision was made on) — it cannot be recovered from "
                    "the observation at update time")
            self.rollout_feats.append(
                np.asarray(feats, dtype=np.float32).reshape(-1))
        self.rollout_stream.append(int(stream))
        self.rollout_taus.append(int(tau))
        self.rollout_masks.append(
            None if action_mask is None
            else np.asarray(action_mask, dtype=bool))
        # ---- QUANTIZED ROLLOUT OBSERVATIONS (2026-09-01) ------------------
        # Same convention the replay buffer has always used for pixels
        # (`ReplayBuffer._obs_uint8`) — deliberately the SAME one, so this
        # codebase has one quantization scheme rather than two that can drift.
        # A row is 49152 floats = 196 KB; as uint8 it is 48 KB. That matters
        # now that rows arrive from every stream and the row floor can hold a
        # rollout open across segments: 2048 rows goes from ~400 MB to
        # ~100 MB, plus the same saving again on the np.stack copy at update
        # time. Quantization error is 1/255, far below the encoder's noise
        # floor, and the agent still SEES full-precision pixels — this is
        # only how the decision is carried to the update.
        # Gated on obs_uint8 (pixel envs only): a vector env's normalized obs
        # can be negative and unbounded, where x255-and-round is nonsense.
        self.rollout_obs.append(
            np.clip(np.round(np.asarray(obs, np.float32) * 255.0),
                    0, 255).astype(np.uint8)
            if self._rollout_uint8 else obs)
        self.rollout_actions.append(action)
        self.rollout_rewards.append(reward)
        self.rollout_dones.append(done)
        self.rollout_log_probs.append(log_prob)
        self.rollout_values.append(value)
        if self.knowledge_dim > 0:
            if knowledge is None:
                knowledge = np.zeros(self.knowledge_dim, dtype=np.float32)
            self.rollout_knowledge.append(
                np.asarray(knowledge, dtype=np.float32).reshape(-1)
            )
        if self.proprio_dim > 0:
            # a step with no self-state recorded gets NEUTRAL zeros, never a
            # dropped row — the rollout arrays must stay index-aligned
            # WIDTH-NORMALISED AT THE WRITE (fix 2026-08-02). Coercing only
            # at the READ was not enough: the rollout buffer is a plain list,
            # so mixed-width rows survived until `np.array(...)` at PPO-update
            # time produced a RAGGED array and killed the run —
            # "inhomogeneous shape after 1 dimensions". Every row must be
            # proprio_dim wide going IN, so the buffer can never become
            # ragged regardless of which path supplied the vector.
            self.rollout_proprio.append(
                self._fit_proprio_width(proprio))

    def _clear_rollout(self) -> None:
        """Drop every rollout column. ONE place, so a column added later
        cannot be cleared on the normal path and forgotten on the skip path —
        a half-cleared rollout is ragged on the next update, and that failure
        surfaces as advantages attached to the wrong observations."""
        for _c in (self.rollout_obs, self.rollout_actions,
                   self.rollout_rewards, self.rollout_dones,
                   self.rollout_log_probs, self.rollout_values,
                   self.rollout_knowledge, self.rollout_proprio,
                   self.rollout_taus, self.rollout_masks,
                   self.rollout_stream, self.rollout_feats):
            _c.clear()

    def should_update(self, env_step_target: int) -> bool:
        """Is the rollout ready for a PPO update?

        ---- WHY THIS IS NOT JUST `env_steps >= n_steps` (2026-09-01) ------
        The env-step trigger was correct as far as it went: counting
        DECISIONS would have let option use delay updates indefinitely. But
        it made the opposite error — it fires on schedule while the number of
        stored ROWS collapses with the option mix. At mean tau ~40 a
        1024-env-step rollout holds ~25 rows, and 25 rows is below any
        sensible minibatch, so the update degenerates to a single full-batch
        gradient step. Rows are scarce EXACTLY when options are used, which
        is the regime this project is trying to get into.

        So both conditions must hold: enough env steps to be on schedule AND
        enough rows to make a real update out of. `max_env_steps_per_update`
        is the escape path — a rollout that somehow never accumulates rows
        (every decision a maximal option) still updates rather than waiting
        forever, so this gate cannot become a latch. Memory is bounded by
        construction: the ceiling is only reachable while rows are FEW, and
        rows are what cost memory.

        Both bounds off (0) == the historical behaviour exactly.
        """
        _steps = self.rollout_env_steps()
        _rows = len(self.rollout_obs)
        # ROW CEILING (2026-09-01). Each row holds a full observation — at
        # 128px RGB that is ~196 KB — and three things now compound: rows come
        # from every stream, the row FLOOR can hold a rollout open across
        # several segments, and the trigger is only evaluated at segment end.
        # Two streams of all-primitive decisions reach ~2048 rows in one
        # segment (~400 MB), plus a second full copy at np.stack time in
        # train_step. That is survivable at 2 clients and not at 4, and 4 is a
        # configuration this repo has run.
        if self.max_rows_per_update > 0 and _rows >= self.max_rows_per_update:
            self.last_update_forced = True
            return True
        if self.max_env_steps_per_update > 0 and \
                _steps >= self.max_env_steps_per_update:
            self.last_update_forced = True
            return True
        self.last_update_forced = False
        if _steps < int(env_step_target):
            return False
        return _rows >= self.min_rows_per_update

    def rollout_env_steps(self) -> int:
        """ENV steps represented by the rollout buffer (sum of taus). The
        update trigger must count env steps, not decisions: under heavy
        option use 1024 decisions could span ~25k env steps. Equals
        len(rollout_obs) whenever all tau == 1 (options off) — parity."""
        if self.rollout_taus:
            return int(sum(self.rollout_taus))
        return len(self.rollout_obs)

    def train_step(self, n_epochs: int = 10,
                   last_value=0.0) -> Dict[str, float]:
        """
        PPO update using collected rollout data.

        Returns training metrics (policy loss, value loss, entropy).
        """
        if len(self.rollout_obs) < 2:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        # ---- EPOCHS SCALE WITH SAMPLE COUNT (2026-07-25 collapse fix) ------
        # The update TRIGGER counts env steps (rollout_env_steps, sum of taus)
        # so option use cannot delay updates — correct. But under heavy option
        # use one stored ROW spans tau~40 env steps, so a 1024-env-step update
        # can fire on as few as ~25 rows. This method has NO minibatching (the
        # configured policy.batch_size is not read here), so `n_epochs` full
        # passes over ~25 samples is ~10 near-identical gradient steps on a
        # tiny, advantage-normalized batch. Measured consequence: the option
        # logits were crushed and the meta-policy stopped selecting options
        # ENTIRELY — ~38,940 masked draws with zero option picks — which then
        # SELF-LOCKS, because a policy that never invokes an option generates
        # no option experience to learn from. Options fired 219 times and then
        # never again for 32,800 steps.
        # Bound the reuse instead: one epoch per ROWS_PER_EPOCH rows, capped at
        # the configured n_epochs. Large batches are unaffected (parity with
        # the old behaviour whenever rows >= n_epochs * ROWS_PER_EPOCH).
        # ---- THE CAP IS FOR THE FULL-BATCH PATH ONLY (2026-09-01) ---------
        # The collapse this guard was written for (2026-07-25) happened with
        # NO minibatching: `n_epochs` full-batch passes over ~25 rows is ~10
        # near-identical gradient steps on one advantage-normalized batch, and
        # it crushed the option logits. That is a property of REUSING one
        # batch, not of taking many steps — and the cap's cure was to remove
        # updates, which on the live config left ONE gradient step per 1024
        # env steps (~5 minutes of wall clock per policy update).
        #
        # So the cap now applies exactly where its evidence came from: the
        # un-minibatched path keeps it, byte-identical. When minibatching is
        # on, each step sees a DIFFERENT 64-row slice, reuse is bounded by
        # n_epochs directly, and the budget below is what bounds total
        # movement — with target_kl as the guard that actually watches the
        # policy rather than counting rows.
        _ROWS_PER_EPOCH = 32
        _rows = len(self.rollout_obs)
        _req = int(n_epochs)
        _mb_cfg = int(self.minibatch_size or 0)
        _will_mb = 0 < _mb_cfg < _rows
        if not _will_mb:
            n_epochs = max(1, min(_req, _rows // _ROWS_PER_EPOCH))
        # Gradient-step budget for the minibatched path. Bounds total movement
        # per rollout independently of how the rows happen to split, so a
        # segment that is mostly options (few, long rows) and one that is
        # mostly primitives (many, short rows) get comparable amounts of
        # learning instead of differing by 40x.
        _max_updates = max(4, _rows // 8) if _will_mb else 10 ** 9
        # ---- MAKE THE GUARD VISIBLE (2026-08-23) --------------------------
        # This cap prevents the collapse above by REMOVING UPDATES, and it
        # reported that at DEBUG — i.e. never, in production. Worse, the
        # loop's segment print annotated any epochs_run < n_epochs as
        # "stopped early (good: the policy hit its movement budget)", which
        # describes target_kl early-stopping. target_kl is 0.0 (withdrawn
        # 2026-08-05), so the ONLY reachable cause is this row cap — and a
        # starved policy was being reported as a healthy one.
        #
        # Under SMDP options one row spans tau env steps, so rows are scarce
        # exactly when options are used: at tau~40 a 1024-env-step update
        # holds ~25 rows -> 25//32 = 0 -> ONE full-batch gradient step. The
        # composition below is what distinguishes "options are eating the
        # rows" from "there simply was not much experience".
        _taus = list(self.rollout_taus) if self.rollout_taus else []
        _opt_rows = sum(1 for t in _taus if int(t) > 1)
        self.last_rows = int(_rows)
        self.last_rows_option = int(_opt_rows)
        self.last_rows_primitive = int(_rows - _opt_rows)
        self.last_tau_mean = (float(sum(_taus)) / len(_taus)) if _taus else 1.0
        self.last_env_steps = int(self.rollout_env_steps())
        self.last_epochs_requested = _req
        self.last_epochs_capped = bool(n_epochs < _req and not _will_mb)
        if self.last_epochs_capped:
            logger.warning(
                "PPO STARVED: %d rows (%d env steps, %d option rows, mean "
                "tau %.1f) -> %d/%d epochs. The tiny-batch collapse guard is "
                "removing updates; this is NOT target_kl early-stopping "
                "(target_kl is off).",
                _rows, self.last_env_steps, _opt_rows, self.last_tau_mean,
                n_epochs, _req)

        # Convert rollout data to tensors (all on self.device). GAE is computed
        # in numpy, so keep a CPU copy of the values (a CUDA tensor's .numpy()
        # would raise) and build the device tensors alongside it.
        # from_numpy shares the stacked array's memory (FloatTensor(np.array())
        # made a second full CPU copy — ~250-500MB per update on 128px obs).
        # DEQUANTIZE HERE, once, for the whole rollout. `.astype(np.float32)`
        # on a uint8 stack already makes the copy this line always made, so
        # the /255 rides along for free.
        _obs_np = np.stack(self.rollout_obs)
        if self._rollout_uint8:
            _obs_np = _obs_np.astype(np.float32) / 255.0
        else:
            _obs_np = _obs_np.astype(np.float32, copy=False)
        obs = torch.from_numpy(_obs_np).to(self.device)
        # arch='rssm': features come from the rollout, not from `obs`. `obs`
        # is still stored and stacked because the forensics (obs_spread) and
        # every other arch need it; it just is not the policy's input here.
        feats_t = (torch.from_numpy(
            np.stack(self.rollout_feats).astype(np.float32, copy=False)
        ).to(self.device) if self.arch == "rssm" else None)
        actions = (torch.LongTensor(self.rollout_actions) if not self.continuous
                   else torch.FloatTensor(self.rollout_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.rollout_log_probs).to(self.device)
        old_values_np = np.array(self.rollout_values, dtype=np.float32)
        old_values = torch.FloatTensor(old_values_np).to(self.device)
        rewards = np.array(self.rollout_rewards, dtype=np.float32)
        dones = np.array(self.rollout_dones, dtype=np.float32)
        # Knowledge features stored alongside each transition (Path A). Kept as
        # a tensor so the augmented input is rebuilt WITH gradient each epoch —
        # that is what lets the knowledge gate learn from the policy loss.
        knowledge = (
            torch.FloatTensor(np.array(self.rollout_knowledge)).to(self.device)
            if self.knowledge_dim > 0
            else None
        )
        proprio_t = (
            torch.FloatTensor(np.array(self.rollout_proprio)).to(self.device)
            if self.proprio_dim > 0 and self.rollout_proprio
            else None
        )

        # Compute GAE advantages (numpy on the CPU values copy).
        # taus: SMDP decision spans (all-ones == shipped behaviour).
        taus_np = (np.array(self.rollout_taus, dtype=np.float32)
                   if self.rollout_taus else None)
        streams_np = (np.array(self.rollout_stream, dtype=np.int64)
                      if self.rollout_stream else None)
        # The rollout arrays must stay index-aligned or the advantages get
        # attached to the wrong observations — the failure mode is silent and
        # looks like "PPO just isn't learning".
        assert streams_np is None or len(streams_np) == len(self.rollout_obs)
        advantages_np = self._compute_gae(rewards, old_values_np, dones,
                                          taus=taus_np, last_value=last_value,
                                          streams=streams_np)
        returns_np = advantages_np + old_values_np

        # ---- NEVER TRAIN ON NaN (2026-09-01) ----------------------------
        # A skipped update costs one rollout; a NaN gradient costs the run AND
        # the checkpoint, because Adam's moments go NaN with the weights and
        # every later update inherits it. The rollout is cleared either way so
        # the next segment starts clean rather than re-feeding the same bad
        # rows. Loud, because a silently skipped update looks exactly like a
        # policy that has stopped learning.
        if not (np.isfinite(advantages_np).all()
                and np.isfinite(returns_np).all()):
            logger.error(
                "PPO update SKIPPED: non-finite advantages/returns "
                "(%d rows, %d bad adv, %d bad ret). The value head is "
                "diverging — check value_clamped_frac and value_loss.",
                len(advantages_np),
                int((~np.isfinite(advantages_np)).sum()),
                int((~np.isfinite(returns_np)).sum()))
            self._clear_rollout()
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "n_updates": 0, "rows": int(len(advantages_np)),
                    "nonfinite_skip": True}

        # Selection-time action masks (options): build ONE (N, A) bool
        # tensor; rows stored as None -> all-True. Skipped entirely when no
        # mask was ever stored (flag off) -> byte-identical.
        masks_t = None
        if (not self.continuous
                and any(m is not None for m in self.rollout_masks)):
            n_act = self.actor.action_head.out_features
            mrows = np.ones((len(self.rollout_masks), n_act), dtype=bool)
            for i, m in enumerate(self.rollout_masks):
                if m is not None:
                    mrows[i, :len(m)] = m[:n_act]
            masks_t = torch.from_numpy(mrows).to(self.device)

        advantages = torch.FloatTensor(advantages_np).to(self.device)
        returns = torch.FloatTensor(returns_np).to(self.device)

        # ---- ENTROPY-COLLAPSE FORENSICS (2026-08-03) --------------------
        # Entropy falls to EXACTLY 0.00/2.48 within a couple of segments of
        # every restart, through minibatching on AND off and entropy_coef
        # 0.01 AND 0.03. That is not a tuning problem. Three candidates, and
        # they are told apart by measurement, not argument:
        #
        #   NOISE AMPLIFICATION — the reward stream is near-constant, so the
        #     RAW advantage spread is ~0; dividing by (std + 1e-8) rescales
        #     what is essentially noise to unit variance and PPO then takes a
        #     full-size gradient step on it every single update.
        #   STATE COLLAPSE — a stationary agent sees ~one observation, so a
        #     deterministic policy is genuinely OPTIMAL for that state and
        #     entropy 0 is correct behaviour, not a bug. Self-reinforcing:
        #     one action -> one state -> one action.
        #   SATURATION — logits already driven so far apart that the softmax
        #     is one-hot and no bounded entropy bonus can pull them back.
        #
        # raw_adv_std distinguishes the first, obs_spread the second,
        # max_prob the third. All three are cheap and computed once per
        # update, never in the step loop.
        _raw_adv_std = float(np.std(advantages_np))
        _raw_adv_absmean = float(np.mean(np.abs(advantages_np)))
        try:
            _obs_spread = float(obs.float().std(dim=0).mean().item())
        except Exception:
            _obs_spread = float("nan")

        # Normalize advantages
        _adv_clipped_frac = 0.0
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # ---- WINSORIZE (2026-09-01) ---------------------------------
            # Z-normalization is computed over the batch, so ONE huge row
            # drags the mean and inflates the std for everything else. On a
            # ~25-row rollout containing the first log break the outlier
            # lands near +4.9 and EVERY OTHER ROW GOES UNIFORMLY NEGATIVE —
            # a segment containing the run's first success actively
            # suppresses every other action taken in it, including the
            # approach and the swing that produced it. Bounding at +-5
            # leaves the log break the largest advantage in the batch
            # (nothing about the ordering changes) while stopping one row
            # from setting the sign of all the others.
            _ADV_LIMIT = 5.0
            _adv_clipped_frac = float(
                (advantages.abs() > _ADV_LIMIT).float().mean().item())
            advantages = advantages.clamp(-_ADV_LIMIT, _ADV_LIMIT)

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        # MINIBATCHED when policy.batch_size is set and smaller than the
        # rollout; otherwise `_chunks = [None]` and every tensor below is used
        # whole, which is the original full-batch path unchanged (no randperm
        # is drawn, so the RNG stream is untouched too).
        _n_rows = obs.shape[0]
        _mb = int(self.minibatch_size or 0)
        _use_mb = 0 < _mb < _n_rows
        _n_updates = 0
        _stop = False
        _epochs_run = 0
        # ACTOR-ONLY KL STOP (2026-09-01). The 2026-08-05 rollback of
        # target_kl diagnosed two faults; this is the second one's fix. Once
        # the policy has moved its budget, the ACTOR freezes but the CRITIC
        # keeps fitting to the end of the schedule. Previously the stop
        # abandoned the whole update, so a guard aimed at the policy left the
        # value function underfit — "larger advantage errors, bigger policy
        # moves" — which made the next update worse, i.e. the guard fed the
        # problem it existed to prevent.
        _actor_frozen = False
        _kl_stopped = False
        for _ in range(n_epochs):
            if _stop:
                break
            _epochs_run += 1
            if _use_mb:
                _perm = torch.randperm(_n_rows, device=self.device)
                _chunks = [_perm[i:i + _mb] for i in range(0, _n_rows, _mb)]
                # A trailing slice of 1-7 rows is noise, not a gradient: its
                # advantage mean is dominated by whichever rows happened to
                # land in it. Fold it away rather than taking a full-size Adam
                # step on it. (`or _chunks[:1]` keeps at least one chunk when
                # the whole rollout is shorter than the floor.)
                _chunks = [c for c in _chunks if len(c) >= 8] or _chunks[:1]
            else:
                _chunks = [None]
            for _ix in _chunks:
                _obs = obs if _ix is None else obs[_ix]
                _act = actions if _ix is None else actions[_ix]
                _olp = old_log_probs if _ix is None else old_log_probs[_ix]
                _adv = advantages if _ix is None else advantages[_ix]
                _ret = returns if _ix is None else returns[_ix]
                _kn = (knowledge if (knowledge is None or _ix is None)
                       else knowledge[_ix])
                _pro = (proprio_t if (proprio_t is None or _ix is None)
                        else proprio_t[_ix])
                _msk = (masks_t if (masks_t is None or _ix is None)
                        else masks_t[_ix])
                _ft = (None if feats_t is None
                       else (feats_t if _ix is None else feats_t[_ix]))

                # Rebuild the (optionally knowledge-augmented) policy input.
                # The gate — and under arch='conv' the encoder — runs WITH
                # gradient here so the whole stack is trained by the PPO loss.
                # arch='rssm' uses the STORED latent: h cannot be recomputed
                # from an observation, and the update must see exactly the
                # input the action was sampled from or every importance ratio
                # is against a different distribution.
                aug = self._augment(
                    _ft if _ft is not None else self._encode(_obs), _kn, _pro)

                # For discrete actions, we need log_prob of the TAKEN action
                features = self.actor.shared(aug)
                if not self.continuous:
                    logits = self._bound_logits(
                        self.actor.action_head(features))
                    if _msk is not None:
                        logits = logits.masked_fill(~_msk, -1e9)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(_act)
                    entropy = dist.entropy().mean()
                else:
                    mean = self.actor.mean_head(features)
                    std = self.actor.log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    new_log_probs = dist.log_prob(_act).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                new_values = self.critic(aug)

                # PPO clipped objective
                _logratio = new_log_probs - _olp
                ratio = _logratio.exp()
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range,
                                    1 + self.clip_range) * _adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- VALUE LOSS IN SYMLOG SPACE (2026-09-01) -------------
                # The head is compared against symlog(return), not the raw
                # return. WHY, in this task's numbers: a felled log pays
                # log_break_reward 20 plus 0.5/tick capped at 300 ticks, so a
                # single stored row can carry ~170 extrinsic -> ~51 after the
                # 0.3 extrinsic weight, against a typical mixed reward of
                # ~0.1. Against a prediction near 0 that is an MSE of ~2600
                # on the one rollout that matters, and at value_coef 0.5 it
                # buried the policy term completely. symlog turns 51 into
                # ~3.95 and 0.1 into ~0.095 — the rare event stays the
                # LARGEST target without being the ONLY one.
                # NOT a change to the incentive: the reward the agent is
                # optimizing is untouched, and GAE still runs entirely in
                # reward space (see _value_of). This is the estimator.
                value_loss = F.mse_loss(
                    new_values,
                    symlog(_ret) if self.value_space == "symlog" else _ret)

                # ACTOR FROZEN once the KL budget is spent, critic still
                # learning — see `_actor_frozen` above. Dropping the policy
                # and entropy terms (rather than breaking out) is what lets
                # the value function finish fitting.
                if _actor_frozen:
                    loss = self.value_coef * value_loss
                else:
                    loss = (policy_loss + self.value_coef * value_loss
                            - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                # ---- TWO CLIPS, NOT ONE (2026-09-01) ---------------------
                # A single clip_grad_norm_ over actor+critic normalises them
                # TOGETHER, so a large value gradient shrinks the policy
                # gradient in the same proportion. With the raw-return head
                # that happened precisely on the first log break: the update
                # carrying the run's only real signal became a value-fitting
                # step with the policy term scaled to near zero. Clipping the
                # groups separately gives each its own budget; the symlog
                # target above makes the spike small in the first place, and
                # this makes it structurally unable to steal the actor's step
                # even if it were not.
                torch.nn.utils.clip_grad_norm_(self._clip_actor, max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self._clip_critic, max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                _n_updates += 1
                # Schulman's k3 estimator: low-variance and non-negative,
                # unlike the naive mean(-logratio) which can go negative and
                # trip an early stop on noise.
                with torch.no_grad():
                    _kl = float(((ratio - 1.0) - _logratio).mean().item())
                self.last_approx_kl = _kl
                if (self.target_kl > 0.0 and _kl > self.target_kl
                        and not _actor_frozen):
                    _actor_frozen = True
                    _kl_stopped = True
                    if not _use_mb:
                        # full-batch path: there is no finer granularity to
                        # continue at, so preserve the historical behaviour
                        # exactly (stop the update) rather than spending the
                        # remaining epochs on critic-only full-batch steps.
                        _stop = True
                        break
                # GRADIENT-STEP BUDGET (minibatched path only; +inf
                # otherwise). Bounds total movement per rollout so the amount
                # of learning does not swing 40x with the option mix.
                if _n_updates >= _max_updates:
                    _stop = True
                    break

        # Clear rollout buffer
        self._clear_rollout()

        self.last_epochs_run = _epochs_run
        _vcf = (self._value_clamp_hits / self._value_clamp_n
                if self._value_clamp_n else 0.0)
        self._value_clamp_hits = self._value_clamp_n = 0
        # How deterministic is the policy NOW, on the states it just trained
        # on? mean max-probability: 1.0 == one-hot == nothing left to sample.
        _max_prob = float("nan")
        try:
            with torch.no_grad():
                _f = self.actor.shared(self._augment(
                    feats_t if feats_t is not None else self._encode(obs),
                    knowledge, proprio_t))
                _lg = self._bound_logits(self.actor.action_head(_f))
                if masks_t is not None:
                    _lg = _lg.masked_fill(~masks_t, -1e9)
                _pr = torch.softmax(_lg, dim=-1)
                _max_prob = float(_pr.max(dim=-1).values.mean().item())
        except Exception:
            pass

        # per-UPDATE means: with minibatching there are n_epochs * n_chunks
        # gradient steps, and dividing by n_epochs would inflate every metric
        # by the chunk count (the entropy readout drives the "<-- COLLAPSED"
        # diagnostic, so a scaled value there is actively misleading).
        _d = max(1, _n_updates)
        return {
            "policy_loss": total_policy_loss / _d,
            "value_loss": total_value_loss / _d,
            "entropy": total_entropy / _d,
            "n_updates": _n_updates,
            "approx_kl": self.last_approx_kl,
            "epochs_run": _epochs_run,
            "raw_adv_std": _raw_adv_std,
            "raw_adv_absmean": _raw_adv_absmean,
            "obs_spread": _obs_spread,
            "max_prob": _max_prob,
            "rows": int(_n_rows),
            # update-rate forensics (2026-08-23): `epochs_capped` is the one
            # that matters — it says the collapse guard removed updates,
            # which reads identically to KL early-stopping without it
            "epochs_requested": int(self.last_epochs_requested),
            "epochs_capped": bool(self.last_epochs_capped),
            "rows_option": int(self.last_rows_option),
            "rows_primitive": int(self.last_rows_primitive),
            "tau_mean": float(self.last_tau_mean),
            "env_steps": int(self.last_env_steps),
            # update-budget forensics (2026-09-01). `epochs_capped` used to be
            # the only signal and it conflated three causes; these separate
            # them: kl_stopped = the policy hit its movement budget (the
            # HEALTHY stop), max_updates = the rollout's step budget ran out,
            # epochs_capped = the full-batch tiny-row guard removed updates
            # (the STARVED case). adv_clipped_frac says how often one row was
            # trying to set the sign of the whole batch.
            "kl_stopped": bool(_kl_stopped),
            "max_updates": int(_max_updates if _max_updates < 10 ** 9 else 0),
            "minibatched": bool(_use_mb),
            "adv_clipped_frac": float(_adv_clipped_frac),
            "value_space": self.value_space,
            # Fraction of value READS that hit the symlog clamp since the last
            # update. Zero is the normal reading and the clamp is a pure
            # safety net; anything non-zero means the critic is diverging and
            # is the finding, not a nuisance to tune away.
            "value_clamped_frac": _vcf,
        }

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        taus=None,
        last_value: float = 0.0,
        streams=None,
    ) -> np.ndarray:
        """
        Generalized Advantage Estimation (GAE).

        Balances bias vs. variance in advantage estimation:
        - lambda=1: high variance, low bias (Monte Carlo-like)
        - lambda=0: low variance, high bias (TD-like)
        - lambda=0.95: good balance (standard choice)

        MULTI-STREAM (2026-09-01): when `streams` is given, rows are grouped
        by stream and the recursion below runs INDEPENDENTLY per group, in
        that group's own chronological order, with that stream's own
        bootstrap from `last_value`. The advantage trace is a statement
        about one trajectory's future; running it over interleaved bodies
        would credit env 0's reward to env 1's action and vice versa — the
        rows are adjacent in a list, not in time. Results are scattered back
        to the original row positions so every other array stays aligned.

        `streams=None` (or a single distinct stream) takes the original
        single-pass path unchanged.
        """
        if streams is not None:
            _st = np.asarray(streams)
            _uniq = np.unique(_st)
            if len(_uniq) > 1:
                out = np.zeros_like(rewards)
                _lv = last_value if isinstance(last_value, dict) else {}
                for s in _uniq:
                    _m = np.flatnonzero(_st == s)
                    _bv = (float(_lv.get(int(s), 0.0)) if _lv
                           else (float(last_value) if int(s) == 0 else 0.0))
                    out[_m] = self._compute_gae(
                        rewards[_m], values[_m], dones[_m],
                        taus=None if taus is None else taus[_m],
                        last_value=_bv, streams=None)
                return out
        if isinstance(last_value, dict):
            last_value = float(last_value.get(0, 0.0))
        # SMDP form (Sutton-Precup-Singh): decision t spanned tau_t env
        # steps and stored the ONLINE-ACCUMULATED discounted reward
        # R_t = sum over i<tau of gamma^i * r_(t+i). Bootstrapping uses
        # gamma^tau, and the GAE trace compounds (gamma*lambda)^tau so the
        # estimator's credit horizon stays fixed in PRIMITIVE time as the
        # option mix drifts. taus all-ones is arithmetically identical to
        # the shipped recursion (regression-anchored in the options smoke).
        if taus is None:
            taus = np.ones_like(rewards)
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # bootstrap: 0.0 on the episodic path (the last transition is a
                # real terminal, and (1-dones[t]) zeroes it anyway) -> BYTE-
                # IDENTICAL. In a lifelong segment that ends mid-trajectory,
                # the loop passes V(s_last) so the advantage trace is unbiased.
                next_value = float(last_value)
            else:
                next_value = values[t + 1]

            g_tau = self.gamma ** taus[t]
            gl_tau = (self.gamma * self.gae_lambda) ** taus[t]
            delta = rewards[t] + g_tau * next_value * (1 - dones[t]) - values[t]
            advantages[t] = (delta
                             + gl_tau * (1 - dones[t]) * last_advantage)
            last_advantage = advantages[t]

        return advantages

    def get_state_dict(self) -> Dict:
        """Get policy weights for saving to skill bank. Conv policies carry
        the encoder + an explicit arch marker so a loader can never confuse
        the two families by accident."""
        sd = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            # WHICH SPACE THE CRITIC SPEAKS (2026-09-01). Absent = a
            # pre-symlog checkpoint. Reading a linear head as symlog is wrong
            # by e^|v| and would look like a plausible value function rather
            # than a broken one, which is exactly the kind of silent
            # mis-scaling this project keeps paying for.
            "value_space": self.value_space,
        }
        if self.conditioner is not None:
            sd["conditioner"] = self.conditioner.state_dict()
        if self.encoder is not None:
            sd["encoder"] = self.encoder.state_dict()
            sd["arch"] = "conv"
            sd["enc_dim"] = int(self.enc_dim)
        elif self.arch == "wm" and self._shared_encoder is not None:
            # SNAPSHOT the shared encoder into the skill. The LIVE system has
            # exactly one encoder (that is the whole point), but a stored skill
            # has to be self-contained: it is loaded into a frozen option slot
            # possibly runs after the world model has moved on, and must keep
            # seeing what it saw when it was competent. ~3.6 MB, against the
            # ~116 MB a flat skill spent on three untrained pixel layers.
            sd["encoder"] = self._shared_encoder.state_dict()
            sd["arch"] = "wm"
            sd["enc_dim"] = int(self.enc_dim)
        return sd

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load policy weights from skill bank.

        ARCH GUARD FIRST (2026-07-26): a flat sd loaded into a conv policy
        (or vice versa) must raise BEFORE any submodule load starts — torch
        >= 2.6 copies matching params before raising on the first mismatch,
        so a late failure leaves a franken-policy. The caller's snapshot/
        restore guard handles the raise; this makes the raise happen at
        param zero. Shape mismatches (knowledge_dim drift) still raise from
        the submodule loads and are handled the same way."""
        # arch="wm" also ships an "encoder" (a snapshot of the shared world
        # model encoder), so the family test is "does this sd carry an
        # encoder" vs "does this policy read encoded features" — NOT
        # `self.encoder is not None`, which is None for wm by design.
        sd_arch = state_dict.get("arch", "conv" if "encoder" in state_dict
                                 else "flat")
        live_encoded = (self.encoder is not None) or (self.arch == "wm")
        if ("encoder" in state_dict) != live_encoded or sd_arch != self.arch:
            raise ValueError(
                f"policy arch mismatch: stored={sd_arch} live={self.arch} "
                f"— refusing before any partial copy")
        self.actor.load_state_dict(state_dict["actor"])
        # ---- VALUE-SPACE MIGRATION (2026-09-01) --------------------------
        # A checkpoint with no `value_space` marker holds a critic trained on
        # RAW returns. Loading it into a symlog policy would silently
        # mis-scale every value by e^|v| — the bootstrap, the advantages and
        # every option's stored value at once — and it would look like a
        # working value function, not a broken one.
        #
        # RE-INITIALISE rather than fall back. The alternative (keep the head
        # linear for the rest of that run) is safe but means the live brain
        # never gets the fix, which is the same as not shipping it. The
        # critic is the single most re-learnable module in the stack — two
        # linear layers refit within a few thousand updates — while the
        # actor, the conditioner and the world-model perception it reads are
        # all preserved untouched. Weigh that against what the linear head is
        # actually worth here: it has been fitting a target that spans 1e-3
        # to 1.7e2 with one gradient step per five minutes.
        _sd_space = state_dict.get("value_space")
        if _sd_space == self.value_space:
            self.critic.load_state_dict(state_dict["critic"])
        else:
            logger.warning(
                "critic value_space mismatch (stored=%s live=%s) — the value "
                "HEAD is re-initialised; actor/conditioner/encoder are loaded "
                "unchanged. Expect value_loss to be large for the first few "
                "updates while V refits; nothing else is lost.",
                _sd_space or "linear(pre-2026-09-01)", self.value_space)
        if self.conditioner is not None and "conditioner" in state_dict:
            self.conditioner.load_state_dict(state_dict["conditioner"])
        if self.encoder is not None:
            self.encoder.load_state_dict(state_dict["encoder"])
        # wm: the stored encoder snapshot is deliberately NOT loaded into the
        # shared module. Writing a skill's frozen perception back into the
        # live world model would let an old skill overwrite the agent's
        # current eyes — the encoder belongs to the world model, and only
        # prediction is allowed to change it.


# ---------------------------------------------------------------------------
# Dream Actor-Critic (latent-space policy for imagination-based training)
# ---------------------------------------------------------------------------

class DreamActorCritic(nn.Module):
    """
    Latent-space actor-critic trained on imagined trajectories (DreamerV3-style).

    Instead of training on real transitions (like PPO above), this trains on
    thousands of "dreamed" trajectories generated by the world model. One real
    environment step can generate many imagined training steps, making this
    massively more sample-efficient.

    Architecture:
      - Actor: latent_state → action distribution
      - Critic: latent_state → scalar value estimate
      - Slow target critic: EMA copy for stable lambda-return targets

    Training loop (called from DevelopmentalAI._dream_train_policy):
      1. Sample posterior states from replay buffer via world model
      2. Imagine H-step trajectory using this actor + world model dynamics
      3. Predict rewards/continues at each imagined step
      4. Compute lambda-returns from imagined rewards
      5. Actor: maximize lambda-returns via REINFORCE
      6. Critic: predict lambda-returns via MSE
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 400,
        continuous: bool = False,
        actor_lr: float = 3e-5,
        critic_lr: float = 3e-5,
        gamma: float = 0.997,
        lambda_: float = 0.95,
        entropy_scale: float = 3e-4,
        target_ema: float = 0.98,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_scale = entropy_scale
        self.target_ema = target_ema

        # ---- Actor network ----
        self.actor_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor_head = nn.Linear(hidden_dim, action_dim)

        # ---- Critic network ----
        self.critic_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---- Slow target critic (EMA for stable training) ----
        self.target_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.target_net.load_state_dict(self.critic_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        # ---- Separate optimizers (DreamerV3-style) ----
        actor_params = list(self.actor_net.parameters())
        if continuous:
            actor_params += list(self.actor_mean.parameters()) + [self.actor_log_std]
        else:
            actor_params += list(self.actor_head.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

    def _update_target(self):
        """Exponential moving average update of target critic."""
        for p, tp in zip(self.critic_net.parameters(), self.target_net.parameters()):
            tp.data.lerp_(p.data, 1.0 - self.target_ema)

    def get_action_dist(self, latent: torch.Tensor):
        """Get action distribution from latent state."""
        features = self.actor_net(latent)
        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)
        else:
            logits = self.actor_head(features)
            return torch.distributions.Categorical(logits=logits)

    def select_action(self, latent: torch.Tensor) -> Tuple[Any, Dict[str, float]]:
        """Select action for real environment interaction from RSSM latent."""
        with torch.no_grad():
            dist = self.get_action_dist(latent)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if self.continuous:
                log_prob = log_prob.sum(-1)
            value = self.critic_net(latent).squeeze(-1)

        if self.continuous:
            return action.squeeze(0).cpu().numpy(), {
                "log_prob": log_prob.item(),
                "value": value.item(),
            }
        else:
            return int(action.item()), {
                "log_prob": log_prob.item(),
                "value": value.item(),
            }

    def policy_fn(self, latent: torch.Tensor) -> torch.Tensor:
        """Action selection for world model imagination — returns tensor."""
        dist = self.get_action_dist(latent)
        action = dist.sample()
        if not self.continuous:
            action = F.one_hot(action, self.action_dim).float()
        return action

    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor,
    ) -> torch.Tensor:
        """
        DreamerV3-style lambda-returns for imagined trajectories.

        V_λ(s_t) = r_t + γ * c_t * ((1-λ) * v(s_{t+1}) + λ * V_λ(s_{t+1}))
        Bootstrap: V_λ(s_{H-1}) = r_{H-1} + γ * c_{H-1} * v(s_{H-1})
        """
        horizon = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        next_return = values[:, -1:]

        for t in reversed(range(horizon)):
            if t == horizon - 1:
                next_val = values[:, t : t + 1]
            else:
                next_val = values[:, t + 1 : t + 2]
            returns[:, t : t + 1] = rewards[:, t : t + 1] + self.gamma * continues[
                :, t : t + 1
            ] * ((1 - self.lambda_) * next_val + self.lambda_ * next_return)
            next_return = returns[:, t : t + 1]

        return returns

    def dream_training_step(
        self, imagined: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Train actor and critic on a single batch of imagined trajectories.

        Args:
            imagined: dict with 'latents' (B,H,D), 'actions' (B,H,A),
                      'rewards' (B,H,1), 'continues' (B,H,1)
                      Rewards should be in real scale (not symlog).
        """
        latents = imagined["latents"].detach()
        actions = imagined["actions"].detach()
        rewards = imagined["rewards"].detach()
        continues = imagined["continues"].detach()
        batch_size, horizon, _ = latents.shape

        # ---- Compute lambda-return targets ----
        with torch.no_grad():
            target_values = self.target_net(latents)
            lambda_returns = self.compute_lambda_returns(
                rewards, target_values, continues
            )

        # ---- Train critic ----
        values = self.critic_net(latents)
        critic_loss = F.mse_loss(values, lambda_returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 100.0)
        self.critic_optimizer.step()

        # ---- Train actor (REINFORCE on imagined advantages) ----
        with torch.no_grad():
            advantages = lambda_returns - self.critic_net(latents)
            adv_flat = advantages.reshape(-1)
            if adv_flat.numel() > 1:
                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        latents_flat = latents.reshape(-1, self.latent_dim)
        dist = self.get_action_dist(latents_flat)

        if self.continuous:
            actions_flat = actions.reshape(-1, self.action_dim)
            log_probs = dist.log_prob(actions_flat).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
        else:
            actions_flat = actions.reshape(-1, self.action_dim)
            action_indices = actions_flat.argmax(-1)
            log_probs = dist.log_prob(action_indices)
            entropy = dist.entropy().mean()

        actor_loss = -(log_probs * adv_flat).mean() - self.entropy_scale * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_params = list(self.actor_net.parameters())
        if self.continuous:
            actor_params += list(self.actor_mean.parameters()) + [self.actor_log_std]
        else:
            actor_params += list(self.actor_head.parameters())
        torch.nn.utils.clip_grad_norm_(actor_params, 100.0)
        self.actor_optimizer.step()

        # ---- Update slow target critic ----
        self._update_target()

        return {
            "dream_actor_loss": actor_loss.item(),
            "dream_critic_loss": critic_loss.item(),
            "dream_entropy": entropy.item(),
            "dream_returns_mean": lambda_returns.mean().item(),
        }
