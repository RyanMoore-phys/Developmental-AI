"""
Dream Visualizer — See What Your AI Is Dreaming
=================================================
Decodes the agent's imagined trajectories so you can watch, plot, and
compare what the world model predicts vs. what actually happens.

Usage:
    from developmental_ai.core import DevelopmentalAI
    from developmental_ai.visualization import DreamVisualizer

    agent = DevelopmentalAI(config_path="configs/default.yaml")
    agent.load_checkpoint("logs/checkpoints")

    viz = DreamVisualizer(agent)

    # Full dashboard saved to disk
    viz.dream_report(save_dir="dream_logs")

    # Interactive: watch a single dream unfold
    viz.plot_single_dream(horizon=30, show=True)
"""

import os
import logging
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class DreamVisualizer:
    """
    Captures, decodes, and visualizes imagined trajectories from the world model.

    Works with both vector observations (CartPole, LunarLander) and pixel
    observations (Atari, rendered MuJoCo). For vector envs it plots decoded
    observation dimensions over time; for pixel envs it renders image grids
    and optional GIF animations.
    """

    def __init__(self, agent):
        self.agent = agent
        self.wm = agent.world_model
        self.device = agent.device
        self.pixel_obs = agent.pixel_obs
        self.obs_labels = getattr(agent, "obs_labels", None)
        self.action_labels = getattr(agent, "action_labels", None)

    # ------------------------------------------------------------------
    # Core: capture a dream
    # ------------------------------------------------------------------

    def capture_dream(
        self,
        horizon: int = 30,
        batch_size: int = 1,
        from_replay: bool = True,
        knowledge: bool = True,
    ) -> Dict[str, Any]:
        """
        Run an imagination trajectory and return everything needed to visualize it.

        Args:
            horizon: How many timesteps to dream forward.
            batch_size: Number of parallel dreams (first one is used for plots).
            from_replay: If True, start from a replay-buffer state.
                         If False, start from the current environment reset.
            knowledge: Whether to condition dreams on the GNN knowledge vector.

        Returns dict with keys:
            latents     (batch, horizon, latent_dim)
            actions     (batch, horizon, action_dim)
            rewards     (batch, horizon, 1)       — real-scale predicted rewards
            continues   (batch, horizon, 1)
            decoded_obs (batch, horizon, obs_dim)  — decoder reconstruction
            starting_obs (batch, obs_dim)          — the real obs the dream started from
            knowledge_vec (batch, k_dim) or None
        """
        self.wm.eval()
        wm_cfg = self.agent.config.get("world_model", {})

        with torch.no_grad():
            # -- Obtain a starting RSSM state --
            if from_replay and len(self.agent.replay_buffer) > 0:
                batch = self.agent.replay_buffer.sample_sequences(
                    batch_size=batch_size,
                    seq_len=wm_cfg.get("sequence_length", 50),
                    device=self.device,
                )
                states, _ = self.wm.observe_sequence(
                    batch["observations"], batch["actions"]
                )
                seq_len = states["h"].shape[1]
                t_idx = torch.randint(0, seq_len, (batch_size,))
                starting_states = {
                    "h": states["h"][torch.arange(batch_size), t_idx],
                    "z": states["z"][torch.arange(batch_size), t_idx],
                }
                starting_obs = batch["observations"][
                    torch.arange(batch_size), t_idx
                ]
            else:
                obs, _ = self.agent.env.reset()
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                if batch_size > 1:
                    obs_t = obs_t.expand(batch_size, -1)
                encoded = self.wm.encoder(obs_t)
                zero_act = torch.zeros(
                    batch_size, self.agent.action_dim, device=self.device
                )
                init_state = self.wm.rssm.initial_state(batch_size, self.device)
                starting_states, _ = self.wm.rssm.observe_step(
                    init_state, zero_act, encoded
                )
                starting_obs = obs_t

            # -- Knowledge conditioning --
            k_vec = None
            if knowledge:
                raw = self.agent.glue.knowledge_integrator.knowledge_vector
                if raw is not None:
                    k_vec = raw.unsqueeze(0).expand(batch_size, -1)

            # -- Dream forward --
            imagined = self.wm.imagine_trajectory(
                starting_states,
                policy_fn=self.agent.dream_actor.policy_fn,
                horizon=horizon,
                knowledge=k_vec,
            )

            # -- Decode latents back to observation space --
            from developmental_ai.world_model.rssm import symexp
            flat_latents = imagined["latents"].reshape(-1, imagined["latents"].shape[-1])
            decoded_flat = self.wm.decoder(flat_latents)
            if not self.pixel_obs:
                decoded_flat = symexp(decoded_flat)
            decoded_obs = decoded_flat.reshape(
                batch_size, horizon, -1
            )

            rewards_real = symexp(imagined["rewards"])

        self.wm.train()

        return {
            "latents": imagined["latents"].cpu().numpy(),
            "actions": imagined["actions"].cpu().numpy(),
            "rewards": rewards_real.cpu().numpy(),
            "continues": imagined["continues"].cpu().numpy(),
            "decoded_obs": decoded_obs.cpu().numpy(),
            "starting_obs": starting_obs.cpu().numpy(),
            "knowledge_vec": k_vec.cpu().numpy() if k_vec is not None else None,
        }

    # ------------------------------------------------------------------
    # Capture a real episode for comparison
    # ------------------------------------------------------------------

    def capture_real_episode(self, max_steps: int = 200) -> Dict[str, np.ndarray]:
        """Run a real episode and record observations, actions, rewards."""
        env = self.agent.env
        obs, _ = env.reset()
        observations, actions, rewards = [obs], [], []

        rssm_state = self.wm.rssm.initial_state(1, self.device)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            encoded = self.wm.encoder(obs_t)
            zero_act = torch.zeros(1, self.agent.action_dim, device=self.device)
            rssm_state, _ = self.wm.rssm.observe_step(
                rssm_state, zero_act, encoded
            )

        for _ in range(max_steps):
            with torch.no_grad():
                if self.agent.dream_training_active:
                    latent = self.wm.rssm.get_latent(rssm_state)
                    action, _ = self.agent.dream_actor.select_action(latent)
                else:
                    action, _ = self.agent.policy.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            if self.agent.is_discrete:
                act_t = torch.zeros(1, self.agent.action_dim, device=self.device)
                act_t[0, int(action)] = 1.0
            else:
                act_t = torch.FloatTensor([action]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
                encoded = self.wm.encoder(next_obs_t)
                rssm_state, _ = self.wm.rssm.observe_step(
                    rssm_state, act_t, encoded
                )

            observations.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

            if terminated or truncated:
                break

        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
        }

    # ------------------------------------------------------------------
    # 1. Decoded dream playback (vector envs)
    # ------------------------------------------------------------------

    def plot_decoded_dream(
        self,
        dream: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        dream_idx: int = 0,
        dims: Optional[List[int]] = None,
    ) -> plt.Figure:
        """
        Plot decoded observation dimensions over the dream horizon.
        Each line is one observation dimension — you can see what the world
        model predicts will happen to cart position, pole angle, etc.
        """
        decoded = dream["decoded_obs"][dream_idx]  # (horizon, obs_dim)
        horizon, obs_dim = decoded.shape

        if dims is None:
            dims = list(range(min(obs_dim, 8)))

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        t = np.arange(horizon)
        for d in dims:
            label = (
                self.obs_labels[d]
                if self.obs_labels and d < len(self.obs_labels)
                else f"obs[{d}]"
            )
            ax.plot(t, decoded[:, d], label=label, linewidth=1.5)

        ax.set_xlabel("Dream timestep")
        ax.set_ylabel("Predicted observation value")
        ax.set_title("Decoded Dream — What the World Model Imagines")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        if standalone:
            fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. Decoded dream playback (pixel envs — image grid)
    # ------------------------------------------------------------------

    def plot_pixel_dream(
        self,
        dream: Dict[str, Any],
        dream_idx: int = 0,
        cols: int = 8,
    ) -> plt.Figure:
        """
        For pixel-based envs: render the decoded dream as an image grid.
        Each cell is one timestep of the imagined trajectory.
        """
        decoded = dream["decoded_obs"][dream_idx]  # (horizon, C*H*W)
        horizon = decoded.shape[0]
        rows = int(np.ceil(horizon / cols))

        ch = self.agent.image_channels
        sz = self.agent.image_size
        frames = decoded.reshape(horizon, ch, sz, sz)
        frames = np.clip(frames, 0, 1)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = np.atleast_2d(axes)

        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.axis("off")
            if i < horizon:
                img = np.transpose(frames[i], (1, 2, 0))
                if ch == 1:
                    ax.imshow(img[:, :, 0], cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(img)
                ax.set_title(f"t={i}", fontsize=7)

        fig.suptitle("Pixel Dream Sequence", fontsize=12)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. Latent space trajectory (PCA)
    # ------------------------------------------------------------------

    def plot_latent_trajectory(
        self,
        dream: Dict[str, Any],
        dream_idx: int = 0,
        method: str = "pca",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Project latent states to 2D and plot the dream trajectory.
        Points are colored by predicted reward — bright = high reward.
        Arrows show the direction of time.
        """
        latents = dream["latents"][dream_idx]  # (horizon, latent_dim)
        rewards = dream["rewards"][dream_idx].flatten()
        horizon = latents.shape[0]

        if method == "pca":
            reducer = PCA(n_components=2)
            projected = reducer.fit_transform(latents)
            explained = reducer.explained_variance_ratio_
            axis_labels = (
                f"PC1 ({explained[0]:.0%} var)",
                f"PC2 ({explained[1]:.0%} var)",
            )
        else:
            from sklearn.manifold import TSNE
            projected = TSNE(n_components=2, perplexity=min(5, horizon - 1)).fit_transform(latents)
            axis_labels = ("t-SNE 1", "t-SNE 2")

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Line colored by reward
        points = projected.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(rewards.min(), rewards.max())
        lc = LineCollection(segments, cmap="RdYlGn", norm=norm, linewidth=2, alpha=0.7)
        lc.set_array(rewards[:-1])
        ax.add_collection(lc)

        # Scatter points sized by timestep (later = bigger)
        sizes = np.linspace(30, 120, horizon)
        sc = ax.scatter(
            projected[:, 0], projected[:, 1],
            c=rewards, cmap="RdYlGn", s=sizes,
            edgecolors="black", linewidth=0.5, zorder=3,
        )

        # Start and end markers
        ax.scatter(*projected[0], marker="s", s=200, c="blue",
                   edgecolors="black", linewidth=2, zorder=4, label="Dream start")
        ax.scatter(*projected[-1], marker="*", s=300, c="red",
                   edgecolors="black", linewidth=2, zorder=4, label="Dream end")

        # Direction arrows at a few intermediate points
        arrow_indices = np.linspace(0, horizon - 2, min(6, horizon - 1), dtype=int)
        for i in arrow_indices:
            dx = projected[i + 1, 0] - projected[i, 0]
            dy = projected[i + 1, 1] - projected[i, 1]
            ax.annotate("", xy=(projected[i + 1, 0], projected[i + 1, 1]),
                        xytext=(projected[i, 0], projected[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Predicted reward")
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title("Dream Trajectory in Latent Space")
        ax.legend(fontsize=8)
        ax.autoscale_view()

        if standalone:
            fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. Dream metrics dashboard
    # ------------------------------------------------------------------

    def plot_dream_metrics(
        self,
        dream: Dict[str, Any],
        dream_idx: int = 0,
    ) -> plt.Figure:
        """
        Per-timestep reward predictions, continuation probabilities, and
        action magnitudes — the narrative arc of one dream.
        """
        rewards = dream["rewards"][dream_idx].flatten()
        continues = dream["continues"][dream_idx].flatten()
        actions = dream["actions"][dream_idx]
        horizon = len(rewards)
        t = np.arange(horizon)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        # Predicted rewards
        ax = axes[0]
        colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in rewards]
        ax.bar(t, rewards, color=colors, alpha=0.8, width=0.8)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Predicted reward")
        ax.set_title("Dream Narrative — Step by Step")
        ax.grid(True, alpha=0.2)

        cumulative = np.cumsum(rewards)
        ax2 = ax.twinx()
        ax2.plot(t, cumulative, color="#3498db", linewidth=2, linestyle="--", label="Cumulative")
        ax2.set_ylabel("Cumulative reward", color="#3498db")
        ax2.legend(fontsize=8, loc="upper left")

        # Continuation probability
        ax = axes[1]
        ax.fill_between(t, continues, alpha=0.4, color="#9b59b6")
        ax.plot(t, continues, color="#9b59b6", linewidth=1.5)
        ax.set_ylabel("P(continue)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
        ax.grid(True, alpha=0.2)

        # Action trace
        ax = axes[2]
        if self.agent.is_discrete:
            chosen = np.argmax(actions, axis=-1)
            ax.step(t, chosen, where="mid", linewidth=1.5, color="#e67e22")
            if self.action_labels:
                ax.set_yticks(range(len(self.action_labels)))
                ax.set_yticklabels(self.action_labels, fontsize=8)
            ax.set_ylabel("Action (discrete)")
        else:
            for d in range(min(actions.shape[-1], 4)):
                ax.plot(t, actions[:, d], label=f"act[{d}]", linewidth=1.2)
            ax.set_ylabel("Action value")
            ax.legend(fontsize=8)
        ax.set_xlabel("Dream timestep")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 5. Real vs. dream comparison
    # ------------------------------------------------------------------

    def plot_real_vs_dream(
        self,
        dream: Dict[str, Any],
        real: Dict[str, np.ndarray],
        dream_idx: int = 0,
        dims: Optional[List[int]] = None,
    ) -> plt.Figure:
        """
        Side-by-side comparison of real episode observations vs. what the
        world model imagined. Shows where the model is accurate and where
        it drifts.
        """
        decoded = dream["decoded_obs"][dream_idx]
        real_obs = real["observations"]
        horizon = min(len(decoded), len(real_obs))
        obs_dim = decoded.shape[-1]

        if dims is None:
            dims = list(range(min(obs_dim, 6)))

        n_dims = len(dims)
        fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2.2 * n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]

        t = np.arange(horizon)

        for i, d in enumerate(dims):
            ax = axes[i]
            label = (
                self.obs_labels[d]
                if self.obs_labels and d < len(self.obs_labels)
                else f"obs[{d}]"
            )

            ax.plot(t, real_obs[:horizon, d], color="#2c3e50", linewidth=1.5,
                    label="Real", alpha=0.9)
            ax.plot(t, decoded[:horizon, d], color="#e74c3c", linewidth=1.5,
                    linestyle="--", label="Dreamed", alpha=0.8)

            # Shade the divergence
            ax.fill_between(
                t, real_obs[:horizon, d], decoded[:horizon, d],
                alpha=0.15, color="#e74c3c",
            )

            ax.set_ylabel(label, fontsize=9)
            if i == 0:
                ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.2)

        axes[0].set_title("Reality vs. Dream — Where the Model Drifts")
        axes[-1].set_xlabel("Timestep")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 6. Reconstruction accuracy (how well does the encoder→decoder round-trip work?)
    # ------------------------------------------------------------------

    def plot_reconstruction(
        self,
        n_sequences: int = 4,
    ) -> plt.Figure:
        """
        Feed real observations through encoder→RSSM→decoder and compare
        the reconstruction to the original. This tests the world model's
        ability to represent states it has actually seen (no imagination).
        """
        wm_cfg = self.agent.config.get("world_model", {})
        seq_len = wm_cfg.get("sequence_length", 15)

        batch = self.agent.replay_buffer.sample_sequences(
            batch_size=n_sequences, seq_len=seq_len, device=self.device,
        )
        real_obs = batch["observations"]

        self.wm.eval()
        with torch.no_grad():
            states, _ = self.wm.observe_sequence(real_obs, batch["actions"])
            latents = torch.cat([states["h"], states["z"]], dim=-1)
            pred = self.wm.decoder(latents.reshape(-1, latents.shape[-1]))
            from developmental_ai.world_model.rssm import symexp
            if not self.pixel_obs:
                pred = symexp(pred)
            pred = pred.reshape(n_sequences, seq_len, -1)

        real_np = real_obs.cpu().numpy()
        pred_np = pred.cpu().numpy()
        obs_dim = real_np.shape[-1]
        dims = list(range(min(obs_dim, 4)))

        fig, axes = plt.subplots(len(dims), 1, figsize=(10, 2.2 * len(dims)), sharex=True)
        if len(dims) == 1:
            axes = [axes]

        t = np.arange(seq_len)
        for i, d in enumerate(dims):
            ax = axes[i]
            label = (
                self.obs_labels[d]
                if self.obs_labels and d < len(self.obs_labels)
                else f"obs[{d}]"
            )
            for s in range(n_sequences):
                alpha = 0.6 if n_sequences > 1 else 0.9
                ax.plot(t, real_np[s, :, d], color="#2c3e50", linewidth=1.2,
                        alpha=alpha, label="Real" if s == 0 else None)
                ax.plot(t, pred_np[s, :, d], color="#27ae60", linewidth=1.2,
                        linestyle="--", alpha=alpha, label="Reconstructed" if s == 0 else None)
            ax.set_ylabel(label, fontsize=9)
            if i == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        mse = np.mean((real_np[:, :, dims] - pred_np[:, :, dims]) ** 2)
        axes[0].set_title(f"Reconstruction Accuracy (MSE={mse:.4f})")
        axes[-1].set_xlabel("Timestep")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 7. Multi-dream latent cloud
    # ------------------------------------------------------------------

    def plot_dream_cloud(
        self,
        n_dreams: int = 10,
        horizon: int = 20,
    ) -> plt.Figure:
        """
        Launch many dreams from different starting points and plot all their
        latent trajectories together. Shows the diversity of imagined futures.
        """
        dream = self.capture_dream(horizon=horizon, batch_size=n_dreams)
        all_latents = dream["latents"]  # (n_dreams, horizon, latent_dim)
        all_rewards = dream["rewards"]  # (n_dreams, horizon, 1)

        flat = all_latents.reshape(-1, all_latents.shape[-1])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(flat).reshape(n_dreams, horizon, 2)

        fig, ax = plt.subplots(figsize=(9, 7))
        cmap = plt.cm.tab10

        for i in range(n_dreams):
            color = cmap(i % 10)
            ax.plot(proj[i, :, 0], proj[i, :, 1],
                    color=color, alpha=0.5, linewidth=1)
            ax.scatter(proj[i, 0, 0], proj[i, 0, 1],
                       marker="o", s=60, color=color, edgecolors="black", zorder=3)
            ax.scatter(proj[i, -1, 0], proj[i, -1, 1],
                       marker="x", s=60, color=color, zorder=3)

            total_r = float(all_rewards[i].sum())
            ax.annotate(f"R={total_r:.1f}",
                        (proj[i, -1, 0], proj[i, -1, 1]),
                        fontsize=7, alpha=0.7)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.set_title(f"Dream Cloud — {n_dreams} Imagined Futures")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 7. GIF animation (pixel envs)
    # ------------------------------------------------------------------

    def save_dream_gif(
        self,
        dream: Dict[str, Any],
        filepath: str = "dream.gif",
        dream_idx: int = 0,
        fps: int = 5,
    ) -> str:
        """Save decoded pixel dream as an animated GIF."""
        try:
            from PIL import Image
        except ImportError:
            logger.warning("Pillow required for GIF export: pip install Pillow")
            return ""

        decoded = dream["decoded_obs"][dream_idx]
        horizon = decoded.shape[0]
        ch = self.agent.image_channels
        sz = self.agent.image_size

        frames_np = decoded.reshape(horizon, ch, sz, sz)
        frames_np = np.clip(frames_np * 255, 0, 255).astype(np.uint8)

        pil_frames = []
        for i in range(horizon):
            img = np.transpose(frames_np[i], (1, 2, 0))
            if ch == 1:
                img = img[:, :, 0]
            pil_frames.append(Image.fromarray(img))

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        pil_frames[0].save(
            filepath, save_all=True, append_images=pil_frames[1:],
            duration=1000 // fps, loop=0,
        )
        logger.info(f"Dream GIF saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # 8. Full dashboard
    # ------------------------------------------------------------------

    def dream_report(
        self,
        save_dir: str = "dream_logs",
        horizon: int = 30,
        n_cloud: int = 8,
        show: bool = False,
    ) -> Dict[str, str]:
        """
        Generate a full dream visualization report.

        Saves all plots to save_dir and returns a dict of filepaths.
        """
        os.makedirs(save_dir, exist_ok=True)
        saved = {}

        print(f"Capturing dream (horizon={horizon})...")
        dream = self.capture_dream(horizon=horizon)

        # --- Decoded observations ---
        if self.pixel_obs:
            fig = self.plot_pixel_dream(dream)
            path = os.path.join(save_dir, "dream_decoded_pixels.png")
            fig.savefig(path, dpi=150)
            saved["decoded_pixels"] = path

            gif_path = os.path.join(save_dir, "dream.gif")
            self.save_dream_gif(dream, gif_path)
            saved["dream_gif"] = gif_path
        else:
            fig = self.plot_decoded_dream(dream)
            path = os.path.join(save_dir, "dream_decoded_obs.png")
            fig.savefig(path, dpi=150)
            saved["decoded_obs"] = path

        if not show:
            plt.close(fig)

        # --- Latent trajectory ---
        print("Plotting latent trajectory...")
        fig = self.plot_latent_trajectory(dream)
        path = os.path.join(save_dir, "dream_latent_trajectory.png")
        fig.savefig(path, dpi=150)
        saved["latent_trajectory"] = path
        if not show:
            plt.close(fig)

        # --- Metrics dashboard ---
        print("Plotting dream metrics...")
        fig = self.plot_dream_metrics(dream)
        path = os.path.join(save_dir, "dream_metrics.png")
        fig.savefig(path, dpi=150)
        saved["metrics"] = path
        if not show:
            plt.close(fig)

        # --- Dream cloud ---
        print(f"Launching {n_cloud} parallel dreams for cloud plot...")
        fig = self.plot_dream_cloud(n_dreams=n_cloud, horizon=horizon)
        path = os.path.join(save_dir, "dream_cloud.png")
        fig.savefig(path, dpi=150)
        saved["dream_cloud"] = path
        if not show:
            plt.close(fig)

        # --- Reconstruction accuracy ---
        if not self.pixel_obs:
            print("Plotting reconstruction accuracy...")
            fig = self.plot_reconstruction()
            path = os.path.join(save_dir, "reconstruction_accuracy.png")
            fig.savefig(path, dpi=150)
            saved["reconstruction"] = path
            if not show:
                plt.close(fig)

        # --- Real vs. dream comparison (fair: same starting state) ---
        if not self.pixel_obs:
            print("Running matched real-vs-dream comparison...")
            real = self.capture_real_episode(max_steps=horizon)
            matched_dream = self.capture_dream(
                horizon=min(horizon, len(real["observations"])),
                from_replay=False,
            )
            fig = self.plot_real_vs_dream(matched_dream, real)
            path = os.path.join(save_dir, "real_vs_dream.png")
            fig.savefig(path, dpi=150)
            saved["real_vs_dream"] = path
            if not show:
                plt.close(fig)

        if show:
            plt.show()

        print(f"\nDream report saved to {save_dir}/")
        for name, path in saved.items():
            print(f"  {name}: {path}")

        return saved

    # ------------------------------------------------------------------
    # Convenience: single dream with interactive display
    # ------------------------------------------------------------------

    def plot_single_dream(
        self, horizon: int = 30, show: bool = True
    ) -> plt.Figure:
        """Quick one-shot: capture a dream and show a combined figure."""
        dream = self.capture_dream(horizon=horizon)

        if self.pixel_obs:
            fig = self.plot_pixel_dream(dream)
        else:
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            self.plot_decoded_dream(dream, ax=ax1)

            ax2 = fig.add_subplot(gs[0, 1])
            self.plot_latent_trajectory(dream, ax=ax2)

            # Metrics in bottom row (full width)
            ax_rew = fig.add_subplot(gs[1, 0])
            rewards = dream["rewards"][0].flatten()
            continues = dream["continues"][0].flatten()
            t = np.arange(len(rewards))
            colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in rewards]
            ax_rew.bar(t, rewards, color=colors, alpha=0.8)
            ax_rew.set_ylabel("Predicted reward")
            ax_rew.set_xlabel("Dream timestep")
            ax_rew.set_title("Reward predictions")
            ax_rew.grid(True, alpha=0.2)

            ax_cont = fig.add_subplot(gs[1, 1])
            ax_cont.fill_between(t, continues, alpha=0.4, color="#9b59b6")
            ax_cont.plot(t, continues, color="#9b59b6", linewidth=1.5)
            ax_cont.set_ylabel("P(continue)")
            ax_cont.set_xlabel("Dream timestep")
            ax_cont.set_title("Episode continuation confidence")
            ax_cont.set_ylim(-0.05, 1.05)
            ax_cont.grid(True, alpha=0.2)

            fig.suptitle(
                f"Dream Visualization — {self.agent.env_name}",
                fontsize=13, fontweight="bold",
            )
            fig.tight_layout()

        if show:
            plt.show()
        return fig
