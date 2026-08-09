"""
Replay Buffer for World Model Training
========================================
Stores transitions (obs, action, reward, done) from real environment
interaction. The world model trains on randomly sampled SEQUENCES from this
buffer, not individual transitions — this is critical because the RSSM
needs temporal context to learn dynamics.

Design decisions:
  - Fixed-capacity circular buffer (oldest experience is overwritten)
  - Samples contiguous sequences, not random individual transitions
  - Stores everything as numpy arrays, converts to torch on sampling
  - Episode boundaries are tracked to avoid sampling across episodes

Performance / quality features (added for the parallel/async refactor):
  - VECTORIZED sampling: valid-start detection and sequence gathering are
    done with numpy fancy-indexing instead of Python loops. The old
    double-loop over the whole buffer on every sample was a real bottleneck.
  - PRIORITIZED EXPERIENCE REPLAY (PER): sequences can be sampled in
    proportion to how badly the world model predicted them (prediction
    error). New experience starts at max priority so it is seen at least
    once. Importance-sampling weights correct the bias this introduces.
    PER is OPT-IN — with prioritized=False the buffer behaves exactly as
    the original uniform sampler.
  - ASYNCHRONOUS sampling: a BackgroundSampler prefetches the next batch in
    a worker thread while the main thread runs the (CPU/GPU-bound) gradient
    step, so sampling and training overlap. This is safe because sampling
    only reads numpy arrays + builds tensors; it never touches model params.
"""

import threading
import queue
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Stores experience for world model training.

    The key difference from a standard RL replay buffer is that we sample
    SEQUENCES of consecutive transitions, because the RSSM needs to
    process temporal sequences to learn dynamics.

    Usage:
        buffer = ReplayBuffer(capacity=100000, obs_dim=4, action_dim=2)
        buffer.add(obs, action, reward, done)
        batch = buffer.sample_sequences(batch_size=16, seq_len=50)

    Prioritized usage:
        batch = buffer.sample_sequences(16, 50, prioritized=True)
        # ... train, get per-sequence prediction error ...
        buffer.update_priorities(batch["start_indices"], seq_len, errors)
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_epsilon: float = 1e-2,
        obs_uint8: bool = False,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # uint8 observation storage (rich-env program, step 8): for [0,1]
        # pixel observations, store quantized uint8 (x255) and dequantize on
        # gather — a 4x RAM cut (100k x 12288-d drops ~4.9GB -> ~1.2GB) at a
        # quantization error of 1/255, far below reconstruction noise. ONLY
        # valid for [0,1] obs (the pixel path); vector envs keep float32
        # (their normalized obs can be negative / unbounded).
        self._obs_uint8 = bool(obs_uint8)

        # Pre-allocate arrays for efficiency
        self.observations = np.zeros(
            (capacity, obs_dim),
            dtype=np.uint8 if self._obs_uint8 else np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # ---- Prioritized Experience Replay state ----
        # Per-transition priority. New transitions get max_priority so they
        # are guaranteed to be sampled at least once before being deprioritized.
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self.per_alpha = per_alpha      # how strongly to prioritize (0 = uniform)
        self.per_beta = per_beta        # importance-sampling correction strength
        self.per_epsilon = per_epsilon  # floor so nothing has zero probability

        # Track episode boundaries so we don't sample across episode breaks
        self.episode_starts = []  # Indices where new episodes begin

        self.position = 0    # Current write position (circular)
        self.size = 0        # How many transitions stored so far
        self._current_episode_start = 0

        # A lock guarding writes so a background sampler can read safely.
        self._lock = threading.Lock()

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """
        Add a single transition to the buffer.

        Args:
            observation: Current observation
            action: Action taken (will be converted to array if scalar)
            reward: Reward received
            done: Whether episode ended
        """
        # Handle scalar/discrete actions by converting to array
        if np.isscalar(action):
            action_array = np.zeros(self.action_dim, dtype=np.float32)
            action_array[int(action)] = 1.0  # One-hot encode discrete actions
        else:
            action_array = np.asarray(action, dtype=np.float32)

        if self._obs_uint8:
            observation = np.clip(
                np.round(np.asarray(observation, np.float32) * 255.0),
                0, 255).astype(np.uint8)

        with self._lock:
            self.observations[self.position] = observation
            self.actions[self.position] = action_array
            self.rewards[self.position] = reward
            self.dones[self.position] = float(done)
            # New experience enters at max priority so PER will sample it soon.
            self.priorities[self.position] = self.max_priority

            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            # Track episode boundaries
            if done:
                self.episode_starts.append(self.position)
                self._current_episode_start = self.position

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device = torch.device("cpu"),
        terminal_fraction: float = 0.25,
        prioritized: bool = False,
        beta: Optional[float] = None,
        reward_fraction: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample random contiguous sequences for world model training.

        Ensures a fraction of sequences include episode termination so the
        continue predictor sees enough done=True examples to learn from.
        This terminal oversampling is preserved even when PER is on, because
        the continue predictor depends on it.

        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence
            device: Torch device for output tensors
            terminal_fraction: Target fraction of sequences that include a done
            prioritized: If True, sample in proportion to transition priority
                (prediction error) and return importance-sampling weights.
            beta: Importance-sampling exponent (defaults to self.per_beta).

        Returns:
            Dict with 'observations', 'actions', 'rewards', 'continues',
            'weights' (importance-sampling weights, all 1.0 when not
            prioritized) and 'start_indices' (buffer indices of each
            sequence start, used to update priorities after training).
            The first four have shape (batch_size, seq_len, ...).
        """
        if self.size < seq_len + 1:
            raise ValueError(
                f"Buffer has {self.size} transitions but needs at least {seq_len + 1}. "
                f"Collect more experience first."
            )

        beta = self.per_beta if beta is None else beta

        # Take a consistent snapshot under the lock (cheap — just views/copies
        # of small index arrays). The heavy gather is done outside the lock.
        with self._lock:
            valid_starts, terminal_starts = self._find_valid_starts(seq_len)
            n_valid = len(valid_starts)
            n_terminal = int(batch_size * terminal_fraction)
            n_reward = int(batch_size * reward_fraction)

            # --- choose terminal sequences (preserves continue-predictor fix) ---
            if len(terminal_starts) > 0 and n_terminal > 0:
                t_probs = self._priority_probs(terminal_starts, seq_len) if prioritized else None
                t_pick = np.random.choice(
                    len(terminal_starts), size=n_terminal, replace=True, p=t_probs
                )
                t_idx = terminal_starts[t_pick]
            else:
                t_idx = np.array([], dtype=np.int64)
                n_terminal = 0

            # --- choose reward-bearing sequences (GOAL-PRIORITIZED REPLAY) ---
            # On sparse-reward tasks the goal transition is rare, so the world
            # model rarely sees the reward/continue/dynamics near the goal. Carve
            # out a fraction of the batch for windows that actually contain a
            # nonzero reward, so the WM gets the goal signal far more often. The
            # terminal oversampling above does NOT cover this: most terminals are
            # timeouts (reward 0) on hard envs.
            if n_reward > 0:
                reward_starts = self._find_reward_starts(seq_len)
                if len(reward_starts) > 0:
                    r_probs = self._priority_probs(reward_starts, seq_len) if prioritized else None
                    r_pick = np.random.choice(
                        len(reward_starts), size=n_reward, replace=True, p=r_probs
                    )
                    r_idx = reward_starts[r_pick]
                else:
                    r_idx = np.array([], dtype=np.int64)
                    n_reward = 0
            else:
                r_idx = np.array([], dtype=np.int64)
                n_reward = 0

            # --- choose the remaining (normal) sequences ---
            n_normal = max(0, batch_size - n_terminal - n_reward)
            source = valid_starts if n_valid > 0 else terminal_starts
            n_probs = self._priority_probs(source, seq_len) if prioritized else None
            if n_normal > 0:
                n_pick = np.random.choice(
                    len(source), size=n_normal, replace=True, p=n_probs
                )
                n_idx = source[n_pick]
            else:
                n_idx = np.array([], dtype=np.int64)

            indices = np.concatenate(
                [a for a in (t_idx, r_idx, n_idx) if len(a) > 0]
            )

            # --- importance-sampling weights (only meaningful for PER) ---
            if prioritized:
                # The true sampling distribution is a MIXTURE over the three
                # strata (terminal / reward / normal), each with its own
                # within-stratum probabilities: P(s) = sum_X frac_X * p_X(s).
                # The old code looked up only the normal stratum, so the
                # oversampled terminal/reward windows silently fell back to a
                # uniform probability — wrong IS weights for exactly the rare
                # high-value windows PER is meant to correct. (The constant
                # buffer-size factor cancels under max-normalization below.)
                batch_n = float(max(1, len(indices)))
                strata = []
                if n_terminal > 0:
                    strata.append((terminal_starts, t_probs, n_terminal))
                if n_reward > 0:
                    strata.append((reward_starts, r_probs, n_reward))
                if len(indices) > n_terminal + n_reward:
                    strata.append(
                        (source, n_probs, len(indices) - n_terminal - n_reward))
                mixture = {}
                for st_starts, st_probs, st_n in strata:
                    frac = st_n / batch_n
                    if st_probs is None:  # uniform fallback within the stratum
                        st_probs = np.full(
                            len(st_starts), 1.0 / max(1, len(st_starts)))
                    for s, p in zip(st_starts, st_probs):
                        mixture[int(s)] = mixture.get(int(s), 0.0) + frac * float(p)
                sampled_p = np.array(
                    [mixture[int(s)] for s in indices], dtype=np.float32)
                weights = sampled_p ** (-beta)
                weights = weights / (weights.max() + 1e-8)
            else:
                weights = np.ones(len(indices), dtype=np.float32)

            # Gather while still holding the lock so writes can't tear a row.
            obs_seqs, act_seqs, rew_seqs, cont_seqs = self._gather(indices, seq_len)

        return {
            "observations": torch.from_numpy(obs_seqs).to(device),
            "actions": torch.from_numpy(act_seqs).to(device),
            "rewards": torch.from_numpy(rew_seqs).to(device),
            "continues": torch.from_numpy(cont_seqs).to(device),
            "weights": torch.from_numpy(weights).to(device),
            "start_indices": indices.astype(np.int64),
        }

    def _gather(
        self, indices: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized gather of contiguous sequences (no Python loops)."""
        # (batch, seq_len) matrix of absolute buffer indices.
        idx = (indices[:, None] + np.arange(seq_len)[None, :]) % self.capacity
        obs_seqs = self.observations[idx]          # (B, T, obs_dim)
        if self._obs_uint8:
            obs_seqs = obs_seqs.astype(np.float32) / 255.0
        act_seqs = self.actions[idx]               # (B, T, action_dim)
        rew_seqs = self.rewards[idx]               # (B, T)
        cont_seqs = 1.0 - self.dones[idx]          # (B, T) continue = not done
        return (
            obs_seqs.astype(np.float32),
            act_seqs.astype(np.float32),
            rew_seqs.astype(np.float32),
            cont_seqs.astype(np.float32),
        )

    def _priority_probs(self, starts: np.ndarray, seq_len: int) -> Optional[np.ndarray]:
        """Probability of sampling each start, proportional to priority^alpha.

        A sequence's priority is the MAX transition priority inside its window
        (so a single high-error transition makes the whole window attractive).
        """
        if len(starts) == 0:
            return None
        idx = (starts[:, None] + np.arange(seq_len)[None, :]) % self.capacity
        seq_pri = self.priorities[idx].max(axis=1) + self.per_epsilon
        scaled = seq_pri ** self.per_alpha
        total = scaled.sum()
        if total <= 0 or not np.isfinite(total):
            return None  # fall back to uniform
        return scaled / total

    def update_priorities(
        self, start_indices: np.ndarray, seq_len: int, errors: np.ndarray
    ) -> None:
        """Update transition priorities after a training step.

        Args:
            start_indices: buffer indices returned by sample_sequences
            seq_len: sequence length used when sampling
            errors: per-sequence prediction error (one scalar per sequence)
        """
        if start_indices is None or len(start_indices) == 0:
            return
        errors = np.asarray(errors, dtype=np.float32).reshape(-1)
        errors = np.abs(errors) + self.per_epsilon
        with self._lock:
            idx = (start_indices[:, None] + np.arange(seq_len)[None, :]) % self.capacity
            # Broadcast each sequence's error across its window.
            self.priorities[idx] = errors[:, None]
            self.max_priority = max(self.max_priority, float(errors.max()))

    def _find_valid_starts(self, seq_len: int):
        """
        Find buffer positions where a sequence of seq_len can be sampled.
        Vectorized with numpy — replaces the old O(size * seq_len) Python loop.

        Returns (normal_starts, terminal_starts) — terminal_starts are
        sequences where done=True occurs at or near the end (last 3 steps),
        used to guarantee the continue predictor sees terminations.
        """
        n_starts = self.size - seq_len
        if n_starts <= 0:
            fallback = np.arange(max(1, self.size - seq_len), dtype=np.int64)
            return fallback, np.array([], dtype=np.int64)

        starts = np.arange(n_starts, dtype=np.int64)
        # Exclude windows that straddle the circular write seam once the
        # buffer has wrapped: a window containing the write position at an
        # offset > 0 splices the newest and oldest transitions as if they
        # were temporally contiguous.
        if self.size == self.capacity:
            offset = (self.position - starts) % self.capacity
            starts = starts[~((offset > 0) & (offset < seq_len))]
        idx = (starts[:, None] + np.arange(seq_len)[None, :]) % self.capacity
        done_window = self.dones[idx] > 0.5            # (n_starts, seq_len)
        any_done = done_window.any(axis=1)
        first_done = np.where(any_done, done_window.argmax(axis=1), -1)

        normal_mask = ~any_done
        # Terminal windows must END exactly on the done: a done earlier than
        # the last step means the window's tail belongs to the NEXT episode,
        # and observe_sequence would roll the GRU straight through the reset,
        # training a spurious done->reset "teleport" into the dynamics.
        terminal_mask = any_done & (first_done == seq_len - 1)

        normal = starts[normal_mask]
        terminal = starts[terminal_mask]

        if len(normal) == 0 and len(terminal) == 0:
            normal = np.arange(max(1, self.size - seq_len), dtype=np.int64)

        return normal, terminal

    def _find_reward_starts(self, seq_len: int, threshold: float = 1e-3):
        """Buffer positions whose length-`seq_len` window contains a nonzero
        reward (|reward| > threshold) — the goal-prioritized-replay candidates."""
        n_starts = self.size - seq_len
        if n_starts <= 0:
            return np.array([], dtype=np.int64)
        starts = np.arange(n_starts, dtype=np.int64)
        # Same write-seam exclusion as _find_valid_starts.
        if self.size == self.capacity:
            offset = (self.position - starts) % self.capacity
            starts = starts[~((offset > 0) & (offset < seq_len))]
        idx = (starts[:, None] + np.arange(seq_len)[None, :]) % self.capacity
        has_reward = (np.abs(self.rewards[idx]) > threshold).any(axis=1)
        # Reward windows must not cross an episode boundary either: on
        # sparse-reward tasks the goal reward is immediately followed by
        # done, so an unchecked reward window usually drags 1+ steps of the
        # NEXT episode in as fake contiguous dynamics — corrupting the model
        # exactly at the goal states this oversampling exists to emphasize.
        # Keep windows with no done, or with the done exactly on the last step.
        done_window = self.dones[idx] > 0.5
        any_done = done_window.any(axis=1)
        first_done = np.where(any_done, done_window.argmax(axis=1), seq_len - 1)
        boundary_ok = ~any_done | (first_done == seq_len - 1)
        return starts[has_reward & boundary_ok]

    def __len__(self) -> int:
        return self.size

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has enough data for a useful training batch."""
        return self.size >= 100


class MultiStreamReplayBuffer:
    """
    Replay buffer for PARALLEL environments.

    With N parallel envs we get N independent streams of experience. Storing
    them in one circular array would interleave timesteps from different envs
    and destroy the temporal contiguity the RSSM needs. So this buffer keeps
    N independent ReplayBuffers (one per env stream) and samples sequences
    that each live wholly inside a single stream.

    It exposes the SAME interface as ReplayBuffer (add / sample_sequences /
    update_priorities / is_ready / __len__) so the rest of the codebase does
    not care which buffer type it is holding. The only difference: add() takes
    an optional `stream` argument, and `start_indices` is returned as an
    (batch, 2) array of [stream_id, index] so priorities can be routed back to
    the right stream.
    """

    def __init__(
        self,
        num_streams: int,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_epsilon: float = 1e-2,
        obs_uint8: bool = False,
    ):
        self.num_streams = num_streams
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # Split capacity across streams so total memory matches the single buffer.
        per_stream_cap = max(1000, capacity // max(1, num_streams))
        self.streams: List[ReplayBuffer] = [
            ReplayBuffer(
                capacity=per_stream_cap,
                obs_dim=obs_dim,
                action_dim=action_dim,
                per_alpha=per_alpha,
                per_beta=per_beta,
                per_epsilon=per_epsilon,
                obs_uint8=obs_uint8,
            )
            for _ in range(num_streams)
        ]

    def add(self, observation, action, reward, done, stream: int = 0) -> None:
        self.streams[stream].add(observation, action, reward, done)

    def __len__(self) -> int:
        return sum(len(s) for s in self.streams)

    @property
    def is_ready(self) -> bool:
        return len(self) >= 100

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device = torch.device("cpu"),
        terminal_fraction: float = 0.25,
        prioritized: bool = False,
        beta: Optional[float] = None,
        reward_fraction: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        # Only sample from streams that have enough data for one sequence.
        ready = [i for i, s in enumerate(self.streams) if s.size >= seq_len + 1]
        if not ready:
            raise ValueError(
                f"No stream has at least {seq_len + 1} transitions yet."
            )

        # Distribute the batch across ready streams (as evenly as possible).
        n_ready = len(ready)
        base = batch_size // n_ready
        remainder = batch_size - base * n_ready
        counts = [base + (1 if k < remainder else 0) for k in range(n_ready)]

        obs_l, act_l, rew_l, cont_l, w_l = [], [], [], [], []
        stream_ids, local_idx = [], []
        for k, stream_i in enumerate(ready):
            count = counts[k]
            if count <= 0:
                continue
            sub = self.streams[stream_i].sample_sequences(
                batch_size=count,
                seq_len=seq_len,
                device=device,
                terminal_fraction=terminal_fraction,
                prioritized=prioritized,
                beta=beta,
                reward_fraction=reward_fraction,
            )
            obs_l.append(sub["observations"])
            act_l.append(sub["actions"])
            rew_l.append(sub["rewards"])
            cont_l.append(sub["continues"])
            w_l.append(sub["weights"])
            stream_ids.append(np.full(count, stream_i, dtype=np.int64))
            local_idx.append(sub["start_indices"])

        start_indices = np.stack(
            [np.concatenate(stream_ids), np.concatenate(local_idx)], axis=1
        )  # (batch, 2) = [stream_id, index]

        return {
            "observations": torch.cat(obs_l, dim=0),
            "actions": torch.cat(act_l, dim=0),
            "rewards": torch.cat(rew_l, dim=0),
            "continues": torch.cat(cont_l, dim=0),
            "weights": torch.cat(w_l, dim=0),
            "start_indices": start_indices,
        }

    def update_priorities(
        self, start_indices: np.ndarray, seq_len: int, errors: np.ndarray
    ) -> None:
        if start_indices is None or len(start_indices) == 0:
            return
        start_indices = np.asarray(start_indices)
        errors = np.asarray(errors, dtype=np.float32).reshape(-1)
        stream_col = start_indices[:, 0]
        idx_col = start_indices[:, 1]
        for stream_i in np.unique(stream_col):
            mask = stream_col == stream_i
            self.streams[int(stream_i)].update_priorities(
                idx_col[mask], seq_len, errors[mask]
            )


class BackgroundSampler:
    """
    Asynchronous prefetcher for replay sequences.

    Runs sampling in a worker thread so it overlaps with the main thread's
    gradient step. Use it inside a training loop like this:

        sampler.request(n_batches)          # enqueue n sampling jobs
        for _ in range(n_batches):
            batch = sampler.get()           # ready (or waits briefly)
            metrics = world_model.train_step(**batch)

    Thread-safety: sampling only reads the buffer's numpy arrays (guarded by
    the buffer's internal lock) and builds CPU/GPU tensors. It never touches
    model parameters, so it cannot race with autograd. For CUDA, host->device
    copies happen on the worker thread; this is fine in practice but if you
    ever see stream issues, set device='cpu' here and move in the main thread.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        prioritized: bool = False,
        terminal_fraction: float = 0.25,
        max_prefetch: int = 4,
        reward_fraction: float = 0.0,
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.prioritized = prioritized
        self.terminal_fraction = terminal_fraction
        self.reward_fraction = reward_fraction

        self._jobs: "queue.Queue[int]" = queue.Queue()
        self._results: "queue.Queue" = queue.Queue(maxsize=max_prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self._jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                batch = self.buffer.sample_sequences(
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                    device=self.device,
                    terminal_fraction=self.terminal_fraction,
                    prioritized=self.prioritized,
                    reward_fraction=self.reward_fraction,
                )
                self._results.put(batch)
            except Exception as e:  # surface sampling errors to the consumer
                self._results.put(e)

    def request(self, n: int = 1) -> None:
        """Enqueue n sampling jobs for the worker to prefetch."""
        for _ in range(n):
            self._jobs.put(1)

    def get(self, timeout: float = 30.0):
        """Block until the next prefetched batch is ready and return it."""
        item = self._results.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        """Stop the worker thread (call on shutdown)."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
