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

import json
import logging
import os
import shutil
import threading
import queue
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Files a saved buffer is made of. `manifest.json` is written LAST and is the
# commit record: a directory without a readable manifest is treated as absent,
# which makes the whole save atomic in effect without needing an atomic
# directory rename. (world_model.pt is written non-atomically and a truncated
# one has already cost this project a silent partial restore — the same
# mistake is not repeated here.)
_BUF_ARRAYS = ("observations", "actions", "rewards", "dones", "priorities")
_BUF_MANIFEST = "manifest.json"


# ---------------------------------------------------------------------------
# Block storage + a memory budget (2026-09-01, cluster migration)
# ---------------------------------------------------------------------------
# WHY THIS EXISTS. `np.zeros((capacity, obs_dim))` at capacity 1e6 and obs_dim
# 49152 reserves ~24.5 GB PER STREAM. calloc hands back lazily-mapped zero
# pages, so nothing fails at startup — RSS just climbs as the buffer fills and
# the process is killed part-way. At ~7 transitions/s that is ~40 hours in,
# i.e. precisely the multi-day runs this project exists to do.
# `buffer_persist_max_gb` caps only what is written to DISK and never touched
# this.
#
# THE HONEST BOUND. A buffer cannot grow without limit. What this gives is:
# grow in blocks, on the training cadence rather than the acting path, up to a
# ceiling derived from the CONTAINER's real memory limit — and when that
# ceiling is reached, fall back to the circular overwrite that shipped before,
# loudly. Degradation instead of death.

class BlockArray:
    """A grow-only array stored as a list of fixed-size blocks.

    Growth appends a block: no reallocation, no copy, and therefore no
    transient 2x spike — which is the whole reason for not simply resizing a
    contiguous array (a 24.5 GB column would need 49 GB to grow by one row).

    Supports the indexing forms this buffer actually uses: integer, 1-D index
    arrays, 2-D (batch, seq) gathers, `[:n]` slices, and boolean-free scatter
    assignment. Deliberately NOT a general ndarray substitute — an unsupported
    form raises rather than silently returning something plausible.
    """

    __slots__ = ("block", "row_shape", "dtype", "fill", "_blocks", "_n")

    def __init__(self, block: int, row_shape: Tuple[int, ...],
                 dtype, fill: float = 0.0, initial_blocks: int = 1):
        self.block = int(block)
        self.row_shape = tuple(row_shape)
        self.dtype = dtype
        self.fill = fill
        self._blocks: List[np.ndarray] = []
        self._n = 0
        for _ in range(max(1, int(initial_blocks))):
            self.add_block()

    # ---- capacity -----------------------------------------------------
    def add_block(self) -> None:
        b = np.empty((self.block,) + self.row_shape, dtype=self.dtype)
        b.fill(self.fill)
        self._blocks.append(b)
        self._n += self.block

    def __len__(self) -> int:
        return self._n

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self._blocks)

    # ---- access -------------------------------------------------------
    def _split(self, idx):
        idx = np.asarray(idx)
        return np.divmod(idx, self.block)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # only the `[:n]` / `[:]` forms the buffer uses
            start, stop, step = key.indices(self._n)
            if step != 1:
                raise IndexError("BlockArray: strided slices unsupported")
            return self[np.arange(start, stop, dtype=np.int64)]
        if np.isscalar(key):
            b, o = divmod(int(key), self.block)
            return self._blocks[b][o]
        b, o = self._split(key)
        out = np.empty(b.shape + self.row_shape, dtype=self.dtype)
        # one gather per touched block; `b` is small (few blocks) so this
        # stays a handful of vectorized copies, not a Python loop over rows
        for bi in np.unique(b):
            m = (b == bi)
            out[m] = self._blocks[int(bi)][o[m]]
        return out

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._n)
            if step != 1:
                raise IndexError("BlockArray: strided slices unsupported")
            key = np.arange(start, stop, dtype=np.int64)
        if np.isscalar(key):
            b, o = divmod(int(key), self.block)
            self._blocks[b][o] = value
            return
        b, o = self._split(key)
        value = np.asarray(value, dtype=self.dtype)
        bcast = value.shape != b.shape + self.row_shape
        for bi in np.unique(b):
            m = (b == bi)
            self._blocks[int(bi)][o[m]] = value if bcast else value[m]

    def any(self) -> bool:
        return any(bool(b.any()) for b in self._blocks)


class _MemoryBudget:
    """How much RAM the replay buffers may collectively occupy.

    ONE budget for the whole process. Every stream asking the machine how big
    it is and taking a fraction is an N-times overshoot, and the fleet size is
    exactly the thing that varies.

    CGROUP FIRST, and that ordering is load-bearing: under a container
    (`/sys/fs/cgroup/...`) the host's RAM is not the limit, and reading the
    host figure is the container-vs-host measurement trap this project has
    already paid for once. On a cluster the container is the normal case.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reserved = 0
        self.limit = 0
        self.source = "unset"

    def configure(self, max_ram_frac: float, hard_max_gb: float) -> None:
        with self._lock:
            if self.limit:
                return          # first configuration wins; streams share it
            total, src = self._detect_total()
            lim = int(total * float(max_ram_frac)) if total else 0
            if hard_max_gb and hard_max_gb > 0:
                cap = int(float(hard_max_gb) * 1e9)
                lim = min(lim, cap) if lim else cap
                src += "+hard_max_gb"
            self.limit, self.source = lim, src
            logger.info(
                "replay memory budget: %.1f GB (%.0f%% of %.1f GB, source=%s)",
                lim / 1e9, 100.0 * float(max_ram_frac),
                (total or 0) / 1e9, src)

    @staticmethod
    def _detect_total() -> Tuple[int, str]:
        # cgroup v2
        try:
            with open("/sys/fs/cgroup/memory.max") as fh:
                v = fh.read().strip()
            if v.isdigit():
                return int(v), "cgroup2"
        except Exception:
            pass
        # cgroup v1
        try:
            with open(
                "/sys/fs/cgroup/memory/memory.limit_in_bytes") as fh:
                v = int(fh.read().strip())
            # v1 reports a sentinel ~2^63 when unlimited
            if 0 < v < (1 << 62):
                return v, "cgroup1"
        except Exception:
            pass
        try:
            import psutil
            return int(psutil.virtual_memory().total), "psutil"
        except Exception:
            pass
        try:
            return (os.sysconf("SC_PHYS_PAGES")
                    * os.sysconf("SC_PAGE_SIZE")), "sysconf"
        except Exception:
            return 0, "unknown"

    def request(self, nbytes: int) -> bool:
        """Reserve `nbytes` if the budget allows. False = at the ceiling."""
        with self._lock:
            if self.limit <= 0:
                return False
            if self.reserved + nbytes > self.limit:
                return False
            self.reserved += nbytes
            return True


MEMORY_BUDGET = _MemoryBudget()


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
        growth: Optional[Dict] = None,
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

        # ---- STORAGE: BLOCKS WHEN GROWABLE, ONE ARRAY WHEN NOT -----------
        # growth disabled -> np.zeros exactly as before, byte-for-byte.
        # growth enabled  -> BlockArray, starting at ONE block rather than the
        # full configured capacity, so a run does not reserve 24.5 GB of
        # lazily-mapped pages up front and then walk RSS into the reaper. The
        # configured `capacity` becomes the STARTING TARGET, not a hard size.
        _g = dict(growth or {})
        self._growable = bool(_g.get("enabled", False))
        self._block = max(1000, int(_g.get("block_transitions", 25000)))
        self._grow_at = float(_g.get("grow_at_frac", 0.85))
        self._grow_pending = False
        self._growth_capped = False
        self._grow_events = 0
        _obs_dtype = np.uint8 if self._obs_uint8 else np.float32
        if self._growable:
            MEMORY_BUDGET.configure(
                float(_g.get("max_ram_frac", 0.6)),
                float(_g.get("hard_max_gb", 0.0)))
            self.capacity = self._block
            self.observations = BlockArray(
                self._block, (obs_dim,), _obs_dtype)
            self.actions = BlockArray(self._block, (action_dim,), np.float32)
            self.rewards = BlockArray(self._block, (), np.float32)
            self.dones = BlockArray(self._block, (), np.float32)
        else:
            self.observations = np.zeros((capacity, obs_dim), dtype=_obs_dtype)
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
            self.rewards = np.zeros(capacity, dtype=np.float32)
            self.dones = np.zeros(capacity, dtype=np.float32)
        # ---- RESTART IS NOT A DEATH (2026-09-01) -------------------------
        # The loop stores `done = terminal or crash_restart`, because for the
        # ADVANTAGE trace both sever the trajectory identically. For the
        # WORLD MODEL they are opposite things: a terminal is an event the
        # continue head should learn to predict; a crash restart is a client
        # rebuild that splices two DIFFERENT WORLDS across one index, and
        # every earlier note in this project says a new mission means a new
        # world. Teaching that splice as dynamics is teaching a fiction, and
        # oversampling it (terminal_fraction) taught it hardest of all.
        # Flagged separately so sequences can EXCLUDE it while `dones` keeps
        # meaning what the GAE path needs it to mean.
        self.max_priority = 1.0
        if self._growable:
            self.restarts = BlockArray(self._block, (), bool, fill=False)
            # PER: a fresh block must start at MAX priority, not 0, or every
            # newly grown slot is unsamplable until something writes it.
            self.priorities = BlockArray(
                self._block, (), np.float32, fill=self.max_priority)
        else:
            self.restarts = np.zeros(capacity, dtype=bool)
            # ---- Prioritized Experience Replay state ----
            # Per-transition priority. New transitions get max_priority so they
            # are guaranteed to be sampled at least once before being
            # deprioritized.
            self.priorities = np.zeros(capacity, dtype=np.float32)
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
        restart: bool = False,
    ) -> None:
        """
        Add a single transition to the buffer.

        Args:
            observation: Current observation
            action: Action taken (will be converted to array if scalar)
            reward: Reward received
            done: Whether episode ended
            restart: this `done` is a CLIENT REBUILD, not a real terminal —
                the frames either side belong to different worlds. Defaults
                False so every existing caller (and every buffer written
                before 2026-09-01) behaves exactly as it did.
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
            self.restarts[self.position] = bool(restart)
            # New experience enters at max priority so PER will sample it soon.
            self.priorities[self.position] = self.max_priority

            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            # GROWTH IS FLAGGED HERE, NEVER PERFORMED HERE. This runs on the
            # acting thread once per env step; allocating a 1.2 GB block in it
            # would stall the agent mid-episode. `maybe_grow()` does the work
            # on the world-model cadence instead. Flagged at 85% so the block
            # is in place BEFORE the write position reaches the end and no
            # transition is ever overwritten while the buffer could have grown.
            if (self._growable and not self._growth_capped
                    and self.size >= self._grow_at * self.capacity):
                self._grow_pending = True

            # Track episode boundaries
            if done:
                self.episode_starts.append(self.position)
                self._current_episode_start = self.position

    def maybe_grow(self) -> bool:
        """Append one block to every column if the buffer is nearly full.

        CALLED FROM THE TRAINING CADENCE, not from add(). That is the whole
        point: allocation happens during the world-model block — the "async
        learning period between intervals" — where a pause costs nothing,
        rather than on the acting thread where it would stall the agent
        mid-episode.

        Returns True if the buffer grew.

        AT THE CEILING it does not raise and does not keep trying: it says so
        once, sets `_growth_capped`, and from then on add() wraps circularly
        exactly as the fixed-capacity buffer always did. The newest N
        transitions are kept and the run continues. A memory guard that kills
        the process is not a guard.
        """
        if not (self._growable and self._grow_pending
                and not self._growth_capped):
            return False
        _need = (self._block * self.obs_dim
                 * (1 if self._obs_uint8 else 4)          # observations
                 + self._block * self.action_dim * 4      # actions
                 + self._block * (4 + 4 + 4 + 1))         # rew/done/pri/restart
        if not MEMORY_BUDGET.request(_need):
            self._growth_capped = True
            self._grow_pending = False
            logger.warning(
                "replay buffer AT ITS MEMORY CEILING at %d transitions "
                "(%.1f GB budget, source=%s). Growth stops here and the "
                "buffer reverts to circular overwrite — the newest %d "
                "transitions are kept and the run continues. Raise "
                "world_model.buffer_growth.max_ram_frac / hard_max_gb, or "
                "give the container more memory, to go further.",
                self.capacity, MEMORY_BUDGET.limit / 1e9,
                MEMORY_BUDGET.source, self.capacity)
            return False
        with self._lock:
            # PUBLISH ORDER IS LOAD-BEARING: every column gets its block
            # BEFORE `capacity` moves. A reader racing this sees the old
            # capacity over a consistent buffer, never an index past the end
            # of a column that has not grown yet.
            for _c in (self.observations, self.actions, self.rewards,
                       self.dones, self.priorities, self.restarts):
                _c.add_block()
            self.capacity += self._block
            self._grow_pending = False
            self._grow_events += 1
        logger.info("replay buffer grew to %d transitions (%.1f GB reserved "
                    "of %.1f GB)", self.capacity,
                    MEMORY_BUDGET.reserved / 1e9, MEMORY_BUDGET.limit / 1e9)
        return True

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
            # ---- QUOTA SHRINKS, IT DOES NOT DUPLICATE (2026-09-01) -------
            # `replace=True` against a small pool is how a fixed FRACTION
            # becomes a fixed set. In lifelong mode `done` fires only on a
            # real death, so the pool is a few dozen windows for a run of
            # millions of steps — and at terminal_fraction 0.25, train_iters
            # 384 and batch_size 8, that is 768 draws per training block,
            # every 250 env steps, from the same handful of death frames.
            # The world model was spending a quarter of its capacity
            # memorizing them.
            # Cap the quota at what the pool can supply WITHOUT repeats; the
            # freed slots go to `n_normal` below (ordinary experience),
            # which is the honest thing to train on when there are no
            # terminals to learn from. When the pool IS large the behaviour
            # is unchanged apart from being repeat-free.
            n_terminal = min(n_terminal, len(terminal_starts))
            if len(terminal_starts) > 0 and n_terminal > 0:
                t_probs = self._priority_probs(terminal_starts, seq_len) if prioritized else None
                t_pick = np.random.choice(
                    len(terminal_starts), size=n_terminal, replace=False,
                    p=t_probs
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
                    # Repeats ARE allowed here, unlike the terminal pool
                    # above, and deliberately: the reward pool is the SIGNAL
                    # this task is starved of (399 breaks, 1 log, ever), so
                    # seeing the same log break several times per batch is
                    # the point rather than the pathology. It is also
                    # self-limiting — the pool grows every time the agent
                    # succeeds, which is exactly when it needs less help.
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

        # ---- A CLIENT REBUILD IS NOT A TERMINAL (2026-09-01) -------------
        # Windows containing a restart row are dropped from BOTH pools. The
        # same argument the write-seam exclusion above already makes: those
        # frames are not temporally contiguous, so nothing true can be
        # learned from the transition across them. Previously they were not
        # merely included, they were the ones OVERSAMPLED — restarts land in
        # `dones`, `dones` feeds terminal_starts, and terminal_fraction 0.25
        # drew from that pool with replacement 768 times per training block.
        # The rarest, most corrupt windows in the buffer had the highest
        # sampling weight.
        if self.restarts.any():
            has_restart = self.restarts[idx].any(axis=1)
            normal_mask = normal_mask & ~has_restart
            terminal_mask = terminal_mask & ~has_restart

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

    # ------------------------------------------------------------------
    # PERSISTENCE (2026-08-23)
    #
    # This buffer had NO save/load path at all, so a crash-relaunch — which
    # the supervisor performs automatically, and which live Python exceptions
    # have caused — restarted world-model training from an EMPTY buffer. On a
    # multi-day run the effective memory was never the configured capacity;
    # it was "however long since the last crash".
    #
    # Stored in CHRONOLOGICAL order (oldest first) rather than as the raw
    # circular layout, so a restore still works when `capacity` changes
    # between runs: the newest min(n, capacity) entries are kept and the rest
    # are dropped exactly as the circular buffer would have dropped them.
    # ------------------------------------------------------------------
    def _chronological(self) -> np.ndarray:
        """Indices oldest-first. Before wrap that is [0, size); after wrap the
        write position is the oldest slot."""
        if self.size < self.capacity:
            return np.arange(self.size, dtype=np.int64)
        return (np.arange(self.capacity, dtype=np.int64)
                + int(self.position)) % self.capacity

    def save(self, path: str, max_transitions: Optional[int] = None) -> int:
        """Write the most recent experience to directory `path`.

        The manifest is written LAST and is the commit record — a reader that
        finds no manifest treats the directory as absent rather than loading
        a half-written buffer.
        """
        with self._lock:
            order = self._chronological()
            if max_transitions is not None and len(order) > int(max_transitions):
                order = order[-int(max_transitions):]      # keep the NEWEST
            n = int(len(order))
            cols = {
                "observations": self.observations[order],
                "actions": self.actions[order],
                "rewards": self.rewards[order],
                "dones": self.dones[order],
                "priorities": self.priorities[order],
                "restarts": self.restarts[order],
            }
            meta = {
                "n": n,
                "obs_dim": int(self.obs_dim),
                "action_dim": int(self.action_dim),
                "obs_uint8": bool(self._obs_uint8),
                "max_priority": float(self.max_priority),
            }
        tmp = path + ".tmp"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        for k, v in cols.items():
            np.save(os.path.join(tmp, k + ".npy"), v)
        with open(os.path.join(tmp, _BUF_MANIFEST), "w") as fh:
            json.dump(meta, fh)                       # COMMIT
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
        return n

    def load(self, path: str) -> int:
        """Restore from `path`. Returns the number of transitions loaded.

        PRE-FLIGHT BEFORE ANY MUTATION, the same discipline the world-model
        restore now uses: a payload whose shapes cannot fill this buffer is
        refused while the buffer is still pristine, rather than half-copied
        into a hybrid that reports success.
        """
        mpath = os.path.join(path, _BUF_MANIFEST)
        if not os.path.isfile(mpath):
            raise FileNotFoundError(
                f"{path} has no {_BUF_MANIFEST} — the save did not commit "
                f"(partial or interrupted write); treating as absent")
        with open(mpath) as fh:
            meta = json.load(fh)
        if int(meta.get("obs_dim", -1)) != int(self.obs_dim):
            raise ValueError(
                f"buffer obs_dim {meta.get('obs_dim')} != {self.obs_dim} — "
                f"the observation changed shape since this was written; "
                f"refusing before touching the buffer")
        if int(meta.get("action_dim", -1)) != int(self.action_dim):
            raise ValueError(
                f"buffer action_dim {meta.get('action_dim')} != "
                f"{self.action_dim} (action_dim moved 10 -> 12 once already) "
                f"— refusing before touching the buffer")
        if bool(meta.get("obs_uint8", False)) != bool(self._obs_uint8):
            raise ValueError(
                f"buffer obs dtype differs (saved obs_uint8="
                f"{meta.get('obs_uint8')}, live={self._obs_uint8}) — the "
                f"stored values would be off by 255x; refusing")
        for k in _BUF_ARRAYS:
            if not os.path.isfile(os.path.join(path, k + ".npy")):
                raise FileNotFoundError(f"{path} is missing {k}.npy")
        cols = {k: np.load(os.path.join(path, k + ".npy")) for k in _BUF_ARRAYS}
        # `restarts` is NOT in _BUF_ARRAYS: it arrived 2026-09-01 and the live
        # pod's persisted buffer predates it. Absent means "nothing is known
        # to be a rebuild splice", which is precisely the pre-change
        # behaviour — so an old buffer resumes identically rather than
        # refusing to load. Making a new column mandatory would have bricked
        # the only copy of the agent's experience.
        _rp = os.path.join(path, "restarts.npy")
        cols["restarts"] = (np.load(_rp) if os.path.isfile(_rp)
                            else np.zeros(int(cols["observations"].shape[0]),
                                          dtype=bool))
        n = int(cols["observations"].shape[0])
        # GROW TO FIT rather than discard (2026-09-01). Truncating to "newest"
        # is right for a shrunken fixed buffer, but with growth on it would
        # throw away experience the machine has room for — and the restored
        # buffer is the one thing on a rebuilt node that cannot be
        # re-collected.
        while (self._growable and not self._growth_capped
               and n > self.capacity):
            self._grow_pending = True
            if not self.maybe_grow():
                break
        if n > self.capacity:                 # capacity shrank: keep newest
            cols = {k: v[-self.capacity:] for k, v in cols.items()}
            n = self.capacity
        with self._lock:
            self.observations[:n] = cols["observations"]
            self.actions[:n] = cols["actions"]
            self.rewards[:n] = cols["rewards"]
            self.dones[:n] = cols["dones"]
            self.priorities[:n] = cols["priorities"]
            self.restarts[:n] = cols["restarts"].astype(bool)
            self.size = n
            self.position = n % self.capacity
            self.max_priority = float(meta.get("max_priority", 1.0)) or 1.0
            # Episode boundaries are deliberately NOT restored: those indices
            # describe the old layout, and `_find_valid_starts` reads `dones`
            # (which IS restored) to avoid sampling across a boundary.
            # Carrying stale indices would be worse than carrying none.
            self.episode_starts = []
            self._current_episode_start = self.position
        return n

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
        growth: Optional[Dict] = None,
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
                # ONE shared MEMORY_BUDGET across every stream — see
                # _MemoryBudget. Each stream grows independently (a busy
                # stream outgrows a quiet one, which is correct), but they
                # draw on the same pool, so N streams cannot each take a
                # fraction of the machine and overshoot by N.
                growth=growth,
            )
            for _ in range(num_streams)
        ]

    def maybe_grow(self) -> bool:
        """Give every stream a chance to grow. Called on the WM cadence."""
        return any(s.maybe_grow() for s in self.streams)

    @property
    def capacity(self) -> int:
        return sum(s.capacity for s in self.streams)

    def add(self, observation, action, reward, done, stream: int = 0,
            restart: bool = False) -> None:
        self.streams[stream].add(observation, action, reward, done,
                                 restart=restart)

    def save(self, path: str, max_transitions: Optional[int] = None) -> int:
        """Persist every stream under `path/stream_<i>`. Returns total written.

        `max_transitions` is PER STREAM, so the cap means the same thing
        however many bodies are feeding the brain.
        """
        os.makedirs(path, exist_ok=True)
        total = 0
        for i, s in enumerate(self.streams):
            total += s.save(os.path.join(path, f"stream_{i}"),
                            max_transitions=max_transitions)
        return total

    def load(self, path: str) -> int:
        """Restore every stream. Missing streams are skipped, not fatal: the
        env count can legitimately change between runs (4 clients broke the
        Paper server once and the fleet was scaled to 2), and a brain that
        refused to start because it now has fewer bodies would be worse than
        one that resumes with the experience it can still account for.
        """
        total = 0
        for i, s in enumerate(self.streams):
            _p = os.path.join(path, f"stream_{i}")
            if not os.path.isdir(_p):
                continue
            total += s.load(_p)
        return total

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
