"""
Distributed Training — Ray-based Parallel Execution
======================================================
Wraps the DevelopmentalAI loop for distributed training using Ray.

Three modes:
  1. Single-node parallel: multiple environments on one machine (Ray local)
  2. Multi-worker: distributed across machines (Ray cluster)
  3. Population-based: multiple agents sharing a skill bank (experimental)

This module handles the orchestration — the DevelopmentalAI loop stays
unchanged, each Ray worker just runs its own instance.

Requires: pip install "ray[rllib]>=2.10.0"

Usage:
    from developmental_ai.core.distributed import DistributedTrainer

    trainer = DistributedTrainer(
        config_path="configs/default.yaml",
        num_workers=4,
    )
    results = trainer.train(total_timesteps=1_000_000)
"""

import os
import logging
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def _train_worker(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ray remote function: runs a DevelopmentalAI instance.

    This is the unit of work — each worker trains independently on its own
    environment instance, accumulating to the shared skill bank if configured.
    """
    from developmental_ai.core.developmental_loop import DevelopmentalAI

    worker_id = config.pop("__worker_id", 0)
    total_timesteps = config.pop("__total_timesteps", 100_000)
    shared_skill_dir = config.pop("__shared_skill_dir", None)

    if shared_skill_dir:
        config.setdefault("skill_bank", {})["storage_dir"] = shared_skill_dir

    log_dir = config.get("loop", {}).get("log_dir", "./logs")
    config.setdefault("loop", {})["log_dir"] = os.path.join(
        log_dir, f"worker_{worker_id}"
    )

    agent = DevelopmentalAI(config=config)
    summary = agent.run(total_timesteps=total_timesteps)
    summary["worker_id"] = worker_id
    return summary


class DistributedTrainer:
    """
    Orchestrates parallel DevelopmentalAI training across Ray workers.

    Each worker runs a fully independent DevelopmentalAI loop with its own
    environment, world model, and curiosity engine. Workers can optionally
    share a skill bank directory so skills discovered by one agent are
    available to all others.

    Usage:
        trainer = DistributedTrainer(
            config_path="configs/default.yaml",
            num_workers=4,
        )
        results = trainer.train(total_timesteps=500_000)
        trainer.shutdown()
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        num_workers: int = 4,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: float = 0,
        shared_skill_bank: bool = True,
        ray_address: Optional[str] = None,
    ):
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray not installed. pip install 'ray[rllib]>=2.10.0'"
            )

        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker
        self.shared_skill_bank = shared_skill_bank

        if config is not None:
            self.base_config = config
        elif config_path is not None:
            import yaml
            with open(config_path, "r") as f:
                self.base_config = yaml.safe_load(f)
        else:
            from developmental_ai.core.developmental_loop import DevelopmentalAI
            self.base_config = DevelopmentalAI._default_config()

        if shared_skill_bank:
            self._shared_skill_dir = os.path.abspath(
                self.base_config.get("skill_bank", {}).get(
                    "storage_dir", "./skill_bank_data"
                )
            )
            os.makedirs(self._shared_skill_dir, exist_ok=True)
        else:
            self._shared_skill_dir = None

        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)
            logger.info(
                f"Ray initialized: {ray.cluster_resources()}"
            )

        self._remote_fn = ray.remote(
            num_cpus=num_cpus_per_worker,
            num_gpus=num_gpus_per_worker,
        )(_train_worker)

    def train(
        self,
        total_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Launch parallel training across all workers.

        Each worker trains for total_timesteps independently. When all
        workers finish, their results are aggregated.

        Returns:
            Aggregated training summary with per-worker details
        """
        if total_timesteps is None:
            total_timesteps = self.base_config.get("loop", {}).get(
                "total_timesteps", 100_000
            )

        per_worker = total_timesteps // self.num_workers

        futures = []
        for i in range(self.num_workers):
            worker_config = dict(self.base_config)
            worker_config["__worker_id"] = i
            worker_config["__total_timesteps"] = per_worker
            if self._shared_skill_dir:
                worker_config["__shared_skill_dir"] = self._shared_skill_dir
            futures.append(self._remote_fn.remote(worker_config))

        logger.info(
            f"Launched {self.num_workers} workers, "
            f"{per_worker} timesteps each "
            f"({total_timesteps} total)"
        )

        results = ray.get(futures)
        return self._aggregate_results(results)

    def train_population(
        self,
        total_timesteps: Optional[int] = None,
        sync_interval: int = 50_000,
    ) -> Dict[str, Any]:
        """
        Population-based training: workers periodically sync skills.

        Unlike plain parallel training, workers run in rounds. After each
        round, the shared skill bank is available to all workers for the
        next round. This enables skill transfer between agents.

        Args:
            total_timesteps: Total timesteps across all rounds
            sync_interval: Timesteps per round before syncing
        """
        if total_timesteps is None:
            total_timesteps = self.base_config.get("loop", {}).get(
                "total_timesteps", 100_000
            )

        remaining = total_timesteps
        all_results: List[Dict[str, Any]] = []
        round_num = 0

        while remaining > 0:
            round_steps = min(sync_interval, remaining)
            per_worker = round_steps // self.num_workers

            futures = []
            for i in range(self.num_workers):
                worker_config = dict(self.base_config)
                worker_config["__worker_id"] = i
                worker_config["__total_timesteps"] = per_worker
                if self._shared_skill_dir:
                    worker_config["__shared_skill_dir"] = self._shared_skill_dir
                futures.append(self._remote_fn.remote(worker_config))

            round_results = ray.get(futures)
            all_results.extend(round_results)

            round_num += 1
            remaining -= round_steps
            total_skills = sum(
                r.get("skill_bank", {}).get("total_skills", 0)
                for r in round_results
            )
            logger.info(
                f"Round {round_num} complete: {round_steps} steps, "
                f"{total_skills} skills in shared bank, "
                f"{remaining} steps remaining"
            )

        return self._aggregate_results(all_results)

    def _aggregate_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine per-worker results into a single summary."""
        import numpy as np

        total_timesteps = sum(r.get("total_timesteps", 0) for r in results)
        total_episodes = sum(r.get("total_episodes", 0) for r in results)
        total_time = max(
            r.get("elapsed_time_seconds", 0) for r in results
        )
        avg_rewards = [
            r.get("avg_episode_reward", 0) for r in results
        ]

        skill_counts = [
            r.get("skill_bank", {}).get("total_skills", 0) for r in results
        ]

        return {
            "num_workers": len(results),
            "total_timesteps": total_timesteps,
            "total_episodes": total_episodes,
            "wall_clock_seconds": total_time,
            "avg_episode_reward": float(np.mean(avg_rewards)),
            "best_worker_reward": float(np.max(avg_rewards)),
            "total_skills_discovered": sum(skill_counts),
            "per_worker": results,
        }

    def shutdown(self) -> None:
        """Shut down the Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shut down")
