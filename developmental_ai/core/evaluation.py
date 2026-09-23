"""
Evaluation Framework — Benchmarking & Learning Curve Analysis
===============================================================
Systematic evaluation of trained DevelopmentalAI agents. Supports:

  1. Eval-only mode: run episodes without training or exploration noise
  2. Benchmark suite: compare configs/checkpoints on standardized metrics
  3. Learning curves: track metric progression over training
  4. Component diagnostics: world model accuracy, curiosity decay, KG growth

Usage:
    from developmental_ai.core.evaluation import Evaluator

    agent = DevelopmentalAI(config_path="configs/default.yaml")
    agent.load_checkpoint("logs/checkpoints")

    evaluator = Evaluator(agent)
    results = evaluator.evaluate(n_episodes=50)
    evaluator.print_report(results)
    evaluator.save_results(results, "eval_results.json")

    # Compare configs:
    comparison = Evaluator.compare_configs(
        ["configs/default.yaml", "configs/lunar_lander.yaml"],
        eval_episodes=20,
        train_timesteps=10000,
    )
"""

import json
import os
import time
import logging
from typing import Dict, Optional, Any, List
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EvalEpisodeRunner:
    """Runs episodes in eval mode: no training, deterministic actions."""

    def __init__(self, agent):
        self.agent = agent

    def run_episode(self, seed: Optional[int] = None) -> Dict[str, float]:
        """Run a single eval episode with no exploration noise or training.

        seed: reproducible episode layout (July 2026 audit H4 — eval episodes
        were previously unseeded, so eval was non-reproducible)."""
        env = self.agent.env
        obs, info = env.reset(seed=seed) if seed is not None else env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        intrinsic_total = 0.0
        prediction_errors = []

        rssm_state = self.agent.world_model.rssm.initial_state(
            1, self.agent.device
        )

        with torch.no_grad():
            init_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(
                self.agent.device
            )
            init_encoded = self.agent.world_model.embed(init_obs_t)
            zero_act = torch.zeros(
                1, self.agent.action_dim, device=self.agent.device
            )
            rssm_state, _ = self.agent.world_model.rssm.observe_step(
                rssm_state, zero_act, init_encoded
            )

        while not done:
            with torch.no_grad():
                if self.agent.dream_training_active:
                    latent = self.agent.world_model.rssm.get_latent(
                        rssm_state
                    )
                    action, _ = self.agent.dream_actor.select_action(latent)
                else:
                    # Greedy/argmax — the docstring's "deterministic actions"
                    # claim is finally true (July 2026 audit H4; the old code
                    # SAMPLED here, adding ±0.2 noise at 20 episodes).
                    # Pass the live knowledge feature so a symbolic agent
                    # (knowledge_dim>0) is evaluated on the SAME conditioning
                    # it trained with — the old call omitted it, silently
                    # evaluating on a zero knowledge vector (July 2026 review).
                    knowledge = self.agent._current_knowledge_feature()
                    action, _ = self.agent.policy.select_action(
                        obs, knowledge=knowledge, deterministic=True)

            next_obs, reward, terminated, truncated, step_info = env.step(
                action
            )
            done = terminated or truncated

            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(
                self.agent.device
            )

            if self.agent.is_discrete:
                act_t = torch.zeros(
                    1, self.agent.action_dim, device=self.agent.device
                )
                act_t[0, int(action)] = 1.0
            else:
                # Mirror the training loop's (1, action_dim) construction —
                # the old torch.FloatTensor([action]) double-wrapped the (A,)
                # array into (1, 1, A), a rank-3 tensor that crashes the WM /
                # ICM on continuous-action eval (July 2026 review).
                act_t = torch.FloatTensor(
                    np.asarray(action, dtype=np.float32)
                ).unsqueeze(0).to(self.agent.device)

            with torch.no_grad():
                intrinsic = self.agent.curiosity.compute_intrinsic_reward(
                    obs_t, act_t, next_obs_t
                ).item()

                encoded = self.agent.world_model.embed(next_obs_t)
                rssm_state, _ = self.agent.world_model.rssm.observe_step(
                    rssm_state, act_t, encoded
                )

                pred_error = self.agent.world_model.get_prediction_error(
                    rssm_state, act_t, next_obs_t
                ).item()
                prediction_errors.append(pred_error)

            intrinsic_total += intrinsic
            episode_reward += reward
            episode_length += 1
            obs = next_obs

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "avg_intrinsic_reward": intrinsic_total / max(1, episode_length),
            "avg_prediction_error": float(np.mean(prediction_errors))
            if prediction_errors else 0.0,
            "terminated_naturally": terminated,
        }


class Evaluator:
    """
    Comprehensive evaluation of a DevelopmentalAI agent.

    Collects:
      - Episode performance: reward, length, success rate
      - World model quality: prediction error magnitude
      - Curiosity state: intrinsic reward level (low = well-explored)
      - Knowledge graph: facts, entities, rules, density
      - Skill bank: total skills, mastery rates
      - Goal hierarchy: completed/failed goals
    """

    def __init__(self, agent):
        self.agent = agent
        self.runner = EvalEpisodeRunner(agent)

    def _success_threshold(self) -> float:
        """Per-env success threshold from config (reward > threshold =
        success). Default 0.0 preserves the old reward>0 behavior for envs
        that don't define one."""
        return float(
            self.agent.config.get("environment", {})
            .get("success_reward_threshold", 0.0))

    def evaluate(
        self,
        n_episodes: int = 50,
        verbose: bool = True,
        base_seed: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Run n_episodes in eval mode and collect comprehensive metrics.

        Episodes are seeded (base_seed + i) for reproducibility, and the env
        wrapper's observation-normalization statistics are frozen for the
        duration so evaluation neither perturbs training state nor drifts.
        """
        self.agent.world_model.eval()
        self.agent.curiosity.eval()
        freeze = getattr(self.agent.env, "freeze_obs_stats", None)
        if freeze is not None:
            freeze(True)

        try:
            episode_results = []
            for i in range(n_episodes):
                result = self.runner.run_episode(seed=base_seed + i)
                episode_results.append(result)
                if verbose and (i + 1) % 10 == 0:
                    avg = np.mean([r["episode_reward"] for r in episode_results])
                    print(f"  Eval {i+1}/{n_episodes}: avg_reward={avg:.2f}")
        finally:
            if freeze is not None:
                freeze(False)
            self.agent.world_model.train()
            self.agent.curiosity.train()

        return self._compile_results(episode_results)

    def _compile_results(
        self, episode_results: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        rewards = [r["episode_reward"] for r in episode_results]
        lengths = [r["episode_length"] for r in episode_results]
        pred_errors = [r["avg_prediction_error"] for r in episode_results]
        intrinsic = [r["avg_intrinsic_reward"] for r in episode_results]

        kg_stats = self.agent.knowledge_graph.get_stats()
        skill_stats = self.agent.skill_bank.get_stats()
        glue_stats = self.agent.glue.stats

        return {
            "timestamp": time.time(),
            "agent": {
                "env_name": self.agent.env_name,
                "total_timesteps": self.agent.total_timesteps,
                "total_episodes": self.agent.total_episodes,
                "dream_training_active": self.agent.dream_training_active,
                "pixel_obs": self.agent.pixel_obs,
            },
            "performance": {
                "n_episodes": len(episode_results),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "median_reward": float(np.median(rewards)),
                "mean_length": float(np.mean(lengths)),
                # Per-env threshold from config (July 2026 audit H4): the old
                # hardcoded reward>0 made e.g. Acrobot (threshold -150) always
                # report 0% success.
                "success_rate": float(np.mean([
                    1.0 if r > self._success_threshold() else 0.0
                    for r in rewards
                ])),
            },
            "world_model": {
                "mean_prediction_error": float(np.mean(pred_errors)),
                "std_prediction_error": float(np.std(pred_errors)),
            },
            "curiosity": {
                "mean_intrinsic_reward": float(np.mean(intrinsic)),
                "exploration_ratio": self.agent.curiosity.stats[
                    "exploration_ratio"
                ],
            },
            "knowledge_graph": kg_stats,
            "skill_bank": skill_stats,
            "goal_hierarchy": {
                "completed": glue_stats["completed_goals"],
                "failed": glue_stats["failed_goals"],
                "stack_depth": glue_stats["goal_stack_depth"],
            },
            "episodes": episode_results,
        }

    def learning_curve(
        self,
        total_timesteps: int,
        eval_interval: int = 5000,
        eval_episodes: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Train the agent and periodically evaluate to build a learning curve.

        Returns a list of eval snapshots, one per eval_interval.
        """
        curve = []
        steps_done = 0

        while steps_done < total_timesteps:
            chunk = min(eval_interval, total_timesteps - steps_done)
            self.agent.run(total_timesteps=self.agent.total_timesteps + chunk, verbose=0)
            steps_done += chunk

            snapshot = self.evaluate(n_episodes=eval_episodes, verbose=False)
            snapshot["training_timesteps"] = self.agent.total_timesteps
            curve.append(snapshot)

            logger.info(
                f"Curve @ {self.agent.total_timesteps} steps: "
                f"reward={snapshot['performance']['mean_reward']:.2f}"
            )

        return curve

    @staticmethod
    def compare_configs(
        config_paths: List[str],
        train_timesteps: int = 50000,
        eval_episodes: int = 20,
    ) -> Dict[str, Any]:
        """
        Train and evaluate multiple configs side-by-side.

        Returns a comparison dict with per-config results.
        """
        from developmental_ai.core.developmental_loop import DevelopmentalAI

        results = {}
        for path in config_paths:
            config_name = os.path.basename(path).replace(".yaml", "")
            logger.info(f"Benchmarking config: {config_name}")

            agent = DevelopmentalAI(config_path=path)
            agent.run(total_timesteps=train_timesteps, verbose=0)

            evaluator = Evaluator(agent)
            eval_result = evaluator.evaluate(
                n_episodes=eval_episodes, verbose=False
            )
            results[config_name] = eval_result

            agent.close()

        return {
            "configs_compared": list(results.keys()),
            "train_timesteps": train_timesteps,
            "eval_episodes": eval_episodes,
            "results": results,
            "ranking": sorted(
                results.keys(),
                key=lambda k: results[k]["performance"]["mean_reward"],
                reverse=True,
            ),
        }

    @staticmethod
    def print_report(results: Dict[str, Any]) -> None:
        """Print a human-readable evaluation report."""
        perf = results["performance"]
        ag = results["agent"]
        wm = results["world_model"]
        cur = results["curiosity"]
        kg = results["knowledge_graph"]
        sb = results["skill_bank"]
        gh = results["goal_hierarchy"]

        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT — {ag['env_name']}")
        print(f"{'='*60}")
        print(f"  Training: {ag['total_timesteps']} steps, "
              f"{ag['total_episodes']} episodes")
        print(f"  Dream policy: {'active' if ag['dream_training_active'] else 'PPO'}")
        print(f"  Pixel obs: {ag['pixel_obs']}")
        print()
        print(f"  Performance ({perf['n_episodes']} eval episodes):")
        print(f"    Mean reward:  {perf['mean_reward']:.2f} "
              f"(+/- {perf['std_reward']:.2f})")
        print(f"    Min/Max:      {perf['min_reward']:.2f} / "
              f"{perf['max_reward']:.2f}")
        print(f"    Median:       {perf['median_reward']:.2f}")
        print(f"    Mean length:  {perf['mean_length']:.0f}")
        print(f"    Success rate: {perf['success_rate']:.1%}")
        print()
        print(f"  World Model:")
        print(f"    Pred error:   {wm['mean_prediction_error']:.4f} "
              f"(+/- {wm['std_prediction_error']:.4f})")
        print()
        print(f"  Curiosity:")
        print(f"    Intrinsic:    {cur['mean_intrinsic_reward']:.4f}")
        print(f"    Explore ratio:{cur['exploration_ratio']:.3f}")
        print()
        print(f"  Knowledge Graph:")
        print(f"    Facts:    {kg['num_facts']}")
        print(f"    Entities: {kg['num_entities']}")
        print(f"    Rules:    {kg['num_action_rules']}")
        print()
        print(f"  Skill Bank:")
        print(f"    Total:    {sb['total_skills']}")
        print(f"    Mastered: {sb['mastered_skills']}")
        if sb.get("skill_names"):
            print(f"    Names:    {', '.join(sb['skill_names'][:5])}")
        print()
        print(f"  Goal Hierarchy:")
        print(f"    Completed: {gh['completed']}")
        print(f"    Failed:    {gh['failed']}")
        print(f"    Depth:     {gh['stack_depth']}")
        print(f"{'='*60}")

    @staticmethod
    def save_results(results: Dict[str, Any], filepath: str) -> None:
        """Save evaluation results to JSON."""
        def _serialize(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=_serialize)
        logger.info(f"Results saved to {filepath}")

    @staticmethod
    def load_results(filepath: str) -> Dict[str, Any]:
        """Load evaluation results from JSON."""
        with open(filepath, "r") as f:
            return json.load(f)
