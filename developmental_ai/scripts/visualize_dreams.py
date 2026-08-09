#!/usr/bin/env python3
"""
CLI tool to visualize what a trained DevelopmentalAI agent dreams.

Usage (from project root):
    # Full report from a checkpoint
    ./venv/bin/python3 -m developmental_ai.scripts.visualize_dreams \
        --config configs/default.yaml \
        --checkpoint logs/checkpoints \
        --output dream_logs

    # Quick single dream (pops up interactive plot)
    ./venv/bin/python3 -m developmental_ai.scripts.visualize_dreams \
        --config configs/default.yaml \
        --checkpoint logs/checkpoints \
        --quick --show

    # Train first, then visualize
    ./venv/bin/python3 -m developmental_ai.scripts.visualize_dreams \
        --config configs/default.yaml \
        --train 5000 \
        --output dream_logs

    # Or use the convenience script:
    ./dream configs/default.yaml --train 5000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(
        description="See what your AI is dreaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to agent config YAML",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint directory to load",
    )
    parser.add_argument(
        "--train", type=int, default=0,
        help="Train for this many timesteps before visualizing (use when no checkpoint)",
    )
    parser.add_argument(
        "--output", default="dream_logs",
        help="Directory to save visualization outputs",
    )
    parser.add_argument(
        "--horizon", type=int, default=30,
        help="How many timesteps to dream forward",
    )
    parser.add_argument(
        "--n-cloud", type=int, default=8,
        help="Number of parallel dreams for the cloud plot",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Show a single combined dream plot instead of full report",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively (requires display)",
    )
    args = parser.parse_args()

    from developmental_ai.core import DevelopmentalAI
    from developmental_ai.visualization import DreamVisualizer

    print(f"Loading agent from {args.config}...")
    agent = DevelopmentalAI(config_path=args.config)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        agent.load_checkpoint(args.checkpoint)
    elif args.train > 0:
        print(f"Training for {args.train} timesteps...")
        agent.run(total_timesteps=args.train, verbose=1)
    else:
        print("Warning: no checkpoint or --train specified. "
              "Dreams from an untrained model will be random noise.")

    viz = DreamVisualizer(agent)

    if args.quick:
        viz.plot_single_dream(horizon=args.horizon, show=args.show)
    else:
        viz.dream_report(
            save_dir=args.output,
            horizon=args.horizon,
            n_cloud=args.n_cloud,
            show=args.show,
        )

    agent.close()


if __name__ == "__main__":
    main()
