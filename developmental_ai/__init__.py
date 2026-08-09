"""
Developmental AI Framework
===========================
A curiosity-driven AI agent that learns like a human child — building skills
progressively through intrinsic motivation, with no fixed training dataset
and no externally-defined reward function.

Architecture: Symbolic + Neural Hybrid World Model
  - RSSM world model (DreamerV3-style) with MLP or CNN encoder/decoder
  - Dream-based policy training (DreamerV3-style imagination rollouts)
  - ICM curiosity engine for intrinsic reward from prediction error
  - Knowledge graph (Neo4j production / in-memory fallback) + PyKEEN embeddings
  - GNN (PyTorch Geometric) for learning over graph topology
  - Symbolic-neural gate (NeSyS-inspired) for blending knowledge sources
  - Skill bank with automatic composition + Hugging Face Hub sharing
  - Custom actor-critic policy with reward mixing
  - LLM integration (Ollama — free, local, open-source) for perception & goals
  - Ray-based distributed training for scaling across workers

Core Loop:
  Environment → Curiosity Engine → Set Own Goal → Explore & Act →
  Observe Result → Prediction Error → Update World Model → Add Skill → Repeat
"""

__version__ = "0.2.1"
