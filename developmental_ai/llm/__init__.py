"""
LLM Integration Module
========================
Optional Claude-powered perception, goal generation, and causal reasoning.
Requires: pip install anthropic + ANTHROPIC_API_KEY environment variable.
"""

from developmental_ai.llm.llm_module import (
    LLMModule,
    LLMPerception,
    LLMGoalGenerator,
    LLMReasoning,
)

__all__ = ["LLMModule", "LLMPerception", "LLMGoalGenerator", "LLMReasoning"]
