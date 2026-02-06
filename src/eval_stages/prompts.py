"""Prompts for evaluation pipeline stages."""

# Default prompt template for Inspect AI evaluation
# Used in Stage 1 (Dataset Preparation) when creating EvalDataset
DEFAULT_EVAL_PROMPT_TEMPLATE = """You are an expert. Solve the following problem.

Problem: {input}

Provide your final answer."""
