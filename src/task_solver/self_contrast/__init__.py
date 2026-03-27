"""Self-Contrast evaluation framework.

Implements the Self-Contrast methodology (arXiv:2401.02009) with five solver
variants for evaluating LLMs on structured benchmarks:

- **V1 -- Base Self-Contrast**: Three perspectives (Top-Down, Bottom-Up,
  Analogical) with contrastive reconciliation.
- **V2 -- Single Agent**: Baseline single-call solver for comparison.
- **V3 -- Tools Integrated**: All perspectives use ``PythonExecutor`` for
  scientific computation.
- **V4 -- Tool Perspective**: A fourth tool-assisted perspective is added
  alongside the original three.
- **V5 -- Single Agent + Tools**: Single agent with full scientific toolkit
  access (code generation, execution, then answer).

Provides a lightweight ``LLMClient`` for multi-provider support (OpenAI,
Anthropic, Google Gemini, Ollama, vec-inf, vLLM) without framework
dependencies.
"""

from src.task_solver.self_contrast.model_client import LLMClient
from src.task_solver.self_contrast.perspectives import (
    BASE_PERSPECTIVES,
    TOOL_PERSPECTIVE,
    PerspectiveResponse,
)
from src.task_solver.self_contrast.solver import SelfContrastSolver


__all__ = [
    "LLMClient",
    "SelfContrastSolver",
    "PerspectiveResponse",
    "BASE_PERSPECTIVES",
    "TOOL_PERSPECTIVE",
]
