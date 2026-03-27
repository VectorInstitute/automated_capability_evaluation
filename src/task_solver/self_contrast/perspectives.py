"""Perspective definitions for the Self-Contrast framework.

Each perspective is a dictionary with ``id``, ``label``, ``guidance``, and
``uses_tools`` keys.  ``PerspectiveResponse`` captures the output of solving
a problem from one perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


BASE_PERSPECTIVES: List[Dict[str, Any]] = [
    {
        "id": "top_down",
        "label": "Top-Down",
        "guidance": (
            "Start from the goal and work backward. Identify the final quantity, "
            "derive the needed formula or decision rule, then plug in the given data."
        ),
        "uses_tools": False,
    },
    {
        "id": "bottom_up",
        "label": "Bottom-Up",
        "guidance": (
            "Start from the given data and compute step by step. Track units and "
            "intermediate values carefully to reach the final answer."
        ),
        "uses_tools": False,
    },
    {
        "id": "analogical",
        "label": "Analogical",
        "guidance": (
            "Solve by analogy to a similar known problem or principle. Use the "
            "analogy to sanity-check formulas and the final answer."
        ),
        "uses_tools": False,
    },
]

TOOL_PERSPECTIVE: Dict[str, Any] = {
    "id": "tool_assisted",
    "label": "Tool-Assisted",
    "guidance": (
        "Use Python scientific computing tools to solve the problem precisely. "
        "Write and execute code using numpy, scipy, sympy, numpy_financial, "
        "or other available libraries to compute the answer."
    ),
    "uses_tools": True,
}


@dataclass
class PerspectiveResponse:
    """Result of solving a problem from a single perspective.

    Attributes
    ----------
    perspective_id : str
        Unique identifier matching a perspective dict's ``"id"`` key.
    label : str
        Human-readable perspective name.
    answer : Optional[str]
        Extracted final answer, or ``None`` on failure.
    rationale : str
        Brief explanation of the reasoning.
    raw_response : str
        Full LLM response text.
    python_code : Optional[str]
        Generated Python code (if any).
    python_output : Optional[str]
        Captured stdout from code execution (if any).
    """

    perspective_id: str
    label: str
    answer: Optional[str]
    rationale: str
    raw_response: str
    python_code: Optional[str] = None
    python_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "perspective_id": self.perspective_id,
            "label": self.label,
            "answer": self.answer,
            "rationale": self.rationale,
            "raw_response": self.raw_response,
            "python_code": self.python_code,
            "python_output": self.python_output,
        }
