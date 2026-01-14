"""Message and state schema for ReCAP-style multi-agent solving."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RecapStep:
    """A single step in the ReCAP plan."""

    step_id: str
    title: str
    description: str
    expected_output: str
    done: bool = False
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalState:
    """Compact, re-injectable state for long-horizon reasoning."""

    task_id: str
    problem: str
    task_type: str
    goal: str
    givens: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    plan: List[RecapStep] = field(default_factory=list)
    # Final outputs
    final_answer: Optional[str] = None
    numerical_answer: Optional[str] = None
    final_reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["plan"] = [s.to_dict() for s in self.plan]
        return d


# --- Runtime messages (autogen) ---


@dataclass
class RecapTask:
    task_id: str
    problem: str
    capability_name: str
    area_name: str
    task_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecapStepRequest:
    task_id: str
    step_id: str
    step_title: str
    step_description: str
    expected_output: str
    # serialized compact state
    state: Dict[str, Any]
    worker_id: str
    enforce_python: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecapStepResult:
    task_id: str
    step_id: str
    worker_id: str
    result: str
    evidence: str
    confidence: float
    assumptions_used: List[str]
    used_python_tool: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecapFinalSolution:
    task_id: str
    problem: str
    task_type: str
    answer: str
    numerical_answer: str
    reasoning: str
    state: Dict[str, Any]
    step_trace: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

