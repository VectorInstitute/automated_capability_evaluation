"""Dataclasses for the diverse task generation pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubTopic:
    """Represents a sub-topic within a capability."""

    name: str
    description: Optional[str] = None


@dataclass
class Combination:
    """Represents a valid (content, difficulty, reasoning) combination."""

    content: str
    difficulty: str
    reasoning: str
    rationale: Optional[str] = None


@dataclass
class Blueprint:
    """Represents a task blueprint for a specific combination."""

    combination_id: int
    subtopic: str
    difficulty: str
    reasoning: str
    blueprint: str
    key_characteristics: List[str] = field(default_factory=list)
    example_question_outline: Optional[str] = None
    rationale: Optional[str] = None


@dataclass
class GeneratedTask:
    """Represents a generated multiple-choice task (internal use)."""

    task_id: str
    blueprint_id: int
    subtopic: str
    difficulty: str
    reasoning: str
    question: str
    choices: Dict[str, str]
    correct_answer: str
    explanation: Optional[str] = None
    alignment_notes: Optional[str] = None


@dataclass
class VerificationResult:
    """Represents the verification result for a task."""

    task_id: str
    subtopic_aligned: bool
    difficulty_aligned: bool
    reasoning_aligned: bool
    choices_appropriate: bool
    overall_aligned: bool
    feedback: str
    suggested_improvements: Optional[str] = None
