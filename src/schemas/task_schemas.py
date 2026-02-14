"""Schemas for task generation stage (Stage 3).

Defines Task dataclass for task. Tasks are concrete evaluation items
that test a capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.schemas.capability_schemas import Capability


@dataclass
class Task:
    """Dataclass for task."""

    task_id: str
    task_statement: str
    capability: Capability
    task_type: Optional[str] = None  # e.g., "multiple_choice", "open_ended"
    solution_type: Optional[str] = None  # e.g., "multiple_choice", "open_ended"
    difficulty: Optional[str] = None  # e.g., "easy", "medium", "hard"
    bloom_level: Optional[str] = None  # e.g., "remember", "understand", ...
    choices: Optional[List[Dict[str, str]]] = (
        None  # [{"label": "A", "solution": "..."}]
    )
    generation_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    # IRT (Item Response Theory) parameters from 3PL fit (optional until analysis is run)
    irt_difficulty: Optional[float] = None
    irt_discrimination: Optional[float] = None
    irt_guessing: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "task_id": self.task_id,
            "task_statement": self.task_statement,
            "task_type": self.task_type,
            "solution_type": self.solution_type,
            "difficulty": self.difficulty,
            "bloom_level": self.bloom_level,
            "choices": self.choices,
            **self.capability.to_dict(),
        }
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        if self.irt_difficulty is not None:
            result["irt_difficulty"] = self.irt_difficulty
        if self.irt_discrimination is not None:
            result["irt_discrimination"] = self.irt_discrimination
        if self.irt_guessing is not None:
            result["irt_guessing"] = self.irt_guessing
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Create from dictionary."""
        capability = Capability.from_dict(data)
        return cls(
            task_id=data["task_id"],
            task_statement=data["task_statement"],
            task_type=data.get("task_type"),
            solution_type=data.get("solution_type"),
            difficulty=data.get("difficulty"),
            bloom_level=data.get("bloom_level"),
            choices=data.get("choices"),
            capability=capability,
            generation_metadata=data.get("generation_metadata", {}),
            irt_difficulty=data.get("irt_difficulty"),
            irt_discrimination=data.get("irt_discrimination"),
            irt_guessing=data.get("irt_guessing"),
        )
