"""Schemas for task generation stage (Stage 3).

Defines Task dataclass for task. Tasks are concrete evaluation items
that test a capability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.schemas.capability_schemas import Capability


@dataclass
class Task:
    """Dataclass for task."""

    task_id: str
    task: str
    task_type: Optional[str] = None  # e.g., "multiple_choice", "open_ended"
    solution_type: Optional[str] = None  # e.g., "multiple_choice", "open_ended"
    difficulty: Optional[str] = None  # e.g., "easy", "medium", "hard"
    bloom_level: Optional[str] = None  # e.g., "remember", "understand", ...
    choices: Optional[List[Dict[str, str]]] = (
        None  # [{"label": "A", "solution": "..."}]
    )
    capability: Capability
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "task_id": self.task_id,
            "task": self.task,
            "task_type": self.task_type,
            "solution_type": self.solution_type,
            "difficulty": self.difficulty,
            "bloom_level": self.bloom_level,
            "choices": self.choices,
            "capability_id": self.capability.capability_id,
            "capability": self.capability.name,
            "capability_description": self.capability.description,
            "area": self.capability.area.name,
            "area_id": self.capability.area.area_id,
            "area_description": self.capability.area.description,
            "domain": self.capability.area.domain.name,
            "domain_id": self.capability.area.domain.domain_id,
        }
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        capability = Capability.from_dict(
            {
                "name": data["capability"],
                "capability_id": data["capability_id"],
                "description": data["capability_description"],
                "area": data["area"],
                "area_id": data["area_id"],
                "area_description": data["area_description"],
                "domain": data["domain"],
                "domain_id": data["domain_id"],
                "domain_description": data.get("domain_description"),
            }
        )
        return cls(
            task_id=data["task_id"],
            task=data["task"],
            task_type=data.get("task_type"),
            solution_type=data.get("solution_type"),
            difficulty=data.get("difficulty"),
            bloom_level=data.get("bloom_level"),
            choices=data.get("choices"),
            capability=capability,
            generation_metadata=data.get("generation_metadata", {}),
        )
