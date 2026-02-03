"""Schemas for Item Response Theory (IRT) analysis.

Defines IRTItemParameters and IRTAnalysis dataclasses. IRT parameters are
context-dependent (dataset, subject models, evaluation settings), so they
are stored in a separate analysis object rather than on the Task class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class IRTItemParameters:
    """IRT parameters for a single task/item."""

    task_id: str
    discrimination: float
    difficulty: float
    guessing: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "discrimination": self.discrimination,
            "difficulty": self.difficulty,
            "guessing": self.guessing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IRTItemParameters:
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            discrimination=float(data["discrimination"]),
            difficulty=float(data["difficulty"]),
            guessing=float(data["guessing"]),
        )


@dataclass
class IRTAnalysis:
    """Complete IRT analysis for a dataset evaluation.

    Stores item parameters per task along with the context (dataset,
    subject models, evaluation settings) so that parameters are
    interpretable and comparable.
    """

    # Context: which evaluation this analysis belongs to
    dataset_id: str
    subject_model_names: List[str]
    evaluation_settings: Dict[str, Any] = field(default_factory=dict)

    # IRT parameters per task (keyed by task_id)
    item_parameters: Dict[str, IRTItemParameters] = field(default_factory=dict)

    # Model fit info (n_items, n_persons, model_type, method, note)
    model_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "subject_model_names": self.subject_model_names,
            "evaluation_settings": self.evaluation_settings,
            "item_parameters": {
                tid: p.to_dict() for tid, p in self.item_parameters.items()
            },
            "model_info": self.model_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IRTAnalysis:
        """Create from dictionary."""
        item_params = {
            tid: IRTItemParameters.from_dict(p)
            for tid, p in data.get("item_parameters", {}).items()
        }
        return cls(
            dataset_id=data["dataset_id"],
            subject_model_names=list(data["subject_model_names"]),
            evaluation_settings=dict(data.get("evaluation_settings", {})),
            item_parameters=item_params,
            model_info=dict(data.get("model_info", {})),
        )

    def get_parameters_for_task(self, task_id: str) -> IRTItemParameters | None:
        """Get IRT parameters for a task by id."""
        return self.item_parameters.get(task_id)
