"""Schemas for evaluation pipeline stages.

Defines dataclasses for evaluation pipeline:
- EvalConfig: Configuration for evaluation run (Stage 0 output)
- EvalDataset: Dataset for one capability (Stage 0 output)
- CapabilityScore: Score for one capability (Stage 2 output)
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvalConfig:
    """Configuration for the evaluation run.

    Created by Eval Stage 0 (Eval Setup). Contains all configuration needed
    to run the evaluation pipeline, including references to generation outputs.
    """

    experiment_id: str
    eval_tag: str
    subject_llms: List[
        Dict[str, str]
    ]  # [{"name": "gpt-4o", "provider": "openai"}, ...]
    judge_llm: Dict[str, str]  # {"name": "gpt-4o-mini", "provider": "openai"}
    validation_tag: str  # Tag from generation Stage 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "eval_tag": self.eval_tag,
            "subject_llms": self.subject_llms,
            "judge_llm": self.judge_llm,
            "validation_tag": self.validation_tag,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            eval_tag=data["eval_tag"],
            subject_llms=data["subject_llms"],
            judge_llm=data["judge_llm"],
            validation_tag=data["validation_tag"],
        )


@dataclass
class EvalDataset:
    """Dataset prepared for Inspect evaluation.

    Created by Eval Stage 0 (Setup and Dataset Preparation). Contains all info
    needed to run Inspect evaluation for one capability.
    """

    area_id: str
    capability_id: str
    capability_name: str
    domain: str
    tasks: List[
        Dict[str, str]
    ]  # [{"id": "task_000", "input": "...", "target": "..."}, ...]
    num_tasks: int
    prompt_template: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "area_id": self.area_id,
            "capability_id": self.capability_id,
            "capability_name": self.capability_name,
            "domain": self.domain,
            "tasks": self.tasks,
            "num_tasks": self.num_tasks,
            "prompt_template": self.prompt_template,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalDataset":
        """Create from dictionary."""
        return cls(
            area_id=data["area_id"],
            capability_id=data["capability_id"],
            capability_name=data["capability_name"],
            domain=data["domain"],
            tasks=data["tasks"],
            num_tasks=data["num_tasks"],
            prompt_template=data["prompt_template"],
        )


@dataclass
class CapabilityScore:
    """Score for a single capability from evaluation.

    Created by Eval Stage 2 (Score Aggregation). Represents the evaluation
    result for one capability with one subject LLM.
    """

    area_id: str
    capability_id: str
    capability_name: str
    subject_llm: str
    mean: float  # 0.0 to 1.0
    std_err: float
    num_tasks: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "area_id": self.area_id,
            "capability_id": self.capability_id,
            "capability_name": self.capability_name,
            "subject_llm": self.subject_llm,
            "mean": self.mean,
            "std_err": self.std_err,
            "num_tasks": self.num_tasks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityScore":
        """Create from dictionary."""
        return cls(
            area_id=data["area_id"],
            capability_id=data["capability_id"],
            capability_name=data["capability_name"],
            subject_llm=data["subject_llm"],
            mean=data["mean"],
            std_err=data["std_err"],
            num_tasks=data["num_tasks"],
        )
