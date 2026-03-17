"""Standardized schemas for ACE pipeline stages.

This module provides standardized data structures for all pipeline stages,
ensuring consistent input/output formats regardless of internal implementation.
"""

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain
from src.schemas.eval_io_utils import (
    get_eval_dir,
    get_experiment_dir,
    load_capability_scores,
    load_eval_config,
    load_eval_dataset,
    save_capability_scores,
    save_eval_config,
    save_eval_dataset,
)

# Evaluation pipeline schemas
from src.schemas.eval_schemas import (
    CapabilityScore,
    EvalConfig,
    EvalDataset,
)
from src.schemas.experiment_schemas import Experiment
from src.schemas.io_utils import (
    load_areas,
    load_capabilities,
    load_domain,
    load_experiment,
    load_solution,
    load_tasks,
    load_validation,
    save_areas,
    save_capabilities,
    save_domain,
    save_experiment,
    save_solution,
    save_tasks,
    save_validation,
)
from src.schemas.metadata_schemas import PipelineMetadata
from src.schemas.solution_schemas import TaskSolution
from src.schemas.task_schemas import Task
from src.schemas.validation_schemas import ValidationResult


__all__ = [
    # Metadata
    "PipelineMetadata",
    # Experiment schemas (Stage 0)
    "Experiment",
    "Domain",
    # Area schemas
    "Area",
    # Capability schemas
    "Capability",
    # Task schemas
    "Task",
    # Solution schemas
    "TaskSolution",
    # Validation schemas
    "ValidationResult",
    # I/O functions - Save (Generation)
    "save_experiment",
    "save_domain",
    "save_areas",
    "save_capabilities",
    "save_tasks",
    "save_solution",
    "save_validation",
    # I/O functions - Load (Generation)
    "load_experiment",
    "load_domain",
    "load_areas",
    "load_capabilities",
    "load_tasks",
    "load_solution",
    "load_validation",
    # Evaluation schemas
    "EvalConfig",
    "EvalDataset",
    "CapabilityScore",
    # I/O functions - Save (Evaluation)
    "save_eval_config",
    "save_eval_dataset",
    "save_capability_scores",
    # I/O functions - Load (Evaluation)
    "load_eval_config",
    "load_eval_dataset",
    "load_capability_scores",
    # Helper functions
    "get_experiment_dir",
    "get_eval_dir",
]
