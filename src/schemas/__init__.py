"""Standardized schemas for ACE pipeline stages.

This module provides standardized data structures for all pipeline stages,
ensuring consistent input/output formats regardless of internal implementation.
"""

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.experiment_schemas import Domain, Experiment
from src.schemas.io_utils import (
    load_areas_output,
    load_capabilities_output,
    load_domain_output,
    load_experiment_output,
    load_solution_output,
    load_tasks_output,
    load_validation_output,
    save_areas_output,
    save_capabilities_output,
    save_domain_output,
    save_experiment_output,
    save_solution_output,
    save_tasks_output,
    save_validation_output,
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
    # I/O functions - Save
    "save_experiment_output",
    "save_domain_output",
    "save_areas_output",
    "save_capabilities_output",
    "save_tasks_output",
    "save_solution_output",
    "save_validation_output",
    # I/O functions - Load
    "load_experiment_output",
    "load_domain_output",
    "load_areas_output",
    "load_capabilities_output",
    "load_tasks_output",
    "load_solution_output",
    "load_validation_output",
]
