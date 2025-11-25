"""I/O utilities for saving and loading pipeline stage outputs."""

import json
from pathlib import Path
from typing import List, Tuple

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.experiment_schemas import Domain, Experiment
from src.schemas.metadata_schemas import PipelineMetadata
from src.schemas.solution_schemas import TaskSolution
from src.schemas.task_schemas import Task
from src.schemas.validation_schemas import ValidationResult


def save_experiment_output(
    experiment: Experiment, metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save experiment output to JSON file.

    Args:
        experiment: Experiment dataclass
        metadata: PipelineMetadata dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        "experiment": experiment.to_dict(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_domain_output(
    domain: Domain, metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save domain output to JSON file.

    Args:
        domain: Domain dataclass
        metadata: PipelineMetadata dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        "domain": domain.to_dict(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_areas_output(
    areas: List[Area], metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save areas output to JSON file.

    Args:
        areas: List of Area dataclasses
        metadata: PipelineMetadata dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        "areas": [area.to_dict() for area in areas],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_capabilities_output(
    capabilities: List[Capability], metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save capabilities output to JSON file.

    Args:
        capabilities: List of Capability dataclasses
        metadata: PipelineMetadata dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        "capabilities": [cap.to_dict() for cap in capabilities],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_tasks_output(
    tasks: List[Task], metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save tasks output to JSON file.

    Args:
        tasks: List of Task dataclasses
        metadata: PipelineMetadata dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        "tasks": [task.to_dict() for task in tasks],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_solution_output(
    task_solution: TaskSolution, metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save solution output to JSON file.

    Args:
        task_solution: TaskSolution dataclass
        metadata: PipelineMetadata dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        **task_solution.to_dict(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_validation_output(
    validation_result: ValidationResult, metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save validation output to JSON file.

    Args:
        validation_result: ValidationResult dataclass
        metadata: PipelineMetadata dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        **validation_result.to_dict(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Load functions


def load_experiment_output(file_path: Path) -> Tuple[Experiment, PipelineMetadata]:
    """Load experiment output from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        Tuple of (Experiment, PipelineMetadata)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = PipelineMetadata.from_dict(data["metadata"])
    experiment = Experiment.from_dict(data["experiment"])
    return experiment, metadata


def load_domain_output(file_path: Path) -> Tuple[Domain, PipelineMetadata]:
    """Load domain output from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        Tuple of (Domain, PipelineMetadata)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = PipelineMetadata.from_dict(data["metadata"])
    domain = Domain.from_dict(data["domain"])
    return domain, metadata


def load_areas_output(file_path: Path) -> Tuple[List[Area], PipelineMetadata]:
    """Load areas output from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        Tuple of (List[Area], PipelineMetadata)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = PipelineMetadata.from_dict(data["metadata"])
    areas = [Area.from_dict(area_data) for area_data in data["areas"]]
    return areas, metadata


def load_capabilities_output(
    file_path: Path,
) -> Tuple[List[Capability], PipelineMetadata]:
    """Load capabilities output from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        Tuple of (List[Capability], PipelineMetadata)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = PipelineMetadata.from_dict(data["metadata"])
    capabilities = [Capability.from_dict(cap_data) for cap_data in data["capabilities"]]
    return capabilities, metadata


def load_tasks_output(file_path: Path) -> Tuple[List[Task], PipelineMetadata]:
    """Load tasks output from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        Tuple of (List[Task], PipelineMetadata)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = PipelineMetadata.from_dict(data["metadata"])
    tasks = [Task.from_dict(task_data) for task_data in data["tasks"]]
    return tasks, metadata


def load_solution_output(file_path: Path) -> Tuple[TaskSolution, PipelineMetadata]:
    """Load solution output from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        Tuple of (TaskSolution, PipelineMetadata)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = PipelineMetadata.from_dict(data["metadata"])
    # Solution files have flattened structure
    # (metadata + all task_solution fields)
    solution_data = {k: v for k, v in data.items() if k != "metadata"}
    task_solution = TaskSolution.from_dict(solution_data)
    return task_solution, metadata


def load_validation_output(
    file_path: Path,
) -> Tuple[ValidationResult, PipelineMetadata]:
    """Load validation output from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        Tuple of (ValidationResult, PipelineMetadata)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = PipelineMetadata.from_dict(data["metadata"])
    # Validation files have flattened structure
    # (metadata + all validation_result fields)
    validation_data = {k: v for k, v in data.items() if k != "metadata"}
    validation_result = ValidationResult.from_dict(validation_data)
    return validation_result, metadata
