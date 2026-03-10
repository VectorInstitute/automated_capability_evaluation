"""I/O utilities for saving and loading evaluation pipeline outputs."""

import json
from pathlib import Path
from typing import List, Tuple

from src.schemas.eval_schemas import (
    CapabilityScore,
    EvalConfig,
    EvalDataset,
)
from src.schemas.metadata_schemas import PipelineMetadata


# Save functions


def save_eval_config(
    config: EvalConfig, metadata: PipelineMetadata, output_path: Path
) -> None:
    """Save eval config to JSON file.

    Args:
        config: EvalConfig dataclass
        metadata: PipelineMetadata dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata.to_dict(),
        **config.to_dict(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_eval_dataset(dataset: EvalDataset, output_path: Path) -> None:
    """Save eval dataset to JSON file.

    Args:
        dataset: EvalDataset dataclass
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)


def save_capability_scores(scores: List[CapabilityScore], output_path: Path) -> None:
    """Save capability scores to JSON file.

    Args:
        scores: List of CapabilityScore dataclasses
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [score.to_dict() for score in scores]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Load functions


def load_eval_config(file_path: Path) -> Tuple[EvalConfig, PipelineMetadata]:
    """Load eval config from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        Tuple of (EvalConfig, PipelineMetadata)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = PipelineMetadata.from_dict(data["metadata"])
    # Config fields are at top level (alongside metadata)
    config_data = {k: v for k, v in data.items() if k != "metadata"}
    config = EvalConfig.from_dict(config_data)
    return config, metadata


def load_eval_dataset(file_path: Path) -> EvalDataset:
    """Load eval dataset from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        EvalDataset dataclass
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvalDataset.from_dict(data)


def load_capability_scores(file_path: Path) -> List[CapabilityScore]:
    """Load capability scores from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns
    -------
        List of CapabilityScore dataclasses
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CapabilityScore.from_dict(item) for item in data]


# Helper functions


def get_experiment_dir(output_base_dir: str, experiment_id: str) -> Path:
    """Get the experiment directory path.

    Args:
        output_base_dir: Base output directory
        experiment_id: Experiment identifier

    Returns
    -------
        Path to experiment directory
    """
    return Path(output_base_dir) / experiment_id


def get_eval_dir(experiment_dir: Path, eval_tag: str) -> Path:
    """Get the eval output directory path.

    Args:
        experiment_dir: Path to experiment directory
        eval_tag: Eval tag

    Returns
    -------
        Path to eval Stage 1 results directory
    """
    return experiment_dir / "eval" / "results" / eval_tag
