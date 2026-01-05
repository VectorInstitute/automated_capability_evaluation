"""Script to compute quality metrics (e.g., benchmark difficulty) from existing scores."""

import json
import logging
import os
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from src.utils import (
    compute_benchmark_consistency,
    compute_benchmark_difficulty,
    compute_benchmark_novelty,
    compute_benchmark_separability,
)
from src.utils import constants
from src.utils.data_utils import get_run_id


logger = logging.getLogger(__name__)


def _collect_accuracies_from_dir(directory: str) -> List[float]:
    """
    Collect all accuracy values from JSON files in a directory (recursively).
    
    Args:
        directory: Directory to walk recursively for JSON files.
        
    Returns:
        List of accuracy values found in the directory.
    """
    accuracies: List[float] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            json_path = os.path.join(root, fname)
            acc = _extract_accuracy_from_inspect_json(json_path)
            if acc is not None:
                accuracies.append(acc)
    return accuracies


def _load_model_accuracies_from_dir(base_dir: str) -> Dict[str, float]:
    """
    Load model accuracies from a directory structure.
    
    Args:
        base_dir: Directory containing per-model subdirectories with JSON files.
        
    Returns:
        Dictionary mapping model name to average accuracy.
    """
    model_to_accuracy: Dict[str, float] = {}
    
    if not os.path.isdir(base_dir):
        logger.warning("Directory does not exist: %s", base_dir)
        return model_to_accuracy
    
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        accuracies = _collect_accuracies_from_dir(model_dir)
        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            model_to_accuracy[model_name] = avg_acc
    
    return model_to_accuracy


def _extract_accuracy_from_inspect_json(json_path: str) -> float | None:
    """Extract the accuracy metric from a single Inspect eval JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", json_path, exc)
        return None

    try:
        # Check if file has results (successful evaluation) or error (failed evaluation)
        if "error" in data or "results" not in data:
            # File has error or no results, skip it
            return None
        
        scores = data["results"]["scores"]
        if not scores:
            return None
        metrics = scores[0]["metrics"]
        acc = metrics["accuracy"]["value"]
        return float(acc)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Failed to extract accuracy from %s: %s", json_path, exc)
        return None


@hydra.main(version_base=None, config_path="cfg", config_name="run_quality_evaluation_cfg")
def main(cfg: DictConfig) -> None:
    """
    Compute benchmark-level quality metrics from saved capability scores.
    """
    run_id = get_run_id(cfg)

    scores_root_dir = getattr(cfg.quality_eval_cfg, "scores_root_dir", None)
    if scores_root_dir:
        base_scores_dir = scores_root_dir
    else:
        base_scores_dir = os.path.join(
            constants.BASE_ARTIFACTS_DIR,
            cfg.quality_eval_cfg.scores_subdir,
            run_id,
        )
        logger.info("Using fallback scores directory: %s", base_scores_dir)

    if not os.path.isdir(base_scores_dir):
        logger.error(
            "Scores directory '%s' does not exist. "
            "Please ensure scores are generated for run_id '%s'.",
            base_scores_dir,
            run_id,
        )
        return

    logger.info("Loading model accuracies from %s", base_scores_dir)

    # For each model directory, walk all JSON files and average their accuracies.
    model_to_accuracy: Dict[str, float] = {}
    # For consistency: map model to list of accuracies per generation
    model_to_generation_accuracies: Dict[str, List[float]] = {}
    
    # Get prior dataset names to exclude them from current dataset
    prior_datasets = getattr(cfg.quality_eval_cfg, "prior_datasets", [])
    prior_dataset_names = set()
    for prior_path in prior_datasets:
        # Extract the directory name from the path
        prior_name = os.path.basename(os.path.normpath(prior_path))
        prior_dataset_names.add(prior_name)
    
    for model_name in os.listdir(base_scores_dir):
        # Skip if this is a prior dataset directory
        if model_name in prior_dataset_names:
            logger.debug("Skipping prior dataset directory: %s", model_name)
            continue
        model_dir = os.path.join(base_scores_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # Check if model_dir contains subdirectories (generations/runs)
        subdirs = [
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d))
        ]
        
        if subdirs:
            # Structure: model_dir/generation_dir/...json files
            # Each subdirectory represents a different dataset generation
            generation_accuracies: List[float] = []
            for gen_dir_name in sorted(subdirs):
                gen_dir = os.path.join(model_dir, gen_dir_name)
                gen_accuracies = _collect_accuracies_from_dir(gen_dir)
                
                if gen_accuracies:
                    avg_gen_acc = sum(gen_accuracies) / len(gen_accuracies)
                    generation_accuracies.append(avg_gen_acc)
                    logger.debug(
                        "Model '%s' generation '%s': %.4f (from %d JSON files)",
                        model_name, gen_dir_name, avg_gen_acc, len(gen_accuracies)
                    )
            
            if generation_accuracies:
                model_to_generation_accuracies[model_name] = generation_accuracies
                # Overall average across all generations
                avg_acc = sum(generation_accuracies) / len(generation_accuracies)
                model_to_accuracy[model_name] = avg_acc
                logger.info(
                    "Model '%s' mean accuracy over %d generations: %.4f",
                    model_name,
                    len(generation_accuracies),
                    avg_acc,
                )
        else:
            # Structure: model_dir/...json files (no generation subdirectories)
            accuracies = _collect_accuracies_from_dir(model_dir)

            if not accuracies:
                logger.warning("No accuracies found for model '%s' in %s", model_name, model_dir)
                continue

            avg_acc = sum(accuracies) / len(accuracies)
            model_to_accuracy[model_name] = avg_acc
            logger.info(
                "Model '%s' mean accuracy over %d JSON files: %.4f",
                model_name,
                len(accuracies),
                avg_acc,
            )

    if not model_to_accuracy:
        logger.error("No valid model accuracies found in %s", base_scores_dir)
        return

    difficulty = compute_benchmark_difficulty(model_to_accuracy)
    separability = compute_benchmark_separability(model_to_accuracy)
    logger.info("Benchmark difficulty: %.4f", difficulty)
    logger.info("Benchmark separability: %.4f", separability)
    
    # Compute consistency if we have multiple generations per model
    if model_to_generation_accuracies:
        try:
            consistency = compute_benchmark_consistency(model_to_generation_accuracies)
            logger.info("Benchmark consistency: %.4f", consistency)
        except ValueError as e:
            logger.warning("Could not compute consistency: %s", e)
    
    # Compute novelty if prior datasets are provided
    prior_datasets = getattr(cfg.quality_eval_cfg, "prior_datasets", [])
    if prior_datasets:
        try:
            logger.info("Loading prior datasets for novelty computation...")
            prior_datasets_accuracies: List[Dict[str, float]] = []
            for prior_dir in prior_datasets:
                prior_acc = _load_model_accuracies_from_dir(prior_dir)
                if prior_acc:
                    prior_datasets_accuracies.append(prior_acc)
                    logger.info(
                        "Loaded prior dataset from %s: %d models",
                        prior_dir, len(prior_acc)
                    )
                else:
                    logger.warning("No accuracies found in prior dataset: %s", prior_dir)
            
            if prior_datasets_accuracies:
                novelty = compute_benchmark_novelty(model_to_accuracy, prior_datasets_accuracies)
                logger.info("Benchmark novelty: %.4f", novelty)
            else:
                logger.warning("No valid prior datasets found, skipping novelty computation.")
        except ValueError as e:
            logger.warning("Could not compute novelty: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error computing novelty: %s", e)


if __name__ == "__main__":
    main()


