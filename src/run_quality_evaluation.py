"""Script to compute quality metrics (e.g., benchmark difficulty) from existing scores."""

import json
import logging
import os
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from src.utils import (
    compute_benchmark_difficulty,
    compute_benchmark_separability,
)
from src.utils import constants
from src.utils.data_utils import get_run_id


logger = logging.getLogger(__name__)


def _extract_accuracy_from_inspect_json(json_path: str) -> float | None:
    """Extract the accuracy metric from a single Inspect eval JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", json_path, exc)
        return None

    try:
        scores = data["results"]["scores"]
        if not scores:
            return None
        metrics = scores[0]["metrics"]
        acc = metrics["accuracy"]["value"]
        return float(acc)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Failed to extract accuracy from %s: %s", json_path, exc)
        return None


@hydra.main(version_base=None, config_path="cfg", config_name="run_quality_cfg")
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
    for model_name in os.listdir(base_scores_dir):
        model_dir = os.path.join(base_scores_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        accuracies: List[float] = []
        for root, _dirs, files in os.walk(model_dir):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                json_path = os.path.join(root, fname)
                acc = _extract_accuracy_from_inspect_json(json_path)
                if acc is not None:
                    accuracies.append(acc)

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


if __name__ == "__main__":
    main()


