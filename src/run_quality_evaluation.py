"""Script to compute quality metrics (e.g., benchmark difficulty) from existing scores."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.generate_embeddings import EmbeddingGenerator, EmbeddingModelName
from src.utils import (
    compute_benchmark_consistency,
    compute_benchmark_difficulty,
    compute_benchmark_novelty,
    compute_benchmark_separability,
    compute_mdm,
    compute_mmd,
    compute_pad,
)
from src.utils import constants
from src.utils.data_utils import get_run_id
from src.utils.diversity_metrics_dataloaders import (
    CapabilityDataloader,
    HuggingFaceDatasetDataloader,
    JSONLDataloader,
    CSVDataloader,
    DatasetDataloader,
    load_texts_from_dataloader,
)


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


def _create_dataloader_from_config(
    data_path: str,
    dataloader_config: Dict[str, Any],
) -> DatasetDataloader:
    """Create a dataloader from configuration.
    
    Args:
        data_path: Path to the data
        dataloader_config: Configuration dict with 'type' and other fields
        
    Returns:
        DatasetDataloader instance
    """
    dataloader_type = dataloader_config.get("type", "capability")
    
    if dataloader_type == "capability":
        return CapabilityDataloader(data_path)
    
    elif dataloader_type == "huggingface":
        from datasets import load_dataset
        dataset_name = dataloader_config.get("dataset_name")
        split = dataloader_config.get("split", "train")
        subset = dataloader_config.get("subset", None)
        dataset = load_dataset(dataset_name, name=subset, split=split)
        
        return HuggingFaceDatasetDataloader(
            dataset=dataset,
            text_field=dataloader_config.get("text_field", "problem"),
        )
    
    elif dataloader_type == "jsonl":
        return JSONLDataloader(
            jsonl_path=data_path,
            name_field=dataloader_config.get("name_field", "name"),
            description_field=dataloader_config.get("description_field", "description"),
            area_field=dataloader_config.get("area_field"),
            instructions_field=dataloader_config.get("instructions_field"),
            task_field=dataloader_config.get("task_field", "problem"),
        )
    
    elif dataloader_type == "csv":
        return CSVDataloader(
            csv_path=data_path,
            name_field=dataloader_config.get("name_field", "name"),
            description_field=dataloader_config.get("description_field", "description"),
            area_field=dataloader_config.get("area_field"),
            instructions_field=dataloader_config.get("instructions_field"),
            task_field=dataloader_config.get("task_field", "problem"),
        )
    
    else:
        raise ValueError(f"Unknown dataloader type: {dataloader_type}")


def _load_capabilities_and_generate_embeddings(
    capabilities_dir: str,
    embedding_model_name: str,
    embed_dimensions: int,
    dataloader_config: Optional[Dict[str, Any]] = None,
    embedding_backend: str = "openai",
) -> tuple[np.ndarray, List[Any]]:
    """
    Load capabilities from directory and generate embeddings.
    
    Supports both capability format (default) and custom dataloaders.
    Always uses the dataloader system for consistency.
    
    Args:
        capabilities_dir: Directory containing capability subdirectories OR path to data file
        embedding_model_name: Name of embedding model to use
        embed_dimensions: Number of embedding dimensions
        dataloader_config: Optional configuration for custom dataloader.
                          If None, defaults to capability format.
        
    Returns:
        Tuple of (embeddings array, list of items/capabilities)
    """
    # Use dataloader system: default to capability format if no config provided
    if dataloader_config:
        logger.info("Using custom dataloader: %s", dataloader_config.get("type", "unknown"))
        dataloader = _create_dataloader_from_config(capabilities_dir, dataloader_config)
    else:
        # Default: use capability format dataloader
        if not os.path.isdir(capabilities_dir):
            logger.error("capabilities_dir must be a directory when using default capability format: %s", capabilities_dir)
            return np.array([]), []
        logger.info("Using capability format dataloader for %s", capabilities_dir)
        dataloader = CapabilityDataloader(capabilities_dir)
    
    # Extract texts using the dataloader
    texts = load_texts_from_dataloader(dataloader)
    
    if not texts:
        logger.warning("No texts extracted from %s", capabilities_dir)
        return np.array([]), []
    
    logger.info("Extracted %d texts for embedding", len(texts))
    
    # Generate embeddings
    logger.info(
        "Generating embeddings using %s (backend=%s)",
        embedding_model_name,
        embedding_backend,
    )
    if embedding_backend.lower() == "openai":
        # Use existing OpenAI-based EmbeddingGenerator
        embedding_generator = EmbeddingGenerator(
            model_name=EmbeddingModelName(embedding_model_name),
            embed_dimensions=embed_dimensions,
        )
        embeddings = embedding_generator.generate_embeddings(texts)
        embeddings_array = np.array([emb.numpy() for emb in embeddings])
    elif embedding_backend.lower() == "huggingface":
        # Use HuggingFace encoder models such as gte-Qwen
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to import sentence_transformers for HuggingFace embeddings: %s",
                exc,
            )
            return np.array([]), []
        
        hf_model = SentenceTransformer(embedding_model_name)
        embeddings_array = hf_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        # Optionally warn if requested dim does not match actual dim
        if embed_dimensions and embeddings_array.shape[1] != embed_dimensions:
            logger.warning(
                "Requested embed_dimensions=%d but HuggingFace model produced %d dims; "
                "using model's native dimension.",
                embed_dimensions,
                embeddings_array.shape[1],
            )
    else:
        logger.error("Unknown embedding_backend: %s", embedding_backend)
        return np.array([]), []
    
    return embeddings_array, []


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
            # Continue to next model if we processed subdirs
            continue
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
    
    # Compute diversity metrics if capabilities directory is provided
    capabilities_dir = getattr(cfg.quality_eval_cfg, "capabilities_dir", None)
    if capabilities_dir:
        metrics_to_compute = getattr(cfg.quality_eval_cfg, "diversity_metrics", ["pad", "mmd", "mdm"])
        embedding_model = getattr(cfg.quality_eval_cfg, "embedding_model", "text-embedding-3-large")
        embedding_backend = getattr(cfg.quality_eval_cfg, "embedding_backend", "openai")
        embed_dimensions = getattr(cfg.quality_eval_cfg, "embedding_dimensions", 3072)
        
        # Get dataloader config if provided
        synth_dataloader_config = getattr(cfg.quality_eval_cfg, "synthetic_dataloader_config", None)
        if synth_dataloader_config:
            synth_dataloader_config = dict(synth_dataloader_config)
        
        logger.info("Computing diversity metrics for capabilities in %s", capabilities_dir)
        
        # Load capabilities and generate embeddings
        synth_embeddings, capabilities = _load_capabilities_and_generate_embeddings(
            capabilities_dir=capabilities_dir,
            embedding_model_name=embedding_model,
            embed_dimensions=embed_dimensions,
            dataloader_config=synth_dataloader_config,
            embedding_backend=embedding_backend,
        )
        
        if len(synth_embeddings) == 0:
            logger.warning("No embeddings generated, skipping diversity metrics")
        else:
            # Check if real data directory/file is provided for comparison
            real_data_dir = getattr(cfg.quality_eval_cfg, "real_data_dir", None)
            real_dataloader_config = getattr(cfg.quality_eval_cfg, "real_dataloader_config", None)
            
            # Check if we have real data: either a valid path OR a dataloader config (for HuggingFace, etc.)
            has_real_data = False
            # Case 1: local path (capability/JSONL/CSV formats)
            if real_data_dir and (os.path.isdir(real_data_dir) or os.path.isfile(real_data_dir)):
                has_real_data = True
            # Case 2: HuggingFace dataset via dataloader (real_data_dir may be None)
            elif real_dataloader_config and real_dataloader_config.get("type") == "huggingface":
                has_real_data = True
            
            if has_real_data:
                # Get real data dataloader config if provided
                if real_dataloader_config:
                    real_dataloader_config = dict(real_dataloader_config)
                
                if real_data_dir:
                    logger.info("Loading real data embeddings from %s", real_data_dir)
                else:
                    logger.info("Loading real data embeddings using dataloader config (no local path)")
                real_embeddings, _ = _load_capabilities_and_generate_embeddings(
                    # For HuggingFace, the capabilities_dir is unused; fallback to empty string
                    capabilities_dir=real_data_dir or "",
                    embedding_model_name=embedding_model,
                    embed_dimensions=embed_dimensions,
                    dataloader_config=real_dataloader_config,
                    embedding_backend=embedding_backend,
                )
                
                if len(real_embeddings) > 0:
                    # Compute metrics that require both synthetic and real data
                    if "pad" in metrics_to_compute:
                        try:
                            pad_score = compute_pad(
                                synth_embeddings,
                                real_embeddings,
                                classifier_name=getattr(cfg.quality_eval_cfg, "pad_classifier", "LogisticRegression"),
                            )
                            logger.info("PAD score: %.4f", pad_score)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Error computing PAD: %s", e)
                    
                    if "mmd" in metrics_to_compute:
                        try:
                            mmd_kernel = getattr(cfg.quality_eval_cfg, "mmd_kernel", "polynomial")
                            mmd_degree = getattr(cfg.quality_eval_cfg, "mmd_degree", 3)
                            mmd_score = compute_mmd(
                                synth_embeddings,
                                real_embeddings,
                                kernel=mmd_kernel,
                                degree=mmd_degree,
                            )
                            logger.info("MMD score (%s kernel): %.4f", mmd_kernel, mmd_score)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Error computing MMD: %s", e)
                else:
                    logger.warning("No real data embeddings generated, skipping comparison metrics")
            else:
                logger.info("No real_data_dir provided, skipping PAD and MMD (require real data)")
            
            # Compute MDM (can be computed without real data - measures internal diversity)
            if "mdm" in metrics_to_compute:
                try:
                    mdm_n_clusters = getattr(cfg.quality_eval_cfg, "mdm_n_clusters", 5)
                    mdm_metric = getattr(cfg.quality_eval_cfg, "mdm_metric", "euclidean")
                    mdm_score = compute_mdm(
                        synth_embeddings,
                        dummy_placeholder=None,
                        n_clusters=mdm_n_clusters,
                        metric=mdm_metric,
                    )
                    logger.info("MDM score (%d clusters, %s metric): %.4f", mdm_n_clusters, mdm_metric, mdm_score)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Error computing MDM: %s", e)


if __name__ == "__main__":
    main()


