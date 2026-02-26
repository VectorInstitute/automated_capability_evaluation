"""Compute quality metrics (e.g., benchmark difficulty) from existing scores."""

import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.generate_embeddings import EmbeddingGenerator, EmbeddingModelName
from src.utils.diversity_metrics_dataloaders import (
    CapabilityDataloader,
    CSVDataloader,
    DatasetDataloader,
    HuggingFaceDatasetDataloader,
    JSONLDataloader,
    load_texts_from_dataloader,
)
from src.utils.quality_evaluation_utils import (
    compute_benchmark_consistency,
    compute_benchmark_difficulty,
    compute_benchmark_novelty,
    compute_benchmark_separability,
    compute_differential_entropy,
    compute_kl_divergence,
    compute_mdm,
    compute_mmd,
    compute_pad,
    fit_umap,
)


logger = logging.getLogger(__name__)


def _as_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert an OmegaConf container-like object to a plain dict.

    Raises if the object cannot be represented as a mapping.
    """
    if isinstance(obj, dict):
        return obj
    container = OmegaConf.to_container(obj, resolve=True)
    if isinstance(container, Mapping):
        mapping = cast(Mapping[str, Any], container)
        return dict(mapping)
    raise TypeError(f"Expected mapping-like config, got: {type(container)}")


def _validate_metric_requirements(cfg: DictConfig) -> None:
    """
    Validate that all required data is provided for the requested metrics.

    Raises ValueError if any required data is missing.
    """
    metrics_to_compute = cfg.get("metrics_to_compute", [])
    if not metrics_to_compute:
        raise ValueError(
            "metrics_to_compute must be specified in config. "
            "Available metrics: difficulty, separability, consistency, novelty, "
            "mdm, entropy, pad, mmd, kl_divergence"
        )

    benchmark_source_cfg = cfg.target_data
    reference_data_source_cfg = cfg.get("reference_datasets")

    # Benchmark metrics (difficulty, separability, consistency) need scores
    benchmark_metrics = {"difficulty", "separability", "consistency"}
    if benchmark_metrics.intersection(metrics_to_compute):
        scores_root_dir = benchmark_source_cfg.get("scores_root_dir")

        if not scores_root_dir:
            raise ValueError(
                "Benchmark metrics "
                f"({benchmark_metrics.intersection(metrics_to_compute)}) "
                "require 'scores_root_dir' to be set in "
                "target_data. "
                "Please provide the path to the directory containing one "
                "subdirectory per subject model."
            )

        base_scores_dir = scores_root_dir

        if not os.path.isdir(base_scores_dir):
            raise ValueError(
                f"Benchmark metrics ({benchmark_metrics.intersection(metrics_to_compute)}) "
                f"require scores directory to exist. "
                f"benchmark scores_root_dir or fallback directory not found: {base_scores_dir}"
            )

        # Check that scores directory contains at least one model subdirectory
        model_dirs = [
            d
            for d in os.listdir(base_scores_dir)
            if os.path.isdir(os.path.join(base_scores_dir, d))
        ]
        if not model_dirs:
            raise ValueError(
                f"Scores directory '{base_scores_dir}' exists but contains no model subdirectories. "
                "Please ensure scores are generated for at least one model."
            )

        # For consistency metric, check that at least one model has generation
        # subdirectories.
        if "consistency" in metrics_to_compute:
            has_generations = False
            for model_name in model_dirs:
                model_dir = os.path.join(base_scores_dir, model_name)
                subdirs = [
                    d
                    for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d))
                ]
                if subdirs:
                    has_generations = True
                    break
            if not has_generations:
                raise ValueError(
                    f"Consistency metric requires multiple generations per model "
                    f"(subdirectories in model directories), but none found in {base_scores_dir}"
                )

    # Internal diversity metrics (mdm, entropy) need capabilities_dir
    internal_metrics = {"mdm", "entropy"}
    if internal_metrics.intersection(metrics_to_compute):
        capabilities_dir = benchmark_source_cfg.get("capabilities_dir")
        if not capabilities_dir:
            raise ValueError(
                f"Internal diversity metrics ({internal_metrics.intersection(metrics_to_compute)}) "
                "require benchmark capabilities_dir"
            )
        if not os.path.isdir(capabilities_dir):
            raise ValueError(
                f"benchmark capabilities_dir does not exist: {capabilities_dir}"
            )
        # Check that capabilities_dir contains at least one capability.json file
        single_cap_json = os.path.join(capabilities_dir, "capability.json")
        if not os.path.exists(single_cap_json):
            # Check subdirectories
            has_capability = False
            for item_name in os.listdir(capabilities_dir):
                item_path = os.path.join(capabilities_dir, item_name)
                if os.path.isdir(item_path):
                    cap_json = os.path.join(item_path, "capability.json")
                    if os.path.exists(cap_json):
                        has_capability = True
                        break
            if not has_capability:
                raise ValueError(
                    f"benchmark capabilities_dir '{capabilities_dir}' exists but contains "
                    "no capability.json files (neither directly nor in subdirectories)"
                )

    # Comparison metrics (pad, mmd, kl_divergence) need benchmark + reference data
    comparison_metrics = {"pad", "mmd", "kl_divergence"}
    if comparison_metrics.intersection(metrics_to_compute):
        capabilities_dir = benchmark_source_cfg.get("capabilities_dir")
        if not capabilities_dir:
            raise ValueError(
                f"Comparison metrics ({comparison_metrics.intersection(metrics_to_compute)}) "
                "require benchmark capabilities_dir"
            )
        if not os.path.isdir(capabilities_dir):
            raise ValueError(
                f"benchmark capabilities_dir does not exist: {capabilities_dir}"
            )
        # Check that capabilities_dir contains at least one capability.json file
        single_cap_json = os.path.join(capabilities_dir, "capability.json")
        if not os.path.exists(single_cap_json):
            # Check subdirectories
            has_capability = False
            for item_name in os.listdir(capabilities_dir):
                item_path = os.path.join(capabilities_dir, item_name)
                if os.path.isdir(item_path):
                    cap_json = os.path.join(item_path, "capability.json")
                    if os.path.exists(cap_json):
                        has_capability = True
                        break
            if not has_capability:
                raise ValueError(
                    f"benchmark capabilities_dir '{capabilities_dir}' exists but contains "
                    "no capability.json files (neither directly nor in subdirectories)"
                )

        if reference_data_source_cfg is None:
            raise ValueError(
                f"Comparison metrics ({comparison_metrics.intersection(metrics_to_compute)}) "
                "require reference_datasets to be configured"
            )

        # Validate each reference source has either path or dataloader
        cfg_container = OmegaConf.to_container(reference_data_source_cfg, resolve=True)
        sources = []
        if isinstance(cfg_container, list):
            sources = cfg_container
        elif isinstance(cfg_container, Mapping):
            sources = [cfg_container]

        if not sources:
            raise ValueError(
                f"Comparison metrics ({comparison_metrics.intersection(metrics_to_compute)}) "
                "require at least one reference_datasets entry"
            )

        for i, src in enumerate(sources):
            src_dict = _as_dict(src)
            name = src_dict.get("name", f"reference_{i}")
            path = src_dict.get("path")
            dataloader = src_dict.get("dataloader")

            has_path = path and (os.path.isdir(path) or os.path.isfile(path))
            has_dataloader = dataloader and dataloader.get("type") == "huggingface"

            if not (has_path or has_dataloader):
                raise ValueError(
                    f"reference_datasets[{i}] ({name}) must have either a valid 'path' "
                    "(existing file/directory) or 'dataloader' with type='huggingface'"
                )

    # Novelty needs reference_datasets with score directories (prior accuracies)
    if "novelty" in metrics_to_compute:
        if reference_data_source_cfg is None:
            raise ValueError(
                "Novelty metric requires reference_datasets (prior accuracies) to be configured"
            )

        cfg_container = OmegaConf.to_container(reference_data_source_cfg, resolve=True)
        sources = []
        if isinstance(cfg_container, list):
            sources = cfg_container
        elif isinstance(cfg_container, Mapping):
            sources = [cfg_container]

        if not sources:
            raise ValueError(
                "Novelty metric requires at least one reference_datasets entry (for prior accuracies)"
            )

        scores_root_dir = benchmark_source_cfg.get("scores_root_dir")
        has_valid_score_dir = False
        checked: List[str] = []
        for i, src in enumerate(sources):
            src_dict = _as_dict(src)
            scores_dir = src_dict.get("scores_dir")
            if not scores_dir:
                src_name = src_dict.get("name")
                if scores_root_dir and src_name:
                    scores_dir = os.path.join(scores_root_dir, src_name)
                else:
                    if not scores_root_dir:
                        checked.append(
                            f"entry {i} (name={src_dict.get('name')}): no scores_dir and scores_root_dir not set"
                        )
                    elif not src_name:
                        checked.append(
                            f"entry {i}: no scores_dir and no name to derive from scores_root_dir"
                        )
                    continue
            if not scores_dir:
                continue
            if not os.path.isdir(scores_dir):
                checked.append(f"{scores_dir!r} (does not exist)")
                continue
            model_dirs = [
                d
                for d in os.listdir(scores_dir)
                if os.path.isdir(os.path.join(scores_dir, d))
            ]
            if not model_dirs:
                checked.append(
                    f"{scores_dir!r} (exists but has no model subdirectories)"
                )
                continue
            has_json = False
            for model_name in model_dirs:
                model_dir = os.path.join(scores_dir, model_name)
                json_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
                if json_files:
                    has_json = True
                    break
            if has_json:
                has_valid_score_dir = True
                break
            checked.append(
                f"{scores_dir!r} (exists, has model subdirs but no .json score files)"
            )

        if not has_valid_score_dir:
            detail = (
                "; ".join(checked)
                if checked
                else "no scores_dir/name derived paths to check"
            )
            raise ValueError(
                "Novelty uses real/reference data via prior accuracies: model scores from evaluating "
                "models on those reference datasets (e.g. MATH-500, MATH-Hard). You must have run that "
                "evaluation and saved scores so they exist at scores_dir (or scores_root_dir/<name>). "
                "Each directory must contain one subdir per model with Inspect eval JSON score files. "
                f"Checked: {detail}. "
                "Either run evaluation on the reference datasets and save scores there, or remove 'novelty' from metrics_to_compute."
            )


def _collect_accuracies_from_inspect_eval_dir(directory: str) -> List[float]:
    """
    Collect accuracy values from Inspect eval JSON files.

    Recursively walks a directory and extracts accuracy values from Inspect eval
    JSON files.

    Single primitive: one dir -> list of accuracies.
    """
    accuracies: List[float] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            json_path = os.path.join(root, fname)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", json_path, exc)
                continue
            try:
                if "error" in data or "results" not in data:
                    continue
                scores = data["results"]["scores"]
                if not scores:
                    continue
                acc = float(scores[0]["metrics"]["accuracy"]["value"])
                accuracies.append(acc)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Failed to extract accuracy from %s: %s", json_path, exc)
    return accuracies


def _load_average_accuracy_per_model_from_scores_dir(base_dir: str) -> Dict[str, float]:
    """
    Load a scores directory with one subdir per model.

    Each model subdir contains Inspect eval JSON files.

    And return model name -> average accuracy. Used for prior (reference) score dirs
    (e.g. novelty).
    Returns empty dict if base_dir does not exist.
    """
    model_to_accuracy: Dict[str, float] = {}
    if not os.path.isdir(base_dir):
        logger.warning("Directory does not exist: %s", base_dir)
        return model_to_accuracy
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        accuracies = _collect_accuracies_from_inspect_eval_dir(model_dir)
        if accuracies:
            model_to_accuracy[model_name] = sum(accuracies) / len(accuracies)
    return model_to_accuracy


def _create_dataloader_from_config(
    data_path: str,
    dataloader_config: Dict[str, Any],
) -> DatasetDataloader:
    """Create a dataloader from configuration.

    Args:
        data_path: Path to the data
        dataloader_config: Configuration dict with 'type' and other fields

    Returns
    -------
        DatasetDataloader instance
    """
    dataloader_type = dataloader_config.get("type", "capability")

    if dataloader_type == "capability":
        return CapabilityDataloader(data_path)

    if dataloader_type == "huggingface":
        from datasets import load_dataset

        dataset_name = dataloader_config.get("dataset_name")
        split = dataloader_config.get("split", "train")
        subset = dataloader_config.get("subset")
        dataset = load_dataset(dataset_name, name=subset, split=split)

        return HuggingFaceDatasetDataloader(
            dataset=dataset,
            text_field=dataloader_config.get("text_field", "problem"),
        )

    if dataloader_type == "jsonl":
        return JSONLDataloader(
            jsonl_path=data_path,
            name_field=dataloader_config.get("name_field", "name"),
            description_field=dataloader_config.get("description_field", "description"),
            area_field=dataloader_config.get("area_field"),
            instructions_field=dataloader_config.get("instructions_field"),
            task_field=dataloader_config.get("task_field", "problem"),
        )

    if dataloader_type == "csv":
        return CSVDataloader(
            csv_path=data_path,
            name_field=dataloader_config.get("name_field", "name"),
            description_field=dataloader_config.get("description_field", "description"),
            area_field=dataloader_config.get("area_field"),
            instructions_field=dataloader_config.get("instructions_field"),
            task_field=dataloader_config.get("task_field", "problem"),
        )

    raise ValueError(f"Unknown dataloader type: {dataloader_type}")


def _load_capabilities_and_generate_embeddings(
    capabilities_dir: str,
    embedding_model_name: str,
    embed_dimensions: int,
    dataloader_config: Optional[Dict[str, Any]] = None,
    embedding_backend: str = "openai",
) -> tuple[np.ndarray, List[str]]:
    """
    Load capabilities from directory and generate embeddings.

    Supports both capability format (default) and custom dataloaders.
    Always uses the dataloader system for consistency.

    Args:
        capabilities_dir: Dir with capability subdirs or path to data file
        embedding_model_name: Name of embedding model to use
        embed_dimensions: Number of embedding dimensions
        dataloader_config: Optional configuration for custom dataloader.
                          If None, defaults to capability format.

    Returns
    -------
        Tuple of (embeddings array, list of extracted texts)
    """
    # Use dataloader system: default to capability format if no config provided
    if dataloader_config:
        logger.info(
            "Using custom dataloader: %s", dataloader_config.get("type", "unknown")
        )
        dataloader = _create_dataloader_from_config(capabilities_dir, dataloader_config)
    else:
        # Default: use capability format dataloader
        if not os.path.isdir(capabilities_dir):
            logger.error(
                "capabilities_dir must be a directory when using default capability format: %s",
                capabilities_dir,
            )
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
            from sentence_transformers import SentenceTransformer
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

    return embeddings_array, texts


def _load_benchmark_scores(
    cfg: DictConfig,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Load benchmark model accuracies from the evaluated scores directory.

    Validation has already run.
    """
    benchmark_source_cfg = cfg.target_data
    scores_root_dir = benchmark_source_cfg.get("scores_root_dir")

    if not scores_root_dir:
        raise ValueError(
            "scores_root_dir must be set in target_data "
            "to load benchmark scores. It should point to a directory that "
            "contains one subdirectory per subject model."
        )

    base_scores_dir = scores_root_dir

    logger.info("Loading model accuracies from %s", base_scores_dir)
    model_to_accuracy: Dict[str, float] = {}
    model_to_generation_accuracies: Dict[str, List[float]] = {}

    for model_name in os.listdir(base_scores_dir):
        model_dir = os.path.join(base_scores_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        subdirs = [
            d
            for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d))
        ]

        if subdirs:
            generation_accuracies: List[float] = []
            for gen_dir_name in sorted(subdirs):
                gen_dir = os.path.join(model_dir, gen_dir_name)
                gen_accuracies = _collect_accuracies_from_inspect_eval_dir(gen_dir)
                if gen_accuracies:
                    avg_gen_acc = sum(gen_accuracies) / len(gen_accuracies)
                    generation_accuracies.append(avg_gen_acc)
                    logger.debug(
                        "Model '%s' generation '%s': %.4f (from %d JSON files)",
                        model_name,
                        gen_dir_name,
                        avg_gen_acc,
                        len(gen_accuracies),
                    )
            if generation_accuracies:
                model_to_generation_accuracies[model_name] = generation_accuracies
                avg_acc = sum(generation_accuracies) / len(generation_accuracies)
                model_to_accuracy[model_name] = avg_acc
                logger.info(
                    "Model '%s' mean accuracy over %d generations: %.4f",
                    model_name,
                    len(generation_accuracies),
                    avg_acc,
                )
            continue

        accuracies = _collect_accuracies_from_inspect_eval_dir(model_dir)
        if not accuracies:
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
        raise RuntimeError(
            f"Unexpected: No valid model accuracies found in {base_scores_dir} "
            "despite validation passing. This may indicate a race condition or file system issue."
        )
    return model_to_accuracy, model_to_generation_accuracies


def _compute_benchmark_metrics(
    model_to_accuracy: Dict[str, float],
    model_to_generation_accuracies: Dict[str, List[float]],
    metrics_to_compute: set,
) -> None:
    """Compute difficulty, separability, and consistency from model accuracies."""
    if "difficulty" in metrics_to_compute:
        difficulty = compute_benchmark_difficulty(model_to_accuracy)
        logger.info("Benchmark difficulty: %.4f", difficulty)

    if "separability" in metrics_to_compute:
        separability = compute_benchmark_separability(model_to_accuracy)
        logger.info("Benchmark separability: %.4f", separability)

    if "consistency" in metrics_to_compute:
        if not model_to_generation_accuracies:
            raise RuntimeError(
                "Unexpected: No model generation accuracies found despite validation passing. "
                "This may indicate a race condition or file system issue."
            )
        consistency = compute_benchmark_consistency(model_to_generation_accuracies)
        logger.info("Benchmark consistency: %.4f", consistency)


def _compute_novelty_metrics(
    cfg: DictConfig,
    benchmark_source_cfg: DictConfig,
    model_to_accuracy: Dict[str, float],
    metrics_to_compute: set,
) -> None:
    """Load prior accuracies and compute one novelty metric using all priors."""
    if "novelty" not in metrics_to_compute:
        return

    reference_data_source_cfg = cfg.get("reference_datasets")
    cfg_container = OmegaConf.to_container(reference_data_source_cfg, resolve=True)
    prior_source_configs = (
        cfg_container if isinstance(cfg_container, list) else [cfg_container]
    )

    scores_root_dir = benchmark_source_cfg.get("scores_root_dir")
    prior_score_dirs: List[str] = []
    for src in prior_source_configs:
        src_dict = _as_dict(src)
        scores_dir = src_dict.get("scores_dir")
        if not scores_dir:
            src_name = src_dict.get("name")
            if scores_root_dir and src_name:
                scores_dir = os.path.join(scores_root_dir, src_name)
        if scores_dir:
            prior_score_dirs.append(str(scores_dir))

    logger.info("Loading prior (previous) accuracies for novelty computation...")
    prior_datasets_accuracies: List[Dict[str, float]] = []
    for prior_dir in prior_score_dirs:
        prior_acc = _load_average_accuracy_per_model_from_scores_dir(prior_dir)
        if not prior_acc:
            raise RuntimeError(
                f"Unexpected: No accuracies found in prior dataset {prior_dir} "
                "despite validation passing. This may indicate a race condition or file system issue."
            )
        prior_datasets_accuracies.append(prior_acc)
        logger.info(
            "Loaded prior dataset from %s: %d models",
            prior_dir,
            len(prior_acc),
        )

    novelty = compute_benchmark_novelty(
        model_to_accuracy,
        cast(List[Mapping[str, float]], prior_datasets_accuracies),
    )
    logger.info("Benchmark novelty: %.4f", novelty)


def _compute_embedding_based_metrics(cfg: DictConfig, metrics_to_compute: set) -> None:
    """Load benchmark and reference embeddings; compute PAD, MMD, KL, MDM, entropy."""
    internal_metrics = {"mdm", "entropy"}
    comparison_metrics = {"pad", "mmd", "kl_divergence"}
    needs_embeddings = bool(
        internal_metrics.intersection(metrics_to_compute)
        or comparison_metrics.intersection(metrics_to_compute)
    )
    if not needs_embeddings:
        return

    benchmark_source_cfg = cfg.target_data
    capabilities_dir = benchmark_source_cfg.get("capabilities_dir")
    embedding_model = cfg.embedding_model
    embedding_backend = cfg.embedding_backend
    embed_dimensions = cfg.embedding_dimensions

    logger.info(
        "Computing embedding-based metrics for capabilities in %s", capabilities_dir
    )
    benchmark_embeddings, capabilities = _load_capabilities_and_generate_embeddings(
        capabilities_dir=capabilities_dir,
        embedding_model_name=embedding_model,
        embed_dimensions=embed_dimensions,
        dataloader_config=None,
        embedding_backend=embedding_backend,
    )
    if len(benchmark_embeddings) == 0:
        raise RuntimeError(
            f"Unexpected: No embeddings generated from {capabilities_dir} "
            "despite validation passing. This may indicate an embedding API/network issue."
        )

    reference_embeddings = None
    reference_embeddings_list: List[np.ndarray] = []
    reference_names: List[str] = []

    if comparison_metrics.intersection(metrics_to_compute):
        reference_data_source_cfg = cfg.get("reference_datasets")
        cfg_container = OmegaConf.to_container(reference_data_source_cfg, resolve=True)
        raw_list = cfg_container if isinstance(cfg_container, list) else [cfg_container]
        reference_source_configs: List[Dict[str, Any]] = []
        for i, src in enumerate(raw_list):
            src_dict = _as_dict(src)
            src_dict.setdefault("name", f"reference_{i}")
            reference_source_configs.append(src_dict)

        for src in reference_source_configs:
            name = src.get("name", "reference")
            reference_data_path = src.get("path")
            reference_dataloader_cfg = src.get("dataloader")
            if reference_dataloader_cfg is not None and not isinstance(
                reference_dataloader_cfg, dict
            ):
                reference_dataloader_cfg = _as_dict(reference_dataloader_cfg)
            if reference_dataloader_cfg is None:
                reference_dataloader_cfg = {}

            if reference_data_path:
                logger.info(
                    "Loading reference data embeddings from %s", reference_data_path
                )
            else:
                logger.info(
                    "Loading reference data embeddings for %s using dataloader config (no local path)",
                    name,
                )
            ref_emb, _ = _load_capabilities_and_generate_embeddings(
                capabilities_dir=reference_data_path or "",
                embedding_model_name=embedding_model,
                embed_dimensions=embed_dimensions,
                dataloader_config=reference_dataloader_cfg,
                embedding_backend=embedding_backend,
            )
            if ref_emb is None or len(ref_emb) == 0:
                raise RuntimeError(
                    f"Failed to generate embeddings for reference source {name}. "
                    "Config validation passed, but embedding generation failed. "
                    "Check embedding API/network connectivity and dataloader configuration."
                )
            reference_embeddings_list.append(ref_emb)
            reference_names.append(name)

        if reference_embeddings_list:
            reference_embeddings = np.vstack(reference_embeddings_list)

            if "pad" in metrics_to_compute:
                for name, ref_emb in zip(reference_names, reference_embeddings_list):
                    pad_score = compute_pad(
                        benchmark_embeddings,
                        ref_emb,
                        classifier_name=cfg.pad_classifier,
                    )
                    logger.info("PAD[%s]: %.4f", name, pad_score)

            if "mmd" in metrics_to_compute:
                mmd_kernel = cfg.mmd_kernel
                mmd_degree = cfg.mmd_degree
                for name, ref_emb in zip(reference_names, reference_embeddings_list):
                    mmd_score = compute_mmd(
                        benchmark_embeddings,
                        ref_emb,
                        kernel=mmd_kernel,
                        degree=mmd_degree,
                    )
                    logger.info(
                        "MMD[%s] (%s kernel): %.4f",
                        name,
                        mmd_kernel,
                        mmd_score,
                    )

    has_reference = reference_embeddings is not None and len(reference_embeddings) > 0
    umap_n_components = cfg.umap_n_components
    umap_n_neighbors = cfg.umap_n_neighbors
    umap_min_dist = cfg.umap_min_dist
    umap_metric = cfg.umap_metric
    need_umap = umap_n_components is not None and (
        "entropy" in metrics_to_compute
        or ("kl_divergence" in metrics_to_compute and has_reference)
    )
    benchmark_reduced = None
    reference_reduced = None
    if need_umap:
        embeddings_to_reduce = [benchmark_embeddings]
        if has_reference:
            assert reference_embeddings is not None
            embeddings_to_reduce.append(reference_embeddings)
        reduced_list = fit_umap(
            embeddings_to_reduce,
            umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
        )
        benchmark_reduced = reduced_list[0]
        reference_reduced = reduced_list[1] if len(reduced_list) > 1 else None

    if "kl_divergence" in metrics_to_compute:
        kl_k = cfg.kl_k
        if reference_reduced is not None:
            assert benchmark_reduced is not None
            kl_benchmark = benchmark_reduced
            kl_reference = reference_reduced
        else:
            kl_benchmark = benchmark_embeddings
            assert reference_embeddings is not None
            kl_reference = reference_embeddings
        assert kl_reference is not None
        kl_score = compute_kl_divergence(kl_benchmark, kl_reference, k=kl_k)
        umap_info = f" (UMAP: {umap_n_components}D)" if umap_n_components else ""
        logger.info("KL divergence score (k=%d)%s: %.4f", kl_k, umap_info, kl_score)

    if "mdm" in metrics_to_compute:
        mdm_n_clusters = cfg.mdm_n_clusters
        mdm_metric = cfg.mdm_metric
        mdm_score = compute_mdm(
            benchmark_embeddings,
            n_clusters=mdm_n_clusters,
            metric=mdm_metric,
        )
        logger.info(
            "MDM score (%d clusters, %s metric): %.4f",
            mdm_n_clusters,
            mdm_metric,
            mdm_score,
        )

    if "entropy" in metrics_to_compute:
        entropy_k = cfg.entropy_k
        entropy_emb = (
            benchmark_reduced if benchmark_reduced is not None else benchmark_embeddings
        )
        entropy_score = compute_differential_entropy(entropy_emb, k=entropy_k)
        umap_info = f" (UMAP: {umap_n_components}D)" if umap_n_components else ""
        logger.info(
            "Differential entropy score (k=%d)%s: %.4f",
            entropy_k,
            umap_info,
            entropy_score,
        )


@hydra.main(
    version_base=None, config_path="cfg", config_name="run_quality_evaluation_cfg"
)
def main(cfg: DictConfig) -> None:
    """Compute benchmark-level quality metrics from saved capability scores."""
    _validate_metric_requirements(cfg)

    metrics_to_compute = set(cfg.metrics_to_compute)
    benchmark_source_cfg = cfg.target_data

    model_to_accuracy, model_to_generation_accuracies = _load_benchmark_scores(cfg)
    _compute_benchmark_metrics(
        model_to_accuracy,
        model_to_generation_accuracies,
        metrics_to_compute,
    )
    _compute_novelty_metrics(
        cfg, benchmark_source_cfg, model_to_accuracy, metrics_to_compute
    )
    _compute_embedding_based_metrics(cfg, metrics_to_compute)


if __name__ == "__main__":
    main()
