"""Compute quality metrics (e.g., benchmark difficulty) from existing scores."""

import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.generate_embeddings import EmbeddingGenerator, EmbeddingModelName
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
from src.utils.data_utils import get_run_id
from src.utils.diversity_metrics_dataloaders import (
    CapabilityDataloader,
    CSVDataloader,
    DatasetDataloader,
    HuggingFaceDatasetDataloader,
    JSONLDataloader,
    load_texts_from_dataloader,
)


logger = logging.getLogger(__name__)


def _collect_accuracies_from_dir(directory: str) -> List[float]:
    """
    Collect all accuracy values from JSON files in a directory (recursively).

    Args:
        directory: Directory to walk recursively for JSON files.

    Returns
    -------
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


def _load_avg_model_accuracies_from_dir(base_dir: str) -> Dict[str, float]:
    """
    Load model accuracies from a directory structure.

    Args:
        base_dir: Directory containing per-model subdirectories with JSON files.

    Returns
    -------
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


@hydra.main(
    version_base=None, config_path="cfg", config_name="run_quality_evaluation_cfg"
)
def main(cfg: DictConfig) -> None:
    """Compute benchmark-level quality metrics from saved capability scores."""
    run_id = get_run_id(cfg)

    # Synthetic benchmark source (scores + capabilities)
    synthetic_cfg = cfg.quality_eval_cfg.synthetic_source
    scores_root_dir = synthetic_cfg.get("scores_root_dir")
    scores_subdir = synthetic_cfg.get("scores_subdir", "scores")

    if scores_root_dir:
        base_scores_dir = scores_root_dir
    else:
        base_scores_dir = os.path.join(
            constants.BASE_ARTIFACTS_DIR,
            scores_subdir,
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

    for model_name in os.listdir(base_scores_dir):
        model_dir = os.path.join(base_scores_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # Check if model_dir contains subdirectories (generations/runs)
        subdirs = [
            d
            for d in os.listdir(model_dir)
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
                        model_name,
                        gen_dir_name,
                        avg_gen_acc,
                        len(gen_accuracies),
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
        # Structure: model_dir/...json files (no generation subdirectories)
        accuracies = _collect_accuracies_from_dir(model_dir)

        if not accuracies:
            logger.warning(
                "No accuracies found for model '%s' in %s", model_name, model_dir
            )
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

    # Compute novelty using score dirs derived from real_data_source.
    novelty_score_dirs: List[str] = []
    real_source_cfg = cfg.quality_eval_cfg.get("real_data_source")
    real_source_configs: List[Mapping[str, Any]] = []
    if real_source_cfg is not None:
        cfg_container = OmegaConf.to_container(real_source_cfg, resolve=True)
        if isinstance(cfg_container, list):
            real_source_configs = cfg_container  # type: ignore[list-item]
        elif isinstance(cfg_container, Mapping):
            real_source_configs = [cfg_container]  # type: ignore[list-item]

    # Use synthetic_source.scores_root_dir for deriving default score dirs
    scores_root_dir = synthetic_cfg.get("scores_root_dir")
    for src in real_source_configs:
        scores_dir = src.get("scores_dir")
        if not scores_dir:
            src_name = src.get("name")
            if scores_root_dir and src_name:
                scores_dir = os.path.join(scores_root_dir, src_name)
        if scores_dir:
            novelty_score_dirs.append(str(scores_dir))

    if novelty_score_dirs:
        try:
            logger.info("Loading prior datasets for novelty computation...")
            prior_datasets_accuracies: List[Dict[str, float]] = []
            prior_labels: List[str] = []
            for prior_dir in novelty_score_dirs:
                prior_acc = _load_avg_model_accuracies_from_dir(prior_dir)
                if prior_acc:
                    prior_datasets_accuracies.append(prior_acc)
                    prior_labels.append(os.path.basename(os.path.normpath(prior_dir)))
                    logger.info(
                        "Loaded prior dataset from %s: %d models",
                        prior_dir,
                        len(prior_acc),
                    )
                else:
                    logger.warning(
                        "No accuracies found in prior dataset: %s", prior_dir
                    )

            if prior_datasets_accuracies:
                novelty_mode = str(
                    cfg.quality_eval_cfg.get("novelty_mode", "combined")
                ).lower()
                if novelty_mode in ("combined", "both"):
                    novelty = compute_benchmark_novelty(
                        model_to_accuracy,
                        cast(
                            List[Mapping[str, float]], prior_datasets_accuracies
                        ),
                    )
                    logger.info("Benchmark novelty (combined): %.4f", novelty)
                if novelty_mode in ("per_dataset", "both"):
                    for label, prior_acc in zip(
                        prior_labels, prior_datasets_accuracies
                    ):
                        n_per = compute_benchmark_novelty(
                            model_to_accuracy, [prior_acc]
                        )
                        logger.info(
                            "Novelty[%s]: %.4f", label, n_per
                        )
            else:
                logger.warning(
                    "No valid real data score dirs found (real_data_source with scores_dir or name), skipping novelty computation."
                )
        except ValueError as e:
            logger.warning("Could not compute novelty: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error computing novelty: %s", e)

    # Compute embedding-based metrics if synthetic capabilities directory is provided
    capabilities_dir = synthetic_cfg.get("capabilities_dir")
    if capabilities_dir:
        internal_diversity_metrics = cfg.quality_eval_cfg.internal_diversity_metrics
        comparison_metrics = cfg.quality_eval_cfg.comparison_metrics
        embedding_model = cfg.quality_eval_cfg.embedding_model
        embedding_backend = cfg.quality_eval_cfg.embedding_backend
        embed_dimensions = cfg.quality_eval_cfg.embedding_dimensions

        logger.info(
            "Computing embedding-based metrics for capabilities in %s", capabilities_dir
        )

        # Load capabilities and generate embeddings
        synth_embeddings, capabilities = _load_capabilities_and_generate_embeddings(
            capabilities_dir=capabilities_dir,
            embedding_model_name=embedding_model,
            embed_dimensions=embed_dimensions,
            dataloader_config=None,
            embedding_backend=embedding_backend,
        )

        if len(synth_embeddings) == 0:
            logger.warning("No embeddings generated, skipping diversity metrics")
        else:
            real_embeddings = None
            # Real data sources for comparison metrics (PAD, MMD, KL)
            real_mode = str(
                cfg.quality_eval_cfg.get("real_comparison_mode", "pooled")
            ).lower()
            real_source_cfg = cfg.quality_eval_cfg.get("real_data_source")

            # Normalize to a list of source configs: each with optional name, path, dataloader.
            # real_data_source can be a single mapping or a list of mappings.
            real_source_configs: List[Dict[str, Any]] = []
            if real_source_cfg is None:
                logger.info(
                    "real_data_source is not set in config; skipping comparison metrics (PAD, MMD, KL)."
                )
            else:
                cfg_container = OmegaConf.to_container(real_source_cfg, resolve=True)
                if isinstance(cfg_container, list):
                    raw_list: List[Any] = cfg_container
                elif isinstance(cfg_container, Mapping):
                    raw_list = [cfg_container]
                else:
                    raw_list = []
                for i, src in enumerate(raw_list):
                    src_dict = dict(src)
                    src_dict.setdefault("name", f"real_{i}")
                    real_source_configs.append(src_dict)

            real_embeddings_list: List[np.ndarray] = []
            real_names: List[str] = []

            # Load embeddings for each real source
            for src in real_source_configs:
                name = src.get("name", "real")
                real_data_path = src.get("path")
                real_dataloader_cfg = src.get("dataloader")
                if real_dataloader_cfg is not None and not isinstance(
                    real_dataloader_cfg, dict
                ):
                    real_dataloader_cfg = dict(
                        OmegaConf.to_container(real_dataloader_cfg, resolve=True)
                    )

                has_real_data = False
                if real_data_path and (
                    os.path.isdir(real_data_path) or os.path.isfile(real_data_path)
                ):
                    has_real_data = True
                elif real_dataloader_cfg and real_dataloader_cfg.get(
                    "type"
                ) == "huggingface":
                    has_real_data = True

                if not has_real_data:
                    logger.info(
                        "Skipping real source %s: no valid path or dataloader (type=huggingface) provided",
                        name,
                    )
                    continue

                if real_dataloader_cfg is None:
                    real_dataloader_cfg = {}

                if real_data_path:
                    logger.info("Loading real data embeddings from %s", real_data_path)
                else:
                    logger.info(
                        "Loading real data embeddings for %s using dataloader config (no local path)",
                        name,
                    )

                emb_real, _ = _load_capabilities_and_generate_embeddings(
                    capabilities_dir=real_data_path or "",
                    embedding_model_name=embedding_model,
                    embed_dimensions=embed_dimensions,
                    dataloader_config=real_dataloader_cfg,
                    embedding_backend=embedding_backend,
                )
                if emb_real is None or len(emb_real) == 0:
                    logger.warning(
                        "No real data embeddings generated for source %s, skipping it",
                        name,
                    )
                    continue

                real_embeddings_list.append(emb_real)
                real_names.append(name)

            if real_embeddings_list:
                # Pooled real embeddings (used for KL + joint UMAP, and for PAD/MMD in 'pooled' mode)
                real_embeddings = np.vstack(real_embeddings_list)

                # Comparison metrics (need both synth and real)
                if "pad" in comparison_metrics:
                    try:
                        if real_mode == "per_dataset" and len(real_embeddings_list) > 1:
                            for name, emb_real in zip(real_names, real_embeddings_list):
                                pad_score = compute_pad(
                                    synth_embeddings,
                                    emb_real,
                                    classifier_name=cfg.quality_eval_cfg.pad_classifier,
                                )
                                logger.info("PAD[%s]: %.4f", name, pad_score)
                        else:
                            pad_score = compute_pad(
                                synth_embeddings,
                                real_embeddings,
                                classifier_name=cfg.quality_eval_cfg.pad_classifier,
                            )
                            logger.info("PAD (pooled real): %.4f", pad_score)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Error computing PAD: %s", e)

                if "mmd" in comparison_metrics:
                    try:
                        mmd_kernel = cfg.quality_eval_cfg.mmd_kernel
                        mmd_degree = cfg.quality_eval_cfg.mmd_degree
                        if real_mode == "per_dataset" and len(real_embeddings_list) > 1:
                            for name, emb_real in zip(real_names, real_embeddings_list):
                                mmd_score = compute_mmd(
                                    synth_embeddings,
                                    emb_real,
                                    kernel=mmd_kernel,
                                    degree=mmd_degree,
                                )
                                logger.info(
                                    "MMD[%s] (%s kernel): %.4f",
                                    name,
                                    mmd_kernel,
                                    mmd_score,
                                )
                        else:
                            mmd_score = compute_mmd(
                                synth_embeddings,
                                real_embeddings,
                                kernel=mmd_kernel,
                                degree=mmd_degree,
                            )
                            logger.info(
                                "MMD (pooled real, %s kernel): %.4f",
                                mmd_kernel,
                                mmd_score,
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Error computing MMD: %s", e)
            elif real_source_configs:
                logger.warning(
                    "No real data embeddings could be generated for any source. "
                    "Check dataloader config (e.g. dataset_name, text_field) and embedding API/network."
                )
            # When real_source_configs is empty we already logged that real_data_source is not set

            # Joint UMAP (for entropy and/or KL in shared space)
            has_real = (
                real_embeddings is not None and len(real_embeddings) > 0
            )
            umap_n_components = cfg.quality_eval_cfg.umap_n_components
            umap_n_neighbors = cfg.quality_eval_cfg.umap_n_neighbors
            umap_min_dist = cfg.quality_eval_cfg.umap_min_dist
            umap_metric = cfg.quality_eval_cfg.umap_metric
            need_umap = umap_n_components is not None and (
                "entropy" in internal_diversity_metrics
                or ("kl_divergence" in comparison_metrics and has_real)
            )
            synth_reduced = None
            real_reduced = None
            if need_umap:
                embeddings_to_reduce = [synth_embeddings]
                if has_real:
                    embeddings_to_reduce.append(real_embeddings)
                reduced_list = fit_umap(
                    embeddings_to_reduce,
                    umap_n_components,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    metric=umap_metric,
                )
                synth_reduced = reduced_list[0]
                real_reduced = reduced_list[1] if len(reduced_list) > 1 else None

            # KL divergence (joint UMAP so synth and real share a space)
            if "kl_divergence" in comparison_metrics and has_real:
                try:
                    kl_k = cfg.quality_eval_cfg.kl_k
                    kl_synth = (
                        synth_reduced if real_reduced is not None else synth_embeddings
                    )
                    kl_real = (
                        real_reduced if real_reduced is not None else real_embeddings
                    )
                    if kl_synth is not None and kl_real is not None:
                        kl_score = compute_kl_divergence(kl_synth, kl_real, k=kl_k)
                    else:
                        kl_score = 0.0
                    umap_info = (
                        f" (UMAP: {umap_n_components}D)" if umap_n_components else ""
                    )
                    logger.info(
                        "KL divergence score (k=%d)%s: %.4f", kl_k, umap_info, kl_score
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Error computing KL divergence: %s", e)

            # Compute internal diversity metrics (only need synthetic data)
            if "mdm" in internal_diversity_metrics:
                try:
                    mdm_n_clusters = cfg.quality_eval_cfg.mdm_n_clusters
                    mdm_metric = cfg.quality_eval_cfg.mdm_metric
                    mdm_score = compute_mdm(
                        synth_embeddings,
                        n_clusters=mdm_n_clusters,
                        metric=mdm_metric,
                    )
                    logger.info(
                        "MDM score (%d clusters, %s metric): %.4f",
                        mdm_n_clusters,
                        mdm_metric,
                        mdm_score,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Error computing MDM: %s", e)

            if "entropy" in internal_diversity_metrics:
                try:
                    entropy_k = cfg.quality_eval_cfg.entropy_k
                    entropy_emb = (
                        synth_reduced if synth_reduced is not None else synth_embeddings
                    )
                    entropy_score = compute_differential_entropy(
                        entropy_emb, k=entropy_k
                    )
                    umap_info = (
                        f" (UMAP: {umap_n_components}D)" if umap_n_components else ""
                    )
                    logger.info(
                        "Differential entropy score (k=%d)%s: %.4f",
                        entropy_k,
                        umap_info,
                        entropy_score,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Error computing differential entropy: %s", e)


if __name__ == "__main__":
    main()
