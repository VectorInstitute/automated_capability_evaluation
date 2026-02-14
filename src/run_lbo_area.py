"""Per-area AL orchestrator."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Literal, Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.capability import Capability
from src.lbo_area import (
    group_by_area,
    plot_single_area_curves,
    run_area_active_learning,
)
from src.utils import constants
from src.utils.capability_discovery_utils import select_complete_capabilities
from src.utils.capability_management_utils import get_previous_capabilities
from src.utils.data_utils import check_cfg, get_run_id
from src.utils.embedding_utils import (
    apply_dimensionality_reduction,
    apply_dimensionality_reduction_to_test_capabilities,
    generate_and_set_capabilities_embeddings,
)
from src.utils.lbo_utils import get_lbo_train_set


Metric = Literal["mse", "ae"]

logger = logging.getLogger(__name__)


def _prepare_capabilities(cfg: DictConfig, run_id: str) -> List[Capability]:
    """Prepare capabilities for active learning."""
    base_capability_dir = os.path.join(
        constants.BASE_ARTIFACTS_DIR,
        f"capabilities_{run_id}",
        cfg.capabilities_cfg.domain,
    )

    capabilities = get_previous_capabilities(
        capability_dir=base_capability_dir,
        score_dir_suffix=run_id,
    )
    logger.info(f"Loaded {len(capabilities)} capabilities.")

    capabilities = select_complete_capabilities(
        capabilities=capabilities,
        strict=False,
        num_tasks_lower_bound=int(
            cfg.capabilities_cfg.num_gen_tasks_per_capability
            * (1 - cfg.capabilities_cfg.num_gen_tasks_buffer)
        ),
    )
    logger.info(f"Selected {len(capabilities)} complete capabilities.")

    generate_and_set_capabilities_embeddings(
        capabilities=capabilities,
        embedding_model_name=cfg.embedding_cfg.embedding_model,
        embed_dimensions=cfg.embedding_cfg.embedding_size,
    )

    for cap in tqdm(capabilities, desc="Loading capability scores"):
        cap.load_scores(subject_llm_name=cfg.subject_llm.name)

    missing_area = [c.name for c in capabilities if getattr(c, "area", None) is None]
    if missing_area:
        raise ValueError(f"Capabilities missing 'area' (first 5): {missing_area[:5]}")

    return capabilities


def _global_split(
    cfg: DictConfig, capabilities: List[Capability]
) -> Tuple[List[Capability], List[Capability]]:
    """Global split of capabilities into train and test sets."""
    train_caps, test_caps = get_lbo_train_set(
        input_data=capabilities,
        train_frac=cfg.lbo_cfg.train_frac,
        stratified=cfg.capabilities_cfg.method == "hierarchical",
        input_categories=[c.area for c in capabilities],
        seed=cfg.exp_cfg.seed,
    )
    return train_caps, test_caps


def _apply_dr(
    cfg: DictConfig,
    train_caps: List[Capability],
    extra_caps_for_fit: List[Capability],
    test_caps: List[Capability],
) -> str:
    """Apply dimensionality reduction."""
    method = cfg.dimensionality_reduction_cfg.no_discovery_reduced_dimensionality_method
    size = cfg.dimensionality_reduction_cfg.no_discovery_reduced_dimensionality_size

    if method == "t-sne":
        _ = apply_dimensionality_reduction(
            capabilities=train_caps + extra_caps_for_fit + test_caps,
            dim_reduction_method_name=method,
            output_dimension_size=size,
            embedding_model_name=cfg.embedding_cfg.embedding_model,
            random_seed=cfg.exp_cfg.seed,
        )
    else:
        dim_model = apply_dimensionality_reduction(
            capabilities=train_caps + extra_caps_for_fit,
            dim_reduction_method_name=method,
            output_dimension_size=size,
            embedding_model_name=cfg.embedding_cfg.embedding_model,
            random_seed=cfg.exp_cfg.seed,
        )
        apply_dimensionality_reduction_to_test_capabilities(
            capabilities=test_caps,
            dim_reduction_method=dim_model,
            embedding_model_name=cfg.embedding_cfg.embedding_model,
        )

    logger.info(f"Dimensionality reduction applied: method={method}, size={size}")
    return str(method)


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """Per-area AL orchestrator."""
    check_cfg(cfg, logger)
    run_id = get_run_id(cfg)

    capabilities = _prepare_capabilities(cfg, run_id)

    global_train, global_test = _global_split(cfg, capabilities)

    num_seed_per_area = getattr(cfg.lbo_cfg, "num_initial_train_per_area", 1)
    train_by_area = group_by_area(global_train)
    test_by_area = group_by_area(global_test)
    logger.info(f"Global train: {len(global_train)}, Global test: {len(global_test)}")

    per_area_sets: Dict[str, Dict[str, List[Capability]]] = {}
    for area, train_list in train_by_area.items():
        test_list = test_by_area.get(area, [])
        k = min(max(1, int(num_seed_per_area)), len(train_list))
        seeds = train_list[:k]
        pool = train_list[k:]
        per_area_sets[area] = {
            "train_all": train_list,
            "seed": seeds,
            "pool": pool,
            "test": test_list,
        }

    all_pools = [c for v in per_area_sets.values() for c in v["pool"]]
    embedding_name = _apply_dr(cfg, global_train, all_pools, global_test)

    results_dir = os.path.join(constants.BASE_ARTIFACTS_DIR, "per_area")
    per_area_dir = os.path.join(results_dir, "per_area")
    os.makedirs(per_area_dir, exist_ok=True)

    iters_per_area = getattr(cfg.lbo_cfg, "num_lbo_runs_per_area", None)
    if iters_per_area is None:
        n_areas = max(1, len(per_area_sets))
        iters_per_area = max(1, int(cfg.lbo_cfg.num_lbo_runs // n_areas))

    metric: Metric = getattr(cfg.lbo_cfg, "area_error_metric", "mse")

    af_tag = "ALM" if cfg.lbo_cfg.acquisition_function == "variance" else "ALC"
    summary: Dict[str, Dict[str, Any]] = {}

    for area, packs in per_area_sets.items():
        seeds = packs["seed"]
        pool = packs["pool"]
        train_all = packs["train_all"]
        test = packs["test"]

        if len(train_all) == 0:
            logger.warning(f"[{area}] No train capabilities; skipping.")
            continue
        logger.info(
            f"[{area}] seeds={len(seeds)}, pool={len(pool)}, test={len(test)}, "
            f"requested_iters={iters_per_area}"
        )
        selected, curves = run_area_active_learning(
            area_name=area,
            train_caps_area=train_all,
            initial_train_area=seeds,
            pool_caps_area=pool,
            test_caps_area=test,
            subject_llm_name=cfg.subject_llm.name,
            embedding_name=embedding_name,
            acquisition_function=cfg.lbo_cfg.acquisition_function,
            num_lbo_iterations=iters_per_area,
            metric=metric,
        )

        out_json = os.path.join(per_area_dir, f"{area}_al_results.json")
        with open(out_json, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "area": area,
                    "metric": metric,
                    "selected_capabilities": [c.name for c in selected],
                    "initial_seed_capabilities": [c.name for c in seeds],
                    "test_capabilities": [c.name for c in test],
                    "train_caps_in_area": [c.name for c in train_all],
                    "error": curves["error"],
                    "avg_std": curves["avg_std"],
                    "cum_cost": curves["cum_cost"],
                    "full_eval_cost_upper": curves["full_eval_cost_upper"][0],
                },
                f,
                indent=4,
            )

        plot_single_area_curves(area, curves, outdir=per_area_dir)

        summary[area] = {
            "json": out_json,
            "error_png": os.path.join(per_area_dir, f"{area}_error_curve.png"),
            "cost_png": os.path.join(per_area_dir, f"{area}_cost_curve.png"),
            "selected_count": len(selected),
            "seed_count": len(seeds),
            "pool_count": len(pool),
            "test_count": len(test),
        }

    index_json = os.path.join(
        per_area_dir, f"per_area_index_{run_id}_{cfg.subject_llm.name}_{af_tag}.json"
    )
    with open(index_json, "w") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Per-area AL finished. Results in: {per_area_dir}")
    logger.info(f"Index: {index_json}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
