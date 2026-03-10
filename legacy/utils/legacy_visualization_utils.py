"""Legacy-only plotting helpers retained for archived scripts and examples."""

from typing import Any, Dict, List

import numpy as np
import torch

from legacy.src.capability import Capability
from src.utils.embedding_utils import (
    hierarchical_2d_visualization,
    save_embedding_heatmap,
    visualize_llm_scores_area_grouped_bar_chart,
    visualize_llm_scores_spider_chart,
)


def plot_capability_scores_spider_and_bar_chart(
    capabilities: List[Capability],
    subject_llm_names: List[str],
    save_dir: str,
    plot_name: str,
    plot_spider_chart: bool = True,
    plot_grouped_bars: bool = True,
) -> None:
    """Plot legacy capability scores by area."""
    llm_scores_by_area: Dict[str, Dict[str, List[float]]] = {}
    for capability in capabilities:
        if capability.area not in llm_scores_by_area:
            llm_scores_by_area[capability.area] = {}
        for llm_name in subject_llm_names:
            if llm_name not in llm_scores_by_area[capability.area]:
                llm_scores_by_area[capability.area][llm_name] = []
            llm_scores_by_area[capability.area][llm_name].append(
                capability.scores[llm_name]["mean"]
            )

    avg_llm_scores_by_area: Dict[str, Dict[str, Any]] = {}
    for area, llm_scores in llm_scores_by_area.items():
        avg_llm_scores_by_area[area] = {}
        for llm_name, scores in llm_scores.items():
            avg_llm_scores_by_area[area][llm_name] = (np.mean(scores), np.std(scores))

    if plot_spider_chart:
        visualize_llm_scores_spider_chart(
            avg_llm_scores_by_area, save_dir, f"{plot_name}_spider_chart"
        )
    if plot_grouped_bars:
        visualize_llm_scores_area_grouped_bar_chart(
            avg_llm_scores_by_area, save_dir, f"{plot_name}_bar_chart"
        )


def plot_hierarchical_capability_2d_embeddings(
    capabilities: List[Capability],
    dim_reduction_method: str,
    plot_name: str,
    save_dir: str,
    show_point_ids: bool,
    save_area_legend: bool = True,
) -> None:
    """Visualize legacy capability embeddings grouped by area."""
    reduced_embeddings = [
        capability.get_embedding(dim_reduction_method) for capability in capabilities
    ]
    area_names = [capability.get_attribute("area") for capability in capabilities]

    embeddings_by_area: dict[str, List[torch.Tensor]] = {}
    points_area_name_ids: dict[str, dict[str, int]] = {}
    for idx, embedding in enumerate(reduced_embeddings):
        area_name = area_names[idx]
        if area_name not in embeddings_by_area:
            embeddings_by_area[area_name] = []
            points_area_name_ids[area_name] = {}
        embeddings_by_area[area_name].append(embedding)
        points_area_name_ids[area_name][capabilities[idx].name] = idx

    hierarchical_2d_visualization(
        embeddings_by_area=embeddings_by_area,
        save_dir=save_dir,
        plot_name=plot_name,
        points_area_name_ids=points_area_name_ids if show_point_ids else None,
        save_area_legend=save_area_legend,
    )


def generate_capability_heatmap(
    capabilities: List[Capability],
    embedding_model_name: str,
    plot_name: str,
    save_dir: str,
    add_squares: bool,
) -> None:
    """Generate and save a legacy capability similarity heatmap."""
    embeddings = [
        capability.get_embedding(embedding_model_name) for capability in capabilities
    ]
    area_names = [capability.area for capability in capabilities]

    embeddings_by_area: dict[str, List[torch.Tensor]] = {}
    capability_names_by_area: dict[str, List[str]] = {}
    for idx, capability in enumerate(capabilities):
        area_name = area_names[idx]
        if area_name not in embeddings_by_area:
            embeddings_by_area[area_name] = []
            capability_names_by_area[area_name] = []
        embeddings_by_area[area_name].append(embeddings[idx])
        capability_names_by_area[area_name].append(capability.name)

    save_embedding_heatmap(
        embeddings_by_area=embeddings_by_area,
        capability_names_by_area=capability_names_by_area,
        save_dir=save_dir,
        plot_name=plot_name,
        add_squares=add_squares,
    )
