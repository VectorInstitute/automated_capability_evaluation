"""Legacy-only plotting helper retained for archived scripts."""

from typing import Any, Dict, List

import numpy as np

from src.capability import Capability
from src.generate_embeddings import (
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
