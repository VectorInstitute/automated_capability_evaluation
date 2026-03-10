"""Visualization utilities for capability analysis and plotting."""

from typing import List

import torch

from src.capability import Capability
from src.generate_embeddings import (
    hierarchical_2d_visualization,
    save_embedding_heatmap,
)


def plot_hierarchical_capability_2d_embeddings(
    capabilities: List[Capability],
    dim_reduction_method: str,
    plot_name: str,
    save_dir: str,
    show_point_ids: bool,
    save_area_legend: bool = True,
) -> None:
    """Visualize the hierarchical capability embeddings.

    Embeddings are retrieved based on the defined dim_reduction_method,
    and they should be 2D.

    Args
    ----
        capabilities (List[Capability]): The list of capabilities.
        dim_reduction_method (str): The dimensionality reduction method to use.
        plot_name (str): The name of the plot to save.
        save_dir (str): The directory to save the plot.
        show_point_ids (bool): Whether to show point IDs in the plot. Set this to
            False for large datasets to avoid cluttering the plot.
        save_area_legend (bool): Whether to save the area legend as a separate plot.

    Returns
    -------
        None
    """
    # Get the reduced embeddings.
    reduced_embeddings = [
        capability.get_embedding(dim_reduction_method) for capability in capabilities
    ]
    area_names = [capability.get_attribute("area") for capability in capabilities]

    # Populate embeddings_by_area, and points_area_name_ids
    embeddings_by_area: dict[str, List[torch.Tensor]] = {}
    points_area_name_ids: dict[str, dict[str, int]] = {}
    for idx in range(len(reduced_embeddings)):
        area_name = area_names[idx]
        if area_name not in embeddings_by_area:
            embeddings_by_area[area_name] = []
            points_area_name_ids[area_name] = {}
        embeddings_by_area[area_name].append(reduced_embeddings[idx])
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
    """
    Generate and save a heatmap of the capabilities based on their embeddings.

    Args:
        capabilities (List[Capability]): the list of capabilities.
        embedding_model_name (str): name of the embedding model used
            to generate the embeddings.
        plot_name (str): name of the plot file to save.
        save_dir (str): directory to save the plot.
        add_squares (bool): whether to add squares to the heatmap.
    """
    # Get the embeddings based on the specified embedding model name.
    embeddings = [
        capability.get_embedding(embedding_model_name) for capability in capabilities
    ]
    # Process capabilities to populate embeddings_by_area and
    # capability_names_by_area.
    area_names = [capability.area for capability in capabilities]
    embeddings_by_area: dict[str, List[torch.Tensor]] = {}
    capability_names_by_area: dict[str, List[str]] = {}
    for idx in range(len(capabilities)):
        embedding_group = area_names[idx]
        if embedding_group not in embeddings_by_area:
            embeddings_by_area[embedding_group] = []
            capability_names_by_area[embedding_group] = []
        embeddings_by_area[embedding_group].append(embeddings[idx])
        capability_names_by_area[embedding_group].append(capabilities[idx].name)

    save_embedding_heatmap(
        embeddings_by_area=embeddings_by_area,
        capability_names_by_area=capability_names_by_area,
        save_dir=save_dir,
        plot_name=plot_name,
        add_squares=add_squares,
    )
