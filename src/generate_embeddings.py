import logging  # noqa: D100
import os
from enum import Enum
from typing import List, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from langchain_openai import OpenAIEmbeddings
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


class EmbeddingModelName(Enum):
    """Enum for OpenAI embedding model names."""

    text_embedding_3_small = "text-embedding-3-small"
    text_embedding_3_large = "text-embedding-3-large"


class DimensionalityReductionTechnique(Enum):
    """Enum for dimensionality reduction techniques."""

    TSNE = "t-sne"
    CUT_EMBEDDING = "cut-embedding"


def normalize_l2(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Normalize a list of PyTorch tensors using L2 norm.

    Args:
        tensors (List[torch.Tensor]): List of input tensors.

    Returns
    -------
        List[torch.Tensor]: List of L2-normalized tensors.
    """
    normalized_tensors = []
    for x in tensors:
        # If x is 1D
        if x.ndimension() == 1:
            norm = torch.norm(x, p=2)
            normalized_tensors.append(x if norm == 0 else x / norm)
        # If x is 2D or higher
        else:
            norm = torch.norm(x, p=2, dim=1, keepdim=True)
            normalized_tensors.append(torch.where(norm == 0, x, x / norm))
    return normalized_tensors


class EmbeddingGenerator:
    """A class to generate embeddings using OpenAI's embedding model."""

    def __init__(
        self,
        model_name: EmbeddingModelName,
        embed_dimensions: int,
    ):
        self.embedding_model = self._load_embedding_model(model_name, embed_dimensions)
        self.embedding_model_name = model_name
        self.embed_dimensions = embed_dimensions

    def _load_embedding_model(
        self,
        model_name: EmbeddingModelName,
        dimensions: int,
    ) -> OpenAIEmbeddings:
        """
        Load the embedding model.

        Args:
            model_name (EmbeddingModelName): The name of the embedding model.
            dimensions (int): The dimensions of the embedding.

        Returns
        -------
            OpenAIEmbeddings: The loaded embedding model.
        """
        # A dimension of 512 for 3-small model shows good performance on MTEB benchmark.
        # Source: https://openai.com/index/new-embedding-models-and-api-updates/
        return OpenAIEmbeddings(model=model_name, dimensions=dimensions)  # type: ignore

    def generate_embeddings(
        self,
        texts: list[str],
    ) -> List[torch.Tensor]:
        """
        Generate and optionally reduce embeddings for a list of texts.

        Args:
            texts (list[str]): A list of texts to generate embeddings for.

        Returns
        -------
            List[torch.Tensor]: A list of embeddings, where each embedding
                                is a torch.Tensor.
        """
        output_float_list = self.embedding_model.embed_documents(texts)
        return [torch.tensor(vec) for vec in output_float_list]


def reduce_embeddings_dimensions(
    embeddings: List[torch.Tensor],
    output_dimensions: int,
    dim_reduction_technique: DimensionalityReductionTechnique = (
        DimensionalityReductionTechnique.TSNE
    ),
    normalize: bool = True,
    perplexity: int = 30,
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    Reduce the dimensionality of the given embeddings.

    Args:
        embeddings (List[torch.Tensor]): A list of embeddings to reduce.
        output_dimensions (int): The number of dimensions to reduce to.
        dim_reduction_technique (DimensionalityReductionTechnique): The
            dimensionality reduction technique to use.
        normalize (bool): Whether to normalize the reduced embeddings.
        perplexity (int): The perplexity parameter for t-SNE.
        seed (int): The random seed for reproducibility.

    Returns
    -------
        List[torch.Tensor]: A list of reduced embeddings as PyTorch tensors.
    """
    # set torch random seed for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if dim_reduction_technique == DimensionalityReductionTechnique.CUT_EMBEDDING:
        reduced_embeddings = [embedding[:output_dimensions] for embedding in embeddings]
    elif dim_reduction_technique == DimensionalityReductionTechnique.TSNE:
        if len(embeddings) < perplexity:
            # Perplexity should always be smaller than the number of samples.
            # If the number of samples is smaller than the default perplexity
            # value, we set the perplexity to the number of samples - 2 since a
            # larger value either throws and error or is too big for the algorithm
            # to work properly.
            perplexity = len(embeddings) - 2
            logger.warning(
                f"Only {len(embeddings)} points are provided for t-SNE\
                perplexity is reduced to the number of points - 2."
            )
        # Convert embeddings to numpy array because that is what t-SNE expects.
        np_embeddings = np.array(embeddings)
        tsne = TSNE(
            n_components=output_dimensions, perplexity=perplexity, random_state=42
        )
        # The output of t-SNE is a numpy array, so we need to convert it back to
        # a list of tensors.
        reduced_embeddings = [
            torch.Tensor(e) for e in tsne.fit_transform(np_embeddings)
        ]

    if normalize:
        reduced_embeddings = normalize_l2(reduced_embeddings)

    return reduced_embeddings


def filter_embeddings(
    embeddings: List[torch.Tensor],
    similarity_threshold: float,
) -> Set[int]:
    """Filter embeddings based on cosine similarity.

    This function removes embeddings that are too similar to each other,
    based on a specified threshold while minimizing the number of
    removed points.

    Args:
        embeddings (List[torch.Tensor]): The list of embedding tensors.
        similarity_threshold (float): The threshold for cosine similarity
                        above which capabilities are considered duplicates.

    Returns
    -------
        Set[int]: A set if indices that should be removed from the
                original list of embeddings.
    """
    # Remove close embeddings.
    similarity_matrix = cosine_similarity(embeddings)
    binary_matrix = (similarity_matrix > similarity_threshold).astype(int)
    # Getting the neighbor pairs, and ignoring the diagonal (self neighbors)
    close_pairs = np.argwhere(
        (binary_matrix == 1) & ~np.eye(binary_matrix.shape[0], dtype=bool)
    )
    # Iterate through the similarity matrix
    num_neighbors = {}
    for row_inx in range(len(similarity_matrix)):
        # Count the number of neighbors for each row and
        # subtract 1 to ignore the diagonal (self connection).
        num_neighbors[row_inx] = sum(binary_matrix[row_inx]) - 1
    # Sort the keys in the dictionary by their values in descending order
    sorted_indices = sorted(num_neighbors, key=lambda x: num_neighbors[x], reverse=True)

    # Eliminate all closely similar neighbors while minimizing the number of
    # removed points.
    idx = -1
    remove_indices = set()
    close_pairs_list = [tuple(pair) for pair in close_pairs]

    while close_pairs_list:
        idx += 1
        # While there are close embeddings (connections),
        # remove the first index from sorted_indices
        current_index = sorted_indices[idx]
        if any(current_index in pair for pair in close_pairs_list):
            remove_indices.add(current_index)
            close_pairs_list = [
                pair for pair in close_pairs_list if current_index not in pair
            ]

    # Remaining points that are left in sorted_indices are filtered embedding indices.
    return set(sorted_indices) - remove_indices


def visualize_embeddings(
    embeddings: List[torch.Tensor],
    save_dir: str,
    plot_name: str,
    point_names: List[str] | None = None,
    seed: int = 42,
) -> None:
    """
    Visualize the embeddings, and make sure they are 2D.

    Args:
        embeddings (List[torch.Tensor]): A list of embeddings to visualize.
        save_dir (str): The directory to save the plot.
        plot_name (str): The name of the plot file.
        point_names (List[str] | None): Optional names for each point in the plot.
        seed (int): The random seed for reproducibility.

    Returns
    -------
        None
    """
    # Check if the embeddings are already 2D
    if embeddings[0].ndimension() > 1 and embeddings[0].size(1) > 2:
        embeddings = reduce_embeddings_dimensions(
            embeddings,
            output_dimensions=2,
            dim_reduction_technique=DimensionalityReductionTechnique.TSNE,
            seed=seed,
        )
    # If point names are provided, annotate each point with its name
    if point_names is not None:
        assert len(point_names) == len(embeddings), (
            "The number of point names must match the number of embeddings."
        )
        x_coords = [embedding[0].item() for embedding in embeddings]
        y_coords = [embedding[1].item() for embedding in embeddings]
        plt.scatter(x_coords, y_coords)
        for i, label in enumerate(point_names):
            plt.text(
                x_coords[i],
                y_coords[i],
                label,
                fontsize=9,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none"},
            )
    else:
        # If no point names are provided, just plot the points
        sns.scatterplot(
            x=[embedding[0].item() for embedding in embeddings],
            y=[embedding[1].item() for embedding in embeddings],
        )

    plt.title("2D Embedding Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close()


def hierarchical_2d_visualization(
    embeddings_by_area: dict[str, List[torch.Tensor]],
    save_dir: str,
    plot_name: str,
    points_area_name_ids: dict[str, dict[str, int]] | None = None,
) -> None:
    """Visualize 2D points grouped by their area.

    Args:
        embeddings_by_area (dict[str, List[torch.Tensor]]): A dictionary where
            keys are areas and values are lists of 2D points (as tensors).
        save_dir (str): The directory to save the plot.
        plot_name (str): The name of the plot file.
        points_area_name_ids (dict[str, dict[str, int]] | None): Optional dictionary
            mapping area names to dictionaries of point names and their IDs.

    """
    # Assert that all points are 2D
    assert all(
        point.ndimension() == 1 and point.size(0) == 2
        for points in embeddings_by_area.values()
        for point in points
    ), "All points must be 2D tensors for visualization."

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", len(embeddings_by_area))

    for i, (area, points) in enumerate(embeddings_by_area.items()):
        points_tensor = torch.stack(points)  # Shape: (N, 2)
        x = points_tensor[:, 0].numpy()
        y = points_tensor[:, 1].numpy()
        plt.scatter(x, y, label=area, color=colors[i], alpha=0.6, edgecolor="k", s=90)

        # Write point IDs on each point.
        if points_area_name_ids is not None:
            area_points = points_area_name_ids.get(area)
            if area_points is not None:
                ids = list(area_points.values())
                for j, (x_coord, y_coord) in enumerate(zip(x, y)):
                    plt.text(
                        x_coord,
                        y_coord,
                        str(ids[j]),
                        fontsize=7,
                        ha="center",
                        va="center",
                        color="black",
                        bbox={"facecolor": "none", "edgecolor": "none"},
                    )

        # Compute cluster center to place label
        center_x = x.mean()
        center_y = y.mean()
        plt.text(
            center_x,
            center_y,
            area,
            fontsize=8,
            weight="normal",
            bbox={"facecolor": colors[i], "alpha": 0.6, "edgecolor": "none"},
            ha="center",
            va="center",
            color="white",
        )

    # If point names are provided, create a legend with names and IDs
    if points_area_name_ids:
        legend_handles = []
        for color_id, (_, names_ids) in enumerate(points_area_name_ids.items()):
            color = colors[color_id]
            for name, point_id in names_ids.items():
                label = f"{point_id}: {name}"
                handle = Line2D(
                    [], [], marker="o", color=color, linestyle="None", label=label
                )
                legend_handles.append(handle)

        plt.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
    plt.title("Hierarchical 2D Embedding Visualization")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.axis("equal")
    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close()


def save_embedding_heatmap(
    embeddings_by_area: dict[str, List[torch.Tensor]],
    capability_names_by_area: dict[str, List[str]],
    save_dir: str,
    plot_name: str,
    add_squares: bool,
    annot_fontsize: int = 14,
    tick_fontsize: int = 16,
) -> None:
    """Generate and save a heatmap of cosine similarity between embeddings.

    This function computes the cosine similarity between a list of
    embeddings and generates a heatmap to visualize the similarity
    matrix. If add_squares is True, it highlights the squares
    along the diagonal corresponding to each area.
    Args:
        embeddings_by_area (dict[str, List[torch.Tensor]]): A dictionary where
            keys are area names and values are lists of embeddings.
        capability_names_by_area (dict[str, List[str]]): A dictionary where
            keys are area names and values are lists of capability names.
        save_dir (str): The directory to save the plot.
        plot_name (str): The name of the plot file.
        add_squares (bool): Whether to add squares around each area's section.
    """
    embeddings = []
    all_capability_names = []
    square_start_indices = {}
    current_idx = 0

    # Process each area to create embedding list and track indices
    for area, tensors in embeddings_by_area.items():
        square_start_indices[area] = current_idx
        embeddings.extend(tensors)

        # Get names for this area and add them to all_capability_names
        if area in capability_names_by_area:
            names = capability_names_by_area[area]
            all_capability_names.extend(names)
        else:
            # Use default names if not provided
            names = [f"{area}_{i}" for i in range(len(tensors))]
            all_capability_names.extend(names)

        current_idx += len(tensors)

    similarity_matrix = cosine_similarity(
        [embedding.numpy() for embedding in embeddings]
    )

    # Calculate figure size based on number of labels
    # to make sure that there's enough space for the text and annotations
    n_elements = len(all_capability_names)
    fig_width = max(12, n_elements * 0.9)
    fig_height = max(10, n_elements * 0.7)

    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=all_capability_names,
        yticklabels=all_capability_names,
        annot_kws={"size": annot_fontsize},
        cbar_kws={"shrink": 0.7},
    )

    # Set X labels and rotate 45 degrees for better visibility
    ax.set_xticklabels(
        all_capability_names,
        rotation=45,
        ha="right",
        fontsize=tick_fontsize,
    )

    # Set Y labels horizontal
    ax.set_yticklabels(
        all_capability_names,
        rotation=0,  # Horizontal text
        ha="right",
        fontsize=tick_fontsize,
    )

    plt.title("Embedding Similarity Heatmap", fontsize=24)

    # tight layout to maximize use of space
    plt.tight_layout(pad=1.0)

    if add_squares:
        # Adding rectangles around each area's section
        # based in square_start_indices that are set
        # when processing each area
        for area, tensors in embeddings_by_area.items():
            start_idx = square_start_indices[area]
            size = len(tensors)
            if size > 0:
                ax.add_patch(
                    Rectangle(
                        (start_idx, start_idx),
                        size,
                        size,
                        fill=False,
                        edgecolor="red",
                        lw=2,
                        clip_on=False,
                    )
                )

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")

    # Save with extra padding to avoid cutoff
    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.8)
    plt.close()
