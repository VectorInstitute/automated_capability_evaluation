import logging  # noqa: D100
import os
import textwrap
from enum import Enum
from typing import Any, Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from langchain_openai import OpenAIEmbeddings
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


class EmbeddingModelName(Enum):
    """Enum for OpenAI embedding model names."""

    text_embedding_3_small = "text-embedding-3-small"
    text_embedding_3_large = "text-embedding-3-large"


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
    assert all(point.size(0) == 2 for point in embeddings), (
        "All points must be 2D tensors for visualization."
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
    save_area_legend: bool = True,
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

    plt.figure(figsize=(7, 6))
    colors = sns.color_palette("husl", len(embeddings_by_area))
    for i, (area, points) in enumerate(embeddings_by_area.items()):
        points_tensor = torch.stack(points)  # Shape: (N, 2)
        x = points_tensor[:, 0].numpy()
        y = points_tensor[:, 1].numpy()
        point_color = "red" if area == "test" else colors[i]
        plt.scatter(x, y, label=area, color=point_color, alpha=0.6, s=90)

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
        # Compute cluster center to place a star
        if area != "test":  # Skip the star for the test area
            center_x = x.mean()
            center_y = y.mean()
            plt.scatter(
                center_x,
                center_y,
                marker="*",
                s=200,  # Size of the star
                color=colors[i],
                edgecolor="black",
                linewidth=1.5,
                label=f"{area} center",
            )

    # If point names are provided, create a legend with names and IDs
    if points_area_name_ids:
        legend_handles = []
        for color_id, (area, names_ids) in enumerate(points_area_name_ids.items()):
            color = colors[color_id]
            for name, point_id in names_ids.items():
                if area == "test":
                    color = "red"  # Use red for test points
                    label = f"{point_id}: {name} (test)"
                else:
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
            prop={"weight": "bold"},
        )
    plt.title(f"{plot_name}", fontsize=30)
    plt.axis("equal")
    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close()

    if save_area_legend:
        fontsize = 23
        fig_width = 12
        text_width = 30
        handles = []
        labels = []

        for i, area in enumerate(embeddings_by_area.keys()):
            if area != "test":
                # Wrap text with increased width parameter
                wrapped_text = "\n".join(textwrap.wrap(area, width=text_width))
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color=colors[i],
                        linestyle="None",
                        markersize=10,
                    )
                )
                labels.append(wrapped_text)

        # Create a figure with specified width but minimal height
        fig = plt.figure(figsize=(fig_width, 0.1))

        # Create a legend that will adjust to the wider figure
        _ = plt.figlegend(
            handles,
            labels,
            loc="center",
            frameon=True,
            fontsize=fontsize,
        )

        # Save with tight bbox
        save_path2 = os.path.join(save_dir, f"{plot_name}_legend.pdf")
        fig.savefig(save_path2, bbox_inches="tight", pad_inches=0.05, format="pdf")
        plt.close(fig)


def save_embedding_heatmap(
    embeddings_by_area: dict[str, List[torch.Tensor]],
    capability_names_by_area: dict[str, List[str]],
    save_dir: str,
    plot_name: str,
    add_squares: bool,
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
    tick_fontsize = 40

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
        annot=False,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=all_capability_names,
        yticklabels=all_capability_names,
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


def visualize_llm_scores_area_grouped_bar_chart(
    data_dict: Dict[str, Dict[Any, Any]],
    save_dir: str,
    plot_name: str,
    show_error_bars: bool = False,
) -> None:
    """
    Save a grouped bar chart of LLM scores by area.

    Args:
        data_dict (Dict[str, Dict]): dictionary containing average LLM scores
            and stds for each area.
        save_dir (str): directory to save the plot.
        plot_name (str): name of the plot file.
        show_error_bars (bool): whether to show error bars on the plot.

    """
    rows = []
    for area, llm_data in data_dict.items():
        for llm, (mean, std) in llm_data.items():
            rows.append({"Area": area, "LLM": llm, "Mean": mean, "Std": std})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(25, 14.5))
    plt.rcParams.update({"font.size": 16})

    sns.set_style("darkgrid")
    sns.set_palette("muted")

    num_llms = len(df["LLM"].unique())
    ncol = max(5, num_llms // 2)

    colors = sns.color_palette("tab10", n_colors=num_llms)

    ax = sns.barplot(
        x="Area",
        y="Mean",
        hue="LLM",
        data=df,
        palette=colors,
        errorbar=("ci", 0),
        dodge=True,
    )

    if show_error_bars:
        bars = ax.patches

        areas = df["Area"].unique()
        llms = df["LLM"].unique()

        error_mapping = {}
        for _, row in df.iterrows():
            key = (row["Area"], row["LLM"])
            error_mapping[key] = row["Std"]
        n_llms = len(llms)

        for i, bar in enumerate(bars):
            area_idx = i // n_llms
            llm_idx = i % n_llms

            if area_idx < len(areas) and llm_idx < len(llms):
                area = areas[area_idx]
                llm = llms[llm_idx]

                std_val = error_mapping.get((area, llm), 0)

                bar_center = bar.get_x() + bar.get_width() / 2
                bar_height = bar.get_height()

                ax.errorbar(
                    x=bar_center,
                    y=bar_height,
                    yerr=std_val,
                    fmt="none",
                    color="black",
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.5,
                )

    ax.set_xlabel("")

    ax.set_xticklabels(
        [textwrap.fill(label, width=20) for label in df["Area"].unique()],
        rotation=45,
        ha="right",
        fontsize=24,
    )

    plt.ylabel("Model Score", fontsize=32)
    ax.tick_params(axis="y", labelsize=22)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        fontsize=22,
        frameon=True,
    )

    plt.tight_layout()
    plt.ylim(0, 1.0)

    plt.subplots_adjust(top=0.85)
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def visualize_llm_scores_spider_chart(
    data: Dict[str, Dict[Any, Any]],
    save_dir: str,
    plot_name: str,
) -> None:
    """
    Generate and save a spider chart for LLM scores.

    Args:
        data (Dict[str, Dict]): dictionary containing average LLM scores
        save_dir (str): directory to save the plot.
        plot_name (str): name of the plot file.

    """
    line_width = 2
    areas = list(data.keys())
    llms = {llm for area_data in data.values() for llm in area_data}
    n = len(areas)
    angles = [n / float(n) * 2 * np.pi for n in range(n)]
    angles += angles[:1]

    _, ax = plt.subplots(figsize=(12, 27), subplot_kw={"polar": True})
    plt.subplots_adjust(left=0.8, right=1.0, top=1.0, bottom=0.8)

    def split_label(label: str, max_length: int = 16) -> str:
        """Split long labels into multiple lines."""
        words = label.split()
        if len(words) == 1 or len(label) <= max_length:
            return label

        lines = []
        current_line: List[str] = []
        current_length = 0

        for word in words:
            if current_length + len(word) + (1 if current_line else 0) <= max_length:
                current_line.append(word)
                current_length += len(word) + (1 if current_line else 0)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    split_areas = [split_label(area) for area in areas]
    plt.xticks(angles[:-1], split_areas, size=15)

    ax.grid(linewidth=0.5, linestyle="dashed", alpha=0.7)
    ax.grid(
        linewidth=0.5,
        linestyle="dashed",
        alpha=0.7,
        which="major",
        axis="x",
    )

    ax.tick_params(axis="x", pad=42)

    palette = {
        "gemini-2.0-flash": "#e6194B",
        "claude-3-7-sonnet-20250219": "#3cb44b",
        "Meta-Llama-3.1-70B-Instruct": "#c875c4",
        "o3-mini": "#4363d8",
        "o1-mini": "#f58231",
    }

    # Plot each LLM
    for _i, llm in enumerate(llms):
        # Extract model name
        model_parts = llm.split(",")
        model_name = model_parts[0].strip()
        line_style = "-"

        values = []
        for area in areas:
            # If the llm has a value for this area, use it, otherwise use 0
            if llm in data[area]:
                values.append(data[area][llm][0])  # Use mean value
            else:
                values.append(0)

        values += values[:1]

        # Plot values
        if model_name in [
            "gemini-2.0-flash",
            "claude-3-7-sonnet-20250219",
            "Meta-Llama-3.1-70B-Instruct",
            "o3-mini",
            "o1-mini",
        ]:
            ax.plot(
                angles,
                values,
                linewidth=line_width,
                linestyle=line_style,
                label=llm,
                color=palette[model_name],
            )

    # Add legend
    plt.legend(loc="center left", bbox_to_anchor=(1.3, 0.5), fontsize=17)
    plt.tight_layout()
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
