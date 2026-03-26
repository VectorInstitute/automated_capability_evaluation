"""Legacy embedding visualization helper functions."""

import os
import textwrap
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.metrics.pairwise import cosine_similarity


def visualize_embeddings(
    embeddings: List[torch.Tensor],
    save_dir: str,
    plot_name: str,
    point_names: List[str] | None = None,
) -> None:
    """Visualize 2D embeddings."""
    assert all(point.size(0) == 2 for point in embeddings), (
        "All points must be 2D tensors for visualization."
    )

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
        sns.scatterplot(
            x=[embedding[0].item() for embedding in embeddings],
            y=[embedding[1].item() for embedding in embeddings],
        )

    plt.title("2D Embedding Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)

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
    """Visualize 2D points grouped by area."""
    assert all(
        point.ndimension() == 1 and point.size(0) == 2
        for points in embeddings_by_area.values()
        for point in points
    ), "All points must be 2D tensors for visualization."

    plt.figure(figsize=(7, 6))
    colors = sns.color_palette("husl", len(embeddings_by_area))
    for i, (area, points) in enumerate(embeddings_by_area.items()):
        points_tensor = torch.stack(points)
        x = points_tensor[:, 0].numpy()
        y = points_tensor[:, 1].numpy()
        point_color = "red" if area == "test" else colors[i]
        plt.scatter(x, y, label=area, color=point_color, alpha=0.6, s=90)

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

        if area != "test":
            center_x = x.mean()
            center_y = y.mean()
            plt.scatter(
                center_x,
                center_y,
                marker="*",
                s=200,
                color=colors[i],
                edgecolor="black",
                linewidth=1.5,
                label=f"{area} center",
            )

    if points_area_name_ids:
        legend_handles = []
        for color_id, (area, names_ids) in enumerate(points_area_name_ids.items()):
            color = colors[color_id]
            for name, point_id in names_ids.items():
                if area == "test":
                    color = "red"
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

        fig = plt.figure(figsize=(fig_width, 0.1))
        _ = plt.figlegend(
            handles,
            labels,
            loc="center",
            frameon=True,
            fontsize=fontsize,
        )

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
    """Generate and save a heatmap of cosine similarity between embeddings."""
    embeddings = []
    all_capability_names = []
    square_start_indices = {}
    current_idx = 0
    tick_fontsize = 40

    for area, tensors in embeddings_by_area.items():
        square_start_indices[area] = current_idx
        embeddings.extend(tensors)

        if area in capability_names_by_area:
            names = capability_names_by_area[area]
            all_capability_names.extend(names)
        else:
            names = [f"{area}_{i}" for i in range(len(tensors))]
            all_capability_names.extend(names)

        current_idx += len(tensors)

    similarity_matrix = cosine_similarity(
        [embedding.numpy() for embedding in embeddings]
    )

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

    ax.set_xticklabels(
        all_capability_names,
        rotation=45,
        ha="right",
        fontsize=tick_fontsize,
    )

    ax.set_yticklabels(
        all_capability_names,
        rotation=0,
        ha="right",
        fontsize=tick_fontsize,
    )

    plt.tight_layout(pad=1.0)

    if add_squares:
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

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.8)
    plt.close()


def visualize_llm_scores_area_grouped_bar_chart(
    data_dict: Dict[str, Dict[Any, Any]],
    save_dir: str,
    plot_name: str,
    show_error_bars: bool = False,
) -> None:
    """Save a grouped bar chart of LLM scores by area."""
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
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def visualize_llm_scores_spider_chart(
    data: Dict[str, Dict[Any, Any]],
    save_dir: str,
    plot_name: str,
) -> None:
    """Generate and save a spider chart for LLM scores."""
    line_width = 2
    areas = list(data.keys())
    llms = {llm for area_data in data.values() for llm in area_data}
    n = len(areas)
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    _, ax = plt.subplots(figsize=(12, 27), subplot_kw={"polar": True})
    plt.subplots_adjust(left=0.8, right=1.0, top=1.0, bottom=0.8)

    def split_label(label: str, max_length: int = 16) -> str:
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

    for _i, llm in enumerate(llms):
        model_parts = llm.split(",")
        model_name = model_parts[0].strip()
        line_style = "-"

        values = []
        for area in areas:
            if llm in data[area]:
                values.append(data[area][llm][0])
            else:
                values.append(0)

        values += values[:1]

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

    plt.legend(loc="center left", bbox_to_anchor=(1.3, 0.5), fontsize=17)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
