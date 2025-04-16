import os  # noqa: D100
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from langchain_openai import OpenAIEmbeddings
from sklearn.manifold import TSNE


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

    Returns
    -------
        List[torch.Tensor]: A list of reduced embeddings as PyTorch tensors.
    """
    # set torch random seed for reproducibility.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if len(embeddings) < perplexity:
        # perplexity should always be smaller than number os samples.
        perplexity = len(embeddings) - 2
        print(
            f"Only {len(embeddings)} points are provided for t-SNE\
              perplexity is reduced to the number of points - 1."
        )
    if dim_reduction_technique == DimensionalityReductionTechnique.CUT_EMBEDDING:
        reduced_embeddings = [embedding[:output_dimensions] for embedding in embeddings]
    elif dim_reduction_technique == DimensionalityReductionTechnique.TSNE:
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
