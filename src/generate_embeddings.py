import os
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_openai import OpenAIEmbeddings
from sklearn.manifold import TSNE


class EmbeddingModelName(Enum):
    text_embedding_3_small = "text-embedding-3-small"
    text_embedding_3_large = "text-embedding-3-large"


class DimensionalityReductionTechnique(Enum):
    TSNE = "T-SNE"
    CUT_EMBEDDING = "CUT_EMBEDDING"


# Taken from https://platform.openai.com/docs/guides/embeddings
def normalize_l2(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x if norm == 0 else x / norm
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(norm == 0, x, x / norm)


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: EmbeddingModelName = EmbeddingModelName.text_embedding_3_small,
        embed_dimensions: int = 512,
    ):
        self.embedding_model = self._load_embedding_model(model_name, embed_dimensions)
        self.embedding_model_name = model_name
        self.embed_dimensions = embed_dimensions

    def _load_embedding_model(
        self, model_name: EmbeddingModelName, dimensions: int = 512
    ):
        # A dimension of 512 for 3-small model shows really good performance on MTEB benchmark.
        # Source: https://openai.com/index/new-embedding-models-and-api-updates/
        # You can still cut down the embedding vector to smaller dimensions, but you have to normalize it.
        return OpenAIEmbeddings(model=model_name, dimensions=dimensions)

    def reduce_embeddings_dimensions(
        self,
        embeddings: List[List[float]],
        output_dimensions: int = 4,
        dim_reduction_technique: DimensionalityReductionTechnique = DimensionalityReductionTechnique.TSNE,
        normalize: bool = True,
        perplexity: int = 30,
    ) -> List[List[float]]:
        """
        Reduce the dimensionality of the given embeddings.

        Args:
            embeddings (List[List[float]]): A list of embeddings to reduce.
            output_dimensions (int): The number of dimensions to reduce to.
            dim_reduction_technique (DimensionalityReductionTechnique): The technique to use for dimensionality reduction.
            normalize (bool): Whether to normalize the reduced embeddings.
            perplexity (int): The perplexity parameter for t-SNE.

        Returns
        -------
            List[List[float]]: A list of reduced embeddings.
        """
        if len(embeddings) < perplexity:
            # perplexity should always be smaller than number os samples.
            perplexity = len(embeddings) - 2
        embeddings = np.array(embeddings)
        if dim_reduction_technique == DimensionalityReductionTechnique.CUT_EMBEDDING:
            reduced_embeddings = np.array(
                [embedding[:output_dimensions] for embedding in embeddings]
            )
        elif dim_reduction_technique == DimensionalityReductionTechnique.TSNE:
            self.tsne = TSNE(
                n_components=output_dimensions, perplexity=perplexity, random_state=42
            )
            reduced_embeddings = self.tsne.fit_transform(embeddings)

        if normalize:
            reduced_embeddings = normalize_l2(reduced_embeddings)

        return reduced_embeddings.tolist()

    def generate_embeddings(
        self,
        texts: list[str],
    ) -> List[List[float]]:
        """
        Generate and optionally reduce embeddings for a list of texts.

        Args:
            texts (list[str]): A list of texts to generate embeddings for.

        Returns
        -------
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        return self.embedding_model.embed_documents(texts)

    def visualize_embeddings(
        self,
        embeddings: List[List[float]],
        save_dir: str,
        plot_name: str,
    ) -> None:
        """
        Visualize the embeddings, and make sure they are 2D.
        Args:
            embeddings (List[List[float]]): A list of embeddings to visualize.
            save_dir (str): The directory to save the plot.
            plot_name (str): The name of the plot file.
        Returns
        -------
            None

        """
        # Check if the embeddings are already 2D
        if len(embeddings[0]) > 2:
            embeddings = self.reduce_embeddings_dimensions(
                embeddings,
                output_dimensions=2,
                dim_reduction_technique=DimensionalityReductionTechnique.TSNE,
            )
        # Plot the 2D embeddings
        df = pd.DataFrame(embeddings, columns=["x", "y"])

        plt.figure(figsize=(10, 8))

        sns.scatterplot(data=df, x="x", y="y", s=50)

        plt.title("T-SNE Visualization of Embeddings")
        plt.xlabel("T-SNE Dim 1")
        plt.ylabel("T-SNE Dim 2")
        plt.legend(loc="best")
        # Save to PDF
        plt.tight_layout()
        plot_path = os.path.join(save_dir, plot_name)
        plt.savefig(plot_path, format="pdf")

        # Optionally close the plot if running in a script or loop
        plt.close()
