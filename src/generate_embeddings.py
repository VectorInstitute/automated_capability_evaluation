from typing import List
import numpy as np
from enum import Enum
from sklearn.manifold import TSNE
from langchain_openai import OpenAIEmbeddings

# Type of embedding model
class EmbeddingModelType(Enum):
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
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

class EmbeddingGenerator:

    def __init__(self, model_name: EmbeddingModelType,
                embed_dimensions: int = 512,
                output_dimensions: int = 4,
                dim_reduction_technique: DimensionalityReductionTechnique = DimensionalityReductionTechnique.TSNE):
        self.embedding_model = self._load_embedding_model(model_name, embed_dimensions)
        self.embedding_model_name = model_name
        self.embed_dimensions = embed_dimensions
        self.output_dimensions = output_dimensions
        self.dim_reduction_technique = dim_reduction_technique
        if self.dim_reduction_technique == DimensionalityReductionTechnique.TSNE:
            self.tsne = TSNE(n_components=output_dimensions, random_state=42)

    def _load_embedding_model(self, model_name: EmbeddingModelType, dimensions: int = 512):
        # A dimension of 512 for 3-small model shows really good performance on MTEB benchmark.
        # Source: https://openai.com/index/new-embedding-models-and-api-updates/
        # You can still cut down the embedding vector to smaller dimensions, but you have to normalize it.
        return OpenAIEmbeddings(model=model_name, dimensions=dimensions)

    def generate_embeddings(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Generate and optionally reduce embeddings for a list of texts.

        Args:
            texts (list[str]): A list of texts to generate embeddings for.
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """

        embeddings = self.embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings)

        if self.embed_dimensions != self.output_dimensions:
            # Do dimensionality reduction
            if (
                self.dim_reduction_technique == DimensionalityReductionTechnique.CUT_EMBEDDING
            ):
                embeddings = embeddings[:, : self.output_dimensions]
            elif self.dim_reduction_technique == DimensionalityReductionTechnique.TSNE:
                assert self.tsne is not None, "T-SNE model is not initialized."
                embeddings = self.tsne.fit_transform(embeddings)

            if normalize:
                embeddings = normalize_l2(embeddings)

        return embeddings.tolist()
