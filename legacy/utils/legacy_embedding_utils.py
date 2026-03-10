"""Legacy embedding and dimensionality-reduction helpers."""

from enum import Enum
from typing import List

import torch
from langchain_openai import OpenAIEmbeddings

from legacy.src.capability import Capability
from legacy.src.dimensionality_reduction import DimensionalityReductionMethod
from legacy.utils import legacy_constants as constants


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
        return OpenAIEmbeddings(model=model_name, dimensions=dimensions)  # type: ignore

    def generate_embeddings(
        self,
        texts: list[str],
    ) -> List[torch.Tensor]:
        """Generate embeddings for a list of representation strings."""
        output_float_list = self.embedding_model.embed_documents(texts)
        return [torch.tensor(vec) for vec in output_float_list]


def apply_dimensionality_reduction(
    capabilities: List[Capability],
    dim_reduction_method_name: str,
    output_dimension_size: int,
    embedding_model_name: str,
    tsne_perplexity: int | None = None,
    random_seed: int = constants.DEFAULT_RANDOM_SEED,
    normalize_output: bool = True,
) -> DimensionalityReductionMethod:
    """Apply dimensionality reduction and set reduced embeddings on capabilities."""
    embeddings = []
    for capability in capabilities:
        embedding = capability.get_embedding(embedding_model_name)
        assert embedding is not None, (
            f"Capability {capability} does not have a valid embedding."
        )
        embeddings.append(embedding)

    dim_reduction = DimensionalityReductionMethod.from_name(
        dim_reduction_method_name,
        output_dimension_size,
        random_seed=random_seed,
        normalize_output=normalize_output,
        tsne_perplexity=tsne_perplexity,
    )
    reduced_embeddings = dim_reduction.fit_transform(embeddings)

    for capability, reduced_embedding in zip(capabilities, reduced_embeddings):
        capability.set_embedding(
            embedding_name=dim_reduction_method_name,
            embedding_tensor=reduced_embedding,
        )
    return dim_reduction


def apply_dimensionality_reduction_to_test_capabilities(
    capabilities: List[Capability],
    dim_reduction_method: DimensionalityReductionMethod,
    embedding_model_name: str,
) -> None:
    """Apply a pre-fit dimensionality reduction method to test capabilities."""
    reduced_embeddings = dim_reduction_method.transform_new_points(
        [capability.get_embedding(embedding_model_name) for capability in capabilities]
    )
    for capability, reduced_embedding in zip(capabilities, reduced_embeddings):
        capability.set_embedding(
            embedding_name=dim_reduction_method.method_name,
            embedding_tensor=reduced_embedding,
        )


def generate_and_set_capabilities_embeddings(
    capabilities: List[Capability],
    embedding_model_name: str,
    embed_dimensions: int,
    rep_string_order: str = "and",
) -> None:
    """Generate legacy capability embeddings and store them on each capability."""
    embedding_generator = EmbeddingGenerator(
        model_name=EmbeddingModelName(embedding_model_name),
        embed_dimensions=embed_dimensions,
    )
    texts = []
    for capability in capabilities:
        capability_dict = capability.to_dict(attribute_names=["name", "description"])
        rep_string = ""
        for char in rep_string_order:
            if char == "a":
                rep_string += capability.area + ", "
            elif char == "n":
                rep_string += capability_dict["name"] + ", "
            elif char == "d":
                rep_string += capability_dict["description"] + ", "
            else:
                raise ValueError(f"Invalid field code: {char}")
        rep_string = rep_string.rstrip(", ")
        if not rep_string:
            raise ValueError("Representation string cannot be empty.")
        texts.append(rep_string)

    embeddings = embedding_generator.generate_embeddings(texts)
    for i, capability in enumerate(capabilities):
        capability.set_embedding(
            embedding_name=embedding_model_name,
            embedding_tensor=embeddings[i],
        )
