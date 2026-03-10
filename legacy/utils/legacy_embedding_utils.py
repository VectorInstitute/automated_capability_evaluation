"""Legacy embedding and dimensionality-reduction helpers."""

from typing import List

from legacy.src.capability import Capability
from legacy.src.dimensionality_reduction import DimensionalityReductionMethod
from src.utils import constants
from src.utils.embedding_utils import EmbeddingGenerator, EmbeddingModelName


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
