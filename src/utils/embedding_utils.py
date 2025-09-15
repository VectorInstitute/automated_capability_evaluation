"""Utility functions for capability embeddings and dimensionality reduction."""

import logging
from typing import List

from src.capability import Capability
from src.dimensionality_reduction import DimensionalityReductionMethod
from src.generate_embeddings import (
    EmbeddingGenerator,
    EmbeddingModelName,
)
from src.utils import constants


logger = logging.getLogger(__name__)


def apply_dimensionality_reduction(
    capabilities: List[Capability],
    dim_reduction_method_name: str,
    output_dimension_size: int,
    embedding_model_name: str,
    tsne_perplexity: int | None = None,
    random_seed: int = constants.DEFAULT_RANDOM_SEED,
    normalize_output: bool = True,
) -> DimensionalityReductionMethod:  # noqa: D205
    """Apply dimensionality reduction to the capabilities.

    This function applies dimensionality reduction on a list of Capabilities.
    The reduced embedding is stored in the `embedding_dict` of
    each capability object with embedding_name corresponding to the dimensionality
    reduction algorithm name.

    Args
    ----
        capabilities (List[Capability]): A list of capabilities with
            valid embeddings.
        dim_reduction_method_name (str): The dimensionality reduction method to use.
        output_dimension_size (int): The number of dimensions to reduce to.
        embedding_model_name (str): The name of the OpenAI embedding model used for
            generating the embeddings.
        tsne_perplexity (int | None): The perplexity parameter for T-SNE.
        random_seed (int): The seed for the random number generator.
        normalize_output (bool): Whether to normalize the output embeddings.

    Returns
    -------
        dim_reduction (DimensionalityReductionMethod):
            The dimensionality reduction object. This object
            can be used to transform new embeddings.
    """
    # First, generate embeddings using the specified embedding model,
    # then apply the dimensionality reduction technique (e.g., T-SNE).
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
    # fit_transform() the dimensionality reduction module on the embeddings.
    reduced_embeddings = dim_reduction.fit_transform(embeddings)

    # Set the reduced embeddings for each capability.
    for capability, reduced_embedding in zip(capabilities, reduced_embeddings):
        capability.set_embedding(
            embedding_name=dim_reduction_method_name, embedding_tensor=reduced_embedding
        )
    return dim_reduction


def apply_dimensionality_reduction_to_test_capabilities(
    capabilities: List[Capability],
    dim_reduction_method: DimensionalityReductionMethod,
    embedding_model_name: str,
) -> None:
    """Apply dimensionality reduction to the test capabilities.

    This function applies dimensionality reduction on a list of Capabilities.
    The reduced embedding is stored in the `embedding_dict` of
    each capability object with embedding_name corresponding to the dimensionality
    reduction algorithm name.

    Args
    ----
        capabilities (List[Capability]): A list of capabilities with
            valid embeddings.
        dim_reduction_method (DimensionalityReductionMethod): The dimensionality
            reduction method to use.
        embedding_model_name (str): The name of the embedding model used for
            generating the embeddings.
    """
    # Apply the dimensionality reduction technique on test capabilities.
    reduced_embeddings = dim_reduction_method.transform_new_points(
        [capability.get_embedding(embedding_model_name) for capability in capabilities]
    )

    # Set the reduced embeddings for each capability.
    for capability, reduced_embedding in zip(capabilities, reduced_embeddings):
        capability.set_embedding(
            embedding_name=dim_reduction_method.method_name,
            embedding_tensor=reduced_embedding,
        )


def generate_and_set_capabilities_embeddings(
    capabilities: List[Capability],
    embedding_model_name: str,
    embed_dimensions: int,
    rep_string_order="and",
) -> None:
    """Generate the capabilities embeddings using the OpenAI embedding model.

    The embedding of each capability is set in the capability object.

    Args
    ----
        capabilities (List[Capability]): The list of capabilities.
        embedding_model_name (str): The name of the embedding model to use.
        embed_dimensions (int): The number of dimensions for the embeddings.
        rep_string_order (str): the order of fields that create the representation string.
            Here, "a": Area, "n": Name, "d": Description.
    """
    # Convert the embedding model name to `EmbeddingModelName` to ensure
    # that the provided model name is valid and supported.
    embedding_generator = EmbeddingGenerator(
        model_name=EmbeddingModelName(
            embedding_model_name
        ),  # Conversion of model name makes sure embedding_model_name is supported.
        embed_dimensions=embed_dimensions,
    )
    # Generate embeddings for the capabilities, all at the same time.
    # Embeddings are generated based on the capability name and description.
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
        print(f"Representation string: {rep_string}")
        texts.append(rep_string)
    embeddings = embedding_generator.generate_embeddings(texts)
    # Set embeddings for capabilities.
    for i, capability in enumerate(capabilities):
        capability.set_embedding(
            embedding_name=embedding_model_name, embedding_tensor=embeddings[i]
        )
