"""Pipeline-facing capability filtering utilities."""

import logging
from typing import Any, List, Tuple

import torch

from src.utils.embedding_utils import filter_embeddings


logger = logging.getLogger(__name__)


def filter_schema_capabilities_by_embeddings(
    capabilities: List[Any],  # List of schema Capability objects
    embeddings: List[torch.Tensor],
    similarity_threshold: float,
) -> Tuple[List[Any], List[int]]:
    """Filter schema capabilities based on embedding similarity.

    This function filters capabilities without mutating them, returning both
    the filtered list and the indices of retained capabilities.

    Args
    ----
        capabilities (List[Any]): The list of schema Capability objects.
        embeddings (List[torch.Tensor]): The embeddings corresponding to capabilities.
        similarity_threshold (float): The threshold for cosine similarity
            above which capabilities are considered duplicates.

    Returns
    -------
        Tuple[List[Any], List[int]]:
            - List of filtered capabilities
            - List of indices of retained capabilities
    """
    if len(capabilities) != len(embeddings):
        raise ValueError(
            f"Number of capabilities ({len(capabilities)}) must match "
            f"number of embeddings ({len(embeddings)})"
        )

    remaining_indices = filter_embeddings(embeddings, similarity_threshold)

    logger.info(
        f"Filtered out {len(capabilities) - len(remaining_indices)} "
        f"capabilities out of {len(capabilities)}"
    )

    filtered_capabilities = [capabilities[i] for i in remaining_indices]
    return filtered_capabilities, list(remaining_indices)
