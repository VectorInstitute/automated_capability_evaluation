"""Embedding helpers used by the current pipeline."""

import logging
from typing import Any, List, Set

import numpy as np
import torch
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


def generate_capability_embeddings(
    capabilities: List[Any],  # List of Capability objects
    embedding_model_name: str,
    embed_dimensions: int,
) -> List[torch.Tensor]:
    """Generate embeddings for capability dataclasses and return tensors."""
    embedding_model = OpenAIEmbeddings(
        model=embedding_model_name,
        dimensions=embed_dimensions,
    )

    texts = []
    for capability in capabilities:
        rep_string = f"{capability.area.area_name}, {capability.capability_name}, {capability.capability_description}"
        logger.debug(f"Representation string: {rep_string}")
        texts.append(rep_string)

    output_float_list = embedding_model.embed_documents(texts)
    return [torch.tensor(vec) for vec in output_float_list]


def filter_embeddings(
    embeddings: List[torch.Tensor],
    similarity_threshold: float,
) -> Set[int]:
    """Filter embeddings based on cosine similarity."""
    similarity_matrix = cosine_similarity(embeddings)
    binary_matrix = (similarity_matrix > similarity_threshold).astype(int)
    close_pairs = np.argwhere(
        (binary_matrix == 1) & ~np.eye(binary_matrix.shape[0], dtype=bool)
    )

    num_neighbors = {}
    for row_inx in range(len(similarity_matrix)):
        num_neighbors[row_inx] = sum(binary_matrix[row_inx]) - 1
    sorted_indices = sorted(num_neighbors, key=lambda x: num_neighbors[x], reverse=True)

    idx = -1
    remove_indices = set()
    close_pairs_list = [tuple(pair) for pair in close_pairs]

    while close_pairs_list:
        idx += 1
        current_index = sorted_indices[idx]
        if any(current_index in pair for pair in close_pairs_list):
            remove_indices.add(current_index)
            close_pairs_list = [
                pair for pair in close_pairs_list if current_index not in pair
            ]

    return set(sorted_indices) - remove_indices
