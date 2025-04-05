import json  # noqa: D100
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.generate_embeddings import (
    DimensionalityReductionTechnique,
    EmbeddingGenerator,
    EmbeddingModelName,
)


def return_json_str(capability_dict):
    return json.dumps(capability_dict, indent=4)


capabilities_dicts = [
    {
        "name": "math_competition_algebra",
        "description": "The math_competition_algebra capability consists of 2931 challenging competition mathematics problems in algebra. It has 5 levels",
        "domain": "math",
    },
    # Very similar to the first capability
    {
        "name": "algebra",
        "description": "math algebra competition questions and solutions consisting of 5 levels of difficulty.",
        "domain": "math",
    },
    # Example taken from https://www.kaggle.com/datasets/linkanjarad/coding-problems-and-solution-python-code
    {
        "name": "natural_language_to_python",
        "description": "The Natural Language to python capability is composed of 3.3k+ coding problems and their corresponding solution in Python code. The dataset includes but not limited to: working with strings, lists, arrays, tuples, dictionaries, CSV, JSON, and utilizing modules including NumPy, BeautifulSoup, tkinter, Pandas, random, os, re, datetime",
        "domain": "code_generation",
    },
]


# Generate embeddings one time for all the tests.
EMBEDDING_SIZE = 256

# Create a list of capability objects
capabilities_json_str = [
    return_json_str(capability_dict) for capability_dict in capabilities_dicts
]
# Generate embeddings for the capabilities
embedding_generator = EmbeddingGenerator(
    model_name=EmbeddingModelName.text_embedding_3_small,
    embed_dimensions=EMBEDDING_SIZE,
)
# Generate embeddings for the capabilities, all at the same time.
embeddings = embedding_generator.generate_embeddings(
    texts=[capability_str for capability_str in capabilities_json_str],
)


def test_capability_embedding():
    """
    Test that embeddings similarities are ax expected and make sense.
    """
    assert len(embeddings) == len(capabilities_json_str), (
        "The number of embeddings generated should be equal to the number of capabilities."
    )
    # Check the length of embeddings
    assert len(embeddings[0]) == EMBEDDING_SIZE, (
        "The length of the embeddings should be equal to the specified embedding dimensions."
    )

    # Calculate the cosine similarity between the embeddings
    capability_0_2_cosine_similarity = cosine_similarity(
        [embeddings[0]], [embeddings[2]]
    )[0][0]
    capability_0_1_cosine_similarity = cosine_similarity(
        [embeddings[0]], [embeddings[1]]
    )[0][0]
    # The first and second capabilities are very similar, so the cosine similarity should be high
    # The value of capability_0_1_cosine_similarity is 0.9049213992192596
    # The value of capability_0_2_cosine_similarity 0.5984262605198356
    assert capability_0_1_cosine_similarity > capability_0_2_cosine_similarity


def test_filtering_logic():
    """
    Test the filtering logic for capabilities.

    This test verifies that the filtering logic correctly identifies and filters out
    duplicate capabilities based on their embeddings.
    """
    similarity_threshold = 0.85

    # Filter the capabilities
    # Remove capabilities with close embeddings.
    embeddings_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_array)
    # Track which capabilities to keep
    keep_indices = []
    removed_indices = set()

    # Directly taken from generate_capabilities.filter_capabilities
    for i in range(len(embeddings)):
        if i in removed_indices:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(embeddings)):
            if j in removed_indices:
                continue
            # If capabilities have similar embeddings, remove the one with lower score.
            if similarity_matrix[i][j] >= similarity_threshold:
                # We select j to remove.
                removed_indices.add(j)

    assert keep_indices == [0, 2]


def test_dimensionality_reduction_tsne():
    """
    Test the dimensionality reduction of the embeddings using t-SNE.
    """
    # Reduce the dimensionality of the embeddings to 2.
    # Perplexity must be less than n_samples. Because we only have 3 samples here,
    # we need to decrease the t-SNE's perplexity parameter. This value is 30 by default.
    reduced_embeddings = embedding_generator.reduce_embeddings_dimensions(
        embeddings,
        output_dimensions=2,
        dim_reduction_technique=DimensionalityReductionTechnique.TSNE,
        perplexity=1,
    )

    # Check that the reduced embeddings have the correct shape.
    assert len(reduced_embeddings) == len(embeddings)
    assert len(reduced_embeddings[0]) == 2

    # Check if the reduced embeddings are still meaningful.

    # Calculate the cosine similarity between the reduced embeddings
    capability_0_2_cosine_similarity = cosine_similarity(
        [reduced_embeddings[0]], [reduced_embeddings[2]]
    )[0][0]
    capability_0_1_cosine_similarity = cosine_similarity(
        [reduced_embeddings[0]], [reduced_embeddings[1]]
    )[0][0]
    # The value of capability_0_1_cosine_similarity is 0.9999215534801661
    # The value of capability_0_2_cosine_similarity 0.-0.9995052575472013
    assert capability_0_1_cosine_similarity > capability_0_2_cosine_similarity


def test_dimensionality_reduction_cut_embedding():
    # OpenAI embedding-3 models are trained with Matryoshka Representation Learning,
    # So we can naturally cut the embedding vector to smaller dimensions, and they
    # should still be meaningful.

    # Reduce the dimensionality of the embeddings to 2 by just cutting them.
    reduced_embeddings = embedding_generator.reduce_embeddings_dimensions(
        embeddings,
        output_dimensions=2,
        dim_reduction_technique=DimensionalityReductionTechnique.CUT_EMBEDDING,
    )
    # Check that the reduced embeddings have the correct shape.
    assert len(reduced_embeddings) == len(embeddings)
    assert len(reduced_embeddings[0]) == 2
    # Check if the reduced embeddings are still meaningful.
    # Calculate the cosine similarity between the reduced embeddings
    capability_0_2_cosine_similarity = cosine_similarity(
        [reduced_embeddings[0]], [reduced_embeddings[2]]
    )[0][0]
    capability_0_1_cosine_similarity = cosine_similarity(
        [reduced_embeddings[0]], [reduced_embeddings[1]]
    )[0][0]
    # The value of capability_0_1_cosine_similarity ia 0.999399141746954
    # The value of capability_0_2_cosine_similarity is 0.9849190137857736
    assert capability_0_1_cosine_similarity > capability_0_2_cosine_similarity


def test_embedding_visualization():
    """
    Test the visualization of the embeddings.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_dir = os.path.join(test_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    plot_name = "embeddings_visualization.pdf"
    embedding_generator.visualize_embeddings(
        embeddings, save_dir=visualization_dir, plot_name=plot_name
    )
    # Check that the visualization was successful.
    plot_dir = os.path.join(visualization_dir, plot_name)
    print(plot_dir)
    assert os.path.exists(plot_dir)
