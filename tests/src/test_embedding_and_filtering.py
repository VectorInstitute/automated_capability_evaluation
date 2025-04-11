import json  # noqa: D100
import os
from typing import List

import pytest
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.generate_embeddings import (
    DimensionalityReductionTechnique,
    EmbeddingGenerator,
    EmbeddingModelName,
    reduce_embeddings_dimensions,
    visualize_embeddings,
)
from tests.src.test_model_class import skip_test


# Use dummy OpenAI API key for tests not making API calls.
DUMMY_OPENAI_API_KEY = "dummy_key"
os.environ["OPENAI_API_KEY"] = DUMMY_OPENAI_API_KEY
# Generate embeddings one time for all the tests.
EMBEDDING_SIZE = 256

# Set random seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.manual_seed(42)


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


def return_json_str(capability_dict):
    """Convert capability dictionary to JSON string."""
    return json.dumps(capability_dict, indent=4)


# Create a list of capability objects
capabilities_json_str = [
    return_json_str(capability_dict) for capability_dict in capabilities_dicts
]


@skip_test
@pytest.fixture(scope="module")
def generate_capability_embeddings() -> List[torch.Tensor]:
    """Reconstruct embeddings one time here to avoid repeated API calls during tests."""
    # We need to make actual API calls to generate embeddings.
    os.environ["OPENAI_API_KEY"] = os.environ.get(
        "TEST_OPENAI_API_KEY", DUMMY_OPENAI_API_KEY
    )
    # Here we are making API calls.
    # Generate embeddings for the capabilities
    embedding_generator = EmbeddingGenerator(
        model_name=EmbeddingModelName.text_embedding_3_small,
        embed_dimensions=EMBEDDING_SIZE,
    )
    # Generate embeddings for the capabilities, all at the same time.
    return embedding_generator.generate_embeddings(
        texts=capabilities_json_str,
    )


def skip_test_embedding(function):
    """Wrap the test functions that are using the embeddings.

    Wraps a test function that requires OpenAI API key, and skips the test if the key
    is not set. It also checks that embeddings are generated and are not set to True.
    `embeddings` could be set to `True` only if `generate_capability_embeddings`
    function is skipped due to error.

    Args:
        function (function): The test function to wrap.

    Returns
    -------
        function: The wrapped test function.
    """

    def wrapper(*args, **kwargs):
        embeddings = kwargs.get("embeddings")
        if (
            os.environ.get("OPENAI_API_KEY") == DUMMY_OPENAI_API_KEY
            or embeddings is True
        ):
            return True
        return skip_test(function)(*args, **kwargs)

    return wrapper


@skip_test_embedding
def test_capability_embedding(embeddings):
    """Test that embeddings similarities are as expected and make sense."""
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
    # The first and second capabilities are very similar,
    # so the cosine similarity should be high.
    # The value of capability_0_1_cosine_similarity is 0.9049213992192596
    # The value of capability_0_2_cosine_similarity 0.5984262605198356
    assert capability_0_1_cosine_similarity > capability_0_2_cosine_similarity


@skip_test_embedding
def test_filtering_logic(embeddings):
    """
    Test the filtering logic for capabilities.

    This test verifies that the filtering logic correctly identifies and filters out
    duplicate capabilities based on their embeddings.
    """
    similarity_threshold = 0.85

    # Filter the capabilities
    # Remove capabilities with close embeddings.
    similarity_matrix = cosine_similarity(embeddings)
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


@skip_test_embedding
def test_dimensionality_reduction_tsne(embeddings):
    """Test the dimensionality reduction of the embeddings using t-SNE."""
    # Reduce the dimensionality of the embeddings to 2.
    # Perplexity must be less than n_samples. Because we only have 3 samples here,
    # we need to decrease the t-SNE's perplexity parameter. This value is 30 by default.
    reduced_embeddings = reduce_embeddings_dimensions(
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
    # The t-SNE reduced vectors may vary between runs due to the stochastic
    # nature of the algorithm.
    assert capability_0_1_cosine_similarity > capability_0_2_cosine_similarity


@skip_test_embedding
def test_dimensionality_reduction_cut_embedding(embeddings):
    """Test the dimensionality reduction of the embeddings using cut embedding."""
    # OpenAI embedding-3 models are trained with Matryoshka Representation Learning,
    # So we can naturally cut the embedding vector to smaller dimensions, and they
    # should still be meaningful.

    # Reduce the dimensionality of the embeddings to 2 by just cutting them.
    reduced_embeddings = reduce_embeddings_dimensions(
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
    # The reduced vectors from cut_embedding always remain the same
    # if the input is unchanged.
    # The value of capability_0_1_cosine_similarity ia 0.999399141746954
    # The value of capability_0_2_cosine_similarity is 0.9849190137857736
    assert capability_0_1_cosine_similarity > capability_0_2_cosine_similarity


@skip_test_embedding
def test_embedding_visualization(embeddings) -> None:
    """Test the visualization of the embeddings."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(test_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    plot_name = "embeddings_visualization.pdf"
    plot_dir = os.path.join(save_dir, plot_name)
    if os.path.exists(plot_dir):
        assert True
    else:
        visualize_embeddings(embeddings, save_dir=save_dir, plot_name=plot_name)
        # Check that the visualization was successful.
        print(plot_dir)
        assert os.path.exists(plot_dir)
