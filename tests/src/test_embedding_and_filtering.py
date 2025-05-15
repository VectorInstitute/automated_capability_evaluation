"""Tests for the capability embedding and filtering."""

import json
import os
from typing import List

import pytest
import torch
from openai import AuthenticationError
from sklearn.metrics.pairwise import cosine_similarity

from src.dimensionality_reduction import DimensionalityReductionMethod
from src.generate_embeddings import (
    EmbeddingGenerator,
    EmbeddingModelName,
    filter_embeddings,
    visualize_embeddings,
)


# Use dummy OpenAI API key for tests not making API calls.
DUMMY_OPENAI_API_KEY = "dummy_key"
os.environ["OPENAI_API_KEY"] = DUMMY_OPENAI_API_KEY
# Generate embeddings one time for all the tests.
EMBEDDING_SIZE = 256

# Set random seed for reproducibility
random_seed = 42
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
torch.manual_seed(random_seed)


# API key error code vars
EC_401 = "Error code: 401"
EC_401_SKIP_MSG = (
    "Skip this test for code check because this test depends on actual API call."
)


def skip_embedding_generation(function):
    """
    Wrap a function to handle AuthenticationError exceptions.

    If an AuthenticationError with error code EC_401 is raised
    (Incorrect API key provided), it prints a skip message and
    does not re-raise the exception. For other AuthenticationError exceptions,
    it re-raises the exception.
    Args:
        function (callable): The test function to be wrapped.

    Returns
    -------
        callable: The wrapped function.
    """

    def wrapper():
        try:
            return function()
        except AuthenticationError as e:
            if EC_401 in str(e):
                print(EC_401_SKIP_MSG)
            else:
                raise e
            return None

    return wrapper


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


@skip_embedding_generation
def generate_embeddings():
    """Generate embeddings once to avoid repeated API calls during tests.

    If an AuthenticationError occurs due to OpenAI API key issues,
    `embeddings` will be set to None.
    """
    # We need to make actual API calls to generate embeddings
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


@pytest.fixture(scope="module")
def embeddings() -> List[torch.Tensor] | bool:
    """Set global embeddings to be used by all the tests here."""
    return generate_embeddings()


def skip_test_embedding(function):
    """Wrap test functions that depend on embeddings.

    This decorator ensures that tests requiring embeddings are skipped if embeddings
    are not available. Embeddings may be unavailable if the OpenAI API key is not set
    or if an error occurred during embedding generation.

    Args:
        function (function): The test function to be wrapped.

    Returns
    -------
        function: The wrapped test function that skips execution if embeddings are None.
    """

    def wrapper(embeddings, *args, **kwargs):
        # Check if embeddings are missing.
        if embeddings is None:
            pytest.skip("Skipping test due to missing embeddings.")
        else:
            return function(embeddings, *args, **kwargs)

    return wrapper


@skip_test_embedding
def test_capability_embedding(embeddings: List[torch.Tensor]):
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
def test_filtering_logic(embeddings: List[torch.Tensor]):
    """
    Unit test the filtering logic for capabilities.

    This test verifies that the filtering logic correctly identifies and filters out
    duplicate capabilities based on their embeddings.
    """
    similarity_threshold = 0.85

    # Filter the capabilities
    # Remove capabilities with close embeddings.
    remaining_indices = filter_embeddings(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold,
    )
    assert list(remaining_indices) == [1, 2]


@skip_test_embedding
def test_dimensionality_reduction_tsne(embeddings: List[torch.Tensor]):
    """Test the dimensionality reduction of the embeddings using t-SNE."""
    # Reduce the dimensionality of the embeddings to 2.
    # Perplexity must be less than n_samples. Because we only have 3 samples here,
    # we need to decrease the t-SNE's perplexity parameter. This value is 30 by default.
    dim_reduction = DimensionalityReductionMethod.from_name(
        "t-sne",
        2,
        random_seed=random_seed,
        normalize_output=True,
        tsne_perplexity=1,
    )
    # fit_transform() the dimensionality reduction module on the embeddings.
    reduced_embeddings = dim_reduction.fit_transform(embeddings)

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
    # nature of the algorithm. Also, three points are not enough to get a good
    # representation of the data. So the below assertion might sometimes fail.
    tolerance = 0.7  # Therefore, we add tolerance.
    assert (
        abs(capability_0_1_cosine_similarity)
        > abs(capability_0_2_cosine_similarity) - tolerance
    )


@skip_test_embedding
def test_dimensionality_reduction_cut_embedding(embeddings: List[torch.Tensor]):
    """Test the dimensionality reduction of the embeddings using cut embedding."""
    # OpenAI embedding-3 models are trained with Matryoshka Representation Learning,
    # So we can naturally cut the embedding vector to smaller dimensions, and they
    # should still be meaningful.

    # Reduce the dimensionality of the embeddings to 2 by just cutting them.
    dim_reduction = DimensionalityReductionMethod.from_name(
        "cut-embeddings",
        2,
        random_seed=random_seed,
        normalize_output=True,
    )
    # fit_transform() the dimensionality reduction module on the embeddings.
    reduced_embeddings = dim_reduction.fit_transform(embeddings)

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
def test_dimensionality_reduction_pca(embeddings: List[torch.Tensor]):
    """Test the dimensionality reduction of the embeddings using PCA."""
    # Reduce the dimensionality of the embeddings to 2.
    dim_reduction = DimensionalityReductionMethod.from_name(
        "pca",
        2,
        random_seed=random_seed,
        normalize_output=True,
    )
    # fit_transform() the dimensionality reduction module on the embeddings.
    reduced_embeddings = dim_reduction.fit_transform(embeddings)

    # Check that the reduced embeddings have the correct shape.
    assert len(reduced_embeddings) == len(embeddings)
    assert len(reduced_embeddings[0]) == 2

    # Try reducing the dimensionality of a new point.
    test_point = torch.randn(EMBEDDING_SIZE)
    reduced_test_point = dim_reduction.transform_new_points([test_point])
    assert len(reduced_test_point) == 1
    assert len(reduced_test_point[0]) == 2


@skip_test_embedding
def test_embedding_visualization(embeddings: List[torch.Tensor]) -> None:
    """Test the visualization of the embeddings."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(test_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    plot_name = "embeddings_visualization_names"
    plot_dir = os.path.join(save_dir, f"{plot_name}.pdf")
    if os.path.isfile(plot_dir):
        assert True
    else:
        names = [capability["name"] for capability in capabilities_dicts]
        try:
            visualize_embeddings(
                embeddings, save_dir=save_dir, plot_name=plot_name, point_names=names
            )
        except Exception as e:
            pytest.fail(f"Visualization failed with error: {e}")
