import json  # noqa: D100
import os
from typing import List

import pytest  # noqa: D100

from src.capability import Capability
from src.generate_capabilities import (
    generate_capability_heatmap,
    plot_hierarchical_capability_2d_embeddings,
)
from src.generate_embeddings import (
    EmbeddingGenerator,
    EmbeddingModelName,
    reduce_embeddings_dimensions,
)


# Check if the "TEST_OPENAI_API_KEY" is set for test purposes.
# If not set, use a dummy key and skip the tests.
DUMMY_OPENAI_API_KEY = "dummy_key"
os.environ["OPENAI_API_KEY"] = os.environ.get(
    "TEST_OPENAI_API_KEY", DUMMY_OPENAI_API_KEY
)
pytestmark = pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") == DUMMY_OPENAI_API_KEY,
    reason="Skipping all the tests due to missing TEST_OPENAI_API_KEY.",
)


EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBED_DIMENSIONS = 512
PERPLEXITY = 8
test_dir = os.path.dirname(os.path.abspath(__file__))
MANUAL_CAPABILITIES_PATH = os.path.join(test_dir, "resources/manual_capabilities.json")


@pytest.fixture(scope="module")
def mock_capabilities():
    """
    Create mock capabilities for testing from MANUAL_CAPABILITIES_PATH resource.

    This fixture loads the capabilities from a JSON file and generates
    embeddings for them using the EmbeddingGenerator class.
    It returns a list of mock capabilities with their embeddings set.
    Three types of embeddings are generated and set:
    - name_embedding: Embedding based on the capability name.
    - name_description_embedding: Embedding based on the capability name:description.
    - json_embedding: Embedding based on the JSON representation of the capability.
    These embeddings are then reduced using t-sne and visualized. This is
    used to validate the meaningfulness of t-sne reduced embeddings.

    """

    class MockCapability(Capability):
        def __init__(self, name, description, area):
            self.name = name
            self.description = description
            self.area = area
            self.embedding_dict = {}

    with open(MANUAL_CAPABILITIES_PATH, "r") as file:
        capabilities_data = json.load(file)

    capabilities = []
    for area, capability_group in capabilities_data.items():
        for name, description in capability_group.items():
            capabilities.append(MockCapability(name, description, area))

    embedding_generator = EmbeddingGenerator(
        model_name=EmbeddingModelName(EMBEDDING_MODEL_NAME),
        embed_dimensions=EMBED_DIMENSIONS,
    )

    # Generate embeddings for the capabilities based on their names.
    texts = [capability.name for capability in capabilities]
    name_embeddings = embedding_generator.generate_embeddings(texts=texts)
    for capability, embedding in zip(capabilities, name_embeddings):
        capability.set_embedding(
            embedding_name="name_embedding",
            embedding_tensor=embedding,
        )
    # Generate embeddings based on names and descriptions
    texts = [
        f"{capability.name}: {capability.description}" for capability in capabilities
    ]
    name_description_embeddings = embedding_generator.generate_embeddings(texts=texts)
    for capability, embedding in zip(capabilities, name_description_embeddings):
        capability.set_embedding(
            embedding_name="name_description_embedding",
            embedding_tensor=embedding,
        )

    # Generate embeddings based on JSON object (name and description).
    texts = [
        capability.to_json_str(attribute_names=["name", "description"])
        for capability in capabilities
    ]
    json_embeddings = embedding_generator.generate_embeddings(texts=texts)
    for capability, embedding in zip(capabilities, json_embeddings):
        capability.set_embedding(
            embedding_name="json_embedding",
            embedding_tensor=embedding,
        )

    return capabilities


def call_visualize(
    mock_capabilities: List[Capability],
    reduced_embedding_name: str,
    plot_name: str,
    show_point_ids: bool,
) -> None:
    """
    Call the visualization function and check if the plot is saved.

    This function checks if the plot is already saved in the specified
    directory. If not, it calls the visualization function to generate
    the plot and save it.

    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(test_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, f"{plot_name}.pdf")
    if os.path.isfile(plot_dir):
        assert True
    else:
        try:
            plot_hierarchical_capability_2d_embeddings(
                capabilities=mock_capabilities,
                dim_reduction_method=reduced_embedding_name,
                plot_name=plot_name,
                save_dir=save_dir,
                show_point_ids=show_point_ids,
            )
        except Exception as e:
            pytest.fail(f"Visualization failed with error: {e}")


def test_tsne_reduce_and_visualize_name_embeddings(
    mock_capabilities: List[Capability],
) -> None:
    """Apply dimensionality reduction to name embeddings and visualize them."""
    name_embeddings = [
        capability.get_embedding("name_embedding") for capability in mock_capabilities
    ]
    reduced_embeddings = reduce_embeddings_dimensions(
        embeddings=name_embeddings, output_dimensions=2, perplexity=PERPLEXITY
    )

    # Set reduced dimensions for each capability
    reduced_embedding_name = "name_embedding_tsne_reduced"
    for idx in range(len(mock_capabilities)):
        mock_capabilities[idx].set_embedding(
            embedding_name=reduced_embedding_name,
            embedding_tensor=reduced_embeddings[idx],
        )

    call_visualize(
        mock_capabilities=mock_capabilities,
        reduced_embedding_name=reduced_embedding_name,
        plot_name="name_embedding_plot",
        show_point_ids=False,
    )


def test_tsne_reduce_and_visualize_name_description_embeddings(
    mock_capabilities: List[Capability],
) -> None:
    """Reduce and visualize name_description embeddings."""
    name_description_embeddings = [
        capability.get_embedding("name_description_embedding")
        for capability in mock_capabilities
    ]
    reduced_embeddings = reduce_embeddings_dimensions(
        embeddings=name_description_embeddings,
        output_dimensions=2,
        perplexity=PERPLEXITY,
    )
    # Set reduced dimensions for each capability
    reduced_embedding_name = "name_description_embedding_tsne_reduced"
    for idx in range(len(mock_capabilities)):
        mock_capabilities[idx].set_embedding(
            embedding_name=reduced_embedding_name,
            embedding_tensor=reduced_embeddings[idx],
        )

    call_visualize(
        mock_capabilities=mock_capabilities,
        reduced_embedding_name=reduced_embedding_name,
        plot_name="name_description_embedding_plot",
        show_point_ids=True,
    )


def test_tsne_reduce_and_visualize_json_embeddings(
    mock_capabilities: List[Capability],
) -> None:
    """Reduce and visualize JSON representation embeddings."""
    json_embeddings = [
        capability.get_embedding("json_embedding") for capability in mock_capabilities
    ]
    reduced_embeddings = reduce_embeddings_dimensions(
        embeddings=json_embeddings, output_dimensions=2, perplexity=PERPLEXITY
    )
    # Set reduced dimensions for each capability
    reduced_embedding_name = "json_embedding_tsne_reduced"
    for idx in range(len(mock_capabilities)):
        mock_capabilities[idx].set_embedding(
            embedding_name=reduced_embedding_name,
            embedding_tensor=reduced_embeddings[idx],
        )

    call_visualize(
        mock_capabilities=mock_capabilities,
        reduced_embedding_name=reduced_embedding_name,
        plot_name="json_embedding_plot",
        show_point_ids=False,
    )


def test_generate_capability_heatmap(
    mock_capabilities: List[Capability],
) -> None:
    """Visualize name_description openai embedding heatmap."""
    plot_name = "heatmap_plot"
    test_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(test_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, f"{plot_name}.pdf")
    if os.path.isfile(plot_dir):
        assert True
    else:
        try:
            generate_capability_heatmap(
                capabilities=mock_capabilities,
                embedding_model_name="name_description_embedding",
                plot_name=plot_name,
                save_dir=save_dir,
                add_squares=True,
            )
        except Exception as e:
            pytest.fail(f"Visualization failed with error: {e}")
