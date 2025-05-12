import pytest  # noqa: D100
import torch

from src.capability import Capability
from src.generate_capabilities import apply_dimensionality_reduction


@pytest.fixture
def mock_capabilities():
    """Create mock capabilities with embeddings for testing."""

    class MockCapability(Capability):
        def __init__(self, name, embedding):
            self.name = name
            self.embedding = embedding
            self.embedding_dict = {}

    return [
        MockCapability("math", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        MockCapability("coding", torch.tensor([4.0, 3.0, 2.0, 1.0])),
        MockCapability("reasoning", torch.tensor([1.0, 1.0, 1.0, 1.0])),
        MockCapability("physics", torch.tensor([2.0, 1.0, 3.0, 1.0])),
    ]


def test_apply_dim_reduction_tsne(mock_capabilities):
    """Test the apply_dimensionality_reduction function with the T-SNE method."""
    dimensionality_reduction_method = "t-sne"
    output_dimensions = 2
    embedding_model_name = "text-embedding-3-small"

    for capability in mock_capabilities:
        capability.set_embedding(
            embedding_model_name,
            capability.embedding,
        )

    print(f"before dim reduction: {mock_capabilities[0].embedding}")
    # Call the function
    apply_dimensionality_reduction(
        capabilities=mock_capabilities,
        dim_reduction_method_name=dimensionality_reduction_method,
        output_dimension_size=output_dimensions,
        embedding_model_name="text-embedding-3-small",
        tsne_perplexity=2,
    )
    print(
        f"after dim reduction: {mock_capabilities[0].embedding_dict[dimensionality_reduction_method]}"
    )

    # Verify that the dim reduction output is set for each capability
    for capability in mock_capabilities:
        assert dimensionality_reduction_method in capability.embedding_dict, (
            f"Encoder output for {dimensionality_reduction_method} not set for capability {capability.name}."
        )
        reduced_embedding = capability.embedding_dict[dimensionality_reduction_method]
        assert isinstance(reduced_embedding, torch.Tensor), (
            f"Reduced embedding for {capability.name} is not a torch.Tensor."
        )
        assert reduced_embedding.shape[0] == output_dimensions, (
            f"Reduced embedding for {capability.name} does not have the correct dimensions."
        )


def test_apply_dim_reduction_pca(mock_capabilities):
    """Test the apply_dimensionality_reduction function for the PCA method."""
    dimensionality_reduction_method = "pca"
    output_dimensions = 2
    embedding_model_name = "text-embedding-3-small"

    for capability in mock_capabilities:
        capability.set_embedding(
            embedding_model_name,
            capability.embedding,
        )

    apply_dimensionality_reduction(
        capabilities=mock_capabilities,
        dim_reduction_method_name=dimensionality_reduction_method,
        output_dimension_size=output_dimensions,
        embedding_model_name="text-embedding-3-small",
    )

    # Verify that the dim reduction output is set for each capability
    for capability in mock_capabilities:
        assert dimensionality_reduction_method in capability.embedding_dict, (
            f"Encoder output for {dimensionality_reduction_method} not set for capability {capability.name}."
        )
        reduced_embedding = capability.get_embedding(dimensionality_reduction_method)
        assert isinstance(reduced_embedding, torch.Tensor), (
            f"Reduced embedding for {capability.name} is not a torch.Tensor."
        )
        assert reduced_embedding.shape[0] == output_dimensions, (
            f"Reduced embedding for {capability.name} does not have the correct dimensions."
        )


def test_apply_dim_reduction_cut_embeddings(mock_capabilities):
    """Test apply_dimensionality_reduction for the cut-embeddings method."""
    dimensionality_reduction_method = "cut-embeddings"
    output_dimensions = 2
    embedding_model_name = "text-embedding-3-small"

    for capability in mock_capabilities:
        capability.set_embedding(
            embedding_model_name,
            capability.embedding,
        )

    apply_dimensionality_reduction(
        capabilities=mock_capabilities,
        dim_reduction_method_name=dimensionality_reduction_method,
        output_dimension_size=output_dimensions,
        embedding_model_name="text-embedding-3-small",
    )

    # Verify that the dim reduction output is set for each capability
    for capability in mock_capabilities:
        assert dimensionality_reduction_method in capability.embedding_dict, (
            f"Encoder output for {dimensionality_reduction_method} not set for capability {capability.name}."
        )
        reduced_embedding = capability.get_embedding(dimensionality_reduction_method)
        assert isinstance(reduced_embedding, torch.Tensor), (
            f"Reduced embedding for {capability.name} is not a torch.Tensor."
        )
        assert reduced_embedding.shape[0] == output_dimensions, (
            f"Reduced embedding for {capability.name} does not have the correct dimensions."
        )

    embeddings = mock_capabilities[0].get_embedding(dimensionality_reduction_method)
    print(f"embeddings: {embeddings}")
    # The first two elements of capability 0 embedding are [1.0, 2.0].
    # The range is 4 - 1 = 3. So normalization of [1, 2] should yield [-1, -1/3].
    actual = torch.tensor([-1.0, -1.0 / 3])
    assert torch.isclose(actual, embeddings).all()
