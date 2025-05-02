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

    # Call the function
    apply_dimensionality_reduction(
        capabilities=mock_capabilities,
        dim_reduction_method_name=dimensionality_reduction_method,
        output_dimension_size=output_dimensions,
        embedding_model_name="text-embedding-3-small",
        tsne_perplexity=2,
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

    capability_0 = mock_capabilities[0]
    encoded_tensor = capability_0.get_embedding(dimensionality_reduction_method)

    assert torch.isclose(
        encoded_tensor, capability_0.get_embedding(dimensionality_reduction_method)
    ).all()


def test_apply_dim_reduction_pca(mock_capabilities):
    """Test the apply_dimensionality_reduction function For the PCA method."""
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
