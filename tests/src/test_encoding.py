import pytest  # noqa: D100
import torch

from src.capability import Capability
from src.generate_capabilities import fit_and_set_encodings


@pytest.fixture
def mock_capabilities():
    """Create mock capabilities with embeddings for testing."""

    class MockCapability(Capability):
        def __init__(self, name, embedding):
            self.name = name
            self.embedding = embedding
            self.encoder_output_dict = {}

    return [
        MockCapability("math", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        MockCapability("coding", torch.tensor([4.0, 3.0, 2.0, 1.0])),
        MockCapability("reasoning", torch.tensor([1.0, 1.0, 1.0, 1.0])),
        MockCapability("physics", torch.tensor([2.0, 1.0, 3.0, 1.0])),
    ]


def test_fit_and_set_encodings_tsne(mock_capabilities):
    """Test the fit_and_set_encodings function with the T-SNE encoder."""
    encoder_model = "t-sne"
    output_dimensions = 2

    # Call the function
    fit_and_set_encodings(
        filtered_capabilities=mock_capabilities,
        encoder_model=encoder_model,
        output_dimensions=output_dimensions,
    )

    # Verify that the encoder output is set for each capability
    for capability in mock_capabilities:
        assert encoder_model in capability.encoder_output_dict, (
            f"Encoder output for {encoder_model} not set for capability {capability.name}."
        )
        reduced_embedding = capability.encoder_output_dict[encoder_model]
        assert isinstance(reduced_embedding, torch.Tensor), (
            f"Reduced embedding for {capability.name} is not a torch.Tensor."
        )
        assert reduced_embedding.shape[0] == output_dimensions, (
            f"Reduced embedding for {capability.name} does not have the correct dimensions."
        )


def test_fit_and_set_encodings_invalid_encoder(mock_capabilities):
    """Test the fit_and_set_encodings function with an invalid encoder model."""
    encoder_model = "invalid-encoder"
    output_dimensions = 2

    with pytest.raises(
        AssertionError, match="Currently, only the t-sne encoder is supported."
    ):
        fit_and_set_encodings(
            filtered_capabilities=mock_capabilities,
            encoder_model=encoder_model,
            output_dimensions=output_dimensions,
        )


def test_capability_encode_function(mock_capabilities):
    """Test the encode function of the Capability class."""
    encoder_model = "t-sne"
    output_dimensions = 2

    # Call the function
    fit_and_set_encodings(
        filtered_capabilities=mock_capabilities,
        encoder_model=encoder_model,
        output_dimensions=output_dimensions,
    )
    capability_0 = mock_capabilities[0]
    encoded_tensor = capability_0.encode(encoder_model_name=encoder_model)

    # Verify that the encoder output is set
    assert "t-sne" in capability_0.encoder_output_dict
    assert isinstance(capability_0.encoder_output_dict["t-sne"], torch.Tensor), (
        "Encoder output is not a torch.Tensor."
    )
    assert torch.isclose(
        encoded_tensor, capability_0.get_encoder_output(encoder_model)
    ).all()
