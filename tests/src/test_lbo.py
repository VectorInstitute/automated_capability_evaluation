"""Unit tests for the LBO module."""

import gpytorch
import pytest
import torch

from src.lbo import LBO  # Import the LBO class


@pytest.fixture
def test_data():
    """Fixture that provides synthetic test data."""
    num_train, input_dim, num_candidates = 10, 2, 5
    torch.manual_seed(42)

    x_train = torch.rand(num_train, input_dim) * 4 - 2
    y_train = torch.sin(3 * x_train[:, 0]) * torch.cos(
        3 * x_train[:, 1]
    ) + 0.1 * torch.randn(num_train)
    x_candidates = torch.rand(num_candidates, input_dim) * 4 - 2

    lbo = LBO(
        x_train=x_train,
        y_train=y_train,
        acquisition_function="variance",
        num_gp_train_iterations=50,
    )

    return x_train, y_train, x_candidates, lbo


def test_initialization(test_data):
    """Test if LBO initializes correctly."""
    x_train, y_train, _, lbo = test_data
    assert lbo.x_train.shape == x_train.shape
    assert lbo.y_train.shape == y_train.shape
    assert lbo.input_dim == x_train.shape[1]
    assert isinstance(lbo.likelihood, gpytorch.likelihoods.GaussianLikelihood)


def compute_loss(lbo):
    """Compute GP loss for current model parameters."""
    lbo.model.train()
    lbo.likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lbo.likelihood, lbo.model)
    output = lbo.model(lbo.x_train)
    return -mll(output, lbo.y_train).item()


def test_gp_training(test_data):
    """Test if GP model training reduces loss."""
    _, _, _, lbo = test_data
    initial_loss = compute_loss(lbo)
    lbo.model = lbo._train_gp()  # Retrain GP
    final_loss = compute_loss(lbo)

    assert final_loss < initial_loss, "Loss should decrease after training."


def test_select_next_point(test_data):
    """Test that LBO selects a valid candidate using variance-based acquisition."""
    _, _, x_candidates, lbo = test_data
    idx, selected_x = lbo.select_next_point(x_candidates)
    assert 0 <= idx < x_candidates.shape[0], "Index should be within candidates."
    assert torch.any(torch.all(x_candidates == selected_x, dim=1)), (
        "Selected point must be from the candidate set."
    )


def test_update_model(test_data):
    """Test if the model updates correctly after adding new data."""
    _, _, _, lbo = test_data
    old_num_train = lbo.x_train.shape[0]
    query_x = torch.rand(lbo.input_dim) * 4 - 2  # A new random query point
    query_y = torch.tensor(
        [torch.sin(3 * query_x[0]) * torch.cos(3 * query_x[1]) + 0.1 * torch.randn(1)]
    )

    lbo.update(query_x, query_y)

    assert lbo.x_train.shape[0] == old_num_train + 1
    assert lbo.y_train.shape[0] == old_num_train + 1
    assert torch.allclose(lbo.x_train[-1], query_x)
    assert torch.allclose(lbo.y_train[-1], query_y)


def test_predict(test_data):
    """Test if the GP model produces reasonable predictions."""
    _, _, x_candidates, lbo = test_data
    mean, std = lbo.predict(x_candidates)

    assert mean.shape == (x_candidates.shape[0],)
    assert std.shape == (x_candidates.shape[0],)
    assert torch.all(std >= 0), "Standard deviation should be non-negative."
