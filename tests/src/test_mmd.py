"""Tests for the Maximum Mean Discrepancy (MMD) utility."""

import numpy as np
import pytest

from src.utils import compute_mmd


def test_mmd_linear_hand_computed():
    """
    Hand-computed MMD test for the linear kernel.

    For k(a,b) = a^T b, we have:
        MMD^2 = || mean(x) - mean(y) ||^2

    Choose 1D samples:
        x = [0, 2]  -> mean(x) = (0 + 2)/2 = 1
        y = [1, 3]  -> mean(y) = (1 + 3)/2 = 2

    Difference in means:
        mean(x) - mean(y) = 1 - 2 = -1

    Therefore:
        MMD^2 = (-1)^2 = 1

    The implementation returns MMD^2 (mean of kernel Gram matrices formula),
    so expected = 1.0.
    """
    x = np.array([[0.0], [2.0]], dtype=np.float64)
    y = np.array([[1.0], [3.0]], dtype=np.float64)

    expected = 1.0

    actual = compute_mmd(x, y, kernel="linear")
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
