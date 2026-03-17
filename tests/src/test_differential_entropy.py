"""Tests for compute_differential_entropy (kNN-based differential entropy)."""

import numpy as np
import pytest

from src.utils import compute_differential_entropy


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_returns_float_and_finite():
    """Return value is a finite float."""
    rng = _rng(0)
    x = rng.normal(size=(300, 16))
    h = compute_differential_entropy(x, k=4)
    assert isinstance(h, float)
    assert np.isfinite(h)


def test_permutation_invariance():
    """Entropy is invariant to row permutation."""
    rng = _rng(1)
    x = rng.normal(size=(250, 8))
    h1 = compute_differential_entropy(x, k=4)

    x_perm = x[rng.permutation(x.shape[0])]
    h2 = compute_differential_entropy(x_perm, k=4)

    assert np.isfinite(h1) and np.isfinite(h2)
    assert abs(h1 - h2) < 1e-10


def test_translation_invariance():
    """Entropy is translation-invariant; kNN estimators should be too."""
    rng = _rng(2)
    x = rng.normal(size=(400, 10))
    shift = rng.normal(size=(1, 10)) * 100.0

    h1 = compute_differential_entropy(x, k=4)
    h2 = compute_differential_entropy(x + shift, k=4)

    assert np.isfinite(h1) and np.isfinite(h2)
    assert abs(h1 - h2) < 1e-6


def test_scaling_increases_entropy():
    """
    Scaling embeddings by a>1 should increase entropy by about d*log(a).

    We don't require exact equality, just the direction and rough magnitude.
    """
    rng = _rng(3)
    n, d = 1200, 6
    x = rng.normal(size=(n, d))
    a = 3.0

    h1 = compute_differential_entropy(x, k=4)
    h2 = compute_differential_entropy(x * a, k=4)

    assert np.isfinite(h1) and np.isfinite(h2)
    assert h2 > h1

    expected_shift = d * np.log(a)
    assert (h2 - h1) == pytest.approx(expected_shift, abs=0.5)


def test_more_spread_more_entropy():
    """Larger variance should yield higher differential entropy."""
    rng = _rng(4)
    x_small = rng.normal(size=(800, 12)) * 0.5
    x_large = rng.normal(size=(800, 12)) * 2.0

    h_small = compute_differential_entropy(x_small, k=4)
    h_large = compute_differential_entropy(x_large, k=4)

    assert np.isfinite(h_small) and np.isfinite(h_large)
    assert h_large > h_small


def test_k_affects_estimate_but_is_finite():
    """Different k values produce finite, usually different estimates."""
    rng = _rng(5)
    x = rng.normal(size=(600, 9))
    h4 = compute_differential_entropy(x, k=4)
    h8 = compute_differential_entropy(x, k=8)

    assert np.isfinite(h4) and np.isfinite(h8)
    assert abs(h4 - h8) > 1e-6


def test_rejects_non_2d_input():
    """Reject non-2D input (e.g. 3D array)."""
    rng = _rng(6)
    x = rng.normal(size=(100, 5, 1))
    with pytest.raises((ValueError, AssertionError)):
        compute_differential_entropy(x, k=4)


def test_rejects_empty_input():
    """Reject empty input array."""
    x = np.empty((0, 10))
    with pytest.raises((ValueError, AssertionError)):
        compute_differential_entropy(x, k=4)


def test_rejects_k_too_large():
    """Reject k larger than n_samples - 1."""
    rng = _rng(7)
    x = rng.normal(size=(10, 3))
    with pytest.raises((ValueError, AssertionError)):
        compute_differential_entropy(x, k=10)


def test_duplicate_points_does_not_nan():
    """
    Duplicate points can cause zero kNN distances -> log(0).

    Depending on your implementation, this might:
      - raise, or
      - return -inf, or
      - remain finite if distances are clipped.
    We only enforce: it should not be NaN (silent failure).
    """
    rng = _rng(8)
    base = rng.normal(size=(80, 4))
    x = np.vstack([base, base[:20]])

    try:
        h = compute_differential_entropy(x, k=4)
    except (ValueError, AssertionError):
        return

    assert not np.isnan(h)


def test_differential_entropy_1d():
    """
    Hand-computed Kozachenko–Leonenko (kNN) differential entropy test.

    We assume the implementation matches the formula:
        H = psi(n) - psi(k) + log(V_d) + d * mean(log(eps))
    where:
      - eps_i is the distance to the (k+1)-th nearest neighbor in X when using
        NearestNeighbors(n_neighbors=k+1) on X and then taking distances[:, k]
        (i.e., self-distance at index 0, first *other* neighbor at index 1 for k=1).
      - V_d is the volume of the unit ball in R^d.

    Choose a tiny 1D dataset with uniform spacing:
        x = [0, 1, 2], n=3, d=1, k=1

    Step 1) kNN distances (k=1):
      For each point, the nearest *other* neighbor is at distance 1:
        eps = [1, 1, 1]
      Therefore:
        mean(log(eps)) = mean(log(1)) = 0

    Step 2) Unit ball volume term in 1D:
      The "unit ball" in 1D is the interval [-1, 1], so:
        V_1 = 2
        log(V_1) = log(2)

    Step 3) Digamma simplification for integers:
      For integer n:
        psi(n) = -gamma + H_{n-1}
      where H_{m} is the m-th harmonic number.

      psi(3) = -gamma + (1 + 1/2) = -gamma + 3/2
      psi(1) = -gamma
      So:
        psi(3) - psi(1) = 3/2

    Step 4) Combine terms:
      H = (psi(3) - psi(1)) + log(2) + 1 * 0
        = 3/2 + log(2)
        ≈ 1.5 + 0.6931471805599453
        = 2.1931471805599454
    """
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    expected = 2.1931471805599454  # 1.5 + ln(2)

    actual = compute_differential_entropy(x, k=1)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
