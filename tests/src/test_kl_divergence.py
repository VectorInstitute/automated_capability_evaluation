"""Tests for the kNN-based KL divergence estimator compute_kl_divergence."""

import numpy as np
import pytest

from src.utils import compute_kl_divergence as kl_divergence


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_returns_python_float():
    """Return value is a finite Python float."""
    rng = _rng(0)
    p = rng.normal(size=(100, 16))
    q = rng.normal(size=(120, 16))
    val = kl_divergence(p, q, k=4, eps=1e-10)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_identical_distributions_near_zero():
    """
    If p and q are the same sample set, KL(P||Q) should be ~0.

    For kNN estimators it won't be exactly 0, so allow a tolerance.
    """
    rng = _rng(1)
    p = rng.normal(size=(300, 8))
    val = kl_divergence(p, p.copy(), k=4, eps=1e-10)
    assert np.isfinite(val)
    assert abs(val) < 0.5


def test_same_distribution_independent_samples_near_zero():
    """Two independent draws from the same distribution -> KL should be small."""
    rng = _rng(2)
    p = rng.normal(size=(400, 12))
    q = rng.normal(size=(450, 12))
    val = kl_divergence(p, q, k=4, eps=1e-10)
    assert np.isfinite(val)
    assert abs(val) < 0.3


def test_shifted_distribution_positive():
    """Shifted q should give KL(P||Q) > 0 (usually noticeably)."""
    rng = _rng(3)
    p = rng.normal(size=(500, 10))
    q = rng.normal(size=(500, 10)) + 1.5
    val = kl_divergence(p, q, k=4, eps=1e-10)
    assert np.isfinite(val)
    assert val > 0.2


def test_not_symmetric_in_general():
    """KL is not symmetric: KL(P||Q) != KL(Q||P) generally."""
    rng = _rng(4)
    p = rng.normal(size=(400, 6))
    q = rng.normal(size=(400, 6)) + 0.8
    pq = kl_divergence(p, q, k=4, eps=1e-10)
    qp = kl_divergence(q, p, k=4, eps=1e-10)
    assert np.isfinite(pq) and np.isfinite(qp)
    assert abs(pq - qp) > 1e-3


def test_permutation_invariance():
    """Reordering rows should not change the result."""
    rng = _rng(5)
    p = rng.normal(size=(300, 7))
    q = rng.normal(size=(320, 7))
    val1 = kl_divergence(p, q, k=4, eps=1e-10)

    p_perm = p[rng.permutation(p.shape[0])]
    q_perm = q[rng.permutation(q.shape[0])]
    val2 = kl_divergence(p_perm, q_perm, k=4, eps=1e-10)

    assert np.isfinite(val1) and np.isfinite(val2)
    assert abs(val1 - val2) < 1e-10


def test_translation_invariance_if_distance_based():
    """
    If the estimator uses only pairwise distances (kNN-style), adding the same.

    offset to both sets should not change KL.
    """
    rng = _rng(6)
    p = rng.normal(size=(350, 9))
    q = rng.normal(size=(360, 9))
    offset = rng.normal(size=(1, 9)) * 10.0

    val1 = kl_divergence(p, q, k=4, eps=1e-10)
    val2 = kl_divergence(p + offset, q + offset, k=4, eps=1e-10)

    assert np.isfinite(val1) and np.isfinite(val2)
    assert abs(val1 - val2) < 1e-6


def test_rejects_bad_shapes():
    """Reject inputs with incompatible shapes."""
    rng = _rng(7)
    p = rng.normal(size=(100, 8))
    q = rng.normal(size=(100, 9))
    with pytest.raises((ValueError, AssertionError)):
        kl_divergence(p, q, k=4, eps=1e-10)


def test_rejects_non_2d_inputs():
    """Reject non-2D input arrays."""
    rng = _rng(8)
    p = rng.normal(size=(100, 8, 1))
    q = rng.normal(size=(120, 8, 1))
    with pytest.raises((ValueError, AssertionError)):
        kl_divergence(p, q, k=4, eps=1e-10)


def test_k_too_large_raises():
    """Reject k that is too large for the sample size."""
    rng = _rng(9)
    p = rng.normal(size=(10, 4))
    q = rng.normal(size=(12, 4))
    with pytest.raises((ValueError, AssertionError)):
        kl_divergence(p, q, k=50, eps=1e-10)


def test_eps_prevents_nan_with_duplicates():
    """Eps should prevent NaN even when duplicates create zero distances."""
    rng = _rng(10)
    base = rng.normal(size=(50, 5))
    p = np.vstack([base, base[:10]])  # duplicates
    q = rng.normal(size=(70, 5))
    val = kl_divergence(p, q, k=4, eps=1e-10)
    assert np.isfinite(val)


def test_kl_value():
    """Manual KL computation for a simple 1D example.

    Manual KL computation for:

    P = [[0], [1], [3]]
    Q = [[10], [11], [13]]
    k = 1, d = 1

    rho = [1, 1, 2]
    nu  = [10, 9, 7]

    KL = (1/3) * (ln(10) + ln(9) + ln(3.5)) + ln(3/2)
       ≈ 2.322989
    """
    p = np.array([[0.0], [1.0], [3.0]])
    q = np.array([[10.0], [11.0], [13.0]])

    expected = 2.322989

    actual = kl_divergence(p, q, k=1, eps=1e-12)

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)
