import numpy as np
import pytest

from src.utils import compute_pad


def test_pad():
    """
    Hand-computed PAD test.

    PAD = 2(1 - 2ε)

    Choose perfectly linearly separable 1D embeddings:

        Synthetic: [-10, -9, -8]
        Real:      [  8,  9, 10]

    These are separable by threshold at 0.

    Classification error ε = 0

    Therefore:
        PAD = 2(1 - 2*0) = 2

    The implementation uses a train/validation split, so the returned value
    may be slightly below 2; we assert it is close to 2 (high separability).
    """
    x_syn = np.array([[-10.0], [-9.0], [-8.0]])
    x_real = np.array([[8.0], [9.0], [10.0]])

    expected = 2.0

    actual = compute_pad(
        x_syn,
        x_real,
        classifier_name="LogisticRegression",
    )

    assert actual == pytest.approx(expected, rel=0.05, abs=0.1)
