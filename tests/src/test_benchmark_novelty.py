import pytest

from src.utils import compute_benchmark_novelty


def test_benchmark_novelty_zero_when_current_matches_a_prior_dataset():
    """
    Hand-computed benchmark novelty test.

    Novelty(D_c, D_prev, M) = 1 - SpearmanCorr(v_c, v_hat_c)

    Construct a case where v_c is exactly equal to one of the prior benchmark
    accuracy vectors. Then linear regression can predict perfectly:

        Let V_prev be a single column vector equal to v_c.
        A linear model v_hat = theta * v_prev + b can fit with theta=1, b=0,
        so v_hat = v_c exactly.

    Therefore:
        SpearmanCorr(v_c, v_hat) = 1  (identical values -> identical ranks)
        Novelty = 1 - 1 = 0
    """
    current = {"modelA": 0.90, "modelB": 0.70, "modelC": 0.80}
    prior1 = {"modelA": 0.90, "modelB": 0.70, "modelC": 0.80}  # exactly the same vector

    expected = 0.0
    actual = compute_benchmark_novelty(current, [prior1])

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_benchmark_novelty_nonzero_hand_computed_case():
    """
    Hand-computed novelty case with non-zero novelty.

    Models: A, B, C, D

    Prior vectors (as "accuracies", just numeric features):
      prior1 x1 = [-1, +1, -1, +1]
      prior2 x2 = [-1, -1, +1, +1]

    Current:
      v_c = [0.1, 0.2, 0.6, 0.3]

    Least-squares with intercept:
      b = mean(v_c) = 0.3
      theta1 = (x1^T v_c) / (x1^T x1) = (-0.2)/4 = -0.05
      theta2 = (x2^T v_c) / (x2^T x2) = (0.6)/4 = 0.15

    Predicted:
      v_hat = b + theta1*x1 + theta2*x2
            = [0.2, 0.1, 0.5, 0.4]

    Spearman ranks:
      rank(v_c)   = [1, 2, 4, 3]
      rank(v_hat) = [2, 1, 4, 3]
      sum d^2 = 2
      rho = 1 - 6*2/(4*(16-1)) = 0.8

    Novelty = 1 - rho = 0.2
    """
    current = {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.3}

    prior1 = {"A": -1.0, "B": 1.0, "C": -1.0, "D": 1.0}
    prior2 = {"A": -1.0, "B": -1.0, "C": 1.0, "D": 1.0}

    expected = 0.2
    actual = compute_benchmark_novelty(current, [prior1, prior2])

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
