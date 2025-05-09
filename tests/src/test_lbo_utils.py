"""
Unit tests for functions from the `src.utils.lbo_utils` module.

The `get_lbo_train_set` function is responsible for splitting input data into
training and remaining datasets based on a specified training fraction and
minimum training size. These tests ensure the function behaves as expected
under various scenarios, including valid inputs, edge cases, and error
conditions.

Test cases included:
- Valid input parameters.
- Insufficient input data.
- Violation of the minimum training size.
- Rounding of the training fraction.
- Empty input data.
- Zero training fraction.
"""

import pytest

from src.utils.lbo_utils import get_lbo_train_set


def test_get_lbo_train_set_valid_input():
    """
    Test the `get_lbo_train_set` function with valid input parameters.

    This test verifies that the function correctly splits the input data into
    training and remaining datasets based on the specified training fraction
    and minimum training size. It also ensures that the training and remaining
    datasets are disjoint.

    It checks the following:
    - The length of the training dataset matches the expected fraction of input data.
    - The length of the remaining dataset is the complement of the training dataset.
    - The training and remaining datasets do not share any common elements.
    """
    input_data = list(range(10))
    train_frac = 0.5
    min_train_size = 2
    train_data, rem_data = get_lbo_train_set(
        input_data, train_frac, min_train_size, seed=42
    )

    assert len(train_data) == int(train_frac * len(input_data))
    assert len(rem_data) == len(input_data) - len(train_data)
    assert set(train_data).isdisjoint(set(rem_data))


def test_get_lbo_train_set_insufficient_input_data():
    """
    Test the `get_lbo_train_set` function for handling insufficient input data.

    This test verifies that the function raises an `AssertionError` with the
    appropriate error message when the input data is insufficient to create a
    training set based on the specified `train_frac` and `min_train_size`.
    """
    input_data = list(range(5))
    train_frac = 0.5
    min_train_size = 1

    with pytest.raises(AssertionError, match="Insufficient input data"):
        get_lbo_train_set(input_data, train_frac, min_train_size, seed=42)


def test_get_lbo_train_set_min_train_size_violation():
    """
    Test the `get_lbo_train_set` function for violating the minimum training size.

    This test verifies that the function raises an `AssertionError` with the
    appropriate error message when the number of training data points is less
    than the recommended minimum training size.
    """
    input_data = list(range(20))
    train_frac = 0.5
    min_train_size = 15

    with pytest.raises(
        AssertionError,
    ):
        get_lbo_train_set(input_data, train_frac, min_train_size, seed=42)


def test_get_lbo_train_set_rounding_train_frac():
    """
    Test the `get_lbo_train_set` function for rounding the training fraction.

    This test verifies that the function correctly rounds the training fraction to two
    decimal places when the fraction is not a multiple of 0.01.

    It checks the following:
    - The length of the training dataset is rounded to 33% of the input data.
    - The length of the remaining dataset is the complement of the training dataset.
    - The training and remaining datasets do not share any common elements.
    """
    input_data = list(range(100))
    train_frac = 0.33333  # Should round to 0.33
    min_train_size = 20
    train_data, rem_data = get_lbo_train_set(
        input_data, train_frac, min_train_size, seed=42
    )

    assert len(train_data) == int(round(train_frac, 2) * len(input_data))
    assert len(rem_data) == len(input_data) - len(train_data)
    assert set(train_data).isdisjoint(rem_data)


def test_get_lbo_train_set_empty_input_data():
    """
    Test the `get_lbo_train_set` function with empty input data.

    This test verifies that the function raises an `AssertionError` with the expected
    error message "Insufficient input data" when provided with an empty input list.
    The test uses a training fraction of 0.5 and a minimum training size of 1.
    """
    input_data = []
    train_frac = 0.5
    min_train_size = 1

    with pytest.raises(AssertionError, match="Insufficient input data"):
        get_lbo_train_set(input_data, train_frac, min_train_size, seed=42)


def test_get_lbo_train_set_zero_train_frac():
    """
    Test the `get_lbo_train_set` function with a zero training fraction.

    This test verifies that the function raises an `AssertionError` when the
    number of training data points is less than the recommended minimum size.
    The input data consists of a range of 10 integers, and the training fraction
    is set to 0.0, which should result in no training data being selected. The
    minimum training size is set to 1.
    """
    input_data = list(range(10))
    train_frac = 0.0
    min_train_size = 1

    with pytest.raises(
        AssertionError,
    ):
        get_lbo_train_set(input_data, train_frac, min_train_size, seed=42)
