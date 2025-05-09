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


def test_get_lbo_train_set_num_train():
    """
    Test the `get_lbo_train_set` function with `num_train` parameter.

    This test verifies that the function correctly splits the input data into
    training and remaining datasets based on the specified number of training
    samples.
    """
    input_data = [f"area_{i}" for i in range(10)]
    num_train = 4
    train_data, rem_data = get_lbo_train_set(input_data, num_train=num_train, seed=42)

    assert len(train_data) == num_train
    assert len(rem_data) == len(input_data) - num_train
    assert set(map(tuple, train_data)).isdisjoint(set(map(tuple, rem_data)))


def test_get_lbo_train_set_stratified_sampling():
    """
    Test the `get_lbo_train_set` function with stratified sampling.

    This test verifies that the function correctly performs stratified sampling
    when the `stratified` parameter is set to `True`.
    """
    input_data = [f"area_{i}" for i in range(10)]
    input_categories = ["A"] * 5 + ["B"] * 5
    train_frac = 0.6
    train_data, rem_data = get_lbo_train_set(
        input_data,
        train_frac=train_frac,
        stratified=True,
        input_categories=input_categories,
        seed=42,
    )

    assert len(train_data) == int(train_frac * len(input_data))
    assert len(rem_data) == len(input_data) - len(train_data)
    assert set(map(tuple, train_data)).isdisjoint(set(map(tuple, rem_data)))

    # Ensure stratification
    train_data_indices = [int(item.split("_")[1]) for item in train_data]
    rem_data_indices = [int(item.split("_")[1]) for item in rem_data]
    train_data_categories = [input_categories[i] for i in train_data_indices]
    rem_data_categories = [input_categories[i] for i in rem_data_indices]
    assert train_data_categories.count("A") == 3
    assert train_data_categories.count("B") == 3
    assert rem_data_categories.count("A") == 2
    assert rem_data_categories.count("B") == 2


def test_get_lbo_train_set_num_train_less_than_categories():
    """
    Test the `get_lbo_train_set` function when `num_train` <= number of categories.

    This test verifies that the function raises a `ValueError` when the number
    of training samples is less than the number of unique categories in the input data.
    """
    input_data = [f"area_{i}" for i in range(3)]
    input_categories = ["A", "B", "C"]
    num_train = 2

    with pytest.raises(
        ValueError,
        match="Number of training samples .* cannot be less than the number of categories",
    ):
        get_lbo_train_set(
            input_data,
            num_train=num_train,
            stratified=True,
            input_categories=input_categories,
            seed=42,
        )


def test_get_lbo_train_set_no_train_frac_or_num_train():
    """
    Test the `get_lbo_train_set` function with neither `train_frac` nor `num_train`.

    This test verifies that the function raises a `ValueError` when both
    `train_frac` and `num_train` are `None`.
    """
    input_data = [f"area_{i}" for i in range(10)]

    with pytest.raises(
        ValueError, match="Either num_train or train_frac must be provided"
    ):
        get_lbo_train_set(input_data, seed=42)


def test_get_lbo_train_set_both_train_frac_and_num_train():
    """
    Test the `get_lbo_train_set` function with both `train_frac` and `num_train`.

    This test verifies that the function raises a `ValueError` when both
    `train_frac` and `num_train` are specified.
    """
    input_data = [f"area_{i}" for i in range(10)]
    train_frac = 0.5
    num_train = 4

    with pytest.raises(
        ValueError,
        match="Both num_train and train_frac are provided, train_frac will be ignored",
    ):
        get_lbo_train_set(
            input_data, train_frac=train_frac, num_train=num_train, seed=42
        )


def test_get_lbo_train_set_missing_input_categories():
    """
    Test `get_lbo_train_set` with `stratified=True` but missing `input_categories`.

    This test verifies that the function raises a `ValueError` when stratified
    sampling is enabled but `input_categories` is not provided.
    """
    input_data = [f"area_{i}" for i in range(10)]
    train_frac = 0.6

    with pytest.raises(
        ValueError,
        match="input_categories must be provided when stratified sampling is enabled",
    ):
        get_lbo_train_set(input_data, train_frac=train_frac, stratified=True, seed=42)
