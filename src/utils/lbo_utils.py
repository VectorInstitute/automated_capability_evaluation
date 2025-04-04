"""
The lbo_utils module for the automated_capability_evaluation project.

It contains utility functions for LBO pipeline.
"""

import random
from typing import Any, List, Tuple


def get_lbo_train_set(
    input_data: List[Any],
    train_frac: float,
    min_train_size: int,
    seed: int = 42,
) -> Tuple[List[Any], List[Any]]:
    """
    Create LBO train partition.

    Get the train set from the input data based on the train fraction.

    Args
    ----
        input_data (List[Any]): The input data.
        train_frac (float): The fraction of data to use for training.
        min_train_size (int): The minimum number of training data points.
        seed (int): The random seed for reproducibility.

    Returns
    -------
        Tuple[List[Any], List[Any]]: A tuple containing the train set
            and the remaining data.
    """
    random.seed(seed)

    # Limit fraction to 2 decimal places,
    # TODO: Revisit this and verify if this constraint is reasonable,
    # if yes, add a warning while validating the config
    train_frac = round(train_frac, 2)
    num_decimal_places = (
        len(str(train_frac).split(".")[1]) if "." in str(train_frac) else 0
    )
    min_input_data = 10**num_decimal_places
    assert len(input_data) >= min_input_data, (
        f"Insufficient input data: {len(input_data)}, "
        + f"based on the given train fraction: {train_frac}."
        + f"Need at least {min_input_data} data points."
    )

    # TODO: Improve the train set selection method
    num_train = int(len(input_data) * train_frac)
    assert num_train >= min_train_size, (
        f"Number of train data points are less than the recommended value: {min_train_size}."
    )
    train_data = random.sample(input_data, num_train)
    rem_data = list(set(input_data) - set(train_data))
    return (train_data, rem_data)
