"""
Tests for the load_data function from the data_utils module.

This module contains a test case for the load_data function, which is responsible for
loading a dataset based on the provided configuration.

Functions:
    test_load_data(): Tests the load_data function to ensure it loads the correct number
    of data samples.
"""

from src.utils import load_data


def test_load_data():
    """
    Test the load_data function to ensure it loads the dataset correctly.

    This test checks if the dataset loaded by the load_data function has the expected
    size.
    The data configuration includes the dataset name, split, subset, and size.

    The expected size is extracted from the data configuration and compared with the
    length of the dataset returned by the load_data function.

    Raises
    ------
        AssertionError: If the length of the dataset does not match the expected size.
    """
    data_cfg = {
        "dataset_name": "qwedsacf/competition_math",
        "split": "train",
        "subset": "",
        "num_repr_samples": 3,
        "streaming": False,
    }
    test_cfg = {
        "size": 12500,
    }

    dataset = load_data(**data_cfg)
    assert len(dataset) == test_cfg["size"]
