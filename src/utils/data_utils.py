"""
The data_utils module for the automatic_benchmark_generation project.

It contains utility functions for loading datasets.
"""

from typing import Any, Dict

from datasets import (
    Dataset,
    load_dataset,  # noqa: D100
)


def load_data(
    dataset_name: str,
    split: str,
    subset: str = "",
    streaming: bool = True,
    **kwargs: Dict[str, Any],
) -> Dataset:
    """
    Load a dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (default is 'train').

    Returns
    -------
        Dataset: The loaded dataset.
    """
    # TODO: Add ability to load datasets from sources other than huggingface
    return load_dataset(
        path=dataset_name, split=split, name=subset, streaming=streaming
    )
