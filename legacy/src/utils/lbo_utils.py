"""
The lbo_utils module for the automated_capability_evaluation project.

It contains utility functions for LBO pipeline.
"""

import random
from collections import defaultdict
from typing import List, Tuple

from src.capability import Capability


def get_lbo_train_set(
    input_data: List[Capability],
    train_frac: float | None = None,
    num_train: int | None = None,
    stratified: bool = False,
    input_categories: List[str] | None = None,
    seed: int = 42,
) -> Tuple[List[Capability], List[Capability]]:
    """
    Create LBO train partition.

    Get the train set from the input data based on the train fraction.

    Args
    ----
        input_data (List[Capability]): The input data to be split into train
            and remaining sets.
        train_frac (float, optional): The fraction of the input data to be used
            for training. If None, num_train must be provided. Defaults to None.
        num_train (int, optional): The number of training samples to be used.
            If None, train_frac must be provided. Defaults to None.
        stratified (bool, optional): Whether to perform stratified sampling
            based on categories. Defaults to False.
        input_categories (List[str], optional): The categories of the input
            data. Required if stratified is True. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns
    -------
        Tuple[List[Capability], List[Capability]]: A tuple containing the train set
            and the remaining data.
    """
    random.seed(seed)

    if num_train is None and train_frac is None:
        raise ValueError(
            "Either num_train or train_frac must be provided to create the train set."
        )
    if num_train is not None:
        if train_frac is not None:
            raise ValueError(
                "Both num_train and train_frac are provided, train_frac will be ignored."
            )
        train_frac = None
        if num_train <= 0:
            raise ValueError("num_train must be a positive integer.")

    if stratified:
        if input_categories is None:
            raise ValueError(
                "input_categories must be provided when stratified sampling is enabled."
            )
        # Group input data by categories
        category_to_items = defaultdict(list)
        for item, category in zip(input_data, input_categories):
            category_to_items[category].append(item)

        if num_train is not None and num_train < len(category_to_items):
            raise ValueError(
                f"Number of training samples ({num_train}) cannot be less than the number of categories ({len(category_to_items)})."
            )

        # Perform stratified sampling
        train_data = []
        for _, items in category_to_items.items():
            if train_frac is not None:
                num_category_train = int(len(items) * train_frac)
                if num_category_train == 0:
                    raise ValueError(
                        f"train_frac {train_frac} is too small for category with {len(items)} items."
                    )
            else:
                assert num_train is not None
                num_category_train = min(
                    len(items), num_train // len(category_to_items)
                )
            train_data.extend(random.sample(items, num_category_train))
    else:
        if train_frac is not None:
            num_train = int(len(input_data) * train_frac)
            if num_train == 0:
                raise ValueError(
                    f"train_frac {train_frac} is too small for input data with {len(input_data)} items."
                )
        assert num_train is not None
        train_data = random.sample(input_data, num_train)

    rem_data = list(set(input_data) - set(train_data))
    return (train_data, rem_data)
