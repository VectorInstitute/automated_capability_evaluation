from datasets import load_dataset


def load_data(dataset_name: str, split: str):
    """
    Load a dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (default is 'train').

    Returns:
        Dataset: The loaded dataset.
    """
    # TODO: Add ability to load datasets from sources other than huggingface
    return load_dataset(dataset_name, split=split, streaming=True)
