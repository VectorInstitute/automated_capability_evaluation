"""
The data_utils module for the automatic_benchmark_generation project.

It contains utility functions for loading datasets.
"""

import json
import logging
import os
import shutil
from typing import Any, Dict

from datasets import (
    Dataset,
    load_dataset,  # noqa: D100
)
from google.cloud import storage
from omegaconf import DictConfig


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


def read_json_file(file_path: str) -> Any:
    """
    Read a JSON file from a GCP bucket or local file system based on the file path.

    Args:
        file_path (str): The path to the JSON file. If it starts with 'gs://',
            it is treated as a GCP bucket path.

    Returns
    -------
        Any: The contents of the JSON file as a dictionary.
    """
    if file_path.startswith("gs://"):
        # Read from GCP bucket
        client = storage.Client()
        bucket_name, blob_name = file_path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        json_data = blob.download_as_text()
        return json.loads(json_data)

    # Read from local file system
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(file_path: str, data: Dict[Any, Any]) -> None:
    """
    Write a dictionary to a JSON file.

    This function handles both GCP bucket paths and local file system paths.

    Args:
        file_path (str): The path to the JSON file. If it starts with 'gs://',
            it is treated as a GCP bucket path.
        data (Dict[Any, Any]): The dictionary to write to the JSON file.
    """
    if file_path.startswith("gs://"):
        # Write to GCP bucket
        client = storage.Client()
        bucket_name, blob_name = file_path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(data, indent=4))
    else:
        # Write to local file system
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


def list_dir(path: str) -> list[str]:
    """
    List the contents of a directory.

    This function handles both GCP bucket paths and local file system paths.

    Args:
        path (str): The path to the directory. If it starts with 'gs://',
            it is treated as a GCP bucket path.

    Returns
    -------
        list: A list of contents in the directory.
    """
    if path.startswith("gs://"):
        # List contents from GCP bucket
        client = storage.Client()
        bucket_name, prefix = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return list({elm.name[(len(prefix) + 1) :].split("/")[0] for elm in blobs})

    # List contents from local file system
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path is not a directory: {path}")
    return os.listdir(path)


def copy_file(src: str, dest: str) -> None:
    """
    Copy a file from source to destination.

    This function handles both GCP bucket paths and local file system paths.

    Args:
        src (str): The source file path. If it starts with 'gs://',
            it is treated as a GCP bucket path.
        dest (str): The destination file path. If it starts with 'gs://',
            it is treated as a GCP bucket path.
    """
    if src.startswith("gs://") and dest.startswith("gs://"):
        # Copy file within GCP buckets
        client = storage.Client()
        src_bucket_name, src_blob_name = src[5:].split("/", 1)
        dest_bucket_name, dest_blob_name = dest[5:].split("/", 1)

        src_bucket = client.bucket(src_bucket_name)
        src_blob = src_bucket.blob(src_blob_name)

        dest_bucket = client.bucket(dest_bucket_name)
        dest_blob = dest_bucket.blob(dest_blob_name)

        dest_blob.rewrite(src_blob)
    elif src.startswith("gs://"):
        # Copy file from GCP bucket to local
        client = storage.Client()
        bucket_name, blob_name = src[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
    elif dest.startswith("gs://"):
        # Copy file from local to GCP bucket
        client = storage.Client()
        bucket_name, blob_name = dest[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(src)
    else:
        # Copy file locally
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)


def path_exists(path: str) -> bool:
    """
    Check if a path exists.

    This function handles both GCP bucket paths and local file system paths.

    Args:
        path (str): The path to check. If it starts with 'gs://',
            it is treated as a GCP bucket path.

    Returns
    -------
        bool: True if the path exists, False otherwise.
    """
    if path.startswith("gs://"):
        # Check existence in GCP bucket
        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return bool(blob.exists())

    # Check existence in local file system
    return os.path.exists(path)


def transfer_inspect_log_to_gcp(src_dir: str, gcp_dir: str) -> None:
    """
    Transfer the inspect log file from local to GCP bucket.

    Args:
        src_dir (str): The source directory containing the file.
        gcp_dir (str): The destination GCP bucket directory.
    """
    src_files = os.listdir(src_dir)
    # Ensure the source directory contains only one file
    assert len(src_files) == 1, (
        f"Expected only one file in {src_dir}, but found {len(src_files)} files."
    )
    # Ensure the file has a .json extension
    src_file = src_files[0]
    if not src_file.endswith(".json"):
        raise ValueError(f"Expected a .json file, but got: {src_file}")

    # Transfer the file to GCP
    src_path = os.path.join(src_dir, src_file)
    dest_path = os.path.join(gcp_dir, src_file)
    copy_file(src_path, dest_path)


def check_cfg(cfg: DictConfig, logger: logging.Logger) -> None:
    """
    Check configuration compatibility.

    Args
    ----
        cfg (DictConfig): The provided configuration.
        logger (logging.Logger): The logger instance to log messages.
    """
    assert cfg.capabilities_cfg.num_gen_capabilities > 0
    assert cfg.capabilities_cfg.num_gen_capabilities_per_run > 0
    assert (
        cfg.capabilities_cfg.num_gen_capabilities
        >= cfg.capabilities_cfg.num_gen_capabilities_per_run
    ), (
        "The total number of capabilities to generate must be greater than or equal to the number of capabilities to generate per run."
    )
    rem_c = (
        cfg.capabilities_cfg.num_gen_capabilities
        % cfg.capabilities_cfg.num_gen_capabilities_per_run
    )
    additional_c = cfg.capabilities_cfg.num_gen_capabilities_per_run - rem_c
    if rem_c != 0:
        logger.warning(f"{additional_c} additional capabilities might be generated.")


def get_run_id(cfg: DictConfig) -> str:
    """
    Generate a unique run ID based on the configuration.

    Args
    ----
        cfg (DictConfig): The provided configuration.

    Returns
    -------
        str: The generated run ID.
    """
    if cfg.exp_cfg.exp_id:
        run_id = str(cfg.exp_cfg.exp_id)
    else:
        run_id = f"{cfg.scientist_llm.name}_C{cfg.capabilities_cfg.num_gen_capabilities}_R{cfg.capabilities_cfg.num_gen_capabilities_per_run}"
        if cfg.capabilities_cfg.method == "hierarchical":
            run_id += f"_A{cfg.capabilities_cfg.num_capability_areas}"
        run_id += f"_T{cfg.capabilities_cfg.num_gen_tasks_per_capability}"
    return run_id
