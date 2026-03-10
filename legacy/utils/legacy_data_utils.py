"""Legacy-only data utility helpers."""

import json
import logging
import os
import shutil
from typing import Any, Dict

from datasets import Dataset, load_dataset
from google.cloud import storage
from omegaconf import DictConfig


def load_data(
    dataset_name: str,
    split: str,
    subset: str = "",
    streaming: bool = True,
    **kwargs: Dict[str, Any],
) -> Dataset:
    """Load a dataset from Hugging Face Hub for legacy workflows."""
    return load_dataset(
        path=dataset_name, split=split, name=subset, streaming=streaming
    )


def read_json_file(file_path: str) -> Any:
    """Read JSON from local disk or a GCS path."""
    if file_path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = file_path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        json_data = blob.download_as_text()
        return json.loads(json_data)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(file_path: str, data: Dict[Any, Any]) -> None:
    """Write JSON to local disk or a GCS path."""
    if file_path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = file_path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(data, indent=4))
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def list_dir(path: str) -> list[str]:
    """List directory contents from local disk or a GCS path."""
    if path.startswith("gs://"):
        client = storage.Client()
        bucket_name, prefix = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return list({elm.name[(len(prefix) + 1) :].split("/")[0] for elm in blobs})

    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path is not a directory: {path}")
    return os.listdir(path)


def copy_file(src: str, dest: str) -> None:
    """Copy files between local paths and/or GCS paths."""
    if src.startswith("gs://") and dest.startswith("gs://"):
        client = storage.Client()
        src_bucket_name, src_blob_name = src[5:].split("/", 1)
        dest_bucket_name, dest_blob_name = dest[5:].split("/", 1)

        src_bucket = client.bucket(src_bucket_name)
        src_blob = src_bucket.blob(src_blob_name)

        dest_bucket = client.bucket(dest_bucket_name)
        dest_blob = dest_bucket.blob(dest_blob_name)

        dest_blob.rewrite(src_blob)
    elif src.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = src[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
    elif dest.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = dest[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(src)
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)


def path_exists(path: str) -> bool:
    """Check path existence for local paths and GCS paths."""
    if path.startswith("gs://"):
        client = storage.Client()
        bucket_name, blob_name = path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return bool(blob.exists())

    return os.path.exists(path)


def transfer_inspect_log_to_gcp(src_dir: str, gcp_dir: str) -> None:
    """Transfer the single Inspect JSON log from local dir to GCS."""
    src_files = os.listdir(src_dir)
    assert len(src_files) == 1, (
        f"Expected only one file in {src_dir}, but found {len(src_files)} files."
    )

    src_file = src_files[0]
    if not src_file.endswith(".json"):
        raise ValueError(f"Expected a .json file, but got: {src_file}")

    src_path = os.path.join(src_dir, src_file)
    dest_path = os.path.join(gcp_dir, src_file)
    copy_file(src_path, dest_path)


def get_run_id(cfg: DictConfig) -> str:
    """Generate the legacy run identifier from config."""
    if cfg.exp_cfg.exp_id:
        return str(cfg.exp_cfg.exp_id)

    run_id = (
        f"{cfg.scientist_llm.name}_"
        f"C{cfg.capabilities_cfg.num_capabilities}_"
        f"R{cfg.capabilities_cfg.num_gen_capabilities_per_run}"
    )
    if cfg.get("areas_cfg", {}).get("num_areas"):
        run_id += f"_A{cfg.areas_cfg.num_areas}"
    run_id += f"_T{cfg.capabilities_cfg.num_gen_tasks_per_capability}"
    return run_id


def check_cfg(cfg: DictConfig, logger: logging.Logger) -> None:
    """Check legacy configuration compatibility."""
    assert getattr(cfg, "exp_cfg", None) is not None, "exp_cfg must be set."
    assert getattr(cfg.exp_cfg, "exp_id", ""), "exp_id must be set in exp_cfg."
    assert getattr(cfg, "global_cfg", None) is not None, "global_cfg must be set."
    assert getattr(cfg.global_cfg, "output_dir", ""), (
        "global_cfg.output_dir must be set."
    )
    assert getattr(cfg.global_cfg, "domain", ""), "global_cfg.domain must be set."
    assert getattr(cfg.global_cfg, "pipeline_type", None) is not None, (
        "global_cfg.pipeline_type must be set."
    )
    assert cfg.capabilities_cfg.num_capabilities > 0
    assert cfg.capabilities_cfg.num_gen_capabilities_per_run > 0
    num_capabilities = int(
        cfg.capabilities_cfg.num_capabilities
        * (1 + cfg.capabilities_cfg.num_capabilities_buffer)
    )
    assert num_capabilities >= cfg.capabilities_cfg.num_gen_capabilities_per_run, (
        "The total number of capabilities to generate must be greater than or equal to the number of capabilities to generate per run."
    )
    rem_c = num_capabilities % cfg.capabilities_cfg.num_gen_capabilities_per_run
    additional_c = cfg.capabilities_cfg.num_gen_capabilities_per_run - rem_c
    if rem_c != 0:
        logger.warning(f"{additional_c} additional capabilities might be generated.")
