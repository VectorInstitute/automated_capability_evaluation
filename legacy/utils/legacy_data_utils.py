"""Legacy-only data utility helpers.

These utilities were moved out of active ``src/utils`` because they are
only used by archived legacy scripts.
"""

import json
import os
from typing import Any, Dict

from google.cloud import storage
from omegaconf import DictConfig


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
