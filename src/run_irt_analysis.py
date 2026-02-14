"""Run IRT analysis on evaluation scores.

Loads **/*.json from data_cfg.scores_dir, builds a response matrix, fits 1PL/2PL/3PL (girth).
Config: src/cfg/irt_cfg.yaml. Override from CLI, e.g. data_cfg.scores_dir=/path output_cfg.output_dir=/out.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import hydra
from omegaconf import DictConfig

from src.schemas.irt_schemas import IRTAnalysis, IRTItemParameters
from src.utils import constants
from src.utils.data_utils import write_json_file
from src.utils.irt_utils import (
    create_response_matrix_from_flat,
    discover_tasks_files,
    extract_question_responses,
    fit_3pl_irt,
    group_response_data_by_capability,
    load_score_files,
    update_tasks_with_irt_and_save,
)

logger = logging.getLogger(__name__)


def _fit_and_build(
    response_matrix: List[List[int]],
    question_ids: List[str],
    model_names: List[str],
    dataset_id: str,
    model_type: str,
    max_iterations: int,
    quadrature_n: int,
    tolerance: float,
) -> IRTAnalysis:
    """Fit IRT and build IRTAnalysis (no save)."""
    fit_result = fit_3pl_irt(
        response_matrix=response_matrix,
        question_ids=question_ids,
        model_names=model_names,
        max_iterations=max_iterations,
        quadrature_n=quadrature_n,
        model_type=model_type,
    )
    evaluation_settings: Dict[str, Any] = {
        "model_type": model_type,
        "max_iterations": max_iterations,
        "quadrature_n": quadrature_n,
        "tolerance": tolerance,
    }
    item_parameters = {
        qid: IRTItemParameters(
            task_id=qid,
            discrimination=p["discrimination"],
            difficulty=p["difficulty"],
            guessing=p["guessing"],
        )
        for qid, p in fit_result["item_parameters"].items()
    }
    return IRTAnalysis(
        dataset_id=dataset_id,
        subject_model_names=model_names,
        evaluation_settings=evaluation_settings,
        item_parameters=item_parameters,
        model_info=fit_result["model_info"],
    )


def run_irt_analysis(
    scores_dir: str,
    model_type: str = "3PL",
    max_iterations: int = 1000,
    quadrature_n: int = 41,
    tolerance: float = 1e-6,
    output_dir: str | None = None,
    output_filename: str = "irt_analysis.json",
    dataset_id: str | None = None,
    per_capability: bool = False,
    capabilities_dir: str | None = None,
) -> IRTAnalysis | Dict[str, IRTAnalysis]:
    """
    Run IRT pipeline: load all **/*.json under scores_dir, extract (model, question)
    responses (custom_scorer C), build matrix, fit IRT.

    Args
    ----
        scores_dir: Base directory containing evaluation JSON files (recursive).
        model_type, max_iterations, quadrature_n, tolerance: IRT fitting options.
        output_dir: If set, save IRTAnalysis here as output_filename.
        output_filename: Output JSON filename (default irt_analysis.json).
        dataset_id: Dataset identifier (default "flat").
        per_capability: If True, group by capability (eval.task), fit IRT separately
            per capability, and save one JSON with all capability analyses. If False,
            one combined response matrix and one IRT fit (current behavior).
        capabilities_dir: If set with output_dir, task JSONs under this dir (**/tasks.json)
            are updated with IRT item parameters and saved to output_dir/updated_tasks/.

    Returns
    -------
        IRTAnalysis when per_capability=False; Dict[capability_name, IRTAnalysis] when True.
    """
    data = load_score_files(scores_dir)
    if not data:
        raise FileNotFoundError(
            f"No JSON score files found under {scores_dir}. "
            "Ensure the directory exists and contains **/*.json."
        )
    response_data, question_info = extract_question_responses(data)
    if not response_data:
        raise ValueError(
            f"No (model, question) responses extracted from {scores_dir}. "
            "Check that score files contain 'samples' and 'eval.task'."
        )

    if per_capability:
        by_capability = group_response_data_by_capability(
            response_data, question_info
        )
        if not by_capability:
            raise ValueError(
                "No capabilities found after grouping. Check question_info has 'task'."
            )
        analyses: Dict[str, IRTAnalysis] = {}
        for cap_name, cap_response_data in by_capability.items():
            if not cap_response_data:
                continue
            response_matrix, question_ids, model_names = (
                create_response_matrix_from_flat(cap_response_data)
            )
            if not response_matrix or not question_ids or not model_names:
                logger.warning(
                    "Skipping capability %s: empty or insufficient matrix",
                    cap_name,
                )
                continue
            # IRT (girth) needs at least 2 items and 2 persons
            if len(question_ids) < 2 or len(model_names) < 2:
                logger.warning(
                    "Skipping capability %s: need at least 2 questions and 2 models (got %d questions, %d models)",
                    cap_name,
                    len(question_ids),
                    len(model_names),
                )
                continue
            try:
                analyses[cap_name] = _fit_and_build(
                    response_matrix=response_matrix,
                    question_ids=question_ids,
                    model_names=model_names,
                    dataset_id=cap_name,
                    model_type=model_type,
                    max_iterations=max_iterations,
                    quadrature_n=quadrature_n,
                    tolerance=tolerance,
                )
                logger.info("Fitted IRT for capability: %s", cap_name)
            except (ValueError, AttributeError, RuntimeError) as e:
                logger.warning(
                    "Skipping capability %s: IRT fit failed (%s)",
                    cap_name,
                    e,
                )
                continue
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, output_filename)
            payload: Dict[str, Any] = {
                "per_capability": True,
                "capabilities": {
                    name: a.to_dict() for name, a in analyses.items()
                },
            }
            write_json_file(out_path, payload)
            logger.info(
                "Per-capability IRT analyses saved to %s (%d capabilities)",
                out_path,
                len(analyses),
            )
        if capabilities_dir and output_dir:
            task_files = discover_tasks_files(capabilities_dir)
            for cap_name, analysis in analyses.items():
                if cap_name not in task_files:
                    continue
                params_plain = {
                    qid: {
                        "difficulty": p.difficulty,
                        "discrimination": p.discrimination,
                        "guessing": p.guessing,
                    }
                    for qid, p in analysis.item_parameters.items()
                }
                update_tasks_with_irt_and_save(
                    capability_name=cap_name,
                    item_parameters=params_plain,
                    task_files=task_files[cap_name],
                    capabilities_dir=capabilities_dir,
                    output_dir=output_dir,
                )
        return analyses

    # Combined: one response matrix, one IRT fit
    response_matrix, question_ids, model_names = create_response_matrix_from_flat(
        response_data
    )
    if not response_matrix or not question_ids or not model_names:
        raise ValueError("Response matrix is empty after flat extraction.")
    analysis = _fit_and_build(
        response_matrix=response_matrix,
        question_ids=question_ids,
        model_names=model_names,
        dataset_id=dataset_id or "flat",
        model_type=model_type,
        max_iterations=max_iterations,
        quadrature_n=quadrature_n,
        tolerance=tolerance,
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)
        write_json_file(out_path, analysis.to_dict())
        logger.info("IRT analysis saved to %s", out_path)
    if capabilities_dir and output_dir and question_info:
        task_files = discover_tasks_files(capabilities_dir)
        cap_to_params: Dict[str, Dict[str, Dict[str, float]]] = {}
        for qid, p in analysis.item_parameters.items():
            cap = question_info.get(qid, {}).get("task", "")
            if cap:
                cap_to_params.setdefault(cap, {})[qid] = {
                    "difficulty": p.difficulty,
                    "discrimination": p.discrimination,
                    "guessing": p.guessing,
                }
        for cap_name, params in cap_to_params.items():
            if cap_name not in task_files:
                continue
            update_tasks_with_irt_and_save(
                capability_name=cap_name,
                item_parameters=params,
                task_files=task_files[cap_name],
                capabilities_dir=capabilities_dir,
                output_dir=output_dir,
            )
    return analysis


def _require_config(cfg: DictConfig) -> None:
    """Validate required config; raise ValueError if any required key is missing or null."""
    missing: List[str] = []

    data_cfg = cfg.get("data_cfg") or {}
    if not data_cfg.get("scores_dir") or str(data_cfg.get("scores_dir", "")).strip().lower() in ("null", ""):
        missing.append("data_cfg.scores_dir")

    output_cfg = cfg.get("output_cfg") or {}
    if not output_cfg.get("output_dir") or str(output_cfg.get("output_dir", "")).strip().lower() == "null":
        missing.append("output_cfg.output_dir")
    if not output_cfg.get("output_filename") or str(output_cfg.get("output_filename", "")).strip() == "":
        missing.append("output_cfg.output_filename")

    irt = cfg.get("irt_cfg") or {}
    if not irt.get("model_type") or str(irt.get("model_type", "")).strip() == "":
        missing.append("irt_cfg.model_type")
    if irt.get("max_iterations") is None:
        missing.append("irt_cfg.max_iterations")
    if irt.get("tolerance") is None:
        missing.append("irt_cfg.tolerance")
    if irt.get("quadrature_n") is None:
        missing.append("irt_cfg.quadrature_n")

    if missing:
        raise ValueError(
            "Missing required config (set in irt_cfg.yaml or override from CLI): " + ", ".join(missing)
        )


@hydra.main(version_base=None, config_path="cfg", config_name="irt_cfg")
def main(cfg: DictConfig) -> None:
    """Run IRT analysis using config from cfg/irt_cfg.yaml (overridable via CLI)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    _require_config(cfg)

    data_cfg = cfg.get("data_cfg") or {}
    output_cfg = cfg.get("output_cfg") or {}
    irt = cfg.get("irt_cfg") or {}

    scores_dir = str(data_cfg["scores_dir"]).strip()
    output_dir = str(output_cfg["output_dir"]).strip()
    output_filename = str(output_cfg["output_filename"]).strip()
    model_type = str(irt["model_type"]).strip()
    max_iterations = int(irt["max_iterations"])
    tolerance = float(irt["tolerance"])
    quadrature_n = int(irt["quadrature_n"])

    dataset_id = data_cfg.get("dataset_id") or cfg.get("dataset_id")
    dataset_id = str(dataset_id).strip() if dataset_id is not None and str(dataset_id).strip().lower() not in ("null", "") else None

    per_capability = bool(data_cfg.get("per_capability", False))
    capabilities_dir_val = data_cfg.get("capabilities_dir")
    capabilities_dir = (
        str(capabilities_dir_val).strip()
        if capabilities_dir_val
        and str(capabilities_dir_val).strip().lower() not in ("null", "")
        else None
    )

    run_irt_analysis(
        scores_dir=scores_dir,
        model_type=model_type,
        max_iterations=max_iterations,
        quadrature_n=quadrature_n,
        tolerance=tolerance,
        output_dir=output_dir,
        output_filename=output_filename,
        dataset_id=dataset_id,
        per_capability=per_capability,
        capabilities_dir=capabilities_dir,
    )
    logger.info(
        "IRT analysis completed (per_capability=%s).",
        per_capability,
    )


if __name__ == "__main__":
    main()
