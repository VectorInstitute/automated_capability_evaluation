"""Run IRT analysis on evaluation results.

Loads all **/*.json under data_cfg.scores_dir, extracts (model, question) responses,
builds a response matrix, and fits 1PL/2PL/3PL IRT (girth). data_cfg.per_capability
controls whether to fit one combined model (false) or one model per capability (true).
Configuration: src/cfg/irt_cfg.yaml. Override from CLI, e.g.:
  python src/run_irt_analysis.py data_cfg.scores_dir=/path/to/scores
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import hydra
from omegaconf import DictConfig

from src.schemas.irt_schemas import IRTAnalysis, IRTItemParameters
from src.utils import constants
from src.utils.data_utils import list_dir, write_json_file
from src.utils.irt_utils import (
    build_response_matrix_from_inspect_score_files,
    create_response_matrix_from_flat,
    discover_tasks_files,
    extract_question_responses,
    fit_3pl_irt,
    group_response_data_by_capability,
    load_score_files,
    update_tasks_with_irt_and_save,
)

logger = logging.getLogger(__name__)


def discover_model_score_files(
    scores_dir: str,
    domain: str,
    capability_name: str,
    model_names: List[str] | None = None,
) -> List[Tuple[str, str]]:
    """
    Discover (model_name, score_json_path) for a capability.

    Expects layout: scores_dir / model_name / domain / capability_name / *.json

    Args
    ----
        scores_dir: Base directory containing model subdirs.
        domain: Capability domain (e.g. "mathematics").
        capability_name: Capability name (e.g. "number_theory_combinatorics").
        model_names: If provided, only these models are used; otherwise
            all subdirs of scores_dir are treated as model names.

    Returns
    -------
        List of (model_name, json_path). Only includes models that have
        at least one .json file in their capability folder.
    """
    if model_names is None:
        try:
            model_names = list_dir(scores_dir)
        except (FileNotFoundError, NotADirectoryError) as e:
            logger.error("Cannot list scores_dir %s: %s", scores_dir, e)
            return []

    result: List[Tuple[str, str]] = []
    for model_name in model_names:
        cap_dir = os.path.join(
            scores_dir, model_name, domain, capability_name
        )
        if not os.path.isdir(cap_dir):
            logger.debug(
                "Skipping model %s: no directory %s",
                model_name,
                cap_dir,
            )
            continue
        try:
            files = [
                f
                for f in list_dir(cap_dir)
                if f.endswith(".json")
            ]
        except Exception as e:
            logger.warning(
                "Skipping model %s: failed to list %s: %s",
                model_name,
                cap_dir,
                e,
            )
            continue
        if not files:
            logger.debug(
                "Skipping model %s: no .json in %s",
                model_name,
                cap_dir,
            )
            continue
        # Use first JSON file (e.g. single eval run per model)
        json_path = os.path.join(cap_dir, files[0])
        result.append((model_name, json_path))
    return result


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


def run_irt_analysis_flat(
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
    Run IRT pipeline using flat loading: load all **/*.json under scores_dir,
    extract (model, question) responses (custom_scorer C), build matrix, fit IRT.

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


def run_irt_analysis(
    scores_dir: str,
    domain: str,
    capability_name: str,
    model_names: List[str] | None = None,
    model_type: str = "3PL",
    max_iterations: int = 2000,
    quadrature_n: int = 41,
    tolerance: float = 1e-6,
    output_dir: str | None = None,
    output_filename: str = "irt_analysis.json",
    dataset_id: str | None = None,
) -> IRTAnalysis:
    """
    Run full IRT pipeline (per-capability): discover score files by domain/capability,
    build matrix, fit IRT, return IRTAnalysis.

    Args
    ----
        scores_dir: Base directory for evaluation scores (layout: scores_dir/model/domain/capability/*.json).
        domain: Capability domain.
        capability_name: Capability name.
        model_names: Optional list of model names; if None, discovered from scores_dir.
        model_type: "1PL", "2PL", or "3PL".
        max_iterations: MML max iterations.
        quadrature_n: Quadrature points.
        tolerance: Convergence tolerance (stored in evaluation_settings).
        output_dir: If set, IRTAnalysis is saved here as output_filename.
        output_filename: Output JSON filename.
        dataset_id: Identifier for the dataset (default: capability_name).

    Returns
    -------
        IRTAnalysis instance with item parameters and context.
    """
    model_score_files = discover_model_score_files(
        scores_dir=scores_dir,
        domain=domain,
        capability_name=capability_name,
        model_names=model_names,
    )
    if not model_score_files:
        raise FileNotFoundError(
            f"No score files found for capability {capability_name} "
            f"(domain={domain}) under {scores_dir}. "
            "Ensure evaluation has been run for at least one model."
        )

    response_matrix, question_ids, names = (
        build_response_matrix_from_inspect_score_files(model_score_files)
    )
    if not response_matrix or not question_ids or not names:
        raise ValueError(
            f"Response matrix is empty for {capability_name}. "
            "Check that all model score files share at least one task id."
        )

    analysis = _fit_and_build(
        response_matrix=response_matrix,
        question_ids=question_ids,
        model_names=names,
        dataset_id=dataset_id or capability_name,
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
    return analysis


def _resolve_output_dir(
    cfg: DictConfig,
    default_dataset_id: str,
) -> str:
    """Resolve output_dir from output_cfg.output_dir or output_dir; default to irt_results/<dataset_id>_<timestamp>."""
    output_cfg = cfg.get("output_cfg") or {}
    out = output_cfg.get("output_dir") if output_cfg else None
    if out is None:
        out = cfg.get("output_dir")
    if out is None or (isinstance(out, str) and out.lower() == "null"):
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        # Use local project directory instead of shared artifacts directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(
            project_root,
            "irt_results",
            f"{default_dataset_id}_{timestamp}",
        )
    return str(out)


def _resolve_output_filename(cfg: DictConfig) -> str:
    """Resolve output filename from output_cfg.output_filename or default irt_analysis.json."""
    output_cfg = cfg.get("output_cfg") or {}
    name = output_cfg.get("output_filename") if output_cfg else None
    if name is None or (isinstance(name, str) and name.strip() == ""):
        return "irt_analysis.json"
    return str(name).strip()


def _resolve_irt_params(cfg: DictConfig) -> tuple[str, int, float, int]:
    """Return (model_type, max_iterations, tolerance, quadrature_n) from irt_cfg with top-level fallback."""
    irt = cfg.get("irt_cfg") or {}
    model_type = str(irt.get("model_type") or cfg.get("model_type", "3PL"))
    max_iterations = int(irt.get("max_iterations") or cfg.get("max_iterations", 2000))
    tolerance = float(irt.get("tolerance", 1e-6))
    quadrature_n = int(irt.get("quadrature_n") or cfg.get("quadrature_n", 41))
    return model_type, max_iterations, tolerance, quadrature_n


@hydra.main(version_base=None, config_path="cfg", config_name="irt_cfg")
def main(cfg: DictConfig) -> None:
    """Run IRT analysis using config from cfg/irt_cfg.yaml (overridable via CLI)."""
    logging.basicConfig(
        level=logging.DEBUG if cfg.get("verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model_type, max_iterations, tolerance, quadrature_n = _resolve_irt_params(cfg)
    output_filename = _resolve_output_filename(cfg)

    data_cfg = cfg.get("data_cfg") or {}
    scores_dir = data_cfg.get("scores_dir") if data_cfg else None
    if not scores_dir or str(scores_dir).strip().lower() in ("null", ""):
        raise ValueError(
            "data_cfg.scores_dir is required. Set it in irt_cfg.yaml or override from CLI, e.g. "
            "python src/run_irt_analysis.py data_cfg.scores_dir=/path/to/scores"
        )
    scores_dir = str(scores_dir).strip()
    dataset_id = cfg.get("dataset_id")
    dataset_id = str(dataset_id) if dataset_id is not None else "flat"
    output_dir = _resolve_output_dir(cfg, dataset_id)
    per_capability = bool(data_cfg.get("per_capability", False))
    capabilities_dir_val = data_cfg.get("capabilities_dir")
    capabilities_dir = (
        str(capabilities_dir_val).strip()
        if capabilities_dir_val
        and str(capabilities_dir_val).strip().lower() not in ("null", "")
        else None
    )

    run_irt_analysis_flat(
        scores_dir=scores_dir,
        model_type=model_type,
        max_iterations=max_iterations,
        quadrature_n=quadrature_n,
        tolerance=tolerance,
        output_dir=output_dir,
        output_filename=output_filename,
        dataset_id=dataset_id if dataset_id != "flat" else None,
        per_capability=per_capability,
        capabilities_dir=capabilities_dir,
    )
    logger.info(
        "IRT analysis completed (per_capability=%s).",
        per_capability,
    )


if __name__ == "__main__":
    main()
