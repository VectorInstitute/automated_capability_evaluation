"""IRT analysis utilities using 3PL model via girth library.

Builds response matrices from inspect score JSON files and fits 1PL/2PL/3PL
IRT models to estimate item parameters (difficulty, discrimination, guessing).

Supports two loading paths:
- Per-capability: discover model/domain/capability score files, then build matrix.
- Flat/glob: load all JSONs under a scores directory, extract responses (custom_scorer C),
  then build matrix for IRT.
"""

import glob
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from src.utils import constants
from src.utils.capability_utils import CAPABILITY_SCORER_MAP
from src.utils.data_utils import read_json_file

logger = logging.getLogger(__name__)

try:
    import girth
    from girth import ability_3pl_eap, rasch_mml, threepl_mml, twopl_mml

    GIRTH_AVAILABLE = True
except ImportError:
    GIRTH_AVAILABLE = False
    logger.warning(
        "girth not available. IRT fitting will fail. Install via: pip install girth"
    )

# Inspect AI uses "C" for correct, "I" for incorrect (CORRECT/INCORRECT from inspect_ai.scorer)
INSPECT_SCORE_CORRECT = "C"


def load_score_files(scores_dir: str) -> List[Dict]:
    """Load all JSON score files from the scores directory (recursive **/*.json).

    Args
    ----
        scores_dir: Path to the scores directory.

    Returns
    -------
        List of loaded JSON dicts; each entry has '_file_path' added.
    """
    if not os.path.exists(scores_dir):
        logger.error("Directory does not exist: %s", scores_dir)
        return []

    pattern = os.path.join(scores_dir, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    logger.info("Found %d JSON score files under %s", len(files), scores_dir)

    data: List[Dict] = []
    errors = 0
    for i, file_path in enumerate(files):
        try:
            with open(file_path, encoding="utf-8") as f:
                file_data = json.load(f)
            file_data["_file_path"] = file_path
            data.append(file_data)
            if (i + 1) % 50 == 0:
                logger.debug("Loaded %d/%d files...", i + 1, len(files))
        except Exception as e:
            errors += 1
            logger.warning("Error loading %s: %s", file_path, e)

    if errors > 0:
        logger.warning("%d files failed to load", errors)
    logger.info("Successfully loaded %d files", len(data))
    return data


def extract_model_name_from_path(file_path: str) -> str:
    """Extract model name from file path (segment after directory containing 'scores').

    Args
    ----
        file_path: Full path to the score file.

    Returns
    -------
        Model name or empty string.
    """
    parts = file_path.replace(os.sep, "/").split("/")
    # Look for any directory containing "scores" (e.g., "scores", "scores_sample")
    for idx, part in enumerate(parts):
        if "scores" in part.lower():
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def extract_question_responses(
    data: List[Dict],
) -> Tuple[Dict[Tuple[str, str], int], Dict[str, Any]]:
    """Extract (model, question_id) -> score from score file data.

    Uses eval.task and sample id; score is 1 if scorer value is 'C', else 0.
    Uses capability-specific scorer name (from CAPABILITY_SCORER_MAP) or default.
    Unique question id is task_name + question_id.

    Args
    ----
        data: List of score file dicts (with '_file_path').

    Returns
    -------
        (response_data, question_info):
        - response_data: (model_name, unique_question_id) -> 0 or 1
        - question_info: unique_question_id -> {task, question_id, input, target}
    """
    response_data: Dict[Tuple[str, str], int] = {}
    question_info: Dict[str, Dict] = {}
    files_processed = 0
    samples_processed = 0

    for file_idx, file_data in enumerate(data):
        if "samples" not in file_data:
            continue

        file_path = file_data.get("_file_path", "")
        model_name = extract_model_name_from_path(file_path)
        if not model_name:
            logger.debug(
                "Skipping file %s: could not extract model name from path",
                file_path,
            )
            continue

        eval_data = file_data.get("eval", {})
        task_name = eval_data.get("task", "unknown")
        capability_name = _clean_task_name(task_name)
        scorer_name = CAPABILITY_SCORER_MAP.get(
            capability_name, constants.DEFAULT_INSPECT_SCORER_NAME
        )
        files_processed += 1

        for sample in file_data["samples"]:
            samples_processed += 1
            question_id = sample.get("id", "")
            if not question_id:
                continue

            unique_question_id = f"{task_name}_{question_id}"

            score_value = 0
            scores = sample.get("scores", {})
            if scorer_name in scores:
                scorer_result = scores[scorer_name]
                if isinstance(scorer_result, dict):
                    value = scorer_result.get("value", "")
                    score_value = 1 if value == INSPECT_SCORE_CORRECT else 0
                elif scorer_result == INSPECT_SCORE_CORRECT:
                    score_value = 1
            elif "custom_scorer" in scores:
                # Fallback to custom_scorer if capability-specific scorer not found
                scorer_result = scores["custom_scorer"]
                if isinstance(scorer_result, dict):
                    value = scorer_result.get("value", "")
                    score_value = 1 if value == INSPECT_SCORE_CORRECT else 0
                elif scorer_result == INSPECT_SCORE_CORRECT:
                    score_value = 1

            response_data[(model_name, unique_question_id)] = score_value
            if unique_question_id not in question_info:
                question_info[unique_question_id] = {
                    "task": task_name,
                    "question_id": question_id,
                    "input": sample.get("input", ""),
                    "target": sample.get("target", ""),
                }

        if (file_idx + 1) % 20 == 0:
            logger.debug(
                "Processed %d/%d files, %d samples...",
                file_idx + 1,
                len(data),
                samples_processed,
            )

    logger.info(
        "Extracted %d unique questions, %d model-question responses from %d files",
        len(question_info),
        len(response_data),
        files_processed,
    )
    return response_data, question_info


def create_response_matrix_from_flat(
    response_data: Dict[Tuple[str, str], int],
) -> Tuple[List[List[int]], List[str], List[str]]:
    """Build response matrix for IRT from (model, question_id) -> score dict.

    Missing (model, question_id) entries are treated as 0.

    Args
    ----
        response_data: (model_name, question_id) -> 0 or 1.

    Returns
    -------
        (response_matrix, question_ids, model_names):
        - response_matrix: 2D list, rows=questions, columns=models
        - question_ids: row order
        - model_names: column order
    """
    models = sorted(set(m for m, _ in response_data.keys()))
    questions = sorted(set(q for _, q in response_data.keys()))

    matrix: List[List[int]] = []
    for question_id in questions:
        row = [response_data.get((model_name, question_id), 0) for model_name in models]
        matrix.append(row)

    logger.info(
        "Created response matrix: %d questions x %d models",
        len(questions),
        len(models),
    )
    return matrix, questions, models


def get_model_question_counts(
    response_data: Dict[Tuple[str, str], int],
) -> Dict[str, int]:
    """Count questions per model from (model, question_id) -> score dict."""
    counts: Dict[str, int] = defaultdict(int)
    for model_name, _ in response_data.keys():
        counts[model_name] += 1
    return dict(counts)


def group_response_data_by_capability(
    response_data: Dict[Tuple[str, str], int],
    question_info: Dict[str, Any],
) -> Dict[str, Dict[Tuple[str, str], int]]:
    """Group (model, question_id) -> score by capability (task name).

    Uses question_info[unique_question_id]["task"] to determine capability.
    Each capability gets a subset of response_data for that task only.

    Args
    ----
        response_data: (model_name, unique_question_id) -> 0 or 1.
        question_info: unique_question_id -> {task, question_id, ...}.

    Returns
    -------
        capability_name -> response_data subset for that capability.
    """
    by_capability: Dict[str, Dict[Tuple[str, str], int]] = {}
    for (model_name, qid), score in response_data.items():
        task = question_info.get(qid, {}).get("task", "unknown")
        if task not in by_capability:
            by_capability[task] = {}
        by_capability[task][(model_name, qid)] = score
    logger.info(
        "Grouped response data into %d capabilities: %s",
        len(by_capability),
        list(by_capability.keys()),
    )
    return by_capability


def _clean_task_name(x: str) -> str:
    """Extract capability/task name from eval task path."""
    return x.split("/")[-1]


def read_inspect_score_samples(
    json_path: str,
) -> Tuple[Dict[str, int], str]:
    """
    Read raw task-level scores from an inspect evaluation JSON file.

    Args
    ----
        json_path: Path to the inspect score JSON file.

    Returns
    -------
        (task_scores, capability_name): Dict mapping task_id to 0/1 score,
        and the capability name from the eval config.
    """
    data = read_json_file(json_path)
    samples = data.get("samples", [])
    eval_info = data.get("eval", {})
    task_name = eval_info.get("master_task") or eval_info.get("task", "")
    capability_name = _clean_task_name(task_name)
    scorer_name = CAPABILITY_SCORER_MAP.get(
        capability_name, constants.DEFAULT_INSPECT_SCORER_NAME
    )

    task_scores: Dict[str, int] = {}
    for sample in samples:
        task_id = sample.get("id")
        if task_id is None:
            continue
        try:
            score_val = sample.get("scores", {}).get(scorer_name, {}).get("value")
            correct = 1 if score_val == INSPECT_SCORE_CORRECT else 0
            task_scores[task_id] = correct
        except (TypeError, KeyError) as e:
            logger.warning(
                "Skipping sample id=%s in %s: %s", task_id, json_path, e
            )
    return task_scores, capability_name


def build_response_matrix_from_inspect_score_files(
    model_score_files: List[Tuple[str, str]],
) -> Tuple[List[List[int]], List[str], List[str]]:
    """
    Build a response matrix from multiple inspect score JSON files (one per model).

    Uses the intersection of task ids across all files so that every cell
    is defined (every model has a score for every task).

    Args
    ----
        model_score_files: List of (model_name, json_file_path) for each subject model.

    Returns
    -------
        (response_matrix, question_ids, model_names):
        - response_matrix: 2D list, rows = questions (tasks), columns = models.
        - question_ids: List of task ids in row order.
        - model_names: List of model names in column order.
    """
    if not model_score_files:
        return [], [], []

    all_task_scores: List[Tuple[str, Dict[str, int]]] = []
    model_names: List[str] = []
    task_id_sets: List[set] = []

    for model_name, json_path in model_score_files:
        task_scores, _ = read_inspect_score_samples(json_path)
        if not task_scores:
            logger.warning(
                "No task scores found for model %s in %s", model_name, json_path
            )
            continue
        all_task_scores.append((model_name, task_scores))
        model_names.append(model_name)
        task_id_sets.append(set(task_scores.keys()))

    if not all_task_scores:
        return [], [], []

    # Use intersection of task ids so every (task, model) has a value
    common_ids = task_id_sets[0]
    for s in task_id_sets[1:]:
        common_ids = common_ids & s
    question_ids = sorted(common_ids)

    if not question_ids:
        logger.warning(
            "No common task ids across model files; cannot build response matrix."
        )
        return [], [], []

    n_items = len(question_ids)
    n_persons = len(model_names)
    response_matrix = [
        [all_task_scores[j][1][qid] for j in range(n_persons)]
        for qid in question_ids
    ]
    return response_matrix, question_ids, model_names


def fit_3pl_irt(
    response_matrix: List[List[int]],
    question_ids: List[str],
    model_names: List[str],
    max_iterations: int = 2000,
    quadrature_n: int = 41,
    model_type: str = "3PL",
) -> Dict[str, Any]:
    """
    Fit 1PL, 2PL, or 3PL IRT model using the girth library.

    For 1PL and 2PL, the corresponding girth MML routines are used.
    For 3PL, the three-parameter logistic model is fit with upper asymptote
    fixed at 1.0.

    Args
    ----
        response_matrix: 2D list, rows = questions, columns = models (0/1).
        question_ids: List of question (task) IDs in row order.
        model_names: List of model names (subjects) in column order.
        max_iterations: Maximum MML iterations.
        quadrature_n: Quadrature points for numerical integration.
        model_type: "1PL", "2PL", or "3PL".

    Returns
    -------
        Dictionary with keys:
        - item_parameters: dict task_id -> {discrimination, difficulty, guessing}
        - model_info: n_items, n_persons, model_type, method, note
    """
    if not GIRTH_AVAILABLE:
        raise ImportError(
            "The 'girth' library is required for IRT fitting. "
            "Install it with: pip install girth"
        )

    model_type = (model_type or "3PL").upper()
    if model_type not in {"1PL", "2PL", "3PL"}:
        raise ValueError(
            f"Unsupported IRT model_type '{model_type}'. "
            "Supported values are '1PL', '2PL', and '3PL'."
        )

    data = np.array(response_matrix, dtype=int)
    n_items, n_persons = data.shape

    # Girth 3PL/2PL/1PL need at least 2 items and 2 persons (otherwise internal .dot fails)
    if n_items < 2 or n_persons < 2:
        raise ValueError(
            f"IRT fitting requires at least 2 items (questions) and 2 persons (models). "
            f"Got n_items={n_items}, n_persons={n_persons}. "
            "Skip this capability or add more data."
        )

    logger.info(
        "Fitting %s IRT model on %d items and %d models ...",
        model_type,
        n_items,
        n_persons,
    )

    options = {
        "max_iteration": int(max_iterations),
        "quadrature_n": int(quadrature_n),
    }

    if model_type == "1PL":
        item_results = rasch_mml(data, options=options)
        difficulty = item_results["Difficulty"]
        discrimination = np.ones_like(difficulty, dtype=float)
        guessing = np.zeros_like(difficulty, dtype=float)
        note = (
            "1PL (Rasch)-style parameters: discrimination fixed to 1, "
            "guessing fixed to 0; upper asymptote fixed at 1.0."
        )
    elif model_type == "2PL":
        item_results = twopl_mml(data, options=options)
        discrimination = item_results["Discrimination"]
        difficulty = item_results["Difficulty"]
        guessing = np.zeros_like(difficulty, dtype=float)
        note = (
            "2PL-style parameters from fit with guessing fixed to 0; "
            "upper asymptote fixed at 1.0."
        )
    else:
        item_results = threepl_mml(data, options=options)
        discrimination = item_results["Discrimination"]
        difficulty = item_results["Difficulty"]
        guessing = item_results.get("Guessing")
        if guessing is None:
            guessing = np.zeros_like(difficulty, dtype=float)
        note = (
            "3PL model: upper asymptote is fixed at 1.0 (not estimated)."
        )

    # Optionally estimate person abilities (not returned)
    ability_3pl_eap(data, difficulty, discrimination, guessing)

    item_parameters: Dict[str, Dict[str, float]] = {}
    for idx, q_id in enumerate(question_ids):
        if idx < len(discrimination):
            item_parameters[q_id] = {
                "discrimination": float(discrimination[idx]),
                "difficulty": float(difficulty[idx]),
                "guessing": float(guessing[idx]),
            }

    model_info = {
        "n_items": n_items,
        "n_persons": n_persons,
        "model_type": model_type,
        "method": "MML (Marginal Maximum Likelihood)",
        "note": note,
    }

    logger.info(
        "IRT %s fit complete. Discrimination range [%.3f, %.3f], "
        "difficulty range [%.3f, %.3f], guessing range [%.3f, %.3f].",
        model_type,
        float(np.min(discrimination)),
        float(np.max(discrimination)),
        float(np.min(difficulty)),
        float(np.max(difficulty)),
        float(np.min(guessing)),
        float(np.max(guessing)),
    )

    return {
        "item_parameters": item_parameters,
        "model_info": model_info,
    }


def calculate_response_statistics(
    response_matrix: List[List[int]],
    question_ids: List[str],
    model_names: List[str],
) -> Dict[str, Any]:
    """Compute basic statistics for the response matrix."""
    matrix = np.array(response_matrix)
    if matrix.size == 0:
        return {
            "question_statistics": {},
            "model_statistics": {},
            "overall": {
                "total_responses": 0,
                "correct_responses": 0,
                "accuracy": 0.0,
                "n_questions": 0,
                "n_models": 0,
            },
        }

    stats: Dict[str, Any] = {
        "question_statistics": {},
        "model_statistics": {},
        "overall": {
            "total_responses": int(matrix.size),
            "correct_responses": int(np.sum(matrix)),
            "accuracy": float(np.mean(matrix)),
            "n_questions": len(question_ids),
            "n_models": len(model_names),
        },
    }

    for idx, question_id in enumerate(question_ids):
        if idx < matrix.shape[0]:
            row = matrix[idx, :]
            stats["question_statistics"][question_id] = {
                "mean_score": float(np.mean(row)),
                "std_score": float(np.std(row)) if row.size > 1 else 0.0,
                "total_correct": int(np.sum(row)),
                "total_attempts": len(row),
            }

    for idx, model_name in enumerate(model_names):
        if idx < matrix.shape[1]:
            col = matrix[:, idx]
            stats["model_statistics"][model_name] = {
                "mean_score": float(np.mean(col)),
                "std_score": float(np.std(col)) if col.size > 1 else 0.0,
                "total_correct": int(np.sum(col)),
                "total_attempts": len(col),
            }

    return stats


def discover_tasks_files(
    capabilities_dir: str,
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """Discover tasks.json files under capabilities_dir and group by capability name.

    Args
    ----
        capabilities_dir: Base directory to search for **/tasks.json.

    Returns
    -------
        capability_name -> [(absolute_file_path, loaded_data)], where data has "metadata" and "tasks".
        Capability name is taken from the first task's capability_name in each file.
    """
    if not os.path.isdir(capabilities_dir):
        logger.warning("Capabilities dir does not exist: %s", capabilities_dir)
        return {}

    pattern = os.path.join(capabilities_dir, "**", "tasks.json")
    files = glob.glob(pattern, recursive=True)
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)

    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            continue
        tasks = data.get("tasks", [])
        if not tasks:
            logger.debug("No tasks in %s", path)
            continue
        # Use capability_name from first task (tasks from one file belong to one capability)
        first = tasks[0]
        cap_name = first.get("capability_name") or first.get("capability_id") or ""
        if not cap_name:
            logger.debug("No capability_name in %s", path)
            continue
        out[cap_name].append((os.path.abspath(path), data))

    logger.info(
        "Discovered tasks files for %d capabilities under %s",
        len(out),
        capabilities_dir,
    )
    return dict(out)


def update_tasks_with_irt_and_save(
    capability_name: str,
    item_parameters: Dict[str, Dict[str, float]],
    task_files: List[Tuple[str, Dict[str, Any]]],
    capabilities_dir: str,
    output_dir: str,
) -> int:
    """Update task dicts with IRT params and save to output_dir/updated_tasks/<rel_path>.

    item_parameters is keyed by unique_question_id (e.g. capability_name_task_id) or task_id.
    Adds irt_difficulty, irt_discrimination, irt_guessing to each matching task.

    Returns
    -------
        Number of files written.
    """
    capabilities_dir_abs = os.path.abspath(capabilities_dir)
    out_subdir = os.path.join(output_dir, "updated_tasks")
    written = 0

    # Set of task identifiers we have IRT params for (normalize to bare task_id for comparison)
    prefix = capability_name + "_"
    irt_task_ids_bare = set()
    for k in item_parameters:
        if k.startswith(prefix):
            irt_task_ids_bare.add(k[len(prefix) :])
        else:
            irt_task_ids_bare.add(k)

    for file_path, data in task_files:
        tasks = data.get("tasks", [])
        if not tasks:
            continue
        file_task_ids = {t.get("task_id") for t in tasks if t.get("task_id")}

        # Check if the capability file's tasks match what we have scores for
        file_ids_match_irt = file_task_ids <= irt_task_ids_bare
        irt_ids_in_file = {
            tid for tid in irt_task_ids_bare if tid in file_task_ids
        }
        irt_ids_not_in_file = irt_task_ids_bare - file_task_ids
        if not file_ids_match_irt or irt_ids_not_in_file:
            logger.warning(
                "Capability %s: task set mismatch in %s. "
                "File has %d task_ids; IRT has %d. "
                "File tasks not in IRT: %s. IRT tasks not in file: %s.",
                capability_name,
                file_path,
                len(file_task_ids),
                len(irt_task_ids_bare),
                sorted(file_task_ids - irt_ids_in_file)[:10]
                + (["..."] if len(file_task_ids - irt_ids_in_file) > 10 else []),
                sorted(irt_ids_not_in_file)[:10]
                + (["..."] if len(irt_ids_not_in_file) > 10 else []),
            )

        updated = 0
        for task in tasks:
            task_id = task.get("task_id")
            if not task_id:
                continue
            unique_id = f"{capability_name}_{task_id}"
            params = item_parameters.get(unique_id) or item_parameters.get(task_id)
            if not params:
                continue
            task["irt_difficulty"] = params.get("difficulty")
            task["irt_discrimination"] = params.get("discrimination")
            task["irt_guessing"] = params.get("guessing")
            updated += 1

        if updated == 0:
            logger.debug(
                "No IRT match for capability %s in %s (task_ids in file may not match scores)",
                capability_name,
                file_path,
            )
            continue

        try:
            rel = os.path.relpath(file_path, capabilities_dir_abs)
        except ValueError:
            rel = os.path.basename(file_path)
        out_path = os.path.join(out_subdir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        written += 1
        logger.info(
            "Updated %d tasks with IRT params for %s -> %s",
            updated,
            capability_name,
            out_path,
        )

    return written
