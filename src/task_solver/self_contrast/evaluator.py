"""Evaluation utilities for the Self-Contrast framework.

Provides ``evaluate_result`` (single problem), ``evaluate_batch`` (full
run), ``print_summary``, and ``save_results``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


log = logging.getLogger(__name__)


# ======================================================================
# Core evaluation logic (self-contained, no external dependency)
# ======================================================================


def normalize_text(text: str) -> str:
    """Lower-case and strip whitespace."""
    if text is None:
        return ""
    return str(text).lower().strip()


def extract_number(text: str) -> Optional[float]:
    """Extract the first number from *text*, handling currency / scientific notation."""
    if text is None:
        return None

    text_str = str(text).strip()
    if text_str.startswith("(") and text_str.endswith(")"):
        text_str = "-" + text_str[1:-1]

    clean = (
        text_str.replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace(" ", "")
        .replace("%", "")
    )

    sci = re.search(r"[-+]?\d*\.?\d+[eE][-+]?\d+", clean)
    if sci:
        try:
            return float(sci.group())
        except ValueError:
            pass

    match = re.search(r"[-+]?\d*\.?\d+", clean)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _has_percent(text: Any) -> bool:
    return text is not None and "%" in str(text)


def evaluate_result(result: Dict[str, Any]) -> bool:
    """Evaluate whether *prediction* matches *ground_truth*.

    Supports ``calcu`` (numerical, 1 % relative tolerance), ``bool``,
    ``mcq``, and generic string comparison.
    """
    prediction = result.get("prediction")
    ground_truth = result.get("ground_truth")
    task_type = result.get("task_type") or result.get("task")

    if prediction is None or ground_truth is None:
        return False

    try:
        if task_type == "calcu":
            return _evaluate_calcu(prediction, ground_truth)
        if task_type == "bool":
            return _evaluate_bool(prediction, ground_truth)
        if task_type == "mcq":
            return _evaluate_mcq(prediction, ground_truth)
        return normalize_text(prediction) == normalize_text(ground_truth)
    except Exception as exc:
        log.warning("Error evaluating result for %s: %s", result.get("id"), exc)
        return False


# ------------------------------------------------------------------
# Task-type specific helpers
# ------------------------------------------------------------------


def _evaluate_calcu(prediction: str, ground_truth: str) -> bool:
    pred_val = extract_number(prediction)
    gt_val = extract_number(ground_truth)
    if pred_val is None or gt_val is None:
        return normalize_text(prediction) == normalize_text(ground_truth)
    if _numeric_match(pred_val, gt_val):
        return True
    # Treat "5%" / "0.05" / "5" as equivalent: if exactly one side is written as
    # a percent, try comparing after a 100x scaling.
    pred_pct = _has_percent(prediction)
    gt_pct = _has_percent(ground_truth)
    if pred_pct != gt_pct:
        if pred_pct and _numeric_match(pred_val / 100.0, gt_val):
            return True
        if gt_pct and _numeric_match(pred_val, gt_val / 100.0):
            return True
    return False


def _numeric_match(pred_val: float, gt_val: float) -> bool:
    if pred_val == gt_val:
        return True
    abs_diff = abs(pred_val - gt_val)
    abs_gt = abs(gt_val)
    if abs_gt < 1.0:
        return abs_diff <= 0.01
    return abs_diff / abs_gt <= 0.01


def _evaluate_bool(prediction: str, ground_truth: str) -> bool:
    pred_str = normalize_text(prediction)
    gt_str = normalize_text(ground_truth)

    true_values = ["1.0", "1", "true", "yes", "t", "y"]
    false_values = ["0.0", "0", "false", "no", "f", "n"]

    pred_true = pred_str in true_values or any(
        re.search(rf"\b{re.escape(v)}\b", pred_str) for v in true_values if len(v) > 1
    )
    pred_false = pred_str in false_values or any(
        re.search(rf"\b{re.escape(v)}\b", pred_str) for v in false_values if len(v) > 1
    )

    gt_true = gt_str in true_values or any(
        re.search(rf"\b{re.escape(v)}\b", gt_str) for v in true_values if len(v) > 1
    )
    gt_false = gt_str in false_values or any(
        re.search(rf"\b{re.escape(v)}\b", gt_str) for v in false_values if len(v) > 1
    )

    if pred_true and gt_true:
        return True
    if pred_false and gt_false:
        return True
    return pred_str == gt_str


def _evaluate_mcq(prediction: str, ground_truth: str) -> bool:
    pred_str = str(prediction).strip().upper()
    gt_str = str(ground_truth).strip().upper()

    gt_letter = None
    gt_match = re.search(r"^([A-Z])$|^([A-Z])[\.\):\s]", gt_str)
    if gt_match:
        gt_letter = gt_match.group(1) or gt_match.group(2)
    elif len(gt_str) == 1 and gt_str.isalpha():
        gt_letter = gt_str

    if gt_letter:
        if pred_str == gt_letter:
            return True
        if re.match(rf"^{gt_letter}[\.\):\s]", pred_str):
            return True
        if re.search(
            rf"(?:answer|choice|option)[\s:]+{gt_letter}\b",
            pred_str,
            re.IGNORECASE,
        ):
            return True
        if pred_str.startswith(gt_letter) and (
            len(pred_str) == 1 or not pred_str[1].isalpha()
        ):
            return True
        match = re.search(rf"\b{gt_letter}\b", pred_str)
        if match and match.start() <= 5:
            return True

    return pred_str == gt_str


# ======================================================================
# Batch evaluation helpers
# ======================================================================


def evaluate_batch(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics over a list of result dicts.

    Returns
    -------
    Dict[str, Any]
        Keys: ``total_processed``, ``total_correct``, ``accuracy``,
        ``per_type`` (dict mapping task type to sub-metrics),
        ``adjudication_rate`` (fraction where contrast triggered
        adjudication).
    """
    total = 0
    correct = 0
    per_type: Dict[str, Dict[str, int]] = {}
    adjudication_count = 0

    for result in results:
        total += 1
        is_correct = result.get("is_correct", False)
        if is_correct:
            correct += 1

        tt = result.get("task_type") or result.get("task") or "unknown"
        if tt not in per_type:
            per_type[tt] = {"total": 0, "correct": 0}
        per_type[tt]["total"] += 1
        if is_correct:
            per_type[tt]["correct"] += 1

        contrast_details = result.get("contrast_details", {})
        final = contrast_details.get("final", {})
        decision_source = final.get("decision_source", "")
        if decision_source == "contrast_adjudication":
            adjudication_count += 1

    per_type_metrics = {}
    for tt, counts in per_type.items():
        per_type_metrics[tt] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0,
        }

    return {
        "total_processed": total,
        "total_correct": correct,
        "accuracy": correct / total if total else 0,
        "per_type": per_type_metrics,
        "adjudication_rate": adjudication_count / total if total else 0,
    }


def print_summary(
    metrics: Dict[str, Any],
    *,
    model: str = "",
    method: str = "Self-Contrast",
    batch_info: str = "",
    execution_time: str = "",
) -> None:
    """Print a formatted evaluation summary to stdout."""
    print("\n" + "=" * 80)
    print(f"{method.upper()} EVALUATION SUMMARY")
    print("=" * 80)
    if model:
        print(f"Model: {model}")
    print(f"Method: {method}")
    if batch_info:
        print(f"Batch: {batch_info}")
    print(f"Total Problems: {metrics['total_processed']}")
    print(f"Correct: {metrics['total_correct']}")
    print(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")

    if metrics.get("per_type"):
        print("\nPer Task-Type:")
        for tt, tm in metrics["per_type"].items():
            print(
                f"  {tt}: {tm['correct']}/{tm['total']} ({tm['accuracy'] * 100:.1f}%)"
            )

    adj_rate = metrics.get("adjudication_rate", 0)
    if adj_rate > 0:
        print(f"\nAdjudication Rate: {adj_rate * 100:.1f}%")

    if execution_time:
        print(f"Execution Time: {execution_time}")
    print("=" * 80)


def save_results(
    metrics: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_path: Path,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist metrics and per-problem results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data: Dict[str, Any] = {"metrics": dict(metrics), "results": results}
    if extra_metadata:
        output_data["metrics"].update(extra_metadata)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log.info("Results saved to %s", output_path)
