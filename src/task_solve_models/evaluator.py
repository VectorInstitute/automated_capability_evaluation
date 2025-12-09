
import re
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove punctuation except for useful ones like dots/commas in numbers
    # But for general text, maybe we want to keep it simple
    return text

def extract_number(text: str) -> float | None:
    """Extract the first number from text, handling currency and percentages."""
    if text is None:
        return None
    # Remove currency symbols and other common non-numeric chars but keep digits, dot, minus
    clean_text = str(text).replace(",", "").replace("$", "").replace("%", "")
    # Find number pattern
    match = re.search(r"[-+]?\d*\.?\d+", clean_text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None

def evaluate_result(result: Dict[str, Any]) -> bool:
    """
    Evaluates if the prediction matches the ground truth.
    """
    prediction = result.get("prediction")
    ground_truth = result.get("ground_truth")
    task_type = result.get("task_type")

    if prediction is None:
        return False

    try:
        if task_type == "calcu":
            # Numerical comparison with tolerance
            pred_val = extract_number(prediction)
            gt_val = extract_number(ground_truth)
            
            if pred_val is None or gt_val is None:
                # Fallback to string comparison if numbers can't be parsed
                return normalize_text(prediction) == normalize_text(ground_truth)
            
            # Allow for small floating point differences or rounding
            # Using 1% relative tolerance or 0.01 absolute tolerance
            if abs(pred_val - gt_val) <= 1e-2 or abs(pred_val - gt_val) / (abs(gt_val) + 1e-9) <= 0.01:
                return True
            return False

        elif task_type == "bool":
            # Normalize boolean answers
            pred_str = normalize_text(prediction)
            gt_str = normalize_text(ground_truth)
            
            true_values = ["1.0", "1", "true", "yes", "t", "y"]
            false_values = ["0.0", "0", "false", "no", "f", "n"]
            
            # Check if prediction maps to True
            pred_is_true = any(v == pred_str or (len(v) > 1 and v in pred_str) for v in true_values)
            pred_is_false = any(v == pred_str or (len(v) > 1 and v in pred_str) for v in false_values)
            
            # Check if ground truth maps to True
            gt_is_true = any(v == gt_str or (len(v) > 1 and v in gt_str) for v in true_values)
            gt_is_false = any(v == gt_str or (len(v) > 1 and v in gt_str) for v in false_values)
            
            if pred_is_true and gt_is_true:
                return True
            if pred_is_false and gt_is_false:
                return True
            
            # If neither mapped clearly, fall back to exact string match
            return pred_str == gt_str

        elif task_type == "mcq":
            # Exact match for single letter
            pred_str = str(prediction).strip().upper()
            gt_str = str(ground_truth).strip().upper()
            
            # Extract just the first letter if the model output "A. Description"
            pred_match = re.search(r"\b([A-Z])\b", pred_str) # Look for standalone letter
            gt_match = re.search(r"\b([A-Z])\b", gt_str)
            
            if pred_match and gt_match:
                return pred_match.group(1) == gt_match.group(1)
            
            # Fallback: check if the ground truth letter is at the start of prediction
            if len(gt_str) == 1 and pred_str.startswith(gt_str):
                return True
                
            return pred_str == gt_str

        else:
            # Default string comparison
            return normalize_text(prediction) == normalize_text(ground_truth)

    except Exception as e:
        log.warning(f"Error evaluating result for {result.get('id')}: {e}")
        return False
