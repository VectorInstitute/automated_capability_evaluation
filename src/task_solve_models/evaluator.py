
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
    """Extract the first number from text, handling currency, percentages, and scientific notation."""
    if text is None:
        return None
    
    text_str = str(text).strip()
    
    # Handle accounting format with parentheses for negatives: (500) = -500
    if text_str.startswith('(') and text_str.endswith(')'):
        text_str = '-' + text_str[1:-1]
    
    # Remove currency symbols, commas, and spaces
    clean_text = text_str.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace(" ", "")
    
    # Remove percentage sign (we'll just extract the number)
    clean_text = clean_text.replace("%", "")
    
    # Try to match scientific notation first (e.g., 1.5e10, 3.2E-5)
    sci_match = re.search(r"[-+]?\d*\.?\d+[eE][-+]?\d+", clean_text)
    if sci_match:
        try:
            return float(sci_match.group())
        except ValueError:
            pass
    
    # Standard number pattern (including decimals and negatives)
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
    Supports: calcu (numerical with tolerance), bool (true/false), mcq (A/B/C/D), and general text.
    """
    prediction = result.get("prediction")
    ground_truth = result.get("ground_truth")
    task_type = result.get("task_type") or result.get("task")  # Handle both field names

    if prediction is None:
        log.debug(f"Evaluation failed for {result.get('id')}: prediction is None")
        return False
    
    if ground_truth is None:
        log.debug(f"Evaluation failed for {result.get('id')}: ground_truth is None")
        return False

    try:
        if task_type == "calcu":
            # Numerical comparison with tolerance
            pred_val = extract_number(prediction)
            gt_val = extract_number(ground_truth)
            
            if pred_val is None or gt_val is None:
                # Fallback to string comparison if numbers can't be parsed
                log.debug(f"Could not extract numbers for {result.get('id')}: pred={prediction}, gt={ground_truth}")
                return normalize_text(prediction) == normalize_text(ground_truth)
            
            # Adaptive tolerance based on magnitude
            # For small numbers (|gt| < 1): use absolute tolerance of 0.01
            # For larger numbers: use 1% relative tolerance
            # Also handle exact matches
            if pred_val == gt_val:
                return True
            
            abs_diff = abs(pred_val - gt_val)
            abs_gt = abs(gt_val)
            
            if abs_gt < 1.0:
                # Small numbers: absolute tolerance
                return abs_diff <= 0.01
            else:
                # Larger numbers: relative tolerance of 1%
                rel_diff = abs_diff / abs_gt
                return rel_diff <= 0.01
            
            return False

        elif task_type == "bool":
            # Normalize boolean answers
            pred_str = normalize_text(prediction)
            gt_str = normalize_text(ground_truth)
            
            # Define true/false values with word boundaries to avoid substring issues
            true_values = ["1.0", "1", "true", "yes", "t", "y"]
            false_values = ["0.0", "0", "false", "no", "f", "n"]
            
            # First check for exact matches
            if pred_str in true_values:
                pred_is_true = True
                pred_is_false = False
            elif pred_str in false_values:
                pred_is_true = False
                pred_is_false = True
            else:
                # Check for word-level matches (use word boundaries)
                pred_is_true = any(re.search(rf'\b{re.escape(v)}\b', pred_str) for v in true_values if len(v) > 1)
                pred_is_false = any(re.search(rf'\b{re.escape(v)}\b', pred_str) for v in false_values if len(v) > 1)
            
            # Same for ground truth
            if gt_str in true_values:
                gt_is_true = True
                gt_is_false = False
            elif gt_str in false_values:
                gt_is_true = False
                gt_is_false = True
            else:
                gt_is_true = any(re.search(rf'\b{re.escape(v)}\b', gt_str) for v in true_values if len(v) > 1)
                gt_is_false = any(re.search(rf'\b{re.escape(v)}\b', gt_str) for v in false_values if len(v) > 1)
            
            if pred_is_true and gt_is_true:
                return True
            if pred_is_false and gt_is_false:
                return True
            
            # If neither mapped clearly, fall back to exact string match
            return pred_str == gt_str

        elif task_type == "mcq":
            # Exact match for single letter (A, B, C, D, E, etc.)
            pred_str = str(prediction).strip().upper()
            gt_str = str(ground_truth).strip().upper()
            
            # Extract ground truth letter (should be a single letter)
            gt_letter = None
            gt_match = re.search(r'^([A-Z])$|^([A-Z])[\.\):\s]', gt_str)
            if gt_match:
                gt_letter = gt_match.group(1) or gt_match.group(2)
            elif len(gt_str) == 1 and gt_str.isalpha():
                gt_letter = gt_str
            
            if gt_letter:
                # Look for this letter in prediction
                # Try exact match first
                if pred_str == gt_letter:
                    return True
                
                # Try "A." or "A)" or "A:" format at start
                if re.match(rf'^{gt_letter}[\.\):\s]', pred_str):
                    return True
                
                # Try patterns like "Answer: A" or "The answer is A"
                if re.search(rf'(?:answer|choice|option)[\s:]+{gt_letter}\b', pred_str, re.IGNORECASE):
                    return True
                
                # Try finding standalone letter at the very start (first 3 chars)
                if pred_str.startswith(gt_letter) and (len(pred_str) == 1 or not pred_str[1].isalpha()):
                    return True
                
                # Check if letter appears early and standalone (within first 5 chars)
                match = re.search(rf'\b{gt_letter}\b', pred_str)
                if match and match.start() <= 5:
                    return True
            
            # Fallback: exact string match
            return pred_str == gt_str

        else:
            # Default string comparison
            return normalize_text(prediction) == normalize_text(ground_truth)

    except Exception as e:
        log.warning(f"Error evaluating result for {result.get('id')}: {e}")
        return False
