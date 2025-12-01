"""IRT analysis using 3PL model via girth library.

This script uses the 'girth' library for 3PL IRT parameter estimation.
"""

import numpy as np
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import girth
    from girth import rasch_mml, twopl_mml, threepl_mml, ability_3pl_eap
    GIRTH_AVAILABLE = True
except ImportError:
    GIRTH_AVAILABLE = False
    logger.warning("girth not available. Please install via: pip install girth")


def fit_3pl_irt(response_matrix: List[List[int]], 
                question_ids: List[str],
                model_names: List[str],
                max_iterations: int = 2000,
                quadrature_n: int = 41,
                model_type: str = "3PL") -> Dict[str, Any]:
    """
    Fit 1PL, 2PL, or 3PL IRT model using the 'girth' library.
    
    For 1PL and 2PL, the corresponding girth MML routines are used directly.
    For 3PL, the three-parameter logistic model is fit with the upper asymptote
    fixed at 1.0 (standard 3PL specification).
    
    Args:
        response_matrix: 2D list where Rows=Questions, Columns=Models (Subjects)
        question_ids: List of question IDs
        model_names: List of model names (subjects)
        
    Returns:
        Dictionary containing standardized IRT parameters.
    """
    if not GIRTH_AVAILABLE:
        raise ImportError("The 'girth' library is required. Install it with: pip install girth")

    model_type = (model_type or "3PL").upper()
    if model_type not in {"1PL", "2PL", "3PL"}:
        raise ValueError(
            f"Unsupported IRT model_type '{model_type}'. "
            "Supported values are '1PL', '2PL', and '3PL'."
        )

    data = np.array(response_matrix, dtype=int)
    
    n_items, n_persons = data.shape
    logger.info(f"Fitting 3PL model via GIRTH on {n_items} items and {n_persons} models...")
    print(f"  -> Response matrix dimensions: {data.shape} (rows=questions, cols=models)")
    print(f"  -> Number of questions (items): {n_items}")
    print(f"  -> Number of models (persons): {n_persons}")
    print(f"  -> Fitting 3PL IRT model via girth on {n_items} items and {n_persons} models...")
    
    try:
        print(f"  -> Estimating item parameters using Marginal Maximum Likelihood (MML)...")

        if model_type == "1PL":
            item_results = rasch_mml(
                data,
                options={
                    'max_iteration': int(max_iterations),
                    'quadrature_n': int(quadrature_n),
                },
            )
            difficulty = item_results['Difficulty']
            discrimination = np.ones_like(difficulty, dtype=float)
            guessing = np.zeros_like(difficulty, dtype=float)

        elif model_type == "2PL":
            item_results = twopl_mml(
                data,
                options={
                    'max_iteration': int(max_iterations),
                    'quadrature_n': int(quadrature_n),
                },
            )
            discrimination = item_results['Discrimination']
            difficulty = item_results['Difficulty']
            guessing = np.zeros_like(difficulty, dtype=float)

        else:  # "3PL"
            item_results = threepl_mml(
                data,
                options={
                    'max_iteration': int(max_iterations),
                    'quadrature_n': int(quadrature_n),
                },
            )
            discrimination = item_results['Discrimination']
            difficulty = item_results['Difficulty']
            guessing = item_results.get('Guessing')
            if guessing is None:
                guessing = np.zeros_like(difficulty, dtype=float)
        
        logger.info("Item parameters estimated successfully.")
        print(f"  -> Item parameters estimated successfully")
        print(f"  -> Estimated parameters for {len(discrimination)} items")
        
        print(f"  -> Discrimination range: [{np.min(discrimination):.3f}, {np.max(discrimination):.3f}], mean: {np.mean(discrimination):.3f}")
        print(f"  -> Difficulty range: [{np.min(difficulty):.3f}, {np.max(difficulty):.3f}], mean: {np.mean(difficulty):.3f}")
        print(f"  -> Guessing range: [{np.min(guessing):.3f}, {np.max(guessing):.3f}], mean: {np.mean(guessing):.3f}")
        
        if np.allclose(discrimination, 1.0, atol=0.01):
            print(f"  -> WARNING: All discrimination values are ~1.0. This may indicate convergence issues.")
        if np.allclose(guessing, 0.0, atol=0.01):
            print(f"  -> WARNING: All guessing values are ~0.0. This may indicate convergence issues.")
        
        # Estimate person abilities (theta) but do not log or return them
        ability_3pl_eap(data, difficulty, discrimination, guessing)
        
    except Exception as e:
        logger.error(f"GIRTH estimation failed: {e}")
        print(f"  -> ERROR: GIRTH estimation failed: {e}")
        raise RuntimeError(f"GIRTH estimation failed. Ensure data is not empty or all zeros. Error: {e}") from e

    print(f"  -> Formatting results...")
    if model_type == "3PL":
        note = '3PL model: upper asymptote is fixed at 1.0 (not estimated)'
    elif model_type == "2PL":
        note = (
            '2PL-style parameters derived from a 3PL fit with guessing fixed to 0; '
            'upper asymptote is fixed at 1.0.'
        )
    else:  # "1PL"
        note = (
            '1PL (Rasch)-style parameters derived from a 3PL fit with discrimination '
            'fixed to 1 and guessing fixed to 0; upper asymptote is fixed at 1.0.'
        )

    results = {
        'item_parameters': {},
        'model_info': {
            'n_items': n_items,
            'n_persons': n_persons,
            'model_type': f'{model_type}',
            'method': 'MML (Marginal Maximum Likelihood)',
            'note': note,
        },
    }

    print(f"  -> Mapping item parameters for {len(question_ids)} questions...")
    for idx, q_id in enumerate(question_ids):
        if idx < len(discrimination):
            results['item_parameters'][q_id] = {
                'discrimination': float(discrimination[idx]),
                'difficulty': float(difficulty[idx]),
                'guessing': float(guessing[idx])
            }
    print(f"  -> Mapped parameters for {len(results['item_parameters'])} items")

    print(f"  -> 3PL IRT analysis completed successfully")
    logger.info("3PL IRT analysis completed successfully.")
    return results


def calculate_statistics(response_matrix: np.ndarray,
                         question_ids: List[str],
                         model_names: List[str]) -> Dict:
    """Calculate basic statistics for the response matrix."""
    matrix = np.array(response_matrix)
    
    stats = {
        'question_statistics': {},
        'model_statistics': {},
        'overall': {
            'total_responses': int(matrix.size),
            'correct_responses': int(np.sum(matrix)),
            'accuracy': float(np.mean(matrix)),
            'n_questions': len(question_ids),
            'n_models': len(model_names)
        }
    }
    
    # Question-level statistics
    for idx, question_id in enumerate(question_ids):
        question_scores = matrix[idx, :]
        stats['question_statistics'][question_id] = {
            'mean_score': float(np.mean(question_scores)),
            'std_score': float(np.std(question_scores)),
            'total_correct': int(np.sum(question_scores)),
            'total_attempts': len(question_scores)
        }
    
    # Model-level statistics
    for idx, model_name in enumerate(model_names):
        model_scores = matrix[:, idx]
        stats['model_statistics'][model_name] = {
            'mean_score': float(np.mean(model_scores)),
            'std_score': float(np.std(model_scores)),
            'total_correct': int(np.sum(model_scores)),
            'total_attempts': len(model_scores)
        }
    
    return stats
