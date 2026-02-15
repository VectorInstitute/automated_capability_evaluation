"""Load and process score data from ACE evaluation outputs."""

import glob
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def load_score_files(scores_dir: str) -> List[Dict]:
    """Load all JSON score files from the scores directory.
    
    Args:
        scores_dir: Path to the scores directory
        
    Returns:
        List of loaded JSON data dictionaries
    """
    if not os.path.exists(scores_dir):
        print(f"  -> ERROR: Directory does not exist: {scores_dir}")
        logger.error(f"Directory does not exist: {scores_dir}")
        return []
    
    pattern = os.path.join(scores_dir, "**/*.json")
    files = glob.glob(pattern, recursive=True)
    print(f"  -> Found {len(files)} JSON files")
    logger.info(f"Found {len(files)} score files")
    
    data = []
    errors = 0
    for i, file_path in enumerate(files):
        try:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                file_data['_file_path'] = file_path
                data.append(file_data)
            if (i + 1) % 50 == 0:
                print(f"  -> Loaded {i + 1}/{len(files)} files...")
        except Exception as e:
            errors += 1
            logger.warning(f"Error loading {file_path}: {e}")
            continue
    
    if errors > 0:
        print(f"  -> Warning: {errors} files failed to load")
    print(f"  -> Successfully loaded {len(data)} files")
    
    return data


def extract_question_responses(data: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Extract question-response matrix from score data.
    
    Args:
        data: List of score file data dictionaries
        
    Returns:
        Tuple of (response_matrix, question_info)
        - response_matrix: Dict mapping (model_name, question_id) -> score (0 or 1)
        - question_info: Dict mapping question_id -> question metadata
    """
    response_matrix = {}
    question_info = {}
    files_processed = 0
    samples_processed = 0
    
    print(f"  -> Processing {len(data)} score files...")
    for file_idx, file_data in enumerate(data):
        if 'samples' not in file_data:
            continue
        
        # Extract model name from file path
        file_path = file_data.get('_file_path', '')
        model_name = extract_model_name(file_path)
        
        if not model_name:
            continue
        
        # Extract capability/task name
        eval_data = file_data.get('eval', {})
        task_name = eval_data.get('task', 'unknown')
        
        if 'samples' not in file_data:
            continue
        
        files_processed += 1
        for sample in file_data['samples']:
            samples_processed += 1
            question_id = sample.get('id', '')
            if not question_id:
                continue
            
            # Create unique question ID: task_name + question_id
            unique_question_id = f"{task_name}_{question_id}"
            
            # Extract score (C = Correct = 1, others = 0)
            scores = sample.get('scores', {})
            score_value = 0
            if 'custom_scorer' in scores:
                scorer_result = scores['custom_scorer']
                if isinstance(scorer_result, dict):
                    value = scorer_result.get('value', '')
                    score_value = 1 if value == 'C' else 0
                elif scorer_result == 'C':
                    score_value = 1
            
            # Store response
            key = (model_name, unique_question_id)
            response_matrix[key] = score_value
            
            # Store question info (only once per question)
            if unique_question_id not in question_info:
                question_info[unique_question_id] = {
                    'task': task_name,
                    'question_id': question_id,
                    'input': sample.get('input', ''),
                    'target': sample.get('target', '')
                }
        
        if (file_idx + 1) % 20 == 0:
            print(f"  -> Processed {file_idx + 1}/{len(data)} files, {samples_processed} samples...")
    
    print(f"  -> Processed {files_processed} files with samples")
    print(f"  -> Total samples processed: {samples_processed}")
    logger.info(f"Extracted {len(question_info)} unique questions")
    logger.info(f"Extracted {len(response_matrix)} model-question responses")
    
    return response_matrix, question_info


def extract_model_name(file_path: str) -> str:
    """Extract model name from file path.
    
    Args:
        file_path: Full path to the score file
        
    Returns:
        Model name or empty string
    """
    parts = file_path.split('/')
    # Look for model name in path (typically after 'scores/')
    try:
        scores_idx = parts.index('scores')
        if scores_idx + 1 < len(parts):
            return parts[scores_idx + 1]
    except ValueError:
        pass
    
    return ''


def create_response_matrix(response_data: Dict[Tuple[str, str], int]) -> Tuple[List[List[int]], List[str], List[str]]:
    """Create a response matrix for IRT analysis.
    
    Args:
        response_data: Dict mapping (model_name, question_id) -> score
        
    Returns:
        Tuple of (response_matrix, model_names, question_ids)
        - response_matrix: 2D list where rows are questions and columns are models
        - model_names: List of model names (column order)
        - question_ids: List of question IDs (row order)
    """
    # Get unique models and questions
    models = sorted(set(model for model, _ in response_data.keys()))
    questions = sorted(set(qid for _, qid in response_data.keys()))
    
    # Create matrix: rows = questions, columns = models
    matrix = []
    for question_id in questions:
        row = []
        for model_name in models:
            key = (model_name, question_id)
            score = response_data.get(key, 0)  # Default to 0 if missing
            row.append(score)
        matrix.append(row)
    
    logger.info(f"Created response matrix: {len(questions)} questions x {len(models)} models")
    
    return matrix, models, questions


def get_model_question_counts(response_data: Dict[Tuple[str, str], int]) -> Dict[str, int]:
    """Get count of questions per model.
    
    Args:
        response_data: Dict mapping (model_name, question_id) -> score
        
    Returns:
        Dict mapping model_name -> question_count
    """
    model_counts = defaultdict(int)
    for model_name, _ in response_data.keys():
        model_counts[model_name] += 1
    return dict(model_counts)

