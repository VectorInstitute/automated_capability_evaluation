"""Utility functions for evaluating benchmark-level metrics."""

from __future__ import annotations

import statistics
from typing import Iterable, List, Mapping, Union

import numpy as np
from scipy.stats import spearmanr


def compute_benchmark_difficulty(
    accuracies: Union[Iterable[float], Mapping[str, float]],
) -> float:
    """
    Compute benchmark difficulty given per-model accuracies.

    The difficulty of a benchmark is defined as:

        DIFFICULTY(D_c, M) = 1 - max_{m in M} acc(LM_m, D_c)

    i.e., one minus the highest accuracy achieved by any model on the benchmark.

    Args:
        accuracies: Either an iterable of accuracy values in [0.0, 1.0] for each model,
            or a mapping from model name to accuracy in [0.0, 1.0].

    Returns:
        A float in [0.0, 1.0] representing the benchmark difficulty.

    Raises:
        ValueError: If no accuracies are provided.
    """
    # Handle Mapping by extracting values, otherwise treat as iterable
    if isinstance(accuracies, Mapping):
        accuracies = accuracies.values()
    
    accuracies = list(accuracies)
    if not accuracies:
        raise ValueError("Cannot compute difficulty: no accuracies provided.")

    best_acc = max(accuracies)
    # Clamp to [0, 1] in case of tiny numerical issues.
    best_acc = max(0.0, min(1.0, best_acc))
    return 1.0 - best_acc


def compute_benchmark_separability(
    accuracies: Union[Iterable[float], Mapping[str, float]],
) -> float:
    """
    Compute benchmark separability given per-model accuracies.

    Separability is defined as the mean absolute deviation of model accuracies
    around their mean:

        SEP(D_c, M) = mean(|v_c - mean(v_c)|)

    where ``v_c`` are the accuracies of different models on the same dataset.

    Args:
        accuracies: Either an iterable of accuracy values in [0.0, 1.0] for each model,
            or a mapping from model name to accuracy in [0.0, 1.0].

    Returns:
        A non-negative float representing separability.

    Raises:
        ValueError: If no accuracies are provided.
    """
    # Handle Mapping by extracting values, otherwise treat as iterable
    if isinstance(accuracies, Mapping):
        accuracies = accuracies.values()
    
    accuracies = list(accuracies)
    if not accuracies:
        raise ValueError("Cannot compute separability: no accuracies provided.")

    mean_acc = sum(accuracies) / len(accuracies)
    abs_devs = [abs(a - mean_acc) for a in accuracies]
    return sum(abs_devs) / len(abs_devs)


def compute_benchmark_consistency(
    model_to_generation_accuracies: Mapping[str, Iterable[float]],
) -> float:
    """
    Compute benchmark consistency given per-model accuracies across multiple dataset generations.

    Consistency measures how stable model performance is across different dataset generations.
    The consistency of a benchmark is defined as:

        CONSISTENCY(D_gen, M) = 1 - 1/n * Σ_{i=1}^n std({performance(m_i) | D_gen,j}_{j=1}^k)

    where:
    - n is the number of models
    - k is the number of dataset generations
    - For each model m_i, we compute the standard deviation of its performance
      across k dataset generations
    - We average these standard deviations across all models
    - We subtract from 1 to get a consistency score (higher is better)

    Args:
        model_to_generation_accuracies: A mapping from model name to an iterable of
            accuracy values, where each accuracy corresponds to the model's performance
            on a different dataset generation. Each model should have the same number
            of generations (k).

    Returns:
        A float in [0.0, 1.0] representing the benchmark consistency.
        Higher values indicate more consistent performance across generations.

    Raises:
        ValueError: If no models are provided, or if models have inconsistent
            numbers of generations, or if any model has fewer than 2 generations
            (std requires at least 2 values).

    Example:
        >>> model_to_accs = {
        ...     "model1": [0.8, 0.82, 0.79],
        ...     "model2": [0.7, 0.71, 0.69],
        ... }
        >>> consistency = compute_benchmark_consistency(model_to_accs)
    """
    if not model_to_generation_accuracies:
        raise ValueError("Cannot compute consistency: no models provided.")

    # Convert to lists and validate
    model_accuracies = {
        model: list(accuracies)
        for model, accuracies in model_to_generation_accuracies.items()
    }

    # Check that all models have the same number of generations
    num_generations = len(next(iter(model_accuracies.values())))
    if num_generations < 2:
        raise ValueError(
            f"Cannot compute consistency: need at least 2 generations per model, "
            f"but found {num_generations}."
        )

    for model, accuracies in model_accuracies.items():
        if len(accuracies) != num_generations:
            raise ValueError(
                f"Inconsistent number of generations: model '{model}' has "
                f"{len(accuracies)} generations, but expected {num_generations}."
            )

    # Compute standard deviation for each model across generations
    model_stds = []
    for model, accuracies in model_accuracies.items():
        if len(accuracies) < 2:
            raise ValueError(
                f"Model '{model}' has fewer than 2 generations, cannot compute std."
            )
        std_dev = statistics.stdev(accuracies)
        model_stds.append(std_dev)

    # Average the standard deviations across all models
    mean_std = sum(model_stds) / len(model_stds)

    # Consistency = 1 - mean_std
    # Clamp to [0, 1] in case of numerical issues
    consistency = max(0.0, min(1.0, 1.0 - mean_std))
    return consistency


def compute_benchmark_novelty(
    current_accuracies: Mapping[str, float],
    prior_datasets_accuracies: List[Mapping[str, float]],
) -> float:
    """
    Compute benchmark novelty by comparing current dataset performance to prior datasets.

    Novelty measures how much new information a dataset reveals about existing models
    over existing benchmarks. The formula is:

        NOVELTY(D_c, D_prev, M) = 1 - RANKCORR(v̂_c, v_c)

    where:
    - v_c is the current dataset's accuracy vector (M×1)
    - V_prev is the prior datasets' accuracy matrix (M×N)
    - v̂_c = V_prev * θ* + b* (predicted from linear regression)
    - RANKCORR is the rank correlation (Spearman correlation)

    If the new accuracy vector v_c is spanned by existing accuracy vectors,
    RANKCORR(v_c, v̂_c) will be close to 1, resulting in low novelty.
    If v_c discovers new patterns in model performance, RANKCORR(v_c, v̂_c)
    will be low, resulting in high novelty.

    Args:
        current_accuracies: A mapping from model name to accuracy on the current
            dataset. This is v_c.
        prior_datasets_accuracies: A list of mappings, where each mapping contains
            model name to accuracy for a prior dataset. This represents V_prev.
            All mappings must contain the same set of models, and these models
            must match the models in current_accuracies.

    Returns:
        A float in [0.0, 1.0] representing the benchmark novelty.
        Higher values indicate more novel/unique performance patterns.

    Raises:
        ValueError: If no prior datasets provided, models don't match, or
            regression fails (e.g., singular matrix).

    Example:
        >>> current = {"model1": 0.8, "model2": 0.6, "model3": 0.7}
        >>> prior1 = {"model1": 0.75, "model2": 0.65, "model3": 0.72}
        >>> prior2 = {"model1": 0.78, "model2": 0.62, "model3": 0.68}
        >>> novelty = compute_benchmark_novelty(current, [prior1, prior2])
    """
    if not prior_datasets_accuracies:
        raise ValueError("Cannot compute novelty: no prior datasets provided.")

    # Get sorted model names to ensure consistent ordering
    current_models = sorted(current_accuracies.keys())
    if not current_models:
        raise ValueError("Cannot compute novelty: current_accuracies is empty.")

    # Validate that all prior datasets have the same models
    for i, prior_acc in enumerate(prior_datasets_accuracies):
        prior_models = sorted(prior_acc.keys())
        if set(prior_models) != set(current_models):
            missing = set(current_models) - set(prior_models)
            extra = set(prior_models) - set(current_models)
            raise ValueError(
                f"Prior dataset {i} has mismatched models. "
                f"Missing: {missing}, Extra: {extra}"
            )

    # Build matrices: V_prev (M×N) and v_c (M×1)
    # M = number of models, N = number of prior datasets
    num_models = len(current_models)
    num_prior = len(prior_datasets_accuracies)

    # V_prev: each column is a prior dataset's accuracies
    V_prev = np.zeros((num_models, num_prior))
    for i, prior_acc in enumerate(prior_datasets_accuracies):
        for j, model in enumerate(current_models):
            V_prev[j, i] = prior_acc[model]

    # v_c: current dataset's accuracies
    v_c = np.array([current_accuracies[model] for model in current_models])

    # Perform linear regression: v_c = V_prev * θ + b
    # We solve: min ||V_prev * θ + b - v_c||²
    # To use np.linalg.lstsq, we reformulate as: [V_prev, 1] * [θ; b] = v_c
    # where 1 is a column vector of ones (for the intercept b)
    
    # Augment design matrix with column of ones for intercept
    ones = np.ones((num_models, 1))
    X = np.hstack([V_prev, ones])
    
    try:
        # Solve using least squares: X * params = v_c
        # params = [θ; b]
        params, residuals, rank, s = np.linalg.lstsq(X, v_c, rcond=None)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Linear regression failed (singular matrix): {e}. "
            "This may happen if prior datasets are linearly dependent."
        ) from e

    # Extract θ and b
    theta = params[:-1]  # First N elements
    b = params[-1]  # Last element (intercept)

    # Compute predicted values: v̂_c = V_prev * θ + b
    v_pred = V_prev @ theta + b

    # Compute rank correlation (Spearman correlation) using scipy
    try:
        rank_corr, _p_value = spearmanr(v_c, v_pred)
    except Exception as e:
        raise ValueError(f"Rank correlation computation failed: {e}") from e

    # Handle edge cases: if correlation is NaN or invalid, novelty is 1.0
    # (NaN occurs when either array has no variation, meaning we can't predict)
    if np.isnan(rank_corr) or not np.isfinite(rank_corr):
        return 1.0

    # Novelty = 1 - rank_correlation
    # Clamp to [0, 1] in case of numerical issues (e.g., negative correlation)
    novelty = max(0.0, min(1.0, 1.0 - rank_corr))
    return novelty


