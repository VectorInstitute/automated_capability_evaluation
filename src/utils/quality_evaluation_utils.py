"""Utility functions for evaluating benchmark-level metrics."""

from __future__ import annotations

from typing import Iterable, Mapping, Union


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


