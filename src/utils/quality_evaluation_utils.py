"""Utility functions for evaluating benchmark-level metrics."""

from __future__ import annotations

import statistics
import warnings
from typing import Iterable, List, Mapping, Optional, Union

import numpy as np
from scipy.stats import spearmanr
from scipy.special import digamma, gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (
    polynomial_kernel,
    rbf_kernel,
    laplacian_kernel,
    linear_kernel,
    sigmoid_kernel,
)
from sklearn.neighbors import NearestNeighbors
import kmedoids
from sklearn.metrics import pairwise_distances

# Optional UMAP import
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    UMAP = None


# Source paper: AutoBencher - https://arxiv.org/abs/2407.08351
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


# Source paper: AutoBencher - https://arxiv.org/abs/2407.08351
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


# Source paper: Data Swarms - https://arxiv.org/abs/2506.00741
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


# Source paper: AutoBencher - https://arxiv.org/abs/2407.08351
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


# ===========================
# ---- Diversity Metrics (PAD, MMD, MDM)
# ===========================

# Source paper: SynQue - https://arxiv.org/abs/2511.03928
def compute_pad(
    x_syn_emb: np.ndarray,
    x_real_emb: np.ndarray,
    classifier_name: str = "LogisticRegression",
) -> float:
    """
    Compute the Proxy-A-Distance (PAD) between two sets of embeddings.
    
    PAD measures the distance between synthetic and real data distributions
    by training a classifier to distinguish between them. Lower values indicate
    more similar distributions.
    
    Args:
        x_syn_emb: Embeddings of synthetic data, shape (n_samples, n_features)
        x_real_emb: Embeddings of real data, shape (n_samples, n_features)
        classifier_name: Classifier to use ("LogisticRegression", "RandomForest", "MLP")
    
    Returns:
        float: PAD value (typically in range [0, 2], lower is better)
    """
    y_syn_train = np.zeros(len(x_syn_emb))
    y_real_train = np.ones(len(x_real_emb))
    x_train = np.concatenate([x_syn_emb, x_real_emb], axis=0)
    y_train = np.concatenate([y_syn_train, y_real_train], axis=0)
    
    # Split into train/validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    # Classifier
    if classifier_name == "LogisticRegression":
        classifier = LogisticRegression(random_state=42, max_iter=1000)
    elif classifier_name == "RandomForest":
        classifier = RandomForestClassifier(random_state=42)
    elif classifier_name == "MLP":
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            max_iter=200,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    
    classifier.fit(x_train, y_train)
    y_pred_proba = classifier.predict_proba(x_val)[:, 1]
    average_loss = np.mean(np.abs(y_pred_proba - y_val))
    return 2 * (1 - 2 * average_loss)


# Source paper: SynQue - https://arxiv.org/abs/2511.03928
def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "polynomial",
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1,
) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two samples: X and Y.
    
    MMD measures the distance between two distributions in a reproducing kernel
    Hilbert space. Lower values indicate more similar distributions.
    
    Args:
        X: First sample, shape (n_samples_X, n_features)
        Y: Second sample, shape (n_samples_Y, n_features)
        kernel: Kernel name ("polynomial", "rbf", "laplacian", "linear", "sigmoid")
        degree: Degree for polynomial kernel (default: 3)
        gamma: Gamma parameter for kernels (default: None, auto)
        coef0: Coef0 for polynomial/sigmoid kernel
    
    Returns:
        float: MMD value (non-negative, lower is better)
    """
    kernel = kernel.lower() if isinstance(kernel, str) else kernel
    if kernel == "polynomial":
        kfunc = polynomial_kernel
        XX = kfunc(X, X, degree=degree, gamma=gamma, coef0=coef0)
        YY = kfunc(Y, Y, degree=degree, gamma=gamma, coef0=coef0)
        XY = kfunc(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == "rbf":
        kfunc = rbf_kernel
        XX = kfunc(X, X, gamma=gamma)
        YY = kfunc(Y, Y, gamma=gamma)
        XY = kfunc(X, Y, gamma=gamma)
    elif kernel == "laplacian":
        kfunc = laplacian_kernel
        XX = kfunc(X, X, gamma=gamma)
        YY = kfunc(Y, Y, gamma=gamma)
        XY = kfunc(X, Y, gamma=gamma)
    elif kernel == "linear":
        kfunc = linear_kernel
        XX = kfunc(X, X)
        YY = kfunc(Y, Y)
        XY = kfunc(X, Y)
    elif kernel == "sigmoid":
        kfunc = sigmoid_kernel
        XX = kfunc(X, X, gamma=gamma, coef0=coef0)
        YY = kfunc(Y, Y, gamma=gamma, coef0=coef0)
        XY = kfunc(X, Y, gamma=gamma, coef0=coef0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)


# Source paper: SynQue - https://arxiv.org/abs/2511.03928
def compute_mdm(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    metric: str = "euclidean",
) -> float:
    """
    Compute the mean distance of points in each cluster to its medoid, then average across clusters.
    
    MDM measures the internal diversity/coherence of a set of embeddings by clustering
    them and computing the average distance to cluster medoids. Lower values indicate
    more coherent/diverse clusters.
    
    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters/medoids to use
        metric: Distance metric for KMedoids ('euclidean', 'cosine', etc.)
    
    Returns:
        float: Mean distance to medoid (averaged over all clusters)
    """
    n_samples = len(embeddings)
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples)
    
    diss = pairwise_distances(embeddings, metric=metric)
    pam_result = kmedoids.fasterpam(diss, n_clusters, random_state=42)
    labels = pam_result.labels
    medoid_indices = pam_result.medoids
    
    total_dist = 0.0
    for i, medoid_idx in enumerate(medoid_indices):
        cluster_points_idx = np.where(labels == i)[0]
        if len(cluster_points_idx) == 0:
            continue
        dists = diss[cluster_points_idx, medoid_idx]
        total_dist += np.mean(dists)
    return total_dist / n_clusters


# ===========================
# ---- Information-Theoretic Metrics (Entropy, KL-Divergence)
# ===========================

def fit_umap_shared(
    embeddings_list: List[np.ndarray],
    n_components: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> List[np.ndarray]:
    """
    Fit UMAP on the concatenation of all embedding arrays, then split back (InfoSynth-style).

    This ensures entropy and KL divergence are comparable across datasets by using
    a single shared low-dimensional space.

    Args:
        embeddings_list: List of embedding matrices, each shape (n_i, n_features).
        n_components: UMAP target dimension.
        n_neighbors: Number of neighbors for UMAP (default: 15).
        min_dist: Minimum distance for UMAP (default: 0.1).
        metric: Distance metric for UMAP (default: "cosine").

    Returns:
        List of reduced embedding arrays in the same order as embeddings_list.
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP is required. Install it with: pip install umap-learn"
        )
    if not embeddings_list:
        return []
    counts = [emb.shape[0] for emb in embeddings_list]
    split_indices = np.cumsum(counts)[:-1]
    combined = np.vstack(embeddings_list)
    if combined.shape[1] <= n_components:
        return [emb.copy() for emb in embeddings_list]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42,
        )
        reduced = umap_model.fit_transform(combined)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    eps = 1e-12
    reduced = reduced / (norms + eps)
    return np.split(reduced, split_indices, axis=0)


# Source paper: InfoSyth - https://arxiv.org/abs/2601.00575
def compute_differential_entropy(embeddings: np.ndarray, k: int = 4) -> float:
    """
    Compute the differential entropy of a set of embeddings using k-nearest neighbors.

    Differential entropy measures the diversity/uncertainty in the embedding distribution.
    Higher values indicate more diverse data. For a shared space across datasets, apply
    UMAP (e.g. fit_umap_shared) to embeddings before calling this function.

    This implementation uses the k-NN estimator for differential entropy:
        H(X) ≈ digamma(N) - digamma(k) + log(volume) + d * mean(log(eps))

    where:
    - N is the number of samples
    - d is the embedding dimension
    - k is the number of neighbors
    - eps is the distance to the k-th nearest neighbor

    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features)
        k: Number of nearest neighbors to use (default: 4)

    Returns:
        float: Differential entropy value (higher is more diverse)
    """
    N, d = embeddings.shape
    if N < k + 1:
        raise ValueError(
            f"Cannot compute entropy: need at least {k + 1} samples, but got {N}."
        )
    
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    eps = distances[:, -1]
    eps[eps == 0] = np.nextafter(0, 1)
    
    log_vol = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)
    entropy = digamma(N) - digamma(k) + log_vol + d * np.mean(np.log(eps))
    return float(entropy)


# Source paper: InfoSyth - https://arxiv.org/abs/2601.00575
def compute_kl_divergence(
    p_embeddings: np.ndarray,
    q_embeddings: np.ndarray,
    k: int = 4,
    eps: float = 1e-10,
) -> float:
    """
    Compute the KL divergence between two sets of embeddings using k-nearest neighbors.

    KL divergence measures how different distribution P is from distribution Q.
    Higher values indicate more novelty (P is more different from Q). For a shared
    space, apply UMAP (e.g. fit_umap_shared) to [P, Q] before calling this function.

    This implementation uses the k-NN estimator for KL divergence:
        KL(P||Q) ≈ (d/n) * sum(log(nu/rho)) + log(m/(n-1))

    where:
    - P is the distribution of p_embeddings (n samples)
    - Q is the distribution of q_embeddings (m samples)
    - d is the embedding dimension
    - rho is the distance to the k-th nearest neighbor in P
    - nu is the distance to the k-th nearest neighbor in Q

    Args:
        p_embeddings: Embeddings of distribution P, shape (n_samples_p, n_features)
        q_embeddings: Embeddings of distribution Q, shape (n_samples_q, n_features)
        k: Number of nearest neighbors to use (default: 4)
        eps: Small epsilon to avoid division by zero (default: 1e-10)

    Returns:
        float: KL divergence value (higher is more novel/different)
    """
    n, d = p_embeddings.shape
    m, _ = q_embeddings.shape
    
    if n < k + 1:
        raise ValueError(
            f"Cannot compute KL divergence: P needs at least {k + 1} samples, but got {n}."
        )
    if m < k:
        raise ValueError(
            f"Cannot compute KL divergence: Q needs at least {k} samples, but got {m}."
        )
    
    # Find k-th nearest neighbor in P for each point in P
    nbrs_p = NearestNeighbors(n_neighbors=k + 1).fit(p_embeddings)
    rho = np.maximum(nbrs_p.kneighbors(p_embeddings)[0][:, k], eps)
    
    # Find k-th nearest neighbor in Q for each point in P
    nbrs_q = NearestNeighbors(n_neighbors=k).fit(q_embeddings)
    nu = np.maximum(nbrs_q.kneighbors(p_embeddings)[0][:, k - 1], eps)
    
    kl_div = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1))
    return float(kl_div)


