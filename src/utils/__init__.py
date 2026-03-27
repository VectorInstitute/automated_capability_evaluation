"""Utilities package exports."""

from __future__ import annotations

from typing import Any


def _missing_quality_eval(*_args: Any, **_kwargs: Any) -> Any:
    """Raise a clear error when optional quality-evaluation deps are missing."""
    raise ImportError(
        "src.utils.quality_evaluation_utils requires optional dependencies that "
        "are not installed in this environment."
    )


compute_benchmark_consistency = _missing_quality_eval
compute_benchmark_difficulty = _missing_quality_eval
compute_benchmark_novelty = _missing_quality_eval
compute_benchmark_separability = _missing_quality_eval
compute_differential_entropy = _missing_quality_eval
compute_kl_divergence = _missing_quality_eval
compute_mdm = _missing_quality_eval
compute_mmd = _missing_quality_eval
compute_pad = _missing_quality_eval
fit_umap = _missing_quality_eval

try:
    from .quality_evaluation_utils import (
        compute_benchmark_consistency as _compute_benchmark_consistency,
    )
    from .quality_evaluation_utils import (
        compute_benchmark_difficulty as _compute_benchmark_difficulty,
    )
    from .quality_evaluation_utils import (
        compute_benchmark_novelty as _compute_benchmark_novelty,
    )
    from .quality_evaluation_utils import (
        compute_benchmark_separability as _compute_benchmark_separability,
    )
    from .quality_evaluation_utils import (
        compute_differential_entropy as _compute_differential_entropy,
    )
    from .quality_evaluation_utils import (
        compute_kl_divergence as _compute_kl_divergence,
    )
    from .quality_evaluation_utils import (
        compute_mdm as _compute_mdm,
    )
    from .quality_evaluation_utils import (
        compute_mmd as _compute_mmd,
    )
    from .quality_evaluation_utils import (
        compute_pad as _compute_pad,
    )
    from .quality_evaluation_utils import (
        fit_umap as _fit_umap,
    )
except Exception:
    pass
else:
    compute_benchmark_consistency = _compute_benchmark_consistency
    compute_benchmark_difficulty = _compute_benchmark_difficulty
    compute_benchmark_novelty = _compute_benchmark_novelty
    compute_benchmark_separability = _compute_benchmark_separability
    compute_differential_entropy = _compute_differential_entropy
    compute_kl_divergence = _compute_kl_divergence
    compute_mdm = _compute_mdm
    compute_mmd = _compute_mmd
    compute_pad = _compute_pad
    fit_umap = _fit_umap
