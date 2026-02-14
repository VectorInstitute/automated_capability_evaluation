"""Run per-area active learning (LBO-based)."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import torch

from src.capability import Capability
from src.lbo import LBO, fit_lbo


Metric = Literal["mse", "ae"]
logger = logging.getLogger(__name__)


def group_by_area(caps: List[Capability]) -> Dict[str, List[Capability]]:
    """Group capabilities by area."""
    buckets: Dict[str, List[Capability]] = defaultdict(list)
    for c in caps:
        if getattr(c, "area", None) is None:
            raise ValueError(f"Capability {c.name} missing 'area'.")
        buckets[c.area].append(c)
    return buckets


def _cap_cost_num_tasks(cap: Capability) -> float:
    return float(len(cap.get_tasks()))


def _sum_cost_num_tasks(caps: List[Capability]) -> float:
    return sum(_cap_cost_num_tasks(c) for c in caps)


def _predict_cap_means(
    lbo_model: LBO, caps: List[Capability], embedding_name: str
) -> torch.Tensor:
    if not caps:
        return torch.empty(0)
    X = torch.stack([c.get_embedding(embedding_name) for c in caps])
    mean, _ = lbo_model.predict(X)
    return mean


def _area_gt_mean(test_caps: List[Capability], subject_llm_name: str) -> float:
    if not test_caps:
        return float("nan")
    vals = [float(c.scores[subject_llm_name]["mean"]) for c in test_caps]
    return sum(vals) / max(1, len(vals))


def _area_pred_mean(
    lbo_model: LBO, test_caps: List[Capability], embedding_name: str
) -> float:
    if not test_caps:
        return float("nan")
    means = _predict_cap_means(lbo_model, test_caps, embedding_name)
    return float(torch.mean(means).item())


def _err(gt: float, pred: float, metric: Metric) -> float:
    if metric == "mse":
        return (gt - pred) ** 2
    if metric == "ae":
        return abs(gt - pred)
    raise ValueError(f"Unsupported metric: {metric}")


def run_area_active_learning(
    area_name: str,
    train_caps_area: List[Capability],
    initial_train_area: List[Capability],
    pool_caps_area: List[Capability],
    test_caps_area: List[Capability],
    subject_llm_name: str,
    embedding_name: str,
    acquisition_function: str = "expected_variance_reduction",
    num_lbo_iterations: int = 10,
    metric: Metric = "mse",
) -> Tuple[List[Capability], Dict[str, List[float]]]:
    """
    Run AL restricted to a single area; returns (selected_caps, curves).

    curves keys: error, avg_std, cum_cost, full_eval_cost_upper.
    """
    if not initial_train_area:
        if pool_caps_area:
            initial_train_area = [pool_caps_area[0]]
            pool_caps_area = pool_caps_area[1:]
        else:
            return [], {
                "error": [],
                "avg_std": [],
                "cum_cost": [],
                "full_eval_cost_upper": [0.0],
            }

    lbo = fit_lbo(
        capabilities=initial_train_area,
        embedding_name=embedding_name,
        subject_llm_name=subject_llm_name,
        acquisition_function=acquisition_function,
    )

    gt = _area_gt_mean(test_caps_area, subject_llm_name)
    if len(test_caps_area) > 0:
        pred0 = _area_pred_mean(lbo, test_caps_area, embedding_name)
        X_test = torch.stack([c.get_embedding(embedding_name) for c in test_caps_area])
        _, std0 = lbo.predict(X_test)
        avg_std0 = float(torch.mean(std0).item())
        base_err = _err(gt, pred0, metric)
    else:
        X_test = None
        avg_std0 = float("nan")
        base_err = float("nan")

    full_eval_cost_upper = _sum_cost_num_tasks(train_caps_area)

    curves: Dict[str, List[float]] = {
        "error": [base_err],
        "avg_std": [avg_std0],
        "cum_cost": [0.0],
        "full_eval_cost_upper": [full_eval_cost_upper],
    }

    selected_caps: List[Capability] = []
    cum_cost = 0.0

    pool_x = (
        torch.stack([c.get_embedding(embedding_name) for c in pool_caps_area])
        if pool_caps_area
        else None
    )
    iters = min(num_lbo_iterations, len(pool_caps_area))
    for i in range(iters):
        logger.info(f"[{area_name}] Iter {i} of {iters}")
        if pool_x is None or pool_x.shape[0] == 0:
            break

        idx, x_sel = lbo.select_next_point(pool_x)
        chosen = pool_caps_area[idx]
        y_sel = float(chosen.scores[subject_llm_name]["mean"])

        pool_caps_area.pop(idx)
        if pool_x.shape[0] > 1:
            pool_x = torch.cat([pool_x[:idx], pool_x[idx + 1 :]], dim=0)
        else:
            pool_x = None
        lbo.update(x_sel, torch.tensor([y_sel]))

        selected_caps.append(chosen)
        cum_cost += _cap_cost_num_tasks(chosen)

        if len(test_caps_area) > 0 and X_test is not None:
            pred_iter = _area_pred_mean(lbo, test_caps_area, embedding_name)
            _, std_iter = lbo.predict(X_test)
            avg_std_iter = float(torch.mean(std_iter).item())
            err_iter = _err(gt, pred_iter, metric)
        else:
            avg_std_iter = float("nan")
            err_iter = float("nan")

        curves["error"].append(err_iter)
        curves["avg_std"].append(avg_std_iter)
        curves["cum_cost"].append(cum_cost)

    return selected_caps, curves


def plot_single_area_curves(
    area: str, curves: Dict[str, List[float]], outdir: str | Path
) -> None:
    """Plot the curves for a single area."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(curves["error"], marker="o")
    plt.xlabel("AL iteration")
    plt.ylabel("Error")
    plt.title(f"{area} — Error vs Iteration")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out / f"{area}_error_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(curves["cum_cost"], marker="o", label="Cumulative cost (#tasks)")
    if curves.get("full_eval_cost_upper"):
        ub = curves["full_eval_cost_upper"][0]
        plt.axhline(y=ub, linestyle="--", label="Full evaluation (upper bound)")
    plt.xlabel("AL iteration")
    plt.ylabel("Cost (#tasks)")
    plt.title(f"{area} — Cumulative Cost vs Iteration")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out / f"{area}_cost_curve.png", dpi=200)
    plt.close()
