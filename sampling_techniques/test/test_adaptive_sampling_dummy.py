from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sampling_techniques.models.tanh_feature_logit import TanhFeatureLogit
from sampling_techniques.models.logistic_regression import logistic_regression
from sampling_techniques.utils.pick_next_points import pick_next_points_near_p05
from sampling_techniques.utils.fit_model import fit_model

from sampling_techniques.utils.utils import extract_features_and_labels
# ----------------------------
# Data / sampling utilities
# ----------------------------

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def split_initial_next(data: Sequence[Dict[str, Any]], n_initial: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return list(data[:n_initial]), list(data[n_initial:])


def generate_ipr_from_next_data_dummy(
    model: TanhFeatureLogit,
    next_data: List[Dict[str, Any]],
    k: int = 4,
    balanced: bool = True,
) -> Tuple[List[Dict[str, Any]], List[float], List[Dict[str, Any]]]:
    """
    Select k points closest to p=0.5 (using the provided model) and return their IPRs.

    Returns:
        sample_points: selected points (dicts)
        ipr_values: overall_ipr for each selected point (same order)
        next_data_remaining: next_data with selected points removed
    """
    k = min(k, len(next_data))  # safety
    sample_points, next_data_remaining = pick_next_points_near_p05(
        model=model,
        next_data=next_data,
        k=k,
        balanced=balanced,
    )
    ipr_values = [p["sim_results"]["overall_ipr"] for p in sample_points]
    return sample_points, ipr_values, next_data_remaining


def append_samples(
    lookahead_times: List[float],
    resofach_values: List[float],
    safe_labels: List[bool],
    sample_points: Sequence[Dict[str, Any]],
    ipr_values: Sequence[float],
    ipr_threshold: float,
) -> Tuple[List[float], List[float]]:
    """
    Mutates the provided arrays by appending new samples. Returns the appended (resofach, lookahead) pairs
    to make plotting highlights easy.
    """
    next_lookahead = [d["x2_lookahead_time"] for d in sample_points]
    next_resofach = [d["x1_resofach"] for d in sample_points]

    lookahead_times.extend(next_lookahead)
    resofach_values.extend(next_resofach)
    safe_labels.extend([ipr >= ipr_threshold for ipr in ipr_values])

    return next_resofach, next_lookahead

def plot_points_and_boundary(
    lookahead_times: Sequence[float],
    resofach_values: Sequence[float],
    safe_labels: Sequence[bool],
    model: TanhFeatureLogit,
    highlight_resofach: Optional[Sequence[float]] = None,
    highlight_lookahead: Optional[Sequence[float]] = None,
    title: str = "Simulation Results Colored by IPR Threshold",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Scatter plot with the p=0.5 decision boundary for the given model.
    Model is assumed to take features ordered as: [lookahead_time, resofach].
    """
    lookahead = np.asarray(lookahead_times, dtype=float)
    resofach = np.asarray(resofach_values, dtype=float)
    safe = np.asarray(safe_labels, dtype=bool)

    colors = np.where(safe, "tab:green", "tab:red")

    plt.figure(figsize=figsize)
    plt.scatter(resofach, lookahead, c=colors, s=40)
    plt.xlabel("resofach (x1_resofach)")
    plt.ylabel("lookahead time (x2_lookahead_time)")
    plt.title(title)

    if highlight_resofach is not None and highlight_lookahead is not None and len(highlight_resofach) > 0:
        plt.scatter(
            np.asarray(highlight_resofach, dtype=float),
            np.asarray(highlight_lookahead, dtype=float),
            c="tab:blue",
            s=50,
            marker="x",
            label="Next Sample Points",
        )

    # Build a grid for contour; note feature order is [lookahead, resofach].
    lookahead_range = np.linspace(float(lookahead.min()), float(lookahead.max()), 400)
    resofach_range = np.linspace(float(resofach.min()), float(resofach.max()), 400)

    # Meshgrid in plotting order (x=resofach, y=lookahead)
    RES_grid, LOOK_grid = np.meshgrid(resofach_range, lookahead_range)

    # Flatten and stack in model-feature order: [lookahead, resofach]
    X_grid = np.vstack([LOOK_grid.ravel(), RES_grid.ravel()]).T
    probs = model.predict_proba(X_grid).reshape(LOOK_grid.shape)

    contour = plt.contour(RES_grid, LOOK_grid, probs, levels=[0.5], colors="black", linewidths=2)
    plt.clabel(contour, inline=True, fontsize=8)

    if highlight_resofach is not None and highlight_lookahead is not None and len(highlight_resofach) > 0:
        plt.legend()

    plt.show()


# ----------------------------
# Adaptive sampling loop
# ----------------------------

@dataclass(frozen=True)
class AdaptiveSamplingConfig:
    ipr_threshold: float = 0.999
    batch_size: int = 4
    balanced: bool = True
    max_samples: int = 128
    min_samples: int = 64
    acc_tol: float = 0.01
    plot_each_iteration: bool = True


def adaptive_sampling(
    initial_data: Sequence[Dict[str, Any]],
    next_data: List[Dict[str, Any]],
    cfg: AdaptiveSamplingConfig,
) -> Tuple[List[float], List[float], List[bool], TanhFeatureLogit, List[Dict[str, Any]]]:
    """
    Runs adaptive sampling until stopping criteria are met.

    Returns:
        lookahead_times, resofach_values, safe_labels, final_model, history
    """
    lookahead_times, resofach_values, safe_labels = extract_features_and_labels(initial_data, cfg.ipr_threshold)
    prev_acc, model = fit_model(lookahead_times, resofach_values, safe_labels)

    history: List[Dict[str, Any]] = [{"iter": 0, "n_samples": len(lookahead_times), "acc": prev_acc, "acc_delta": None}]

    print("Validation Accuracy (initial):", prev_acc)

    if cfg.plot_each_iteration:
        plot_points_and_boundary(
            lookahead_times=lookahead_times,
            resofach_values=resofach_values,
            safe_labels=safe_labels,
            model=model,
            title="Initial samples + p=0.5 boundary",
        )

    rep = 0
    while True:
        n_samples = len(lookahead_times)

        if n_samples >= cfg.max_samples:
            print(f"Stopping: reached max_samples={cfg.max_samples} (n_samples={n_samples})")
            break

        if not next_data:
            print("Stopping: no more next_data to sample from.")
            break

        rep += 1
        print(f"=== Adaptive Sampling Iteration {rep} (n_samples={n_samples}) ===")

        # Pick new points using current model
        sample_points, ipr_values, next_data = generate_ipr_from_next_data_dummy(
            model=model,
            next_data=next_data,
            k=cfg.batch_size,
            balanced=cfg.balanced,
        )

        # Append to training data
        next_resofach, next_lookahead = append_samples(
            lookahead_times=lookahead_times,
            resofach_values=resofach_values,
            safe_labels=safe_labels,
            sample_points=sample_points,
            ipr_values=ipr_values,
            ipr_threshold=cfg.ipr_threshold,
        )

        # Refit
        acc, model = fit_model(lookahead_times, resofach_values, safe_labels)
        acc_change = abs(acc - prev_acc)

        n_samples_after = len(lookahead_times)
        print(f"Validation Accuracy {rep}: {acc:.4f} (Δ={acc_change:.4f})")

        history.append({"iter": rep, "n_samples": n_samples_after, "acc": acc, "acc_delta": acc_change})

        # Early stop (after reaching min_samples)
        if n_samples_after >= cfg.min_samples and acc_change < cfg.acc_tol:
            print(
                f"Stopping: n_samples={n_samples_after} >= min_samples={cfg.min_samples} "
                f"and Δ={acc_change:.4f} < {cfg.acc_tol}"
            )
            break

        prev_acc = acc

        if cfg.plot_each_iteration:
            plot_points_and_boundary(
                lookahead_times=lookahead_times,
                resofach_values=resofach_values,
                safe_labels=safe_labels,
                model=model,
                highlight_resofach=next_resofach,
                highlight_lookahead=next_lookahead,
                title=f"Iteration {rep}: samples + boundary + next picks",
            )

    return lookahead_times, resofach_values, safe_labels, model, history


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    data_path = "results/example/initial_samples_results_1.5_0.5_0.95_24.json"
    n_initial_samples = 32

    cfg = AdaptiveSamplingConfig(
        ipr_threshold=0.999,
        batch_size=4,
        balanced=True,
        max_samples=128,
        min_samples=64,
        acc_tol=0.01,
        plot_each_iteration=True,  # flip to False for speed
    )

    data = load_json(data_path)
    initial_data, next_data = split_initial_next(data, n_initial_samples)

    lookahead_times, resofach_values, safe_labels, model, history = adaptive_sampling(
        initial_data=initial_data,
        next_data=next_data,
        cfg=cfg,
    )

    # Final plot
    plot_points_and_boundary(
        lookahead_times=lookahead_times,
        resofach_values=resofach_values,
        safe_labels=safe_labels,
        model=model,
        title="Final samples + p=0.5 boundary",
    )


if __name__ == "__main__":
    main()
