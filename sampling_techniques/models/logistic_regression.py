# sampling_techniques/models/logistic_regression.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sampling_techniques.models.tanh_feature_logit import TanhFeatureLogit
from sampling_techniques.utils.scaling import uniform_norm

from sampling_techniques.models.utils import (
    _validate_and_stack_inputs,
    _train_val_split,
    _bounds_from_train,
)

from sampling_techniques.models.utils import (
    _adam_optimize_tanh_feature_logit, AdamConfig
)

# -----------------------------
# Public API (same signature)
# -----------------------------
def logistic_regression(
    lookahead_time_array,
    resofach_array,
    safe_array,   # bool target: True if IPR >= 0.999 else False
    *,
    seed: int = 0,
    val_fraction: float = 0.2,
    max_iter: int = 5000,
    lr: float = 5e-2,
    l2: float = 1e-4,
    patience: int = 250,
    min_delta: float = 1e-6,
) -> Tuple[float, dict]:
    """
    Fit TanhFeatureLogit on:
      X_raw[:,0] = lookahead_time (TLA)
      X_raw[:,1] = resofach (alpha)
    target y = safe_array (bool)

    Returns
    -------
    acc : float
      Validation accuracy at threshold 0.5.
    params : dict
      JSON-safe serialization compatible with TanhFeatureLogit.from_dict(params).
    """
    X_raw, y = _validate_and_stack_inputs(lookahead_time_array, resofach_array, safe_array)
    Xtr_raw, ytr, Xva_raw, yva = _train_val_split(X_raw, y, seed=seed, val_fraction=val_fraction)

    # bounds/stats from train split only
    bounds, stats = _bounds_from_train(Xtr_raw)

    # shared scaling util
    Xtr_scaled = uniform_norm(bounds, Xtr_raw)
    Xva_scaled = uniform_norm(bounds, Xva_raw)

    # Convert scaling-space -> model-hat [-1,1] by subtracting mids
    # (This matches the model refactor you adopted in TanhFeatureLogit.)
    x1_mid = 0.5 * (bounds.x1_min + bounds.x1_max)
    x2_mid = 0.5 * (bounds.x2_min + bounds.x2_max)

    t_tr = (Xtr_scaled[:, 0:1] - x1_mid)  # tla_hat
    a_tr = (Xtr_scaled[:, 1:2] - x2_mid)  # alpha_hat
    t_va = (Xva_scaled[:, 0:1] - x1_mid)
    a_va = (Xva_scaled[:, 1:2] - x2_mid)

    ytr_c = ytr.reshape(-1, 1)
    yva_c = yva.reshape(-1, 1)

    best, best_val_bce, best_val_acc = _adam_optimize_tanh_feature_logit(
        a_tr, t_tr, ytr_c,
        a_va, t_va, yva_c,
        max_iter=max_iter,
        l2=l2,
        patience=patience,
        min_delta=min_delta,
        cfg=AdamConfig(lr=lr),
    )

    model = TanhFeatureLogit(
        w_alpha_tanh=float(best["w_alpha"]),
        k_alpha=float(best["k"]),
        alpha0_n=float(best["a0"]),
        w_tla=float(best["w_tla"]),
        b=float(best["b"]),
        alpha_mean=float(stats["alpha_mean"]),
        alpha_min=float(stats["alpha_min"]),
        alpha_max=float(stats["alpha_max"]),
        tla_mean=float(stats["tla_mean"]),
        tla_min=float(stats["tla_min"]),
        tla_max=float(stats["tla_max"]),
        best_bce=float(best_val_bce),
        acc=float(best_val_acc),
    )

    return model.acc, model.to_dict()
