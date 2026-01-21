from dataclasses import dataclass
from types import SimpleNamespace
from typing import Tuple
from dataclasses import dataclass

import numpy as np

# -----------------------------
# Small, reusable utilities
# -----------------------------
def _validate_and_stack_inputs(
    lookahead_time_array,
    resofach_array,
    safe_array,
    *,
    min_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    tla = np.asarray(lookahead_time_array, dtype=float).ravel()
    alpha = np.asarray(resofach_array, dtype=float).ravel()
    safe_raw = np.asarray(safe_array).ravel()

    if not (tla.shape == alpha.shape == safe_raw.shape):
        raise ValueError(
            f"Input shapes must match. Got tla={tla.shape}, alpha={alpha.shape}, safe={safe_raw.shape}"
        )

    mask = np.isfinite(tla) & np.isfinite(alpha)
    tla, alpha, safe_raw = tla[mask], alpha[mask], safe_raw[mask]

    if tla.size < min_samples:
        raise ValueError("Not enough valid samples after filtering NaNs/Infs.")

    y = safe_raw.astype(bool).astype(float)  # 1 safe, 0 unsafe
    X_raw = np.column_stack([tla, alpha])    # X[:,0]=tla, X[:,1]=alpha
    return X_raw, y


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    n = len(idx)
    n_val = max(1, int(round(val_fraction * n)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def _bounds_from_train(Xtr_raw: np.ndarray) -> tuple[SimpleNamespace, dict]:
    """
    Build bounds for uniform_norm, plus keep min/max/mean stats for serialization.
    Convention: x1=tla (col0), x2=alpha (col1)
    """
    tla_min = float(np.min(Xtr_raw[:, 0]))
    tla_max = float(np.max(Xtr_raw[:, 0]))
    alpha_min = float(np.min(Xtr_raw[:, 1]))
    alpha_max = float(np.max(Xtr_raw[:, 1]))

    bounds = SimpleNamespace(
        x1_min=tla_min,
        x1_max=tla_max,
        x2_min=alpha_min,
        x2_max=alpha_max,
    )

    stats = dict(
        tla_min=tla_min,
        tla_max=tla_max,
        tla_mean=0.5 * (tla_min + tla_max),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_mean=0.5 * (alpha_min + alpha_max),
    )
    return bounds, stats


def _sigmoid(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return 1.0 / (1.0 + np.exp(-u))


def _bce_loss(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# -----------------------------
# Adam optimizer as a helper
# -----------------------------
@dataclass
class AdamConfig:
    lr: float = 5e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


def _adam_optimize_tanh_feature_logit(
    alpha_hat_tr: np.ndarray,
    tla_hat_tr: np.ndarray,
    y_tr: np.ndarray,
    alpha_hat_va: np.ndarray,
    tla_hat_va: np.ndarray,
    y_va: np.ndarray,
    *,
    max_iter: int,
    l2: float,
    patience: int,
    min_delta: float,
    cfg: AdamConfig,
) -> tuple[dict, float, float]:
    """
    Optimize params for the model:

      lin = w_alpha * tanh(k * (alpha_hat - a0)) + w_tla * tla_hat + b
      p   = sigmoid(lin)

    Returns:
      best_params dict, best_val_bce, best_val_acc
    """
    # Parameters to learn
    w_alpha, k, a0, w_tla, b = 0.0, 1.0, 0.0, 0.0, 0.0

    m = np.zeros(5, dtype=float)
    v = np.zeros(5, dtype=float)

    best = dict(w_alpha=w_alpha, k=k, a0=a0, w_tla=w_tla, b=b)
    best_val_bce = float("inf")
    best_val_acc = 0.0
    no_improve = 0

    ntr = max(1, y_tr.shape[0])

    for it in range(1, max_iter + 1):
        # ---------- forward (train) ----------
        u = k * (alpha_hat_tr - a0)
        th = np.tanh(u)

        # lin = w_alpha*tanh(k*(a-a0)) + w_tla*t + b
        lin = w_alpha * th + w_tla * tla_hat_tr + b
        p = _sigmoid(lin)

        # dL/dlin
        dlin = (p - y_tr) / ntr

        # ---------- gradients ----------
        gw_alpha = float(np.sum(dlin * th))
        gw_tla   = float(np.sum(dlin * tla_hat_tr))
        gb       = float(np.sum(dlin))

        dth_du = (1.0 - th**2)
        common = dlin * (w_alpha * dth_du)  # dL/du

        gk  = float(np.sum(common * (alpha_hat_tr - a0)))
        ga0 = float(np.sum(common * (-k)))

        # L2 on linear weights
        gw_alpha += 2.0 * l2 * w_alpha
        gw_tla   += 2.0 * l2 * w_tla

        g = np.array([gw_alpha, gk, ga0, gw_tla, gb], dtype=float)

        # ---------- Adam update ----------
        m = cfg.beta1 * m + (1.0 - cfg.beta1) * g
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * (g * g)
        mhat = m / (1.0 - cfg.beta1**it)
        vhat = v / (1.0 - cfg.beta2**it)

        step = cfg.lr * mhat / (np.sqrt(vhat) + cfg.eps)

        w_alpha -= float(step[0])
        k       -= float(step[1])
        a0      -= float(step[2])
        w_tla   -= float(step[3])
        b       -= float(step[4])

        # keep k sane
        k = float(np.clip(k, 1e-3, 50.0))

        # ---------- validation ----------
        u_va = k * (alpha_hat_va - a0)
        th_va = np.tanh(u_va)
        lin_va = w_alpha * th_va + w_tla * tla_hat_va + b
        p_va = _sigmoid(lin_va)

        val_bce = _bce_loss(p_va, y_va)
        val_pred = (p_va.ravel() >= 0.5).astype(int)
        val_acc = float(np.mean(val_pred == y_va.ravel().astype(int)))

        if val_bce < best_val_bce - min_delta:
            best_val_bce = val_bce
            best_val_acc = val_acc
            best = dict(w_alpha=w_alpha, k=k, a0=a0, w_tla=w_tla, b=b)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best, float(best_val_bce), float(best_val_acc)