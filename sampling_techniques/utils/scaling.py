import numpy as np

def uniform_norm(bound, X_raw: np.ndarray) -> np.ndarray:
    """
    Normalize raw inputs from [min, max] to [mid-1, mid+1].

    Column convention:
        X_raw[:, 0] = x1
        X_raw[:, 1] = x2

    Returns Xn with:
        Xn[:, 0] = x1_scaled in [x1_mid-1, x1_mid+1]
        Xn[:, 1] = x2_scaled in [x2_mid-1, x2_mid+1]
    """
    X = np.asarray(X_raw, dtype=float)
    Xn = X.copy()

    r_x1 = (bound.x1_max - bound.x1_min) + 1e-12
    r_x2 = (bound.x2_max - bound.x2_min) + 1e-12

    x1_mid = 0.5 * (bound.x1_min + bound.x1_max)
    x2_mid = 0.5 * (bound.x2_min + bound.x2_max)

    x1_lo, x1_hi = x1_mid - 1.0, x1_mid + 1.0
    x2_lo, x2_hi = x2_mid - 1.0, x2_mid + 1.0

    # scale x1 (raw col 0) from [x1_min, x1_max] -> [x1_lo, x1_hi]
    Xn[:, 0] = (X[:, 0] - bound.x1_min) / r_x1 * (x1_hi - x1_lo) + x1_lo

    # scale x2 (raw col 1) from [x2_min, x2_max] -> [x2_lo, x2_hi]
    Xn[:, 1] = (X[:, 1] - bound.x2_min) / r_x2 * (x2_hi - x2_lo) + x2_lo

    return Xn


def uniform_denorm(bound, X_scaled: np.ndarray) -> np.ndarray:
    """
    Inverse of uniform_norm.

    Inputs:
        X_scaled[:, 0] = x1_scaled in [x1_mid-1, x1_mid+1]
        X_scaled[:, 1] = x2_scaled in [x2_mid-1, x2_mid+1]

    Returns:
        X_raw[:, 0] = x1
        X_raw[:, 1] = x2
    """
    Xn = np.asarray(X_scaled, dtype=float)
    X = Xn.copy()

    r_x1 = (bound.x1_max - bound.x1_min) + 1e-12
    r_x2 = (bound.x2_max - bound.x2_min) + 1e-12

    x1_mid = 0.5 * (bound.x1_min + bound.x1_max)
    x2_mid = 0.5 * (bound.x2_min + bound.x2_max)

    x1_lo, x1_hi = x1_mid - 1.0, x1_mid + 1.0
    x2_lo, x2_hi = x2_mid - 1.0, x2_mid + 1.0

    # recover x1 from x1_scaled
    X[:, 0] = (Xn[:, 0] - x1_lo) / (x1_hi - x1_lo) * r_x1 + bound.x1_min

    # recover x2 from x2_scaled
    X[:, 1] = (Xn[:, 1] - x2_lo) / (x2_hi - x2_lo) * r_x2 + bound.x2_min

    return X