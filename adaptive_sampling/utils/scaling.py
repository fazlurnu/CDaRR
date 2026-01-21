import numpy as np

def uniform_norm(self, X_raw: np.ndarray) -> np.ndarray:
    """
    Normalize raw inputs using min-max scaling to [-1, 1].

    X_raw[:, 0] = x1
    X_raw[:, 1] = x2

    Returns Xn with:
        Xn[:, 0] = x1_hat
        Xn[:, 1] = x2_hat
    """
    
    X = np.asarray(X_raw, dtype=float)
    Xn = X.copy()

    r_x1 = (self.x1_max - self.x1_min) + 1e-12
    r_x2 = (self.x2_max - self.x2_min) + 1e-12

    # x1_hat from raw col 1
    Xn[:, 0] = 2.0 * (X[:, 1] - self.x1_min) / r_x1 - 1.0

    # x2_hat from raw col 0
    Xn[:, 1] = 2.0 * (X[:, 0] - self.x2_min) / r_x2 - 1.0

    return Xn

def uniform_denorm(self, X_scaled: np.ndarray) -> np.ndarray:
    """
    Inverse of uniform_norm.

    X_scaled[:, 0] = x1_hat ∈ [-1, 1]
    X_scaled[:, 1] = x2_hat ∈ [-1, 1]

    Returns:
        X_raw[:, 0] = x1
        X_raw[:, 1] = x2
    """
    Xn = np.asarray(X_scaled, dtype=float)
    X = Xn.copy()

    r_x1 = (self.x1_max - self.x1_min) + 1e-12
    r_x2 = (self.x2_max - self.x2_min) + 1e-12

    # recover x1 from x1_hat
    X[:, 0] = 0.5 * (Xn[:, 0] + 1.0) * r_x1 + self.x1_min

    # recover x2 from x2_hat
    X[:, 1] = 0.5 * (Xn[:, 1] + 1.0) * r_x2 + self.x2_min

    return X