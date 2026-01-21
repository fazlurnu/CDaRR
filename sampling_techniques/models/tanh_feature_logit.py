from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import numpy as np

# Shared scaling utilities (exist in your repo)
from sampling_techniques.utils.scaling import uniform_norm, uniform_denorm


@dataclass
class TanhFeatureLogit:
    """
    Binary classifier with:
      X_raw[:,0] = lookahead_time (TLA)   -> x1
      X_raw[:,1] = resofach (alpha)       -> x2

    Uses shared scaling utils, but the model itself operates on *model-hat space*
    in [-1, 1] for each feature.

    If your shared uniform_norm maps raw -> [mid-1, mid+1], we convert:
        x_hat = x_scaled - mid
    so that (mid-1) -> -1 and (mid+1) -> +1.

    Decision:
      lin = w_alpha_tanh * tanh(k_alpha * (alpha_hat - alpha0_n)) + w_tla * tla_hat + b
      p   = sigmoid(lin)
    """

    # learned params
    w_alpha_tanh: float
    k_alpha: float
    alpha0_n: float
    w_tla: float
    b: float

    # normalization stats (train-set)
    alpha_mean: float
    alpha_min: float
    alpha_max: float
    tla_mean: float
    tla_min: float
    tla_max: float

    # optional metrics
    best_bce: float = float("nan")
    acc: float = float("nan")

    MODEL_VERSION: str = "1.3"

    @staticmethod
    def _sigmoid(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return 1.0 / (1.0 + np.exp(-u))

    @staticmethod
    def _check_X(X: np.ndarray, name: str = "X") -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N,2). Got {X.shape}")
        return X

    def _bounds(self) -> SimpleNamespace:
        # Column convention: X[:,0]=TLA (x1), X[:,1]=alpha (x2)
        return SimpleNamespace(
            x1_min=float(self.tla_min),
            x1_max=float(self.tla_max),
            x2_min=float(self.alpha_min),
            x2_max=float(self.alpha_max),
        )

    @staticmethod
    def _mids(bounds: SimpleNamespace) -> tuple[float, float]:
        x1_mid = 0.5 * (float(bounds.x1_min) + float(bounds.x1_max))
        x2_mid = 0.5 * (float(bounds.x2_min) + float(bounds.x2_max))
        return x1_mid, x2_mid

    def _scaled_to_hat(self, X_scaled: np.ndarray, bounds: SimpleNamespace) -> np.ndarray:
        """
        Convert scaling-space (whatever uniform_norm returns; expected [mid-1, mid+1])
        into model-hat space [-1, 1] by subtracting mids.
        """
        Xs = self._check_X(X_scaled, "X_scaled")
        x1_mid, x2_mid = self._mids(bounds)
        Xh = Xs.copy()
        Xh[:, 0] = Xh[:, 0] - x1_mid
        Xh[:, 1] = Xh[:, 1] - x2_mid
        return Xh

    def _hat_to_scaled(self, X_hat: np.ndarray, bounds: SimpleNamespace) -> np.ndarray:
        """
        Convert model-hat space [-1, 1] into scaling-space by adding mids.
        """
        Xh = self._check_X(X_hat, "X_hat")
        x1_mid, x2_mid = self._mids(bounds)
        Xs = Xh.copy()
        Xs[:, 0] = Xs[:, 0] + x1_mid
        Xs[:, 1] = Xs[:, 1] + x2_mid
        return Xs

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X = self._check_X(X_raw, "X_raw")
        bounds = self._bounds()

        # shared normalization (raw -> scaling-space)
        X_scaled = uniform_norm(bounds, X)

        # scaling-space -> model-hat [-1,1]
        X_hat = self._scaled_to_hat(X_scaled, bounds)

        tla_hat = X_hat[:, 0:1]    # (N,1) in [-1,1]
        alpha_hat = X_hat[:, 1:2]  # (N,1) in [-1,1]

        u = self.k_alpha * (alpha_hat - self.alpha0_n)
        th = np.tanh(u)
        lin = self.w_alpha_tanh * th + self.w_tla * tla_hat + self.b
        return self._sigmoid(lin).ravel()

    def predict(self, X_raw: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_raw) >= float(threshold)).astype(int)

    def denorm_inputs(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Invert scaling-space back to raw units using uniform_denorm.
        Expects X_scaled in the SAME scaling-space produced by uniform_norm
        (i.e., not model-hat space unless you first convert via _hat_to_scaled).
        """
        bounds = self._bounds()
        Xs = self._check_X(X_scaled, "X_scaled")
        return uniform_denorm(bounds, Xs)

    def hat_to_raw(self, X_hat: np.ndarray) -> np.ndarray:
        """
        Convenience: model-hat [-1,1] -> raw units.
        """
        bounds = self._bounds()
        Xs = self._hat_to_scaled(X_hat, bounds)
        return uniform_denorm(bounds, Xs)

    def raw_to_hat(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Convenience: raw units -> model-hat [-1,1].
        """
        X = self._check_X(X_raw, "X_raw")
        bounds = self._bounds()
        Xs = uniform_norm(bounds, X)
        return self._scaled_to_hat(Xs, bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "TanhFeatureLogit",
            "model_version": self.MODEL_VERSION,
            "params": {
                "w_alpha_tanh": float(self.w_alpha_tanh),
                "k_alpha": float(self.k_alpha),
                "alpha0_n": float(self.alpha0_n),
                "w_tla": float(self.w_tla),
                "b": float(self.b),
            },
            "normalization": {
                "alpha_mean": float(self.alpha_mean),
                "alpha_min": float(self.alpha_min),
                "alpha_max": float(self.alpha_max),
                "tla_mean": float(self.tla_mean),
                "tla_min": float(self.tla_min),
                "tla_max": float(self.tla_max),
            },
            "metrics": {
                "best_bce": float(self.best_bce),
                "acc": float(self.acc),
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TanhFeatureLogit":
        if d.get("type") != "TanhFeatureLogit":
            raise ValueError(f"Unexpected model type: {d.get('type')}")
        p = d["params"]
        n = d["normalization"]
        m = d.get("metrics", {})
        return cls(
            w_alpha_tanh=p["w_alpha_tanh"],
            k_alpha=p["k_alpha"],
            alpha0_n=p["alpha0_n"],
            w_tla=p["w_tla"],
            b=p["b"],
            alpha_mean=n["alpha_mean"],
            alpha_min=n["alpha_min"],
            alpha_max=n["alpha_max"],
            tla_mean=n["tla_mean"],
            tla_min=n["tla_min"],
            tla_max=n["tla_max"],
            best_bce=m.get("best_bce", float("nan")),
            acc=m.get("acc", float("nan")),
        )