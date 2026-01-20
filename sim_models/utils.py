import numpy as np
from typing import Any, Optional

def _normalize_indices(indices: Any, n: int) -> np.ndarray:
    """
    Normalize indices to a 1D int array.
    - None => all indices [0..n-1]
    - np.where(mask) => tuple => take [0]
    """
    if indices is None:
        return np.arange(n, dtype=int)
    if isinstance(indices, tuple):
        indices = indices[0]
    return np.asarray(indices, dtype=int)


def _resize_1d(old: np.ndarray, n: int, fill_value: float = np.nan) -> np.ndarray:
    """Resize a 1D array preserving existing values; new entries filled with fill_value."""
    if old is None or old.size == 0:
        return np.full(n, fill_value, dtype=float)
    out = np.full(n, fill_value, dtype=old.dtype)
    k = min(old.size, n)
    out[:k] = old[:k]
    return out


def _resize_list(old: list, n: int, fill_value: str = "") -> list:
    """Resize a list preserving existing values; new entries filled with fill_value."""
    if old is None or len(old) == 0:
        return [fill_value for _ in range(n)]
    if len(old) > n:
        return old[:n]
    if len(old) < n:
        return old + [fill_value for _ in range(n - len(old))]
    return old