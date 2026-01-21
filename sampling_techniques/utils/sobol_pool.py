import numpy as np
from scipy.stats import qmc

def generate_sobol_points(
    bounds,
    n_samples: int = 1024,
    seed: int = 42,
):
    """
    Generate Sobol samples scaled to [mid-1, mid+1] for both dimensions,
    where mid is computed from the provided bounds.

    Parameters
    ----------
    bounds : SimpleNamespace (or any object with attributes)
        Must provide: x1_min, x1_max, x2_min, x2_max
    n_samples : int
        Number of Sobol samples (preferably a power of 2).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, 2) with columns [x1, x2],
        where each dimension is in [mid-1, mid+1].
    """
    sampler = qmc.Sobol(d=2, seed=seed)

    # Sobol works best with powers of 2; keep your existing behavior:
    samples_unit = sampler.random(n=n_samples)

    x1_mid = 0.5 * (bounds.x1_min + bounds.x1_max)
    x2_mid = 0.5 * (bounds.x2_min + bounds.x2_max)

    lower_bounds = np.array([x1_mid - 1.0, x2_mid - 1.0], dtype=float)
    upper_bounds = np.array([x1_mid + 1.0, x2_mid + 1.0], dtype=float)

    samples_scaled = qmc.scale(samples_unit, lower_bounds, upper_bounds)
    return samples_scaled
