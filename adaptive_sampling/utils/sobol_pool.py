import numpy as np
from scipy.stats import qmc

def generate_sobol_samples(
    n_samples: int = 1024,
    seed: int = 42,
):
    """
    Generate Sobol samples scaled to [-1, 1] for both dimensions.

    Parameters
    ----------
    n_samples : int
        Number of Sobol samples (preferably a power of 2).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, 2) with columns [x, y],
        where x, y ∈ [-1, 1].
    """
    sampler = qmc.Sobol(d=2, seed=seed)
    samples_unit = sampler.random(n=n_samples)

    lower_bounds = np.array([-1.0, -1.0])
    upper_bounds = np.array([ 1.0,  1.0])

    samples_scaled = qmc.scale(samples_unit, lower_bounds, upper_bounds)
    return samples_scaled


