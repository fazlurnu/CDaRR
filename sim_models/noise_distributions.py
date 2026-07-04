"""Pluggable position-noise distributions for the ADS-L sensor layer.

A *distribution* is any callable ``(n, ci95, rng) -> ndarray`` returning ``n``
two-dimensional measurement errors in metres (east, north). This mirrors the
functional CNS noise layer used in CDaRR_FP (``sim_models/cns/distributions.py``)
so the exp3/exp4 noise-model sweep (normal / latency-bias / mixture-Gaussian)
can be reproduced here.

``NoiseModel`` supplies accuracy as a 95% confidence interval (``ci95``) and
converts to a per-axis 1-sigma with :data:`CI95_TO_STD_2D`, matching the factor
already used in ``NoiseModel`` / ``ADSL``.
"""
import numpy as np

# 95% radial CI -> per-axis 1-sigma for a 2D isotropic Gaussian:
# sqrt(-2 ln 0.05) = 2.4477...  (same constant as ADSL.CI95_TO_STD_2D)
CI95_TO_STD_2D = 2.448


def gaussian(n, ci95, rng) -> np.ndarray:
    """Zero-mean isotropic 2D normal error in metres, shape ``(n, 2)``."""
    if n == 0:
        return np.empty((0, 2))
    std = float(ci95) / CI95_TO_STD_2D
    return rng.standard_normal((n, 2)) * std


def make_mixture_gaussian(tail_ratio=3.0, tail_weight=0.1):
    """Two-component zero-mean isotropic Gaussian mixture with a preserved 2D radial ci95.

    With probability ``(1 - tail_weight)``: draw from N(0, sigma1^2 I).
    With probability ``tail_weight``:       draw from N(0, sigma2^2 I),
    sigma2 = tail_ratio * sigma1 (a wider tail component).

    sigma1 is solved by bisection so the 95th percentile of the 2D radial
    distance equals ``ci95`` exactly, preserving the same containment guarantee
    as :func:`gaussian`. Ported verbatim from CDaRR_FP.

    Constraint solved:  p*exp(-u) + (1-p)*exp(-u/k^2) = 0.05
    with u = ci95^2 / (2 sigma1^2), k = tail_ratio, p = 1 - tail_weight.
    """
    if not 0.0 < tail_weight < 1.0:
        raise ValueError(f"tail_weight must be in (0, 1), got {tail_weight}")
    if tail_ratio <= 1.0:
        raise ValueError(f"tail_ratio must be > 1, got {tail_ratio}")

    p = 1.0 - tail_weight
    k = float(tail_ratio)
    _cache = {}

    def _sigma1(ci95_val):
        key = round(ci95_val, 8)
        if key in _cache:
            return _cache[key]
        lo, hi = ci95_val * 1e-5, ci95_val
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            u = ci95_val ** 2 / (2.0 * mid ** 2)
            val = p * np.exp(-u) + (1.0 - p) * np.exp(-u / k ** 2)
            if val < 0.05:
                lo = mid
            else:
                hi = mid
        _cache[key] = 0.5 * (lo + hi)
        return _cache[key]

    def mixture_gaussian(n, ci95, rng) -> np.ndarray:
        if n == 0:
            return np.empty((0, 2))
        s1 = _sigma1(float(ci95))
        s2 = k * s1
        use_tail = rng.random(n) < tail_weight
        sigmas = np.where(use_tail, s2, s1).reshape(n, 1)
        return rng.standard_normal((n, 2)) * sigmas

    return mixture_gaussian
