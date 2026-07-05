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
import math

import numpy as np

# 95% radial CI -> per-axis 1-sigma for a 2D isotropic Gaussian:
# sqrt(-2 ln 0.05) = 2.4477...  (same constant as ADSL.CI95_TO_STD_2D)
CI95_TO_STD_2D = 2.448


def gaussian(n, ci95, rng, trk_rad=None) -> np.ndarray:
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

    def mixture_gaussian(n, ci95, rng, trk_rad=None) -> np.ndarray:
        if n == 0:
            return np.empty((0, 2))
        s1 = _sigma1(float(ci95))
        s2 = k * s1
        use_tail = rng.random(n) < tail_weight
        sigmas = np.where(use_tail, s2, s1).reshape(n, 1)
        return rng.standard_normal((n, 2)) * sigmas

    return mixture_gaussian


def _radial_cdf(r, sigma_along, sigma_cross, n_grid=4001):
    """P(sqrt(X^2 + Y^2) <= r) for independent X ~ N(0, sigma_along^2),
    Y ~ N(0, sigma_cross^2). Computed by numerical integration (no closed
    form exists for sigma_along != sigma_cross)."""
    if sigma_along == sigma_cross:
        return 1.0 - math.exp(-r ** 2 / (2.0 * sigma_along ** 2))
    x = np.linspace(-r, r, n_grid)
    fx = np.exp(-0.5 * (x / sigma_along) ** 2) / (sigma_along * math.sqrt(2.0 * math.pi))
    y_bound = np.sqrt(np.maximum(r ** 2 - x ** 2, 0.0))
    z = y_bound / (sigma_cross * math.sqrt(2.0))
    erf_z = np.array([math.erf(v) for v in z])
    integrand = fx * erf_z  # 2*Phi(z*sqrt2) - 1 == erf(z)
    return float(np.trapz(integrand, x))


def make_anisotropic_gaussian(var_ratio=3.0):
    """Anisotropic 2D Gaussian position error, oriented along each aircraft's
    track: along-track variance is ``var_ratio`` times cross-track variance,
    while the overall 95% radial containment still matches ``ci95`` (same
    guarantee as :func:`gaussian` / :func:`make_mixture_gaussian`).

    Requires ``trk_rad``, the per-sample aircraft track angle in radians
    (shape ``(n,)``), to rotate the along-/cross-track components into
    east/north.

    ``sigma_cross`` is solved by bisection (via :func:`_radial_cdf`) so the
    95th percentile of the 2D radial distance equals ``ci95`` exactly.
    """
    if var_ratio <= 1.0:
        raise ValueError(f"var_ratio must be > 1, got {var_ratio}")

    std_ratio = math.sqrt(var_ratio)
    _cache = {}

    def _sigma_cross(ci95_val):
        key = round(ci95_val, 8)
        if key in _cache:
            return _cache[key]
        lo, hi = ci95_val * 1e-5, ci95_val
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            val = _radial_cdf(ci95_val, std_ratio * mid, mid)
            # CDF decreases as sigma (mid) grows, so shrink the interval the
            # opposite way from the tail-probability bisection in
            # make_mixture_gaussian.
            if val < 0.95:
                hi = mid
            else:
                lo = mid
        _cache[key] = 0.5 * (lo + hi)
        return _cache[key]

    def anisotropic_gaussian(n, ci95, rng, trk_rad) -> np.ndarray:
        if n == 0:
            return np.empty((0, 2))
        sigma_cross = _sigma_cross(float(ci95))
        sigma_along = std_ratio * sigma_cross
        along = rng.standard_normal(n) * sigma_along
        cross = rng.standard_normal(n) * sigma_cross
        trk = np.asarray(trk_rad, dtype=float)
        east  = along * np.sin(trk) + cross * np.cos(trk)
        north = along * np.cos(trk) - cross * np.sin(trk)
        return np.stack([east, north], axis=1)

    return anisotropic_gaussian


def make_anisotropic_mixture_gaussian(var_ratio=3.0, tail_ratio=3.0, tail_weight=0.1):
    """Combines :func:`make_anisotropic_gaussian` and :func:`make_mixture_gaussian`:
    a two-component Gaussian mixture where every component has the same
    along-/cross-track variance ratio ``var_ratio`` (oriented per-aircraft by
    ``trk_rad``), and the tail component's axes are both scaled up by
    ``tail_ratio`` relative to the core component (same shape, larger spread),
    drawn with probability ``tail_weight``.

    ``sigma_cross`` (of the core component) is solved by bisection so the
    overall radial 95th percentile still equals ``ci95`` -- same containment
    guarantee as the other distributions in this module.
    """
    if var_ratio <= 1.0:
        raise ValueError(f"var_ratio must be > 1, got {var_ratio}")
    if not 0.0 < tail_weight < 1.0:
        raise ValueError(f"tail_weight must be in (0, 1), got {tail_weight}")
    if tail_ratio <= 1.0:
        raise ValueError(f"tail_ratio must be > 1, got {tail_ratio}")

    std_ratio = math.sqrt(var_ratio)
    p = 1.0 - tail_weight
    k = float(tail_ratio)
    _cache = {}

    def _sigma_cross(ci95_val):
        key = round(ci95_val, 8)
        if key in _cache:
            return _cache[key]
        lo, hi = ci95_val * 1e-5, ci95_val
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            val = (p * _radial_cdf(ci95_val, std_ratio * mid, mid)
                   + (1.0 - p) * _radial_cdf(ci95_val, std_ratio * k * mid, k * mid))
            if val < 0.95:
                hi = mid
            else:
                lo = mid
        _cache[key] = 0.5 * (lo + hi)
        return _cache[key]

    def anisotropic_mixture_gaussian(n, ci95, rng, trk_rad) -> np.ndarray:
        if n == 0:
            return np.empty((0, 2))
        s1 = _sigma_cross(float(ci95))
        s2 = k * s1
        use_tail = rng.random(n) < tail_weight
        sigma_cross = np.where(use_tail, s2, s1)
        sigma_along = std_ratio * sigma_cross
        along = rng.standard_normal(n) * sigma_along
        cross = rng.standard_normal(n) * sigma_cross
        trk = np.asarray(trk_rad, dtype=float)
        east  = along * np.sin(trk) + cross * np.cos(trk)
        north = along * np.cos(trk) - cross * np.sin(trk)
        return np.stack([east, north], axis=1)

    return anisotropic_mixture_gaussian
