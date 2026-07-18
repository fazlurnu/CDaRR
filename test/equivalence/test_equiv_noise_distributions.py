"""L0 equivalence: cns.distributions vs. sim_models.noise_distributions.

cns/distributions.py is a *verbatim* copy (byte-identical, diff-confirmed at
commit time) rather than a re-transcription, so this is mostly a regression
guard: same-seeded Generator into old and new must produce identical arrays
AND leave the RNG in the identical final state (same draw count) -- per
refactor_fp.md's L0 spec, this matters because a noise-model swap changing
draw counts would desynchronize the rest of a run's RNG stream (see
KNOWN_ISSUES.md's RNG ledger discussion).
"""
import numpy as np
import pytest

from cns import distributions as new
from sim_models import noise_distributions as old

CI95_LEVELS = [3.0, 10.0, 15.0]
N = 5000


def _same_state(rng_a, rng_b):
    return rng_a.bit_generator.state == rng_b.bit_generator.state


@pytest.mark.fast
@pytest.mark.parametrize("ci95", CI95_LEVELS)
def test_gaussian_matches(ci95):
    rng_old, rng_new = np.random.default_rng(1), np.random.default_rng(1)
    a = old.gaussian(N, ci95, rng_old)
    b = new.gaussian(N, ci95, rng_new)
    assert np.array_equal(a, b)
    assert _same_state(rng_old, rng_new)


@pytest.mark.fast
def test_gaussian_matches_empty():
    rng_old, rng_new = np.random.default_rng(2), np.random.default_rng(2)
    assert np.array_equal(old.gaussian(0, 10.0, rng_old), new.gaussian(0, 10.0, rng_new))
    assert _same_state(rng_old, rng_new)


@pytest.mark.fast
@pytest.mark.parametrize("ci95", CI95_LEVELS)
@pytest.mark.parametrize("tail_ratio,tail_weight", [(3.0, 0.10), (2.0, 0.25)])
def test_mixture_gaussian_matches(ci95, tail_ratio, tail_weight):
    old_f = old.make_mixture_gaussian(tail_ratio, tail_weight)
    new_f = new.make_mixture_gaussian(tail_ratio, tail_weight)
    rng_old, rng_new = np.random.default_rng(3), np.random.default_rng(3)
    a = old_f(N, ci95, rng_old)
    b = new_f(N, ci95, rng_new)
    assert np.array_equal(a, b)
    assert _same_state(rng_old, rng_new)


@pytest.mark.fast
@pytest.mark.parametrize("ci95", CI95_LEVELS)
@pytest.mark.parametrize("var_ratio", [9.0, 4.0])
def test_anisotropic_gaussian_matches(ci95, var_ratio):
    old_f = old.make_anisotropic_gaussian(var_ratio)
    new_f = new.make_anisotropic_gaussian(var_ratio)
    trk_rad = np.random.default_rng(0).uniform(0, 2 * np.pi, N)
    rng_old, rng_new = np.random.default_rng(4), np.random.default_rng(4)
    a = old_f(N, ci95, rng_old, trk_rad)
    b = new_f(N, ci95, rng_new, trk_rad)
    assert np.array_equal(a, b)
    assert _same_state(rng_old, rng_new)


@pytest.mark.fast
@pytest.mark.parametrize("ci95", CI95_LEVELS)
def test_anisotropic_mixture_gaussian_matches(ci95):
    old_f = old.make_anisotropic_mixture_gaussian(9.0, 3.0, 0.10)
    new_f = new.make_anisotropic_mixture_gaussian(9.0, 3.0, 0.10)
    trk_rad = np.random.default_rng(0).uniform(0, 2 * np.pi, N)
    rng_old, rng_new = np.random.default_rng(5), np.random.default_rng(5)
    a = old_f(N, ci95, rng_old, trk_rad)
    b = new_f(N, ci95, rng_new, trk_rad)
    assert np.array_equal(a, b)
    assert _same_state(rng_old, rng_new)


@pytest.mark.fast
def test_bisection_sigmas_match():
    """The internal _sigma1/_sigma_cross bisection caches -- exposed indirectly
    via the first-drawn sample's scale -- must match to full precision."""
    for ci95 in CI95_LEVELS:
        old_f = old.make_mixture_gaussian(3.0, 0.10)
        new_f = new.make_mixture_gaussian(3.0, 0.10)
        r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
        a = old_f(20000, ci95, r1)
        b = new_f(20000, ci95, r2)
        # Empirical std as a proxy for the bisected sigma1 matching to high precision.
        assert round(float(np.std(a)), 12) == round(float(np.std(b)), 12)
