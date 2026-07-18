"""L0 equivalence: crr.prob_math vs. the legacy math functions in
sim_models/crr_resumenav_probabilistic_ftr.py.

These are dense log-space numerical routines (overflow/underflow-safe angular
density integration) copied by hand into the new module -- a transcription
slip here would be easy to make and easy to miss without exact-value fuzzing
across many geometries, including the harsh regimes the log-space tricks
exist for (high SNR, near-zero b, x=0, both Ktheta values used in production).
"""
import numpy as np
import pytest

from crr import prob_math as new
from sim_models import crr_resumenav_probabilistic_ftr as old

SEED = 20260717
N_FUZZ = 1000


def _rng():
    return np.random.default_rng(SEED)


@pytest.mark.fast
def test_Phi_matches():
    rng = _rng()
    x = rng.uniform(-50, 50, size=N_FUZZ)
    assert np.array_equal(old.Phi(x), new.Phi(x))
    # scalar path
    for v in (-10.0, -1e-9, 0.0, 1e-9, 10.0, 1e6, -1e6):
        assert old.Phi(v) == new.Phi(v)


@pytest.mark.fast
def test_to_cov_matches():
    rng = _rng()
    cases = [None, 0.0, 3.0, -2.5]
    for c in cases:
        old_r, new_r = old._to_cov(c), new._to_cov(c)
        assert np.array_equal(old_r, new_r), f"scalar case {c}"
    for _ in range(50):
        vec = rng.uniform(0.1, 20.0, size=2)
        assert np.array_equal(old._to_cov(vec), new._to_cov(vec))
        mat = rng.uniform(-5, 5, size=(2, 2))
        assert np.array_equal(old._to_cov(mat), new._to_cov(mat))
    with pytest.raises(ValueError):
        new._to_cov(np.zeros(3))


@pytest.mark.fast
def test_regularize_spd_matches():
    rng = _rng()
    for _ in range(200):
        S = rng.uniform(-10, 10, size=(2, 2))
        eps = rng.choice([1e-9, 1e-10, 1e-6])
        assert np.array_equal(old._regularize_spd(S, eps=eps), new._regularize_spd(S, eps=eps))


def _random_mu_sigma(rng, snr_scale=1.0):
    mu = rng.uniform(-5, 5, size=2) * snr_scale
    std = rng.uniform(0.1, 10.0, size=2)
    corr = rng.uniform(-0.9, 0.9)
    Sigma = np.array([[std[0] ** 2, corr * std[0] * std[1]],
                       [corr * std[0] * std[1], std[1] ** 2]])
    return mu, Sigma


@pytest.mark.fast
def test_log_p_theta_projected_normal_matches():
    rng = _rng()
    theta = np.linspace(0.0, 2 * np.pi, 256, endpoint=False)
    for _ in range(N_FUZZ // 4):
        mu, Sigma = _random_mu_sigma(rng)
        old_v = old.log_p_theta_projected_normal(theta, mu, Sigma)
        new_v = new.log_p_theta_projected_normal(theta, mu, Sigma)
        assert np.array_equal(old_v, new_v)

    # Harsh regime: high SNR (|mu|/sigma >> 1) -- the log-space code path exists for this.
    for _ in range(N_FUZZ // 4):
        mu, Sigma = _random_mu_sigma(rng, snr_scale=1000.0)
        Sigma = Sigma * 1e-6  # tiny variance -> huge SNR
        old_v = old.log_p_theta_projected_normal(theta, mu, Sigma)
        new_v = new.log_p_theta_projected_normal(theta, mu, Sigma)
        assert np.array_equal(old_v, new_v)

    # Harsh regime: near-zero b (mu close to the origin, isotropic-ish Sigma).
    for _ in range(N_FUZZ // 4):
        mu = rng.uniform(-1e-8, 1e-8, size=2)
        std = rng.uniform(0.5, 5.0, size=2)
        Sigma = np.diag(std ** 2)
        old_v = old.log_p_theta_projected_normal(theta, mu, Sigma)
        new_v = new.log_p_theta_projected_normal(theta, mu, Sigma)
        assert np.array_equal(old_v, new_v)


@pytest.mark.fast
def test_p_theta_projected_normal_matches():
    rng = _rng()
    theta = np.linspace(0.0, 2 * np.pi, 100, endpoint=False)
    for _ in range(200):
        mu, Sigma = _random_mu_sigma(rng)
        assert np.array_equal(
            old.p_theta_projected_normal(theta, mu, Sigma),
            new.p_theta_projected_normal(theta, mu, Sigma),
        )


@pytest.mark.fast
@pytest.mark.parametrize("Ktheta", [248, 256])
def test_analytical_dcpa_prob_gt_matches(Ktheta):
    rng = _rng()
    for _ in range(N_FUZZ // 2):
        x = rng.uniform(0.0, 200.0)
        mu_r = rng.uniform(-500, 500, size=2)
        Sigma_r, _ = _random_mu_sigma(rng)  # reuse shape helper for a plausible cov
        _, Sigma_r = _random_mu_sigma(rng)
        mu_v = rng.uniform(-30, 30, size=2)
        _, Sigma_v = _random_mu_sigma(rng)
        old_p = old.analytical_dcpa_prob_gt(x, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=Ktheta)
        new_p = new.analytical_dcpa_prob_gt(x, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=Ktheta)
        assert old_p == new_p, f"mismatch at x={x}, Ktheta={Ktheta}"

    # x = 0 edge case
    mu_r, Sigma_r = np.array([10.0, 5.0]), np.eye(2) * 4.0
    mu_v, Sigma_v = np.array([2.0, -1.0]), np.eye(2) * 1.0
    assert old.analytical_dcpa_prob_gt(0.0, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=Ktheta) == \
        new.analytical_dcpa_prob_gt(0.0, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=Ktheta)

    # x < 0 edge case (short-circuit return 1.0)
    assert old.analytical_dcpa_prob_gt(-5.0, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=Ktheta) == \
        new.analytical_dcpa_prob_gt(-5.0, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=Ktheta) == 1.0


@pytest.mark.fast
def test_analytical_past_cpa_prob_matches():
    rng = _rng()
    for _ in range(N_FUZZ // 2):
        mu_rel = rng.uniform(-100, 100, size=2)
        nu_rel = rng.uniform(-20, 20, size=2)
        _, Sigma_rel = _random_mu_sigma(rng)
        _, Sigma_nu = _random_mu_sigma(rng)
        old_p = old.analytical_past_cpa_prob(mu_rel, Sigma_rel, nu_rel, Sigma_nu)
        new_p = new.analytical_past_cpa_prob(mu_rel, Sigma_rel, nu_rel, Sigma_nu)
        assert old_p == new_p

    # eps short-circuit branch (s2 ~ 0)
    zero_cov = np.zeros((2, 2))
    for m_dir in (np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 0.0])):
        old_p = old.analytical_past_cpa_prob(m_dir, zero_cov, np.array([1.0, 0.0]), zero_cov)
        new_p = new.analytical_past_cpa_prob(m_dir, zero_cov, np.array([1.0, 0.0]), zero_cov)
        assert old_p == new_p
