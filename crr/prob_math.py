''' Core probability math for the probabilistic FTR recovery criterion.

Fresh, verbatim copy of the math functions in
sim_models/crr_resumenav_probabilistic_ftr.py (not an import -- see
cd/statebased.py's docstring for why; these are dense log-space numerical
routines where a transcription slip would be easy to make and hard to spot,
so test_equiv_prob_math.py fuzzes every function against the original across
many random geometries before anything downstream trusts this module).
'''
import math

import numpy as np

try:
    from scipy.special import erf as sp_erf
    _erf = sp_erf
except Exception:
    _erf = np.vectorize(math.erf)

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def Phi(x):
    """Standard normal CDF, vectorized."""
    return 0.5 * (1.0 + _erf(np.asarray(x) / SQRT2))

def _to_cov(s, dim=2):
    """
    Convert scalar/std-vector/cov-matrix into a (dim,dim) covariance matrix.
    Accepts:
      - scalar std: s -> (s^2) I
      - (dim,) stds: -> diag(stds^2)
      - (dim,dim) cov: -> itself
    """
    if s is None:
        return np.zeros((dim, dim), float)

    if np.isscalar(s):
        return (float(s) ** 2) * np.eye(dim)

    s = np.asarray(s, float)
    if s.shape == (dim,):
        return np.diag(s ** 2)
    if s.shape == (dim, dim):
        return s

    raise ValueError(f"Invalid covariance/std shape: {s.shape}")

def _regularize_spd(S, eps=1e-9):
    """Make covariance numerically SPD-ish (adds eps*I)."""
    S = np.asarray(S, float).reshape(2, 2)
    return 0.5 * (S + S.T) + eps * np.eye(2)

def log_p_theta_projected_normal(theta, mu, Sigma):
    """
    Log of the projected-normal angular density p_Theta(theta).

    Numerically stable version that works in log-space to avoid the
    exp(-large) * exp(+large) overflow/underflow that occurs when
    the velocity SNR (|mu|/sigma) is high.

    theta : array in [0, 2pi)
    mu    : (2,) mean velocity
    Sigma : (2,2) velocity covariance (SPD)

    Returns log p(theta), shape (K,).
    """
    mu = np.asarray(mu, float).reshape(2)
    Sigma = _regularize_spd(Sigma, eps=1e-10)

    detS = float(np.linalg.det(Sigma))
    if detS <= 0:
        Sigma = _regularize_spd(Sigma, eps=1e-6)
        detS = float(np.linalg.det(Sigma))
        if detS <= 0:
            raise ValueError("Sigma_v must be positive definite.")

    Q = np.linalg.inv(Sigma)
    c = float(mu @ Q @ mu)

    u = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2,K)
    Qu = Q @ u                                            # (2,K)
    a = np.sum(u * Qu, axis=0)                             # (K,)
    b = (u.T @ (Q @ mu))                                   # (K,)

    a = np.maximum(a, 1e-15)
    z = b / np.sqrt(a)

    # --- log-space computation of term = 1/a + (b*sqrt(2pi)/a^1.5)*exp(0.5*z^2)*Phi(z) ---
    # term1 = 1/a
    log_term1 = -np.log(a)

    # term2 = |b| * sqrt(2pi) / a^1.5 * exp(0.5*z^2) * Phi(z)
    log_phi_z = np.log(np.maximum(Phi(z), 1e-300))
    log_term2_abs = (np.log(np.maximum(np.abs(b), 1e-300))
                     + math.log(SQRT2PI)
                     - 1.5 * np.log(a)
                     + 0.5 * z * z
                     + log_phi_z)

    sign_b = np.sign(b)

    # log(term) via log-sum-exp when b >= 0, log-sub-exp when b < 0
    log_term = np.where(
        sign_b >= 0,
        np.logaddexp(log_term1, log_term2_abs),
        # b < 0: term = 1/a - |term2|; guaranteed positive by theory
        log_term1 + np.log(np.maximum(
            1.0 - np.exp(np.minimum(log_term2_abs - log_term1, 500)),
            1e-300
        ))
    )

    log_const = -math.log(2.0 * math.pi) - 0.5 * math.log(detS)
    log_p = log_const - 0.5 * c + log_term

    return log_p


def p_theta_projected_normal(theta, mu, Sigma):
    """
    Angle density p_Theta(theta) for V ~ N(mu, Sigma) in R^2 (projected normal).
    theta: array in [0,2pi). Sigma must be SPD-ish.
    Returns p(theta) (not necessarily normalized unless you normalize in caller).

    This is a convenience wrapper around log_p_theta_projected_normal.
    """
    return np.exp(log_p_theta_projected_normal(theta, mu, Sigma))

def analytical_dcpa_prob_gt(x, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=248):
    """
    Compute P(DCPA > x) for unconstrained CPA model:
      d_CPA = r - v*(r.v)/(v.v), D = ||d_CPA||,
    with r ~ N(mu_r, Sigma_r), v ~ N(mu_v, Sigma_v), independent.
    Uses 1D integration over theta (direction of v) and a folded-normal tail.

    Parameters
    ----------
    x : float (threshold, x >= 0)
    mu_r : (2,) mean relative position [dx, dy]
    Sigma_r : (2,2) covariance of relative position
    mu_v : (2,) mean relative velocity [du, dv]
    Sigma_v : (2,2) covariance of relative velocity
    Ktheta : number of angle samples

    Returns
    -------
    float : P(D > x)
    """
    x = float(x)
    if x < 0:
        return 1.0

    mu_r = np.asarray(mu_r, float).reshape(2)
    Sigma_r = _regularize_spd(Sigma_r, eps=1e-9)

    mu_v = np.asarray(mu_v, float).reshape(2)
    Sigma_v = _regularize_spd(Sigma_v, eps=1e-9)

    theta = np.linspace(0.0, 2.0 * math.pi, int(Ktheta), endpoint=False)
    dtheta = 2.0 * math.pi / float(Ktheta)

    # Log-space density and normalization to avoid underflow/overflow
    # when velocity SNR is high (peaked p_Theta)
    log_pth = log_p_theta_projected_normal(theta, mu_v, Sigma_v)
    log_pth_max = np.max(log_pth)

    if not np.isfinite(log_pth_max):
        # Fallback: uniform weights that sum to 1
        weights = np.full_like(theta, 1.0 / float(Ktheta))
    else:
        # Normalize in log-space: w_k = exp(log_pth_k) / sum_j(exp(log_pth_j))
        # so that sum(w_k) = 1  (a discrete probability distribution over theta)
        log_pth_shifted = log_pth - log_pth_max
        pth_shifted = np.exp(log_pth_shifted)
        pth_sum_shifted = float(np.sum(pth_shifted))
        if pth_sum_shifted <= 0:
            weights = np.full_like(theta, 1.0 / float(Ktheta))
        else:
            weights = pth_shifted / pth_sum_shifted

    u_perp = np.stack([-np.sin(theta), np.cos(theta)], axis=0)  # (2,K)

    m = (u_perp.T @ mu_r)                                       # (K,)
    s2 = np.sum(u_perp * (Sigma_r @ u_perp), axis=0)            # (K,)
    s = np.sqrt(np.maximum(s2, 1e-15))                          # (K,)

    z1 = (x - m) / s
    z0 = (-x - m) / s
    cdf = Phi(z1) - Phi(z0)
    cdf = np.clip(cdf, 0.0, 1.0)

    tail = 1.0 - cdf
    p = float(np.sum(tail * weights))
    return float(np.clip(p, 0.0, 1.0))

def analytical_past_cpa_prob(mu_rel, Sigma_rel, nu_rel, Sigma_nu, eps=1e-12):
    """
    First-order (delta-method) approximation of P(t_CPA < 0).
    """
    mu_rel = np.asarray(mu_rel, float).reshape(2)
    nu_rel = np.asarray(nu_rel, float).reshape(2)

    Sigma_rel = _regularize_spd(Sigma_rel, eps=1e-9)
    Sigma_nu  = _regularize_spd(Sigma_nu,  eps=1e-9)

    m = float(mu_rel @ nu_rel)
    s2 = float(nu_rel @ Sigma_rel @ nu_rel + mu_rel @ Sigma_nu @ mu_rel)

    if not np.isfinite(s2) or s2 <= eps:
        if m < 0.0:
            return 1.0
        if m > 0.0:
            return 0.0
        return 0.5

    s = math.sqrt(s2)
    p = float(Phi(-m / s))
    return float(np.clip(p, 0.0, 1.0))
