import numpy as np
import math
import bluesky as bs

from sim_models.crr_recovery_base import (
    get_desired_ownship_velocity,
    compute_pair_positions,
    get_pair_dxdy,
    apply_recovery,
    record_initial_intruder_velocity,
)

# -------------------------
# Core math helpers
# -------------------------

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

def p_theta_projected_normal(theta, mu, Sigma):
    """
    Angle density p_Theta(theta) for V ~ N(mu, Sigma) in R^2 (projected normal).
    theta: array in [0,2pi). Sigma must be SPD-ish.
    Returns p(theta) (not necessarily normalized unless you normalize in caller).
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

    z = np.clip(z, -12.0, 12.0)

    term = 1.0 / a + (b * SQRT2PI / (a ** 1.5)) * np.exp(0.5 * z * z) * Phi(z)
    const = 1.0 / (2.0 * math.pi * math.sqrt(detS))
    p = const * np.exp(-0.5 * c) * term
    return p

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

    pth = p_theta_projected_normal(theta, mu_v, Sigma_v)
    pth_sum = float(np.sum(pth) * dtheta)
    if pth_sum <= 0 or not np.isfinite(pth_sum):
        pth = np.full_like(theta, 1.0 / (2.0 * math.pi))
    else:
        pth = pth / pth_sum

    u_perp = np.stack([-np.sin(theta), np.cos(theta)], axis=0)  # (2,K)

    m = (u_perp.T @ mu_r)                                       # (K,)
    s2 = np.sum(u_perp * (Sigma_r @ u_perp), axis=0)            # (K,)
    s = np.sqrt(np.maximum(s2, 1e-15))                          # (K,)

    z1 = (x - m) / s
    z0 = (-x - m) / s
    cdf = Phi(z1) - Phi(z0)
    cdf = np.clip(cdf, 0.0, 1.0)

    tail = 1.0 - cdf
    p = float(np.sum(tail * pth) * dtheta)
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


# -------------------------
# Recovery method
# -------------------------
def resumenav_probabilistic_ftr(reso, conf, ownship, intruder):
    """
    Probabilistic version of resumenav_double_criteria:
      crit1 := P(DCPA > rpz | intruder keeps current velocity) > threshold
      crit2 := P(DCPA > rpz | intruder reverts to initial velocity) > threshold

    Expected optional conf fields:
      - conf.sigma_r : relative position std (scalar or 2x2 cov)
      - conf.sigma_v : relative velocity std (scalar or 2x2 cov)
      - conf.dcpa_prob_threshold (default 0.9)
      - conf.dcpa_prob_Ktheta (default 256)
    """
    record_initial_intruder_velocity(reso, conf, intruder)

    pair_dxdy = compute_pair_positions(conf)
    vod_cache = {}

    # Pull covariance models from conf
    Sigma_r = None
    for name in ("Sigma_r", "sigma_r", "Sigma_pos", "sigma_pos", "pos_std", "sigma_position"):
        if hasattr(conf, name):
            Sigma_r = getattr(conf, name)
            break
    Sigma_r = _regularize_spd(_to_cov(Sigma_r), eps=1e-6)

    Sigma_v = None
    for name in ("Sigma_v", "sigma_v", "Sigma_vel", "sigma_vel", "vel_std", "sigma_velocity"):
        if hasattr(conf, name):
            Sigma_v = getattr(conf, name)
            break
    Sigma_v = _regularize_spd(_to_cov(Sigma_v), eps=1e-6)

    prob_threshold = float(getattr(conf, "dcpa_prob_threshold", 0.9))
    Ktheta = int(getattr(conf, "dcpa_prob_Ktheta", 256))

    delpairs = set()
    changeactive = {}

    for conflict in reso.resopairs:
        idx1, idx2 = bs.traf.id2idx(conflict)

        if idx1 < 0:
            delpairs.add(conflict)
            reso._intr_init_vel.pop(conflict, None)
            continue

        if idx2 < 0:
            delpairs.add(conflict)
            reso._intr_init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
            continue

        dx, dy = get_pair_dxdy(conflict, pair_dxdy, ownship, intruder, idx1, idx2)
        rpz = float(np.max(conf.rpz[[idx1, idx2]]))
        Vo_u, Vo_v = get_desired_ownship_velocity(ownship, idx1, vod_cache)

        Vi_c_u = float(intruder.gseast[idx2])
        Vi_c_v = float(intruder.gsnorth[idx2])

        mu_r = np.array([dx, dy], dtype=float)

        # Criterion 1: intruder maintains current velocity (Vi,c)
        mu_v1 = np.array([Vo_u - Vi_c_u, Vo_v - Vi_c_v], dtype=float)
        p1 = analytical_dcpa_prob_gt(rpz, mu_r, Sigma_r, mu_v1, Sigma_v, Ktheta=Ktheta)
        crit1 = (p1 > prob_threshold)

        # Criterion 2: intruder reverts to initial velocity (Vi,i)
        Vi_i_u, Vi_i_v = reso._intr_init_vel.get(conflict, (Vi_c_u, Vi_c_v))
        mu_v2 = np.array([Vo_u - float(Vi_i_u), Vo_v - float(Vi_i_v)], dtype=float)
        p2 = analytical_dcpa_prob_gt(rpz, mu_r, Sigma_r, mu_v2, Sigma_v, Ktheta=Ktheta)
        crit2 = (p2 > prob_threshold)

        if crit1 and crit2:
            delpairs.add(conflict)
            reso._intr_init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
        else:
            changeactive[idx1] = True

    apply_recovery(changeactive, reso, delpairs)
    return delpairs
