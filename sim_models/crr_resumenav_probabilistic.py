import numpy as np
import math
import bluesky as bs

# -------------------------
# Core math helpers
# -------------------------

# Vectorized erf for Phi
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
        # last-resort regularization
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

    # Mild clipping to avoid overflow in exp(0.5 z^2) for extreme z
    z = np.clip(z, -12.0, 12.0)

    term = 1.0 / a + (b * SQRT2PI / (a ** 1.5)) * np.exp(0.5 * z * z) * Phi(z)
    const = 1.0 / (2.0 * math.pi * math.sqrt(detS))
    p = const * np.exp(-0.5 * c) * term
    return p

def analytical_dcpa_prob_gt(x, mu_r, Sigma_r, mu_v, Sigma_v, Ktheta=248):
    """
    Compute P(DCPA > x) for unconstrained CPA model:
      d_CPA = r - v*(r·v)/(v·v), D = ||d_CPA||,
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

    # Direction density for v
    pth = p_theta_projected_normal(theta, mu_v, Sigma_v)
    # Normalize to integrate to 1 over [0,2pi)
    pth_sum = float(np.sum(pth) * dtheta)
    if pth_sum <= 0 or not np.isfinite(pth_sum):
        # fallback: uniform if numerical issues occur
        pth = np.full_like(theta, 1.0 / (2.0 * math.pi))
    else:
        pth = pth / pth_sum

    # Perpendicular unit vectors to direction theta
    u_perp = np.stack([-np.sin(theta), np.cos(theta)], axis=0)  # (2,K)

    # Y_perp(theta) = u_perp^T r ~ N(m, s^2)
    m = (u_perp.T @ mu_r)                                       # (K,)
    s2 = np.sum(u_perp * (Sigma_r @ u_perp), axis=0)            # (K,)
    s = np.sqrt(np.maximum(s2, 1e-15))                          # (K,)

    # CDF of |N(m,s^2)| at x: Phi((x-m)/s) - Phi((-x-m)/s)
    z1 = (x - m) / s
    z0 = (-x - m) / s
    cdf = Phi(z1) - Phi(z0)
    cdf = np.clip(cdf, 0.0, 1.0)

    tail = 1.0 - cdf
    p = float(np.sum(tail * pth) * dtheta)
    # Clamp just in case
    return float(np.clip(p, 0.0, 1.0))

def analytical_past_cpa_prob(mu_rel, Sigma_rel, nu_rel, Sigma_nu, eps=1e-12):
    """
    First-order (delta-method) approximation of P(t_CPA < 0)
    with:
      mu_rel = E[intruder - ownship]  (meters)
      nu_rel = E[ownship - intruder]  (m/s)

    Define S := (rel_pos · rel_vel) = x_rel · nu_rel.
    Then t_CPA = S / ||nu_rel||^2, hence t_CPA < 0  <=>  S < 0.

    Approx:
      S ~ N(m, s^2)
      m  = mu_rel^T nu_rel
      s^2 = nu_rel^T Sigma_rel nu_rel + mu_rel^T Sigma_nu mu_rel

    So:
      P(t_CPA < 0) = P(S < 0) = Phi(-m/s)
    """
    mu_rel = np.asarray(mu_rel, float).reshape(2)
    nu_rel = np.asarray(nu_rel, float).reshape(2)

    Sigma_rel = _regularize_spd(Sigma_rel, eps=1e-9)
    Sigma_nu  = _regularize_spd(Sigma_nu,  eps=1e-9)

    m = float(mu_rel @ nu_rel)
    s2 = float(nu_rel @ Sigma_rel @ nu_rel + mu_rel @ Sigma_nu @ mu_rel)

    if not np.isfinite(s2) or s2 <= eps:
        # Degenerate fallback: S is (almost) deterministic
        if m < 0.0:
            return 1.0
        if m > 0.0:
            return 0.0
        return 0.5

    s = math.sqrt(s2)
    p = float(Phi(-m / s))   # <-- key sign
    return float(np.clip(p, 0.0, 1.0))


# -------------------------
# Modified method
# -------------------------
def resumenav_probabilistic(reso, conf, ownship, intruder):
    """
    Probabilistic version of resumenav_double_criteria:
      crit1 := P(DCPA > rpz | intruder keeps current velocity) > 0.99
      crit2 := P(DCPA > rpz | intruder reverts to initial velocity) > 0.99

    Uses mu_r=[dx,dy] and mu_v=[du,dv] from each criterion's relative geometry.
    Covariances are taken from conf if available, otherwise defaulted/regularized.

    Expected optional conf fields (any that exist will be used):
      - conf.Sigma_r or conf.sigma_r or conf.sigma_pos or conf.pos_std
      - conf.Sigma_v or conf.sigma_v or conf.sigma_vel or conf.vel_std
      - conf.dcpa_prob_threshold (default 0.99)
      - conf.dcpa_prob_Ktheta (default 4096)
    """
    if not hasattr(reso, "_intr_init_vel"):
        reso._intr_init_vel = {}

    curpairs = set(conf.confpairs)
    newpairs = curpairs - reso.resopairs
    reso.resopairs.update(curpairs)

    for pair in newpairs:
        idx1, idx2 = bs.traf.id2idx(pair)
        if idx1 >= 0 and idx2 >= 0:
            # Vi,i recorded at conflict initiation
            reso._intr_init_vel[pair] = (float(intruder.gseast[idx2]),
                                         float(intruder.gsnorth[idx2]))

    pair_dxdy = {}
    if len(conf.confpairs) > 0:
        q = np.radians(conf.qdr)
        dxs = conf.dist * np.sin(q)
        dys = conf.dist * np.cos(q)
        pair_dxdy = dict(zip(conf.confpairs, zip(dxs.tolist(), dys.tolist())))

    vod_cache = {}

    def _val(a, idx):
        try:
            return float(a[idx])
        except Exception:
            return None

    def get_Vo_d(idx):
        if idx in vod_cache:
            return vod_cache[idx]

        trk = None
        if hasattr(ownship, "seltrk"):
            trk = _val(ownship.seltrk, idx)
        if trk is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "trk"):
            trk = _val(ownship.ap.trk, idx)
        if trk is None:
            trk = _val(ownship.trk, idx)

        spd = None
        if hasattr(ownship, "selspd"):
            spd = _val(ownship.selspd, idx)
        if spd is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "tas"):
            spd = _val(ownship.ap.tas, idx)
        if spd is None:
            spd = _val(getattr(ownship, "gs", None), idx)
        if spd is None:
            spd = float(np.hypot(ownship.gseast[idx], ownship.gsnorth[idx]))

        r = np.radians(trk)
        u = spd * np.sin(r)
        v = spd * np.cos(r)
        vod_cache[idx] = (u, v)
        return u, v

    # Pull (or default) covariance models for relative position/velocity.
    # You can wire these to your actual uncertainty model by putting them on conf.
    Sigma_r = None
    for name in ("Sigma_r", "sigma_r", "Sigma_pos", "sigma_pos", "pos_std", "sigma_position"):
        if hasattr(conf, name):
            Sigma_r = getattr(conf, name)
            break
    Sigma_r = _to_cov(Sigma_r)  # default zero if None

    Sigma_v = None
    for name in ("Sigma_v", "sigma_v", "Sigma_vel", "sigma_vel", "vel_std", "sigma_velocity"):
        if hasattr(conf, name):
            Sigma_v = getattr(conf, name)
            break
    Sigma_v = _to_cov(Sigma_v)  # default zero if None

    # A little regularization helps avoid singular matrices
    Sigma_r = _regularize_spd(Sigma_r, eps=1e-6)
    Sigma_v = _regularize_spd(Sigma_v, eps=1e-6)

    prob_threshold = float(getattr(conf, "dcpa_prob_threshold", 0.999))
    Ktheta = int(getattr(conf, "dcpa_prob_Ktheta", 248))

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

        # Relative position (dx,dy)
        if conflict in pair_dxdy:
            dx, dy = pair_dxdy[conflict]
            dx = float(dx)
            dy = float(dy)
        else:
            re = 6371000.0
            dlon = float(intruder.lon[idx2] - ownship.lon[idx1])
            dlat = float(intruder.lat[idx2] - ownship.lat[idx1])
            latm = 0.5 * np.radians(float(intruder.lat[idx2] + ownship.lat[idx1]))
            dx = re * np.radians(dlon) * np.cos(latm)
            dy = re * np.radians(dlat)

        rpz = float(np.max(conf.rpz[[idx1, idx2]]))

        # Ownship velocity (east,north)
        Vo_u, Vo_v = get_Vo_d(idx1)

        # Intruder current velocity
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
        
        # Criterion 3: probabilistic past-CPA (P(t_CPA < 0) > threshold)
        mu_rel = mu_r  # intruder - ownship (already)

        # IMPORTANT: ownship - intruder (matches paper sign for Phi(-m/s))
        nu_rel = np.array([
            float(ownship.gseast[idx1] - intruder.gseast[idx2]),
            float(ownship.gsnorth[idx1] - intruder.gsnorth[idx2])
        ], dtype=float)

        p3 = analytical_past_cpa_prob(mu_rel, Sigma_r, nu_rel, Sigma_v)
        crit3 = (p3 > prob_threshold)

        if crit1 and crit2 and crit3:
            delpairs.add(conflict)
            reso._intr_init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
        else:
            changeactive[idx1] = True

    for idx, active in changeactive.items():
        reso.active[idx] = active
        if not active:
            iwpid = bs.traf.ap.route[idx].findact(idx)
            if iwpid != -1:
                bs.traf.ap.route[idx].direct(idx, bs.traf.ap.route[idx].wpname[iwpid])

    reso.resopairs -= delpairs
    # print(delpairs)
    return delpairs