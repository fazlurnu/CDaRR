''' Position/velocity measurement noise -- functional core.

Redesign (not just a move) of the sim_models/noise_model.py NoiseModel class
into pure "arrays in -> arrays out" functions (refactor_fp.md, Phase 2's cns/
target). The legacy class mutated a shared ``msg`` object in place; these
functions take the relevant truth/current arrays plus an explicit
``np.random.Generator`` and return the new values for the requested indices,
touching nothing.

``latency_bias`` is kept separable from ``position_noise`` because the legacy
code calls it two different ways: baked into ``add_position_noise`` (bias
applied on top of a fresh truth-based noise draw), and standalone via
``add_latency_bias`` -- the live sim loop calls the latter directly on a
message that already holds a *relayed*, previously-noised position (freshly
received aircraft only; see sim/pairwise_stochastic/get_ipr_stochastic_env.py
and my-observation.md #13), without redrawing the random component. Both
call shapes are preserved here as separate pure functions.
'''
import numpy as np

# 95% radial CI -> per-axis 1-sigma (matches ADSL.CI95_TO_STD_2D / adsl_module.py).
CI95_TO_STD_2D = 2.448

_M_PER_DEG_LAT = 111_320.0


def make_covariance(std):
    ''' Isotropic 2D covariance matrix for a per-axis std (matches
    NoiseModel.__init__'s pos_cov/vel_cov construction exactly). '''
    return np.array([[std**2, 0.0], [0.0, std**2]], dtype=float)


def latency_bias(lat, trk, gs, idx, latency_s):
    ''' Deterministic along-track latency bias (lat/lon degrees) for aircraft
    in idx, using TRUTH lat/trk/gs at idx for magnitude/direction (bias
    convention: -latency_s * gs, i.e. the reported position lags behind
    truth). Pure. Returns (lat_bias_deg, lon_bias_deg), each shape
    (idx.size,) -- all-zero if latency_s == 0 or idx is empty, matching the
    legacy add_latency_bias's no-op guard.
    '''
    if idx.size == 0 or not latency_s:
        return np.zeros(idx.size), np.zeros(idx.size)

    trk_rad = np.deg2rad(np.asarray(trk[idx], dtype=float))
    gs_ = np.asarray(gs[idx], dtype=float)
    bias_at = -latency_s * gs_
    east_bias = bias_at * np.sin(trk_rad)
    north_bias = bias_at * np.cos(trk_rad)

    mean_lat = np.asarray(lat[idx], dtype=float)
    lat_bias_deg = north_bias / _M_PER_DEG_LAT
    coslat = np.maximum(np.cos(np.deg2rad(mean_lat)), 1e-6)
    lon_bias_deg = east_bias / (_M_PER_DEG_LAT * coslat)

    return lat_bias_deg, lon_bias_deg


def position_noise(lat, lon, trk, gs, idx, pos_cov, rng, pos_dist=None, pos_ci95=None, latency_s=0.0):
    ''' New (noised, latency-biased) lat/lon for aircraft in idx, drawn
    against TRUTH lat/lon/trk/gs. Pure given rng (no array is mutated).
    Matches the legacy add_position_noise's full behaviour, including its
    unconditional trailing latency-bias application.

    ``pos_dist`` is a callable ``(n, pos_ci95, rng, trk_rad) -> (n, 2)``
    metres east/north; ``None`` uses the default isotropic Gaussian via
    ``pos_cov``. Returns (new_lat, new_lon), each shape (idx.size,); empty
    arrays if idx is empty.
    '''
    if idx.size == 0:
        return np.array([]), np.array([])

    trk_rad = np.deg2rad(np.asarray(trk[idx], dtype=float))

    # Draw (east_m, north_m): custom distribution if provided, else Gaussian.
    if pos_dist is None:
        xy = rng.multivariate_normal((0.0, 0.0), pos_cov, size=int(idx.size))
    else:
        xy = pos_dist(int(idx.size), pos_ci95, rng, trk_rad)
    east_m = xy[:, 0]
    north_m = xy[:, 1]

    mean_lat = np.asarray(lat[idx], dtype=float)

    lat_noise_deg = north_m / _M_PER_DEG_LAT
    coslat = np.cos(np.deg2rad(mean_lat))
    coslat = np.maximum(coslat, 1e-6)
    lon_noise_deg = east_m / (_M_PER_DEG_LAT * coslat)

    new_lat = lat[idx] + lat_noise_deg
    new_lon = lon[idx] + lon_noise_deg

    lat_bias_deg, lon_bias_deg = latency_bias(lat, trk, gs, idx, latency_s)
    new_lat = new_lat + lat_bias_deg
    new_lon = new_lon + lon_bias_deg

    return new_lat, new_lon


def velocity_noise(trk, gs, idx, vel_cov, rng):
    ''' New (noised) gsnorth/gseast for aircraft in idx, from the given (CURRENT
    -- typically already-copied-from-truth) trk/gs at those indices. Pure
    given rng. Returns (new_gsnorth, new_gseast), each shape (idx.size,);
    empty arrays if idx is empty.
    '''
    if idx.size == 0:
        return np.array([]), np.array([])

    vxy = rng.multivariate_normal((0.0, 0.0), vel_cov, size=int(idx.size))
    north_noise = vxy[:, 0]
    east_noise = vxy[:, 1]

    trk_rad = np.deg2rad(trk[idx])
    gs_ = gs[idx]

    new_gsnorth = gs_ * np.cos(trk_rad) + north_noise
    new_gseast = gs_ * np.sin(trk_rad) + east_noise

    return new_gsnorth, new_gseast
