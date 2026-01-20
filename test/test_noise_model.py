# test/test_adsl.py
#
# Plain Python tests (no pytest) for refactored ADSL model.
# Requirements:
# 1) Noise model: print that ~95% of samples are within specified confidence interval
#    - Position: radius in meters <= confidence_interval
#    - Velocity: radius in m/s <= confidence_interval_velo
#
# Traffic truth is pulled from PairwiseHorConflict()._get_states()

import numpy as np

import bluesky as bs
from envs.pairwise_conflict import PairwiseHorConflict

# Import ADSL from wherever you placed it after refactor.
# (You have sim_models/cns_adsl.py and sim_models/adsl_module.py in your tree.)

from sim_models.adsl_module import ADSL


def _ensure_bluesky_inited():
    if not getattr(bs, "_joblib_inited", False):
        bs.init(mode="sim", detached=True)
        bs._joblib_inited = True


def _make_pairwise_env():
    # Keep consistent with your usage snippet (values don’t matter much for this test)
    width = 1
    height = 1
    horizontal_sep = 50       # m
    lookahead_time = 15       # s
    init_speed_ownship = 10.2889   # m/s
    init_speed_intruder = 10.2889  # m/s
    dpsi = 180
    aircraft_type = "M600"
    SIMDT_FACTOR = 1.0

    return PairwiseHorConflict(
        pair_width=width,
        pair_height=height,
        asas_pzr_m=horizontal_sep,
        dtlookahead=lookahead_time + 1,
        init_speed_ownship=init_speed_ownship,
        init_speed_intruder=init_speed_intruder,
        init_dpsi=dpsi,
        aircraft_type_ownship=aircraft_type,
        simdt_factor=SIMDT_FACTOR,
    )


def _pos_error_radius_m(lat_true, lon_true, lat_meas, lon_meas):
    """
    Compute horizontal position error radius (meters) using the same small-angle
    approximation used in the noise model (111320 m/deg + cos(lat) for lon).
    """
    # north in meters
    north_m = (lat_meas - lat_true) * 111_320.0

    # east in meters
    coslat = np.cos(np.deg2rad(lat_true))
    coslat = np.maximum(coslat, 1e-6)
    east_m = (lon_meas - lon_true) * 111_320.0 * coslat

    return np.sqrt(east_m * east_m + north_m * north_m)


def _vel_error_radius(msg, states, idx):
    """
    Compute 2D velocity error radius for indices idx:
      (noisy gsnorth/gseast) - (truth derived from gs/trk)
    """
    trk_rad = np.deg2rad(states.trk[idx])
    gs = states.gs[idx]
    vn_true = gs * np.cos(trk_rad)
    ve_true = gs * np.sin(trk_rad)

    vn_err = msg.gsnorth[idx] - vn_true
    ve_err = msg.gseast[idx] - ve_true
    return np.sqrt(vn_err * vn_err + ve_err * ve_err)


def test_noise_model(pairwise, confidence_interval, confidence_interval_velo, seed=1234):
    """
    Empirically verify that ~95% of *2D radius* errors are within the specified CI
    (given the CI->STD conversion used in your ADSL code).
    """
    states = pairwise._get_states()
    n = int(states.ntraf)

    # Set reception_prob=1.0 to always update so we sample noise every call
    adsl = ADSL(
        confidence_interval=confidence_interval,
        confidence_interval_velo=confidence_interval_velo,
        reception_prob=1.0,
        seed=seed,
    )

    # IMPORTANT: first update is noise too, but we want many samples
    # We'll run many updates on the same truth (states) to sample measurement noise.
    N_SAMPLES = 50_000  # total samples per aircraft is this count
    # We will collect radii across all aircraft and all iterations.
    pos_radii = np.empty(N_SAMPLES * n, dtype=float)
    vel_radii = np.empty(N_SAMPLES * n, dtype=float)

    # Prime (first update)
    adsl.update_from_truth(states)

    k = 0
    all_idx = np.arange(n, dtype=int)
    for _ in range(N_SAMPLES):
        adsl.update_from_truth(states)

        # Position radius vs truth
        pr = _pos_error_radius_m(states.lat, states.lon, adsl.msg.lat, adsl.msg.lon)
        pos_radii[k : k + n] = pr

        # Velocity radius vs truth derived from gs/trk
        vr = _vel_error_radius(adsl.msg, states, all_idx)
        vel_radii[k : k + n] = vr

        k += n

    pos_frac = float(np.mean(pos_radii <= confidence_interval))
    vel_frac = float(np.mean(vel_radii <= confidence_interval_velo))

    print("\n=== Noise model check ===")
    print(f"Position CI (95%) target radius <= {confidence_interval:.3f} m")
    print(f"Empirical fraction within CI: {pos_frac * 100:.2f}%")

    print(f"\nVelocity CI (95%) target radius <= {confidence_interval_velo:.3f} m/s")
    print(f"Empirical fraction within CI: {vel_frac * 100:.2f}%")

    # Loose sanity bounds: should be close to 95% (allow Monte Carlo + approximation wiggle)
    # Tighten if you like.
    assert 0.93 <= pos_frac <= 0.97, f"Position CI fraction not ~95%: {pos_frac:.4f}"
    assert 0.93 <= vel_frac <= 0.97, f"Velocity CI fraction not ~95%: {vel_frac:.4f}"


def main():
    _ensure_bluesky_inited()
    pairwise = _make_pairwise_env()

    # Choose some test parameters
    confidence_interval = 30.0          # meters (position 95% radius)
    confidence_interval_velo = 5.0      # m/s (velocity 95% radius)

    test_noise_model(
        pairwise,
        confidence_interval=confidence_interval,
        confidence_interval_velo=confidence_interval_velo,
        seed=42,
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
