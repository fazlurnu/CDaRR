"""L1 equivalence: cns.noise vs. the legacy sim_models.noise_model.NoiseModel.

NoiseModel is pure numpy (no bluesky dependency), so this uses lightweight
synthetic truth/message objects rather than the real sim env. Compares the new
pure functions' *outputs* against the legacy class's resulting msg.lat/lon/
gsnorth/gseast after the same-seeded call, across every pos_dist option, both
latency on/off, and the standalone add_latency_bias relay use case.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from cns.noise import make_covariance, latency_bias, position_noise, velocity_noise
from cns.distributions import make_mixture_gaussian, make_anisotropic_gaussian
from sim_models.noise_model import NoiseModel
from sim_models.adsl_message import ADSLMessage

N = 12
CI95_TO_STD_2D = 2.448


def _truth(n, seed=0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        lat=rng.uniform(40.0, 41.0, n),
        lon=rng.uniform(-70.0, -69.0, n),
        trk=rng.uniform(0, 360, n),
        gs=rng.uniform(5.0, 15.0, n),
    )


def _msg_from_truth(states, n):
    msg = ADSLMessage()
    msg.ensure_size(n)
    idx = np.arange(n)
    msg.lat[idx] = states.lat[idx]
    msg.lon[idx] = states.lon[idx]
    msg.trk[idx] = states.trk[idx]
    msg.gs[idx] = states.gs[idx]
    return msg


POS_DIST_CASES = [
    ("none", None),
    ("mixture", make_mixture_gaussian(3.0, 0.10)),
    ("anisotropic", make_anisotropic_gaussian(9.0)),
]


@pytest.mark.fast
@pytest.mark.parametrize("label,pos_dist", POS_DIST_CASES)
@pytest.mark.parametrize("latency_s", [0.0, 0.1])
def test_position_noise_matches_legacy(label, pos_dist, latency_s):
    states = _truth(N, seed=1)
    idx = np.arange(N)
    pos_std = 10.0 / CI95_TO_STD_2D

    old_msg = _msg_from_truth(states, N)
    old_rng = np.random.default_rng(42)
    old_nm = NoiseModel(pos_std_m=pos_std, vel_std_ms=1.0, rng=old_rng,
                        pos_dist=pos_dist, latency_s=latency_s)
    old_nm.add_position_noise(old_msg, states, idx)

    new_rng = np.random.default_rng(42)
    pos_cov = make_covariance(pos_std)
    pos_ci95 = pos_std * CI95_TO_STD_2D
    new_lat, new_lon = position_noise(
        states.lat, states.lon, states.trk, states.gs, idx,
        pos_cov, new_rng, pos_dist=pos_dist, pos_ci95=pos_ci95, latency_s=latency_s,
    )

    assert np.array_equal(old_msg.lat[idx], new_lat), f"{label} lat mismatch"
    assert np.array_equal(old_msg.lon[idx], new_lon), f"{label} lon mismatch"
    assert old_rng.bit_generator.state == new_rng.bit_generator.state


@pytest.mark.fast
def test_position_noise_matches_legacy_empty_idx():
    states = _truth(N, seed=2)
    idx = np.array([], dtype=int)
    pos_std = 10.0 / CI95_TO_STD_2D
    pos_cov = make_covariance(pos_std)

    new_lat, new_lon = position_noise(
        states.lat, states.lon, states.trk, states.gs, idx, pos_cov, np.random.default_rng(0),
    )
    assert new_lat.size == 0 and new_lon.size == 0


@pytest.mark.fast
def test_latency_bias_standalone_matches_legacy():
    """The relay use case: bias applied to an already-noised message, not truth."""
    states = _truth(N, seed=3)
    idx = np.arange(N)
    pos_std = 10.0 / CI95_TO_STD_2D
    latency_s = 0.1

    # Build an "already noised" message (simulating a relay from another node).
    relayed = _msg_from_truth(states, N)
    relayed.lat[idx] = states.lat[idx] + 1e-5
    relayed.lon[idx] = states.lon[idx] - 1e-5

    old_msg = ADSLMessage()
    old_msg.ensure_size(N)
    old_msg.lat[idx] = relayed.lat[idx].copy()
    old_msg.lon[idx] = relayed.lon[idx].copy()

    old_nm = NoiseModel(pos_std_m=pos_std, vel_std_ms=1.0, rng=np.random.default_rng(0),
                        latency_s=latency_s)
    old_nm.add_latency_bias(old_msg, states, idx)

    lat_bias, lon_bias = latency_bias(states.lat, states.trk, states.gs, idx, latency_s)
    new_lat = relayed.lat[idx] + lat_bias
    new_lon = relayed.lon[idx] + lon_bias

    assert np.array_equal(old_msg.lat[idx], new_lat)
    assert np.array_equal(old_msg.lon[idx], new_lon)


@pytest.mark.fast
def test_latency_bias_zero_is_noop():
    states = _truth(N, seed=4)
    idx = np.arange(N)
    lat_bias, lon_bias = latency_bias(states.lat, states.trk, states.gs, idx, 0.0)
    assert np.array_equal(lat_bias, np.zeros(N))
    assert np.array_equal(lon_bias, np.zeros(N))


@pytest.mark.fast
def test_velocity_noise_matches_legacy():
    states = _truth(N, seed=5)
    idx = np.arange(N)
    vel_std = 1.0 / CI95_TO_STD_2D

    old_msg = _msg_from_truth(states, N)
    old_rng = np.random.default_rng(7)
    old_nm = NoiseModel(pos_std_m=10.0, vel_std_ms=vel_std, rng=old_rng)
    old_nm.add_velocity_noise(old_msg, idx)

    new_rng = np.random.default_rng(7)
    vel_cov = make_covariance(vel_std)
    new_gsnorth, new_gseast = velocity_noise(states.trk, states.gs, idx, vel_cov, new_rng)

    assert np.array_equal(old_msg.gsnorth[idx], new_gsnorth)
    assert np.array_equal(old_msg.gseast[idx], new_gseast)
    assert old_rng.bit_generator.state == new_rng.bit_generator.state


@pytest.mark.fast
def test_velocity_noise_matches_legacy_empty_idx():
    vel_cov = make_covariance(1.0 / CI95_TO_STD_2D)
    new_gsnorth, new_gseast = velocity_noise(
        np.array([]), np.array([]), np.array([], dtype=int), vel_cov, np.random.default_rng(0))
    assert new_gsnorth.size == 0 and new_gseast.size == 0
