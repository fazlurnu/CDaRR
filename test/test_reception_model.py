# test/test_adsl.py
#
# Plain Python tests (no pytest) for refactored ADSL model.
# Requirements:
# 1) Bernoulli update: after first tick, per-aircraft update frequency ~ reception_prob
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

def test_bernoulli_update(pairwise, reception_prob, seed=4321):
    """
    Empirically verify Bernoulli update after the first tick:
    - First tick: all updated (no packet loss)
    - Subsequent ticks: each aircraft updated with probability reception_prob
    """
    states = pairwise._get_states()
    n = int(states.ntraf)

    adsl = ADSL(
        confidence_interval=0.0,          # noise doesn't matter here
        confidence_interval_velo=0.0,
        reception_prob=reception_prob,
        seed=seed,
    )

    # First update should update everyone (no packet loss)
    idx0 = adsl.update_from_truth(states)
    assert np.array_equal(idx0, np.arange(n)), "First update was not full (should have no packet loss)"

    # Now test reception frequency over many ticks
    T = 20_000
    counts = np.zeros(n, dtype=int)

    for _ in range(T):
        idx = adsl.update_from_truth(states)   # after first tick, Bernoulli applies
        counts[idx] += 1

    freqs = counts / float(T)
    mean_freq = float(np.mean(freqs))
    std_freq = float(np.std(freqs))

    print("\n=== Bernoulli update check ===")
    print(f"Target reception_prob: {reception_prob:.3f}")
    print(f"Empirical per-aircraft update freq: mean={mean_freq:.4f}, std={std_freq:.4f}")
    print("Per-aircraft frequencies:", np.round(freqs, 4).tolist())

    # Typical tolerance: ~ +/- 0.01 to 0.02 with T=20k; we’ll use 0.03 to be safe.
    tol = 0.03
    assert abs(mean_freq - reception_prob) <= tol, (
        f"Mean update frequency {mean_freq:.4f} not within {tol} of {reception_prob:.4f}"
    )

def main():
    _ensure_bluesky_inited()
    pairwise = _make_pairwise_env()

    # Choose some test parameters
    reception_prob = 0.95               # after first tick

    test_bernoulli_update(
        pairwise,
        reception_prob=reception_prob,
        seed=42,
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
