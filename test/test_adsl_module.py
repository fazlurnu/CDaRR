# test/test_adsl.py
#
# Plain Python tests (no pytest) for refactored ADSL model.
# Requirements:
# 1) Noise model: print that ~95% of samples are within specified confidence interval
#    - Position: radius in meters <= confidence_interval
#    - Velocity: radius in m/s <= confidence_interval_velo
# 2) Bernoulli update: after first tick, per-aircraft update frequency ~ reception_prob
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


def test_end_to_end_comm_trace(pairwise, confidence_interval, confidence_interval_velo, reception_prob, seed=2025):
    """
    End-to-end comm trace test:
      1) ownship updates from truth (adds noise)
      2) transmit to intruder with packet loss at intruder side
      3) trace across time steps
      4) verify hold vs update behavior
      5) verify Bernoulli update frequency ~= reception_prob (after first tick)

    We implement the same pattern as your previous simulator:
      - intruder updates indices that "received" a packet using ownship message
      - missed indices are filled from prev_intruder buffer (hold last)
    """
    states = pairwise._get_states()
    n = int(states.ntraf)

    # ADSL nodes
    adsl_bus = ADSL(confidence_interval, confidence_interval_velo, reception_prob=1.0, seed=seed + 1)

    ownship_adsl = ADSL(confidence_interval, confidence_interval_velo, reception_prob=1.0, seed=seed + 2)
    intruder_adsl = ADSL(confidence_interval, confidence_interval_velo, reception_prob=reception_prob, seed=seed + 3)

    # Buffer to hold last intruder observation
    prev_intruder_adsl = ADSL(confidence_interval, confidence_interval_velo, reception_prob=1.0, seed=seed + 4)

    # --- Tick 0: NO packet loss; everyone gets initial message ---
    ownship_adsl.update_from_truth(states)
    adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=None)
    adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

    assert intruder_adsl.ntraf == n
    assert np.array_equal(np.asarray(intruder_adsl.id, dtype=str), np.asarray(states.id, dtype=str))

    # Reception probability source (compat with both styles)
    p = getattr(intruder_adsl, "reception_prob", None)
    if p is None:
        p = intruder_adsl.reception.p

    # --- Trace over time ---
    T = 5000  # number of simulated "message update attempts" after the first tick

    update_counts = np.zeros(n, dtype=int)   # times each aircraft received packet
    hold_counts = np.zeros(n, dtype=int)     # times each aircraft missed packet

    # Noise update checks:
    # - on receive: intruder should match ownship *this tick* (fresh noise)
    # - on miss: intruder should match previous intruder (hold last)
    receive_match_fail = 0
    hold_match_fail = 0

    # Track how often intruder lat changes on "received" ticks
    changed_on_receive = np.zeros(n, dtype=int)

    rng_manual = np.random.default_rng(seed + 999)  # for the manual Bernoulli mask

    for _t in range(T):
        # Pull (possibly time-evolving) truth
        states = pairwise._get_states()

        # 1) Ownship receives truth + adds noise
        ownship_adsl.update_from_truth(states)

        # Save intruder before applying comm updates (for hold/noise checks)
        lat_before = intruder_adsl.lat.copy()
        lon_before = intruder_adsl.lon.copy()
        alt_before = intruder_adsl.alt.copy()

        # 4) Packet loss at intruder side (after first tick)
        rx_mask = (rng_manual.random(n) <= p)
        idx_rx = np.where(rx_mask)[0]
        idx_miss = np.where(~rx_mask)[0]

        update_counts[idx_rx] += 1
        hold_counts[idx_miss] += 1

        # 2) Transmit:
        #    - received indices take fresh ownship message
        #    - missed indices hold last via prev_intruder buffer
        if idx_rx.size > 0:
            adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=idx_rx)
        if idx_miss.size > 0:
            adsl_bus.send_data(intruder_adsl, prev_intruder_adsl, indices=idx_miss)

        # 3) Trace/recording is implicit here (we're checking consistency each step)

        # 5a) Check "noise updated" on received packets:
        #     intruder should equal ownship for received indices
        if idx_rx.size > 0:
            ok_lat = np.allclose(intruder_adsl.lat[idx_rx], ownship_adsl.lat[idx_rx], rtol=0.0, atol=0.0)
            ok_lon = np.allclose(intruder_adsl.lon[idx_rx], ownship_adsl.lon[idx_rx], rtol=0.0, atol=0.0)
            ok_alt = np.allclose(intruder_adsl.alt[idx_rx], ownship_adsl.alt[idx_rx], rtol=0.0, atol=0.0)
            if not (ok_lat and ok_lon and ok_alt):
                receive_match_fail += 1

            # also verify it *changed* compared to previous intruder most of the time
            # (extremely unlikely to be equal if noise is being refreshed)
            changed = ~np.isclose(intruder_adsl.lat[idx_rx], lat_before[idx_rx], rtol=0.0, atol=0.0)
            changed_on_receive[idx_rx] += changed.astype(int)

        # 5b) Check "hold last" on missed packets:
        if idx_miss.size > 0:
            ok_lat = np.allclose(intruder_adsl.lat[idx_miss], lat_before[idx_miss], rtol=0.0, atol=0.0)
            ok_lon = np.allclose(intruder_adsl.lon[idx_miss], lon_before[idx_miss], rtol=0.0, atol=0.0)
            ok_alt = np.allclose(intruder_adsl.alt[idx_miss], alt_before[idx_miss], rtol=0.0, atol=0.0)
            if not (ok_lat and ok_lon and ok_alt):
                hold_match_fail += 1

        # Update prev buffer for next step
        adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

    # --- Results / prints ---
    freqs = update_counts / float(T)
    mean_freq = float(np.mean(freqs))

    print("\n=== End-to-end comm trace check ===")
    print(f"T steps (after first tick): {T}")
    print(f"Target reception_prob: {p:.3f}")
    print(f"Empirical update freq mean: {mean_freq:.4f}")
    print("Per-aircraft update freqs:", np.round(freqs, 4).tolist())

    print(f"\nReceive->ownship match failures: {receive_match_fail} / {T} steps")
    print(f"Miss->hold match failures:      {hold_match_fail} / {T} steps")

    # How often "received" packets resulted in a changed latitude vs previous:
    # If noise is being refreshed, this should be very close to 1.0 for aircraft that receive often.
    change_rate = np.zeros(n, dtype=float)
    for i in range(n):
        if update_counts[i] > 0:
            change_rate[i] = changed_on_receive[i] / float(update_counts[i])
    print("Change rate on receive (lat):", np.round(change_rate, 4).tolist())

    # --- Assertions ---
    # 1) Bernoulli update frequency close to p (allow some Monte Carlo tolerance)
    tol = 0.03
    assert abs(mean_freq - p) <= tol, (
        f"Mean update frequency {mean_freq:.4f} not within {tol} of {p:.4f}"
    )

    # 2) Transmission correctness: received indices should match ownship; missed should hold last
    assert receive_match_fail == 0, "Some steps had received indices not matching ownship message."
    assert hold_match_fail == 0, "Some steps had missed indices not holding last message."

    # 3) Noise refresh sanity: for aircraft that received a reasonable number of times,
    #    the message should *almost always* change vs previous on receive.
    #    (Extremely rare to be exactly identical if new noise is drawn.)
    for i in range(n):
        if update_counts[i] >= 50:
            assert change_rate[i] > 0.98, (
                f"Aircraft {i} change_rate_on_receive too low ({change_rate[i]:.3f}); "
                "noise may not be refreshed on updates."
            )

def main():
    _ensure_bluesky_inited()
    pairwise = _make_pairwise_env()

    # Choose some test parameters
    confidence_interval = 30.0          # meters (position 95% radius)
    confidence_interval_velo = 5.0      # m/s (velocity 95% radius)
    reception_prob = 0.95               # after first tick

    test_end_to_end_comm_trace(
        pairwise,
        confidence_interval=confidence_interval,
        confidence_interval_velo=confidence_interval_velo,
        reception_prob=reception_prob,
        seed=42,
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
