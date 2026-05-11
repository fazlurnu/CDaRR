"""
test_stochastic_sim_single_job_sweep.py

Copy-paste runnable script.
Sweeps over (confidence_interval_pos, confidence_interval_velo) for the ADSL model,
runs the pairwise sim with MVP resolution, prints summary metrics per case, and
(optionally) plots for ONE selected case to avoid spamming figures.

Assumptions:
- Your project provides:
    from sim_models.cd_statebased import StateBased
    from sim_models.cr_mvp import MVP
    from envs.pairwise_conflict import PairwiseHorConflict
    from sim_models.adsl_module import ADSL
    from sim_models.reception_model import ReceptionModel
    from sim.utils import _check_tcpa_tinhor_per_pair, done_with_timeout
- BlueSky is installed and importable as `bluesky as bs`
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import bluesky as bs

from sim_models.cd_statebased import StateBased
from sim_models.cr_mvp import MVP
from envs.pairwise_conflict import PairwiseHorConflict
from sim_models.adsl_module import ADSL
from sim_models.reception_model import ReceptionModel

from sim.utils import _check_tcpa_tinhor_per_pair, done_with_timeout

def reset_between_cases(pairwise=None):
    # Reset env first (if provided)
    if pairwise is not None:
        try:
            pairwise.reset()
        except Exception:
            pass
        try:
            pairwise.close()
        except Exception:
            pass

    # Reset BlueSky traffic
    try:
        bs.traf.reset()
        bs.sim.reset()
    except Exception:
        pass
    
# ----------------------------
# USER SETTINGS
# ----------------------------
confidence_interval_list = [1.5, 5, 15, 46.3, 46.3 * 2]  # meters
confidence_interval_velo_list = [0.5, 1.5, 5]            # m/s

# Scenario parameters
width = 10
height = 10
horizontal_sep = 50.0
lookahead_time = 15.0
init_speed_ownship = 10.2889
init_speed_intruder = 10.2889
dpsi = 180
aircraft_type = "M600"

SIMDT_FACTOR = 4.0
DONE_TIMEOUT = 10.0
reception_prob = 0.95

# Run length
tmax = lookahead_time * 100.0

# Reproducibility
seed_base = 42

# Plot only one case (to avoid many figures)
PLOT_ONE_CASE = True
PLOT_CI = confidence_interval_list[-1]        # which CI to plot
PLOT_CIV = confidence_interval_velo_list[0]   # which CIV to plot


# ----------------------------
# INIT BLUESKY ONCE
# ----------------------------
if not getattr(bs, "_joblib_inited", False):
    bs.init(mode="sim", detached=True)
    bs._joblib_inited = True

# Require turn-rate limits (your existing check pattern)
try:
    _ = bs.traf.MAX_TR
    _ = bs.traf.MAX_DTR2
except AttributeError as e:
    raise RuntimeError(
        "Required BlueSky attributes missing: bs.traf.MAX_TR and/or bs.traf.MAX_DTR2."
    ) from e

# Apply limits (as in your script)
bs.traf.MAX_TR = 15
bs.traf.MAX_DTR2 = 10


def run_one_case(ci_pos: float, ci_vel: float, *, seed: int, do_plots: bool):
    # Fresh models per case (avoid state leakage across sweeps)
    conf_detection = StateBased()
    conf_detection_groundtruth = StateBased()
    conf_resolution = MVP()

    # Fresh environment per case
    pairwise = PairwiseHorConflict(
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

    simdt = float(bs.settings.simdt) * float(SIMDT_FACTOR)
    event_dt = float(bs.settings.asas_dt)

    # Timers / termination
    sim_timer_second = 0.0
    done_start_time = None

    # ADSL nodes (fresh per case)
    adsl_bus = ADSL(ci_pos, ci_vel, reception_prob=1.0, seed=seed + 1)
    ownship_adsl = ADSL(ci_pos, ci_vel, reception_prob=1.0, seed=seed + 2)
    intruder_adsl = ADSL(ci_pos, ci_vel, reception_prob=reception_prob, seed=seed + 3)
    prev_intruder_adsl = ADSL(ci_pos, ci_vel, reception_prob=1.0, seed=seed + 4)

    # Reception model (fresh RNG per case so results are deterministic per (ci_pos,ci_vel))
    rx_rng = np.random.default_rng(seed + 999)
    intruder_adsl.reception = ReceptionModel(reception_prob=reception_prob, rng=rx_rng)

    # Recording
    distance_array = []
    time_list = []
    lat_list, lon_list = [], []
    gs_list, hdg_list = [], []

    update_counts = None
    asas_ticks = 0
    last_n_active = 0

    eps = np.finfo(float).eps * 100
    next_event_t = 0.0

    # --- Tick 0 init message (no loss) ---
    states0 = pairwise._get_states()
    n = int(states0.ntraf)
    update_counts = np.zeros(n, dtype=int)

    ownship_adsl.update_from_truth(states0)
    adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=None)
    adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

    # --- Main loop ---
    while sim_timer_second < tmax:
        states = pairwise._get_states()
        n = int(states.ntraf)

        if sim_timer_second + eps >= next_event_t:
            # Comms update
            ownship_adsl.update_from_truth(states)

            idx_rx = intruder_adsl.reception.sample_indices(n)
            rx_mask = np.zeros(n, dtype=bool)
            rx_mask[idx_rx] = True
            idx_miss = np.where(~rx_mask)[0]

            if idx_rx.size > 0:
                adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=idx_rx)
                update_counts[idx_rx] += 1
            if idx_miss.size > 0:
                adsl_bus.send_data(intruder_adsl, prev_intruder_adsl, indices=idx_miss)

            # Hold buffer update
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)
            asas_ticks += 1

            # Detector "sees" noisy ADSL states
            ownship_obs = ownship_adsl
            intruder_obs = intruder_adsl

            # Detect + resolve (noisy)
            conf_detection.detect(ownship_obs, intruder_obs, horizontal_sep, 100.0, lookahead_time)
            reso = conf_resolution.resolve(conf_detection, ownship_obs, intruder_obs, bs.settings.asas_marh)

            # Ground-truth detect for termination metrics
            conf_detection_groundtruth.detect(states, states, horizontal_sep, 100.0, lookahead_time)
            done_now, n_active = _check_tcpa_tinhor_per_pair(
                bs.traf.id,
                conf_detection_groundtruth.tcpa_all,
                conf_detection_groundtruth.tinhor_all,
            )
            last_n_active = int(n_active)

            done_start_time, should_stop = done_with_timeout(
                done_now=done_now,
                done_start_time=done_start_time,
                sim_timer_second=sim_timer_second,
                done_timeout=DONE_TIMEOUT,
                verbose=False,
            )
            # Note: we still step + record below, then break if should_stop is true.
            # Advance next event time
            missed = (
                int(np.floor((sim_timer_second - next_event_t) / event_dt)) + 1
                if sim_timer_second > next_event_t
                else 1
            )
            next_event_t += missed * event_dt
        else:
            # Keep previous reso between event ticks
            # (If your pairwise.step requires a reso always, this stays from last event tick.)
            pass

        # Step + record (keep time and distance lengths aligned)
        dist = pairwise.step(reso)
        distance_array.append(dist)

        time_list.append(sim_timer_second)
        lat_list.append(states.lat)
        lon_list.append(states.lon)
        gs_list.append(states.gs)
        hdg_list.append(states.hdg)

        # Break after recording if stop latched
        if done_start_time is not None and (sim_timer_second - done_start_time) >= DONE_TIMEOUT:
            break

        sim_timer_second += simdt

    # Cleanup and metrics
    pairwise.reset()

    distance_array = np.asarray(distance_array)[:, :pairwise.nb_pair]
    nb_pair = int(pairwise.nb_pair)

    min_dist_per_pair = np.min(distance_array, axis=0)
    n_los = int(np.sum(min_dist_per_pair < horizontal_sep))
    ipr = 1.0 - (n_los / float(nb_pair))  # since all pairs are "conflicts" by construction

    emp_p = float("nan")
    if asas_ticks > 0:
        emp_p = float(np.mean(update_counts / float(asas_ticks)))

    # Plot for this case (optional)
    if do_plots:
        # Distance: all pairs
        Tdist = distance_array.shape[0]
        t_plot = np.asarray(time_list[:Tdist], dtype=float)

        plt.figure(figsize=(12, 6))
        for i in range(nb_pair):
            plt.plot(t_plot, distance_array[:, i], color="tab:blue", alpha=0.1)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title(f"All-pairs distance | CI={ci_pos} m, CIV={ci_vel} m/s | DPSI={dpsi}, RPZ={horizontal_sep} m")
        plt.axhline(y=horizontal_sep, color="r", linestyle="--")
        plt.tight_layout()
        plt.show()

        # GS + HDG: all aircraft, same figure with subplots
        gs_arr = np.asarray(gs_list)
        hdg_arr = np.asarray(hdg_list)
        t_plot2 = np.asarray(time_list, dtype=float)

        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(2, 1, 1)
        for k in range(gs_arr.shape[1]):
            ax1.plot(t_plot2, gs_arr[:, k], color="tab:blue", alpha=0.1)
        ax1.set_ylabel("Ground Speed (m/s)")
        ax1.set_title("GS (all aircraft)")

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        for k in range(hdg_arr.shape[1]):
            ax2.plot(t_plot2, hdg_arr[:, k], color="tab:blue", alpha=0.1)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Heading (deg)")
        ax2.set_title("HDG (all aircraft)")

        plt.tight_layout()
        plt.show()

        # Trajectories: ownship-centric per pair (ownship start at 0,0; intruder relative)
        lat_arr = np.asarray(lat_list)
        lon_arr = np.asarray(lon_list)
        T, ntraf = lat_arr.shape
        R = 6371000.0

        plt.figure(figsize=(7, 7))
        for p in range(nb_pair):
            i_own = 2 * p
            i_int = 2 * p + 1
            if i_int >= ntraf:
                break

            lat0 = float(lat_arr[0, i_own])
            lon0 = float(lon_arr[0, i_own])
            lat0r = np.deg2rad(lat0)

            x_own = np.deg2rad(lon_arr[:, i_own] - lon0) * R * np.cos(lat0r)
            y_own = np.deg2rad(lat_arr[:, i_own] - lat0) * R

            x_int = np.deg2rad(lon_arr[:, i_int] - lon0) * R * np.cos(lat0r)
            y_int = np.deg2rad(lat_arr[:, i_int] - lat0) * R

            plt.plot(x_own, y_own, color="tab:blue", alpha=0.1)
            plt.plot(x_int, y_int, color="tab:blue", alpha=0.1)

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("East (m) relative to ownship start")
        plt.ylabel("North (m) relative to ownship start")
        plt.title(f"Ownship-centric trajectories | CI={ci_pos} m, CIV={ci_vel} m/s")
        plt.tight_layout()
        plt.show()

    pairwise.close()

    return {
        "ci": ci_pos,
        "civ": ci_vel,
        "ipr": ipr,
        "n_los": n_los,
        "nb_pair": nb_pair,
        "emp_p": emp_p,
        "asas_ticks": asas_ticks,
        "active_end": last_n_active,
    }


def main():
    results = []

    for ci in confidence_interval_list:
        for civ in confidence_interval_velo_list:
            do_plots = False
            if PLOT_ONE_CASE and (ci == PLOT_CI) and (civ == PLOT_CIV):
                do_plots = True

            reset_between_cases()
            out = run_one_case(ci, civ, seed=seed_base, do_plots=do_plots)
            results.append(out)

            print(
                f"CI={out['ci']:>6.2f} m | CIV={out['civ']:>4.1f} m/s | "
                f"IPR={out['ipr']:.4f} (LOS={out['n_los']}/{out['nb_pair']}) | "
                f"emp_p={out['emp_p']:.3f} | active_end={out['active_end']}"
            )

    # Optional: quick best/worst by IPR
    best = max(results, key=lambda r: r["ipr"])
    worst = min(results, key=lambda r: r["ipr"])

    print("\n=== Sweep summary ===")
    print(f"Best  IPR: {best['ipr']:.4f} at CI={best['ci']}, CIV={best['civ']}")
    print(f"Worst IPR: {worst['ipr']:.4f} at CI={worst['ci']}, CIV={worst['civ']}")


if __name__ == "__main__":
    main()
