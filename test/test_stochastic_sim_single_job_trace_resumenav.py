'''
The goal of this test is to make sure the MVP works well under pairwise
deterministic scenario for diff dpsi. Also do some plotting.
'''

import numpy as np
import pandas as pd
from collections import Counter

from sim_models.cd_statebased import StateBased
from envs.pairwise_conflict import PairwiseHorConflict
from sim_models.cr_mvp import MVP
from sim_models.cr_vo import VO
from sim_models.adsl_module import ADSL
from sim_models.reception_model import ReceptionModel

from sim.utils import _check_tcpa_tinhor_per_pair, done_with_timeout, get_configs

import matplotlib.pyplot as plt
from geopy.distance import geodesic

import bluesky as bs
from bluesky.tools.aero import kts

if not getattr(bs, "_joblib_inited", False):
    bs.init(mode="sim", detached=True)
    bs._joblib_inited = True

cfg = get_configs()

# Simulation grid and conflict parameters
width = 10
height = 10
horizontal_sep = cfg.horizontal_sep  # in meters
lookahead_time = 60
init_speed_ownship = cfg.init_speed_ownship # m/s
init_speed_intruder = cfg.init_speed_intruder # m/s
dpsi = 2 # degrees
aircraft_type = cfg.aircraft_type

SIMDT_FACTOR = cfg.SIMDT_FACTOR

# this is to terminate the sim
# if the tcpa and tin has been negative for DONE_TIMEOUT sec
# we can terminate sim

DONE_TIMEOUT = cfg.DONE_TIMEOUT
done_start_time = None

try:
    max_tr = bs.traf.MAX_TR
    max_dtr2 = bs.traf.MAX_DTR2
    print(f"Make sure this is available (only for M600): max_tr: {max_tr}, max_dtr2: {max_dtr2}")
except AttributeError as e:
    raise RuntimeError(
        "Required BlueSky turn rate limit attributes are missing: "
        "bs.traf.MAX_TR and/or bs.traf.MAX_DTR2. "
        "Please update/install a BlueSky version that includes them."
    ) from e

# Conflict detection and resolution setup
# Unfortunately I have to use two StateBased CD
# One is feeded to the conf_reso
# Another one is to keep track of the sim
# Conflict detection is fixed in your config right now (StateBased)
conf_detection = StateBased()
conf_detection_groundtruth = StateBased()

# Conflict resolution chosen by config
if cfg.resolution_model == "MVP":
    conf_resolution = MVP()
elif cfg.resolution_model == "VO":
    conf_resolution = VO()
else:
    raise ValueError(f"Unsupported resolution model: {cfg.resolution_model}")

stats = {}

for dpsi in range(2, 46, 2):
    pairwise = PairwiseHorConflict(
            pair_width=width, pair_height=height,
            asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time + 1,
            init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
            init_dpsi=dpsi, aircraft_type_ownship=aircraft_type,
            simdt_factor=SIMDT_FACTOR
        )

    # now we do conflict resolution using MVP
    simdt = bs.settings.simdt * SIMDT_FACTOR
    tmax = 300

    sim_timer_second = 0.0

    is_done = False

    # ADSL nodes
    seed = 44

    confidence_interval = cfg.confidence_interval
    confidence_interval_velo = cfg.confidence_interval_velo
    reception_prob = 1.0

    # "bus" is just used to copy messages (send_data)
    adsl_bus = ADSL(confidence_interval, confidence_interval_velo, reception_prob=1.0, seed=seed + 1)

    ownship_adsl = ADSL(confidence_interval, confidence_interval_velo, reception_prob=1.0, seed=seed + 2)
    intruder_adsl = ADSL(confidence_interval, confidence_interval_velo, reception_prob=reception_prob, seed=seed + 3)

    prev_intruder_adsl = ADSL(confidence_interval, confidence_interval_velo, reception_prob=1.0, seed=seed + 4)

    # Include reception model
    rx_rng = np.random.default_rng(seed + 999)
    intruder_adsl.reception = ReceptionModel(reception_prob=reception_prob, rng=rx_rng)

    # --- record trajectory ---
    distance_array = []
    time_list, id_list = [], []
    lat_list, lon_list = [], []
    gs_list, hdg_list = [], []

    # For reception stats (per-aircraft, per ASAS tick)
    update_counts = None
    asas_ticks = 0

    eps = np.finfo(float).eps * 100
    next_event_t = 0.0
    event_dt = float(bs.settings.asas_dt)

    initialized = False

    # ============================================================
    # TP/FP/FN/TN (micro, over ALL directed DRI<->DRO pairs)
    # - Directed pairs counted separately: (DRIxxx, DROyyy) != (DROyyy, DRIxxx)
    # - TN requires a universe size: |U| = 2 * n_dri * n_dro
    #   We infer n_dri and n_dro from the IDs present in the scenario.
    # ============================================================
    micro_cm = Counter(TP=0, FP=0, FN=0, TN=0)
    cm_ticks = 0
    U_size = None  # will be computed once we see the IDs

    def _infer_universe_size_from_ids(ids_iterable):
        """Infer |U| = 2 * n_dri * n_dro from IDs like DRI000, DRO003, etc."""
        ids = [str(x) for x in ids_iterable]
        n_dri = sum(s.startswith("DRI") for s in ids)
        n_dro = sum(s.startswith("DRO") for s in ids)
        return 2 * n_dri * n_dro, n_dri, n_dro

    while sim_timer_second < tmax:
        states = pairwise._get_states() # speed is in m/s
        n = int(states.ntraf)

        if not initialized:
            update_counts = np.zeros(n, dtype=int)
            ownship_adsl.update_from_truth(states)
            adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=None)
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

            # Compute universe size once (needed for TN)
            # Uses the set of aircraft currently in the sim.
            U_size, n_dri, n_dro = _infer_universe_size_from_ids(states.id)
            print(f"[CM] Universe size |U| = 2*n_dri*n_dro = {U_size} (n_dri={n_dri}, n_dro={n_dro})")

            initialized = True

        if sim_timer_second + eps >= next_event_t:
            # --- ADSL comm update ---
            ownship_adsl.update_from_truth(states)

            # reception_model.py decides who receives
            idx_rx = intruder_adsl.reception.sample_indices(n)
            rx_mask = np.zeros(n, dtype=bool)
            rx_mask[idx_rx] = True
            idx_miss = np.where(~rx_mask)[0]

            # Apply comms:
            if idx_rx.size > 0:
                adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=idx_rx)
                update_counts[idx_rx] += 1
            if idx_miss.size > 0:
                adsl_bus.send_data(intruder_adsl, prev_intruder_adsl, indices=idx_miss)

            # Update hold buffer for next tick
            adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

            asas_ticks += 1

            # Choose what the conflict detection observes
            ownship_obs = ownship_adsl
            intruder_obs = intruder_adsl

            # --- detect/resolve using (possibly noisy) observations ---
            conf_detection.detect(ownship_obs, intruder_obs, horizontal_sep, 100.0, lookahead_time)
            conf_detection_groundtruth.detect(states, states, horizontal_sep, 100.0, lookahead_time)

            reso = conf_resolution.resolve(conf_detection, ownship_obs, intruder_obs, bs.settings.asas_marh)

            delpairs_noise = conf_resolution.resumenav(conf_detection, ownship_obs, intruder_obs)
            delpairs_true = conf_resolution.resumenav(conf_detection_groundtruth, states, states)

            # print(round(sim_timer_second, 2), delpairs_noise, delpairs_true)

            # ----------------------------
            # Confusion matrix update (micro)
            # ----------------------------
            # Ensure these are sets (they should already be)
            pred = set(delpairs_noise)
            gt   = set(delpairs_true)

            tp = len(pred & gt)
            fp = len(pred - gt)
            fn = len(gt - pred)

            # TN depends on the full directed-pair universe size
            # |U| = 2 * n_dri * n_dro
            tn = (U_size - tp - fp - fn) if U_size is not None else 0

            micro_cm["TP"] += tp
            micro_cm["FP"] += fp
            micro_cm["FN"] += fn
            micro_cm["TN"] += tn
            cm_ticks += 1

            # advance next_event_t exactly like you did
            missed = (
                int(np.floor((sim_timer_second - next_event_t) / event_dt)) + 1
                if sim_timer_second > next_event_t
                else 1
            )
            next_event_t += missed * event_dt

        # Step en
        distance_ = pairwise.step(reso)
        distance_array.append(distance_)
        
        # Record truth trajectory
        time_list.append(sim_timer_second)
        id_list.append(states.id)
        lat_list.append(states.lat)
        lon_list.append(states.lon)
        gs_list.append(states.gs)
        hdg_list.append(states.hdg)

        # Done logic
        done_now, n_active = _check_tcpa_tinhor_per_pair(
                            bs.traf.id, conf_detection_groundtruth.tcpa_all, conf_detection_groundtruth.tinhor_all
        )

        done_start_time, should_stop = done_with_timeout(
            done_now=done_now,
            done_start_time=done_start_time,
            sim_timer_second=sim_timer_second,
            done_timeout=DONE_TIMEOUT,
            verbose=True,
        )
        if should_stop:
            break

        sim_timer_second += simdt

    # ----------------------------
    # Cleanup
    # ----------------------------
    pairwise.reset()

    # Convert distances, first pair only (same as yours)
    distance_array = np.array(distance_array)[:, :pairwise.nb_pair]
    Tdist = distance_array.shape[0]
    t_plot = np.array(time_list[:Tdist], dtype=float)

    print("Horizontal margin: ", bs.settings.asas_marh)

    # Calculate IPR
    min_dist_per_pair = np.min(distance_array, axis=0)  # (nb_pair,)
    n_los = int(np.sum(min_dist_per_pair < horizontal_sep))
    n_conflict = int(pairwise.nb_pair)  # by your definition

    ipr = 1.0 - (n_los / float(n_conflict))  # (n_conflict - n_los)/n_conflict

    print(f"Worst CPA: {min(min_dist_per_pair):.2f} m")
    print(f"dpsi: {dpsi}, IPR: {ipr:.4f}, (LOS={n_los}/{n_conflict})")

    distance_pair0 = distance_array[:, 0]
    pairwise.close()

    # ----------------------------
    # Print reception stats (new)
    # ----------------------------
    # update_counts counts how many times each aircraft received (excluding tick0 initialization)
    if asas_ticks > 0:
        emp_p = float(np.mean(update_counts / float(asas_ticks)))
    else:
        emp_p = float("nan")

    # print("\n=== Reception stats ===")
    # print(f"Target reception_prob: {reception_prob:.3f}")
    # print(f"ASAS ticks: {asas_ticks}")
    # print(f"Empirical mean reception: {emp_p:.3f}")

    # ----------------------------
    # Print TP/FP/FN/TN stats (new)
    # ----------------------------
    # print("\n=== Conflict-pair detection confusion (micro, directed DRI<->DRO) ===")
    # print(f"CM ticks evaluated: {cm_ticks}")
    # if U_size is not None:
    #     print(f"Universe size per tick |U|: {U_size} (2*n_dri*n_dro)")
    #     print(f"Total binary decisions: {cm_ticks * U_size}")
    # print(f"TP={micro_cm['TP']}  FP={micro_cm['FP']}  FN={micro_cm['FN']}  TN={micro_cm['TN']}")

    # Optional derived rates (guard division by zero)
    den_pos = micro_cm["TP"] + micro_cm["FN"]
    den_pred = micro_cm["TP"] + micro_cm["FP"]
    den_all = micro_cm["TP"] + micro_cm["FP"] + micro_cm["FN"] + micro_cm["TN"]
    den_threat = micro_cm["TP"] + micro_cm["FP"] + micro_cm["FN"]

    tpr = (micro_cm["TP"] / den_pos) if den_pos else float("nan")      # recall
    ppv = (micro_cm["TP"] / den_pred) if den_pred else float("nan")    # precision
    acc = (micro_cm["TP"] + micro_cm["TN"]) / den_all if den_all else float("nan")
    csi = (micro_cm["TP"] / den_threat) if den_threat else 0.0

    # print(f"Precision={ppv:.4f}  Recall={tpr:.4f}  Accuracy={acc:.6f}")

    fpr = micro_cm["FP"] / (micro_cm["FP"] + micro_cm["TN"]) if (micro_cm["FP"] + micro_cm["TN"]) else float("nan")
    # print(f"FPR={fpr:.6e}  (lower is better)")

    stats[dpsi] = {"recall": tpr, "precision": ppv,
                   "accuracy": acc, "FPR": fpr,
                   "csi": csi,
                   "true_pos": micro_cm["TP"], "true_neg": micro_cm["TN"],
                   "false_pos": micro_cm["FP"], "false_neg": micro_cm["FN"],
                   "ipr": ipr}

# Convert {dpsi: {metric: value}} -> table
df = (
    pd.DataFrame.from_dict(stats, orient="index")
      .rename_axis("dpsi")
      .reset_index()
      .sort_values("dpsi")
)

# Optional: if you really want percent units in the CSV (your plots label [%])
# df[["recall", "precision", "accuracy", "FPR", "csi"]] *= 100.0

out_csv = "results/misc/resumenav_stats_by_dpsi.csv"
df.to_csv(out_csv, index=False)
print(f"Saved stats to: {out_csv}")

dpsis = sorted(stats.keys())
accuracy = [stats[d]["accuracy"] for d in dpsis]
fpr = [stats[d]["FPR"] for d in dpsis]
recall = [stats[d]["csi"] for d in dpsis]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(dpsis, fpr, marker="o")
ax1.set_ylabel("FPR [%]")
# ax1.set_title("False Positive Rate [%]")
# ax1.grid(True)

ax2.plot(dpsis, accuracy, marker="o")
ax2.set_ylabel("Accuracy [%]")
# ax2.grid(True)

plt.show()