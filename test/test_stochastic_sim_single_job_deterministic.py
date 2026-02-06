
'''
The goal of this test is to make sure the MVP works well under pairwise
deterministic scenario for diff dpsi. Also do some plotting.
'''

import numpy as np
from sim_models.cd_statebased import StateBased
from envs.pairwise_conflict import PairwiseHorConflict
from sim_models.cr_mvp import MVP
from sim_models.cr_vo import VO
from sim_models.adsl_module import ADSL
from sim_models.reception_model import ReceptionModel
from sim_models.crr_resumenav_heuristic import resumenav
from sim_models.crr_resumenav_ftr import resumenav_double_criteria
from sim_models.crr_resumenav_ftr_pastcpa import resumenav_triple_criteria

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
lookahead_time = 120
init_speed_ownship = cfg.init_speed_ownship # m/s
init_speed_intruder = cfg.init_speed_intruder # m/s
dpsi = 90
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

pairwise = PairwiseHorConflict(
        pair_width=width, pair_height=height,
        asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time + 1,
        init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
        init_dpsi=dpsi, aircraft_type_ownship=aircraft_type,
        simdt_factor=SIMDT_FACTOR
    )

# now we do conflict resolution using MVP
simdt = bs.settings.simdt * SIMDT_FACTOR
tmax = lookahead_time * cfg.tmax_factor  # e.g., 4x lookahead time

sim_timer_second = 0.0

is_done = False

# ADSL nodes
seed = 42

confidence_interval = cfg.confidence_interval
confidence_interval_velo = cfg.confidence_interval_velo
reception_prob = cfg.reception_prob

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

while sim_timer_second < 600:
    states = pairwise._get_states() # speed is in m/s
    n = int(states.ntraf)

    if not initialized:
        update_counts = np.zeros(n, dtype=int)
        ownship_adsl.update_from_truth(states)
        adsl_bus.send_data(intruder_adsl, ownship_adsl, indices=None)
        adsl_bus.send_data(prev_intruder_adsl, intruder_adsl, indices=None)

        initialized = True

    if sim_timer_second + eps >= next_event_t:
        # --- ADSL comm update (this is the only real change vs your deterministic test) ---
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

        # --- your existing detect/resolve using (possibly noisy) observations ---
        conf_detection.detect(ownship_obs, intruder_obs, horizontal_sep, 100.0, lookahead_time)
        conf_detection_groundtruth.detect(states, states, horizontal_sep, 100.0, lookahead_time)

        reso = conf_resolution.resolve(conf_detection, ownship_obs, intruder_obs, 1.05)

        conf_detection.sigma_r = ownship_adsl.pos_std + ownship_adsl.pos_std
        conf_detection.sigma_v = ownship_adsl.vel_std + ownship_adsl.vel_std

        delpairs_noise = resumenav_triple_criteria(conf_resolution, conf_detection, ownship_obs, intruder_obs)

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

    print(f"{sim_timer_second:.2f}, done={done_now}, active_pairs={np.sum(n_active)}")

    done_start_time, should_stop = done_with_timeout(
        done_now=done_now,
        done_start_time=done_start_time,
        sim_timer_second=sim_timer_second,
        done_timeout=DONE_TIMEOUT,
        verbose=True,
    )
    if should_stop:
        break

    # print(f"{sim_timer_second:.2f}, done={done_now}")

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

print(f"CPA: {min_dist_per_pair} m")
print(f"IPR: {ipr:.4f}  (LOS={n_los}/{n_conflict})")

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

print("\n=== Reception stats ===")
print(f"Target reception_prob: {reception_prob:.3f}")
print(f"ASAS ticks: {asas_ticks}")
print(f"Empirical mean reception: {emp_p:.3f}")
# print("Per-aircraft reception freq:", np.round(update_counts / max(1, asas_ticks), 3).tolist())

# ----------------------------
# Plotting (DRO = tab:blue, DRI = tab:red)
# ----------------------------

# --- Plot distance of ALL pairs over time (distance is per-pair, keep as blue) ---
plt.figure(figsize=(12, 6))
for i in range(pairwise.nb_pair):
    plt.plot(t_plot, distance_array[:, i], color="tab:blue", alpha=0.1)
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.title(f"Distance for ALL pairs over time - DPSI {dpsi} deg, RPZ {horizontal_sep} m")
plt.axhline(y=horizontal_sep, color="r", linestyle="--")
plt.tight_layout()
plt.show()

# --- Plot GS and HDG for ALL aircraft (DRO blue, DRI red) ---
T = len(time_list)
t_plot = np.array(time_list, dtype=float)
nb = pairwise.nb_pair

gs_arr  = np.asarray(gs_list)   # (T, ntraf)
hdg_arr = np.asarray(hdg_list)  # (T, ntraf)

# Determine colors based on IDs at t=0 (stable ordering)
id0 = np.asarray(id_list[0], dtype=str)  # e.g. ["DRO000", "DRI000", ...]
is_dro = np.char.startswith(id0, "DRO")
is_dri = np.char.startswith(id0, "DRI")

# Wrap heading to [-180, 180): 359 -> -1
hdg_wrapped = ((hdg_arr + 180.0) % 360.0) - 180.0

plt.figure(figsize=(12, 6))

# GS subplot
ax1 = plt.subplot(2, 1, 1)
nplot = min(gs_arr.shape[1], 2 * nb)
for k in range(nplot):
    color = "tab:blue" if is_dro[k] else ("tab:red" if is_dri[k] else "tab:blue")
    ax1.plot(t_plot, gs_arr[:T, k], color=color, alpha=0.1)
ax1.set_ylabel("Ground Speed (m/s)")
ax1.set_title(f"Ground Speed (DRO blue, DRI red) - DPSI {dpsi} deg, RPZ {horizontal_sep} m")

# HDG subplot
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
for k in range(nplot):
    color = "tab:blue" if is_dro[k] else ("tab:red" if is_dri[k] else "tab:blue")
    ax2.plot(t_plot, hdg_wrapped[:T, k], color=color, alpha=0.1)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Heading (deg)")
ax2.set_title(f"Heading (DRO blue, DRI red) - DPSI {dpsi} deg, RPZ {horizontal_sep} m")

plt.tight_layout()
plt.show()

# --- Trajectories in ownship-centric frame (per pair):
#     DRO (ownship) = tab:blue, DRI (intruder) = tab:red ---
lat_arr = np.asarray(lat_list)  # (T, ntraf)
lon_arr = np.asarray(lon_list)  # (T, ntraf)
T, ntraf = lat_arr.shape
nb = pairwise.nb_pair
R = 6371000.0

plt.figure(figsize=(7, 7))

for p in range(nb):
    i_own = 2 * p
    i_int = 2 * p + 1
    if i_int >= ntraf:
        break

    # Origin = ownship position at t=0 for this pair
    lat0 = float(lat_arr[0, i_own])
    lon0 = float(lon_arr[0, i_own])
    lat0r = np.deg2rad(lat0)

    x_own = np.deg2rad(lon_arr[:, i_own] - lon0) * R * np.cos(lat0r)
    y_own = np.deg2rad(lat_arr[:, i_own] - lat0) * R

    x_int = np.deg2rad(lon_arr[:, i_int] - lon0) * R * np.cos(lat0r)
    y_int = np.deg2rad(lat_arr[:, i_int] - lat0) * R

    plt.plot(x_own, y_own, color="tab:blue", alpha=0.1)  # DRO
    plt.plot(x_int, y_int, color="tab:red", alpha=0.1)   # DRI

plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("East (m) relative to ownship start")
plt.ylabel("North (m) relative to ownship start")
plt.title(f"Ownship-centric trajectories (DRO blue, DRI red) - DPSI {dpsi}, RPZ {horizontal_sep} m")
plt.tight_layout()
plt.show()