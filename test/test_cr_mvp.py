
'''
The goal of this test is to make sure the MVP works well under pairwise
deterministic scenario for diff dpsi. Also do some plotting.
'''

import numpy as np
from sim_models.cd_statebased import StateBased
from envs.pairwise_conflict import PairwiseHorConflict
from sim_models.cr_mvp import MVP
from bluesky.tools.aero import kts

import matplotlib.pyplot as plt
from geopy.distance import geodesic

import bluesky as bs

from typing import Optional, Tuple

if not getattr(bs, "_joblib_inited", False):
    bs.init(mode="sim", detached=True)
    bs._joblib_inited = True

# Conflict detection and resolution setup
conf_detection = StateBased()
conf_resolution = MVP()

# Simulation grid and conflict parameters
width = 1
height = 1
horizontal_sep = 50  # in meters
lookahead_time = 15  # seconds
init_speed_ownship = 10.2889 # m/s (20 kts)
init_speed_intruder = 10.2889 # m/s (20 kts)
dpsi = 180  # degrees
aircraft_type = 'M600'

SIMDT_FACTOR = 1.0
DONE_TIMEOUT = 10.0
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

bs.traf.MAX_TR = 15
bs.traf.MAX_DTR2 = 10

# input conf_detection.tcpa_all, conf_detection.tinhor_all
def _check_tcpa_tinhor_per_pair(id, tcpa, tinhor):
    id = np.asarray(id, dtype=str)

    is_dro = np.char.startswith(id, "DRO")
    is_dri = np.char.startswith(id, "DRI")

    # Extract numeric suffix (e.g. '000', '001')
    num = np.array([s[3:] for s in id])

    # Build diagonal mask: DROxxx <-> DRIxxx only
    mask = (
        (is_dro & is_dri) == False  # placeholder, explained below
    )

    mask = np.zeros((len(id), len(id)), dtype=bool)

    for i in range(len(id)):
        if is_dro[i]:
            # Find matching DRI with same suffix
            match = np.where(is_dri & (num == num[i]))[0]
            if match.size:
                j = match[0]
                mask[i, j] = True
                mask[j, i] = True   # optional symmetry

    tcpa_sel = np.where(mask, tcpa, np.nan)
    tinhor_sel = np.where(mask, tinhor, np.nan)

    valid = ~np.isnan(tcpa_sel)

    if not np.any(valid):
        return False

    return np.all(tcpa_sel[valid] < 0) and np.all(tinhor_sel[valid] < 0)

def done_with_timeout(done_now: bool,
                      done_start_time: Optional[float],
                      sim_timer_second: float,
                      done_timeout: float,
                      *,
                      verbose: bool = False) -> Tuple[Optional[float], bool]:
    # Latch
    if done_now:
        if done_start_time is None:
            done_start_time = sim_timer_second
    else:
        done_start_time = None

    # Stop only after timeout
    should_stop = (
        done_start_time is not None
        and (sim_timer_second - done_start_time) >= done_timeout
    )

    if should_stop and verbose:
        print("Done + timeout reached, stopping simulation")

    return done_start_time, should_stop


pairwise = PairwiseHorConflict(
        pair_width=width, pair_height=height,
        asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time + 1,
        init_speed_ownship=init_speed_ownship, init_speed_intruder=init_speed_intruder,
        init_dpsi=dpsi, aircraft_type_ownship=aircraft_type,
        simdt_factor=SIMDT_FACTOR
    )

# now we do conflict resolution using MVP
simdt = bs.settings.simdt * SIMDT_FACTOR
tmax = lookahead_time * 4 # run for multiple times of the lookahead time

distance_array = []
distance_geodesic = []

sim_timer_second = 0.0

# --- record trajectory ---
time_list, id_list = [], []
lat_list, lon_list = [], []
gs_list, hdg_list = [], []

eps = np.finfo(float).eps * 100 
next_event_t = 0.0
event_dt = bs.settings.asas_dt

is_done = False

while sim_timer_second < tmax:
        states = pairwise._get_states() # speed is in m/s

        if sim_timer_second + eps >= next_event_t:
            # we add one to the lookahead time below so that the drone is iniated in conflict
            conf_detection.detect(states, states, horizontal_sep, 100, lookahead_time)
            reso = conf_resolution.resolve(conf_detection, states, states, bs.settings.asas_marh)
            reso_hdg, reso_spd, _, _, resopairs = reso

            # reso = None  # disable resolution for testing

            missed = int(np.floor((sim_timer_second - next_event_t) / event_dt)) + 1 if sim_timer_second > next_event_t else 1
            next_event_t += missed * event_dt
        
        distance_ = pairwise.step(reso)

        done_now, _n_active = _check_tcpa_tinhor_per_pair(
                        bs.traf.id, conf_detection.tcpa_all, conf_detection.tinhor_all
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

        print(f"{sim_timer_second}, {done_now}")
        distance_array.append(distance_)
        time_list.append(sim_timer_second)
        id_list.append(states.id)
        lat_list.append(states.lat)
        lon_list.append(states.lon)
        gs_list.append(states.gs)
        hdg_list.append(states.hdg)

        sim_timer_second += simdt

pairwise.reset()

# Do some plotting, only take the first pair
distance_array = np.array(distance_array)[:, :pairwise.nb_pair]

# Print smallest distance reached by every pair
for i in range(pairwise.nb_pair):
    print(f"Pair {i} min distance: {np.min(distance_array[:, i])} m")

distance_array = distance_array[:, 0]  # first pair only

pairwise.close()

# # Plot gs and hdg of first pair over time
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_list, [gs[0] for gs in gs_list], label='Ownship GS')
plt.plot(time_list, [gs[1] for gs in gs_list], label='Intruder GS')
plt.ylabel("Ground Speed (m/s)")
plt.title(f"Ground Speed of First Pair over Time - DPSI {dpsi} deg, RPZ {horizontal_sep} m")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time_list, [hdg[0] for hdg in hdg_list], label='Ownship HDG')
plt.plot(time_list, [hdg[1] for hdg in hdg_list], label='Intruder HDG')
plt.xlabel("Time (s)")
plt.ylabel("Heading (deg)")
plt.title(f"Heading of First Pair over Time - DPSI {dpsi} deg, RPZ {horizontal_sep} m")
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(time_list, distance_array)
plt.xlabel("Time (s)")
plt.ylabel("Distance between pair 0 (m)")
plt.title(f"Distance between first pair over time - DPSI {dpsi} deg, RPZ {horizontal_sep} m")
plt.axhline(y=horizontal_sep, color='r', linestyle='--', label='Horizontal Separation')
plt.show()

# Do plotting of trajectory of first pair with lat/lon and geodesic distance
first_pair_lat = np.array([lat_list[i][0:2] for i in range(len(lat_list))])
first_pair_lon = np.array([lon_list[i][0:2] for i in range(len(lon_list))])

geodesic_distances = []
for i in range(len(first_pair_lat)):
    dist = geodesic((first_pair_lat[i][0], first_pair_lon[i][0]),
                    (first_pair_lat[i][1], first_pair_lon[i][1])).meters
    geodesic_distances.append(dist)

plt.plot(first_pair_lon[:, 0], first_pair_lat[:, 0], label='Ownship')
plt.plot(first_pair_lon[:, 1], first_pair_lat[:, 1], label='Intruder')

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title(f"Trajectory with Horizontal Separation ({horizontal_sep} m)")
plt.legend()
plt.show()

data = np.column_stack((
    first_pair_lon[:, 0],
    first_pair_lat[:, 0],
    first_pair_lon[:, 1],
    first_pair_lat[:, 1],
))

header = "ownship_lon,ownship_lat,intruder_lon,intruder_lat"

np.savetxt(
    f"trajectory_{dpsi}.csv",
    data,
    delimiter=",",
    header=header,
    comments=""
)