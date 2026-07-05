import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from bluesky.tools import geo
from bluesky.tools.aero import cas2tas, casormach2tas, fpm, kts

import bluesky as bs

import json

M2NM = 1/1852
NM2M = 1852

DCPA_M = 0  # initial dcpa in meters

ALT = 100

## read params
with open("envs/pairwise_params.json", "r") as f:
    params = json.load(f)

# Access parameters
start_lat = params["start_lat"]
start_lon = params["start_lon"]
delta_lat_lon = params["delta_lat_lon"]

class PairwiseHorConflict():
    """ 
    PairwiseHorConflict
    """

    def __init__(self, 
                pair_width: int, pair_height: int,      ## number of spawned aircraft
                asas_pzr_m: float, dtlookahead: float,  ## separation standard params
                init_speed_ownship: float, init_speed_intruder: float,      ## aircraft params
                aircraft_type_ownship: str, aircraft_type_intruder: str = None,     ## aircraft params
                init_dpsi: float = None, simdt_factor: int = 1) -> None:
        
        self.traj = {}

        self.nb_pair = pair_width * pair_height

        self.asas_pzr_m = asas_pzr_m
        self.dtlookahead = dtlookahead

        # Per-pair speeds: accept a scalar (same speed for every pair) or a
        # length-nb_pair array (per-pair heterogeneity, e.g. exp3). Broadcast to
        # a length-nb_pair array so both creation and _do_action can index by pair.
        self.init_speed_ownship = np.broadcast_to(
            np.asarray(init_speed_ownship, dtype=float), (self.nb_pair,)
        ).copy()
        self.init_speed_intruder = np.broadcast_to(
            np.asarray(init_speed_intruder, dtype=float), (self.nb_pair,)
        ).copy()

        # we measure the distance
        self.distance_array = np.zeros((self.nb_pair))

        self.aircraft_type_ownship = aircraft_type_ownship
        if(aircraft_type_intruder == None):
            self.aircraft_type_intruder = aircraft_type_ownship
        else:
            self.aircraft_type_intruder = aircraft_type_intruder

        # this is to generate the heading, 0 for ownship
        # init_dpsi for intruder if init_dpsi is not None
        # init_dpsi may be a scalar (same crossing angle for every pair) or a
        # length-nb_pair array (per-pair crossing angle, e.g. exp3). Even indices
        # are ownships (heading 0); odd indices are intruders (crossing angle).
        if init_dpsi is not None:
            dpsi_arr = np.broadcast_to(
                np.asarray(init_dpsi, dtype=float), (self.nb_pair,)
            )
            self.init_heading = np.zeros(2 * self.nb_pair)
            self.init_heading[1::2] = dpsi_arr
        else:
            self.init_heading = np.array([
                                            0 if i % 2 == 0 else np.random.randint(0, 360)
                                            for i in range(2 * pair_width * pair_height)
                                        ])

        # set conflict definition
        bs.settings.asas_pzr = self.asas_pzr_m * M2NM
        bs.settings.asas_dtlookahead = self.dtlookahead

        # set simulation time step, and enable fast-time running
        simdt = bs.settings.simdt * simdt_factor
        bs.stack.stack(f"DT {simdt}")

        dcpa = DCPA_M * M2NM

        # create drones
        counter = 0
        idx = 0

        # Precompute IDs only once
        self.ownship_ids   = []
        self.intruder_ids  = []

        self.ownship_idx = []
        self.intruder_idx = []

        for i in range(pair_width):
            for j in range(pair_height):
                ownship_id = f"DRO{counter:03}"
                intruder_id = f"DRI{counter:03}"                

                aclats = start_lat + i * delta_lat_lon
                aclons = start_lon + j * delta_lat_lon
                
                ## the heading of this one is always zero
                bs.traf.cre(acid=ownship_id, actype= self.aircraft_type_ownship, aclat=aclats, aclon=aclons,
                    achdg=self.init_heading[idx], acalt=ALT, acspd=self.init_speed_ownship[counter])
                self.ownship_ids.append(ownship_id)
                self.ownship_idx.append(idx)

                idx += 1

                ## make intruder, dpsi is random
                bs.traf.creconfs(acid=intruder_id, actype = self.aircraft_type_intruder, targetidx=bs.traf.id2idx(ownship_id),
                                dpsi=self.init_heading[idx], dcpa = dcpa, tlosh = bs.settings.asas_dtlookahead, spd = self.init_speed_intruder[counter])
                self.intruder_ids.append(intruder_id)
                self.intruder_idx.append(idx)

                idx += 1
                
                counter += 1
            
    def reset(self) -> None:      
        bs.traf.reset()

    def _get_states(self):
        return bs.traf
    
    def _do_action(self, action):
        if(action != None):
            reso_hdg, reso_spd, _, _, resopairs = action

        ntraf = self._get_states().ntraf

        for i in range(ntraf):
            target_id = bs.traf.id[i]

            # Per-pair nominal speed: DRO### / DRI### -> pair index ###.
            pair_idx = int(target_id[3:])
            if("DRO" in target_id):
                nom_spd = self.init_speed_ownship[pair_idx]
            else:
                nom_spd = self.init_speed_intruder[pair_idx]

            if action != None:
                if(any(target_id in pair for pair in resopairs)):
                    bs.stack.stack(f"HDG {target_id}, {reso_hdg[i]}")
                    bs.stack.stack(f"SPD {target_id}, {reso_spd[i] / kts}") # this should be in kts
                else:
                    bs.stack.stack(f"HDG {target_id}, {self.init_heading[i]}")
                    bs.stack.stack(f"SPD {target_id}, {nom_spd / kts}") # this should be in kts
            else:
                bs.stack.stack(f"HDG {target_id}, {self.init_heading[i]}")
                bs.stack.stack(f"SPD {target_id}, {nom_spd / kts}") # speed in kts

    def _update_distance(self) -> None:
        # Gather lat/lon arrays for all pairs
        lat_dro = np.array([bs.traf.lat[idx] for idx in self.ownship_idx])
        lon_dro = np.array([bs.traf.lon[idx] for idx in self.ownship_idx])
        lat_dri = np.array([bs.traf.lat[idx] for idx in self.intruder_idx])
        lon_dri = np.array([bs.traf.lon[idx] for idx in self.intruder_idx])

        # Compute all distances in one vectorized call
        dist = geo.latlondist_matrix(
            np.asmatrix(lat_dro),
            np.asmatrix(lon_dro),
            np.asmatrix(lat_dri),
            np.asmatrix(lon_dri)
        )

        # Store results in meters
        self.distance_array[:] = np.diag(dist) * NM2M

    def step(self, action) -> np.ndarray:
        
        self._do_action(action)

        bs.sim.step()

        self._update_distance()

        return np.array(self.distance_array)
    
    def close(self) -> None:
        # Do NOT quit the BlueSky simulator here.
        # Just clear traffic/state that this env created.
        bs.traf.reset()


# ---------------------------------------------------------------------------
# PairwiseHorConflictDist
# ---------------------------------------------------------------------------
class PairwiseHorConflictDist:
    """
    Like PairwiseHorConflict but creates intruders at a fixed initial distance
    via creconfs_dist, with per-pair randomized dpsi (0–360°), ownship/intruder
    speeds (speed_range m/s), and dcpa (dcpa_range_m metres).
    """

    def __init__(self,
                 pair_width: int, pair_height: int,
                 asas_pzr_m: float, dtlookahead: float,
                 dist_nm: float,
                 aircraft_type_ownship: str,
                 aircraft_type_intruder: str = None,
                 speed_range: tuple = (7.7167, 18.0056),  # 15–35 kts in m/s
                 dcpa_range_m: tuple = (0.0, 50.0),
                 simdt_factor: int = 1,
                 rng: np.random.Generator = None) -> None:

        self.nb_pair = pair_width * pair_height
        self.asas_pzr_m = asas_pzr_m
        self.dtlookahead = dtlookahead
        self.dist_nm = dist_nm
        self.distance_array = np.zeros(self.nb_pair)
        self.aircraft_type_ownship = aircraft_type_ownship
        self.aircraft_type_intruder = aircraft_type_ownship if aircraft_type_intruder is None else aircraft_type_intruder

        if rng is None:
            rng = np.random.default_rng()

        n = self.nb_pair
        dpsi_arr          = rng.uniform(0, 360, n)
        self.ownship_speeds  = rng.uniform(*speed_range, n)   # m/s
        self.intruder_speeds = rng.uniform(*speed_range, n)   # m/s
        dcpa_nm_arr          = rng.uniform(*dcpa_range_m, n) * M2NM

        # init_heading[2k] = ownship heading (0°), init_heading[2k+1] = intruder abs track = dpsi
        self.init_heading = np.zeros(2 * n)
        for k in range(n):
            self.init_heading[2 * k + 1] = dpsi_arr[k]

        self.ownship_ids  = []
        self.intruder_ids = []
        self.ownship_idx  = []
        self.intruder_idx = []

        bs.settings.asas_pzr = self.asas_pzr_m * M2NM
        bs.settings.asas_dtlookahead = self.dtlookahead
        simdt = bs.settings.simdt * simdt_factor
        bs.stack.stack(f"DT {simdt}")

        counter = 0
        idx = 0
        for i in range(pair_width):
            for j in range(pair_height):
                ownship_id  = f"DRO{counter:03}"
                intruder_id = f"DRI{counter:03}"

                aclats = start_lat + i * delta_lat_lon
                aclons = start_lon + j * delta_lat_lon

                bs.traf.cre(
                    acid=ownship_id,
                    actype=self.aircraft_type_ownship,
                    aclat=aclats, aclon=aclons,
                    achdg=0.0, acalt=ALT,
                    acspd=self.ownship_speeds[counter],
                )
                self.ownship_ids.append(ownship_id)
                self.ownship_idx.append(idx)
                idx += 1

                bs.traf.creconfs_dist(
                    acid=intruder_id,
                    actype=self.aircraft_type_intruder,
                    targetidx=bs.traf.id2idx(ownship_id),
                    dpsi=dpsi_arr[counter],
                    dcpa=dcpa_nm_arr[counter],
                    dist=self.dist_nm,
                    spd=self.intruder_speeds[counter],
                )
                self.intruder_ids.append(intruder_id)
                self.intruder_idx.append(idx)
                idx += 1

                counter += 1

    def reset(self) -> None:
        bs.traf.reset()

    def _get_states(self):
        return bs.traf

    def _update_distance(self) -> None:
        lat_dro = np.array([bs.traf.lat[i] for i in self.ownship_idx])
        lon_dro = np.array([bs.traf.lon[i] for i in self.ownship_idx])
        lat_dri = np.array([bs.traf.lat[i] for i in self.intruder_idx])
        lon_dri = np.array([bs.traf.lon[i] for i in self.intruder_idx])

        dist = geo.latlondist_matrix(
            np.asmatrix(lat_dro), np.asmatrix(lon_dro),
            np.asmatrix(lat_dri), np.asmatrix(lon_dri),
        )
        self.distance_array[:] = np.diag(dist) * NM2M

    def _do_action(self, action) -> None:
        if action is not None:
            reso_hdg, reso_spd, _, _, resopairs = action

        ntraf = self._get_states().ntraf
        for i in range(ntraf):
            target_id = bs.traf.id[i]
            pair_idx  = int(target_id[3:])
            is_own    = target_id.startswith("DRO")
            nom_spd   = self.ownship_speeds[pair_idx] if is_own else self.intruder_speeds[pair_idx]

            if action is not None and any(target_id in pair for pair in resopairs):
                bs.stack.stack(f"HDG {target_id}, {reso_hdg[i]}")
                bs.stack.stack(f"SPD {target_id}, {reso_spd[i] / kts}")
            else:
                bs.stack.stack(f"HDG {target_id}, {self.init_heading[i]}")
                bs.stack.stack(f"SPD {target_id}, {nom_spd / kts}")

    def step(self, action) -> np.ndarray:
        self._do_action(action)
        bs.sim.step()
        self._update_distance()
        return np.array(self.distance_array)

    def close(self) -> None:
        bs.traf.reset()
