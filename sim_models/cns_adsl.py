''' Communication, navigation, surveillance model. '''
import numpy as np
from bluesky.tools import geo
from bluesky.tools.aero import nm

class ADSL():
    """ 
    
    """
    def __init__(self, confidence_interval, confidence_interval_velo,
                 reception_prob = 1.0):
        # Calculate standard deviation from confidence interval
        # For 2D, 95% confidence interval is approximately 2.448 standard deviations
        self.reception_prob = reception_prob

        self.std_dev = confidence_interval / 2.448
        self.velo_std_dev = confidence_interval_velo / 2.448

        self.ntraf = 0
        self.lat     = np.array([])  # latitude [deg]
        self.lon     = np.array([])  # longitude [deg]
        self.alt     = np.array([])  # altitude [m]
        self.hdg     = np.array([])  # traffic heading [deg]
        self.trk     = np.array([])  # track angle [deg]
        self.gs      = np.array([])  # ground speed [m/s]
        self.gseast  = np.array([])  # ground speed east [m/s]
        self.gsnorth = np.array([])  # ground speed north [m/s]
        self.vs      = np.array([])  # vertical speed [m/s]
        self.id      = []  # identifier (string)

        self.first_update_done = False
        
    def _get_noisy_pos(self, states, update_idx=None):
        self.ntraf = states.ntraf

        # Make update_idx a plain 1D index array
        if update_idx is None:
            update_idx = np.arange(self.ntraf)
            first = True
        else:
            # if caller passes np.where(...), it may be a tuple
            if isinstance(update_idx, tuple):
                update_idx = update_idx[0]
            update_idx = np.asarray(update_idx, dtype=int)
            first = False

        # On first call, allocate/copy everything once
        if first or self.lat.size == 0:
            self.lat = states.lat.copy()
            self.lon = states.lon.copy()
            self.alt = states.alt.copy()
            self.hdg = states.hdg.copy()
            self.trk = states.trk.copy()
            self.gs  = states.gs.copy()
            self.tas = states.tas.copy()
            self.vs  = states.vs.copy()
            self.id  = states.id.copy()

            self.gseast  = states.gseast.copy()
            self.gsnorth = states.gsnorth.copy()

            self.perf = states.perf
            self.ap = states.ap
            self.selalt = states.selalt
        else:
            # On partial update, refresh ONLY the aircraft that received a packet
            self.alt[update_idx] = states.alt[update_idx]
            self.hdg[update_idx] = states.hdg[update_idx]
            self.trk[update_idx] = states.trk[update_idx]
            self.gs[update_idx]  = states.gs[update_idx]
            self.tas[update_idx] = states.tas[update_idx]
            self.vs[update_idx]  = states.vs[update_idx]
            # id typically doesn't change

        # ---- Position noise only for updated aircraft ----
        cov = np.array([[self.std_dev**2, 0.0],
                        [0.0, self.std_dev**2]])

        # draw len(update_idx) samples at once: shape (K, 2)
        xy = np.random.multivariate_normal((0.0, 0.0), cov, size=len(update_idx))
        x = xy[:, 0]
        y = xy[:, 1]

        mean_lat = states.lat[update_idx]
        lat_noise = y / 111320.0

        coslat = np.cos(np.deg2rad(mean_lat))
        coslat = np.maximum(coslat, 1e-6)  # avoid blow-up near poles
        lon_noise = x / (111320.0 * coslat)

        self.lat[update_idx] = states.lat[update_idx] + lat_noise
        self.lon[update_idx] = states.lon[update_idx] + lon_noise

    def _get_noisy_velo(self, states, update_array=None):
        self.ntraf = states.ntraf

        cov_velo = np.array([[self.velo_std_dev**2, 0],
                            [0, self.velo_std_dev**2]])

        if update_array is not None:
            idx = update_array[0] if isinstance(update_array, tuple) else update_array

            vx_noise, vy_noise = np.random.multivariate_normal((0, 0), cov_velo, len(idx)).T

            self.gsnorth[idx] = self.gs[idx] * np.cos(np.deg2rad(self.trk[idx])) + vx_noise
            self.gseast[idx]  = self.gs[idx] * np.sin(np.deg2rad(self.trk[idx])) + vy_noise
        else:
            vx_noise, vy_noise = np.random.multivariate_normal((0, 0), cov_velo, self.ntraf).T
            self.gsnorth = self.gs * np.cos(np.deg2rad(self.trk)) + vx_noise
            self.gseast  = self.gs * np.sin(np.deg2rad(self.trk)) + vy_noise
    
    def _get_noisy_states(self, states):
        ## Still buggy the update_prob_cond, actually require the comm uncertainty to be assymetrical
        update_prob_cond = (np.random.random(size = states.ntraf) <= self.reception_prob)
        up = np.where(update_prob_cond)
        
        if not self.first_update_done:
            self._get_noisy_pos(states)
            self._get_noisy_velo(states)

            self.first_update_done = True
        else:
            self._get_noisy_pos(states, up)
            self._get_noisy_velo(states, up)

    def send_data(self, dst_adsl, src_adsl, indices=None):
        """Copy measurement from src to dst. This simulates sending data from ownship to intruder
        If indices is None: full copy. Else, only update specific aircraft.
        """

        MEAS_FIELDS = [
                        "ntraf", "id", "lat", "lon", "alt", "hdg", "trk", "gs", "vs",
                        "gseast", "gsnorth"
                    ]
        
        for f in MEAS_FIELDS:
            src_val = getattr(src_adsl, f)
            dst_val = getattr(dst_adsl, f)

            if isinstance(src_val, np.ndarray):
                if indices is None:
                    setattr(dst_adsl, f, src_val.copy())
                else:
                    if dst_val.shape != src_val.shape:
                        setattr(dst_adsl, f, src_val.copy())
                    else:

                        dst_val[indices] = src_val[indices]
            else:
                if indices is None:
                    setattr(dst_adsl, f, src_val)
                else:
                    if isinstance(src_val, list):
                        for i in indices:
                            dst_val[i] = src_val[i]

        dst_adsl.first_update_done = True