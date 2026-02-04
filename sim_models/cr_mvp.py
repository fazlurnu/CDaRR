''' Conflict resolution based on the Modified Voltage Potential algorithm. '''
import numpy as np
from bluesky import stack
import bluesky as bs
from bluesky.core import Entity

class MVP(Entity):
    """ 
    Conflict Resolution - Modified Voltage Potential

    Taken from BlueSky library, but made "adjustable" as a separate script here
    """
    def __init__(self):
        super().__init__()
        # [-] switch to limit resolution to the horizontal direction
        self.swresohoriz = True
        # [-] switch to use only speed resolutions (works with swresohoriz = True)
        self.swresospd = False
        # [-] switch to use only heading resolutions (works with swresohoriz = True)
        self.swresohdg = False
        # [-] switch to limit resolution to the vertical direction
        self.swresovert = False

        self.resopairs = set()  # Resolved conflicts that are still before CPA

        # Resolution factors:
        # set < 1 to maneuver only a fraction of the resolution
        # set > 1 to add a margin to separation values
        self.resofach = bs.settings.asas_marh
        self.resofacv = bs.settings.asas_marv
        
        # Switches to guarantee last reso zone commands keep valid if cd zone changes
        self.resodhrelative = True  # Size of resolution zone dh, vertically, set relative to CD zone
        self.resorrelative  = True  # Size of resolution zone r, vertically, set relative to CD zone

        with self.settrafarrays():
            self.resooffac = np.array([], dtype=bool)
            self.noresoac = np.array([], dtype=bool)
            # whether the autopilot follows ASAS or not
            self.active = np.array([], dtype=bool)
            self.trk = np.array([])  # heading provided by the ASAS [deg]
            self.tas = np.array([])  # speed provided by the ASAS (eas) [m/s]
            self.alt = np.array([])  # alt provided by the ASAS [m]
            self.vs = np.array([])   # vspeed provided by the ASAS [m/s]

    def resumenav(self, conf, ownship, intruder):
        # print(self.resopairs)
        self.resopairs.update(conf.confpairs)
        # print(self.resopairs)

        delpairs = set()
        changeactive = dict()

        # smallest relative angle between vectors of heading a and b
        def anglediff(a, b):
            d = a - b
            if d > 180:
                return anglediff(a, b + 360)
            elif d < -180:
                return anglediff(a + 360, b)
            else:
                return d

        # Look at all conflicts, also the ones that are solved but CPA is yet to come
        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)
            # If the ownship aircraft is deleted remove its conflict from the list
            if idx1 < 0:
                delpairs.add(conflict)
                continue

            if idx2 >= 0:
                # Distance vector using flat earth approximation
                re = 6371000.
                dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                      np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                              ownship.lat[idx1])),
                                      np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

                # Relative velocity vector
                vrel = np.array([intruder.gseast[idx2] - ownship.gseast[idx1],
                                 intruder.gsnorth[idx2] - ownship.gsnorth[idx1]])

                # Check if conflict is past CPA
                past_cpa = np.dot(dist, vrel) > 0.0

                rpz = np.max(conf.rpz[[idx1, idx2]])
                # hor_los:
                # Aircraft should continue to resolve until there is no horizontal
                # LOS. This is particularly relevant when vertical resolutions
                # are used.
                hdist = np.linalg.norm(dist)
                hor_los = hdist < rpz

                # Bouncing conflicts:
                # If two aircraft are getting in and out of conflict continously,
                # then they it is a bouncing conflict. ASAS should stay active until
                # the bouncing stops.
                is_bouncing = \
                    abs(anglediff(ownship.trk[idx1], intruder.trk[idx2])) < 30.0 and \
                    hdist < rpz * self.resofach

            # Start recovery for ownship if intruder is deleted, or if past CPA
            # and not in horizontal LOS or a bouncing conflict
            if idx2 >= 0 and (not past_cpa or hor_los or is_bouncing):
                # Enable ASAS for this aircraft
                changeactive[idx1] = True
            else:
                # Switch ASAS off for ownship if there are no other conflicts
                # that this aircraft is involved in.
                changeactive[idx1] = changeactive.get(idx1, False)
                # If conflict is solved, remove it from the resopairs list
                delpairs.add(conflict)

        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            self.active[idx] = active
            if not active:
                # Waypoint recovery after conflict: Find the next active waypoint
                # and send the aircraft to that waypoint.
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:  # To avoid problems if there are no waypoints
                    bs.traf.ap.route[idx].direct(
                        idx, bs.traf.ap.route[idx].wpname[iwpid])

        # Remove pairs from the list that are past CPA or have deleted aircraft
        self.resopairs -= delpairs

        return delpairs

    def resumenav_double_criteria(self, conf, ownship, intruder):
        if not hasattr(self, "_intr_init_vel"):
            self._intr_init_vel = {}

        curpairs = set(conf.confpairs)
        newpairs = curpairs - self.resopairs
        self.resopairs.update(curpairs)

        for pair in newpairs:
            idx1, idx2 = bs.traf.id2idx(pair)
            if idx1 >= 0 and idx2 >= 0:
                # Vi,i recorded at conflict initiation
                self._intr_init_vel[pair] = (float(intruder.gseast[idx2]),
                                            float(intruder.gsnorth[idx2]))

        def calculate_dcpa(dx, dy, du, dv):
            dv2 = du * du + dv * dv
            if abs(dv2) < 1e-6:
                dv2 = 1e-6
            tcpa = -(du * dx + dv * dy) / dv2
            dist2 = dx * dx + dy * dy
            dcpa2 = abs(dist2 - tcpa * tcpa * dv2)
            return float(np.sqrt(dcpa2)), float(tcpa)

        pair_dxdy = {}
        if len(conf.confpairs) > 0:
            q = np.radians(conf.qdr)
            dxs = conf.dist * np.sin(q)
            dys = conf.dist * np.cos(q)
            pair_dxdy = dict(zip(conf.confpairs, zip(dxs.tolist(), dys.tolist())))

        vod_cache = {}

        def _val(a, idx):
            try:
                return float(a[idx])
            except Exception:
                return None

        def get_Vo_d(idx):
            if idx in vod_cache:
                return vod_cache[idx]

            trk = None
            if hasattr(ownship, "seltrk"):
                trk = _val(ownship.seltrk, idx)
            if trk is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "trk"):
                trk = _val(ownship.ap.trk, idx)
            if trk is None:
                trk = _val(ownship.trk, idx)

            spd = None
            if hasattr(ownship, "selspd"):
                spd = _val(ownship.selspd, idx)
            if spd is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "tas"):
                spd = _val(ownship.ap.tas, idx)
            if spd is None:
                spd = _val(getattr(ownship, "gs", None), idx)
            if spd is None:
                spd = float(np.hypot(ownship.gseast[idx], ownship.gsnorth[idx]))

            r = np.radians(trk)
            u = spd * np.sin(r)
            v = spd * np.cos(r)
            vod_cache[idx] = (u, v)
            return u, v

        delpairs = set()
        changeactive = {}

        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)

            if idx1 < 0:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                continue

            if idx2 < 0:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                changeactive[idx1] = changeactive.get(idx1, False)
                continue

            if conflict in pair_dxdy:
                dx, dy = pair_dxdy[conflict]
                dx = float(dx)
                dy = float(dy)
            else:
                re = 6371000.0
                dlon = float(intruder.lon[idx2] - ownship.lon[idx1])
                dlat = float(intruder.lat[idx2] - ownship.lat[idx1])
                latm = 0.5 * np.radians(float(intruder.lat[idx2] + ownship.lat[idx1]))
                dx = re * np.radians(dlon) * np.cos(latm)
                dy = re * np.radians(dlat)

            rpz = float(np.max(conf.rpz[[idx1, idx2]]))
            Vo_u, Vo_v = get_Vo_d(idx1)

            Vi_c_u = float(intruder.gseast[idx2])
            Vi_c_v = float(intruder.gsnorth[idx2])

            # Criterion 1: intruder maintains current velocity (Vi,c)
            du1 = Vo_u - Vi_c_u
            dv1 = Vo_v - Vi_c_v
            Dcpa1, _ = calculate_dcpa(dx, dy, du1, dv1)
            crit1 = Dcpa1 > rpz

            # Criterion 2: intruder reverts to initial velocity (Vi,i) logged at start
            Vi_i_u, Vi_i_v = self._intr_init_vel.get(conflict, (Vi_c_u, Vi_c_v))
            du2 = Vo_u - Vi_i_u
            dv2 = Vo_v - Vi_i_v
            Dcpa2, _ = calculate_dcpa(dx, dy, du2, dv2)
            crit2 = Dcpa2 > rpz

            if crit1 and crit2:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                changeactive[idx1] = changeactive.get(idx1, False)
            else:
                changeactive[idx1] = True

        for idx, active in changeactive.items():
            self.active[idx] = active
            if not active:
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:
                    bs.traf.ap.route[idx].direct(idx, bs.traf.ap.route[idx].wpname[iwpid])

        self.resopairs -= delpairs
        return delpairs
    
    def resumenav_triple_criteria(self, conf, ownship, intruder):
        if not hasattr(self, "_intr_init_vel"):
            self._intr_init_vel = {}

        curpairs = set(conf.confpairs)
        newpairs = curpairs - self.resopairs
        self.resopairs.update(curpairs)

        for pair in newpairs:
            idx1, idx2 = bs.traf.id2idx(pair)
            if idx1 >= 0 and idx2 >= 0:
                # Vi,i recorded at conflict initiation
                self._intr_init_vel[pair] = (float(intruder.gseast[idx2]),
                                            float(intruder.gsnorth[idx2]))

        def calculate_dcpa(dx, dy, du, dv):
            dv2 = du * du + dv * dv
            if abs(dv2) < 1e-6:
                dv2 = 1e-6
            tcpa = -(du * dx + dv * dy) / dv2
            dist2 = dx * dx + dy * dy
            dcpa2 = abs(dist2 - tcpa * tcpa * dv2)
            return float(np.sqrt(dcpa2)), float(tcpa)

        pair_dxdy = {}
        if len(conf.confpairs) > 0:
            q = np.radians(conf.qdr)
            dxs = conf.dist * np.sin(q)
            dys = conf.dist * np.cos(q)
            pair_dxdy = dict(zip(conf.confpairs, zip(dxs.tolist(), dys.tolist())))

        vod_cache = {}

        def _val(a, idx):
            try:
                return float(a[idx])
            except Exception:
                return None

        def get_Vo_d(idx):
            if idx in vod_cache:
                return vod_cache[idx]

            trk = None
            if hasattr(ownship, "seltrk"):
                trk = _val(ownship.seltrk, idx)
            if trk is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "trk"):
                trk = _val(ownship.ap.trk, idx)
            if trk is None:
                trk = _val(ownship.trk, idx)

            spd = None
            if hasattr(ownship, "selspd"):
                spd = _val(ownship.selspd, idx)
            if spd is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "tas"):
                spd = _val(ownship.ap.tas, idx)
            if spd is None:
                spd = _val(getattr(ownship, "gs", None), idx)
            if spd is None:
                spd = float(np.hypot(ownship.gseast[idx], ownship.gsnorth[idx]))

            r = np.radians(trk)
            u = spd * np.sin(r)
            v = spd * np.cos(r)
            vod_cache[idx] = (u, v)
            return u, v

        delpairs = set()
        changeactive = {}

        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)

            if idx1 < 0:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                continue

            if idx2 < 0:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                changeactive[idx1] = changeactive.get(idx1, False)
                continue

            if conflict in pair_dxdy:
                dx, dy = pair_dxdy[conflict]
                dx = float(dx)
                dy = float(dy)
            else:
                re = 6371000.0
                dlon = float(intruder.lon[idx2] - ownship.lon[idx1])
                dlat = float(intruder.lat[idx2] - ownship.lat[idx1])
                latm = 0.5 * np.radians(float(intruder.lat[idx2] + ownship.lat[idx1]))
                dx = re * np.radians(dlon) * np.cos(latm)
                dy = re * np.radians(dlat)

            rpz = float(np.max(conf.rpz[[idx1, idx2]]))
            Vo_u, Vo_v = get_Vo_d(idx1)

            Vi_c_u = float(intruder.gseast[idx2])
            Vi_c_v = float(intruder.gsnorth[idx2])

            # Distance vector using flat earth approximation
            re = 6371000.
            dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                    np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                            ownship.lat[idx1])),
                                    np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

            # Relative velocity vector
            vrel = np.array([intruder.gseast[idx2] - ownship.gseast[idx1],
                                intruder.gsnorth[idx2] - ownship.gsnorth[idx1]])

            # Check if conflict is past CPA
            past_cpa = np.dot(dist, vrel) > 0.0

            # Criterion 1: intruder maintains current velocity (Vi,c)
            du1 = Vo_u - Vi_c_u
            dv1 = Vo_v - Vi_c_v
            Dcpa1, _ = calculate_dcpa(dx, dy, du1, dv1)
            crit1 = Dcpa1 > rpz

            # Criterion 2: intruder reverts to initial velocity (Vi,i) logged at start
            Vi_i_u, Vi_i_v = self._intr_init_vel.get(conflict, (Vi_c_u, Vi_c_v))
            du2 = Vo_u - Vi_i_u
            dv2 = Vo_v - Vi_i_v
            Dcpa2, _ = calculate_dcpa(dx, dy, du2, dv2)
            crit2 = Dcpa2 > rpz

            if crit1 and crit2 and past_cpa:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                changeactive[idx1] = changeactive.get(idx1, False)
            else:
                changeactive[idx1] = True

        for idx, active in changeactive.items():
            self.active[idx] = active
            if not active:
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:
                    bs.traf.ap.route[idx].direct(idx, bs.traf.ap.route[idx].wpname[iwpid])

        self.resopairs -= delpairs
        return delpairs
    
    def resumenav_double_criteria_dummy(self, conf, ownship, intruder):
        if not hasattr(self, "_intr_init_vel"):
            self._intr_init_vel = {}

        curpairs = set(conf.confpairs)
        newpairs = curpairs - self.resopairs
        self.resopairs.update(curpairs)

        for pair in newpairs:
            idx1, idx2 = bs.traf.id2idx(pair)
            if idx1 >= 0 and idx2 >= 0:
                # Vi,i recorded at conflict initiation
                self._intr_init_vel[pair] = (float(intruder.gseast[idx2]),
                                            float(intruder.gsnorth[idx2]))

        def calculate_dcpa(dx, dy, du, dv):
            dv2 = du * du + dv * dv
            if abs(dv2) < 1e-6:
                dv2 = 1e-6
            tcpa = -(du * dx + dv * dy) / dv2
            dist2 = dx * dx + dy * dy
            dcpa2 = abs(dist2 - tcpa * tcpa * dv2)
            return float(np.sqrt(dcpa2)), float(tcpa)

        pair_dxdy = {}
        if len(conf.confpairs) > 0:
            q = np.radians(conf.qdr)
            dxs = conf.dist * np.sin(q)
            dys = conf.dist * np.cos(q)
            pair_dxdy = dict(zip(conf.confpairs, zip(dxs.tolist(), dys.tolist())))

        vod_cache = {}

        def _val(a, idx):
            try:
                return float(a[idx])
            except Exception:
                return None

        def get_Vo_d(idx):
            if idx in vod_cache:
                return vod_cache[idx]

            trk = None
            if hasattr(ownship, "seltrk"):
                trk = _val(ownship.seltrk, idx)
            if trk is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "trk"):
                trk = _val(ownship.ap.trk, idx)
            if trk is None:
                trk = _val(ownship.trk, idx)

            spd = None
            if hasattr(ownship, "selspd"):
                spd = _val(ownship.selspd, idx)
            if spd is None and hasattr(ownship, "ap") and hasattr(ownship.ap, "tas"):
                spd = _val(ownship.ap.tas, idx)
            if spd is None:
                spd = _val(getattr(ownship, "gs", None), idx)
            if spd is None:
                spd = float(np.hypot(ownship.gseast[idx], ownship.gsnorth[idx]))

            r = np.radians(trk)
            u = spd * np.sin(r)
            v = spd * np.cos(r)
            vod_cache[idx] = (u, v)
            return u, v

        delpairs = set()
        changeactive = {}

        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)

            if idx1 < 0:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                continue

            if idx2 < 0:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                changeactive[idx1] = changeactive.get(idx1, False)
                continue

            if conflict in pair_dxdy:
                dx, dy = pair_dxdy[conflict]
                dx = float(dx)
                dy = float(dy)
            else:
                re = 6371000.0
                dlon = float(intruder.lon[idx2] - ownship.lon[idx1])
                dlat = float(intruder.lat[idx2] - ownship.lat[idx1])
                latm = 0.5 * np.radians(float(intruder.lat[idx2] + ownship.lat[idx1]))
                dx = re * np.radians(dlon) * np.cos(latm)
                dy = re * np.radians(dlat)

            rpz = float(np.max(conf.rpz[[idx1, idx2]]))
            Vo_u, Vo_v = get_Vo_d(idx1)

            Vi_c_u = float(intruder.gseast[idx2])
            Vi_c_v = float(intruder.gsnorth[idx2])

            # Criterion 1: intruder maintains current velocity (Vi,c)
            du1 = Vo_u - Vi_c_u
            dv1 = Vo_v - Vi_c_v
            Dcpa1, _ = calculate_dcpa(dx, dy, du1, dv1)
            crit1 = Dcpa1 > rpz

            # Criterion 2: intruder reverts to initial velocity (Vi,i) logged at start
            Vi_i_u, Vi_i_v = self._intr_init_vel.get(conflict, (Vi_c_u, Vi_c_v))
            du2 = Vo_u - Vi_i_u
            dv2 = Vo_v - Vi_i_v
            Dcpa2, _ = calculate_dcpa(dx, dy, du2, dv2)
            crit2 = Dcpa2 > rpz

            if crit1 and crit2:
                delpairs.add(conflict)
                self._intr_init_vel.pop(conflict, None)
                changeactive[idx1] = changeactive.get(idx1, False)
            else:
                changeactive[idx1] = True

        for idx, active in changeactive.items():
            self.active[idx] = active
            if not active:
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:
                    bs.traf.ap.route[idx].direct(idx, bs.traf.ap.route[idx].wpname[iwpid])

        return delpairs
    
    def resolve(self, conf, ownship, intruder, resofach):
        # here always update the resolution factor for horizontal
        # might be handy for future implementations when resofach
        # can change durting simulation
        
        self.resofach = resofach
        ''' Resolve all current conflicts '''
        # Initialize an array to store the resolution velocity vector for all A/C
        dv = np.zeros((ownship.ntraf, 3))

        # Initialize an array to store time needed to resolve vertically
        timesolveV = np.ones(ownship.ntraf) * 1e9

        # Call MVP function to resolve conflicts-----------------------------------
        for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)

            # If A/C indexes are found, then apply MVP on this conflict pair
            # Because ADSB is ON, this is done for each aircraft separately
            if idx1 > -1 and idx2 > -1:
                dv_mvp, tsolV = self.MVP(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2)
                if tsolV < timesolveV[idx1]:
                    timesolveV[idx1] = tsolV

                # Cooperative behavior: halve vertical component, then apply to ownship
                dv_mvp[2] = 0.5 * dv_mvp[2]
                dv[idx1] = dv[idx1] - dv_mvp

                # Check the noreso aircraft. Nobody avoids noreso aircraft.
                # But noreso aircraft will avoid other aircraft
                if self.noresoac[idx2]:
                    dv[idx1] = dv[idx1] + dv_mvp

                # Check the resooff aircraft. These aircraft will not do resolutions.
                if self.resooffac[idx1]:
                    dv[idx1] = 0.0

        # Determine new speed and limit resolution direction for all aircraft-------

        # Resolution vector for all aircraft, cartesian coordinates
        dv = np.transpose(dv)

        # The old speed vector, cartesian coordinates
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])

        # The new speed vector, cartesian coordinates
        newv = v + dv

        # Limit resolution direction if required-----------------------------------

        # Compute new speed vector in polar coordinates based on desired resolution
        if self.swresohoriz:  # horizontal resolutions
            if self.swresospd and not self.swresohdg:  # SPD only
                newtrack = ownship.trk
                newgs    = np.sqrt(newv[0, :]**2 + newv[1, :]**2)
                newvs    = ownship.vs
            elif self.swresohdg and not self.swresospd:  # HDG only
                newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
                newgs    = ownship.gs
                newvs    = ownship.vs
            else:  # SPD + HDG
                newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
                newgs    = np.sqrt(newv[0, :]**2 + newv[1, :]**2)
                newvs    = ownship.vs
        elif self.swresovert:  # vertical resolutions
            newtrack = ownship.trk
            newgs    = ownship.gs
            newvs    = newv[2, :]
        else:  # horizontal + vertical
            newtrack = (np.arctan2(newv[0, :], newv[1, :]) * 180 / np.pi) % 360
            newgs    = np.sqrt(newv[0, :]**2 + newv[1, :]**2)
            newvs    = newv[2, :]

        # Determine ASAS module commands for all aircraft--------------------------

        # Cap the velocity
        newgscapped = np.maximum(ownship.perf.vmin, np.minimum(ownship.perf.vmax, newgs))

        # Cap the vertical speed
        vscapped = np.maximum(ownship.perf.vsmin, np.minimum(ownship.perf.vsmax, newvs))

        # Calculate if Autopilot selected altitude should be followed.
        asasalttemp = vscapped * timesolveV + ownship.alt
        signdvs = np.sign(vscapped - ownship.ap.vs * np.sign(ownship.selalt - ownship.alt))
        signalt = np.sign(asasalttemp - ownship.selalt)
        alt = np.where(np.logical_or(signdvs == 0, signdvs == signalt), asasalttemp, ownship.selalt)

        # Only update asas alt when actually in conflict
        altCondition = np.logical_and(timesolveV < conf.dtlookahead, np.abs(dv[2, :]) > 0.0)
        alt[altCondition] = asasalttemp[altCondition]

        # If resolutions are limited in horizontal direction, keep alt at selected alt
        alt = alt * (1 - self.swresohoriz) + ownship.selalt * self.swresohoriz

        return newtrack, newgscapped, vscapped, alt, self.resopairs

    def MVP(self, ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2):
        """Modified Voltage Potential (MVP) resolution method"""
        # Preliminary calculations-------------------------------------------------
        rpz_m = np.max(conf.rpz[[idx1, idx2]] * self.resofach)
        hpz_m = np.max(conf.hpz[[idx1, idx2]] * self.resofacv)
        dtlook = conf.dtlookahead[idx1]
        qdr = np.radians(qdr)

        # Relative position vector between id1 and id2
        drel = np.array([np.sin(qdr) * dist,
                         np.cos(qdr) * dist,
                         intruder.alt[idx2] - ownship.alt[idx1]])

        # Relative velocity vector
        v1 = np.array([ownship.gseast[idx1], ownship.gsnorth[idx1], ownship.vs[idx1]])
        v2 = np.array([intruder.gseast[idx2], intruder.gsnorth[idx2], intruder.vs[idx2]])
        vrel = v2 - v1

        # Horizontal resolution----------------------------------------------------

        dcpa  = drel + vrel * tcpa
        dabsH = np.sqrt(dcpa[0] * dcpa[0] + dcpa[1] * dcpa[1])

        iH = rpz_m - dabsH

        threshold = 0.001
        if dabsH <= threshold:
            dabsH = threshold
            dcpa[0] = drel[1] / dist * dabsH
            dcpa[1] = -drel[0] / dist * dabsH

        if rpz_m < dist and dabsH < dist:
            erratum = np.cos(np.arcsin(rpz_m / dist) - np.arcsin(dabsH / dist))
            dv1 = ((rpz_m / erratum - dabsH) * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = ((rpz_m / erratum - dabsH) * dcpa[1]) / (abs(tcpa) * dabsH)
        else:
            dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)

        # Vertical resolution------------------------------------------------------

        iV = hpz_m if abs(vrel[2]) > 0.0 else hpz_m - abs(drel[2])
        tsolV = abs(drel[2] / vrel[2]) if abs(vrel[2]) > 0.0 else tLOS

        if tsolV > dtlook:
            tsolV = tLOS
            iV = hpz_m

        dv3 = np.where(abs(vrel[2]) > 0.0,
                       (iV / tsolV) * (-vrel[2] / abs(vrel[2])),
                       (iV / tsolV))

        dv = np.array([dv1, dv2, dv3])
        return dv, tsolV
