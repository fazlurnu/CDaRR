import numpy as np
import math
import bluesky as bs

def resumenav_triple_criteria(reso, conf, ownship, intruder):
    if not hasattr(reso, "_intr_init_vel"):
        reso._intr_init_vel = {}

    curpairs = set(conf.confpairs)
    newpairs = curpairs - reso.resopairs
    reso.resopairs.update(curpairs)

    for pair in newpairs:
        idx1, idx2 = bs.traf.id2idx(pair)
        if idx1 >= 0 and idx2 >= 0:
            # Vi,i recorded at conflict initiation
            reso._intr_init_vel[pair] = (float(intruder.gseast[idx2]),
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

    for conflict in reso.resopairs:
        idx1, idx2 = bs.traf.id2idx(conflict)

        if idx1 < 0:
            delpairs.add(conflict)
            reso._intr_init_vel.pop(conflict, None)
            continue

        if idx2 < 0:
            delpairs.add(conflict)
            reso._intr_init_vel.pop(conflict, None)
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
        Vi_i_u, Vi_i_v = reso._intr_init_vel.get(conflict, (Vi_c_u, Vi_c_v))
        du2 = Vo_u - Vi_i_u
        dv2 = Vo_v - Vi_i_v
        Dcpa2, _ = calculate_dcpa(dx, dy, du2, dv2)
        crit2 = Dcpa2 > rpz

        # Criterion 3: conflict is past CPA
        # Distance vector using flat earth approximation
        re = 6371000.
        dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                        ownship.lat[idx1])),
                                np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

        # Relative velocity vector
        vrel = np.array([ownship.gseast[idx1] - intruder.gseast[idx2],
                         ownship.gsnorth[idx1] - intruder.gsnorth[idx2]])

        # Check if conflict is past CPA
        crit3 = np.dot(dist, vrel) < 0.0

        if crit1 and crit2 and crit3:
            delpairs.add(conflict)
            reso._intr_init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
        else:
            changeactive[idx1] = True

    for idx, active in changeactive.items():
        reso.active[idx] = active
        if not active:
            iwpid = bs.traf.ap.route[idx].findact(idx)
            if iwpid != -1:
                bs.traf.ap.route[idx].direct(idx, bs.traf.ap.route[idx].wpname[iwpid])

    reso.resopairs -= delpairs
    return delpairs