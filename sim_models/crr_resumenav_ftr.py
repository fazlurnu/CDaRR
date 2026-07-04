import numpy as np
import bluesky as bs

from sim_models.crr_recovery_base import (
    get_desired_ownship_velocity,
    compute_pair_positions,
    get_pair_dxdy,
    apply_recovery,
    record_initial_intruder_velocity,
)


def calculate_dcpa(dx, dy, du, dv):
    dv2 = du * du + dv * dv
    if abs(dv2) < 1e-6:
        dv2 = 1e-6
    tcpa = -(du * dx + dv * dy) / dv2
    dist2 = dx * dx + dy * dy
    dcpa2 = abs(dist2 - tcpa * tcpa * dv2)
    return float(np.sqrt(dcpa2)), float(tcpa)


def resumenav_double_criteria(reso, conf, ownship, intruder):
    record_initial_intruder_velocity(reso, conf, intruder)

    pair_dxdy = compute_pair_positions(conf)
    vod_cache = {}

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

        dx, dy = get_pair_dxdy(conflict, pair_dxdy, ownship, intruder, idx1, idx2)
        rpz = float(np.max(conf.rpz[[idx1, idx2]]))
        Vo_u, Vo_v = get_desired_ownship_velocity(ownship, idx1, vod_cache)

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

        if crit1 and crit2:
            delpairs.add(conflict)
            reso._intr_init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
        else:
            changeactive[idx1] = True

    apply_recovery(changeactive, reso, delpairs)
    return delpairs
