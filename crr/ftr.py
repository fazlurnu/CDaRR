'''Double-criteria FTR conflict recovery -- functional core.

Fresh copy of the Phase-1 extraction in sim_models/crr_resumenav_ftr.py.
'''
from dataclasses import replace

import numpy as np

from .common import (
    RecoveryState, default_id2idx,
    get_desired_ownship_velocity, compute_pair_positions, get_pair_dxdy,
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


def ftr_release_decision(dx, dy, rpz, Vo_u, Vo_v, Vi_c_u, Vi_c_v, Vi_i_u, Vi_i_v):
    ''' Double-criteria FTR release decision for one conflict pair.

    Pure. True iff BOTH: (1) the projected CPA stays outside rpz assuming the
    intruder holds its current velocity (Vi,c), and (2) the same assuming the
    intruder reverts to its velocity at conflict initiation (Vi,i).
    '''
    # Criterion 1: intruder maintains current velocity (Vi,c)
    du1 = Vo_u - Vi_c_u
    dv1 = Vo_v - Vi_c_v
    Dcpa1, _ = calculate_dcpa(dx, dy, du1, dv1)
    crit1 = Dcpa1 > rpz

    # Criterion 2: intruder reverts to initial velocity (Vi,i) logged at start
    du2 = Vo_u - Vi_i_u
    dv2 = Vo_v - Vi_i_v
    Dcpa2, _ = calculate_dcpa(dx, dy, du2, dv2)
    crit2 = Dcpa2 > rpz

    return crit1 and crit2


def step(conf, ownship, intruder, state: RecoveryState, id2idx=default_id2idx):
    ''' Pure double-criteria FTR recovery decision.

    Returns (new_state, changeactive, delpairs) -- same contract as crr.cpa.step.
    '''
    state = record_initial_intruder_velocity(state, conf, intruder, id2idx=id2idx)

    pair_dxdy = compute_pair_positions(conf)
    vod_cache = {}

    delpairs = set()
    changeactive = {}
    init_vel = dict(state.init_vel)

    for conflict in state.resopairs:
        idx1, idx2 = id2idx(conflict)

        if idx1 < 0:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            continue

        if idx2 < 0:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
            continue

        dx, dy = get_pair_dxdy(conflict, pair_dxdy, ownship, intruder, idx1, idx2)
        rpz = float(np.max(conf.rpz[[idx1, idx2]]))
        Vo_u, Vo_v = get_desired_ownship_velocity(ownship, idx1, vod_cache)

        Vi_c_u = float(intruder.gseast[idx2])
        Vi_c_v = float(intruder.gsnorth[idx2])
        Vi_i_u, Vi_i_v = init_vel.get(conflict, (Vi_c_u, Vi_c_v))

        release = ftr_release_decision(dx, dy, rpz, Vo_u, Vo_v, Vi_c_u, Vi_c_v, Vi_i_u, Vi_i_v)

        if release:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
        else:
            changeactive[idx1] = True

    new_state = replace(state, resopairs=state.resopairs - delpairs, init_vel=init_vel)
    return new_state, changeactive, delpairs
