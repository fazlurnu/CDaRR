'''Probabilistic double-criteria FTR conflict recovery -- functional core.

Fresh copy of the Phase-1 extraction in
sim_models/crr_resumenav_probabilistic_ftr.py.

Deliberate change from the legacy code: the original smuggled ``sigma_r``,
``sigma_v``, ``dcpa_prob_threshold``, and ``dcpa_prob_Ktheta`` onto the
``conf`` object via ad-hoc attribute assignment
(``conf_detection.sigma_r = sigma_r_worldview``, etc, in
``sim/pairwise_stochastic/get_ipr_stochastic_env.py``) and then searched
several candidate attribute names via ``hasattr``/``getattr`` -- impossible
with a frozen :class:`cd.common.ConflictData`, and exactly the "no hidden
parameter channels" principle refactor_fp.md calls for (section 1, point 5).
These are now ordinary keyword parameters. The only two attribute names the
legacy multi-name search ever actually resolved in this codebase's real call
sites are ``sigma_r``/``sigma_v`` (grep-verified); the other legacy candidate
names (``Sigma_pos``, ``sigma_position``, ...) were never exercised, so
dropping them changes no observable behavior for this pipeline.
'''
from dataclasses import replace

import numpy as np

from .common import (
    RecoveryState, default_id2idx,
    get_desired_ownship_velocity, compute_pair_positions, get_pair_dxdy,
    record_initial_intruder_velocity,
)
from .prob_math import _to_cov, _regularize_spd, analytical_dcpa_prob_gt


def probabilistic_ftr_release_decision(mu_r, Sigma_r, Sigma_v, rpz,
                                        Vo_u, Vo_v, Vi_c_u, Vi_c_v, Vi_i_u, Vi_i_v,
                                        prob_threshold, Ktheta):
    ''' Probabilistic double-criteria release decision for one conflict pair.

    Pure. True iff BOTH: (1) P(DCPA > rpz | intruder keeps current velocity) >
    prob_threshold, and (2) the same assuming the intruder reverts to its
    velocity at conflict initiation.
    '''
    # Criterion 1: intruder maintains current velocity (Vi,c)
    mu_v1 = np.array([Vo_u - Vi_c_u, Vo_v - Vi_c_v], dtype=float)
    p1 = analytical_dcpa_prob_gt(rpz, mu_r, Sigma_r, mu_v1, Sigma_v, Ktheta=Ktheta)
    crit1 = (p1 > prob_threshold)

    # Criterion 2: intruder reverts to initial velocity (Vi,i)
    mu_v2 = np.array([Vo_u - float(Vi_i_u), Vo_v - float(Vi_i_v)], dtype=float)
    p2 = analytical_dcpa_prob_gt(rpz, mu_r, Sigma_r, mu_v2, Sigma_v, Ktheta=Ktheta)
    crit2 = (p2 > prob_threshold)

    return crit1 and crit2


def step(conf, ownship, intruder, state: RecoveryState,
         sigma_r=None, sigma_v=None, prob_threshold=0.9, Ktheta=256,
         id2idx=default_id2idx):
    ''' Pure probabilistic double-criteria FTR recovery decision.

    ``sigma_r``/``sigma_v`` accept the same scalar/vector/(2,2)-matrix forms
    as the legacy code's ``_to_cov`` (``None`` -> zero covariance). Returns
    (new_state, changeactive, delpairs) -- same contract as crr.cpa.step /
    crr.ftr.step.
    '''
    state = record_initial_intruder_velocity(state, conf, intruder, id2idx=id2idx)

    pair_dxdy = compute_pair_positions(conf)
    vod_cache = {}

    Sigma_r = _regularize_spd(_to_cov(sigma_r), eps=1e-6)
    Sigma_v = _regularize_spd(_to_cov(sigma_v), eps=1e-6)

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

        mu_r = np.array([dx, dy], dtype=float)

        release = probabilistic_ftr_release_decision(
            mu_r, Sigma_r, Sigma_v, rpz, Vo_u, Vo_v, Vi_c_u, Vi_c_v, Vi_i_u, Vi_i_v,
            prob_threshold, Ktheta)

        if release:
            delpairs.add(conflict)
            init_vel.pop(conflict, None)
            changeactive[idx1] = changeactive.get(idx1, False)
        else:
            changeactive[idx1] = True

    new_state = replace(state, resopairs=state.resopairs - delpairs, init_vel=init_vel)
    return new_state, changeactive, delpairs
