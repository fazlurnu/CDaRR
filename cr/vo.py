''' Conflict resolution based on the Velocity Obstacle (VO) algorithm --
functional core.

Fresh copy of the Phase-1 extraction in sim_models/cr_vo.py (see
cd/statebased.py's docstring for why this is a copy, not an import).
'''
from typing import Tuple

import numpy as np
from math import atan2, asin, sqrt, sin, cos
from shapely.geometry import Point, LineString
from shapely.affinity import translate
from shapely.ops import nearest_points

from .common import ResolutionCommand, ResolutionParams, select_command, cap_velocities, resolve_altitude
from crr.common import RecoveryState, default_id2idx
from crr.cpa import step as cpa_step


def get_cc_tp(ownship_position, intruder_position, rpz):
    """Tangent points of the collision-cone circle (radius rpz, centred on
    intruder_position) as seen from ownship_position. Pure."""
    dx = intruder_position.x - ownship_position.x
    dy = intruder_position.y - ownship_position.y

    d = sqrt(dx**2 + dy**2)

    if(d > rpz):
        theta = atan2(dy, dx)
        beta = asin(rpz/d)
        side = sqrt(d**2 - rpz**2)

        tp_1_x = ownship_position.x + side * cos(theta - beta)
        tp_1_y = ownship_position.y + side * sin(theta - beta)
        tp_2_x = ownship_position.x + side * cos(theta + beta)
        tp_2_y = ownship_position.y + side * sin(theta + beta)

        return Point(tp_1_x, tp_1_y), Point(tp_2_x, tp_2_y)

    else:
        return None, None


def vo_pair(ownship, intruder, conf, qdr, dist, tLOS, idx1, idx2, resofach, resofacv):
    """Velocity-Obstacle (VO) resolution for a single conflict pair.

    Pure: reads only its arguments. Returns (dv, tsolV) -- the 3D resolution
    velocity [dv_east, dv_north, dv_vert] for the ownship, and the vertical
    time-to-solve.
    """
    rpz = np.max(conf.rpz[[idx1, idx2]] * resofach)
    hpz = np.max(conf.hpz[[idx1, idx2]] * resofacv)
    dtlook = conf.dtlookahead[idx1]

    # Convert qdr from degrees to radians
    qdr = np.radians(qdr)

    # Relative position vector between id1 and id2
    drel = np.array([np.sin(qdr) * dist, \
                    np.cos(qdr) * dist, \
                    intruder.alt[idx2] - ownship.alt[idx1]])

    ownship_position = Point(0, 0)

    intruder_position = Point(drel[1], drel[0])


    tp_1, tp_2 = get_cc_tp(ownship_position, intruder_position, rpz)

    ownship_velocity = Point(ownship.gsnorth[idx1], ownship.gseast[idx1])
    intruder_velocity = Point(intruder.gsnorth[idx2], intruder.gseast[idx2])

    method = 0

    if (tp_1 is not None) and (tp_2 is not None):
        vo_0 = translate(ownship_position, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
        vo_1 = translate(tp_1, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
        vo_2 = translate(tp_2, xoff = intruder_velocity.x, yoff = intruder_velocity.y)

        vo_line_1 = LineString([vo_0, vo_1])
        vo_line_2 = LineString([vo_0, vo_2])

        # method = 0: opt, 1: spd change, 2: hdg change
        if(method == 0):
            cp_1 = nearest_points(vo_line_1, ownship_velocity)[0]
            cp_2 = nearest_points(vo_line_2, ownship_velocity)[0]

            cp_1_dist = cp_1.distance(ownship_velocity)
            cp_2_dist = cp_2.distance(ownship_velocity)

            if(cp_1_dist <= cp_2_dist):
                cp = cp_1
            else:
                cp = cp_2

        dv1 = ownship_velocity.y - cp.y
        dv2 = ownship_velocity.x - cp.x

    else:
        dv1 = 0
        dv2 = 0

    # Vertical resolution
    vrel_vs = ownship.vs[idx1] - intruder.vs[idx2]

    iV = hpz if abs(vrel_vs) > 0.0 else hpz - abs(drel[2])
    tsolV = abs(drel[2] / vrel_vs) if abs(vrel_vs) > 0.0 else tLOS

    if tsolV > dtlook:
        tsolV = tLOS
        iV = hpz

    dv3 = (iV / tsolV) * (-vrel_vs / abs(vrel_vs)) if abs(vrel_vs) > 0.0 else (iV / tsolV)

    dv = np.array([dv1, dv2, dv3])

    return dv, tsolV


def resolve(conf, ownship, intruder, params: ResolutionParams, state: RecoveryState,
            noresoac=None, resooffac=None, id2idx=default_id2idx
            ) -> Tuple[ResolutionCommand, RecoveryState, dict]:
    ''' Resolve all current conflicts with VO.

    VO quirk (F6, refactor_fp.md): the legacy VO.resolve() calls its own
    past-CPA-style release step (self.resumenav) internally at the end of
    every resolve, IN ADDITION to whatever recovery model the simulation
    loop configures and calls afterward -- so under VO resolution, the pair
    set is processed by past-CPA release logic twice per tick. Preserved
    as-is here via an internal crr.cpa.step call using the SAME resofach as
    resolution (the legacy code shares one self.resofach attribute between
    resolve() and resumenav()); not "fixed".

    Priority rules (params.swprio) are NOT ported -- no config in this
    pipeline ever sets swprio True, so the branch is unexercised; raises
    NotImplementedError rather than silently producing the wrong answer.

    Returns (command, new_state, changeactive) -- changeactive is the F6
    step's decision, for the caller to apply via crr.common.apply_recovery
    if it wants the active-flag/waypoint-recovery side effects (a no-op in
    this pipeline, see refactor_fp.md's F4 note, but exposed for parity).
    '''
    if params.swprio:
        raise NotImplementedError(
            "VO priority rules (swprio/applyprio) are not ported -- no config "
            "in this pipeline sets swprio True. See refactor_fp.md Phase 1's "
            "cr_vo.py commit for the same scoping decision.")

    ntraf = ownship.ntraf
    if noresoac is None:
        noresoac = np.zeros(ntraf, dtype=bool)
    if resooffac is None:
        resooffac = np.zeros(ntraf, dtype=bool)

    # Initialize an array to store the resolution velocity vector for all A/C
    dv = np.zeros((ntraf, 3))

    # Initialize an array to store time needed to resolve vertically
    timesolveV = np.ones(ntraf) * 1e9

    # Call vo function to resolve conflicts-----------------------------------
    for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
        idx1 = ownship.id.index(ac1)
        idx2 = intruder.id.index(ac2)

        # If A/C indexes are found, then apply vo on this conflict pair
        # Because ADSB is ON, this is done for each aircraft separately
        if idx1 >-1 and idx2 > -1:
            dv_vo, tsolV = vo_pair(ownship, intruder, conf, qdr, dist, tLOS, idx1, idx2,
                                    params.resofach, params.resofacv)

            if tsolV < timesolveV[idx1]:
                timesolveV[idx1] = tsolV

            # since cooperative, the vertical resolution component can be halved, and then dv_vo can be added
            dv_vo[2] = 0.5 * dv_vo[2]
            dv[idx1] = dv[idx1] - dv_vo

            # Check the noreso aircraft. Nobody avoids noreso aircraft.
            # But noreso aircraft will avoid other aircraft
            if noresoac[idx2]:
                dv[idx1] = dv[idx1] + dv_vo

            # Check the resooff aircraft. These aircraft will not do resolutions.
            if resooffac[idx1]:
                dv[idx1] = 0.0

    # Determine new speed and limit resolution direction for all aicraft-------

    # Resolution vector for all aircraft, cartesian coordinates
    dv = np.transpose(dv)

    # The old speed vector, cartesian coordinates
    v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])

    # The new speed vector, cartesian coordinates
    newv = v + dv

    # Limit resolution direction if required-----------------------------------
    newtrack, newgs, newvs = select_command(
        newv, ownship, params.swresohoriz, params.swresospd, params.swresohdg, params.swresovert)

    # Determine ASAS module commands for all aircraft--------------------------
    newgscapped, vscapped = cap_velocities(newgs, newvs, ownship.perf)

    alt = resolve_altitude(ownship, vscapped, timesolveV, conf.dtlookahead, dv[2, :], params.swresohoriz)

    command = ResolutionCommand(trk=newtrack, gs_capped=newgscapped, vs_capped=vscapped, alt=alt)

    # F6: VO's built-in past-CPA release step, on top of whatever recovery
    # model the caller runs afterward.
    new_state, changeactive, _delpairs = cpa_step(
        conf, ownship, intruder, state, params.resofach, id2idx=id2idx)

    return command, new_state, changeactive
