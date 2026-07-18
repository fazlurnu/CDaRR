''' Conflict resolution based on the Modified Voltage Potential (MVP) algorithm --
functional core.

Fresh copy of the Phase-1 extraction in sim_models/cr_mvp.py (see
refactor_fp.md, Phase 2 -- not an import, so old-vs-new equivalence is a real
check, not a tautology). Unlike CDaRR_FP's cr/mvp.py, this preserves the full
algorithm this repo actually uses: vertical resolution, noreso/resooff
handling, and all four swresohoriz/spd/hdg/vert branches. CDaRR_FP's version
is a simplified horizontal-only rewrite and is NOT a valid reference here.
'''
import numpy as np

from .common import ResolutionCommand, select_command, cap_velocities, resolve_altitude


def mvp_pair(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2, resofach, resofacv):
    """Modified Voltage Potential (MVP) resolution for a single conflict pair.

    Pure: reads only its arguments. Returns (dv, tsolV) -- the 3D resolution
    velocity [dv_east, dv_north, dv_vert] for the ownship, and the vertical
    time-to-solve.
    """
    # Preliminary calculations-------------------------------------------------
    rpz_m = np.max(conf.rpz[[idx1, idx2]] * resofach)
    hpz_m = np.max(conf.hpz[[idx1, idx2]] * resofacv)
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


def resolve(conf, ownship, intruder, params, noresoac=None, resooffac=None) -> ResolutionCommand:
    ''' Resolve all current conflicts with MVP.

    Pure: `noresoac`/`resooffac` (per-aircraft bool arrays) default to
    all-False, matching this pipeline's actual usage -- neither switch is ever
    set by any config (see refactor_fp.md's risk table). No self, no
    resopairs -- recovery-state threading is crr's job (F1 composition order
    lives in cdarr.core), not resolution's.
    '''
    ntraf = ownship.ntraf
    if noresoac is None:
        noresoac = np.zeros(ntraf, dtype=bool)
    if resooffac is None:
        resooffac = np.zeros(ntraf, dtype=bool)

    # Initialize an array to store the resolution velocity vector for all A/C
    dv = np.zeros((ntraf, 3))

    # Initialize an array to store time needed to resolve vertically
    timesolveV = np.ones(ntraf) * 1e9

    # Call MVP function to resolve conflicts-----------------------------------
    for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
        idx1 = ownship.id.index(ac1)
        idx2 = intruder.id.index(ac2)

        # If A/C indexes are found, then apply MVP on this conflict pair
        # Because ADSB is ON, this is done for each aircraft separately
        if idx1 > -1 and idx2 > -1:
            dv_mvp, tsolV = mvp_pair(ownship, intruder, conf, qdr, dist, tcpa, tLOS, idx1, idx2,
                                      params.resofach, params.resofacv)
            if tsolV < timesolveV[idx1]:
                timesolveV[idx1] = tsolV

            # Cooperative behavior: halve vertical component, then apply to ownship
            dv_mvp[2] = 0.5 * dv_mvp[2]
            dv[idx1] = dv[idx1] - dv_mvp

            # Check the noreso aircraft. Nobody avoids noreso aircraft.
            # But noreso aircraft will avoid other aircraft
            if noresoac[idx2]:
                dv[idx1] = dv[idx1] + dv_mvp

            # Check the resooff aircraft. These aircraft will not do resolutions.
            if resooffac[idx1]:
                dv[idx1] = 0.0

    # Determine new speed and limit resolution direction for all aircraft-------

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

    return ResolutionCommand(trk=newtrack, gs_capped=newgscapped, vs_capped=vscapped, alt=alt)
