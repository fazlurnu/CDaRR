''' State-based conflict detection. '''
import numpy as np
from bluesky.tools import geo
from bluesky.tools.aero import nm

# Large finite value masking the diagonal (ownship-vs-itself) so an aircraft never
# detects a conflict with its own track. Distinct from the 1e8 sentinels below.
_BIG = 1e9
_EPS = 1e-6


def _velocity_components(trk, gs, n):
    ''' Decompose track (deg) / ground-speed into (east, north) m/s row vectors, shape (1, n). '''
    trkrad = np.radians(trk)
    u = gs * np.sin(trkrad).reshape((1, n))
    v = gs * np.cos(trkrad).reshape((1, n))
    return u, v


def relative_bearing_distance(ownship, intruder, eye):
    ''' Bearing (deg) and distance (m) from every ownship to every intruder.

    The diagonal is pushed to _BIG via `eye` so self-pairs are never flagged.
    '''
    qdr, dist = geo.kwikqdrdist_matrix(np.asmatrix(ownship.lat), np.asmatrix(ownship.lon),
                                np.asmatrix(intruder.lat), np.asmatrix(intruder.lon))

    # Convert back to array to allow element-wise array multiplications later on
    # Convert to meters and add large value to own/own pairs
    qdr = np.asarray(qdr)
    dist = np.asarray(dist) * nm + _BIG * eye
    return qdr, dist


def horizontal_conflict(ownship, intruder, qdr, dist, rpz, eye):
    ''' Horizontal CPA geometry and entry/exit times of the protected zone.

    Returns (swhorconf, tcpa, dcpa2, dtinhor, tinhor, touthor, vrel, rpz_mat).
    `dtinhor` is exposed separately (not folded into `tinhor`) because callers need
    the *unmasked* half-in-zone time even outside a formal conflict (see `tinhor_all`).
    '''
    # Calculate horizontal closest point of approach (CPA)
    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)  # is pos j rel to i
    dy = dist * np.cos(qdrrad)  # is pos j rel to i

    # Note: ownship/intruder track+speed are reshaped by their OWN ntraf, not a
    # shared one -- ownship.ntraf and intruder.ntraf may differ transiently.
    ownu, ownv = _velocity_components(ownship.trk, ownship.gs, ownship.ntraf)
    intu, intv = _velocity_components(intruder.trk, intruder.gs, intruder.ntraf)

    du = ownu - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
    dv = ownv - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < _EPS, _EPS, dv2)  # limit lower absolute value
    vrel = np.sqrt(dv2)

    tcpa = -(du * dx + dv * dy) / dv2 + _BIG * eye

    # Calculate distance^2 at CPA (minimum distance^2)
    dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

    # Check for horizontal conflict
    # RPZ can differ per aircraft, get the largest value per aircraft pair
    rpz_mat = np.asarray(np.maximum(np.asmatrix(rpz), np.asmatrix(rpz).transpose()))
    R2 = rpz_mat * rpz_mat
    swhorconf = dcpa2 < R2  # conflict or not

    # Calculate times of entering and leaving horizontal conflict
    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
    dtinhor = dxinhor / vrel

    tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf
    touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)  # set very large if no conf

    return swhorconf, tcpa, dcpa2, dtinhor, tinhor, touthor, vrel, rpz_mat


def vertical_conflict(ownship, intruder, hpz, eye):
    ''' Vertical separation and entry/exit times of the protected zone.

    Returns (dalt, tinver, toutver, hpz_mat).
    '''
    # Vertical crossing of disk (-dh,+dh)
    dalt = ownship.alt.reshape((1, ownship.ntraf)) - \
        intruder.alt.reshape((1, intruder.ntraf)).T  + _BIG * eye

    dvs = ownship.vs.reshape(1, ownship.ntraf) - \
        intruder.vs.reshape(1, intruder.ntraf).T
    dvs = np.where(np.abs(dvs) < _EPS, _EPS, dvs)  # prevent division by zero

    # Check for passing through each others zone
    # hPZ can differ per aircraft, get the largest value per aircraft pair
    hpz_mat = np.asarray(np.maximum(np.asmatrix(hpz), np.asmatrix(hpz).transpose()))
    tcrosshi = (dalt + hpz_mat) / -dvs
    tcrosslo = (dalt - hpz_mat) / -dvs
    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)

    return dalt, tinver, toutver, hpz_mat


class StateBased():
    """ 
    Conflict Detection - State Based

    Taken from BlueSky library, but made "adjustable" as a separate script here
    """
    def __init__(self):
        self.rpz = -1
        self.hpz = -1
        self.dtlookahead = -1

        self.confpairs = list()
        self.lospairs = list()
        self.qdr = np.array([])
        self.dist = np.array([])
        self.dcpa = np.array([])
        self.tcpa = np.array([])
        self.tLOS = np.array([])

        self.confpairs_unique = {}
    
    def detect(self, ownship, intruder, rpz, hpz, dtlookahead):
        ''' Conflict detection between ownship (traf) and intruder (traf/adsb).'''

        self.rpz = np.array([rpz] * (ownship.ntraf))
        self.hpz = np.array([hpz] * (ownship.ntraf))
        self.dtlookahead = [dtlookahead] * (ownship.ntraf)

        # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
        eye = np.eye(ownship.ntraf)

        # Horizontal conflict ------------------------------------------------------
        qdr, dist = relative_bearing_distance(ownship, intruder, eye)

        swhorconf, tcpa, dcpa2, dtinhor, tinhor, touthor, vrel, rpz_mat = horizontal_conflict(
            ownship, intruder, qdr, dist, rpz, eye)

        # this is to check the end condition, tcpa and tin should be < 0
        self.tcpa_all = tcpa
        self.tinhor_all = tcpa - dtinhor

        # Vertical conflict --------------------------------------------------------
        dalt, tinver, toutver, hpz_mat = vertical_conflict(ownship, intruder, hpz, eye)

        # Combine vertical and horizontal conflict----------------------------------
        tinconf = np.maximum(tinver, tinhor)
        toutconf = np.minimum(toutver, touthor)

        swconfl = np.array(swhorconf * (tinconf <= toutconf) * (toutconf > 0.0) *
                           np.asarray(tinconf < np.asmatrix(dtlookahead).T) * (1.0 - eye), dtype=bool)

        # --------------------------------------------------------------------------
        # Update conflict lists
        # --------------------------------------------------------------------------
        # Ownship conflict flag and max tCPA
        self.inconf = np.any(swconfl, 1)
        self.tcpamax = np.max(tcpa * swconfl, 1)

        # Select conflicting pairs: each a/c gets their own record
        self.confpairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swconfl))]
        self.confpairs_unique = {frozenset(pair) for pair in self.confpairs}
        swlos = (dist < rpz_mat) * (np.abs(dalt) < hpz_mat)
        self.lospairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swlos))]

        self.qdr = qdr[swconfl]
        self.dist = dist[swconfl]

        self.dcpa = np.sqrt(dcpa2[swconfl])
        self.tcpa = tcpa[swconfl]
        self.tLOS = tinconf[swconfl]
