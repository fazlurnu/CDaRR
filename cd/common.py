''' Shared data type for conflict-detection output. '''
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class ConflictData:
    ''' Immutable snapshot of one detect() call's output.

    Field names mirror the legacy StateBased instance attributes of the same
    name (rpz, hpz, dtlookahead, confpairs, qdr, dist, dcpa, tcpa, tLOS,
    inconf, tcpamax, confpairs_unique, lospairs, tcpa_all, tinhor_all), so
    ConflictData is a drop-in replacement wherever `conf` is read today
    (conf.confpairs, conf.rpz[[i, j]], conf.dtlookahead[idx1], ...) -- see
    refactor_fp.md section 4 (the cdarr() composition).

    `eq=False`: several fields are numpy arrays, whose `==` returns an
    elementwise array rather than a bool; the auto-generated dataclass
    `__eq__` would raise on comparison. Equivalence tests compare fields
    individually (np.array_equal, etc.), never the dataclass as a whole.
    '''
    rpz: np.ndarray
    hpz: np.ndarray
    dtlookahead: list
    confpairs: list
    confpairs_unique: frozenset
    lospairs: list
    qdr: np.ndarray
    dist: np.ndarray
    dcpa: np.ndarray
    tcpa: np.ndarray
    tLOS: np.ndarray
    inconf: np.ndarray
    tcpamax: np.ndarray
    tcpa_all: np.ndarray
    tinhor_all: np.ndarray
