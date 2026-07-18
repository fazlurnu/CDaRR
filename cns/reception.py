''' Packet reception sampling -- functional core.

Fresh copy of sim_models/reception_model.py's ReceptionModel, as a single
pure function (refactor_fp.md's exact target shape:
``sample_received(n, ci95, rng) -> ndarray[int]``).
'''
import numpy as np


def sample_received(n, p, rng):
    ''' Indices of aircraft that receive a packet this tick (Bernoulli per
    aircraft, probability p). Pure given rng. Returns an int array, empty if
    n <= 0.
    '''
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if n <= 0:
        return np.array([], dtype=int)
    mask = rng.random(n) <= p
    return np.where(mask)[0]
