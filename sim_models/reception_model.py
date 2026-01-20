import numpy as np

class ReceptionModel:
    """Packet reception sampler (Bernoulli per aircraft)."""

    def __init__(self, reception_prob: float, rng: np.random.Generator):
        if not (0.0 <= reception_prob <= 1.0):
            raise ValueError("reception_prob must be in [0, 1]")
        self.p = float(reception_prob)
        
        # this is just to generate random number to be used
        # in the sample_indices
        self.rng = rng

    def sample_indices(self, n: int) -> np.ndarray:
        """Return indices of aircraft that receive a packet this tick."""
        if n <= 0:
            return np.array([], dtype=int)
        mask = self.rng.random(n) <= self.p
        return np.where(mask)[0]