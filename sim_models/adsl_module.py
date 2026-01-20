import numpy as np
from sim_models.utils import _normalize_indices
from typing import Any, Optional

from sim_models.noise_model import NoiseModel
from sim_models.reception_model import ReceptionModel
from sim_models.adsl_message import ADSLMessage

class ADSL:
    """
    ADS-L / ADS-B-like measurement node.

    Rules:
    1) First update: NO packet loss; all aircraft get initial measurement (noisy).
    2) Later updates: packet loss applied; only received aircraft are updated & re-noised.

    Parameters
    ----------
    confidence_interval : float
        Position 95% confidence interval (meters assumed).
    confidence_interval_velo : float
        Velocity 95% confidence interval (m/s assumed).
    reception_prob : float
        Per-aircraft reception probability after the first update.
    seed : Optional[int]
        RNG seed for reproducibility.
    """
    CI95_TO_STD_2D = 2.448  # same factor as your original comment/code

    def __init__(
        self,
        confidence_interval: float,
        confidence_interval_velo: float,
        reception_prob: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)

        pos_std = float(confidence_interval) / self.CI95_TO_STD_2D
        vel_std = float(confidence_interval_velo) / self.CI95_TO_STD_2D

        self.msg = ADSLMessage()
        self.reception = ReceptionModel(reception_prob=reception_prob, rng=self.rng)
        self.noise = NoiseModel(pos_std_m=pos_std, vel_std_ms=vel_std, rng=self.rng)

        self.first_update_done = False

    # Compatibility: allow direct access like self.lat, self.lon, etc.
    def __getattr__(self, name: str) -> Any:
        if hasattr(self.msg, name):
            return getattr(self.msg, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def ntraf(self) -> int:
        return self.msg.ntraf

    def update_from_truth(self, states: Any) -> np.ndarray:
        """
        Update this node's message from truth `states`.

        Returns indices updated this tick.
        """
        n = int(states.ntraf)
        self.msg.ensure_size(n)

        # Requirement (1): first update is full, no packet loss
        if not self.first_update_done:
            idx = np.arange(n, dtype=int)
        else:
            idx = self.reception.sample_indices(n)

        # Copy truth for updated aircraft, then add noise for those aircraft
        self.msg.copy_from_states(states, idx)
        self.noise.add_position_noise(self.msg, states, idx)
        self.noise.add_velocity_noise(self.msg, idx)

        self.first_update_done = True
        return idx

    # Backwards-compatible method name (if your code calls this)
    def _get_noisy_states(self, states: Any) -> None:
        self.update_from_truth(states)

    # Comms: copy message between nodes/modules
    @staticmethod
    def send_data(dst_adsl: "ADSL", src_adsl: "ADSL", indices: Any = None) -> None:
        """
        Copy measurements from src to dst (simulates sending data).
        - indices is None => full copy
        - indices provided => patch copy for those aircraft
        """
        if indices is None:
            dst_adsl.msg.copy_from_message(src_adsl.msg, idx=None)
        else:
            idx = _normalize_indices(indices, src_adsl.ntraf)
            dst_adsl.msg.ensure_size(src_adsl.ntraf)
            dst_adsl.msg.copy_from_message(src_adsl.msg, idx=idx)

        dst_adsl.first_update_done = True

    def transmit_to(self, dst_adsl: "ADSL", indices: Any = None) -> None:
        """Convenience wrapper: transmit this node's message to dst."""
        self.send_data(dst_adsl=dst_adsl, src_adsl=self, indices=indices)