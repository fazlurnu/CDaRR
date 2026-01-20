import numpy as np
from sim_models.adsl_message import ADSLMessage
from typing import Any

class NoiseModel:
    """
    Applies measurement noise.
    - Position: 2D Gaussian (meters) converted to lat/lon degrees.
    - Velocity: 2D Gaussian added to North/East ground-speed components (m/s).
    """

    def __init__(self, pos_std_m: float, vel_std_ms: float, rng: np.random.Generator):
        self.rng = rng
        self.pos_cov = np.array([[pos_std_m**2, 0.0],
                                 [0.0, pos_std_m**2]], dtype=float)
        self.vel_cov = np.array([[vel_std_ms**2, 0.0],
                                 [0.0, vel_std_ms**2]], dtype=float)

    def add_position_noise(self, msg: ADSLMessage, states: Any, idx: np.ndarray) -> None:
        """Add noise to msg.lat/msg.lon for aircraft in idx."""
        if idx.size == 0:
            return

        # Draw (east_m, north_m)
        xy = self.rng.multivariate_normal((0.0, 0.0), self.pos_cov, size=int(idx.size))
        east_m = xy[:, 0]
        north_m = xy[:, 1]

        mean_lat = np.asarray(states.lat[idx], dtype=float)

        lat_noise_deg = north_m / 111_320.0
        coslat = np.cos(np.deg2rad(mean_lat))
        coslat = np.maximum(coslat, 1e-6)
        lon_noise_deg = east_m / (111_320.0 * coslat)

        msg.lat[idx] = states.lat[idx] + lat_noise_deg
        msg.lon[idx] = states.lon[idx] + lon_noise_deg

    def add_velocity_noise(self, msg: ADSLMessage, idx: np.ndarray) -> None:
        """
        Add noise to msg.gsnorth/msg.gseast for aircraft in idx.
        Uses current stored msg.gs and msg.trk (last-known) at those indices.
        """
        if idx.size == 0:
            return

        vxy = self.rng.multivariate_normal((0.0, 0.0), self.vel_cov, size=int(idx.size))
        north_noise = vxy[:, 0]
        east_noise  = vxy[:, 1]

        trk_rad = np.deg2rad(msg.trk[idx])
        gs = msg.gs[idx]

        msg.gsnorth[idx] = gs * np.cos(trk_rad) + north_noise
        msg.gseast[idx]  = gs * np.sin(trk_rad) + east_noise