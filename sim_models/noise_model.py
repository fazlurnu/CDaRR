import numpy as np
from sim_models.adsl_message import ADSLMessage
from typing import Any, Callable, Optional

# 95% radial CI -> per-axis 1-sigma (matches ADSL.CI95_TO_STD_2D).
CI95_TO_STD_2D = 2.448

class NoiseModel:
    """
    Applies measurement noise.
    - Position: 2D Gaussian (meters) converted to lat/lon degrees.
    - Velocity: 2D Gaussian added to North/East ground-speed components (m/s).

    Optional (exp3/exp4 noise-model sweep):
    - ``pos_dist``: a callable ``(n, ci95, rng, trk_rad) -> (n, 2)`` in metres
      that replaces the default Gaussian position draw (e.g. mixture-Gaussian,
      anisotropic along-/cross-track Gaussian). ``trk_rad`` is the per-sample
      aircraft track angle in radians, for distributions that need to orient
      themselves relative to heading; distributions that don't need it may
      ignore the argument. When ``pos_dist`` is ``None`` the original
      2D-Gaussian behaviour is used unchanged.
    - ``latency_s``: ADS-B position reporting latency in seconds. Adds a
      per-aircraft along-track bias of ``-latency_s * gs`` (metres), i.e. the
      reported position lags behind truth. ``0.0`` disables it.
    """

    def __init__(
        self,
        pos_std_m: float,
        vel_std_ms: float,
        rng: np.random.Generator,
        pos_dist: Optional[Callable[[int, float, np.random.Generator], np.ndarray]] = None,
        latency_s: float = 0.0,
    ):
        self.rng = rng
        self.pos_cov = np.array([[pos_std_m**2, 0.0],
                                 [0.0, pos_std_m**2]], dtype=float)
        self.vel_cov = np.array([[vel_std_ms**2, 0.0],
                                 [0.0, vel_std_ms**2]], dtype=float)
        self.pos_dist = pos_dist
        self.latency_s = float(latency_s)
        # ci95 equivalent of pos_std_m, for pluggable distributions that take a CI.
        self._pos_ci95 = float(pos_std_m) * CI95_TO_STD_2D

    def add_position_noise(self, msg: ADSLMessage, states: Any, idx: np.ndarray) -> None:
        """Add noise to msg.lat/msg.lon for aircraft in idx."""
        if idx.size == 0:
            return

        trk_rad = np.deg2rad(np.asarray(states.trk[idx], dtype=float))

        # Draw (east_m, north_m): custom distribution if provided, else Gaussian.
        if self.pos_dist is None:
            xy = self.rng.multivariate_normal((0.0, 0.0), self.pos_cov, size=int(idx.size))
        else:
            xy = self.pos_dist(int(idx.size), self._pos_ci95, self.rng, trk_rad)
        east_m = xy[:, 0]
        north_m = xy[:, 1]

        mean_lat = np.asarray(states.lat[idx], dtype=float)

        lat_noise_deg = north_m / 111_320.0
        coslat = np.cos(np.deg2rad(mean_lat))
        coslat = np.maximum(coslat, 1e-6)
        lon_noise_deg = east_m / (111_320.0 * coslat)

        msg.lat[idx] = states.lat[idx] + lat_noise_deg
        msg.lon[idx] = states.lon[idx] + lon_noise_deg

        # Along-track latency bias: reported position lags truth by latency_s * gs.
        # Factored out into its own method so callers that need to apply *only*
        # the deterministic bias (e.g. to a received/relayed copy, without
        # redrawing the random position-noise component) can call it directly.
        self.add_latency_bias(msg, states, idx)

    def add_latency_bias(self, msg: ADSLMessage, states: Any, idx: np.ndarray) -> None:
        """Add *only* the deterministic along-track latency bias to msg.lat/lon
        for aircraft in idx (no new random draw). No-op if ``latency_s == 0``.

        Uses this NoiseModel's ``latency_s`` and the (true) groundspeed/track
        in ``states`` for the bias magnitude/direction, matching the
        along-track convention in ``add_position_noise``.
        """
        if idx.size == 0 or not self.latency_s:
            return

        trk_rad = np.deg2rad(np.asarray(states.trk[idx], dtype=float))
        gs = np.asarray(states.gs[idx], dtype=float)
        bias_at = -self.latency_s * gs
        east_bias = bias_at * np.sin(trk_rad)
        north_bias = bias_at * np.cos(trk_rad)

        mean_lat = np.asarray(states.lat[idx], dtype=float)
        lat_bias_deg = north_bias / 111_320.0
        coslat = np.maximum(np.cos(np.deg2rad(mean_lat)), 1e-6)
        lon_bias_deg = east_bias / (111_320.0 * coslat)

        msg.lat[idx] = msg.lat[idx] + lat_bias_deg
        msg.lon[idx] = msg.lon[idx] + lon_bias_deg

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