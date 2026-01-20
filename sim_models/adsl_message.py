import numpy as np
from typing import Any, Optional
from sim_models.utils import _resize_1d, _resize_list

class ADSLMessage:
    """
    Holds the *last known* (noisy) measurements for a traffic set.
    Arrays are 1D and aligned by aircraft index.

    This message format does not contain all message in ADS-L tech spec.
    This is just for the purpose of simulation
    Also there's no resampling here
    """

    def __init__(self) -> None:
        self.lat = np.array([], dtype=float)      # deg
        self.lon = np.array([], dtype=float)      # deg
        self.alt = np.array([], dtype=float)      # m
        self.hdg = np.array([], dtype=float)      # deg
        self.trk = np.array([], dtype=float)      # deg
        self.gs  = np.array([], dtype=float)      # m/s
        self.tas = np.array([], dtype=float)      # m/s
        self.vs  = np.array([], dtype=float)      # m/s

        self.gseast  = np.array([], dtype=float)  # m/s
        self.gsnorth = np.array([], dtype=float)  # m/s

        self.id: list[str] = []                   # identifiers (strings)

        # Convenience references (typically shared objects, not per-aircraft arrays)
        # Non related to ADS-L, but required by BlueSky, to be scrapped later, hopefully
        self.perf: Any = None
        self.ap: Any = None
        self.selalt: Any = None

    @property
    def ntraf(self) -> int:
        return int(self.lat.size)

    @property
    def empty(self) -> bool:
        return self.ntraf == 0

    def ensure_size(self, n: int) -> None:
        """Ensure all arrays/lists have length n, preserving existing values."""
        if n < 0:
            raise ValueError("n must be non-negative")

        self.lat = _resize_1d(self.lat, n)
        self.lon = _resize_1d(self.lon, n)
        self.alt = _resize_1d(self.alt, n)
        self.hdg = _resize_1d(self.hdg, n)
        self.trk = _resize_1d(self.trk, n)
        self.gs  = _resize_1d(self.gs, n)
        self.tas = _resize_1d(self.tas, n)
        self.vs  = _resize_1d(self.vs, n)

        self.gseast  = _resize_1d(self.gseast, n)
        self.gsnorth = _resize_1d(self.gsnorth, n)

        self.id = _resize_list(self.id, n, fill_value="")

    def copy_from_states(self, states: Any, idx: np.ndarray) -> None:
        """
        Copy truth-state fields for aircraft in idx into this message.
        Assumes ensure_size(states.ntraf) has already been called.
        """
        self.lat[idx] = states.lat[idx]
        self.lon[idx] = states.lon[idx]
        self.alt[idx] = states.alt[idx]
        self.hdg[idx] = states.hdg[idx]
        self.trk[idx] = states.trk[idx]
        self.gs[idx]  = states.gs[idx]
        self.tas[idx] = states.tas[idx]
        self.vs[idx]  = states.vs[idx]

        # Keep optional/shared references aligned with your original model
        self.perf = getattr(states, "perf", None)
        self.ap = getattr(states, "ap", None)
        self.selalt = getattr(states, "selalt", None)

        # IDs: patch per index (works whether states.id is list-like or array-like)
        if hasattr(states, "id"):
            sid = states.id
            for i in idx:
                self.id[int(i)] = sid[int(i)]

        # Optional: if truth carries east/north already, store those too
        if hasattr(states, "gseast") and hasattr(states, "gsnorth"):
            self.gseast[idx] = states.gseast[idx]
            self.gsnorth[idx] = states.gsnorth[idx]

    def copy_from_message(self, other: "ADSLMessage", idx: Optional[np.ndarray] = None) -> None:
        """
        Copy from another message.
        - idx is None: full copy
        - idx provided: patch those indices; if sizes mismatch, fall back to full copy
        """
        if idx is None:
            self.lat = other.lat.copy()
            self.lon = other.lon.copy()
            self.alt = other.alt.copy()
            self.hdg = other.hdg.copy()
            self.trk = other.trk.copy()
            self.gs  = other.gs.copy()
            self.tas = other.tas.copy()
            self.vs  = other.vs.copy()

            self.gseast  = other.gseast.copy()
            self.gsnorth = other.gsnorth.copy()

            self.id = other.id.copy()

            self.perf = other.perf
            self.ap = other.ap
            self.selalt = other.selalt
            return

        if self.ntraf != other.ntraf:
            self.copy_from_message(other, idx=None)
            return

        self.lat[idx] = other.lat[idx]
        self.lon[idx] = other.lon[idx]
        self.alt[idx] = other.alt[idx]
        self.hdg[idx] = other.hdg[idx]
        self.trk[idx] = other.trk[idx]
        self.gs[idx]  = other.gs[idx]
        self.tas[idx] = other.tas[idx]
        self.vs[idx]  = other.vs[idx]

        self.gseast[idx]  = other.gseast[idx]
        self.gsnorth[idx] = other.gsnorth[idx]

        for i in idx:
            self.id[int(i)] = other.id[int(i)]