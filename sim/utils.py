import numpy as np
from typing import Optional, Tuple, Union
import json
from pathlib import Path
import bluesky as bs
from types import SimpleNamespace

import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            
def _check_tcpa_tinhor_per_pair(id, tcpa, tinhor):
    """
    Returns:
      done_now (bool): True if for all matched pairs tcpa < 0 and tinhor < 0
      n_active (int): number of matched pairs with tcpa > 0 and tinhor > 0
    """
    id = np.asarray(id, dtype=str)

    is_dro = np.char.startswith(id, "DRO")
    is_dri = np.char.startswith(id, "DRI")

    # numeric suffix, e.g. '000'
    num = np.array([s[3:] for s in id])

    # Build mapping suffix -> indices
    dro_idx = {num[i]: i for i in range(len(id)) if is_dro[i]}
    dri_idx = {num[i]: i for i in range(len(id)) if is_dri[i]}

    common = sorted(set(dro_idx.keys()) & set(dri_idx.keys()))
    if not common:
        return False, 0

    tcpa_pairs = np.empty(len(common), dtype=float)
    tin_pairs  = np.empty(len(common), dtype=float)

    for k, suf in enumerate(common):
        i = dro_idx[suf]
        j = dri_idx[suf]
        tcpa_pairs[k] = tcpa[i, j]
        tin_pairs[k]  = tinhor[i, j]

    # "Done" condition (same as your original)
    done_now = bool(np.all(tcpa_pairs < 0) and np.all(tin_pairs < 0))

    # Count how many are still "active conflicts ahead"
    is_active = (tcpa_pairs > 0) & (tin_pairs > 0)

    return done_now, is_active

def done_with_timeout(done_now: bool,
                      
                      done_start_time: Optional[float],
                      sim_timer_second: float,
                      done_timeout: float,
                      *,
                      verbose: bool = False) -> Tuple[Optional[float], bool]:
    # Latch
    if done_now:
        if done_start_time is None:
            done_start_time = sim_timer_second
    else:
        done_start_time = None

    # Stop only after timeout
    should_stop = (
        done_start_time is not None
        and (sim_timer_second - done_start_time) >= done_timeout
    )

    if should_stop and verbose:
        print("Done + timeout reached, stopping simulation")

    return done_start_time, should_stop

def get_configs(config_path: Union[str, Path] = "sim_configs/sim_config.json") -> SimpleNamespace:
    """
    Load simulation configuration from JSON, apply BlueSky settings,
    and return a flattened config dictionary for use in the simulation.

    Parameters
    ----------
    config_path : str or Path
        Path to sim_config.json

    Returns
    -------
    dict
        Flattened configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        cfg = json.load(f)

    # -------------------------------------------------
    # Scenario
    # -------------------------------------------------
    scenario = cfg["scenario"]
    width = scenario["width"]
    height = scenario["height"]
    dpsi = scenario["dpsi_deg"]
    horizontal_sep = scenario["horizontal_sep_m"]

    # -------------------------------------------------
    # Aircraft
    # -------------------------------------------------
    aircraft = cfg["aircraft"]
    aircraft_type = aircraft["type"]
    init_speed_ownship = aircraft["init_speed_ownship_mps"]
    init_speed_intruder = aircraft["init_speed_intruder_mps"]

    # -------------------------------------------------
    # Simulation
    # -------------------------------------------------
    simulation = cfg["simulation"]
    SIMDT_FACTOR = simulation["simdt_factor"]
    DONE_TIMEOUT = simulation["done_timeout_s"]
    tmax_factor = simulation["tmax_factor"]

    # -------------------------------------------------
    # CDR / ASAS settings
    # -------------------------------------------------
    cdr = cfg["cdr_settings"]
    lookahead_time = cdr["lookahead_time_s"]
    asas_marh = cdr["asas_marh"]

    # -------------------------------------------------
    # ADSL
    # -------------------------------------------------
    adsl = cfg["adsl"]
    seed = adsl["seed"]
    confidence_interval = adsl["confidence_interval_m"]
    confidence_interval_velo = adsl["confidence_interval_velo_mps"]
    reception_prob = adsl["reception_prob"]

    # -------------------------------------------------
    # Conflict models
    # -------------------------------------------------
    conflict_models = cfg["conflict_models"]
    detection_model = conflict_models["detection"]
    resolution_model = conflict_models["resolution"]
    recovery_model = conflict_models["recovery"]

    # -------------------------------------------------
    # Apply BlueSky settings (must be after bs.init)
    # -------------------------------------------------
    bluesky_cfg = cfg["bluesky"]
    bs.traf.MAX_TR = bluesky_cfg["max_tr"]
    bs.traf.MAX_DTR2 = bluesky_cfg["max_dtr2"]
    bs.settings.asas_marh = asas_marh

    return SimpleNamespace(
        # Scenario
        width=width,
        height=height,
        dpsi=dpsi,
        horizontal_sep=horizontal_sep,

        # Aircraft
        aircraft_type=aircraft_type,
        init_speed_ownship=init_speed_ownship,
        init_speed_intruder=init_speed_intruder,

        # Simulation
        SIMDT_FACTOR=SIMDT_FACTOR,
        DONE_TIMEOUT=DONE_TIMEOUT,
        tmax_factor=tmax_factor,
        lookahead_time=lookahead_time,

        # ADSL
        confidence_interval=confidence_interval,
        confidence_interval_velo=confidence_interval_velo,
        reception_prob=reception_prob,

        # Conflict models
        detection_model=detection_model,
        resolution_model=resolution_model,
        recovery_model=recovery_model
    )

