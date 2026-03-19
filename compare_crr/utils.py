"""Utilities for the compare_crr experiment runner."""

import os
import numpy as np


def to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


def build_output_path(output_dir, ci, civ, reception_prob, gamma, recovery_model):
    """Build the output JSON file path for a given parameter combination.

    Parameters
    ----------
    output_dir : str
        Base output directory.
    ci : float
        Position confidence interval.
    civ : float
        Velocity confidence interval.
    reception_prob : float
        Reception probability.
    gamma : float or None
        Threshold probability (None for deterministic methods).
    recovery_model : str
        Recovery model name.

    Returns
    -------
    str : full path to the output JSON file.
    """
    # Sanitize recovery model name for use in filenames
    recovery_tag = recovery_model.lower().replace(" ", "_")

    parts = [
        f"samples_results",
        f"{ci}",
        f"{civ}",
        f"{reception_prob}",
    ]
    if gamma is not None:
        parts.append(f"{gamma}")
    parts.append(recovery_tag)

    filename = "_".join(parts) + ".json"
    return os.path.join(output_dir, filename)
