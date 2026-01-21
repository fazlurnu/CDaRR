import matplotlib.pyplot as plt
import numpy as np
import os

import json

def to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj

from sampling_techniques.sampling.generate_ipr_sobol_samples import generate_ipr_sobol_samples
from types import SimpleNamespace

import numpy as np
import pandas as pd

# let's define the bounds here and we generate the samples accordingly

# x1 is resofach, x2 is lookahead time
bounds = SimpleNamespace(
    x1_min=0.9,
    x1_max=1.2,
    x2_min=1.0,
    x2_max=120.0
)

confidence_interval_list = [1.5, 15.0]
confidence_interval_velo_list = [0.5, 1.5]
dpsi_list = [180, 24, 4, 2]

for ci in confidence_interval_list:
    for civ in confidence_interval_velo_list:
        for dpsi in dpsi_list:
            print("Generating initial samples for CI:", ci, "CIV:", civ, "DPSI:", dpsi)

            params = SimpleNamespace(
                confidence_interval=ci,
                confidence_interval_velo=civ,
                reception_prob=0.95,
                dpsi=dpsi,
            )

            results = generate_ipr_sobol_samples(bounds, params, n_samples=512, seed=42)

            results_clean = to_python(results)

            output_dir = "results/tests"
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(
                output_dir,
                f"initial_samples_results_"
                f"{params.confidence_interval}_"
                f"{params.confidence_interval_velo}_"
                f"{params.reception_prob}_"
                f"{params.dpsi}.json"
            )

            with open(output_path, "w") as f:
                json.dump(results_clean, f, indent=2)