"""
Experiment runner for comparing conflict recovery methods.

Runs Monte Carlo simulations across the full parameter matrix:
  - Recovery methods: CPA, FTR, Probabilistic FTR
  - Position 95% CI: {3, 10} m
  - Velocity 95% CI: {1, 3} m/s
  - Threshold gamma: {0.5, 0.75, 0.9, 0.99, 0.999} (probabilistic only)
  - Crossing angles: 2-42 by 2, 45-180 by 5

Usage:
    nohup python -u -m compare_crr.compare_crr > compare_crr.log 2>&1 &
"""

import os
import json
from types import SimpleNamespace

from compare_crr.generate_ipr_from_random_dpsi import generate_ipr_from_random_dpsi
from compare_crr.utils import to_python, build_output_path

# ---------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------
CONFIG_PATH = "sim_configs/sim_config_ultimate200.json"

EXPERIMENTS = [
    # {
    #     "recovery_models": ["Probabilistic FTR"],
    #     "ci_list": [3, 10],
    #     "civ_list": [1, 3],
    #     "gamma_list": [0.999],
    #     "output_dir_template": "results/tests_tin_1.5tlookahead_ulti200_gamma{gamma}",
    # },
    {
        "recovery_models": ["CPA", "FTR"],
        "ci_list": [3, 10],
        "civ_list": [1, 3],
        "gamma_list": [None],
        "output_dir": "results/tests_tin_1.5tlookahead_ulti200",
    },
]

# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------
def run_experiments():
    for experiment in EXPERIMENTS:
        recovery_models = experiment["recovery_models"]
        ci_list = experiment["ci_list"]
        civ_list = experiment["civ_list"]
        gamma_list = experiment["gamma_list"]

        for recovery_model in recovery_models:
            for civ in civ_list:
                for ci in ci_list:
                    for gamma in gamma_list:
                        # Determine output directory
                        if "output_dir_template" in experiment:
                            output_dir = experiment["output_dir_template"].format(gamma=gamma)
                        else:
                            output_dir = experiment["output_dir"]

                        output_path = build_output_path(
                            output_dir, ci, civ, 1.0, gamma, recovery_model
                        )

                        # Skip if already computed
                        if os.path.exists(output_path):
                            print(f"SKIP (exists): {output_path}")
                            continue

                        print(
                            f"Running: recovery={recovery_model}, CI={ci}, CIV={civ}, "
                            f"gamma={gamma}"
                        )

                        params = SimpleNamespace(
                            confidence_interval=ci,
                            confidence_interval_velo=civ,
                            reception_prob=1.0,
                            config_path=CONFIG_PATH,
                            threshold_probability=gamma,
                            recovery_model=recovery_model,
                        )

                        results = generate_ipr_from_random_dpsi(params)
                        results_clean = to_python(results)

                        os.makedirs(output_dir, exist_ok=True)

                        with open(output_path, "w") as f:
                            json.dump(results_clean, f, indent=2)

                        print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_experiments()
