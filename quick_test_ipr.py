'''Quick IPR check at crossing angle = 2 deg, 100 pairs x 4 runs, all 3 methods.

Ad hoc comparison script, mirrors the quick_test_ipr.py used in CDaRR_FP_New /
CDaRR_FP_Old, adapted to this project's run_multiple_jobs / get_ipr_stochastic_env
API (NB_PAIR is fixed at 100 pairs internally — see sim/pairwise_stochastic/run_multiple_jobs.py).
'''
from sim.pairwise_stochastic.run_multiple_jobs import run_multiple_jobs

ANGLE   = 2
N_RUNS  = 4
N_JOBS  = 4
CI, CIV = 3, 1   # pos3_vel1 uncertainty level
GAMMA   = 0.999
CONFIG_PATH = "sim_configs/sim_config.json"

METHODS = ["CPA", "FTR", "Probabilistic FTR"]

print(f'{"Method":<20} {"Overall IPR":>12}')
print('-' * 33)
for method in METHODS:
    res = run_multiple_jobs(
        n_runs=N_RUNS, n_jobs=N_JOBS,
        asas_marh=1.05,
        confidence_interval=CI, confidence_interval_velo=CIV,
        reception_prob=1.0,
        lookahead_time=120,
        dpsi=ANGLE,
        config_path=CONFIG_PATH,
        recovery_model=method,
        threshold_probability=GAMMA,
    )
    print(f'{method:<20} {res["overall_ipr"]:>12.4f}')
