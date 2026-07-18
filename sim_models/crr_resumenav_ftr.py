''' Deprecated forwarding shim (refactor_fp.md Phase 5).

This module moved to `legacy.sim_models.crr_resumenav_ftr` -- it was the frozen
reference implementation the functional-core port (cd/cr/crr/cns/cdarr) was
verified against, and the live simulation loop
(sim/pairwise_stochastic/loop.py, wired in by
sim/pairwise_stochastic/get_ipr_stochastic_env.py's Phase 4 switchover) no
longer uses it. Kept importable at this path, unchanged, as a grace-window
forward -- so the equivalence test suite (test/equivalence/*.py, which
compares the new core against this exact code) and any other existing caller
keep working without edits. New code should import from
`legacy.sim_models.crr_resumenav_ftr` directly.
'''
from legacy.sim_models.crr_resumenav_ftr import *  # noqa: F401,F403
