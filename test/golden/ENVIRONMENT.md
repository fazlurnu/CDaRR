# Reference environment — reproducibility baseline

Pins the environment in which the **frozen reference code** (branch `main` at the freeze
tag) is run to produce the golden baselines. The refactored code on `fp-refactor` must
reproduce those baselines bit-for-bit *in this same environment* — see
[refactor_fp.md](../../refactor_fp.md) §5.1.

## Freeze point
- Reference tag: `pre-fp-refactor`
- Reference commit: `90d8b3e8590f5dbdac9ccf0cc64d9defbc42d593` ("Add FP refactor plan")
- Refactor branch: `fp-refactor` (worktree at `/Users/mfrahman/Projects/CDaRR_fp-refactor`)
- Captured: 2026-07-17

## Machine / interpreter
- Platform: macOS-26.5.2-arm64 (Darwin arm64)
- Python: 3.11.15
- Conda env: `cdarr`

## Pinned packages (numerics-relevant)
- numpy 2.3.5
- scipy 1.17.1
- joblib 1.5.3
- shapely 2.1.2

Full snapshots alongside this file: `conda-env.yml`, `pip-freeze.txt`.

## BlueSky fork (vendored dependency, not pip-managed)
- Path: `/Users/mfrahman/Projects/bluesky`
- Branch: `CDaRR`
- Commit: `800f47d4af92a3f17adae943a4f9ebf09307c5ca`

## Foundation smoke test (first reproducibility evidence)
`get_ipr_stochastic_env(dpsi=90, seed=44, MVP + FTR, config=test_config_2x2.json)`,
run twice in one process:
- ipr = 1.0, distance_array shape = (963, 4), sim_timer = 192.4, n_active = 0
- distance_array SHA-256[:16] = `8c35ca66d7b35268` — identical across both runs.

Encoded as a pytest determinism guard in `test/equivalence/test_smoke.py`.

## How to reproduce
    conda activate cdarr
    # BlueSky fork must be at commit 800f47d on branch CDaRR
    cd /Users/mfrahman/Projects/CDaRR_fp-refactor
    python -m pytest test/equivalence
