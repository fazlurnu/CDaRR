# Conflict Detection and Resolution Robustas

Use this [bluesky version](https://github.com/fazlurnu/bluesky.git), branch CDaRR, this one has turn rate limiter specific for M600.

<p align="center">
  <img src="CDARR.png" alt="CDARR" width="200">
</p>

## Architecture

The conflict detection / resolution / recovery core is a set of pure
functional packages, composed by `cdarr/core.py`:

- `cd/` — conflict detection (state-based CPA geometry)
- `cr/` — conflict resolution (MVP, VO)
- `crr/` — conflict recovery / resume-navigation (Past-CPA, FTR, Probabilistic FTR)
- `cns/` — communication/navigation/surveillance (position/velocity noise,
  packet reception, the ADS-L message algebra)
- `cdarr/` — composes the above into `cdarr()` / `cdarr_step()`

`sim/pairwise_stochastic/loop.py` is the simulation shell built on this
core; `sim/pairwise_stochastic/get_ipr_stochastic_env.py` is a thin,
signature-compatible wrapper over it, so every existing caller
(`experiments/*`, `compare_crr/*`, `sim/pairwise_stochastic/run_multiple_jobs.py`)
needs no changes.

The original Entity-class implementation (`sim_models/`) is preserved,
unchanged, under `legacy/sim_models/` — it's the frozen reference the
functional core is verified against (`test/equivalence/`), reproducing it
bit-for-bit. The old `sim_models/*.py` import paths still work as thin
forwarding shims during the deprecation grace window.

See [refactor_fp.md](refactor_fp.md) for the full refactor plan and
rationale, and `test/golden/KNOWN_ISSUES.md` for known behavioral notes.

