# Known issues — frozen reference behavior

Documented, deliberately-unfixed characteristics of the pipeline at the freeze tag
`pre-fp-refactor`. The refactor **preserves** these (reproduce-first); fixing them is a
separate, behavior-changing decision — see [refactor_fp.md](../../refactor_fp.md) §9.

## KI-1 — In-process, order-dependent nondeterminism

**Symptom.** Repeated `get_ipr_stochastic_env` / `run_multiple_jobs` calls in the *same
process* can produce different floating-point trajectories. A call's result depends on
what ran before it in that process. A *fresh process running a fixed sequence* is fully
reproducible.

**Evidence (2×2 config, found during Phase 0).**
- `capture_goldens.py` then `capture_goldens.py --verify` in two separate processes:
  all 16 cases MATCH.
- The same case (`c01`) run back-to-back in one process: different `distance_array`
  bytes (identical shape, identical IPR — a floating-point trajectory difference).
- `np.random.seed(...)` before each run does **not** fix it → the leak is bluesky global
  state that `pairwise.reset()` (which only calls `bs.traf.reset()`) does not clear, **not**
  numpy's global RNG.

**Scope / impact.**
- Aggregate metrics (IPR over ~100k pairs) are statistically robust.
- Per-run / per-pair values (e.g. `worst_cpa`) are **not** bit-reproducible when more than
  one repetition runs in the same joblib worker:
  - **Exposed:** exp3 / exp4 / exp5 (`N_RUNS=1000`, `N_JOBS=100` → ~10 reps/worker).
  - **Not exposed:** compare_crr, exp1, exp2 (`n_runs == n_jobs` → 1 rep/worker, fresh
    process each).
- The precise numeric impact on published figures has **not** been quantified.

**Mitigation (adopted).**
- Golden capture and every equivalence check run in a **fixed-order, fresh process**.
- The pytest golden test shells out to `capture_goldens.py --verify` (subprocess) instead
  of running cases in-process.
- The n_jobs-invariance guard is marked `xfail` pointing here.

**Status.** Deferred. Candidate root causes to investigate if a fix is wanted: residual
`bs.traf` per-aircraft arrays not zeroed on reset; accumulated bluesky `Entity`
registrations from creating a fresh MVP/VO/StateBased each call; or `bs.sim` state not
reset. A fix changes results and requires re-baselining the goldens and re-running
exp3/4/5.
