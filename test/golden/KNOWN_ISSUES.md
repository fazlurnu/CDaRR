# Known issues — frozen reference behavior

Documented characteristics of the pipeline found relative to the freeze tag
`pre-fp-refactor`. See [refactor_fp.md](../../refactor_fp.md) §9 for the general
reproduce-first policy; KI-1 below is the one exception (see "Status").

## KI-1 — Cross-run recovery-state leak via the MVP/VO singleton (RESOLVED)

**Symptom.** Repeated `get_ipr_stochastic_env` / `run_multiple_jobs` calls in the *same
process* can produce different floating-point trajectories. A call's result depends on
what ran before it in that process. A *fresh process running a fixed sequence* was fully
reproducible, which is what let this hide behind the Phase 0 subprocess-based goldens.

**Root cause (found while starting Phase 1).** BlueSky's `EntityMeta` makes `MVP`/`VO`
process-wide singletons: `MVP()` in `_create_cdr_models` does not construct a fresh
object after the first call — it returns the persistent instance, `__init__` having
already run once. Its recovery bookkeeping (`resolution.resopairs`,
`resolution._intr_init_vel`) therefore survives from one `get_ipr_stochastic_env` call to
the next. Aircraft IDs are reused every run (`DRO###`/`DRI###`), so a new run's first
`resolve()` can see pairs already present in a stale `resopairs` — skipping the
"new pair" branch that records the initial intruder velocity — and recovery then reuses
the *previous* run's `_intr_init_vel` for that pair. `StateBased()` (detection) is
unaffected: its instance-level arrays are fully overwritten by every `.detect()` call, so
staleness never carries a stale value forward.

**Fix.** `_create_cdr_models` (`sim/pairwise_stochastic/get_ipr_stochastic_env.py`) now
resets `resolution.resopairs = set()` and `resolution._intr_init_vel = {}` immediately
after constructing/fetching the singleton, so every `get_ipr_stochastic_env` call starts
resolution from a clean state regardless of what ran before it in the process.

**Evidence the leak existed.**
- `capture_goldens.py` then `capture_goldens.py --verify` in two separate processes:
  all 16 cases MATCH (this is why Phase 0 didn't catch it).
- The same case (`c01`) run back-to-back in one process (pre-fix): different
  `distance_array` bytes, identical shape/IPR — a floating-point trajectory difference.
  `np.random.seed(...)` before each run did **not** fix it, confirming the leak was
  bluesky object state, not numpy's global RNG.
- Verified against the frozen reference (`git stash` the fix line, re-verify goldens):
  6/16 golden cases changed value with the fix applied — `c02, c09, c10, c12, c15, c16`
  (every FTR / Probabilistic-FTR case; CPA cases were unaffected since Past-CPA recovery
  doesn't use `_intr_init_vel`).

**Magnitude on aggregate IPR (measured, 2026-07-17).** `test/golden/leak_impact.py`, an
A/B (fixed vs. leaky) over 4 conditions mirroring exp3/exp4/exp5 at `n_runs=12,
n_jobs=4` (~3 reps/worker; the published exp3/4/5 runs use `N_RUNS=1000, N_JOBS=100`,
~10 reps/worker — this batch is a conservative lower bound on contamination):

| condition | fixed | leaky | delta |
|---|---|---|---|
| exp3_normal_prob | 1.00000 | 1.00000 | +0.00000 |
| exp3_normal_ftr | 0.96083 | 0.96083 | +0.00000 |
| exp4_normal_ftr | 0.96333 | 0.96417 | +0.00083 |
| exp5_mismatch_prob | 0.99667 | 0.99667 | +0.00000 |

At n=1200 pairs/condition and IPR≈0.96, the Monte Carlo standard error on the aggregate
is ≈0.006; the one nonzero delta is ~1/7 of that — indistinguishable from noise at this
sample size. **Reading:** the leak visibly perturbs individual run trajectories (per the
golden mismatches above) but was not shown to move the aggregate IPR metric that
exp3/4/5 actually report, at least not by more than this batch could resolve. This was
judged sufficient evidence to fix forward rather than run a larger confirmatory batch;
raw per-condition results in `leak_impact_fixed.json` / `leak_impact_leaky.json`.

**Scope.**
- **Exposed:** exp3 / exp4 / exp5 (`N_RUNS=1000`, `N_JOBS=100` → ~10 reps/worker).
- **Not exposed:** compare_crr, exp1, exp2 (`n_runs == n_jobs` → 1 rep/worker, fresh
  process each — no leak possible regardless of the singleton).

**Status.** Fixed in the reference pipeline (commit after this file was last edited —
see `git log -- sim/pairwise_stochastic/get_ipr_stochastic_env.py`). This is a
**deliberate exception** to reproduce-first: the pre-fix behavior was an accidental
process-state leak, not an intentional part of the algorithm, so the fix is treated as a
reference correction rather than an algorithmic change the refactor must preserve. The
16 golden baselines were re-captured against the fixed reference and re-verified for
subprocess reproducibility (2 independent runs, all MATCH) before Phase 1 resumed. If
`refactor_fp.md`'s exp3/4/5 numbers are ever regenerated for the paper, they should be
regenerated with this fix in place.
