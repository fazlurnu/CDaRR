# Refactor Progress

Mirrors `REFACTOR_TODO.md`. Updated after each human approval.

Status: `[x]` approved · `[~]` in progress · `[ ]` pending

---

## PHASE 0 — Test suite modernization (§10)

### §10.4 Shared infrastructure

- [x] **P0.1** `requirements-dev.txt` — created; contains `pytest`
- [x] **P0.2** `pytest.ini` — created; `testpaths = test`, `addopts = -q`
- [x] **P0.3** `conftest.py` — created; `--plot` option, `bluesky_init` autouse session fixture, Agg backend
- [x] **P0.4** DELETE `sim_models/test_sim_models.py` — deleted (was empty)

### §10.5 BROKEN — fix stale imports / moved API

- [ ] **P0.5** `test/test_stochastic_sim_single_job_deterministic.py`
- [ ] **P0.6** `test/test_stochastic_sim_single_job_probabilistic.py`
- [ ] **P0.7** `test/test_stochastic_sim_single_job_trace_resumenav.py`

### §10.5 CLEAN-UP — guards / plots / asserts

- [ ] **P0.8** `test/test_stochastic_sim_single_simple.py`
- [ ] **P0.9** `test/test_stochastic_sim_multi_job.py`
- [ ] **P0.10** `test/test_stochastic_sim_multi_job_dist.py`
- [ ] **P0.11** `test/test_stochastic_sim_single_job.py`
- [ ] **P0.12** `test/test_stochastic_sim_forloop_sanitycheck.py`
- [ ] **P0.13** `test/test_cr_mvp.py`
- [ ] **P0.14** `test/test_cd_statebased.py`

### §10.5 Ensure pytest compatibility

- [ ] **P0.15** `test/test_adsl_module.py`
- [ ] **P0.16** `test/test_noise_model.py`
- [ ] **P0.17** `test/test_reception_model.py`

---

## STEP 0 — Golden master baseline (§5)

- [ ] **S0.1** `tests/golden/capture_golden.py`

---

## STEP 1 — `cdarr/noise.py` (§6 STEP 1)

- [ ] **S1.1** `cdarr/__init__.py`
- [ ] **S1.2** `cdarr/protocols.py`
- [ ] **S1.3** `cdarr/noise.py`
- [ ] **S1.4** `test/test_equiv_noise.py`

---

## STEP 2 — `cdarr/sensing.py` :: Observation (§6 STEP 2)

- [ ] **S2.1** `cdarr/sensing.py`

---

## STEP 3 — `cdarr/sensing.py` :: sensors (§6 STEP 3)

- [ ] **S3.1** `cdarr/sensing.py` *(modify — add sensors)*
- [ ] **S3.2** `test/test_equiv_sensor.py`

---

## STEP 4 — `cdarr/recovery.py` (§6 STEP 4)

- [ ] **S4.1** `cdarr/recovery.py`
- [ ] **S4.2** `test/test_equiv_recovery.py`

---

## STEP 5 — `cdarr/scenarios.py` (§6 STEP 5)

- [ ] **S5.1** `cdarr/scenarios.py`
- [ ] **S5.2** `test/test_equiv_scenario.py`

---

## STEP 6 — `cdarr/experiment.py` (§6 STEP 6)

- [ ] **S6.1** `cdarr/experiment.py`
- [ ] **S6.2** `cdarr/__init__.py` *(modify — add exports)*
- [ ] **S6.3** `test/test_equiv_experiment.py`

---

## STEP 7 — `cdarr/registry.py` + `from_config` (§6 STEP 7)

- [ ] **S7.1** `cdarr/registry.py`
- [ ] **S7.2** `cdarr/experiment.py` *(modify — add `from_config`)*
- [ ] **S7.3** `test/test_equiv_from_config.py`

---

## STEP 8 — Deprecation shims + repoint consumers (§6 STEP 8)

- [ ] **S8.1** `sim/pairwise_stochastic/get_ipr_stochastic_env.py` *(shims)*
- [ ] **S8.2** `sim/pairwise_stochastic/run_multiple_jobs.py` *(verify/minor modify)*
- [ ] **S8.3** `test/test_equiv_shims.py`
