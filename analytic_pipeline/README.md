# TBI-TRACT analytic pipeline

This directory contains the manuscript-facing analysis workflow. Files are numbered in the order a reviewer or reproducing analyst should read or run them; the numbering no longer reflects historical workstation script numbers used during development.

## Local configuration

Copy `R/00_config.example.R` to `R/00_config.R` and edit the paths for your environment. Patient-level ACS TQIP/TQP data are not distributed with this repository.

Several model-development steps use Python XGBoost through `reticulate` and expect a CUDA-capable environment named `tbi-tract-xgb-gpu`. CPU-only reproduction would require adapting the compute settings without changing the analytic definitions.

## Numbered entrypoints and implementation sources

The files in `01_cohort/` through `05_reporting/` are stable, reviewer-facing entrypoints. Each states the purpose of the step in manuscript terminology and executes the corresponding source in `implementation/`.

`implementation/` preserves the executable analysis code used during development. Some implementation files retain historical internal object names or output-directory names; these are preserved for provenance rather than cosmetically rewriting tested analysis code. They are not alternative model specifications.

## Primary execution order

| Step | File | Purpose |
|---:|---|---|
| 01 | `01_cohort/01_recover_registry_fields.R` | Recover source fields required downstream and repair final disposition where hospital disposition is absent. |
| 02 | `01_cohort/02_adjudicate_cohort_and_predictors.R` | Construct the candidate adult TBI cohort and adjudicate candidate predictors. |
| 03 | `01_cohort/03_freeze_analysis_dataset.R` | Reconcile the final cohort, define manuscript endpoints, quantify selection/case-mix differences, and save the frozen analysis dataset. |
| 04 | `01_cohort/04_verify_icp_endpoint.R` | Confirm that the invasive ICP endpoint is exactly EVD or intraparenchymal bolt. |
| 05 | `02_model_development/05_temporal_hyperparameter_tuning.R` | Evaluate the prespecified XGBoost structural grid using rolling-origin temporal folds. |
| 06 | `02_model_development/06_select_classification_predictors.R` | Select the final classification predictor policy, including endpoint-specific race/ethnicity/payer context. |
| 07 | `02_model_development/07_select_duration_predictors.R` | Select the final predictor policy for continuous duration models. |
| 08 | `03_sensitivity/08_covid_period_sensitivity.R` | Evaluate temporal/COVID-period sensitivity. |
| 09 | `03_sensitivity/09_duration_predictor_ablation.R` | Evaluate duration-model sensitivity to predictor-family removal. |
| 10 | `03_sensitivity/10_final_robustness_analyses.R` | Generate final subgroup, airway-state, survivor-only, and composite-outcome sensitivity analyses. |
| 11 | `04_deployment/11_fit_deployment_models.R` | Fit final deployment models after model structure and predictor policies are fixed. |
| 12 | `04_deployment/12_extend_hospital_los_cv.R` | Extend boosting-round selection for the hospital-LOS duration model after the initial search reached its cap. |
| 13 | `04_deployment/13_reconcile_hospital_los_manifest.R` | Verify and reconcile the final 4,341-round hospital-LOS deployment metadata. |
| 14 | `05_reporting/14_build_manuscript_metrics.R` | Build manuscript-facing temporal performance and calibration outputs from the final prediction caches. |
| 15 | `05_reporting/15_build_reporting_qc.R` | Produce cohort-flow, missingness, reproducibility, and reporting QC outputs. |
| 16 | `05_reporting/16_compare_ridge_baseline.R` | Fit the penalized-regression comparator using the same development/evaluation split and endpoint-specific predictor policies. |
| 17 | `05_reporting/17_build_figures.R` | Generate the manuscript figure package from finalized reporting artifacts. |

## Classification predictor-family ablation

The original source file for the classification predictor-family ablation was not retained in the archived analysis project. The complete output set used by the manuscript is provided in `03_sensitivity/predictor_ablation/results/`. Step 09 automatically stages the selected reference-architecture file it requires when that output is not already present locally.

No reconstructed replacement source code is presented.

## Development archive

`archive/development_history/` contains superseded model-development experiments retained for provenance. These files are **not** part of the primary execution sequence and should not be interpreted as alternative final specifications.

## Reporting artifacts

`05_reporting/` also contains:

- `model_artifact_checksums.csv` — checksums for the final model and encoder files;
- `TBI_TRACT_tables_and_etables.docx` — final table/eTable reference output.

The exact historical script that assembled the Word table document was not retained.

## Interpretation of temporal performance

The 2022–2024 performance estimates are internal rolling-origin temporal evaluation results. Because later temporal results informed final predictor adjudication, they are not independent external validation. External validation requires application of the fixed deployment models to an independent health system or population.
