# TBI-TRACT

**Trauma Resource and Acute Care Trajectory Calculator for Adults With Traumatic Brain Injury**

TBI-TRACT is a multivariable prediction framework for early inpatient trajectory and resource use after traumatic brain injury. This repository contains the Shiny application, final deployment model objects and encoders, and the analysis code supporting the accompanying manuscript.

## Study snapshot

- **Data source:** ACS TQIP/TQP Participant Use Files, 2020–2024
- **Analytic cohort:** 755,880 direct-presenting adults aged 18–89 years with traumatic intracranial injury and an observable index-hospital trajectory
- **Prediction time:** after the initial trauma-center evaluation and diagnostic workup
- **Internal temporal evaluation:** rolling-origin evaluation cohorts from 2022 through 2024; the 2024 cohort contained 151,874 patients
- **Outputs:** discharge disposition; hospital length-of-stay trajectory; ICU trajectory; mechanical-ventilation trajectory; continuous hospital, ICU, and ventilator duration; EVD or intraparenchymal ICP bolt utilization; and craniotomy/craniectomy
- **Validation status:** independent external and prospective validation remain pending

### Headline 2024 discrimination

| Output | AUROC |
|---|---:|
| Post-acute facility discharge | 0.785 |
| Death/hospice discharge | 0.939 |
| Any ICU use | 0.853 |
| ICU ≥8 days | 0.885 |
| Any mechanical ventilation | 0.940 |
| Ventilation ≥8 days | 0.916 |
| Hospital LOS ≥28 days | 0.875 |
| EVD or intraparenchymal ICP bolt | 0.925 |
| Craniotomy/craniectomy | 0.894 |

For 2024, median-prediction mean absolute error was 4.54 days for hospital LOS, 3.41 days for ICU LOS conditional on ICU use, and 4.79 days for ventilator duration conditional on ventilation.

## Repository structure

- `app.R` — Shiny entry point and clinician-facing input presentation
- `app_core.R` — application prediction logic, model loading, validation, and output rendering
- `data/models/` — final XGBoost deployment models
- `data/encoders/` — endpoint-specific preprocessing metadata and encoder objects
- `analytic_pipeline/` — numbered manuscript analysis workflow and reproducibility documentation
- `www/` — application assets

The analysis workflow is documented in [`analytic_pipeline/README.md`](analytic_pipeline/README.md). The numbered files there are ordered by execution rather than by the historical workstation script numbers used during model development.

## Reproducibility

The manuscript reports **internal rolling-origin temporal evaluation**, not random cross-validation and not independent external validation. Structural model choices were assessed across temporally ordered development/evaluation folds. After predictor and endpoint specifications were fixed, deployment models were fit on the full 2020–2024 cohort; internal cross-validation on the full development cohort was used only to choose the number of boosting rounds for deployment.

The invasive ICP endpoint is defined as **EVD or intraparenchymal ICP bolt**. Brain-tissue oxygen and jugular venous-bulb monitoring are not included in that endpoint.

Model-file checksums are provided in `analytic_pipeline/05_reporting/model_artifact_checksums.csv`.

## Data availability

The ACS TQIP/TQP Participant Use Files are not redistributed. Reproduction requires independent access to the applicable 2020–2024 source files and compliance with ACS data-use terms. This repository contains no patient-level TQIP/TQP data.

## Archival limitations

One historical source file used for the classification predictor-family ablation was not retained in the archived analysis project. The complete manuscript-facing output set from that analysis is provided under `analytic_pipeline/03_sensitivity/predictor_ablation/`; no reconstructed source code is presented. The exact table-assembly script was also not retained; the final table/eTable document is included as a reference output.

The prediction tool is intended to support counseling and anticipatory resource planning. It is not a stand-alone basis for treatment decisions.
