# TBI-TRACT

**Trauma Resource and Acute Care Trajectory Calculator for Adults With Traumatic Brain Injury**

This repository contains the deployed TBI-TRACT Shiny application, locked XGBoost model/encoder objects, and the manuscript-facing analytic code package for the 2020–2024 ACS TQIP/TQP study. Patient-level ACS data are not included.

## Clinical model at a glance

- **Population:** 755,880 direct-presenting adults aged 18–89 years with traumatic intracranial injury and an observable index-hospital trajectory.
- **Prediction time:** after the initial trauma-center evaluation and diagnostic workup.
- **Temporal evaluation:** rolling-origin evaluation in 2022, 2023, and 2024; the 2024 cohort contained 151,874 patients.
- **Outputs:** discharge disposition; hospital LOS trajectory; ICU trajectory; ventilation trajectory; continuous hospital/ICU/ventilator duration; EVD or intraparenchymal ICP bolt utilization; and craniotomy/craniectomy.
- **Validation status:** internally evaluated temporally; independent external/prospective validation remains pending. The tool supports counseling/resource planning and is not a treatment recommendation engine.

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

2024 median-prediction MAE was 4.54 days for hospital LOS, 3.41 days for ICU LOS conditional on ICU use, and 4.79 days for ventilator duration conditional on ventilation.

## App

`app.R` loads the locked JSON models under `data/models/` and preprocessing metadata under `data/encoders/`. The interface enforces the study age range and bounded physiologic inputs, collapses duplicate unknown/other encoding sentinels into clinician-readable choices, and performs server-side range validation before prediction.

The final predictor policy excludes helmet and respiratory-assistance variables. Race, ethnicity, and payer are used only by disposition and categorical hospital-LOS models as social/health-system context.

## Analytic code

The manuscript-facing reproducibility code is maintained in the repository under the analytic-pipeline files/directories. It includes:

- raw-field/disposition repair and cohort/predictor adjudication code;
- final methods-completion and EVD/BOLT endpoint construction/QC;
- temporal model-development and sensitivity scripts;
- final deployment fitting and HLOS metadata reconciliation;
- the canonical Ridge comparator;
- manuscript metric/QC and final figure-generation code; and
- the exact final table/eTable reference artifact when distributed with the release package.

The original source file for the classification predictor-family stress test (script 18) was not recoverable from the retained working export. Its manuscript-facing summary outputs are retained separately, and this limitation is documented rather than reconstructing or fabricating historical code.

## Data availability

The ACS TQIP/TQP Participant Use Files are not redistributed. Reproduction requires independent access to the applicable 2020–2024 source files and compliance with ACS data-use terms. The repository contains no patient-level TQIP/TQP records.

## Reproducibility notes

Manuscript-facing performance comes from rolling-origin forward-temporal evaluation, not random cross-validation. After architecture/predictor policies were locked, final deployment models were fit on 2020–2024; full-development cross-validation was used only to choose boosting rounds for deployment. The final hospital-LOS duration model uses 4,341 boosting rounds.

The invasive ICP endpoint is **EVD or intraparenchymal ICP bolt only**. Brain-tissue oxygen and jugular venous-bulb monitoring are not part of the modeled endpoint.
