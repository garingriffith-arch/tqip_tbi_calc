# Code and data availability

This repository provides the analysis code used for cohort construction, endpoint and predictor definition, temporal model development, sensitivity analyses, deployment fitting, manuscript metrics, reproducibility checks, and figure generation for TBI-TRACT.

The ACS TQIP/TQP Participant Use Files are not redistributed. Reproduction requires independent access to the applicable 2020–2024 source files under the relevant ACS data-use terms. No patient-level TQIP/TQP records are included in this repository.

Two archival limitations are documented explicitly:

1. The original R source file for the classification predictor-family ablation was not retained in the archived analysis project. The complete manuscript-facing output set from that analysis is included under `analytic_pipeline/03_sensitivity/predictor_ablation/`; no reconstructed source code is presented.
2. The exact script used to assemble the final Word table/eTable document was not retained. The final table/eTable file is included under `analytic_pipeline/05_reporting/` as a reference output.

These archival limitations do not affect the deployed model objects, endpoint definitions, reported manuscript metrics, or the figure-generation workflow.
