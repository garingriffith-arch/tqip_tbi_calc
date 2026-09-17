# TBI-TRACT step 03: freeze the final analysis dataset
#
# Reconciles final cohort flow, derives manuscript endpoint definitions, audits
# selection/case-mix differences, and saves the frozen analysis dataset used by
# subsequent model-development scripts.

implementation <- file.path("analytic_pipeline", "implementation", "03_freeze_analysis_dataset.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
