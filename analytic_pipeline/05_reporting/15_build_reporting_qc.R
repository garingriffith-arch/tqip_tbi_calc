# TBI-TRACT step 15: build reporting and reproducibility QC outputs
#
# Produces complete cohort flow, outcome observation counts, predictor missingness,
# deployment reproducibility metadata, model-file checksums, and reporting QC.

implementation <- file.path("analytic_pipeline", "implementation", "15_build_reporting_qc.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
