# TBI-TRACT step 13: reconcile hospital-LOS deployment metadata
#
# Verifies agreement between the selected hospital-LOS boosting rounds, saved
# encoder metadata, and deployment manifest. No model is refit in this QC step.

implementation <- file.path("analytic_pipeline", "implementation", "13_reconcile_hospital_los_manifest.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
