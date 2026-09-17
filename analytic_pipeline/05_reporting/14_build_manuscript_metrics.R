# TBI-TRACT step 14: build manuscript performance and calibration metrics
#
# Builds manuscript-facing performance and calibration outputs from the final
# rolling-origin prediction caches. Reported estimates are internal temporal
# evaluation results, not random-CV out-of-fold predictions.

implementation <- file.path("analytic_pipeline", "implementation", "14_build_manuscript_metrics.R.gz")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
con <- gzfile(implementation, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
