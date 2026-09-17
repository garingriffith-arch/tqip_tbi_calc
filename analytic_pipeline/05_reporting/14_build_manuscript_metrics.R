# TBI-TRACT manuscript metric generation
#
# Builds the manuscript-facing temporal performance and calibration outputs from
# the final rolling-origin prediction caches. The complete source is stored in
# the adjacent compressed R file and is executed by this entrypoint.
#
# Reported predictions are rolling-origin forward-temporal evaluation results;
# they are not conventional random-cross-validation out-of-fold predictions.

implementation <- file.path(
  "analytic_pipeline",
  "05_reporting",
  "14_build_manuscript_metrics_impl.R.gz"
)

if (!file.exists(implementation)) {
  stop("Missing manuscript-metric implementation: ", implementation, call. = FALSE)
}

con <- gzfile(implementation, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
