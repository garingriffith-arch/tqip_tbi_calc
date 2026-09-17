# =============================================================================
# 26_run_TBI_TRACT_deployment_temporal_CV_GPU_v2.R
# PUBLIC ENTRY POINT — FINAL CLEANED DEPLOYMENT PIPELINE
#
# The exact cleaned implementation is stored alongside this entry point as a
# gzip-compressed R source file. It preserves the final deployment analysis while
# using manuscript-concordant forward-temporal structural-selection terminology.
# Run from the TBI-TRACT project root.
# =============================================================================

impl_candidates <- c(
  file.path(
    "analytic_pipeline",
    "deployment",
    "26_run_TBI_TRACT_deployment_temporal_CV_GPU_v2_impl.R.gz"
  ),
  "26_run_TBI_TRACT_deployment_temporal_CV_GPU_v2_impl.R.gz"
)
impl_file <- impl_candidates[file.exists(impl_candidates)][1L]
if (length(impl_file) == 0L || is.na(impl_file)) {
  stop("Could not find deployment implementation source.", call. = FALSE)
}
con <- gzfile(impl_file, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
