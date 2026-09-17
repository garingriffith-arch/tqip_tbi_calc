# =============================================================================
# 27_build_TBI_TRACT_manuscript_output_data.R
# PUBLIC ENTRY POINT — FINAL MANUSCRIPT OUTPUT PIPELINE
#
# The exact cleaned implementation is stored alongside this entry point as a
# gzip-compressed R source file. It preserves all legacy output filenames needed
# downstream while describing manuscript-facing predictions as rolling-origin
# forward-temporal evaluation predictions rather than conventional OOF estimates.
# Run from the TBI-TRACT project root.
# =============================================================================

impl_candidates <- c(
  file.path(
    "analytic_pipeline",
    "reporting",
    "27_build_TBI_TRACT_manuscript_output_data_impl.R.gz"
  ),
  "27_build_TBI_TRACT_manuscript_output_data_impl.R.gz"
)
impl_file <- impl_candidates[file.exists(impl_candidates)][1L]
if (length(impl_file) == 0L || is.na(impl_file)) {
  stop("Could not find manuscript-output implementation source.", call. = FALSE)
}
con <- gzfile(impl_file, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
