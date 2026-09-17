# =============================================================================
# 29_build_TBI_TRACT_final_figures.R
# FINAL MANUSCRIPT FIGURE PIPELINE
#
# This is the public entry point for the exact locked final figure implementation.
# The implementation is stored alongside this file as a gzip-compressed R source
# to preserve the exact final V4 figure code, including the finalized eFigure 1
# cohort-flow geometry and combined Figure 3 corrections.
#
# Run from the TBI-TRACT project root, consistent with the rest of the pipeline.
# =============================================================================

impl_candidates <- c(
  file.path(
    "analytic_pipeline",
    "reporting",
    "29_build_TBI_TRACT_final_figures_impl.R.gz"
  ),
  "29_build_TBI_TRACT_final_figures_impl.R.gz"
)

impl_file <- impl_candidates[file.exists(impl_candidates)][1L]

if (length(impl_file) == 0L || is.na(impl_file)) {
  stop(
    "Could not find 29_build_TBI_TRACT_final_figures_impl.R.gz.",
    call. = FALSE
  )
}

con <- gzfile(impl_file, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
