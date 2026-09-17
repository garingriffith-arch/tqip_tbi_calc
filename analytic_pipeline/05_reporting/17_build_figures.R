# TBI-TRACT step 17: generate manuscript figures
#
# Generates the main and supplementary figure package from finalized reporting
# outputs. Figure numbering and cohort-flow presentation match the submission
# package.

implementation <- file.path("analytic_pipeline", "implementation", "17_build_figures.R.gz")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
con <- gzfile(implementation, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
