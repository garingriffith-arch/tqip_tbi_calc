# TBI-TRACT manuscript figure generation
#
# Generates the main and supplementary figures from finalized reporting outputs.
# The complete figure source is stored in the adjacent compressed R file and is
# executed by this entrypoint.

implementation <- file.path(
  "analytic_pipeline",
  "05_reporting",
  "17_build_figures_impl.R.gz"
)

if (!file.exists(implementation)) {
  stop("Missing figure-generation implementation: ", implementation, call. = FALSE)
}

con <- gzfile(implementation, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
