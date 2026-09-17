# TBI-TRACT step 02: adjudicate cohort and candidate predictors
#
# Constructs the candidate adult TBI cohort and adjudicates predictor availability
# for the intended post-evaluation prediction time. Run from the repository root.

implementation <- file.path("analytic_pipeline", "implementation", "02_adjudicate_cohort_and_predictors.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
