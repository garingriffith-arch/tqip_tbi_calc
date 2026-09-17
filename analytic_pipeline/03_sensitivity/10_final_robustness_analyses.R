# TBI-TRACT step 10: final robustness analyses
#
# Generates the final prediction cache and manuscript-reported robustness analyses,
# including subgroup performance, airway-state sensitivity, survivor-only analyses,
# and composite severe-outcome sensitivities.

implementation <- file.path("analytic_pipeline", "implementation", "10_final_robustness_analyses.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
