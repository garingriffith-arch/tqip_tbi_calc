# TBI-TRACT step 16: compare the penalized-regression baseline
#
# Fits Ridge comparators using the same 2020–2023 development cohort, 2024
# evaluation cohort, endpoint definitions, and endpoint-specific predictor policy
# as the primary XGBoost models.

implementation <- file.path("analytic_pipeline", "implementation", "16_compare_ridge_baseline.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
