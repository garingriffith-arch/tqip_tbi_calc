# TBI-TRACT step 07: select the final duration-model predictor policy
#
# Compares candidate clinical predictor sets for hospital LOS, ICU LOS, and
# ventilator-duration quantile models using the same rolling-origin design.

implementation <- file.path("analytic_pipeline", "implementation", "07_select_duration_predictors.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
