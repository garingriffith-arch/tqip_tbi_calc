# TBI-TRACT step 06: select the final classification predictor policy
#
# Compares clinically motivated predictor sets across the rolling-origin temporal
# folds and selects the endpoint-specific classification specification used by
# the final deployment models.

implementation <- file.path("analytic_pipeline", "implementation", "06_select_classification_predictors.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
