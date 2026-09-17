# TBI-TRACT step 05: rolling-origin structural hyperparameter tuning
#
# Evaluates the prespecified XGBoost structural grid across temporally ordered
# development/evaluation folds. This step supports model development; it is not
# independent external validation.

implementation <- file.path("analytic_pipeline", "implementation", "05_temporal_hyperparameter_tuning.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
