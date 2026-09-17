# TBI-TRACT deployment model fitting
#
# Fits the final deployment models after the endpoint definitions, predictor
# policies, and structural hyperparameters have been fixed. The complete source
# is stored in the adjacent compressed R file and is executed by this entrypoint.
# Internal 5-fold cross-validation in this step is used only to select boosting
# rounds for deployment; it is not reported as manuscript validation.

implementation <- file.path(
  "analytic_pipeline",
  "04_deployment",
  "11_fit_deployment_models_impl.R.gz"
)

if (!file.exists(implementation)) {
  stop("Missing deployment implementation: ", implementation, call. = FALSE)
}

con <- gzfile(implementation, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
