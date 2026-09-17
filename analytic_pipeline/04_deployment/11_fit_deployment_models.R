# TBI-TRACT step 11: fit final deployment models
#
# Fits the fixed deployment specifications on the complete 2020–2024 cohort.
# Internal cross-validation in this step is used only to select boosting rounds
# for deployment and is not reported as independent validation.

implementation <- file.path("analytic_pipeline", "implementation", "11_fit_deployment_models.R.gz")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
con <- gzfile(implementation, open = "rt")
on.exit(close(con), add = TRUE)
source(con, local = FALSE, echo = FALSE)
