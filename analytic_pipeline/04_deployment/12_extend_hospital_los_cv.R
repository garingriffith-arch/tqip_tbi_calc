# TBI-TRACT step 12: extend hospital-LOS boosting-round selection
#
# Extends the internal deployment cross-validation search for the hospital-LOS
# quantile model after the initial search reached its boosting-round cap. This
# changes deployment-round selection only; manuscript temporal performance is not refit.

implementation <- file.path("analytic_pipeline", "implementation", "12_extend_hospital_los_cv.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
