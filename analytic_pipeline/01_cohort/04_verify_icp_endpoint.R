# TBI-TRACT step 04: verify the invasive ICP endpoint
#
# Confirms that the frozen endpoint is EVD or intraparenchymal ICP bolt only.
# Brain-tissue oxygen and jugular venous-bulb monitoring are not included.

implementation <- file.path("analytic_pipeline", "implementation", "04_verify_icp_endpoint.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
