# TBI-TRACT step 01: recover registry fields and repair final disposition
#
# Recovers source fields required downstream and repairs final disposition when
# hospital discharge disposition is absent. Run from the repository root after
# configuring R/00_config.R.

implementation <- file.path("analytic_pipeline", "implementation", "01_recover_registry_fields.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
