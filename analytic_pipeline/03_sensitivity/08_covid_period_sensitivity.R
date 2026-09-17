# TBI-TRACT step 08: COVID-period temporal sensitivity
#
# Evaluates whether model performance materially differs across pandemic-era and
# later periods using the prespecified temporal analysis framework.

implementation <- file.path("analytic_pipeline", "implementation", "08_covid_period_sensitivity.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
