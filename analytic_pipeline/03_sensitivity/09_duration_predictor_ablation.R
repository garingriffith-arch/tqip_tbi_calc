# TBI-TRACT step 09: duration-model predictor-family ablation
#
# Reproduces the continuous-duration ablation analysis. The original source for
# the preceding classification ablation was not retained; its required selected
# architecture output is preserved in this repository and is staged automatically
# if it is not already present in the local analysis output directory.

config_candidates <- c(file.path(getwd(), "R", "00_config.R"), "R/00_config.R")
config_file <- config_candidates[file.exists(config_candidates)][1L]
if (length(config_file) == 0L || is.na(config_file)) {
  stop("Could not find R/00_config.R.", call. = FALSE)
}
source(config_file)

reference_file <- file.path(
  "analytic_pipeline", "03_sensitivity", "predictor_ablation", "results",
  "04_SELECTED_REFERENCE_HYPERPARAMETERS_BY_FOLD_ENDPOINT.csv"
)
stress_dir <- file.path(output_dir, "TBI_TRACT_PREDICTOR_STRESS_TEST")
expected_file <- file.path(stress_dir, basename(reference_file))

if (!file.exists(expected_file)) {
  if (!file.exists(reference_file)) stop("Missing preserved ablation output: ", reference_file, call. = FALSE)
  dir.create(stress_dir, recursive = TRUE, showWarnings = FALSE)
  ok <- file.copy(reference_file, expected_file, overwrite = FALSE)
  if (!isTRUE(ok)) stop("Could not stage preserved ablation output.", call. = FALSE)
}

implementation <- file.path("analytic_pipeline", "implementation", "09_duration_predictor_ablation.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
