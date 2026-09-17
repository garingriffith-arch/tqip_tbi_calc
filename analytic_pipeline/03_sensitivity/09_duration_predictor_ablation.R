# TBI-TRACT step 09: duration-model predictor-family ablation
#
# Reproduces the continuous-duration ablation analysis. The source for the
# preceding classification ablation was not retained, so its preserved output
# set is staged automatically into the local analysis output directory when
# needed. No reconstructed classification-ablation source code is used.

config_candidates <- c(file.path(getwd(), "R", "00_config.R"), "R/00_config.R")
config_file <- config_candidates[file.exists(config_candidates)][1L]
if (length(config_file) == 0L || is.na(config_file)) {
  stop("Could not find R/00_config.R.", call. = FALSE)
}
source(config_file)

reference_dir <- file.path(
  "analytic_pipeline", "03_sensitivity", "predictor_ablation", "results"
)
stress_dir <- file.path(output_dir, "TBI_TRACT_PREDICTOR_STRESS_TEST")
required_reference <- file.path(
  stress_dir,
  "04_SELECTED_REFERENCE_HYPERPARAMETERS_BY_FOLD_ENDPOINT.csv"
)

if (!file.exists(required_reference)) {
  if (!dir.exists(reference_dir)) {
    stop("Missing preserved classification-ablation outputs: ", reference_dir, call. = FALSE)
  }
  dir.create(stress_dir, recursive = TRUE, showWarnings = FALSE)
  entries <- list.files(reference_dir, full.names = TRUE, all.files = TRUE, no.. = TRUE)
  copied <- file.copy(entries, stress_dir, recursive = TRUE, overwrite = FALSE)
  if (length(copied) != length(entries) || any(!copied)) {
    stop("Could not stage the complete preserved classification-ablation output set.", call. = FALSE)
  }
}

implementation <- file.path("analytic_pipeline", "implementation", "09_duration_predictor_ablation.R")
if (!file.exists(implementation)) stop("Missing implementation: ", implementation, call. = FALSE)
source(implementation, local = FALSE, echo = FALSE)
