# =============================================================================
# 26c_reconcile_TBI_TRACT_hospital_LOS_manifest.R
#
# PURPOSE
#   Repair a bookkeeping-only name-collision from script 26b.
#
# ISSUE
#   The extended Hospital LOS CV selected 4,341 rounds in the current run, and
#   the rounds table / encoder were updated. However, this statement in 26b:
#
#       selected_rounds = selected_rounds
#
#   can resolve the RHS to the existing data.table column rather than the local
#   scalar, leaving the deployment manifest at the old 2,500-round value.
#
# ACTION
#   - Require agreement between the rounds table and saved Hospital LOS encoder.
#   - Update the deployment manifest to that verified round count.
#   - If the reporting/QC reproducibility manifest exists, patch it too.
#
# IMPORTANT
#   NO model is refit.
#   NO manuscript-facing prediction is changed.
# =============================================================================

rm(list = ls())
gc()

if (!requireNamespace("data.table", quietly = TRUE)) {
  stop("Package 'data.table' is required.", call. = FALSE)
}

suppressPackageStartupMessages({
  library(data.table)
})

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)

config_file <- config_candidates[file.exists(config_candidates)][1L]

if (length(config_file) == 0L || is.na(config_file)) {
  stop("Could not find R/00_config.R.", call. = FALSE)
}

source(config_file)

deploy_dir <- file.path(
  output_dir,
  "TBI_TRACT_FINAL_DEPLOYMENT_CV"
)

report_dir <- file.path(
  output_dir,
  "TBI_TRACT_REPORTING_QC_COMPLETION"
)

rounds_file <- file.path(
  deploy_dir,
  "03_FULL_DEVELOPMENT_5FOLD_CV_ROUNDS.csv"
)

manifest_file <- file.path(
  deploy_dir,
  "04_FINAL_DEPLOYMENT_MODEL_MANIFEST.csv"
)

encoder_file <- file.path(
  deploy_dir,
  "encoders",
  "hospital_los_encoder.rds"
)

needed <- c(
  rounds_file,
  manifest_file,
  encoder_file
)

if (!all(file.exists(needed))) {
  stop(
    "Missing prerequisite file(s):\n",
    paste(
      needed[
        !file.exists(needed)
      ],
      collapse = "\n"
    ),
    call. = FALSE
  )
}

rounds_table <- fread(rounds_file)
manifest <- fread(manifest_file)
encoder <- readRDS(encoder_file)

round_row <- rounds_table[
  endpoint_id == "hospital_los"
]

manifest_row <- manifest[
  endpoint_id == "hospital_los"
]

if (nrow(round_row) != 1L) {
  stop(
    "Could not uniquely identify hospital_los in rounds table.",
    call. = FALSE
  )
}

if (nrow(manifest_row) != 1L) {
  stop(
    "Could not uniquely identify hospital_los in deployment manifest.",
    call. = FALSE
  )
}

rounds_verified <- as.integer(
  round_row$selected_rounds_5fold_cv[1L]
)

encoder_rounds <- as.integer(
  encoder$selected_rounds
)

if (
  is.na(rounds_verified) ||
  is.na(encoder_rounds)
) {
  stop(
    "Hospital LOS selected-round count is missing.",
    call. = FALSE
  )
}

if (rounds_verified != encoder_rounds) {
  stop(
    "Hospital LOS round-count mismatch: rounds table = ",
    rounds_verified,
    "; encoder = ",
    encoder_rounds,
    ". Do not patch manifest until reconciled.",
    call. = FALSE
  )
}

old_manifest_rounds <- as.integer(
  manifest_row$selected_rounds[1L]
)

# Use a differently named scalar to avoid data.table column-name capture.
verified_selected_rounds <- rounds_verified
verified_model_path <- file.path(
  deploy_dir,
  "models",
  "hospital_los.json"
)
verified_encoder_path <- encoder_file

if (!file.exists(verified_model_path)) {
  stop(
    "Hospital LOS deployment model JSON is missing: ",
    verified_model_path,
    call. = FALSE
  )
}

manifest[
  endpoint_id == "hospital_los",
  `:=`(
    selected_rounds =
      verified_selected_rounds,
    model_path =
      verified_model_path,
    encoder_path =
      verified_encoder_path
  )
]

fwrite(
  manifest,
  manifest_file
)

# Patch downstream reporting/QC manifest if it already exists.
report_manifest_file <- file.path(
  report_dir,
  "09_MODEL_REPRODUCIBILITY_MANIFEST.csv"
)

if (file.exists(report_manifest_file)) {
  report_manifest <- fread(
    report_manifest_file
  )

  if (
    sum(
      report_manifest$endpoint_id ==
        "hospital_los"
    ) != 1L
  ) {
    stop(
      "Could not uniquely identify hospital_los in reporting manifest.",
      call. = FALSE
    )
  }

  report_manifest[
    endpoint_id == "hospital_los",
    `:=`(
      selected_rounds =
        verified_selected_rounds,
      selected_rounds_internal_5fold_cv =
        verified_selected_rounds,
      model_path =
        verified_model_path,
      encoder_path =
        verified_encoder_path
    )
  ]

  fwrite(
    report_manifest,
    report_manifest_file
  )
}

qc <- data.table(
  endpoint_id = "hospital_los",
  old_manifest_selected_rounds =
    old_manifest_rounds,
  verified_rounds_table =
    rounds_verified,
  verified_encoder_rounds =
    encoder_rounds,
  corrected_manifest_selected_rounds =
    verified_selected_rounds,
  model_exists =
    file.exists(
      verified_model_path
    ),
  encoder_exists =
    file.exists(
      verified_encoder_path
    ),
  status =
    "PASS"
)

qc_file <- file.path(
  report_dir,
  "21_HOSPITAL_LOS_MANIFEST_RECONCILIATION.csv"
)

dir.create(
  report_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

fwrite(
  qc,
  qc_file
)

cat(
  "\nHospital LOS manifest reconciliation: PASS\n",
  "Old manifest rounds: ",
  old_manifest_rounds,
  "\nVerified extended rounds: ",
  verified_selected_rounds,
  "\nNo model was refit.\n",
  "QC: ",
  qc_file,
  "\n",
  sep = ""
)
