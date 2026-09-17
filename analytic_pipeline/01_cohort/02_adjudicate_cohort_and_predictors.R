# =============================================================================
# 04_adjudicate_TBI_TRACT_admission_cohort_and_predictors.R
#
# TBI-TRACT PRE-MODEL ADJUDICATION GATE
# ------------------------------------
# This script FITS NO MODEL.
#
# Its purpose is to freeze the methodological logic of the revised TBI-TRACT
# admission-era model BEFORE any final performance is examined.
#
# It audits and adjudicates:
#   1) Direct presentation vs interfacility transfer-in
#   2) Transfer-out / AMA / custody / ED-other trajectory exclusions
#   3) ICD-10-CM intracranial and extracranial injury phenotypes
#   4) PUF_PREEXISTINGCONDITIONS numeric code decoding via each year's
#      PUF_TRAUMA_LOOKUP.csv (rather than guessing code meanings)
#   5) Race / ethnicity / payer role
#   6) Redundant and leakage-prone predictors
#   7) The complete methodological repair register originating from the
#      original TBI-TRACT audit
#
# IMPORTANT
# ---------
# No predictor is retained because it improves AUROC in this script.
# Decisions are based on prespecified clinical/time-zero/deployability rules.
#
# Proposed primary target population:
#   - S06 TBI
#   - recorded age 18-89 years (ACS PUF suppresses AGEYEARS >89)
#   - DIRECT presentation to the index trauma center
#   - complete observable index-center trajectory:
#       exclude AMA
#       exclude transfer to another acute-care hospital
#       exclude ED-other/institutional/custody
#
# This script should be run BEFORE the definitive model-building script.
# =============================================================================

rm(list = ls())
gc()

suppressPackageStartupMessages({
  library(data.table)
})

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

config_candidates <- c(
  file.path(getwd(), "R", "00_config.R"),
  "R/00_config.R"
)

config_file <- config_candidates[file.exists(config_candidates)][1]

if (is.na(config_file) || length(config_file) == 0L) {
  stop(
    "Could not find R/00_config.R. Run from the TBI-TRACT project root.",
    call. = FALSE
  )
}

source(config_file)

audit_years <- 2020:2024

audit_dir <- file.path(
  output_dir,
  "pre_model_adjudication_TBI_TRACT_admission"
)

dir.create(
  audit_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

logical_cores <- parallel::detectCores(logical = TRUE)
if (is.na(logical_cores)) logical_cores <- 32L
threads <- max(1L, min(28L, logical_cores - 4L))
setDTthreads(threads)

cat("\n============================================================\n")
cat("TBI-TRACT ADMISSION MODEL: PRE-MODEL ADJUDICATION GATE\n")
cat("NO MODEL WILL BE FIT\n")
cat("============================================================\n")
cat("Threads: ", threads, "\n", sep = "")
cat("Output:  ", audit_dir, "\n\n", sep = "")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

safe_num <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

normalize_id <- function(x) {
  out <- trimws(as.character(x))
  out <- gsub("\\.0$", "", out)
  out[out %in% c("", "NA", "NaN", "<NA>", "Inf", "-Inf")] <- NA_character_
  out
}

norm_code <- function(x) {
  toupper(gsub("[^A-Z0-9]", "", as.character(x)))
}

norm_text <- function(x) {
  tolower(trimws(as.character(x)))
}

first_existing <- function(paths) {
  hit <- paths[file.exists(paths)]
  if (length(hit) == 0L) return(NA_character_)
  normalizePath(hit[1], winslash = "/", mustWork = TRUE)
}

header_names <- function(path) {
  tryCatch(
    names(fread(path, nrows = 0, showProgress = FALSE)),
    error = function(e) character()
  )
}

extract_year_from_path <- function(path) {
  m <- regexpr(
    "(?<![0-9])(2020|2021|2022|2023|2024)(?![0-9])",
    path,
    perl = TRUE
  )
  if (m[1] == -1L) return(NA_integer_)
  as.integer(regmatches(path, m))
}

first_matching_col <- function(nms, candidates) {
  nms_upper <- toupper(nms)
  cand_upper <- toupper(candidates)
  idx <- match(cand_upper, nms_upper)
  idx <- idx[!is.na(idx)]
  if (length(idx) == 0L) return(NA_character_)
  nms[idx[1]]
}

is_present_condition_row <- function(
    answer,
    value,
    answer_biu = NULL
) {
  a <- norm_text(answer)
  v <- norm_text(value)

  an <- safe_num(answer)
  vn <- safe_num(value)

  positive <- (
    a %in% c("yes", "y", "true", "present") |
      v %in% c("yes", "y", "true", "present") |
      (!is.na(an) & an == 1) |
      (!is.na(vn) & vn == 1)
  )

  # When harmonization has already reduced the row to an explicit condition
  # code and there is no recognizable answer field, preserve the row rather
  # than declaring it absent. The raw lookup audit will show whether this is
  # occurring.
  no_answer_information <- (
    (is.na(answer) | trimws(as.character(answer)) == "") &
      (is.na(value) | trimws(as.character(value)) == "")
  )

  positive | no_answer_information
}

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

repaired_core_file <- file.path(
  data_out_dir,
  "tbi_tract_patient_core_repaired_2020_2024.rds"
)

if (!file.exists(repaired_core_file)) {
  stop(
    "Repaired project-local patient core not found:\n  ",
    repaired_core_file,
    "\nRun 00d_repair_30min_raw_fields_and_final_disposition.R first.",
    call. = FALSE
  )
}

icd_dx_file <- first_existing(c(
  file.path(
    warehouse_dir,
    "data",
    "harmonized_icd_diagnoses_2007_2024.rds"
  ),
  file.path(
    warehouse_dir,
    "harmonized_icd_diagnoses_2007_2024.rds"
  )
))

if (is.na(icd_dx_file)) {
  stop(
    "Could not locate harmonized ICD diagnosis table.",
    call. = FALSE
  )
}

preexisting_file <- first_existing(c(
  file.path(
    warehouse_dir,
    "data",
    "harmonized_preexisting_conditions_2007_2024.rds"
  ),
  file.path(
    warehouse_dir,
    "harmonized_preexisting_conditions_2007_2024.rds"
  )
))

if (is.na(preexisting_file)) {
  stop(
    "Could not locate harmonized pre-existing conditions table.",
    call. = FALSE
  )
}

source_manifest <- data.table(
  source = c(
    "Repaired patient core",
    "Harmonized ICD diagnoses",
    "Harmonized pre-existing conditions"
  ),
  path = c(
    repaired_core_file,
    icd_dx_file,
    preexisting_file
  )
)

fwrite(
  source_manifest,
  file.path(
    audit_dir,
    "00_source_manifest.csv"
  )
)

# -----------------------------------------------------------------------------
# METHOD 1: Prespecified predictor-adjudication rules
# -----------------------------------------------------------------------------

adjudication_rules <- data.table(
  criterion = c(
    "Prediction-time availability",
    "Temporal integrity",
    "Clinical deployability",
    "Measurement validity",
    "Clinical relevance",
    "Transportability",
    "Nonredundancy",
    "Ethical/use-case suitability",
    "Performance independence"
  ),
  rule = c(
    "Predictor must plausibly be available during the initial trauma-center evaluation / near-admission diagnostic workup.",
    "Predictor must represent baseline patient/injury state rather than a downstream complication, treatment, or resource outcome.",
    "A clinician or deployed calculator must be able to supply the information without access to retrospective registry-only severity summaries.",
    "TQIP variable or derivation must measure the intended construct consistently enough for the proposed use.",
    "There must be a plausible clinical relationship with subsequent resource trajectory.",
    "Avoid features whose meaning depends strongly on local coding/practice when an interpretable alternative exists.",
    "Do not include multiple deterministic/near-deterministic encodings of the same construct merely because all are available.",
    "Do not use social/structural attributes as individual treatment/resource determinants when the intended bedside use could reproduce historical inequities; instead evaluate model performance across those groups.",
    "Predictor inclusion/exclusion is decided before final 2024 performance; no univariable p-value, stepwise selection, or 2024 AUROC screening is permitted."
  )
)

fwrite(
  adjudication_rules,
  file.path(
    audit_dir,
    "01_PRESPECIFIED_PREDICTOR_ADJUDICATION_RULES.csv"
  )
)

# -----------------------------------------------------------------------------
# METHOD 2: Full methodological repair register
# -----------------------------------------------------------------------------

repair_register <- data.table(
  issue = c(
    "Adult cohort restriction",
    "Transfer-in heterogeneity",
    "Transfer-out incomplete trajectory",
    "AMA incomplete trajectory",
    "Custody / ED-other institutional trajectory",
    "ED disposition reconstruction",
    "Helmet derivation",
    "GCS assessment qualifiers",
    "Supplemental oxygen",
    "Initial temperature",
    "Complete-case training",
    "Training/deployment missingness mismatch",
    "Age redundancy",
    "GCS redundancy",
    "Vital-sign threshold redundancy",
    "ICD injury phenotype handling",
    "Pre-existing condition code mapping",
    "Pre-existing condition count redundancy",
    "Constant S06 predictor",
    "AIS/ISS registry severity summaries",
    "Race / ethnicity",
    "Payer",
    "ICP endpoint definition",
    "ICP unknown/N/A handling",
    "ICU endpoint coherence",
    "Ventilation endpoint coherence",
    "Admission respiratory-state leakage for ventilation",
    "Hospital LOS threshold",
    "Death handling for duration outcomes",
    "2024 temporal test contamination",
    "Random 80/20 headline validation",
    "Early stopping / reported validation reuse",
    "Boosting-round truncation",
    "Internal uncertainty / bootstrap terminology",
    "GCS severity subgroup validation",
    "Fairness / subgroup validation",
    "COVID-era sensitivity",
    "Deployment encoder mismatch"
  ),
  locked_action = c(
    "Use S06 TBI with recorded AGEYEARS 18-89; transparently report ACS suppression above age 89.",
    "EXCLUDE interfacility transfer-in patients from the primary cohort; remove transfer status as a predictor.",
    "EXCLUDE patients transferred from the index center to another short-term acute-care hospital.",
    "EXCLUDE AMA.",
    "EXCLUDE ED-other/institutional/custody because trajectory is not clinically comparable/fully observable.",
    "Use repaired final disposition combining ED and hospital disposition fields.",
    "Use recovered Helmet worn / No helmet / Unknown based on all PROTDEV_* fields.",
    "Use GCS E/V/M plus recovered qualifiers; do not silently reinterpret intubated/sedated examinations.",
    "Use recovered initial supplemental oxygen status.",
    "Use recovered initial temperature after range QC.",
    "No primary complete-case analysis; retain incomplete predictor rows.",
    "Numeric NA remains NA for XGBoost; categorical missingness is explicit Unknown; no later arbitrary zero/default fill.",
    "Use continuous age only; do not also enter age groups.",
    "Use GCS eye/verbal/motor plus qualifiers; total/severity reserved for description/subgroups.",
    "Use raw validated physiologic measurements; do not additionally enter deterministic hypotension/hypoxia/tachycardia/RR flags.",
    "Retain clinically interpretable acute injury phenotypes that satisfy time-zero rules; separately adjudicate evolving states such as edema/herniation.",
    "Decode numeric PUF_PREEXISTINGCONDITIONS using yearly PUF_TRAUMA_LOOKUP; never infer labels from numeric codes alone.",
    "Use selected individual PMHx conditions; do not additionally enter a selected-condition count.",
    "Remove dx_any_s06_intracranial after S06 cohort restriction because it is constant.",
    "Exclude AIS, ISS, max-AIS, AIS region counts, and related retrospective registry severity summaries from primary prediction.",
    "Do not use as bedside predictors; retain for descriptive/fairness evaluation where sample size permits.",
    "Do not use as bedside predictor; retain for subgroup/transportability evaluation.",
    "True pressure monitoring = EVD and/or intraparenchymal monitor; exclude PbtO2 and jugular bulb.",
    "Preserve unknown timing/status rather than silently coding unknown as no when endpoint ascertainment is incomplete.",
    "Use a single multiclass trajectory: No ICU / ICU 1-7 d / ICU >=8 d; derive any-ICU probability from the same distribution.",
    "Use a single multiclass trajectory: No ventilation / ventilation 1-7 d / ventilation >=8 d; derive any-ventilation probability from the same distribution.",
    "For ventilation trajectory, exclude direct admission respiratory-assistance/intubation indicators that reveal current ventilation state; audit final endpoint-specific exclusion set.",
    "Primary threshold remains >=28 days based on development-only distribution and clinical interpretability; document rationale. Do not choose threshold from 2024 performance.",
    "Deaths remain in index-hospital duration/resource outcomes; short duration from early death is part of observed trajectory.",
    "2024 is untouched until final evaluation.",
    "Do not use random split as headline validation.",
    "2020-2022 early-stopping training; 2023 iteration selection; refit 2020-2023; 2024 final temporal evaluation.",
    "Use sufficiently high round ceiling with early stopping; previous audit showed natural optima <4000 rounds.",
    "Use bootstrap/CIs for uncertainty with precise terminology; do not call non-refit resampling 'model stability'.",
    "Report performance in mild, moderate, severe, and moderate/severe TBI strata without using severity category as a predictor.",
    "Evaluate calibration/discrimination across race/ethnicity, sex, payer, and other sufficiently sized groups; subgroup attributes need not be predictors.",
    "Prespecified sensitivity: develop/refit the identical model using 2022-2023 only and evaluate on the same 2024 temporal test set.",
    "Freeze the exact encoder/missingness rules used in training and deployment."
  ),
  status = c(
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "AUDIT NOW",
    "AUDIT NOW",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "AUDIT NOW",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED",
    "LOCKED"
  )
)

fwrite(
  repair_register,
  file.path(
    audit_dir,
    "02_COMPLETE_METHODS_REPAIR_REGISTER.csv"
  )
)

# -----------------------------------------------------------------------------
# Load repaired core and identify S06 cohort
# -----------------------------------------------------------------------------

cat("Reading repaired patient core...\n")
core <- as.data.table(
  readRDS(repaired_core_file)
)
core <- core[
  admission_year %in% audit_years
]

for (v in intersect(
  c("inc_key", "harmonized_patient_id"),
  names(core)
)) {
  core[, (v) := normalize_id(get(v))]
}

cat("Reading ICD diagnosis table...\n")
dx <- as.data.table(
  readRDS(icd_dx_file)
)
dx <- dx[
  admission_year %in% audit_years
]

for (v in intersect(
  c("inc_key", "harmonized_patient_id"),
  names(dx)
)) {
  dx[, (v) := normalize_id(get(v))]
}

dx[, code := norm_code(diagnosis_code)]

id_vars <- c(
  "admission_year",
  "inc_key",
  "harmonized_patient_id"
)

s06_ids <- unique(
  dx[
    startsWith(code, "S06"),
    ..id_vars
  ]
)
s06_ids[, s06_tbi := 1L]

pre_n <- nrow(core)
core <- merge(
  core,
  s06_ids,
  by = id_vars,
  all.x = TRUE,
  sort = FALSE
)

if (nrow(core) != pre_n) {
  stop(
    "S06 merge changed patient-core row count.",
    call. = FALSE
  )
}

core[, s06_tbi :=
       fifelse(is.na(s06_tbi), 0L, s06_tbi)]

core[, age_audit :=
       safe_num(age)]

adult_s06 <- core[
  s06_tbi == 1L &
    !is.na(age_audit) &
    age_audit >= 18 &
    age_audit <= 89
]

# -----------------------------------------------------------------------------
# Transfer-in audit
# -----------------------------------------------------------------------------

if (!"transfer_clean" %in% names(adult_s06)) {
  stop(
    "transfer_clean is not present in the repaired core. ",
    "Cannot adjudicate transfer-in exclusion.",
    call. = FALSE
  )
}

adult_s06[, transfer_in_status := {
  x <- trimws(as.character(transfer_clean))
  x_lower <- tolower(x)

  # Explicitly classify missing/blank/unrecognized values as unknown.
  # data.table::fifelse() propagates NA in the test expression, so relying on
  # a nested final "else" would leave true NA values unclassified.
  fcase(
    is.na(x) | x == "" | x_lower %in% c("na", "n/a", "unknown", "unk", "not known", "not recorded"),
      "Transfer-in status unknown",
    x_lower %in% c("yes", "y", "1", "true"),
      "Transfer-in: Yes",
    x_lower %in% c("no", "n", "0", "false"),
      "Direct presentation: No transfer-in",
    default = "Transfer-in status unknown"
  )
}]

transfer_in_qc <- adult_s06[
  ,
  .N,
  by = .(
    admission_year,
    transfer_in_status
  )
][
  order(
    admission_year,
    transfer_in_status
  )
]

transfer_in_qc[
  ,
  pct_within_year :=
    100 * N / sum(N),
  by = admission_year
]

fwrite(
  transfer_in_qc,
  file.path(
    audit_dir,
    "03_TRANSFER_IN_STATUS_BY_YEAR.csv"
  )
)


# Preserve the exact incoming transfer_clean values so any unusual harmonized
# coding can be reviewed rather than silently interpreted.
transfer_raw_qc <- adult_s06[
  ,
  .N,
  by = .(
    admission_year,
    transfer_clean_raw = as.character(transfer_clean),
    transfer_in_status
  )
][
  order(
    admission_year,
    transfer_in_status,
    -N
  )
]

fwrite(
  transfer_raw_qc,
  file.path(
    audit_dir,
    "03B_TRANSFER_RAW_VALUE_AUDIT.csv"
  )
)

# Explicitly inspect the repaired trajectory status before applying transfer-in.
trajectory_qc <- adult_s06[
  ,
  .N,
  by = .(
    admission_year,
    project_trajectory_status_repaired
  )
][
  order(
    admission_year,
    project_trajectory_status_repaired
  )
]

fwrite(
  trajectory_qc,
  file.path(
    audit_dir,
    "04_TRAJECTORY_STATUS_BEFORE_TRANSFER_IN_EXCLUSION.csv"
  )
)

# Candidate primary cohort:
# - direct presentations only
# - known direct presentation
# - repaired retained complete trajectory
primary_candidate <- adult_s06[
  transfer_in_status ==
    "Direct presentation: No transfer-in" &
    project_trajectory_status_repaired ==
      "Retain"
]

# Mutually exclusive hierarchical exclusion reasons so the cohort flow reconciles
# exactly. Transfer-in status is adjudicated first because the target population
# is specifically direct index-center presentation; among direct presentations,
# downstream trajectory exclusions are then applied.
adult_s06[, primary_cohort_reason := fcase(
  transfer_in_status == "Transfer-in: Yes",
    "Excluded: transfer-in Yes",

  transfer_in_status == "Transfer-in status unknown",
    "Excluded: transfer-in status unknown",

  !is.na(project_trajectory_status_repaired) &
    project_trajectory_status_repaired == "Exclude: AMA",
    "Excluded: AMA",

  !is.na(project_trajectory_status_repaired) &
    project_trajectory_status_repaired == "Exclude: short-term acute-care transfer",
    "Excluded: transfer-out to another acute-care hospital",

  !is.na(project_trajectory_status_repaired) &
    project_trajectory_status_repaired == "Exclude: custody/ED-other institutional",
    "Excluded: ED-other/institutional/custody",

  !is.na(project_trajectory_status_repaired) &
    project_trajectory_status_repaired == "Retain",
    "Retained: direct-presentation complete trajectory",

  default = "Excluded: other unresolved trajectory status"
)]

# Hard QC: every row must now have exactly one nonmissing hierarchical reason.
if (
  anyNA(adult_s06$primary_cohort_reason) ||
    any(trimws(adult_s06$primary_cohort_reason) == "")
) {
  stop(
    "At least one patient still lacks a hierarchical cohort reason.",
    call. = FALSE
  )
}

reason_counts <- adult_s06[
  ,
  .N,
  by = primary_cohort_reason
]

flow_order <- c(
  "Excluded: transfer-in Yes",
  "Excluded: transfer-in status unknown",
  "Excluded: AMA",
  "Excluded: transfer-out to another acute-care hospital",
  "Excluded: ED-other/institutional/custody",
  "Excluded: other unresolved trajectory status",
  "Retained: direct-presentation complete trajectory"
)

cohort_flow <- rbind(
  data.table(
    step = "S06 TBI with recorded age 18-89",
    N = nrow(adult_s06)
  ),
  data.table(
    step = flow_order,
    N = vapply(
      flow_order,
      function(z) {
        val <- reason_counts[
          primary_cohort_reason == z,
          N
        ]
        if (length(val) == 0L) 0L else as.integer(val[1])
      },
      integer(1)
    )
  )
)

if (
  sum(cohort_flow[-1, N]) !=
    cohort_flow[1, N]
) {
  fwrite(
    reason_counts,
    file.path(
      audit_dir,
      "ERROR_cohort_reason_counts.csv"
    )
  )

  stop(
    paste0(
      "Hierarchical cohort-flow counts do not reconcile. ",
      "Expected ", cohort_flow[1, N],
      " rows but classified ", sum(cohort_flow[-1, N]),
      ". See ERROR_cohort_reason_counts.csv."
    ),
    call. = FALSE
  )
}

if (
  cohort_flow[
    step == "Retained: direct-presentation complete trajectory",
    N
  ] != nrow(primary_candidate)
) {
  stop(
    "Retained cohort-flow count does not match primary_candidate.",
    call. = FALSE
  )
}

fwrite(
  cohort_flow,
  file.path(
    audit_dir,
    "05_PRIMARY_COHORT_FLOW_DIRECT_PRESENTATIONS.csv"
  )
)

fwrite(
  adult_s06[
    ,
    .N,
    by = .(
      admission_year,
      primary_cohort_reason
    )
  ][
    order(
      admission_year,
      primary_cohort_reason
    )
  ],
  file.path(
    audit_dir,
    "05B_PRIMARY_COHORT_EXCLUSION_REASON_BY_YEAR.csv"
  )
)

candidate_by_year <- primary_candidate[
  ,
  .N,
  by = admission_year
][
  order(admission_year)
]

fwrite(
  candidate_by_year,
  file.path(
    audit_dir,
    "06_PRIMARY_CANDIDATE_COHORT_BY_YEAR.csv"
  )
)

# -----------------------------------------------------------------------------
# ICD injury phenotype derivation and adjudication
# -----------------------------------------------------------------------------

cat("Deriving ICD injury phenotypes for adjudication...\n")

# Structural / acute injury patterns.
dx[, dx_concussion_tmp :=
     startsWith(code, "S060")]
dx[, dx_cerebral_edema_traumatic_tmp :=
     startsWith(code, "S061")]
dx[, dx_diffuse_axonal_injury_tmp :=
     startsWith(code, "S062")]
dx[, dx_focal_contusion_or_iph_tmp :=
     startsWith(code, "S063")]
dx[, dx_epidural_hematoma_tmp :=
     startsWith(code, "S064")]
dx[, dx_subdural_hematoma_tmp :=
     startsWith(code, "S065")]
dx[, dx_subarachnoid_hemorrhage_tmp :=
     startsWith(code, "S066")]
dx[, dx_other_intracranial_injury_tmp :=
     startsWith(code, "S068") |
       startsWith(code, "S069")]

# G93.5/G93.6 can reflect evolving/non-traumatic states and therefore undergo
# separate temporal-integrity adjudication.
dx[, dx_brain_compression_herniation_tmp :=
     startsWith(code, "G935") |
       startsWith(code, "G936")]

dx[, dx_skull_fracture_any_tmp :=
     startsWith(code, "S02")]
dx[, dx_vault_skull_fracture_tmp :=
     startsWith(code, "S020")]
dx[, dx_base_skull_fracture_tmp :=
     startsWith(code, "S021")]
dx[, dx_facial_fracture_tmp :=
     startsWith(code, "S02") &
       !startsWith(code, "S020") &
       !startsWith(code, "S021")]
dx[, dx_open_wound_head_tmp :=
     startsWith(code, "S01")]

dx[, dx_spinal_cord_injury_tmp :=
     startsWith(code, "S14") |
       startsWith(code, "S24") |
       startsWith(code, "S34")]
dx[, dx_neck_vascular_injury_tmp :=
     startsWith(code, "S15")]
dx[, dx_thoracic_injury_tmp :=
     grepl("^S2[0-9]", code)]
dx[, dx_abdominal_pelvic_injury_tmp :=
     grepl("^S3[0-9]", code)]
dx[, dx_upper_extremity_injury_tmp :=
     grepl("^S[456][0-9]", code)]
dx[, dx_lower_extremity_injury_tmp :=
     grepl("^S[789][0-9]", code)]

dx_summary <- dx[
  ,
  .(
    dx_any_s06_intracranial =
      as.integer(any(
        startsWith(code, "S06"),
        na.rm = TRUE
      )),
    dx_concussion =
      as.integer(any(
        dx_concussion_tmp,
        na.rm = TRUE
      )),
    dx_cerebral_edema_traumatic =
      as.integer(any(
        dx_cerebral_edema_traumatic_tmp,
        na.rm = TRUE
      )),
    dx_diffuse_axonal_injury =
      as.integer(any(
        dx_diffuse_axonal_injury_tmp,
        na.rm = TRUE
      )),
    dx_focal_contusion_or_iph =
      as.integer(any(
        dx_focal_contusion_or_iph_tmp,
        na.rm = TRUE
      )),
    dx_epidural_hematoma =
      as.integer(any(
        dx_epidural_hematoma_tmp,
        na.rm = TRUE
      )),
    dx_subdural_hematoma =
      as.integer(any(
        dx_subdural_hematoma_tmp,
        na.rm = TRUE
      )),
    dx_subarachnoid_hemorrhage =
      as.integer(any(
        dx_subarachnoid_hemorrhage_tmp,
        na.rm = TRUE
      )),
    dx_other_intracranial_injury =
      as.integer(any(
        dx_other_intracranial_injury_tmp,
        na.rm = TRUE
      )),
    dx_brain_compression_herniation =
      as.integer(any(
        dx_brain_compression_herniation_tmp,
        na.rm = TRUE
      )),
    dx_skull_fracture_any =
      as.integer(any(
        dx_skull_fracture_any_tmp,
        na.rm = TRUE
      )),
    dx_vault_skull_fracture =
      as.integer(any(
        dx_vault_skull_fracture_tmp,
        na.rm = TRUE
      )),
    dx_base_skull_fracture =
      as.integer(any(
        dx_base_skull_fracture_tmp,
        na.rm = TRUE
      )),
    dx_facial_fracture =
      as.integer(any(
        dx_facial_fracture_tmp,
        na.rm = TRUE
      )),
    dx_open_wound_head =
      as.integer(any(
        dx_open_wound_head_tmp,
        na.rm = TRUE
      )),
    dx_spinal_cord_injury =
      as.integer(any(
        dx_spinal_cord_injury_tmp,
        na.rm = TRUE
      )),
    dx_neck_vascular_injury =
      as.integer(any(
        dx_neck_vascular_injury_tmp,
        na.rm = TRUE
      )),
    dx_thoracic_injury =
      as.integer(any(
        dx_thoracic_injury_tmp,
        na.rm = TRUE
      )),
    dx_abdominal_pelvic_injury =
      as.integer(any(
        dx_abdominal_pelvic_injury_tmp,
        na.rm = TRUE
      )),
    dx_upper_extremity_injury =
      as.integer(any(
        dx_upper_extremity_injury_tmp,
        na.rm = TRUE
      )),
    dx_lower_extremity_injury =
      as.integer(any(
        dx_lower_extremity_injury_tmp,
        na.rm = TRUE
      ))
  ),
  by = id_vars
]

dx_summary[, dx_intracranial_hemorrhage_any :=
  as.integer(
    dx_focal_contusion_or_iph == 1L |
      dx_epidural_hematoma == 1L |
      dx_subdural_hematoma == 1L |
      dx_subarachnoid_hemorrhage == 1L
  )
]

dx_summary[, dx_multiple_intracranial_patterns :=
  as.integer(
    dx_focal_contusion_or_iph +
      dx_epidural_hematoma +
      dx_subdural_hematoma +
      dx_subarachnoid_hemorrhage +
      dx_other_intracranial_injury >= 2
  )
]

dx_summary[, dx_polyregion_injury_count :=
  rowSums(.SD),
  .SDcols = c(
    "dx_any_s06_intracranial",
    "dx_facial_fracture",
    "dx_thoracic_injury",
    "dx_abdominal_pelvic_injury",
    "dx_spinal_cord_injury",
    "dx_upper_extremity_injury",
    "dx_lower_extremity_injury"
  )
]

dx_summary[, dx_polyregion_2plus :=
  as.integer(
    dx_polyregion_injury_count >= 2
  )
]
dx_summary[, dx_polyregion_3plus :=
  as.integer(
    dx_polyregion_injury_count >= 3
  )
]

# Restrict prevalence audit to candidate direct-presentation cohort.
primary_ids <- unique(
  primary_candidate[, ..id_vars]
)

dx_primary <- merge(
  dx_summary,
  primary_ids,
  by = id_vars,
  all = FALSE
)

injury_vars <- setdiff(
  names(dx_summary),
  c(
    id_vars,
    "dx_polyregion_injury_count"
  )
)

injury_prevalence <- rbindlist(
  lapply(
    injury_vars,
    function(v) {
      dx_primary[
        ,
        .(
          N = .N,
          n_positive = sum(
            get(v) == 1L,
            na.rm = TRUE
          ),
          prevalence_pct =
            100 * mean(
              get(v) == 1L,
              na.rm = TRUE
            )
        ),
        by = admission_year
      ][, variable := v]
    }
  ),
  fill = TRUE
)

setcolorder(
  injury_prevalence,
  c(
    "variable",
    "admission_year",
    "N",
    "n_positive",
    "prevalence_pct"
  )
)

fwrite(
  injury_prevalence[
    order(
      variable,
      admission_year
    )
  ],
  file.path(
    audit_dir,
    "07_ICD_INJURY_PHENOTYPE_PREVALENCE_BY_YEAR.csv"
  )
)


fwrite(
  dx_primary[
    ,
    .(
      N = .N,
      mean_polyregion_count =
        mean(dx_polyregion_injury_count, na.rm = TRUE),
      median_polyregion_count =
        median(dx_polyregion_injury_count, na.rm = TRUE),
      p75_polyregion_count =
        as.numeric(
          quantile(
            dx_polyregion_injury_count,
            0.75,
            na.rm = TRUE
          )
        ),
      p90_polyregion_count =
        as.numeric(
          quantile(
            dx_polyregion_injury_count,
            0.90,
            na.rm = TRUE
          )
        )
    ),
    by = admission_year
  ][
    order(admission_year)
  ],
  file.path(
    audit_dir,
    "07B_ICD_POLYREGION_COUNT_DISTRIBUTION_BY_YEAR.csv"
  )
)

injury_adjudication <- data.table(
  variable = c(
    "dx_concussion",
    "dx_cerebral_edema_traumatic",
    "dx_diffuse_axonal_injury",
    "dx_focal_contusion_or_iph",
    "dx_epidural_hematoma",
    "dx_subdural_hematoma",
    "dx_subarachnoid_hemorrhage",
    "dx_other_intracranial_injury",
    "dx_brain_compression_herniation",
    "dx_skull_fracture_any",
    "dx_vault_skull_fracture",
    "dx_base_skull_fracture",
    "dx_facial_fracture",
    "dx_open_wound_head",
    "dx_spinal_cord_injury",
    "dx_neck_vascular_injury",
    "dx_thoracic_injury",
    "dx_abdominal_pelvic_injury",
    "dx_upper_extremity_injury",
    "dx_lower_extremity_injury",
    "dx_any_s06_intracranial",
    "dx_intracranial_hemorrhage_any",
    "dx_multiple_intracranial_patterns",
    "dx_polyregion_injury_count",
    "dx_polyregion_2plus",
    "dx_polyregion_3plus"
  ),
  proposed_role = c(
    "PRIMARY INCLUDE",
    "TEMPORAL REVIEW",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "TEMPORAL REVIEW",
    "PRIMARY INCLUDE",
    "EXCLUDE REDUNDANT",
    "EXCLUDE REDUNDANT",
    "PRIMARY INCLUDE",
    "CLINICAL-RELEVANCE REVIEW",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "PRIMARY INCLUDE",
    "EXCLUDE CONSTANT",
    "EXCLUDE REDUNDANT",
    "EXCLUDE REDUNDANT",
    "EXCLUDE REDUNDANT",
    "EXCLUDE REDUNDANT",
    "EXCLUDE REDUNDANT"
  ),
  rationale = c(
    "Acute traumatic intracranial injury phenotype; exists at injury and is clinically interpretable.",
    "Can be visible on initial imaging but can also evolve after admission; final ICD diagnosis lacks time-of-recognition.",
    "Acute traumatic injury phenotype.",
    "Acute traumatic injury phenotype.",
    "Acute traumatic injury phenotype.",
    "Acute traumatic injury phenotype.",
    "Acute traumatic injury phenotype.",
    "Preserves patients coded with other/unspecified acute S06 injury patterns.",
    "G93.5/G93.6 may reflect evolving or non-traumatic compression/edema states and lacks diagnosis timestamp.",
    "Acute structural injury; use one broad skull-fracture flag rather than simultaneously entering vault/base subclasses.",
    "Deterministic refinement of broad skull-fracture flag; avoid simultaneous redundant encodings.",
    "Deterministic refinement of broad skull-fracture flag; avoid simultaneous redundant encodings.",
    "Acute extracranial injury phenotype.",
    "Time-valid but may add limited resource information beyond other head-injury variables; adjudicate clinically rather than by 2024 AUROC.",
    "Acute extracranial injury phenotype.",
    "Acute extracranial injury phenotype.",
    "Acute extracranial injury phenotype.",
    "Acute extracranial injury phenotype.",
    "Acute extracranial injury phenotype.",
    "Acute extracranial injury phenotype.",
    "Constant by definition after S06 cohort restriction.",
    "Derived composite of included hemorrhage flags; avoid redundant encoding.",
    "Derived composite of included intracranial flags; avoid redundant encoding.",
    "Derived count of included injury-region flags; avoid redundant encoding.",
    "Thresholded derivative of polyregion count.",
    "Thresholded derivative of polyregion count."
  )
)

fwrite(
  injury_adjudication,
  file.path(
    audit_dir,
    "08_ICD_INJURY_PHENOTYPE_ADJUDICATION.csv"
  )
)

# -----------------------------------------------------------------------------
# Pre-existing condition lookup decoding
# -----------------------------------------------------------------------------

cat("Reading harmonized pre-existing conditions...\n")
pc <- as.data.table(
  readRDS(preexisting_file)
)
pc <- pc[
  admission_year %in% audit_years
]

for (v in intersect(
  c("inc_key", "harmonized_patient_id"),
  names(pc)
)) {
  pc[, (v) := normalize_id(get(v))]
}

# Raw lookup discovery.
raw_search_roots <- unique(c(
  project_dir,
  data_dir,
  dirname(warehouse_dir),
  "C:/Users/garin/OneDrive/OHSU/Research/TQIP/Griffith_G-selected",
  "C:/OneDrive/garingriffith/OneDrive/OHSU/Research/TQIP/Griffith_G-selected",
  "C:/Users/garin/OneDrive/OHSU/Research/TQIP/TQIP Harmony",
  "C:/OneDrive/garingriffith/OneDrive/OHSU/Research/TQIP/TQIP Harmony"
))

raw_search_roots <- raw_search_roots[
  dir.exists(raw_search_roots)
]

lookup_files <- unique(
  unlist(
    lapply(
      raw_search_roots,
      function(root) {
        list.files(
          root,
          pattern = "^PUF_TRAUMA_LOOKUP\\.csv$",
          recursive = TRUE,
          full.names = TRUE,
          ignore.case = TRUE
        )
      }
    )
  )
)

lookup_inventory <- if (
  length(lookup_files) > 0L
) {
  rbindlist(
    lapply(
      lookup_files,
      function(f) {
        nm <- header_names(f)
        data.table(
          path = normalizePath(
            f,
            winslash = "/",
            mustWork = FALSE
          ),
          admission_year =
            extract_year_from_path(f),
          n_columns = length(nm),
          columns =
            paste(nm, collapse = " | ")
        )
      }
    ),
    fill = TRUE
  )
} else {
  data.table(
    path = character(),
    admission_year = integer(),
    n_columns = integer(),
    columns = character()
  )
}

fwrite(
  lookup_inventory[
    order(admission_year, path)
  ],
  file.path(
    audit_dir,
    "09_PUF_TRAUMA_LOOKUP_SOURCE_INVENTORY.csv"
  )
)

# Select one lookup per year, preferring the largest file if duplicates exist.
selected_lookup <- rbindlist(
  lapply(
    audit_years,
    function(yr) {
      cand <- lookup_inventory[
        admission_year == yr
      ]

      if (nrow(cand) == 0L) {
        return(
          data.table(
            admission_year = yr,
            selected_path = NA_character_
          )
        )
      }

      cand[, file_size :=
             file.info(path)$size]
      setorder(
        cand,
        -file_size,
        path
      )

      cand[1, .(
        admission_year,
        selected_path = path
      )]
    }
  )
)

fwrite(
  selected_lookup,
  file.path(
    audit_dir,
    "10_SELECTED_PUF_TRAUMA_LOOKUP_BY_YEAR.csv"
  )
)

decoded_lookup_list <- list()
lookup_decode_status <- list()

for (i in seq_len(nrow(selected_lookup))) {
  yr <- selected_lookup$admission_year[i]
  f <- selected_lookup$selected_path[i]

  if (is.na(f) || !file.exists(f)) {
    lookup_decode_status[[
      length(lookup_decode_status) + 1L
    ]] <- data.table(
      admission_year = yr,
      status = "Lookup file not found",
      format_col = NA_character_,
      code_col = NA_character_,
      label_col = NA_character_
    )
    next
  }

  lk <- fread(
    f,
    na.strings = c("", "NA"),
    showProgress = FALSE
  )

  nms <- names(lk)

  format_col <- first_matching_col(
    nms,
    c(
      "FMTNAME",
      "FORMATNAME",
      "FORMAT_NAME",
      "FORMAT",
      "FMT_NAME"
    )
  )

  code_col <- first_matching_col(
    nms,
    c(
      "START",
      "VALUE",
      "CODE",
      "FORMATVALUE",
      "FORMAT_VALUE"
    )
  )

  label_col <- first_matching_col(
    nms,
    c(
      "LABEL",
      "DESCRIPTION",
      "DESC",
      "TEXT",
      "FORMATLABEL",
      "FORMAT_LABEL"
    )
  )

  lookup_decode_status[[
    length(lookup_decode_status) + 1L
  ]] <- data.table(
    admission_year = yr,
    status = if (
      anyNA(
        c(
          format_col,
          code_col,
          label_col
        )
      )
    ) {
      "Required lookup columns not automatically recognized"
    } else {
      "Decoded"
    },
    format_col = format_col,
    code_col = code_col,
    label_col = label_col
  )

  if (
    anyNA(
      c(
        format_col,
        code_col,
        label_col
      )
    )
  ) {
    next
  }

  fmt <- norm_text(
    lk[[format_col]]
  )

  keep <- grepl(
    "pre.*exist|comorbid",
    fmt
  )

  sub <- lk[keep]

  if (nrow(sub) == 0L) {
    # Fallback: find labels that resemble known preexisting-condition names.
    labels_all <- norm_text(
      lk[[label_col]]
    )

    keep <- grepl(
      paste0(
        "bleed|anticoag|copd|chronic obstruct|diabet|hypertens|",
        "smok|tobacco|function.*depend|dement|heart failure|",
        "dialysis|cirrhos|steroid"
      ),
      labels_all
    )

    sub <- lk[keep]
  }

  if (nrow(sub) > 0L) {
    decoded_lookup_list[[
      length(decoded_lookup_list) + 1L
    ]] <- data.table(
      admission_year = yr,
      lookup_format =
        as.character(
          sub[[format_col]]
        ),
      condition_code =
        normalize_id(
          sub[[code_col]]
        ),
      condition_label =
        as.character(
          sub[[label_col]]
        )
    )
  }
}

lookup_decode_status_dt <- rbindlist(
  lookup_decode_status,
  fill = TRUE
)

fwrite(
  lookup_decode_status_dt,
  file.path(
    audit_dir,
    "11_PREEXISTING_LOOKUP_DECODE_STATUS.csv"
  )
)

decoded_lookup <- if (
  length(decoded_lookup_list) > 0L
) {
  unique(
    rbindlist(
      decoded_lookup_list,
      fill = TRUE
    )
  )
} else {
  data.table(
    admission_year = integer(),
    lookup_format = character(),
    condition_code = character(),
    condition_label = character()
  )
}

fwrite(
  decoded_lookup[
    order(
      admission_year,
      safe_num(condition_code),
      condition_code
    )
  ],
  file.path(
    audit_dir,
    "12_DECODED_PREEXISTING_CONDITION_LOOKUP.csv"
  )
)

# Harmonized condition-code inventory.
#
# IMPORTANT: In the current Harmony build, condition_code is blank for the
# PUF_PREEXISTINGCONDITIONS rows, while the actual ACS numeric condition code is
# carried in condition_name and condition_group_clean (e.g., 4, 11, 19, 23).
# Derive a robust code source instead of assuming condition_code is populated.
get_pc_chr <- function(nm) {
  if (nm %in% names(pc)) {
    normalize_id(pc[[nm]])
  } else {
    rep(NA_character_, nrow(pc))
  }
}

pc_code_candidates <- data.table(
  condition_code = get_pc_chr("condition_code"),
  condition_name = get_pc_chr("condition_name"),
  condition_group_clean = get_pc_chr("condition_group_clean"),
  condition_answer_biu = get_pc_chr("condition_answer_biu"),
  condition_value = get_pc_chr("condition_value")
)

is_integer_like_code <- function(x) {
  !is.na(x) & grepl("^[0-9]+$", x)
}

pc[, condition_code_norm := {
  c1 <- pc_code_candidates$condition_code
  c2 <- pc_code_candidates$condition_name
  c3 <- pc_code_candidates$condition_group_clean

  out <- fifelse(
    is_integer_like_code(c1),
    c1,
    fifelse(
      is_integer_like_code(c2),
      c2,
      fifelse(
        is_integer_like_code(c3),
        c3,
        NA_character_
      )
    )
  )
  out
}]

pc[, condition_code_source := {
  c1 <- pc_code_candidates$condition_code
  c2 <- pc_code_candidates$condition_name
  c3 <- pc_code_candidates$condition_group_clean

  fcase(
    is_integer_like_code(c1), "condition_code",
    is_integer_like_code(c2), "condition_name",
    is_integer_like_code(c3), "condition_group_clean",
    default = "UNRESOLVED"
  )
}]

pc[, present_row :=
  is_present_condition_row(
    condition_answer,
    condition_value,
    condition_answer_biu
  )
]

pc_code_source_qc <- pc[
  ,
  .N,
  by = .(
    admission_year,
    condition_code_source
  )
][
  order(
    admission_year,
    condition_code_source
  )
]

fwrite(
  pc_code_source_qc,
  file.path(
    audit_dir,
    "12B_PMHX_HARMONIZED_CODE_SOURCE_QC.csv"
  )
)

pc_present <- pc[
  present_row == TRUE
]

pc_inventory <- pc_present[
  ,
  .(
    row_count = .N,
    unique_patients =
      uniqueN(
        paste(
          admission_year,
          inc_key,
          harmonized_patient_id,
          sep = "|"
        )
      )
  ),
  by = .(
    admission_year,
    condition_code_norm
  )
][
  order(
    admission_year,
    safe_num(condition_code_norm),
    condition_code_norm
  )
]

fwrite(
  pc_inventory,
  file.path(
    audit_dir,
    "13_HARMONIZED_PREEXISTING_CODE_INVENTORY.csv"
  )
)

# Merge decoded labels where available.
if (nrow(decoded_lookup) > 0L) {
  pc_labeled <- merge(
    pc_inventory,
    decoded_lookup[
      ,
      .(
        admission_year,
        condition_code_norm =
          normalize_id(condition_code),
        condition_label
      )
    ],
    by = c(
      "admission_year",
      "condition_code_norm"
    ),
    all.x = TRUE
  )
} else {
  pc_labeled <- copy(
    pc_inventory
  )
  pc_labeled[, condition_label :=
               NA_character_]
}

fwrite(
  pc_labeled[
    order(
      admission_year,
      safe_num(condition_code_norm),
      condition_code_norm
    )
  ],
  file.path(
    audit_dir,
    "14_HARMONIZED_PREEXISTING_CODES_WITH_LABELS.csv"
  )
)

# Prespecified PMHx concept map.
#
# Use exact ACS lookup labels rather than broad regex matching. This prevents
# concept cross-contamination (for example, "coagul" matching "Anticoagulant
# Therapy" when mapping "Bleeding Disorder") and makes the derivation auditable.
pmhx_concepts <- data.table(
  concept = c(
    "Bleeding disorder",
    "Anticoagulant therapy",
    "COPD",
    "Diabetes",
    "Hypertension",
    "Current smoker",
    "Functional dependence",
    "Dementia",
    "Congestive heart failure",
    "Chronic renal failure",
    "Cirrhosis",
    "Chronic steroid use"
  ),
  exact_acs_label = c(
    "Bleeding Disorder",
    "Anticoagulant Therapy",
    "Chronic Obstructive Pulmonary Disease",
    "Diabetes Mellitus",
    "Hypertension",
    "Current Smoker",
    "Functionally Dependent Health Status",
    "Dementia",
    "Congestive Heart Failure",
    "Chronic Renal Failure",
    "Cirrhosis",
    "Steroid Use"
  ),
  proposed_role = c(
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN",
    "PRIMARY RETAIN"
  )
)

pmhx_matches <- merge(
  decoded_lookup[
    ,
    .(
      admission_year,
      condition_code_norm = normalize_id(condition_code),
      condition_label
    )
  ],
  pmhx_concepts[
    ,
    .(
      concept,
      exact_acs_label,
      proposed_role
    )
  ],
  by.x = "condition_label",
  by.y = "exact_acs_label",
  all.y = TRUE,
  allow.cartesian = TRUE
)

setcolorder(
  pmhx_matches,
  c(
    "concept",
    "proposed_role",
    "admission_year",
    "condition_code_norm",
    "condition_label"
  )
)

# Add observed harmonized row counts after the exact lookup mapping.
pmhx_matches <- merge(
  pmhx_matches,
  pc_labeled[
    ,
    .(
      admission_year,
      condition_code_norm,
      row_count,
      unique_patients
    )
  ],
  by = c(
    "admission_year",
    "condition_code_norm"
  ),
  all.x = TRUE
)

fwrite(
  pmhx_matches[
    order(
      concept,
      admission_year,
      condition_code_norm
    )
  ],
  file.path(
    audit_dir,
    "15_PMHX_CONCEPT_TO_ACS_CODE_MAPPING.csv"
  )
)

# Build PMHx prevalence in the candidate direct-presentation cohort using the
# decoded concept mappings, but only when labels were successfully decoded.
pmhx_prevalence_list <- list()

if (
  any(
    !is.na(
      pmhx_matches$condition_code_norm
    )
  )
) {
  primary_key <- unique(
    primary_candidate[, ..id_vars]
  )

  pc_primary <- merge(
    pc_present,
    primary_key,
    by = id_vars,
    all = FALSE
  )

  for (i in seq_len(
    nrow(pmhx_concepts)
  )) {
    concept_id <- pmhx_concepts$concept[i]

    # IMPORTANT: use a local variable named concept_id so data.table does not
    # resolve both sides of "concept == concept" to the column itself.
    mapped_codes <- unique(
      pmhx_matches[
        concept == concept_id &
          !is.na(condition_code_norm),
        condition_code_norm
      ]
    )

    if (length(mapped_codes) == 0L) next

    positive_ids <- unique(
      pc_primary[
        condition_code_norm %in%
          mapped_codes,
        ..id_vars
      ]
    )
    positive_ids[, positive := 1L]

    tmp <- merge(
      primary_key,
      positive_ids,
      by = id_vars,
      all.x = TRUE
    )
    tmp[
      is.na(positive),
      positive := 0L
    ]

    prev <- tmp[
      ,
      .(
        N = .N,
        n_positive =
          sum(positive == 1L),
        prevalence_pct =
          100 * mean(
            positive == 1L
          )
      ),
      by = admission_year
    ]
    prev[, concept := concept_id]

    pmhx_prevalence_list[[
      length(pmhx_prevalence_list) + 1L
    ]] <- prev
  }
}

pmhx_prevalence <- if (
  length(pmhx_prevalence_list) > 0L
) {
  rbindlist(
    pmhx_prevalence_list,
    fill = TRUE
  )
} else {
  data.table(
    admission_year = integer(),
    N = integer(),
    n_positive = integer(),
    prevalence_pct = numeric(),
    concept = character()
  )
}

fwrite(
  pmhx_prevalence[
    order(
      concept,
      admission_year
    )
  ],
  file.path(
    audit_dir,
    "16_PMHX_PREVALENCE_IN_DIRECT_PRIMARY_COHORT.csv"
  )
)


# Every selected PMHx concept must map in every study year.
expected_selected_pmhx_qc <- merge(
  CJ(
    concept = pmhx_concepts$concept,
    admission_year = audit_years,
    unique = TRUE
  ),
  pmhx_matches[
    ,
    .(
      concept,
      admission_year,
      condition_code_norm,
      condition_label
    )
  ],
  by = c(
    "concept",
    "admission_year"
  ),
  all.x = TRUE
)

expected_selected_pmhx_qc[, mapping_ok :=
  !is.na(condition_code_norm) &
    !is.na(condition_label)
]

fwrite(
  expected_selected_pmhx_qc[
    order(
      concept,
      admission_year
    )
  ],
  file.path(
    audit_dir,
    "16B_SELECTED_PMHX_MAPPING_QC.csv"
  )
)

# -----------------------------------------------------------------------------
# Race / ethnicity / payer policy
# -----------------------------------------------------------------------------

social_policy <- data.table(
  variable_domain = c(
    "Race",
    "Ethnicity",
    "Payer / insurance"
  ),
  primary_predictor_role = c(
    "EXCLUDE FROM BEDSIDE PREDICTION",
    "EXCLUDE FROM BEDSIDE PREDICTION",
    "EXCLUDE FROM BEDSIDE PREDICTION"
  ),
  analytic_role = c(
    "Retain for descriptive reporting and fairness assessment of discrimination/calibration when group sizes permit.",
    "Retain for descriptive reporting and fairness assessment of discrimination/calibration when group sizes permit.",
    "Retain for subgroup/transportability assessment, particularly disposition and LOS; do not let historical access patterns become an individual clinical resource determinant."
  ),
  rationale = c(
    "Can encode structural inequities rather than patient biology; intended clinical use does not require race to determine treatment/resource need.",
    "Same use-case concern as race; evaluate model performance rather than using ethnicity to generate bedside resource predictions.",
    "Strongly tied to observed access to post-acute resources and therefore risks predicting system constraints rather than clinical need."
  )
)

fwrite(
  social_policy,
  file.path(
    audit_dir,
    "17_RACE_ETHNICITY_PAYER_POLICY.csv"
  )
)

# -----------------------------------------------------------------------------
# Preliminary predictor dictionary
# -----------------------------------------------------------------------------

baseline_predictors <- data.table(
  domain = c(
    rep("Demographics", 2),
    rep("Injury context", 2),
    rep("Neurologic status", 8),
    rep("Initial physiology", 7)
  ),
  variable = c(
    "age",
    "sex_clean",
    "mechanism_clean",
    "helmet_use_recovered",
    "gcs_eye_clean",
    "gcs_verbal_clean",
    "gcs_motor_clean",
    "gcsq_intubated_recovered",
    "gcsq_sedated_paralyzed_recovered",
    "gcsq_eye_obstruction_recovered",
    "gcsq_unknown_recovered",
    "pupil_clean",
    "sbp_clean",
    "pulse_clean",
    "rr_clean",
    "spo2_clean",
    "respiratoryassistance_clean",
    "supplemental_oxygen_recovered",
    "temperature_c_recovered"
  ),
  proposed_role = "PRIMARY INCLUDE",
  note = c(
    "Continuous recorded age; cohort limited to recorded 18-89.",
    "Direct clinical demographic.",
    "Admission injury context.",
    "Recovered from all PROTDEV fields with explicit Unknown.",
    "Use component, not total/severity duplicate.",
    "Use component, not total/severity duplicate.",
    "Use component, not total/severity duplicate.",
    "Assessment qualifier; endpoint-specific leakage review for ventilation trajectory.",
    "Assessment qualifier.",
    "Assessment qualifier.",
    "Assessment qualifier.",
    "Admission neurologic status.",
    "Raw validated value; no deterministic hypotension flag.",
    "Raw validated value; no deterministic tachycardia flag.",
    "Raw validated value; no deterministic abnormal-RR flag.",
    "Raw validated value; no deterministic hypoxia flag.",
    "Admission respiratory state; exclude from ventilation trajectory if it directly reveals current ventilation.",
    "Recovered initial oxygen context.",
    "Recovered initial temperature."
  )
)

injury_dict <- injury_adjudication[
  ,
  .(
    domain = "ICD injury phenotype",
    variable,
    proposed_role,
    note = rationale
  )
]

pmhx_dict <- pmhx_concepts[
  ,
  .(
    domain = "Pre-existing conditions",
    variable = concept,
    proposed_role,
    note = paste0(
      "Exact ACS lookup label: ",
      exact_acs_label,
      ". Use explicit Unknown/not documented handling at deployment."
    )
  )
]

excluded_structural <- data.table(
  domain = c(
    "Transfer",
    "Demographics/social",
    "Demographics/social",
    "Demographics/social",
    "Registry severity",
    "Registry severity",
    "Derived redundancy",
    "Derived redundancy",
    "Derived redundancy"
  ),
  variable = c(
    "transfer_clean",
    "race_clean",
    "ethnicity_clean",
    "insurance_clean",
    "ISS / ISS-derived variables",
    "AIS / AIS-derived variables",
    "age_group",
    "gcs_total / gcs_severity",
    "selected PMHx count"
  ),
  proposed_role = c(
    "REMOVE PREDICTOR; EXCLUDE TRANSFER-IN PATIENTS",
    "FAIRNESS EVALUATION ONLY",
    "FAIRNESS EVALUATION ONLY",
    "SUBGROUP/TRANSPORTABILITY ONLY",
    "EXCLUDE",
    "EXCLUDE",
    "EXCLUDE REDUNDANT",
    "EXCLUDE REDUNDANT",
    "EXCLUDE REDUNDANT"
  ),
  note = c(
    "Time-zero is heterogeneous after prior-facility care.",
    "Not required for bedside resource-need prediction.",
    "Not required for bedside resource-need prediction.",
    "May encode access constraints rather than clinical resource need.",
    "Retrospective registry severity summary.",
    "Retrospective registry severity summary.",
    "Deterministic transform of age.",
    "Deterministic/near-deterministic transforms of component GCS.",
    "Redundant with individual selected conditions."
  )
)

predictor_dictionary <- rbindlist(
  list(
    baseline_predictors,
    injury_dict,
    pmhx_dict,
    excluded_structural
  ),
  fill = TRUE
)

fwrite(
  predictor_dictionary,
  file.path(
    audit_dir,
    "18_PRELIMINARY_PREDICTOR_ADJUDICATION_DICTIONARY.csv"
  )
)

# -----------------------------------------------------------------------------
# Hard-stop / review flags
# -----------------------------------------------------------------------------

pmhx_mapping_pass <- (
  nrow(lookup_decode_status_dt) == length(audit_years) &&
    all(lookup_decode_status_dt$status == "Decoded") &&
    nrow(pc_labeled) > 0L &&
    any(!is.na(pc_labeled$condition_label)) &&
    nrow(pmhx_prevalence) > 0L &&
    nrow(expected_selected_pmhx_qc) ==
      nrow(pmhx_concepts) * length(audit_years) &&
    all(expected_selected_pmhx_qc$mapping_ok)
)

review_flags <- data.table(
  item = c(
    "Transfer-in direct-presentation definition",
    "PMHx lookup + harmonized-code mapping complete",
    "Cerebral edema timing",
    "Brain compression/herniation timing",
    "Open wound head clinical relevance",
    "Ventilation trajectory direct-state leakage"
  ),
  status = c(
    "LOCKED: exclude transfer-in Yes and unknown",
    if (pmhx_mapping_pass) "PASS" else "REVIEW REQUIRED",
    "REVIEW REQUIRED before primary model",
    "REVIEW REQUIRED before primary model",
    "REVIEW REQUIRED before primary model",
    "REVIEW REQUIRED before primary model; likely exclude respiratory assistance and intubation qualifier"
  )
)

fwrite(
  review_flags,
  file.path(
    audit_dir,
    "19_ITEMS_REQUIRING_FINAL_CLINICAL_ADJUDICATION.csv"
  )
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

summary_lines <- c(
  "TBI-TRACT ADMISSION MODEL PRE-MODEL ADJUDICATION COMPLETE",
  "",
  "NO MODEL WAS FIT.",
  "",
  paste0(
    "Recorded-age 18-89 S06 cohort before transfer/trajectory exclusions: ",
    format(
      nrow(adult_s06),
      big.mark = ","
    )
  ),
  paste0(
    "Candidate primary direct-presentation complete-trajectory cohort: ",
    format(
      nrow(primary_candidate),
      big.mark = ","
    )
  ),
  "",
  "Transfer policy:",
  "  Exclude transfer-in Yes.",
  "  Exclude unknown transfer-in status.",
  "  Exclude transfer-out, AMA, and ED-other/institutional/custody.",
  "  Remove transfer_clean from predictor set.",
  "",
  "ICD injury policy:",
  "  Retain acute clinically interpretable injury phenotypes.",
  "  Exclude constant/redundant composites.",
  "  Review cerebral edema and brain compression/herniation separately for temporal integrity.",
  "",
  "PMHx policy:",
  "  Decode numeric ACS condition codes from yearly PUF_TRAUMA_LOOKUP.",
  "  Restore selected individual baseline conditions after mapping is verified.",
  "  Do not use a redundant selected-condition count.",
  "",
  "Race/ethnicity/payer:",
  "  Do not use as bedside predictors.",
  "  Retain for fairness/subgroup/transportability evaluation.",
  "",
  "COVID sensitivity remains prespecified.",
  "",
  "Most important files to review before final model build:",
  "  02_COMPLETE_METHODS_REPAIR_REGISTER.csv",
  "  05_PRIMARY_COHORT_FLOW_DIRECT_PRESENTATIONS.csv",
  "  08_ICD_INJURY_PHENOTYPE_ADJUDICATION.csv",
  "  12_DECODED_PREEXISTING_CONDITION_LOOKUP.csv",
  "  15_PMHX_CONCEPT_TO_ACS_CODE_MAPPING.csv",
  "  16_PMHX_PREVALENCE_IN_DIRECT_PRIMARY_COHORT.csv",
  "  18_PRELIMINARY_PREDICTOR_ADJUDICATION_DICTIONARY.csv",
  "  19_ITEMS_REQUIRING_FINAL_CLINICAL_ADJUDICATION.csv",
  "",
  "STOP HERE. Review the adjudication outputs before fitting the definitive model."
)

writeLines(
  summary_lines,
  file.path(
    audit_dir,
    "ADJUDICATION_SUMMARY.txt"
  )
)

cat("\n")
cat(
  paste(
    summary_lines,
    collapse = "\n"
  )
)
cat(
  "\n\nOutput:\n  ",
  audit_dir,
  "\n",
  sep = ""
)
